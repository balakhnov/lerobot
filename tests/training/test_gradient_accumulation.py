#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.utils import cycle

accelerate = pytest.importorskip("accelerate")
Accelerator = accelerate.Accelerator
DistributedType = accelerate.DistributedType
GradientAccumulationPlugin = accelerate.utils.GradientAccumulationPlugin
AcceleratorState = accelerate.state.AcceleratorState
GradientState = accelerate.state.GradientState

train_module = importlib.import_module("lerobot.scripts.lerobot_train")


@pytest.fixture(autouse=True)
def reset_accelerate_state():
    """Accelerate uses shared state, so every test must start from a clean process state."""
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


class TinyRegressionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 1, bias=False)
        self.update_calls = 0

    def forward(self, batch):
        prediction = self.projection(batch["input"])
        loss = torch.nn.functional.mse_loss(prediction, batch["target"])
        return loss, {"component_loss": loss.item()}

    def update(self):
        self.update_calls += 1


class TinyDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        value = float(index + 1)
        return {
            "input": torch.tensor([value, -0.5 * value]),
            "target": torch.tensor([0.25 * value]),
        }


class SkippedStepAccelerator:
    """Minimal accelerator double for an AMP-overflow optimizer step."""

    sync_gradients = True
    optimizer_step_was_skipped = True

    def accumulate(self, model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model, keep_fp32_wrapper=True):
        return model


def make_metrics_tracker(batch_size: int = 1) -> MetricsTracker:
    return MetricsTracker(
        batch_size=batch_size,
        num_frames=10,
        num_episodes=1,
        metrics={
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
        },
        initial_step=0,
    )


def make_accelerator(gradient_accumulation_steps: int) -> Accelerator:
    return Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            sync_with_dataloader=False,
        ),
        step_scheduler_with_optimizer=False,
        cpu=True,
    )


def run_update(
    policy,
    batch,
    optimizer,
    accelerator,
    scheduler=None,
    grad_clip_norm=0.0,
    train_metrics=None,
):
    return train_module.update_policy(
        train_metrics=train_metrics or make_metrics_tracker(),
        policy=policy,
        batch=batch,
        optimizer=optimizer,
        grad_clip_norm=grad_clip_norm,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )


def test_gradient_accumulation_config_defaults_to_one(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/test"),
        policy=DiffusionConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "output",
    )

    assert cfg.gradient_accumulation_steps == 1


@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, True])
def test_gradient_accumulation_config_rejects_non_positive_values(tmp_path, invalid_value):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/test"),
        policy=DiffusionConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "output",
    )
    cfg.gradient_accumulation_steps = invalid_value

    with pytest.raises(ValueError, match="gradient_accumulation_steps.*positive integer"):
        cfg.validate()


def test_gradient_accumulation_config_round_trips_through_pretrained_config(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/test"),
        policy=DiffusionConfig(device="cpu", push_to_hub=False),
        gradient_accumulation_steps=4,
    )

    cfg._save_pretrained(tmp_path)
    restored = TrainPipelineConfig.from_pretrained(tmp_path)

    assert restored.gradient_accumulation_steps == 4


def test_resolved_accelerator_disables_dataloader_boundary_sync():
    cfg = SimpleNamespace(
        gradient_accumulation_steps=3,
        trainable_config=SimpleNamespace(device="cpu", dtype="float32", use_amp=False),
    )

    accelerator = train_module._resolve_accelerator(cfg, accelerator=None)

    assert accelerator.gradient_accumulation_steps == 3
    assert accelerator.gradient_state.sync_with_dataloader is False
    assert accelerator.step_scheduler_with_optimizer is False


def test_injected_accelerator_must_match_configured_accumulation_steps():
    cfg = SimpleNamespace(gradient_accumulation_steps=2)
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=1,
        distributed_type=DistributedType.NO,
    )

    with pytest.raises(ValueError, match="configured.*2.*accelerator.*1"):
        train_module._resolve_accelerator(cfg, accelerator=accelerator)


def test_injected_accelerator_must_disable_dataloader_boundary_sync():
    cfg = SimpleNamespace(gradient_accumulation_steps=2)
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=2,
        distributed_type=DistributedType.NO,
        gradient_state=SimpleNamespace(sync_with_dataloader=True),
    )

    with pytest.raises(ValueError, match="sync_with_dataloader.*False"):
        train_module._resolve_accelerator(cfg, accelerator=accelerator)


def test_injected_accelerator_must_not_auto_step_scheduler():
    cfg = SimpleNamespace(gradient_accumulation_steps=2)
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=2,
        distributed_type=DistributedType.NO,
        gradient_state=SimpleNamespace(sync_with_dataloader=False),
        step_scheduler_with_optimizer=True,
    )

    with pytest.raises(ValueError, match="step_scheduler_with_optimizer.*False"):
        train_module._resolve_accelerator(cfg, accelerator=accelerator)


def test_compatible_injected_accelerator_is_reused():
    cfg = SimpleNamespace(gradient_accumulation_steps=2)
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=2,
        distributed_type=DistributedType.NO,
        gradient_state=SimpleNamespace(sync_with_dataloader=False),
        step_scheduler_with_optimizer=False,
    )

    assert train_module._resolve_accelerator(cfg, accelerator=accelerator) is accelerator


def test_gradient_accumulation_rejects_fsdp():
    cfg = SimpleNamespace(gradient_accumulation_steps=2)
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=2,
        distributed_type=DistributedType.FSDP,
    )

    with pytest.raises(NotImplementedError, match="FSDP.*gradient accumulation"):
        train_module._resolve_accelerator(cfg, accelerator=accelerator)


def test_fsdp_without_gradient_accumulation_remains_supported():
    cfg = SimpleNamespace(gradient_accumulation_steps=1)
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=1,
        distributed_type=DistributedType.FSDP,
    )

    assert train_module._resolve_accelerator(cfg, accelerator=accelerator) is accelerator


def test_optimizer_scheduler_and_policy_hook_run_only_on_sync_step():
    accelerator = make_accelerator(gradient_accumulation_steps=2)
    policy = TinyRegressionPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
    clip_grad_norm = Mock(wraps=accelerator.clip_grad_norm_)
    accelerator.clip_grad_norm_ = clip_grad_norm
    train_metrics = make_metrics_tracker()
    initial_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
    initial_lr = scheduler.get_last_lr()[0]

    run_update(
        policy,
        {
            "input": torch.tensor([[1.0, 2.0]]),
            "target": torch.tensor([[0.5]]),
        },
        optimizer,
        accelerator,
        scheduler,
        grad_clip_norm=1.0,
        train_metrics=train_metrics,
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(initial_parameters, policy.parameters(), strict=True)
    )
    assert scheduler.get_last_lr()[0] == initial_lr
    assert accelerator.unwrap_model(policy).update_calls == 0
    clip_grad_norm.assert_not_called()

    run_update(
        policy,
        {
            "input": torch.tensor([[-1.0, 0.5]]),
            "target": torch.tensor([[-0.2]]),
        },
        optimizer,
        accelerator,
        scheduler,
        grad_clip_norm=1.0,
        train_metrics=train_metrics,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, policy.parameters(), strict=True)
    )
    assert scheduler.get_last_lr()[0] == pytest.approx(initial_lr * 0.5)
    assert accelerator.unwrap_model(policy).update_calls == 1
    clip_grad_norm.assert_called_once()
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert train_metrics.loss.count == 2
    assert train_metrics.update_s.count == 2
    assert train_metrics.grad_norm.count == 1
    assert train_metrics.lr.count == 1
    assert train_metrics.component_loss.count == 2


def test_gradient_accumulation_one_preserves_single_batch_behavior():
    accelerator = make_accelerator(gradient_accumulation_steps=1)
    policy = TinyRegressionPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
    initial_parameters = [parameter.detach().clone() for parameter in policy.parameters()]

    run_update(
        policy,
        {
            "input": torch.tensor([[1.0, 2.0]]),
            "target": torch.tensor([[0.5]]),
        },
        optimizer,
        accelerator,
        scheduler,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, policy.parameters(), strict=True)
    )
    assert scheduler.get_last_lr()[0] == pytest.approx(0.05)
    assert accelerator.unwrap_model(policy).update_calls == 1


def test_loss_metric_aggregates_every_microbatch():
    torch.manual_seed(0)
    policy = TinyRegressionPolicy()
    reference_policy = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = make_accelerator(gradient_accumulation_steps=2)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    train_metrics = make_metrics_tracker()
    batches = [
        {
            "input": torch.tensor([[1.0, 2.0]]),
            "target": torch.tensor([[0.5]]),
        },
        {
            "input": torch.tensor([[-1.0, 0.5]]),
            "target": torch.tensor([[-0.2]]),
        },
    ]
    expected_losses = [reference_policy(batch)[0].item() for batch in batches]

    for batch in batches:
        run_update(
            policy,
            batch,
            optimizer,
            accelerator,
            train_metrics=train_metrics,
        )

    assert train_metrics.loss.count == 2
    assert train_metrics.loss.avg == pytest.approx(sum(expected_losses) / 2)
    assert train_metrics.component_loss.count == 2
    assert train_metrics.component_loss.avg == pytest.approx(sum(expected_losses) / 2)


def test_accumulated_update_matches_single_large_batch_update():
    torch.manual_seed(0)
    accumulated_policy = TinyRegressionPolicy()
    baseline_policy = copy.deepcopy(accumulated_policy)
    first_batch = {
        "input": torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
        "target": torch.tensor([[0.5], [-0.2]]),
    }
    second_batch = {
        "input": torch.tensor([[0.25, -2.0], [3.0, 1.0]]),
        "target": torch.tensor([[0.1], [0.75]]),
    }

    accelerator = make_accelerator(gradient_accumulation_steps=2)
    accumulated_optimizer = torch.optim.SGD(accumulated_policy.parameters(), lr=0.1)
    accumulated_policy, accumulated_optimizer = accelerator.prepare(accumulated_policy, accumulated_optimizer)
    run_update(
        accumulated_policy,
        first_batch,
        accumulated_optimizer,
        accelerator,
        grad_clip_norm=0.1,
    )
    run_update(
        accumulated_policy,
        second_batch,
        accumulated_optimizer,
        accelerator,
        grad_clip_norm=0.1,
    )

    baseline_optimizer = torch.optim.SGD(baseline_policy.parameters(), lr=0.1)
    large_batch = {
        "input": torch.cat([first_batch["input"], second_batch["input"]]),
        "target": torch.cat([first_batch["target"], second_batch["target"]]),
    }
    loss, _ = baseline_policy(large_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(baseline_policy.parameters(), max_norm=0.1)
    baseline_optimizer.step()

    for accumulated_parameter, baseline_parameter in zip(
        accelerator.unwrap_model(accumulated_policy).parameters(),
        baseline_policy.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(accumulated_parameter, baseline_parameter)


def test_dataloader_epoch_boundary_does_not_force_a_partial_update():
    cfg = SimpleNamespace(
        gradient_accumulation_steps=2,
        trainable_config=SimpleNamespace(device="cpu", dtype="float32", use_amp=False),
    )
    accelerator = train_module._resolve_accelerator(cfg, accelerator=None)
    policy = TinyRegressionPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    dataloader = DataLoader(TinyDataset(length=5), batch_size=1, shuffle=False)
    policy, optimizer, scheduler, dataloader = accelerator.prepare(policy, optimizer, scheduler, dataloader)
    iterator = cycle(dataloader)

    for _ in range(3):
        for _ in range(cfg.gradient_accumulation_steps):
            run_update(policy, next(iterator), optimizer, accelerator, scheduler)

    assert accelerator.unwrap_model(policy).update_calls == 3
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1 * 0.5**3)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_skipped_amp_optimizer_step_does_not_advance_scheduler_or_policy_hook():
    accelerator = SkippedStepAccelerator()
    policy = TinyRegressionPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    optimizer.step = Mock()
    optimizer.zero_grad = Mock()
    scheduler = Mock()

    train_module.update_policy(
        train_metrics=make_metrics_tracker(),
        policy=policy,
        batch={
            "input": torch.tensor([[1.0, 2.0]]),
            "target": torch.tensor([[0.5]]),
        },
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )

    optimizer.step.assert_called_once_with()
    scheduler.step.assert_not_called()
    assert policy.update_calls == 0
