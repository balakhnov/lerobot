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

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers", reason="transformers is required (install lerobot[pi])")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

BATCH_SIZE = 2
NUM_FLOW_SAMPLES = 3
CHUNK_SIZE = 2
ACTION_DIM = 3
PADDED_ACTION_DIM = 4
IMAGE_SIZE = 28


def make_tiny_config(*, num_flow_samples: int, padded_action_dim: int = ACTION_DIM) -> PI05Config:
    return PI05Config(
        paligemma_variant="gemma_tiny",
        action_expert_variant="gemma_tiny",
        num_hidden_layers=2,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        max_action_dim=padded_action_dim,
        image_resolution=(IMAGE_SIZE, IMAGE_SIZE),
        num_flow_samples=num_flow_samples,
        device="cpu",
    )


def make_model_inputs():
    generator = torch.Generator().manual_seed(123)
    images = [torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)]
    img_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool)]
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    masks = torch.tensor([[True, True, True], [True, True, False]])
    actions = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM, generator=generator)
    noise = torch.randn(
        BATCH_SIZE,
        NUM_FLOW_SAMPLES,
        CHUNK_SIZE,
        ACTION_DIM,
        generator=generator,
    )
    time = torch.tensor([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]])
    return images, img_masks, tokens, masks, actions, noise, time


def test_num_flow_samples_must_be_positive():
    with pytest.raises(ValueError, match="num_flow_samples must be at least 1"):
        make_tiny_config(num_flow_samples=0)


def test_cached_prefix_training_rejects_gradient_checkpointing():
    model = PI05Pytorch(make_tiny_config(num_flow_samples=NUM_FLOW_SAMPLES))
    model.gradient_checkpointing_enable()
    model.train()

    with pytest.raises(RuntimeError, match="does not support gradient checkpointing"):
        model(None, None, None, None, None, None, None)


def test_single_flow_sample_preserves_legacy_forward_contract():
    model = PI05Pytorch(make_tiny_config(num_flow_samples=1)).eval()
    images, img_masks, tokens, masks, actions, noise, time = make_model_inputs()

    with torch.no_grad():
        legacy_losses = model(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise[:, 0],
            time[:, 0],
        )
        sample_axis_losses = model(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise[:, :1],
            time[:, :1],
        )

    assert legacy_losses.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    torch.testing.assert_close(legacy_losses, sample_axis_losses[:, 0], rtol=0, atol=0)


def test_vectorized_flow_samples_match_explicit_losses_and_gradients():
    vectorized_model = PI05Pytorch(make_tiny_config(num_flow_samples=NUM_FLOW_SAMPLES))
    explicit_model = PI05Pytorch(make_tiny_config(num_flow_samples=1))
    explicit_model.load_state_dict(deepcopy(vectorized_model.state_dict()))
    vectorized_model.train()
    explicit_model.train()

    images, img_masks, tokens, masks, actions, noise, time = make_model_inputs()

    vectorized_losses = vectorized_model(images, img_masks, tokens, masks, actions, noise, time)
    explicit_losses = torch.cat(
        [
            explicit_model(
                images,
                img_masks,
                tokens,
                masks,
                actions,
                noise[:, sample_index : sample_index + 1],
                time[:, sample_index : sample_index + 1],
            )
            for sample_index in range(NUM_FLOW_SAMPLES)
        ],
        dim=1,
    )

    assert vectorized_losses.shape == (
        BATCH_SIZE,
        NUM_FLOW_SAMPLES,
        CHUNK_SIZE,
        ACTION_DIM,
    )
    torch.testing.assert_close(vectorized_losses, explicit_losses, rtol=1e-5, atol=1e-6)

    vectorized_losses.mean().backward()
    explicit_losses.mean().backward()
    for (vectorized_name, vectorized_parameter), (explicit_name, explicit_parameter) in zip(
        vectorized_model.named_parameters(), explicit_model.named_parameters(), strict=True
    ):
        assert vectorized_name == explicit_name
        if vectorized_parameter.grad is None or explicit_parameter.grad is None:
            assert vectorized_parameter.grad is None and explicit_parameter.grad is None
            continue
        torch.testing.assert_close(
            vectorized_parameter.grad,
            explicit_parameter.grad,
            rtol=2e-4,
            atol=2e-6,
            msg=lambda message, name=vectorized_name: f"Gradient mismatch for {name}: {message}",
        )


def test_policy_samples_independent_flows_and_reduces_to_batch(monkeypatch):
    config = make_tiny_config(
        num_flow_samples=NUM_FLOW_SAMPLES,
        padded_action_dim=PADDED_ACTION_DIM,
    )
    image_key = f"{OBS_IMAGES}.camera"
    config.input_features = {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    policy = PI05Policy(config)
    captured = {}

    def sample_noise(shape, device):
        captured["noise_shape"] = tuple(shape)
        return torch.arange(torch.tensor(shape).prod(), dtype=torch.float32, device=device).reshape(shape)

    def sample_time(batch_size, device):
        captured["time_batch_size"] = batch_size
        return torch.arange(batch_size, dtype=torch.float32, device=device)

    expected_losses = torch.arange(
        BATCH_SIZE * NUM_FLOW_SAMPLES * CHUNK_SIZE * PADDED_ACTION_DIM,
        dtype=torch.float32,
    ).reshape(BATCH_SIZE, NUM_FLOW_SAMPLES, CHUNK_SIZE, PADDED_ACTION_DIM)

    def model_forward(images, img_masks, tokens, masks, actions, noise, time):
        captured["noise"] = noise
        captured["time"] = time
        return expected_losses

    monkeypatch.setattr(policy.model, "sample_noise", sample_noise)
    monkeypatch.setattr(policy.model, "sample_time", sample_time)
    monkeypatch.setattr(policy.model, "forward", model_forward)

    batch = {
        image_key: torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
        OBS_LANGUAGE_TOKENS: torch.ones(BATCH_SIZE, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(BATCH_SIZE, 3, dtype=torch.bool),
        ACTION: torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM),
    }
    per_sample_loss, loss_dict = policy(batch, reduction="none")

    truncated_losses = expected_losses[..., :ACTION_DIM]
    expected_per_sample_loss = truncated_losses.mean(dim=(1, 2, 3))
    expected_loss_per_dim = truncated_losses.mean(dim=(0, 1, 2))

    assert captured["noise_shape"] == (
        BATCH_SIZE,
        NUM_FLOW_SAMPLES,
        CHUNK_SIZE,
        PADDED_ACTION_DIM,
    )
    assert captured["time_batch_size"] == BATCH_SIZE * NUM_FLOW_SAMPLES
    assert captured["noise"].shape == captured["noise_shape"]
    assert captured["time"].shape == (BATCH_SIZE, NUM_FLOW_SAMPLES)
    assert torch.unique(captured["time"]).numel() == BATCH_SIZE * NUM_FLOW_SAMPLES
    assert per_sample_loss.shape == (BATCH_SIZE,)
    torch.testing.assert_close(per_sample_loss, expected_per_sample_loss)
    torch.testing.assert_close(torch.tensor(loss_dict["loss_per_dim"]), expected_loss_per_dim)
    assert loss_dict["loss"] == pytest.approx(expected_per_sample_loss.mean().item())


def test_num_flow_samples_does_not_change_inference():
    single_sample_model = PI05Pytorch(make_tiny_config(num_flow_samples=1)).eval()
    multi_sample_model = PI05Pytorch(make_tiny_config(num_flow_samples=NUM_FLOW_SAMPLES)).eval()
    multi_sample_model.load_state_dict(deepcopy(single_sample_model.state_dict()))

    images, img_masks, tokens, masks, _actions, noise, _time = make_model_inputs()
    inference_noise = noise[:, 0]
    with torch.no_grad():
        single_sample_actions = single_sample_model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            noise=inference_noise,
            num_steps=1,
        )
        multi_sample_actions = multi_sample_model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            noise=inference_noise,
            num_steps=1,
        )

    torch.testing.assert_close(single_sample_actions, multi_sample_actions, rtol=0, atol=0)
