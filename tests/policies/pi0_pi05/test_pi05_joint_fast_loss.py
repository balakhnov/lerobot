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

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytest.importorskip("transformers", reason="transformers is required (install lerobot[pi])")

from transformers import DynamicCache  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.vla_utils import make_att_2d_masks, trim_past_key_values  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

BATCH_SIZE = 2
CHUNK_SIZE = 2
ACTION_DIM = 3
IMAGE_SIZE = 28
FAST_TOKENS = 4


def make_tiny_config(*, use_fast_auxiliary: bool = True, num_flow_samples: int = 1) -> PI05Config:
    return PI05Config(
        paligemma_variant="gemma_tiny",
        action_expert_variant="gemma_tiny",
        num_hidden_layers=1,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        max_action_dim=ACTION_DIM,
        image_resolution=(IMAGE_SIZE, IMAGE_SIZE),
        num_flow_samples=num_flow_samples,
        use_fast_auxiliary=use_fast_auxiliary,
        max_action_tokens=FAST_TOKENS,
        fast_loss_weight=0.25,
        device="cpu",
    )


def make_model_inputs(*, num_flow_samples: int = 1):
    generator = torch.Generator().manual_seed(123)
    images = [torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)]
    img_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool)]
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    masks = torch.tensor([[True, True, True], [True, True, False]])
    actions = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM, generator=generator)
    noise = torch.randn(
        BATCH_SIZE,
        num_flow_samples,
        CHUNK_SIZE,
        ACTION_DIM,
        generator=generator,
    )
    time = torch.linspace(0.2, 0.8, BATCH_SIZE * num_flow_samples).reshape(BATCH_SIZE, num_flow_samples)
    fast_tokens = torch.tensor([[11, 12, 13, 0], [21, 22, 0, 0]], dtype=torch.long)
    fast_mask = torch.tensor([[True, True, True, False], [True, True, False, False]], dtype=torch.bool)
    return images, img_masks, tokens, masks, actions, noise, time, fast_tokens, fast_mask


def test_fast_auxiliary_config_validation():
    with pytest.raises(ValueError, match="fast_loss_weight must be non-negative"):
        PI05Config(fast_loss_weight=-1)
    with pytest.raises(ValueError, match="max_action_tokens must be at least 1"):
        PI05Config(max_action_tokens=0)
    with pytest.raises(ValueError, match="incompatible with train_expert_only"):
        PI05Config(use_fast_auxiliary=True, train_expert_only=True)


def test_prepare_fast_inputs_shifts_right_and_preserves_target_mask():
    model = PI05Pytorch(make_tiny_config())
    *_unused, fast_tokens, fast_mask = make_model_inputs()

    fast_inputs, fast_input_mask = model.prepare_fast_inputs(fast_tokens, fast_mask)

    bos_token_id = model.paligemma_with_expert.paligemma.config.text_config.bos_token_id
    expected_inputs = torch.cat(
        [
            torch.full((BATCH_SIZE, 1), bos_token_id, dtype=torch.long),
            fast_tokens[:, :-1],
        ],
        dim=1,
    )
    expected_input_mask = torch.cat([torch.ones(BATCH_SIZE, 1, dtype=torch.bool), fast_mask[:, :-1]], dim=1)
    torch.testing.assert_close(fast_inputs, expected_inputs)
    torch.testing.assert_close(fast_input_mask, expected_input_mask)


def test_fast_prefix_attention_is_causal_and_conditioning_cannot_see_fast_tokens():
    model = PI05Pytorch(make_tiny_config()).eval()
    images, img_masks, tokens, masks, *_rest, fast_tokens, fast_mask = make_model_inputs()
    fast_inputs, fast_input_mask = model.prepare_fast_inputs(fast_tokens, fast_mask)

    _embs, pad_masks, att_masks = model.embed_prefix(
        images,
        img_masks,
        tokens,
        masks,
        fast_inputs=fast_inputs,
        fast_input_masks=fast_input_mask,
    )
    structural_mask = make_att_2d_masks(pad_masks, att_masks)
    fast_start = structural_mask.shape[-1] - FAST_TOKENS

    assert not structural_mask[:, :fast_start, fast_start:].any()
    fast_to_conditioning = structural_mask[:, fast_start:, :fast_start].any(dim=-1)
    assert fast_to_conditioning[fast_input_mask].all()
    for fast_index in range(FAST_TOKENS):
        query = fast_start + fast_index
        valid_query = fast_input_mask[:, fast_index]
        assert structural_mask[valid_query, query, fast_start : query + 1].any(dim=-1).all()
        if query + 1 < structural_mask.shape[-1]:
            assert not structural_mask[:, query, query + 1 :].any()


def test_fast_loss_matches_manual_masked_cross_entropy():
    model = PI05Pytorch(make_tiny_config()).eval()
    *_unused, fast_tokens, fast_mask = make_model_inputs()
    hidden_size = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    hidden = torch.randn(BATCH_SIZE, FAST_TOKENS, hidden_size, generator=torch.Generator().manual_seed(7))

    fast_loss, fast_accuracy = model.compute_fast_loss(hidden, fast_tokens, fast_mask)

    valid = fast_mask
    logits = F.linear(
        hidden[valid],
        model.paligemma_with_expert.paligemma.lm_head.weight,
    ).float()
    per_token_loss = F.cross_entropy(logits, fast_tokens[valid], reduction="none")
    example_indices = valid.nonzero(as_tuple=False)[:, 0]
    expected_loss = torch.zeros(BATCH_SIZE).index_add(0, example_indices, per_token_loss)
    expected_loss = expected_loss / valid.sum(dim=1)
    expected_accuracy = torch.zeros(BATCH_SIZE).index_add(
        0,
        example_indices,
        (logits.argmax(dim=-1) == fast_tokens[valid]).float(),
    )
    expected_accuracy = expected_accuracy / valid.sum(dim=1)

    torch.testing.assert_close(fast_loss, expected_loss)
    torch.testing.assert_close(fast_accuracy, expected_accuracy)


def test_trim_past_key_values_removes_tail_and_preserves_gradients():
    cache = DynamicCache()
    keys = torch.arange(10.0, requires_grad=True).reshape(1, 1, 5, 2)
    values = torch.arange(10.0, 20.0, requires_grad=True).reshape(1, 1, 5, 2)
    keys.retain_grad()
    values.retain_grad()
    cache.update(keys, values, 0)

    trimmed = trim_past_key_values(cache, tail_tokens=2)
    trimmed_keys, trimmed_values, _ = next(iter(trimmed))

    torch.testing.assert_close(trimmed_keys, keys[:, :, :3])
    torch.testing.assert_close(trimmed_values, values[:, :, :3])
    (trimmed_keys.sum() + trimmed_values.sum()).backward()
    torch.testing.assert_close(keys.grad[:, :, :3], torch.ones_like(keys[:, :, :3]))
    torch.testing.assert_close(values.grad[:, :, :3], torch.ones_like(values[:, :, :3]))
    torch.testing.assert_close(keys.grad[:, :, 3:], torch.zeros_like(keys[:, :, 3:]))
    torch.testing.assert_close(values.grad[:, :, 3:], torch.zeros_like(values[:, :, 3:]))


def test_teacher_forced_fast_tokens_do_not_change_flow_loss():
    model = PI05Pytorch(make_tiny_config()).eval()
    inputs = make_model_inputs()
    images, img_masks, tokens, masks, actions, noise, time, fast_tokens, fast_mask = inputs
    alternate_fast_tokens = fast_tokens.clone()
    alternate_fast_tokens[fast_mask] += 100

    with torch.no_grad():
        flow_loss, fast_loss, _ = model(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            fast_tokens,
            fast_mask,
        )
        alternate_flow_loss, alternate_fast_loss, _ = model(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            alternate_fast_tokens,
            fast_mask,
        )

    torch.testing.assert_close(flow_loss, alternate_flow_loss, rtol=0, atol=0)
    assert not torch.equal(fast_loss, alternate_fast_loss)


def test_flow_loss_gradients_reach_vlm_without_knowledge_insulation():
    model = PI05Pytorch(make_tiny_config())
    inputs = make_model_inputs()
    flow_loss, _fast_loss, _fast_accuracy = model(*inputs)

    flow_loss.mean().backward()

    vlm_k_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.k_proj.weight
    assert vlm_k_proj.grad is not None
    assert bool((vlm_k_proj.grad != 0).any())


def test_joint_policy_combines_per_example_losses_once_across_flow_samples(monkeypatch):
    num_flow_samples = 3
    config = make_tiny_config(num_flow_samples=num_flow_samples)
    image_key = f"{OBS_IMAGES}.camera"
    config.input_features = {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    policy = PI05Policy(config)
    flow_losses = torch.arange(
        BATCH_SIZE * num_flow_samples * CHUNK_SIZE * ACTION_DIM,
        dtype=torch.float32,
    ).reshape(BATCH_SIZE, num_flow_samples, CHUNK_SIZE, ACTION_DIM)
    fast_losses = torch.tensor([2.0, 4.0])
    fast_accuracy = torch.tensor([0.5, 1.0])

    monkeypatch.setattr(
        policy.model,
        "sample_noise",
        lambda shape, device: torch.zeros(shape, device=device),
    )
    monkeypatch.setattr(
        policy.model,
        "sample_time",
        lambda batch_size, device: torch.ones(batch_size, device=device),
    )

    def model_forward(
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        fast_action_tokens,
        fast_action_masks,
    ):
        assert fast_action_tokens.shape == (BATCH_SIZE, FAST_TOKENS)
        assert fast_action_masks.shape == (BATCH_SIZE, FAST_TOKENS)
        return flow_losses, fast_losses, fast_accuracy

    monkeypatch.setattr(policy.model, "forward", model_forward)
    batch = {
        image_key: torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
        OBS_LANGUAGE_TOKENS: torch.ones(BATCH_SIZE, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(BATCH_SIZE, 3, dtype=torch.bool),
        ACTION_TOKENS: torch.ones(BATCH_SIZE, FAST_TOKENS, dtype=torch.long),
        ACTION_TOKEN_MASK: torch.ones(BATCH_SIZE, FAST_TOKENS, dtype=torch.bool),
        ACTION: torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM),
    }

    per_example_loss, metrics = policy(batch, reduction="none")
    expected_flow_loss = flow_losses.mean(dim=(1, 2, 3))
    expected_total = expected_flow_loss + config.fast_loss_weight * fast_losses

    torch.testing.assert_close(per_example_loss, expected_total)
    assert metrics["flow_loss"] == pytest.approx(expected_flow_loss.mean().item())
    assert metrics["fast_loss"] == pytest.approx(fast_losses.mean().item())
    assert metrics["fast_accuracy"] == pytest.approx(fast_accuracy.mean().item())
    assert metrics["loss"] == pytest.approx(expected_total.mean().item())


def test_flow_only_model_contract_is_unchanged():
    config = make_tiny_config(use_fast_auxiliary=False)
    model = PI05Pytorch(config).eval()
    images, img_masks, tokens, masks, actions, noise, time, *_fast = make_model_inputs()

    with torch.no_grad():
        losses = model(images, img_masks, tokens, masks, actions, noise, time)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (BATCH_SIZE, 1, CHUNK_SIZE, ACTION_DIM)
