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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("transformers", reason="transformers is required (install lerobot[pi])")

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import DEVICE, require_hf_token  # noqa: E402

DATASET_REPO_ID = "lerobot/pusht_image"
TRAIN_STEPS = 200


@require_hf_token
def test_pi05_overfits_one_dataset_batch():
    set_seed(42)

    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    config = PI05Config(
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        num_hidden_layers=1,
        chunk_size=8,
        n_action_steps=8,
        num_inference_steps=1,
        max_action_dim=metadata.features[ACTION]["shape"][0],
        image_resolution=(28, 28),
        train_expert_only=True,
        device=DEVICE,
    )
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        episodes=[0],
        delta_timestamps=resolve_delta_timestamps(config, metadata),
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)))
    policy = make_policy(config, ds_meta=dataset.meta)
    dataset_stats = deepcopy(dataset.meta.stats)
    for key in (OBS_STATE, ACTION):
        dataset_stats[key].setdefault("q01", dataset_stats[key]["min"])
        dataset_stats[key].setdefault("q99", dataset_stats[key]["max"])
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset_stats)

    assert len(policy.model.paligemma_with_expert.paligemma.model.language_model.layers) == 1
    assert len(policy.model.paligemma_with_expert.gemma_expert.model.layers) == 1
    assert len(policy.model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers) == 1

    target_actions = batch[ACTION].clone()
    batch = preprocessor(batch)

    # At t=1 with zero noise, one Euler step maps the learned velocity directly
    # back to the target action, making the overfit assertion deterministic.
    policy.model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    policy.model.sample_time = lambda batch_size, device: torch.ones(batch_size, device=device)

    optimizer = torch.optim.Adam(
        (parameter for parameter in policy.parameters() if parameter.requires_grad), lr=1e-4
    )
    policy.train()
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad()
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()

    normalized_actions = policy.predict_action_chunk(batch)
    torch.testing.assert_close(normalized_actions, batch[ACTION], rtol=0, atol=5e-3)

    generated_actions = postprocessor(normalized_actions)
    torch.testing.assert_close(generated_actions, target_actions, rtol=5e-3, atol=0.1)
