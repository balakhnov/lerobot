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

import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from lerobot.datasets import LeRobotDataset, add_features
from lerobot.utils.constants import REWARD


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURE_NAME = REWARD
DEFAULT_NUM_REWARD_SAMPLES = 5
OVERWRITE_OUTPUT_DIR = True


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Add success reward labels to the last samples of every episode in a dataset."
    )
    parser.add_argument(
        "dataset_name",
        help=(
            "Source dataset name. Use either a repo id like HuggingFaceVLA/smol-libero "
            "or a bare name like smol-libero."
        ),
    )
    parser.add_argument("--num-reward-samples", type=int, default=DEFAULT_NUM_REWARD_SAMPLES)
    return parser.parse_args()


def source_repo_id_from_dataset_name(dataset_name: str) -> str:
    return dataset_name


def output_repo_id_from_dataset_name(dataset_name: str) -> str:
    source_repo_id = source_repo_id_from_dataset_name(dataset_name)
    namespace, name = source_repo_id.split("/", maxsplit=1)
    return f"{namespace}/{name}_with_success_reward"


def output_dir_from_dataset_name(dataset_name: str) -> Path:
    source_repo_id = source_repo_id_from_dataset_name(dataset_name)
    name = source_repo_id.split("/", maxsplit=1)[1]
    dir_name = f"lerobot_{name.replace('-', '_')}_with_success_reward"
    return REPO_ROOT / "output" / dir_name


def label_scalar(value) -> float:
    if isinstance(value, list):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def format_label_counts(dataset: LeRobotDataset) -> str:
    labels = dataset.hf_dataset.with_format(None)[FEATURE_NAME]
    counts: dict[int, int] = {}
    for label in labels:
        key = int(label_scalar(label))
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    return ", ".join(
        f"label={label}: {count} ({100 * count / total:.1f}%)" for label, count in sorted(counts.items())
    )


def main():
    args = parse_args()
    source_repo_id = source_repo_id_from_dataset_name(args.dataset_name)
    output_repo_id = output_repo_id_from_dataset_name(args.dataset_name)
    output_dir = output_dir_from_dataset_name(args.dataset_name)

    dataset = LeRobotDataset(source_repo_id)
    if args.num_reward_samples <= 0:
        raise ValueError("--num-reward-samples must be positive.")

    if output_dir.exists():
        if not OVERWRITE_OUTPUT_DIR:
            raise FileExistsError(f"{output_dir} already exists. Remove it or set OVERWRITE_OUTPUT_DIR = True.")
        shutil.rmtree(output_dir)

    episode_lengths = {
        int(episode["episode_index"]): int(episode["length"]) for episode in dataset.meta.episodes
    }

    def compute_reward(row_dict, episode_index, frame_index):
        del row_dict
        episode_length = episode_lengths[int(episode_index)]
        reward_start_index = max(episode_length - args.num_reward_samples, 0)
        return float(int(frame_index) >= reward_start_index)

    dataset_with_reward = add_features(
        dataset,
        features={
            FEATURE_NAME: (
                compute_reward,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    print(f"Created {dataset_with_reward.repo_id} at {dataset_with_reward.root}")
    print(f"Added feature: {FEATURE_NAME}")
    print(f"Labeled the last {args.num_reward_samples} sample(s) in each successful episode as reward")
    print(f"Label counts: {format_label_counts(dataset_with_reward)}")


if __name__ == "__main__":
    main()
