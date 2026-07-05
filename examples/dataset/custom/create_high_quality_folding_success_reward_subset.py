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

import pandas as pd
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _copy_and_reindex_episodes_metadata, _write_parquet
from lerobot.utils.constants import REWARD


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_REPO_ID = "lerobot/high_quality_folding"
DEFAULT_SOURCE_CACHE_DIR = REPO_ROOT / "output/high_quality_folding_selected_cache"
DEFAULT_OUTPUT_REPO_ID = "lerobot/high_quality_folding_100_success_reward"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/lerobot_high_quality_folding_100_success_reward"
DEFAULT_NUM_EPISODES = 100
DEFAULT_NUM_REWARD_SAMPLES = 5
REWARD_FEATURE_INFO = {"dtype": "float32", "shape": (1,), "names": None}


def label_scalar(value) -> float:
    if isinstance(value, list):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def format_label_counts(dataset: LeRobotDataset) -> str:
    labels = dataset.hf_dataset.with_format(None)[REWARD]
    counts: dict[int, int] = {}
    for label in labels:
        key = int(label_scalar(label))
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    return ", ".join(
        f"label={label}: {count} ({100 * count / total:.1f}%)" for label, count in sorted(counts.items())
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Create a 100-episode high_quality_folding subset with success reward labels. "
            "Default command: uv run python examples/dataset/custom/create_high_quality_folding_success_reward_subset.py"
        )
    )
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE_DIR)
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--num-reward-samples", type=int, default=DEFAULT_NUM_REWARD_SAMPLES)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the output directory already exists instead of replacing it.",
    )
    return parser.parse_args()


def ensure_clean_output_dir(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Remove it or omit --no-overwrite.")
    shutil.rmtree(path)


def link_file(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    dst_path.hardlink_to(src_path)


def copy_and_reindex_data_with_reward(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    num_reward_samples: int,
) -> dict[int, dict]:
    file_to_episodes: dict[Path, set[int]] = {}
    for old_idx in episode_mapping:
        file_path = src_dataset.meta.get_data_file_path(old_idx)
        file_to_episodes.setdefault(file_path, set()).add(old_idx)

    all_task_indices = set()
    for src_path in file_to_episodes:
        df = pd.read_parquet(src_dataset.root / src_path, columns=["episode_index", "task_index"])
        mask = df["episode_index"].isin(list(episode_mapping))
        all_task_indices.update(df[mask]["task_index"].unique().tolist())

    tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in all_task_indices]
    dst_meta.save_episode_tasks(list(set(tasks)))

    task_mapping = {}
    for old_task_idx in range(len(src_dataset.meta.tasks)):
        task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    episode_lengths = {
        episode_mapping[old_idx]: int(src_dataset.meta.episodes[old_idx]["length"])
        for old_idx in episode_mapping
    }

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    for src_path in tqdm(sorted(file_to_episodes), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)
        episodes_to_keep = file_to_episodes[src_path]
        df = df[df["episode_index"].isin(list(episode_mapping))].copy().reset_index(drop=True)
        if len(df) == 0:
            continue

        df["episode_index"] = df["episode_index"].replace(episode_mapping)
        df["index"] = range(global_index, global_index + len(df))
        df["task_index"] = df["task_index"].replace(task_mapping)

        episode_lengths_for_rows = df["episode_index"].map(episode_lengths)
        reward_start_indices = (episode_lengths_for_rows - num_reward_samples).clip(lower=0)
        df[REWARD] = (df["frame_index"] >= reward_start_indices).astype("float32")

        first_ep_old_idx = min(episodes_to_keep)
        src_ep = src_dataset.meta.episodes[first_ep_old_idx]
        chunk_idx = src_ep["data/chunk_index"]
        file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / dst_meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, dst_path, dst_meta)

        for ep_old_idx in episodes_to_keep:
            ep_new_idx = episode_mapping[ep_old_idx]
            ep_df = df[df["episode_index"] == ep_new_idx]
            episode_data_metadata[ep_new_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(df)

    return episode_data_metadata


def link_videos_and_collect_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}
    linked_paths: set[Path] = set()

    for video_key in src_dataset.meta.video_keys:
        for old_idx in tqdm(sorted(episode_mapping), desc=f"Linking {video_key} videos"):
            src_ep = src_dataset.meta.episodes[old_idx]
            new_idx = episode_mapping[old_idx]

            src_rel_path = src_dataset.meta.get_video_file_path(old_idx, video_key)
            if src_rel_path not in linked_paths:
                link_file(src_dataset.root / src_rel_path, dst_meta.root / src_rel_path)
                linked_paths.add(src_rel_path)

            episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_ep[
                f"videos/{video_key}/chunk_index"
            ]
            episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_ep[
                f"videos/{video_key}/file_index"
            ]
            episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                f"videos/{video_key}/from_timestamp"
            ]
            episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                f"videos/{video_key}/to_timestamp"
            ]

    return episodes_video_metadata


def create_subset_with_reward(
    src_dataset: LeRobotDataset,
    selected_episodes: list[int],
    output_repo_id: str,
    output_dir: Path,
    num_reward_samples: int,
) -> LeRobotDataset:
    features = {**src_dataset.meta.features, REWARD: REWARD_FEATURE_INFO}
    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=src_dataset.meta.fps,
        features=features,
        robot_type=src_dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(src_dataset.meta.video_keys) > 0,
        chunks_size=src_dataset.meta.chunks_size,
        data_files_size_in_mb=src_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=src_dataset.meta.video_files_size_in_mb,
    )
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_episodes)}

    data_metadata = copy_and_reindex_data_with_reward(
        src_dataset,
        dst_meta,
        episode_mapping,
        num_reward_samples,
    )
    video_metadata = link_videos_and_collect_metadata(src_dataset, dst_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(src_dataset, dst_meta, episode_mapping, data_metadata, video_metadata)

    return LeRobotDataset(output_repo_id, root=output_dir)


def main():
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError(f"--num-episodes must be positive, got {args.num_episodes}.")
    if args.num_reward_samples <= 0:
        raise ValueError(f"--num-reward-samples must be positive, got {args.num_reward_samples}.")

    selected_episodes = list(range(args.num_episodes))

    ensure_clean_output_dir(args.output_dir, overwrite=not args.no_overwrite)

    source_dataset = LeRobotDataset(
        args.source_repo_id,
        root=args.source_cache_dir,
        episodes=selected_episodes,
        download_videos=True,
    )
    print(
        f"Downloaded/loaded selected source files for {source_dataset.num_episodes} episodes "
        f"and {source_dataset.num_frames} frames."
    )

    dataset_with_reward = create_subset_with_reward(
        source_dataset,
        selected_episodes,
        args.output_repo_id,
        args.output_dir,
        args.num_reward_samples,
    )

    print(f"Created {dataset_with_reward.repo_id} at {dataset_with_reward.root}")
    print(f"Episodes: {dataset_with_reward.meta.total_episodes}")
    print(f"Frames: {dataset_with_reward.meta.total_frames}")
    print(f"Added feature: {REWARD}")
    print(f"Labeled the last {args.num_reward_samples} sample(s) in each episode as reward=1")
    print(f"Label counts: {format_label_counts(dataset_with_reward)}")


if __name__ == "__main__":
    main()
