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

"""
Create a side-by-side video for a random dataset episode:
left side is one camera view, right side is the reward classifier's P(reward).

This script is a companion to examples/dataset/custom/train_success_reward_classifier.py
and expects that script's output directory to contain a trained classifier.

Usage:
    uv run python examples/dataset/custom/create_success_reward_video.py

    uv run python examples/dataset/custom/create_success_reward_video.py \
        HuggingFaceVLA/smol-libero \
        --episode 3 \
        --camera-key observation.images.image2
"""

from __future__ import annotations

import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.rewards import RewardModelConfig
from lerobot.datasets import LeRobotDataset
from lerobot.rewards.classifier.modeling_classifier import Classifier
from lerobot.utils.constants import OBS_IMAGE
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_NAME = "HuggingFaceVLA/smol-libero"
PREDICT_BATCH_SIZE = 64
GRAPH_BG = (248, 248, 248)
GRAPH_GRID = (215, 215, 215)
GRAPH_AXIS = (70, 70, 70)
GRAPH_LINE = (80, 80, 80)
GRAPH_REWARD_LINE = (50, 110, 220)
GRAPH_MARKER = (30, 30, 220)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Create a reward-classifier video for one dataset episode.")
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default=DEFAULT_DATASET_NAME,
        help=(
            "Source dataset name. Use the same value passed to "
            "examples/dataset/custom/add_episode_success_reward_feature.py."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Reward dataset repo id. Defaults to <dataset_name>_with_success_reward.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Local reward dataset root. Defaults to "
            "output/lerobot_<dataset-name>_with_success_reward."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Reward classifier directory. Defaults to output/<reward-dataset-root-name>_classifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated videos. Defaults to output/<classifier-dir-name>_videos.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--camera-key", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=PREDICT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, mps, or xpu.",
    )
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


def classifier_output_dir_from_dataset_root(dataset_root: Path) -> Path:
    return REPO_ROOT / "output" / f"{dataset_root.name}_classifier"


def video_output_dir_from_model_dir(model_dir: Path) -> Path:
    return REPO_ROOT / "output" / f"{model_dir.name}_videos"


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return auto_select_torch_device()
    if not is_torch_device_available(requested_device):
        raise ValueError(f"Requested device '{requested_device}' is not available.")
    return torch.device(requested_device)


def load_classifier(model_dir: Path, device: torch.device) -> Classifier:
    config = RewardModelConfig.from_pretrained(model_dir)
    config.device = str(device)
    return Classifier.from_pretrained(model_dir, config=config)


def image_keys_from_model(model: Classifier) -> tuple[str, ...]:
    image_keys = tuple(key for key in model.config.input_features if key.startswith(OBS_IMAGE))
    if not image_keys:
        raise ValueError("Reward classifier config does not contain image input features.")
    return image_keys


def image_size_from_model(model: Classifier, image_keys: tuple[str, ...]) -> tuple[int, int]:
    image_sizes = set()
    for key in image_keys:
        shape = tuple(model.config.input_features[key].shape)
        if len(shape) != 3:
            raise ValueError(f"Expected CHW image shape for '{key}', got {shape}.")
        image_sizes.add((int(shape[-2]), int(shape[-1])))

    if len(image_sizes) != 1:
        raise ValueError(f"All model image inputs must share one size, got {sorted(image_sizes)}.")
    return image_sizes.pop()


def resolve_episode(dataset: LeRobotDataset, requested_episode: int | None, seed: int | None) -> int:
    episode_indices = [int(idx) for idx in dataset.meta.episodes["episode_index"]]
    if requested_episode is not None:
        if requested_episode not in episode_indices:
            raise ValueError(
                f"Episode {requested_episode} not found. Available range: {episode_indices[:3]}..."
            )
        return requested_episode

    rng = random.Random(seed)
    return rng.choice(episode_indices)


def episode_frame_range(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    episode_indices = [int(idx) for idx in dataset.meta.episodes["episode_index"]]
    row_index = episode_indices.index(episode_index)
    return (
        int(dataset.meta.episodes["dataset_from_index"][row_index]),
        int(dataset.meta.episodes["dataset_to_index"][row_index]),
    )


def resolve_camera_key(
    dataset: LeRobotDataset,
    model_image_keys: tuple[str, ...],
    requested_camera_key: str | None,
) -> str:
    camera_keys = tuple(dataset.meta.camera_keys)
    if requested_camera_key is not None:
        if requested_camera_key not in camera_keys:
            raise ValueError(
                f"Camera key '{requested_camera_key}' not found. Available cameras: {camera_keys}"
            )
        return requested_camera_key

    for key in model_image_keys:
        if key in camera_keys:
            return key
    if camera_keys:
        return camera_keys[0]
    raise ValueError("Dataset has no camera keys.")


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] not in (1, 3):
        raise ValueError(f"Expected channel-first image tensor, got shape {tuple(image.shape)}")

    array = image.permute(1, 2, 0).numpy()
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.max(initial=0.0) <= 1.0:
            array *= 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return array


def make_model_batch(
    frame_batch: list[dict[str, torch.Tensor]],
    image_keys: tuple[str, ...],
    image_size: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = {}
    for key in image_keys:
        images = torch.stack([frame[key] for frame in frame_batch]).float().to(device)
        batch[key] = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
    return batch


@torch.no_grad()
def predict_reward_probabilities(
    model: Classifier,
    frames: list[dict[str, torch.Tensor]],
    image_keys: tuple[str, ...],
    image_size: tuple[int, int],
    batch_size: int,
) -> list[float]:
    device = next(model.parameters()).device
    probabilities: list[float] = []
    for start in range(0, len(frames), batch_size):
        batch = make_model_batch(frames[start : start + batch_size], image_keys, image_size, device)
        outputs = model.predict([batch[key] for key in image_keys])
        probabilities.extend(outputs.probabilities.detach().cpu().float().view(-1).tolist())
    return probabilities


def graph_points(probabilities: list[float], width: int, height: int) -> np.ndarray:
    margin_left = 58
    margin_right = 24
    margin_top = 34
    margin_bottom = 46
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, height - margin_top - margin_bottom)

    xs = np.linspace(margin_left, margin_left + plot_width, num=len(probabilities))
    ys = margin_top + (1.0 - np.asarray(probabilities)) * plot_height
    return np.column_stack([xs, ys]).round().astype(np.int32)


def draw_reward_probability_graph(
    probabilities: list[float],
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    graph = np.full((height, width, 3), GRAPH_BG, dtype=np.uint8)
    margin_left = 58
    margin_right = 24
    margin_top = 34
    margin_bottom = 46
    x0 = margin_left
    x1 = width - margin_right
    y0 = height - margin_bottom
    y1 = margin_top

    cv2.rectangle(graph, (x0, y1), (x1, y0), GRAPH_AXIS, 1, cv2.LINE_AA)
    for label, value in (("1.0", 1.0), ("0.5", 0.5), ("0.0", 0.0)):
        y = int(round(y1 + (1.0 - value) * (y0 - y1)))
        cv2.line(graph, (x0, y), (x1, y), GRAPH_GRID, 1, cv2.LINE_AA)
        cv2.putText(graph, label, (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAPH_AXIS, 1, cv2.LINE_AA)

    if probabilities:
        points = graph_points(probabilities, width, height)
        cv2.polylines(graph, [points], isClosed=False, color=GRAPH_LINE, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(
            graph,
            [points[: frame_index + 1]],
            isClosed=False,
            color=GRAPH_REWARD_LINE,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        x, y = points[frame_index]
        cv2.line(graph, (x, y1), (x, y0), (190, 190, 190), 1, cv2.LINE_AA)
        cv2.circle(graph, (x, y), 6, GRAPH_MARKER, -1, cv2.LINE_AA)

    probability = probabilities[frame_index]
    cv2.putText(
        graph,
        f"P(reward) = {probability:.3f}",
        (margin_left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        GRAPH_AXIS,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        graph,
        f"frame {frame_index + 1}/{len(probabilities)}",
        (margin_left, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        GRAPH_AXIS,
        1,
        cv2.LINE_AA,
    )
    return graph


def write_side_by_side_video(
    frames_rgb: list[np.ndarray],
    probabilities: list[float],
    fps: int,
    output_path: Path,
    episode_index: int,
    camera_key: str,
) -> None:
    if not frames_rgb:
        raise ValueError("No frames to write.")
    if len(frames_rgb) != len(probabilities):
        raise ValueError(f"Got {len(frames_rgb)} frames and {len(probabilities)} probabilities.")

    frame_height, frame_width = frames_rgb[0].shape[:2]
    graph_width = frame_width
    output_width = frame_width + graph_width
    output_height = frame_height

    if output_width % 2:
        output_width -= 1
        graph_width -= 1
    if output_height % 2:
        output_height -= 1
        frame_height -= 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    camera_label = camera_key.rsplit(".", maxsplit=1)[-1]
    overlay_text = f"ep {episode_index} | {camera_label}"

    try:
        for idx, frame_rgb in enumerate(frames_rgb):
            frame_rgb = frame_rgb[:frame_height, :frame_width]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame_bgr,
                overlay_text,
                (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                overlay_text,
                (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            graph = draw_reward_probability_graph(probabilities, idx, graph_width, output_height)
            writer.write(np.concatenate([frame_bgr, graph], axis=1))
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    dataset_repo_id = args.dataset_repo_id or output_repo_id_from_dataset_name(args.dataset_name)
    dataset_root = args.dataset_root or output_dir_from_dataset_name(args.dataset_name)
    model_dir = args.model_dir or classifier_output_dir_from_dataset_root(dataset_root)
    output_dir = args.output_dir or video_output_dir_from_model_dir(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Run examples/dataset/custom/train_success_reward_classifier.py first."
        )

    device = resolve_device(args.device)
    model = load_classifier(model_dir, device)
    model.eval()
    model_image_keys = image_keys_from_model(model)
    model_image_size = image_size_from_model(model, model_image_keys)

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    for key in model_image_keys:
        if key not in dataset.features:
            raise ValueError(f"Model expects '{key}', but the dataset only has: {tuple(dataset.features)}")

    episode_index = resolve_episode(dataset, args.episode, args.seed)
    from_idx, to_idx = episode_frame_range(dataset, episode_index)
    camera_key = resolve_camera_key(dataset, model_image_keys, args.camera_key)

    frames_rgb: list[np.ndarray] = []
    model_frames: list[dict[str, torch.Tensor]] = []
    for dataset_index in range(from_idx, to_idx):
        item = dataset[dataset_index]
        frames_rgb.append(tensor_to_rgb_uint8(item[camera_key]))
        model_frames.append({key: item[key] for key in model_image_keys})

    probabilities = predict_reward_probabilities(
        model,
        model_frames,
        model_image_keys,
        model_image_size,
        batch_size=args.batch_size,
    )

    output_path = args.output_path
    if output_path is None:
        output_path = output_dir / f"reward_probability_episode_{episode_index:06d}.mp4"

    write_side_by_side_video(
        frames_rgb,
        probabilities,
        fps=int(dataset.fps),
        output_path=output_path,
        episode_index=episode_index,
        camera_key=camera_key,
    )
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
