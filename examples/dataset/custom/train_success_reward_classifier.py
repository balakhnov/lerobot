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

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig
from lerobot.rewards.classifier.modeling_classifier import Classifier
from lerobot.utils.constants import REWARD

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_NAME = "HuggingFaceVLA/smol-libero"

IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (3, *IMAGE_SIZE)
MODEL_NAME = "lerobot/resnet10"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
NUM_WORKERS = 0
STEPS = 200
LEARNING_RATE = 1e-4
LOG_FREQ = 1
VAL_FREQ = 100
VAL_FRACTION = 0.1
SPLIT_SEED = 0
LABEL_WEIGHTS = {0: 1.0, 1: 10.0}


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a classifier on a dataset with success reward labels.")
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


def label_key(label: float) -> int | float:
    if float(label).is_integer():
        return int(label)
    return round(label, 4)


def split_episode_indices(
    total_episodes: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if total_episodes < 2:
        raise ValueError("Need at least 2 episodes to create disjoint train and validation splits.")
    if not 0 < val_fraction < 1:
        raise ValueError(f"VAL_FRACTION must be between 0 and 1, got {val_fraction}.")

    generator = torch.Generator().manual_seed(seed)
    episode_indices = torch.randperm(total_episodes, generator=generator).tolist()
    num_val_episodes = max(1, round(total_episodes * val_fraction))
    num_val_episodes = min(num_val_episodes, total_episodes - 1)

    val_episodes = sorted(episode_indices[:num_val_episodes])
    train_episodes = sorted(episode_indices[num_val_episodes:])
    if set(train_episodes) & set(val_episodes):
        raise RuntimeError("Episode split leaked episodes across train and validation sets.")
    return train_episodes, val_episodes


def compute_label_counts(dataset: LeRobotDataset) -> dict[int | float, int]:
    counts: dict[int | float, int] = {}
    labels = dataset.hf_dataset.with_format(None)[REWARD]
    for label in labels:
        key = label_key(torch.as_tensor(label).flatten()[0].item())
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def format_label_counts(label_counts: dict[int | float, int]) -> str:
    total = sum(label_counts.values())
    return ", ".join(
        f"label={label}: {count} ({100 * count / total:.1f}%)"
        for label, count in label_counts.items()
    )


def format_label_weights(label_weights: dict[int | float, float]) -> str:
    return ", ".join(
        f"label={label}: weight={weight:g}" for label, weight in sorted(label_weights.items())
    )


def image_keys_from_metadata(metadata: LeRobotDatasetMetadata) -> tuple[str, ...]:
    image_keys = tuple(metadata.camera_keys)
    if not image_keys:
        raise ValueError("Dataset metadata does not contain any image or video camera features.")
    return image_keys


def print_dataset_info(
    dataset_repo_id: str,
    metadata: LeRobotDatasetMetadata,
    train_dataset: LeRobotDataset,
    val_dataset: LeRobotDataset,
    train_episodes: list[int],
    val_episodes: list[int],
    image_keys: tuple[str, ...],
) -> None:
    print(f"Dataset repo_id={dataset_repo_id}")
    print(f"Dataset root={metadata.root}")
    print(
        f"Dataset total: {metadata.total_episodes} episodes, "
        f"{metadata.total_frames} frames, fps={metadata.fps}"
    )
    print(f"Image keys: {', '.join(image_keys)}")
    print(
        f"Split seed={SPLIT_SEED}, val_fraction={VAL_FRACTION}: "
        f"{len(train_episodes)} train episodes ({len(train_dataset)} frames) and "
        f"{len(val_episodes)} validation episodes ({len(val_dataset)} frames)."
    )
    print(f"Train episodes: {train_episodes}")
    print(f"Validation episodes: {val_episodes}")
    print(f"Train label counts: {format_label_counts(compute_label_counts(train_dataset))}")
    print(f"Validation label counts: {format_label_counts(compute_label_counts(val_dataset))}")
    print(f"Label weights: {format_label_weights(LABEL_WEIGHTS)}")


def prepare_batch(
    batch: dict[str, torch.Tensor],
    image_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    batch = {
        key: value.to(DEVICE, non_blocking=True)
        for key, value in batch.items()
        if key in image_keys or key == REWARD
    }
    for key in image_keys:
        batch[key] = F.interpolate(
            batch[key].float(),
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )
    batch[REWARD] = batch[REWARD].float().view(-1)
    return batch


def compute_batch_predictions_and_loss(
    model: Classifier,
    batch: dict[str, Tensor],
    label_weights: dict[int | float, float],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    images, labels = model.extract_images_and_labels(batch)
    outputs = model.predict(images)

    if model.config.num_classes == 2:
        unweighted_losses = F.binary_cross_entropy_with_logits(outputs.logits, labels, reduction="none")
        predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
    else:
        unweighted_losses = F.cross_entropy(outputs.logits, labels.long(), reduction="none")
        predictions = torch.argmax(outputs.logits, dim=1)

    sample_weights = torch.tensor(
        [label_weights.get(label_key(label.item()), 1.0) for label in labels],
        dtype=unweighted_losses.dtype,
        device=unweighted_losses.device,
    )
    weighted_losses = unweighted_losses * sample_weights

    return predictions, weighted_losses, labels, sample_weights


def format_eval_label_stats(label_stats: dict[int | float, dict[str, float | int]]) -> str:
    parts = []
    for label, stats in sorted(label_stats.items()):
        parts.append(
            f"label={label}: loss={stats['loss']:.4f} "
            f"weight={stats['weight']:.2f} "
            f"accuracy={stats['accuracy']:.2f} "
            f"correct={stats['correct']}/{stats['total']}"
        )
    return "; ".join(parts)


@torch.no_grad()
def validate(
    model: Classifier,
    dataloader: DataLoader,
    image_keys: tuple[str, ...],
) -> dict[str, float | int | dict]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    label_loss_sums: dict[int | float, float] = {}
    label_weight_sums: dict[int | float, float] = {}
    label_correct: dict[int | float, int] = {}
    label_totals: dict[int | float, int] = {}
    for batch in dataloader:
        batch = prepare_batch(batch, image_keys)
        predictions, losses, labels, sample_weights = compute_batch_predictions_and_loss(
            model,
            batch,
            LABEL_WEIGHTS,
        )
        correct = predictions == labels

        total_loss += losses.sum().item()
        total_correct += correct.sum().item()
        total_samples += labels.size(0)

        for label, loss, sample_weight, is_correct in zip(
            labels,
            losses,
            sample_weights,
            correct,
            strict=True,
        ):
            key = label_key(label.item())
            label_loss_sums[key] = label_loss_sums.get(key, 0.0) + loss.item()
            label_weight_sums[key] = label_weight_sums.get(key, 0.0) + sample_weight.item()
            label_correct[key] = label_correct.get(key, 0) + int(is_correct.item())
            label_totals[key] = label_totals.get(key, 0) + 1

    model.train(was_training)
    if total_samples == 0:
        raise ValueError("Validation dataloader produced no samples.")

    label_metrics = {}
    for label, total in label_totals.items():
        correct = label_correct[label]
        label_metrics[label] = {
            "loss": label_loss_sums[label] / total,
            "weight": label_weight_sums[label] / total,
            "accuracy": 100 * correct / total,
            "correct": correct,
            "total": total,
        }

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100 * total_correct / total_samples,
        "correct": total_correct,
        "total": total_samples,
        "by_label": label_metrics,
    }


def main():
    args = parse_args()
    dataset_repo_id = args.dataset_repo_id or output_repo_id_from_dataset_name(args.dataset_name)
    dataset_root = args.dataset_root or output_dir_from_dataset_name(args.dataset_name)
    output_dir = classifier_output_dir_from_dataset_root(dataset_root)

    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    image_keys = image_keys_from_metadata(metadata)
    train_episodes, val_episodes = split_episode_indices(
        metadata.total_episodes,
        VAL_FRACTION,
        SPLIT_SEED,
    )
    train_dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=train_episodes)
    val_dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=val_episodes)

    if len(train_dataset) < BATCH_SIZE:
        raise ValueError(
            f"Training split has {len(train_dataset)} frames, fewer than BATCH_SIZE={BATCH_SIZE}; "
            "reduce BATCH_SIZE or VAL_FRACTION."
        )
    print_dataset_info(
        dataset_repo_id,
        metadata,
        train_dataset,
        val_dataset,
        train_episodes,
        val_episodes,
        image_keys,
    )
    print(f"Classifier output dir: {output_dir}")

    cfg = RewardClassifierConfig(
        input_features={
            key: PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE) for key in image_keys
        },
        output_features={REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,))},
        model_name=MODEL_NAME,
        model_type="cnn",
        num_cameras=len(image_keys),
        device=DEVICE,
        learning_rate=LEARNING_RATE,
    )

    model = Classifier(cfg).to(DEVICE)
    model.train()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )
    optimizer = torch.optim.Adam(model.get_optim_params(), lr=LEARNING_RATE)

    step = 0
    while step < STEPS:
        for batch in train_dataloader:
            batch = prepare_batch(batch, image_keys)
            predictions, losses, labels, _ = compute_batch_predictions_and_loss(
                model,
                batch,
                LABEL_WEIGHTS,
            )
            loss = losses.mean()
            correct = (predictions == labels).sum().item()
            total = labels.size(0)
            metrics = {
                "accuracy": 100 * correct / total,
                "correct": correct,
                "total": total,
            }

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step % LOG_FREQ == 0 or step == 1:
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"accuracy={metrics['accuracy']:.2f} "
                    f"correct={metrics['correct']}/{metrics['total']}"
                )

            if step % VAL_FREQ == 0 or step == STEPS:
                val_metrics = validate(model, val_dataloader, image_keys)
                print(
                    f"validation step={step} loss={val_metrics['loss']:.4f} "
                    f"accuracy={val_metrics['accuracy']:.2f} "
                    f"correct={val_metrics['correct']}/{val_metrics['total']}"
                )
                print(f"validation by label: {format_eval_label_stats(val_metrics['by_label'])}")

            if step >= STEPS:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Saved reward classifier to {output_dir}")


if __name__ == "__main__":
    main()
