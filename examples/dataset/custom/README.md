# Custom Dataset Reward Examples

This directory contains small, directly runnable examples for creating and using
a visual success reward classifier on LeRobot datasets.

The examples assume that the selected episodes are successful demonstrations.
They label the last few frames of each episode as positive reward and all earlier
frames as zero reward.

## Scripts

- `add_episode_success_reward_feature.py` adds `next.reward` labels to an existing dataset.
- `train_success_reward_classifier.py` trains a `lerobot/resnet10` reward classifier from those labels.
- `create_success_reward_video.py` renders a side-by-side episode video with the classifier's `P(reward)` curve.
- `create_high_quality_folding_success_reward_subset.py` creates a smaller labeled subset from `lerobot/high_quality_folding`.

## Basic Workflow

Create a reward-labeled dataset:

```bash
uv run python examples/dataset/custom/add_episode_success_reward_feature.py HuggingFaceVLA/smol-libero
```

By default this writes:

```text
output/lerobot_smol_libero_with_success_reward
```

Train a reward classifier:

```bash
uv run python examples/dataset/custom/train_success_reward_classifier.py HuggingFaceVLA/smol-libero
```

By default this saves:

```text
output/lerobot_smol_libero_with_success_reward_classifier
```

Create a reward probability video:

```bash
uv run python examples/dataset/custom/create_success_reward_video.py \
    HuggingFaceVLA/smol-libero \
    --model-dir output/lerobot_smol_libero_with_success_reward_classifier \
    --episode 0
```

The video contains the selected camera view on the left and the classifier's
reward probability over time on the right.

## Labeling Options

Use `--num-reward-samples` to control how many final frames in each episode are
labeled as reward:

```bash
uv run python examples/dataset/custom/add_episode_success_reward_feature.py \
    HuggingFaceVLA/smol-libero \
    --num-reward-samples 10
```

The output dataset repo id is derived from the source repo id by appending
`_with_success_reward`.

## Training Notes

The trainer uses constants near the top of
`train_success_reward_classifier.py` for the main knobs:

- `IMAGE_SIZE`
- `BATCH_SIZE`
- `STEPS`
- `LEARNING_RATE`
- `VAL_FRACTION`
- `LABEL_WEIGHTS`

Train and validation splits are made by episode, so frames from the same episode
do not appear in both splits.

To train on a custom local reward dataset, pass both the repo id and root:

```bash
uv run python examples/dataset/custom/train_success_reward_classifier.py \
    my-org/my-dataset \
    --dataset-repo-id my-org/my-dataset_with_success_reward \
    --dataset-root output/lerobot_my_dataset_with_success_reward
```

## High-Quality Folding Subset

Create a 100-episode labeled subset from `lerobot/high_quality_folding`:

```bash
uv run python examples/dataset/custom/create_high_quality_folding_success_reward_subset.py
```

By default this writes:

```text
output/lerobot_high_quality_folding_100_success_reward
```

You can adjust the subset size and label window:

```bash
uv run python examples/dataset/custom/create_high_quality_folding_success_reward_subset.py \
    --num-episodes 50 \
    --num-reward-samples 10
```

Then train with:

```bash
uv run python examples/dataset/custom/train_success_reward_classifier.py \
    lerobot/high_quality_folding \
    --dataset-repo-id lerobot/high_quality_folding_100_success_reward \
    --dataset-root output/lerobot_high_quality_folding_100_success_reward
```

## Video Options

Pick the episode, camera, output path, and device:

```bash
uv run python examples/dataset/custom/create_success_reward_video.py \
    HuggingFaceVLA/smol-libero \
    --model-dir output/lerobot_smol_libero_with_success_reward_classifier \
    --episode 3 \
    --camera-key observation.images.image2 \
    --output-path output/reward_probability_episode_000003.mp4 \
    --device auto
```
