# Full fine-tuning of the released SO-101 checkpoint

Use `leyland_700_fsdp_h100_lerobot.json` in this checkout to fine-tune
[`black-forest-labs/flux-3-action-so101`](https://huggingface.co/black-forest-labs/flux-3-action-so101)
with `lerobot-train` on one node with eight H100 80 GB GPUs. The transformer
and its existing action/history heads train without LoRA. The video VAE and
text encoder remain frozen, as in the Flux3 training implementation.

`policy.path` loads the checkpoint's model config, weights and saved processor
pipelines. It preserves its history contract, camera layout, action representation,
normalization and pinned shared-encoder references. No generic-base export,
fresh-head initialization or dataset conversion is needed for a LeRobot v3 dataset.
The standalone `leyland_700_fsdp_h100.json` is a separate generic-base recipe;
it has not been changed to match this pretrained-policy workflow.

## Dataset settings

Replace `dataset.repo_id` and `dataset.root` with your dataset. For a Hub-only
dataset, set `dataset.root` to null. Use six measured-state and six absolute
command channels in matching joint order/units, gripper last, at 30 Hz.
The saved processors turn five joint commands into consecutive command deltas;
the gripper stays absolute. Do not convert your stored actions to deltas first.

The checkpoint expects separate cameras named `observation.images.scene` and
`observation.images.wrist`, scene first. If your names differ, add a top-level
`rename_map`, for example **only if** `top` is the fixed scene camera and
`gripper_cam` is the wrist-mounted camera:

```json
"rename_map": {
  "observation.images.top": "observation.images.scene",
  "observation.images.gripper_cam": "observation.images.wrist"
}
```

Do not change the checkpoint's camera order or pre-tile your videos. Its
256 x 512 canvas is assembled from the two separate views. It uses eight
observed ticks, two visual snapshots, past-command conditioning and 42 future
actions; inference executes 32 before replanning. The saved normalization is
retained, not recomputed from your 700 demonstrations. Dataset v3 describes
storage, not action units/calibration: ensure these remain compatible with the
checkpoint. See the [SO-101 guide](https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md).

## Launch on eight GPUs

Install the `training` and `flux3` extras and a compatible NATTEN wheel as
described in the local [Flux3 guide](../../docs/source/flux3.mdx). Ensure access
to the SO-101 Hub checkpoint and its shared base encoders.

```bash
uv run torchrun --standalone --nproc_per_node=8 --no-python \
  .venv/bin/lerobot-train \
  --config_path=examples/flux3/leyland_700_fsdp_h100_lerobot.json
```

This launches `lerobot-train` once per GPU. FSDP2 uses `dp_replicate=1`,
`dp_shard=8` and wraps `ModeBlock` and `SingleStreamBlock` modules. Plain
`lerobot-train` without a distributed launcher cannot satisfy this topology.

## Batch and schedule

- Global batch: 8 GPUs x 4 samples/GPU x 16 accumulation = **512**.
- Budget: 160,000 microsteps = 10,000 optimizer updates.
- Backbone freeze: **0**. Both pretrained parameter groups participate from
  the first update; no separate head-only training phase is used.
- Backbone and head warmup: **3,200 microsteps = 200 optimizer updates** each.
- Save and offline-validation interval: 8,000 microsteps = 500 updates.

The earlier 1,000-update freeze was for fresh heads and is removed here.
Warmup is a learning-rate ramp, not a frozen phase. A short 200-update ramp is
a starting heuristic for full-model adaptation, not a requirement or a tested
BFL recipe for this dataset. The trunk's scheduler starts at zero on the first
microbatch, then ramps; with accumulation 16 the first optimizer update already
has a nonzero trunk learning rate. To omit warmup entirely, set both
`scheduler_warmup_steps` and `scheduler_warmup_steps_heads` to zero.

All durations count microsteps in this checkout; the scheduler advances every
microbatch. Keep intervals and phase lengths divisible by accumulation.
FP32 master parameters (`policy.dtype=float32`) use BF16 mixed precision.
Both the backbone and pretrained heads use a peak learning rate of 5e-5
(`optimizer_lr=0.00005`, `optimizer_lr_heads_multiplier=1.0`). This replaces
the original 1.3e-4 / 6.5e-4 rates with a more conservative continuation
starting point, not a validated optimum for your dataset. Use validation and
robot rollouts to tune it.
The policy's own gradient checkpointing is enabled; leave the accelerator's
separate activation-checkpointing placeholder at its default.

## Implementation limits

EMA is disabled because this trainer rejects EMA under FSDP. No unsupported
wrist-camera-dropout, VAE microbatching or episode-visit fields are added.
LeRobot uses its standard AdamW preset and Accelerate FSDP mixed precision;
it does not reproduce BFL's standalone fused optimizer or separate FP32
timestep-embedding FSDP group. This is not an identical reproduction of the
standalone recipe.

The last 20% of episodes per task are held out, with at most 2,048 samples per
offline validation. Environment evaluation is disabled during sharded training.
W&B is enabled in `pickplace`. Checkpoints contain full safetensors and DCP
model shards, with sharded optimizer state; plan disk space because this
checkout has no automatic checkpoint-retention setting. The 10,000-update
budget is inherited from the reference, not an estimate of what 700 episodes
need. Select checkpoints using validation and safety-limited robot rollouts.

## Smoke run

Use a separate output directory for a 20-update test:

```bash
uv run torchrun --standalone --nproc_per_node=8 --no-python \
  .venv/bin/lerobot-train \
  --config_path=examples/flux3/leyland_700_fsdp_h100_lerobot.json \
  --steps=320 --save_freq=320 --eval_steps=0 --log_freq=16 \
  --policy.scheduler_warmup_steps=80 \
  --policy.scheduler_warmup_steps_heads=80 \
  --output_dir=/data/outputs/flux3_leyland_700_fsdp_lerobot_smoke
```

GPU capacity, distributed forward/backward, Hub checkpoint loading and
checkpoint save/load still need validation on the H100 machine. If batch 4
exhausts memory, use batch 1 with accumulation 64 and multiply all
microstep-based durations/phases/intervals by four to preserve global batch
512 and the update budget.

## Resume

```bash
uv run torchrun --standalone --nproc_per_node=8 --no-python \
  .venv/bin/lerobot-train \
  --config_path=/data/outputs/flux3_leyland_700_fsdp_lerobot_run1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

Do not resume the smoke checkpoint into the full run. Resume with the saved
full-run config to retain its processors, topology and schedule.
