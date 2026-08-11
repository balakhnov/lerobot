import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot import GrootConfig, GrootPolicy

DEVICE = "cuda"
MODEL_ID = "nvidia/gr00t17-lerobot-libero_object-640"
BASE_MODEL_ID = "nvidia/GR00T-N1.7-3B"
DATASET_ID = "lerobot/libero"
WARMUP_STEPS = 2
INFERENCE_STEPS = 10
PROFILE_STEPS = 3
EXPERIMENTS_DIR = "groot_inference_experiments"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile-action-head", "--compile_action_head", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-backbone", "--compile_backbone", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--compile-mode", "--compile_mode", default="max-autotune")
    parser.add_argument("--compile-backend", "--compile_backend", default="inductor")
    parser.add_argument(
        "--compile-fullgraph", "--compile_fullgraph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", "--output_dir", default=EXPERIMENTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_start = time.perf_counter()
    started_at = datetime.now().astimezone()
    compiled_targets = [
        target
        for enabled, target in (
            (args.compile_backbone, "backbone"),
            (args.compile_action_head, "action_head"),
        )
        if enabled
    ]
    compile_label = f"compiled_{'_'.join(compiled_targets)}" if compiled_targets else "eager"
    run_name = f"{started_at:%Y%m%d_%H%M%S_%f}_{compile_label}"
    experiment_dir = Path(args.output_dir) / run_name
    experiment_dir.mkdir(parents=True)
    trace_path = experiment_dir / "trace.json"
    report_path = experiment_dir / "report.json"
    console_output = []

    def log(message):
        message = str(message)
        print(message)
        console_output.append(message)

    log(f"Experiment directory: {experiment_dir}")

    gpu = torch.cuda.get_device_properties(DEVICE)
    log(f"GPU: {gpu.name}")
    log(f"GPU memory: {gpu.total_memory / 1024**3:.1f} GiB")
    log(f"Compute capability: {gpu.major}.{gpu.minor}")
    log(f"CUDA: {torch.version.cuda}")
    log(f"PyTorch: {torch.__version__}")

    dataset = LeRobotDataset(DATASET_ID, episodes=[0])
    observation = dataset[0]
    image_keys = sorted(key for key in observation if key.startswith("observation.images."))
    image_shapes = {key: tuple(observation[key].shape) for key in image_keys}
    instruction = observation["task"]
    observation.pop("action")

    config = GrootConfig.from_pretrained(MODEL_ID)
    config.base_model_path = BASE_MODEL_ID
    config.device = DEVICE
    config.compile_action_head = args.compile_action_head
    config.compile_backbone = args.compile_backbone
    config.compile_mode = args.compile_mode
    config.compile_backend = args.compile_backend
    config.compile_fullgraph = args.compile_fullgraph

    log(f"Compile backbone prefill: {config.compile_backbone}")
    log(f"Compile action head: {config.compile_action_head}")
    if config.compile_backbone or config.compile_action_head:
        log(f"Compile mode: {config.compile_mode}")
        log(f"Compile backend: {config.compile_backend}")
        log(f"Compile fullgraph: {config.compile_fullgraph}")

    model = GrootPolicy.from_pretrained(MODEL_ID, config=config)
    model.to(DEVICE).eval()

    preprocess, _ = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={
            "groot_n1_7_vlm_encode_v1": {"device": DEVICE},
            "device_processor": {"device": DEVICE},
        },
    )
    observation = preprocess(observation)

    action_head = model._groot_model.action_head
    noise_shape = (
        observation["input_ids"].shape[0],
        action_head.config.action_horizon,
        action_head.action_dim,
    )
    log(f"Backbone sequence length: {observation['input_ids'].shape[-1]}")
    log(f"Images: {len(image_keys)}")
    for key in image_keys:
        log(f"  {key}: {image_shapes[key]}")
    log(f"Instruction: {instruction}")
    log(f"Instruction length: {len(instruction)} characters, {len(instruction.split())} words")
    log(f"Action head noise shape: {noise_shape}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            model.predict_action_chunk(observation)
    torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start

    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(INFERENCE_STEPS):
            actions = model.predict_action_chunk(observation)
        torch.cuda.synchronize()

    total_inference_time = time.perf_counter() - start
    inference_time = total_inference_time / INFERENCE_STEPS
    peak_memory_allocated = torch.cuda.max_memory_allocated(DEVICE)
    peak_memory_reserved = torch.cuda.max_memory_reserved(DEVICE)
    log(f"Action shape: {actions.shape}")
    log(f"Warmup time: {warmup_time:.2f} s")
    log(f"Average inference time: {inference_time * 1000:.2f} ms")

    profile_start = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(PROFILE_STEPS):
                with record_function("groot_predict_action_chunk"):
                    model.predict_action_chunk(observation)
        torch.cuda.synchronize()

    profile_time = time.perf_counter() - profile_start
    profiler_summary = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=20)
    log(profiler_summary)
    profiler.export_chrome_trace(str(trace_path))
    log(f"Chrome trace saved to {trace_path}")
    log(f"Experiment report saved to {report_path}")

    report = {
        "started_at": started_at.isoformat(),
        "artifacts": {
            "experiment_directory": str(experiment_dir),
            "trace": str(trace_path),
            "report": str(report_path),
        },
        "parameters": {
            "model_id": MODEL_ID,
            "base_model_id": BASE_MODEL_ID,
            "dataset_id": DATASET_ID,
            "device": DEVICE,
            "warmup_steps": WARMUP_STEPS,
            "inference_steps": INFERENCE_STEPS,
            "profile_steps": PROFILE_STEPS,
            "compile_backbone": args.compile_backbone,
            "compile_action_head": args.compile_action_head,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
            "compile_fullgraph": args.compile_fullgraph,
        },
        "gpu": {
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "total_memory_gib": gpu.total_memory / 1024**3,
            "compute_capability": f"{gpu.major}.{gpu.minor}",
            "peak_memory_allocated_bytes": peak_memory_allocated,
            "peak_memory_reserved_bytes": peak_memory_reserved,
        },
        "runtime": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "inputs": {
            "backbone_sequence_length": observation["input_ids"].shape[-1],
            "input_ids_shape": list(observation["input_ids"].shape),
            "image_count": len(image_keys),
            "image_shapes": {key: list(shape) for key, shape in image_shapes.items()},
            "instruction": instruction,
            "instruction_characters": len(instruction),
            "instruction_words": len(instruction.split()),
            "action_head_noise_shape": list(noise_shape),
        },
        "outputs": {"action_shape": list(actions.shape)},
        "timings": {
            "warmup_seconds": warmup_time,
            "inference_total_seconds": total_inference_time,
            "inference_average_seconds": inference_time,
            "inference_average_ms": inference_time * 1000,
            "profiling_seconds": profile_time,
            "experiment_total_seconds": time.perf_counter() - experiment_start,
        },
        "profiler_summary": profiler_summary,
        "console_output": "\n".join(console_output),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
