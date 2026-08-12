import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import draccus
import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_groot.yaml"
EXPERIMENTS_DIR = "inference_experiments"


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def _load_policy(config: dict[str, Any]) -> tuple[PreTrainedConfig, str]:
    policy_values = config["policy"].copy()
    policy_path = policy_values.pop("path", None) or policy_values.pop("pretrained_path")

    policy_config = PreTrainedConfig.from_pretrained(
        policy_path,
        revision=policy_values.pop("revision", None),
    )
    config_values = draccus.encode(policy_config, PreTrainedConfig)
    policy_values.pop("type", None)
    config_values.update(policy_values)
    policy_config = draccus.decode(PreTrainedConfig, config_values)
    policy_config.pretrained_path = Path(policy_path)
    return policy_config, str(policy_path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _benchmark_inference(
    model: PreTrainedPolicy,
    observation: dict[str, Any],
    device: torch.device,
    warmup_steps: int,
    inference_steps: int,
) -> tuple[torch.Tensor, float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _synchronize(device)
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model.predict_action_chunk(observation)
    _synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(inference_steps):
            actions = model.predict_action_chunk(observation)
    _synchronize(device)
    inference_seconds = time.perf_counter() - start
    return actions, warmup_seconds, inference_seconds


def _profile_inference(
    model: PreTrainedPolicy,
    observation: dict[str, Any],
    device: torch.device,
    steps: int,
    trace_path: Path,
) -> tuple[str, float]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(steps):
                with record_function("predict_action_chunk"):
                    model.predict_action_chunk(observation)
        _synchronize(device)

    elapsed_seconds = time.perf_counter() - start
    summary = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=20)
    profiler.export_chrome_trace(str(trace_path))
    return summary, elapsed_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config-path", default=DEFAULT_CONFIG_PATH, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    policy_config, policy_path = _load_policy(config)
    rename_map = config.get("rename_map", {})

    dataset_config = config["dataset"]
    dataset_id = dataset_config["repo_id"]
    episodes = dataset_config.get("episodes", [0])

    benchmark_config = config.get("benchmark", {})
    warmup_steps = benchmark_config.get("warmup_steps", 2)
    inference_steps = benchmark_config.get("inference_steps", 10)
    profile_steps = benchmark_config.get("profile_steps", 3)
    output_dir = Path(benchmark_config.get("output_dir", EXPERIMENTS_DIR))

    device = torch.device(policy_config.device)
    experiment_start = time.perf_counter()
    started_at = datetime.now().astimezone()
    run_name = f"{started_at:%Y%m%d_%H%M%S_%f}"
    experiment_dir = output_dir / run_name
    experiment_dir.mkdir(parents=True)
    trace_path = experiment_dir / "trace.json"
    report_path = experiment_dir / "report.json"
    console_output = []

    def log(message: object) -> None:
        message = str(message)
        print(message)
        console_output.append(message)

    log(f"Experiment directory: {experiment_dir}")

    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        log(f"GPU: {gpu.name}")
        log(f"GPU memory: {gpu.total_memory / 1024**3:.1f} GiB")
        log(f"Compute capability: {gpu.major}.{gpu.minor}")
    log(f"Device: {device}")
    log(f"CUDA: {torch.version.cuda}")
    log(f"PyTorch: {torch.__version__}")

    dataset = LeRobotDataset(dataset_id, episodes=episodes)
    observation = dataset[0]
    image_keys = sorted(key for key in observation if key.startswith("observation.images."))
    image_shapes = {key: tuple(observation[key].shape) for key in image_keys}
    instruction = observation["task"]
    observation.pop("action")

    policy_config_values = draccus.encode(policy_config, PreTrainedConfig)
    log(f"Policy config: {policy_config_values}")

    model = make_policy(cfg=policy_config, ds_meta=dataset.meta, rename_map=rename_map)
    model.to(device).eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    if policy_config.type == "groot":
        preprocessor_overrides["groot_n1_7_vlm_encode_v1"] = {"device": str(device)}
    preprocess, _ = make_pre_post_processors(
        model.config, policy_path, preprocessor_overrides=preprocessor_overrides
    )
    observation = preprocess(observation)

    log(f"Images: {len(image_keys)}")
    for key in image_keys:
        log(f"  {key}: {image_shapes[key]}")
    log(f"Instruction: {instruction}")
    log(f"Instruction length: {len(instruction)} characters, {len(instruction.split())} words")
    log(f"Input tensor shapes: {_tensor_shapes(observation)}")

    actions, warmup_time, total_inference_time = _benchmark_inference(
        model, observation, device, warmup_steps, inference_steps
    )
    average_inference_time = total_inference_time / inference_steps
    peak_memory_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None

    log(f"Action shape: {actions.shape}")
    log(f"Warmup time: {warmup_time:.2f} s")
    log(f"Average inference time: {average_inference_time * 1000:.2f} ms")

    profiler_summary, profile_time = _profile_inference(model, observation, device, profile_steps, trace_path)
    log(profiler_summary)
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
            "policy_path": policy_path,
            "policy_type": policy_config.type,
            "policy_config": policy_config_values,
            "dataset_id": dataset_id,
            "rename_map": rename_map,
            "device": str(device),
            "warmup_steps": warmup_steps,
            "inference_steps": inference_steps,
            "profile_steps": profile_steps,
        },
        "gpu": {
            "name": gpu.name if gpu else None,
            "total_memory_bytes": gpu.total_memory if gpu else None,
            "total_memory_gib": gpu.total_memory / 1024**3 if gpu else None,
            "compute_capability": f"{gpu.major}.{gpu.minor}" if gpu else None,
            "peak_memory_allocated_bytes": peak_memory_allocated,
            "peak_memory_reserved_bytes": peak_memory_reserved,
        },
        "runtime": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "inputs": {
            "input_tensor_shapes": _tensor_shapes(observation),
            "image_count": len(image_keys),
            "image_shapes": {key: list(shape) for key, shape in image_shapes.items()},
            "instruction": instruction,
            "instruction_characters": len(instruction),
            "instruction_words": len(instruction.split()),
        },
        "outputs": {"action_shape": list(actions.shape)},
        "timings": {
            "warmup_seconds": warmup_time,
            "inference_total_seconds": total_inference_time,
            "inference_average_seconds": average_inference_time,
            "inference_average_ms": average_inference_time * 1000,
            "profiling_seconds": profile_time,
            "experiment_total_seconds": time.perf_counter() - experiment_start,
        },
        "profiler_summary": profiler_summary,
        "console_output": "\n".join(console_output),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
