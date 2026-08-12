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
from lerobot.policies import make_policy, make_pre_post_processors

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_groot.yaml"
EXPERIMENTS_DIR = "inference_experiments"


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected {config_path} to contain a YAML mapping.")
    return config


def _load_policy(config: dict[str, Any]):
    policy_values = config.get("policy")
    if not isinstance(policy_values, dict):
        raise ValueError("config.yaml must contain a 'policy' mapping with a 'path' entry.")

    policy_values = dict(policy_values)
    policy_path = policy_values.pop("path", policy_values.pop("pretrained_path", None))
    if not policy_path:
        raise ValueError("config.yaml must set policy.path to a LeRobot policy checkpoint or Hub repository.")

    policy_config = PreTrainedConfig.from_pretrained(
        policy_path,
        revision=policy_values.pop("revision", None),
    )
    config_values = draccus.encode(policy_config, PreTrainedConfig)
    configured_type = policy_values.pop("type", None)
    if configured_type is not None and configured_type != config_values["type"]:
        raise ValueError(
            f"config.yaml policy.type={configured_type!r} does not match the checkpoint policy type "
            f"{config_values['type']!r}."
        )
    config_values.update(policy_values)
    policy_config = draccus.decode(PreTrainedConfig, config_values)
    policy_config.pretrained_path = Path(policy_path)
    return policy_config, str(policy_path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config-path", default=DEFAULT_CONFIG_PATH, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(args.config)
    policy_config, policy_path = _load_policy(config)

    dataset_values = config.get("dataset", {})
    if not isinstance(dataset_values, dict) or not dataset_values.get("repo_id"):
        raise ValueError("config.yaml must contain dataset.repo_id.")
    dataset_id = str(dataset_values["repo_id"])
    episodes = dataset_values.get("episodes", [0])

    benchmark_values = config.get("benchmark", {})
    if not isinstance(benchmark_values, dict):
        raise ValueError("config.yaml 'benchmark' must be a mapping when provided.")
    warmup_steps = int(benchmark_values.get("warmup_steps", 2))
    inference_steps = int(benchmark_values.get("inference_steps", 10))
    profile_steps = int(benchmark_values.get("profile_steps", 3))
    output_dir = benchmark_values.get("output_dir", EXPERIMENTS_DIR)

    device = torch.device(policy_config.device)
    experiment_start = time.perf_counter()
    started_at = datetime.now().astimezone()
    run_name = f"{started_at:%Y%m%d_%H%M%S_%f}"
    experiment_dir = Path(output_dir) / run_name
    experiment_dir.mkdir(parents=True)
    trace_path = experiment_dir / "trace.json"
    report_path = experiment_dir / "report.json"
    console_output = []

    def log(message):
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

    model = make_policy(cfg=policy_config, ds_meta=dataset.meta)
    model.to(device).eval()

    preprocessor_overrides = {"device_processor": {"device": str(device)}}
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model.predict_action_chunk(observation)
    _synchronize(device)
    warmup_time = time.perf_counter() - warmup_start

    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(inference_steps):
            actions = model.predict_action_chunk(observation)
        _synchronize(device)

    total_inference_time = time.perf_counter() - start
    inference_time = total_inference_time / inference_steps
    peak_memory_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    log(f"Action shape: {actions.shape}")
    log(f"Warmup time: {warmup_time:.2f} s")
    log(f"Average inference time: {inference_time * 1000:.2f} ms")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profile_start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(profile_steps):
                with record_function("predict_action_chunk"):
                    model.predict_action_chunk(observation)
        _synchronize(device)

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
            "policy_path": policy_path,
            "policy_type": policy_config.type,
            "policy_config": policy_config_values,
            "dataset_id": dataset_id,
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
