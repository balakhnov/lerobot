# Inference benchmark

`examples/benchmark_inference.py` measures policy inference latency on one sample from a
LeRobot dataset. It reports preprocessing, model inference, postprocessing, and their
end-to-end sum independently. The benchmark currently includes ready-to-run eager and
compiled configurations for GR00T N1.7.

## Requirements

- An NVIDIA GPU with enough memory for the selected policy.
- A working CUDA-enabled PyTorch installation.
- Access to the configured Hugging Face model and dataset repositories.
- LeRobot's GR00T dependencies.

Install the project dependencies and authenticate with Hugging Face when required:

```bash
uv sync --locked --extra groot
hf auth login
```

## Run the benchmark

Run eager PyTorch inference:

```bash
uv run python examples/benchmark_inference.py \
    --config configs/benchmark_groot.yaml
```

Run compiled inference:

```bash
uv run python examples/benchmark_inference.py \
    --config configs/benchmark_groot_compile.yaml
```

The first compiled warm-up can take several minutes while PyTorch builds and tunes the
compiled graphs. Compilation time is included in the reported warm-up duration but is
excluded from the timed inference samples.

## Measurement method

The benchmark loads the first frame of the configured dataset episode and uses the same
observation for every iteration. For the provided GR00T configurations this observation
contains:

- batch size 1;
- two 256 x 256 RGB camera images;
- the episode's language instruction;
- an eight-dimensional robot state;
- four flow-matching denoising steps;
- a 16-step action output chunk.

It performs 10 full-pipeline warm-up calls followed by 100 measured calls. Every measured
iteration runs these stages:

1. `preprocess(observation)`
2. `model.predict_action_chunk(processed_observation)`
3. `postprocess(actions)`

CUDA is synchronized immediately before the iteration and after every stage. This ensures
that asynchronous GPU work is charged to the stage that launched it. The reported E2E
sample is calculated as:

```text
preprocess_ms + model_ms + postprocess_ms
```

This component-sum definition follows NVIDIA's GR00T component benchmark methodology. It
also means that E2E includes the synchronization overhead at each component boundary; it
is not an unsynchronized throughput measurement.

For each component and E2E, the benchmark reports:

- median;
- mean and population standard deviation;
- minimum;
- maximum.

Use the median when comparing with NVIDIA's published general GR00T N1.7 table. The
profiler runs after the latency benchmark and covers only `predict_action_chunk`; profiler
overhead is not included in the latency results.

## Configuration

The policy and dataset are selected in the YAML file:

```yaml
policy:
  path: nvidia/gr00t17-lerobot-libero_object-640
  base_model_path: nvidia/GR00T-N1.7-3B
  device: cuda

dataset:
  repo_id: lerobot/libero
  episodes: [0]

benchmark:
  warmup_steps: 10
  inference_steps: 100
  profile_steps: 3
  output_dir: groot_inference_experiments
```

The compiled GR00T configuration controls the action head and backbone independently:

```yaml
policy:
  compile_action_head: true
  compile_action_head_mode: reduce-overhead
  compile_backbone: true
  compile_backbone_mode: reduce-overhead
  compile_fullgraph: true
```

Supported compile-mode strings are those accepted by the installed PyTorch version, such
as `default`, `reduce-overhead`, and `max-autotune`. The deprecated `compile_mode` field is
still accepted for old checkpoints and applies the same mode to both components.

To benchmark only one compiled component, disable the other one. For example, this is
closer to NVIDIA's published `torch.compile` benchmark, which compiles the action head but
leaves the backbone eager:

```yaml
policy:
  compile_action_head: true
  compile_action_head_mode: max-autotune
  compile_backbone: false
```

## Outputs

Each run creates a timestamped directory under `benchmark.output_dir`:

```text
groot_inference_experiments/<timestamp>/
|-- report.json
`-- trace.json
```

`report.json` contains:

- policy, dataset, hardware, CUDA, and PyTorch configuration;
- input and output tensor shapes;
- summary statistics for every measured stage;
- all 100 raw timing samples in milliseconds;
- peak allocated and reserved CUDA memory;
- the text profiler summary.

`trace.json` is a Chrome/PyTorch profiler trace. Open it with Perfetto or another compatible
trace viewer to inspect CPU and CUDA operations.

## Comparing results with NVIDIA

For a meaningful comparison, keep these variables identical:

- GPU model and power mode;
- checkpoint and base-model revision;
- number and resolution of camera inputs;
- instruction and state dimensions;
- denoising steps and action horizon;
- precision and attention implementation;
- eager, compiled, or TensorRT execution mode;
- warm-up count and summary statistic.

The default local configuration uses a LeRobot LIBERO Object checkpoint. NVIDIA's published
general N1.7 table uses the original `GR00T-N1.7-LIBERO/libero_10` checkpoint, so the numbers
are not strictly apples-to-apples. A closer LeRobot configuration can use:

```yaml
policy:
  path: nvidia/gr00t17-lerobot-libero_10-640
```

The executable NVIDIA LIBERO benchmark path selects two camera views despite its published
table being captioned "1 camera." Record the actual input tensor shapes from `report.json`
when presenting results.

The benchmark does not currently measure TensorRT. Compare its eager and compiled results
only with the corresponding NVIDIA modes.
