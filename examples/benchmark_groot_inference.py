import time

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot import GrootConfig, GrootPolicy

DEVICE = "cuda"
MODEL_ID = "nvidia/gr00t17-lerobot-libero_object-640"
BASE_MODEL_ID = "nvidia/GR00T-N1.7-3B"
DATASET_ID = "lerobot/libero"
WARMUP_STEPS = 2
INFERENCE_STEPS = 10


def main():
    dataset = LeRobotDataset(DATASET_ID, episodes=[0])
    observation = dataset[0]
    observation.pop("action")

    config = GrootConfig.from_pretrained(MODEL_ID)
    config.base_model_path = BASE_MODEL_ID
    config.device = DEVICE

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

    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            model.predict_action_chunk(observation)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(INFERENCE_STEPS):
            actions = model.predict_action_chunk(observation)
        torch.cuda.synchronize()

    inference_time = (time.perf_counter() - start) / INFERENCE_STEPS
    print(f"Action shape: {actions.shape}")
    print(f"Average inference time: {inference_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
