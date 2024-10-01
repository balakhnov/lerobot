"""
This scripts distilate diffusion policy with progressive distilation algorithm
"""
import argparse
from pathlib import Path
import torch
import tqdm
from huggingface_hub import snapshot_download
import copy
import math
from lerobot.common.envs.factory import make_env
from omegaconf import DictConfig


import wandb
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.modeling_diffusion_distillate import DiffusionPolicyDistillate
from lerobot.scripts.eval import eval_policy

# TODO: implement original algorithm, not some approximation
# TODO: test back_step function
# TODO: test shape and functionality of weighting function
# TODO: evaluate performance of teacher model as function of numbers of steps
# visualisation

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, required=True)
parser.add_argument("--input_policy_path", type=str, required=True)
args = parser.parse_args()

NUM_STUDENT_INFERENCE_STEPS = args.num_steps

training_steps = 1000
log_freq = 100
num_eval_episods = 10

# Create a directory to store the video of the evaluation
output_directory = Path(f"outputs/distil/example_pusht_diffusion_{NUM_STUDENT_INFERENCE_STEPS}")
output_directory.mkdir(parents=True, exist_ok=True)

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}
dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=2,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Download the diffusion policy for pusht environment
pretrained_policy_path = Path(args.input_policy_path)
if pretrained_policy_path.as_posix() == "lerobot/diffusion_pusht":
    pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")
# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.

print(f'policy path: {pretrained_policy_path}')
student_policy = DiffusionPolicyDistillate.from_pretrained(pretrained_policy_path)
student_policy.diffusion.change_noise_scheduler_type('DDIM')
student_policy.diffusion.num_inference_steps = NUM_STUDENT_INFERENCE_STEPS
student_policy.train()
student_policy.to(device)
optimizer = torch.optim.Adam(student_policy.parameters(), lr=1e-4)

teacher_policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
teacher_policy.diffusion.change_noise_scheduler_type('DDIM')
teacher_policy.diffusion.num_inference_steps = 2*student_policy.diffusion.num_inference_steps
teacher_policy.eval()
teacher_policy.to(device)

env_cfg = DictConfig({'env':{'name': 'pusht', 
           'task': 'PushT-v0', 
           'image_size': 96, 
           'state_dim': 2, 
           'action_dim': 2, 
           'fps': '${fps}', 
           'episode_length': 300, 
           'gym': {'obs_type': 'pixels_agent_pos', 
                   'render_mode': 'rgb_array', 
                   'visualization_width': 384, 
                   'visualization_height': 384}},
                   'eval':{'n_episodes': 10, 'batch_size': 10, 'use_async_envs': False}})
env = make_env(cfg=env_cfg)

# log metrics to wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="diffusion_distillate",

    # track hyperparameters and run metadata
    config={
    }
)

# Reset policies and environmens to prepare for rollout
teacher_policy.reset()
student_policy.reset()

step = 0
pbar = tqdm.tqdm(total=training_steps)
done = False
while not done:
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = student_policy.forward(batch, teacher_policy=teacher_policy)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        info = {}
        if step > 0 and step % log_freq == 0:
            # Save a policy checkpoint.
            student_policy.save_pretrained(output_directory)
            # eval student
            student_eval_info = eval_policy(
                env,
                student_policy,
                num_eval_episods,
                max_episodes_rendered=2,
                videos_dir=Path(output_directory) / "student_videos",
                start_seed=42,
            )
            # eval teacher
            teacher_eval_info = eval_policy(
                env,
                teacher_policy,
                num_eval_episods,
                max_episodes_rendered=2,
                videos_dir=Path(output_directory) / "teacher_videos",
                start_seed=42,
            )
            student_policy.train()
            info['teacher_success_rate'] = teacher_eval_info["aggregated"]["pc_success"]
            info['student_success_rate'] = student_eval_info["aggregated"]["pc_success"]
        info['loss'] = loss.item()
        info['log_loss'] = math.log(loss.item())
        info['step'] = step
        step += 1
        pbar.update()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})  # Add this line to show loss
        wandb.log(info)

        if step >= training_steps:
            done = True
            break
pbar.close()
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
