"""
This scripts distilate diffusion policy with progressive distilation algorithm
"""

from pathlib import Path
import torch
import tqdm
from huggingface_hub import snapshot_download
import copy

import wandb
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.modeling_diffusion_distillate import DiffusionPolicyDistillate

# TODO: implement particular case for 2 steps
# TODO: find out how to work with number of steps(in reference, in our case)
# TODO: understand algorithm in progressive distilation what exactly do we train
# TODO: evaluate performance of teacher model as function of numbers of steps
# visualisation

device = torch.device("cpu")

training_steps = 5000
log_freq = 100

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/distil/example_pusht_diffusion")
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
    num_workers=0,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Download the diffusion policy for pusht environment
pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

teacher_policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
teacher_policy.diffusion.num_inference_steps = 2
teacher_policy.eval()
teacher_policy.to(device)

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
student_policy = DiffusionPolicyDistillate.from_pretrained(pretrained_policy_path)
student_policy.diffusion.num_inference_steps = 1
student_policy.train()
student_policy.to(device)
optimizer = torch.optim.Adam(student_policy.parameters(), lr=1e-4)

# log metrics to wandb
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="diffusion_distillate",

#     # track hyperparameters and run metadata
#     config={
#     }
# )

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
        if step % log_freq == 0:
            # Save a policy checkpoint.
            student_policy.save_pretrained(output_directory)
        info = {}
        info['loss'] = loss.item()
        info['step'] = step
        step += 1
        pbar.update()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})  # Add this line to show loss
        # wandb.log(info)

        if step >= training_steps:
            done = True
            break
pbar.close()
# [optional] finish the wandb run, necessary in notebooks
# wandb.finish()
