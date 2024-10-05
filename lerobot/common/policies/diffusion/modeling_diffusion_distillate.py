#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import copy
import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel


class DiffusionPolicyDistillate(
    DiffusionPolicy
):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "diffusion_distillate"

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config=config, dataset_stats=dataset_stats)
        self.diffusion = DiffusionModelDistillate(config)


    def forward(self, batch: dict[str, Tensor], teacher_policy: DiffusionPolicy) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch, teacher_model=teacher_policy.diffusion)
        return {"loss": loss}


class DiffusionModelDistillate(DiffusionModel):
    def __init__(self, config: DiffusionConfig):
        super().__init__(config=config)
    
    def distilation_timesteps(self, shape: tuple):
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        indexes = torch.randint(
            low=0,
            high=self.num_inference_steps,
            size=shape
        ).long()
        return self.noise_scheduler.timesteps[indexes]

    def compute_loss(self, batch: dict[str, Tensor], teacher_model: DiffusionModel) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        batch_size = batch["observation.state"].shape[0]
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)
        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)        
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)
        
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        teacher_model.noise_scheduler.set_timesteps(teacher_model.num_inference_steps)
        
        timesteps = self.distilation_timesteps(shape=(trajectory.shape[0],)).to(device=trajectory.device)

        # noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        
        # one student DDIM step
        student_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        model_output = self.unet(
                student_trajectory,
                timesteps,
                global_cond=global_cond,
            )
        
        # snr = self.noise_scheduler.snr(timestep=timesteps)
        # w = 1 + snr ** 2
        # # # Compute previous image: x_t -> x_t-1
        # student_trajectory_prev = self.noise_scheduler.step(model_output, timesteps, student_trajectory).prev_sample
        
        # two teacher DDIM step
        teacher_trajectory = teacher_model.noise_scheduler.add_noise(trajectory, eps, timesteps)
        model_output_1 = teacher_model.unet(
                teacher_trajectory,
                timesteps,
                global_cond=global_cond,
            )
        # Compute previous image: x_t -> x_t-1
        teacher_trajectory_prev = teacher_model.noise_scheduler.step(model_output_1, timesteps, teacher_trajectory).prev_sample

        prev_timesteps = timesteps - teacher_model.noise_scheduler.config.num_train_timesteps // teacher_model.noise_scheduler.num_inference_steps
        prev_timesteps = torch.clip(prev_timesteps,
                                    min = 0,
                                    max = self.config.num_train_timesteps)
        
        model_output_2 = teacher_model.unet(
                teacher_trajectory_prev,
                prev_timesteps,
                global_cond=global_cond,
            )
        # Compute previous image: x_t -> x_t-1
        teacher_ouptut = teacher_model.noise_scheduler.step(model_output_2, prev_timesteps, teacher_trajectory_prev)
        
        teacher_trajectory_prev_prev = teacher_ouptut.prev_sample
        predicted_model_output = self.noise_scheduler.step_back(prev_sample=teacher_trajectory_prev_prev,
                                                             sample=teacher_trajectory,
                                                             timestep=timesteps)
        
        loss = F.mse_loss(predicted_model_output, model_output, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
