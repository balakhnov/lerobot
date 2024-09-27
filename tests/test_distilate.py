import pytest
import torch
from torch import nn
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion_distillate import DiffusionPolicyDistillate


@pytest.fixture
def model():
    cfg = DiffusionConfig()
    policy = DiffusionPolicyDistillate(cfg)
    policy.diffusion.num_inference_steps = 10
    return policy.diffusion

def test_distilation_timesteps_shape(model):
    shape = (10,)
    timesteps = model.distilation_timesteps(shape)
    print(timesteps.shape)
    assert timesteps.shape == shape + 1

def test_distilation_timesteps_range(model):
    shape = (100,)
    timesteps = model.distilation_timesteps(shape)
    assert torch.all(timesteps >= 0)
    assert torch.all(timesteps < model.noise_scheduler.config.num_train_timesteps)

def test_distilation_timesteps_randomness(model):
    shape = (1000,)
    timesteps1 = model.distilation_timesteps(shape)
    timesteps2 = model.distilation_timesteps(shape)
    assert not torch.all(timesteps1 == timesteps2)

@pytest.mark.parametrize("num_inference_steps", [10, 50, 100])
def test_distilation_timesteps_num_inference_steps(model, num_inference_steps):
    model.num_inference_steps = num_inference_steps
    shape = (10,)
    timesteps = model.distilation_timesteps(shape)
    assert timesteps.shape == shape
    assert torch.all(timesteps < model.noise_scheduler.config.num_train_timesteps)
