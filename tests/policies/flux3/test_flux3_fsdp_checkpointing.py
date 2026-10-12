# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.flux3.f3 import transformer


def test_checkpoint_recompute_preserves_fsdp_compute_dtype(monkeypatch):
    class CastOnceBlock(nn.Module):
        """Simulate FSDP casting only the original forward's inputs."""

        def __init__(self):
            super().__init__()
            self.seen_dtypes = []

        def _get_fsdp_state(self):
            return SimpleNamespace(
                _mp_policy=SimpleNamespace(cast_forward_inputs=True, param_dtype=torch.bfloat16)
            )

        def forward(self, x, nested, *, bias):
            self.seen_dtypes.append((x.dtype, nested[0].dtype, nested[1].dtype, bias.dtype))
            if len(self.seen_dtypes) == 1:
                x = x.to(torch.bfloat16)
                bias = bias.to(torch.bfloat16)
            return (x + bias).sin() * nested[0]

    monkeypatch.setattr(transformer, "FSDPModule", CastOnceBlock)
    block = CastOnceBlock()
    runner = SimpleNamespace(training=True, gradient_checkpointing=True)
    x = torch.randn(2, 4, requires_grad=True)
    bias = torch.randn(2, 4, requires_grad=True)
    modulation = torch.ones(2, 4)
    indices = torch.arange(4)
    result = transformer.JointSingleSeq._run_block(runner, block, x, (modulation, indices), bias=bias)
    result.sum().backward()
    expected = (torch.bfloat16, torch.bfloat16, torch.int64, torch.bfloat16)
    assert block.seen_dtypes == [expected, expected]
    reference = x.detach().bfloat16() + bias.detach().bfloat16()
    torch.testing.assert_close(x.grad, reference.cos().float())
    torch.testing.assert_close(bias.grad, x.grad)
