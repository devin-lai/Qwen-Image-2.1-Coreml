# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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
# Modified by Devin Lai for fixed-shape Core ML inference. See NOTICE.

"""Flow-matching Euler sampling for the Qwen-Image-2.1 configuration.

Adapted from Diffusers' FlowMatchEulerDiscreteScheduler with exponential dynamic
shifting and shift_terminal=0.02. The configuration is included here so inference
does not need the original checkpoint. See NOTICE and tests/test_parity.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# scheduler/scheduler_config.json, Qwen/Qwen-Image-2.1
NUM_TRAIN_TIMESTEPS = 1000
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 8192
BASE_SHIFT = 0.5
MAX_SHIFT = 0.9
SHIFT_TERMINAL = 0.02


def resolution_mu(target_tokens: int) -> float:
    """The resolution-dependent shift: more image tokens, more noise early on.

    Linear in the token count between the two anchor points the config names.
    """
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    return target_tokens * slope + BASE_SHIFT - slope * BASE_IMAGE_SEQ_LEN


@dataclass
class FlowMatchEulerScheduler:
    """Sigmas and timesteps for one sampling run.

    `sigmas` has `num_steps + 1` entries — the extra terminal zero is what the
    last step integrates down to.
    """

    sigmas: torch.Tensor
    timesteps: torch.Tensor

    @classmethod
    def make(cls, num_steps: int, target_tokens: int) -> FlowMatchEulerScheduler:
        if num_steps < 2:
            raise ValueError("this scheduler requires at least 2 denoising steps")
        mu = resolution_mu(target_tokens)
        # The pipeline's own sigma grid: linear from 1.0 down to 1/num_steps.
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps).astype(np.float32)

        # Exponential dynamic shift.
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))

        # Stretch the schedule so it terminates exactly at `shift_terminal`
        # instead of wherever the shift happened to leave it.
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - SHIFT_TERMINAL)
        sigmas = 1 - (one_minus_z / scale_factor)

        sigmas = torch.from_numpy(np.asarray(sigmas)).to(torch.float32)
        timesteps = sigmas * NUM_TRAIN_TIMESTEPS
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        return cls(sigmas=sigmas, timesteps=timesteps)

    def __len__(self) -> int:
        return len(self.timesteps)

    def step(self, model_output: torch.Tensor, index: int, sample: torch.Tensor) -> torch.Tensor:
        """One Euler step along the probability-flow ODE.

        Upcast to fp32 before integrating: the increment is a small multiple of a
        large sample, and accumulating it in the model's dtype loses the tail of
        every step.
        """
        sample = sample.to(torch.float32)
        dt = self.sigmas[index + 1] - self.sigmas[index]
        return (sample + dt * model_output).to(model_output.dtype)
