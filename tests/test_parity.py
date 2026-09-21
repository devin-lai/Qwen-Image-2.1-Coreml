"""Compare the host-side math with the pinned Diffusers reference.

The rotary table, timestep sinusoid, and Euler sampler should agree exactly.
Reference comparisons skip if Diffusers is absent; install the torch-reference
extra to run the complete suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_image_coreml.layout import (
    fold_rope_sign,
    rope_for_bucket,
    text_image_layout,
    time_proj,
)
from qwen_image_coreml.scheduler import (
    FlowMatchEulerScheduler,
    resolution_mu,
)

diffusers = pytest.importorskip("diffusers", reason="parity checks need diffusers installed")

TEXT_LEN, LATENT, STEPS = 31, 64, 40
TARGET_TOKENS = LATENT * LATENT


def _reference_rope():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope

    return QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])


def test_rope_table_is_bit_identical():
    """The ported rotary table equals `QwenImage21Rope.forward`, element for element."""
    img_mask = torch.zeros(1, TEXT_LEN + TARGET_TOKENS // 4, dtype=torch.bool)
    img_mask[0, TEXT_LEN:] = True
    repeats = torch.where(img_mask, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

    expected = _reference_rope()(
        [(1, LATENT, LATENT)], image_pad_mask, device=torch.device("cpu")
    )

    from qwen_image_coreml.layout import rope_freqs

    actual = rope_freqs([(1, LATENT, LATENT)], image_pad_mask)

    assert actual.shape == expected.shape
    assert torch.equal(actual.real, expected.real)
    assert torch.equal(actual.imag, expected.imag)


def test_folded_sign_multiplies_the_same_pairs():
    """Folding the sign into `sin` is an identity, not an approximation.

    `[x1, x0] * [-s0, +s0]` and `[-x1, x0] * [s0, s0]` multiply the same pairs of
    numbers, so the two forms of the rotation must agree to `max_abs = 0`.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, 128)
    layout = text_image_layout(TEXT_LEN, LATENT, LATENT, fold=False)
    cos, sin = layout["cos_target"], layout["sin_target"]

    # The stack form: build the rotated vector explicitly, then multiply.
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape(x.shape)
    stack_form = x * cos + rotated * sin

    # The fold form: swap each pair with one flip, sign already in the table.
    swapped = x.reshape(*x.shape[:-1], -1, 2).flip(-1).reshape(x.shape)
    fold_form = x * cos + swapped * fold_rope_sign(sin)

    assert torch.equal(stack_form, fold_form)


def test_padded_layout_keeps_the_image_in_place():
    """Padding the prompt must not move the target image's rotary positions."""
    unpadded = text_image_layout(TEXT_LEN, LATENT, LATENT)
    padded = rope_for_bucket(TEXT_LEN, 64, LATENT, LATENT)

    assert torch.equal(padded["cos_target"], unpadded["cos_target"])
    assert torch.equal(padded["sin_target"], unpadded["sin_target"])
    # Padded slots carry the identity rotation, not zeros.
    assert torch.equal(padded["cos_prefix_padded"][TEXT_LEN:], torch.ones(64 - TEXT_LEN, 128))
    assert torch.equal(padded["sin_prefix_padded"][TEXT_LEN:], torch.zeros(64 - TEXT_LEN, 128))


def test_timestep_projection_matches_the_model():
    """`time_proj` equals `QwenImage21TemporalTimesteps`, on the real schedule.

    Checked on the *real* timesteps rather than on 1.0: 1.0 is exactly
    representable in fp16, which is how a broken timestep path verifies clean.
    """
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21TemporalTimesteps,
    )

    module = QwenImage21TemporalTimesteps(timestep_dim=256)
    schedule = FlowMatchEulerScheduler.make(STEPS, TARGET_TOKENS)
    for t_value in schedule.timesteps:
        timestep = (t_value.expand(1) / 1000).to(torch.float32)
        with torch.no_grad():
            expected = module(timestep)
        assert torch.equal(time_proj(timestep), expected), f"t={float(t_value)}"


def _reference_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
        shift_terminal=0.02,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
    )


def test_sigma_schedule_is_bit_identical():
    """The ported sampler's sigmas and timesteps equal diffusers'."""
    reference = _reference_scheduler()
    reference.set_timesteps(
        sigmas=np.linspace(1.0, 1 / STEPS, STEPS), mu=resolution_mu(TARGET_TOKENS)
    )
    reference.set_begin_index(0)

    ours = FlowMatchEulerScheduler.make(STEPS, TARGET_TOKENS)

    assert torch.equal(ours.timesteps, reference.timesteps)
    assert torch.equal(ours.sigmas, reference.sigmas)


def test_euler_step_is_bit_identical():
    """Forty steps of the ported sampler track diffusers' exactly."""
    reference = _reference_scheduler()
    reference.set_timesteps(
        sigmas=np.linspace(1.0, 1 / STEPS, STEPS), mu=resolution_mu(TARGET_TOKENS)
    )
    reference.set_begin_index(0)
    ours = FlowMatchEulerScheduler.make(STEPS, TARGET_TOKENS)

    torch.manual_seed(0)
    mine = torch.randn(1, 256, 64)
    theirs = mine.clone()

    for index, t_value in enumerate(ours.timesteps):
        torch.manual_seed(index)
        model_output = torch.randn_like(mine)
        mine = ours.step(model_output, index, mine)
        theirs = reference.step(model_output, t_value, theirs, return_dict=False)[0]
        assert torch.equal(mine, theirs), f"diverged at step {index}"
