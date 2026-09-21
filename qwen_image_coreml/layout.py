# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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

"""Host-side rotary tables, attention masks, prompt padding, and timestep projection.

The rotary table and timestep projection are adapted from Diffusers. They run
once per layout or timestep on the CPU and supply real-valued inputs to Core ML.
The parity tests compare them with the reference implementation.
"""

from __future__ import annotations

import math

import torch

# Additive mask value. `exp()` of it underflows to zero well inside fp16's range,
# so a masked key contributes nothing without ever producing a NaN.
NEG = -1.0e4

# From the checkpoint's `transformer/config.json`. Vendored rather than read so
# the demo needs no part of the PyTorch checkpoint on disk.
ROPE_THETA = 10000
ROPE_AXES_DIM = (16, 56, 56)
HEADS = 32
HEAD_DIM = 128
LATENT_CHANNELS = 64
VAE_SCALE = 16  # one latent token per 16x16 pixel tile


# --------------------------------------------------------------------- rotary


def _rope_params(index: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    freqs = torch.outer(
        index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _rope_tables(theta: int = ROPE_THETA, axes_dim=ROPE_AXES_DIM) -> list[torch.Tensor]:
    """One complex frequency table per axis, indexed by position.

    Positions run 0..8191 forward and then -1..-1024 backward, so a negative
    index addresses the tail of the table — which is how the image grid, centred
    on zero, reaches its negative coordinates.
    """
    pos_index = torch.arange(8192)
    neg_index = torch.arange(1024).flip(0) * -1 - 1
    return [
        torch.cat([_rope_params(pos_index, dim, theta), _rope_params(neg_index, dim, theta)], dim=0)
        for dim in axes_dim
    ]


def rope_freqs(img_shapes, image_pad_mask: torch.Tensor) -> torch.Tensor:
    """3-axis (frame, height, width) rotary frequencies over the joint sequence.

    Text tokens advance a shared position on all three axes. Each image block
    freezes the frame axis at the position the preceding text reached and lays
    its tokens out on a height/width grid centred on zero, so a block's spatial
    positions do not depend on where it sits in the sequence.
    """
    tables = _rope_tables()
    frame_index: list[int] = []
    image_height_index: list[int] = []
    image_width_index: list[int] = []
    cursor, position = 0, 0
    total_len = image_pad_mask.shape[-1]
    is_image_token = image_pad_mask.tolist()

    for _, height, width in img_shapes:
        block_start = is_image_token.index(True, cursor)
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len

        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)

        image_height_index.extend(
            [h for h in range(-(height - height // 2), height // 2) for _ in range(width)]
        )
        image_width_index.extend(
            [w for _ in range(height) for w in range(-(width - width // 2), width // 2)]
        )

    if cursor < total_len:
        frame_index.extend(range(position, position + total_len - cursor))

    frame = torch.tensor(frame_index, dtype=torch.long)
    height_idx = frame.clone()
    width_idx = frame.clone()
    height_idx[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long)
    width_idx[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long)

    return torch.cat([tables[0][frame], tables[1][height_idx], tables[2][width_idx]], dim=-1)


def fold_rope_sign(sin: torch.Tensor) -> torch.Tensor:
    """`sin` with the rotation's pair-swap sign folded in: `[-s0, +s0, -s1, +s1, ...]`.

    A complex multiply on interleaved real pairs is
    `x*cos + [-x1, x0, -x3, x2, ...]*sin`. Building that rotated vector inside the
    graph costs a reshape, a split, a negate, a stack and a reshape — five ops over
    a 33 MB tensor, twice per block. Swapping the pairs instead
    (`[x1, x0, x3, x2, ...]`) is one `flip`, and the minus sign moves onto this
    host-side table, where it is free:

        [x1, x0, ...] * [-s0, +s0, ...] == [-x1, x0, ...] * [s0, s0, ...]

    Both sides multiply the same pairs of numbers, so this is bit-identical rather
    than an approximation — `sin` is already `repeat_interleave`d, so `s0` covers
    both slots of its pair. Measured: 4.07 ms -> 1.16 ms per application, and a
    decode step applies it 64 times (query and key, 32 blocks).

    The shipped packages are built with the folded table. Feeding them the raw
    imaginary part instead rotates every query and key the wrong way.
    """
    if sin.shape[-1] % 2:
        raise ValueError(f"RoPE table must have an even last dim, got {sin.shape[-1]}")
    sign = torch.tensor([-1.0, 1.0], dtype=sin.dtype, device=sin.device)
    return sin * sign.repeat(sin.shape[-1] // 2)


def text_image_layout(text_len: int, latent_h: int, latent_w: int, *, fold: bool = True) -> dict:
    """The text-to-image layout: `text_len` prompt tokens, then one target image."""
    target_tokens = latent_h * latent_w
    img_shapes = [(1, latent_h, latent_w)]
    # The vision-language sequence: False at text, True at the image slots the
    # pipeline appends, one per 2x2 group of latent tokens.
    img_mask = torch.zeros(1, text_len + target_tokens // 4, dtype=torch.bool)
    img_mask[0, text_len:] = True

    repeats = torch.where(img_mask, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
    freqs = rope_freqs(img_shapes, image_pad_mask)
    cos = torch.repeat_interleave(freqs.real, 2, dim=-1).float()
    sin = torch.repeat_interleave(freqs.imag, 2, dim=-1).float()
    if fold:
        sin = fold_rope_sign(sin)

    seq_len = int(image_pad_mask.shape[0])
    assert seq_len == text_len + target_tokens, (seq_len, text_len, target_tokens)
    return {
        "rope_form": "fold" if fold else "stack",
        "text_len": text_len,
        "target_tokens": target_tokens,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "cos": cos,
        "sin": sin,
        "cos_prefix": cos[:text_len].contiguous(),
        "sin_prefix": sin[:text_len].contiguous(),
        "cos_target": cos[text_len:].contiguous(),
        "sin_target": sin[text_len:].contiguous(),
    }


def rope_for_bucket(
    real_text_len: int, bucket: int, latent_h: int, latent_w: int, *, fold: bool = True
) -> dict:
    """RoPE tables laid out for a prompt right-padded to a fixed bucket.

    The real positions come from the *real* prompt length, so a padded slot gets
    an arbitrary (masked-out) position and the image block stays exactly where the
    unpadded model would put it. Shorten the prompt and the image's rotary
    positions do not move.
    """
    layout = text_image_layout(real_text_len, latent_h, latent_w, fold=fold)
    dim = layout["cos"].shape[-1]
    cos = torch.zeros(bucket + layout["target_tokens"], dim)
    sin = torch.zeros_like(cos)
    cos[:real_text_len] = layout["cos_prefix"]
    sin[:real_text_len] = layout["sin_prefix"]
    cos[bucket:] = layout["cos_target"]
    sin[bucket:] = layout["sin_target"]
    # Padded rows get the identity rotation rather than zeros: an all-zero cos/sin
    # would send those keys to the origin, and while they are masked out of the
    # softmax anyway, an all-zero row is a needless denormal source.
    cos[real_text_len:bucket] = 1.0
    layout.update(
        bucket=bucket,
        cos_prefix_padded=cos[:bucket].contiguous(),
        sin_prefix_padded=sin[:bucket].contiguous(),
        cos_target=cos[bucket:].contiguous(),
        sin_target=sin[bucket:].contiguous(),
    )
    return layout


# ---------------------------------------------------------------------- masks


def causal_bias(length: int, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Additive `(1, 1, L, L)` prefix mask: causal, and never attending to padding."""
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool))
    if valid is not None:
        allowed = allowed & valid.view(1, -1)
    return torch.where(allowed, 0.0, NEG)[None, None]


def decode_bias(text_len: int, target_tokens: int, valid: torch.Tensor | None = None):
    """Additive `(1, 1, 1, text_len + target_tokens)` mask for a decode step.

    Target queries see the whole prefix and their own block, so the only thing to
    exclude is right-padded prompt positions. Returns `None` when nothing is
    padded, which lets Core ML take its unmasked SDPA path.
    """
    if valid is None or bool(valid.all()):
        return None
    keep = torch.ones(text_len + target_tokens, dtype=torch.bool)
    keep[:text_len] = valid
    return torch.where(keep, 0.0, NEG).view(1, 1, 1, -1)


def pad_prompt(embeds: torch.Tensor, bucket: int):
    """Right-pad prompt embeddings to a fixed bucket, and report which rows are real.

    A Core ML model has one static shape; prompts do not. Padding to a bucket and
    masking the padded keys keeps one artifact usable for every prompt up to that
    length.
    """
    batch, length, dim = embeds.shape
    if length > bucket:
        raise ValueError(f"prompt of {length} tokens exceeds the {bucket}-token bucket")
    padded = embeds.new_zeros(batch, bucket, dim)
    padded[:, :length] = embeds
    valid = torch.zeros(bucket, dtype=torch.bool)
    valid[:length] = True
    return padded, valid


# ------------------------------------------------------------------- timestep


def time_proj(
    timestep: torch.Tensor, dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0
) -> torch.Tensor:
    """The timestep sinusoid, on the host, in fp32.

    This stays out of the Core ML graph on purpose. It is parameter-free — a
    cosine bank — but it multiplies the timestep by 1000 before taking the
    sinusoid, so a timestep that arrives already rounded to fp16 (spacing 4.9e-4
    near 1.0) becomes a phase error of up to 0.21 rad. Measured, that is a 34.9 dB
    projection: by far the worst number in the pipeline, for 256 floats of
    arithmetic. The `Embed` package therefore takes the `[-1, 1]` result instead
    of the timestep, and fp16 carries that at ~70 dB.

    It also hides from a naive parity check. The obvious example timestep, 1.0, is
    exactly representable in fp16, so a model built this way verifies clean and
    only the real schedule (0.9869..., 0.9735...) is wrong.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    )
    args = (time_factor * timestep.float())[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def latent_hw(height: int, width: int) -> tuple[int, int]:
    return height // VAE_SCALE, width // VAE_SCALE
