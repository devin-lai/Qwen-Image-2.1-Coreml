"""Text-to-image inference with a cached prompt prefix and four decode chunks.

The prefix uses t = 0 and cannot attend to target-image tokens. Its KV tensors
can be reused across denoising steps for the same prompt.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .layout import (
    LATENT_CHANNELS,
    causal_bias,
    decode_bias,
    latent_hw,
    pad_prompt,
    rope_for_bucket,
)
from .runtime import Denoiser, load_mlmodel, predict, prefix_cache
from .scheduler import FlowMatchEulerScheduler

DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models"


@dataclass
class Result:
    image: Image.Image
    latents: torch.Tensor
    timings: dict = field(default_factory=dict)


def load_prompt_embeds(path: Path) -> tuple[torch.Tensor, str]:
    """Read a `*_embeds.npz` written by `encode_prompt.py`."""
    with np.load(path) as data:
        embeds = torch.from_numpy(data["prompt_embeds"]).float()
        prompt = str(data["prompt"][0]) if "prompt" in data else ""
    return embeds, prompt


def unpack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`(1, h*w, 64)` -> `(1, 64, h, w)`, the pipeline's `_unpack_latents`."""
    batch, _, channels = latents.shape
    return latents.transpose(1, 2).reshape(batch, channels, h, w)


def initial_latents(seed: int, latent_h: int, latent_w: int) -> torch.Tensor:
    """Seeded noise, packed to `(1, tokens, 64)`.

    Draw and pack the noise in the same order as the reference pipeline.
    """
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(
        1, 1, LATENT_CHANNELS, latent_h, latent_w, generator=generator, dtype=torch.float32
    )
    # `_pack_latents`, made contiguous because a transposed view reaches Core ML
    # as strided data its Metal backend rejects with an abort, not an exception.
    return latents.view(1, LATENT_CHANNELS, latent_h * latent_w).transpose(1, 2).contiguous()


def generate(
    embeds: torch.Tensor,
    *,
    models_dir: Path = DEFAULT_MODELS,
    steps: int = 40,
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
    bucket: int = 64,
    chunks: int = 4,
    compute_units: str = "cpu_and_gpu",
    cache_path: Path | None = None,
    progress: bool = True,
) -> Result:
    """Run the full text-to-image path and return the decoded image."""
    started_total = time.perf_counter()
    models_dir = Path(models_dir)
    if (height, width, bucket, chunks) != (1024, 1024, 64, 4):
        raise ValueError("the released models require 1024x1024, bucket=64, and chunks=4")
    if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != 4096:
        raise ValueError("prompt embeddings must have shape (1, tokens, 4096)")
    if not 1 <= embeds.shape[1] <= bucket:
        raise ValueError(f"prompt must contain 1 to {bucket} tokens; got {embeds.shape[1]}")
    if not torch.isfinite(embeds).all():
        raise ValueError("prompt embeddings must be finite")
    embeds = embeds.detach().cpu().float()
    latent_h, latent_w = latent_hw(height, width)
    scheduler = FlowMatchEulerScheduler.make(steps, latent_h * latent_w)
    text_len = embeds.shape[1]
    # `bucket` must match the bucket the packages were converted for — growing it
    # to fit a long prompt would just fail later, inside Core ML, on a shape
    # mismatch. `pad_prompt` raises here instead, naming both numbers.
    layout = rope_for_bucket(text_len, bucket, latent_h, latent_w, fold=True)
    target = layout["target_tokens"]
    padded, valid = pad_prompt(embeds, bucket)
    dbias = decode_bias(bucket, target, valid)
    if dbias is None:
        dbias = torch.zeros(1, 1, 1, bucket + target)
    pbias = causal_bias(bucket, valid)

    timings: dict[str, float] = {}

    # ---- prefix: the prompt's KV cache, once ------------------------------
    started = time.perf_counter()
    print("[prefix] ...", flush=True)
    cache_k, cache_v = prefix_cache(
        models_dir,
        chunks,
        compute_units,
        padded,
        pbias,
        layout["cos_prefix_padded"],
        layout["sin_prefix_padded"],
        cache_path,
    )
    timings["prefix"] = time.perf_counter() - started
    print(
        f"[prefix] {timings['prefix']:.1f}s  kv cache {tuple(cache_k.shape)} "
        f"{cache_k.dtype}  {cache_k.nbytes / 1e6:.0f} MB x2",
        flush=True,
    )

    # ---- denoise ----------------------------------------------------------
    latents = initial_latents(seed, latent_h, latent_w)

    started = time.perf_counter()
    denoiser = Denoiser(models_dir, chunks, compute_units)
    denoiser.bind(layout["cos_target"], layout["sin_target"], cache_k, cache_v, dbias)
    timings["denoiser_load"] = time.perf_counter() - started

    step_times = []
    started = time.perf_counter()
    print(f"[denoise] {steps} steps ...", flush=True)
    for index, t_value in enumerate(scheduler.timesteps):
        t0 = time.perf_counter()
        timestep = (t_value.expand(1) / 1000).to(torch.float32)
        noise = denoiser.step(latents, timestep)
        latents = scheduler.step(noise, index, latents)
        step_times.append(time.perf_counter() - t0)
        if progress and index % 10 == 0:
            print(
                f"    step {index:>3}  {step_times[-1]:.2f}s  "
                f"latents |x|max {latents.abs().max():.3f}",
                flush=True,
            )
    timings["denoise_total"] = time.perf_counter() - started
    timings["step_median"] = float(np.median(step_times))
    print(
        f"[denoise] {timings['denoise_total']:.1f}s  "
        f"median step {timings['step_median']:.2f}s",
        flush=True,
    )

    del denoiser
    gc.collect()

    unpacked = unpack_latents(latents, latent_h, latent_w)

    # ---- decode -----------------------------------------------------------
    started = time.perf_counter()
    print("[vae] ...", flush=True)
    vae = load_mlmodel(
        models_dir / f"QwenImage21_VAEDecoder_{height}x{width}.mlpackage", compute_units
    )
    pixels = np.asarray(predict(vae, unpacked)[0])
    del vae
    gc.collect()
    timings["vae"] = time.perf_counter() - started
    print(f"[vae] {timings['vae']:.1f}s", flush=True)

    rgba = np.clip(pixels[0].transpose(1, 2, 0), 0, 1)
    image = Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), mode="RGBA")
    timings["total"] = time.perf_counter() - started_total
    return Result(image=image, latents=unpacked, timings=timings)
