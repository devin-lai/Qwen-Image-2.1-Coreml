#!/usr/bin/env python3
"""Check that this install reproduces the reference sample.

    python verify.py

Regenerates the neon-sign prompt at 1024x1024 with seed 42 and compares latents
with `assets/reference/`. Exact agreement was measured on the reference M5;
other hardware or runtime versions may differ. The VAE runs but its pixels are
not compared. Use `--steps 4` for the shorter reference job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qwen_image_coreml import generate, load_prompt_embeds
from qwen_image_coreml.pipeline import DEFAULT_MODELS
from qwen_image_coreml.runtime import COMPUTE_UNITS

HERE = Path(__file__).resolve().parent


def psnr(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    if reference.shape != actual.shape:
        return float("-inf")
    if not np.isfinite(actual).all():
        return float("-inf")
    mse = float(np.mean((reference - actual) ** 2))
    if mse == 0.0:
        return float("inf")
    peak = float(np.max(np.abs(reference)))
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse) if peak else float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=str(DEFAULT_MODELS))
    ap.add_argument("--steps", type=int, default=40, help="must match a shipped reference")
    ap.add_argument("--compute-units", default="cpu_and_gpu", choices=COMPUTE_UNITS)
    ap.add_argument("--out", default=None, help="also write the generated image here")
    args = ap.parse_args()

    reference_path = HERE / "assets" / "reference" / f"neon_sign_seed42_{args.steps}steps.npz"
    if not reference_path.exists():
        have = sorted(p.name for p in (HERE / "assets" / "reference").glob("*.npz"))
        raise SystemExit(f"no reference for --steps {args.steps}; have: {', '.join(have)}")

    embeds, prompt = load_prompt_embeds(HERE / "assets" / "prompts" / "neon_sign.npz")
    print(f'reference job: "{prompt}"  seed 42, {args.steps} steps, 1024x1024\n')

    result = generate(
        embeds,
        models_dir=Path(args.models),
        steps=args.steps,
        seed=42,
        compute_units=args.compute_units,
        cache_path=None,
    )
    if args.out:
        result.image.save(args.out)
        print(f"wrote {args.out}")

    with np.load(reference_path) as data:
        expected = data["latents"]
    actual = result.latents.numpy()

    if actual.shape != expected.shape:
        raise SystemExit(f"\nFAIL  shape {actual.shape}, expected {expected.shape}")

    max_abs = float(np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max())
    print("\n" + "-" * 56)
    print(f"  latents   {actual.shape}")
    print(f"  max |diff| {max_abs:.3e}   PSNR {psnr(expected, actual):.2f} dB")
    if max_abs == 0.0:
        print("\nPASS  bit-identical to the reference.")
        return
    print(
        "\nFAIL  latents do not match the reference exactly.\n"
        "      Check the model revision and --compute-units (reference: cpu_and_gpu).\n"
        "      Different hardware, macOS, or runtime versions can also change results."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
