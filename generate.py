#!/usr/bin/env python3
"""Generate an image with the Core ML models.

    python generate.py --prompt-embeds assets/prompts/neon_sign.npz --out neon.png

Precomputed embeddings are included in assets/prompts/. Use encode_prompt.py to
prepare a new prompt with the upstream text encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_image_coreml import generate, load_prompt_embeds
from qwen_image_coreml.pipeline import DEFAULT_MODELS
from qwen_image_coreml.runtime import COMPUTE_UNITS

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--prompt-embeds",
        default="assets/prompts/neon_sign.npz",
        help="a .npz from encode_prompt.py (default: %(default)s)",
    )
    ap.add_argument("--out", default="output.png", help="where to write the image")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=1024, help="square side, in pixels")
    ap.add_argument("--bucket", type=int, default=64,
                    help="prompt-length bucket the packages were built for")
    ap.add_argument("--models", default=str(DEFAULT_MODELS), help="directory of .mlpackages")
    ap.add_argument("--compute-units", default="cpu_and_gpu", choices=COMPUTE_UNITS)
    ap.add_argument(
        "--prefix-cache",
        default=None,
        help="path to cache the prompt's KV tensors (67 MB); reused on later runs. "
             "Defaults to .cache/ in this checkout; 'none' recomputes each run.",
    )
    ap.add_argument("--save-latents", default=None, help="also write the raw latents as .npz")
    ap.add_argument("--timings", default=None, help="write per-stage timings as JSON")
    args = ap.parse_args()

    embeds_path = Path(args.prompt_embeds)
    embeds, prompt = load_prompt_embeds(embeds_path)
    print(f'prompt: "{prompt}"  ({embeds.shape[1]} tokens)')

    if args.prefix_cache == "none":
        cache_path = None
    elif args.prefix_cache:
        cache_path = Path(args.prefix_cache)
    else:
        cache_path = HERE / ".cache" / f"{embeds_path.stem}_prefix.npz"

    result = generate(
        embeds,
        models_dir=Path(args.models),
        steps=args.steps,
        seed=args.seed,
        height=args.size,
        width=args.size,
        bucket=args.bucket,
        compute_units=args.compute_units,
        cache_path=cache_path,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out)
    print(f"wrote {out}  ({result.timings['total']:.1f}s total)")

    if args.save_latents:
        np.savez(args.save_latents, latents=result.latents.numpy())
    if args.timings:
        Path(args.timings).write_text(json.dumps(result.timings, indent=2))


if __name__ == "__main__":
    main()
