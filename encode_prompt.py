#!/usr/bin/env python3
"""Encode a text prompt for generate.py using the upstream Qwen3-VL encoder.

    python encode_prompt.py "a red fox in the snow" --name fox

Requires the optional torch-reference dependencies and the upstream text-encoder
weights. Saved embeddings can be reused without loading the encoder again.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = os.environ.get("QWEN_IMAGE_21_CHECKPOINT", "Qwen/Qwen-Image-2.1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt", help="the text to encode")
    ap.add_argument("--name", default=None, help="output stem (default: derived from the prompt)")
    ap.add_argument("--out-dir", default=str(HERE / "assets" / "prompts"))
    ap.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="local path or hub id of the PyTorch model (default: %(default)s)",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    from diffusers import QwenImage21Pipeline

    started = time.perf_counter()
    print(f"loading the text encoder from {args.checkpoint} ...", flush=True)
    # vae/transformer/scheduler are not needed to encode a prompt, and passing
    # None keeps 15 GB of weights this process would otherwise load off the heap.
    pipe = QwenImage21Pipeline.from_pretrained(
        args.checkpoint,
        vae=None,
        transformer=None,
        scheduler=None,
        dtype=getattr(torch, args.dtype),
        low_cpu_mem_usage=True,
    )
    pipe.text_encoder.to(args.device).eval()
    print(f"  loaded in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    with torch.no_grad():
        embeds, mask, image_pad_mask = pipe.encode_prompt(
            prompt=[args.prompt], device=torch.device(args.device)
        )
    print(f"  encoded in {time.perf_counter() - started:.1f}s", flush=True)

    embeds = embeds.float().cpu().numpy().astype(np.float32)
    mask_np = (
        np.ones(embeds.shape[:2], dtype=np.bool_)
        if mask is None
        else mask.cpu().numpy().astype(np.bool_)
    )

    if embeds.shape[1] > 64:
        raise SystemExit(
            f"prompt is {embeds.shape[1]} tokens; the shipped packages are built for a "
            "64-token bucket. Shorten it, or convert a package with a larger bucket."
        )

    name = args.name or "_".join(args.prompt.lower().split()[:4]).strip(".,\"'")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez(
        path,
        prompt_embeds=embeds,
        prompt_embeds_mask=mask_np,
        image_pad_mask=image_pad_mask.cpu().numpy().astype(np.bool_),
        prompt=np.array([args.prompt]),
        dtype=np.array([args.dtype]),
    )
    print(f"wrote {path}  ({embeds.shape[1]} tokens, {path.stat().st_size / 1e6:.1f} MB)")
    print(f"\n  python generate.py --prompt-embeds {path} --out {name}.png")


if __name__ == "__main__":
    main()
