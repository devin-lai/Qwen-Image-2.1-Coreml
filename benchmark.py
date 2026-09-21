#!/usr/bin/env python3
"""Measure decode latency per chunk and compute unit.

    python benchmark.py --hold --label local

Use --hold to keep all decode chunks resident as in the generation loop.
Compare runs in the same session where possible; timings vary with system load.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from qwen_image_coreml.layout import (
    HEAD_DIM,
    HEADS,
    LATENT_CHANNELS,
    decode_bias,
    rope_for_bucket,
)
from qwen_image_coreml.pipeline import DEFAULT_MODELS
from qwen_image_coreml.runtime import load_mlmodel, n_inputs, open_chunk, predict

HERE = Path(__file__).resolve().parent


def time_predict(model, args, repeats: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        predict(model, *args)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        predict(model, *args)
        samples.append(time.perf_counter() - started)
    samples = np.array(samples)
    return {
        "median_ms": float(np.median(samples) * 1000),
        "p10_ms": float(np.percentile(samples, 10) * 1000),
        "p90_ms": float(np.percentile(samples, 90) * 1000),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=str(DEFAULT_MODELS))
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--bucket", type=int, default=64, help="prompt-length bucket the model was built for")
    ap.add_argument("--size", type=int, default=1024, help="square side, in pixels")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--units", default="cpu_and_gpu")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--hold", action="store_true", help="also time every chunk with all of them resident")
    ap.add_argument("--label", default=None, help="write benchmarks/bench_<label>.json")
    args = ap.parse_args()

    models_dir = Path(args.models)
    latent = args.size // 16
    layout = rope_for_bucket(args.bucket, args.bucket, latent, latent)
    target = layout["target_tokens"]
    dbias = torch.zeros(1, 1, 1, args.bucket + target)
    if (explicit := decode_bias(args.bucket, target, None)) is not None:
        dbias = explicit

    # Random inputs of the right shape: latency does not depend on the values,
    # and this keeps the benchmark independent of any prompt or checkpoint.
    torch.manual_seed(0)
    latents = torch.randn(1, target, LATENT_CHANNELS)
    hidden = torch.randn(1, target, 4096)
    temb = torch.randn(1, 4096) * 0.1
    modulation = torch.randn(1, 16384) * 0.1
    per_chunk = 32 // args.chunks
    cache_k = torch.randn(per_chunk, 1, args.bucket, HEADS, HEAD_DIM) * 0.5
    cache_v = torch.randn(per_chunk, 1, args.bucket, HEADS, HEAD_DIM) * 0.5

    def feed_for(model, i):
        feed = [
            latents if i == 0 else hidden,
            modulation,
            temb,
            layout["cos_target"],
            layout["sin_target"],
            cache_k,
            cache_v,
        ]
        if n_inputs(model) == 8:
            feed.append(dbias)
        return feed

    results: dict = {"target_tokens": target, "bucket": args.bucket, "chunks": args.chunks}
    for units in (u.strip() for u in args.units.split(",")):
        print(f"\n=== compute units: {units} ===")
        per_unit: dict = {}
        total, ok = 0.0, True
        for i in range(args.chunks):
            model = None
            try:
                started = time.perf_counter()
                model = open_chunk(models_dir, i, args.chunks, "decode", units)
                load_s = time.perf_counter() - started
                stats = time_predict(model, feed_for(model, i), args.repeats)
                stats["load_s"] = load_s
                per_unit[f"decode_{i}"] = stats
                total += stats["median_ms"]
                print(
                    f"  Blocks_{i}of{args.chunks}: {stats['median_ms']:7.1f} ms "
                    f"(p10 {stats['p10_ms']:.0f} / p90 {stats['p90_ms']:.0f}), load {load_s:.1f}s"
                )
            except FileNotFoundError as exc:
                print(f"  chunk {i}: {exc}")
                ok = False
            except Exception as exc:
                # A compute unit the runtime refuses is a result, not a crash.
                print(f"  Blocks_{i}of{args.chunks}: FAILED on {units}: {type(exc).__name__}: {exc}")
                per_unit[f"decode_{i}"] = {"error": f"{type(exc).__name__}: {exc}"}
                ok = False
            finally:
                del model
                gc.collect()

        if ok and args.hold:
            print("  with all chunks resident:")
            held = [open_chunk(models_dir, i, args.chunks, "decode", units) for i in range(args.chunks)]
            resident = 0.0
            for i, model in enumerate(held):
                stats = time_predict(model, feed_for(model, i), args.repeats)
                per_unit[f"decode_{i}_resident"] = stats
                resident += stats["median_ms"]
                print(f"    Blocks_{i}of{args.chunks}: {stats['median_ms']:7.1f} ms")
            per_unit["step_total_resident_ms"] = resident
            print(f"  -> one step, all chunks resident: {resident:.1f} ms")
            held.clear()
            gc.collect()

        if ok:
            per_unit["step_total_ms"] = total
            print(f"  -> one denoising step: {total:.1f} ms  ({40 * total / 1000:.1f}s for 40 steps)")
        results[units] = per_unit

        if not args.skip_vae:
            vae_path = models_dir / f"QwenImage21_VAEDecoder_{args.size}x{args.size}.mlpackage"
            if vae_path.exists():
                model = None
                try:
                    model = load_mlmodel(vae_path, units)
                    stats = time_predict(
                        model, [torch.randn(1, LATENT_CHANNELS, latent, latent)], 3, warmup=1
                    )
                    results[units]["vae"] = stats
                    print(f"  VAE decode {args.size}px: {stats['median_ms']:.0f} ms")
                except Exception as exc:
                    print(f"  VAE decode: FAILED on {units}: {type(exc).__name__}")
                finally:
                    del model
                    gc.collect()

    out_dir = HERE / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"bench_{args.label}.json" if args.label else "bench.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
