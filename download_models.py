#!/usr/bin/env python3
"""Fetch the converted `.mlpackage`s into `models/`.

    python download_models.py                 # everything, 14.7 GB
    python download_models.py --only vae      # just the VAE decoder, 0.5 GB

Packages and their license files are downloaded from Hugging Face. Re-running
the command resumes interrupted transfers and checks the requested revision.
Set `QWEN_COREML_REPO` to use a different model repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = os.environ.get("QWEN_COREML_REPO", "devin-lai/Qwen-Image-2.1-Coreml")

PACKAGES = {
    "blocks": [f"QwenImage21_Blocks_{i}of4.mlpackage" for i in range(4)],
    "embed": ["QwenImage21_Embed.mlpackage"],
    "vae": ["QwenImage21_VAEDecoder_1024x1024.mlpackage"],
}
SIZES_GB = {"blocks": 14.06, "embed": 0.17, "vae": 0.51}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="hub repo id (default: %(default)s)")
    ap.add_argument("--dest", default=str(HERE / "models"))
    ap.add_argument(
        "--only",
        nargs="+",
        choices=sorted(PACKAGES),
        default=sorted(PACKAGES),
        help="which groups to fetch (default: all)",
    )
    ap.add_argument("--revision", default=None, help="a branch, tag or commit on the hub repo")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub") from None

    wanted = [name for group in args.only for name in PACKAGES[group]]
    total = sum(SIZES_GB[group] for group in args.only)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    present_gb = sum(
        path.stat().st_size
        for name in wanted
        for path in (dest / name).rglob("*")
        if path.is_file()
    ) / 1e9
    remaining_gb = max(0.0, total - present_gb)
    free_gb = shutil.disk_usage(dest).free / 1e9
    print(f"checking {len(wanted)} package(s), ~{total:.1f} GB, in {dest}")
    print(f"  {free_gb:.0f} GB free on this volume")
    if free_gb < remaining_gb * 1.1:
        raise SystemExit(
            f"not enough space: need ~{remaining_gb:.1f} GB more plus headroom, "
            f"{free_gb:.0f} GB free"
        )

    # Let the Hub client check every file. A directory can exist after an
    # interrupted transfer and is not evidence that its weights are complete.
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(dest),
        allow_patterns=[f"{name}/**" for name in wanted] + ["LICENSE", "Notice", "SHA256SUMS"],
    )

    for name in wanted:
        path = dest / name
        for relative in (
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin",
        ):
            item = path / relative
            if not item.is_file() or item.stat().st_size == 0:
                raise SystemExit(f"incomplete package: {item}; retry the download")
        print(f"  ok {name}")
    print("\n  python generate.py --out output.png")


if __name__ == "__main__":
    main()
