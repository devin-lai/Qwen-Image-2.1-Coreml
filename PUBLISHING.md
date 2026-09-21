# Repository maintenance

There are two independent repositories:

- GitHub: `devin-lai/Qwen-Image-2.1-Coreml`, containing code, sample assets, and
  benchmark records.
- Hugging Face: `devin-lai/Qwen-Image-2.1-Coreml`, containing six `.mlpackage`
  directories, the model card, license, notices, and file checksums.

The local Hugging Face checkout is nested under `Qwen-Image-2.1-Coreml/` and is
ignored by the outer repository. It is not a submodule. The normal download
location, `models/`, is also ignored.

## GitHub

Run the host tests and lint before committing:

```bash
python -m pip install -e '.[test,torch-reference]'
python -m pytest tests/ -v
ruff check .
```

With the local model checkout available, check inference as well:

```bash
python verify.py --models ./Qwen-Image-2.1-Coreml --steps 4
```

Keep weights, compiled models, credentials, prompt KV caches, and generated
outputs out of this repository. The samples under `assets/` are intentional.

### Discovery and sharing

Keep the GitHub About section aligned with the README and the measured release:

- Description: `Run Qwen-Image-2.1 locally on Apple silicon Macs with Core ML. 1024x1024 text-to-image, FP16 models and Python inference. Measured 2.4-2.6x faster denoising steps vs PyTorch bf16/MPS on M5 (32 GB). Preconverted weights on Hugging Face.`
- Website: `https://huggingface.co/devin-lai/Qwen-Image-2.1-Coreml`
- Topics: `qwen`, `qwen-image`, `qwen-image-2-1`, `coreml`, `core-ml`,
  `coremltools`, `apple-silicon`, `macos`, `text-to-image`, `image-generation`,
  `generative-ai`, `diffusion-models`, `on-device-ai`, `local-ai`,
  `model-conversion`, `python`.

[GitHub topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics)
help people discover related projects. Use topics for capabilities this release
actually provides; revise them when supported platforms or features change.

The social preview is [`assets/social-preview.png`](assets/social-preview.png),
with editable source in [`assets/social-preview.svg`](assets/social-preview.svg).
It is 1280 × 640 and under 1 MB. Upload the PNG under repository **Settings →
General → Social preview**; committing the image alone does not set the preview.
GitHub documents this setting in its
[social preview guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).
With librsvg installed, regenerate the PNG after changing the SVG:

```bash
rsvg-convert assets/social-preview.svg -o assets/social-preview.png
```

Keep the benchmark hardware, baseline precision, and denoising-only qualifier
in descriptions and preview images. The release does not establish an
end-to-end speedup or performance on other Macs. Update the English README,
Chinese quickstart, and preview together when those claims change.

For a project announcement, start with a real gallery output and this factual
summary, adjusting it for the community where it will be shared:

> Run Qwen-Image-2.1 locally on Apple silicon with Core ML. Includes preconverted
> FP16 models, a Python CLI, four reproducible sample prompts, and raw benchmark
> records. On an M5 MacBook Pro with 32 GB, median denoising steps were 2.4–2.6×
> faster than PyTorch bf16/MPS. Looking for hardware reports from other Macs.
> Code: Apache-2.0; weights: Qwen Research License, non-commercial research and
> evaluation. https://github.com/devin-lai/Qwen-Image-2.1-Coreml

## Hugging Face

Git LFS stores `*.bin` and `*.mlmodel`. The package manifests and documentation
remain ordinary Git files. Stage and commit from inside the model checkout so
the two histories stay separate.

When updating packages, keep their Core ML author, license, source, and
modification metadata current. Retain the upstream license and `Notice` files,
and regenerate `SHA256SUMS` for the distributed package files. Validate them
from the root of the model checkout:

```bash
shasum -a 256 -c SHA256SUMS
git lfs fsck
```

The model card uses `license: other` and links to its local Qwen Research
License. The weights are restricted to non-commercial research and evaluation
unless a separate commercial license is obtained. The license also restricts
use of Qwen as the primary name of derivative products; retain descriptive
wording such as "Core ML conversion of Qwen-Image-2.1" and review the repository
identifier before publication.

A model update should include a fresh inference check. Exact latent agreement
is a numerical reference check, not a file-integrity check; use `SHA256SUMS` for
the latter.

Publishing is a separate step. Local commits do not upload Git or LFS objects.
