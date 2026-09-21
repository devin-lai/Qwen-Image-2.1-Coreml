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
