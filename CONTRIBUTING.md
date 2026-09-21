# Contributing

Help make local image generation with Core ML easier to reproduce on more Macs.
Useful contributions include hardware measurements, installation fixes,
documentation improvements, and examples with reproducible settings.

## Share a hardware benchmark

Use the [hardware report form](https://github.com/devin-lai/Qwen-Image-2.1-Coreml/issues/new?template=benchmark_report.yml).
Successful runs and failures on untested configurations are both useful.

After following the [quickstart](README.md#quickstart), run:

```bash
python generate.py --steps 40 --seed 42 --out output.png --timings timings.json
python benchmark.py --hold --label local
```

Attach `timings.json` and `benchmarks/bench_local.json`, along with:

- Mac model, chip, unified memory, and macOS version.
- Python, coremltools, and PyTorch versions; repository commit and model revision.
- The exact commands, prompt, compute units, and whether the prefix cache was warm.
- Other workloads, power mode, and any observed memory pressure or swap.

Keep denoising latency, prefix computation, model loading, and total command
wall time distinct. Compare alternatives on the same machine with the same
prompt, seed, steps, and compute settings. Include raw measurements, not just a
speedup ratio. [Existing records](benchmarks/README.md) show the context for
the published results.

## Report a bug or propose an improvement

Search [existing issues](https://github.com/devin-lai/Qwen-Image-2.1-Coreml/issues)
first. Use the [bug report form](https://github.com/devin-lai/Qwen-Image-2.1-Coreml/issues/new?template=bug_report.yml)
for failures, or open a blank issue for an idea. Include the smallest command
that reproduces the problem, the error, and your environment. Remove credentials
and private paths from logs before sharing.

## Submit a pull request

Fork the repository, create a branch, and keep each pull request focused on one
change. Explain the problem, the resulting behavior, and how you verified it.
For code changes, run the host tests and lint:

```bash
python -m pip install -e '.[test,torch-reference]' ruff
python -m pytest tests/ -v
ruff check .
```

Host tests do not need model weights. For inference changes, also run
`python verify.py --steps 4` on a supported Mac with the models installed and
include the result. For documentation changes, check links, commands, and
GitHub rendering. Keep English and [Chinese quickstart](README.zh-CN.md)
instructions consistent when changing installation or supported features.

Do not commit model weights, compiled packages, prompt KV caches, or local
credentials. Add gallery samples deliberately with their prompt, seed, steps,
hardware, and model revision. Conversion scripts live in
[torch2coreml](https://github.com/devin-lai/torch2coreml); this repository contains
inference code and measurement records.

## Attribution

Retain the code and model license distinctions described in
[the README](README.md#license-and-attribution). Contributions to the inference
code use this repository's Apache-2.0 license; converted weights remain subject
to the Qwen Research License. Preserve notices for any adapted third-party code.
