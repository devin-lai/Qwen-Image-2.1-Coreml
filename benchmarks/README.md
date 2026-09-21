# Measurement records

These records were collected on an M5 MacBook Pro with 32 GB of memory,
macOS 27.0, coremltools 9.0, and PyTorch 2.11. Numerical results are retained as
recorded. The local absolute output path in `optimization_study.txt` has been
shortened to a relative path.

| File | Contents |
| --- | --- |
| `step_latency_ios18.json` | Decode and VAE timings for the released recipe |
| `step_latency_ios17.json` | Timings for the earlier conversion recipe |
| `step_latency_resident.json` | Decode timings with all chunks held in memory |
| `pipeline_timing_session1.json`, `pipeline_timing_session2.json` | Stage timings for two 40-step generations |
| `precision_step_coreml.json`, `precision_step_torch.json` | One denoising step at t = 1.0 against an fp32 CPU reference |
| `precision_vae.json` | Converted VAE output against an fp32 reference |
| `consistency.json` | Repeatability and comparison with the PyTorch trajectory |
| `build_vs_build.json` | Comparison of the released and earlier Core ML builds |
| `optimization_study.json`, `optimization_study.txt` | Conversion variants tested on two or four blocks |
| `package_manifest.json` | Package input and output names, shapes, and dtypes |

`generate.py --timings` and `benchmark.py --label` can produce new runtime
measurements. The conversion study and precision harness that produced the other
records are not included in this repository.

## Interpreting the results

The saved pipeline stage timings exclude text encoding and denoiser loading.
The VAE stage includes loading; the VAE microbenchmark measures warmed-up
prediction. Do not substitute one for the other when adding up a run.

The two saved denoising runs took 221.2 s and 249.7 s. Treat this as observed
session variation and compare builds in the same session where possible.
`--hold` keeps all decode chunks resident, matching the generation loop.

The optimization study's `step` column projects full-model latency from
smaller block tests. Its quantized and Neural Engine results are not full-model
release measurements. Only the fp16 packages are distributed.

The records use Python's JSON representation of `Infinity` for zero-error PSNR.
Strict JSON parsers may need to handle that value explicitly.
