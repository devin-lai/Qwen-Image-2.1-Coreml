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
| `pytorch_denoise_timing.json` | PyTorch bf16/MPS denoising baseline, with KV caching enabled |
| `weight_sharing.json` | Transformer package sizes before and after merging prefix/decode functions |
| `precision_step_coreml.json`, `precision_step_torch.json` | One denoising step at t = 1.0 against an fp32 CPU reference |
| `precision_vae.json` | Converted VAE output against an fp32 reference |
| `consistency.json` | Repeatability and comparison with the PyTorch trajectory |
| `build_vs_build.json` | Comparison of the released and earlier Core ML builds |
| `optimization_study.json`, `optimization_study.txt` | Conversion variants tested on two or four blocks |
| `package_manifest.json` | Package input and output names, shapes, and dtypes |

`generate.py --timings` and `benchmark.py --label` can produce new runtime
measurements. The conversion study and precision harness that produced the other
records are not included in this repository.

## Performance summary

The opening tables in the GitHub README and Hugging Face model card use these
records. The PyTorch baseline and weight-sharing record were copied unchanged
from the conversion study's `logs/ref_latents_timing.json` and `logs/merge.json`.
The study's README supplied the initial comparisons; the values below are
calculated from the saved measurements.

| Published metric | Calculation and source |
| --- | --- |
| 2.4–2.6× faster median denoising steps | 14.254 s in [the PyTorch baseline](pytorch_denoise_timing.json), divided by 5.434 s and 5.949 s in [Core ML session 1](pipeline_timing_session1.json) and [session 2](pipeline_timing_session2.json), gives 2.62× and 2.40×. |
| 221–250 s for 40 denoising steps | The two Core ML `denoise_total` values are 221.213 s and 249.667 s, or 3.69 and 4.16 minutes. |
| 1.91 s VAE prediction | The warmed median is 1908.002 ms in [the resident-model benchmark](step_latency_resident.json). Model loading is excluded. |
| 15.8 dB higher noise-prediction PSNR | 71.554 dB in [Core ML precision](precision_step_coreml.json) minus 55.797 dB for bf16 in [PyTorch precision](precision_step_torch.json). Both compare the same one-step output with an fp32 CPU reference at `t = 1.0`. |
| About 6.1× lower RMS error | With the same reference peak, the RMS error ratio is `10 ** ((71.554 - 55.797) / 20) = 6.14`. This is a numerical precision result for that input, not a perceptual image-quality score. |
| 50% less transformer storage | Summing [the four merge records](weight_sharing.json) gives 28.0205 GB for separate prefix/decode packages and 14.0617 GB for merged packages: a 49.82% reduction without weight quantization. |
| 14.74 GB total download | The transformer chunks, timestep embedding, and VAE decoder listed in [the package manifest](package_manifest.json), rounded to two decimal places. All sizes use decimal GB. |

The latency comparison uses medians from separate sessions on the same M5 Mac.
It is an observed ratio, not a controlled back-to-back comparison. The PyTorch
baseline uses eager bf16 execution on MPS with KV caching; Core ML uses fp16 on
`cpu_and_gpu`. This compares those execution paths, including their precision
and graph differences.

The 40-step Core ML times exclude the prompt prefix, model loading, and text
encoding. Prefix computation took 27.90 s and 33.34 s in the two sessions and can
be cached for reuse. The PyTorch loop extracts its prefix on the first step, so
its 577.63 s total has a different boundary; the headline speedup uses the
median steps instead. No total command wall-time speedup is claimed.

The older study's 17× VAE comparison used a 32.3 s fp32 CPU stage that included
loading, against a warmed fp16 GPU prediction. The opening table therefore
reports the directly measured Core ML latency without that multiplier.

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
