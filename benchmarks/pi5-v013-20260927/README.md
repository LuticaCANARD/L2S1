# Raspberry Pi 5: published L2S1 v0.1.3 validation

The official `l2s1-runtime-linux-arm64@0.1.3` bundle and public
`l2s1-sdk==0.1.3` wheel ran real Gemma 3 1B Q8 inference on a physical
Raspberry Pi 5 with 4 GB RAM and Debian 13 ARM64 on 2026-09-27.
No source compilation, system package changes, or service changes were needed.
The existing model and earlier installation were preserved.

Release commit: `57a73657bd9ad4d97b29e12d1400b304839d0d27`.
Runtime archive SHA-256:
`1bc1639be1a83bab96dcac2de86cd87ed50c4065d3cbe17375237188cc5c3f43`.
The archive was checked against the published `release.json`; the SDK also
validated the extracted runtime manifest before launching it.

## Measured result

| Metric | Fresh | Fixed-schema prefix reuse |
| --- | ---: | ---: |
| Request p50, three decisions | 16.874 s | 4.091 s |
| Request p95 (six samples only) | 19.194 s | 7.470 s |
| Peak native process RSS | 1.124 GiB | 1.133 GiB |
| Maximum system swap used | 0 MiB | 0 MiB |
| Reused prefix tokens, measured requests | 0 | 1,829 / 2,394 input tokens |
| Raw top-1 | 55.6% | 55.6% |
| Coverage | 88.9% | 88.9% |
| Accepted-only accuracy | 62.5% | 62.5% |
| Active undervoltage samples | 59 / 122 | 31 / 44 |
| Active throttling samples | 59 / 122 | 31 / 44 |

Observed p50 ratio: **4.125x**. All nine unique decisions had exactly equal
evidence between modes, including candidate scores. Each mode also reproduced
its evidence exactly across repeated inputs (nine repeat comparisons).

## Scope and limits

This is a small installation and reuse smoke benchmark: the first three
warehouse cases of `decision-rules-v1`, two measured passes per mode, six requests
and 18 decisions per mode. Repeats are not independent quality evidence.
Each mode excludes loading and one initial warmup request. Latency includes the
local Python SDK / stdio round trip. Settings: context 2048, batch/ubatch 256,
four CPU threads, one resident process at a time. Fresh ran before fixed reuse;
this was not a randomized or power-controlled comparison.

**Active undervoltage and throttling occurred in both modes.** The result describes
this power setup, not unthrottled Pi performance. Maximum sampled temperature was
67.75 C. Telemetry is sampled approximately once per second; short events may be
missed. Memory uses the native child's reported process high-water RSS.

Both modes use the same published SIMD-enabled binary. This measurement does
not isolate SIMD speedup. It also does not establish a regression against the
older Pi source-build benchmark: binary build settings, source, cases and power
conditions differ. Gemma 3's low accuracy in this small sample does not establish
suitability for unattended decisions.

## Evidence and execution notes

`verified-summary.json` is the authoritative summary, regenerated directly from
`fresh-responses.json`, `fixed-responses.json`, and their telemetry files with
`report.py`. `identity.json` binds the model, suite and native runtime. Models,
binaries and Python dependencies are not included here.

The initial full-suite attempt was stopped after its warmup because it was too
slow for this installation check; its logs are in `stopped-full-suite/` and are
excluded from the measured comparison. The short fresh run completed and saved
all responses, but its original summary code used `option_id` instead of the
SDK's score `id`. That reporting error is preserved in `benchmark.log`.
`benchmark-fixed.py` corrected the field and ran the fixed mode. `report.py`
reconstructed both summaries without altering or repeating the native predictions.
The original executed scripts are retained for provenance.

The isolated installation remains at
`/home/lutica/l2s1-v013-validation-20260927` on the tested Pi, including the venv,
verified runtime bundle, scripts and results. The benchmark's native processes
were closed on completion. No password is stored in this evidence.

## Retrospective comparison with the earlier Pi run

The earlier 12-case suite reported 4.766 s p50. Restricting its raw samples to
exactly warehouse-01/02/03 and passes 0/1 matches this smoke run's six requests:

| Three-case subset | Earlier source, fresh | v0.1.3 portable, fresh | v0.1.3 portable, reuse | Current baseline, fresh | Current kernels, fresh | Current OpenMP, fresh | Current combined, fresh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p50 / three-decision request | 4.743 s | 16.874 s | 4.091 s | 15.643 s | 4.450 s | 15.685 s | 4.459 s |
| Raw top-1 | 55.6% | 55.6% | 55.6% | 55.6% | 55.6% | 55.6% | 55.6% |
| Coverage | 100% | 88.9% | 88.9% | 88.9% | 100.0% | 88.9% | 100.0% |
| Accepted-only accuracy | 55.6% | 62.5% | 62.5% | 62.5% | 55.6% | 62.5% | 55.6% |

Historical columns contain six requests each. Current columns contain 12 requests each from a same-source four-build crossover, with identical stdio, three-case warmup, compute settings and explicit fresh execution. Active undervoltage/throttling remained present. The current columns are controlled against each other at the software level, not against the historical columns. See the [fresh kernel investigation](../pi5-arm64-fresh-20260927/README.md) for p95, memory, power/clock telemetry, all 36 decisions, numerical changes and regression scope.

The portable fresh path was **3.56x slower** than the older build. With reuse,
latency was 13.8% lower than that historical fresh result. The 4.125x ratio is
strictly a comparison within v0.1.3; it is not improvement over the earlier build.
Same model and suite SHA-256 hashes, host/kernel, threads/context/batch and input
token counts were confirmed. Native implementation, compiler/kernel selection,
transport and warmup still differ, and neither measurement controlled the power
supply. This is observational evidence, not an isolated build regression test.

A read-only inspection of the older Pi build confirmed `GGML_NATIVE=ON`,
`GGML_OPENMP=ON`, and `-mcpu=cortex-a76+crc+crypto+dotprod+noi8mm+nosve+nosme`.
The v0.1.3 portable release configuration explicitly turns `GGML_NATIVE` and
`GGML_OPENMP` off. This makes lost CPU-specific optimization a strong hypothesis;
the [same-source follow-up](../pi5-arm64-fresh-20260927/README.md) now isolates the ARM kernel configuration and OpenMP as separate software factors. Power remains uncontrolled, so it does not establish a sole cause for the exact historical ratio. The new Rust NEON evidence scan does not replace the
GGML matrix kernels that dominate inference.

All nine unique top-1 predictions match the older build, but probability values
differ. In warehouse-02/dispatch_priority, the incorrect `low` prediction changed
from 0.9134 to 0.6658 and now abstains at the same 0.8 threshold. Thus the higher
accepted-only accuracy is filtering an error, not higher raw prediction accuracy.
Historical process RSS (2.10 GiB) was a whole-suite peak from another executable;
it cannot establish a matched-subset 46% memory reduction.

See `historical-comparison.json` and the original
`../pi5-gemma3-20260927/runs/gemma3-q8-rules/benchmark.json` for the source samples.
