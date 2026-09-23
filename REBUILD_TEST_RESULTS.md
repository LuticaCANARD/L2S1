# Rebuild and GPT-OSS test results

Completed: 2026-09-22 (Asia/Seoul). Tested the current working source with GPT-OSS-20B MXFP4 and the local RTX 3080 CUDA runtime. Model context 2,048; batch 256; four threads.

## Build and automated checks

- Removed this package’s release artifacts, recompiled the Rust/C++ package, rebuilt CLI and examples, and reran release tests. All passed.
- The four recompiled executables matched the archived executables byte for byte (SHA256).
- Default-feature Rust tests: 11 passed. Llama-feature release tests: 19 passed. Python evaluator/runner tests: 8 passed.
- Formatting and all-target release Clippy (`-D warnings`) passed. External llama.cpp Jinja headers emitted unused-function C++ warnings.
- Real-model CUDA native test: passed. It explicitly checks request isolation, A–Z token mapping, chunking at batches 32/512 with the unchanged 0.02 probability tolerance, invalid profiles, and overlong input.

## State-first prefix reuse

- CUDA test exit code: 0.
- Fresh/reuse comparisons: 45; changed raw top-1: 0; changed accepted selections: 0.
- Maximum option-probability difference: 0.0; maximum candidate-mass difference: 0.0.
- Actual reused prefix tokens across reused decisions: 4096.
- Uses state-first v2, 12 synthetic requests plus long-state and identical-prompt cases. Includes recovery after invalid/overlong requests. This checks cache equivalence, not general classification accuracy.

## Frozen Kaggle AG News 400 articles

The default `legacy + fresh` path uses the GPT-OSS final-channel prefill (`gpt-oss-final-prefill-decision-v1`). The exact same frozen requests, separate gold labels, checkpoint, and thresholds were retained. This is one local CUDA pass, not a state-first v2 Kaggle measurement.

| Metric | Result |
| --- | ---: |
| Correct / all | 129/400 (32.25%) |
| Wrong accepted answers | 43 |
| Abstained | 228 (57.00%) |
| Accepted accuracy | 129/172 (75.00%) |
| Coverage | 43.00% |
| Raw top-1 accuracy before abstention | 55.25% |
| Errors / missing | 0 / 0 |
| p50 / p95 per article | 1525.25 / 4594.70 ms |

The previous final-prefill v1 run had 134 correct, 41 wrong and 225 abstentions (33.50% correct/all; 76.57% accepted accuracy). The rebuilt run has 1.25 percentage points lower correct/all, while raw top-1 accuracy remains 55.25%. The earlier and current binaries differ; this comparison does not isolate the cause of numerical changes.

Compared with the previous final-prefill v1 run: 27 accepted selections changed; maximum option-probability difference 0.38478454145009744. Abstentions remain in correct/all; accepted accuracy excludes them. These figures are dataset accuracy, not calibrated per-answer confidence.

## Reproducibility

- Model SHA256: `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
- Evaluator SHA256: `7c60b41200667dd1a162c5c41baeff302ea1571c69f454a7a2c71d8dbdda7aa0`.
- Request SHA256: `1489716ed040e95087a6e973819ac1f346f39651db50cd2908da2b723da50a54`.
- Source changes during verification: [].
- Exact sources, binaries, hashes, commands, logs, per-article predictions, scores and fresh/reuse samples are preserved locally in `results/rebuild-20260921/` (Git-ignored).
- Existing model, dataset and result files were retained. No GitHub publication was performed for this verification.
