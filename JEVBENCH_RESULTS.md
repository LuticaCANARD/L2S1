# Gemma 4 JevBench public results

Completed 2026-09-23 KST on `100.66.64.91` (`lucatagpu`), NVIDIA GeForce RTX 3060 12 GiB, driver 595.71.05. Current project package `l2s1` was rebuilt from a frozen copy of the working source, including uncommitted changes, in a fresh directory with no prior Cargo artifacts. The source was checked again after the run: no core source drift occurred during evaluation.

## Rebuild and integration checks

- Default Rust tests: 21 passed.
- Llama-feature release tests: 30 passed; 13 optional real-model tests were not automatically run.
- All-target release Clippy: passed. Upstream C++ helper warnings remain in the build log.
- Gemma 4 CUDA native integration: explicitly invoked and passed, including request isolation, candidate mapping, unchanged chunk-consistency tolerance, profile rejection, and overlong-input rejection.
- New JevBench mapping checks: five passed, locally and on the server.
- Pinned upstream protocol tests: 17 passed.

The project Rust code and C++ bridge were compiled again. The existing matching llama.cpp shared libraries were reused at commit `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`; this is not a claim that the entire CUDA toolkit or llama.cpp was rebuilt.

## Model and evaluation configuration

- Gemma 4 E2B IT Q8_0; SHA256 `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`.
- Stock model, no LoRA, output head, or learned calibration.
- JevBench revision `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`; all 231 published items attempted exactly once.
- Original state/rubric/label order preserved; expected answers and rationales excluded from model input.
- Legacy prompt, fresh execution, context 8,192, batch/microbatch 256, four threads, FlashAttention off, request batch size one.
- Default policy thresholds retained: top candidate probability 0.8 and full-vocabulary candidate mass 0.05.
- Largest actual input: 3,981 tokens. Zero truncated inputs, missing records, duplicate records, invalid distributions, or inference errors.

## Official scoring functions on the public subset

| Public tier | Correct / attempted | Accuracy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Easy | 48 / 48 | 100.00% | ~0 | ~0 |
| Original | 61 / 72 | 84.72% | 0.2511 | 0.1318 |
| Hard | 48 / 111 | 43.24% | 1.0238 | 0.4910 |
| All public items | 157 / 231 | 67.97% | 0.5702 | 0.2768 |

Accuracy is the upstream argmax-over-labels metric, before the project's abstention policy. All 231 probability distributions also passed the stricter upstream sum tolerance; no renormalization was needed. Brier uses the multiclass sum convention and can exceed one. Lower Brier and ECE are better.

The model was overconfident on this dataset: 199 decisions had top-candidate probability at least 0.9, with mean confidence 99.38% but accuracy 73.87%. High native candidate probabilities therefore did not establish reliable correctness, especially on the hard tier.

Additional diagnostics:

- Original paraphrase pairs: 30/36 agreed (83.33%); both answers correct for 28/36 pairs (77.78%).
- Ordinal expected-value MAE: 0.4910 over the ordinal items.
- Hard probability questions with explicit gold distributions: mean total variation distance 0.6504 over ten items.

## Project abstention policy, reported separately

| Outcome | Count |
| --- | ---: |
| Correct accepted decisions | 151 |
| Wrong accepted decisions | 58 |
| Abstentions | 22 |
| Errors | 0 |

Coverage was 209/231 (90.48%); accepted accuracy was 151/209 (72.25%). The defaults reject some uncertain outputs, but still accept many errors. No threshold was selected or adjusted using these labels.

## Local inference latency

| Public tier | p50 | p95 |
| --- | ---: | ---: |
| Easy | 41.20 ms | 44.59 ms |
| Original | 41.67 ms | 45.97 ms |
| Hard | 144.22 ms | 922.20 ms |
| All public items | 46.00 ms | 698.53 ms |

Each request contains one decision. Timings measure the serial Rust inference call, including input preparation, with model loading and one warmup excluded. There is no HTTP/network latency in these figures; they are not directly comparable to the board's endpoint timings or its estimated load adjustments. This is one run, not a repeatability/performance SLA study.

## Audit and limits

The pinned official JevBench CLI independently summarized the saved normalized records and matched all compared accuracy, validity, calibration, latency, and paraphrase aggregates. A second calculation from the original predictions and labels reproduced 157 correct, the Brier score, and ECE.

This covers the 231 published items, not all 534 official items. The unavailable held-out and non-redistributed tasks were not invented or replaced. Local cost remains unknown (`null`); no official overall score or ranking is claimed. The model was not fine-tuned or calibrated on JevBench.

Local evidence: `results/jevbench-20260923/run-gemma4/` and `results/jevbench-20260923/evidence/`. Remote evidence: `~/personal/skid/jevbench-20260923/`. Exact source/binary hashes are in `evidence/rebuild-manifest.json`; model/dataset/evaluator identities and the inference command are in `run-gemma4/manifest.json`.

See [JEVBENCH.md](JEVBENCH.md) for mapping details and reproduction commands.
