# Q8 output-head experiment — 2026-09-23

The frozen Gemma4 E2B Q8 model improved from **323/400 (80.75%) to 335/400 (83.75%)** with a task-specific linear head trained on deployment hidden states. A small affine transform of existing candidate logits reached 327/400 (81.75%). The hidden head was selected by development NLL before opening the final test results. No base weights or default behavior changed.

This is a supervised airline-sentiment classifier on a frozen LLM, not an RLCD/Jev reproduction or evidence of general reasoning improvement. The +3.00 percentage-point top-1 gain has paired exact McNemar p = 0.08069 (26 corrections, 14 regressions); this 400-case pilot does not establish a population-level improvement at the conventional 5% level. The previous LoRA experiment used a different holdout, so its 77.00%/75.75% scores are not direct comparisons with this table.

## Fixed protocol

- Local RTX 3080; Gemma4 E2B `Q8_0`, GGUF SHA-256 `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`.
- llama.cpp `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`; fresh/legacy prompt; context 2048, batch/ubatch 256, four CPU threads, FlashAttention off.
- Kaggle Twitter Airline Sentiment archive SHA-256 `c0dbee48cac32110a607430dc5c1941a1728383adef20a893ded91adbbba2de2`. Seed 20260924. Excluded all 2,500 normalized texts used by both earlier airline experiments.
- New disjoint texts: train 900, development 300, calibration 400, test 400. Three cyclic training option orders produce 2,700 feature rows. The order probe uses 60 of the test texts × three orders.
- Actual frozen Q8 features, CPU float64 multinomial linear fitting. Standardization fitted on training only and folded into the exported parameters. Fixed L2 grid 0.001/0.01/0.1/1; lowest development NLL selects penalty and head. Calibration temperature fits only its own 400 examples.
- The hidden head has 4,611 parameters; the affine head has 12. CPU fitting and temperature selection for all methods took 11.98 seconds, excluding GPU feature extraction.

The initial general-embedding extraction path was discarded because it forced vocabulary projections for all input tokens. The measured pipeline uses unmasked NextN hidden extraction, preserving base graph shapes. It still computes the full vocabulary logits to retain the base candidate-mass gate; it does not remove the expensive vocabulary projection.

## Accuracy and calibration

Every row uses its independently fitted calibration temperature. Top-1 counts include all 400 cases, irrespective of abstention.

| Method | Correct / 400 | Top-1 | Temperature | Test NLL ↓ | Brier ↓ | ECE, 10 equal bins ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Base + temperature | 323 | 80.75% | 6.36494 | 0.52855 | 0.28664 | 6.16% |
| Affine candidate logits | 327 | 81.75% | 0.96129 | 0.48337 | 0.27144 | 4.49% |
| Hidden-state linear head | **335** | **83.75%** | 1.13939 | **0.43922** | **0.24143** | **2.98%** |

For comparison, uncalibrated base NLL is 2.03983 and ECE 19.00%; raw base confidence is overconfident. Comparing abstention against that uncalibrated confidence would give a misleading advantage to the baseline. Temperature alone cannot change top-1 ranking.

The hidden head chose L2=0.01 with development NLL 0.42749. The affine head chose L2=0.001 with development NLL 0.49656. Hidden test NLL improved by 16.90% relative to calibrated base NLL; this remains a result on this split, not a production probability guarantee.

## Abstention

The same base candidate-mass threshold of 0.05 is retained. Counts are correct accepted / wrong accepted / abstained; all sum to 400.

| Confidence threshold | Base + temperature | Hidden head | Base accepted accuracy | Head accepted accuracy |
|---|---:|---:|---:|---:|
| 0.6 | 308 / 51 / 41 | 310 / 41 / 49 | 85.79% | 88.32% |
| 0.7 | 278 / 42 / 80 | 286 / 35 / 79 | 86.88% | 89.10% |
| 0.8 | 241 / 26 / 133 | **251 / 23 / 126** | 90.26% | **91.61%** |
| 0.9 | 153 / 5 / 242 | 178 / 7 / 215 | 96.84% | 96.22% |

At 0.8, coverage rises from 66.75% to 68.50%, abstentions fall by seven, ten more correct decisions are returned and three fewer incorrect decisions are returned. Abstentions do not decrease at every threshold: at 0.6 the head abstains eight more times while reducing errors. At 0.9 coverage improves but accepted accuracy is slightly lower.

At an equal diagnostic 80% coverage, accuracy is 86.88% base versus 89.06% head. At 60% coverage it is 92.92% versus 93.33%. These rankings are descriptive test diagnostics, not deployed thresholds selected using test labels.

## Option order and task scope

All three methods agree across all three cyclic orders on 55/60 probe texts. Across the 180 dependent calls, correct counts are base 156, affine 156, hidden 159. The head improves some classifications but does not improve this consistency count.

On the separate AG News benchmark, all 400 results with the airline head loaded are exactly identical to the base path and the previous experiment's base executable results (310/400 top-1). This proves scoped fallback on these requests, not general cross-domain improvement.

The artifact is explicitly scoped to `airline_sentiment` with the original instruction and option meanings. Other task IDs retain base scoring. Changed instructions/options under the trained ID, mismatched GGUF hash/compute/device/prompt, non-fresh execution, and combining LoRA with a head are rejected. This does not detect out-of-domain tweets submitted under the same task ID.

## Speed

Three paired, isolated GPU runs evaluate the same 400 requests after one warmup. Pair order is base/head, head/base, base/head. Startup, file hashing, warmup and serialization are excluded from the forward timing.

| Method | 400-case times (seconds) | Median total | Median total / 400 |
|---|---|---:|---:|
| Base | 15.419, 15.533, 15.646 | 15.533 s | 38.83 ms |
| Hidden head | 15.810, 15.477, 15.944 | 15.810 s | 39.53 ms |

Median overhead is **1.79%**, about 0.69 ms per request. The individual runs overlap; this is a small machine-local timing difference, not a universal latency guarantee. All repeated predictions/scores are exactly identical. All 400 ordinary base results also exactly equal their feature-export counterparts.

Startup is materially different: base loading takes about **1.38 s**, while loading plus the output head's full 4.7 GiB GGUF SHA-256 validation takes about **19.63 s**. Reuse a loaded `LlamaBackend` or the JSONL evaluation process to amortize this one-time validation. A one-request-per-process CLI pays the startup cost each time. This implementation improves decisions with low steady-state overhead; it does not deliver a major inference speedup.

## Runtime verification

The three deployed JSON artifacts were evaluated through Rust/CUDA on the 400 test requests and 180 permutation requests each. Independently calculated Python/float64 head logits match runtime values to a maximum absolute difference of 3.25e-14; calibrated probabilities match within 5.22e-15. Base candidate masses match exactly. Native tests also compare feature-enabled and ordinary inference across short and multi-batch inputs and check request/error isolation. The final native integration test passed checks for a rejected wrong GGUF hash, failed reloads, task-signature changes, execution/prompt drift, scoped fallback and an exact match between direct head arithmetic and inference.

Validation completed: 25 regular Rust tests with the llama feature, 17 without it, two explicit real-Gemma CUDA integration tests, 14 existing Python benchmark/training tests, strict Clippy, release build and the main CLI invocation. Native/model tests remain explicit opt-in tests; the ordinary suite does not silently load a model.

## Artifacts and reproducibility

- Protocol and IDs: `results/output-head-20260923/data/selection.json`.
- Training choices: `results/output-head-20260923/heads/training.json`.
- Selected deployment artifact: `results/output-head-20260923/heads/selected.json` (identical to `hidden.json`).
- Alternative head and temperature-only artifact: `heads/logit_affine.json`, `heads/base_temperature.json`.
- Feature exports: `features/{train,dev,calibration,test,probe}.jsonl`.
- Final comparison and numeric equivalence checks: `comparison.json`.
- Three-run timings, exact base equivalence and task fallback: `timings.json`.
- Source/executable hashes: `source-provenance.json`, `final-provenance.json`; all artifacts remain local under ignored `results/`.

See [OUTPUT_HEAD.md](OUTPUT_HEAD.md) for the exact runtime contract, limitations and reproduction commands. Data are balanced and deduplicated by normalized exact text only. A separate, larger holdout and domain-shift testing are needed before claiming a general improvement.
