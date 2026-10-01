# Recorded model evidence audit — 2026-09-26

[English](../en/MODEL_AUDIT_20260926.md) · [한국어](../ko/MODEL_AUDIT_20260926.md) · [日本語](../ja/MODEL_AUDIT_20260926.md)

This is a source-data audit of measurements recorded on September 23–25, 2026, not a new inference run. Evidence paths in code are repository-relative local artifacts. Public aggregates and existing reports are linked below.

Raw = correct highest-probability candidate / all items. Accepted accuracy = accepted correct / accepted. Coverage = accepted / all. The tables retain the underlying counts; p50 is milliseconds unless stated otherwise.

## Same-condition RTX 3060 matrix — 2026-09-23

All 22 completed configurations × 231 public JevBench items (5,082 responses) were recounted. The public selection contains 20 configurations / 4,620 responses: raw correct 2,924; accepted 3,122; accepted correct 2,279; accepted wrong 843; abstained 1,498; errors 0. Across the original 23 attempted runtime configurations, 22 completed and the default GPT-OSS CUDA Graph attempt failed after 129 predictions; its separate Graphs-disabled run completed.

| Model | Raw correct / 231 | Accepted correct / accepted | Coverage | p50 ms | GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B Q8_0 | 184/231 (79.65%) | 129/138 (93.48%) | 138/231 (59.74%) | 86.88 | 5039 |
| Qwen3.5 9B Q4_K_M | 180/231 (77.92%) | 154/167 (92.22%) | 167/231 (72.29%) | 130.55 | 5637 |
| Gemma4 E4B Q4_K_M | 179/231 (77.49%) | 164/194 (84.54%) | 194/231 (83.98%) | 86.62 | 3989 |
| Qwen3.5 4B Q4_K_M | 176/231 (76.19%) | 130/140 (92.86%) | 140/231 (60.61%) | 88.47 | 3381 |
| Gemma4 E2B Q8_0 | 157/231 (67.97%) | 151/209 (72.25%) | 209/231 (90.48%) | 46.79 | 3025 |
| Qwen3.5 2B Q8_0 | 144/231 (62.34%) | 83/98 (84.69%) | 98/231 (42.42%) | 40.99 | 2465 |

Qwen3.5 4B Q8 leads raw accuracy in this matrix and returns 129 correct answers among 138 accepted. Gemma4 E4B Q4 returns 164 correct answers among 194 accepted at about the same p50 and 1,050 MiB less sampled GPU memory. Qwen3.5 9B Q4 combines 92.22% accepted accuracy with 72.29% coverage. Gemma4 E2B Q8 offers a 46.79 ms p50 at 67.97% raw accuracy.

The 22 completed configurations share request and evaluator hashes, RTX 3060 12 GiB, context 8192, batch/ubatch 256, four threads, FlashAttention off, fresh/legacy prompts, embedded model templates and policy thresholds 0.8 / 0.05. Inference is serial Rust scoring; loading and one warmup are excluded. GPU MiB is maximum whole-board usage sampled every 200 ms, including loading and warmup. GPT-OSS uses `GGML_CUDA_DISABLE_GRAPHS=1`.

Raw accuracy, accepted counts, multiclass Brier, ECE with 10 bins, serial p50/p95, request hashes and GPU CSV maxima match the recorded summaries. Separate idle reruns of Qwen3.5 4B Q8 and Gemma4 E2B reproduce every candidate probability; their p50 values are 86.99 / 46.08 ms and remain separate from the original matrix.

[Public RTX 3060 summary](../../benchmarks/rtx3060-20260926/summary.json) · [RTX 3060 report](RTX3060_BENCHMARK.md)

For Qwen3.5 9B, Q4_K_M and Q8_0 both score 180/231 (77.92%) raw correct. Their sampled board peaks are 5,637 / 8,817 MiB: Q4 uses 36.07% less in these runs. This compares aggregate accuracy, not identical output distributions.

## Gemma 31B / 26B mixed CPU/GPU placement — 2026-09-23

These two runs share the same 231 JevBench requests and evaluator with each other. Their evaluator differs from the preceding matrix. Both use context 8192, batch/ubatch 256, eight threads, fresh/legacy prompts, read-mode loading and policy 0.8 / 0.05.

| Model / placement | Raw correct / 231 | Accepted correct / accepted | Coverage | p50 / p95 ms | Board GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 31B Q4_K_M / GPU 24 | 207/231 (89.61%) | 206/228 (90.35%) | 228/231 (98.70%) | 2312.37 / 24590.96 | 10897 |
| Gemma4 26B A4B UD-Q4_K_M / CPU expert 18 | 196/231 (84.85%) | 191/220 (86.82%) | 220/231 (95.24%) | 653.19 / 7338.46 | 10557 |

Gemma31B uses 23 repeating layers plus the output layer on GPU, with 37 repeating layers on CPU. Its full run records 15.92 GiB engine peak RSS, 10.60 GiB sampled process GPU allocation and zero sampled process swap. Board memory in the table is separate from process allocation; RSS excludes VRAM. Gemma26B uses 18 CPU expert layers. Its six alternating warm-filesystem loading checks record median peak RSS of 16.17 GiB (`auto`) and 9.35 GiB (`read`); this is an RSS loading comparison.

## Complete-label intent classification — 2026-09-23

RTX 3060; four models × BANKING77 English 200 items / MASSIVE Korean 200 items = 1,600 responses. Each request includes all 77 / 60 labels. All raw, accepted and p50 counts match the native predictions. Context 8192, batch/ubatch 256, four threads, fresh/legacy, FlashAttention off, policy 0.8 / 0.05; model loading and warmup excluded. This intent evaluator is distinct from the JevBench evaluators.

| Model | Dataset | Raw correct / 200 | Accepted correct / accepted | Coverage | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 E2B Q8_0 | BANKING77 en | 123/200 (61.50%) | 119/181 (65.75%) | 181/200 (90.50%) | 214.15 |
| Gemma4 E2B Q8_0 | MASSIVE ko | 103/200 (51.50%) | 100/181 (55.25%) | 181/200 (90.50%) | 157.52 |
| Gemma4 E4B Q8_0 | BANKING77 en | 130/200 (65.00%) | 127/173 (73.41%) | 173/200 (86.50%) | 371.72 |
| Gemma4 E4B Q8_0 | MASSIVE ko | 143/200 (71.50%) | 136/174 (78.16%) | 174/200 (87.00%) | 277.05 |
| Gemma4 E4B Q4_K_M | BANKING77 en | 130/200 (65.00%) | 124/172 (72.09%) | 172/200 (86.00%) | 383.61 |
| Gemma4 E4B Q4_K_M | MASSIVE ko | 138/200 (69.00%) | 129/168 (76.79%) | 168/200 (84.00%) | 287.00 |
| Qwen3 8B Q8_0 | BANKING77 en | 111/200 (55.50%) | 107/175 (61.14%) | 175/200 (87.50%) | 877.64 |
| Qwen3 8B Q8_0 | MASSIVE ko | 98/200 (49.00%) | 96/173 (55.49%) | 173/200 (86.50%) | 593.45 |

Gemma4 E4B Q8 reaches 65.0% on BANKING77, tied with E4B Q4, and leads these four models on MASSIVE Korean at 71.5%. Gemma4 E2B Q8 offers 214.15 / 157.52 ms p50 for English / Korean.

[Intent report](INTENT_BENCHMARK.md)

## Image classification baselines — 2026-09-24 / 25

One original JPEG per serial local HTTP request, loaded-model timing including the first request, startup excluded, no warmup, default policy 0.8 / 0.05. Each dataset keeps its own class set and fixed baseline question. September 24 uses RTX 3060; September 25 TrashNet uses RTX 3080.

| Dataset / GPU | Model | Raw correct / all | Accepted correct / accepted | Coverage | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cats/dogs / RTX 3060 | Gemma4 E2B Q8_0 | 69/70 (98.57%) | 69/70 (98.57%) | 70/70 (100%) | 99.18 |
| Caltech 30 classes / RTX 3060 | Gemma4 E2B Q8_0 | 122/150 (81.33%) | 122/148 (82.43%) | 148/150 (98.67%) | 195.28 |
| TrashNet 6 classes / RTX 3080 | Qwen3-VL 2B Q8_0 | 95/120 (79.17%) | 91/115 (79.13%) | 115/120 (95.83%) | 101.08 |
| TrashNet 6 classes / RTX 3080 | Gemma4 E2B Q8_0 | 64/120 (53.33%) | 64/113 (56.64%) | 113/120 (94.17%) | 79.12 |

Cats/dogs uses 70 validation images (24 cats / 46 dogs), context 2048; reversed option order also yields 69/70 with p50 98.75 ms. Caltech uses 30 classes × five images, context 4096; its 3,855 MiB GPU record is a point sample. TrashNet uses six classes × 20 images. Qwen3-VL 2B Q8 has the highest raw score among the three recorded TrashNet baseline models. Subsequent traits prompts and batch experiments are separate exploratory protocols. These are independent datasets and evaluators, with HTTP latency separate from the Rust matrix.

[Vision report](VISION_BENCHMARK.md)

## Synthetic shared-state prefix reuse — 2026-09-25

RTX 3080, context 2048, batch/ubatch 32, four threads, state-first prompt and disabled preparation cache. The example state contains 200 filler words; 16 questions have different instructions and IDs. Five alternating-order rounds each have one untimed warmup. Times are median whole-run milliseconds, including preparation and session creation/drop, excluding model loading and report serialization.

| Model | Fresh ms | Shared session ms | Ratio | Reused / input tokens |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8_0 | 1461.77 | 531.49 | 2.75× | 3840/5789 |
| Gemma4 E2B Q8_0 | 2735.54 | 985.42 | 2.78× | 3840/5751 |

The original JSON records reproduce these medians. Across 210 paired decisions per model, raw logits, candidate probabilities, candidate mass, selected values and abstention reasons match exactly. This is an explicit native session with actual KV prefix reuse and synthetic input; task accuracy and retained-session memory were not measured.

[Shared-state report](../../benchmarks/shared-state-cache-20260925/REPORT.md)

## Audited source paths

The Python filenames below identify the historical runs. Duplicate Python implementations have been removed; use `l2s1-tools report-jevbench-matrix` for recounts and `l2s1-tools jevbench-public run` for current execution.

- RTX 3060: `results/jevbench-matrix-20260923/REPORT.json`, `environment.json`; `<configuration>/{manifest.json,summary.json,tasks-with-gold.jsonl,requests.jsonl,predictions.jsonl,gpu-memory.csv,inference.stderr.log}`. Recount: `scripts/report_jevbench_matrix.py`; GPU sampling: `scripts/jevbench_public.py`.

- Gemma31B: `results/large-model-20260923T141422Z/REPORT.json`, `remote/jevbench/{manifest.json,summary.json,predictions.jsonl,tasks-with-gold.jsonl,gpu-memory.csv}`. Gemma26B: `results/gemma26-lowrss-20260923T135227Z/REPORT.json`, `remote/jevbench-read/{manifest.json,summary.json,predictions.jsonl,tasks-with-gold.jsonl,gpu-memory.csv}`.

- Intent: `results/intent-wide-20260923/prepared/gold.jsonl`, `runs/<model>/{manifest.json,summary.json,predictions.jsonl,scored.jsonl}`.

- Images: `benchmarks/cats-dogs-vision-20260924/{summary.json,observations.jsonl}` and `dog-cat/`; `benchmarks/caltech101-vision-20260924/{summary.json,observations.jsonl,selection.json}`; `benchmarks/trashnet-vision-20260925/{gemma4,qwen3vl,smolvlm}/{summary.json,observations.jsonl}`.

- Shared state: `benchmarks/shared-state-cache-20260925/{qwen3-0.6b-sm86.json,gemma4-e2b-sm86.json}`; workload: `examples/benchmark_shared_state_cache.rs`.
