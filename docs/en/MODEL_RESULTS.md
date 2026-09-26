<a id="recorded-model-results"></a>
# Recorded model results

[English](MODEL_RESULTS.md) · [한국어](../ko/MODEL_RESULTS.md) · [日本語](../ja/MODEL_RESULTS.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)


[English introduction](../../README.md) · [한국어 소개](../../README.ko.md) · [Usage guide](GUIDE.md)

These are historical measurements at their recorded revisions and settings.
They are not measurements of the current checkout or an official leaderboard.

<a id="recorded-model-comparison"></a>
## Recorded model comparison

The September 23, 2026 JevBench matrix measured **22 GGUF checkpoints on all 231 public items** using the same frozen project build on an RTX 3060 12 GiB. All 5,082 predictions in the completed comparison runs were valid, with no inference errors or truncation. The 23 runtime configurations include one failed default GPT-OSS attempt and its successful CUDA Graphs-disabled recovery.

The original 22 matrix rows use identical request and evaluator hashes, fresh/legacy execution, context 8192, batch/ubatch 256, four threads and FlashAttention off, without reasoning-token generation, LoRA, an output head or learned calibration. The explicit GPT-OSS exception is marked below. Rows marked † are separate September 24 runs; ‡ is a separate September 23 run.

| Checkpoint | Argmax accuracy | Hard accuracy | Accepted wrong | Abstained / 231 | p50 / p95 ms | typed-decisions raw / coverage / accepted accuracy / correct-all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Gemma 4 31B Q4_K_M](https://huggingface.co/google/gemma-4-31B-it) † | 89.61% | 78.38% | 22 | 3 | 2287.40 / 24524.56 | — |
| Gemma 4 26B A4B UD-Q4_K_M ‡ | 84.85% | 70.27% | 29 | 11 | 787.01 / 8814.01 | — |
| Qwen3.5-4B-Q8_0 | 79.65% | 61.26% | 9 | 93 | 86.88 / 1137.30 | — |
| Qwen3.5-9B-Q4_K_M | 77.92% | 58.56% | 13 | 64 | 130.55 / 1697.14 | — |
| Qwen3.5-9B-Q8_0 | 77.92% | 56.76% | 14 | 66 | 124.68 / 1613.79 | — |
| gemma-4-E4B-it-Q4_K_M | 77.49% | 55.86% | 30 | 37 | 86.62 / 1186.47 | — |
| gemma-4-E4B-it-Q8_0 | 76.62% | 54.05% | 32 | 37 | 85.04 / 1146.21 | — |
| Qwen3.5-4B-Q4_K_M | 76.19% | 55.86% | 10 | 91 | 88.47 / 1175.18 | — |
| [Ternary Bonsai 27B Q2_g64](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) † | 75.32% | 53.15% | 15 | 80 | 187.15 / 2462.18 | — |
| Qwen3.8-27B-UD-IQ2_XXS | 73.16% | 47.75% | 18 | 84 | 414.61 / 5460.22 | — |
| Qwen3-8B-Q8_0 | 71.43% | 48.65% | 56 | 15 | 121.45 / 1817.94 | — |
| [Bonsai 27B Q1_0](https://huggingface.co/prism-ml/Bonsai-27B-gguf) † | 71.00% | 46.85% | 14 | 91 | 184.51 / 2489.64 | — |
| gemma-4-E2B-it-Q8_0 | 67.97% | 43.24% | 58 | 22 | 46.79 / 701.11 | 54.30% / 92.75% / 55.69% / 51.65% § |
| Qwen3-4B-Q8_0 | 65.80% | 44.14% | 63 | 29 | 85.26 / 1349.28 | — |
| Ministral-3-8B-Instruct-2512-Q4_K_M | 65.37% | 47.75% | 23 | 93 | 441.43 / 2399.70 | — |
| gpt-oss-20b-Q4_K_M (CUDA Graphs off) | 64.94% | 48.65% | 31 | 82 | 181.93 / 2262.28 | — |
| Qwen3.5-2B-Q8_0 | 62.34% | 48.65% | 15 | 133 | 40.99 / 513.03 | — |
| gemma-3-4b-it-Q8_0 | 59.74% | 36.04% | 88 | 9 | 74.71 / 879.92 | — |
| Phi-4-mini-instruct.Q8_0 | 56.71% | 42.34% | 30 | 115 | 70.50 / 971.95 | — |
| Qwen3.5-0.8B-Q8_0 | 51.95% | 42.34% | 16 | 183 | 27.93 / 350.53 | — |
| SmolLM3-3B-Q8_0 | 46.32% | 31.53% | 41 | 127 | 71.47 / 884.76 | — |
| Llama-3.2-3B-Instruct-Q8_0 | 43.72% | 30.63% | 46 | 147 | 68.18 / 884.75 | — |
| gemma-3-1b-it-Q8_0 | 38.96% | 28.83% | 121 | 30 | 25.06 / 309.24 | — |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 33.77% | 36.04% | 1 | 230 | 30.08 / 702.65 | — |
| Qwen3-0.6B-Q8_0 | 31.60% | 31.53% | 129 | 41 | 34.01 / 445.12 | 31.25% / 55.60% / 34.35% / 19.10% § |
| SmolLM2-135M-Instruct-Q8_0 | 30.30% | 29.73% | 8 | 211 | 14.88 / 253.65 | — |

§ The September 26 [typed-decisions measurement](TYPED_DECISIONS_BENCHMARK.md) covers all 400 test cases / 2,000 judgments on an RTX 3080 with a separate frozen direct-mode evaluator. This added column uses a different dataset and hardware from JevBench. Its case p50/p95 were 322.64/370.98 ms for Gemma 4 E2B and 173.46/204.22 ms for Qwen3 0.6B; the existing latency column remains JevBench latency. Raw accuracy is measured before abstention; coverage, accepted accuracy, and correct/all measure the acceptance policy separately.

The † rows used the same public dataset (SHA-256 `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`) and byte-identical 231 request JSONL (SHA-256 `6f96c4fc2b924ec0bef4aaa94c25b2456522909fdbebffe0c0df8fa44e5c2faa`). Candidate-argmax scores were 164/231 for Bonsai Q1_0, 174/231 for Ternary Bonsai Q2_g64 and 207/231 for Gemma 4 31B Q4_K_M. All 693 predictions were valid, with no inference errors or truncation; independent recounts reproduced each tier total. The Gemma 4 31B run also reproduced every candidate probability and policy value from its earlier validated run. The default policy accepted 140/151/228 decisions respectively, of which 126/136/206 were correct.

These runs used fresh/legacy execution, context 8192, batch/ubatch 256, and no LoRA, output head or learned calibration. The Bonsai runs used an RTX 3080 with full GPU offload and four threads; Gemma 4 31B used an RTX 3060 with 24 GPU layers, read-mode loading and eight threads. Evaluator binaries also differed, so the † latency figures are not a controlled speed comparison with each other or the original matrix. Their local, gitignored evidence is in `results/bonsai-27b-20260924/` and `results/jevbench-gemma31-rust-20260924/`; these artifacts are not included in the repository.

The ‡ 26B run used the same 231 public task IDs on an RTX 3060 with 18 CPU expert layers and eight threads. It used a different request serialization and evaluator build from the original matrix and † reruns, so the table combines task scores from distinct runs, not a controlled latency comparison. All 231 decisions completed without errors or truncation; the default policy accepted 220, including 191 correct. See the [26B measurement details](JEVBENCH.md#gemma-4-26b-a4b-separate-run-2026-09-23). Its raw evidence is local and gitignored.

Qwen3.5-4B Q8_0 had the highest argmax accuracy in the original 22-checkpoint matrix: 184/231 (79.65%), including 68/111 Hard items (61.26%). Its default policy accepted 138 decisions: 129 correct and 9 wrong, for 93.48% accepted accuracy at 59.74% coverage. Qwen3.5-9B Q4_K_M covered 72.29% with 13 accepted errors; Gemma4 E2B covered 90.48% with 58 accepted errors. Accuracy before abstention, accepted accuracy and coverage answer different questions.

GPT-OSS 20B Q4_K_M exhausted GPU memory in `cudaGraphInstantiate` after 129 predictions. With `GGML_CUDA_DISABLE_GRAPHS=1`, it completed all 231 at 64.94% accuracy and a sampled peak of 11,901 MiB. Its first 129 probability distributions were identical to the failed run. Keep this runtime exception when reproducing its result.

After model downloads finished, complete reruns of Qwen3.5-4B Q8_0 and Gemma4 E2B produced identical probabilities for every item. Their confirmation p50/p95 latencies were 86.99/1139.49 ms and 46.08/699.14 ms respectively; the table retains the original matrix timings.

These are public-subset, local inference measurements, not an official full-suite score, rank or production validation. Latency excludes loading and warmup; small models may exceed their training context. For the original 22-checkpoint matrix, raw predictions, model/source hashes, memory samples, failed-attempt evidence and independent accuracy/Brier/ECE recount are in the local, gitignored `results/jevbench-matrix-20260923/` directory; they are not included in this repository. The measured source snapshot is kept with those artifacts, and later working-tree optimizations are outside this frozen comparison. See the [evaluation method](JEVBENCH.md).

<a id="additional-recorded-model-measurements"></a>
## Additional recorded model measurements

The [complete-label intent evaluation](INTENT_BENCHMARK.md#measured-results) used 200 BANKING77 English and 200 MASSIVE Korean examples per checkpoint on the RTX 3060. Each request included all 77 or 60 official labels. Accuracy below is raw top-choice accuracy before the abstention policy; the p50 values are local inference milliseconds. The four original configurations and separate 26B run completed both samples without inference errors or truncation.

| Checkpoint | BANKING77 correct / 200 | MASSIVE Korean correct / 200 | English / Korean p50 ms |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 123 (61.5%) | 103 (51.5%) | 214.15 / 157.52 |
| Gemma 4 E4B Q8_0 | 130 (65.0%) | 143 (71.5%) | 371.72 / 277.05 |
| Gemma 4 E4B Q4_K_M | 130 (65.0%) | 138 (69.0%) | 383.61 / 287.00 |
| Gemma 4 26B A4B UD-Q4_K_M ‡ | 152 (76.0%) | 156 (78.0%) | 2880.52 / 2134.52 |
| Qwen3-8B Q8_0 | 111 (55.5%) | 98 (49.0%) | 877.64 / 593.45 |

The ‡ 26B row used 18 CPU expert layers, eight threads, and a separate evaluator run. Its p50 values are first-call medians; immediate repeated calls with request-local cache measured 669.15 / 656.70 ms and preserved all 400 result evidence objects. These timings are not controlled comparisons with the four original rows.

The [CPU/GPU cache check](LAYA_BENCHMARK.md) additionally measured one larger checkpoint with 24 GPU layers and eight CPU threads. Times below cover two typed cases and their immediate repeats.

| Checkpoint | Batch 256 fresh | Batch 64 fresh / prefix reuse | Batch 128 fresh / prefix reuse |
| --- | ---: | ---: | ---: |
| Gemma 4 31B Q4_K_M | 41.677 s | 108.552 / 69.904 s | 62.438 / 42.770 s |

Both same-batch reuse comparisons preserved all 20 probability vectors and selections. This small repeated-input check measures cache behavior, not labeled task accuracy. Neither cached setting beat the batch-256 fresh baseline.

These intent and cache measurements use different tasks and settings from the JevBench matrix above. Their detailed protocols and local-artifact limits are in the linked guides.
