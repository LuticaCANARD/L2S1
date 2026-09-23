# JevBench multi-model measurements

Host: `100.66.64.91`, RTX 3060 12 GiB. Public JevBench: 231 items (48 Easy, 72 Original, 111 Hard).

Planned configurations: 22. Scored: 9. Failed before complete scoring: 0. Pending: 13.

| Model | Correct / 231 | Accuracy | Hard | ECE ↓ | p50 ms | p95 ms | Peak GPU MiB | Accepted wrong | Abstained | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-Q8_0.gguf | 184 | 79.65% | 61.26% | 0.0469 | 86.88 | 1137.30 | 5039 | 9 | 93 | 0 |
| Qwen3.5-4B-Q4_K_M.gguf | 176 | 76.19% | 55.86% | 0.0619 | 88.47 | 1175.18 | 3381 | 10 | 91 | 0 |
| gemma-4-E2B-it-Q8_0.gguf | 157 | 67.97% | 43.24% | 0.2768 | 46.79 | 701.11 | 3025 | 58 | 22 | 0 |
| Qwen3.5-2B-Q8_0.gguf | 144 | 62.34% | 48.65% | 0.1084 | 40.99 | 513.03 | 2465 | 15 | 133 | 0 |
| Qwen3.5-0.8B-Q8_0.gguf | 120 | 51.95% | 42.34% | 0.1216 | 27.93 | 350.53 | 1317 | 16 | 183 | 0 |
| gemma-3-1b-it-Q8_0.gguf | 90 | 38.96% | 28.83% | 0.5594 | 25.06 | 309.24 | 1663 | 121 | 30 | 0 |
| tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 78 | 33.77% | 36.04% | 0.2369 | 30.08 | 702.65 | 1297 | 1 | 230 | 0 |
| Qwen3-0.6B-Q8_0.gguf | 73 | 31.60% | 31.53% | 0.5880 | 34.01 | 445.12 | 1857 | 129 | 41 | 0 |
| SmolLM2-135M-Instruct-Q8_0.gguf | 70 | 30.30% | 29.73% | 0.3772 | 14.88 | 253.65 | 595 | 8 | 211 | 0 |

## Method and limits

- Same frozen public dataset, Rust evaluator binary, legacy layout, fresh requests, context 8192, batch/ubatch 256, four threads, FlashAttention off, one warmup, and default abstention thresholds.
- Embedded model templates and the existing Auto prompt profile are used. No generation of reasoning tokens, LoRA, learned head, calibration, or benchmark-based prompt tuning.
- Accuracy is argmax over candidate probabilities before abstention. Accepted errors, abstentions, and probability calibration remain separate.
- Native predictions were independently recounted against gold labels; Brier and ECE were recomputed. Requests and binary hashes match across scored models.
- Latency is serial local inference, excluding loading and warmup. Downloads can overlap runs; these single-run timings are not an SLA or directly comparable to HTTP leaderboard timings.
- Peak GPU memory is whole-board usage sampled every 200 ms, including baseline; short peaks may be missed.
- This is a selected hardware/runtime feasibility matrix, not every published model or every quantization. Small models may run beyond their training context; logs retain warnings.
- Public subset only: no claim of the official full-suite score or rank. Ranking on this dataset is model selection evidence, not an independent production validation.

## Pending

- Qwen3.5-9B-Q4_K_M
- Qwen3.5-9B-Q8_0
- Qwen3-4B-Q8_0
- Qwen3-8B-Q8_0
- gemma-4-E4B-it-Q8_0
- gemma-4-E4B-it-Q4_K_M
- gemma-3-4b-it-Q8_0
- Llama-3.2-3B-Instruct-Q8_0
- Phi-4-mini-instruct.Q8_0
- Ministral-3-8B-Instruct-2512-Q4_K_M
- SmolLM3-3B-Q8_0
- gpt-oss-20b-Q4_K_M
- Qwen3.8-27B-UD-IQ2_XXS
