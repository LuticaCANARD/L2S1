# Local Gemma 4 parallel AG News evaluation

Date: 2026-09-22. Gemma 4 E2B IT Q8_0 on the local RTX 3080. Same frozen 400 Kaggle AG News articles, 100 per class, with gold labels separate from model input. No remote inference was performed.

All modes use the same legacy model prompt, thresholds, checkpoint, context 2,048 per question, decode batch 256 and four CPU threads. Each configuration runs once, after one untimed warmup batch. Independent article states are never concatenated or merged into a new prompt.

| Mode / request batch | Correct | Wrong | Abstained | Correct/all | Accepted accuracy | Raw top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh-1 | 305 | 83 | 12 | 76.25% | 78.61% | 77.50% |
| parallel-4 | 304 | 81 | 15 | 76.00% | 78.96% | 77.50% |
| parallel-16 | 305 | 81 | 14 | 76.25% | 79.02% | 78.25% |
| parallel-32 | 305 | 80 | 15 | 76.25% | 79.22% | 77.75% |

Accepted accuracy excludes abstentions; correct/all includes all 400. These results are specific to this public dataset and checkpoint, not calibrated probabilities or general model accuracy.

| Mode | Sum of measured inference batch times | Articles/s | Amortized ms/article | Article completion p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| fresh-1 | 16.692 s | 23.96 | 41.73 | 39.96 / 55.55 ms |
| parallel-4 | 15.045 s | 26.59 | 37.61 | 149.08 / 159.70 ms |
| parallel-16 | 16.773 s | 23.85 | 41.93 | 668.88 / 696.91 ms |
| parallel-32 | 19.973 s | 20.03 | 49.93 | 1592.18 / 1615.07 ms |

**Throughput and latency differ.** Each article completes when its entire batch finishes. Dividing batch time by its size gives amortized compute cost, not response latency. Inference sums count each batch once and exclude model load, initial warmup, input cloning and output serialization. Any later context resize remains included. No queue accumulation or network delay was measured. These are single-pass measurements, not a stable service SLA.

| Mode | Changed selections/abstentions vs fresh | Changed raw top-1 | Max probability delta | Existing equivalence criterion |
| --- | ---: | ---: | ---: | --- |
| fresh-1 | 0 | 0 | 0.00000000 | PASS |
| parallel-4 | 9 | 4 | 0.51335377 | FAIL |
| parallel-16 | 6 | 3 | 0.49686515 | FAIL |
| parallel-32 | 11 | 2 | 0.51165619 | FAIL |

- Current fresh run versus the preceding Gemma rebuild run: 0 selections/abstentions changed.
- All configurations completed 400 unique expected IDs with no inference errors or missing cases. Correct/wrong/abstained counts were independently recounted.
- Candidate IDs/token IDs and logical input lengths match across modes. The Rust preparation function is identical; only native scheduling changes.
- Explicit multi-request API contract passed on Gemma 4 CUDA: separate states, repeated IDs across requests, output regrouping, partial waves, invalid/overlong request recovery, empty batch and width-one equivalence.
- Rust tests, all-target release Clippy and formatting passed. Parallel mode remains experimental; the previous synthetic fixture already demonstrated a serial abstention turning into an accepted wrong answer.

## Reproduce

```sh
LLAMA_CPP_DIR=/path/to/llama.cpp LLAMA_LIB_DIR=/path/to/llama.cpp/build-cuda/bin \
  cargo build --release --locked --features llama --example evaluate_jsonl
target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --input results/kaggle-ag-news/requests.jsonl --output /tmp/gemma-parallel.jsonl \
  --cuda --prompt-layout legacy --execution-mode parallel \
  --parallel-width 16 --request-batch-size 16 --warmup
```

- Evaluator SHA256: `59ae875a3aca828fbcf9a4f710e6bbf8874233b87d54f128621cca92a6769c32`.
- Checkpoint SHA256: `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`.
- Frozen requests SHA256: `1489716ed040e95087a6e973819ac1f346f39651db50cd2908da2b723da50a54`.
- Full predictions, commands, batch timings, confusion matrices, hashes and logs: `results/kaggle-parallel-local-20260922/` (local, Git-ignored).
