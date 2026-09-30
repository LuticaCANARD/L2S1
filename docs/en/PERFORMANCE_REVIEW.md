# Performance review — 2026-09-30

[English](PERFORMANCE_REVIEW.md) · [한국어](../ko/PERFORMANCE_REVIEW.md) · [日本語](../ja/PERFORMANCE_REVIEW.md)

Review of main `6b2fcfa` / released v0.2.0. The experiment table records the
original review; the implementation follow-up below links subsequent measurements. Existing measurements use the
revisions named below. [Evidence and replay](../../benchmarks/decision-performance-20260930/README.md).

## Baseline and failure modes

| Model | Correct / 231 | Accuracy | p50 / p95 ms | Accepted correct / accepted |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 159 | 68.83% | 40.41 / 485.74 | 153 / 213 |
| Gemma 4 12B QAT Q4_0 | 194 | 83.98% | 93.48 / 1359.39 | 181 / 203 |
| Laya English 421M | 133 | 57.58% | 45.18 / 71.63 | 133 / 231 (no abstention policy) |

JevBench revision `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`: 48 easy,
72 original, 111 hard public items, one run each. This is not the full 534-item
leaderboard. Gemma: L2S1 `74334ec`, RTX 3080 10 GiB, fresh/legacy, context 4096,
batch/ubatch 256, four threads, FlashAttention off, native Rust in-process.
Laya: official SDK 0.3.21, checkpoint `55cf4c4e`, PyTorch 2.8.0 CUDA/BF16 autocast,
in-process Python, default 512-token budget. Loading and one warmup excluded.
Laya truncated 57 items; Gemma truncated none. Separate runs, different kernels,
quantization and context limits prevent a controlled engine-only comparison.

12B still misses 11/15 temporal/numeric tasks, 6/18 multi-hop tasks and 6/19
long-policy tasks. It accepts 22 wrong answers. E2B has 37 wrong answers with
top score >=0.99; 12B has 10. Candidate scores are not correctness guarantees.

## Recommended experiment order

| Priority | Experiment | Existing implementation / change needed | Required evidence |
| --- | --- | --- | --- |
| 1 | Profile and tune 12B memory/prefill | FlashAttention off/on, batch/ubatch 128/256, width 1/2/3, `set_parallel_context_dynamic`; existing controls | Per-stage time, actual GPU placement, peak memory, p50/p95, raw and accepted answer changes |
| 2 | Reuse fixed questions or shared evidence | Existing resident fixed-schema reuse; `shared` and `parallel_prefix_session()` for exact common prefixes | Fresh vs split-cold vs split-warm, actual reused tokens, same-schema changing states, numerical and error-recovery checks |
| 3 | Improve hard-task accuracy | Validate concise criteria and numeric/time preprocessing; task-specific LoRA or output head using separate training data | Freeze on development data, then evaluate untouched grouped holdout; include 12B and a smaller model |
| 4 | Calibrate acceptance and escalation | Existing family/task calibration plus application routing | Risk versus coverage, wrong accepted count, accepted-correct/all, end-to-end latency and compute cost on holdout |

The small synthetic rule study (36 unique decisions, repeated three times) found
12B parallel/state-first slower than fresh/state-first: p50 313.23 vs 243.59 ms.
E2B improved from 105.52 to 72.01 ms on that fixture. Every configuration reported
zero reused prefix tokens. The 12B parallel board-memory sample reached 10013 MiB
on a 10 GiB GPU, including desktop memory; this motivates a memory experiment,
but does not prove memory pressure caused the slowdown. Do not make parallel
execution the universal fast default on this evidence.

Code pointers: `src/llama.rs` compute settings and dynamic parallel context;
`src/llama/shared_decision.rs` fixed-schema sessions;
`src/llama/parallel_session.rs` cross-call shared-prefix retention;
`src/llama/interchange.rs` configuration/calibration identity. Short prefixes
below the batch boundary can legitimately reuse zero tokens. Token alignment
can change numeric scores and requires separate validation.

## Why a simple two-model cascade is insufficient

Replay the existing E2B acceptance rule (top probability >=0.8, candidate mass
>=0.05, no top tie), sending only abstentions to 12B and retaining its policy:

| Fixed policy | Routed to 12B | Raw correct | Accepted correct / accepted | Abstained |
| --- | ---: | ---: | ---: | ---: |
| E2B then 12B | 18 / 231 | 166 / 231 (71.86%) | 159 / 222 (71.62%) | 9 |
| 12B alone | 231 / 231 | 194 / 231 (83.98%) | 181 / 203 (89.16%) | 28 |

This is arithmetic on recorded predictions, not a new deployed run. No latency,
co-resident VRAM, loading or routing-cost gain is claimed. High-confidence wrong
E2B answers remain on the small model. Any new router must be fitted on separate
data and tested with both models' memory requirements and actual transport.

## Experiment contract and boundaries

Keep model, dataset, source, runtime and adapter hashes. Change one setting at a
time before combined tests; interleave baseline/candidate order and measure at
least three timing runs with identical warmup. Repeated items are not additional
accuracy samples. Report request length, errors/truncation, raw accuracy,
coverage, accepted accuracy, accepted-correct/all, wrong accepted, p50/p95,
throughput and memory. Compare same-model score/probability drift explicitly.
Set a latency target and a permissible accuracy/coverage change before running;
reject a candidate that violates either. Public JevBench already examined here
is a regression set, not an untouched selection holdout.

Start with experiment 1 and 2; their controls already exist. Token-level KV
quantization would require a new bridge/configuration/identity path and numerical
validation; it is not an exposed L2S1 optimization today. Speculative decoding
targets multi-token generation and is a low priority for direct one-step scoring.
Threshold calibration can change acceptance without fixing raw predictions.

For the separate Laya comparison, test an explicit long-context checkpoint or
windowing and record changed model/context, latency and aggregation; never assume
recovering truncated tokens will recover the missing accuracy. The upstream
[Laya model card](https://huggingface.co/convaiinnovations/laya) documents the
different checkpoint limits. Upstream [llama.cpp performance guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md)
also recommends checking actual GPU offload and thread counts; its generation
numbers are not L2S1 latency estimates.

## Implementation follow-up

The first three experiments now have [runnable tooling and explicit structured-fact preprocessing](../DECISION_PERFORMANCE.md). See the linked study for measured results and limits.
