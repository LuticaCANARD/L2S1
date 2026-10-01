# Decision-model architecture review — 2026-09-27

[English](en/DECISION_MODEL_REVIEW.md) · [한국어](ko/DECISION_MODEL_REVIEW.md) · [日本語](ja/DECISION_MODEL_REVIEW.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

**L2S1 currently provides a portable, inspectable decision runtime; this does not establish the accuracy, calibration or cost structure of a trained decision model.** The requested [five-model comparison](https://fornewchallenge.tistory.com/1593) motivated this review. [Kev's primary documentation](https://github.com/jaredpalmer/kev#highlights) describes learned decision behavior and held-out calibration. Those are useful design targets; cross-project headline scores and latency ratios are not controlled measurements of our runtime. No external model code or weights were imported.

## Findings and priorities

| Priority | Evidence in this repository | Consequence and next step |
| --- | --- | --- |
| P0: Confidence is not operational risk | [DecisionPolicy and scoring](../src/decision.rs) use conditional candidate softmax, a mass floor and one global pair of thresholds. [HTTP](../src/http/contract.rs) maps `target_error_rate` to `1 - rate`; it already reports `guaranteed: false`. | The name still suggests a stronger contract than the implementation. Prefer explicit score thresholds; a future genuine risk policy needs a model/task-bound artifact, independent calibration and held-out error/coverage measurement. Scalar temperature changes confidence but cannot repair raw argmax errors. |
| P1: Repeated decoder work | [LlamaBackend::decide](../src/llama.rs) evaluates each question separately in the default fresh mode. Parallel mode batches independent sequences; [shared-state sessions](../src/llama/shared_state.rs) reuse exact complete token batches under explicit lifetime rules. | One request with many questions is not inherently one shared encoder pass. Measure 1/4/16/64 questions and short/long states, including p95 and memory. A shared state encoder plus dynamic question/option heads is a separate learned backend experiment, not a default execution-mode switch. Preserve question isolation. |
| P1: Answer codes couple semantics to tokenization | [Multi-letter scoring](../src/llama/code_sequences.rs) evaluates full continuation paths and disallows parallel/state-restore, calibration and output-head combinations. [Semantic mixtures](../src/consensus.rs) require extra passes and weight them by candidate mass. | Larger candidate sets cost additional forward work; mass also depends on answer-code paths and formatting. Mass is not an out-of-domain detector. Rotation mixtures are an opt-in mitigation, not proven order invariance. Measure paired semantic-label flips, accepted-selection flips, probability TVD and total latency before adopting them. A learned dynamic option scorer is a candidate alternative. |
| P1: HTTP scheduling differs from the worker API | [HTTP serve](../src/http.rs) owns a bounded queue but executes local jobs one by one, while [BackendWorker](../src/worker.rs) has separate opt-in microbatching. `receiver.recv()` has no inference deadline; socket timeouts do not cancel queued/native work. | Long jobs can delay short jobs; a disconnected client can leave work running. Unify admission/scheduling, carry deadlines and skip expired queued work before native dispatch. In-flight cancellation requires native support. Verify queue p95, fairness, memory and state isolation under concurrent load. Explicit batch endpoints do not automatically batch independent HTTP requests. |
| P1: Evaluation implementations have drifted | The Rust Laya scorer used last-option ties and raw-argmax correctness for accepted results, contradicting its documented semantics. It omitted already documented NLL, 15-bin ECE and per-kind metrics. | Fixed in this change, with regression tests. Existing published full-test results reproduced unchanged. Keep a shared versioned metric contract and cross-tool fixtures so Rust/Python reports cannot silently diverge again. |
| P2: Specialist evidence is not transfer evidence | [Training preparation](../training/src/l2s1_training/prepare_jev_data.py) separates exact states/IDs but explicitly cannot establish absence of near duplicates or pretrained exposure. The pinned pilot covers four seen workflows. | Add source/workflow-held-out, Korean, insufficient-evidence and adversarial-state evaluation before claiming general decision quality. Preserve untouched test labels. A model-selected `DONE` still needs an application-side observable completion check; schema validity alone cannot prove a task succeeded. |

The [current output head](../src/output_head.rs) binds to fixed option definitions, instruction and decision ID. It is a useful specialist classifier, but replacing it with a dynamic state/action contrastive head would change the training objective, feature extraction and serving contract. We have not implemented or validated that architecture in this review.

## Delivered changes

The Laya scorer now respects source-order raw ties, scores native accepted answers independently, and emits `raw_top1_accuracy`, `correct_accepted`, `accepted_correct_all`, `nll_hard`, `ece_hard_15` and `by_type`. Missing/failed labeled decisions stay in planned denominators. Reports record scorer/input hashes.

Both Laya and public JevBench reports now include `risk_coverage`: the raw argmax is ranked by maximum candidate probability, with equal-confidence samples kept together. The curve and maximum coverage at empirical 1%, 5% and 10% error budgets are descriptive diagnostics. They ignore native mass/tie gates and are separate from actual acceptance metrics. They use test labels to describe a frontier, so their thresholds must not be shipped as deployment policies or interpreted as statistical guarantees. Empty acceptance has null risk, not zero risk. Repeated cases remain separate cache-diagnostic passes in Laya reports.

## Recorded-output replay

Re-scoring both existing [typed-decisions full-test runs](TYPED_DECISIONS_BENCHMARK.md) reproduced all published raw and accepted counts, with no invalid outputs across 4,000 recorded decisions. This is an offline replay of September 26 outputs, not new inference, training, calibration, a speed measurement or evidence of an accuracy improvement.

Gemma 4 E2B accepted 1,855/2,000 decisions, including 822 wrong (44.31% accepted error). Even selecting the best threshold after seeing these test labels, a maximum-probability ranking achieved only 120/2,000 coverage (6%) at empirical risk <=5%: six errors among 120 retained predictions. Qwen3 0.6B accepted 1,112 with 730 wrong (65.65%); its corresponding empirical <=5% frontier retained only 1/2,000 (0.05%). These results concern these uncalibrated checkpoint/task configurations, not all L2S1 models. Teacher labels are synthetic, and decisions share states, so no independence-based uncertainty bound is claimed.

The compact [replay evidence](../benchmarks/decision-review-20260927/summary.json) records input/scorer hashes and error-budget counts. Complete curves are generated locally under `results/decision-review-20260927/`. Reproduce without loading a model:

```sh
cargo build --locked -p l2s1-tools
mkdir -p results/decision-review-20260927
target/debug/l2s1-tools laya-benchmark score \
  --prepared results/typed-decisions-20260926/prepared \
  --predictions results/typed-decisions-20260926/gemma4-e2b/predictions.jsonl \
  --output results/decision-review-20260927/gemma4-e2b.json
```

Use `qwen3-06b` for the other recorded run. Raw inference artifacts are local ignored files; the compact evidence does not claim to bundle them. The metric regression suite and tools clippy check pass. Runtime scheduling, trained dynamic heads and deployable risk calibration remain open engineering work.

## Prefix reuse and the supplied M5 Max comparison

The user supplied measurements from 2026-09-27 on an M5 Max, 128 GB, macOS 27.0 (26A428), AC/High Power. These are external local measurements on an uncommitted `local/metal-all-benchmarks` branch, not a rerun or an audit of per-case outputs here. GGUF/Metal and MLX differ in quantization and prompts as well as engine. The reported decision-rules p50 is per three-decision request; JevBench p50 is per item. Do not combine those units.

For Gemma 4 26B-A4B, reported decision-rules fresh p50 is 506.0 ms versus jv's 166.6 ms (3.04x); JevBench reverses this ordering: 153.1 versus 193.4 ms. This supports investigating workload structure before attributing the gap to llama.cpp. The summary's roughly 1.5x ratio with prefix reuse lacks a detailed prefix-reuse row, layout/batch settings and reused-token counts. It is a useful hypothesis, not proof of the fraction spent ingesting rules. The reported 0.313–0.457 ms jv HTTP probes measure jv transport/parsing, not L2S1 inference overhead.

The concrete local issue is request lifetime: ordinary `decide()` clears KV before and after each request, even in prefix-reuse mode. Existing `shared_state()` fixes the state and varies the questions; it does not express fixed rules/options across changing states. The new [`shared_decision()` session](SEMIF_ALGORITHM.md#fixed-schema-sessions) supplies that lifetime without changing prompts, labels or thresholds. Batch alignment still limits reuse. This native scoped API does not automatically add HTTP pooling, schema routing or SDK KV sessions.

Performance and correctness remain separate priorities. In the supplied M5 report, 26B reaches 100% raw top-1 on 36 synthetic rule decisions but only 67.5% on 2,000 typed decisions; 95.7% coverage with 68.7% accepted accuracy means 31.3% of accepted typed decisions are wrong. Prefix reuse cannot fix that risk. Small-model differences between engines require controlled prompt/template, token, quantization and candidate-order comparisons before blaming a backend or claiming a learned capability gap. MLX implementation is out of scope at the user's request.

The [231-item RTX 3080 JevBench check](JEVBENCH.md#rtx3080-rerun-20260927) found only 6.48–7.20% token reuse at batch 64. It preserved matched-batch outputs but was slower than batch-256 fresh for both Qwen3 0.6B and Gemma 4 E2B. Batch size itself changed several top-1 answers. Thus the CPU rule-fixture gain is workload-specific; keep batching and accuracy checks in the optimization loop.
