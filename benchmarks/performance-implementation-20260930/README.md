# Decision performance implementation — 2026-09-30

Implements the first three items from the [performance review](../../docs/PERFORMANCE_REVIEW.md): a gated compute tuner, resident evaluator sessions, and opt-in structured arithmetic/time/graph facts. [Usage](../../docs/DECISION_PERFORMANCE.md).

Base: main `6b2fcfa`; exact measured source, binary, model, input and runtime hashes are retained in the artifacts below. No model weights, universal compute defaults, prompt defaults, or acceptance thresholds were changed.

## Conditions and reporting

RTX 3080 10 GiB, i9-9900K, four inference threads, WSL2, CUDA 12.4.131, driver 596.21. Pinned llama.cpp `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. Native Rust in-process, one model process at a time, one ordinary untimed warmup batch, three timed passes in alternating configuration order. Context 4096 throughout measured studies. Models: Gemma 4 E2B Q8_0 and Gemma 4 12B QAT Q4_0.

Native placement logs confirm E2B 36/36 layers and 12B 49/49 layers offloaded. CUDA model buffers: 2342.30 / 6637.69 MiB; CPU-mapped model buffers: 2788.00 / 787.50 MiB respectively. Layer offload is not a claim that every model byte resides on GPU. Reported GPU maxima sample the whole board, including desktop and loading; RSS/HWM samples include loading. They are separate from timed inference and are not allocator-exact peaks.

Tables use median of run p50s and worst run nearest-rank p95. Full ranges, stage timings, throughput, memory and probability drift are in `summary.json`. Repeats assess timing/stability: they do not multiply the accuracy denominator. Policy: top probability >=0.8, candidate mass >=0.05, no top tie. Scores are not guarantees of correctness.

## 1. Compute tuning

Development fixture: 12 requests / 36 unique rule decisions, with varied audit-history lengths. All eight profiles retained 36/36 raw and accepted-correct decisions in all three runs. Layout was explicitly Legacy in every profile, so compute comparisons did not change prompt order.

| Profile | p50 ms | worst p95 ms | GPU sampled max MiB |
| --- | ---: | ---: | ---: |
| baseline | 492.07 | 991.02 | 9593.0 |
| flash | 470.93 | 904.83 | 9461.0 |
| flash-ub128 | 487.94 | 1049.19 | 9335.0 |
| flash-b128 | 486.81 | 1041.51 | 9335.0 |
| parallel-dynamic-1 | 485.96 | 1041.40 | 9337.0 |
| parallel-dynamic-2 | 526.29 | 1063.76 | 9282.0 |
| parallel-dynamic-3 | 530.66 | 1097.71 | 9289.0 |
| parallel-static-3 | 5700.29 | 12752.71 | 9593.0 |

FlashAttention alone reduced median p50 by 4.30%, below the predeclared 5% gate. The tuner selected **baseline**. Probability/mass drift stayed below 0.02 on this small fixture. This does not establish equivalence on other tasks; JevBench below demonstrates that limit.

Dynamic allocation greatly reduced the fixed width-3 parallel slowdown, but parallel still did not beat fresh on this fixture. High board memory and the static-context slowdown were observed together; no page-migration trace was collected to establish its internal cause.

## 2. Actual native prefix reuse

Each profile evaluates 12 changing states, all 12/12 raw and accepted-correct. The schema/manual is deliberately long enough for batch-aligned reuse; short unrelated production prompts can reuse zero tokens. Batch/ubatch 128, FA on, Legacy. Fixed schema uses one request per call; shared evidence uses two requests per call, width 2 and dynamic context. Shared-evidence latency is **completion latency for the two-request call**, not divided by two.

The resident scope opens after ordinary warmup, so its first measured call is cold. `split-cold` uses the same execution mode without retention across calls. Counts below are actual reused tokens per pass, including within-call sharing where applicable.

| Model / fixture | Mode | p50 ms | worst p95 ms | Reused tokens/pass | Max probability drift vs fresh |
| --- | --- | ---: | ---: | ---: | ---: |
| e2b / schema | fresh | 99.98 | 120.16 | 0 | 0.000000 |
| e2b / schema | split-cold | 100.91 | 134.75 | 0 | 0.000000 |
| e2b / schema | split-resident | 29.43 | 111.89 | 4224 | 0.000000 |
| e2b / shared | fresh | 284.29 | 305.74 | 0 | 0.000000 |
| e2b / shared | split-cold | 159.19 | 198.66 | 3840 | 0.000000 |
| e2b / shared | split-resident | 37.60 | 182.23 | 6400 | 0.000000 |
| 12b / schema | fresh | 227.30 | 236.81 | 0 | 0.000000 |
| 12b / schema | split-cold | 227.74 | 236.95 | 0 | 0.000000 |
| 12b / schema | split-resident | 60.96 | 232.26 | 4224 | 0.000000 |
| 12b / shared | fresh | 623.18 | 633.44 | 0 | 0.000000 |
| 12b / shared | split-cold | 320.42 | 366.61 | 3840 | 0.000005 |
| 12b / shared | split-resident | 45.04 | 363.38 | 6400 | 0.000005 |

Both models passed fixed-schema and measured FA-on resident-parallel error recovery and post-drop isolation tests. Dynamic allocation was primed for the recovery reference: allocation growth itself clears KV and can legitimately alter reuse counts. Cold-first and subsequent-call timings remain separate in the JSON evidence.

## 3. Structured-input accuracy

The helper performs checked i64 arithmetic, signed UTC-second differences and directed shortest paths using explicit JSON pointers. It preserves source fields and adds auditable facts. The model still chooses an option and applies its policy. This measures the value of supplied computational evidence; it is not a new general reasoning capability or a result on natural-language JevBench.

Development: 36 instances (12 per family). Freeze rule: enable a model/family only if raw correct increases, wrong accepted does not increase, and p95 is at most twice baseline. E2B enabled numeric/time and rejected graph (wrong accepted 1→4 on development). 12B enabled all three. Selection files were frozen before holdout evaluation.

Holdout: **120 unique instances per model**, 40 per family, disjoint numeric ranges and graph node namespaces. Same templates are reused, so these results do not establish cross-domain generalization. Selection was not changed after observing holdout.

| Model / mode | Raw correct / 120 | Accepted correct / accepted | Wrong accepted | Abstained | p50 / worst p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| e2b / raw | 50/120 (41.67%) | 25/52 | 27 | 68 | 34.81 / 64.88 |
| e2b / facts | 80/120 (66.67%) | 62/77 | 15 | 43 | 35.81 / 66.45 |
| 12b / raw | 77/120 (64.17%) | 62/78 | 16 | 42 | 86.43 / 152.67 |
| 12b / facts | 106/120 (88.33%) | 98/103 | 5 | 17 | 89.20 / 158.44 |

e2b: 34 fixed, 4 regressed; paired gain 25.00 percentage points, instance-bootstrap 95% interval [15.83, 34.17]. Shared templates limit this interval's scope. Timing includes preprocessing; model loading and upfront input validation are excluded.

12b: 29 fixed, 0 regressed; paired gain 24.17 percentage points, instance-bootstrap 95% interval [16.67, 31.67]. Shared templates limit this interval's scope. Timing includes preprocessing; model loading and upfront input validation are excluded.

## JevBench public regression

Pinned revision `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`: 48 easy + 72 original + 111 hard = **231 unique items**, not the full 534-item suite. This public set had already been examined; it is a regression set, not the untouched selection holdout. Original requests were retained and structured facts were disabled. Fresh/Legacy, context 4096, batch/ubatch 256, four threads. The off/on comparison was fixed before reading new public outcomes.

| Model / FA | Raw correct / 231 | Accepted correct / accepted | Wrong accepted | Abstained | p50 / worst p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| e2b / off | 159/231 (68.83%) | 153/213 | 60 | 18 | 36.91 / 452.69 |
| e2b / on | 158/231 (68.40%) | 149/208 | 59 | 23 | 34.94 / 377.93 |
| 12b / off | 194/231 (83.98%) | 181/203 | 22 | 28 | 89.59 / 1351.48 |
| 12b / on | 195/231 (84.42%) | 182/203 | 21 | 28 | 86.56 / 1131.02 |

e2b FA-on vs off: 2 top-1 changes, 7 accepted-selection changes, max probability drift 0.317868. Counts were stable in all three repeats. Speed does not justify silently changing compute/calibration identity or acceptance behavior.

12b FA-on vs off: 6 top-1 changes, 2 accepted-selection changes, max probability drift 0.199328. Counts were stable in all three repeats. Speed does not justify silently changing compute/calibration identity or acceptance behavior.

The Rust adapter's first run for all four model/config pairs was independently checked against pinned upstream `score_task` and `summarize`, including accuracy, calibration metrics and latency. No truncation or failed predictions occurred. Baseline counts reproduce the earlier 159/231 E2B and 194/231 12B results.

## Validation and known failing profile

New deterministic-fact unit tests, evaluator input-contract tests, tuner failure/selection tests, native resident recovery/isolation and fixed-layout compute-resize tests passed. The compute test now explicitly fixes Legacy layout, avoiding an invalid comparison between different prompts. CI also rebuilds the evaluator with the native CPU backend and replays the committed evidence.

**Known failed existing test:** `tests/parallel.rs::real_model_prefix_sharing` on E2B, FA off, batch 256, width 4, static context 2048, Legacy: probability drift versus fresh was **0.1142807672**, exceeding its existing **0.05** limit. The limit was not relaxed and this profile is not claimed equivalent. The measured resident tests instead isolate cross-call retention within the same FA-on profile. This PR adds tooling around existing native sessions; it does not claim to fix that legacy fresh/parallel drift.

## Evidence and reproduction

- [Summary](summary.json): all 3-pass metrics, family counts, drift, uncertainty and memory.
- [Records](records.jsonl), [record hash](records-sha256.json), [provenance](provenance.json).
- [Protocol](protocol.json), [JevBench protocol](jevbench-protocol.json), [E2B selection](e2b-accuracy-selection.json), [12B selection](12b-accuracy-selection.json).
- `hardware.json`, `implementation-hashes.json`, `validation.json` and `jevbench-audit.json` record the environment and checks. JevBench data is MIT licensed; see [license](JEVBENCH-LICENSE).

From the repository root, run `python3 benchmarks/performance-implementation-20260930/verify.py` to recompute every reported run without a model. `prepare.py` deterministically recreates synthetic inputs and gold. Build the evaluator with `cargo build --release --locked --features llama-cuda --example evaluate_jsonl`.

Use `scripts/benchmark_decision_performance.py` with the checked-in input/gold/matrix files, model GGUF, native library directory, a new output directory and `--repeats 3`. Use `--evaluation-only` for schema/shared/accuracy/JevBench studies. Compute tuning omits that flag and writes a gated `selection.json`. Run E2B and 12B sequentially.

After separate `accuracy-dev` runs under `<results>/e2b` and `<results>/12b`, run `freeze_accuracy.py --results <results> --selection-dir <results>/selections` before evaluating the generated per-model `selected-holdout.jsonl` with `accuracy-matrix.json` and `holdout-gold.json`.

For JevBench, use `l2s1-tools jevbench-public prepare` with the pinned upstream checkout, evaluate its unchanged `requests.jsonl` with `jevbench-matrix.json` and `jevbench-gold.json`, then use `jevbench-public score`. `audit_jevbench.py --upstream <checkout> --results <results>` checks upstream parity. `report.py --results <results>` exports compact records and metrics; `render.py` generates this report. All scripts require only the Python standard library except the pinned upstream scorer's own dependencies.
