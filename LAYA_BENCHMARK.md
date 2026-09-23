# Laya/Jev task adapter and cache evaluation

`scripts/laya_benchmark.py` evaluates the public tasks used by
[Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark)
with the local L2S1 JSONL evaluator. It does not load Laya, train on evaluation
labels, or produce the separate JevBench composite score.

| Suite | Frozen evaluation data | Scope |
| --- | --- | --- |
| `phish` | `AreLit/PhishNChips`, `core_emails.csv` | 2,000 emails, balanced binary labels |
| `typed` | `LocalLLaMA/typed-decisions`, `all/test` | 400 cases, 2,000 binary/choice/ordinal decisions |
| `probes` | Luni's literal probe definitions | Nine requests, 14 unique questions: grounding, contradiction and routing variants |

Dataset and benchmark revisions are pinned in the adapter. Fetch records source
URLs and file SHA256 values. Probe definitions are parsed as literal Python AST
values; downloaded Python code is never imported or executed. Datasets and
generated evidence belong under ignored `results/`, not in the source package.
Source datasets retain their own terms; no model checkpoint is downloaded.

## Prepare and run

The adapter and scoring use the Python standard library. Only preparation of the
typed Parquet file requires `pyarrow` in your Python environment.

```sh
python3 scripts/laya_benchmark.py fetch --suite typed --output results/laya/typed-source
python3 scripts/laya_benchmark.py prepare --suite typed \
  --source results/laya/typed-source --output results/laya/typed

cargo build --release --locked --features llama --example evaluate_jsonl
python3 scripts/laya_benchmark.py run \
  --prepared results/laya/typed --output results/laya/fresh \
  --evaluator target/release/examples/evaluate_jsonl --model /path/to/model.gguf
```

Use `phish` or `probes` in the same fetch/prepare flow. For CUDA, supply `--cuda`
and the placement required by your checkpoint, such as `--gpu-layers 24
--model-load-mode read`. Use the matching native libraries described in the
[build guide](README.md#build). A small integration check can use `prepare
--limit 4`; its manifest and report explicitly mark the result as a subset.

Each JSONL case contains all its decisions over one state. The evaluator no
longer forces each question into a separate request. Only case IDs, state,
question instructions and criteria enter inference; gold labels, distributions,
workflow tags and rationales stay in a separate file. Option order, binary
polarity and ordinal levels are preserved. The engine's defaults remain legacy
prompts, fresh execution and disabled preparation caching.

Every run records evaluator/model/input hashes, the command, backend settings,
all predictions, errors, timings and preparation-cache counter snapshots. Missing,
duplicate, unexpected, truncated or malformed outputs cannot silently improve
accuracy. Overall accuracy counts missing/failed labeled decisions against the
planned denominator; valid-only accuracy is named separately. A failed run keeps
its evidence and exits unsuccessfully.

## Scoring boundaries

- Raw accuracy is computed before abstention. Accepted accuracy, wrong accepted
  decisions and coverage are separate.
- Binary ties use the upstream `p_true >= 0.5` rule. Choice/ordinal ties use the
  original option order. Native acceptance still follows the engine policy.
- Phishing reports tie-aware AUROC, recall, precision and hard-label Brier.
  The upstream handwritten AUROC ranks ties arbitrarily; this adapter gives ties
  half credit instead.
- Typed tasks report hard accuracy, soft-target Brier/TVD/KL/soft accuracy for
  binary and choice questions, and expected-level MAE/within-one for ordinal
  questions. Rounded soft targets are normalized. Hard-label Brier is separate.
- `ece_hard` uses ten equal-width bins against argmax correctness. It is not a
  measurement of agreement with the soft teacher distribution, and is not assumed
  interchangeable with an unspecified upstream ECE implementation.
- Probe failures are reported per check. The complement and stability tests are
  heuristics: some paired questions are not strict logical complements, and
  routing variants also change rubric wording. Grounding outputs are reused for
  the overconfidence check, which explicitly uses L2S1 entropy confidence.
  No universal "11-test accuracy" or teacher-agreement ceiling is inferred.
- GPU 5090 timings published by the source are not directly comparable to a
  local CPU/3060 run. Per-case latency includes all questions and full batch
  completion; dividing by questions or batch size is amortized work, not latency.

## Compare caching without changing the task

Use immediate repeats only for cache diagnostics; repeat zero is the quality
result. Later repeats are never counted as additional independent examples.

```sh
python3 scripts/laya_benchmark.py prepare --suite typed \
  --source results/laya/typed-source --output results/laya/cache-input \
  --limit 4 --repeats 2

python3 scripts/laya_benchmark.py run \
  --prepared results/laya/cache-input --output results/laya/state-fresh \
  --evaluator target/release/examples/evaluate_jsonl --model /path/to/model.gguf \
  --prompt-layout state-first --execution-mode fresh --batch 64 \
  --cache-bytes 67108864

python3 scripts/laya_benchmark.py run \
  --prepared results/laya/cache-input --output results/laya/state-reuse \
  --evaluator target/release/examples/evaluate_jsonl --model /path/to/model.gguf \
  --prompt-layout state-first --execution-mode prefix-reuse --batch 64 \
  --cache-bytes 67108864

python3 scripts/laya_benchmark.py compare --prepared results/laya/cache-input \
  --baseline results/laya/state-fresh --candidate results/laya/state-reuse \
  --output results/laya/cache-comparison.json
```

The comparison requires identical checkpoint, evaluator, inputs, prompt identity,
compute settings and policy. It checks every probability, candidate mass, raw
top choice and accepted selection. Changing layout or batch size needs its own
fresh baseline: it is not a cache-only comparison. A 0.02 probability/mass
regression tolerance never excuses a changed top choice or accepted selection.
Exact equality is also reported. Warmup is untimed and its preparation entries
are cleared before measurement. KV is still cleared at normal request boundaries.

## Why a CPU workload can show little cache benefit

CPU placement does not disable KV reuse. The native implementation compares exact
token prefixes before dispatching work to any CPU/GPU layers. There are three
different kinds of work to distinguish:

| Mechanism | Work saved | Current boundary |
| --- | --- | --- |
| Preparation cache | Prompt compilation/tokenization and answer-boundary mappings | Exact state plus decision for prompt entries; model/configuration scoped and byte bounded |
| Prefix KV reuse | Recomputing transformer states for shared input tokens, including CPU layers | Questions within a request, or an explicit immutable-state session |
| Result memoization | Entire inference | Not enabled by this adapter; would measure a different workload |

With the default prefill batch of 256, a common prefix of 255 tokens saves **zero**
tokens. Reuse is rounded down to a complete original batch to preserve numerical
behavior; the final token is always evaluated. A single question per request,
ordinary separate calls, legacy instruction-first layout, changed early tokens,
or unsupported recurrent/hybrid memory can therefore leave reuse at zero.
Preparation hits alone do not show that transformer work was avoided. Inspect
`reused_prefix_tokens` and `native_ms`, not just cache-hit counts.

A CPU integration check on September 24, 2026 used SmolLM2 135M Q8_0 on an
i5-12600KF, four threads, state-first prompts, the first two typed test cases
(five decisions each), and one immediate repeat. At batch 256, preparation
caching reduced preparation from 16.45 to 7.93 ms, but total inference stayed
around 3.4 seconds and KV reuse was zero. At batch 64, fresh versus prefix reuse
took 3.899 versus 2.389 seconds (1.63x), with 2,048 reused tokens. All 20
probability vectors, candidate masses, top choices and accepted selections were
exactly equal within that same-batch comparison. This small serial check
demonstrates CPU reuse, not a 31B or production speed guarantee; batch 64 also
made the fresh path slower than batch 256.

The same server also checked Gemma 4 31B Q4_K_M with CPU/GPU split (24 offloaded
layers, eight threads, read loading). On the same two typed cases and their
repeats, batch 64 fresh/reuse took 108.552/69.904 seconds (1.55x); batch 128 took
62.438/42.770 seconds (1.46x). Both same-batch cache comparisons preserved all
20 probability vectors, candidate masses and selections exactly. However, the
matching batch-256 fresh cases took 41.677 seconds. Neither smaller-batch cached
configuration beat that baseline. Across batch sizes, maximum probability
changes were 0.00165/0.00504 with no changed selections on this small sample.
Keep batch 256 as the default; CPU cache effectiveness does not imply a net gain
after changing prefill geometry. All 15 integration configurations (350 decision
records including repeats) completed without errors or truncation. This covers
small typed/phishing samples and the complete behavioral probe set, not full
inference on the 4,000 labeled decisions prepared by the adapter.

## Speed work in priority order

1. **Keep one model resident and preserve case-level grouping.** The adapter now
   does this. For changing questions about an immutable state across API calls,
   use `SharedStateSession`; normal requests remain isolated.
2. **Measure state-first with a same-layout fresh baseline.** It exposes shared
   evidence before question-specific suffixes. Changing layout can change model
   predictions, so it remains opt-in.
3. **Test smaller prefill batches on short CPU states.** Batch 64/128 can make
   short prefixes reusable, but also reduces matrix-multiplication efficiency.
   Compare fresh and cached at each size; do not remove the batch-alignment guard.
4. **Use bounded preparation caching for repeated inputs and fixed schemas.**
   Candidate mappings can hit across changed states, while exact prompt entries
   will miss. If native inference dominates, token preparation savings alone
   cannot materially change total latency.
   Also inspect `boundary_and_other_ms`: it includes validation, request-boundary
   clearing and other work outside the phase timers. The native `sd_clear` zeros
   KV buffers, including host buffers, and ordinary batch/request boundaries can
   call it repeatedly. Removing redundant clears or separating logical reset
   from physical zeroing is a follow-up optimization requiring isolation/error
   recovery tests; this adapter does not change those semantics.
5. **Evaluate parallel width under the actual memory budget.** It can improve
   throughput, but grows KV memory and can change numerical results. On a 31B
   model already near 12 GiB VRAM, more offloaded layers or sequences may fail.
   `--request-batch-size` alone does not create native parallelism in fresh mode.
6. **Match context allocation to measured input lengths before increasing GPU
   offload.** Short classification inputs may not need an 8,192-token KV buffer.
   A smaller explicit context can free GPU memory for more layers, reducing CPU
   computation. Check every input and answer-code prefix; never silently truncate.
   Recheck quality and allocation with the new context/placement combination.
7. **Consider explicit fixed-schema prefix sessions as a later feature.** Ordinary
   cross-request KV reuse is not implemented here. Such an API needs exact
   token-prefix checks, model/template/adapter identity, bounded memory, explicit
   ownership and error invalidation. Do not confuse it with an answer cache or
   reuse cached answers for semantically similar inputs.

Read loading changes allocation and startup RSS, not the amount of transformer
inference. Quantization, CPU thread count and layer placement are separate tuning
axes; each must preserve an independently checked quality baseline. A smaller
first-stage model with 31B fallback would also change the inference policy and
needs its own accuracy/coverage evaluation.
