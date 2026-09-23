# Parallel question execution

The optional `parallel` mode evaluates independent questions using separate llama.cpp sequence IDs. It shares the loaded model, computes an exact common prompt prefix once per wave, and puts suffix tokens from multiple questions into each decode batch. It does not start multiple threads that concurrently mutate one llama context.

```sh
cargo run --release --locked --features llama -- \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --input examples/warehouse.json --device cuda \
  --prompt-layout state-first --execution-mode parallel --parallel-width 4
```

Defaults remain `legacy` prompts and `fresh` execution. Prompt layout and execution mode are separate choices: moving state changes the prompt; parallelism changes the execution schedule. Compare all execution modes at the same layout.

## Implementation

- Rust prepares and validates each question's tokens and A–Z single-token continuations. User state and question data retain special-token parsing protection.
- At most `parallel_width` questions enter each wave (default 4, allowed 1–32). The last wave can be smaller. Results retain request order and IDs, including when prompts finish in different decode batches.
- A common prefix is found by comparing exact token IDs across the whole wave, retaining at least one suffix token per question. It is rounded down to a complete original prefill batch. The first sequence evaluates it; `llama_memory_seq_cp` shares its KV entries with the other sequences.
- Suffix tokens have independent sequence IDs and original absolute positions. Their attention sees their own sequence and shared prefix, not another question's suffix. Final full-vocabulary logits are copied using each question's final token index in the relevant decode batch.
- Memory is cleared between waves, API calls, and after failures. There is no persistent cross-request cache. Recurrent/hybrid models are explicitly unsupported in this mode.
- The model stays loaded once. The context is recreated lazily when sequence capacity changes, with nominal token capacity `context * width`. `context` remains the per-question input limit. Higher widths increase KV/attention memory and may fail allocation; there is no silent CPU fallback or input truncation.
- `backend.parallel_width` records the configured wave limit (serial modes report 1). `reused_prefix_tokens` is zero for the wave's first question, which paid for the shared prefix, and the shared token count for its followers. Summed `input_tokens - reused_prefix_tokens` accounts for actual submitted tokens.

This uses the pinned llama.cpp [batch, sequence, memory-copy and per-token logits API](https://github.com/ggml-org/llama.cpp/blob/3d82ef62d47fd74e18f36c5eccbdcf965b617b17/include/llama.h). It is independent sequence batching with serial waves, not a replacement model architecture or calibrated-decision training.

### Independent request batches

`backend.decide_batch(&requests)` evaluates multiple requests without merging
their states or changing their question prompts. Parallel mode flattens the
questions into isolated sequences, processes bounded waves, and restores the
original request/decision grouping. Repeated decision IDs in different requests
are allowed. Only exact token prefixes can be shared within this explicit call;
no KV survives the call. A failure returns an error for the whole batch, without
partial response results. An empty batch returns an empty result list.

The JSONL evaluator supports `--execution-mode parallel --parallel-width 16
--request-batch-size 16`, plus an optional `--warmup` batch. This enables a fair
comparison on the same independent article requests. It records full batch
completion latency for each article, batch identity, and amortized compute time
separately. Dividing a batch's elapsed time by its size measures amortized cost,
not individual response latency. See [KAGGLE_PARALLEL_RESULTS.md](KAGGLE_PARALLEL_RESULTS.md)
for the 400-article comparison.

## Validation and measurement

```sh
export LLAMA_CPP_DIR=/path/to/llama.cpp
export LLAMA_LIB_DIR="$LLAMA_CPP_DIR/build-cuda/bin"
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  cargo test --release --locked --offline --features llama \
  --test parallel real_model_parallel_contract -- --ignored --nocapture
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  SKID_PARALLEL_WIDTH=4 SKID_PARALLEL_OUTPUT=/tmp/parallel.json \
  cargo test --release --locked --offline --features llama \
  --test parallel real_model_parallel_measurement -- --ignored --nocapture
```

The contract test covers mixed typed questions, output IDs/order, unequal prompt lengths, a partial wave, shared-prefix reuse, A–Z candidate mapping, untrusted token-like text, cross-question isolation, changed-state request isolation, error recovery after an earlier wave completed, invalid widths, and width-one equivalence to fresh execution.

The measurement uses 1, 4, 16 and 32 questions on short and long warehouse states. Questions repeat three warehouse criteria to measure scaling; they are not 32 distinct ground-truth tasks. Each mode has one untimed warmup (including any context allocation) and three timed repetitions. Mode order alternates between configurations. Detailed JSON retains every score and latency. It reports serial-versus-parallel differences instead of asserting that distinct batch shapes are numerically identical; 0.02 remains the existing probability/mass comparison threshold and any changed top-1 or accepted selection fails the reported equivalence criterion.

These measurements exclude model/context startup from steady-state latency, do not include a remote API/network, and do not establish Jev performance parity or calibrated confidence. The preexisting Kaggle reports used one question per request and do not measure this mode.

## Gemma 4 CUDA results — 2026-09-22

Checkpoint: Gemma 4 E2B IT Q8_0, RTX 3080, four threads, per-question context 2,048, decode batch 256, state-first v2. Times below are median total request milliseconds across three measured repetitions after one warmup. Width 4 bounds each wave to four questions; width 32 permits all tested questions in one wave.

| State | Questions | Fresh | Serial prefix reuse | Parallel width 4 | Parallel width 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 1 | 32.55 | 23.85 | 24.16 | 25.29 |
| Short | 4 | 127.87 | 137.52 | 94.29 | 104.31 |
| Short | 16 | 520.74 | 527.01 | 400.80 | 372.02 |
| Short | 32 | 1058.13 | 1050.37 | 815.43 | 835.62 |
| Long | 1 | 117.70 | 118.47 | 124.14 | 118.28 |
| Long | 4 | 521.60 | 350.85 | 260.42 | 256.91 |
| Long | 16 | 2050.54 | 1055.77 | 1050.23 | 927.45 |
| Long | 32 | 4059.32 | 2080.84 | 2113.40 | 2121.91 |

Serial references in this table come from the width-32 experiment. Width-4 results are from its separate experiment, so small differences across experiments are not controlled comparisons. Each JSON preserves its own paired serial references.

| Width | Paired decision comparisons | Changed top-1 | Changed selected/abstained | Max probability delta | Max mass delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 318 | 0 | 0 | 0.00020950 | 0.00000353 |
| 32 | 318 | 0 | 0 | 0.00022185 | 0.00000375 |

The thresholds were not relaxed. These are observed fixture comparisons, not a guarantee that parallel mode preserves every model’s scores.

### Labeled synthetic fixture

| Mode | Correct / 36 | Wrong | Abstained | Raw top-1 correct |
| --- | ---: | ---: | ---: | ---: |
| fresh | 33 | 2 | 1 | 34 |
| prefix_reuse | 33 | 2 | 1 | 34 |
| parallel | 33 | 3 | 0 | 33 |

This is the existing 12-request/36-decision synthetic rules fixture, evaluated at state-first v2. It is not the legacy Kaggle accuracy benchmark.

**The broader labeled fixture does not pass serial-equivalence.** For `warehouse-01 / dispatch_priority`, fresh execution abstained with probabilities medium 0.47755 and high 0.51107; parallel execution selected the wrong medium level with probability 0.82361. The maximum candidate-probability difference was 0.34606, exceeding the unchanged 0.02 threshold. Fresh and serial prefix reuse had 33 correct, 2 wrong, 1 abstained; parallel had 33 correct, 3 wrong, 0 abstained. The earlier repeated-warehouse scaling measurements therefore cannot establish general equivalence. Keep parallel mode experimental and explicitly opt-in. Do not increase confidence or equivalence thresholds merely to hide this discrepancy.

### Verification evidence

- Gemma 4 CUDA contract test passed. Ordinary default-feature and llama-feature Rust tests, formatting, and all-target release Clippy passed.
- Detailed scores, repetitions and source/executable hashes: `results/parallel-20260922/` (local, Git-ignored).
- No model training or probability calibration was performed. Parallel suffix processing reduces repeated execution cost but does not implement Jev’s training or establish API-level performance parity.
