# Parallel question execution

The optional `parallel` mode evaluates independent questions using separate llama.cpp sequence IDs. It shares the loaded model, computes an exact common prompt prefix once per wave, and puts suffix tokens from multiple questions into each decode batch. It does not start multiple threads that concurrently mutate one llama context.

```sh
cargo run --release --locked --features llama-cuda -- \
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
- The model stays loaded once. By default, the context is recreated lazily when sequence capacity changes, with nominal token capacity `context * width`. `context` remains the per-question input limit. Higher widths increase KV/attention memory and may fail allocation; there is no silent CPU fallback or input truncation.
- `backend.parallel_width` records the configured wave limit (serial modes report 1). `reused_prefix_tokens` is zero for the wave's first question, which paid for the shared prefix, and the shared token count for its followers. Summed `input_tokens - reused_prefix_tokens` accounts for actual submitted tokens.

This uses the pinned llama.cpp [batch, sequence, memory-copy and per-token logits API](https://github.com/ggml-org/llama.cpp/blob/3d82ef62d47fd74e18f36c5eccbdcf965b617b17/include/llama.h). It is independent sequence batching with serial waves, not a replacement model architecture or calibrated-decision training.

### Dynamic parallel KV context

`--parallel-context-dynamic` is an opt-in memory setting for `parallel` mode. After tokenizing each wave, it reserves at most `sum(input_tokens) + batch` KV slots, capped at the legacy `context * width` reservation. llama.cpp may pad the requested size. The sum counts shared prefixes more than once, so this is conservative. `--context` still validates each individual question; no input is truncated.

The native context grows when a later wave needs more slots. It retains that high-water capacity for later waves with the same effective question count to avoid repeated context rebuilds. Switching between default and dynamic mode recreates the context, and both modes clear request-local KV after inference. `backend.parallel_context_dynamic` identifies the opt-in result; `backend.parallel_context_tokens` reports the currently allocated padded context in dynamic mode. The model weights remain loaded.

This changes the llama.cpp context size and may change logits or decisions. Compare the exact model, prompts, policy and input set against the default parallel mode before using the memory setting for decisions. It is a memory optimization; it does not promise a throughput improvement.

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
not individual response latency.

## Validation and measurement

```sh
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  cargo test --release --locked --offline --features llama-cuda \
  --test parallel real_model_parallel_contract -- --ignored --nocapture
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  SKID_PARALLEL_WIDTH=4 SKID_PARALLEL_OUTPUT=/tmp/parallel.json \
  cargo test --release --locked --offline --features llama-cuda \
  --test parallel real_model_parallel_measurement -- --ignored --nocapture
```

The contract test covers mixed typed questions, output IDs/order, unequal prompt lengths, a partial wave, shared-prefix reuse, A–Z candidate mapping, untrusted token-like text, cross-question isolation, changed-state request isolation, error recovery after an earlier wave completed, invalid widths, and width-one equivalence to fresh execution.

The measurement uses 1, 4, 16 and 32 questions on short and long warehouse states. Questions repeat three warehouse criteria to measure scaling; they are not 32 distinct ground-truth tasks. Each mode has one untimed warmup (including any context allocation) and three timed repetitions. Mode order alternates between configurations. Detailed JSON retains every score and latency. It reports serial-versus-parallel differences instead of asserting that distinct batch shapes are numerically identical; 0.02 remains the existing probability/mass comparison threshold and any changed top-1 or accepted selection fails the reported equivalence criterion.

These measurements exclude model/context startup from steady-state latency, do not include a remote API/network, and do not establish Jev performance parity or calibrated confidence.

## Numerical limitations

Parallel execution changes batch shapes and can change probabilities, top choices and accepted decisions. Previous labeled-fixture checks observed a fresh abstention becoming a wrong accepted answer; the broader fixture failed the existing 0.02 probability/mass tolerance. The mode remains experimental and opt-in. Compare it against fresh execution on your exact checkpoint and workload, preserving the same policy and equivalence thresholds.
