<a id="parallel-question-execution"></a>
# Parallel question execution

[English](PARALLEL_EXECUTION.md) · [한국어](../ko/PARALLEL_EXECUTION.md) · [日本語](../ja/PARALLEL_EXECUTION.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)


The supported `parallel` mode evaluates independent questions using separate llama.cpp sequence IDs. It shares the loaded model, computes an exact common prompt prefix once per wave, and puts suffix tokens from multiple questions into each decode batch. It does not start multiple threads that concurrently mutate one llama context.

```sh
cargo run --release --locked --features llama-cuda -- \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --input examples/warehouse.json --device cuda \
  --prompt-layout state-first --execution-mode parallel --parallel-width 4
```

The default execution mode remains `fresh`. Since 0.2.0, text `parallel` execution uses the `state-first` layout unless `--prompt-layout` (`set_prompt_layout`) selects one explicitly; every other mode, and a backend with a vision projector, keeps `legacy`. Prompt layout and execution mode are still separate choices: moving state changes the prompt; parallelism changes the execution schedule. Compare all execution modes at the same layout.

<a id="implementation"></a>
## Implementation

- Rust prepares and validates each question's tokens and A–Z single-token continuations. User state and question data retain special-token parsing protection.
- At most `parallel_width` questions enter each wave (default 4, allowed 1–32). The last wave can be smaller. Results retain request order and IDs, including when prompts finish in different decode batches.
- Shared prefixes form a tree over exact token IDs, retaining at least one suffix token per question: the whole wave shares its common prefix, and questions that agree for longer keep sharing after another question diverges. By default each shared segment is rounded down to complete original prefill batches. One sequence evaluates each segment; `llama_memory_seq_cp` shares its KV entries with the other sequences in that group.
- Suffix tokens have independent sequence IDs and original absolute positions. Their attention sees their own sequence and shared prefix, not another question's suffix. Final full-vocabulary logits are copied using each question's final token index in the relevant decode batch.
- Memory is cleared at API call boundaries and after failures. Between the waves of one call, only the root shared prefix of the previous wave's first sequence is kept, and only when the next wave's first prompt starts with the same exact tokens. Retention across calls requires an explicit `ParallelPrefixSession`. Recurrent/hybrid models are explicitly unsupported in this mode.
- The model stays loaded once. By default, the context is recreated lazily when sequence capacity changes, with nominal token capacity `context * width`. `context` remains the per-question input limit. Higher widths increase KV/attention memory and may fail allocation; there is no silent CPU fallback or input truncation.
- `backend.parallel_width` records the configured wave limit (serial modes report 1). `reused_prefix_tokens` counts tokens a question did not evaluate itself in that call: zero for the question that evaluated a shared prefix, and the shared token count for the questions that copied it. A retained prefix counts as reused for every question, so the first question of a later wave or session call can be nonzero. Summed `input_tokens - reused_prefix_tokens` accounts for actual evaluated tokens; do not read a single zero as "sharing failed".

This uses the pinned llama.cpp [batch, sequence, memory-copy and per-token logits API](https://github.com/ggml-org/llama.cpp/blob/3d82ef62d47fd74e18f36c5eccbdcf965b617b17/include/llama.h). It is independent sequence batching with serial waves, not a replacement model architecture or calibrated-decision training.

<a id="prefix-sharing-controls-020"></a>
### Prefix sharing controls (0.2.0)

These settings change only which exact token prefixes are shared, never the prompts' contents (except the documented default layout above).

- **Layout.** `legacy` serializes `instruction` and `options` before `state`, and typed detail puts `decision_kind` first, so a wave mixing questions diverges within a few tokens and usually shares nothing. `state-first` puts the state first. If an explicitly selected `legacy` layout shares no token in a multi-question parallel call, the backend prints one warning to stderr. Changing the layout changes predictions: re-evaluate accuracy and bind calibration to the new `prompt_version`.
- **Fixed JSON order.** Both layouts serialize the payload fields and every nested object of `state` and `shared` with sorted keys, independent of serde_json's `preserve_order` feature. Cargo unifies features across the dependency graph, so previously any dependency enabling `preserve_order` could silently change `legacy` prompts (and therefore decisions and calibration). The bytes equal the earlier default build's output.
- **Shared input.** `DecisionRequest.shared` (JSON `"shared"`) holds evidence common to many requests or questions, such as knowledge or examples. It is always the first field of the data segment in both layouts, so callers no longer need key-name ordering tricks inside `state`. Absent or `null` leaves the prompt unchanged. The ONNX backend rejects it.
- **Wave order.** `--parallel-wave-order prefix` (default, `ParallelWaveOrder::Prefix`) sorts prepared prompts by token sequence before forming waves so questions with long common prefixes share a wave. `request` restores consecutive request-order waves. Results always keep request and decision order; wave membership can change scores slightly because decode batches differ.
- **Alignment.** `--parallel-prefix-alignment batch` (default) rounds every shared segment down to complete prefill batches, so shared KV is computed with the same decode boundaries as serial prefill; each question re-evaluates up to `batch - 1` common tokens per segment. `token` shares every common token. It is faster for long common prefixes but can change scores slightly; validate decisions first. Token-aligned nested segments shorter than 16 reused tokens are not split into a separate decode.
- **Retention across calls.** `backend.parallel_prefix_session()` returns a `ParallelPrefixSession` whose `decide` / `decide_batch` keep the latest root shared prefix in sequence 0 between calls. A later call whose first prompt (after wave ordering) starts with the same tokens does not evaluate it again. The session requires `parallel` mode, keeps the configured width as sequence capacity, and clears native KV on creation, failure and drop. Responses report `backend.parallel_prefix_retained`. The same call sequence produces the same scores.

Local measurement (Qwen3-0.6B Q8_0, CPU, 4 threads, batch 512, width 8; three calls of 8 requests sharing a ~1,300-token `shared` knowledge field, explicit `legacy` layout; one run that partly shared the CPU with another test job, so times are indicative while reuse counts are exact; not a general speedup claim): fresh 243.9 s; parallel `batch` 64.6 s (21,504 of 33,110 input tokens reused); parallel `token` 16.1 s (28,728); session `batch` 51.7 s (24,576); session `token` 1.6 s (32,832). With a 512-token batch, batch alignment re-evaluated up to 511 common tokens per question, which dominated the remaining work here.

`backend.parallel_prefix_alignment` and `backend.parallel_wave_order` report non-default settings; absent fields mean `batch` and request order (the behavior before 0.2.0).

<a id="dynamic-parallel-kv-context"></a>
### Dynamic parallel KV context

`--parallel-context-dynamic` is an opt-in memory setting for `parallel` mode. After tokenizing each wave, it reserves at most `sum(input_tokens) + batch` KV slots, capped at the legacy `context * width` reservation. llama.cpp may pad the requested size. The sum counts shared prefixes more than once, so this is conservative. `--context` still validates each individual question; no input is truncated.

The native context grows when a later wave needs more slots. It retains that high-water capacity for later waves with the same effective question count to avoid repeated context rebuilds. Switching between default and dynamic mode recreates the context, and both modes clear request-local KV after inference. `backend.parallel_context_dynamic` identifies the opt-in result; `backend.parallel_context_tokens` reports the currently allocated padded context in dynamic mode. The model weights remain loaded.

This changes the llama.cpp context size and may change logits or decisions. Compare the exact model, prompts, policy and input set against the default parallel mode before using the memory setting for decisions. It is a memory optimization; it does not promise a throughput improvement.

<a id="independent-request-batches"></a>
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

<a id="independent-image-batches"></a>
### Independent image batches

With a matching vision projector loaded, `backend.decide_vision_batch(&requests,
&images)` scores each request from its own state and one original image. In
`parallel` mode, decisions enter bounded native waves; `fresh` mode evaluates
them serially. HTTP exposes the same path when each decision references one
image via `media_ids` and the listener uses `--execution-mode parallel
--parallel-width 4`. Four request media items with four independent decisions
then share native execution instead of merely sharing an HTTP response.

Image decode and multimodal tokenization retain the trusted prompt control-token
boundary. Compatible projector chunks enter llama.cpp mtmd's batch encoder;
projector shape and token limits may split these into smaller encoder batches.
All wave questions use independent decoder sequence IDs and per-sequence image
positions, with their own full-vocabulary final logits or compact candidate
scores and full-vocabulary normalizers. Vision currently does
not share prompt-prefix KV, so `reused_prefix_tokens` remains zero. Noncausal
image chunks must fit the configured token batch and microbatch in full; an
unsupported layout or recurrent model returns an error rather than switching to
serial execution. More than 26 options still require `fresh` vision execution.

Image waves use separate KV streams. Dynamic context sizing reserves the
longest multimodal input plus one token batch of headroom in each stream,
bounded by the per-question limit. Both the token count and position span
are checked against that limit.
Each question remains bounded by `--context`; memory and image embeddings are
cleared after the batch or an error. Final partial waves preserve request and
decision order. Batch failures return no partial responses.

`backend.vision_batch_metrics()` exposes counters for the latest native wave.
Parallel image HTTP responses also include these under
`backend.details.vision_batch`, labeled `scope: "last_native_wave"`:
`projector_encode_calls`, `projector_batch_max`, `decoder_calls`,
`decoder_batch_max_sequences`, `projector_reused_chunks`, `kv_clear_calls`, and
`kv_clear_skipped`. The two clear counters have
`kv_clear_scope: "since_last_native_vision_start"`: subsequent cleanup or configuration calls
can increment them before the next native vision call resets them.
Preparation-cache counters alongside them have a separate
`preparation_cache_scope` covering the backend lifetime since configuration.
These counters establish the actual encoder and
decoder batch shapes; the number of images in an HTTP request alone does not.
Measure total batch completion latency and amortized milliseconds per image
separately, and compare scores, top choices, abstentions, and task accuracy
against `fresh` on the same images.

The [120-image TrashNet measurement](benchmarks/trashnet-vision-20260925/REPORT.md#native-four-image-batching-2026-09-26)
verified four native decoder sequences on RTX 3080, with lower amortized
processing times, but all three CUDA checkpoints failed the existing numerical
equivalence criterion. Selected values or raw rankings changed. Image batching
is supported within its model and layout limits; native batch counters and isolation tests do not establish
score equivalence or task accuracy.

<a id="optimization-components"></a>
### Optimization components

The optimized four-image profile includes exact prompt preparation caching,
compact evidence transfer and dirty-only physical KV clearing. These components
are tested separately using the existing setters at an unchanged compute
configuration. Their score equality does not establish fresh-versus-parallel
parity. The serial cache/compact comparison did not establish a speedup;
these components are included in `--vision-optimized`.

Fresh image HTTP responses expose `backend.details.vision_preparation` with a
backend-lifetime cache snapshot and `vision_kv_clear` for the last native call.
All serial media groups receive the same final snapshot so the common response
metadata remains consistent. These fields do not assert image or KV reuse.

`--vision-projector-reuse` is a supported setting for explicit projector reuse in parallel
mode: it independently encodes each unique chunk and reuses identical chunks
within one wave. It does not preserve the serial decoder's numerical execution.

<a id="combined-vision-optimizations"></a>
### Combined vision optimizations

`--vision-optimized` requires a matching projector and an explicitly selected
CUDA or Metal device. It selects parallel width 4, dynamic per-stream context,
token batch/microbatch 1024, Flash Attention on, compact evidence, an 8 MiB
preparation cache and identical-image projector reuse. Per-question context
defaults to 4096. Rust callers construct the backend with
`ComputeOptions::vision_optimized()`, load the projector, then call
`enable_vision_optimizations()`.

Native memory tracks writes before entering every decode or snapshot restore,
including operations that fail after partial writes. `sd_clear` always
invalidates logical cached tokens, but only zeros physical KV when dirty.
Successful vision calls own their native cleanup; a Rust guard clears on
validation/preparation/scoring errors and unwinding. This skips duplicate
clears while keeping request and session isolation.

The preparation cache retains exact rendered state/decision prompts and answer
boundary mappings; it stores no image bytes or KV. Images with identical bytes
and matching ordered chunk/position metadata can share immutable projector
embeddings inside one wave. In this reuse mode, each unique chunk uses a
single-chunk encoder batch so changing another image cannot change its encoder
batch shape or slot. This selects an alternative to projector batching; decoder
batching is retained. Decoder KV streams and image positions remain
independent. `backend.vision_projector_reuse` reports the enabled setting and
`projector_reused_chunks` reports actual reused chunks, which can exceed image
count for projectors that create a global image and tiles.

Compact transfer copies only each decision's candidates plus its complete
vocabulary normalizer into Rust. llama.cpp still computes the full vocabulary
and transfers its output to the host. The optimized profile does not combine
text-only KV prefix sessions or snapshot restoration with independent vision
streams. These are alternative execution modes; hybrid/recurrent models and
wide answer codes are explicitly rejected by this profile.

Both parallel attention and projector batch shapes can change numerical
results. Keep fresh as the default and compare the combined profile against
fresh on the original inputs. Full-versus-compact equivalence at an identical
compute configuration is checked separately from fresh-versus-optimized
equivalence. Metal execution and performance still require real hardware tests.

<a id="validation-and-measurement"></a>
## Validation and measurement

Vision isolation and numerical equivalence are separate tests. Set
`SKID_VISION_MODEL`, `SKID_VISION_MMPROJ`, and optionally `SKID_CUDA=1`, then run:

```sh
cargo test --release --locked --features llama-cuda --test vision \
  real_vision_batch_preserves_image_state_order_and_recovers_after_errors \
  -- --ignored --test-threads=1
cargo test --release --locked --features llama-cuda --test vision \
  real_vision_batch_equivalence_on_color_fixture -- --ignored --test-threads=1
python3 benchmarks/trashnet-vision-20260925/evaluate_batch.py --help
```

The equivalence test can fail on CUDA even when the isolation test passes.
The evaluator retains such failures in its comparison instead of increasing
the tolerance or replacing the batched result with serial inference.

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

<a id="numerical-limitations"></a>
## Numerical limitations

Parallel execution changes batch shapes and can change probabilities, top choices and accepted decisions. Previous labeled-fixture checks observed a fresh abstention becoming a wrong accepted answer; the broader fixture failed the existing 0.02 probability/mass tolerance. The mode is supported and selected explicitly with `--execution-mode parallel`. Compare it against fresh execution on your exact checkpoint and workload, preserving the same policy and equivalence thresholds.
