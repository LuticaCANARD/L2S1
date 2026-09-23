# Model interchangeability in L2S1

L2S1 keeps the application decision contract while replacing a compatible local GGUF model. Identity and preflight explain compatibility; quality and threshold suitability still require a labeled workload.

## Inspect, validate, run

Build with the existing `llama` feature and matching llama.cpp source/libraries. Examples use the compiled binary:

```sh
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf --inspect
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --preflight
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --diagnostics
```

`LlamaBackend::inspect()` reports checkpoint SHA-256 (including the GGUF tokenizer), embedded-template hash, effective prompt profile/version, bridge/runtime build hash, loaded llama/ggml library content hash, adapter/head hashes, device and compute configuration. Checkpoint and loaded-library hashes are computed once at backend registration. Startup therefore reads the complete checkpoint; optimized builds are recommended. Keep registered weight/adapter/runtime files immutable throughout their use. This identity is content provenance, not a signature or protection against concurrent file modification. GPU driver, firmware and operating-system identity are not fingerprinted. The CPU device label identifies the CPU backend, not a unique processor model. A configuration match does not establish identical behavior on every physical machine.

Loaded-library discovery currently uses Linux's dynamic loader. The native backend fails explicitly when verified runtime discovery is unavailable; the pure Rust scoring, calibration and worker modules do not require Linux or llama.cpp. The runtime build hash binds bridge/scoring/prompt/calibration sources, dependency lockfile, Jinja sources, linked core libraries, the compiled bridge archive (including transitive headers and C++ compilation), target/profile/Rust flags and compiler version; loaded-library hashes also cover the dynamically loaded ggml device backends.

`preflight()` uses the same model-bound preparation as inference. It checks request structure, execution support, active artifacts, context length and unique stable single-token continuations at the actual answer boundary. It returns token counts, candidate mappings and prompt-token fingerprints without inference. Capability flags are metadata-based, not a guarantee of model quality or numerical equivalence. Hybrid/recurrent models reject parallel execution; prefix reuse explicitly falls back to fresh. State restore is experimental even when exposed as an available operation. A loaded head still enforces its existing fresh-only/device/compute restrictions.

The existing `DecisionBackend::decide()` and response JSON remain available. `decide_detailed()` adds a separate envelope with model identity, per-decision requested/effective execution, fallback reason, evidence origin, calibration ID, request-local timing, and snapshot accounting. It never returns raw prompt text. Preparation/token hashes in preflight are outside inference timings; a parallel batch's native time is counted once. Ordinary abstention stays a successful result. Detailed failures distinguish invalid requests, unsupported capabilities, incompatible artifacts, context overflow, invalid score evidence and native failure. CLI `--preflight`/`--diagnostics` return request-stage failure JSON on stdout and a nonzero exit status; model loading and malformed JSON can still fail before this envelope.

## Exact evidence and policy

`ExactEvidence::from_logits()` creates validated evidence from a complete vocabulary vector. The shared scorer owns normalization, binary/choice/ordinal interpretation, tie handling and both acceptance gates. It preserves original f64 summation order. Evidence keeps semantic IDs, token IDs, native logits, vocabulary size and the full-vocabulary log normalizer. Candidate-only scores, partial top-k results and generated numeric estimates cannot instantiate this exact-evidence type through deserialization or public field mutation.

The full-vocabulary candidate mass remains independent of calibrated probabilities. Existing output heads retain their base-model mass gate and use the same policy scorer. Evidence is bound to its prepared decision/model by the native backend's ownership and diagnostic envelope; it is not a portable authenticated inference receipt.

## Optional execution optimizations

All optimizations are explicit. Defaults remain legacy prompts, fresh execution, full evidence transfer and disabled preparation caching. Applications can keep their repeated/fixed-schema workloads, and can also change schemas between requests. No learned state encoder or new model checkpoint is required.

### Bounded preparation reuse

```rust,ignore
backend.set_preparation_cache(l2s1::PreparationCacheConfig {
    max_entries: 128,
    max_bytes: 8 * 1024 * 1024,
});
let response = backend.decide(&request)?;
let cache_stats = backend.preparation_cache_stats();
backend.clear_preparation_cache();
```

The backend owns two FIFO caches. Complete preparation is keyed by the exact serialized state and decision. Candidate mapping is keyed by the actual assistant-boundary text and candidate count. Model/tokenizer/template ownership is local to the backend, and relevant configuration changes invalidate cached preparation. Different questions, option counts, instructions and states use ordinary preparation on a miss. Only successful preparation is cached; predictions and native KV state are never stored here. Returned token vectors are still copied.

`max_entries` applies independently to each cache, so their combined entry count can reach twice that limit. One combined `max_bytes` budget allocates three quarters to complete prompts and one quarter to candidate mappings. Accounting includes retained key/token allocations and inline entries; allocator metadata, spare queue capacity, temporary input keys and returned clones are outside this bound. Oversized entries bypass storage without evicting useful entries. A zero entry or byte limit disables storage. `clear_preparation_cache()` removes entries but retains hit/miss/eviction counters; `set_preparation_cache()` replaces both caches and resets those counters. Ordinary request boundaries retain prepared tokens, while clearing native KV state.

CLI equivalents are `--preparation-cache-bytes 8388608 --preparation-cache-entries 128`. The byte default is zero, so caching is disabled until requested. CLI invocations each load a new backend; reuse across calls requires a resident library backend or worker.

### Compact native evidence transfer

```rust,ignore
backend.set_evidence_transfer(l2s1::EvidenceTransfer::Compact)?;
```

`--evidence-transfer compact` selects the same path in the CLI. `Full` remains the default. Compact mode computes the full-vocabulary log normalizer in the C++ bridge and returns only that normalizer, vocabulary size and candidate logits to Rust. It preserves the meaning of `candidate_mass`; it does not replace the denominator with candidate-only normalization. Native logits still cover the entire vocabulary on the host: this removes a host-buffer copy into Rust, not the vocabulary output projection or llama.cpp's device-to-host work. Compare numeric deltas on the actual runtime rather than assuming identical floating-point implementations.

Compact mode supports fresh and prefix-reuse execution, including an explicit shared-state session. It rejects state-restore, parallel execution and learned output heads. The selected transfer mode is part of model/calibration identity and is reported in compact responses; full-mode JSON omits the added field for compatibility. Select the mode before registering calibration. Switching to an incompatible mode fails rather than silently reusing an artifact fitted with another transfer mode.

### Explicit immutable-state sessions

```rust,ignore
backend.set_execution_mode(l2s1::ExecutionMode::PrefixReuse);
backend.set_prompt_layout(l2s1::PromptLayout::StateFirst);
{
    let mut session = backend.shared_state(state)?;
    let first = session.decide(first_questions)?;
    let next = session.decide(different_questions)?;
} // Native KV state is cleared here.
```

`SharedStateSession` exclusively borrows one backend and owns one immutable JSON state. Each `decide(Vec<Decision>)` call can change decision IDs, kinds, instructions, option counts and criteria. IDs must be unique within a call and can recur in later calls. The model, artifacts, policy and prompt configuration cannot change during the borrow. Each question still receives its independently compiled prompt; only complete prefill batches from an exact common token prefix are reused, and questions never attend to earlier answers.

Sessions require an explicitly selected `PrefixReuse` mode and reject recurrent/hybrid memory and output heads. They preserve the chosen prompt layout and evidence transfer mode. State-first layout can expose a longer common prefix when questions change, but changing layout can change predictions. This is the existing causal decoder's prefix reuse across explicit session calls, not a bidirectional state encoder or a small learned reading head. Reuse depends on actual matching tokens; one short question need not become faster.

Creation, a failed call and drop clear native KV state. After a failed call the same session can accept another valid call. Ordinary `DecisionBackend::decide()` remains isolated at request boundaries; sessions do not change its lifetime contract. `reused_prefix_tokens` in each result reports actual reuse.

## Scoped scalar calibration

`ScalarCalibration::fit()` fits a positive temperature by calibration-set NLL, with a bounded search from 0.05 to 20 and a temperature-1 fallback if fitting would worsen NLL. It supports all three decision kinds. Each artifact binds the complete model/configuration fingerprint, exact task signature (including option order and ordinal values), native score semantics, calibration-record hash and source-group hashes. Another model, quantization, adapter, template/profile, compute setting or execution mode rejects the artifact at request time. Changing configuration does not silently drop a loaded calibration.

```sh
cargo run --release --offline --example fit_calibration -- fit-input.json task-temperature.json
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --calibration task-temperature.json --diagnostics
```

`fit-input.json` contains `id`, `model` (the `identity` from inspection), `decision` (one exact request decision), `calibration`, and independent `held_out` arrays. Each record contains `group`, `raw_logits` in task-option order, and zero-based `correct_option`. Supply logits measured with that exact model/configuration. The fitter cannot establish their provenance from numeric records alone. It rejects overlapping source-group IDs and mismatched held-out option counts before writing the artifact and prints held-out NLL/Brier before and after calibration. Group IDs must group paraphrases and option variants by original source; exact-ID checks do not detect mislabeled groups or near duplicates.

`evaluate_policy()` additionally accepts measured base candidate mass and reports coverage, accepted accuracy (null when nothing is accepted), and top-choice accuracy at a given `DecisionPolicy`. Use several thresholds for a held-out acceptance curve. Fitting metrics are explicitly labeled as fit metrics, not test accuracy. No trained or universally suitable temperature is bundled.

Register artifacts with `register_calibration()`/`load_calibration()`; the CLI accepts repeated `--calibration`. At most one artifact per task/ID is accepted. Other task IDs keep native scoring. A matching ID with changed task meaning fails. Raw logits and base mass are retained; calibrated scoring and artifact ID are explicit. A scalar calibration cannot be stacked with an output head. `clear_calibrations()` is an explicit removal operation.

## Experimental whole-sequence restore

```sh
l2s1 --model MODEL.gguf --input examples/warehouse.json \
  --prompt-layout state-first --execution-mode state-restore \
  --snapshot-limit-bytes 268435456 --diagnostics
```

State restore finds an exact prefix common to all decisions in one request, rounded down to complete prefill batches. It computes that prefix, copies the whole sequence state using llama.cpp's sequence-state API, clears and restores it before each suffix, and retains independent answer logits. At least the final token is always evaluated. The first decision pays for prefix prefill and reports zero reused tokens; subsequent decisions report the shared prefix. Single-decision or short-common-prefix requests execute fresh.

Snapshots never leave the native call or survive a request, error, context resize, prompt change or adapter change. Only one snapshot buffer exists. Its size is checked against the configured byte limit before allocation (default 256 MiB; zero forces budget fallback when a common prefix exists). Missing save/restore support or a budget excess triggers explicit fresh fallback; native decode failure remains an error. Diagnostics expose snapshot bytes, save/restore/prefill/suffix wall times and restore count. This bounds the snapshot buffer, not total RSS, GPU allocation, logits buffers or llama.cpp's internal scratch memory. Measure whole-process peak RSS separately.

This is serial state restoration, including for hybrid memory. It does not enable hybrid parallel sequences. No speedup is promised: state copying can cost more than recomputation. Compare exact models/configurations against fresh execution before deployment. State-first prompts also remain opt-in because their changed ordering can change predictions independently of execution mode.

## Bounded ownership and scheduling

`BackendWorker::spawn()` accepts a backend factory, queue capacity, maximum serialized request bytes, shared `MemoryBudget`, and a positive model memory reservation estimate. The factory constructs the backend inside a dedicated thread; `LlamaBackend` remains non-Send/non-Sync. Requests are serialized on that owner thread. `submit()` rejects a full queue immediately and returns a `DecisionTicket` for admitted work. `close()` drains admitted work, joins the thread and releases its owned backend/reservation. Dropping a ticket does not cancel work. Failed construction and panics release reservations; a stopped worker reports disconnection.

```rust,ignore
let budget = l2s1::MemoryBudget::new(2 * 1024 * 1024 * 1024);
let mut worker = l2s1::BackendWorker::spawn(
    8, 1024 * 1024, &budget, 1024 * 1024 * 1024,
    move || l2s1::llama::LlamaBackend::load(
        &model_path, 2048, 256, 4, false, l2s1::DecisionPolicy::default()),
)?;
let response = worker.submit(request)?.wait()?;
worker.close()?;
```

`BackendWorker::spawn_batched()` adds `BatchPolicy` before the factory argument and requires `BatchDecisionBackend`:

```rust,ignore
let policy = l2s1::BatchPolicy {
    max_requests: 4,
    max_input_tokens: 8192,
    max_wait: std::time::Duration::from_millis(2),
};
let mut worker = l2s1::BackendWorker::spawn_batched(
    8, 1024 * 1024, &budget, 1024 * 1024 * 1024, policy,
    move || {
        let mut backend = l2s1::llama::LlamaBackend::load(
            &model_path, 2048, 256, 4, false, l2s1::DecisionPolicy::default())?;
        backend.set_execution_mode(l2s1::ExecutionMode::Parallel);
        backend.set_parallel_width(4)?;
        Ok(backend)
    },
)?;
```

Token admission uses actual model preparation and sums every decision prompt, rather than estimating from characters or request count. Requests exceeding the per-batch token limit fail through their own tickets; an otherwise valid request that does not fit the remaining batch waits for the next one. Results preserve input request order and grouping. `max_wait` budgets collection from the first request's admission time; it is not an inference deadline, a hard preflight-time limit or an end-to-end latency guarantee. Zero wait skips collection of additional requests. Queue and serialized-byte limits still apply.

With `LlamaBackend`, only explicitly configured `Parallel` mode merges admitted requests into native waves. Fresh, prefix-reuse and state-restore modes remain serial and preserve request isolation. Native parallel failures fail the affected batch without replaying inference. Parallel mode retains its model restrictions, extra KV memory and measured score-drift risks. Token admission and collection also cost time; batching alone promises neither lower latency nor greater throughput.

Reservations must include weights, KV/recurrent state, outputs and native scratch space with margin. They enforce admission against an operator-supplied estimate, not an operating-system memory limit. Queue limits bound admitted request count and serialized input size, not the caller's preexisting allocations. Closing can wait for a running native call; no forced cancellation is provided. To change models, drain/close a worker and construct another, or reserve sufficient budget for both explicitly. No HTTP service, remote provider, persistent cross-model cache or live hot-swap policy is introduced.

## Validation and compatibility

See the [verification guide](VERIFICATION.md). The same conformance test accepts colon-separated model paths, checks all result kinds/mappings, artifacts, failures, recovery, snapshot limits and fresh-versus-optimized scores. Prefix reuse/state restore retain the existing 0.02 probability/mass criterion plus unchanged top-choice and accepted results. Parallel execution has [known batch-shape drift](PARALLEL_EXECUTION.md); it reports the unchanged equivalence criterion separately and remains opt-in.

```sh
cargo test --locked --offline
cargo test --release --locked --offline --features llama
L2S1_CONFORMANCE_MODELS=MODEL_A.gguf:MODEL_B.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

Use [`examples/benchmark_optimizations.rs`](examples/benchmark_optimizations.rs) to compare the optional paths with local models:

```sh
cargo run --release --locked --offline --features llama \
  --example benchmark_optimizations -- \
  --model MODEL_A.gguf --model MODEL_B.gguf \
  --output /tmp/l2s1-optimizations.json --repeats 3
```

The output must be new. One backend remains resident per model. The harness uses short/long synthetic warehouse states and 1/4/16 questions cycling choice, binary and ordinal schemas, rotates path order, and runs one untimed warmup immediately before each measured path/scenario/round. It records full responses, identities, cache counters, input/reused tokens, timing and maximum probability/mass/logit differences. Cached/compact paths are compared with legacy fresh; shared-state calls are compared with state-first fresh. Amortized time per question is not individual request latency. Identical warmup inputs exercise repeated preparation; they do not establish cache gains for new questions. This harness does not measure worker batching or labeled task accuracy.

`ExecutionMode::StateRestore` and the added `EvidenceTransfer` configuration require downstream exhaustive matches and manually constructed Rust metadata structs to be updated where applicable. Default full-transfer JSON omits the added transfer field. Legacy prompt/fresh execution defaults and base score semantics are preserved. Test fixtures and contract equivalence do not establish production or labeled workload quality.
