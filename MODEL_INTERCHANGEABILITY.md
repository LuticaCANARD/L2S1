# Model interchangeability in L2S1

L2S1 keeps the application decision contract while replacing a compatible local GGUF model. Identity and preflight explain compatibility; quality and threshold suitability still require a labeled workload. All seven recommended areas from [the source review](MODEL_INTERCHANGEABILITY_REVIEW.md) now have implementations. No upstream implementation was copied.

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

Snapshots never leave the native call or survive a request, error, context resize, prompt change or adapter change. Only one snapshot buffer exists. Its size is checked against the configured byte limit before allocation (default 256 MiB; zero forces budget fallback when a common prefix exists). Missing save/restore support or a budget excess triggers explicit fresh fallback; native decode failure remains an error. Diagnostics expose snapshot bytes, save/restore/prefill/suffix wall times and restore count. This bounds the snapshot buffer, not total RSS, GPU allocation, logits buffers or llama.cpp's internal scratch memory. Whole-process peak RSS in the validation report is a separate measure.

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

Reservations must include weights, KV/recurrent state, outputs and native scratch space with margin. They enforce admission against an operator-supplied estimate, not an operating-system memory limit. Queue limits bound admitted request count and serialized input size, not the caller's preexisting allocations. Closing can wait for a running native call; no forced cancellation is provided. To change models, drain/close a worker and construct another, or reserve sufficient budget for both explicitly. No HTTP service, remote provider, persistent cross-model cache or live hot-swap policy is introduced.

## Validation and compatibility

See [implementation validation](MODEL_INTERCHANGEABILITY_RESULTS.md). The same conformance test accepts colon-separated model paths, checks all result kinds/mappings, artifacts, failures, recovery, snapshot limits and fresh-versus-optimized scores. Prefix reuse/state restore retain the existing 0.02 probability/mass criterion plus unchanged top-choice and accepted results. Parallel execution has [known batch-shape drift](PARALLEL_EXECUTION.md); it reports the unchanged equivalence criterion separately and remains opt-in.

```sh
cargo test --locked --offline
cargo test --release --locked --offline --features llama
L2S1_CONFORMANCE_MODELS=MODEL_A.gguf:MODEL_B.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

`ExecutionMode::StateRestore` is a new Rust enum variant, so downstream exhaustive matches must handle it. The previous error enum and response structs are unchanged. Legacy prompt/fresh execution defaults and base score semantics are preserved. Test fixtures and contract equivalence do not establish production or labeled workload quality.
