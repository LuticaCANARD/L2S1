# Model interchangeability in L2S1

L2S1 keeps the application decision contract while replacing a compatible local GGUF model. Identity and preflight explain compatibility; quality and threshold suitability still require a labeled workload.

## Inspect, validate, run

Build with the `llama` feature for CPU or `llama-cuda` for CUDA. The sys dependency builds matching llama.cpp source and libraries. Examples use the compiled binary:

```sh
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf --inspect
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --preflight
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --diagnostics
```

`LlamaBackend::inspect()` reports checkpoint SHA-256 (including the GGUF tokenizer), embedded-template hash, effective prompt profile/version, bridge/runtime build hash, loaded llama/ggml library content hash, adapter/head hashes, device and compute configuration. Checkpoint and loaded-library hashes are computed once at backend registration. Startup therefore reads the complete checkpoint; optimized builds are recommended. Keep registered weight/adapter/runtime files immutable throughout their use. This identity is content provenance, not a signature or protection against concurrent file modification. GPU driver, firmware and operating-system identity are not fingerprinted. The CPU device label identifies the CPU backend, not a unique processor model. A configuration match does not establish identical behavior on every physical machine.

Loaded-library discovery currently uses Linux's dynamic loader. The native backend fails explicitly when verified runtime discovery is unavailable; the pure Rust scoring, calibration and worker modules do not require Linux or llama.cpp. The runtime build hash binds bridge/scoring/prompt/calibration sources, dependency lockfile, Jinja sources, linked core libraries, the compiled bridge archive (including transitive headers and C++ compilation), target/profile/Rust flags and compiler version; loaded-library hashes also cover the dynamically loaded ggml device backends.

`preflight()` uses the same model-bound preparation as inference. It checks request structure, execution support, active artifacts, context length and stable candidate continuations at the actual answer boundary. Up to 26 candidates use the existing single-token mapping. Larger sets use fixed-width uppercase codes and return `candidate_token_sequences` instead of scalar `candidate_token_ids`; the token sequences must be unique and prefix-free. These decisions support fresh or prefix-reuse execution with full evidence without output heads, scalar calibration or feature export. `encode_decision_sequences()` exposes the complete paths; the old scalar export API explicitly rejects wide codes. See [the sequence scoring contract](INTENT_BENCHMARK.md). Preflight returns token counts, candidate mappings and prompt-token fingerprints without inference. Capability flags are metadata-based, not a guarantee of model quality or numerical equivalence. Hybrid/recurrent models reject parallel execution; prefix reuse explicitly falls back to fresh. State restoration is available as an explicit serial mode with full evidence transfer. A loaded head still enforces its existing fresh-only/device/compute restrictions.

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

The backend owns three FIFO caches: text preparation, vision prompt preparation and candidate mappings. Complete preparation is keyed by the exact serialized state and decision. Candidate mapping is keyed by the actual assistant-boundary text and candidate count. Model/tokenizer/template ownership is local to the backend, and relevant configuration changes invalidate cached preparation. Different questions, option counts, instructions and states use ordinary preparation on a miss. Only successful preparation is cached; predictions and native KV state are never stored here. Returned token vectors are still copied. Single-letter mappings and multi-letter token paths share these limits; wide mappings count both the outer vector allocation and every token-path allocation. Cache hits do not bypass execution-mode or artifact checks.

`max_entries` applies independently to each cache, so their combined entry count can reach three times that limit. One combined `max_bytes` budget allocates half to text prompts, one quarter to vision prompts and one quarter to candidate mappings. Accounting includes retained key/token allocations and inline entries; allocator metadata, spare queue capacity, temporary input keys and returned clones are outside this bound. Oversized entries bypass storage without evicting useful entries. A zero entry or byte limit disables storage. `clear_preparation_cache()` removes entries but retains hit/miss/eviction counters; `set_preparation_cache()` replaces all three caches and resets those counters. Ordinary request boundaries retain prepared tokens, while clearing native KV state.

CLI equivalents are `--preparation-cache-bytes 8388608 --preparation-cache-entries 128`. The byte default is zero, so caching is disabled until requested. CLI invocations each load a new backend; reuse across calls requires a resident library backend or worker.

### Compact native evidence transfer

```rust,ignore
backend.set_evidence_transfer(l2s1::EvidenceTransfer::Compact)?;
```

`--evidence-transfer compact` selects the same path in the CLI. `Full` remains the default. Compact mode computes the full-vocabulary log normalizer in the C++ bridge and returns only that normalizer, vocabulary size and candidate logits to Rust. It preserves the meaning of `candidate_mass`; it does not replace the denominator with candidate-only normalization. Native logits still cover the entire vocabulary on the host: this removes a host-buffer copy into Rust, not the vocabulary output projection or llama.cpp's device-to-host work. Compare numeric deltas on the actual runtime rather than assuming identical floating-point implementations.

Text compact mode supports fresh and prefix-reuse execution, including an explicit shared-state session. It rejects text state-restore/parallel execution and learned output heads. Vision compact mode supports fresh and parallel execution for single-token answer codes (at most 26 options); wide image codes require fresh/full evidence. Every image sequence retains its own full-vocabulary normalizer. The selected transfer mode is part of model/calibration identity and is reported in compact responses; full-mode JSON omits the added field for compatibility. Select the mode before registering calibration. Switching to an incompatible mode fails rather than silently reusing an artifact fitted with another transfer mode.

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

Creation, a failed call and drop clear native KV state. After a failed call the same session can accept another valid call. Ordinary `DecisionBackend::decide()` remains isolated at request boundaries; sessions do not change its lifetime contract. `reused_prefix_tokens` in each result reports actual reuse. For multi-letter codes it reports reuse on the root prompt evaluation; `code_evaluated_tokens` counts decoded tokens across every code-prefix branch. Wide codes require full evidence. `session.preparation_cache_stats()` and `session.take_timings()` expose counters while the session is borrowed.

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

## Request-local state restoration

```sh
l2s1 --model MODEL.gguf --input examples/warehouse.json \
  --prompt-layout state-first --execution-mode state-restore \
  --snapshot-limit-bytes 268435456 --diagnostics
```

State restore finds an exact prefix common to all decisions in one request, rounded down to complete prefill batches. It computes that prefix and saves the whole sequence state using llama.cpp's sequence-state API. The first decision continues from the original prefill; each later decision clears and restores the saved state before evaluating its suffix. Each answer has independent logits, and at least the final token is always evaluated. The first decision pays for prefix prefill and reports zero reused tokens; subsequent decisions report the shared prefix. The `restores` diagnostic counts actual state loads, so a successful N-decision request reports N-1 restores. Single-decision or short-common-prefix requests execute fresh.

Snapshots never leave the native call or survive a request, error, context resize, prompt change or adapter change. Only one snapshot buffer exists. Its size is checked against the configured byte limit before allocation (default 256 MiB; zero forces budget fallback when a common prefix exists). Missing save/restore support or a budget excess triggers explicit fresh fallback; native decode failure remains an error. Diagnostics expose snapshot bytes, save/restore/prefill/suffix wall times and restore count. This bounds the snapshot buffer, not total RSS, GPU allocation, logits buffers or llama.cpp's internal scratch memory. Measure whole-process peak RSS separately.

This is serial state restoration, including for hybrid memory. It requires full evidence transfer and does not enable hybrid parallel sequences or persistent shared-state sessions. It is an explicit execution mode; `prefix-reuse` continues to use KV rollback and falls back to fresh on recurrent/hybrid models. No speedup is promised: state copying can cost more than recomputation. Compare exact models/configurations against fresh execution before deployment. State-first prompts also remain opt-in because their changed ordering can change predictions independently of execution mode. The [Bonsai RTX 3060 validation](benchmarks/bonsai-state-restore-20260925/REPORT.md) exercises the hybrid path with real state saves and restores.

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

## CPU/GPU placement

CUDA loading accepts `--gpu-layers N` and `--cpu-moe-layers N`. The first limits
GPU layer placement; the second keeps the expert weights of the first N MoE
layers in CPU RAM while retaining normal placement for attention and shared
weights. These settings can be combined. This controls weight residency; llama.cpp may still offload operations using host weights to CUDA during prefill. Omit both to preserve the original
CUDA loading behavior. CPU-only loading rejects CUDA placement requests.

```sh
l2s1 --model models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --device cuda \
  --cpu-moe-layers 18 --context 8192 --batch 256 --threads 8 \
  --preparation-cache-bytes 67108864 --input request.json
```

The Rust fields are `ComputeOptions.gpu_layers: Option<u32>` and
`cpu_moe_layers: u32`; manually constructed structs must supply `None` and `0`
for legacy defaults. Old JSON remains readable, and default serialization omits
both fields. Non-default values appear in response compute metadata and model
identity, so calibration/head/worker identities distinguish placements.

CPU expert splitting requires an MoE checkpoint and N no greater than its layer
count. Patterns use llama.cpp's expert tensor naming and remain owned by the
loaded engine. This is explicit weight placement, not a CPU fallback after CUDA
allocation failure. The native log reports actual CPU/GPU buffers; requested
layer counts alone do not prove a VRAM budget. CPU RAM, KV and compute buffers
must also fit. On the RTX 3060 12 GiB, the 26B Q4 checkpoint with context 8192 and batch 256 failed context allocation with 14 CPU expert layers; 18 layers completed all 400 intent cases (200 BANKING77 English and 200 MASSIVE Korean), each with a first call and one cached repeat. All 400 cached calls preserved the complete decision evidence exactly. This validates the tested placement and context, not arbitrary context lengths or concurrent model instances.

Quantized CPU and GPU kernels can produce different scores. A Qwen3 0.6B Q8
probe with CPU-resident layers exceeded the existing 0.02 probability-difference
criterion against full CUDA (maximum 0.028); this is not an equivalence claim.
Evaluate accuracy on the intended placement. Preparation and session cache
checks compare against fresh inference with that same placement.

## Model loading and peak host RSS

`--model-load-mode read` selects llama.cpp's ordinary read/upload path instead of
its automatic memory mapping. It changes how weights reach their existing CPU
and CUDA buffers; it does not change the checkpoint, quantization, layer split,
context, or inference algorithm. `auto` remains the default. This option is
available in the CLI, JSONL evaluator, intent-cache benchmark and JevBench harness.

For the 26B CPU/GPU split, add it to the placement command:

```sh
l2s1 --model models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --device cuda \
  --cpu-moe-layers 18 --model-load-mode read \
  --context 8192 --batch 256 --threads 8 --input request.json
```

The Rust field is `ComputeOptions.model_load_mode: ModelLoadMode`. Existing Rust
struct literals must add `ModelLoadMode::Auto`; old JSON remains readable and
default serialization omits the field. Explicit `read` is recorded in compute
metadata and model/artifact identity. The original native entry points preserve
automatic loading for external reference callers.

Reduced process RSS is not the same as reduced total physical memory demand:
read loading uses allocated backend buffers, potentially CUDA-pinned host memory,
while mapped weights are clean file-backed pages. The OS file cache remains
outside process RSS. Measure loading high-water RSS, steady inference RSS and
memory pressure separately on the intended host. Do not assume read loading is
better for every checkpoint or a CPU-only deployment.

On one RTX 3060 host, three alternating warm-cache loads per mode of the same
Gemma 4 26B A4B Q4 checkpoint, with 18 CPU expert layers, measured median peak
engine RSS of 16.174 GiB in `auto` and 9.349 GiB in `read` (42.2% lower).
Median short post-load RSS was 10.332 versus 9.334 GiB; model loading took
10.361 versus 13.931 seconds. All 231 public JevBench evidence objects from
the `read` run exactly matched the prior `auto` run. Sampled system
`MemAvailable` at the engine RSS peak was 28.903 GiB in `auto` and 19.941 GiB
in `read`, so the lower process RSS is not evidence of lower total physical
memory demand. This was a local Linux/CUDA measurement, not a Metal result or
a general load-time guarantee. The full local report and source hashes remain
gitignored at `results/gemma26-lowrss-20260923T135227Z/REPORT.md`.

## Vision preparation and throughput profiles

Vision callers can select `--vision-preserving` (or `enable_vision_preserving_optimizations()`) to retain fresh batch256/FlashOff execution while caching exact preparation and copying compact evidence. `--vision-optimized` selects the separate experimental parallel throughput profile. Both require a matching projector and at most 26 answer options; model, compute settings and hardware must match the reference used for numerical validation. Neither profile caches prior inference results.
