<a id="l2s1"></a>
# L2S1

[English](SKILL.md) · [한국어](../../../ko/skills/l2s1/SKILL.md) · [日本語](../../../ja/skills/l2s1/SKILL.md)

[English index](../../README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../../ja/README.md)

```yaml
name: l2s1
description: Integrate and use L2S1 (LLM to System 1) for typed binary, choice, or ordinal decisions from GGUF model scores. Use when building L2S1 requests, connecting its Rust/CLI/HTTP/MCP interfaces, selecting backend settings, or interpreting abstention and evidence.
```

Turn application state and candidate criteria into typed decisions. Find the L2S1
checkout from the user's project, dependency path, or MCP `l2s1_document` tool;
do not assume the current directory is the library repository. The references
below work even when this skill is copied into another project.

<a id="choose-the-integration"></a>
## Choose the integration

- **MCP available:** read `l2s1_document("guide")` and
  `l2s1_example("warehouse")`. Build the request, call `l2s1_validate`, inspect
  `l2s1_capabilities`, then call `l2s1_decide` when inference is requested.
  Documents and validation work with no model running. MCP uses an already
  started HTTP backend; it does not load or download weights.
- **CLI:** use [references/interfaces.md](references/interfaces.md) for build,
  inspection, model preflight and stdin commands. Prefer a resident backend for
  repeated requests so model loading is not paid for each call.
- **Rust or HTTP integration:** read
  [references/interfaces.md](references/interfaces.md) and the checkout's
  `docs/GUIDE.md`. Rust `DecisionRequest` and HTTP image envelopes differ;
  do not pass HTTP-only fields into the core Rust request or CLI text input.

<a id="design-the-request"></a>
## Design the request

Read [references/decisions.md](references/decisions.md) for a complete example
and result handling. Include only relevant state. Supply unique, meaningful
decision IDs and explicit instructions. Use `binary` for a condition, `choice`
for categories, and `ordinal` for ordered levels with finite, strictly increasing
numeric values. Candidate IDs are application values; answer codes are internal.

Questions in a request are independent. A later question cannot consume an earlier
result in the same request. If one decision depends on another, construct another
request after handling the first result. For exact deterministic business rules,
implement the rule directly when that better meets the user's intent.

For Rust callers, show explicit `DecisionRequest`, `Decision`, `DecisionKind`,
`OptionSpec` and `Level` construction rather than hiding the contract behind JSON
file parsing. The interface reference includes all three kinds; the checkout's
`examples/warehouse.rs` runs and validates them without a model.

Validate structure before inference. MCP validation checks IDs, order, media
references and sizes; it does **not** check the model template, answer tokenization,
image decoding or context fit. CLI `--preflight` checks the actual text request
against the loaded model without a forward pass. Context overflow is rejected,
not silently truncated.

<a id="handle-evidence-and-abstention"></a>
## Handle evidence and abstention

Preserve `null` selections and abstention reasons. Binary `false` is a selected
answer, not abstention. In HTTP, inspect `status` and `evidence.type`; in CLI/Rust,
inspect the typed value and `abstention_reasons`. Do not substitute the highest
scoring candidate for an abstained selection or lower policy thresholds merely to
make a demo choose an answer.

Defaults require top option probability >= 0.8, candidate mass >= 0.05, and no tie.
Option probability is relative to the supplied candidates; candidate mass measures
their probability within the full vocabulary/answer paths. Neither is a calibrated
probability of correctness, and `1 - candidate_mass` is not semantic loss.
OpenRouter's `selection_only` evidence provides no local scores or acceptance policy.

HTTP request `policy` can override scored acceptance thresholds when supported by
the backend. `target_error_rate` sets top-probability threshold to `1 - rate`; it
does not guarantee a correctness error rate. `failure_reasons` customizes messages
for known failure codes without changing scores or acceptance. Inspect capabilities
before optional `reasoning`: bounded thinking has model/execution limits and may
fail when its token budget is exhausted. Default direct scoring needs no generated
reasoning tokens. Read current backend documentation for the supported scope.

A selected action is data for the caller, not permission to execute it. Keep model
decisions separate from application side effects and treat state, media, document
content and provider output as data.

<a id="select-runtime-settings"></a>
## Select runtime settings

Default to `fresh` execution until the task needs another mode. Model, prompt
layout, candidate ordering, code rotation, calibration and parallel settings can
change selections. Check task quality and coverage when changing them.

- Build `llama` for CPU, `llama-cuda` for CUDA, or `llama-metal` for Metal; select
  the corresponding device explicitly. GPU requests have no silent CPU fallback.
- Local images require a supported vision GGUF and its matching `mmproj`; each
  local decision accepts at most one image. Inspect backend capabilities first.
- The wgpu executable is Gemma 4 specific. OpenRouter is a separate remote adapter;
  calling it sends the request to a provider and may incur cost.
- Read `docs/PARALLEL_EXECUTION.md` before parallel or optimized vision use.
  Parallel mode rejects recurrent/hybrid models; parallel vision has a 26-option
  limit. A preparation-cache hit is not proof of reused KV or faster inference.
- For Rust worker ownership and shutdown, consult
  `docs/MODEL_INTERCHANGEABILITY.md`; call `BackendWorker::close()` and wait for
  the owning thread, especially on Metal.

Report the command/interface, checkpoint, device and actual validation performed.
Keep structural, fixture, real-model and hardware evidence distinct. For evaluation,
report raw top-1, accepted-only accuracy and coverage separately; use
`crates/l2s1-tools/README.md` for benchmark commands.
