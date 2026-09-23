# L2S1 model interchangeability design review

Reviewed on 2026-09-23. The objective is to replace a compatible model while preserving the application's decision request, result meaning, and acceptance policy. Loading another file is only part of that contract: tokenizer behavior, available inference operations, score provenance, and calibration also need explicit boundaries.

This is a source review and proposed design, not an implementation or comparative benchmark. Local observations refer to the working tree on top of `72fdef3eabaae579d640cb7b51d18a29dc83791e`, including its existing uncommitted output-head work. Upstream files were read at the revisions below. No upstream model, server, or test suite was run for this review.

Implementation follow-up: all seven recommended design areas are now implemented; see [the API and operational contract](MODEL_INTERCHANGEABILITY.md) and [validation results](MODEL_INTERCHANGEABILITY_RESULTS.md). The review below preserves the pre-implementation assessment and its rejected/deferred alternatives.

## Sources

| Project | Reviewed revision | Main source files | Source license |
| --- | --- | --- | --- |
| [Open Alternative to Jev](https://github.com/ikermoel/open-alternative-jev/tree/4a85df1831b537343c9e133b5150fa3a3b1ce98e) | `4a85df1831b537343c9e133b5150fa3a3b1ce98e` | `so1/decider.py`, `so1/backends/base.py`, `so1/backends/vllm.py`, `so1/prompting.py`, `so1/calibration.py`, `tests/test_vllm_backend.py` | Apache-2.0 |
| [SemIf](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37) | `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` | `src/semif_phase1/direct.py`, `src/semif_phase1/llamacpp_backend.py`, `docs/CALIBRATION.md`, `tests/test_llamacpp.py` | MIT |
| [decidr](https://github.com/devanmolsharma/decidr/tree/58df99474cebbc96601cb8d5e64cfcb63fffbe75) | `58df99474cebbc96601cb8d5e64cfcb63fffbe75` | `src/decidr/backend.py`, `src/decidr/core.py`, `src/decidr/token_cache.py` | MIT |
| [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python/tree/e1d4cc938204b22fc5a3c3aca7044072fe3f712d) | `e1d4cc938204b22fc5a3c3aca7044072fe3f712d` | `src/system_one_adapter/providers/base.py`, `src/system_one_adapter/_client.py`, `src/system_one_adapter/_utils/error_handling.py`, `tests/test_provider_lifecycle.py` | MIT |

The proposals below adapt architectural ideas; this review imports no upstream code. Source license labels describe the reviewed repositories, not their model weights.

## Existing L2S1 boundaries to preserve

- [DecisionBackend](src/decision.rs) already exposes a common `decide()` contract. Binary, choice, and ordinal results, semantic option IDs, candidate-relative probabilities, full-vocabulary candidate mass, and abstention already exist.
- [LlamaBackend](src/llama.rs) already renders embedded GGUF templates and has explicit Qwen and GPT-OSS profiles. `prepare()` checks candidate tokenization at the assistant continuation, rather than trusting isolated letter tokens. Missing templates and oversized requests fail explicitly.
- [Native execution](native/bridge.cpp) already supports fresh execution, batch-aligned prefix reuse, and isolated parallel sequences. Prefix reuse falls back to fresh for recurrent/hybrid memory; parallel execution rejects those models. Cache state is cleared at request boundaries and on errors.
- [OutputHead](src/output_head.rs) and [its runtime contract](OUTPUT_HEAD.md) already bind a learned head to model SHA-256, task meaning, device, compute options, and prompt version. Temperature-only correction is representable by an identity `logit_affine` head. Head use currently requires fresh execution; hidden features are Gemma4-specific. The existing calibration scripts and separate train/development/calibration/test data should be extended rather than duplicated.

These are existing capabilities, not benefits supplied by the proposed work. Interchangeability should preserve these guarantees; it does not promise that different models make identical decisions or have interchangeable thresholds.

## Recommended adoption order

| Priority | Design | Source of the idea | L2S1 change | Expected effort |
| --- | --- | --- | --- | --- |
| P0 | Model identity, capability report, and request preflight | SemIf's verified tokenizer/metadata; decidr's explicit backend requirements | Explain compatibility before a workload runs | Medium |
| P0 | Separate inference evidence from decision policy | Open Alternative to Jev's narrow backend protocol; System One Adapter's provider boundary | Keep probability and abstention semantics in one place | Medium |
| P0 | One conformance suite across model families | Upstream backend equivalence and lifecycle tests | Make model replacement a tested contract | Medium |
| P1 | Reusable, scoped calibration artifacts | SemIf's workload calibration; Open Alternative to Jev's scaler | Generalize existing L2S1 calibration without transferring it across models | Medium |
| P1 | Request-local execution diagnostics and typed failures | SemIf's provenance; System One Adapter's attempt records | Expose actual execution, fallback, and failure causes | Small–medium |
| P2 | Whole-sequence state restore for hybrid models | SemIf's llama.cpp backend | Evaluate an opt-in reuse path where suffix removal is unsupported | Medium–large, experimental |
| P2 | Bounded backend ownership and scheduling | decidr's shared worker pool; System One Adapter's lifecycle | Useful if L2S1 adds a service or remote backend | Medium, conditional |

### 1. Model identity and preflight

[SemIf verifies GGUF/reference-tokenizer agreement](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/src/semif_phase1/llamacpp_backend.py) and records model/checkpoint provenance. [decidr makes logprob availability an explicit backend requirement](https://github.com/devanmolsharma/decidr/blob/58df99474cebbc96601cb8d5e64cfcb63fffbe75/src/decidr/backend.py). Adopt early validation and explicit requirements, while retaining L2S1's self-contained GGUF tokenizer and template.

Propose a model inspection API and a CLI preflight command. Separate three results:

1. **Declared capabilities:** architecture, context, score availability, supported execution modes, LoRA and hidden-feature restrictions.
2. **Validated request compatibility:** rendered template, actual answer-boundary token IDs, option count, context usage, policy requirements, head/calibration binding.
3. **Measured quality:** a separately run labeled workload. Successful loading or a tiny scoring probe is not evidence of task accuracy.

Use an immutable identity containing checkpoint checksum, template/tokenizer identity, prompt profile/version, runtime build identity, and active adapter/head identity. Record device and effective compute configuration for reproducibility. A checksum may be computed once during registration or verified artifact loading; do not rehash multi-gigabyte weights for each decision. Distinguish verified identity from path-only information.

Report, for example, that a checkpoint supports fresh decisions but not parallel sequences, or that the request's candidate codes fail the continuation check. Current `configure_profile()` and `prepare()` supply much of this logic; expose them through a structured report instead of introducing another formatter. Retain request-time checks because options and context length vary. Do not replace actual checks with a model-name allowlist.

### 2. Inference evidence and decision policy

[Open Alternative to Jev's backend](https://github.com/ikermoel/open-alternative-jev/blob/4a85df1831b537343c9e133b5150fa3a3b1ce98e/so1/backends/base.py) returns candidate scores; its [Decider](https://github.com/ikermoel/open-alternative-jev/blob/4a85df1831b537343c9e133b5150fa3a3b1ce98e/so1/decider.py) owns orchestration and normalization. This is useful separation, but its minimal score contract is insufficient for L2S1's full-vocabulary mass gate.

Keep the public `DecisionBackend` facade and introduce an internal evidence boundary first:

```text
DecisionRequest
  -> model-bound prompt preparation
  -> inference backend
  -> candidate evidence and score provenance
  -> optional matching calibration/head
  -> shared DecisionPolicy
  -> DecisionResponse
```

The evidence should identify semantic options, candidate token mapping, score origin, completeness, normalizer/mass availability, and effective model identity. For the current native path, retain exact candidate logits and the full-vocabulary log normalizer:

```text
p(option_i | supplied options) = exp(z_i - logsumexp(candidate logits))
candidate_mass = exp(logsumexp(candidate logits) - logsumexp(full vocabulary))
```

Different evidence kinds must be distinguishable: raw logits, full-vocabulary logprobs, candidate-normalized logprobs, incomplete top-k observations, and model-generated numeric answers. Candidate-normalized probabilities alone cannot reconstruct full-vocabulary mass. An API returning partial scores must not silently supply `candidate_mass = 1`, fictitious `raw_logit` values, or invented token IDs.

Initially keep the existing exact local result contract and reject backends that cannot satisfy it. If partial/API evidence is later supported, design an explicit versioned result with optional evidence and an explicit policy for unavailable mass. Adding optional JSON fields can preserve deserialization, but adding fields to public Rust structs or variants to exhaustive enums can break source compatibility; provide constructors/versioned types or plan the API revision deliberately.

A native implementation could eventually return candidate logits plus a computed full-vocabulary normalizer instead of copying the entire vector into Rust. This would reduce a bridge copy/allocation; it does not by itself remove the vocabulary projection or GPU-to-host transfer. Preserve numerical behavior and measure before claiming a speed improvement.

### 3. Model replacement conformance suite

[Open Alternative to Jev checks HF/vLLM agreement](https://github.com/ikermoel/open-alternative-jev/blob/4a85df1831b537343c9e133b5150fa3a3b1ce98e/tests/test_vllm_backend.py); [SemIf tests exact cache identity and restoration](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/tests/test_llamacpp.py). Reuse the testing pattern, not their tolerances or checkpoint-specific expected answers.

Build on L2S1's existing native, decision, parallel, prefix-reuse, and output-head tests. Run the same public contract over multiple template/tokenizer families, including the existing Qwen, Gemma, SmolLM2, and GPT-OSS paths. Include a recurrent/hybrid checkpoint when validating memory capabilities.

Acceptance criteria:

- The same request preserves decision IDs, semantic option IDs, output kinds, ordinal values, and abstention explanations across models. Predictions need not match across different models.
- Base candidate probabilities and full-vocabulary mass agree with an independent reference calculation for the same logits. Reject duplicate, unstable, missing, and multi-token candidate mappings.
- Missing templates, unsupported modes, oversize inputs, and incompatible artifacts produce specific errors without silent template/device/model substitution.
- Model or template changes invalidate prepared prompts and model-specific artifacts; a failed request cannot contaminate the next request.
- Same-model fresh/reuse/parallel comparisons report probability deltas, top-choice changes, accepted-decision changes, and request ordering effects. Published upstream tolerances are not imported blindly.
- Adapter/head compatibility is tested separately from base-model compatibility.

Publish a capability matrix with exact checkpoint/runtime identities and evidence status. Keep fixture/synthetic, real-model contract, and labeled quality results separate. This matrix is a more useful interchangeability claim than “supports any model.”

### 4. Calibration that survives application code changes, not model changes

[SemIf's calibration design](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/docs/CALIBRATION.md) keeps raw inference intact and evaluates workload-specific temperature scaling with group-disjoint folds. [Open Alternative to Jev](https://github.com/ikermoel/open-alternative-jev/blob/4a85df1831b537343c9e133b5150fa3a3b1ce98e/so1/calibration.py) exposes fitting and application as a separate component.

L2S1 already has temperature fitting and deployment-bound output-head artifacts. Extract/generalize that machinery into an optional scalar calibration artifact usable by ordinary binary, choice, and ordinal decisions, instead of requiring a learned choice head for temperature-only use.

Bind the artifact to verified model identity, prompt/template and score semantics, workload/task signature, execution/compute configuration where relevant, and calibration-data provenance. Replacing weights, quantization, adapter, or template must reject a mismatched artifact. Do not silently reuse a temperature selected for Gemma with Qwen, or silently discard an explicitly requested incompatible artifact.

Preserve raw scores and base mass; record the applied artifact. A positive scalar temperature preserves mathematical argmax, but threshold acceptance, near-tie handling, coverage, and accepted accuracy can change. Measure those outcomes, not just ECE. Split paraphrases and option-order variants by source example to avoid leakage. Retain independent final-test evaluation, NLL/Brier metrics, and per-model coverage/accepted-accuracy curves. Calibration is evidence for its declared workload, not a universal correctness guarantee.

### 5. Diagnostics and failures at the request boundary

[System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/src/system_one_adapter/providers/base.py) separates one provider call from orchestration and uses request-local attempt capture. Its [client](https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/src/system_one_adapter/_client.py) separates malformed-output correction from transport retries. Adopt that separation if API backends are added; native incompatibility errors should never trigger generic retry or model substitution.

For local execution, add a diagnostics envelope with model/prompt-token fingerprints, requested/effective execution modes, cache fallback reason, timing stages, evidence completeness, and artifact IDs. L2S1 currently records requested mode and reused-token count; zero reused tokens alone cannot distinguish a short prefix from unsupported memory. Keep diagnostics request-local and record batch timing once. Default to metadata/hashes; raw prompts and provider payloads should require explicit tracing.

Distinguish unsupported capability, incompatible artifact, context exceeded, invalid score evidence, backend failure, and ordinary policy abstention. An unavailable score is not a low-confidence prediction. Structure error details without forcing existing Rust callers into an unplanned exhaustive-enum migration.

### 6. Hybrid-model reuse through state restoration

[SemIf's llama.cpp backend](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/src/semif_phase1/llamacpp_backend.py) saves a complete sequence prefix and restores it before each suffix. This offers a candidate path for models whose recurrent state cannot support partial suffix removal. L2S1 currently uses fresh execution for that case.

Evaluate this as a separate opt-in mode, scoped to one request. Bind snapshots to exact tokens, model/context identity, prompt configuration, adapters, and runtime; clear them on errors and configuration changes. Measure snapshot bytes, copy time, prefill time, suffix time, and peak memory. Check same-model score and acceptance agreement against fresh execution on short and long inputs. Restore overhead may erase the benefit, and this serial technique does not establish parallel-sequence support. Preserve fresh fallback until compatibility and performance are demonstrated for the exact runtime/checkpoint.

### 7. Backend lifecycle and bounded scheduling

[decidr uses one bounded executor per client](https://github.com/devanmolsharma/decidr/blob/58df99474cebbc96601cb8d5e64cfcb63fffbe75/src/decidr/core.py). [System One Adapter's lifecycle tests](https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/tests/test_provider_lifecycle.py) cover owned versus injected providers, concurrent closure, and failed construction.

This becomes useful if L2S1 exposes a long-running service: use bounded queues and a configured memory budget, keep requests bound to one model instance, and release only resources the service owns. Existing `LlamaBackend` intentionally is not `Send`/`Sync`; do not place its context behind a thread pool without an ownership design. A dedicated worker can own a model/context and serialize access. Model selection does not require keeping every model resident. Live replacement additionally needs draining/rollback and memory planning; defer it unless the product requires it.

## Designs to reject or defer

| Upstream behavior | Decision for L2S1 | Reason |
| --- | --- | --- |
| [vLLM missing-label floor scores](https://github.com/ikermoel/open-alternative-jev/blob/4a85df1831b537343c9e133b5150fa3a3b1ce98e/so1/backends/vllm.py) | Reject in the exact scoring path | The source substitutes `min(returned scores) - 1` for missing labels. That is an estimate, not observed probability mass. Constrained candidate probabilities also do not provide the original full-vocabulary mass. |
| [Packed multi-turn questions and ChatML defaults](https://github.com/ikermoel/open-alternative-jev/blob/4a85df1831b537343c9e133b5150fa3a3b1ce98e/so1/prompting.py) | Do not make them the default | Later questions can attend to earlier questions/placeholders. Preserve isolated decisions and embedded-template support; any packing experiment needs its own scoring/prompt version and order-interference evaluation. |
| [decidr's semantic-ID prefix walk and early stop](https://github.com/devanmolsharma/decidr/blob/58df99474cebbc96601cb8d5e64cfcb63fffbe75/src/decidr/core.py) | Defer to a separately named scoring method | It can use several requests and stop before fully scoring an option string. This is not equivalent to L2S1's single-token candidate distribution. Preserve the separation between application IDs and output codes. |
| [Persistent token cache keyed by model name and option ID](https://github.com/devanmolsharma/decidr/blob/58df99474cebbc96601cb8d5e64cfcb63fffbe75/src/decidr/token_cache.py) | Do not copy the identity scheme | Verified token reuse is useful, but a model name omits tokenizer/template revisions and endpoint identity. L2S1 already has the local tokenizer, so speculative remote token discovery has little immediate value. |
| Generated JSON probabilities as a replacement for native logits | Separate optional adapter only | Generated numeric estimates have different semantics. They must not populate the exact native evidence fields or silently share calibration. |
| Fixed-model hidden-head optimization as the main interface | Keep as an optional specialization | A Gemma4-specific feature path is useful for a task, but cannot become a requirement for switching ordinary GGUF models. |

## Implementation sequence and proof boundary

First, expose identity/capability/preflight reports and extend the existing cross-model contract suite. Next, introduce the internal evidence boundary with a native adapter while retaining current public behavior. Then generalize calibration and add effective-execution diagnostics. Only after those foundations should remote providers, hybrid snapshots, or a multi-model service be considered.

Before a refactor is accepted, compare the same models, prompts, compute settings, and requests against the current implementation. Preserve base scoring, candidate mass, abstention, and legacy defaults. Evaluate quality and performance changes in separate opt-in experiments. This review establishes source-level design opportunities; it establishes no new model compatibility, accuracy, latency, or production claim.
