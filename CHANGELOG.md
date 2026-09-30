# Changelog

## 0.2.0 (2026-09-30)

Parallel prefix sharing. Several defaults change prompts or wave schedules, so
decisions, scores and calibration identities can differ from 0.1.4. Re-evaluate
workload accuracy and recalibrate before upgrading a deployment. See
[parallel prefix sharing](docs/PARALLEL_EXECUTION.md#prefix-sharing-controls-020).

### Changed (can change results)

- Text `parallel` execution now defaults to the `state-first` prompt layout
  when no layout is selected (`--prompt-layout`, `set_prompt_layout`). The
  legacy layout puts the question before the state, so waves of mixed
  questions shared almost no prefix (0% reuse in measurements). Other modes
  and backends with a vision projector keep `legacy`. `prompt_layout` and
  `prompt_version` report the resolved layout, so calibration bound to
  `legacy` parallel prompts is rejected rather than silently misapplied. An
  explicitly selected `legacy` layout that reuses no token in a multi-question
  parallel call prints one warning.
- Parallel waves are formed from prompts sorted by token sequence
  (`ParallelWaveOrder::Prefix`, `--parallel-wave-order prefix`), so questions
  with common prefixes share a wave. `request` restores the 0.1.4 schedule.
  Results keep request order.
- Shared prefixes inside a wave form a tree: questions that agree for longer
  keep sharing after another question diverges, instead of the whole wave
  falling back to one common prefix.
- Between the waves of one parallel call, the previous wave's root shared
  prefix is kept when the next wave starts with the same tokens. KV is still
  cleared at call boundaries. `reused_prefix_tokens` of a later wave's first
  question can therefore be nonzero.

### Added

- `LlamaBackend::load_vocab_only` loads only the GGUF tokenizer, chat template
  and metadata (llama.cpp `vocab_only`) for `encode_decision`,
  `encode_decision_sequences` and `preflight`; inference returns an error.
  `export_decision_tokens` uses it, so exporting no longer loads the weights.
  Tokens and `model_identity` match a default CPU load.
- `l2s1-train run --backend mlx` trains decision adapters on Apple Silicon
  with MLX, writes PEFT-format adapters for the existing GGUF LoRA conversion,
  and evaluates on Metal (`training[mlx]`, `l2s1-train doctor --mlx`).
- `DecisionRequest.shared` (JSON `"shared"`): evidence common to many requests
  or questions, always rendered first in the data segment in both layouts.
  Callers no longer need to name `state` keys so that they sort first. Absent
  or `null` leaves prompts unchanged; the ONNX backend rejects it.
  `DecisionRequest::new` and `DecisionRequest::input` helpers and the public
  `PromptInput` type accompany it.
- `ParallelPrefixAlignment::Token` (`--parallel-prefix-alignment token`)
  shares every common token instead of rounding each shared segment down to a
  complete prefill batch. Faster for long common prefixes, but shared KV no
  longer uses serial decode boundaries; `batch` remains the default.
- `LlamaBackend::parallel_prefix_session()` returns a `ParallelPrefixSession`
  that keeps the root shared prefix between `decide` / `decide_batch` calls,
  so repeated calls (for example triage batches with the same knowledge and
  examples) do not evaluate it again.
- `BackendInfo.parallel_prefix_alignment`, `parallel_wave_order` and
  `parallel_prefix_retained` report these settings.
- `LlamaBackend::prompt_layout()` and `reset_prompt_layout()`.
- `FamilyCalibration`: one temperature per decision kind and option count,
  fitted on labeled records from at least two tasks, applied to tasks without
  a task-specific `ScalarCalibration` (which still takes precedence). The
  artifact reports leave-one-task-out NLL, Brier and ECE for every fitting
  task. CLI `--family-calibration`, `register_family_calibration()`,
  `load_family_calibration()` and the `fit_family_calibration` example.
- `CalibrationMetrics.ece`: top-label expected calibration error over 10
  confidence bins, reported by every calibration fit and held-out evaluation.

### Fixed

- `ModelIdentity` now includes `parallel_prefix_alignment` and
  `parallel_wave_order`. Both change parallel scores, so a calibration fitted
  under one setting is rejected under another instead of being misapplied.
  Default (`batch`, request order) values are omitted, so fingerprints of
  non-parallel configurations are unchanged.
- Prompt JSON (payload fields and every nested object in `state` and
  `shared`) is serialized with sorted keys regardless of serde_json's
  `preserve_order` feature. Cargo unifies features across the dependency
  graph, so a single dependency enabling it previously changed `legacy`
  prompts, decisions and calibration baselines silently. Output bytes equal
  the previous default build.

### Breaking API changes

- `DecisionRequest` has a new public field; struct literals need
  `shared: None` (or use `DecisionRequest::new`).
- `l2s1_llama_sys::sd_forward_parallel` takes a `flags` argument
  (`SD_PARALLEL_TOKEN_PREFIX`, `SD_PARALLEL_RETAIN_PREFIX`).
- `LlamaBackend::encode_decision` and `encode_decision_sequences` accept
  `impl Into<PromptInput>`; `&serde_json::Value` still works.
- `ModelIdentity` and `CalibrationMetrics` have new public fields; struct
  literals need `parallel_prefix_alignment`, `parallel_wave_order` and `ece`.
