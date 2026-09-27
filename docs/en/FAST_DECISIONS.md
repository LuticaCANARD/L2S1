# Fixed prefixes, ONNX decisions and validated routing

[English](../en/FAST_DECISIONS.md) · [한국어](../ko/FAST_DECISIONS.md) · [日本語](../ja/FAST_DECISIONS.md)

Compatible resident text servers (`--listen` / `--stdio`, including SDK `load`) automatically use bounded fixed-schema prefix KV reuse when no execution mode is specified. One-shot CLI calls and the low-level Rust backend retain `fresh`. Explicit execution modes, vision/projector settings, calibration/output heads and compact evidence preserve their existing paths; recurrent/hybrid models fall back to fresh with a startup message. Use `--execution-mode fresh` to opt out, or `--fixed-schema` to require support. Native parallel batching still requires `--execution-mode parallel`. This changes the split plan and can change scores; check capabilities and `usage.reused_prefix_tokens`.

This default is a source change after v0.1.3. Published v0.1.3 requires explicit `fixedSchema: true` / `fixed_schema=True`; use a newly built binary until the next release.

## Fixed-schema GGUF sessions

Build with `llama` (CPU), `llama-cuda` or `llama-metal`. Start a resident server:

```sh
l2s1 --model model.gguf --listen 127.0.0.1:8080
```

`FixedSchemaBackend` uses the same explicit prefix/suffix split on cold and warm calls, independently of the prefill batch size. It preserves complete prompt tokenization and matches actual token prefixes, including BPE boundaries. One native KV context is active. Up to eight prefix snapshots occupy at most 256 MiB; schema preparation has a separate 64-entry/4 MiB limit. Eviction recomputes the prefix. Answers are never cached. `clear()` clears retained KV/snapshots; use a dedicated backend/server per trust domain.

This execution plan can change scores compared with unsplit fresh execution. Validate both **flat fresh versus split cold** and **split cold versus split warm**. The latter is the cache equivalence check. Explicit fixed-schema requests reject recurrent/hybrid models, existing calibration artifacts and output heads; calibrations fitted to the old plan must not silently transfer. Full evidence is required. More than 26 options and images use the existing path with KV cleared. `shared_decision()` retains its original batch-aligned behavior.

TypeScript: `L2S1.load({ model, fixedSchema: true, binaryPath: "./target/release/l2s1" })`. Python: `LoadOptions(model=..., fixed_schema=True, binary_path="./target/release/l2s1")`. Both select prefix-reuse automatically when execution mode is omitted, and support stdio or HTTP. `prepare()` alone still does not create KV state. Check `/v1/capabilities.prefix_reuse` and per-result `usage.reused_prefix_tokens`.

Explicit `fixedSchema: true` / `fixed_schema=True` requires this plan and errors on incompatible configurations. Omit the option for automatic selection; `false` / `False` selects fresh when no explicit execution mode is given. Existing cascade policies must match the exact serving plan: preserve their previous explicit settings or refit and validate the policy.

## Laya ONNX backend

```sh
cargo build --release --features llama,onnx --bin l2s1 --bin l2s1-onnx
python3 scripts/fetch_laya_onnx.py --output models/laya-en
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so
l2s1-onnx --model-dir models/laya-en --min-top-probability 0 --listen 127.0.0.1:8081
```

ONNX Runtime >=1.24 is loaded dynamically; install its CPU/GPU distribution separately. CUDA requires the `onnx-cuda` feature, a compatible CUDA Runtime/provider and `--cuda`; CPU fallback is not silently enabled. Fetch `--precision fp16` into a separate directory for the CUDA fast path. GGUF CUDA independently requires `llama-cuda`.

The downloader pins graph, original weights, tokenizer and calibration hashes. Models are Apache-2.0; the derived graph and JSON formatting provenance are in `third_party/ollaya`. The backend supports Laya marker heads, not arbitrary ONNX networks. State tokenization is shared; each question remains an independent encoder row. Batches are bounded by 32,768 padded input tokens and 128 decisions. L2S1 sorts structured JSON keys; exact Ollaya parity requires the same key order. Overlong states, instructions and options are rejected rather than silently truncated.

Evidence type `discriminative` includes candidate logits/probabilities, schema identity, applied temperature and estimates. It has **no vocabulary token IDs or candidate mass**. It does not accept the GGUF request policy or `target_error_rate`. The separate maximum-probability threshold is not an error guarantee. Model artifacts and the ONNX runtime library are included in the serving identity. Validate target hardware and deployment precision.

## Calibrated fast/slow routing

First run the fast server above and the intended slow server. Prepare disjoint calibration and validation JSONL files: `{id, request, expected: {decision_id: option_id}}`. Binary labels are the strings `false` and `true`; ordinal labels are level IDs.

```sh
python3 scripts/calibrate_cascade.py --fast-url http://127.0.0.1:8081 \
  --slow-url http://127.0.0.1:8080 --calibration calibration.jsonl \
  --validation heldout.jsonl --max-error 0.05 --output cascade.json
```

The script checks disjoint IDs/exact states, fits thresholds only on calibration labels and rejects schemas failing the frozen threshold on validation. This cannot establish absence of near duplicates, domain leakage or future distribution shift. Error limits are empirical held-out measurements, not statistical guarantees; inspect sample size and coverage. No valid rule means no fast acceptance.

Restart the slow server with `--fast-model-dir models/laya-en --cascade-policy cascade.json` (and `--fast-cuda` for CUDA). Keep all slow settings unchanged. The binary needs both relevant backend features. `--fast-model-dir` without a policy runs slow-only. Model/runtime identity mismatch fails startup. Only validated exact schemas can use the fast result; low confidence, abstention, unsupported inputs and fast-backend errors fall back to the slow model. Images go directly to a vision-capable slow model. Slow failures propagate. Every result records its chosen backend, route and reason, preserving decision order. Request-local acceptance policy overrides are rejected for cascades.

HTTP now coalesces compatible text/direct requests only when the backend advertises enabled native batching: at most eight jobs, 128 decisions, 44 MiB and 1 ms collection. Prefix sessions and cascades currently execute serially. Queued jobs expire after 180 seconds; an already running native call cannot be interrupted. Throughput, single-request latency and p95 must be measured separately.

## Verification

`benchmark_fixed_schema` compares flat fresh, split cold and split reuse on the original multi-decision rule requests. `benchmark_onnx` records warmed in-process timing and input/score evidence for reference comparisons. These are local diagnostics; they do not establish general task accuracy or an Ollaya headline speed on different hardware.


## Local measurements (2026-09-27)

On the local WSL2 CPU, Qwen3 0.6B Q8_0 processed all 36 decision-rules judgments in 29.132 s fresh, 31.062 s split cold and 14.962 s with split reuse (1.95x versus fresh). Batch size remained 256; 2,799/4,983 tokens were reused, with no selection or probability changes. These are whole-pass totals from one measured pass after warmup, not request p50 or M5 Max results.

On RTX 3080, the English Laya fp16 path measured 19.179 ms request p50 versus Ollaya's 19.567 ms, with p95 22.165/23.338 ms. The same 20 synthetic ticket requests contain five questions each, repeated twice after 20 warmups. CPU and CUDA token IDs, marker positions and logits match exactly; probability differences are below 2.23e-16. L2S1 timing includes probability scoring, while the reference excludes it. This establishes local parity, not general accuracy or a universal speed advantage.

Real HTTP testing confirmed fast acceptance, schema/input fallback, Python SDK decoding and up to six coalesced requests among 16 concurrent calls. The calibration test used toy data and does not validate production quality. See the [measurement summary](../../benchmarks/fast-decisions-20260927/summary.json) and [provenance](../../benchmarks/fast-decisions-20260927/provenance.json). Full records and reproduction commands are in the repository's `benchmarks/fast-decisions-20260927` directory.


## SIMD evidence postprocessing

Full-vocabulary logit validation and maximum scanning now use runtime AVX2/NEON dispatch, with scalar fallback. Exponentials and sequential f64 sums keep their original order. Local AVX2 measurements show about 10x faster scanning and 1.17–1.26x faster complete normalization for 32k–262k vocabularies; these are **not whole-inference speedups**. All 36 real SmolLM2 rule results match the pre-SIMD binary exactly. See the [SIMD measurement](../../benchmarks/evidence-simd-20260927/summary.json) and [native parity](../../benchmarks/evidence-simd-20260927/native-parity.json). NEON timing remains unmeasured.
