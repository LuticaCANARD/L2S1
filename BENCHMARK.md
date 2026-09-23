# Local model comparison benchmark

The `decision-rules-v1` benchmark compares existing GGUF checkpoints on **12 requests with 36 labeled decisions**. It measures rule-following correctness, abstention, repeated-output consistency, and inference latency. It does not download or distribute models.

The default is the legacy v1 prompt. Optional v2 state-first prompts and optional prefix reuse are documented in [SEMIF_ALGORITHM.md](SEMIF_ALGORITHM.md). Pass `--prompt-layout state-first` for v2, then `--execution-mode fresh` (default) or `--execution-mode prefix-reuse` to the Python runner, using separate output directories for comparisons. Reports record the requested mode, logical input tokens, reused prefix tokens, and actual evaluated tokens. Compare the same prompt version, model, device, and batch setting; changing the prompt can change accuracy independently of cache reuse.

The English fixture in `tests/fixtures/decision_benchmark.json` covers two domains:

- Warehouse: storage selection, cold-chain requirements, and dispatch priorities at 0, 6, 7, 24, 25, and 48 hours.
- Access control: role-to-scope mapping, boolean editing permissions, and failed-attempt review priorities at 0, 1, 2, 3, and 4 attempts.

Every request has one choice, one binary, and one ordinal decision. Choice option order varies; ordinal values remain sorted as required by the API. Some cases contain irrelevant notes. Expected answers are semantic option IDs, independent of A/B/C token positions. Ordinary Rust tests independently recompute every label from the scenario rules.

This is a small synthetic rule-following benchmark, not a general reasoning, coding, multilingual, safety, or production-accuracy evaluation. Do not tune against these labels and present the resulting numbers as held-out accuracy. Real-model consistency tests in `tests/native.rs` remain separate.

## Run the five-model matrix

Prerequisites: Rust dependencies already cached, CMake, a C++17 compiler, and locally acquired checkpoints. The bundled sys dependency builds the pinned llama.cpp source; CUDA runs additionally need the CUDA toolkit. Model paths are listed in `tests/fixtures/benchmark_models.json`; no Hugging Face client, account, or network request is used by the runner. Review the separate licenses before obtaining a model.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools benchmark-models \
  --device cpu cuda \
  --iterations 3 \
  --warmup 1
```

The runner builds a release test executable once with `--locked --offline` and the CUDA feature when requested, hashes each model, and runs one model/device at a time. CPU is the default device. CUDA is explicit and cannot silently fall back to CPU. The native backend reuses one loaded model for the whole run; Rust process startup, Cargo compilation, loading, and warmup are excluded from steady-state timings.

Select a subset or change settings:

```sh
target/release/l2s1-tools benchmark-models \
  --model gemma4 --model qwen3 \
  --device cuda \
  --iterations 10 --warmup 2 \
  --context 2048 --batch 256 --threads 4 \
  --min-top-probability 0.8 --min-candidate-mass 0.05 \
  --timeout 1800 \
  --output results/benchmark/my-comparison
```

`--iterations` means full measured passes over all 12 requests. `--warmup` means full excluded passes, in addition to one separately timed first request. Case order rotates deterministically between measured passes. Repeats help characterize timing and consistency; they do not increase the number of independent accuracy examples.

Output directories must be new so stale reports cannot be mistaken for current results. The default is a timestamped directory under ignored `results/benchmark/`. `--timeout` applies separately to each model/device, including warmup. Missing files, native errors, and timeouts remain visible as failed rows; the runner continues with other models and returns a nonzero exit code if any run fails. Low accuracy is reported as a measurement, not a test failure.

For custom checkpoints, pass `--manifest /path/to/models.json`:

```json
{
  "models": [
    {"id": "my-model", "path": "/absolute/path/to/chat-model.gguf"}
  ]
}
```

Relative checkpoint paths resolve against the repository root, including in custom manifests. IDs must be unique lowercase names containing only letters, digits, hyphens, and underscores. The benchmark uses the backend's automatic prompt profile, so models receive the same semantic tasks through their respective templates. Compare template/token counts as well as model sizes.

## Reports and metric definitions

Each directory contains `summary.md`, `summary.json`, a build log, and per-model/device JSON and native logs. Checkpoint SHA256, fixture SHA256, executable SHA256, available repository/llama.cpp revisions, device, quantization description, settings, policy, individual labels, predictions, probabilities, and latency samples are retained. Raw local reports can contain local model paths; review them before publishing.

| Metric | Definition |
| --- | --- |
| Coverage | Accepted decisions / all measured decisions |
| Abstention rate | Abstained decisions / all measured decisions |
| Accepted accuracy | Correct accepted decisions / accepted decisions; `null` / `n/a` if none accepted |
| Correct / all | Correct accepted decisions / all decisions; abstentions do not count as correct |
| Raw top-1 accuracy | Highest-scoring semantic option versus the label, before policy abstention; ties count incorrect |
| Correct accepted/s | Correct accepted decisions / measured inference seconds |
| Decisions/s | All completed decisions / measured inference seconds, including abstentions |
| Request p50/p95 | Nearest-rank percentiles of full `decide` calls, each containing three sequential decisions |
| Input tokens/s | Input tokens processed / inference seconds; **not** generated tokens/s |
| Repeat consistency | Accepted output and raw top-1 changes versus the first pass for the same case/decision |

Quality counts and fractions also appear by domain and by decision kind. Full samples permit per-case inspection. Repeated outputs may keep the same selected option while scores change; this report does not replace the native suite's probability/batch checks.

Loading and the first request are reported separately. Timings include prompt preparation, prefill, logits transfer, and scoring, but exclude JSON report serialization. The OS page cache is not flushed; load times are not cold-disk measurements. Full context capacity, process RSS, peak VRAM, and concurrent serving throughput are not measured. Hardware, quantization, context, batch, thread count, and policy must be held consistent for meaningful comparisons. There are no arbitrary hardware speed thresholds.

## Test the benchmark without models

```sh
cargo test --locked --offline --test benchmark
python3 -m unittest discover -s scripts -p 'test_benchmark_models.py'
```

These validate fixture labels and metric denominators, including ties and all-abstaining output, plus manifest validation and failure-preserving reports. They do not establish real inference correctness.

The actual model benchmark is an ignored Rust test in `tests/benchmark.rs`. It can also run directly:

```sh
SKID_MODEL=/absolute/path/to/model.gguf \
SKID_CUDA=1 \
SKID_BENCH_ITERATIONS=3 \
SKID_BENCH_WARMUP=1 \
SKID_BENCH_OUTPUT=results/benchmark/single-model.json \
  cargo test --release --locked --offline --features llama --test benchmark \
  native::model_decision_benchmark -- --exact --ignored --nocapture
```

Direct runs accept `SKID_CONTEXT`, `SKID_BATCH`, `SKID_THREADS`, `SKID_MIN_TOP_PROBABILITY`, and `SKID_MIN_CANDIDATE_MASS`. The Python runner adds checkpoint hashes and the comparison table. The earlier `tests/performance.rs` remains available for repeated measurements of the single warehouse example.
