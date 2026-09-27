<a id="local-model-comparison-benchmark"></a>
# Local model comparison benchmark

[English](BENCHMARK.md) · [한국어](../ko/BENCHMARK.md) · [日本語](../ja/BENCHMARK.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)


The `decision-rules-v1` benchmark compares existing GGUF checkpoints on **12 requests with 36 labeled decisions**. It measures rule-following correctness, abstention, repeated-output consistency, and inference latency. It does not download or distribute models.

The default is the legacy v1 prompt. Optional v2 state-first prompts and optional prefix reuse are documented in [SEMIF_ALGORITHM.md](SEMIF_ALGORITHM.md). Pass `--prompt-layout state-first` for v2, then `--execution-mode fresh` (default) or `--execution-mode prefix-reuse` to the Rust runner, using separate output directories for comparisons. Reports record the requested mode, logical input tokens, reused prefix tokens, and actual evaluated tokens. Compare the same prompt version, model, device, and batch setting; changing the prompt can change accuracy independently of cache reuse.

The English fixture in `tests/fixtures/decision_benchmark.json` covers two domains:

- Warehouse: storage selection, cold-chain requirements, and dispatch priorities at 0, 6, 7, 24, 25, and 48 hours.
- Access control: role-to-scope mapping, boolean editing permissions, and failed-attempt review priorities at 0, 1, 2, 3, and 4 attempts.

Every request has one choice, one binary, and one ordinal decision. Choice option order varies; ordinal values remain sorted as required by the API. Some cases contain irrelevant notes. Expected answers are semantic option IDs, independent of A/B/C token positions. Ordinary Rust tests independently recompute every label from the scenario rules.

This is a small synthetic rule-following benchmark, not a general reasoning, coding, multilingual, safety, or production-accuracy evaluation. Do not tune against these labels and present the resulting numbers as held-out accuracy. Real-model consistency tests in `tests/native.rs` remain separate.

<a id="recorded-windows-rtx-5090-results"></a>
## Recorded Windows / RTX 5090 results

The user supplied this historical summary, recorded at **2026-09-26 16:21:17 UTC** (2026-09-27 01:21:17 KST), and identified the GPU as **NVIDIA GeForce RTX 5090**. Windows x86_64; CPU model and memory measurements are unavailable. The GPU identification is user-confirmed; no device log was supplied.

| Model | Device | Status | Coverage | Accepted accuracy | Correct / all | Raw top-1 | p50 ms/request | p95 ms/request | Decisions/s | Correct accepted/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smollm2 | cpu | ok | 13.9% | 20.0% | 2.8% | 38.9% | 7156.9 | 7805.6 | 0.41 | 0.01 |
| smollm2 | cuda | ok | 11.1% | 25.0% | 2.8% | 38.9% | 41.6 | 44.9 | 71.61 | 1.99 |
| qwen3 | cpu | ok | 77.8% | 35.7% | 27.8% | 33.3% | 22681.9 | 25154.9 | 0.13 | 0.04 |
| qwen3 | cuda | ok | 83.3% | 33.3% | 27.8% | 41.7% | 44.7 | 47.0 | 66.42 | 18.45 |
| gemma3 | cpu | ok | 91.7% | 51.5% | 47.2% | 50.0% | 33004.2 | 36213.8 | 0.09 | 0.04 |
| gemma3 | cuda | ok | 91.7% | 57.6% | 52.8% | 61.1% | 67.7 | 70.7 | 44.37 | 23.42 |
| tinyllama | cpu | ok | 0.0% | n/a | 0.0% | 38.9% | 9513.6 | 10583.5 | 0.31 | 0.00 |
| tinyllama | cuda | ok | 0.0% | n/a | 0.0% | 38.9% | 34.2 | 36.3 | 87.59 | 0.00 |
| gemma4 | cpu | timeout | — | — | — | — | — | — | — | — |
| gemma4 | cuda | ok | 97.2% | 94.3% | 91.7% | 94.4% | 67.6 | 80.7 | 43.90 | 40.24 |

Each completed run has 12 distinct requests / 36 labeled decisions, repeated 3 times: 36 timed requests / 108 measured decisions. Repeats are not independent accuracy examples. Loading and warmups are excluded. Settings: `legacy`, `fresh`, context 2048, batch 256, 4 threads, thresholds 0.8 / 0.05, timeout 1800 seconds.

Gemma4 CUDA is the strongest candidate within this fixture: 33 correct accepted, 2 wrong accepted, and 1 abstention per unchanged pass. Accepted accuracy is 33/35 (94.3%), correct/all is 33/36 (91.7%), raw top-1 is 34/36 (94.4%), and coverage is 35/36 (97.2%). The 67.6 ms p50 measures a three-decision request. Decisions/s includes abstentions. TinyLlama accepts nothing, so accepted accuracy is `n/a`; Gemma4 CPU timed out and has no completed metrics.

All 10 table rows match the supplied summary. The arithmetic audit checks 324 timing samples, counts, ratios, nearest-rank percentiles, and throughput. The fixture SHA256 matches after LF → Windows CRLF conversion. Every completed run reports 0 changes in 72 repeat comparisons, but Qwen3 and Gemma3 raw top-1 and SmolLM2 coverage differ between CPU and CUDA. The referenced per-run JSON and logs were not supplied; per-case predictions, scores, device placement, checkpoint/executable hashes, and native consistency were not independently verified. Inference was not rerun. This is synthetic fixture evidence, not general model quality.

[Summary JSON](../../benchmarks/decision-rules-windows-20260926/summary.json) · [Aggregate audit](../../benchmarks/decision-rules-windows-20260926/audit.json) · [Source and hardware](../../benchmarks/decision-rules-windows-20260926/provenance.json)

```sh
node web/scripts/verify-decision-rules.mjs
```


<a id="recorded-apple-m5-max-results"></a>
## Recorded macOS / Apple M5 Max results

Recorded at **2026-09-27 04:33:25 UTC** (13:33:25 KST) on an **Apple M5 Max** with 128 GB unified memory, macOS 27.0, in High Power mode on AC power. Settings match the Windows run above: `legacy`, `fresh`, context 2048, batch 256, 4 threads, thresholds 0.8 / 0.05, 3 passes, 1 warmup, timeout 1800 seconds. Metal runs use `--device cpu metal` from this repository revision. Gemma 4 26B-A4B Q4_K_M ran as a sixth checkpoint through a custom manifest.

| Model | Device | Status | Coverage | Accepted accuracy | Correct / all | Raw top-1 | p50 ms/request | p95 ms/request | Decisions/s | Correct accepted/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smollm2 | cpu | ok | 8.3% | 33.3% | 2.8% | 38.9% | 153.3 | 167.7 | 19.58 | 0.54 |
| smollm2 | metal | ok | 13.9% | 40.0% | 5.6% | 38.9% | 21.8 | 23.2 | 137.29 | 7.63 |
| qwen3 | cpu | ok | 83.3% | 33.3% | 27.8% | 41.7% | 474.3 | 522.1 | 6.26 | 1.74 |
| qwen3 | metal | ok | 83.3% | 33.3% | 27.8% | 36.1% | 42.9 | 43.7 | 69.99 | 19.44 |
| gemma3 | cpu | ok | 91.7% | 54.5% | 50.0% | 50.0% | 610.4 | 679.0 | 4.83 | 2.42 |
| gemma3 | metal | ok | 88.9% | 53.1% | 47.2% | 50.0% | 55.6 | 60.7 | 52.82 | 24.94 |
| tinyllama | cpu | ok | 0.0% | n/a | 0.0% | 38.9% | 1399.0 | 1556.7 | 2.13 | 0.00 |
| tinyllama | metal | ok | 0.0% | n/a | 0.0% | 38.9% | 59.6 | 60.7 | 50.23 | 0.00 |
| gemma4 | cpu | ok | 94.4% | 100.0% | 94.4% | 97.2% | 1784.4 | 2377.2 | 1.65 | 1.56 |
| gemma4 | metal | ok | 94.4% | 97.1% | 91.7% | 94.4% | 203.8 | 246.4 | 14.68 | 13.46 |
| gemma4-26b-a4b | cpu | ok | 100.0% | 100.0% | 100.0% | 100.0% | 5837.7 | 6489.0 | 0.52 | 0.52 |
| gemma4-26b-a4b | metal | ok | 100.0% | 100.0% | 100.0% | 100.0% | 506.0 | 647.7 | 5.75 | 5.75 |

Gemma 4 26B-A4B answers all 36 decisions correctly in every pass on both devices (Metal p50 506.0 ms). Gemma 4 E2B accepts 34/36 per pass on both devices: CPU gets all 34 correct, Metal 33 correct and 1 wrong. Every run reports 0 changes in 72 repeat comparisons; CPU and Metal still differ slightly for several checkpoints. Every Metal report records `backend.offload_device: "Apple M5 Max"`. Per-run reports and logs are kept locally and not published. This is synthetic fixture evidence, not general model quality.

[Results and limits](../../benchmarks/decision-rules-macos-m5max-20260927/README.md) · [Summary JSON](../../benchmarks/decision-rules-macos-m5max-20260927/summary.json) · [Source and hardware](../../benchmarks/decision-rules-macos-m5max-20260927/provenance.json)


<a id="run-the-five-model-matrix"></a>
## Run the five-model matrix

Prerequisites: Rust dependencies already cached, CMake, a C++17 compiler, and locally acquired checkpoints. The bundled sys dependency builds the pinned llama.cpp source; CUDA runs additionally need the CUDA toolkit. Model paths are listed in `tests/fixtures/benchmark_models.json`; no Hugging Face client, account, or network request is used by the runner. Review the separate licenses before obtaining a model.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools benchmark-models \
  --device cpu cuda \
  --iterations 3 \
  --warmup 1
```

The runner builds a release test executable once with `--locked --offline` and the CUDA feature when requested, hashes each model, and runs one model/device at a time. CPU is the default device. CUDA is explicit and cannot silently fall back to CPU. On macOS, `--device cpu metal` builds once with the `llama-metal` feature; CUDA and Metal cannot be combined in one run. Check `backend.offload_device` in each Metal report to confirm GPU placement. The native backend reuses one loaded model for the whole run; Rust process startup, Cargo compilation, loading, and warmup are excluded from steady-state timings.

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

<a id="reports-and-metric-definitions"></a>
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

<a id="test-the-benchmark-without-models"></a>
## Test the benchmark without models

```sh
cargo test --locked --offline --test benchmark
cargo test --locked --offline -p l2s1-tools
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

Direct runs accept `SKID_DEVICE` (`cpu`, `cuda`, or `metal`; build with the matching feature), `SKID_CONTEXT`, `SKID_BATCH`, `SKID_THREADS`, `SKID_MIN_TOP_PROBABILITY`, and `SKID_MIN_CANDIDATE_MASS`. The Rust runner adds checkpoint hashes and the comparison table. The earlier `tests/performance.rs` remains available for repeated measurements of the single warehouse example.

## Comparing execution and cache settings

`benchmark-models` accepts `--device metal` on macOS (builds `llama-metal`), and `--execution-mode fresh|prefix-reuse|parallel|state-restore`. CUDA and Metal require separate builds. `--parallel-width` is 1–32 (default 3); increasing it increases KV capacity and memory pressure. Parallel and state-restore are opt-in measurements and can be slower than fresh.

`--evidence-transfer full|compact` defaults to full; compact text evidence requires fresh or prefix-reuse. It keeps the full-vocabulary probability normalizer but returns compact candidate evidence. `--preparation-cache-bytes 8388608` enables bounded preparation caching (default 0, disabled; at most 128 entries per cache). Per-run reports retain cache counters before/after the measured passes; warmup can populate them. Preparation hits avoid rendering/tokenization, not transformer inference, and repeating the same fixture can overstate gains for new states. Check actual `reused_prefix_tokens` separately.

For the direct ignored test, the corresponding environment variables are `SKID_DEVICE`, `SKID_EXECUTION_MODE`, `SKID_PARALLEL_WIDTH`, `SKID_EVIDENCE_TRANSFER` and `SKID_PREPARATION_CACHE_BYTES`. `SKID_CUDA` remains a legacy fallback when `SKID_DEVICE` is absent. Ordinary requests remain isolated even with prefix-reuse; use the separate [fixed-schema session experiment](SEMIF_ALGORITHM.md#fixed-schema-sessions) to measure reuse across changing states. Its grouped timings are not the original three-decision request p50.
