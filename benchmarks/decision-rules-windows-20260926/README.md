# Windows / RTX 5090 decision rules benchmark

Recorded at **2026-09-26 16:21:17 UTC** (2026-09-27 01:21:17 KST). Suite: `decision-rules-v1`. GPU: **NVIDIA GeForce RTX 5090**, identified by the user after supplying the summary; no device log was supplied. Platform: Windows x86_64. CPU model and GPU memory usage are unavailable.

## Measured results

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

The five exact checkpoint filenames, hashes, original latency samples, executable hash, revisions, settings, and aggregate counts are in [summary.json](summary.json). Hardware/source details are in [provenance.json](provenance.json); the aggregate audit is in [audit.json](audit.json).

## How to use these results

- Gemma 4 E2B Q8_0 on RTX 5090 is the strongest candidate **within these 36 synthetic decisions**: each unchanged pass has 33 correct accepted answers, 2 wrong accepted answers, and 1 abstention. Raw top-1 is 34/36. Accepted accuracy is 33/35 (94.3%), while correct/all is 33/36 (91.7%). Its 67.6 ms p50 measures a request with three decisions; 43.90 decisions/s includes abstentions and corresponds to 14.63 requests/s. Correct accepted throughput is 40.24/s.
- Gemma 3 CUDA accepts 33/36 decisions and gets 19/33 accepted answers correct. Qwen3 CUDA accepts 30/36 and gets 10/30 correct. SmolLM2 CUDA accepts 4/36 and gets 1/4 correct. These measurements do not support choosing them for autonomous execution of these rules under the recorded policy.
- TinyLlama abstains on all 36 decisions in every pass on both devices. Its high throughput produces zero correct accepted answers; accepted accuracy stays `n/a`.
- Gemma4 CPU exceeded the 1,800 second per-run timeout. No completed CPU quality or latency measurement exists for that checkpoint.
- Start a separate evaluation on representative application data before selecting a model or changing thresholds. Do not tune on these labels and then describe the result as held-out accuracy.

## Arithmetic and evidence checks

The supplied Markdown table agrees with all 10 summary rows. The audit recomputes 9 completed runs from 324 request timing samples: count totals, all accuracy fractions, nearest-rank p50/p95, min/max/mean, and both throughput metrics. Each completed run contains 12 requests × 3 passes = 36 timed requests and 108 measured decisions. There are only **12 independent requests / 36 distinct labeled decisions** per model/device.

All completed runs report 0 changed outputs across 72 repeat comparisons. This supports the per-pass counts above, but neither proves probability equivalence nor passes the native consistency suite. CPU and CUDA summaries differ: Qwen3 raw top-1 is 12/36 versus 15/36, Gemma3 is 18/36 versus 22/36, and SmolLM2 accepts 5/36 versus 4/36. Per-case predictions and score differences cannot be checked because the referenced per-run JSON and logs were not supplied.

The recorded fixture SHA256 `cfe9219e45cfb66435c68945daf92f079693b8524b3da654c879f68f7ce53fe8` equals the current fixture after converting LF to Windows CRLF. The LF hash is `f9d5380fdafe1bb686c9f4f1d44003a9f71c6f47fddd59063c62d3301b7d6c04`. This verifies fixture identity up to line endings; it does not verify the historical binary or checkpoint contents. Their hashes and revisions are retained as supplied. The recorded repository revision was located while preparing the PR; the historical executable and per-run files were not supplied.

Verification uses Node.js built-ins and needs no model or GPU:

```sh
node web/scripts/verify-decision-rules.mjs
```

## Reproduction settings

Sequential model/device runs; `legacy` prompt layout; `fresh` execution; context 2048; batch 256; 4 threads; 3 measured passes; 1 excluded warmup pass; `min_top_probability=0.8`; `min_candidate_mass=0.05`; timeout 1800 seconds. Model loading and warmups are excluded from request timings. No concurrent serving, memory, or cold-start benchmark is claimed.

For a new run on the current source, follow [the runner instructions](../../docs/BENCHMARK.md). Preserve this historical summary and use a new output directory:

```sh
target/release/l2s1-tools benchmark-models \
  --device cpu cuda --iterations 3 --warmup 1 \
  --prompt-layout legacy --execution-mode fresh \
  --context 2048 --batch 256 --threads 4 \
  --min-top-probability 0.8 --min-candidate-mass 0.05 \
  --timeout 1800 --output results/benchmark/new-decision-rules-run
```

This import audits supplied aggregates; inference was not rerun. Status `ok` means execution completed. All results concern a small synthetic rule suite, not general model quality or production approval. No model weights, local model paths, or unavailable per-run files are exported.
