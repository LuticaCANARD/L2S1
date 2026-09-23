# Decision benchmark validation results

Legacy v1-prompt results. The optional state-first v2 algorithm has separate measurements in [SEMIF_ALGORITHM.md](SEMIF_ALGORITHM.md); these numbers are not a validation of the new prompt.

Date: 2026-09-21. Actual local inference using the five existing GGUF files on CPU and CUDA. These measurements use the new labeled suite, not the earlier single warehouse fixture.

- Hardware: Intel Core i9-9900K, four inference threads; NVIDIA RTX 3080, native GPU access on Linux/WSL.
- Release build; context 2,048; batch 256; minimum top probability 0.8; minimum candidate mass 0.05.
- Per model/device: one separately timed first request, one excluded full warmup pass, then two measured passes. This yields 24 measured requests / 72 decisions from 12 unique requests / 36 unique decisions.
- All ten matrix runs completed: 720 measured decisions total. Execution success does not mean the answers were correct.
- Models ran sequentially. Normal workstation activity, including lightweight source checks, was not isolated. These are smoke benchmark measurements, not stable hardware rankings or cold-cache loading tests.

Suite: `decision-rules-v1`. Sequential runs; load and warmups excluded from request timings.

| Model | Device | Status | Coverage | Accepted accuracy | Correct / all | Raw top-1 | p50 ms/request | p95 ms/request | Decisions/s | Correct accepted/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smollm2 | cuda | ok | 22.2% | 37.5% | 8.3% | 38.9% | 57.4 | 67.4 | 52.89 | 4.41 |
| smollm2 | cpu | ok | 19.4% | 42.9% | 8.3% | 38.9% | 697.1 | 1624.2 | 3.55 | 0.30 |
| qwen3 | cuda | ok | 83.3% | 33.3% | 27.8% | 38.9% | 57.5 | 69.9 | 51.08 | 14.19 |
| qwen3 | cpu | ok | 83.3% | 33.3% | 27.8% | 36.1% | 2105.2 | 2339.5 | 1.39 | 0.39 |
| gemma3 | cuda | ok | 88.9% | 53.1% | 47.2% | 52.8% | 65.7 | 84.5 | 44.60 | 21.06 |
| gemma3 | cpu | ok | 80.6% | 55.2% | 44.4% | 52.8% | 2767.0 | 3338.0 | 1.05 | 0.47 |
| tinyllama | cuda | ok | 0.0% | n/a | 0.0% | 38.9% | 50.1 | 52.8 | 59.52 | 0.00 |
| tinyllama | cpu | ok | 0.0% | n/a | 0.0% | 38.9% | 3534.3 | 4597.0 | 0.80 | 0.00 |
| gemma4 | cuda | ok | 94.4% | 97.1% | 91.7% | 97.2% | 98.5 | 110.1 | 30.17 | 27.66 |
| gemma4 | cpu | ok | 94.4% | 100.0% | 94.4% | 97.2% | 7475.0 | 8580.2 | 0.40 | 0.38 |

Each request contains three decisions. Accuracy denominators include repeated passes; repeats are not independent examples. Accepted accuracy is n/a when every answer abstains. Raw top-1 ignores policy abstention, with ties counted incorrect. Decisions/s includes abstentions. These synthetic rule tasks do not establish general model quality. Status 'ok' means execution completed, not that every answer was correct or the native consistency suite passed.

Per-run JSON includes all labels, predictions, scores, group metrics, timings, and repeat-consistency counts. Failures and timeouts remain in summary.json and per-run logs. No model weights are copied.

Repeat consistency: 0 changed selected/top-1 outputs across 360 comparisons with first-pass outputs. This says nothing about probability equality or accuracy.

The previously recorded Gemma 3 and TinyLlama CUDA batch-consistency failures remain unresolved; this benchmark uses a fixed batch of 256. See [VERIFICATION.md](VERIFICATION.md).

Checkpoint SHA256 values and individual output records are in the excluded local `results/benchmark/decision-rules-v1-validation/` directory. Model sources and file specifications are in [MODEL_SPECS.md](MODEL_SPECS.md). Run commands and metric definitions are in [BENCHMARK.md](BENCHMARK.md). No model weights are distributed.

Fixture SHA256: `f9d5380fdafe1bb686c9f4f1d44003a9f71c6f47fddd59063c62d3301b7d6c04`. llama.cpp revision: `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.

Validation: 14 ordinary Rust tests and 3 Python runner tests passed; formatting and Clippy with `-D warnings` passed. The runner also retained both missing-file and invalid-GGUF failures and returned exit status 1 in a separate negative run.
