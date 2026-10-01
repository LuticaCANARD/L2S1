# macOS / Apple M5 Max decision rules benchmark

Recorded at **2026-09-27 04:33:25 UTC** (13:33:25 KST); the run finished at 04:43:23 UTC. Suite: `decision-rules-v1`. Hardware: **Apple M5 Max**, 128 GB unified memory, macOS 27.0 (26A428). Power: High Power mode (`pmset` `powermode 2`) on AC power, captured before and after the run.

## Measured results

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

Each completed run has 12 distinct requests / 36 labeled decisions, repeated 3 times: 36 timed requests / 108 measured decisions. Repeats are not independent accuracy examples. Loading and warmups are excluded. Settings: `legacy`, `fresh`, context 2048, batch 256, 4 threads, thresholds 0.8 / 0.05, timeout 1800 seconds. These match the [Windows / RTX 5090 record](../decision-rules-windows-20260926/README.md).

The exact checkpoint filenames, hashes, latency samples, executable hash, revisions, settings, and aggregate counts are in [summary.json](summary.json). Hardware, power, and model sources are in [provenance.json](provenance.json). The five reference checkpoints and `gemma4-26b-a4b` ran as two invocations of the same executable; `summary.json` lists both under `sources`.

## How to use these results

- Gemma 4 26B-A4B Q4_K_M accepts and answers all 36 decisions correctly in every pass on both devices. Its Metal p50 is 506.0 ms per three-decision request.
- Gemma 4 E2B Q8_0 accepts 34 of 36 decisions per pass on both devices. CPU gets all 34 correct; Metal gets 33 correct and 1 wrong. Its Metal p50 is 203.8 ms.
- CPU and Metal differ slightly for some checkpoints: Qwen3 raw top-1, Gemma 3 coverage, SmolLM2 coverage, and Gemma 4 E2B accepted accuracy. Every run reports 0 changes in 72 repeat comparisons, so the differences are between devices, not between passes.
- TinyLlama abstains on all 36 decisions in every pass on both devices; accepted accuracy stays `n/a`.
- Start a separate evaluation on representative application data before selecting a model or changing thresholds. Do not tune on these labels and then describe the result as held-out accuracy.

## Evidence and limits

- Every Metal per-run report records `backend.offload_device: "Apple M5 Max"`; every CPU report records `offload_requested: false`.
- Checkpoint SHA256 values in `summary.json` match the local files. The fixture SHA256 is the LF variant, equal to `fixture_lf_sha256` in the [Windows audit](../decision-rules-windows-20260926/audit.json).
- Two rows (Gemma 4 E2B Metal and 26B CPU) were recomputed from the per-run latency samples and counts: nearest-rank p50 and coverage match.
- Per-run JSON reports and native logs stay local and are not published, because they contain local paths. The native consistency suite was not run.
- An earlier run on the same day with the same settings produced identical quality metrics but different latency; its power mode was not recorded, so it is not published and no power-mode effect is claimed.
- Process RSS, unified memory use, and thermal state are not measured. This is synthetic fixture evidence, not general model quality.
