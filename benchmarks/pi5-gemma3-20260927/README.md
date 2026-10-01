# Raspberry Pi 5 4GB: Gemma3 1B Q8 CPU run

Completed on the physical `hilbert-no-1` Pi, using source `7b071fd9c8c5c06e9a567c59ac587ee74166be64`
and llama.cpp `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. Model SHA-256:
`b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`. This is the same Q8 model file used in the supplied
Windows benchmark, but a different source revision, OS, architecture and power condition.
No LoRA adapter is loaded; this is not a fine-tuning result.

| Metric | Measured value |
| --- | ---: |
| Coverage | 91.7% |
| Accepted accuracy | 54.5% |
| Correct accepted / all | 50.0% |
| Raw top-1 | 55.6% |
| p50 / request (3 decisions) | 4.766 s |
| p95 / request | 5.197 s |
| Decisions / second (including abstentions) | 0.620 |
| Correct accepted / second | 0.310 |
| Peak process RSS, including loading/warmup | 2.100 GiB |
| Maximum system swap used | 0.0 MiB |
| Maximum sampled temperature | 72.15 C |

`decision-rules-v1`: 12 unique cases / 36 decisions,
repeated 3 times. One first request and one full warmup pass are excluded
from latency. Threads 4, context 2048, batch 256,
legacy prompt layout and fresh execution. Coverage and accuracy include repeats;
repeats are not additional independent evidence. The benchmark completed; this does
not imply the full native consistency suite passed.

Power telemetry sampled once per second, including load/warmup: active undervoltage
in 128 of 243 samples; active throttling
in 128 samples. Historical bits were already set
before the run. Brief events between samples can be missed. Inspect power supply
and cable before treating this as an unthrottled hardware benchmark.

All labels/predictions/scores and repeat checks: `runs/gemma3-q8-rules/benchmark.json`.
Command/settings: `runs/gemma3-q8-rules/command.json`. Peak RSS uses Linux wait4
resource usage for the benchmark process. Full sampled telemetry and exit status
are in that directory. Model and adapter weights are excluded from this evidence.

The Pi installation remains at `/home/lutica/l2s1-pi-20260927`.

## Jev output verification and reporting fix

The installed package ran on the Pi's Python 3.13.5. Twelve package tests passed
(`package-tests-fixed.log`). An additional native request containing Choice,
Noul and Score produced all three output types in `jev-demo-report/jev-answers.jsonl`.
This tiny format example matched **1 of 3 labels**; the Choice and Score answers
were wrong despite high confidence. It is not an accuracy improvement claim.
The native request took 3.277 seconds after one warmup. This sample is separate
from the 36-decision rules benchmark above.

The live run exposed a reporter bug: custom Score labels with probabilities but
no redundant `gold.score` field caused a failed report. The fix derives the target
expectation from the supplied distribution when that optional field is absent;
explicit historical targets remain unchanged. Reporting now exits nonzero for
failed cases while retaining failure records and denominators. The fixed report
reuses the original native predictions; native inference was not rerun or altered.
Original failed-report evidence remains in the local/remote results directory.

`reporter-identity.json` binds the corrected report to the patched reporter file.
The native binary and rules results still refer to the exact earlier commit above,
before the subsequent native bridge refactor on main.

## Reproduction

The source archive was `git archive` of the pinned repository commit. `setup.sh`
installed CMake 4.2.3 and Rust 1.98.1 into the dedicated user directory and built
the CPU binaries with two build jobs using GCC 14.2.0. No system services were
changed. The model was copied from the existing local file and hash-checked.
`run_measurement.py` contains the exact three-pass benchmark and monitoring setup;
`runs/gemma3-q8-rules/command.json` records its resolved command/environment.
The setup script's `stable` toolchain follows the installer channel; pin 1.98.1
when reproducing the original compiler. Rust, native source, shared libraries and
binaries must remain matched. `run_jev_demo.sh` describes the separate example;
rerun into new output paths because reporting never overwrites existing results.
