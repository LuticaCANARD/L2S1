# Pi 5 state-cache validation and state-restore overhead

This follows the [portable ARM64 fresh investigation](../pi5-arm64-fresh-20260927/README.md).
Both sides already use the optimized runtime-dispatched ARM kernels, with
OpenMP off and `GGML_NATIVE=OFF`. The baseline is PR #77 commit
`4f6c02fa84cfc192a4f7a49e3a5e14e5eef8c649`; the candidate adds the two Rust
state-restore changes in this PR. Pinned llama.cpp remains
`3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.

## Measured results

| Mode | Short p50 / p95 (n=12) | Long-state p50 / p95 (n=4) | Block peak RSS |
| --- | ---: | ---: | ---: |
| Fresh | 4.458 / 4.939 s | 14.414 / 14.604 s | 2.118 GiB |
| Prefix reuse | 4.504 / 4.893 s | 8.978 / 9.043 s | 2.118 GiB |
| State restore before | 4.489 / 4.966 s | 9.184 / 9.198 s | 2.126 GiB |
| State restore after | 4.482 / 4.901 s | 9.053 / 9.159 s | 2.126 GiB |

The long-state request reuses **512 / 1,306 input tokens** (256 tokens for each of two later decisions). Snapshot size is **6.506 MiB**. State restoration after the fix is **1.59x** as fast as matched fresh execution in this study. Prefix reuse has a similar request time and avoids snapshot copying; these measurements do not establish a state-restore advantage over prefix reuse for this dense Gemma model. Short requests reuse zero tokens at batch 256 and have no cache speedup.

| State-restore work per three-decision request | Before | After |
| --- | ---: | ---: |
| Median restore time (two restores, long state) | 18.138 ms | 5.753 ms |
| Median save time (long state) | 3.829 ms | 3.399 ms |
| Zero-budget request p50 / p95 (n=4) | 17.382 / 17.440 s | 14.546 / 14.570 s |
| Redundant prefix prefill with zero budget, median | 2.694 s | 0.000 s |

The measured restore stage is 68.3% shorter. Normal long-state request p50 is only 1.4% lower; with four samples and uncontrolled power, the whole-request difference cannot be assigned precisely to a 12.4 ms copy-cost reduction. The structural zero-budget improvement removes an entire unused prefix prefill: request p50 falls 16.3%, with exact fresh fallback results and zero reused tokens. Nonzero insufficient budgets still need prefill to determine the actual snapshot size.

The `zero-budget` input is the same long request with a snapshot limit of zero. That option controls state restoration only: it does not disable `prefix-reuse`, so prefix timings in that group are not budget-fallback results.

## Full-suite quality and numerical equivalence

| Mode, 36 unique decisions | Raw top-1 | Coverage | Accepted-only accuracy | Correct accepted / all |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 19/36 (52.8%) | 36/36 (100.0%) | 19/36 (52.8%) | 52.8% |
| Prefix reuse | 19/36 (52.8%) | 36/36 (100.0%) | 19/36 (52.8%) | 52.8% |
| State restore before | 19/36 (52.8%) | 36/36 (100.0%) | 19/36 (52.8%) | 52.8% |
| State restore after | 19/36 (52.8%) | 36/36 (100.0%) | 19/36 (52.8%) | 52.8% |

All saved same-input comparisons (full suite, long state, budget fallback, repeated rounds and explicit sessions) have zero changed top-1 labels, acceptance statuses and values. Maximum probability delta: **0**; candidate-mass delta: **0**; raw-logit delta: **0**. The regression threshold remains 0.02 with unchanged label/acceptance checks. The whole short suite has no aligned shared batch; the long-state measurements and native regression independently exercise real restoration.

These quality rows use state-first prompts. They are not the earlier legacy-layout scores, and caching does not improve model accuracy or make the high coverage reliable.

## Explicit shared-state session

| State, three changing questions | Fresh p50 / p95 | Shared-session p50 / p95 | Reused tokens per request |
| --- | ---: | ---: | ---: |
| warehouse-01 | 4.408 / 4.520 s | 4.475 / 4.488 s | 0 |
| warehouse-01-long-state | 14.496 / 14.861 s | 8.989 / 9.139 s | 512 |

Four measured runs per row/mode, with alternating order. Fresh evaluates a three-decision request; the shared path makes three one-question Rust calls within one immutable-state session. The session is created and dropped inside the measured outer call. Same-input numerical comparisons are exact, including the changed question instructions. This does not add automatic cross-request caching to the CLI or SDK.

## Power and temperature limits

| Mode | Active undervoltage / samples | Active throttling / samples | Temperature range | ARM clock range | Max system swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh | 82/166 | 82/166 | 65.55–72.70 C | 1.000–2.400 GHz | 10.0 MiB |
| Prefix reuse | 67/123 | 67/123 | 67.20–72.15 C | 1.000–2.400 GHz | 10.0 MiB |
| State restore before | 79/156 | 79/156 | 66.10–72.70 C | 1.000–2.400 GHz | 10.0 MiB |
| State restore after | 77/142 | 77/142 | 65.55–72.70 C | 1.000–2.400 GHz | 10.0 MiB |

Active undervoltage/throttling remained present. Balanced order, common software settings and warmup help compare this setup, but these are not normal-power performance guarantees. Short telemetry events may be missed.

## What changed

1. A zero snapshot budget now enters fresh fallback before computing the common
   prefix. Previously that prefix was computed and discarded, then recomputed
   by fresh evaluation. The diagnostic still reports `snapshot_memory_budget`,
   but `prefill_ms`, `save_ms`, snapshot bytes and restores are zero.
2. Dense KV sequence restoration no longer clears the entire KV buffer before
   each restore. In the pinned upstream `llama-kv-cache.cpp`, single-sequence
   `state_read_meta` removes the destination sequence and restores its cells.
   The extra buffer memset was redundant. The existing clear is retained for
   recurrent/hybrid models and `L2S1_FORCE_KV_CLEAR=1`. Request boundaries and
   error cleanup still clear native state. No batch boundary or scoring policy
   changes.

These changes do not introduce an automatic cache across unrelated requests.
Three mechanisms are deliberately distinguished:

| Mechanism | Lifetime | Work reused |
| --- | --- | --- |
| `prefix-reuse` | One ordinary request | Complete batches in an exact token prefix, using KV suffix removal |
| `state-restore` | One ordinary request | One common prefix snapshot restored before independent suffixes |
| Explicit `shared_state` session | Several Rust calls borrowing one immutable state | Exact prefix KV across changing questions; cleared on creation, error and drop |

Resident fixed-schema caching from PR #76 is a separate split plan. Its earlier
timings and fresh-versus-fixed numerical differences are not substituted for
this experiment. No result/answer cache is used.

## Method and measurement boundary

Physical Pi 5 4GB; Gemma 3 1B Q8_0, SHA-256
`b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`;
context 2048, batch/ubatch 256, four CPU threads, CPU, full evidence, identical
`state-first` layout and unchanged policy. This layout differs from the legacy
layout in the earlier fresh benchmark. Compare modes within this study; do not
multiply its ratios by the earlier kernel improvement or claim identical
predictions across prompt layouts.

The resident `examples/state_cache_probe.rs` harness exposes existing Rust
diagnostics over the same newline JSON transport for all ordinary modes. It
records snapshot bytes and save/restore/prefill/suffix timings, which ordinary
wire responses do not expose. Time includes the diagnostic preflight, native
evaluation and response serialization, but excludes model loading. The model
path is identical; matching native libraries are loaded from the installed npm
packages, with executable/library hashes and loaded paths retained per block.
`measured-probe.rs` preserves the exact compiled harness source; the example
has the same behavior with rustfmt formatting. The diagnostic probe is a
separately built example executable, not the shipped CLI; `installed-smoke.py` additionally exercises the installed production CLI.

Four configurations (fresh, prefix, old restore, new restore) run in four
balanced Latin-square rounds: 0/1/3/2, 1/2/0/3, 2/3/1/0, 3/0/2/1. Each block
idles ten seconds, warms warehouse-01 and the long-state request once each,
then measures warehouse-01/02/03, the long state, and its zero-budget version.
The long state adds 300 `packing` words to an irrelevant field in warehouse-01;
its three decisions share one complete 256-token batch. There are 12 short
requests, four long requests and four zero-budget requests per configuration.
p95 is nearest-rank; especially at n=4 it is just a small-sample maximum.

The quality phase checks all 36 unique fixture decisions separately. Repeated
requests do not increase quality sample counts. A separate four-round paired
study compares a fresh three-decision request with three one-question Rust
calls inside an explicit shared-state session; the outer diagnostic transport
and input prompts are the same, while the Rust API call boundaries differ.
The fixture questions have different instructions. This measures an explicit
in-process state lifetime, not a network cache, eviction policy or concurrency.

Telemetry records temperature, ARM clock, active/historical undervoltage and
throttling, RSS/high-water RSS and system swap once per second. Peak RSS includes
loading and warmup, and is a whole-block maximum, not attribution of memory to
an individual short/long request. Power and temperature are observed, not
controlled. Short power events can be missed. No compilation overlaps the
performance measurements. The exploratory `smoke-*` blocks used the prior
60 C / 120-second cooldown and no warmup; they are excluded from the timed
comparison and quality tables.

## Evidence and reproduction

Validation completed on the physical Pi:

- The installed production CLI passed explicit fresh, state-restore, zero-budget
  fallback and legacy-layout smoke checks, including cold/warm evidence equality
  and recovery after an oversized request. The reused-token counts are
  respectively 0, 512, 0 and 0; state-first evidence exactly matches fresh.
- The new real-model state-restore regression and existing fixed-schema
  regression passed. State restoration also passed with
  `L2S1_FORCE_KV_CLEAR=1`. The new regression covers differing suffix lengths,
  duplicates, zero/insufficient budgets, changed state, errors and single-decision
  fallback, without weakening the existing 0.02 numerical threshold.
- Both installed manifests were verified. All native shared libraries are
  byte-identical before/after; `identity.json` records the changed Rust source
  hashes and probe/package identities. The source hashes match this checkout.

Local x86 validation also passed the real-model regression with SmolLM2 and
Gemma 3, library tests (37 passed, one ignored), CLI tests (two passed), sys-crate
tests (three passed), strict Clippy, formatting and affected documentation links.
The raw test logs are retained. `validate.py` checks measurement counts, actual
reuse, fallback behavior, score comparisons, source identity and installed smoke
results; it does not rerun inference.

Build the baseline diagnostic example at `4f6c02f` with the same probe source
added, `L2S1_PORTABLE_BUILD=1 L2S1_ARM64_DISPATCH=1 L2S1_OPENMP=0`, and retain it
with the previous installed kernel package. `build-after.sh` builds, packages,
npm-installs and verifies the candidate and matching probe. The scripts preserve
the concrete isolated Pi paths used for this study. The packages retain version
0.1.3 locally; no existing installation is replaced and no release is published.

`run.sh` executes the balanced comparison, full quality suite, explicit session
comparison, and then native regression tests. `measure.py` reuses the unchanged
sensor implementation from the earlier study (the Pi task root's `measure.py`).
`installed-smoke.py` runs after all measurements. Regenerate the report with
`python3 report.py .`, validate its invariants with `python3 validate.py`, and
check captured files with `sha256sum -c SHA256SUMS`.

No weights, binaries, dependency archives or credentials are committed. This is
physical Pi CPU evidence; it does not validate the changed dense restore path
on CUDA or Metal, or add new recurrent/hybrid hardware validation.
