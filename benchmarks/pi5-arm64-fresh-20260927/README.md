# Pi 5 portable ARM64 fresh inference investigation

This study rebuilds the current L2S1 source four ways and measures actual npm
runtime packages installed on the same physical Pi. It does not publish a new
release. The candidate packages retain the source version `0.1.3` locally; they
are not the already published v0.1.3 artifacts.

Base source: `46851481926b388ee1b7f5dd5439c7413dfdbc9c` (main after PR #76),
plus this PR's native build/loader changes. All four builds use identical source;
only `L2S1_ARM64_DISPATCH` and `L2S1_OPENMP` differ. Upstream llama.cpp remains
`3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.

| Build | `L2S1_ARM64_DISPATCH` | `L2S1_OPENMP` | Native CPU tuning |
| --- | ---: | ---: | --- |
| baseline | 0 | 0 | Off |
| kernels (proposed portable default) | 1 | 0 | Off; runtime ISA dispatch |
| openmp | 0 | 1 | Off |
| combined | 1 | 1 | Off; runtime ISA dispatch |

All four set `L2S1_PORTABLE_BUILD=1`. No build sets `GGML_NATIVE=ON`.
The dispatch builds use upstream `GGML_BACKEND_DL` / `GGML_CPU_ALL_VARIANTS`.
Pi selects `armv8.2_2` (dotprod + FP16 vector arithmetic); unsupported
SVE/i8mm/SME variants score zero. The baseline only reports NEON/ARM_FMA, without
dotprod or FP16 vector arithmetic. These are GGML matrix kernels. The Rust SIMD
logit scan is unchanged.

In the pinned upstream `ggml-cpu/repack.cpp`, Q8_0 tensors with suitable shapes
use the 4x4 repacked kernel when NEON and dotprod are available; without dotprod
or i8mm this branch returns no Q8 repack type. The specialized buffer allocates
storage and repacks the weights. Thus this kernel factor includes dotprod,
FP16 vector arithmetic and the corresponding repack path together. It does not
isolate their individual contributions. That distinction matters for both
memory and numerical comparisons.

## Measured result

| Current source, fresh only | p50 / request | p95 / request | Peak RSS | Active undervoltage / measured samples |
| --- | ---: | ---: | ---: | ---: |
| baseline | 15.643 s | 18.051 s | 1.123 GiB | 83 / 190 |
| kernels | 4.450 s | 4.948 s | 2.113 GiB | 30 / 53 |
| openmp | 15.685 s | 17.869 s | 1.124 GiB | 83 / 190 |
| combined | 4.459 s | 4.902 s | 2.113 GiB | 26 / 53 |

With OpenMP off in both builds, runtime-selected ARM kernels reduced fresh p50 by **71.6% (3.52x)** in this experiment. All measured fresh responses report zero reused prefix tokens. OpenMP alone did not approach the kernel improvement; see per-round medians and ratios in `summary.json`. The default therefore enables ARM dispatch and keeps OpenMP off, with no new OpenMP runtime dependency.

Active undervoltage/throttling occurred during this experiment. Software settings, inputs, warmup and transport are matched and order is balanced, but the power supply was not controlled. These results support the missing ARM kernel path as a major contributor under this setup; they do not quantify its sole share of the historical 3.56x slowdown or establish normal-power Pi performance.

| Build | Temperature range | ARM clock range | Active throttling / samples | Maximum swap |
| --- | ---: | ---: | ---: | ---: |
| baseline | 64.45–69.40 C | 1.000–2.400 GHz | 83 / 190 | 10.7 MiB |
| kernels | 64.45–69.95 C | 1.000–2.400 GHz | 30 / 53 | 10.7 MiB |
| openmp | 63.90–68.85 C | 1.000–2.400 GHz | 83 / 190 | 10.7 MiB |
| combined | 65.55–69.40 C | 1.000–2.400 GHz | 26 / 53 | 10.7 MiB |

## Quality and numerical changes

Each row below uses all **36 unique decisions**, separately from repeated timing samples.

| Fresh build | Raw top-1 | Coverage | Accepted-only accuracy | Correct accepted / all |
| --- | ---: | ---: | ---: | ---: |
| baseline | 19/36 (52.8%) | 31/36 (86.1%) | 17/31 (54.8%) | 47.2% |
| kernels | 20/36 (55.6%) | 33/36 (91.7%) | 18/33 (54.5%) | 50.0% |
| openmp | 19/36 (52.8%) | 31/36 (86.1%) | 17/31 (54.8%) | 47.2% |
| combined | 20/36 (55.6%) | 33/36 (91.7%) | 18/33 (54.5%) | 50.0% |

Baseline versus kernels changes **1 raw top-1 label** and **4 acceptance statuses**. The maximum option-probability difference is **0.322919**; maximum candidate-mass difference is **0.000470**. Kernel changes are not numerical identity, and this is not a model accuracy improvement claim. On warehouse-02/dispatch_priority, the same wrong `low` top-1 rises from about 0.666 to 0.913 and is accepted at the unchanged 0.8 threshold.

The specialized Q8 repack path also increases measured peak RSS from about 1.12 to 2.11 GiB. This is a matched current-source observation; the historical whole-suite peak was not previously a matched memory comparison. Recheck confidence-based policies or calibration with the selected backend; runtime and loaded-library fingerprints change with the native implementation.

- `baseline-vs-openmp`: 36 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.
- `kernels-vs-combined`: 36 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.
- `kernels-fresh-vs-prefix`: 36 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.
- `kernels-fresh-vs-fixed`: 36 unique decisions; top-1 changes 3, status changes 1; max probability delta 0.81946608, mass delta 0.00049977276.
- `kernels-prefix-repeat`: 36 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.
- `kernels-fixed-repeat`: 36 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.
- `baseline-vs-masked-fallback`: 9 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.
- `historical-vs-kernels`: 36 unique decisions; top-1 changes 0, status changes 0; max probability delta 0, mass delta 0.

The optimized fresh run matches the earlier native build across all 36 raw top-1 labels, acceptance statuses, option probabilities, candidate masses and raw logits. Historical wire value objects have extra fields, so this is numerical/semantic equality, not byte-identical response envelopes.

| Kernel build execution plan, 36 unique decisions | Raw top-1 | Coverage | Accepted-only accuracy | Correct accepted / all |
| --- | ---: | ---: | ---: | ---: |
| fresh | 20/36 | 33/36 | 18/33 | 18/36 |
| request-local prefix | 20/36 | 33/36 | 18/33 | 18/36 |
| fixed-schema prefix | 19/36 | 32/36 | 18/32 | 18/36 |

The short batch-256 request-local prefix fixture reuses zero tokens, so equality there alone does not prove KV reuse. The separate batch-32 native test includes long-state and identical-prompt cases to exercise real reuse. Fixed-schema mode reuses 6,250 tokens over its two measured suite passes. Its different split plan changes scores relative to fresh (including three raw top-1 labels); the cold/warm regression checks consistency within that plan and passes with exactly equal repeated evidence. Fresh-versus-fixed numerical equivalence is **not** claimed. No probability tolerance is relaxed to hide differences.

## Method

Gemma 3 1B Q8_0, SHA-256
`b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`;
context 2048, batch/ubatch 256, four CPU threads, CPU device, legacy prompt,
identical direct stdio transport, default policy (top probability 0.8,
candidate mass 0.05). Every performance command explicitly passes
`--execution-mode fresh`, and every response must report zero reused tokens.

Each fresh process warms up warehouse-01/02/03 once, then measures those three
requests. Each request has three decisions. Four balanced Latin-square rounds
place each build in every position once and balance adjacent ordered pairs:

1. baseline, kernels, combined, openmp
2. kernels, openmp, baseline, combined
3. openmp, combined, kernels, baseline
4. combined, baseline, openmp, kernels

There are 12 measured requests per build. Loading/warmup is excluded. Before
each block, the runner waits for <=60 C or records a 120-second timeout. All
compilation finishes before the measurements start. Accuracy uses unique cases,
not repeated requests. Separate quality runs cover all 12 cases / 36 decisions.
The complete suite is also repeated in request-local prefix and fixed-schema
modes for the proposed kernel build.

Telemetry samples actual ARM clock, scaling frequency, temperature, process
RSS/high-water RSS, system swap, and active/historical undervoltage/throttling
bits approximately once per second. Short events can be missed. Latency is
end-to-end stdio time, including scoring and serialization. p95 uses the nearest
rank and is only a small-sample descriptive statistic, not a service objective.
Memory includes model loading and warmup for each process.

## Runtime identity and safety

`build-identity.json` records each installed manifest, compiler flags, upstream
source digest, dependency paths, instruction counts and runtime CPU scores.
Each process also saves `/proc/PID/maps` and hashes its actual llama/GGML/mtmd
libraries; the harness rejects paths outside that process's installed package.
The native build cache includes dispatch/OpenMP options, and each package comes
from one complete CMake install. Shared libraries from different builds are not
interchanged.

The ARMv8 fallback is always packaged. The upstream feature scorer is compiled
without variant ISA flags and with `-fno-lto`, before specialized code can run.
`mask-features.c` is a test-only HWCAP override used to exercise fallback on the
Pi. It is not an emulator and cannot establish execution on another physical
ARM CPU. Disassembly and runtime selection provide complementary evidence.

OpenMP builds are experimental comparison packages and resolve their GCC OpenMP
runtime through the Pi's system `libgomp.so.1`. The proposed portable default
keeps OpenMP off. Package verification checks every shared module with `ldd`,
including modules invisible to the executable's static dependency list. It also
uses an existing invalid GGUF to initialize and verify actual CPU dispatch;
`--version` or a nonexistent model alone would not exercise that boundary.

## Historical evidence

`historical-benchmark.json`, `historical-identity.json` and
`historical-command.json` preserve the earlier source-build evidence used here.
`historical-live-identity.json` rechecks its real executable hash, linked
libraries and CMake configuration, and verifies the published v0.1.3 archive
and runtime file hashes against the earlier manifest. The executable SHA matches
its recorded identity. The upstream source revision is unchanged; no alternative
llama.cpp revision is introduced by this fix.

The old matched subset was **4.743 s fresh**, released v0.1.3 was **16.874 s
fresh**, and released prefix reuse was **4.091 s**. The 4.125x ratio describes
released portable fresh versus released reuse only. Old and released runs had
different source, transport/warmup and uncontrolled power; neither their exact
3.56x gap nor normal-power hardware performance can be attributed to one build
option from that historical comparison alone.

`historical-binary-inspection.json` inspects the actual installed old and released libraries: the old native CPU module contains 1,184 `sdot` instructions and reports DOTPROD; the published v0.1.3 module contains zero and does not report DOTPROD. The candidate's selected `armv8.2_2` module contains 1,200; its ARMv8 fallback contains zero. Instruction counts describe these binaries, not executed instruction frequencies. `identity-checks.json` records the source/release hash checks; shared objects do not independently attest a Git SHA. `l2s1-source-identity.json` confirms the Pi's native build inputs match this PR.

## Completed validation and deployment scope

- All four candidate tarballs were built, npm-installed and verified on the physical Pi. The timing/quality runs used those installed packages, and loaded-library maps/hashes stayed within each package.
- Pi Rust library: 37 passed, one model-dependent test ignored by that command; CLI: two passed. The two explicitly invoked Gemma tests both passed: fixed-schema cold/warm at batch 256, and request-local reuse at batch 32. The latter compared 45 decisions with 6,272 reused tokens, zero changed top-1/selected decisions, and zero probability/mass deltas. Existing 0.02 tolerances were unchanged.
- Installed Python SDK 0.1.3 loaded the candidate runtime successfully. Explicit fresh reused zero tokens; omitted execution mode selected automatic fixed-schema reuse (322 warm reused tokens). Both modes reproduced cold/warm evidence exactly (`sdk-smoke.json`).
- Feature-masked fallback passed nine fresh decisions with exact baseline scores; missing fallback and a missing dependency visible only through a dynamic CPU module were correctly rejected by the package verifier.
- Local x86 regression also passed: core library 28, native library 37 plus one ignored, CLI two, release-pipeline tests seven, strict Clippy and real SmolLM2 fixed/prefix tests (`local-validation.json`).

These are local candidate packages built with Debian 13 / GCC 14.2, not artifacts downloaded from the changed GitHub Actions workflow. The proposed ARM64 workflow uses Ubuntu 24.04 / GCC 14 because the pinned upstream variant list includes ARMv9.2/SME. This raises the distribution/toolchain baseline relative to the old Ubuntu 22.04 recipe. The tested candidate requires GLIBC_2.38 and GLIBCXX_3.4.29; the official Ubuntu-built artifact's exact symbol floor still needs inspection after that workflow runs. CPU ISA fallback does not imply compatibility with older glibc distributions. OpenMP comparison packages resolved system `libgomp1:arm64` 14.2.0-19; the proposed default has no libgomp dependency. No new release was published.

## Reproduction and artifacts

`build.sh` captures the four-way build/package commands; `install-measure.sh`
installs each local tarball through npm into a different directory before
running `inspect-builds.py`, `measure.py`, `report.py` and native regression tests.
Use fresh output directories. No weights, executables, credentials or dependency
archives are committed. The scripts contain the concrete paths of this study.

The Pi required libclang 19 and its matching standard headers for bindgen. Both
Debian packages were extracted under the study's `deps/` directory without a
system installation. Builds used the existing isolated Rust/CMake toolchain;
`LIBCLANG_PATH=.../deps/usr/lib/aarch64-linux-gnu` and
`BINDGEN_EXTRA_CLANG_ARGS=-resource-dir=.../deps/usr/lib/llvm-19/lib/clang/19` were
set when invoking `build.sh`. The initial missing-libclang/header attempts
preceded all measurement and produced no samples.

Regenerate the derived report with `python3 report.py .` from this directory,
then run `python3 validate.py` and `sha256sum -c SHA256SUMS`.
Raw responses, sampled telemetry, command lines, loaded-library identities and
native regression output accompany the report. Historical outputs are never
substituted for the new predictions.

## State-cache follow-up

The [state-cache study](../pi5-state-cache-20260928/README.md) compares fresh, request-local prefix reuse, state restoration before/after two overhead fixes, and explicit shared-state sessions on the optimized ARM64 build. It uses the same state-first layout within each comparison and reports short-prefix misses, snapshot memory and copy costs. Its diagnostic transport and layout differ from this legacy-layout study, so the timing ratios must not be multiplied or merged as one controlled experiment.
