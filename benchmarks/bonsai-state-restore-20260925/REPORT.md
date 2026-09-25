# Bonsai request-local state restoration on RTX 3060

Measured on 2026-09-25 on `lucatagpu`, an NVIDIA GeForce RTX 3060 12 GiB with driver 595.71.05. The hybrid checkpoint falls back to fresh execution in `prefix-reuse` mode. The explicit `state-restore` mode saves the common prefix's complete llama.cpp sequence state, runs the first suffix on that state, and restores it for each later suffix. The state exists only during one native request. If the snapshot cannot be used, execution falls back to fresh.

## Setup

- L2S1 measured build from commit `a12a99b` (same implementation rebased as `7be9a29`); CUDA binary SHA-256 `edeb9b1e61afb72a0db94d71201f6d0baf63e4cfd3eb3c60bb7cf3c1e4edbdde`. llama.cpp source revision `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.
- Ternary Bonsai 2 27B Q1_0 GGUF SHA-256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`, fully offloaded to CUDA.
- The fixed [16-decision warehouse request](request-16.json) and its first decision alone. The fixture uses a shared state prefix and choice decisions. Flags: `--device cuda --context 4096 --batch 256 --ubatch 256 --threads 8 --prompt-layout state-first`; the second path adds `--execution-mode state-restore`. Both use full evidence transfer.
- One warmup and three measured loopback HTTP requests per count and mode, with one resident model per mode. Timings include HTTP and JSON handling but exclude startup and model loading. Modes ran sequentially. GPU memory was sampled during each run. The [raw summary](summary.json) has all measured wall times and result comparisons.

## Results

| Mode | 1 decision p50 | 16 decisions p50 | 16-decision speedup | Reused prefix tokens for 16 | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh/full | 757.0 ms | 12,207.9 ms | 1.00× | 0 | 4,309 MiB |
| state-restore/full | 766.1 ms | 6,976.4 ms | **1.75×** | 3,840 | 4,309 MiB |

The three 16-decision client times were 12,156.0/12,207.9/12,226.3 ms for fresh and 6,973.4/6,976.4/6,981.5 ms for state restoration. The restored request saved a 173,678,124-byte state snapshot and loaded it 15 times for the later 15 decisions; the first decision used the original prefilled state. Its diagnostic timings were 338.7 ms prefill, 257.2 ms save, 1,225.7 ms restore, and 5,158.5 ms suffix execution. The one-decision request had no reused tokens.

For all measured requests, the restored path had **zero** changed selections and raw top choices and a **zero** maximum difference in option probabilities and candidate mass versus fresh. All 16 raw top choices matched the fixture's simple rules; the default policy accepted 15/16 decisions in both paths. The real-model regression additionally compared per-option token IDs, logits, probabilities, candidate mass, values and abstention reasons. It passed on this checkpoint. With a zero-byte snapshot limit it reported `snapshot_memory_budget`, ran fresh with zero reused tokens, and produced the same results; restoring the normal limit allowed reuse again.

## Reproduction and limits

On a CUDA host with the same GGUF, run the ignored regression using `L2S1_BONSAI_MODEL=/path/to/Bonsai-27B-Q1_0.gguf cargo test --release --locked --features llama-cuda --test state_restore_bonsai -- --ignored --nocapture`. For the CLI diagnostics, use the flags above with `--input` pointing to the linked request and add `--diagnostics`. The full benchmark outputs and test logs remain on the measured host under `~/personal/skid/state-restore-stable-20260925/results/`.

This is execution equivalence and latency for one synthetic fixture on one GGUF and GPU. It does not measure labeled task quality, other hybrid checkpoints, cross-request reuse, or a production latency distribution. The snapshot limit controls the snapshot buffer, not total process memory. `state-restore` remains opt-in; `prefix-reuse` still reports hybrid fallback on Bonsai.
