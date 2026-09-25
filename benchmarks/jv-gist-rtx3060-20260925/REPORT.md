# `jv.py` comparison and RTX 3060 CUDA measurement

Measured on 2026-09-25. The reference is [EcmaXp's `jv.py` gist](https://gist.github.com/EcmaXp/042d6b841a88d44187c53bffbda2ba32), revision `59241722f7f44eb65176011f48cf4990eb10ab81` (file SHA-256 `8a44930e4233a13795258793cd1a879db0a08df8b25e2a462a8bcce011c102e5`). That implementation explicitly requires Apple Silicon macOS and Metal and disables CPU fallback. It cannot run on this Linux RTX 3060 server, so **there is no measured `jv.py` versus L2S1 speed ratio**. The CUDA numbers below compare L2S1 execution paths on the same host and GGUF for each model.

## What the implementations do

| Area | `jv.py` reference | L2S1 CUDA path measured here |
| --- | --- | --- |
| Runtime and weights | MLX/Metal; Ternary-Bonsai-2-27B MLX 2-bit or Gemma 4 26B QAT 4-bit | llama.cpp/CUDA; Bonsai 27B Q1_0 GGUF or Gemma 4 26B UD-Q4_K_M GGUF |
| Shared state | Tokenized state prefix retained across requests in an 8-entry, 2 GiB LRU KV cache | `state-first` request-local prefix reuse; no cross-request KV cache |
| Multiple questions | Dynamic suffix batches subject to KV and token budgets | Sequential `fresh`, request-local `prefix-reuse`, or experimental parallel sequences |
| Scoring | Softmax over selected candidate logits | Candidate probabilities plus full-vocabulary `candidate_mass`; `compact` transfers fewer logits to Rust while preserving that mass |
| API | Jev-compatible `POST /v1/systemone` | Typed `POST /v1/decisions` and `GET /v1/capabilities` |

The two Bonsai files are different checkpoints, and the Gemma files use different quantization and runtimes. The fixture uses analogous choice decisions, but its wire payload and prompt are L2S1-specific. The numbers cannot establish which implementation or model is faster.

## Machine and method

- Host: `lucatagpu`, NVIDIA GeForce RTX 3060 12 GiB, driver 595.71.05; CUDA toolkit 13.2.2, compute capability build `sm_86`. CPU: Intel Core i5-12600KF. No other GPU process was present when the runs began.
- L2S1 source: commit `c41925a92dc59feae30537fd5cc1f71215ba6c5e` (merged by PR #30). llama.cpp source: `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. CUDA binary SHA-256: `582dcc2be98483aad0e3731dd278dac30b42429041ab52666d70a86ce8bf69fa`.
- Bonsai GGUF SHA-256: `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`; fully offloaded. Gemma GGUF SHA-256: `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`; 18 MoE layers on CPU to fit in 12 GiB. [Inspect artifacts](bonsai-inspect.json) and [Gemma inspect](gemma-inspect.json) record device identity and supported modes.
- One synthetic warehouse state and either 1 or 16 choice decisions. The [16-decision request](request-16.json) fixes the complete input; the 1-decision request is its first decision with the same state. Both model runs have the same request hash (`323c238174203a15d8e5d1aaec4f2ba7a6e2cd3af7b3aa53222f336f2f1841bd` for 16 decisions).
- Flags: `--device cuda --context 4096 --batch 256 --ubatch 256 --threads 8 --prompt-layout state-first --parallel-width 2` for Gemma; the Bonsai run used width 4 but did not enter parallel mode. Gemma also used `--cpu-moe-layers 18`. All runs used one resident model per mode.
- One warmup plus three timed HTTP requests per mode and count. The reported p50 is the median of three client wall times, including loopback HTTP and JSON handling but excluding process startup/model loading. Mode order was fixed, so these small samples show this fixture's behavior rather than a service-level latency distribution. Startup took about 2.9–3.0 s for Bonsai and 10.2–10.4 s for Gemma. GPU memory was sampled every 0.2 s.

## Results

| GGUF | Mode | 1 decision p50 | 16 decisions p50 | 16-decision speedup vs fresh | Actual `reused_prefix_tokens` for 16 | Peak GPU memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bonsai Q1_0 | fresh/full | 760.9 ms | 12,215.8 ms | 1.00× | 0 | 4,309 MiB |
| Bonsai Q1_0 | prefix/full | 767.0 ms | 12,227.4 ms | 1.00× | 0 | 4,309 MiB |
| Bonsai Q1_0 | prefix/compact | 766.7 ms | 12,234.0 ms | 1.00× | 0 | 4,309 MiB |
| Gemma 4 26B Q4_K_M | fresh/full | 1,565.7 ms | 24,961.0 ms | 1.00× | 0 | 9,581 MiB |
| Gemma 4 26B Q4_K_M | prefix/full | 1,600.2 ms | 12,725.9 ms | **1.96×** | 3,840 | 9,581 MiB |
| Gemma 4 26B Q4_K_M | parallel/full, width 2 | 1,591.8 ms | 17,058.9 ms | 1.46× | 2,048 | 10,513 MiB |
| Gemma 4 26B Q4_K_M | prefix/compact | 1,569.3 ms | 12,657.8 ms | **1.97×** | 3,840 | 9,581 MiB |

[Bonsai summary](bonsai-summary.json) and [Gemma summary](gemma-summary.json) include the three repeated timings' mean, median, minimum, maximum, nearest-rank p95, accepted counts, usage, startup, sampled temperature and GPU memory, and output parity. With only three measured samples, p95 is simply their maximum.

Bonsai reports `recurrent_or_hybrid: true` and `prefix_reuse_fallback: recurrent_or_hybrid_memory`; its measured reuse was zero and latency did not improve. Parallel mode is absent from its inspected supported modes. Gemma reports 3,840 actually reused tokens for prefix mode. Both prefix modes and compact mode matched fresh selections, raw top choices, candidate probabilities and candidate mass exactly. Gemma width-2 parallel kept all selections and raw top choices; its maximum option-probability delta was `1.87e-8`, and maximum candidate-mass delta was `6.04e-8`.

For this simple labeled fixture, raw top choice matched the rule in all 16 decisions for both models. The default policy accepted 15/16 Bonsai decisions and 16/16 Gemma decisions. This verifies the fixture and output stability; it is not an independent task-quality benchmark. Candidate mass is not correctness probability.

Gemma width 4 could not initialize the 16-decision parallel context on this card: CUDA reported a failed 504.75 MiB buffer allocation (`cudaMalloc failed: out of memory`; [log excerpt](parallel-width4-oom.txt)). Width 2 completed at the same model placement. This is a resource limit for this setup, not an inference result.

## Reproduction and boundary

The [driver](bench.py) was run from an isolated server directory containing `source/` (the L2S1 commit above), `results/`, and the model paths recorded in the driver. Set `LD_LIBRARY_PATH` to the native llama.cpp build's `lib` directory and the local CUDA toolkit `lib`. Run `BENCH_MODEL=bonsai python3 bench.py` or `BENCH_MODEL=gemma python3 bench.py`; `BENCH_RESUME=1` resumes only completed, identity-matched modes after an interrupted run. The actual Gemma measurement was resumed after the width-4 OOM. The complete per-mode response and sampler logs remain on the 3060 server at `~/personal/skid/jv-gist-compare-20260925/results/`.

The reference's cross-request KV retention and its suffix batching are meaningful design differences. To measure a direct runtime comparison, run the pinned gist and its pinned MLX weights on Apple Silicon with a separately controlled model/task protocol; an RTX 3060 alone cannot supply that result.
