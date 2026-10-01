# Generic GGUF CUDA path: two-model smoke measurement

[English](../../docs/en/benchmarks/gguf-cuda-20260925/README.md) · [한국어](../../docs/ko/benchmarks/gguf-cuda-20260925/README.md) · [日本語](../../docs/ja/benchmarks/gguf-cuda-20260925/README.md)

[English index](../../docs/en/README.md) · [한국어 색인](../../docs/ko/README.md) · [日本語索引](../../docs/ja/README.md)

This run checks that the existing llama.cpp CUDA executable can produce typed
decisions from two different GGUF model families. It is one invocation per
model, without warmup, on the three decisions in [`examples/warehouse.json`](../../examples/warehouse.json)
(SHA-256 `68e3bffae421112de90658c62c5d52293e53b719f7610abb5377b9ad4bbd4944`).
The expected values from the fixture are `chilled`, `true`, and `high`.

## Runtime and method

- NVIDIA GeForce RTX 3060 (12 GB); each response reports this device as the CUDA offload target.
- L2S1 CUDA executable copied to the server on 2026-09-24, SHA-256 `b99f8f1e40d6ae542ed780eb3868e72a1c93f735ff488663b073c2335578af62`. The server copy has no Git metadata, so this is a check of the existing CUDA path, not a build of this PR commit. Both responses record compiled runtime SHA-256 `771abedbe730b88af57cc1bc35543c4353b30382de7bf9c475a78e9814e66043` and loaded runtime SHA-256 `ae19e1bb27af03e2057408582ca17f35a748b7fec5aab75db5bef03763a026d8`.
- Fresh execution, context 2048, batch/ubatch 256, 4 CPU threads, FlashAttention off, default decision policy. No prompt or runtime options changed between models.
- Command: `l2s1 --model /path/to/model.gguf --device cuda --diagnostics --input examples/warehouse.json`. The binary used matching llama.cpp libraries from its copied `lib/` directory and CUDA runtime libraries from the GPU server.
- `native_ms` is the request-local inference timer in the response. Wall time includes loading, model hashing, JSON output, and process startup.

## Results

| Model | GGUF SHA-256 | Native inference | Wall time | Correct / all | Selected / all | Correct / selected | Raw top-1 / all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | 130.73 ms | 1.29 s | 0/3 | 3/3 | 0/3 | 0/3 |
| SmolLM2-135M Instruct Q8_0 | `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83` | 111.14 ms | 0.76 s | 0/3 | 0/3 | undefined | 0/3 |

Qwen selected `ambient`, `false`, and `low`. SmolLM2 abstained on all three
decisions; its accepted-only accuracy is undefined. The result proves that
both GGUFs ran through the CUDA decision path. Three synthetic decisions and
one timing sample per model do not establish task accuracy or stable latency.

Full model identities, option scores, candidate masses, selections, abstention
reasons, and stage timings are in [`qwen3-q8-rtx3060.json`](qwen3-q8-rtx3060.json)
and [`smollm2-q8-rtx3060.json`](smollm2-q8-rtx3060.json). Model weights are
not included.
