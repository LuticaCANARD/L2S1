<a id="direct-vision-http-benchmark"></a>
# Direct vision HTTP benchmark

[English](VISION_BENCHMARK.md) · [한국어](../ko/VISION_BENCHMARK.md) · [日本語](../ja/VISION_BENCHMARK.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)


This is a **local synthetic smoke benchmark**, measured on 2026-09-24 on `lucatagpu` (RTX 3060 12 GiB, driver 595.71.05). It compares the CPU and CUDA paths of the same CUDA-enabled L2S1 executable, Gemma 4 E2B Q8_0 GGUF, matching `mmproj`, request schema and host. The model and projector SHA-256 values are recorded in the [raw CPU](../../benchmarks/vision-20260924/cpu.json) and [raw CUDA](../../benchmarks/vision-20260924/cuda.json) reports. Model weights are not in this repository.

The [runner](../../scripts/benchmark_vision_http.py) starts an isolated loopback HTTP server for each device, waits for `/healthz`, sends four excluded warmup requests, then times 30 serial `POST /v1/decisions` requests alternating the repository's red and blue 64×64 PNGs. It prepares JSON/base64 before timing. Each timed interval includes HTTP request/response transfer, image encoding, model inference, scoring and response serialization. A single loaded model handles each run; no parallel clients or batching across requests are measured. Both runs used context 2048, batch 256, four CPU threads and `--model-load-mode read`. The GPU run preceded the CPU run.

| Device | p50 | p95, nearest rank | Mean | Startup to `/healthz` | Fixture selections | GPU memory while loaded | Process peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 CUDA | 96.054 ms | 96.236 ms | 96.021 ms | 5.013 s | 30/30 | 3,813 MiB | 3.47 GiB |
| CPU on same host | 3,066.295 ms | 3,162.112 ms | 3,079.159 ms | 5.263 s | 30/30 | 20 MiB | 5.94 GiB |

The CPU/CUDA p50 ratio was **31.9×** for this repeated two-image workload. The 30 selections repeat only two labeled solid-color images; 30/30 is an interface smoke check, not 30 independent examples or a general vision accuracy result. Process RSS and GPU memory are different measures and cannot be added or treated as total system memory. Startup includes process launch, model/projector loading and readiness polling; the latency columns exclude startup and warmup. This single serial run does not establish concurrent throughput or a production latency guarantee.

<a id="reproduce"></a>
## Reproduce

Build L2S1 with `--features llama-cuda`, supply a matching vision GGUF and `mmproj`, and make the build's matching native shared libraries available through `LD_LIBRARY_PATH`. Use the same binary for both device modes. The runner refuses to overwrite existing reports and saves every warmup/measured observation plus the model, projector, binary and image hashes.

```sh
python3 scripts/benchmark_vision_http.py \
  --binary target/release/l2s1 \
  --model /path/to/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --red-image tests/fixtures/vision_red_64.png \
  --blue-image tests/fixtures/vision_blue_64.png \
  --device cuda --warmup 4 --iterations 30 \
  --output results/vision-http-cuda.json
```

Repeat with `--device cpu`, another output path and a free `--listen` address. The checked-in reports were produced from the vision implementation before it was extracted into PR #12; their binary SHA-256 identifies that exact build. The PR #12 source passes the isolated CPU contract tests, while these timing numbers describe the recorded host build and configuration.
