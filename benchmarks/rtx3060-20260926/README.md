# RTX 3060 public benchmark artifacts

Measured 2026-09-23 · exported 2026-09-26 · JevBench public 231 items · 20 model configurations · 4,620 decisions.

- [summary.json](summary.json): verified aggregate counts, score percentages, latency, sampled whole-device GPU memory, GGUF file bytes, and runtime configuration.
- [manifest.json](manifest.json): original artifact SHA-256 values, checkpoint hashes, file-size provenance, frozen evaluator identity, and portable command arguments.
- [English report](../../docs/en/RTX3060_BENCHMARK.md) · [한국어](../../docs/ko/RTX3060_BENCHMARK.md) · [日本語](../../docs/ja/RTX3060_BENCHMARK.md).

```sh
python3 scripts/export_rtx3060_web.py
python3 scripts/export_rtx3060_web.py --check
```
