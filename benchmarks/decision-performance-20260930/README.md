# RTX 3080 decision performance review

Measured 2026-09-29/30 KST; review of v0.2.0 on 2026-09-30.

[English review](../../docs/en/PERFORMANCE_REVIEW.md) · [한국어 검토](../../docs/ko/PERFORMANCE_REVIEW.md) · [日本語の検討](../../docs/ja/PERFORMANCE_REVIEW.md)

This directory contains 693 derived scored records: three models on the same
231 public JevBench items. No private tasks or input text are included.
The records preserve measured probabilities, correctness, latency and acceptance.
`provenance.json` identifies original file hashes, revisions, hardware and settings.

| Model | Correct / 231 | p50 / p95 ms | Accepted correct / accepted | Wrong accepted | Abstained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 159 | 40.41 / 485.74 | 153 / 213 | 60 | 18 |
| Gemma 4 12B QAT Q4_0 | 194 | 93.48 / 1359.39 | 181 / 203 | 22 | 28 |
| Laya English 421M | 133 | 45.18 / 71.63 | 133 / 231 | 98 | 0 |

Gemma uses the L2S1 0.8/0.05 policy; Laya has no additional abstention policy.
Laya's default 512-token budget truncates 57 items; Gemma's 4096-token context
truncates none. Different runtimes, context and quantization; one timed pass per
model, separate runs, no fixed clocks. Model loading and one warmup are excluded.
These are public-subset measurements, not the full 534-item leaderboard.

Recompute the counts, per-family errors, high-confidence errors and fixed-policy
cascade replay with `python3 benchmarks/decision-performance-20260930/summarize.py`.
The cascade replay does not measure deployment latency or memory savings.

[Summary](../../benchmarks/decision-performance-20260930/summary.json) · [Provenance](../../benchmarks/decision-performance-20260930/provenance.json) · [Scored records](../../benchmarks/decision-performance-20260930/records.jsonl)

JevBench attribution: https://github.com/fstandhartinger/jevbench at
`f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`; MIT notice in `JEVBENCH-LICENSE`.
