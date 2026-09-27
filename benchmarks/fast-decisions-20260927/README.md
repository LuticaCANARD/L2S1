# Fixed prefix and Laya integration diagnostics, 2026-09-27

These local measurements validate the new execution paths. They are not JevBench scores, general task accuracy, M5 Max results, or production error guarantees. See [summary](summary.json) and [provenance](provenance.json).

## Fixed-schema prefix

Qwen3 0.6B Q8_0, CPU, 4 threads, context 2048, batch/ubatch 256. Original decision-rules-v1 multi-decision requests: 36 judgments per complete pass. Each path has one full warmup and one measured pass; these times are **whole-pass totals**, not request p50.

| Execution | Whole pass |
| --- | ---: |
| Flat fresh | 29.132 s |
| Explicit split, cold | 31.062 s |
| Explicit split, prefix reuse | 14.962 s |

Reuse is 1.95x faster than flat fresh and 2.08x faster than matched split cold here. It reuses 2,799 of 4,983 tokens. Cold/warm selections, probabilities and candidate mass match exactly; flat/split selections and probabilities also match in this run. One native context and at most eight host prefix snapshots are retained. The cache begins empty for each measured reuse pass. An earlier run overlapping compilation was discarded; Metal/CUDA native prefix performance still needs hardware-specific measurement.

## Laya against Ollaya

Same English Laya graph, tokenizer, original weights, calibration, input strings and questions. 20 synthetic ticket inputs, five questions each; 20 warmup requests then two measured passes: 40 requests / 200 judgments per engine. CUDA uses fp16 on RTX 3080 10 GB; CPU uses fp32. Both use ORT 1.30.0, four intra-op threads, no HTTP/model loading in the timing.

| Runtime | CUDA request p50 | CUDA request p95 |
| --- | ---: | ---: |
| L2S1 ONNX | 19.179 ms | 22.165 ms |
| Ollaya runner | 19.567 ms | 23.338 ms |

This is comparable local speed, not evidence that L2S1 is generally faster. L2S1 includes probability scoring; the reference times tokenization and forward only. CPU records are retained to audit parity, but their variable tails are not used for a speed claim. All token IDs, marker positions and raw logits match exactly on **both** devices. Maximum probability difference is 2.23e-16; no top-1 changes. These unlabelled synthetic parity inputs do not measure accuracy.

Real HTTP verification generated a policy from 18 toy calibration rows and 18 separate toy validation rows, then exercised fast acceptance, unregistered-schema fallback, over-context fast failure with GGUF fallback, and Python SDK decoding. Six of 16 simultaneous independent requests were observed in one native batch. This demonstrates batching/routing, not a throughput gain or p95 guarantee. The toy policy is deliberately not published as a deployable artifact.

## Reproduce

Set `ORT_DYLIB_PATH` to an installed ONNX Runtime >=1.24 library and configure compatible GPU provider libraries for CUDA. This WSL host also required the matching Windows-driver directory before the stale Linux NVIDIA libraries in `LD_LIBRARY_PATH`; no system libraries were changed. Paths are machine-specific.

```sh
cargo build --release --features llama,onnx-cuda --example benchmark_onnx --example benchmark_fixed_schema
python3 scripts/fetch_laya_onnx.py --precision fp16 --output models/laya-en-fp16
target/release/examples/benchmark_onnx --model-dir models/laya-en-fp16 --cuda \
  --input benchmarks/fast-decisions-20260927/requests.jsonl --output onnx.json
target/release/examples/benchmark_fixed_schema --model models/Qwen3-0.6B-Q8_0.gguf \
  --rounds 1 --batch 256 --output prefix.json
python3 benchmarks/fast-decisions-20260927/audit.py
```

For the reference, check out Ollaya revision `6d121dd45162b222bc4deb971c5975875f538fa8`, copy `l2s1_compare.rs` from this directory into `crates/ollaya-runner/examples/`, and build that example with `--features cuda-dynamic`. Arguments are model directory, this directory's `ollaya.jsonl`, and `cuda` or `cpu`. The complete token/logit/probability records are retained in the four `.json.gz` files; `audit.py` recomputes parity without running inference. Model provenance and license notes are in `scripts/fetch_laya_onnx.py` and `third_party/ollaya/NOTICE`.
