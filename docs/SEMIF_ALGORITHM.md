# State-first decisions and request-local prefix reuse

[English](en/SEMIF_ALGORITHM.md) · [한국어](ko/SEMIF_ALGORITHM.md) · [日本語](ja/SEMIF_ALGORITHM.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

The prefix-reuse mode applies SemIf's evidence-first prompt and serial prefix-cache ideas to the existing Rust/libllama decision backend. The scoring contract remains single-token conditional softmax with full-vocabulary candidate mass, typed results, and explicit abstention.

## Reference and scope

Reviewed [SemIf](https://github.com/TheoLeeCJ/SemIf) source on 2026-09-21:

- [`core.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/core.py): evidence before criterion and options. Reviewed Git blob `6e93b16dcd4ab56c7a859c9046c48c6731d7180c`.
- [`serial.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/serial.py): exact prefix validation, independent suffix evaluation, and candidate-mass diagnostics. Reviewed Git blob `015023b7e9d616e6ace0a600548e6fe33ad7f98f`.
- [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py): parallel shared-state branch layout. Reviewed Git blob `6e6870305d1b7693cead70b30637c0fec2684354`.

SemIf is MIT-licensed, copyright 2026 TheoLeeCJ. This update is an independent Rust/C++ implementation of these algorithmic ideas; it does not vendor SemIf code, model weights, or datasets. Its published performance and accuracy are not measurements of this project. This implementation evaluates suffixes serially; parallel execution is described separately in [PARALLEL_EXECUTION.md](PARALLEL_EXECUTION.md).

## Algorithm

1. For opt-in `state-first`, serialize a typed payload in the fixed order `state`, `instruction`, `options`. A Rust struct guarantees that order even when serde_json map features differ. Preserve structured JSON and tokenize all request data with special-token parsing disabled.
2. Apply the existing model-specific chat template and assistant boundary. Qwen3 non-thinking and GPT-OSS Harmony final prefill remain supported. Validate each A–Z answer code as a stable single-token continuation.
3. For `fresh`, clear memory and evaluate the full prompt. For opt-in `prefix-reuse`, compare the complete tokenized prompt against the previous decision's tokens. Never infer equality from state hashes or raw-text lengths.
4. Round the exact common prefix down to a complete original prefill batch. Keep at least one final token for evaluation, even for identical prompts. This preserves the same suffix batch boundaries as fresh inference. A shared prefix shorter than `--batch` receives no reuse.
5. Remove the previous suffix with `llama_memory_seq_rm` and evaluate the remaining tokens at their original absolute positions. For recurrent/hybrid models or unsuccessful suffix removal, clear memory and evaluate fresh. Clear both native cache metadata and KV memory at every request boundary and after any failure.
6. Read the final full-vocabulary logits and apply the unchanged scoring and abstention policy. `input_tokens` counts logical tokens; `reused_prefix_tokens` counts actual reused tokens; their difference is the number evaluated. `backend.execution_mode` records the requested mode, including when reuse falls back to fresh.

The defaults remain `--prompt-layout legacy --execution-mode fresh`, preserving the original prompt behavior. Select `--prompt-layout state-first` independently to use the versioned v2 prompt. Both layouts support either execution mode, though legacy prompts offer less reusable evidence. Prior v1 benchmark outputs must not be relabeled as v2 results. Moving evidence changes model predictions independently of cache reuse. Scores remain uncalibrated, and a speedup does not establish better semantic accuracy.

## Reproduce

Use the pinned sys dependency build described in the README:

```sh
cargo test --locked --offline
cargo test --release --locked --offline --features llama

SKID_MODEL=models/Qwen3-0.6B-Q8_0.gguf SKID_CUDA=0 \
  SKID_REUSE_OUTPUT=/tmp/qwen3-reuse.json \
  cargo test --release --locked --offline --features llama \
  --test prefix_reuse -- --ignored --nocapture
```

Set `SKID_CUDA=1` with `--features llama-cuda` for CUDA; GPU access is required. Set `SKID_BATCH=32` to check smaller prefill batches (default: 256). The opt-in native test uses 12 synthetic labeled requests (36 decisions), one six-decision long-state request, and one three-decision long-state identical-prompt request. It compares probabilities, candidate mass, raw top-1 and accepted selections, and checks error/request isolation. Its existing 0.02 probability/mass tolerance is a regression check, not an accuracy guarantee; any changed top-1 or accepted selection also fails the test. Timing order alternates between modes, with one warmup request per mode. Each request is measured once per mode, so timings are smoke measurements rather than stable percentiles.

For labeled accuracy, coverage, latency distributions, and repeated runs:

```sh
target/release/l2s1-tools benchmark-models --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode fresh --output results/v2-fresh
target/release/l2s1-tools benchmark-models --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode prefix-reuse --output results/v2-reuse
```

Use separate output directories and preserve model hashes, prompt versions, device, batch, context, and policy. The runner records reused and evaluated token counts; logical input-token throughput alone does not measure actual compute saved.
