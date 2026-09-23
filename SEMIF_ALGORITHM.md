# State-first decisions and request-local prefix reuse

The September 2026 update applies SemIf's evidence-first prompt and serial prefix-cache ideas to the existing Rust/libllama decision backend. The scoring contract remains single-token conditional softmax with full-vocabulary candidate mass, typed results, and explicit abstention. No custom head, training, or answer generation was added.

## Reference and scope

Reviewed [SemIf](https://github.com/TheoLeeCJ/SemIf) source on 2026-09-21:

- [`core.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/core.py): evidence before criterion and options. Reviewed Git blob `6e93b16dcd4ab56c7a859c9046c48c6731d7180c`.
- [`serial.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/serial.py): exact prefix validation, independent suffix evaluation, and candidate-mass diagnostics. Reviewed Git blob `015023b7e9d616e6ace0a600548e6fe33ad7f98f`.
- [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py): parallel shared-state branch layout. Reviewed Git blob `6e6870305d1b7693cead70b30637c0fec2684354`.

SemIf is MIT-licensed, copyright 2026 TheoLeeCJ. This update is an independent Rust/C++ implementation of these algorithmic ideas; it does not vendor SemIf code, model weights, or datasets. Its published performance and accuracy are not measurements of this project. This implementation evaluates suffixes serially; parallel branches, cross-request caching, training, and probability calibration remain outside its scope.

## Algorithm

1. For opt-in `state-first`, serialize a typed payload in the fixed order `state`, `instruction`, `options`. A Rust struct guarantees that order even when serde_json map features differ. Preserve structured JSON and tokenize all request data with special-token parsing disabled.
2. Apply the existing model-specific chat template and assistant boundary. Qwen3 non-thinking and GPT-OSS Harmony final prefill remain supported. Validate each A–Z answer code as a stable single-token continuation.
3. For `fresh`, clear memory and evaluate the full prompt. For opt-in `prefix-reuse`, compare the complete tokenized prompt against the previous decision's tokens. Never infer equality from state hashes or raw-text lengths.
4. Round the exact common prefix down to a complete original prefill batch. Keep at least one final token for evaluation, even for identical prompts. This preserves the same suffix batch boundaries as fresh inference. A shared prefix shorter than `--batch` receives no reuse.
5. Remove the previous suffix with `llama_memory_seq_rm` and evaluate the remaining tokens at their original absolute positions. For recurrent/hybrid models or unsuccessful suffix removal, clear memory and evaluate fresh. Clear both native cache metadata and KV memory at every request boundary and after any failure.
6. Read the final full-vocabulary logits and apply the unchanged scoring and abstention policy. `input_tokens` counts logical tokens; `reused_prefix_tokens` counts actual reused tokens; their difference is the number evaluated. `backend.execution_mode` records the requested mode, including when reuse falls back to fresh.

The defaults remain `--prompt-layout legacy --execution-mode fresh`, preserving the original prompt behavior. Select `--prompt-layout state-first` independently to use the versioned v2 prompt. Both layouts support either execution mode, though legacy prompts offer less reusable evidence. Prior v1 benchmark outputs must not be relabeled as v2 results. Moving evidence changes model predictions independently of cache reuse. Scores remain uncalibrated, and a speedup does not establish better semantic accuracy.

## Reproduce

Use the same matching llama.cpp source and library build described in the README:

```sh
export LLAMA_CPP_DIR=/path/to/llama.cpp
export LLAMA_LIB_DIR="$LLAMA_CPP_DIR/build-cuda/bin"

cargo test --locked --offline
cargo test --release --locked --offline --features llama

SKID_MODEL=models/Qwen3-0.6B-Q8_0.gguf SKID_CUDA=0 \
  SKID_REUSE_OUTPUT=/tmp/qwen3-reuse.json \
  cargo test --release --locked --offline --features llama \
  --test prefix_reuse -- --ignored --nocapture
```

Set `SKID_CUDA=1` for CUDA; GPU access is required. Set `SKID_BATCH=32` to check smaller prefill batches (default: 256). The opt-in native test uses 12 synthetic labeled requests (36 decisions), one six-decision long-state request, and one three-decision long-state identical-prompt request. It compares probabilities, candidate mass, raw top-1 and accepted selections, and checks error/request isolation. Its existing 0.02 probability/mass tolerance is a regression check, not an accuracy guarantee; any changed top-1 or accepted selection also fails the test. Timing order alternates between modes, with one warmup request per mode. Each request is measured once per mode, so timings are smoke measurements rather than stable percentiles.

For labeled accuracy, coverage, latency distributions, and repeated runs:

```sh
python3 scripts/benchmark_models.py --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode fresh --output results/v2-fresh
python3 scripts/benchmark_models.py --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode prefix-reuse --output results/v2-reuse
```

Use separate output directories and preserve model hashes, prompt versions, device, batch, context, and policy. The runner records reused and evaluated token counts; logical input-token throughput alone does not measure actual compute saved.

## Validation results

Validation on 2026-09-21 used the same local checkpoints as `MODEL_SPECS.json`, llama.cpp `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`, an Intel Core i9-9900K (four inference threads), and an RTX 3080. Context: 2,048. Policy: top probability 0.8, candidate mass 0.05. Release build, sequential runs, normal workstation activity. No weights were downloaded.

All **12 runs / 540 fresh-versus-reuse decision comparisons passed**: five CUDA models at batches 256 and 32, plus SmolLM2 and Qwen3 on CPU at batch 256. Maximum observed option-probability and candidate-mass differences were **0.0**; no raw top-1 or accepted selections changed. Identical prompts, changed states, overlong-input failure recovery, empty-request failure recovery, and request isolation passed. This is observed equality on these fixtures, not a guarantee for every model or device.

The table measures only the two long-state requests (nine decisions total per mode). It compares fresh and reused execution at the **same** batch size and v2 prompt. Each pair is a single smoke measurement; the ratios are not general throughput guarantees.

| Model | Device | Batch | Fresh long-state ms | Reused long-state ms | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| smollm2 | cuda | 256 | 617.5 | 339.2 | 1.82x |
| qwen3 | cuda | 256 | 638.9 | 336.0 | 1.90x |
| gemma3 | cuda | 256 | 617.8 | 335.7 | 1.84x |
| gemma4 | cuda | 256 | 1094.9 | 558.4 | 1.96x |
| tinyllama | cuda | 256 | 795.4 | 374.0 | 2.13x |
| smollm2 | cuda | 32 | 1470.4 | 568.2 | 2.59x |
| qwen3 | cuda | 32 | 1529.9 | 513.5 | 2.98x |
| gemma3 | cuda | 32 | 1698.2 | 544.3 | 3.12x |
| gemma4 | cuda | 32 | 2910.9 | 943.9 | 3.08x |
| tinyllama | cuda | 32 | 1885.8 | 578.6 | 3.26x |
| smollm2 | cpu | 256 | 10460.3 | 4925.3 | 2.12x |
| qwen3 | cpu | 256 | 36486.4 | 16143.4 | 2.26x |

Short shared prefixes below one batch receive no reuse. Batch 256 saved no tokens on the 12 short labeled requests in this fixture; its speed benefit came from the long-state requests. A smaller batch enables shorter-prefix reuse but changes fresh-inference kernel shapes too, so compare modes at the same batch setting.

An initial unaligned CUDA implementation failed the unchanged 0.02 probability tolerance on all five models (maximum difference about 0.139); SmolLM2 changed four accepted selections. The final implementation rounds reuse down to original batch boundaries. Those failed measurements were retained locally, and the tolerance was not relaxed.

### Prompt quality is a separate measurement

The pre-update v1 executable was preserved and rerun on the same 36 labeled CPU decisions. The v2 results below use fresh execution at batch 256; reuse matched them exactly. Counts distinguish unconditional top-1 accuracy from accuracy among accepted decisions.

| Model / CPU | Prompt | Accepted / 36 | Correct accepted / accepted | Correct raw top-1 / 36 |
| --- | --- | ---: | ---: | ---: |
| smollm2 | v1 legacy | 7/36 | 3/7 | 14/36 |
| smollm2 | v2 state-first | 9/36 | 5/9 | 14/36 |
| qwen3 | v1 legacy | 30/36 | 10/30 | 13/36 |
| qwen3 | v2 state-first | 34/36 | 10/34 | 11/36 |

State-first improved accepted correctness for SmolLM2 on this tiny fixture but reduced Qwen3 raw top-1 accuracy. It is therefore opt-in; default legacy behavior is preserved. These results do not establish improved general accuracy, and the fixture is not held-out calibration data.

### Evidence and limits

- Detailed local outputs: `results/semif/{model}.{cpu|cuda}.batch{32|256}.json` and matching logs. They contain every score and timing. Initial rejected results use `*.cuda.unaligned.json`; v1 outputs use `{smollm2,qwen3}-v1.jsonl`. These working artifacts are ignored by Git.
- All five GGUF SHA256 hashes were rechecked against `VERIFICATION.md`. The Python runner's state-first/reuse smoke run also passed on SmolLM2 CPU at batch 32: 6,033 logical tokens, 1,920 reused tokens, and 4,113 evaluated tokens across the 36 labeled decisions.
- Default-feature and llama-feature tests, formatting, Clippy with `-D warnings`, Python runner tests, CLI option checks, and documentation export were checked. Native tests are opt-in and were run explicitly for the matrix above.
- The current v2 real-model matrix does not include GPT-OSS or a recurrent/hybrid checkpoint. Their template/fallback paths have code coverage or source checks only; prior GPT-OSS v1 evidence remains separate. No parallel GPU branch execution or production workload was measured.
