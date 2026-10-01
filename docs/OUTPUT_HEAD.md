# Frozen deployment output heads

[English](en/OUTPUT_HEAD.md) · [한국어](ko/OUTPUT_HEAD.md) · [日本語](ja/OUTPUT_HEAD.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

An opt-in `--output-head <JSON>` scores a single explicitly identified choice task with a small learned classifier. The GGUF body stays frozen. No adapter is loaded unless requested; ordinary decisions retain the original scoring path.

Two heads are supported:

- `logit_affine`: learn a matrix and bias over the base candidate logits, mapped by semantic option ID. An identity matrix implements temperature-only calibration.
- `hidden`: learn a matrix and bias over Gemma4's final post-output-normalization hidden state. The E2B feature has 1,536 dimensions; a three-class head has 4,611 parameters.

Both compute `softmax((W x + b) / temperature)`. Artifact rows use semantic option IDs, so an option permutation changes code-to-class mapping correctly. Permuting the prompt can still change the underlying model features; it is measured separately.

`candidate_mass` remains the original full-vocabulary LM candidate mass. It is a separate retained compatibility gate, **not** a probability supplied by the new classifier. Temperature calibrates the candidate distribution only. The response identifies the head in `calibration_id`, records its path in `backend.output_head_path`, and uses `learned_hidden_softmax_with_base_mass_v1` or `learned_logit_affine_softmax_with_base_mass_v1`. `scores[].raw_logit` is the head score before temperature; `option_probability` includes temperature.

## Runtime contract

The artifact binds the exact GGUF SHA-256, device description, compute options, prompt version, task ID, instruction and semantic option IDs/criteria. Reordering options is permitted. A different task ID uses the base model; reusing the trained ID with a changed instruction or option meaning fails explicitly. The task ID is a caller assertion: text from a different domain under that ID is not automatically detected.

Only fresh execution is supported with a loaded head. LoRA and output heads cannot be combined. Changed prompt/execution settings after loading are checked again before inference. Hidden-feature export currently supports Gemma4 only. Use the same pinned llama.cpp revision and revalidate/retrain after runtime changes; the JSON does not fingerprint the shared library or GPU driver.

The native bridge uses the pinned llama.cpp staging API `llama_set_embeddings_nextn(..., true, false)` and `llama_get_embeddings_nextn_ith` to read Gemma4's post-norm feature. Unmasked extraction preserves the base graph shapes and transfers hidden rows from the final decode batch; Rust receives only its last row. Normal embedding mode would force vocabulary outputs for all input tokens and is deliberately not used. The full vocabulary projection remains necessary for the retained mass gate, so this implementation does not claim to eliminate the LM head or reproduce Jev/RLCD latency.

## Reproduce the pilot

Build with the bundled llama.cpp commit `3d82ef62d47fd74e18f36c5eccbdcf965b617b17` and CUDA feature:

```bash
cargo build --release --offline --features llama-cuda --examples --bin l2s1
cargo build --release --offline -p l2s1-tools

target/release/l2s1-tools prepare-output-head \
  --source results/kaggle-airline-20260922 \
  --previous results/finetune-20260923/data \
  --output results/output-head-20260923/data

mkdir -p results/output-head-20260923/features
for split in train dev calibration test probe; do
  target/release/examples/export_decision_features \
    --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
    --input "results/output-head-20260923/data/$split.jsonl" \
    --output "results/output-head-20260923/features/$split.jsonl"
done

# Python environment with PyTorch (CPU training) and the standard library.
python3 scripts/train_output_head.py \
  --data results/output-head-20260923/data \
  --features results/output-head-20260923/features \
  --output results/output-head-20260923/heads

target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda --warmup \
  --output-head results/output-head-20260923/heads/selected.json \
  --input results/output-head-20260923/data/test.jsonl \
  --output results/output-head-20260923/selected-test.jsonl
```

A concrete main-CLI call for the selected artifact is:

```bash
target/release/l2s1 --model models/gemma-4-E2B-it-Q8_0.gguf \
  --device cuda --output-head results/output-head-20260923/heads/selected.json \
  --input results/output-head-20260923/example-request.json
```

The main CLI accepts the same artifact with `--device cuda --output-head ...`; its input is a `DecisionRequest`, not a JSONL benchmark envelope. `LlamaBackend::extract_features` exports one fresh base request with its ordinary decision response, and `load_output_head` activates the explicit artifact.

The pilot excludes all normalized exact texts used in both earlier airline experiments. It freezes 900 training texts (three cyclic option orders = 2,700 calls), 300 development texts, 400 calibration texts and 400 test texts. A 60-test-text permutation probe makes 180 calls, not 180 independent cases. Splits are balanced and are not the original population distribution. Semantic near-duplicates are not excluded.

Feature standardization uses training data only. Float64 CPU LBFGS fits cross entropy plus `L2/2 * ||W||²`; the fixed grid is 0.001, 0.01, 0.1 and 1. Each head's penalty and the selected head use lowest development NLL before calibration. Temperature uses calibration only. Standardization is folded into the exported weights and bias. Final test labels never select hyperparameters, head, or temperature. The generated `selection.json` and `heads/training.json` record the selection and fitting details.

Model weights, raw data, features and learned artifacts stay under ignored `models/` and `results/`. Dataset terms and model terms remain separate from this repository's source-code license.
