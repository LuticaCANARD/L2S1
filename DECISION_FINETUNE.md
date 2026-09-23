# Supervised decision LoRA pilot

This experiment trains Gemma 4 E2B IT for the existing airline sentiment decision
interface. It follows the public goal of typed probabilistic decisions; it does
not reproduce TypeSafe's proprietary RLCD or Jev architecture. It adds no new
parallel attention architecture and makes no claim of general decision ability.

## Frozen protocol

- Source: the pinned Kaggle Twitter US Airline Sentiment archive already used by
  `AIRLINE_BENCHMARK_RESULTS.md`. Dataset text and weights remain in ignored
  `results/`; they are not redistributed by this repository.
- Exclude all 800 previous calibration/evaluation examples by normalized text.
  Remove exact normalized duplicates and conflicting-label text groups.
- Seed 20260923: 900 unique training tweets (300 per class), 400 new calibration
  tweets and 400 new test tweets. Splits are disjoint by normalized text. This is
  an intentionally balanced, single-domain sample, not natural class prevalence.
- Each training tweet appears in all three cyclic option orders (2,700 training
  inputs). A fixed 60-case test subset is evaluated in all three orders (180
  diagnostic calls; these are not additional independent cases).
- `export_decision_tokens` uses the real Rust/GGUF prompt renderer and tokenizer.
  Transformers consumes those exact input and candidate token IDs, avoiding a
  reimplementation of production prompts. Labels are stored separately.
- Base model: `google/gemma-4-E2B-it`, revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Training uses an NF4 base with bf16
  compute, frozen large matrices in bf16 and fp32 normalization/adapters.
- LoRA rank 8, alpha 16, dropout 0, language-model attention Q/V projections only.
  One epoch, microbatch 1, accumulation 12, learning rate 0.0001 linearly decaying
  to zero, AdamW with zero weight decay, gradient norm clipping at 1.
- Loss at the decision position: candidate cross-entropy plus 0.1 times negative
  log full-vocabulary candidate mass. No loss on prompt tokens or generated text.
- Fixed final checkpoint, no test-based checkpoint/hyperparameter selection.
  A training-only smoke run checks GPU/memory/gradient compatibility first;
  its adapter is discarded and the actual run starts from the original base.
- Fit each checkpoint's temperature on its new calibration split only. Report
  raw top-1, NLL, Brier, ECE, coverage and accepted accuracy at thresholds 0.6–1.0.
  Candidate mass remains unchanged and its 0.05 gate remains active.
- Compare base and trained checkpoints in the same runtime/precision. Treat
  Transformers NF4 and llama.cpp Q8 as distinct comparisons; conversion effects
  are not training gains. No application default is changed by this pilot.
- Run the previous AG News 400-case development benchmark as an additional
  out-of-domain regression check on the GGUF base and adapter. It is not a new
  untouched test set and does not fit this experiment's adapter or temperature.

Public pretraining exposure and near-duplicate text are not excluded. A single
training seed cannot establish reproducibility across random initializations.
New test data comes from the same historical dataset and does not establish
out-of-domain or production accuracy. Current API calibration is an offline
artifact and is not silently applied to application responses.

## Entry points

1. `scripts/prepare_decision_finetune.py` freezes the data and protocol.
2. `examples/export_decision_tokens.rs` exports production input/candidate IDs.
3. `scripts/train_decision_lora.py` downloads the pinned model, runs a smoke test
   or the full paired experiment, and saves the adapter and raw measurements.
4. `scripts/report_decision_finetune.py` fits calibration-only temperatures and
   reports held-out metrics and option-order diagnostics.

The CLI and JSONL evaluator accept `--lora path/to/adapter.gguf`. Convert a PEFT
adapter with the matching llama.cpp `convert_lora_to_gguf.py --base <config-dir>`
tool. One adapter is supported per backend at scale 1; it is reattached whenever
parallel execution resizes the context. Response metadata records `lora_path`.
Without this option the existing base-model path is unchanged. Loading an adapter
does not apply temperature calibration automatically.

Artifacts for the current run are under `results/finetune-20260923/`. The remote
training directory is `~/skid-desion-finetune-20260923` on the authorized host.
The dependency lock and exact executed scripts accompany the results.

For a new artifact directory (this reproduces the same split, not a new test set):

```sh
python3 scripts/prepare_decision_finetune.py \
  --source results/kaggle-airline-20260922 --output results/decision-pilot-new/data
for split in train calibration test probe; do
  target/release/examples/export_decision_tokens \
    --model models/gemma-4-E2B-it-Q8_0.gguf \
    --input "results/decision-pilot-new/data/$split.jsonl" \
    --output "results/decision-pilot-new/data/$split-tokens.jsonl"
done
python3 scripts/prepare_decision_finetune.py \
  --output results/decision-pilot-new/data --seal-tokens
# Run in the locked training environment on the GPU host:
python scripts/train_decision_lora.py --data results/decision-pilot-new/data \
  --output results/decision-pilot-new/pilot --cache results/decision-pilot-new/hf-cache
python3 scripts/report_decision_finetune.py --data results/decision-pilot-new/data \
  --run results/decision-pilot-new/pilot
```
