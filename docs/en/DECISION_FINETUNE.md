<a id="supervised-decision-lora-pilot"></a>
# Supervised decision LoRA pilot

[English](DECISION_FINETUNE.md) · [한국어](../ko/DECISION_FINETUNE.md) · [日本語](../ja/DECISION_FINETUNE.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)


This experiment trains Gemma 4 E2B IT for the existing airline sentiment decision
interface. It follows the public goal of typed probabilistic decisions; it does
not reproduce TypeSafe's proprietary RLCD or Jev architecture. It adds no new
parallel attention architecture and makes no claim of general decision ability.

<a id="frozen-protocol"></a>
## Frozen protocol

- Source: the pinned Kaggle Twitter US Airline Sentiment archive prepared by
  `target/release/l2s1-tools kaggle-airline`. Dataset text and weights remain in ignored
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

<a id="entry-points"></a>
## Entry points

1. `target/release/l2s1-tools prepare-decision-finetune` freezes the data and protocol.
2. `examples/export_decision_tokens.rs` exports production input/candidate IDs.
3. `scripts/train_decision_lora.py` downloads the pinned model, runs a smoke test
   or the full paired experiment, and saves the adapter and raw measurements.
4. `target/release/l2s1-tools report-decision-finetune` fits calibration-only temperatures and
   reports held-out metrics and option-order diagnostics.

The CLI and JSONL evaluator accept `--lora path/to/adapter.gguf`. Convert a PEFT
adapter with the matching llama.cpp `convert_lora_to_gguf.py --base <config-dir>`
tool. One adapter is supported per backend at scale 1; it is reattached whenever
parallel execution resizes the context. Response metadata records `lora_path`.
Without this option the existing base-model path is unchanged. Loading an adapter
does not apply temperature calibration automatically.

Keep generated artifacts, the dependency lock and exact executed scripts under
an ignored local `results/` directory.

For a new artifact directory (this reproduces the same split, not a new test set):

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools prepare-decision-finetune \
  --source results/kaggle-airline-20260922 --output results/decision-pilot-new/data
for split in train calibration test probe; do
  target/release/examples/export_decision_tokens \
    --model models/gemma-4-E2B-it-Q8_0.gguf \
    --input "results/decision-pilot-new/data/$split.jsonl" \
    --output "results/decision-pilot-new/data/$split-tokens.jsonl"
done
target/release/l2s1-tools prepare-decision-finetune \
  --output results/decision-pilot-new/data --seal-tokens
# Run in the locked training environment on the GPU host:
python scripts/train_decision_lora.py --data results/decision-pilot-new/data \
  --output results/decision-pilot-new/pilot --cache results/decision-pilot-new/hf-cache
target/release/l2s1-tools report-decision-finetune --data results/decision-pilot-new/data \
  --run results/decision-pilot-new/pilot
```

<a id="frozen-synthetic-accuracy-study"></a>
## Frozen synthetic accuracy study

The new study is separate from the airline pilot and the earlier warehouse
fixtures. It generates 480 train, 120 dev, 120 calibration and 180 test logical
cases across six domains. Both threshold values and wording-template families
are disjoint across splits. Each case has paired natural-language and symbolic
criteria; these are two representations of one case, not independent samples.
Choice, binary and ordinal targets are balanced, including exact and neighboring
integer/fractional boundaries. Ordinal values retain their sorted order.

The following workflow requires a configured native build and an existing local
Gemma 4 E2B IT checkpoint at the pinned revision above. Training additionally
requires the GPU Python environment with compatible PyTorch, Transformers,
bitsandbytes and PEFT. Record its dependency versions with the experiment. Keep
all data, token exports, teacher responses, adapters and reports under ignored
`results/`; keep base GGUF weights under ignored `models/`. The new scripts refuse
to overwrite their output files/directories and do not download checkpoints.

Prepare the study and compare prompt configurations on **dev only**:

```sh
study=results/accuracy-study-new
model=models/gemma-4-E2B-it-Q8_0.gguf
checkpoint=/path/to/local/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7
variant=natural
mkdir -p "$study"
target/release/l2s1-tools prepare-accuracy-study --output "$study/data"
target/release/l2s1-tools prepare-accuracy-study --output "$study/data" --verify
cargo build --release --locked --features llama-cuda \
  --example evaluate_accuracy --example export_decision_tokens

target/release/examples/evaluate_accuracy --model "$model" --cuda \
  --input "$study/data/dev-$variant-requests.jsonl" \
  --output "$study/dev-$variant-predictions.jsonl" \
  --prompt-details minimal,typed,typed-examples --layouts legacy,state-first \
  --all-rotations
target/release/l2s1-tools report-accuracy-study --data "$study/data" \
  --predictions "$study/dev-$variant-predictions.jsonl" \
  --split dev --variant "$variant" --output "$study/dev-$variant-report.json" \
  --select "$study/dev-$variant-selection.json"
```

Use `symbolic` for the paired representation experiment. If comparing both
variants, finish both dev reports and freeze the winning variant before reading
calibration/test predictions. Selection ranks raw top-1, then accepted-correct
fraction over all cases, then median compute-path latency, with a deterministic
final tie break. Only configurations covering the whole split are eligible:
rotation 2 covers three-option tasks only and cannot win against full-split
configurations. `--select` rejects calibration and test splits. Timings exclude
model loading and output serialization; they are not end-to-end service latency.

Read the frozen choice and export **training requests only**, including their
code rotations. The exporter uses the production GGUF tokenizer. The trainer's
`--detail` uses underscores; native CLI enum values use hyphens.

```sh
selection="$study/dev-$variant-selection.json"
detail=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["setting"]["prompt_detail"])' "$selection")
detail_cli=$(python3 -c 'import sys; print(sys.argv[1].replace("_", "-"))' "$detail")
layout=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["setting"]["prompt_layout"].replace("_", "-"))' "$selection")
target/release/examples/export_decision_tokens --model "$model" \
  --input "$study/data/train-$variant-requests.jsonl" \
  --output "$study/train-tokens.jsonl" --all-rotations \
  --prompt-detail "$detail_cli" --prompt-layout "$layout"

python3 - "$study" "$variant" "$detail" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
seal = dict(schema_version=1, split='train', variant=sys.argv[2], detail=sys.argv[3],
            manifest_sha256=digest(root/'data/manifest.json'),
            token_sha256=digest(root/'train-tokens.jsonl'),
            hf_model='google/gemma-4-E2B-it',
            hf_revision='3e22461f65e89153144f8adb70e3b8c2cc9845a7')
with (root/'train-token-seal.json').open('x') as out:
    json.dump(seal, out, sort_keys=True, indent=2)
    out.write('\n')
PY
```

Generate thinking-teacher answers, run a discarded smoke check, then train a
fresh adapter. Only teacher final answers matching independent **train** labels
are admitted. Generated rationales are logged but never used as student targets;
missing or wrong teacher answers are not replaced with oracle labels. The same
accepted examples and protocol must bind smoke and final training.

`--teacher-batch-size` enables left-padded teacher generation (default 1). Each
answer is trimmed independently at its first EOS before validation. Use a separate
limited pilot to size GPU memory; the teacher report records peak CUDA allocation
and batch timing. Incomplete pilots are not accepted as training inputs.

```sh
train_stage() {
  python3 scripts/train_accuracy_lora.py "$@" \
    --data "$study/data" --variant "$variant" --detail "$detail" \
    --tokens "$study/train-tokens.jsonl" --token-seal "$study/train-token-seal.json" \
    --checkpoint "$checkpoint"
}
train_stage teacher --output "$study/teacher" --max-new-tokens 384
train_stage smoke --teacher "$study/teacher" --output "$study/smoke"
train_stage train --teacher "$study/teacher" \
  --smoke-report "$study/smoke/complete.json" --output "$study/train"
python3 "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" "$study/train/adapter" \
  --base "$checkpoint" --outfile "$study/adapter.gguf" --outtype f16
```

This conversion command needs a full llama.cpp checkout in `LLAMA_CPP_DIR`; the bundled native build snapshot omits conversion tools. The inference build uses the bundled source unless `L2S1_LLAMA_CPP_SOURCE` or the legacy `LLAMA_CPP_DIR` override is set.

A successful training loss, smoke check or conversion does not establish a
native accuracy improvement. Evaluate the converted adapter with the same GGUF,
compute configuration and frozen prompt settings as the base. For example:

```sh
target/release/examples/evaluate_accuracy --model "$model" --cuda \
  --lora "$study/adapter.gguf" \
  --input "$study/data/test-$variant-requests.jsonl" \
  --output "$study/test-$variant-adapter.jsonl" \
  --prompt-details "$detail_cli" --layouts "$layout" --all-rotations
target/release/l2s1-tools report-accuracy-study --data "$study/data" \
  --predictions "$study/test-$variant-adapter.jsonl" \
  --split test --variant "$variant" --output "$study/test-$variant-adapter-report.json"
```

Run the identical native evaluation without `--lora` into separate base output
files. Compare only the previously frozen single rotation or ensemble; additional
rotation rows are diagnostics, not new opportunities to select on test. If fitting
calibration or abstention thresholds, finish that work using the calibration
split before the final test evaluation; the reporter itself fits neither.
Report raw top-1, NLL/Brier, coverage, accepted accuracy, accepted-correct/all and
rotation sensitivity together. NF4 teacher/training results and GGUF adapter
results have different runtime/precision boundaries. This workflow makes no
measured accuracy or performance claim by itself.
