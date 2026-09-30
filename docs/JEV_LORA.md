# Jev-type decision LoRA training

The official `l2s1-training` package provides the `l2s1-train` CLI.
Install from this repository or its built wheel; the package is not yet published
to PyPI. Its schema version 1 covers data manifests, model registries and run
reports. Existing script entry points remain compatibility wrappers.

The same pipeline supports all five local benchmark models: SmolLM2, Qwen3,
Gemma3, TinyLlama and Gemma4. A model profile pins its original Hugging Face
checkpoint, GGUF filename, Transformers loader and language attention adapter
targets. Each model gets its own native token export, discarded smoke run,
adapter, GGUF conversion and paired base/adapter evaluation. `--models all`
runs them sequentially and retains failures in the matrix summary.

The first real-weight pilot uses Gemma4 E2B. The other four profiles have CPU
architecture/gradient checks using tiny random models; those checks are not
pretrained-model, NF4, conversion or quality evidence for those models.

The [completed Gemma4 pilot](../benchmarks/jev-lora-20260927/README.md) improved raw label agreement from 54.30% to 57.85% and reduced soft KL from 3.7739 to 0.7544. Coverage fell from 92.75% to 48.35%, and correct accepted/all fell from 51.65% to 32.30%. Noul supplied the raw-accuracy gain; Choice and Score raw accuracy slightly declined. The existing rules regression also lost raw accuracy. The adapter remains an opt-in experiment.

## Ollaya reference and output

Reference: [ollaya-dev/ollaya at f9e2d11](https://github.com/ollaya-dev/ollaya/tree/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea).
In particular, its [answer renderer](https://github.com/ollaya-dev/ollaya/blob/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea/crates/ollaya-decision/src/answer.rs),
[API types](https://github.com/ollaya-dev/ollaya/blob/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea/crates/ollaya-api/src/decide.rs),
and [Winnow family](https://github.com/ollaya-dev/ollaya/blob/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea/docs/families/winnow.md)
inform the output contract and token-logit approach.

| Type | Native decision | Jev-shaped answer |
| --- | --- | --- |
| Choice | `choice` | `choice`, `probabilities`, `confidence` |
| Noul | `binary` | `noul = P(true)`; no confidence field |
| Score | zero-based `ordinal` | expected index in `score`, plus `legend`, `probabilities`, `confidence` |

Choice/Score confidence is `(K * max(p) - 1) / (K - 1)`, clamped to [0, 1].
It differs from L2S1's entropy confidence. Wire numbers use four decimal places;
winner selection and metrics use the original probabilities. A 300-case check
against compiled, unmodified Ollaya decision source matched all output JSONs.
This proves synthetic answer-rendering parity, not HTTP API or model parity.

`report_jev.py` writes one record per case to `jev-answers.jsonl`, with answers
keyed by question ID. `l2s1_policy` separately preserves native acceptance,
abstention reasons, candidate mass and entropy confidence. A raw Jev-shaped
answer is not permission to bypass an abstention. `usage.output_tokens` is zero:
the runtime reads candidate logits and the serializer constructs the answer.
This training package adds no HTTP endpoint or shared multiquestion forward pass.

## Frozen first experiment

- Data: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90),
  pinned revision `c76749ec58bd8c3d2ea706b31c333a9059c38f90`.
- Select 30 train cases from each of four workflows by a seeded ID hash:
  120 cases / 600 decisions. Reserve another 80 cases for development and leave
  1,000 unused. Evaluate the independent public test split: 400 cases / 2,000
  decisions. Exact canonical-state and ID overlap is rejected across all splits.
- Only state, instructions and criteria enter inference. Teacher probabilities
  supply soft targets; factors, gold labels and teacher metadata never enter prompts.
- This is a specialist trained on those four workflows. Test agreement measures
  agreement with synthetic teacher labels, not general decision correctness or a
  comparison with an unseen-workflow generalist. Near duplicates and pretraining
  exposure are not excluded.
- Export all cyclic answer-code assignments with the native renderer. Choose one
  assignment per training decision by a label-independent hash. Shared semantic
  labels, model identity, tokens and export source are validated before training.
- One epoch; LoRA rank 8, alpha 16, Q/V attention projections, dropout 0;
  microbatch 1, accumulation 12; AdamW learning rate 1e-4 decays to zero;
  soft-target candidate cross entropy plus 0.1 negative log candidate mass.
- NF4 frozen base, bf16 compute where supported (fp16 otherwise), fp32 adapters
  and normalization parameters. CUDA is required for this training path; the
  Apple Silicon path below keeps the data, loss and schedule but uses an MLX
  4-bit affine base and fp32 adapters.
- A 12-example smoke adapter is discarded. Final training restarts from the base.
  Fixed final checkpoint; no test/development selection, threshold fitting or
  temperature fitting. These probabilities are not claimed calibrated.
- Re-evaluate both checkpoints in the same GGUF runtime and precision, with
  policy thresholds 0.8 top probability / 0.05 candidate mass. Report raw label
  accuracy, coverage, accepted accuracy, correct/all, hard and soft probability
  errors, per-type/per-workflow metrics, and full request latency.

## Run another model or the whole matrix

Install the lightweight package for data preparation, profiles and reporting:

```sh
python -m pip install ./training
l2s1-train --version
l2s1-train profiles
l2s1-train doctor
```

For training, use a separate Python 3.11+ environment, install the appropriate
CUDA PyTorch wheel for the machine, then `python -m pip install './training[cuda,parquet]'`.
The CUDA extra pins the stack used in the pilot (PyTorch 2.14.0,
Transformers 5.17.0, PEFT 0.21.0, bitsandbytes 0.50.2, Accelerate 1.15.0).
`l2s1-train doctor --cuda` reports missing dependencies and CUDA availability.
Native exporter/evaluator and matching llama.cpp converter/libraries remain
separate prerequisites. Existing local checkpoints can be reused offline.
This does not establish Windows GPU training validation; installed CLI contracts
are checked separately from real CUDA execution.

On Apple Silicon, install `'./training[mlx]'` instead, build the native tools
with `--features llama-metal`, check with `l2s1-train doctor --mlx`, and add
`--backend mlx` to `l2s1-train run`. The trainer consumes the same sealed token
export, writes a PEFT-format adapter (`alpha = scale * rank`) that the same
`convert_lora_to_gguf.py` stage converts, and evaluates with `--metal`, so the
adapter loads through `--lora` without fusing a full model copy. The exporter
loads only the GGUF vocabulary, so a large model is not loaded twice. Weights
load lazily and are quantized to 4 bits as they stream in, so Gemma 4 12B trains
in about 13 GiB. Gemma 4 checkpoints (`gemma4_unified`) need mlx-lm from main
(ml-explore/mlx-lm#1349), which requires Transformers 5.7+; the pinned llama.cpp
converters ran in that same environment (Transformers 5.17). A pinned-data run on
Gemma 4 12B IT is recorded in
[jev-lora-mlx-gemma4-12b-20260930](../benchmarks/jev-lora-mlx-gemma4-12b-20260930/README.md).
Use a custom `--profiles` registry for checkpoints outside the bundled profiles.

`--eval-execution parallel` scores each case's decisions together instead of
one fresh call each. It changes probabilities slightly (at most 0.05 measured,
top-1 unchanged), so compare base and adapter under the same mode. The speedup
comes from sharing the state prefix and therefore needs `state-first` prompts;
with the pilot's `legacy` layout it measured no faster than `fresh`. Batching
several training examples per forward pass was also measured and rejected:
on Gemma 4 12B it was 2.6× slower and, because MLX quantized matmuls round
differently by batch shape, changed the loss.

Download the pinned dataset's `all/train-00000-of-00001.parquet` and
`all/test-00000-of-00001.parquet` to a source directory as `train.parquet` and
`test.parquet`. Retain the dataset card there as `README.md`.

```sh
l2s1-train prepare --source results/jev-source --output results/jev-data

# Explicit opt-in download; this does not train or change application defaults.
l2s1-train fetch --models all \
  --checkpoint-root results/jev-checkpoints --output results/jev-fetch

cargo build --release --features llama-cuda \
  --example export_decision_tokens --example evaluate_jsonl

l2s1-train run --models all \
  --checkpoint-root results/jev-checkpoints --data results/jev-data \
  --gguf-root models --output results/jev-matrix \
  --converter /path/to/matching/llama.cpp/convert_lora_to_gguf.py \
  --library-path /path/to/matching/llama.cpp/lib \
  --rules-fixture tests/fixtures/decision_benchmark.json
```

Use `--models gemma4` for one model, or list any subset. Checkpoints live under
`<checkpoint-root>/<pinned-revision>/`. Model weights remain local and are not
copied into Git. Fetch may require the user's existing Hugging Face access for
restricted checkpoints; no credentials are embedded or requested by the script.
Each output directory must be new. A stage has a default two-hour timeout;
change it with `--stage-timeout SECONDS`. Live status is atomically persisted
before and after each stage. Errors and timeouts retain logs and continue to
the next model. Ctrl-C/SIGTERM stop the active child process, mark it cancelled,
and leave remaining models `not_started`. Exit codes: 0 complete, 1 failure or
timeout, 2 invalid command arguments, 130 cancellation. Hard process kills or
power loss leave the last `running` status for diagnosis; they cannot be recorded
as graceful cancellation. Automatic resumption is not supported; use a new output
directory. Inspect `<model>/run.json`, stage logs and
`summary.json` for failures; `status: ok` means completion, not improvement.
The optional rules fixture adds a paired regression run on the earlier 36
decisions. It is previously seen regression evidence, not another untouched test.

To add another supported causal model, supply `--profiles custom-models.json`
using the structure of `training/src/l2s1_training/jev_model_profiles.json`. Pin the exact HF
revision and matching GGUF, choose the safe Auto loader and real adapter target
modules. Remote Python code is disabled. Unsupported token mappings, missing
adapter modules and failed GGUF conversions stop that model's run. Merely adding
a profile does not establish runtime support: execute smoke, conversion and
native evaluation before reporting a model as validated.

```sh
python -m unittest discover -s scripts -p test_jev_lora.py
# Run inside the training environment; uses tiny random CPU models, no downloads.
python -m unittest discover -s scripts -p test_jev_model_architectures.py
```

## Your own Jev data

Provide three nonempty JSONL files with independent case IDs and states. Each
line has `id`, `workflow`, `state`, `questions` and `gold`. `questions` maps unique
question IDs to Jev types; `gold` maps the same IDs to `type`, a string `label`,
and a full `probabilities` object. For Score, the optional numeric `score`
is the target expectation; when omitted, reporting derives it from the supplied
probabilities. Reports retain failed cases and exit nonzero if any case fails.
Probabilities must be finite, nonnegative,
and sum to one within 1e-4. Criteria and instructions must be nonempty strings.
Case/question IDs cannot contain `/`, which separates native training IDs.

```json
{"id":"train-1","workflow":"routing","state":{"items":2},"questions":{"multiple":{"type":"noul","instructions":"Are there multiple items?"}},"gold":{"multiple":{"type":"noul","label":"true","probabilities":{"false":0.1,"true":0.9}}}}
```

Choice uses a criteria object keyed by label; Noul uses `false`/`true`; Score uses
an ordered criteria array and zero-based string labels (`"0"`, `"1"`, ...).
See [`training/examples`](../training/examples) for all three types. The example
splits only demonstrate the file format, not a useful training dataset.

```sh
l2s1-train prepare --train my-data/train.jsonl \
  --development my-data/development.jsonl --test my-data/test.jsonl \
  --output results/my-jev-data
```

Use that directory as `run --data`. Preparation rejects duplicate IDs and exact
canonical states both within and across splits. Native inference files contain
only state and questions; training targets remain separate. Before training,
request content, split hashes, IDs, counts and the native token seal are verified.
The fixed protocol above applies to custom datasets too: one final epoch, no
holdout fitting or checkpoint selection. Editing a manifest to request an
unsupported protocol fails instead of silently ignoring parameters.

## Installation and compatibility checks

```sh
python -m pip install build
python -m build training
python -m pip install --force-reinstall --no-deps training/dist/l2s1_training-0.1.0-py3-none-any.whl
l2s1-train doctor --exporter target/release/examples/export_decision_tokens \
  --evaluator target/release/examples/evaluate_jsonl \
  --converter /path/to/llama.cpp/convert_lora_to_gguf.py
python -m unittest discover -s training/tests
```

Set `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows) for the matching native libraries
when checking binaries with `doctor`; `run --library-path` sets this for its child
processes. Doctor checks help flags, dependency imports and CUDA visibility; only
an actual smoke/conversion/evaluation establishes model support.
CI builds wheel and sdist, installs the wheel outside the source tree on Linux
and Windows, and checks the CLI, data contracts and failure lifecycle. Artifacts
contain source/tools only, never checkpoints or adapters. Historical pilot
reports retain their original code hashes and results; packaging does not claim
a new real-weight experiment. The pilot adapter remains opt-in because coverage
and the rules regression declined.

## Raspberry Pi inference

[Physical Pi 5 4GB evidence](../benchmarks/pi5-gemma3-20260927/README.md) is available
for **base Gemma3 1B Q8 CPU inference**, including Jev file rendering. Peak process
RSS was 2.10 GiB with no swap used; the rules test's p50 was 4.766 seconds per
three-decision request. Undervoltage/throttling occurred during the measurement.
Raw accuracy was 55.6% and accepted accuracy 54.5%; this is not a validated
Gemma3 adapter or a deployment-quality claim. The CUDA-only training runner still
requires a separate GPU machine. The Pi installation uses the lightweight package
for data/reporting and a separately built ARM64 CPU runtime for inference.
