# Typed-decisions: recorded full-test measurements

[English](en/TYPED_DECISIONS_BENCHMARK.md) · [한국어](ko/TYPED_DECISIONS_BENCHMARK.md) · [日本語](ja/TYPED_DECISIONS_BENCHMARK.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

On September 26, 2026, two checkpoints each completed all **400 test cases / 2,000 decisions** from [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90/all). These are synthetic, teacher-labelled workflow judgments, not human-labelled deployment outcomes. There were no inference errors or truncated outputs in either run.

The score is hard argmax agreement before abstention. It is separate from the JevBench public-subset score. The test split was not used for training, calibration fitting or prompt selection.

## Results

| Checkpoint | Raw top-1 | Coverage | Correct / accepted | Correct accepted / all | Accepted wrong | Case p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 1086/2000 (54.30%) | 92.75% | 55.69% | 1033/2000 (51.65%) | 822 | 322.64 / 370.98 |
| Qwen3 0.6B Q8_0 | 625/2000 (31.25%) | 55.60% | 34.35% | 382/2000 (19.10%) | 730 | 173.46 / 204.22 |

One case contains five decisions. Latency is completion of the entire case, **not** latency per question or amortized cost. Loading and one untimed warmup case are excluded. The default policy is top option probability >= 0.8, candidate mass >= 0.05, and no tie. Scores are uncalibrated.

These measurements show substantial overconfidence: raising a model-score threshold cannot by itself guarantee a requested real-world error rate. Accepted-only accuracy is slightly higher than raw accuracy for both models, but the accepted subsets still contain many errors: 822 for Gemma and 730 for Qwen.

## By decision kind

| Checkpoint | Kind | Raw correct / planned | Coverage | Correct / accepted |
| --- | --- | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | binary | 304/600 (50.67%) | 99.83% | 50.75% |
| Gemma 4 E2B Q8_0 | choice | 337/600 (56.17%) | 94.83% | 57.64% |
| Gemma 4 E2B Q8_0 | ordinal | 445/800 (55.62%) | 85.88% | 58.37% |
| Qwen3 0.6B Q8_0 | binary | 264/600 (44.00%) | 62.33% | 42.25% |
| Qwen3 0.6B Q8_0 | choice | 161/600 (26.83%) | 61.50% | 33.88% |
| Qwen3 0.6B Q8_0 | ordinal | 200/800 (25.00%) | 46.12% | 26.83% |

The 2,000 decisions comprise 600 binary, 600 choice and 800 ordinal questions. Binary raw labels use p_true >= 0.5; choice and ordinal ties use original option order. Native accepted correctness scores the actual selected value, independently of that raw tie rule.

## By workflow

| Checkpoint | Workflow | Raw correct / planned |
| --- | --- | ---: |
| Gemma 4 E2B Q8_0 | agent_trace_observability | 239/500 (47.80%) |
| Gemma 4 E2B Q8_0 | customer_service | 328/500 (65.60%) |
| Gemma 4 E2B Q8_0 | invoice_processing | 222/500 (44.40%) |
| Gemma 4 E2B Q8_0 | security_incidents | 297/500 (59.40%) |
| Qwen3 0.6B Q8_0 | agent_trace_observability | 170/500 (34.00%) |
| Qwen3 0.6B Q8_0 | customer_service | 155/500 (31.00%) |
| Qwen3 0.6B Q8_0 | invoice_processing | 186/500 (37.20%) |
| Qwen3 0.6B Q8_0 | security_incidents | 114/500 (22.80%) |

## Environment and provenance

Both runs used the same frozen native evaluator on NVIDIA GeForce RTX 3080 10 GiB, driver 596.21, an Intel Core i9-9900K under WSL2, and four CPU threads. Settings: CUDA full offload, context 8192, batch/microbatch 256, Flash Attention off, legacy/minimal prompt, fresh serial execution, read loading, no LoRA, output head, learned calibration or preparation cache. Models were run sequentially. Existing display/host GPU use was present; these are local single-run timings.

The evaluator was frozen from the `fb3ad4e` source before optional thinking-mode implementation. These full-test numbers are **direct mode** measurements. Thinking mode has separate bounded generation and demo checks; no full-test thinking accuracy gain is claimed. Final source changes do not retroactively change the measured binary.

- Evaluator SHA-256: `7fb6e5237188de8161a3025f6b0ce7e16e7b2e3274520a19879e95a115d221ba`.
- Dataset Parquet SHA-256: `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`.
- Request JSONL SHA-256: `a677142b77f72f4445d79d10213d79e7de564f223553bb3a4c88105a181f7e6d`.
- Provenance: [manifest](../benchmarks/typed-decisions-20260926/manifest.json). The checked-in `gemma4-e2b-run.json` and `qwen3-06b-run.json` contain the original commands and model hashes. The public website manifest also embeds these two run records with portable command paths.
- Public downloads: [Gemma summary JSON](../benchmarks/typed-decisions-20260926/gemma4-e2b-summary.json), [Qwen summary JSON](../benchmarks/typed-decisions-20260926/qwen3-06b-summary.json), [Gemma 2,000 scored decisions](../benchmarks/typed-decisions-20260926/gemma4-e2b-scored.jsonl), [Qwen 2,000 scored decisions](../benchmarks/typed-decisions-20260926/qwen3-06b-scored.jsonl), and the manifest above. Scored records contain probabilities, raw/native selections, correctness, mass, workflow and latency; they contain no original state or model-generated thought text. Public summaries and scored JSONL preserve the checked-in bytes; the public manifest embeds run metadata and replaces absolute native-library and command paths with basenames. Its `public_export` field records these changes; command flags and hashes are preserved.
- Full native prediction JSONL and logs remain in ignored `results/typed-decisions-20260926/`. These raw files and the frozen executable are not bundled in the repository.

Summaries additionally report hard Brier, hard NLL (natural logs, probability floor 1e-12), ECE using 10 and 15 equal-width max-probability bins, soft-target Brier/TVD/KL for binary/choice, and expected-level MAE/within-one for ordinal. The ECE definition is stated explicitly; equality with another project's ECE implementation is not assumed.

## Ollaya reference figures

[Ollaya's published GGUF study](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/families/llm-logits.md#measured-quality-typed-decisions-test-400-rows--2000-questions) reports 56.6% for Gemma 4 E2B Q8_0, 71.7% for Gemma 4 12B Q4_0, and 59.1% for decider-2B on its 400-case / 2,000-question typed test. These are **source-reported reference figures**, not Ollaya runs performed here. Prompt, runtime, calibration and evaluator protocols differ; the source does not establish byte-identical requests or a controlled latency comparison with L2S1.

Our Gemma E2B result is 54.3%. This evidence does not establish a L2S1 accuracy advantage. Models of different sizes, task-specific fine-tunes, and independently measured latencies should not be presented as a runtime-only comparison.

## Reproduce

Use the fetch/prepare/run/score commands in [the task adapter guide](LAYA_BENCHMARK.md#prepare-and-run). Run the full `typed` suite without `--limit` or extra repeats. Preserve one resident checkpoint, the recorded parameters and untouched test labels.

```sh
target/release/l2s1-tools laya-benchmark run \
  --prepared results/laya/typed --output results/laya/typed-gemma-e2b \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
  --context 8192 --batch 256 --threads 4 --model-load-mode read
```

Build the evaluator with `llama-cuda` and make its matching native libraries available. Reproduction from a newer source is a new measurement, rather than verification of the frozen executable's timings.
