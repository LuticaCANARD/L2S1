# Kaggle AG News evaluation

[English](en/KAGGLE_BENCHMARK.md) · [한국어](ko/KAGGLE_BENCHMARK.md) · [日本語](ja/KAGGLE_BENCHMARK.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

This evaluates the existing decision engine on natural news articles rather than the synthetic rule fixture. It is a zero-shot classification experiment, not a trained Kaggle competition submission.

## Data and frozen protocol

Source: [AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset), version 2. The downloaded archive contains 120,000 training rows and 7,600 test rows, with four classes: World, Sports, Business, and Science/Technology.

The preparation script checks the downloaded archive SHA256, removes test articles whose normalized title and description match training articles, removes duplicate test texts, and samples 100 test articles per class with seed `20260921`. In this archive, 10 test articles overlapped with training. The frozen evaluation has 400 distinct articles and an always-one-class baseline of 25%.

No training examples, demonstrations, labels, or expected answers are sent to the model. Requests contain only title, description, a fixed instruction, and the same four category descriptions in a fixed order. Expected labels are stored separately in `selection.json`. The sample, prompt, category order, and thresholds were frozen before inference and were not tuned on the results. Public historical data may have appeared in model pretraining; removing this dataset's train/test overlap does not establish absence of pretraining contamination.

Each local GGUF runs once over the same 400 articles using CUDA, context 2048, batch 256, and four CPU threads. Models run sequentially. Default abstention thresholds are unchanged: top candidate probability 0.8 and full-vocabulary candidate mass 0.05. The inference code is unchanged. Qwen3.8 uses UD-IQ2_XXS; Gemma 3 and 4 use Q8_0; GPT-OSS uses MXFP4. These are different model sizes and quantizations, not a controlled comparison of original model families.

## Metrics

- **Correct / all:** accepted correct predictions divided by all 400 articles. Abstentions, errors, and missing outputs remain in the denominator.
- **Accepted accuracy:** accepted correct predictions divided by accepted predictions. Undefined when all predictions abstain.
- **Coverage:** accepted predictions divided by 400.
- **Raw top-1 accuracy:** highest-probability candidate against the label even if the application abstains; ties count incorrect. This is not the application's returned-answer accuracy.
- **Confusion matrix:** per-class predictions, abstentions, errors, and missing outcomes.
- **95% Wilson interval:** descriptive uncertainty for each reported proportion, conditional on this sample/protocol; not a guarantee about other domains or model training contamination.
- **Latency:** end-to-end `decide` time, including prompt construction, prefill, logits transfer, and scoring, excluding model loading. The first article is included. No generated tokens are measured.

Every article outcome is flushed to JSONL immediately. A model has a 30-minute timeout; partial results retain their missing denominator. Failed or incomplete runs must not be presented as completed benchmarks. Existing output files are never overwritten.

## Reproduce

Data and detailed predictions stay under the ignored `results/` directory; they are not included in the source repository. Download the exact archive from Kaggle; preparation fails if its hash differs from the reviewed version.

```sh
mkdir -p results/kaggle-ag-news
curl --fail --location \
  https://www.kaggle.com/api/v1/datasets/download/amananandrai/ag-news-classification-dataset \
  --output results/kaggle-ag-news/dataset.zip
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools kaggle-ag-news prepare

cargo build --release --locked --features llama-cuda --example evaluate_jsonl
target/release/l2s1-tools kaggle-ag-news run
target/release/l2s1-tools kaggle-ag-news report
```

Run only one model with `--model gemma4`, or repeat `--model`. Use a fresh directory with `--folder` to preserve earlier results. The Rust command expects the existing local model filenames documented in `crates/l2s1-tools/src/kaggle_ag_news.rs`; it does not download weights.

The prepared source archive, individual CSVs, frozen requests, selected row IDs, model files, executable, and source files have SHA256 provenance in `results/kaggle-ag-news/`. `runtime.json` records the llama.cpp revision and dirty working-tree status; the Git commit alone does not describe all tested code. `runner-used.py` preserves the exact runner used for this experiment, before the report-generation command was added.

Validation of the evaluator:

```sh
python3 -m unittest discover -s scripts -p test_kaggle_ag_news.py -v
cargo clippy --release --locked --features llama-cuda --example evaluate_jsonl -- -D warnings
```

Scoring tests cover denominator handling, all-abstained outputs, duplicate/unknown IDs, split overlap removal, deterministic sampling, and confidence interval edge cases. Run the model-specific checks in [VERIFICATION.md](VERIFICATION.md) separately; a successful fixed-batch classification run does not establish consistency across execution modes.
