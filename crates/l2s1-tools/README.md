# L2S1 experiment tools

[English](../../docs/en/crates/l2s1-tools/README.md) · [한국어](../../docs/ko/crates/l2s1-tools/README.md) · [日本語](../../docs/ja/crates/l2s1-tools/README.md)

[English index](../../docs/en/README.md) · [한국어 색인](../../docs/ko/README.md) · [日本語索引](../../docs/ja/README.md)

`l2s1-tools` is a repository-only Rust CLI for preparing frozen datasets, running the local GGUF evaluator, and independently recounting saved predictions. It does not link llama.cpp; commands that perform inference launch the repository's `evaluate_jsonl` example. Build it with:

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools --help
```

Run commands from the repository root, or pass `--root /path/to/L2S1`. The binary is excluded from crate publication. It writes generated data and reports to user-selected paths, normally under ignored `results/`. Existing output paths are generally rejected to preserve frozen evidence. Training and direct PyTorch model probes remain in Python.

| Former Python script | Rust subcommand |
| --- | --- |
| `benchmark_models.py` | `benchmark-models` |
| `kaggle_ag_news.py` | `kaggle-ag-news prepare/run/report` |
| `kaggle_airline.py` | `kaggle-airline prepare/run/report` |
| `calibrate_ag_news.py` | `calibrate-ag-news` |
| `prepare_decision_finetune.py` | `prepare-decision-finetune` |
| `prepare_accuracy_study.py` | `prepare-accuracy-study` |
| `prepare_output_head.py` | `prepare-output-head` |
| `evaluate_decision_lora.py` | `evaluate-decision-lora` |
| `evaluate_intents.py` | `evaluate-intents` |
| `evaluate_intents_wide.py` | `evaluate-intents-wide` |
| `analyze_intent_rotation.py` | `analyze-intent-rotation` |
| `jevbench_public.py` | `jevbench-public prepare/score/run` |
| `jevbench_matrix.py` | `jevbench-matrix download/run` |
| `laya_benchmark.py` | `laya-benchmark fetch/prepare/score/compare/run` |
| `probe_airline_order.py` | `probe-airline-order` |
| `report_accuracy_study.py` | `report-accuracy-study` |
| `report_airline.py` | `report-airline` |
| `report_decision_finetune.py` | `report-decision-finetune` |
| `report_intents.py` | `report-intents` |
| `report_intents_wide.py` | `report-intents-wide` |
| `report_jevbench_matrix.py` | `report-jevbench-matrix` |
| `report_output_head.py` | `report-output-head` |
| `summarize_output_head_runtime.py` | `summarize-output-head-runtime` |
| `tune_compute.py` | `tune-compute` |

The old Python files remain as comparison references and for training modules that import their helpers. Rust is the documented path for new data preparation, benchmarking, and reporting. The fixed accuracy-study seed reproduces all 18 frozen files byte-for-byte. AG News, airline, decision fine-tuning, output-head, and intent preparation were checked against existing local artifacts. JevBench and Laya preparation and saved-prediction scoring were also checked against pinned public data. These checks do not establish fresh model inference, network download, or GPU execution for the new CLI; those paths require the corresponding host and model files.

A few JSON reports differ only in floating-point reduction order or key formatting. Reports include separate denominators for coverage, failures, and candidate probabilities; a local synthetic/public score is not a production accuracy claim. JevBench uses a Rust transcription of the pinned public scorer and explicitly does not compute a full official composite score. Laya probe cases are a small frozen rendering of the pinned `probe.py` definitions; preparation verifies the downloaded source hash before using them.
