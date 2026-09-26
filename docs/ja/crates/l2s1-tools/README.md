<a id="l2s1-experiment-tools"></a>
# L2S1 実験ツール

[English](../../../en/crates/l2s1-tools/README.md) · [한국어](../../../ko/crates/l2s1-tools/README.md) · [日本語](README.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

`l2s1-tools` は、凍結データセットを準備し、ローカル GGUF エバリュエーターを実行し、保存された予測を個別に再カウントするためのリポジトリ専用 Rust CLI です。 llama.cpp にはリンクしません。推論を実行するコマンドは、リポジトリの `evaluate_jsonl` サンプルを起動します。以下を使用して構築します。

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools --help
```

リポジトリ ルートからコマンドを実行するか、`--root /path/to/L2S1` を渡します。バイナリはクレートの公開から除外されます。生成されたデータとレポートは、通常は無視される `results/` の下でユーザーが選択したパスに書き込まれます。既存の出力パスは通常、凍結された証拠を保存するために拒否されます。トレーニングおよび直接の PyTorch モデル プローブは Python に残ります。

| 従来の Python スクリプト | Rust サブコマンド |
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

古い Python ファイルは、比較参照として、およびヘルパーをインポートするトレーニング モジュールとして残ります。 Rust は、新しいデータの準備、ベンチマーク、レポートの文書化されたパスです。修正された 正解率 スタディ シードは、すべての 18 凍結ファイルをバイト単位で再現します。 AG ニュース、航空会社、判断 微調整、出力 ヘッド、およびインテントの準備が、既存のローカル アーティファクトに対してチェックされました。 JevBench および Laya の準備と保存された予測スコアも、ピン留めされた公開データに対してチェックされました。これらのチェックでは、新しい CLI の新しいモデル推論、ネットワーク ダウンロード、または GPU の実行は確立されません。これらのパスには、対応するホスト ファイルとモデル ファイルが必要です。

いくつかの JSON レポートは、浮動小数点のリダクション順序またはキーの形式のみが異なります。レポートには、採用率、失敗、および候補の確率の個別の分母が含まれます。ローカル合成/公開スコアは、本番環境 正解率 クレームではありません。 JevBench は、固定された公開スコアラーの Rust 転写を使用し、完全な公式総合スコアを明示的に計算しません。 Laya プローブ ケースは、固定された `probe.py` 定義の小さな凍結レンダリングです。準備では、ダウンロードしたソース ハッシュを使用する前に検証します。
