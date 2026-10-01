<a id="kaggle-ag-news-evaluation"></a>
# Kaggle AG ニュースの評価

[English](../en/KAGGLE_BENCHMARK.md) · [한국어](../ko/KAGGLE_BENCHMARK.md) · [日本語](KAGGLE_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

これは、合成ルール フィクスチャー ではなく、自然ニュース記事に関する既存の 判断 エンジンを評価します。これはゼロショット分類実験であり、訓練された Kaggle コンテストの提出物ではありません。

<a id="data-and-frozen-protocol"></a>
## データと凍結されたプロトコル

出典: Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) の [AG ニュース分類データセット、バージョン 2。ダウンロードしたアーカイブには、World、Sports、Business、Science/Technology の 4 つのクラスを持つ 120,000 トレーニング行と 7,600 テスト行が含まれています。

準備スクリプトは、ダウンロードされたアーカイブ SHA256 をチェックし、正規化されたタイトルと説明がトレーニング記事と一致するテスト記事を削除し、重複したテスト テキストを削除して、シード `20260921` を持つクラスごとに 100 テスト記事をサンプリングします。このアーカイブでは、 10 テスト記事がトレーニングと重なっています。凍結された評価には、400 の個別の記事と、25% の常に 1 クラスのベースラインが含まれます。

トレーニング例、デモンストレーション、ラベル、または期待される回答はモデルに送信されません。リクエストには、タイトル、説明、固定の指示、および同じ 4 つのカテゴリの説明のみが固定の順序で含まれます。予期されるラベルは、`selection.json` に個別に保存されます。サンプル、プロンプト、カテゴリの順序、およびしきい値は推論前に固定されており、結果には調整されませんでした。公開されている履歴データがモデルの事前トレーニングに使用されている可能性があります。このデータセットのトレーニング/テストの重複を削除しても、事前トレーニングの汚染がないことは証明されません。

各ローカル GGUF は、CUDA、コンテキスト 2048、バッチ 256、および 4 つの CPU スレッドを使用して、同じ 400 記事に対して 1 回実行されます。モデルは順次実行されます。デフォルトの 判断保留 しきい値は変更されていません: 最上位候補確率 0.8 および 全語彙 候補の確率質量 0.05。推論コードは変更されません。 Qwen3.8 は UD-IQ2_XXS を使用します。 Gemma 3 および 4 は Q8_0 を使用します。 GPT-OSS は MXFP4 を使用します。これらは異なるモデル サイズと量子化であり、元のモデル ファミリの制御された比較ではありません。

<a id="metrics"></a>
## メトリクス

- **Correct / all:** が受け入れた正しい予測をすべての 400 記事で割ったもの。判断保留、エラー、出力の欠落は分母に残ります。
- **Accepted 正解率:** 受け入れられた正しい予測を受け入れられた予測で割ったもの。すべての予測が判断保留された場合は未定義。
- **カバレッジ:** が受け入れた予測を 400 で割ったもの。
- **Raw トップ-1 正解率:** アプリケーションが判断保留した場合でも、ラベルに対する最も確率の高い候補。同点カウントが正しくありません。これは、アプリケーションが返す応答 正解率 ではありません。
- **混同マトリックス:** クラスごとの予測、判断保留、エラー、欠落した結果。
- **95% ウィルソン区間:** 報告された各割合の記述的不確実性 (このサンプル/プロトコルの条件付き)。他のドメインまたはモデル トレーニングの汚染について保証するものではありません。
- **Latency:** エンドツーエンドの `decide` 時間 (プロンプト構築、プレフィル、logits 転送、スコアリングを含む) (モデルの読み込みを除く)。最初の記事が収録されています。生成されたトークンは測定されません。

すべての記事の結果はすぐに JSONL にフラッシュされます。モデルには 30 分のタイムアウトがあります。部分的な結果では、不足している分母が保持されます。失敗した実行または不完全な実行を、完了したベンチマークとして提示してはなりません。既存の出力ファイルが上書きされることはありません。

<a id="reproduce"></a>
## 再現する

データと詳細な予測は、無視される `results/` ディレクトリに残ります。これらはソース リポジトリには含まれていません。 Kaggle から正確なアーカイブをダウンロードします。ハッシュがレビュー済みバージョンと異なる場合、準備は失敗します。

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

`--model gemma4` でモデルを 1 つだけ実行するか、`--model` を繰り返します。以前の結果を保存するには、`--folder` を持つ新しいディレクトリを使用します。 Rust コマンドは、`crates/l2s1-tools/src/kaggle_ag_news.rs` に記載されている既存のローカル モデル ファイル名を想定します。重みはダウンロードされません。

準備されたソース アーカイブ、個々の CSV、凍結されたリクエスト、選択された行 ID、モデル ファイル、実行可能ファイル、およびソース ファイルには、`results/kaggle-ag-news/` の SHA256 来歴があります。 `runtime.json` は、llama.cpp リビジョンとダーティ作業ツリーのステータスを記録します。 Git コミットだけでは、テストされたすべてのコードが記述されるわけではありません。 `runner-used.py` は、レポート生成コマンドが追加される前に、この実験に使用された正確なランナーを保存します。

評価者の検証:

```sh
python3 -m unittest discover -s scripts -p test_kaggle_ag_news.py -v
cargo clippy --release --locked --features llama-cuda --example evaluate_jsonl -- -D warnings
```

スコアリング テストは、分母の処理、全判断保留出力、重複/不明 ID、分割オーバーラップ除去、判断論的サンプリング、および信頼区間のエッジ ケースをカバーします。 [VERIFICATION.md](VERIFICATION.md) でモデル固有のチェックを個別に実行します。固定バッチ分類の実行が成功しても、実行モード間で一貫性は確立されません。
