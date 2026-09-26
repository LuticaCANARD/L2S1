<a id="caltech-101-30-way-vision-decision-benchmark"></a>
# Caltech-101: 30 ウェイ ビジョン 判断 ベンチマーク

[English](../../../en/benchmarks/caltech101-vision-20260924/README.md) · [한국어](../../../ko/benchmarks/caltech101-vision-20260924/README.md) · [日本語](README.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

これは、L2S1 HTTP ビジョン API の探索的なゼロショット画像分類実行です。これは、26 オプションの 1 トークン アルファベット制限を超えて、前のビジョン PR で導入された 30 オプション コード シーケンス パスを実行します。

<a id="data-and-protocol"></a>
## データとプロトコル

- 出典: [Caltech-101 Kaggle ミラー](https://www.kaggle.com/datasets/imbikramsaha/caltech-101)、出典 ZIP SHA-256 `c29bfcb9f72b1b03bdfb6b871907ba08ec436bd4049870c9e0424a6f489d4855`。
- 推論前に、`scripts/prepare_caltech101_vision.py` はシード `20260924` を使用して、101 オブジェクト クラスの 30 をサンプリングし、クラスごとに 5 つのイメージをサンプリングしました。 `BACKGROUND_Google` のみが除外されました。サンプル ZIP SHA-256 は `421e1a1ad05a288a57ad5e837a4f22d1941854ca561fe67a2b1be0131de4b1de` です。
- 元の JPEG バイトは、`image_base64` として `POST /v1/decisions` に 1 つずつ送信されました。各リクエストでは、アルファベット順に並べられた同じ 30 オプションが使用されました。画像ファイル名も 正解 もモデルに送信されませんでした。ウォームアップ要求は使用されませんでした。
- このミラーは、この実行に対してホールドアウト スプリットを提供しません。これらの 150 画像は、提供されたアーカイブからサンプリングされたものです。事前トレーニングの重複は不明です。結果は、一般的な分類パフォーマンスではなく、この固定サンプルとプロンプトを測定します。バランスのとれたランダム選択ベースラインは、1/30 (3.33%) です。

`selection.json` は標本に含まれる全ファイル名と画像ハッシュを固定します。`observations.jsonl` には各 HTTP 応答の候補スコア 30 個、選択した候補、生の最上位候補、レイテンシー、判断保留の理由が入ります。画像とモデル重みは Git に含めません。

<a id="runtime"></a>
## ランタイム

| 項目 | 値 |
| --- | --- |
| コード | `7e33eca` (ビジョンワイドコード) |
| モデル | Gemma 4 E2B IT Q8_0 GGUF、SHA-256 `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a` |
| プロジェクター | 一致するマルチモーダル GGUF、SHA-256 `9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292` |
| GPU | NVIDIA GeForce RTX 3060、CUDA オフロードがすべての HTTP 応答で確認されました |
| 推論 | 新しいリクエスト コンテキスト、コンテキスト 4096、バッチ 256、 4 CPU スレッド、モデル ロード モード `read`、デフォルト ポリシー (`min_candidate_mass=0.05`、`min_top_probability=0.8`) |
| 実行 | 一度に 1 つのローカル HTTP リクエスト。起動はリクエストのレイテンシーから除外されます |

GPU は 3855 MiB を使用し、実行の終了近くに 74 °C を報告しました。サーバーは、5014 ms の後、`/healthz` に到達しました。これらは単一実行の観察です。

<a id="results"></a>
## 結果

| 指標 | 結果 |
| --- | ---: |
| 画像・クラス | 150 / 30 (それぞれ 5) |
| すべての画像を修正します | 122/150 (81.33%) |
| 採用率 | 148/150 (98.67%) |
| 承認された判断のうち 正解率 | 122/148 (82.43%) |
| 判断保留 | 2/150、パゴダ画像上の両方の `low_top_probability` |
| 生のトップ-1、判断保留ポリシーを無視 | 122/150 (81.33%) |
| ロードされたモデルの HTTP レイテンシ | 196.609 msを意味します。 p50 195.282 ms; p95 最も近いランク 200.662 ms |
| コードプレフィックスの評価 | 1 per image for all 150 requests |

クラスごとのカウントは `summary.json` です。このサンプルでは、​​`flamingo_head`、`kangaroo`、および `okapi` の 3 つのクラスのスコアが 0/5 でした。最初のリクエストはレイテンシ (最大 308.351 ms) に含まれます。以前の 2 クラスの猫/犬のスコアとの比較は示唆されていません。選択肢とサンプルは大幅に異なります。

<a id="reproduce"></a>
## 再現する

Kaggle ミラーを ZIP としてダウンロードし、上記のソース ハッシュを確認します。リポジトリ ルートから、CUDA 対応の `l2s1` バイナリと一致する GGUF ファイルを使用して:

```sh
python3 scripts/prepare_caltech101_vision.py --source /path/to/dataset.zip --output /path/to/sample-dir
python3 scripts/benchmark_caltech101_vision.py \
  --selection /path/to/sample-dir/selection.json \
  --sample /path/to/sample-dir/sample-30x5.zip \
  --binary /path/to/l2s1 --model /path/to/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --output /path/to/new-result-dir
```

ベンチマークは既存の出力ディレクトリの再利用を拒否し、スコア付けする前にサンプル ZIP とイメージごとの SHA-256 ハッシュを検証します。生の観察結果、概要、サーバー ログが出力ディレクトリに書き込まれます。再実行を比較するには、最初にモデル/プロジェクター/ソースのハッシュと正確なクラスの順序を確認します。
