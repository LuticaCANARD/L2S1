<a id="typed-decisions-recorded-full-test-measurements"></a>
# Typed-decisions: 全テストの測定記録

[English](../en/TYPED_DECISIONS_BENCHMARK.md) · [한국어](../ko/TYPED_DECISIONS_BENCHMARK.md) · [日本語](TYPED_DECISIONS_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

2026 年 9 月 26 日、2 つのチェックポイントがそれぞれ [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90/all) の **400 テストケース / 2,000 判断**の全体を完了しました。これは教師モデルがラベルを付けた合成ワークフロー判断で、人がラベルを付けた実運用の結果ではありません。両実行に推論エラーや出力の切り詰めはありませんでした。

スコアは判断保留前の hard argmax 一致率です。JevBench 公開サブセットのスコアとは別です。テスト分割は学習、校正の学習、プロンプト選択に使っていません。

<a id="results"></a>
## 結果

| チェックポイント | ポリシー適用前の top-1 | 採用率 | 正解 / 採用 | 採用した正解 / 全体 | 採用した不正解 | ケース p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 1086/2000 (54.30%) | 92.75% | 55.69% | 1033/2000 (51.65%) | 822 | 322.64 / 370.98 |
| Qwen3 0.6B Q8_0 | 625/2000 (31.25%) | 55.60% | 34.35% | 382/2000 (19.10%) | 730 | 173.46 / 204.22 |

1 ケースには 5 判断があります。レイテンシはケース全体の完了時間で、**質問ごとの時間や分割平均コストではありません。** 読み込みと時間を測らないウォームアップ 1 ケースは除外します。デフォルトは最上位候補確率 >= 0.8、候補質量 >= 0.05、同点なしです。スコアは未校正です。

測定は大きな過信を示します。モデルスコアのしきい値を上げるだけで、要求する実エラー率を保証できません。両モデルとも採用判断の正解率は未加工の正解率より少し高いですが、採用した部分集合にも Gemma 822、Qwen 730 の不正解があります。

<a id="by-decision-kind"></a>
## 判断 種類別

| チェックポイント | 種類 | ポリシー適用前の正解 / 計画 | 採用率 | 正解 / 採用 |
| --- | --- | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | binary | 304/600 (50.67%) | 99.83% | 50.75% |
| Gemma 4 E2B Q8_0 | choice | 337/600 (56.17%) | 94.83% | 57.64% |
| Gemma 4 E2B Q8_0 | ordinal | 445/800 (55.62%) | 85.88% | 58.37% |
| Qwen3 0.6B Q8_0 | binary | 264/600 (44.00%) | 62.33% | 42.25% |
| Qwen3 0.6B Q8_0 | choice | 161/600 (26.83%) | 61.50% | 33.88% |
| Qwen3 0.6B Q8_0 | ordinal | 200/800 (25.00%) | 46.12% | 26.83% |

2,000 判断は binary 600、choice 600、ordinal 800 です。二値の未加工ラベルは p_true >= 0.5、choice・ordinal の同点は元の候補順序を使います。ネイティブの採用正解は、この未加工の同点規則とは独立に実際の選択値を採点します。

<a id="by-workflow"></a>
## ワークフロー別

| チェックポイント | ワークフロー | ポリシー適用前の正解 / 計画 |
| --- | --- | ---: |
| Gemma 4 E2B Q8_0 | agent_trace_observability | 239/500 (47.80%) |
| Gemma 4 E2B Q8_0 | customer_service | 328/500 (65.60%) |
| Gemma 4 E2B Q8_0 | invoice_processing | 222/500 (44.40%) |
| Gemma 4 E2B Q8_0 | security_incidents | 297/500 (59.40%) |
| Qwen3 0.6B Q8_0 | agent_trace_observability | 170/500 (34.00%) |
| Qwen3 0.6B Q8_0 | customer_service | 155/500 (31.00%) |
| Qwen3 0.6B Q8_0 | invoice_processing | 186/500 (37.20%) |
| Qwen3 0.6B Q8_0 | security_incidents | 114/500 (22.80%) |

<a id="environment-and-provenance"></a>
## 環境と来歴

両実行は同じ固定ネイティブ評価器を使用しました。NVIDIA GeForce RTX 3080 10 GiB、ドライバー 596.21、WSL2 上の Intel Core i9-9900K、CPU 4 スレッドです。設定は CUDA 全オフロード、コンテキスト 8192、バッチ・マイクロバッチ 256、Flash Attention 無効、legacy/minimal プロンプト、fresh 直列実行、read 読み込みです。LoRA、出力ヘッド、学習済み校正、準備キャッシュは使いません。モデルは順番に実行しました。既存の画面・ホストの GPU 使用があり、ローカルの単発測定です。

評価器は任意の thinking 実装前の `fb3ad4e` ソースで固定しました。全テスト数値は **direct モード**の測定です。Thinking は別途上限付き生成とデモの確認があり、全テストの正解率向上は主張しません。最終ソース変更は測定済みバイナリーを遡って変えません。

- 評価者 SHA-256: `7fb6e5237188de8161a3025f6b0ce7e16e7b2e3274520a19879e95a115d221ba`。
- データセット Parquet SHA-256: `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`。
- リクエスト JSONL SHA-256: `a677142b77f72f4445d79d10213d79e7de564f223553bb3a4c88105a181f7e6d`。
- 来歴: [manifest](../../benchmarks/typed-decisions-20260926/manifest.json)。チェックインされた `gemma4-e2b-run.json` および `qwen3-06b-run.json` には、元のコマンドとモデル ハッシュが含まれています。公開 Web サイトのマニフェストには、これら 2 つの実行レコードと移植可能なコマンド パスも埋め込まれています。
- パブリック ダウンロード: [Gemma 概要 JSON](../../benchmarks/typed-decisions-20260926/gemma4-e2b-summary.json)、[Qwen 概要 JSON](../../benchmarks/typed-decisions-20260926/qwen3-06b-summary.json)、 [Gemma 判断 2,000 件の採点記録](../../benchmarks/typed-decisions-20260926/gemma4-e2b-scored.jsonl)、[Qwen 判断 2,000 件の採点記録](../../benchmarks/typed-decisions-20260926/qwen3-06b-scored.jsonl)、および上記のマニフェスト。スコアリングされたレコードには、確率、生/ネイティブ の選択、正確さ、質量、ワークフロー、レイテンシーが含まれます。これらには、元の状態やモデルによって生成された思考テキストは含まれません。公開サマリーとスコア付き JSONL では、チェックインされたバイトが保持されます。パブリック マニフェストには実行メタデータが埋め込まれ、絶対 ネイティブ ライブラリとコマンド パスがベース名に置き換えられます。その `public_export` フィールドには、これらの変更が記録されます。コマンドのフラグとハッシュは保持されます。
- 完全な ネイティブ 予測 JSONL とログは無視された `results/typed-decisions-20260926/` のままになります。これらの生ファイルと凍結された実行可能ファイルはリポジトリにバンドルされていません。

要約は hard Brier、hard NLL（自然対数、確率下限 1e-12）、最大確率の等幅 10・15 ビンによる ECE、binary・choice の soft-target Brier/TVD/KL、ordinal の期待レベル MAE・within-one も報告します。ECE 定義は明記され、他プロジェクトの実装との同一性は仮定しません。

<a id="ollaya-reference-figures"></a>
## Ollaya 参考数値

[Ollaya の公開 GGUF 評価](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/families/llm-logits.md#measured-quality-typed-decisions-test-400-rows--2000-questions)は、独自の 400 ケース / 2,000 質問 typed テストで、Gemma 4 E2B Q8_0 56.6%、Gemma 4 12B Q4_0 71.7%、decider-2B 59.1% を報告します。これは**参照元に報告された参考数値**で、ここで実行した Ollaya 測定ではありません。プロンプト・ランタイム・校正・評価手順が異なり、L2S1 と同一バイトのリクエストや条件を揃えたレイテンシ比較を示しません。

こちらの Gemma E2B 結果は 54.3% です。この根拠は L2S1 の正解率優位を実証しません。規模の異なるモデル、タスク別の調整、独立したレイテンシ測定を、ランタイムだけの比較として示してはいけません。

<a id="reproduce"></a>
## 再現する

[タスクアダプターのガイド](LAYA_BENCHMARK.md#prepare-and-run)の fetch・prepare・run・score コマンドを使います。`--limit` や追加反復なしで全 `typed` 評価を実行します。1 つの常駐チェックポイント、記録されたパラメーター、変更しないテストラベルを維持します。

```sh
target/release/l2s1-tools laya-benchmark run \
  --prepared results/laya/typed --output results/laya/typed-gemma-e2b \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
  --context 8192 --batch 256 --threads 4 --model-load-mode read
```

`llama-cuda` で評価器をビルドし、対応するネイティブライブラリを提供します。新しいソースからの再現は新しい測定で、固定バイナリーの時間の検証ではありません。
