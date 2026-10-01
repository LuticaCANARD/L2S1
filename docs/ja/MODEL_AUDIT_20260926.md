# モデル実測データの再検証 — 2026-09-26

[English](../en/MODEL_AUDIT_20260926.md) · [한국어](../ko/MODEL_AUDIT_20260926.md) · [日本語](../ja/MODEL_AUDIT_20260926.md)

2026-09-23〜25に記録した測定の元データを再検証した文書です。新しい推論実行ではありません。コード表記の証拠パスはリポジトリ基準のローカル資料です。公開集計と既存のレポートへのリンクを示します。

Rawはポリシー適用前の最上位候補の正答/全件、受理精度は受理正答/受理件数、coverageは受理/全件です。表は元の件数を併記します。特記がなければp50の単位はmsです。

## 同一条件のRTX 3060比較 — 2026-09-23

完了した22設定 × JevBench公開231問、計5,082応答を再集計しました。公開一覧は20設定 / 4,620応答で、raw正答2,924、受理3,122、受理正答2,279、受理誤答843、保留1,498、エラー0です。元の実行設定は23件で、22件が完了しました。既定のGPT-OSS CUDA Graph設定は129応答後に失敗し、Graphを無効にした別設定は完了しました。

| モデル | Raw正答 / 231 | 正答 / 受理 | Coverage | p50 ms | GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B Q8_0 | 184/231 (79.65%) | 129/138 (93.48%) | 138/231 (59.74%) | 86.88 | 5039 |
| Qwen3.5 9B Q4_K_M | 180/231 (77.92%) | 154/167 (92.22%) | 167/231 (72.29%) | 130.55 | 5637 |
| Gemma4 E4B Q4_K_M | 179/231 (77.49%) | 164/194 (84.54%) | 194/231 (83.98%) | 86.62 | 3989 |
| Qwen3.5 4B Q4_K_M | 176/231 (76.19%) | 130/140 (92.86%) | 140/231 (60.61%) | 88.47 | 3381 |
| Gemma4 E2B Q8_0 | 157/231 (67.97%) | 151/209 (72.25%) | 209/231 (90.48%) | 46.79 | 3025 |
| Qwen3.5 2B Q8_0 | 144/231 (62.34%) | 83/98 (84.69%) | 98/231 (42.42%) | 40.99 | 2465 |

この比較ではQwen3.5 4B Q8のraw精度が最も高く、受理138件のうち129件が正答です。Gemma4 E4B Q4はほぼ同じp50と1,050 MiB少ないGPU最大標本で、受理194件中164件の正答を返します。Qwen3.5 9B Q4は受理精度92.22%とcoverage72.29%を両立します。Gemma4 E2B Q8はraw67.97%、p50 46.79 msです。

完了した22設定はrequest/evaluator hash、RTX 3060 12 GiB、context 8192、batch/ubatch 256、CPU threads 4、FlashAttention off、fresh/legacy prompt、内蔵モデルテンプレート、ポリシー0.8 / 0.05を共有します。時間はRustの直列判定で、モデル読込とウォームアップ1回を除きます。GPU MiBは読込とウォームアップを含む200 ms間隔のボード全体使用量の最大標本です。GPT-OSSは`GGML_CUDA_DISABLE_GRAPHS=1`を使用します。

Raw精度、受理件数、多クラスBrier、10区間ECE、直列p50/p95、request hash、GPU CSVの最大値は記録済み集計と一致します。ダウンロード終了後の別実行では、Qwen3.5 4B Q8とGemma4 E2Bの全候補確率が一致し、p50は86.99 / 46.08 msでした。元の比較時間とは別に保持します。

[公開RTX 3060集計](../../benchmarks/rtx3060-20260926/summary.json) · [RTX 3060レポート](RTX3060_BENCHMARK.md)

Qwen3.5 9BのQ4_K_MとQ8_0は、raw正答がともに180/231 (77.92%)です。GPUボード最大標本は5,637 / 8,817 MiBで、この実行ではQ4が36.07%少なくなっています。集計精度の比較であり、出力分布の同一性を示すものではありません。

## Gemma 31B / 26BのCPU・GPU混合配置 — 2026-09-23

この2実行は同じJevBench 231件のrequestとevaluatorを共有します。evaluatorは上のモデル比較と異なります。context 8192、batch/ubatch 256、CPU threads 8、fresh/legacy prompt、read読込、ポリシー0.8 / 0.05です。

| モデル / 配置 | Raw正答 / 231 | 正答 / 受理 | Coverage | p50 / p95 ms | ボードGPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 31B Q4_K_M / GPU 24 | 207/231 (89.61%) | 206/228 (90.35%) | 228/231 (98.70%) | 2312.37 / 24590.96 | 10897 |
| Gemma4 26B A4B UD-Q4_K_M / CPU expert 18 | 196/231 (84.85%) | 191/220 (86.82%) | 220/231 (95.24%) | 653.19 / 7338.46 | 10557 |

Gemma31Bは反復層23個と出力層をGPU、反復層37個をCPUに配置します。全実行のengine peak RSSは15.92 GiB、プロセスGPU割当の最大標本は10.60 GiB、プロセスswap標本は0です。表のボードメモリとプロセス割当を区別し、RSSにVRAMは含みません。Gemma26BはCPU expert 18層を使用します。warm filesystemで読込方式を交互に実行した6回の検査では、peak RSS中央値が`auto` 16.17 GiB / `read` 9.35 GiBでした。これは読込方式のRSS比較です。

## 全ラベルのintent分類 — 2026-09-23

RTX 3060で4モデル × BANKING77英語200件 / MASSIVE韓国語200件、計1,600応答を評価しました。各requestに77 / 60個の全ラベルを渡します。Raw、受理件数、p50はnative元データと一致します。context 8192、batch/ubatch 256、CPU threads 4、fresh/legacy、FlashAttention off、ポリシー0.8 / 0.05で、モデル読込とウォームアップを除きます。Intent evaluatorはJevBench evaluatorsとは別です。

| モデル | データセット | Raw正答 / 200 | 正答 / 受理 | Coverage | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 E2B Q8_0 | BANKING77 en | 123/200 (61.50%) | 119/181 (65.75%) | 181/200 (90.50%) | 214.15 |
| Gemma4 E2B Q8_0 | MASSIVE ko | 103/200 (51.50%) | 100/181 (55.25%) | 181/200 (90.50%) | 157.52 |
| Gemma4 E4B Q8_0 | BANKING77 en | 130/200 (65.00%) | 127/173 (73.41%) | 173/200 (86.50%) | 371.72 |
| Gemma4 E4B Q8_0 | MASSIVE ko | 143/200 (71.50%) | 136/174 (78.16%) | 174/200 (87.00%) | 277.05 |
| Gemma4 E4B Q4_K_M | BANKING77 en | 130/200 (65.00%) | 124/172 (72.09%) | 172/200 (86.00%) | 383.61 |
| Gemma4 E4B Q4_K_M | MASSIVE ko | 138/200 (69.00%) | 129/168 (76.79%) | 168/200 (84.00%) | 287.00 |
| Qwen3 8B Q8_0 | BANKING77 en | 111/200 (55.50%) | 107/175 (61.14%) | 175/200 (87.50%) | 877.64 |
| Qwen3 8B Q8_0 | MASSIVE ko | 98/200 (49.00%) | 96/173 (55.49%) | 173/200 (86.50%) | 593.45 |

Gemma4 E4B Q8はBANKING77でE4B Q4と同率の65.0%、MASSIVE韓国語では4モデル中最高の71.5%です。Gemma4 E2B Q8の英語 / 韓国語p50は214.15 / 157.52 msです。

[Intentレポート](INTENT_BENCHMARK.md)

## 画像分類のbaseline — 2026-09-24 / 25

元のJPEGを1枚ずつローカルHTTPで直列処理しました。読込済みモデルの初回requestを含み、起動時間を除き、ウォームアップは実施していません。ポリシーは0.8 / 0.05で、各データセットのクラスと固定baseline質問を維持します。9月24日はRTX 3060、9月25日のTrashNetはRTX 3080です。

| データセット / GPU | モデル | Raw正答 / 全件 | 正答 / 受理 | Coverage | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cats/dogs / RTX 3060 | Gemma4 E2B Q8_0 | 69/70 (98.57%) | 69/70 (98.57%) | 70/70 (100%) | 99.18 |
| Caltech 30 classes / RTX 3060 | Gemma4 E2B Q8_0 | 122/150 (81.33%) | 122/148 (82.43%) | 148/150 (98.67%) | 195.28 |
| TrashNet 6 classes / RTX 3080 | Qwen3-VL 2B Q8_0 | 95/120 (79.17%) | 91/115 (79.13%) | 115/120 (95.83%) | 101.08 |
| TrashNet 6 classes / RTX 3080 | Gemma4 E2B Q8_0 | 64/120 (53.33%) | 64/113 (56.64%) | 113/120 (94.17%) | 79.12 |

猫/犬はvalidation 70枚（猫24 / 犬46）、context 2048です。選択肢を逆順にした実行でも69/70、p50 98.75 msでした。Caltechは30クラス × 5枚、context 4096で、GPU 3,855 MiBの記録は1時点の標本です。TrashNetは6クラス × 20枚です。Qwen3-VL 2B Q8は記録された3つのTrashNet baselineモデル中、rawが最高です。後続のtraits promptとbatch実験は別の探索プロトコルです。それぞれ独立したデータセット・evaluatorの結果で、HTTP時間はRust比較時間と区別します。

[画像評価レポート](VISION_BENCHMARK.md)

## 合成入力の共有状態prefix再利用 — 2026-09-25

RTX 3080、context 2048、batch/ubatch 32、CPU threads 4、state-first prompt、preparation cache無効です。例の状態にfiller 200単語を追加し、16問は異なる指示とIDを持ちます。実行順序を交互にした5ラウンドで、各経路に非計測ウォームアップ1回を実施しました。時間は準備とsession生成・破棄を含む全実行msの中央値で、モデル読込とレポート直列化は除きます。

| モデル | Fresh ms | 共有session ms | 比率 | 再利用 / 入力token |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8_0 | 1461.77 | 531.49 | 2.75× | 3840/5789 |
| Gemma4 E2B Q8_0 | 2735.54 | 985.42 | 2.78× | 3840/5751 |

元のJSONの各ラウンド記録から表の中央値を再現しました。モデルごとの210対の判定で、raw logit、候補確率、candidate mass、選択値、保留理由が完全一致します。実際のKV prefixを再利用する明示的native sessionの合成入力評価で、タスク精度と保持sessionのメモリは測定していません。

[共有状態レポート](../../benchmarks/shared-state-cache-20260925/REPORT.md)

## 再検証した元データのパス

以下の Python ファイル名は当時の実行の出典です。重複する Python 実装は削除済みです。現在の再集計には `l2s1-tools report-jevbench-matrix`、実行には `l2s1-tools jevbench-public run` を使用します。

- RTX 3060: `results/jevbench-matrix-20260923/REPORT.json`, `environment.json`; `<configuration>/{manifest.json,summary.json,tasks-with-gold.jsonl,requests.jsonl,predictions.jsonl,gpu-memory.csv,inference.stderr.log}`. Recount: `scripts/report_jevbench_matrix.py`; GPU sampling: `scripts/jevbench_public.py`.

- Gemma31B: `results/large-model-20260923T141422Z/REPORT.json`, `remote/jevbench/{manifest.json,summary.json,predictions.jsonl,tasks-with-gold.jsonl,gpu-memory.csv}`. Gemma26B: `results/gemma26-lowrss-20260923T135227Z/REPORT.json`, `remote/jevbench-read/{manifest.json,summary.json,predictions.jsonl,tasks-with-gold.jsonl,gpu-memory.csv}`.

- Intent: `results/intent-wide-20260923/prepared/gold.jsonl`, `runs/<model>/{manifest.json,summary.json,predictions.jsonl,scored.jsonl}`.

- Images: `benchmarks/cats-dogs-vision-20260924/{summary.json,observations.jsonl}` and `dog-cat/`; `benchmarks/caltech101-vision-20260924/{summary.json,observations.jsonl,selection.json}`; `benchmarks/trashnet-vision-20260925/{gemma4,qwen3vl,smolvlm}/{summary.json,observations.jsonl}`.

- Shared state: `benchmarks/shared-state-cache-20260925/{qwen3-0.6b-sm86.json,gemma4-e2b-sm86.json}`; workload: `examples/benchmark_shared_state_cache.rs`.
