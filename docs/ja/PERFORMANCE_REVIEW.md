# 性能改善の検討 — 2026-09-30

[English](../en/PERFORMANCE_REVIEW.md) · [한국어](../ko/PERFORMANCE_REVIEW.md) · [日本語](PERFORMANCE_REVIEW.md)

main `6b2fcfa` / v0.2.0 のコードに基づく検討です。以下の表は当初の改善候補です。実装後の測定結果は末尾の続報を参照してください。既存の測定リビジョンは別記します。[証拠と再集計](../../benchmarks/decision-performance-20260930/README.md)。

## 基準値と失敗傾向

| モデル | 正解 / 231 | 正解率 | p50 / p95 ms | 採用正解 / 採用 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 159 | 68.83% | 40.41 / 485.74 | 153 / 213 |
| Gemma 4 12B QAT Q4_0 | 194 | 83.98% | 93.48 / 1359.39 | 181 / 203 |
| Laya English 421M | 133 | 57.58% | 45.18 / 71.63 | 133 / 231 (保留ポリシーなし) |

JevBench `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84` の公開231問（Easy 48、Original 72、Hard 111）を各1回測定。全534問の公式順位ではありません。GemmaはL2S1 `74334ec`、RTX 3080 10 GiB、fresh/legacy、context 4096、batch/ubatch 256、4 threads、FlashAttention off、Rust in-process。Layaは公式SDK 0.3.21、checkpoint `55cf4c4e`、PyTorch 2.8.0 CUDA/BF16 autocast、Python in-process、既定512トークン。読み込みと1回のウォームアップは除外。Layaは57問で入力を切り詰め、Gemmaは0問でした。ランタイム・量子化・文脈上限・実行時点が異なり、エンジン単独の統制比較ではありません。

12Bもtemporal/numericで11/15、multi-hopで6/18、long-policyで6/19を誤答し、採用誤答は22件でした。スコア0.99以上の誤答はE2Bが37件、12Bが10件です。候補スコアは正解確率の保証ではありません。

## 実験の優先順位

| 順序 | 実験 | 現在の機能・作業 | 必要な証拠 |
| --- | --- | --- | --- |
| 1 | 12Bのメモリ・prefill設定 | 既存のFlashAttention off/on、batch/ubatch 128/256、width 1/2/3、`set_parallel_context_dynamic` | 段階別時間、GPU配置、メモリ、p50/p95、正解・採用の変化 |
| 2 | 固定質問・共通入力の再利用 | 常駐fixed-schema、`shared`、`parallel_prefix_session()` | fresh/split-cold/split-warm、実再利用トークン、状態変更と障害復旧の数値検証 |
| 3 | 難問の正解率 | 簡潔な基準、数値・時刻の前処理、別データによるLoRA/output head | 開発データで選択を固定し、独立したgroup holdoutで12Bと小型モデルを評価 |
| 4 | 採用と上位モデルへの移送 | 既存family/task calibrationとアプリ側ルーティング | 誤答率と採用率、採用誤答、採用正解/全体、実遅延と費用 |

合成ルール実験（固有36判断を3回反復）では12Bのparallel/state-first p50は313.23 msで、fresh/state-firstの243.59 msより遅く、E2Bは105.52→72.01 msでした。全設定で再利用トークンは0。12B並列時のボード全体メモリ観測値はデスクトップ込み10013 MiBでした。メモリ調査の理由にはなりますが、遅延原因を確定しません。並列実行を万能な高速既定値にはしません。

コード: `src/llama.rs`（compute・動的context）、`src/llama/shared_decision.rs`（固定質問）、`src/llama/parallel_session.rs`（呼び出し間prefix）、`src/llama/interchange.rs`（設定・較正の識別）。batch境界より短いprefixは再利用0になり得ます。token alignmentは数値を変えるため別途検証します。

## 単純な二段階構成の限界

E2Bの既存ポリシー（top probability >=0.8、candidate mass >=0.05、同率なし）が保留した場合だけ12Bに移し、そのポリシーを維持して過去の予測を再集計しました。

| ポリシー | 12Bに移送 | 正解 | 採用正解 / 採用 | 保留 |
| --- | ---: | ---: | ---: | ---: |
| E2B → 12B | 18 / 231 | 166 / 231 (71.86%) | 159 / 222 (71.62%) | 9 |
| 12B単独 | 231 / 231 | 194 / 231 (83.98%) | 181 / 203 (89.16%) | 28 |

これは記録済み予測の計算で、実際の二段階推論ではありません。同時ロード、読み込み・ルーティング費用、速度向上は未測定です。高確信度のE2B誤答が残ります。新しいルーターは別データで選択し、実際のメモリと通信費用を検証する必要があります。

## 実験の条件と境界

モデル・データ・ソース・ランタイム・adapterのハッシュを固定し、設定を一つずつ比較してから組み合わせます。baseline/candidateの順序を交替し、同じウォームアップで最低3回計時。反復を独立した正解率標本に数えません。入力長、エラー・切り詰め、正解率、採用率、採用正解率、採用正解/全体、採用誤答、p50/p95、throughput、メモリと確率変化を報告します。遅延目標と許容する正解率・採用率の変化は実験前に固定します。既に調査した公開JevBenchは回帰データであり、未使用holdoutではありません。

最初は既存設定で実行できる1・2を試します。KV量子化は現在公開されたL2S1設定ではなく、bridge・設定識別・数値検証が必要です。speculative decodingは多トークン生成向けで、一段階の直接採点では低優先度です。閾値較正は採用を変えますが元の誤答を自動修正しません。

Layaの追加比較では長文checkpointやwindowingを明示し、モデル・文脈・集計・遅延の変更を記録します。切り詰め解消が正解率を回復するとは仮定しません。[公式Laya](https://huggingface.co/convaiinnovations/laya)はcheckpoint別上限を説明しています。[llama.cppのガイド](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md)はGPU配置とthread数の確認を推奨しますが、生成速度の値をL2S1遅延の予測に使いません。

## 実装の続報

優先度1〜3向けの[チューニング・常駐セッション・構造化データの前処理](../DECISION_PERFORMANCE.md)を追加しました。測定結果と適用範囲はリンク先の実験レポートを参照してください。
