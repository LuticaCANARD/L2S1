# Prefix の再利用、ONNX 判定、検証済みルーティング

[English](../en/FAST_DECISIONS.md) · [한국어](../ko/FAST_DECISIONS.md) · [日本語](FAST_DECISIONS.md)

三つとも明示的に有効化する機能です。既存 GGUF の既定値は維持し、MLX は実装しません。

## 固定スキーマ

環境に応じて `llama`、`llama-cuda`、`llama-metal` でビルドします。

```sh
l2s1 --model model.gguf --execution-mode prefix-reuse --fixed-schema --listen 127.0.0.1:8080
```

`FixedSchemaBackend` は cold/warm の両方を同じ prefix/suffix 境界で評価し、大きな prefill batch を維持します。完全なプロンプトをトークン化して実際の token prefix を比較するため、BPE 境界を保ちます。アクティブな KV context は一つ、prefix snapshot は最大 8 個・256MiB、スキーマ準備キャッシュは別途 64 個・4MiB です。追い出された prefix は再計算し、答えは保存しません。`clear()` で KV と snapshot を消去します。信頼領域ごとに専用サーバーを使ってください。

分割自体が既存 fresh の点数を変える可能性があります。**既存 fresh と split cold**、**split cold と split warm**を別々に比較します。後者がキャッシュの同等性検査です。recurrent/hybrid、既存 calibration、output head は拒否し、full evidence を要求します。26 個を超える候補と画像は KV を消去した既存経路で処理します。`shared_decision()` の batch 境界に合わせる従来動作は維持します。

TypeScript は `L2S1.load({ model, fixedSchema: true, binaryPath: "./target/release/l2s1" })`、Python は `LoadOptions(model=..., fixed_schema=True, binary_path="./target/release/l2s1")` を使います。実行モード省略時は prefix-reuse を選び、stdio/HTTP に対応します。`prepare()` だけでは KV は作りません。capabilities の `prefix_reuse` と結果の `usage.reused_prefix_tokens` を確認してください。

## Laya ONNX

```sh
cargo build --release --features llama,onnx --bin l2s1 --bin l2s1-onnx
python3 scripts/fetch_laya_onnx.py --output models/laya-en
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so
l2s1-onnx --model-dir models/laya-en --min-top-probability 0 --listen 127.0.0.1:8081
```

ONNX Runtime 1.24 以降を別途インストールします。CUDA は `onnx-cuda`、互換 CUDA ライブラリ、`--cuda` が必要です。CPU に黙って切り替えません。CUDA 用 FP16 graph は別ディレクトリへ `--precision fp16` で取得します。GGUF CUDA は別途 `llama-cuda` が必要です。

ダウンローダーは graph、元の重み、tokenizer、calibration のハッシュを検証します。モデルは Apache-2.0、出典は `third_party/ollaya` に記録します。Laya marker head に対応し、任意の ONNX network は対象外です。state のトークン化のみ共有し、質問ごとの encoder 入力は独立です。batch は padding 込み 32,768 tokens・128 判定までです。構造化 JSON のキーは整列するため、Ollaya 比較でもキー順を合わせます。長すぎる state、規則、候補は切り詰めず拒否します。

`discriminative` evidence は候補 logits/確率、schema ID、適用 temperature、推定値を持ち、vocabulary token ID と candidate mass は持ちません。GGUF 用のリクエスト policy と `target_error_rate` は拒否します。最大確率の閾値も誤り率の保証ではありません。モデルと ONNX Runtime ライブラリを serving identity に含め、実際のハードウェアと精度で検証してください。

## 高速モデルと低速モデル

高速サーバーと目的の低速サーバーを先に起動します。独立した calibration/validation JSONL は `{id, request, expected: {decision_id: option_id}}` 形式です。binary の正解は文字列 `false`/`true`、ordinal は level ID です。

```sh
python3 scripts/calibrate_cascade.py --fast-url http://127.0.0.1:8081 \
  --slow-url http://127.0.0.1:8080 --calibration calibration.jsonl \
  --validation heldout.jsonl --max-error 0.05 --output cascade.json
```

ID と同一 state の重複を拒否し、閾値は calibration のみで選びます。固定閾値が validation で失敗した規則は除外します。類似重複、学習データ汚染、運用分布変化までは証明しません。誤り率は held-out 標本の経験値であり、統計的保証ではありません。標本数と coverage を確認してください。

低速サーバーの設定を維持し、`--fast-model-dir models/laya-en --cascade-policy cascade.json` を追加して再起動します。高速 CUDA は `--fast-cuda` も追加します。両バックエンドの機能が必要です。policy がない場合は低速モデルのみ使い、モデル/runtime ID が異なれば起動を拒否します。検証済みの同一スキーマだけで高速結果を採用します。低確率、棄権、未対応入力、高速モデルの失敗は低速モデルへ、画像は vision 対応の低速モデルへ直接送ります。低速モデルの失敗は伝播します。各結果は採用モデルと理由を記録し、入力順を維持します。cascade ではリクエストごとの受理 policy を上書きできません。

HTTP は native batch が有効な場合のみ text/direct リクエストを集約します。最大 8 リクエスト・128 判定・44MiB・1ms です。prefix session と cascade は現在逐次実行します。待機中のジョブは 180 秒で期限切れになりますが、実行中の native 処理は中断できません。スループット、単独リクエスト遅延、p95 は別々に測ります。

## 検証

`benchmark_fixed_schema` は元の複数判定リクエストで fresh、split cold、split reuse を比較します。`benchmark_onnx` は準備・推論・点数化の時間と入出力証拠を記録します。ローカル診断であり、一般的な正解率や別ハードウェアの Ollaya 代表速度を保証しません。


## ローカル測定 (2026-09-27)

WSL2 CPU上のQwen3 0.6B Q8_0では、decision-rulesの36判断全体がfresh 29.132秒、split cold 31.062秒、split reuse 14.962秒でした。fresh比で1.95倍の改善です。batch 256を維持し、4,983トークン中2,799を再利用して、選択・確率の変化はありませんでした。ウォームアップ後の1回の全体測定であり、リクエストp50やM5 Maxの結果ではありません。

RTX 3080の英語Laya fp16経路はリクエストp50が19.179ms、同じモデルのOllayaは19.567msで、p95はそれぞれ22.165/23.338msでした。5問ずつ含む合成チケット入力20件を、20回のウォームアップ後に2周実行しました。CPUとCUDAの両方でトークンID・marker位置・logitが完全一致し、確率差は2.23e-16未満でした。L2S1の時間には確率計算を含み、比較runnerでは除外しています。ローカルな同等性の検証であり、一般的な精度や普遍的な速度優位を示すものではありません。

実HTTPでfast採用、スキーマ・入力fallback、Python SDKの受信を検証し、同時リクエスト16件中最大6件の一括実行を確認しました。calibrationは試験用データのため、本番品質の根拠にはなりません。[測定概要](../../benchmarks/fast-decisions-20260927/summary.json)と[出典](../../benchmarks/fast-decisions-20260927/provenance.json)を参照してください。全記録と再現コマンドはリポジトリの`benchmarks/fast-decisions-20260927`にあります。


## SIMDによるスコア後処理

全vocabularyのlogit検証・最大値探索にAVX2/NEONの実行時選択とscalar fallbackを導入しました。指数関数と逐次f64加算の順序は維持します。ローカルAVX2測定では3.2万〜26.2万vocabularyの探索が約10倍、正規化全体が1.17〜1.26倍高速化しました。**モデル推論全体の高速化率ではありません。** 実SmolLM2による規則判断36件の結果は適用前と完全一致しました。[SIMD測定](../../benchmarks/evidence-simd-20260927/summary.json)と[native同等性](../../benchmarks/evidence-simd-20260927/native-parity.json)を参照してください。NEONの速度は未測定です。
