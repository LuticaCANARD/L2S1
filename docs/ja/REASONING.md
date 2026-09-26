<a id="direct-scoring-and-bounded-thinking"></a>
# 直接スコア計算と上限付きの思考

[English](../en/REASONING.md) · [한국어](../ko/REASONING.md) · [日本語](REASONING.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

Direct モードは引き続きデフォルトで、既存のプロンプトのバイト列を維持します。閉じた空の思考ブロックの後で候補コードを評価し、推論シーケンスを生成しません。明示的な `thinking` モードではまずトークンを greedy 生成し、モデル固有の `</think>` トークンを待ち、最後の区切りを評価してから、全語彙 logits で同じ型付き候補集合を評価します。最終回答コードは生成せず、スコアを計算します。

初期の thinking 対応は、チャットテンプレートが `enable_thinking` をサポートする **dense Qwen3 テキスト GGUF** に限定されます。対応モデルで自動選択される `qwen3` プロンプトプロファイルを使います。画像の thinking、他のアーキテクチャ、26 個を超える候補、プレフィックス再利用、並列実行、状態復元、準備キャッシュ、簡略化した根拠情報、LoRA、出力ヘッド、学習済み校正、隠れた特徴量の出力は拒否します。画像の判断は互換性のあるビジョンモデルとプロジェクターで direct モードを使います。

<a id="cli"></a>
## CLI

```sh
cargo build --release --locked --features llama --bin l2s1
target/release/l2s1 --model models/Qwen3-0.6B-Q8_0.gguf \
  --input request.json --reasoning-mode thinking --max-reasoning-tokens 128
```

従来の経路には `--reasoning-mode direct` を使います。思考予算のデフォルトは 128、範囲は 1–1024 です。入力、全トークン予算、最後の区切りが設定されたコンテキストに収まる必要があります。入力の切り詰めは無効です。予算内で思考を閉じなかった場合、生成トークン数と予算を含む `reasoning_limit` で失敗します。閉じる前に生成終了トークンが出た場合は `reasoning_incomplete` で失敗します。どちらも合成の終了トークンを挿入せず、未完了のシーケンスを回答に変換しません。

<a id="library-and-http"></a>
## ライブラリと HTTP

```rust,ignore
backend.set_reasoning(l2s1::ReasoningOptions {
    mode: l2s1::ReasoningMode::Thinking,
    max_tokens: 128,
})?;
let response = backend.decide(&request)?;
backend.set_reasoning(l2s1::ReasoningOptions::default())?;
```

JSON HTTP リクエストは `"reasoning":{"mode":"thinking","max_tokens":128}` を受け付けます。アダプターは失敗時も含め、各リクエストの後に以前の推論モードを復元します。対応モデル情報は capability エンドポイントで確認できます。ビジョンリクエストで thinking を選ぶと明示的なエラーになります。

成功した CLI・ライブラリの結果には `reasoning` が含まれます。HTTP の結果では `usage.reasoning` に配置します。例は次のとおりです。

```json
{"mode":"thinking","generated_tokens":88,"completed":true}
```

数にはモデルが生成した終了トークンを含みます。信頼された最後の区切りと、生成しない回答コードは含みません。`input_tokens` は準備された入力プロンプトを数えます。思考の内容は返しません。プロンプト・成果物の識別情報には `thinking-greedy-v1` と予算を含め、direct モードの校正が黙って再利用されるのを防ぎます。

<a id="verified-boundary"></a>
## 検証した範囲

2026 年 9 月 26 日、実際の Qwen3-0.6B Q8_0 CPU ランタイムは、猫のテキストについての二値質問を 88 生成トークンで完了しました。1 トークンの予算では明示的に失敗しました。その失敗後と完了した思考の後に direct に戻すと、同じバックエンドインスタンスで全候補確率が完全に再現されました。thinking の繰り返しでもトークン使用量と確率が完全に再現されました。不正なキャッシュ・実行の組み合わせは拒否され、正しく復旧しました。

同じ質問は RTX 3080 の実際の CUDA HTTP ブラウザーデモでも 96 生成トークンで完了し、1 トークンの予算ではユーザーのカスタム `reasoning_limit` 文言とともに HTTP 422 を返しました。CPU と CUDA の生成トークン数が同じとは仮定しません。これは完了確認であり、条件を揃えたレイテンシやデバイス間の数値同等性の測定ではありません。

これはネイティブの生成と隔離の回帰検証であり、品質ベンチマークや thinking が正解率を改善するという主張ではありません。公開の typed-decisions 測定は direct を使います。思考後も、候補確率と `candidate_mass` は従来の条件付きスコアの意味を維持します。

実モデルの回帰検証は明示的に実行してください。

```sh
L2S1_TEST_REASONING_MODEL=/path/to/Qwen3-0.6B-Q8_0.gguf \
  cargo test --locked --features llama --test reasoning_llama -- --nocapture
```

ローカルの direct・thinking・limit の生の根拠は、gitignore 対象の `results/reasoning-native-20260926/` ディレクトリにあります。
