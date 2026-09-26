<a id="ollaya-comparison-and-l2s1-opportunities"></a>
# Ollaya との比較と L2S1 の機会

[English](../en/OLLAYA_COMPARISON.md) · [한국어](../ko/OLLAYA_COMPARISON.md) · [日本語](OLLAYA_COMPARISON.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

この比較は 2026 年 9 月 26 日に確認した [Ollaya リビジョン 8989f88](https://github.com/ollaya-dev/ollaya/tree/8989f88d92bd2191c548fa915b6a897db0a85f32) を参照します。フィールドの不在についての記述は、その時点で文書化された API に関するもので、将来のすべてのバックエンドについてではありません。

Ollaya はすでに、モデルレジストリ、再開可能なモデル取得、デーモンのライフサイクル、言語の振り分け、デスクトップアプリ、MCP、TypeSafe 互換クライアントなど、より充実した配布体験を提供します。これらの利便性の再実装は大きな製品開発です。ソースに報告されたモデル時間や同等性の測定は、L2S1 が行った測定ではありません。

| 機会 | Ollaya の参照 | L2S1 の機能と根拠 |
| --- | --- | --- |
| 画像に基づく型付き判断 | 文書のリクエストは状態と型付き質問。この API には画像メディアのリクエスト仕様がない | ネイティブのプロジェクターによる choice・binary・ordinal 画像リクエスト、実際の写真デモと画像評価。公開写真例の誤りを明示。 |
| 入力が収まらない場合の明示的な拒否 | API §5.3 はモデルのコンテキストに合わせた状態の切り詰めを説明。ネイティブ /api/decide は state_truncated を報告するが /v1 は報告できない | コンテキスト超過を拒否し、モデルスコア応答に切り詰め情報と明示的な保留を維持。普遍的な品質向上を推測せず、該当バックエンドで検証。 |
| 計算に対するユーザーの制御 | README はテキスト生成なしの判断推論を説明 | Direct または上限付きネイティブ Qwen3 思考。実際の 88 トークン完了と上限エラーを確認。全評価で thinking の正解率向上を測定したとは主張しない。 |
| 採用基準と失敗理由の制御 | API は安定したエラーコードとモデルの temperature を説明 | リクエストごとの最上位スコア・質量しきい値、要求エラー率のしきい値変換、null 保留、標準理由コードに添えるユーザー文言。生の根拠は維持。実エラー率は正解ラベルでの検証が必要。 |
| 再現可能な品質と採用の根拠 | 公開 typed-decisions モデルスコア | 全 400 ケース / 2,000 判断の direct 実行、ハッシュ、ダウンロード可能な判断単位の記録。未加工の正解率、採用率、採用判断の正解率、正解/全体を分離。 |
| ブラウザー内テキスト実行 | 参照 README・API はネイティブデーモンとクライアント・バックエンド実行を説明 | Pages の Qwen3 ONNX WebGPU デモは要求時だけ取得し worker でローカル処理。SwiftShader ソフトウェアで実 direct 推論を確認し、ハードウェア GPU 速度・GGUF 数値同等性は未検証。 |

**L2S1 の正解率の優位性は実証されていません。** 同じ規模の Gemma 4 E2B Q8_0 は、この L2S1 手順で 54.3%、Ollaya の報告では独自の手順で 56.6% です。プロンプト、モデル実行、校正、評価器が異なるため、参考比較であり、条件を揃えたランタイム実験ではありません。Qwen3 0.6B はここでは 31.25% でした。[全ベンチマーク](TYPED_DECISIONS_BENCHMARK.md)を参照してください。

現在の強みは、制御・確認可能な画像とテキストの判断です。明示的な失敗境界、選べる計算量、変更しない生のスコア根拠、再利用可能なネイティブ実行が中心です。正確性、速度、数値同等性を主張する前に、タスク別の校正と、同一モデル・同一リクエストの条件を揃えた比較を優先してください。

出典: [Ollaya README](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/README.md)、[API §5.3](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/api.md#53-model-specific-limits)、[GGUF 測定](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/families/llm-logits.md)。
