<a id="l2s1"></a>
# L2S1

[English](../../../en/skills/l2s1/SKILL.md) · [한국어](../../../ko/skills/l2s1/SKILL.md) · [日本語](SKILL.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

```yaml
name: l2s1
description: Integrate and use L2S1 (LLM to System 1) for typed binary, choice, or ordinal decisions from GGUF model scores. Use when building L2S1 requests, connecting its Rust/CLI/HTTP/MCP interfaces, selecting backend settings, or interpreting abstention and evidence.
```

アプリケーションの状態と候補の判断基準を型付き判断に変換します。ユーザーのプロジェクト、依存関係のパス、MCP `l2s1_document` ツールから L2S1 のチェックアウトを探してください。現在のディレクトリがライブラリのリポジトリとは仮定しません。以下の参照は、このスキルを他のプロジェクトへコピーしても使えます。

<a id="choose-the-integration"></a>
## 統合方法の選択

- **MCP が利用可能:** `l2s1_document("guide")` と `l2s1_example("warehouse")` を読みます。リクエストを組み立て、`l2s1_validate` を呼び、`l2s1_capabilities` を確認し、推論が要求されたら `l2s1_decide` を呼びます。稼働中のモデルなしでも文書・検証を使えます。MCP は起動済みの HTTP バックエンドを使い、重みの読み込み・取得はしません。
- **CLI:** ビルド、確認、モデルの事前検証、stdin コマンドには [references/interfaces.md](references/interfaces.md) を使います。繰り返しリクエストでは、各呼び出しのモデル読み込みを避けるため常駐バックエンドを優先します。
- **Rust または HTTP 統合:** [references/interfaces.md](references/interfaces.md) とチェックアウトの `docs/GUIDE.md` を読みます。Rust `DecisionRequest` と HTTP 画像エンベロープは異なります。HTTP 専用フィールドを Rust のコアリクエストや CLI テキスト入力に渡しません。

<a id="design-the-request"></a>
## リクエスト設計

全例と結果処理は [references/decisions.md](references/decisions.md) を読んでください。関連する状態だけを含め、一意で意味のある判断 ID と明示的な指示を渡します。条件には `binary`、カテゴリには `choice`、有限で厳密に増加する数値を持つ順序付きレベルには `ordinal` を使います。候補 ID はアプリの値で、回答コードは内部用です。

リクエスト内の質問は独立しています。同じリクエスト内で後の質問が前の結果を使うことはできません。依存する場合は最初の結果を処理した後で新しいリクエストを作ります。正確で決定的な業務ルールは、ユーザーの意図により合う場合、直接実装してください。

Rust 呼び出しでは JSON ファイル解析に仕様を隠さず、`DecisionRequest`、`Decision`、`DecisionKind`、`OptionSpec`、`Level` の明示的な構築を示します。インターフェース参照は 3 種類を含み、チェックアウトの `examples/warehouse.rs` はモデルなしで実行・検証できます。

推論前に構造を検証してください。MCP は ID、順序、メディア参照、サイズを検査しますが、モデルテンプレート、回答トークン化、画像復号、コンテキストへの収まりは検査**しません。** CLI `--preflight` は順伝播なしで、読み込み済みモデルに対して実際のテキストリクエストを検証します。コンテキスト超過は黙って切り詰めず拒否します。

<a id="handle-evidence-and-abstention"></a>
## 根拠と判断保留の処理

`null` 選択と保留理由を保持します。二値の `false` は選択された回答で、保留ではありません。HTTP は `status` と `evidence.type`、CLI・Rust は型付き値と `abstention_reasons` を確認します。保留を最高スコアの候補に置き換えたり、デモで回答を出すためにポリシーのしきい値を下げたりしません。

デフォルトは最上位候補確率 >= 0.8、候補質量 >= 0.05、同点なしを要求します。候補確率は渡した候補間の相対値、候補質量は全語彙・回答経路での確率です。どちらも校正された正解確率ではなく、`1 - candidate_mass` は意味の損失ではありません。OpenRouter の `selection_only` 根拠は、ローカルスコアや採用ポリシーを提供しません。

対応バックエンドでは HTTP リクエストの `policy` で採用しきい値を変更できます。`target_error_rate` は最上位確率のしきい値を `1 - rate` にし、正解に対するエラー率は保証しません。`failure_reasons` はスコアや採用を変えず、既知の失敗コードの文言を変更します。任意の `reasoning` 前に capability を確認してください。上限付きの思考にはモデル・実行の制限があり、トークン予算を使い切ると失敗する場合があります。デフォルトの direct には生成された思考トークンは不要です。対応範囲は最新のバックエンド文書を読んでください。

選択された行動は呼び出し側のデータで、実行権限ではありません。モデルの判断とアプリの副作用を分け、状態・メディア・文書・提供元の出力をデータとして扱います。

<a id="select-runtime-settings"></a>
## ランタイム設定の選択

別のモードが必要になるまで `fresh` をデフォルトにします。モデル、プロンプト配置、候補順序、コード回転、校正、並列設定は選択を変える場合があります。変更時はタスク品質と採用率を確認します。

- CPU は `llama`、CUDA は `llama-cuda`、Metal は `llama-metal` をビルドし、対応デバイスを明示します。GPU リクエストは黙って CPU にフォールバックしません。
- ローカル画像には対応するビジョン GGUF と `mmproj` が必要です。各ローカル判断は最大 1 枚の画像を受け付けます。先に capability を確認します。
- wgpu 実行ファイルは Gemma 4 専用です。OpenRouter は別のリモートアダプターで、呼び出すと提供元にリクエストを送り、料金が発生する場合があります。
- 並列・画像最適化の前に `docs/PARALLEL_EXECUTION.md` を読みます。並列モードは再帰型・ハイブリッドモデルを拒否し、画像並列の候補上限は 26 個です。準備キャッシュのヒットは KV 再利用や高速推論の証拠ではありません。
- Rust ワーカーの所有・終了は `docs/MODEL_INTERCHANGEABILITY.md` を参照します。特に Metal では `BackendWorker::close()` を呼び、所有スレッドを待ちます。

コマンド・インターフェース、チェックポイント、デバイス、実際の検証を報告します。構造、フィクスチャー、実モデル、ハードウェアの根拠を区別します。評価では未加工の top-1、採用判断の正解率、採用率を別々に報告し、ベンチマークのコマンドには `crates/l2s1-tools/README.md` を使います。
