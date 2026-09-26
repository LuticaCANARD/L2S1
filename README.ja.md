# L2S1 — LLM to System 1

**ローカル GGUF モデルのスコアを、型付きの判断結果に変換します。**

[English](README.md) · [한국어](README.ko.md) · 日本語 · [ドキュメント](docs/ja/README.md) · [モデルの測定結果](docs/ja/MODEL_RESULTS.md)

L2S1 は、ローカルのチャットモデルで二値・選択・順序付きの判断を行う Rust ライブラリと CLI です。JSON 形式の状態、質問、候補ごとの判断基準を渡すと、型付きの値とモデルのスコアを返します。採用ポリシーを満たさない場合は、判断を保留し、その理由を明示します。

メッセージの分類、リクエストの振り分け、条件の確認、順序付きのレベル評価に使用できます。質問と候補 ID は、アプリケーションがリクエストごとに指定します。互換性のある GGUF モデルに変更しても、リクエストと結果の型を維持できます。

[Python SDK](sdks/python/README.md)、[native batch](docs/ja/BATCHING_API_REVIEW.md)、[GitHub Release・npm・PyPI・Cargo 配布パイプライン](docs/ja/RELEASE_PIPELINE.md)を提供します。

## クイックスタート

Rust 2024 エディションに対応した最新の stable ツールチェーン、CMake 3.24 以上、C++17 コンパイラ、互換性のある chat/instruct GGUF が必要です。初回のネイティブビルドでは、固定されたリビジョンの llama.cpp ソースをダウンロードします。モデルの重みは別途用意してください。

```sh
git clone https://github.com/LuticaCANARD/L2S1.git
cd L2S1
cargo build --release --locked --features llama --bin l2s1

./target/release/l2s1 \
  --model /path/to/chat-model.gguf \
  --input examples/warehouse.json
```

[倉庫のリクエスト例](examples/warehouse.json)では、同じ荷物について、保管エリア、温度管理の要否、出荷優先度をそれぞれ独立して判断します。最初の判断は次のような構造です。

```json
{
  "state": { "storage_requirement": "chilled" },
  "decisions": [{
    "id": "storage_zone",
    "instruction": "Select the storage zone matching storage_requirement.",
    "kind": {
      "type": "choice",
      "options": [
        { "id": "ambient", "criterion": "Ambient storage is required." },
        { "id": "chilled", "criterion": "Chilled storage is required." },
        { "id": "frozen", "criterion": "Frozen storage is required." }
      ]
    }
  }]
}
```

[記録済みの Gemma 4 CUDA 実行](examples/warehouse.gemma4.cuda.output.json)では、`storage_zone` に対して次の結果が返されました（抜粋）。

```json
{
  "id": "storage_zone",
  "value": { "type": "choice", "selected": "chilled" },
  "abstention_reasons": []
}
```

CLI の完全な応答には、バックエンド情報、ポリシー、候補のスコア、確率質量、トークン使用量も含まれます。予測とスコアはモデルや設定によって変わります。実際の用途に対応する正解ラベル付きの例で検証してください。

## 主な機能

- **型付きの出力。** 二値判定、カテゴリ選択、順序付きのレベル評価に共通のリクエスト形式を使います。意味を表す選択肢 ID は、互換性のあるモデル間で維持されます。
- **ローカルモデルのスコアを直接使用。** モデルのアシスタント応答の開始位置でスコアを読み取ります。候補が多く、回答コードが複数トークンに分かれる場合は、コード全体の尤度を計算します。
- **明示的な判断保留。** 選択を保留してもスコアを保持し、その理由を返します。デフォルトのポリシーでは、候補間の相対確率と、全語彙における候補の確率質量の両方を確認します。
- **テキストと画像。** 対応するビジョンモデルと、それに適合する `mmproj` GGUF を使って静止画像を判断できます。
- **Rust、TypeScript、CLI、HTTP。** アプリケーション内でバックエンドを常駐させる、[TypeScript パッケージ](docs/ja/typescript/README.md)を Node.js から使う、ファイルや標準入力から JSON を渡す、バージョン付きの判断 API を提供するといった方法を選べます。
- **実行内容の確認。** モデルの識別情報、リクエストの事前検証、診断情報を確認できます。オプションのキャッシュ、状態復元、並列実行、LoRA、タスク別の校正には、それぞれ明示された利用条件があります。

## 判断の型とスコア

| 型 | 入力 | ローカルでの結果 |
| --- | --- | --- |
| `binary` | 偽・真の判断基準 | Boolean または `null` と `p_true` |
| `choice` | 候補 ID と判断基準 | 選択された ID または `null` |
| `ordinal` | 数値が増加する順に並べたレベル | 選択されたレベル ID または `null` と期待値 |

各判断は独立して評価されます。すべての型で候補のスコアと判断保留の理由を返します。候補が 26 個以下なら `A`–`Z`、それより多い場合は `AA`–`ZZ` などの固定長コードを使います。トークン化とコンテキストの制限は、選択したモデルに対して確認します。

デフォルトの採用判定には、次の 2 つのスコアを使います。

- `option_probability`: 指定した候補間での相対確率です。
- `candidate_mass`: 全語彙または回答コードの経路において、候補に割り当てられた確率質量です。

デフォルトのポリシーでは、最上位候補の確率が **0.8 以上**、候補の確率質量が **0.05 以上**で、最上位候補に同点がないことを要求します。条件を満たさない場合、選択値は `null` になります。これらはモデルのスコアであり、正解する確率として校正された値ではありません。オプションの校正は、特定のモデル、設定、タスクに紐付きます。

数式、回答コード、検証ルールは[判断の仕様](docs/ja/GUIDE.md#the-decision-contract)を参照してください。

## バックエンドとハードウェア

| バックエンド | モデルと入力 | ハードウェア | 結果の根拠 |
| --- | --- | --- | --- |
| `llama` / `llama-cuda` / `llama-metal` | 互換性のある GGUF チャットモデル。画像には対応するプロジェクターが必要 | CPU、NVIDIA CUDA、Apple Metal | ローカルモデルのスコア |
| `wgpu` | Gemma 4 テキスト GGUF。対応するビジョンプロジェクターを追加可能 | ネイティブの wgpu GPU アダプター | ローカルモデルのスコア |
| `openrouter` | 要求されたモダリティに対応するプロバイダーのモデル | リモート API | 選択結果のみ。確率は返さない |

デフォルトの Cargo feature は空です。ローカル CLI を使うには `llama` を有効にします。モデルは固定されたランタイムでサポートされ、利用可能なチャットテンプレートを持ち、選択したデバイスとコンテキストに収まる必要があります。wgpu アダプターは Gemma 4 専用です。モデルを変更すると、予測、レイテンシ、トークン化、校正も変わる可能性があります。

ローカル GPU 推論では、対応する feature でビルドし、デバイスを指定します。

```sh
# NVIDIA CUDA: CUDA ツールキットが必要です。
cargo build --release --locked --features llama-cuda --bin l2s1
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --device cuda --input examples/warehouse.json

# macOS Metal: Metal コンパイラを含む Xcode ツールチェーンが必要です。
cargo build --release --locked --features llama-metal --bin l2s1
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --device metal --input examples/warehouse.json
```

デフォルトは CPU です。GPU を明示的に指定する場合、その GPU が利用可能である必要があります。初回ビルドではネイティブのソースを取得します。オフラインビルドでは `L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp` を設定できます。ネイティブライブラリ、配布パッケージ、CUDA アーキテクチャごとのビルドについては[ビルドガイド](docs/ja/GUIDE.md#build)を参照してください。

[wgpu](docs/ja/GUIDE.md#optional-wgpu-backend) には専用の `l2s1-wgpu` 実行ファイルを使います。[OpenRouter](docs/ja/GUIDE.md#openrouter-adapter) には `l2s1-openrouter` を使い、`OPENROUTER_API_KEY` を設定します。OpenRouter の応答は共通の型付き HTTP エンベロープを使い、根拠は `selection_only` です。ローカルの確率ポリシーは適用されません。

## モデルの確認と実行

```sh
# 読み込んだモデルと機能を確認します。
./target/release/l2s1 --model /path/to/chat-model.gguf --inspect

# モデルの順伝播を実行せずに、実際のリクエストを検証します。
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --input examples/warehouse.json --preflight

# 実行診断を有効にして実行します。
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --input examples/warehouse.json --diagnostics
```

`--input -` は標準入力を読み取ります。結果は標準出力へ、ネイティブのログは標準エラー出力へ送られます。設定されたコンテキストを超える入力は、切り詰めずに拒否します。計算設定と構造化されたエラーについては[確認と診断](docs/ja/GUIDE.md#inspect-validate-and-run)を参照してください。

## HTTP と画像入力

ローカルサーバーを起動します。

```sh
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --listen 127.0.0.1:8080
```

別のターミナルからリクエストを送ります。

```sh
curl -sS -H 'Content-Type: application/json' \
  --data-binary @examples/warehouse.json \
  http://127.0.0.1:8080/v1/decisions
```

サーバーは `POST /v1/decisions`、`GET /v1/capabilities`、`GET /healthz` を公開します。HTTP 応答には API バージョン、リクエスト ID、バックエンド、ポリシー、根拠を含む型付きの結果が入ります。ループバックにバインドするか、リモートアクセスには認証付きのリバースプロキシを使ってください。

静止画像には、ビジョンモデルと、それに適合するプロジェクターを読み込みます。

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/projector.gguf --image photo.jpg \
  --input examples/warehouse.json
```

HTTP の画像リクエストでは、名前付きの `media` と判断ごとの `media_ids` を使います。ローカルバックエンドは、判断ごとに 1 枚の画像を受け付けます。ペイロード、制限、バックエンドごとの動作については[画像と HTTP の仕様](docs/ja/GUIDE.md#direct-image-input-and-http-api)を参照してください。

## TypeScript から使う

[`@l2s1/node` パッケージ](docs/ja/typescript/README.md)は、現在の OS とアーキテクチャに対応するビルド済みの Rust ランタイムを選択し、型付きの `load()`、`decide()`、`capabilities()`、`close()` を提供します。ビルドワークフローで作成されたラッパーとランタイムの tarball をインストールしてください。npm にはまだ公開されていません。GGUF の重みは別途用意します。同じアプリケーション API で、`connect()` は HTTP サーバーを使い、`fromBackend()` はカスタムバックエンドを受け付けます。

Node.js 24 以上と Bash または Zsh を使い、リポジトリのルートから次のコマンドを実行してください。ローカル CPU ランタイムと TypeScript パッケージをビルドし、例の型を検査してから[倉庫の分類例](sdks/typescript/examples/warehouse.ts)を実行します。モデルのパスは、用意した GGUF ファイルの絶対パスに置き換えてください。`npm pack` は `sdks/typescript/l2s1-node-0.1.0.tgz` を生成します。

```sh
cargo build --release --locked --features llama --bin l2s1
cd sdks/typescript
npm ci
npm run build
npm run check
L2S1_BINARY=../../target/release/l2s1 \
  node examples/warehouse.ts /absolute/path/to/chat-model.gguf
npm pack
cd ../..
```

```ts
import { L2S1 } from '@l2s1/node';

const engine = await L2S1.load({
  model: '/path/to/chat-model.gguf',
});
try {
  const response = await engine.decide({
    state: { x: 1 },
    decisions: [{ id: 'positive', instruction: 'Is x positive?',
      kind: { type: 'binary', false_label: 'x <= 0', true_label: 'x > 0' } }],
  });
  console.log(response.results);
} finally { await engine.close(); }
```

既存のサーバーやブラウザーアプリケーションから使う場合は、`@l2s1/node/http` から `L2S1Client` をインポートします。インストール、画像、推論、エラー、移植性については[パッケージガイド](docs/ja/typescript/README.md)を参照してください。

## Rust から使う

`l2s1` 依存関係の `llama` feature を有効にします。一度読み込んだバックエンドを保持し、繰り返しリクエストに使います。

```rust
use std::path::Path;
use l2s1::{
    Decision, DecisionBackend, DecisionKind, DecisionPolicy, DecisionRequest,
    Level, OptionSpec, llama::LlamaBackend,
};
use serde_json::{Map, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = DecisionRequest {
        state: Value::Object(Map::from_iter([
            ("shipment_id".into(), Value::from("BOX-103")),
            ("storage_requirement".into(), Value::from("chilled")),
            ("hours_until_dispatch".into(), Value::from(4)),
        ])),
        decisions: vec![
            Decision {
                id: "storage_zone".into(),
                instruction: "Select the storage zone that matches the shipment's storage_requirement.".into(),
                kind: DecisionKind::Choice {
                    options: vec![
                        OptionSpec {
                            id: "ambient".into(),
                            criterion: "The shipment requires ambient storage.".into(),
                        },
                        OptionSpec {
                            id: "chilled".into(),
                            criterion: "The shipment requires chilled storage.".into(),
                        },
                        OptionSpec {
                            id: "frozen".into(),
                            criterion: "The shipment requires frozen storage.".into(),
                        },
                    ],
                },
            },
            Decision {
                id: "cold_chain_required".into(),
                instruction: "Does this shipment need temperature-controlled storage? Chilled and frozen shipments do; ambient shipments do not.".into(),
                kind: DecisionKind::Binary {
                    false_label: "No temperature control is required.".into(),
                    true_label: "Temperature control is required.".into(),
                },
            },
            Decision {
                id: "dispatch_priority".into(),
                instruction: "Choose the priority using hours_until_dispatch and the exact thresholds in the levels.".into(),
                kind: DecisionKind::Ordinal {
                    levels: vec![
                        Level {
                            id: "low".into(),
                            criterion: "More than 24 hours remain until dispatch.".into(),
                            value: 0.0,
                        },
                        Level {
                            id: "medium".into(),
                            criterion: "More than 6 hours and at most 24 hours remain until dispatch.".into(),
                            value: 1.0,
                        },
                        Level {
                            id: "high".into(),
                            criterion: "At most 6 hours remain until dispatch.".into(),
                            value: 2.0,
                        },
                    ],
                },
            },
        ],
    };
    request.validate()?;
    let mut backend = LlamaBackend::load(
        Path::new("/path/to/chat-model.gguf"),
        2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    let response = backend.decide(&request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
```

依存関係に `serde_json` を追加してください。リクエストは Rust で直接組み立てるため、入力ファイルは不要です。モデル不要の[実行可能な例](examples/warehouse.rs)でも同じリクエストを組み立てて検証できます。`cargo run --locked --example warehouse` で実行してください。専用スレッドでバックエンドを所有し、受け付けるリクエスト数を制限する場合は `BackendWorker` を使います。Metal ではプロセス終了前にバックエンドを解放してください。ワーカーを使う場合は `close()` を呼び出し、所有スレッドの終了を待ちます。ライフサイクルとネイティブリンクの詳細は [Rust 統合](docs/ja/GUIDE.md#rust-integration)を参照してください。

## 実行モードと画像処理のスループット

llama.cpp バックエンドは、4 つの実行モードに対応しています。

| モード | 用途 |
| --- | --- |
| `fresh` | 空のシーケンス状態から独立して評価する。デフォルト |
| `prefix-reuse` | 1 つのリクエスト内で、完全に一致する共通トークンプレフィックスを再利用する |
| `state-restore` | 共通プレフィックスの状態を保存し、独立したサフィックスの評価前に復元する |
| `parallel` | 独立した質問を、それぞれ分離されたシーケンスでまとめて処理する |

GPU での画像処理では、`--vision-optimized` により、4 つのデコーダーストリーム、動的なコンテキスト確保、Flash Attention、コンパクトな根拠情報、サイズ制限付きの前処理キャッシュ、同一画像のプロジェクター処理の再利用を有効にします。

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/projector.gguf --device cuda --vision-optimized \
  --image photo.jpg --input examples/warehouse.json
```

並列実行とビジョンプロファイルはサポートされた機能ですが、`fresh` とは数値計算の実行経路が異なるため、スコアや選択結果が変わることがあります。並列モードは再帰型・ハイブリッドモデルを拒否し、画像の並列処理では選択肢数が最大 26 個に制限されます。ビジョンプロファイルには CUDA または Metal と、互換性のある GPU カーネルが必要です。Metal での性能は未検証です。使用するチェックポイントで、タスクの品質と採用率を検証してください。モデル依存のテストは明示的に有効にする必要があり、重みをダウンロードしません。[実行とメモリ](docs/ja/GUIDE.md#execution-and-memory)、[画像処理の最適化](docs/ja/GUIDE.md#optimized-vision)、[検証](docs/ja/VERIFICATION.md)を参照してください。

[画像とテキストのデモを開く](https://n2s1.luticalab.net/demo): 実際に記録されたモデル応答を確認し、対応するローカルのテキスト推論では直接回答または上限付きの思考を選び、採用しきい値や失敗時のメッセージを編集できます。[デモのセットアップ](docs/ja/IMAGE_DEMO.md) · [推論の仕様](docs/ja/REASONING.md)。Pages サイトは記録済みの結果を配信します。新たに推論を実行するには、ドキュメントに記載されたローカルのネイティブサーバーが必要です。

## 記録済みの測定

| 評価 | 記録された範囲 | レポート |
| --- | --- | --- |
| JevBench 公開サブセット | 元の測定表: 22 チェックポイント × 231 項目、5,082 件の有効な予測 | [モデルの測定結果](docs/ja/MODEL_RESULTS.md)、[手法](docs/ja/JEVBENCH.md) |
| 意図分類 | 英語 77 ラベル、韓国語 60 ラベル。チェックポイントごとに各言語 200 例 | [意図分類ベンチマーク](docs/ja/INTENT_BENCHMARK.md) |
| typed-decisions | テスト分割全体: モデルごとに 400 ケース / 2,000 判断。採用ポリシー適用前の正解率は Gemma 4 E2B が 54.30%、Qwen3 0.6B が 31.25% | [手順と結果](docs/ja/TYPED_DECISIONS_BENCHMARK.md) |
| 画像の判断 | 静止画像の分類と実行モードの比較 | [画像ベンチマーク](docs/ja/VISION_BENCHMARK.md)、[TrashNet 評価](docs/ja/benchmarks/trashnet-vision-20260925/REPORT.md) |

これらは、記載されたリビジョン、ハードウェア、設定で行ったローカル実験の記録です。正解率、採用された判断の正解率、採用率は分けて報告してください。一部の生データはローカルに保持され、gitignore の対象です。レポートに保存場所と再現手順を記載しています。JevBench 公開サブセットの結果は、公式の全スイートのスコアや順位ではありません。

## ドキュメント

AI エージェントには、環境をまたいで利用できる [L2S1 スキル](docs/ja/skills/l2s1/SKILL.md)と、オプションの stdio MCP アダプターを使えます。MCP はドキュメント、型付きリクエストの検証、常駐 HTTP バックエンドによる判断を提供します。[エージェントのセットアップ](docs/ja/AGENT_INTEGRATION.md)を参照してください。

| トピック | ドキュメント |
| --- | --- |
| ビルド、リクエスト、Rust API、HTTP、ランタイムオプション | [ガイド](docs/ja/GUIDE.md) |
| モデル識別、事前検証、校正、ワーカーの所有権 | [モデルの交換](docs/ja/MODEL_INTERCHANGEABILITY.md) |
| モデル別のテストコマンド | [検証](docs/ja/VERIFICATION.md) |
| プレフィックス再利用と並列実行 | [プレフィックスアルゴリズム](docs/ja/SEMIF_ALGORITHM.md)、[並列実行](docs/ja/PARALLEL_EXECUTION.md) |
| 学習による特化 | [LoRA 学習](docs/ja/DECISION_FINETUNE.md)、[出力ヘッド](docs/ja/OUTPUT_HEAD.md) |
| データセットの準備とレポートツール | [l2s1-tools](docs/ja/crates/l2s1-tools/README.md) |
| 記録されたモデル比較とその限界 | [モデルの測定結果](docs/ja/MODEL_RESULTS.md) |

## リポジトリと開発

| パス | 役割 |
| --- | --- |
| [`src/decision.rs`](src/decision.rs) | 型付きリクエスト、結果、共通のスコア計算、ポリシー |
| [`src/llama.rs`](src/llama.rs)、[`src/wgpu.rs`](src/wgpu.rs) | ローカル推論バックエンド |
| [`src/http.rs`](src/http.rs)、[`src/http/contract.rs`](src/http/contract.rs) | HTTP サーバーとバージョン付きの通信仕様 |
| [`crates/l2s1-llama-sys/`](crates/l2s1-llama-sys) | 固定リビジョンの llama.cpp ビルドとネイティブブリッジ |
| [`crates/l2s1-tools/`](crates/l2s1-tools) | データセットとベンチマークのツール |
| [`examples/`](examples) | リクエスト、保存済みの出力、統合例 |
| [`sdks/`](sdks/README.md) | 言語 SDK: TypeScript/Node.js と Python |
| [`docs/`](docs/README.md) | 詳細ガイド、設計ノート、評価レポート |
| [`web/`](web) | Svelte のドキュメントサイト |

Rust のみで実行できる検証は `cargo test --locked` で実行します。ネイティブおよびモデル別の検証は [VERIFICATION.md](docs/ja/VERIFICATION.md)に記載しています。不具合を報告する際は、実行コマンド、チェックポイントと量子化、ランタイムとデバイス、エラーを添えて [GitHub Issue](https://github.com/LuticaCANARD/L2S1/issues) を作成してください。ドキュメントを更新する際は、英語・韓国語・日本語の README の内容を揃えてください。

## ライセンス

L2S1 のソースは [MIT ライセンス](LICENSE)です。モデルの重みにはそれぞれ独自のライセンスがあり、同梱されません。ネイティブコンポーネントを配布する際は[第三者のライセンス表記](THIRD_PARTY_LICENSES.txt)を保持してください。[LICENSING.md](docs/ja/LICENSING.md)も参照してください。
