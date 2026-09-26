<a id="build-and-interfaces"></a>
# ビルドとインターフェース

[English](../../../../en/skills/l2s1/references/interfaces.md) · [한국어](../../../../ko/skills/l2s1/references/interfaces.md) · [日本語](interfaces.md)

[English index](../../../../en/README.md) · [한국어 색인](../../../../ko/README.md) · [日本語索引](../../../README.md)

確認した L2S1 チェックアウトからコマンドを実行します。モデルパスはユーザー・アプリが用意し、重みは同梱しません。Rust は 2024 エディション、ネイティブは CMake 3.24 以上と C++17 が必要です。フラグを変更するときは CLI `--help` を確認します。

```sh
cargo build --release --locked --features llama --bin l2s1
./target/release/l2s1 --model /path/to/model.gguf --inspect
./target/release/l2s1 --model /path/to/model.gguf --input request.json --preflight
./target/release/l2s1 --model /path/to/model.gguf --input request.json --diagnostics
# Or pipe a text request to --input -; JSON stdout, native logs stderr.
```

GPU ビルドは `llama-cuda`・`llama-metal` に置き換え、`--device cuda`・`--device metal` で実行します。オフラインでは `L2S1_LLAMA_CPP_SOURCE=/path/to/matching/llama.cpp` を設定します。ソース・ヘッダー・ネイティブブリッジ・ライブラリは一緒に再ビルドし、任意のランタイムライブラリに差し替えません。ログで ccache が書き込み不可なら、書き込み可能な `CCACHE_DIR` または `CCACHE_DISABLE=1` を使います。Linux ローダーには生成した対応ライブラリのディレクトリを `LD_LIBRARY_PATH` に追加する必要がある場合があります。`docs/GUIDE.md` を参照します。

<a id="resident-http"></a>
## 常駐 HTTP

```sh
./target/release/l2s1 --model /path/to/model.gguf --listen 127.0.0.1:8080
curl -sS http://127.0.0.1:8080/v1/capabilities
curl -sS -H 'Content-Type: application/json' --data-binary @request.json \
  http://127.0.0.1:8080/v1/decisions
```

ヘルスは `GET /healthz` です。未認証のネイティブ HTTP はループバックに保持します。リモートにはアプリの認証済みリバースプロキシを使います。画像は `--mmproj /path/to/matching-projector.gguf` で起動します。HTTP は base64 メディア、CLI は `--image /path/to/photo.jpg` です。テキスト CLI の事前検証は HTTP 画像を検証しません。

<a id="mcp"></a>
## MCP

Python 3.10 以上の仮想環境に `mcp/requirements.txt` を入れ、その Python で `/path/to/L2S1/mcp/server.py` を起動します。`--backend-url http://127.0.0.1:8080` を使います。サーバーがリポジトリ内なら `--root /path/to/L2S1` は任意です。Stdio MCP は別のアダプタープロセスで、ネイティブ HTTP エンドポイントではありません。`/v1/decisions` を Streamable HTTP MCP URL として登録しません。

ツールは `l2s1_document`、`l2s1_example`、`l2s1_validate`、`l2s1_capabilities`、`l2s1_decide` です。リソースは `l2s1://schema/request`、`l2s1://docs/guide`、`l2s1://examples/warehouse` などです。`design_decision` はタスクから有効なリクエストを作る支援です。設定は `docs/AGENT_INTEGRATION.md` を参照します。アダプターは運用者の選んだ origin に接続し、シェルコマンド、ファイル書き込み、ダウンロード、モデル読み込み、自動再試行を行いません。

<a id="rust"></a>
## Rust

`l2s1` 依存関係に対応する feature を有効にし、`serde_json` を追加します。Rust の型で直接構築し、入力ファイルや JSON 解析は不要です。一度読み込んで複数判断に保持します。

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
        Path::new("/path/to/model.gguf"),
        2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    let response = backend.decide(&request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    drop(backend);
    Ok(())
}
```

GPU・ビジョン・オプション・ワーカー所有には `docs/GUIDE.md` と `docs/MODEL_INTERCHANGEABILITY.md` の最新 API を使います。校正と出力ヘッドはモデル・タスク・設定に紐付き、GGUF の変更後は有効性を維持しません。
