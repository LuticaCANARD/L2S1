<a id="ai-agent-integration"></a>
# AI エージェント統合

[English](../en/AGENT_INTEGRATION.md) · [한국어](../ko/AGENT_INTEGRATION.md) · [日本語](AGENT_INTEGRATION.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

L2S1 は環境をまたいで使える[スキル](skills/l2s1/SKILL.md)と stdio MCP アダプターを含みます。スキルはリクエスト設計、Rust・CLI・HTTP の選択、実モデルの検証、判断保留と根拠の保持を案内します。MCP は管理されたリポジトリの文書と型付きツールを公開し、エージェントがライブラリを調べたり、稼働中のバックエンドを呼び出したりする際にシェルアクセスを不要にします。

<a id="install-the-skill"></a>
## スキルのインストール

`skills/l2s1` ディレクトリ全体をエージェントのスキルディレクトリにコピーしてください。`.agents/skills` でプロジェクトのスキルを検出するエージェントでは、L2S1 のチェックアウトから次を実行します。

```sh
mkdir -p /path/to/consumer-project/.agents/skills
cp -R skills/l2s1 /path/to/consumer-project/.agents/skills/l2s1
```

`references/` と `agents/` を `SKILL.md` と一緒に保持してください。他のスキルがない配置先を選び、既存のコピーを置き換える前に確認してください。自動検出が有効で、名前付きスキルに対応するクライアントでは `$l2s1` で呼び出せます。MCP がなくても Rust、CLI、HTTP で動作します。配布用ソースはリポジトリの `skills/l2s1` にあり、エージェントへのインストールは別の手順です。

<a id="install-and-connect-mcp"></a>
## MCP のインストールと接続

アダプターは [公式 MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) の v1 API と [stdio トランスポート](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)を使います。Python 3.10 以上が必要です。依存関係はネイティブ Rust ビルドから分離されます。

```sh
python3 -m venv /path/to/l2s1-mcp-venv
/path/to/l2s1-mcp-venv/bin/python -m pip install -r /path/to/L2S1/mcp/requirements.txt
```

stdio に対応する MCP クライアントに以下のコマンドを登録します。絶対パスを置き換えれば、作業ディレクトリを仮定する必要はありません。`mcpServers` JSON 設定を使うクライアントでは以下を使えます。

```json
{
  "mcpServers": {
    "l2s1": {
      "command": "/path/to/l2s1-mcp-venv/bin/python",
      "args": [
        "/path/to/L2S1/mcp/server.py",
        "--backend-url", "http://127.0.0.1:8080"
      ]
    }
  }
}
```

クライアントがアダプターを起動します。`python /path/to/L2S1/mcp/server.py --help` でコマンドを確認できます。`--help` なしでは標準入力の MCP メッセージを待ちます。標準出力はプロトコルメッセージ専用です。

文書、例、スキーマ、オフライン検証はすぐに使えます。推論にはネイティブ HTTP バックエンドを別途起動します。CPU の例です。

```sh
cargo build --release --locked --features llama --bin l2s1
./target/release/l2s1 --model /path/to/model.gguf --listen 127.0.0.1:8080
```

バックエンドがモデルを所有し、呼び出し間も常駐します。必要に応じて `llama-cuda` と `--device cuda`、または `llama-metal` と `--device metal` を使います。画像には対応する `--mmproj` も指定します。アダプターのデフォルトは `http://127.0.0.1:8080`、タイムアウトは 180 秒で、`--backend-url` と `--timeout` で設定できます。コピーしたサーバーは `--root /path/to/L2S1` を使えます。管理された文書と例にはソースのチェックアウトが必要です。

ネイティブ API は JSON HTTP で、**Streamable HTTP MCP エンドポイントではありません。** アダプターはバックエンドの起動、重みの取得、選んだ動作の実行、失敗した推論の再試行を行いません。設定された origin だけに接続し、HTTP プロキシや認証情報を継承しません。ネイティブ HTTP はループバックに置いてください。このアダプターはリモート認証を提供しません。OpenRouter を設定したバックエンドは状態とメディアをリモートの提供元へ送り、料金が発生する場合があります。

<a id="agent-interface"></a>
## エージェントのインターフェース

| ツール | 動作 |
| --- | --- |
| `l2s1_document(name)` | 許可リストの `overview`、`guide`、`models`、`verification`、`parallel`、`tools`、`agents`、`skill` を読む |
| `l2s1_example(name)` | `warehouse` テキストリクエストまたは `image` HTTP テンプレートを取得 |
| `l2s1_validate(request)` | モデルなしで v1 の構造、ID、順序、メディア参照、サイズ、任意の推論・ポリシーフィールドを検査 |
| `l2s1_capabilities()` | 常駐バックエンドの根拠タイプ、モダリティ対応、制限を取得 |
| `l2s1_decide(request)` | 保留、スコア、使用量を含むネイティブ HTTP 結果を変更せず返す |

2 つのリクエストツールは、ツール検出時に `binary`、`choice`、`ordinal`、画像フィールドの詳細な JSON Schema を公開します。静的リソースは同じ文書を `l2s1://docs/<name>`、例を `l2s1://examples/<name>`、スキーマを `l2s1://schema/request` で公開します。リソースをエージェントに公開しないクライアントには同等のツールがあります。`design_decision(task)` プロンプトはリクエスト設計を支援します。

アダプターは HTTP v1 の `state`、`decisions`、任意の `media`、判断ごとの `media_ids`、任意の `reasoning`、`policy`、`target_error_rate`、`failure_reasons` に対応します。オプション機能の前に capability を確認してください。古いバックエンドは未対応フィールドを拒否する場合があります。リクエストのエラー率はスコアのしきい値で、正解保証ではありません。未知のフィールドは拒否します。HTTP の上限である 128 判断、4 メディア、復号画像ごとに 8 MiB、本文ごとに 44 MiB を検査します。検証は構造的です。画像の復号、モデル・トークン・コンテキストの事前検証、バックエンドごとの制限にはランタイムが必要です。テンプレートの画像のプレースホルダーは、実際の標準 base64 バイト列に置き換えます。

通常はガイドと例を読み、リクエストを設計・検証し、バックエンドの capability を確認して `l2s1_decide` を呼びます。保留した結果も `status: "abstained"` と null 選択値を含む成功したツール実行です。転送、検証、バックエンドの失敗は MCP ツールエラーになります。バックエンドの HTTP ステータスと構造化エラー本文はエラーメッセージに保持されます。タイムアウト後も推論が続く可能性があり、アダプターは自動で再送しません。

<a id="verification"></a>
## 検証

```sh
python /path/to/L2S1/mcp/server.py --help
python -m unittest discover -s mcp -p 'test_*.py' -v
python /path/to/L2S1/mcp/smoke_client.py
```

仮想環境の Python で実行してください。テストは HTTP フィクスチャーと subprocess stdio の公式 SDK クライアントを使い、検出、スキーマ・リソース、検証エラー、根拠・保留の無変更返却、バックエンドエラーとプロンプト、任意のポリシー・推論フィールド、再試行しないタイムアウトを検証します。CI はモデルの重みなしで Python 3.10 と 3.14 に同じテストを実行する設定です。スモーククライアントは実際の HTTP バックエンドを必要とし、MCP → HTTP → モデルの実経路で倉庫リクエスト 1 件を検証します。特定のモデルの回答を仮定せず、選択と保留を報告します。
