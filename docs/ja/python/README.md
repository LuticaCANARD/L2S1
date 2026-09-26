<a id="l2s1-for-python"></a>
# Python 用 L2S1

[English](../../en/python/README.md) · [한국어](../../ko/python/README.md) · [日本語](README.md)

Python 3.11+ の非同期 SDK です。`@l2s1/node` と同じ常駐 Rust エンジンと
HTTP v1 JSON を使い、Pydantic の実行時検証と PEP 561 型情報を提供します。
[l2s1-sdk 0.1.1](https://pypi.org/project/l2s1-sdk/0.1.1/) を PyPI に公開しました。import は `import l2s1` を維持します。

<a id="install"></a>
## インストール

PyPI からインストールします。

```sh
pip install l2s1-sdk==0.1.1
```

ソースまたはビルド済み wheel からインストール:

```sh
python -m pip install ./sdks/python
python -m pip install ./sdks/python/dist/l2s1_sdk-0.1.1-py3-none-any.whl
```

wheel は Python SDK を含みます。Rust 実行ファイルと GGUF は別途用意します。
`load()` は `binary_path` または PATH の `l2s1` を使います。`runtime_dir` に
展開した `@l2s1/runtime-<platform>` を指定すれば同じ実行ファイルとネイティブ
ライブラリを使えます。バージョン・OS・CPU・デバイス・チェックサムを検証します。
Node.js は不要です。モデル取得、コンパイル、インストール時スクリプトはありません。

<a id="repeat-a-decision-with-new-data"></a>
## 入力だけを変えて繰り返す

```python
import asyncio
from pydantic import BaseModel, ConfigDict
from l2s1 import BinaryKind, BinaryValue, Decision, L2S1, LoadOptions

class Temperature(BaseModel):
    model_config = ConfigDict(strict=True)
    temperature_c: float

async def main() -> None:
    async with await L2S1.load(LoadOptions(
        model="/path/to/model.gguf", binary_path="/path/to/l2s1",
        execution_mode="parallel",
    )) as engine:
        plan = engine.prepare([Decision(
            id="cold", instruction="Is temperature_c below 10?",
            kind=BinaryKind(false_label="At least 10.", true_label="Below 10."),
        )], state_type=Temperature)
        first = await plan.decide(Temperature(temperature_c=6))
        second = await plan.decide(Temperature(temperature_c=15))
        results = await plan.decide_batch([
            Temperature(temperature_c=2), Temperature(temperature_c=20),
        ])
        result = first.results[0]
        if isinstance(result.value, BinaryValue):
            print(result.value.value)  # bool | None

asyncio.run(main())
```

`prepare()` は固定した定義をコピーします。`state_type` を指定すると
`PreparedDecision[Temperature]` を返し、mypy は呼び出しの型、SDK は実行時の
モデル型を検証します。省略すると JSON 値を受け取ります。入力検証は正答を
保証しません。この準備はトークンのコンパイルや永続 KV 再利用ではありません。

`load()` は既定で `transport="stdio"` を使います。ビルド済み Rust 実行ファイルを
一度起動し、stdin/stdout の JSON で通信します。ポート・HTTP サーバー・Node.js は
不要です。同一プロセスの PyO3 拡張ではありません。ローカル HTTP 接続は
`transport="http"`、共有サーバーは `connect()` を使います。

`decide_batch()` は配列を一度で Rust native parallel 実行器へ渡します。
`execution_mode="parallel"` と `parallel_width` を指定します。HTTP では
`POST /v1/decision-batches` に `{"requests": [...]}` を送ります。state・ID・media・
ポリシーは要求ごとに独立し、結果は入力順です。別要求で同じ ID を使えます。
envelope は `execution: "native_parallel"`、`timeout_ms` はバッチ全体に適用します。
直列 fallback や自動再試行はありません。未対応は `batch_unsupported`、parallel
モード無効は `batch_not_enabled` です。カスタム native adapter は `BatchDecisionBackend` を実装します。

最大 128 要求・合計 128 decisions です。現在は direct reasoning と判断ごとに
最大 26 選択肢に対応します。全て text、または各判断に画像が一つと対応する
projector が必要です。text/image 混在は拒否します。全 wire 入力を実行前に検証し、
実行失敗はバッチ全体の失敗です。実行済み wave は巻き戻し・再実行しません。
`asyncio.gather(decide(...))` は自動バッチではありません。
[バッチ API の検討](../BATCHING_API_REVIEW.md)を参照してください。

<a id="connect-send-existing-json-or-adapt-another-transport"></a>
## HTTP 接続とカスタムバックエンド

```python
from l2s1 import DecisionRequest, L2S1

async with L2S1.connect("http://127.0.0.1:8080") as engine:
    capabilities = await engine.capabilities()
    request = DecisionRequest.model_validate(payload)
    response = await engine.decide(request, timeout_ms=180_000)
    payload_for_typescript = response.model_dump(exclude_unset=True)
```

`payload` は TypeScript と同じ JSON です。`load/connect/from_backend` は同じ API を
提供します。`DecisionBackend` Protocol は非同期 `decide/capabilities` を要求し、
任意の同期・非同期 `close()` があれば facade がリソースを所有します。独立した
HTTP トランスポートとして `L2S1Client` も提供します。

| TypeScript | Python |
| --- | --- |
| `load({ model, binaryPath })` | `await load(LoadOptions(model=..., binary_path=...))` |
| `connect({ baseUrl })` | `connect(base_url)` |
| `fromBackend(backend)` | `from_backend(backend)` |
| `decide(request)` | `await decide(request)` |
| `prepare<State>(decisions)` | `prepare(decisions, state_type=State)` |
| `plan.decide(state)` | `await plan.decide(state)` |
| `decideBatch(items)` | `await decide_batch(items)` |
| async disposal | `async with` / `await close()` |
| `AbortSignal` | asyncio task のキャンセル |

Python のメソッドとオプションは snake case、JSON キーは共通です。`false`、`null`、
結果の順番・ID、`model_scored/selection_only` を保持します。
`model_dump(exclude_unset=True)` で省略された拡張項目を保ち、判断種類の既定
`type` は常にシリアライズします。

画像、reasoning、要求ポリシーは `capabilities()` で対応を確認してください。
指定された制御を黙って無視しません。スコアと `target_error_rate` は正答率の
保証ではありません。`L2S1Error` はサーバーの `code/status/request_id/user_reason` を
保持し、ネットワーク・キャンセルは HTTPX/asyncio の例外を保持します。自動再試行はありません。

`close()` は冪等です。HTTP クライアントを閉じるとローカル要求をキャンセルし、
共有サーバーは動き続けます。所有エンジンを閉じると Rust 子プロセスを終了して
待機します。キャンセル・timeout は native 推論停止を保証しません。既定の stdio は
ポートを開かず stderr を読み続けます。明示的 HTTP のみ loopback を使います。
stdio は最大 16 未完了呼び出しで、timeout 後も native 応答までスロットを保持します。

<a id="build-and-verify"></a>
## ビルドと検証

```sh
python -m pip install './sdks/python[dev]'
python -m mypy --config-file sdks/python/pyproject.toml sdks/python/src sdks/python/examples sdks/python/tests/typecheck.py
python -m unittest discover -s sdks/python/tests -v
python -m build sdks/python
cargo build --locked --example typescript_fixture
npm --prefix sdks/typescript ci
npm --prefix sdks/typescript run build
L2S1_TEST_BINARY="$PWD/target/debug/examples/typescript_fixture" \
  python -m unittest discover -s sdks/python/tests -v
```

Windows は実行ファイルに `.exe` を付けます。Rust fixture はモデルを使わず、
実際のプロセス・stdio/HTTP バッチ・スコア計算と Python↔TypeScript JSON 一致を検証します。
GGUF/CUDA/Metal の品質・性能証拠とは別です。CI は Linux・macOS・Windows、
Python 3.11/3.14 を対象として wheel/sdist を保存します。ローカルの成功は他の
プラットフォームの成功を意味しません。リリースパイプラインが検証済み wheel/sdist を PyPI に公開します。

[配布パイプライン](../RELEASE_PIPELINE.md)は設定済み trusted publisher で GitHub Release・npm・PyPI・Cargo に検証済み配布物を公開します。[v0.1.1](https://github.com/LuticaCANARD/L2S1/releases/tag/v0.1.1) をインストールできます。
