# 繰り返す判断と native batching

Rust・TypeScript・Python は同じモデルを常駐させ、入力だけを変更できます。
TS `prepare<State>(decisions)` と Python `prepare(decisions, state_type=State)` は
固定定義を保存します。Python は Pydantic 検証と PEP 561 型情報を提供します。

| 境界 | API | 実行 |
| --- | --- | --- |
| Rust library | `LlamaBackend::decide_batch(&requests)` | parallel は独立 native sequences、他モードは直列 |
| ローカル TS | `load({ executionMode: 'parallel' })` → `decideBatch(requests)` | 既定 stdio、一度で配列を Rust プロセスへ渡す |
| ローカル Python | `await load(LoadOptions(execution_mode="parallel", ...))` → `decide_batch(requests)` | 既定 stdio、共通 Rust 実行ファイル・runtime bundle |
| HTTP | `POST /v1/decision-batches`, `{"requests": [...]}` | native parallel の配列呼び出し |
| Rust 収集 worker | `BackendWorker::spawn_batched()` | 要求数・入力 token・待機時間で制限する microbatch |

## ポートなしのビルド済みモード

`l2s1 --model model.gguf --execution-mode parallel --stdio` を常駐させます。
SDK `load()` はこれを既定とし stdin/stdout JSON lines で通信します。Python に
HTTP サーバーや Node.js は不要です。同一プロセスの N-API/PyO3 拡張ではありません。
Rust はライブラリを直接リンクできます。明示的ローカル HTTP は
`transport: 'http'` / `transport="http"`、共有サーバーは `connect()` を使います。

RPC は `{id:"local-1", op:"decide_batch", body:{requests:[...]}}` です。
`health`・`capabilities`・`decide`・`decide_batch` を提供し同じ id と `result` または
`error` を返します。stdout は RPC、stderr はログ専用です。SDK は最大 16 未完了
呼び出しで、timeout・キャンセル後も応答までスロットを保持します。終了時に子を
回収します。timeout は native 推論停止を保証しません。

## Native batch 契約

全 wire 入力を実行前に検証し media groups を平坦化して native batch を一度
呼びます。Rust は独立 sequences を wave ごとに処理します。state・media・ID・
policy・failure messages は要求ごとに独立し、結果は入力順です。別要求で同じ ID を
使えます。要求 policy は自分の応答のみに適用し原始スコアを保ちます。

成功は `{api_version:1, request_id, execution:"native_parallel", responses:[...]}`。
各項目は既存 v1 応答で `request_id = batch-id/入力インデックス` です。stdio/HTTP は
同じ wire 契約で、TS/Python の相互運用を提供します。

- 最大 128 要求、合計 128 decisions、body 44 MiB。
- `parallel_width` / `parallelWidth` は wave 幅です。大きい batch は複数 wave になります。
- 現在の llama.cpp parallel は direct reasoning、判断ごとに最大 26 選択肢です。
- 全て text、または各判断に一画像と対応 projector が必要です。text/image 混在を
  拒否し、既存の recurrent/hybrid parallel 制約も維持します。
- `capabilities().batch` は対応・有効状態・media・reasoning・上限を表します。
- 未対応は `batch_unsupported`、parallel 無効は `batch_not_enabled`。直列 fallback・
  自動再試行はありません。カスタム backend は native batch メソッドを実装します。
- timeout はバッチ全体です。実行失敗は全体失敗で、実行済み wave は巻き戻し・
  再実行しません。項目別 Result API とは別契約です。

## 繰り返しと性能の証拠

`Promise.all(engine.decide(...))`・`asyncio.gather(...)` は別要求をキューへ入れ、
現在の stdio/HTTP は自動結合しません。明示的 `decideBatch()` / `decide_batch()` を
使います。自動結合には wire media・reasoning・policy・deadline と既存 bounded
Rust worker の adapter が必要です。

`prepare()` は定義・型の再利用で token コンパイルや永続 KV ではありません。
state の変更で既存 `(state, decision)` 準備 cache は miss になります。
`SharedStateSession` は state を固定し質問を変える別 API です。native wave は
完全一致 token prefix のみ共有し、suffix sequences は独立です。

batch shape は確率・選択・abstention を変える可能性があります。処理量・完了時間・
p50/p95・メモリ・スコア差・top-1・policy acceptance を別々に測定します。fixture の
stdio/HTTP と TS↔Python JSON 検査はモデル性能の証拠ではありません。実 GGUF CPU
smoke は CUDA/Metal 性能や数値同等性を検証しません。
