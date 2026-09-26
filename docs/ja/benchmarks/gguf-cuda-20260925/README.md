<a id="generic-gguf-cuda-path-two-model-smoke-measurement"></a>
# 汎用 GGUF CUDA パス: 2 モデル スモーク 測定

[English](../../../en/benchmarks/gguf-cuda-20260925/README.md) · [한국어](../../../ko/benchmarks/gguf-cuda-20260925/README.md) · [日本語](README.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

この実行では、既存の llama.cpp CUDA 実行可能ファイルが 2 つの異なる GGUF モデル ファミリから型付き判断を生成できることを確認します。これは、[`examples/warehouse.json`](../../../../examples/warehouse.json) (SHA-256 `68e3bffae421112de90658c62c5d52293e53b719f7610abb5377b9ad4bbd4944`) の 3 つの判断に対して、ウォームアップなしでモデルごとに 1 回の呼び出しです。 フィクスチャー からの期待値は、`chilled`、`true`、および `high` です。

<a id="runtime-and-method"></a>
## ランタイムとメソッド

- NVIDIA GeForce RTX 3060 (12 GB);各応答は、このデバイスを CUDA オフロード ターゲットとして報告します。
- L2S1 CUDA 実行可能ファイルが、2026-09-24、SHA-256 `b99f8f1e40d6ae542ed780eb3868e72a1c93f735ff488663b073c2335578af62` 上のサーバーにコピーされました。サーバー コピーには Git メタデータがないため、これは既存の CUDA パスのチェックであり、この PR コミットのビルドではありません。どちらの応答も、コンパイルされたランタイム SHA-256 `771abedbe730b88af57cc1bc35543c4353b30382de7bf9c475a78e9814e66043` とロードされたランタイム SHA-256 `ae19e1bb27af03e2057408582ca17f35a748b7fec5aab75db5bef03763a026d8` を記録します。
- 新規実行、コンテキスト 2048、バッチ/ubatch 256、 4 CPU スレッド、FlashAttendant オフ、デフォルト 判断 ポリシー。モデル間でプロンプトまたはランタイム オプションが変更されることはありません。
- コマンド: `l2s1 --model /path/to/model.gguf --device cuda --diagnostics --input examples/warehouse.json`。バイナリは、コピーされた `lib/` ディレクトリからの一致する llama.cpp ライブラリと、GPU サーバーからの CUDA ランタイム ライブラリを使用しました。
- `native_ms` は、応答内のリクエストローカル推論タイマーです。所要時間には、ロード、モデルのハッシュ、JSON 出力、プロセスの起動が含まれます。

<a id="results"></a>
## 結果

| モデル | GGUF SHA-256 | ネイティブ推論 | 経過時間 | 採用した正解 / 全体 | 選択 / 全体 | 正解 / 選択 | ポリシー適用前の top-1 / 全体 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | 130.73 ms | 1.29 s | 0/3 | 3/3 | 0/3 | 0/3 |
| SmolLM2-135M Instruct Q8_0 | `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83` | 111.14 ms | 0.76 s | 0/3 | 0/3 | 未定義 | 0/3 |

Qwen は、`ambient`、`false`、および `low` を選択しました。 SmolLM2 は 3 つの判断すべてにおいて判断保留した。 採用された判断の正解率 は未定義です。結果は、両方の GGUF が CUDA 判断 パスを介して実行されたことを証明します。モデルごとに 3 つの総合的な判断と 1 つのタイミング サンプルでは、​​タスク 正解率 または安定したレイテンシは確立されません。

完全なモデル ID、オプション スコア、候補質量、選択、判断保留 理由、およびステージ タイミングは、[`qwen3-q8-rtx3060.json`](../../../../benchmarks/gguf-cuda-20260925/qwen3-q8-rtx3060.json) および [`smollm2-q8-rtx3060.json`](../../../../benchmarks/gguf-cuda-20260925/smollm2-q8-rtx3060.json) にあります。モデルの重量は含まれておりません。
