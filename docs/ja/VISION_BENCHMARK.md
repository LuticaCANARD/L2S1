<a id="direct-vision-http-benchmark"></a>
# 画像の直接推論 HTTP ベンチマーク

[English](../en/VISION_BENCHMARK.md) · [한국어](../ko/VISION_BENCHMARK.md) · [日本語](VISION_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

これは 2026 年 9 月 24 日に `lucatagpu`（RTX 3060 12 GiB、ドライバー 595.71.05）で測定した **ローカルの合成スモークベンチマーク**です。同じ CUDA 対応 L2S1 実行ファイル、Gemma 4 E2B Q8_0 GGUF、対応する `mmproj`、リクエストスキーマ、ホストの CPU と CUDA 経路を比較します。モデル・プロジェクターの SHA-256 は [CPU 生レポート](../../benchmarks/vision-20260924/cpu.json)と [CUDA 生レポート](../../benchmarks/vision-20260924/cuda.json)に記録されています。モデルの重みはリポジトリにありません。

[実行スクリプト](../../scripts/benchmark_vision_http.py)は、デバイスごとに独立したループバック HTTP サーバーを起動し、`/healthz` を待ち、測定から除く 4 回のウォームアップを送信します。その後リポジトリの赤・青 64×64 PNG を交互に使い、30 回の直列 `POST /v1/decisions` を測定します。JSON・base64 は測定前に準備します。区間には HTTP 転送、画像エンコード、モデル推論、スコア計算、応答の直列化を含みます。各実行は読み込み済みの 1 モデルで処理し、並列クライアントやリクエスト間のバッチは測定しません。両方ともコンテキスト 2048、バッチ 256、CPU 4 スレッド、`--model-load-mode read` を使用しました。GPU を CPU より先に実行しました。

| デバイス | p50 | p95 最近順位 | 平均 | 起動から `/healthz` | フィクスチャーの選択 | 読み込み中の GPU メモリ | プロセス最大 RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 CUDA | 96.054 ms | 96.236 ms | 96.021 ms | 5.013 s | 30/30 | 3,813 MiB | 3.47 GiB |
| 同じホストの CPU | 3,066.295 ms | 3,162.112 ms | 3,079.159 ms | 5.263 s | 30/30 | 20 MiB | 5.94 GiB |

この 2 画像の反復処理で CPU/CUDA の p50 比は **31.9×** でした。30 選択はラベル付き単色画像 2 枚の反復です。30/30 はインターフェースのスモーク確認で、30 独立例や一般的な画像正解率ではありません。RSS と GPU メモリは異なる尺度で、加算したり総システムメモリとみなしたりできません。起動時間にはプロセス起動、モデル・プロジェクター読み込み、準備確認を含みます。レイテンシ列は起動とウォームアップを除きます。この 1 回の直列実行は、同時処理量や本番レイテンシを保証しません。

<a id="reproduce"></a>
## 再現

`--features llama-cuda` で L2S1 をビルドし、対応する画像 GGUF と `mmproj` を用意し、ビルドに対応したネイティブ共有ライブラリを `LD_LIBRARY_PATH` で利用可能にします。両デバイスで同じ実行ファイルを使います。実行スクリプトは既存レポートを上書きせず、全ウォームアップ・測定観測とモデル・プロジェクター・実行ファイル・画像のハッシュを保存します。

```sh
python3 scripts/benchmark_vision_http.py \
  --binary target/release/l2s1 \
  --model /path/to/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --red-image tests/fixtures/vision_red_64.png \
  --blue-image tests/fixtures/vision_blue_64.png \
  --device cuda --warmup 4 --iterations 30 \
  --output results/vision-http-cuda.json
```

別の出力パスと空いている `--listen` アドレスで `--device cpu` を繰り返します。保存済みのレポートは PR #12 に分離する前の画像実装で生成されました。実行ファイル SHA-256 がそのビルドを識別します。PR #12 ソースは独立した CPU 契約テストに合格していますが、これらの時間は記録されたホストのビルドと設定に関する値です。
