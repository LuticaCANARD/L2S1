<a id="verification-guide"></a>
# 検証ガイド

[English](../en/VERIFICATION.md) · [한국어](../ko/VERIFICATION.md) · [日本語](VERIFICATION.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

使用する予定の正確な チェックポイント、ランタイム、および実行構成のチェックを実行します。単体テスト、ネイティブ 契約テスト、ラベル付きタスク評価は、さまざまな質問に答えます。

<a id="build-and-general-checks"></a>
## ビルドと一般的なチェック

固定された llama.cpp リビジョンは `3d82ef62d47fd74e18f36c5eccbdcf965b617b17` です。 CMake FetchContent は、最初のデフォルト ビルドでそれをダウンロードし、アーカイブ ハッシュを検証します。 sys 依存関係により、一致するソース、ヘッダー、共有ライブラリが構築されます。 CPU および CUDA ビルドは Linux 上で実行されます。 `llama-metal` は、macOS 上の Metal デバイスと同じ ネイティブ API を構築します。オフライン チェックにローカル チェックアウトを使用するか、別の上流リビジョンをテストするには、`L2S1_LLAMA_CPP_SOURCE` を設定します。

```sh
cargo fmt --all -- --check
cargo test --locked --offline
cargo test --release --locked --offline --features llama
cargo clippy --release --locked --offline --all-targets --features llama -- -D warnings
python3 -m unittest discover -s scripts -p 'test_*.py'
```

GPU チェックの場合は、`--features llama-cuda` でビルドし、ネイティブ テストには `SKID_CUDA=1` を使用します。 CPU のみのビルドは、CUDA リクエストを拒否します。

Mac では、`cargo build --release --locked --features wgpu --bin l2s1-wgpu` と `cargo build --release --locked --features llama-metal --bin l2s1` を使用して両方の Metal ルートを構築します。 1 つ目は Gemma 4 の `WGPU_BACKEND=metal --require-metal` で実行し、2 つ目は一般的な GGUF モデルの `--device metal` で実行します。実際の Metal 推論と等価性には、一致する GGUF ファイルを備えた Mac が必要です。 Gemma 4 リクエストローカル モードと同等の場合は、`L2S1_WGPU_GEMMA_MODEL` および `L2S1_WGPU_GEMMA_MMPROJ` を設定してから、`cargo test --release --locked --features wgpu --test wgpu_execution -- --ignored` を実行します。

CUDA 上の別の GGUF ファミリの場合は、新しいモデル パスで同じバイナリを使用します。タスクの品質を比較する前に、`--inspect`、`--preflight --input examples/warehouse.json`、および実際の 判断 リクエストを実行します。以下のマルチモデル適合性テストでは、コロンで区切られた GGUF パスと `SKID_CUDA=1` を受け入れます。成功したロードまたは有限の 候補の確率質量 は、ラベル付きタスクの正確性ではなく、この 判断 パスとの互換性を確立します。

直接静止画入力の場合は、互換性のあるビジョン GGUF と、それに一致する `mmproj` GGUF を提供します。無視されたテストでは 2 つの異なる PNG が使用され、画像コンテンツが生の logits に影響を与えるかどうかがチェックされ、無効な画像の後の回復がチェックされます。タスク 正解率 は確立されません。

```sh
SKID_VISION_MODEL=/path/to/vision-model.gguf \
SKID_VISION_MMPROJ=/path/to/mmproj.gguf \
cargo test --locked --offline --features llama --test vision -- --ignored
```

許可された CUDA ホスト上の同じコントラクトには、`--features llama-cuda` と `SKID_CUDA=1` を使用します。 HTTP パスには、`/healthz` および `/v1/capabilities` のライブ ループバック チェック、`media` という名前の有効な `POST /v1/decisions`、および HTTP 400 を返す不正なリクエストがさらに必要です。 `error.code` および `error.request_id` と。

Offline Cargo コマンドには、事前に設定された CMake ソース キャッシュ、またはローカル チェックアウトに設定された `L2S1_LLAMA_CPP_SOURCE` も必要です。一般的なテストの実行では、モデル ファイルを必要とするテストがスキップされます。重みをダウンロードしたり、実際のモデルの互換性を確立したりすることはありません。アップストリーム C++ ヘルパーの警告は、Rust lint の結果とは別のものです。

<a id="model-contract-checks"></a>
## モデル契約チェック

```sh
L2S1_CONFORMANCE_MODELS=/path/to/model-a.gguf:/path/to/model-b.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/l2s1-conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

`SKID_CUDA=1`をCUDAに設定します。適合スイートは、セマンティック ID、すべての 判断 種類、トークン マッピング、アーティファクト バインディング、コンテキスト障害、リカバリ、スナップショット制限、および実行診断をチェックします。 0.02 の既存の確率/質量許容値、変更されていない上位の選択肢、および変更されていない受け入れられた結果を使用して、最適化されたモードと新規実行を比較します。並行した相違点は個別に報告されます。完了したレポートは自動的に等価パスにはなりません。

CUDA ホスト上の検証されたハイブリッド Bonsai パスに対して、実際のモデルの状態復元回帰を実行します。

```sh
L2S1_BONSAI_MODEL=/path/to/Bonsai-27B-Q1_0.gguf \
  cargo test --release --locked --features llama-cuda \
  --test state_restore_bonsai -- --ignored --nocapture
```

16-判断 共有状態リクエストを、新しいスコアとポリシー判断、実際に再利用されたトークン、スナップショットのバイト制限、強制フォールバック後のリカバリと照合してチェックします。これは、一般的なタスク 正解率 ではなく、1 つの フィクスチャー での実行と等価です。

特定の実行パスには、[プレフィックス再利用チェック](SEMIF_ALGORITHM.md#reproduce)および[パラレルコントラクトチェック](PARALLEL_EXECUTION.md#validation-and-measurement)を使用してください。パラレル モードには既知の数値的な違いがあり、オプトインのままです。 事前検証 をロードまたは渡すモデルでも、推論とラベル付きワークロード評価が必要です。

<a id="optional-optimization-checks"></a>
## オプションの最適化チェック

```sh
SKID_MODEL=/path/to/model.gguf \
  cargo test --release --locked --offline --features llama \
  --test native_compact --test optimization_contract --test shared_state \
  --test worker_native -- --include-ignored --test-threads=1

cargo run --release --locked --offline --features llama \
  --example benchmark_optimizations -- \
  --model /path/to/model-a.gguf --model /path/to/model-b.gguf \
  --output /tmp/l2s1-optimizations.json --repeats 3
```

チェックポイント ごとに契約チェックを繰り返します。 CUDA を測定する場合は、テストには `SKID_CUDA=1` を使用し、ベンチマークには `--cuda` を使用します。この契約には、正確なキャッシュされたトークンの準備、動的な命令とオプションの順序、コンパクト/完全な証拠の等価性、セッション エラーのクリーンアップ、ネイティブ ワーカー チケットの相関関係が含まれます。コンパクト転送では、全語彙 ノーマライザーが保持されます。ホスト側のバッファ コピーを Rust に削除します。語彙の射影や GPU からホストへの転送は削除されません。

このベンチマークは、短い合成状態と拡張された合成状態を持つ 1、 4、および 16 の質問を測定し、実行順序をローテーションし、読み込みを除外し、生の応答、ID、ステージ タイミング、およびキャッシュ統計を記録します。キャッシュ測定では、ウォームアップ後に意図的に繰り返し入力を使用します。共有状態測定では、1 つのセッション内で異なる質問を含む個別の呼び出しが発行されます。各最適化を同じレイアウトの新しいベースラインと比較します。状態優先プロンプトは、再利用とは独立して予測を変更できます。質問ごとの償却時間はスタンドアロンのリクエスト レイテンシーではなく、これらのワークロードはタスク 正解率 または 本番環境 スループットを確立しません。レポートは上書きを拒否し、無視されるローカル出力ディレクトリに属します。

<a id="task-quality-and-evidence"></a>
## タスクの品質と証拠

- [合成ベンチマーク](BENCHMARK.md): ルールフィクスチャーの正確性、判断保留、一貫性および待ち時間。
- [AG ニュース評価](KAGGLE_BENCHMARK.md): 凍結された分類プロトコル。
- [JevBench 評価](JEVBENCH.md): パブリック タスク マッピング、アップストリーム スコアリング、および個別の受け入れメトリクス。

チェックポイント、ランタイム、プロンプト、構成の ID を、無視されたローカル出力ディレクトリ内の生の予測とラベルとともに保持します。レイテンシー、採用率、採用された判断の正解率、および 候補間の相対 信頼性を分けてください。生成されたレポートとホスト固有の実行計画はソース ドキュメントではありません。 README には、コンパクトなモデル比較の概要が含まれています。これは公式のリーダーボードや 本番環境-正解率 の主張ではありません。
