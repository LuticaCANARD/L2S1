<a id="l2s1-llamacpp-native-dependency"></a>
# L2S1 llama.cpp ネイティブ 依存関係

[English](../../../en/crates/l2s1-llama-sys/README.md) · [한국어](../../../ko/crates/l2s1-llama-sys/README.md) · [日本語](README.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

このクレートはモデル所有権、テキスト・ビジョンの実行、prefix reuse、状態スナップショット、thinking 制限と compact evidence を Rust で実装します。llama.cpp/GGML/mtmd をビルドし、同じヘッダーから Rust ABI バインディングを生成します。小さな `native/chat.cpp` は upstream Jinja レンダリングを維持し、`native/exception.cpp` は C++ 例外が Rust フレームを越えないよう処理します。実行計画とスコアリングは Rust にあります。CMake FetchContent は、ggml-org/llama.cpp コミット `3d82ef62d47fd74e18f36c5eccbdcf965b617b17` をダウンロードし、ビルドする前にソース アーカイブの SHA-256 を検証します。 `cmake/CMakeLists.txt` および `UPSTREAM_COMMIT` を参照してください。 ネイティブ ソースはこのリポジトリにチェックインされていません。上流のソース アーカイブにはそのライセンスが含まれています。このクレートには `THIRD_PARTY_LICENSES.txt` が含まれています。

デフォルトは CPU のみです。CUDA は `l2s1/llama-cuda`、macOS の Metal は `l2s1/llama-metal` を有効にします。CMake 3.24 以上、C++17 コンパイラー、および bindgen 用の Clang/libclang が必要です。Debian/Ubuntu では `libclang-dev`、macOS では Homebrew `llvm`、Windows では MSVC と LLVM を使い、必要に応じて `LIBCLANG_PATH` を指定してください。CUDA には CUDA ツールキットも必要です。1 ビルドで CUDA と Metal を同時に有効にできません。macOS と Linux の `l2s1` ビルドスクリプトは、llama.cpp ライブラリのディレクトリを実行ファイルの rpath に埋め込みます。インストールしたネイティブライブラリも、依存する llama.cpp・GGML ライブラリを同じディレクトリで探します。下流の実行ファイルはビルドスクリプトから `DEP_L2S1_LIBDIR` を読み、独自の rpath を追加できます。ネイティブログのデフォルトは警告です。起動前に `L2S1_LOG=error|warn|info|debug|off` で変更します。モデル読み込み失敗には最後の llama.cpp エラーを含みます。CUDA ビルドは配布成果物向けにホスト固有アーキテクチャの選択を無効にします。既知の対象には `L2S1_CUDA_ARCHITECTURES`（例: `86`）で対象アーキテクチャを制限します。`L2S1_NATIVE_COMPILER_LAUNCHER` に `ccache` の絶対パスなどを指定して llama.cpp の C・C++・CUDA コンパイルをラップできます。`scripts/build_cuda_arch.sh` はインストール済みの `ccache` を自動検出します。Rust のキャッシュには `RUSTC_WRAPPER=sccache` を明示的に指定します。その Rust ラッパーを選択すると `cc` crate も 残っている C++ Jinja アダプターに `sccache` を使います。

最初のデフォルト ビルドでは、固定されたソースをダウンロードするためにネットワーク アクセスが必要です。オフライン ビルドまたは別の llama.cpp ソース チェックアウトの場合は、`L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp` を設定します。新しい変数が設定されていない場合、従来の `LLAMA_CPP_DIR` が受け入れられます。選択したチェックアウトには、`include/llama.h`、`src/llama-ext.h`、`tools/mtmd/mtmd.h`、`common/jinja`、およびそれらの依存関係が含まれている必要があります。このクレートは、そのソースから独自の ネイティブ ライブラリを構築します。個別に構築されたライブラリがそのヘッダーと混合されることはありません。代替アップストリーム リビジョンは互換性が保証されていないため、使用する前に ネイティブ 契約テストに合格する必要があります。

このビルドには、静止画像を直接入力するための `libmtmd` が含まれており、それを同じ `libllama` リビジョンにリンクします。ビデオサポートは無効になっています。ソースおよびバイナリ パッケージには、一致する `libmtmd` および GGML 共有ライブラリが含まれている必要があります。

この ネイティブ パッケージは、バージョン対応リリースに応じて `l2s1` パッケージより前に公開します。事前にビルドされた実行可能ファイルには、適切なローダー パスでパッケージ化された ネイティブ 共有ライブラリも必要です。ソース パッケージのビルドは、それ自体では再配置可能なバイナリ アーカイブを生成しません。


安定した `sd_*` ABI は Rust 呼び出し側から引き続き利用できます。以前の `native/bridge.cpp` は `src/bridge.rs`、`src/text.rs`、`src/vision.rs` に置き換わりました。ネイティブ割り当てはスコープを持つ所有者が管理し、エラー経路では汚染され得る KV 状態を消去します。生成したバインディングと Rust 実装のソースはランタイム fingerprint に含まれます。llama.cpp 自体の Rust 再実装ではありません。型付きの常駐プロセス呼び出しは [C++ SDK](../../../../sdks/cpp/README.md)、以前の C++ 実装との比較範囲と再現方法は[移行検証](../../../RUST_BRIDGE_MIGRATION.md)を参照してください。

<a id="portable-linux-arm64"></a>
## 汎用 Linux ARM64 ビルド

`L2S1_PORTABLE_BUILD=1` は `GGML_NATIVE=OFF` を維持します。Linux aarch64 では、ARMv8 基本カーネルを含む upstream の ARM CPU モジュールをビルドします。GGML は Linux HWCAP/HWCAP2 を確認してから dotprod、FP16、SVE、i8mm、SME のカーネルを選択します。機能判定コードは、その ISA オプションを使わず LTO を無効にして別途コンパイルします。ブリッジはリンクされた GGML と同じディレクトリのモジュールを読み込むため、バイナリを移動するときは対応するライブラリ一式も移動してください。Cargo テストも同じライブラリ一式を使います。

ARMv9.2/SME を含むこの upstream カーネル一覧は GCC 14 で検証しました。Linux ARM64 パッケージの CI は Ubuntu 24.04 と GCC 14 を使い、互換性のある glibc と C++ ランタイムが必要です。以前の Linux ディストリビューションではソースビルドが必要になる場合があります。

`L2S1_ARM64_DISPATCH=0` は比較測定用に従来の汎用 CPU 実装を選択します。`L2S1_OPENMP=0` / `1` は OpenMP を独立して無効化 / 有効化します。両設定はネイティブビルドのキャッシュキーに含まれます。汎用ビルドでは既定で OpenMP を無効にします。独自の OpenMP パッケージには、GCC の `libgomp.so.1` など対応するコンパイラの OpenMP ランタイムが必要です。パッケージ検証は、動的に読み込む CPU モジュールを含むすべての共有ライブラリの依存関係を確認します。異なるビルドや upstream revision のライブラリを混在させないでください。

Rust の evidence SIMD 検査は GGML の行列演算を高速化しません。CPU カーネル比較では `--execution-mode fresh` を明示してください。常駐 prefix reuse は実行する演算量自体を変えます。
