<a id="l2s1-llamacpp-native-dependency"></a>
# L2S1 llama.cpp ネイティブ 依存関係

[English](../../../en/crates/l2s1-llama-sys/README.md) · [한국어](../../../ko/crates/l2s1-llama-sys/README.md) · [日本語](README.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

このクレートは、同じソース リビジョンから llama.cpp および L2S1 の C++ ブリッジをビルドします。 CMake FetchContent は、ggml-org/llama.cpp コミット `3d82ef62d47fd74e18f36c5eccbdcf965b617b17` をダウンロードし、ビルドする前にソース アーカイブの SHA-256 を検証します。 `cmake/CMakeLists.txt` および `UPSTREAM_COMMIT` を参照してください。 ネイティブ ソースはこのリポジトリにチェックインされていません。上流のソース アーカイブにはそのライセンスが含まれています。このクレートには `THIRD_PARTY_LICENSES.txt` が含まれています。

デフォルトは CPU のみです。CUDA は `l2s1/llama-cuda`、macOS の Metal は `l2s1/llama-metal` を有効にします。CMake 3.24 以上と C++17 コンパイラーが必要で、CUDA には CUDA ツールキットも必要です。1 ビルドで CUDA と Metal を同時に有効にできません。macOS と Linux の `l2s1` ビルドスクリプトは、llama.cpp ライブラリのディレクトリを実行ファイルの rpath に埋め込みます。インストールしたネイティブライブラリも、依存する llama.cpp・GGML ライブラリを同じディレクトリで探します。下流の実行ファイルはビルドスクリプトから `DEP_L2S1_LIBDIR` を読み、独自の rpath を追加できます。ネイティブログのデフォルトは警告です。起動前に `L2S1_LOG=error|warn|info|debug|off` で変更します。モデル読み込み失敗には最後の llama.cpp エラーを含みます。CUDA ビルドは配布成果物向けにホスト固有アーキテクチャの選択を無効にします。既知の対象には `L2S1_CUDA_ARCHITECTURES`（例: `86`）で対象アーキテクチャを制限します。`L2S1_NATIVE_COMPILER_LAUNCHER` に `ccache` の絶対パスなどを指定して llama.cpp の C・C++・CUDA コンパイルをラップできます。`scripts/build_cuda_arch.sh` はインストール済みの `ccache` を自動検出します。Rust のキャッシュには `RUSTC_WRAPPER=sccache` を明示的に指定します。その Rust ラッパーを選択すると `cc` crate も L2S1 C++ ブリッジに `sccache` を使います。

最初のデフォルト ビルドでは、固定されたソースをダウンロードするためにネットワーク アクセスが必要です。オフライン ビルドまたは別の llama.cpp ソース チェックアウトの場合は、`L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp` を設定します。新しい変数が設定されていない場合、従来の `LLAMA_CPP_DIR` が受け入れられます。選択したチェックアウトには、`include/llama.h`、`src/llama-ext.h`、`tools/mtmd/mtmd.h`、`common/jinja`、およびそれらの依存関係が含まれている必要があります。このクレートは、そのソースから独自の ネイティブ ライブラリを構築します。個別に構築されたライブラリがそのヘッダーと混合されることはありません。代替アップストリーム リビジョンは互換性が保証されていないため、使用する前に ネイティブ 契約テストに合格する必要があります。

このビルドには、静止画像を直接入力するための `libmtmd` が含まれており、それを同じ `libllama` リビジョンにリンクします。ビデオサポートは無効になっています。ソースおよびバイナリ パッケージには、一致する `libmtmd` および GGML 共有ライブラリが含まれている必要があります。

この ネイティブ パッケージは、バージョン対応リリースに応じて `l2s1` パッケージより前に公開します。事前にビルドされた実行可能ファイルには、適切なローダー パスでパッケージ化された ネイティブ 共有ライブラリも必要です。ソース パッケージのビルドは、それ自体では再配置可能なバイナリ アーカイブを生成しません。
