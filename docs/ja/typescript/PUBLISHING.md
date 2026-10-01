<a id="npm-publishing-review"></a>
# npm 公開の検討

[English](../../en/typescript/PUBLISHING.md) · [한국어](../../ko/typescript/PUBLISHING.md) · [日本語](PUBLISHING.md)

[English index](../../en/README.md) · [한국어 색인](../../ko/README.md) · [日本語索引](../README.md)

ラッパーとプラットフォームランタイムは公開 npm パッケージとして配布できます。通常の npm tarball、ESM exports、TypeScript 宣言、任意のプラットフォーム依存関係を使います。最終ユーザーに Rust・C++ コンパイラーは不要です。公開には、ネイティブプラットフォームビルドの合格と npm scope・公開権限の取得が必要です。

<a id="package-layout"></a>
## パッケージ構成

| パッケージ | 内容 | ランタイム |
| --- | --- | --- |
| `@l2s1/node` | JavaScript、宣言、ソース、README、MIT ライセンス | Node.js 22 以上。HTTP サブパスはブラウザー向けにバンドル可能 |
| `@l2s1/runtime-linux-x64` | Rust 実行ファイルと対応する共有ライブラリ | CPU、glibc Linux x64 |
| `@l2s1/runtime-linux-arm64` | Rust 実行ファイルと対応する共有ライブラリ | CPU、glibc Linux arm64 |
| `@l2s1/runtime-darwin-x64` | Rust 実行ファイルと対応する共有ライブラリ | CPU、macOS x64 |
| `@l2s1/runtime-darwin-arm64` | Rust 実行ファイルと対応する共有ライブラリ | CPU・Metal、macOS arm64 |
| `@l2s1/runtime-win32-x64` | Rust 実行ファイルと対応する DLL | CPU、Windows x64 |

ラッパーは完全に一致するランタイムのバージョンを参照します。`os`、`cpu`、Linux `libc` のメタデータで非互換の任意パッケージのインストールを防ぎます。ランタイム解決はパッケージ識別、バージョン、要求デバイスを検証します。ランタイムがなければ明示的なエラーとなり、HTTP・カスタムバックエンドはランタイムのインストールなしで動作します。CUDA は別途ビルドした実行ファイルを `binaryPath` で指定して使えます。

ラッパーと生成ランタイムは `publishConfig.access = public` と npm registry URL を使います。npm によれば scoped パッケージのデフォルトは非公開のため、公開時には public アクセスを明示します。[npm scoped パッケージの文書](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/)

<a id="release-conditions"></a>
## リリース条件

1. npm `@l2s1` ユーザー・組織 scope の所有権または公開権限を確認します。GitHub リポジトリの所有者が npm scope の所有者とは限りません。未認証の registry 404 は、scope が利用可能という証拠ではありません。
2. リリースコミットを `v*` バージョンタグで push するか手動で実行します。4 ワークフローはタグ push・手動実行だけで開始し、PR・ブランチ push は CI を開始しません。`typescript-runtimes.yml` の 5 つのネイティブビルド・インストールジョブを通します。ローカル Linux ビルドは macOS・Windows・arm64 成果物の検証ではありません。
3. 全 tarball を `npm publish --dry-run --access public --ignore-scripts` で確認します。バージョン、ファイル、ライセンス、モデルの重みとインストールスクリプトがないことを確認します。
4. 同じバージョンのランタイム tarball 5 つを先に公開し、ラッパーを公開します。ラッパーを先に公開すると、npm の任意依存関係の処理によりローカル推論が使えないインストールになる場合があります。
5. 新しいプロジェクトで registry 名からラッパーをインストールし、自動ランタイム選択と実モデルの判断を検証します。最終 registry テストには実際の公開が必要で、ローカル tarball のインストールは準備の根拠です。

公開 scoped パッケージには npm アカウント、scope 権限、対応する公開認証が必要です。2026-09-26 の確認ではローカル npm CLI は未認証（`npm whoami` は `ENEEDAUTH`）、未認証の `npm view @l2s1/node` は 404 でした。この検討でパッケージは公開していません。


<a id="binary-and-license-boundaries"></a>
## バイナリーとライセンスの範囲

ネイティブパッケージはプロジェクトの MIT ライセンス、第三者の表記、SHA-256 ファイル一覧を含みます。エンジン・ヘッダー・ブリッジ・共有ライブラリは、同じ固定 llama.cpp ソースでビルドします。モデルの重みは含めず、それぞれの条件に従って別途用意します。HTTP 専用ラッパーは Node・ブラウザー API を使い、第三者の JavaScript ランタイムライブラリは追加しません。

Linux ビルドは Ubuntu 22.04 を使います。対応する glibc・C++ ランタイム要件を維持し、リリース前にサポートを表明する最も古いディストリビューションでテストしてください。Alpine・musl は同梱対象外です。Windows は Microsoft Visual C++ x64 ランタイムが必要です。macOS 配布・Metal 対応はネイティブジョブで確認します。一般 CPU カーネルはビルドホストの ISA を仮定せず、ホスト最適化ビルドと性能が異なる場合があります。

基本の main ブランチの HTTP 仕様は状態・判断・メディアに対応します。任意の reasoning・policy 拡張は、アプリが選び、サーバーが対応するときだけ送ります。古いサーバーは未対応フィールドを拒否します。起動時のポリシーは `L2S1.load({ policy })` で引き続き利用できます。

<a id="verification-scope"></a>
## 検証範囲

リリース CI は TypeScript と例の型、HTTP 転送、カスタムバックエンドの経路、起動失敗・取消・終了処理、実際の Rust HTTP の検証とスコア計算、梱包したラッパー・ランタイム tarball のオフラインインストールを確認します。ネイティブパッケージ検証はファイルハッシュ、実行ファイルの版、バンドル内での Linux 共有ライブラリ解決を確認します。プラットフォーム CI は不正モデルで起動を確認し、モデルはダウンロードしません。これは実行ファイルの読み込み確認で、推論品質の確認ではありません。

実際の GGUF スモークテストには別途重みが必要です。保留を含む型付き結果とスコア構造を検証し、タスク正解率や GPU 動作は実証しません。

`release.yml` は検証済み npm・PyPI・Cargo パッケージを公開し、その後 GitHub Release を公開します。アカウントと trusted publisher は別途設定します。[配布パイプライン](../RELEASE_PIPELINE.md)を参照してください。
