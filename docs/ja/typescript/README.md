<a id="l2s1-for-typescript"></a>
# L2S1 TypeScript

[English](../../en/typescript/README.md) · [한국어](../../ko/typescript/README.md) · [日本語](README.md)

[English index](../../en/README.md) · [한국어 색인](../../ko/README.md) · [日本語索引](../README.md)

`@l2s1/node` は、Node.js の既存の Rust 推論エンジンを使用します。 1 つのモデル プロセスは呼び出しをまたいで常駐します。バイナリ、選択、順序付き、画像、ポリシー、推論、および証拠フィールドは、Rust HTTP v1 スキーマを使用します。検証、トークン化、スコアリング、判断保留 および GPU の実行は Rust に残ります。

Node.js 22+ が必要です。ラッパーは、現在の OS および CPU アーキテクチャ用のオプションの事前構築済みランタイム パッケージを選択します。各ランタイム パッケージには、Rust 実行可能ファイルと、対応する llama.cpp/GGML 共有ライブラリが含まれています。インストールではコンパイル スクリプトやダウンロード スクリプトは実行されません。モデルウェイトは別途提供されます。これらのパッケージは配布用に準備されていますが、npm には公開されていません。

Scope 権限、パッケージ順序、認証、プラットフォームのリリース条件は [npm 公開の検討](PUBLISHING.md)を参照してください。

<a id="install-from-this-repository"></a>
## このリポジトリからインストールする

エンド ユーザーの場合は、ビルド ワークフローによって生成されたラッパーとプラットフォーム ランタイム tarball をインストールします。たとえば、Linux x64 の場合は次のようになります。

```sh
npm install ./l2s1-node-0.1.1.tgz ./l2s1-runtime-linux-x64-0.1.1.tgz
```

パッケージが公開された後、`npm install @l2s1/node` はオプションの依存関係を通じてランタイムを選択します。オプションの依存関係を有効にしておきます。ランタイムとラッパーのバージョンは一致する必要があります。

カスタム ネイティブ ビルドの場合は、リポジトリ ルートで Rust 実行可能ファイルをビルドします。

```sh
cargo build --release --locked --features llama --bin l2s1
```

NVIDIA CUDA の場合は、`--features llama-cuda` を使用します。 macOS Metal の場合は、`--features llama-metal` を使用します。ツールチェーンの要件と共有ライブラリについては、[native ビルド ガイド ](../GUIDE.md#build) を参照してください。配布するときは、実行可能ファイルとそれに必要な ネイティブ ライブラリを一緒に保管してください。

TypeScript パッケージをビルドしてパックします。

```sh
cd sdks/typescript
npm ci
npm run build
npm pack
# In your application:
npm install /path/to/L2S1/sdks/typescript/l2s1-node-0.1.1.tgz
```

<a id="load-a-local-model"></a>
## ローカルモデルをロードする

```ts
import { L2S1 } from '@l2s1/node';

const engine = await L2S1.load({
  model: '/path/to/chat-model.gguf',
  device: 'cpu',
});
try {
  const response = await engine.decide({
    state: { temperature_c: 6 },
    decisions: [{
      id: 'cold',
      instruction: 'Is temperature_c below 10?',
      kind: {
        type: 'binary',
        false_label: 'Temperature is at least 10.',
        true_label: 'Temperature is below 10.',
      },
    }],
  });
  for (const result of response.results) {
    console.log(result.id, result.value, result.status);
    if (result.evidence.type === 'model_scored') {
      console.log(result.evidence.estimate.p_true, result.evidence.candidate_mass);
    }
  }
} finally {
  await engine.close();
}
```

`load()` はインストール済み OS/CPU ランタイムまたは `binaryPath` を使います。既定の
`transport: 'stdio'` はビルド済み Rust 実行ファイルと stdin/stdout JSON で通信し、
ポートを開きません。同一プロセスの N-API binding ではありません。明示的な
`transport: 'http'` は loopback サーバーを開きます。モデルプロセスを再利用し、
`close()` で終了を待ちます。単一呼び出しの timeout・キャンセルは native 推論停止を
保証しません。stdio は最大 16 未完了呼び出しで、timeout 後も native 応答までスロットを保持します。

ビルド ワークフローは、Linux x64/arm64 (glibc)、macOS x64/arm64、および Windows x64 をカバーします。 Linux および Windows パッケージは CPU を公開します。 macOS arm64 は、CPU および Metal を公開します。 CUDA およびその他のカスタム ビルドは `binaryPath` を使用します。 Linux パッケージにはシステム glibc/C++ ランタイムが必要です。 Windows パッケージには Microsoft Visual C++ x64 ランタイムが必要です。サポートされていないプラットフォームでは、明らかなエラーが発生します。これらはワークフローのターゲットです。 1 つのプラットフォームでのローカル検証では、他のプラットフォームのアーティファクトが CI に合格したことは確立されません。

明示的なリソース管理をサポートする TypeScript アプリケーションは、`await using engine = await L2S1.load(...)` を書き込むことができます。これはスコープの終了時に `close()` を呼び出します。パッケージはESMです。

`LoadOptions` は、CPU/CUDA/Metal、コンテキスト/バッチ/スレッド数、ビジョン プロジェクター (`mmproj`)、LoRA、実行モード、並列幅、プロンプト レイアウト/詳細、および起動ポリシーを公開します。あまり一般的ではない Rust フラグは `extraArgs` で渡すことができます。 `--stdio` / `--listen` は予約されています。 `startupTimeoutMs` のデフォルトは 120,000 および `timeoutMs` から 180,000 です。 `signal` を起動するとロードがキャンセルされます。 `onStderr` は、ネイティブ ログ チャンクを受信します。呼び出しでは、2 番目の引数として `{ signal, timeoutMs }` を受け入れます。失敗した推論リクエストが自動的に再試行されることはありません。

ビルド後に、Node.js 24 の TypeScript サポートを使用して、[warehouse サンプル ](../../../sdks/typescript/examples/warehouse.ts) を実行します。

```sh
L2S1_BINARY=/absolute/path/to/l2s1 node examples/warehouse.ts /path/to/chat-model.gguf
```

<a id="connect-to-a-server"></a>
## サーバーに接続する

アプリケーション コードは、バックエンドを切り替えながら 1 つの API を維持できます。

```ts
import { L2S1, type DecisionBackend } from '@l2s1/node';

const local = await L2S1.load({ model: '/path/to/model.gguf' });
const remote = L2S1.connect({ baseUrl: 'https://inference.example.com/l2s1' });
// Both expose decide(request), capabilities(), close() and async disposal.
// const response = await remote.decide(request);

function useCustomBackend(backend: DecisionBackend) {
  return L2S1.fromBackend(backend);
}
```

`DecisionBackend` には、エクスポートされた応答タイプを返す非同期 `decide(request, options)` メソッドと `capabilities(options)` メソッドが必要です。オプションの `close()` フックは、所有されているリソースを解放します。これにより、アプリケーションの 判断 コードを変更せずに、カスタム IPC、RPC、またはその他のランタイム アダプターが許可されます。 `fromBackend()` は、ライフサイクル所有権をファサードに転送します。 HTTP 接続を閉じると、このクライアントの要求がキャンセルされ、共有リモート サーバーは実行されたままになります。

<a id="repeat-fixed-decisions-with-new-state"></a>
## 入力だけを変えて固定した判断を繰り返す

```ts
const batchEngine = await L2S1.load({
  model: '/path/to/model.gguf', executionMode: 'parallel', parallelWidth: 4,
});
type Temperature = { temperature_c: number };
const plan = batchEngine.prepare<Temperature>([{
  id: 'cold', instruction: 'Is temperature_c below 10?',
  kind: { type: 'binary', false_label: 'At least 10.', true_label: 'Below 10.' },
}]);
const first = await plan.decide({ temperature_c: 6 });
const second = await plan.decide({ temperature_c: 15 });
const responses = await plan.decideBatch([{ temperature_c: 2 }, { temperature_c: 20 }]);
// Different definitions per item: await batchEngine.decideBatch(requests).
await batchEngine.close();
```

`prepare()` は固定定義と state 型を再利用します。token コンパイルや永続 KV 再利用では
ありません。`decideBatch()` は stdio または HTTP `/v1/decision-batches` で配列全体を
一度渡し、native parallel を実行します。`executionMode: 'parallel'` と `parallelWidth` を
指定します。state・ID・media・ポリシーは独立し結果は入力順です。timeout はバッチ全体に
適用します。直列 fallback・自動再試行はなく、`batch_unsupported` または
`batch_not_enabled` を返します。カスタム backend は任意の `decideBatch()` を実装します。

最大 128 要求・合計 128 判断で、direct reasoning と判断ごとに最大 26 選択肢です。
全て text、または各判断に画像が一つと対応する projector が必要です。text/image 混在は
拒否します。全 wire 入力を実行前に検証し、実行失敗はバッチ全体の失敗です。実行済み
wave を巻き戻し・再実行しません。`capabilities().batch` を確認してください。
`Promise.all(decide(...))` は自動バッチではありません。
[バッチ API の検討](../BATCHING_API_REVIEW.md)と [Python SDK](../python/README.md)を参照してください。



既存の Rust サーバーには `@l2s1/node/http` を使用します。このサブパスにはノードの組み込みインポートがなく、ブラウザー用にバンドルすることもできます。ブラウザ呼び出しには、CORS を提供する同一オリジン プロキシまたはリバース プロキシが必要です。 Rust サーバーは CORS ヘッダーを追加しません。これにより、ブラウザ内で ネイティブ 推論は実行されません。

```ts
import { L2S1Client, L2S1Error } from '@l2s1/node/http';

const client = new L2S1Client({
  baseUrl: 'http://127.0.0.1:8080',
  // headers: { Authorization: 'Bearer proxy-token' },
});
await client.health();
const capabilities = await client.capabilities();
// await client.decide(request, { signal: abortController.signal });
```

`L2S1Error` は、Rust 障害の `code`、`status`、`requestId`、および `userReason` を保持します。トランスポートのキャンセルとネットワーク エラーでは、元のフェッチ エラーが保持されます。クライアントは、バージョン管理された応答エンベロープ、判断 注文/ID/種類、結果ステータス、および証拠の種類をチェックします。モデル スコアとプロバイダーの選択には、個別の TypeScript 証拠タイプがあります。 `selection_only` には確率がありません。クライアントは、Rust wgpu および OpenRouter HTTP サーバーもサポートします。

<a id="policies-reasoning-and-images"></a>
## ポリシー、理由、イメージ

リクエスト `policy` は `{ min_top_probability, min_candidate_mass }` を使用します。両方を指定する必要があります。 `target_error_rate` は `min_top_probability = 1 - rate` にマップされます。正確性を保証するものではありません。 `failure_reasons` は、既知の 判断保留/障害コードのメッセージを提供します。 `reasoning: { mode: 'thinking', max_tokens: 128 }` には、バックエンド とモデル広告のサポートが必要です。サポートされていないリクエストは Rust で失敗します。オプション機能を選択する前に、`capabilities()` をお読みください。

同梱の v1 エンジンは要求ごとの policy、エラー予算、失敗メッセージを受け付けます。reasoning は direct のみを宣言し、thinking 要求は明示的に拒否します。古いサーバーは未対応フィールドに HTTP 400 を返します。`L2S1.load({ policy })` で起動時の既定 policy も設定できます。クライアントは要求された制御を黙って無視しません。

画像は、データ URL プレフィックスのない標準の Base64 を使用します。

```ts
const request = {
  state: { task: 'classify the image' },
  media: [{ type: 'image' as const, id: 'photo', data_base64: imageBytes.toString('base64') }],
  decisions: [{
    id: 'cat', instruction: 'Is there a cat in the image?', media_ids: ['photo'],
    kind: { type: 'binary' as const, false_label: 'No cat.', true_label: 'A cat is present.' },
  }],
};
// Load with { model: visionModel, mmproj: matchingProjector } before deciding.
```

省略 `media_ids` はすべての要求メディアを使用します。 `[]` はテキストのみを選択します。 Rust は、メディア、判断、本体およびモデルの制限を適用します。 [HTTP契約](../GUIDE.md#direct-image-input-and-http-api)を参照してください。

<a id="verification-and-portability"></a>
## 検証と移植性

```sh
npm run check
npm test
npm run test:rust
npm pack --dry-run
```

`test:rust` は、小さな判断論的な Rust バックエンド を構築し、マネージド ノード呼び出しを通じて実際の Rust スコアリング、検証、および HTTP エンベロープを実行します。 Rust ツールチェーンは必要ですが、モデルや C++ ツールチェーンは必要ありません。これは フィクスチャー の証拠であり、モデル品質の証拠ではありません。

オプションの実モデル スモーク (リポジトリ ルートで、最初に ネイティブ バイナリをビルドします):

```sh
cd sdks/typescript
L2S1_BINARY=/absolute/path/to/l2s1 L2S1_MODEL=/path/to/chat-model.gguf node --test test/model.integration.mjs
```

WASM は別個のランタイム ポートです。 WASM 用の Rust ラッパーをコンパイルしても、C++ 推論エンジンはパッケージ化されず、CUDA/Metal の実行は保持されません。 WASM ディストリビューションには、個別に構築された推論エンジン、その JS/WASM 境界、および CPU/WebGPU パスが必要です。このパッケージは、代わりに事前に構築された ネイティブ ランタイム パッケージを使用します。

<a id="build-distribution-artifacts"></a>
## 配布成果物のビルド

ワークフローは `v*` バージョンタグの push または手動実行で開始します。[ランタイム ワークフロー](../../../.github/workflows/typescript-runtimes.yml) は、5 つのプラットフォームすべてを構築し、ラッパー/ランタイム `.tgz` ファイルをワークフロー アーティファクトとしてアップロードします。パッケージは公開しません。ラッパーを公開する前に、一致するバージョンですべてのランタイム パッケージを公開します。このワークフローは、チェックサム、実行可能ファイルの起動、新しい npm プロジェクトへのインストール、および自動ランタイム解決を検証します。 CI での無効なモデルの起動はモデル推論の検証ではありません。

現在のプラットフォームをローカルに構築するには、リポジトリ ルートで実行します。

```sh
# On macOS arm64, use llama-metal and --devices cpu,metal to include Metal.
L2S1_PORTABLE_BUILD=1 cargo build --release --locked --features llama \
  --bin l2s1 --message-format=json-render-diagnostics > native-build.jsonl
node sdks/typescript/scripts/bundle-runtime.mjs --cargo-log native-build.jsonl
node sdks/typescript/scripts/verify-runtime.mjs sdks/typescript/runtime-packages/linux-x64
cd sdks/typescript/runtime-packages/linux-x64
npm pack --pack-destination ../../ --ignore-scripts
cd ../..
npm pack
npm run test:package -- l2s1-node-0.1.1.tgz l2s1-runtime-linux-x64-0.1.1.tgz
```

ランタイムを再構築するときは、新しい出力ディレクトリを使用します。 `L2S1_PORTABLE_BUILD=1` は、ビルドホスト CPU 命令と OpenMP 依存関係を無効にします。一般的な CPU カーネルは、ホストに最適化されたカスタム ビルドよりも遅い可能性があります。各アーティファクトには、ライセンス通知と SHA-256 マニフェストが含まれています。ベリファイアは、Linux llama.cpp/GGML の依存関係がバンドル ディレクトリから解決されることを確認します。

[配布パイプライン](../RELEASE_PIPELINE.md)で SDK と native runtime の自動公開・認証設定を確認してください。
