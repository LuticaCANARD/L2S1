<a id="l2s1--llm-to-system-1-website"></a>
# L2S1 — LLM to System 1 ウェブサイト

[English](../../en/web/README.md) · [한국어](../../ko/web/README.md) · [日本語](README.md)

[English index](../../en/README.md) · [한국어 색인](../../ko/README.md) · [日本語索引](../README.md)

英語の SvelteKit 紹介ページは `adapter-static` でプリレンダリングします。契約の例には合成確率と明記し、性能表は記録済みモデルテストの要約です。`/webgpu` は英語・韓国語・日本語で Qwen3 0.6B の実ブラウザ推論と、大きな GGUF ローカルサーバーの明示的な選択を提供します。ブラウザモードの入力は worker 内に残り、ローカルモードの入力は分析時だけ送信します。モデルは要求時にダウンロードし、サイトに同梱しません。[実行ガイド](../WEBGPU_DEMO.md)を参照してください。

```sh
npm ci
npm run check
npm run lint
npm run build
npm run dev
```

生成された `build/` ディレクトリを静的ホストにデプロイします。このセットアップでは展開は実行されていません。

ブラウザのチェックの場合:

```sh
npx playwright install chromium
npm run build
npm test
```

`PLAYWRIGHT_CHROMIUM_EXECUTABLE` は、既にインストールされている互換性のある Chromium を選択できます。ブラウザー テストでは、JavaScript を使用しないプリレンダリング、キーボード制御の 判断保留、すべての 判断 タイプ、クリップボードのフィードバック、ドキュメントのダウンロード、デスクトップ/モバイルのオーバーフローがカバーされます。スクリーンショットは `test-results/` に保存されます。

`scripts/sync-content.mjs` は、ルートの英語・韓国語・日本語 README、ライセンス、`docs/` の最上位 Markdown、全 `docs/en/`、`docs/ko/`、`docs/ja/` 言語ディレクトリ、公開のパッケージ・サイト README、ベンチマークの README・レポート、倉庫 JSON 例を `static/docs/` にコピーします。リポジトリ相対の文書リンクはパスを保持します。各言語ディレクトリに索引と完全な公開文書本文があり、同じ文書の言語切り替えと原文見出しのアンカーを保持します。モデルディレクトリや元のモデル出力はコピーしません。明示した 5 ファイルの許可リストで、typed-decisions 要約、4,000 採点記録、移植可能な manifest を `/benchmarks/typed-decisions-20260926/` に公開します。要約・採点 JSONL は保存済みのバイトを維持し、manifest は両モデルの実行記録と絶対パス正規化の説明を含みます。記録に元の状態・生成思考テキストはありません。`src/lib/benchmarks.json` は実測の一部をパスなしで要約し、新しく完了した性能実行からのみ更新します。

サイトのソースはリポジトリの MIT ライセンスに従います。フレームワークと依存関係の通知は、`THIRD_PARTY_LICENSES.txt` に個別に記録され、静的ビルドにコピーされます。ビルド ツールとテストは、同梱されるクライアント ランタイムの一部ではありません。
