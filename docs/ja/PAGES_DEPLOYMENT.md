<a id="cloudflare-pages-deployment"></a>
# Cloudflareページの導入

[English](../en/PAGES_DEPLOYMENT.md) · [한국어](../ko/PAGES_DEPLOYMENT.md) · [日本語](PAGES_DEPLOYMENT.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

SvelteKit Web アプリは `adapter-static` を使用し、`web/build` を Cloudflare Pages プロジェクト `l2s1` に公開します。要求された 本番環境 ホスト名は `https://l2s1.luticalab.net` です。

公開デモには、記録された画像とテキストの判断が含まれます。これらは、ネイティブ L2S1 HTTP サーバーから保存された結果であり、ページ内のモデル推論ではありません。ライブ推論には、同じオリジン プロキシの背後で個別に実行される ネイティブ L2S1 サーバーが必要です: ビジョン用の `/inference` とテキスト用の `/text-inference`。 Vite 開発サーバーは、これらのプロキシをローカルホスト ポート 8080 および
8081. Static Pages は Vite プロキシを提供しません。ライブ モードが機能するには、本番環境 ページ関数または同等のプロキシが、到達可能な HTTPS ネイティブ バックエンド に接続されている必要があります。現在の UI は、任意のリモート API URL を受け入れません。 Pages アセットのデプロイメントを GGUF ランタイムのデプロイメントとして説明しないでください。

`/webgpu` はユーザーのブラウザーでテキストモデルを実行します。ネイティブ推論プロキシは不要で、明示的に固定 ONNX モデルとランタイムをダウンロードすると Web Worker が WebGPU 推論を実行します。Pages に画像推論を追加したり、ネイティブ GGUF サーバーを配備したりはしません。[WebGPU 実行](WEBGPU_DEMO.md)を参照してください。

<a id="account-and-authentication"></a>
## アカウントと認証

2026-09-26 で、Cloudflare の API は、`luticalab.net` がアカウント `cb2875b9abef08939d6a42e711a3600d` のアクティブなゾーンであることを確認しました。ゾーン ID は `f3f8abbf6f629072dc472e9afef8309a` です。これらの ID は公開構成識別子であり、資格情報ではありません。

API トークンを **Account → Cloudflare Pages → Edit** でそのアカウントにスコープ設定し、**Zone → DNS → Edit** で `luticalab.net` にスコープ設定して使用します。ゾーン読み取りアクセスは、変更前のゾーンを確認するのにも役立ちます。資格情報を環境または Wrangler 資格情報ストアに保持します。決してこのファイルまたは `wrangler.jsonc` に追加しないでください。

2026-09-26 で、ユーザーはページの読み取り/書き込み権限と DNS 読み取り/編集権限を使用して正式な `cf` OAuth ログインを完了しました。ログイン後、ページと要求された DNS レコードの両方が正常にクエリされました。以前の Wrangler ログインにはページ/DNS 権限がありませんでした。成功した `whoami` は、展開へのアクセスを証明しませんでした。公式 `cf` CLI は、次のスコープでの共有ページ/DNS OAuth ログインをサポートしています。

```bash
cf auth login --force --no-browser --scopes account:read user:read zone:read pages:read pages:write dns_records:read dns_records:edit
cf auth whoami
CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d cf pages projects list
cf --zone f3f8abbf6f629072dc472e9afef8309a dns records list --name l2s1.luticalab.net
```

出力された OAuth URL を、ポート 8877 でこのマシンのローカルホスト コールバックに到達できるブラウザで開きます。 `cf` と Wrangler CLI 資格情報ストアは別個です。一方にログインしても、もう一方の OAuth ログインは自動的に置き換えられません。

<a id="build-and-upload"></a>
## ビルドしてアップロードする

Wrangler 4.141.0 またはレビュー済みの新しいバージョンを使用してください。リポジトリのルートから:

```bash
npm --prefix web ci
npm --prefix web run check
npm --prefix web run lint
npm --prefix web run build
npm --prefix web test
cd web
export CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d
npx wrangler@4.141.0 whoami
npx wrangler@4.141.0 pages project list --json
```

`l2s1` プロジェクトはすでに存在します。同等の新しい Pages プロジェクトが存在しない場合に再作成するには、それを一度作成します。

```bash
npx wrangler@4.141.0 pages project create l2s1 --production-branch main --force
```

Wrangler 4.141.0 は、デフォルトで新しいプロジェクトの作成をワーカーに委任します。作成専用の `--force` オプションは、明示的に要求された Pages の展開を保持します。既存の Pages プロジェクトの後続のコマンドで `--force` を渡さないでください。

検証済みのビルドを 本番環境 ブランチにアップロードします。

```bash
npx wrangler@4.141.0 pages deploy ./build --project-name l2s1 --branch main --commit-dirty=true
npx wrangler@4.141.0 pages deployment list --project-name l2s1
```

`--commit-dirty=true` は、ローカルのコミットされていないビルドがアップロードされたことを記録します。変更がコミットされた後は、実際の G​​it の状態に応じて省略するか、設定します。 `l2s1.pages.dev` が使用できない場合、プロジェクトは接尾辞付きの `pages.dev` ホスト名を受け取ることがあります。 Cloudflareから返されたホスト名を想定するのではなく、それを使用します。

このワークフローではダイレクト アップロードを使用します。 Cloudflareは、後で直接アップロードプロジェクトをGit統合に変換することをサポートしていません。自動アップロードは、CI で Wrangler を実行することで実装できます。

<a id="associate-the-production-hostname"></a>
## 本番環境 ホスト名を関連付ける

2026-09-26 に本番サイトをプロジェクト `l2s1` と URL `https://l2s1.luticalab.net` に修正しました。プロジェクト ID は `d15c7797-4208-4388-9a7c-44a3df22e496` です。プロキシ CNAME は `l2s1.pages.dev` を指し、DNS レコード ID は `58376692674167fd35459ab8d5519f5a` です。

旧 `n2s1` プロジェクトは既存のリンクを正しい URL にリダイレクトするためだけに維持します。このプロジェクトには `web/legacy/` を個別にアップロードし、本サイトのビルドはアップロードしません。パスとクエリ文字列を保持します。

```bash
npx wrangler@4.141.0 pages deploy ./legacy --project-name n2s1 --branch main
```

まず、`l2s1.luticalab.net` を Pages プロジェクトの **Custom ドメイン** に追加します。ページの関連付けが存在した後でのみ、実際の 本番環境 `pages.dev` ホスト名を指す `l2s1.luticalab.net` の DNS CNAME レコードを作成または確認します。 Cloudflareですでにホストされているゾーンの場合、ダッシュボードはカスタムドメインフローの一部としてこのCNAMEを作成できます。まず既存のレコードを確認し、別のサービスを指すレコードを確認せずに上書きしないようにしてください。

自動ドメイン関連付けの場合、Cloudflare の API は、`POST /accounts/{account_id}/pages/projects/{project_name}/domains` と `{"name":"l2s1.luticalab.net"}` を受け入れます。 DNS レコードは、ゾーン DNS API を通じて個別に管理されます。現在、Wrangler には Pages のカスタム ドメイン サブコマンドがありません。

`cf` CLI を認証すると、そのドメイン コマンドも使用できるようになります。

```bash
CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d cf pages projects domains create l2s1 --name l2s1.luticalab.net
```

Pages のカスタム ドメインの関連付けを行わずに DNS CNAME のみを追加するだけでは不十分で、522 エラーが発生する可能性があります。

<a id="verify-after-upload"></a>
## アップロード後に確認する

1. ページ展開リストで 本番環境 展開が成功したことを確認します。
2. Pages のカスタム ドメインとその証明書がアクティブであることを確認します。
3. `l2s1.luticalab.net` を解決し、その HTTPS URL をリクエストします。
4. ブラウザで 本番環境 URL を開き、記録されたイメージ デモを実行し、判断 モードに切り替えて、ベンチマーク データとダウンロード可能なアーティファクトを確認します。
5. パブリック API が接続され、デプロイされたブラウザーからのイメージ要求が成功した後にのみ、ライブ ネイティブ 推論を個別に検証されたものとして扱います。

<a id="official-references"></a>
## 公式リファレンス

- [Pages Wrangler 構成](https://developers.cloudflare.com/pages/functions/wrangler-configuration/)
- [直接アップロード](https://developers.cloudflare.com/pages/get-started/direct-upload/)
- [カスタム ドメイン](https://developers.cloudflare.com/pages/configuration/custom-domains/)
- [Pages ドメイン API](https://developers.cloudflare.com/api/resources/pages/subresources/projects/subresources/domains/methods/create/) を追加
