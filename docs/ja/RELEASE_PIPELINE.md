# SDK 配布パイプライン

[English](../en/RELEASE_PIPELINE.md) · [한국어](../ko/RELEASE_PIPELINE.md) · [日本語](RELEASE_PIPELINE.md)

[release.yml](../../.github/workflows/release.yml) は安定版 `vMAJOR.MINOR.PATCH` の tag
push、または既存 tag を指定する手動実行で起動します。全 job は同じ commit を
checkout し、Cargo・Python・npm・runtime の version 一致を検証します。prerelease
は拒否します。この workflow を commit してから新しい tag を push します。

Linux x64/arm64、macOS x64/arm64、Windows x64 の runtime をビルドし Rust/TS の
検査・インストール試験を実行します。Python 3.11/3.14 を 3 OS で検証し native
Cargo crate を package・検証します。10 archive と runtime 内部 checksum を確認し、
`SHA256SUMS` と `release.json` を生成してから公開を始めます。npm は 5 runtime →
`@l2s1/node`、PyPI は `l2s1-sdk` wheel/sdist、crates.io は `l2s1-llama-sys` → index 待機 →
`l2s1` の順です。全 registry 成功後に GitHub draft release へ全 asset を添付して
公開します。公開済み release は上書きしません。

## 初期設定

GitHub environments `npm`・`pypi`・`crates-io` を使います。registry で所有者
`LuticaCANARD`、repository `L2S1`、workflow `release.yml`、対応 environment を
trusted publisher に登録します。registry アカウント・公開権限は別途必要です。

- npm: **6 package 全て**を登録し直接 `npm publish` を許可します。workflow は npm 11 を
  インストールします。新規 package は environment secret `NPM_TOKEN` で初回公開し、
  OIDC 登録後に token を削除できます。
- PyPI: `l2s1-sdk` に登録します。新規は pending publisher を使え、token は不要です。
- crates.io: 2 crate を登録します。初回は environment secret `CARGO_REGISTRY_TOKEN` を
  使えます。OIDC 登録後に削除すると一時 token を使います。内部 `l2s1-tools` は公開しません。
- GitHub Release: 最後の job のみ built-in token に contents write を与えます。

## Cargo の配布範囲

checkout の WGPU は未公開の git crate `rullama-engine` を使っています。
`prepare_cargo_release.py` は checkout を変更せず別 staging に native 配布を作ります。
公開 feature は `llama`・`llama-cuda`・`llama-metal`・`openrouter` で stdio/native batch を
含みます。WGPU は checkout 専用です。同一 build の `.crate` と staging source を使い、
公開前の再 package と検証済み archive の SHA-256 が一致する必要があります。
モデルと build 出力は含みません。

## 起動と失敗時

Cargo root/sys manifests、Python pyproject/export/runtime version、npm package/lock と
runtime optional dependencies を同一 version にします。

```sh
python scripts/prepare_release.py --tag v0.1.1
python -m unittest discover -s scripts -p test_release_pipeline.py -v
git tag v0.1.1
git push origin v0.1.1
```

registry 間は単一 transaction ではありません。一部成功後に他が失敗したら同じ検証済み
asset で失敗 job を再実行します。自動削除・rollback はしません。既存 version は npm
integrity・PyPI SHA-256・crates.io archive SHA-256 の完全一致時のみ省略します。
認証・通信・サーバー障害を「package なし」と扱いません。内容違いは新 version が必要です。
失敗 job の再実行は成功した build artifact を保持します。

ローカル検証は hosted macOS/Windows/arm64 CI・registry 認証・実公開の成功証拠では
ありません。CI は Rust fixture と invalid-model 起動を使い、モデルを取得しません。
実 GGUF CPU smoke は別検証です。

[npm OIDC](https://docs.npmjs.com/trusted-publishers/) ·
[PyPI OIDC](https://docs.pypi.org/trusted-publishers/using-a-publisher/) ·
[crates.io auth](https://github.com/rust-lang/crates-io-auth-action) ·
[GitHub draft release](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)
