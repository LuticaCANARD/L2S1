<a id="license"></a>
# ライセンス

[English](../en/LICENSING.md) · [한국어](../ko/LICENSING.md) · [日本語](LICENSING.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

L2S1 のソースコードは、`Cargo.toml` に記載された [MIT ライセンス](../../LICENSE)に従います。

[THIRD_PARTY_LICENSES.txt](../../THIRD_PARTY_LICENSES.txt) には、依存関係のライセンス全文と著作者表記が含まれます。該当するコンポーネントを配布する際は、適用される表記を保持してください。

`l2s1-llama-sys` のソースパッケージは、ビルド中に固定された llama.cpp アーカイブを取得します。アーカイブには上流のライセンスが含まれ、crate にも[第三者の表記](../../crates/l2s1-llama-sys/THIRD_PARTY_LICENSES.txt)が含まれます。ネイティブの依存関係を配布する際は、これらを保持してください。

リポジトリ専用の `l2s1-tools` パッケージは、固定された公開ベンチマークの採点方法に従い、小規模な固定 Laya プローブのレンダリングを含みます。[表記](crates/l2s1-tools/NOTICE.md)には JevBench の MIT 著作者表記と参照元が含まれます。全データセットやチェックポイントは同梱しません。

<a id="model-weights"></a>
## モデルの重み

このリポジトリと Cargo パッケージは、モデルの重みを含みません。ユーザーがローカルのモデルファイルを用意します。`models/`、`*.gguf`、`*.safetensors` は `.gitignore` と Cargo パッケージ設定で除外されます。

プロジェクトの MIT ライセンスはソースコードに適用されます。別途取得したモデルは、それぞれのライセンスに従います。
