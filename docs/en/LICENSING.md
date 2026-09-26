<a id="license"></a>
# License

[English](LICENSING.md) · [한국어](../ko/LICENSING.md) · [日本語](../ja/LICENSING.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)


L2S1 source code is licensed under the [MIT License](../../LICENSE), as declared in `Cargo.toml`.

[THIRD_PARTY_LICENSES.txt](../../THIRD_PARTY_LICENSES.txt) contains the dependency license texts and attribution notices. Preserve the applicable notices when distributing those components.

The `l2s1-llama-sys` source package fetches a pinned llama.cpp archive during its build. The archive contains the upstream license; the crate also includes [third-party notices](../../crates/l2s1-llama-sys/THIRD_PARTY_LICENSES.txt). Preserve those notices when distributing the native dependency.

The repository-only `l2s1-tools` package follows pinned public benchmark scoring and includes a small frozen Laya probe rendering. Its [notices](crates/l2s1-tools/NOTICE.md) include the JevBench MIT attribution and source references. Full datasets and checkpoints are not bundled.

<a id="model-weights"></a>
## Model weights

This repository and its Cargo package do not include model weights. Users supply their own local model files. `models/`, `*.gguf` and `*.safetensors` are excluded by `.gitignore` and the Cargo package configuration.

The project's MIT license applies to its source code; separately obtained models retain their own licenses.
