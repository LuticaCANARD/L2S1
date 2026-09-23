# License

L2S1 source code is licensed under the [MIT License](LICENSE), as declared in `Cargo.toml`.

[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) contains the dependency license texts and attribution notices. Preserve the applicable notices when distributing those components.

The `l2s1-llama-sys` source package includes a pinned llama.cpp source snapshot, its upstream license, and its own [third-party notices](crates/l2s1-llama-sys/THIRD_PARTY_LICENSES.txt). Preserve those notices when distributing the native dependency.

## Model weights

This repository and its Cargo package do not include model weights. Users supply their own local model files. `models/`, `*.gguf` and `*.safetensors` are excluded by `.gitignore` and the Cargo package configuration.

The project's MIT license applies to its source code; separately obtained models retain their own licenses.
