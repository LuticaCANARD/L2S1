# Licensing assessment

Reviewed on 2026-09-21 against this `Cargo.lock` and llama.cpp commit `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.

## Conclusion

The project's own source can be distributed under MIT, including commercially. The reviewed llama.cpp integration does not require changing the project's license. Preserve upstream notices when distributing copies or substantial portions of dependency code, including compiled code. This conclusion concerns the reviewed software composition; it does not relicense model weights or separately supplied runtimes.

The project previously declared `license = "MIT"` in Cargo.toml but had no license text. [LICENSE](LICENSE) now supplies that text. [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) preserves dependency license texts and copyright notices.

## Reviewed code

| Component | Terms found | Distribution handling |
| --- | --- | --- |
| llama.cpp / ggml | MIT | Preserve the ggml authors' notice and MIT text. |
| Jinja, JSON wrapper, Unicode helpers compiled by `build.rs` | llama.cpp MIT | These are compiled into this project's adapter, so notices matter even if libllama is supplied separately. |
| nlohmann/json 3.12.0 | MIT | Preserve its license and bundled component copyright notices. |
| llamafile CPU SGEMM, YaRN, ggllm.cpp tokenizer ancestry | MIT notices in the reviewed runtime sources/upstream projects | Relevant notices are included for runtime distributions. |
| 33 Cargo dependencies in the lockfile | MIT available for each; most offer an alternative Apache-2.0 license | The collected texts select the MIT option. Includes build-time and target-specific dependencies conservatively. |
| unicode-ident data | Additional Unicode-3.0 terms | Its Unicode copyright and permission text is included as well as MIT. |

No dependency in that reviewed Cargo graph requires licensing this project's own code under GPL or another copyleft license. This is not a license audit of every optional llama.cpp backend. The local runtime enables x86 CPU, llamafile, OpenMP, and CUDA; additional backends or dependency changes require a fresh notice review.

## Source and binary packages

Cargo explicitly excludes `models/`, `target/`, and `results/`. Relying only on `.gitignore` was insufficient in this workspace: the initial package inventory included the GGUF weights. Source packages retain the project license and third-party notices.

When distributing a binary, ship these notices alongside it. If bundling libllama/ggml, retain their applicable notices as well. The current executable dynamically uses system libraries and NVIDIA CUDA libraries; they are not included in the Cargo package and do not become MIT-licensed. Bundling NVIDIA components requires checking the installed toolkit's redistributable list and [NVIDIA CUDA license](https://docs.nvidia.com/cuda/eula/index.html). Likewise, retain the licenses applicable to any system/runtime libraries you choose to bundle. No assembled binary installer or container image was audited here.

## Gemma 3 model weights

The tested `ggml-org/gemma-3-1b-it-GGUF` repository identifies its license as `gemma`. The [Gemma Terms of Use](https://ai.google.dev/gemma/terms), including Section 3.1, apply to this checkpoint and its derivatives independently of the application's MIT license.

Redistribution requires providing the agreement, passing through its use restrictions, and marking modified files. Non-hosted distributions also need the specified Notice text:

> Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

Hosted/API distribution is also covered by the agreement. Keeping weights out of the source package does not remove obligations when operating a Gemma service. The model was downloaded locally for testing; no weights are included in the source package. This assessment is specific to Gemma 3 and does not assume all Gemma generations use identical terms.

## Gemma 4 model weights

Google's [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and [Gemma 4 license](https://ai.google.dev/gemma/apache_2) identify Apache-2.0, unlike the Gemma 3 terms above. The reviewed `ggml-org/gemma-4-E2B-it-GGUF` checkpoint also declares Apache-2.0.

The application can retain MIT while those weights retain Apache-2.0. If redistributing the weights or a derivative, supply the Apache license, preserve applicable attribution/NOTICE content, and identify modified files as required by Section 4. Apache's patent and trademark provisions still apply. Do not describe an application-plus-weights bundle as having only MIT terms. The model weights remain excluded from this source package.

## Sources

- [llama.cpp MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE); the actual reviewed text was read from the pinned local checkout.
- [nlohmann/json license](https://github.com/nlohmann/json/blob/v3.12.0/LICENSE.MIT); the checked-in llama.cpp license file and header notices were also inspected.
- Cargo metadata and each dependency's license files in the local registry for the exact locked versions.
- [YaRN license](https://github.com/jquesnelle/yarn/blob/master/LICENSE) and [ggllm.cpp license](https://github.com/cmp-nct/ggllm.cpp/blob/master/LICENSE).
- [Gemma checkpoint](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/tree/f9c28bcd85737ffc5aef028638d3341d49869c27) and the official terms linked above.

## Separate Svelte website

The MIT-licensed introduction site in `web/` is excluded from the Rust crate. Its installed npm dependency notices are recorded separately in `web/THIRD_PARTY_LICENSES.txt` and included in the static site's documentation assets. The site includes only an illustrative scoring demo, aggregate performance measurements, documentation, and JSON examples; it contains no model weights or inference service. Regenerate its dependency notices when changing the npm lockfile.
