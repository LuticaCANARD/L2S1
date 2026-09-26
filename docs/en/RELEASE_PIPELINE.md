# SDK release pipeline

[English](RELEASE_PIPELINE.md) · [한국어](../ko/RELEASE_PIPELINE.md) · [日本語](../ja/RELEASE_PIPELINE.md)

[release.yml](../../.github/workflows/release.yml) runs on a stable `vMAJOR.MINOR.PATCH`
tag push or manual dispatch naming an existing tag. All jobs check out the same
resolved commit. Tag and Cargo/Python/npm/runtime versions must match; prereleases
are rejected. Commit this workflow before pushing a new version tag.

The pipeline builds Linux x64/arm64, macOS x64/arm64 and Windows x64 runtime packs,
runs Rust/TS checks and installation tests, tests Python 3.11/3.14 on all three
OSes, then packages and verifies native Cargo crates. It validates ten archives,
including runtime file hashes, and generates `SHA256SUMS` and `release.json`.
Only then does registry publication start: npm publishes five runtimes before
`@l2s1/node`; PyPI publishes the `l2s1-sdk` wheel/sdist; crates.io publishes
`l2s1-llama-sys` then waits for the index before `l2s1`. After all three succeed,
GitHub receives a draft release, all assets are attached, and the release is
published. An already published release is never overwritten.

## Authentication setup

Create/use GitHub environments `npm`, `pypi`, `crates-io`. Configure trusted
publishers for owner `LuticaCANARD`, repository `L2S1`, workflow `release.yml`,
and the corresponding environment in each registry. Registry account ownership
is independent from GitHub. The workflow does not create these external accounts.

- npm: configure **all six packages**. Permit direct `npm publish` in allowed
  actions. The workflow installs npm 11. New packages can bootstrap using the
  environment secret `NPM_TOKEN`; configure OIDC afterwards and remove the token.
- PyPI: configure `l2s1-sdk`, using a pending publisher for a new project. No token is required.
- crates.io: configure both crates. New crates can bootstrap using environment
  secret `CARGO_REGISTRY_TOKEN`. Remove it after configuring OIDC; the action
  issues a temporary token. Internal `l2s1-tools` retains `publish = false`.
- GitHub Release: the built-in token has contents write only in the final job.

## Cargo scope and verification

The checkout's WGPU backend uses the unpublished git crate `rullama-engine`.
`prepare_cargo_release.py` creates a separate native-only staging tree without
changing the checkout. Published features are `llama`, `llama-cuda`, `llama-metal`,
`openrouter`, including stdio and native batching. WGPU remains checkout-only.
Both `.crate` archives and the staged source come from one build. Publication
repackages that source and compares SHA-256 to the verified archives first.
Weights and build outputs are excluded.

## Run and recover

Align versions in root/sys Cargo manifests, Python pyproject/exported/runtime
versions, npm package/lock and all runtime optional dependencies.

```sh
python scripts/prepare_release.py --tag v0.1.1
python -m unittest discover -s scripts -p test_release_pipeline.py -v
git tag v0.1.1
git push origin v0.1.1
```

Registries are not one transaction. If one publishes and another fails, rerun
failed jobs with the same verified files; nothing is automatically deleted or
rolled back. Existing versions are skipped only if npm integrity, PyPI file hashes
or crates.io archive hash match exactly. Auth, network and server errors do not
mean “package absent”. Conflicting bytes require a new version. Re-running failed
jobs preserves successful build artifacts.

Local checks do not establish hosted macOS/Windows/arm64 CI or registry auth and
publication success. CI uses deterministic Rust fixtures and invalid-model native
startup, not downloaded model weights. Real local GGUF CPU smoke is separate.

[Draft-first immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases),
[npm OIDC](https://docs.npmjs.com/trusted-publishers/),
[PyPI OIDC](https://docs.pypi.org/trusted-publishers/using-a-publisher/),
[crates.io authentication](https://github.com/rust-lang/crates-io-auth-action),
[Cargo dependency rules](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html).

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)
