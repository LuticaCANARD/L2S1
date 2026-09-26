<a id="npm-publishing-review"></a>
# npm publishing review

[English](PUBLISHING.md) · [한국어](../../ko/typescript/PUBLISHING.md) · [日本語](../../ja/typescript/PUBLISHING.md)

[English index](../README.md) · [한국어 색인](../../ko/README.md) · [日本語索引](../../ja/README.md)





The wrapper and platform runtimes can be distributed as public npm packages. The implementation uses ordinary npm tarballs, ESM exports, TypeScript declarations and optional platform dependencies. End users do not need a Rust/C++ compiler. Publication is still conditional on passing the native platform builds and obtaining npm scope/publishing access.

<a id="package-layout"></a>
## Package layout

| Package | Contents | Runtime |
| --- | --- | --- |
| `@l2s1/node` | JavaScript, declarations, source, README, MIT license | Node.js 22+; HTTP subpath can be bundled for browsers |
| `@l2s1/runtime-linux-x64` | Rust executable and matching shared libraries | CPU, glibc Linux x64 |
| `@l2s1/runtime-linux-arm64` | Rust executable and matching shared libraries | CPU, glibc Linux arm64 |
| `@l2s1/runtime-darwin-x64` | Rust executable and matching shared libraries | CPU, macOS x64 |
| `@l2s1/runtime-darwin-arm64` | Rust executable and matching shared libraries | CPU/Metal, macOS arm64 |
| `@l2s1/runtime-win32-x64` | Rust executable and matching DLLs | CPU, Windows x64 |

The wrapper references exact matching runtime versions. `os`, `cpu` and Linux `libc` metadata prevent incompatible optional packages being installed. Runtime resolution validates package identity, version and requested device. A missing runtime produces an explicit error; HTTP and custom backends work without installing a runtime. CUDA is available through a separately built executable supplied with `binaryPath`.

Both wrapper and generated runtimes use `publishConfig.access = public` and the npm registry URL. npm documents that scoped packages default to private visibility, so public publication must explicitly use public access. [npm scoped package documentation](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/)

<a id="release-conditions"></a>
## Release conditions

1. Confirm ownership or publishing rights for the npm `@l2s1` user/organization scope. A GitHub repository owner is not automatically the owner of an npm scope. An unauthenticated registry 404 does not prove the scope is available.
2. Start release checks by pushing the release commit with a `v*` version tag or manually dispatching the workflows for that commit. `release.yml` invokes reusable runtime and Python checks for that exact tag commit. PRs and branch pushes do not publish packages. Pass the five native build/install jobs in `typescript-runtimes.yml`. A local Linux build is not verification of macOS, Windows or arm64 artifacts.
3. Check all tarballs using `npm publish --dry-run --access public --ignore-scripts`. Verify the version, files, licenses, absence of model weights and absence of install scripts.
4. Publish the five runtime tarballs first at the matching version, then publish the wrapper. Publishing the wrapper before its runtimes would allow npm's optional dependency handling to leave local inference unavailable.
5. Install the wrapper by registry name into a clean project and verify automatic runtime selection and a real-model decision. This final registry test requires actual publication; local tarball installation is preparation evidence.

Public scoped publishing needs an npm account, scope permissions and supported publication authentication. As checked on 2026-09-26, the local npm CLI is not authenticated (`npm whoami` returns `ENEEDAUTH`), and an unauthenticated `npm view @l2s1/node` returns 404. No package was published during this review.

For CI publication, npm supports GitHub Actions trusted publishing through OIDC. It requires npm 11.5.1+, Node.js 22.14.0+, the corresponding package trusted-publisher settings and workflow `id-token: write`. The `release.yml` pipeline publishes verified npm, PyPI and Cargo packages, then GitHub Release. Registry accounts and trusted publishers require separate setup. See [the release pipeline](../RELEASE_PIPELINE.md). [npm trusted publishing documentation](https://docs.npmjs.com/trusted-publishers/)

<a id="binary-and-license-boundaries"></a>
## Binary and license boundaries

Native packages carry the project MIT license, third-party notices and a SHA-256 file manifest. The bundled engine, headers, bridge and shared libraries are built from the same pinned llama.cpp source. Model weights are not included and must be supplied under their own terms. The HTTP-only wrapper uses Node/browser platform APIs and introduces no third-party JavaScript runtime library.

Linux build jobs use Ubuntu 22.04; retain the corresponding glibc/C++ runtime requirements and test the oldest advertised distribution before release. Alpine/musl is outside the bundled targets. Windows requires the Microsoft Visual C++ x64 runtime. macOS deployment/Metal support must be established by its native jobs. General CPU kernels avoid build-host ISA assumptions; performance may differ from a host-optimized build.

The base main-branch HTTP contract supports state, decisions and media. Optional request reasoning/policy extensions are forwarded only when the application chooses them and the server supports them; an older server rejects unsupported fields. Startup policy remains available through `L2S1.load({ policy })`.

<a id="verification-scope"></a>
## Verification scope

Release CI checks TypeScript and example types, HTTP transport, custom backend routing, startup failure/cancellation/cleanup, real Rust HTTP validation/scoring and offline installation of packed wrapper/runtime tarballs. Native package verification checks file hashes, executable version and Linux shared-library resolution from the bundle. The platform CI tests startup with an invalid model and has no model download; that confirms executable loading rather than inference quality.

Real GGUF smoke tests require separately supplied weights. They verify typed results and score structure, including abstentions, and do not establish task accuracy or GPU behavior.
