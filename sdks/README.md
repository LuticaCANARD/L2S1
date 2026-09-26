# L2S1 SDKs

| SDK | Package | Guide |
| --- | --- | --- |
| TypeScript / Node.js | `@l2s1/node` | [typescript/README.md](typescript/README.md) |
| Python | `l2s1` | [python/README.md](python/README.md) |

Both SDKs call the same resident Rust engine over stdio or HTTP and expose native batching.
Rust library and Cargo package sources remain in the repository root and `crates/`.

Run build, test and release commands from the repository root unless a guide specifies an SDK directory.
See [the release pipeline](../docs/RELEASE_PIPELINE.md) for npm, PyPI, Cargo and GitHub Release publication.
