# L2S1 SDKs

| SDK | Install command | Guide |
| --- | --- | --- |
| TypeScript / Node.js | `npm install @l2s1/node@0.1.1` | [typescript/README.md](typescript/README.md) |
| Python | `pip install l2s1-sdk==0.1.1` (`import l2s1`) | [python/README.md](python/README.md) |

For the Rust library, run `cargo add l2s1@0.1.1 --features llama` in your application.

Both SDKs call the same resident Rust engine over stdio or HTTP and expose native batching.
Rust library and Cargo package sources remain in the repository root and `crates/`.

Run build, test and release commands from the repository root unless a guide specifies an SDK directory.
See [the release pipeline](../docs/RELEASE_PIPELINE.md) for npm, PyPI, Cargo and GitHub Release publication.
