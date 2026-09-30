# L2S1 SDKs

| SDK | Install command | Guide |
| --- | --- | --- |
| TypeScript / Node.js | `npm install @l2s1/node@0.2.0` | [typescript/README.md](typescript/README.md) |
| C++17 | CMake target `l2s1::cpp` (source package) | [cpp/README.md](cpp/README.md) |
| Python | `pip install l2s1-sdk==0.2.0` (`import l2s1`) | [python/README.md](python/README.md) |

For the Rust library, run `cargo add l2s1@0.2.0 --features llama` in your application.

All SDKs use the same resident Rust engine and expose native batching. C++ uses stdio; Python and TypeScript also support HTTP.
Python, TypeScript and C++ `load()` inherit automatic fixed-schema prefix reuse
from a compatible runtime when the execution mode is omitted. Schema changes
clear retained prefixes; state changes keep reuse. See the
[runtime/version requirements and opt-out](../docs/DECISION_PERFORMANCE.md#keep-native-prefixes-between-calls).
Rust library and Cargo package sources remain in the repository root and `crates/`.

Run build, test and release commands from the repository root unless a guide specifies an SDK directory.
See [the release pipeline](../docs/RELEASE_PIPELINE.md) for npm, PyPI, Cargo and GitHub Release publication.
