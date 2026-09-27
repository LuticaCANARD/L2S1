# Exact SIMD evidence scan, 2026-09-27

The native CPU model build already uses llama.cpp's `-march=native` kernels on this host. This change targets the remaining Rust **full-vocabulary evidence postprocessing**: fuse logit validation and maximum reduction, dispatch to x86-64 AVX2 or AArch64 NEON, and retain a scalar fallback for unsupported CPUs and inputs shorter than 64 floats. It needs no global `target-cpu=native` flag or additional runtime library. Compact native evidence and ONNX encoder kernels are unchanged.

The maximum is selected from the original f32 values. Exponentials, f64 conversion and sequential summation retain the original arithmetic and order; no approximate SIMD exponential, fast-math or reassociation is introduced. NaN/+infinity are rejected and negative infinity remains valid for excluded vocabulary entries. SIMD source changes participate in the native serving fingerprint.

Measured on Intel i9-9900K under WSL2 (4 visible vCPUs), AVX2, release build. Each sample averages 100 evaluations; one warmup pair and 11 measured pairs alternate legacy/SIMD order. All logits are deterministic synthetic values. Full samples and source digests are in [summary.json](summary.json).

| Vocabulary size | Validation + maximum speedup | Full normalizer, legacy / SIMD | Normalizer speedup |
| --- | ---: | ---: | ---: |
| 32,000 | 10.23x | 204.24 / 163.61 µs | 1.25x |
| 128,256 | 9.96x | 761.07 / 605.27 µs | 1.26x |
| 262,144 | 10.26x | 1,627.45 / 1,389.90 µs | 1.17x |

These are postprocessing measurements, **not end-to-end model inference gains**. The remaining exponential/reduction work dominates this stage. There is no NEON latency claim. CI runs the same numerical tests on Linux, macOS and Windows; CPU dispatch determines the exercised ISA.

Tests cover unaligned slices, lengths/tails, large vocabularies, extreme values, signed zero, subnormals, every invalid-value lane and bit-identical f64 normalizers. A paired SmolLM2 Q8_0 CPU run checks all 36 decision-rules judgments against the pre-SIMD binary; its report is `native-parity.json`.

```sh
cargo test --locked --lib evidence::
cargo run --release --locked --example benchmark_evidence_simd > simd.json
```
