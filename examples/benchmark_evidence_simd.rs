//! Isolate full-vocabulary CPU normalization; no model inference speed claim.
#[path = "../src/evidence/simd.rs"]
mod simd;
use std::{hint::black_box, time::Instant};

fn legacy_max(values: &[f32]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return None;
    }
    Some(
        values
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max),
    )
}
fn vector_max(values: &[f32]) -> Option<f64> {
    simd::maximum(values).map(f64::from)
}
fn measure(values: &[f32], maximum: fn(&[f32]) -> Option<f64>, normalize: bool) -> f64 {
    let start = Instant::now();
    for _ in 0..100 {
        let input = black_box(values);
        let max = maximum(input).unwrap();
        black_box(if normalize {
            max + input
                .iter()
                .map(|&v| (v as f64 - max).exp())
                .sum::<f64>()
                .ln()
        } else {
            max
        });
    }
    start.elapsed().as_secs_f64() * 1e6 / 100.0
}
fn main() {
    let mut records = Vec::new();
    for n in [32000, 128256, 262144] {
        let mut seed = 123_u32;
        let values: Vec<_> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as f64 / u32::MAX as f64 * 160.0 - 80.0) as f32
            })
            .collect();
        assert_eq!(legacy_max(&values), vector_max(&values));
        for normalize in [false, true] {
            let mut scalar = Vec::new();
            let mut vector = Vec::new();
            for round in 0..12 {
                // Alternate ordering; discard the first pair as warmup.
                let (a, b) = if round % 2 == 0 {
                    (
                        measure(&values, legacy_max, normalize),
                        measure(&values, vector_max, normalize),
                    )
                } else {
                    let b = measure(&values, vector_max, normalize);
                    (measure(&values, legacy_max, normalize), b)
                };
                if round > 0 {
                    scalar.push(a);
                    vector.push(b);
                }
            }
            let median = |v: &[f64]| {
                let mut v = v.to_vec();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            records.push(serde_json::json!({"vocabulary":n,"scope":if normalize{"full_log_normalizer"}else{"validation_and_maximum"},
                "legacy_median_us":median(&scalar),"simd_median_us":median(&vector),"speedup":median(&scalar)/median(&vector),
                "legacy_samples_us":scalar,"simd_samples_us":vector}));
        }
    }
    let isa = if cfg!(target_arch = "x86_64") {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                "avx2"
            } else {
                "scalar"
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            "scalar"
        }
    } else if cfg!(target_arch = "aarch64") {
        "neon-or-scalar"
    } else {
        "scalar"
    };
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"isa":isa,"iterations_per_sample":100,"paired_samples":11,
        "scope":"CPU postprocessing only; model inference, HTTP and candidate mapping excluded. Synthetic logits; no accuracy claim.","records":records})).unwrap());
}
