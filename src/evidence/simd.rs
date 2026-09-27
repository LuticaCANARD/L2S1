//! Exact logit validation and maximum scan; probability arithmetic stays scalar f64.

pub(crate) fn maximum(values: &[f32]) -> Option<f32> {
    if values.len() >= 64 {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: dispatch checks the ISA; loads only address complete slice chunks.
            return unsafe { avx2(values) };
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: dispatch checks the ISA; loads only address complete slice chunks.
            return unsafe { neon(values) };
        }
    }
    scalar(values)
}

fn scalar(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut maximum = f32::NEG_INFINITY;
    for &value in values {
        if value.is_nan() || value == f32::INFINITY {
            return None;
        }
        maximum = maximum.max(value);
    }
    Some(maximum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(values: &[f32]) -> Option<f32> {
    use std::arch::x86_64::*;
    let mut maximum = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut invalid = _mm256_setzero_ps();
    let finite = _mm256_set1_ps(f32::MAX);
    let (chunks, tail) = values.as_chunks::<8>();
    for chunk in chunks {
        // SAFETY: every chunk contains eight readable f32 values; alignment is unrestricted.
        let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        // Unordered or greater than the largest finite value: NaN or +infinity.
        invalid = _mm256_or_ps(invalid, _mm256_cmp_ps::<_CMP_NLE_UQ>(v, finite));
        maximum = _mm256_max_ps(maximum, v);
    }
    if _mm256_movemask_ps(invalid) != 0 {
        return None;
    }
    let mut lanes = [0.0; 8];
    // SAFETY: lanes has space for exactly eight f32 values.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), maximum) };
    let mut maximum = lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &v in tail {
        if v.is_nan() || v == f32::INFINITY {
            return None;
        }
        maximum = maximum.max(v);
    }
    Some(maximum)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(values: &[f32]) -> Option<f32> {
    use std::arch::aarch64::*;
    let mut maximum = vdupq_n_f32(f32::NEG_INFINITY);
    let mut invalid = vdupq_n_u32(0);
    let finite = vdupq_n_f32(f32::MAX);
    let (chunks, tail) = values.as_chunks::<4>();
    for chunk in chunks {
        // SAFETY: every chunk contains four readable f32 values; alignment is unrestricted.
        let v = unsafe { vld1q_f32(chunk.as_ptr()) };
        invalid = vorrq_u32(
            invalid,
            vorrq_u32(vcgtq_f32(v, finite), vmvnq_u32(vceqq_f32(v, v))),
        );
        maximum = vmaxq_f32(maximum, v);
    }
    if vmaxvq_u32(invalid) != 0 {
        return None;
    }
    let mut maximum = vmaxvq_f32(maximum);
    for &v in tail {
        if v.is_nan() || v == f32::INFINITY {
            return None;
        }
        maximum = maximum.max(v);
    }
    Some(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tails_unaligned_inputs_and_extreme_values_match_scalar() {
        let values: Vec<_> = (0..521)
            .map(|i| match i % 13 {
                0 => f32::NEG_INFINITY,
                1 => f32::MIN,
                2 => f32::MAX,
                3 => -0.0,
                4 => f32::from_bits(1),
                _ => (i as f32 - 300.0) / 17.0,
            })
            .collect();
        for offset in 0..8 {
            for len in 0..=513 {
                let input = &values[offset..offset + len];
                assert_eq!(maximum(input), scalar(input), "offset={offset} len={len}");
            }
        }
        assert_eq!(maximum(&[f32::NEG_INFINITY; 129]), Some(f32::NEG_INFINITY));
    }

    #[test]
    fn invalid_values_in_every_vector_lane_and_tail_are_rejected() {
        for bad in [f32::NAN, f32::INFINITY, f32::from_bits(0xffc01234)] {
            for at in 0..137 {
                let mut input = [1.0; 137];
                input[at] = bad;
                assert_eq!(maximum(&input), None, "at={at}");
            }
        }
    }

    #[test]
    fn full_normalizer_keeps_original_f64_bits() {
        let mut seed = 123_u32;
        for n in [64, 65, 127, 32000, 128256, 262144] {
            let input: Vec<_> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed as f64 / u32::MAX as f64 * 160.0 - 80.0) as f32
                })
                .collect();
            let old_max = input
                .iter()
                .map(|&v| v as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let new_max = maximum(&input).unwrap() as f64;
            let normalize = |max: f64| {
                max + input
                    .iter()
                    .map(|&v| (v as f64 - max).exp())
                    .sum::<f64>()
                    .ln()
            };
            assert_eq!(normalize(old_max).to_bits(), normalize(new_max).to_bits());
        }
    }
}
