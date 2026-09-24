//! Fixed-width uppercase answer codes sized to the complete candidate set.
use crate::{Error, Result};

/// A-Z for up to 26 options, AA-ZZ for up to 676, AAA-ZZZ for up to
/// 17,576, and so on. There is no alphabet-derived candidate-count ceiling.
pub fn option_code_width(option_count: usize) -> usize {
    let mut width = 1;
    let mut last = option_count.saturating_sub(1);
    while last >= 26 {
        last /= 26;
        width += 1;
    }
    width
}

pub fn option_code(index: usize, option_count: usize) -> Result<String> {
    if option_count < 2 || index >= option_count {
        return Err(Error::Invalid("invalid option code index or count".into()));
    }
    let mut number = index;
    let mut bytes = vec![b'A'; option_code_width(option_count)];
    for byte in bytes.iter_mut().rev() {
        *byte += (number % 26) as u8;
        number /= 26;
    }
    Ok(String::from_utf8(bytes).expect("uppercase ASCII"))
}

pub(crate) fn option_code_index(code: &str, option_count: usize) -> Option<usize> {
    if code.len() != option_code_width(option_count) {
        return None;
    }
    let mut index = 0usize;
    for byte in code.bytes() {
        if !byte.is_ascii_uppercase() {
            return None;
        }
        index = index
            .checked_mul(26)?
            .checked_add(usize::from(byte - b'A'))?;
    }
    (index < option_count).then_some(index)
}

/// Score complete, prefix-free canonical token sequences. Shared prefixes are
/// evaluated once; every edge uses the full-vocabulary normalizer, not a
/// shortlist normalizer. Mixing code widths would violate this contract.
#[cfg(any(feature = "llama", feature = "wgpu", test))]
pub(crate) fn sequence_log_probabilities(
    paths: &[Vec<i32>],
    mut logits: impl FnMut(&[i32]) -> Result<Vec<f32>>,
) -> Result<Vec<f64>> {
    use std::collections::BTreeMap;
    validate_code_paths(paths)?;
    let mut prefixes: BTreeMap<Vec<i32>, Vec<(usize, i32)>> = BTreeMap::new();
    for (i, path) in paths.iter().enumerate() {
        for (j, &token) in path.iter().enumerate() {
            prefixes
                .entry(path[..j].to_vec())
                .or_default()
                .push((i, token));
        }
    }
    let mut scores = vec![0.0; paths.len()];
    for (prefix, edges) in prefixes {
        let values = logits(&prefix)?;
        if values.is_empty() || values.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            return Err(Error::Backend("invalid sequence vocabulary logits".into()));
        }
        let max = values
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let normalizer = max
            + values
                .iter()
                .map(|&v| (f64::from(v) - max).exp())
                .sum::<f64>()
                .ln();
        if !normalizer.is_finite() {
            return Err(Error::Backend("invalid sequence normalizer".into()));
        }
        for (index, token) in edges {
            let value = usize::try_from(token).ok().and_then(|i| values.get(i));
            match value {
                Some(v) if v.is_finite() => scores[index] += f64::from(*v) - normalizer,
                _ => {
                    return Err(Error::Backend(
                        "invalid sequence candidate token or logit".into(),
                    ));
                }
            }
        }
    }
    Ok(scores)
}

#[cfg(any(feature = "llama", feature = "wgpu", test))]
pub(crate) fn validate_code_paths(paths: &[Vec<i32>]) -> Result<()> {
    let mut ordered: Vec<_> = paths.iter().collect();
    ordered.sort();
    if paths.len() < 2
        || paths
            .iter()
            .any(|p| p.is_empty() || p.iter().any(|&t| t < 0))
        || ordered.windows(2).any(|p| p[1].starts_with(p[0]))
    {
        return Err(Error::Backend(
            "answer code token sequences must be nonempty, unique and prefix-free".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_scores_use_all_vocabulary_mass_and_share_prefixes() {
        // Canonical tokenizations have different token lengths. Candidate 2
        // is a single merged token; candidates 0/1 share their first token.
        let paths = vec![vec![0, 0], vec![0, 1], vec![2]];
        let mut calls = Vec::new();
        let scores = sequence_log_probabilities(&paths, |prefix| {
            calls.push(prefix.to_vec());
            Ok(if prefix.is_empty() {
                vec![0.0, 0.0, 0.0, 0.0]
            } else {
                vec![0.0, 0.0]
            })
        })
        .unwrap();
        assert_eq!(calls, vec![vec![], vec![0]]);
        let mass: f64 = scores.iter().map(|v| v.exp()).sum();
        assert!((mass - 0.5).abs() < 1e-12);
        assert!((scores[0].exp() / mass - 0.25).abs() < 1e-12);
        assert!((scores[2].exp() / mass - 0.5).abs() < 1e-12);
    }

    #[test]
    fn overlapping_paths_and_nonfinite_evidence_are_rejected() {
        for paths in [
            vec![vec![1], vec![1, 2]],
            vec![vec![1], vec![1]],
            vec![vec![], vec![1]],
        ] {
            assert!(
                sequence_log_probabilities(&paths, |_| panic!("must reject before inference"))
                    .is_err()
            );
        }
        assert!(
            sequence_log_probabilities(&[vec![0], vec![1]], |_| Ok(vec![0.0, f32::NAN])).is_err()
        );
    }
}
