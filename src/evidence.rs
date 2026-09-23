//! Exact native evidence. Partial top-k and generated probabilities cannot enter this type.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateObservation {
    pub option_id: String,
    pub token_id: i32,
    pub raw_logit: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExactEvidence {
    candidates: Vec<CandidateObservation>,
    full_vocabulary_log_normalizer: f64,
    vocabulary_size: usize,
    pub source: &'static str,
}
impl ExactEvidence {
    /// Only the trusted native boundary may supply a compact full-vocabulary
    /// summary; partial/provider scores cannot construct this evidence publicly.
    #[cfg(any(feature = "llama", test))]
    pub(crate) fn from_native_summary(
        decision: &Decision,
        logits: &[f32],
        tokens: &[i32],
        vocabulary_size: usize,
        normalizer: f64,
    ) -> Result<Self> {
        DecisionRequest {
            state: serde_json::Value::Null,
            decisions: vec![decision.clone()],
        }
        .validate()?;
        let options = decision.options();
        if logits.len() != options.len()
            || tokens.len() != options.len()
            || !normalizer.is_finite()
            || vocabulary_size < tokens.len()
            || logits.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Backend("invalid compact native evidence".into()));
        }
        let mut seen = HashSet::new();
        let mut candidates = Vec::with_capacity(tokens.len());
        for ((option, &token_id), &logit) in options.iter().zip(tokens).zip(logits) {
            if token_id < 0 || token_id as usize >= vocabulary_size || !seen.insert(token_id) {
                return Err(Error::Backend("invalid compact candidate mapping".into()));
            }
            candidates.push(CandidateObservation {
                option_id: option.id.clone(),
                token_id,
                raw_logit: logit as f64,
            });
        }
        let max = logits
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let candidate_lse = max
            + logits
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln();
        if normalizer + 1e-10 < candidate_lse {
            return Err(Error::Backend(
                "candidate mass exceeds full vocabulary mass".into(),
            ));
        }
        Ok(Self {
            candidates,
            full_vocabulary_log_normalizer: normalizer,
            vocabulary_size,
            source: "native_full_vocabulary_logits_v1",
        })
    }
    pub fn from_logits(decision: &Decision, logits: &[f32], tokens: &[i32]) -> Result<Self> {
        DecisionRequest {
            state: serde_json::Value::Null,
            decisions: vec![decision.clone()],
        }
        .validate()?;
        let options = decision.options();
        if logits.is_empty()
            || tokens.len() != options.len()
            || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
        {
            return Err(Error::Backend("invalid logits or candidate count".into()));
        }
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for (option, &token_id) in options.iter().zip(tokens) {
            if token_id < 0 || token_id as usize >= logits.len() || !seen.insert(token_id) {
                return Err(Error::Backend(
                    "candidate tokens must be unique and in vocabulary".into(),
                ));
            }
            let raw_logit = logits[token_id as usize] as f64;
            if !raw_logit.is_finite() {
                return Err(Error::Backend("candidate logit is not finite".into()));
            }
            candidates.push(CandidateObservation {
                option_id: option.id.clone(),
                token_id,
                raw_logit,
            });
        }
        // Same order and f64 arithmetic as the original full-logit scorer.
        let max = logits
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let full_vocabulary_log_normalizer = max
            + logits
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln();
        Ok(Self {
            candidates,
            full_vocabulary_log_normalizer,
            vocabulary_size: logits.len(),
            source: "native_full_vocabulary_logits_v1",
        })
    }
    pub fn candidates(&self) -> &[CandidateObservation] {
        &self.candidates
    }
    pub fn score(
        &self,
        decision: &Decision,
        input_tokens: usize,
        policy: &DecisionPolicy,
    ) -> Result<DecisionResult> {
        let options = decision.options();
        if options.len() != self.candidates.len()
            || options
                .iter()
                .zip(&self.candidates)
                .any(|(o, c)| o.id != c.option_id)
        {
            return Err(Error::Backend(
                "evidence semantic option mapping mismatch".into(),
            ));
        }
        DecisionRequest {
            state: serde_json::Value::Null,
            decisions: vec![decision.clone()],
        }
        .validate()?;
        let scores: Vec<_> = self.candidates.iter().map(|c| c.raw_logit).collect();
        let tokens: Vec<_> = self.candidates.iter().map(|c| c.token_id).collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let candidate_lse = max + scores.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
        let mass = (candidate_lse - self.full_vocabulary_log_normalizer)
            .exp()
            .clamp(0.0, 1.0);
        crate::decision::score_candidate_logits(
            decision,
            &scores,
            &tokens,
            input_tokens,
            mass,
            policy,
        )
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    #[test]
    fn trusted_summary_preserves_full_softmax_and_rejects_inconsistent_mass() {
        let request: DecisionRequest =
            serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
        let logits = [-3.0f32, 0.7, -1.0, 2.0, 0.4, f32::NEG_INFINITY];
        let normalizer = 2.0
            + logits
                .iter()
                .map(|&x| (x as f64 - 2.0).exp())
                .sum::<f64>()
                .ln();
        for decision in request.decisions {
            let tokens = &([3, 1, 4][..decision.options().len()]);
            let selected: Vec<_> = tokens.iter().map(|&i| logits[i as usize]).collect();
            let full = ExactEvidence::from_logits(&decision, &logits, tokens)
                .unwrap()
                .score(&decision, 42, &DecisionPolicy::default())
                .unwrap();
            let compact = ExactEvidence::from_native_summary(
                &decision,
                &selected,
                tokens,
                logits.len(),
                normalizer,
            )
            .unwrap()
            .score(&decision, 42, &DecisionPolicy::default())
            .unwrap();
            assert_eq!(
                serde_json::to_value(full).unwrap(),
                serde_json::to_value(compact).unwrap()
            );
            for invalid_normalizer in [f64::NAN, f64::INFINITY, -100.0] {
                assert!(
                    ExactEvidence::from_native_summary(
                        &decision,
                        &selected,
                        tokens,
                        logits.len(),
                        invalid_normalizer
                    )
                    .is_err()
                );
            }
            let duplicates = vec![3; tokens.len()];
            assert!(
                ExactEvidence::from_native_summary(
                    &decision,
                    &selected,
                    &duplicates,
                    logits.len(),
                    normalizer
                )
                .is_err()
            );
        }
    }
}
