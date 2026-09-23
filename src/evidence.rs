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
