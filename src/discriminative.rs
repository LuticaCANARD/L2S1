//! Evidence from a trained candidate head, without vocabulary token IDs or mass.
use crate::{Decision, DecisionKind, Error, Result};
use serde_json::{Value, json};

pub fn score(
    decision: &Decision,
    logits: &[f32],
    temperature: f64,
    threshold: f64,
    calibration_id: &str,
    input_tokens: usize,
) -> Result<Value> {
    let options = decision.options();
    if options.len() < 2
        || logits.len() != options.len()
        || logits.iter().any(|v| !v.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
    {
        return Err(Error::Backend(
            "invalid discriminative head scores or policy".into(),
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut probabilities: Vec<f64> = logits
        .iter()
        .map(|&z| ((z as f64 - max) / temperature).exp())
        .collect();
    let sum: f64 = probabilities.iter().sum();
    for p in &mut probabilities {
        *p /= sum;
    }
    let mut best = 0;
    for i in 1..probabilities.len() {
        if probabilities[i] > probabilities[best] {
            best = i;
        }
    }
    let pmax = probabilities[best];
    let mut reasons = Vec::new();
    if pmax < threshold {
        reasons.push("low_top_probability");
    }
    if probabilities.iter().filter(|&&p| p == pmax).count() > 1 {
        reasons.push("tied_candidates");
    }
    let accepted = reasons.is_empty();
    let (value, estimate) = match &decision.kind {
        DecisionKind::Binary { .. } => (
            json!({"type":"binary","value":if accepted{Some(best==1)}else{None}}),
            json!({"p_true":probabilities[1]}),
        ),
        DecisionKind::Choice { .. } => (
            json!({"type":"choice","selected":if accepted{Some(&options[best].id)}else{None}}),
            json!({}),
        ),
        DecisionKind::Ordinal { levels } => (
            json!({"type":"ordinal","selected":if accepted{Some(&options[best].id)}else{None}}),
            json!({"expected_value":levels.iter().zip(&probabilities).map(|(l,p)|l.value*p).sum::<f64>()}),
        ),
    };
    let entropy = -probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|p| p * p.ln())
        .sum::<f64>()
        / (probabilities.len() as f64).ln();
    Ok(
        json!({"id":decision.id,"value":value,"status":if accepted{"selected"}else{"abstained"},"abstention_reasons":reasons,
        "evidence":{"type":"discriminative","schema_sha256":crate::cascade::schema_id(decision)?,"scores":options.iter().zip(logits).zip(&probabilities).map(|((o,z),p)|json!({"id":o.id,"raw_logit":z,"option_probability":p})).collect::<Vec<_>>(),
            "top_option_probability":pmax,"entropy_confidence":1.0-entropy,"scoring_method":"laya_marker_softmax_v1",
            "calibration_id":calibration_id,"temperature":temperature,"truncated":false,"estimate":estimate},
        "usage":{"input_tokens":input_tokens,"reused_prefix_tokens":0}}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_abstain_and_no_language_model_evidence_is_invented() {
        let d = Decision {
            id: "x".into(),
            instruction: "x".into(),
            kind: DecisionKind::Binary {
                false_label: "no".into(),
                true_label: "yes".into(),
            },
        };
        let value = score(&d, &[1.0, 1.0], 1.0, 0.0, "test", 10).unwrap();
        assert_eq!(value["status"], "abstained");
        assert!(value["value"]["value"].is_null());
        assert!(value["evidence"].get("candidate_mass").is_none());
        assert!(value["evidence"]["scores"][0].get("token_id").is_none());
        assert_eq!(
            score(&d, &[0.0, 4.0], 1.0, 0.8, "test", 10).unwrap()["value"]["value"],
            true
        );
        assert!(score(&d, &[f32::NAN, 0.0], 1.0, 0.0, "test", 10).is_err());
    }
}
