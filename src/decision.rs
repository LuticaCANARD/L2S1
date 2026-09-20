use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("inference failed: {0}")]
    Backend(String),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub state: serde_json::Value,
    pub decisions: Vec<Decision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub id: String,
    pub instruction: String,
    pub kind: DecisionKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionKind {
    Binary {
        false_label: String,
        true_label: String,
    },
    Choice {
        options: Vec<OptionSpec>,
    },
    Ordinal {
        levels: Vec<Level>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionSpec {
    pub id: String,
    pub criterion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Level {
    pub id: String,
    pub criterion: String,
    pub value: f64,
}

impl Decision {
    pub fn options(&self) -> Vec<OptionSpec> {
        match &self.kind {
            DecisionKind::Binary {
                false_label,
                true_label,
            } => vec![
                OptionSpec {
                    id: "false".into(),
                    criterion: false_label.clone(),
                },
                OptionSpec {
                    id: "true".into(),
                    criterion: true_label.clone(),
                },
            ],
            DecisionKind::Choice { options } => options.clone(),
            DecisionKind::Ordinal { levels } => levels
                .iter()
                .map(|l| OptionSpec {
                    id: l.id.clone(),
                    criterion: l.criterion.clone(),
                })
                .collect(),
        }
    }
}

impl DecisionRequest {
    pub fn validate(&self) -> Result<()> {
        if self.decisions.is_empty() {
            return Err(Error::Invalid("decisions must not be empty".into()));
        }
        let mut ids = HashSet::new();
        for d in &self.decisions {
            if d.id.trim().is_empty() || !ids.insert(&d.id) || d.instruction.trim().is_empty() {
                return Err(Error::Invalid(
                    "decision IDs must be unique and instructions nonempty".into(),
                ));
            }
            let options = d.options();
            if !(2..=26).contains(&options.len()) {
                return Err(Error::Invalid(
                    "each decision requires 2..=26 options".into(),
                ));
            }
            let mut option_ids = HashSet::new();
            for o in &options {
                if o.id.trim().is_empty()
                    || o.criterion.trim().is_empty()
                    || !option_ids.insert(&o.id)
                {
                    return Err(Error::Invalid(
                        "option IDs must be unique and criteria nonempty".into(),
                    ));
                }
            }
            if let DecisionKind::Ordinal { levels } = &d.kind
                && (levels.iter().any(|l| !l.value.is_finite())
                    || levels.windows(2).any(|l| l[0].value >= l[1].value))
            {
                return Err(Error::Invalid(
                    "ordinal values must be finite and strictly increasing".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Thresholds are user policy, not guarantees of correctness.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionPolicy {
    pub min_top_probability: f64,
    pub min_candidate_mass: f64,
}

impl Default for DecisionPolicy {
    fn default() -> Self {
        Self {
            min_top_probability: 0.8,
            min_candidate_mass: 0.05,
        }
    }
}

impl DecisionPolicy {
    pub fn validate(&self) -> Result<()> {
        for p in [self.min_top_probability, self.min_candidate_mass] {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(Error::Invalid(
                    "policy thresholds must be finite values in [0, 1]".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionValue {
    Binary {
        p_true: f64,
        value: Option<bool>,
    },
    Choice {
        selected: Option<String>,
    },
    Ordinal {
        expected_value: f64,
        selected: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptionScore {
    pub id: String,
    pub code: String,
    pub token_id: i32,
    pub raw_logit: f64,
    pub option_probability: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstentionReason {
    LowCandidateMass,
    LowTopProbability,
    TiedCandidates,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionResult {
    pub id: String,
    pub value: DecisionValue,
    pub scores: Vec<OptionScore>,
    pub candidate_mass: f64,
    pub top_option_probability: f64,
    pub entropy_confidence: f64,
    pub abstention_reasons: Vec<AbstentionReason>,
    pub scoring_method: String,
    pub calibration_id: Option<String>,
    pub input_tokens: usize,
    pub truncated: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BackendInfo {
    pub model_path: String,
    pub model_description: String,
    #[serde(default)]
    pub model_architecture: String,
    #[serde(default)]
    pub prompt_profile: String,
    pub prompt_version: String,
    pub runtime: String,
    pub offload_requested: bool,
    pub offload_device: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub backend: BackendInfo,
    pub policy: DecisionPolicy,
    pub results: Vec<DecisionResult>,
}

pub trait DecisionBackend {
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse>;
}

/// Scores are conditional on the supplied candidate set. No calibration is implied.
pub fn score_logits(
    decision: &Decision,
    logits: &[f32],
    candidate_tokens: &[i32],
    input_tokens: usize,
    policy: &DecisionPolicy,
) -> Result<DecisionResult> {
    DecisionRequest {
        state: serde_json::Value::Null,
        decisions: vec![decision.clone()],
    }
    .validate()?;
    policy.validate()?;
    let options = decision.options();
    if candidate_tokens.len() != options.len()
        || logits.is_empty()
        || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
    {
        return Err(Error::Backend("invalid logits or candidate count".into()));
    }
    let mut seen = HashSet::new();
    let mut selected_logits = Vec::new();
    for &id in candidate_tokens {
        if id < 0 || id as usize >= logits.len() || !seen.insert(id) {
            return Err(Error::Backend(
                "candidate tokens must be unique and in vocabulary".into(),
            ));
        }
        let z = logits[id as usize] as f64;
        if !z.is_finite() {
            return Err(Error::Backend("candidate logit is not finite".into()));
        }
        selected_logits.push(z);
    }
    let log_sum_exp = |values: &[f64]| {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
    };
    let candidate_lse = log_sum_exp(&selected_logits);
    let full_lse = log_sum_exp(&logits.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let mass = (candidate_lse - full_lse).exp().clamp(0.0, 1.0);
    let p: Vec<f64> = selected_logits
        .iter()
        .map(|z| (z - candidate_lse).exp())
        .collect();
    let best = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    let mut reasons = Vec::new();
    if mass < policy.min_candidate_mass {
        reasons.push(AbstentionReason::LowCandidateMass);
    }
    if p[best] < policy.min_top_probability {
        reasons.push(AbstentionReason::LowTopProbability);
    }
    if p.iter().filter(|&&v| (v - p[best]).abs() < 1e-12).count() > 1 {
        reasons.push(AbstentionReason::TiedCandidates);
    }
    let accepted = reasons.is_empty();
    let value = match &decision.kind {
        DecisionKind::Binary { .. } => DecisionValue::Binary {
            p_true: p[1],
            value: accepted.then_some(best == 1),
        },
        DecisionKind::Choice { .. } => DecisionValue::Choice {
            selected: accepted.then(|| options[best].id.clone()),
        },
        DecisionKind::Ordinal { levels } => DecisionValue::Ordinal {
            expected_value: levels.iter().zip(&p).map(|(l, p)| l.value * p).sum(),
            selected: accepted.then(|| options[best].id.clone()),
        },
    };
    let entropy = -p
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|v| v * v.ln())
        .sum::<f64>();
    Ok(DecisionResult {
        id: decision.id.clone(),
        value,
        scores: options
            .iter()
            .enumerate()
            .map(|(i, o)| OptionScore {
                id: o.id.clone(),
                code: ((b'A' + i as u8) as char).to_string(),
                token_id: candidate_tokens[i],
                raw_logit: selected_logits[i],
                option_probability: p[i],
            })
            .collect(),
        candidate_mass: mass,
        top_option_probability: p[best],
        entropy_confidence: (1.0 - entropy / (p.len() as f64).ln()).clamp(0.0, 1.0),
        abstention_reasons: reasons,
        scoring_method: "single_token_conditional_softmax_v1".into(),
        calibration_id: None,
        input_tokens,
        truncated: false,
    })
}
