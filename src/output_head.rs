//! Small task-specific heads over frozen deployment features.
use crate::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputHead {
    pub version: u32,
    pub id: String,
    /// `hidden` or `logit_affine`. Rows always use semantic option IDs.
    pub feature_kind: String,
    pub model_sha256: String,
    pub device: String,
    pub compute: ComputeOptions,
    pub prompt_version: String,
    pub decision_id: String,
    pub instruction: String,
    pub options: Vec<OptionSpec>,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub temperature: f64,
}

impl OutputHead {
    pub fn validate(&self, hidden_size: usize) -> Result<()> {
        DecisionRequest {
            state: serde_json::Value::Null,
            decisions: vec![Decision {
                id: self.decision_id.clone(),
                instruction: self.instruction.clone(),
                kind: DecisionKind::Choice {
                    options: self.options.clone(),
                },
            }],
        }
        .validate()?;
        self.compute.validate()?;
        let dim = match self.feature_kind.as_str() {
            "hidden" => hidden_size,
            "logit_affine" => self.options.len(),
            _ => return Err(Error::Invalid("unknown output head feature kind".into())),
        };
        if self.version != 1
            || self.id.is_empty()
            || dim == 0
            || self.model_sha256.len() != 64
            || !self.model_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.weights.len() != self.options.len()
            || self.bias.len() != self.options.len()
            || self
                .weights
                .iter()
                .any(|w| w.len() != dim || w.iter().any(|v| !v.is_finite()))
            || self.bias.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("malformed output head".into()));
        }
        Ok(())
    }

    /// Different tasks retain base behavior. A matching ID with changed meaning
    /// must fail instead of silently applying an incompatible classifier.
    pub fn applies_to(&self, decision: &Decision) -> Result<bool> {
        if decision.id != self.decision_id {
            return Ok(false);
        }
        let options = decision.options();
        if !matches!(decision.kind, DecisionKind::Choice { .. })
            || decision.instruction != self.instruction
            || options.len() != self.options.len()
            || options.iter().any(|o| {
                !self
                    .options
                    .iter()
                    .any(|x| x.id == o.id && x.criterion == o.criterion)
            })
        {
            return Err(Error::Invalid(
                "output head task signature does not match".into(),
            ));
        }
        Ok(true)
    }

    pub fn apply(
        &self,
        decision: &Decision,
        features: &[f32],
        base: DecisionResult,
        policy: &DecisionPolicy,
    ) -> Result<DecisionResult> {
        if !self.applies_to(decision)? {
            return Ok(base);
        }
        self.validate(features.len())?;
        let x: Vec<f64> = if self.feature_kind == "hidden" {
            features.iter().map(|&x| x as f64).collect()
        } else {
            self.options
                .iter()
                .map(|o| {
                    base.scores
                        .iter()
                        .find(|s| s.id == o.id)
                        .map(|s| s.raw_logit)
                        .ok_or_else(|| Error::Backend("missing base option score".into()))
                })
                .collect::<Result<_>>()?
        };
        if self.weights.iter().any(|w| w.len() != x.len()) || x.iter().any(|v| !v.is_finite()) {
            return Err(Error::Backend("invalid output head features".into()));
        }
        let canonical: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.bias)
            .map(|(w, b)| w.iter().zip(&x).map(|(a, x)| a * x).sum::<f64>() + b)
            .collect();
        let options = decision.options();
        let raw: Vec<f64> = options
            .iter()
            .map(|o| canonical[self.options.iter().position(|x| x.id == o.id).unwrap()])
            .collect();
        let calibrated: Vec<f64> = raw.iter().map(|z| z / self.temperature).collect();
        let tokens: Vec<i32> = base.scores.iter().map(|s| s.token_id).collect();
        let mut result = crate::decision::score_candidate_logits(
            decision,
            &calibrated,
            &tokens,
            base.input_tokens,
            base.candidate_mass,
            policy,
        )?;
        for (s, z) in result.scores.iter_mut().zip(raw) {
            s.raw_logit = z;
        }
        result.scoring_method = format!("learned_{}_softmax_with_base_mass_v1", self.feature_kind);
        result.calibration_id = Some(self.id.clone());
        result.reused_prefix_tokens = base.reused_prefix_tokens;
        Ok(result)
    }
}
