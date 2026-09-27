//! Artifact-bound, schema-specific fast/slow routing. No policy means slow-only.
use crate::{Decision, DecisionRequest, Error, Result, http::HttpDecisionBackend};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub fn schema_id(decision: &Decision) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(decision).map_err(|e| Error::Invalid(e.to_string()))?)
    ))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CascadeRule {
    pub schema_sha256: String,
    pub min_probability: f64,
    pub calibration_dataset_sha256: String,
    pub validation_dataset_sha256: String,
    pub validation_items: usize,
    pub accepted: usize,
    pub errors: usize,
    /// Empirical held-out error limit, not a statistical or production guarantee.
    pub max_empirical_error: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CascadePolicy {
    pub version: u32,
    pub fast_artifact_id: String,
    pub slow_artifact_id: String,
    pub rules: Vec<CascadeRule>,
}
fn digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
impl CascadePolicy {
    pub fn validate(&self, fast: &Value, slow: &Value) -> Result<()> {
        if self.version != 1
            || !digest(&self.fast_artifact_id)
            || !digest(&self.slow_artifact_id)
            || fast["artifact_id"].as_str() != Some(&self.fast_artifact_id)
            || slow["artifact_id"].as_str() != Some(&self.slow_artifact_id)
        {
            return Err(Error::Invalid(
                "cascade policy does not match loaded model/runtime artifacts".into(),
            ));
        }
        let mut schemas = std::collections::HashSet::new();
        for r in &self.rules {
            if !digest(&r.schema_sha256)
                || !schemas.insert(&r.schema_sha256)
                || !digest(&r.calibration_dataset_sha256)
                || !digest(&r.validation_dataset_sha256)
                || r.calibration_dataset_sha256 == r.validation_dataset_sha256
                || !r.min_probability.is_finite()
                || !(0.0..=1.0).contains(&r.min_probability)
                || !r.max_empirical_error.is_finite()
                || !(0.0..=1.0).contains(&r.max_empirical_error)
                || r.accepted == 0
                || r.accepted > r.validation_items
                || r.errors > r.accepted
                || r.errors as f64 / r.accepted as f64 > r.max_empirical_error
            {
                return Err(Error::Invalid("cascade rule requires unique schemas and separate successful held-out validation".into()));
            }
        }
        Ok(())
    }
}
pub struct CascadeBackend<F, S> {
    fast: F,
    slow: S,
    policy: Option<CascadePolicy>,
    caps: Value,
}
impl<F: HttpDecisionBackend, S: HttpDecisionBackend> CascadeBackend<F, S> {
    pub fn new(fast: F, slow: S, policy: Option<CascadePolicy>) -> Result<Self> {
        let f = fast.capabilities();
        let s = slow.capabilities();
        if let Some(p) = &policy {
            p.validate(&f, &s)?;
        }
        let mut caps = s.clone();
        caps["backend"] = json!({"runtime":"cascade","model":"schema-routed"});
        caps["evidence"] = "mixed".into();
        caps["batch"]["supported"] = false.into();
        caps["batch"]["enabled"] = false.into();
        caps["request_policy"] = json!({"supported":false,"target_error_rate":"use a held-out validated cascade policy"});
        caps["routing"] = json!({"enabled":policy.is_some(),"fast":f,"slow":s,"guaranteed":false});
        Ok(Self {
            fast,
            slow,
            policy,
            caps,
        })
    }
}
impl<F: HttpDecisionBackend, S: HttpDecisionBackend> HttpDecisionBackend for CascadeBackend<F, S> {
    fn capabilities(&self) -> Value {
        self.caps.clone()
    }
    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> Result<Value> {
        request.validate()?;
        let mut results: Vec<Option<Value>> = vec![None; request.decisions.len()];
        let mut eligible = Vec::new();
        let mut thresholds = Vec::new();
        if images.is_empty()
            && let Some(policy) = &self.policy
        {
            for (i, d) in request.decisions.iter().enumerate() {
                let id = schema_id(d)?;
                if let Some(r) = policy.rules.iter().find(|r| r.schema_sha256 == id) {
                    eligible.push(i);
                    thresholds.push(r.min_probability);
                }
            }
        }
        let mut fast_failed = false;
        if !eligible.is_empty() {
            let fast_request = DecisionRequest {
                state: request.state.clone(),
                decisions: eligible
                    .iter()
                    .map(|&i| request.decisions[i].clone())
                    .collect(),
            };
            match self.fast.decide_json(&fast_request, &[]) {
                Ok(output) => {
                    let items = output["results"]
                        .as_array()
                        .filter(|a| a.len() == eligible.len())
                        .ok_or_else(|| {
                            Error::Backend("fast backend result count mismatch".into())
                        })?;
                    for ((&i, &threshold), item) in eligible.iter().zip(&thresholds).zip(items) {
                        if item["id"] != request.decisions[i].id {
                            return Err(Error::Backend(
                                "fast backend result identity mismatch".into(),
                            ));
                        }
                        if item["status"] == "selected"
                            && item["abstention_reasons"]
                                .as_array()
                                .is_some_and(Vec::is_empty)
                            && item["evidence"]["top_option_probability"]
                                .as_f64()
                                .is_some_and(|p| p.is_finite() && p >= threshold && p <= 1.0)
                        {
                            let mut item = item.clone();
                            item["routing"] = json!({"stage":"fast","reason":"validated_schema_threshold","backend":output["backend"],"threshold":threshold,"guaranteed":false});
                            results[i] = Some(item);
                        }
                    }
                }
                Err(_) => {
                    fast_failed = true;
                }
            }
        }
        let remaining: Vec<usize> = results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.is_none().then_some(i))
            .collect();
        if !remaining.is_empty() {
            let slow_request = DecisionRequest {
                state: request.state.clone(),
                decisions: remaining
                    .iter()
                    .map(|&i| request.decisions[i].clone())
                    .collect(),
            };
            let output = self.slow.decide_json(&slow_request, images)?;
            let items = output["results"]
                .as_array()
                .filter(|a| a.len() == remaining.len())
                .ok_or_else(|| Error::Backend("slow backend result count mismatch".into()))?;
            for (&i, item) in remaining.iter().zip(items) {
                if item["id"] != request.decisions[i].id {
                    return Err(Error::Backend(
                        "slow backend result identity mismatch".into(),
                    ));
                }
                let reason = if !images.is_empty() {
                    "image"
                } else if !eligible.contains(&i) {
                    "unvalidated_schema"
                } else if fast_failed {
                    "fast_backend_failure"
                } else {
                    "fast_abstention_or_threshold"
                };
                let mut item = item.clone();
                item["routing"] = json!({"stage":"slow","reason":reason,"backend":output["backend"],"guaranteed":false});
                results[i] = Some(item);
            }
        }
        Ok(
            json!({"backend":{"runtime":"cascade","model":"schema-routed","details":{"fast":self.caps["routing"]["fast"]["backend"],"slow":self.caps["routing"]["slow"]["backend"],"guaranteed":false}},"policy":null,"results":results}),
        )
    }
}
