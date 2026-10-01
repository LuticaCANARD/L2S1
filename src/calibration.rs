//! Scalar temperature calibration; retains base full-vocabulary mass and raw logits.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub fn task_fingerprint(decision: &Decision) -> String {
    crate::interoperability::digest(&serde_json::to_vec(decision).expect("serializable decision"))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRecord {
    pub group: String,
    pub raw_logits: Vec<f64>,
    pub correct_option: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationMetrics {
    pub examples: usize,
    pub nll: f64,
    pub brier: f64,
    /// Top-label expected calibration error over 10 equal-width confidence
    /// bins. Absent in artifacts written before it was reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ece: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarCalibration {
    pub version: u32,
    pub id: String,
    pub model_fingerprint: String,
    pub task_fingerprint: String,
    pub decision_id: String,
    pub option_count: usize,
    pub temperature: f64,
    pub score_kind: String,
    pub calibration_data_sha256: String,
    pub calibration_group_sha256: Vec<String>,
    pub uncalibrated_fit_metrics: CalibrationMetrics,
    pub calibrated_fit_metrics: CalibrationMetrics,
}
impl ScalarCalibration {
    pub fn fit(
        id: String,
        model: &ModelIdentity,
        task: &Decision,
        records: &[CalibrationRecord],
    ) -> Result<Self> {
        DecisionRequest {
            shared: None,
            state: serde_json::Value::Null,
            decisions: vec![task.clone()],
        }
        .validate()?;
        validate_records(records, task.options().len())?;
        if id.trim().is_empty() {
            return Err(Error::Invalid("calibration ID is required".into()));
        }
        let temperature = fit_temperature(records)?;
        let before = calibration_metrics(records, 1.0)?;
        let groups = group_digests(records);
        Ok(Self {
            version: 1,
            id,
            model_fingerprint: model.fingerprint(),
            task_fingerprint: task_fingerprint(task),
            decision_id: task.id.clone(),
            option_count: task.options().len(),
            temperature,
            score_kind: "native_full_vocabulary_logits_v1".into(),
            calibration_data_sha256: crate::interoperability::digest(
                &serde_json::to_vec(records).map_err(|e| Error::Invalid(e.to_string()))?,
            ),
            calibration_group_sha256: groups,
            uncalibrated_fit_metrics: before,
            calibrated_fit_metrics: calibration_metrics(records, temperature)?,
        })
    }
    pub fn validate(&self, model: &ModelIdentity) -> Result<()> {
        if !(2..=26).contains(&self.option_count)
            || self.version != 1
            || self.id.trim().is_empty()
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.model_fingerprint != model.fingerprint()
            || self.score_kind != "native_full_vocabulary_logits_v1"
            || self.calibration_data_sha256.len() != 64
            || self.calibration_group_sha256.is_empty()
        {
            return Err(Error::Invalid(
                "incompatible scalar calibration identity, score kind or temperature".into(),
            ));
        }
        Ok(())
    }
    pub fn applies_to(&self, decision: &Decision) -> Result<bool> {
        if decision.id != self.decision_id {
            return Ok(false);
        }
        if task_fingerprint(decision) != self.task_fingerprint {
            return Err(Error::Invalid("calibration task signature mismatch".into()));
        }
        Ok(true)
    }
    /// Evaluate independent records and reject source-group leakage.
    pub fn evaluate_held_out(&self, records: &[CalibrationRecord]) -> Result<CalibrationMetrics> {
        validate_records(records, self.option_count)?;
        if records.iter().any(|r| {
            self.calibration_group_sha256
                .contains(&crate::interoperability::digest(r.group.as_bytes()))
        }) {
            return Err(Error::Invalid(
                "held-out groups overlap calibration groups".into(),
            ));
        }
        calibration_metrics(records, self.temperature)
    }
    pub fn apply(
        &self,
        decision: &Decision,
        base: DecisionResult,
        policy: &DecisionPolicy,
    ) -> Result<DecisionResult> {
        if !self.applies_to(decision)? {
            return Ok(base);
        }
        apply_temperature(
            decision,
            base,
            policy,
            self.temperature,
            &self.id,
            "temperature_softmax_with_base_mass_v1",
        )
    }
}
fn apply_temperature(
    decision: &Decision,
    base: DecisionResult,
    policy: &DecisionPolicy,
    temperature: f64,
    id: &str,
    method: &str,
) -> Result<DecisionResult> {
    if !temperature.is_finite()
        || temperature <= 0.0
        || base.scoring_method != "single_token_conditional_softmax_v1"
        || base.calibration_id.is_some()
    {
        return Err(Error::Invalid(
            "calibration requires uncalibrated native scores and positive temperature".into(),
        ));
    }
    let options = decision.options();
    if base.id != decision.id
        || base.scores.len() != options.len()
        || base.scores.iter().zip(&options).any(|(s, o)| s.id != o.id)
    {
        return Err(Error::Invalid("calibration score mapping mismatch".into()));
    }
    let scores: Vec<_> = base
        .scores
        .iter()
        .map(|s| s.raw_logit / temperature)
        .collect();
    let tokens: Vec<_> = base.scores.iter().map(|s| s.token_id).collect();
    let mut result = crate::decision::score_candidate_logits(
        decision,
        &scores,
        &tokens,
        base.input_tokens,
        base.candidate_mass,
        policy,
    )?;
    for (s, raw) in result.scores.iter_mut().zip(&base.scores) {
        s.raw_logit = raw.raw_logit;
    }
    result.reused_prefix_tokens = base.reused_prefix_tokens;
    result.truncated = base.truncated;
    result.scoring_method = method.into();
    result.calibration_id = Some(id.into());
    Ok(result)
}
/// Minimize NLL over log-temperature on a bounded interval. The identity
/// temperature is kept unless fitting improves the fit-set NLL.
fn fit_temperature(records: &[CalibrationRecord]) -> Result<f64> {
    let (mut lo, mut hi) = (0.05_f64.ln(), 20.0_f64.ln());
    for _ in 0..80 {
        let a = lo + (hi - lo) / 3.0;
        let b = hi - (hi - lo) / 3.0;
        if calibration_metrics(records, a.exp())?.nll < calibration_metrics(records, b.exp())?.nll {
            hi = b;
        } else {
            lo = a;
        }
    }
    let temperature = ((lo + hi) / 2.0).exp();
    if calibration_metrics(records, temperature)?.nll > calibration_metrics(records, 1.0)?.nll {
        return Ok(1.0);
    }
    Ok(temperature)
}
fn group_digests(records: &[CalibrationRecord]) -> Vec<String> {
    let mut groups: Vec<_> = records
        .iter()
        .map(|r| crate::interoperability::digest(r.group.as_bytes()))
        .collect();
    groups.sort();
    groups.dedup();
    groups
}
fn validate_records(records: &[CalibrationRecord], width: usize) -> Result<()> {
    if records.is_empty()
        || width < 2
        || records.iter().any(|r| {
            r.group.trim().is_empty()
                || r.raw_logits.len() != width
                || r.correct_option >= width
                || r.raw_logits.iter().any(|z| !z.is_finite())
        })
    {
        return Err(Error::Invalid("invalid calibration records".into()));
    }
    let groups: HashSet<_> = records.iter().map(|r| &r.group).collect();
    if groups.is_empty() {
        return Err(Error::Invalid("calibration requires source groups".into()));
    }
    Ok(())
}
pub fn calibration_metrics(
    records: &[CalibrationRecord],
    temperature: f64,
) -> Result<CalibrationMetrics> {
    validate_records(records, records.first().map_or(0, |r| r.raw_logits.len()))?;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::Invalid(
            "temperature must be positive and finite".into(),
        ));
    }
    const BINS: usize = 10;
    let (mut nll, mut brier) = (0.0, 0.0);
    // (count, summed confidence, correct) per top-probability bin.
    let mut bins = [(0usize, 0.0f64, 0usize); BINS];
    for r in records {
        let z: Vec<_> = r.raw_logits.iter().map(|z| z / temperature).collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + z.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
        nll += lse - z[r.correct_option];
        brier += z
            .iter()
            .enumerate()
            .map(|(i, z)| ((z - lse).exp() - f64::from(i == r.correct_option)).powi(2))
            .sum::<f64>();
        // Ties resolve to the first option, like the raw argmax elsewhere.
        let top = (0..z.len()).fold(0, |best, i| if z[i] > z[best] { i } else { best });
        let confidence = (z[top] - lse).exp();
        let bin = ((confidence * BINS as f64) as usize).min(BINS - 1);
        bins[bin].0 += 1;
        bins[bin].1 += confidence;
        bins[bin].2 += usize::from(top == r.correct_option);
    }
    if !nll.is_finite() || !brier.is_finite() {
        return Err(Error::Invalid("calibration scores overflow".into()));
    }
    let ece = bins
        .iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|&(_, confidence, correct)| (confidence - correct as f64).abs())
        .sum::<f64>()
        / records.len() as f64;
    Ok(CalibrationMetrics {
        examples: records.len(),
        nll: nll / records.len() as f64,
        brier: brier / records.len() as f64,
        ece: Some(ece),
    })
}

/// The decision kind name used by family calibration scopes.
pub fn decision_kind_name(decision: &Decision) -> &'static str {
    match decision.kind {
        DecisionKind::Binary { .. } => "binary",
        DecisionKind::Choice { .. } => "choice",
        DecisionKind::Ordinal { .. } => "ordinal",
    }
}

/// Labeled raw-logit records for one task of a calibration family.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyCalibrationTask {
    pub decision: Decision,
    pub records: Vec<CalibrationRecord>,
}

/// One task scored with a temperature fitted on the other tasks only, with
/// records from the held-out task's source groups also excluded from fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskHeldOutMetrics {
    pub decision_id: String,
    pub task_fingerprint: String,
    pub temperature: f64,
    pub uncalibrated: CalibrationMetrics,
    pub calibrated: CalibrationMetrics,
}

/// Temperature calibration shared by every task of one decision kind and
/// option count, for tasks without their own labeled records.
///
/// A task-specific `ScalarCalibration` takes precedence. Fitting pools the
/// records of at least two distinct tasks; `leave_one_task_out` reports how a
/// temperature fitted without each task performs on that task, which is the
/// evidence for applying it to unseen tasks. Like scalar calibration it is
/// bound to the model/configuration fingerprint, keeps raw logits and the
/// base candidate mass, and calibrates only the candidate distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyCalibration {
    pub version: u32,
    pub id: String,
    pub model_fingerprint: String,
    pub decision_kind: String,
    pub option_count: usize,
    pub temperature: f64,
    pub score_kind: String,
    pub calibration_data_sha256: String,
    pub calibration_group_sha256: Vec<String>,
    pub calibration_task_sha256: Vec<String>,
    pub uncalibrated_fit_metrics: CalibrationMetrics,
    pub calibrated_fit_metrics: CalibrationMetrics,
    pub leave_one_task_out: Vec<TaskHeldOutMetrics>,
}

impl FamilyCalibration {
    pub fn fit(id: String, model: &ModelIdentity, tasks: &[FamilyCalibrationTask]) -> Result<Self> {
        if id.trim().is_empty() {
            return Err(Error::Invalid("calibration ID is required".into()));
        }
        let first = &tasks
            .first()
            .ok_or_else(|| Error::Invalid("family calibration requires tasks".into()))?
            .decision;
        let kind = decision_kind_name(first);
        let option_count = first.options().len();
        if !(2..=26).contains(&option_count) {
            return Err(Error::Invalid(
                "family calibration supports 2 to 26 options".into(),
            ));
        }
        let mut fingerprints = Vec::with_capacity(tasks.len());
        for task in tasks {
            DecisionRequest {
                shared: None,
                state: serde_json::Value::Null,
                decisions: vec![task.decision.clone()],
            }
            .validate()?;
            if decision_kind_name(&task.decision) != kind
                || task.decision.options().len() != option_count
            {
                return Err(Error::Invalid(
                    "family tasks must share decision kind and option count".into(),
                ));
            }
            validate_records(&task.records, option_count)?;
            fingerprints.push(task_fingerprint(&task.decision));
        }
        let mut distinct = fingerprints.clone();
        distinct.sort();
        distinct.dedup();
        if distinct.len() != tasks.len() || distinct.len() < 2 {
            return Err(Error::Invalid(
                "family calibration requires at least two distinct tasks".into(),
            ));
        }
        let records: Vec<_> = tasks.iter().flat_map(|t| t.records.clone()).collect();
        let temperature = fit_temperature(&records)?;
        let mut leave_one_task_out = Vec::with_capacity(tasks.len());
        for (held, fingerprint) in tasks.iter().zip(&fingerprints) {
            let held_groups: HashSet<_> = held.records.iter().map(|r| &r.group).collect();
            let fit: Vec<_> = tasks
                .iter()
                .filter(|t| !std::ptr::eq(*t, held))
                .flat_map(|t| &t.records)
                .filter(|r| !held_groups.contains(&r.group))
                .cloned()
                .collect();
            if fit.is_empty() {
                return Err(Error::Invalid(
                    "every task needs records from other tasks and source groups".into(),
                ));
            }
            let held_temperature = fit_temperature(&fit)?;
            leave_one_task_out.push(TaskHeldOutMetrics {
                decision_id: held.decision.id.clone(),
                task_fingerprint: fingerprint.clone(),
                temperature: held_temperature,
                uncalibrated: calibration_metrics(&held.records, 1.0)?,
                calibrated: calibration_metrics(&held.records, held_temperature)?,
            });
        }
        Ok(Self {
            version: 1,
            id,
            model_fingerprint: model.fingerprint(),
            decision_kind: kind.into(),
            option_count,
            temperature,
            score_kind: "native_full_vocabulary_logits_v1".into(),
            calibration_data_sha256: crate::interoperability::digest(
                &serde_json::to_vec(tasks).map_err(|e| Error::Invalid(e.to_string()))?,
            ),
            calibration_group_sha256: group_digests(&records),
            calibration_task_sha256: distinct,
            uncalibrated_fit_metrics: calibration_metrics(&records, 1.0)?,
            calibrated_fit_metrics: calibration_metrics(&records, temperature)?,
            leave_one_task_out,
        })
    }
    pub fn validate(&self, model: &ModelIdentity) -> Result<()> {
        if !(2..=26).contains(&self.option_count)
            || self.version != 1
            || self.id.trim().is_empty()
            || !matches!(self.decision_kind.as_str(), "binary" | "choice" | "ordinal")
            || (self.decision_kind == "binary" && self.option_count != 2)
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.model_fingerprint != model.fingerprint()
            || self.score_kind != "native_full_vocabulary_logits_v1"
            || self.calibration_data_sha256.len() != 64
            || self.calibration_group_sha256.is_empty()
            || self.calibration_task_sha256.len() < 2
        {
            return Err(Error::Invalid(
                "incompatible family calibration identity, scope or temperature".into(),
            ));
        }
        Ok(())
    }
    pub fn applies_to(&self, decision: &Decision) -> bool {
        decision_kind_name(decision) == self.decision_kind
            && decision.options().len() == self.option_count
    }
    /// Evaluate independent records and reject source-group leakage.
    pub fn evaluate_held_out(&self, records: &[CalibrationRecord]) -> Result<CalibrationMetrics> {
        validate_records(records, self.option_count)?;
        if records.iter().any(|r| {
            self.calibration_group_sha256
                .contains(&crate::interoperability::digest(r.group.as_bytes()))
        }) {
            return Err(Error::Invalid(
                "held-out groups overlap calibration groups".into(),
            ));
        }
        calibration_metrics(records, self.temperature)
    }
    /// Calibrate a result that no task-specific calibration handled; results
    /// already calibrated or outside the family scope are returned unchanged.
    pub fn apply(
        &self,
        decision: &Decision,
        base: DecisionResult,
        policy: &DecisionPolicy,
    ) -> Result<DecisionResult> {
        if !self.applies_to(decision) || base.calibration_id.is_some() {
            return Ok(base);
        }
        apply_temperature(
            decision,
            base,
            policy,
            self.temperature,
            &self.id,
            "family_temperature_softmax_with_base_mass_v1",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPolicyRecord {
    pub observation: CalibrationRecord,
    pub base_candidate_mass: f64,
}
#[derive(Debug, Serialize)]
pub struct CalibrationPolicyMetrics {
    pub examples: usize,
    pub accepted: usize,
    pub coverage: f64,
    pub accepted_accuracy: Option<f64>,
    pub raw_top_accuracy: f64,
}
impl ScalarCalibration {
    /// A held-out policy curve can call this with several probability/mass thresholds.
    /// Both the unchanged base mass gate and the calibrated probability/tie gate apply.
    pub fn evaluate_policy(
        &self,
        records: &[CalibrationPolicyRecord],
        policy: &DecisionPolicy,
    ) -> Result<CalibrationPolicyMetrics> {
        policy.validate()?;
        let observations: Vec<_> = records.iter().map(|r| r.observation.clone()).collect();
        self.evaluate_held_out(&observations)?;
        let (mut accepted, mut accepted_correct, mut correct) = (0, 0, 0);
        for r in records {
            if !r.base_candidate_mass.is_finite() || !(0.0..=1.0).contains(&r.base_candidate_mass) {
                return Err(Error::Invalid(
                    "invalid held-out base candidate mass".into(),
                ));
            }
            let z: Vec<_> = r
                .observation
                .raw_logits
                .iter()
                .map(|z| z / self.temperature)
                .collect();
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = max + z.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
            let p: Vec<_> = z.iter().map(|z| (z - lse).exp()).collect();
            let best = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
            let is_correct = best == r.observation.correct_option;
            correct += usize::from(is_correct);
            if r.base_candidate_mass >= policy.min_candidate_mass
                && p[best] >= policy.min_top_probability
                && p.iter().filter(|&&v| (v - p[best]).abs() < 1e-12).count() == 1
            {
                accepted += 1;
                accepted_correct += usize::from(is_correct);
            }
        }
        Ok(CalibrationPolicyMetrics {
            examples: records.len(),
            accepted,
            coverage: accepted as f64 / records.len() as f64,
            accepted_accuracy: (accepted > 0).then(|| accepted_correct as f64 / accepted as f64),
            raw_top_accuracy: correct as f64 / records.len() as f64,
        })
    }
}
