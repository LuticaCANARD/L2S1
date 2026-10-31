//! Decision accuracy tuning API: temperature calibration, selective abstention
//! policy optimization, logit-affine output head fitting, and strategy sweeps.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn default_group() -> String {
    "default".into()
}

/// Tuning strategy to execute on the labeled dataset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TuningMode {
    /// Fit scalar temperature calibration, logit-affine output head, and
    /// selective policy thresholds; recommend the best-performing method.
    #[default]
    Auto,
    /// Fit scalar temperature calibration and optimize `DecisionPolicy` thresholds.
    PolicyAndTemperature,
    /// Fit an L2-regularized logit-affine `OutputHead` to improve raw top-1
    /// accuracy, followed by temperature scaling and `DecisionPolicy` tuning.
    LogitAffine,
    /// Sweep prompt detail levels and option-rotation consensus mixtures on the
    /// backend before fitting calibration and policy thresholds.
    StrategySweep,
}

/// One labeled example for tuning or held-out evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneExample {
    /// Source group identifier used to prevent fit/held-out data leakage.
    #[serde(default = "default_group")]
    pub group: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared: Option<serde_json::Value>,
    #[serde(default)]
    pub state: serde_json::Value,
    /// Expected option ID (string) or boolean (`false`/`true` for binary tasks).
    pub expected: serde_json::Value,
    /// Optional pre-collected raw candidate logits in semantic option order.
    /// When omitted, the active backend evaluates `state` to obtain logits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_logits: Option<Vec<f64>>,
    /// Optional pre-collected full-vocabulary candidate mass in `(0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_mass: Option<f64>,
}

/// Accuracy tuning request for library, CLI, stdio, and HTTP `/v1/tune`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneRequest {
    /// Identifier for generated calibration / output head artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Shared context inherited by examples that omit their own `shared` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared: Option<serde_json::Value>,
    pub decision: Decision,
    pub examples: Vec<TuneExample>,
    /// Optional independent validation examples with disjoint `group` values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub held_out: Vec<TuneExample>,
    #[serde(default)]
    pub mode: TuningMode,
    /// Minimum accepted accuracy target in `(0, 1]` for policy threshold selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_accepted_accuracy: Option<f64>,
    /// Minimum coverage (accepted fraction) in `[0, 1]` for policy selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_coverage: Option<f64>,
    /// L2 regularization strength toward the identity transform for `logit_affine`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l2_regularization: Option<f64>,
    /// Number of cyclic option-code rotations (`1..=26`) to evaluate in `strategy_sweep`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_passes: Option<usize>,
    /// When true on a resident backend, apply the tuned policy and calibration/head in place.
    #[serde(default)]
    pub apply: bool,
}

/// Combined accuracy and calibration quality metrics for a scored split.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuningSplitMetrics {
    pub examples: usize,
    pub raw_top_accuracy: f64,
    pub accepted: usize,
    pub errors: usize,
    pub coverage: f64,
    pub accepted_accuracy: Option<f64>,
    pub nll: f64,
    pub brier: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ece: Option<f64>,
}

/// One point on the selective classification accuracy/coverage frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyEvaluation {
    pub policy: DecisionPolicy,
    pub examples: usize,
    pub accepted: usize,
    pub errors: usize,
    pub coverage: f64,
    pub accepted_accuracy: Option<f64>,
    pub raw_top_accuracy: f64,
}

/// Evaluation of one prompt/rotation inference strategy during `strategy_sweep`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyEvaluation {
    pub name: String,
    pub prompt_detail: PromptDetail,
    pub rotation_passes: usize,
    pub raw_top_accuracy: f64,
    pub uncalibrated_nll: f64,
    pub calibrated_nll: f64,
    pub calibrated_ece: Option<f64>,
    pub temperature: f64,
    pub recommended_policy: DecisionPolicy,
    pub accepted_accuracy: Option<f64>,
    pub coverage: f64,
    pub elapsed_ms: f64,
}

/// Output of the decision accuracy tuning pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneResponse {
    pub id: String,
    pub decision_id: String,
    pub mode: TuningMode,
    /// `"temperature_calibration"`, `"logit_affine_head"`, or `"semantic_mixture"`.
    pub recommended_method: String,
    pub recommended_policy: DecisionPolicy,
    /// Whether `recommended_policy` satisfied `target_accepted_accuracy` and `min_coverage`.
    pub target_met: bool,
    /// Whether the tuned configuration was applied to the live backend.
    pub applied: bool,
    pub uncalibrated_fit: TuningSplitMetrics,
    pub tuned_fit: TuningSplitMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncalibrated_held_out: Option<TuningSplitMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tuned_held_out: Option<TuningSplitMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_calibration: Option<ScalarCalibration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_head: Option<OutputHead>,
    pub policy_frontier: Vec<PolicyEvaluation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strategies: Vec<StrategyEvaluation>,
}

impl TuneRequest {
    pub fn validate(&self) -> Result<()> {
        DecisionRequest {
            shared: self.shared.clone(),
            state: serde_json::Value::Null,
            decisions: vec![self.decision.clone()],
        }
        .validate()?;
        let option_count = self.decision.options().len();
        if !(2..=26).contains(&option_count) {
            return Err(Error::Invalid(
                "accuracy tuning supports decisions with 2 to 26 options".into(),
            ));
        }
        if let Some(id) = &self.id
            && id.trim().is_empty()
        {
            return Err(Error::Invalid(
                "tuning artifact ID must not be empty".into(),
            ));
        }
        if self.examples.is_empty() {
            return Err(Error::Invalid(
                "tuning requires at least one example".into(),
            ));
        }
        if let Some(target) = self.target_accepted_accuracy
            && (!target.is_finite() || !(0.0..=1.0).contains(&target) || target == 0.0)
        {
            return Err(Error::Invalid(
                "target_accepted_accuracy must be a finite value in (0, 1]".into(),
            ));
        }
        if let Some(min_cov) = self.min_coverage
            && (!min_cov.is_finite() || !(0.0..=1.0).contains(&min_cov))
        {
            return Err(Error::Invalid(
                "min_coverage must be a finite value in [0, 1]".into(),
            ));
        }
        if let Some(l2) = self.l2_regularization
            && (!l2.is_finite() || l2 <= 0.0)
        {
            return Err(Error::Invalid(
                "l2_regularization must be finite and positive".into(),
            ));
        }
        if let Some(passes) = self.rotation_passes
            && !(1..=26).contains(&passes)
        {
            return Err(Error::Invalid(
                "rotation_passes must be between 1 and 26".into(),
            ));
        }
        for example in self.examples.iter().chain(&self.held_out) {
            if example.group.trim().is_empty() {
                return Err(Error::Invalid("example group must not be empty".into()));
            }
            resolve_expected_option(&self.decision, &example.expected)?;
            if let Some(logits) = &example.raw_logits
                && (logits.len() != option_count || logits.iter().any(|z| !z.is_finite()))
            {
                return Err(Error::Invalid(
                    "example raw_logits must be finite and match decision option count".into(),
                ));
            }
            if let Some(mass) = example.candidate_mass
                && (!mass.is_finite() || !(0.0..=1.0).contains(&mass) || mass == 0.0)
            {
                return Err(Error::Invalid(
                    "example candidate_mass must be finite and in (0, 1]".into(),
                ));
            }
        }
        if !self.held_out.is_empty() {
            let fit_groups: HashSet<_> = self
                .examples
                .iter()
                .map(|e| crate::interoperability::digest(e.group.as_bytes()))
                .collect();
            if self
                .held_out
                .iter()
                .any(|e| fit_groups.contains(&crate::interoperability::digest(e.group.as_bytes())))
            {
                return Err(Error::Invalid(
                    "held-out groups overlap calibration groups".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn effective_id(&self) -> String {
        self.id
            .as_ref()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("tune-{}", self.decision.id))
    }
}

pub fn resolve_expected_option(decision: &Decision, expected: &serde_json::Value) -> Result<usize> {
    let options = decision.options();
    if let Some(id) = expected.as_str() {
        return options
            .iter()
            .position(|o| o.id == id)
            .ok_or_else(|| Error::Invalid(format!("unknown expected option ID {id:?}")));
    }
    if let Some(flag) = expected.as_bool()
        && matches!(decision.kind, DecisionKind::Binary { .. })
    {
        return Ok(usize::from(flag));
    }
    Err(Error::Invalid(
        "expected must be a valid option ID string (or boolean for binary decisions)".into(),
    ))
}

fn to_policy_records(
    decision: &Decision,
    examples: &[TuneExample],
) -> Result<Vec<CalibrationPolicyRecord>> {
    examples
        .iter()
        .map(|example| {
            let raw_logits = example.raw_logits.clone().ok_or_else(|| {
                Error::Invalid("missing raw_logits for offline tuning record".into())
            })?;
            let correct_option = resolve_expected_option(decision, &example.expected)?;
            Ok(CalibrationPolicyRecord {
                observation: CalibrationRecord {
                    group: example.group.clone(),
                    raw_logits,
                    correct_option,
                },
                base_candidate_mass: example.candidate_mass.unwrap_or(1.0),
            })
        })
        .collect()
}

/// Evaluate policy and calibration metrics for a set of policy records at `temperature`.
pub fn evaluate_split_metrics(
    records: &[CalibrationPolicyRecord],
    temperature: f64,
    policy: &DecisionPolicy,
) -> Result<TuningSplitMetrics> {
    policy.validate()?;
    let observations: Vec<_> = records.iter().map(|r| r.observation.clone()).collect();
    let cal = calibration_metrics(&observations, temperature)?;
    let eval = evaluate_policy_at_temperature(records, temperature, policy)?;
    Ok(TuningSplitMetrics {
        examples: eval.examples,
        raw_top_accuracy: eval.raw_top_accuracy,
        accepted: eval.accepted,
        errors: eval.errors,
        coverage: eval.coverage,
        accepted_accuracy: eval.accepted_accuracy,
        nll: cal.nll,
        brier: cal.brier,
        ece: cal.ece,
    })
}

pub fn evaluate_policy_at_temperature(
    records: &[CalibrationPolicyRecord],
    temperature: f64,
    policy: &DecisionPolicy,
) -> Result<PolicyEvaluation> {
    policy.validate()?;
    if records.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::Invalid(
            "policy evaluation requires nonempty records and positive temperature".into(),
        ));
    }
    let (mut accepted, mut accepted_correct, mut correct) = (0usize, 0usize, 0usize);
    for r in records {
        if !r.base_candidate_mass.is_finite()
            || !(0.0..=1.0).contains(&r.base_candidate_mass)
            || r.observation.raw_logits.len() < 2
            || r.observation.correct_option >= r.observation.raw_logits.len()
            || r.observation.raw_logits.iter().any(|z| !z.is_finite())
        {
            return Err(Error::Invalid("invalid policy evaluation record".into()));
        }
        let z: Vec<_> = r
            .observation
            .raw_logits
            .iter()
            .map(|z| z / temperature)
            .collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + z.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
        let p: Vec<_> = z.iter().map(|z| (z - lse).exp()).collect();
        let best = (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best });
        let is_correct = best == r.observation.correct_option;
        correct += usize::from(is_correct);
        if r.base_candidate_mass >= policy.min_candidate_mass
            && p[best] >= policy.min_top_probability
            && (policy.is_forced_classification()
                || p.iter().filter(|&&v| (v - p[best]).abs() < 1e-12).count() == 1)
        {
            accepted += 1;
            accepted_correct += usize::from(is_correct);
        }
    }
    let errors = accepted.saturating_sub(accepted_correct);
    Ok(PolicyEvaluation {
        policy: policy.clone(),
        examples: records.len(),
        accepted,
        errors,
        coverage: accepted as f64 / records.len() as f64,
        accepted_accuracy: (accepted > 0).then(|| accepted_correct as f64 / accepted as f64),
        raw_top_accuracy: correct as f64 / records.len() as f64,
    })
}

/// Build the Pareto frontier of selective classification policies and select
/// the best `DecisionPolicy` for the requested accuracy/coverage constraints.
pub fn optimize_policy(
    records: &[CalibrationPolicyRecord],
    temperature: f64,
    target_accepted_accuracy: Option<f64>,
    min_coverage: Option<f64>,
) -> Result<(DecisionPolicy, bool, Vec<PolicyEvaluation>)> {
    if records.is_empty() {
        return Err(Error::Invalid(
            "policy optimization requires records".into(),
        ));
    }
    let mut prob_candidates = vec![0.0];
    let mut mass_candidates = vec![0.0];
    for r in records {
        let z: Vec<_> = r
            .observation
            .raw_logits
            .iter()
            .map(|z| z / temperature)
            .collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + z.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
        let top_p = z
            .iter()
            .map(|z| (z - lse).exp())
            .fold(0.0_f64, f64::max)
            .clamp(0.0, 1.0);
        // Round to 4 decimal places for clean, reproducible policy thresholds.
        let rounded_p = (top_p * 10000.0).floor() / 10000.0;
        let rounded_m = ((r.base_candidate_mass * 10000.0).floor() / 10000.0).clamp(0.0, 1.0);
        prob_candidates.push(rounded_p);
        mass_candidates.push(rounded_m);
    }
    for fixed in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] {
        prob_candidates.push(fixed);
    }
    for fixed_mass in [0.01, 0.05, 0.1, 0.25, 0.5] {
        mass_candidates.push(fixed_mass);
    }
    prob_candidates.sort_by(f64::total_cmp);
    prob_candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    mass_candidates.sort_by(f64::total_cmp);
    mass_candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    // Keep candidate grid bounded if dataset is large.
    if mass_candidates.len() > 16 {
        let step = mass_candidates.len() / 12;
        mass_candidates = mass_candidates.into_iter().step_by(step.max(1)).collect();
        if *mass_candidates.first().unwrap_or(&1.0) > 0.0 {
            mass_candidates.insert(0, 0.0);
        }
    }

    let mut evaluations = Vec::new();
    for &min_p in &prob_candidates {
        for &min_m in &mass_candidates {
            let policy = DecisionPolicy {
                min_top_probability: min_p,
                min_candidate_mass: min_m,
            };
            let eval = evaluate_policy_at_temperature(records, temperature, &policy)?;
            if eval.accepted > 0 {
                evaluations.push(eval);
            }
        }
    }
    if evaluations.is_empty() {
        let forced =
            evaluate_policy_at_temperature(records, temperature, &DecisionPolicy::default())?;
        return Ok((DecisionPolicy::default(), false, vec![forced]));
    }

    // Extract Pareto frontier: sort by coverage descending, then accuracy descending, then lowest thresholds.
    evaluations.sort_by(|a, b| {
        b.coverage
            .total_cmp(&a.coverage)
            .then_with(|| {
                b.accepted_accuracy
                    .unwrap_or(0.0)
                    .total_cmp(&a.accepted_accuracy.unwrap_or(0.0))
            })
            .then_with(|| {
                a.policy
                    .min_top_probability
                    .total_cmp(&b.policy.min_top_probability)
            })
            .then_with(|| {
                a.policy
                    .min_candidate_mass
                    .total_cmp(&b.policy.min_candidate_mass)
            })
    });

    let mut frontier: Vec<PolicyEvaluation> = Vec::new();
    let mut best_acc_so_far = -1.0_f64;
    for eval in &evaluations {
        let acc = eval.accepted_accuracy.unwrap_or(0.0);
        if acc > best_acc_so_far + 1e-12 {
            best_acc_so_far = acc;
            if let Some(last) = frontier.last_mut()
                && (last.coverage - eval.coverage).abs() < 1e-12
            {
                if acc > last.accepted_accuracy.unwrap_or(0.0) {
                    *last = eval.clone();
                }
                continue;
            }
            frontier.push(eval.clone());
        }
    }

    let min_cov = min_coverage.unwrap_or(0.0);
    let (recommended, target_met) = if let Some(target_acc) = target_accepted_accuracy {
        // Maximize coverage among policies meeting both target_accepted_accuracy and min_coverage.
        let feasible: Vec<_> = evaluations
            .iter()
            .filter(|e| {
                e.coverage + 1e-12 >= min_cov
                    && e.accepted_accuracy.unwrap_or(0.0) + 1e-12 >= target_acc
            })
            .collect();
        if let Some(best) = feasible.first() {
            (best.policy.clone(), true)
        } else {
            // Fall back to highest accuracy that meets min_coverage, or overall highest accuracy.
            let cov_feasible: Vec<_> = evaluations
                .iter()
                .filter(|e| e.coverage + 1e-12 >= min_cov)
                .collect();
            let pool = if cov_feasible.is_empty() {
                evaluations.iter().collect::<Vec<_>>()
            } else {
                cov_feasible
            };
            let best = pool
                .into_iter()
                .max_by(|a, b| {
                    a.accepted_accuracy
                        .unwrap_or(0.0)
                        .total_cmp(&b.accepted_accuracy.unwrap_or(0.0))
                        .then_with(|| a.coverage.total_cmp(&b.coverage))
                })
                .unwrap();
            (best.policy.clone(), false)
        }
    } else if min_coverage.is_some() {
        // Maximize accepted_accuracy subject to coverage >= min_cov.
        let feasible: Vec<_> = evaluations
            .iter()
            .filter(|e| e.coverage + 1e-12 >= min_cov)
            .collect();
        if let Some(best) = feasible.into_iter().max_by(|a, b| {
            a.accepted_accuracy
                .unwrap_or(0.0)
                .total_cmp(&b.accepted_accuracy.unwrap_or(0.0))
                .then_with(|| a.coverage.total_cmp(&b.coverage))
        }) {
            (best.policy.clone(), true)
        } else {
            (DecisionPolicy::default(), false)
        }
    } else {
        // Default utility: if forced classification has 0 errors, keep forced (0.0, 0.0).
        // Otherwise pick the frontier point maximizing accuracy gain * coverage (with coverage >= 0.5).
        let forced = &evaluations[0];
        if forced.errors == 0 {
            (DecisionPolicy::default(), true)
        } else {
            let base_acc = forced.raw_top_accuracy;
            let best = frontier
                .iter()
                .filter(|e| e.coverage >= 0.5)
                .max_by(|a, b| {
                    let score_a = (a.accepted_accuracy.unwrap_or(0.0) - base_acc) * a.coverage;
                    let score_b = (b.accepted_accuracy.unwrap_or(0.0) - base_acc) * b.coverage;
                    score_a
                        .total_cmp(&score_b)
                        .then_with(|| a.coverage.total_cmp(&b.coverage))
                })
                .unwrap_or(forced);
            if best.accepted_accuracy.unwrap_or(0.0) > base_acc + 1e-9 {
                (best.policy.clone(), true)
            } else {
                (DecisionPolicy::default(), true)
            }
        }
    };

    Ok((recommended, target_met, frontier))
}

fn transform_logits(raw: &[f64], weights: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(bias)
        .map(|(w, &b)| w.iter().zip(raw).map(|(&a, &x)| a * x).sum::<f64>() + b)
        .collect()
}

fn apply_transform_to_records(
    records: &[CalibrationPolicyRecord],
    weights: &[Vec<f64>],
    bias: &[f64],
) -> Vec<CalibrationPolicyRecord> {
    records
        .iter()
        .map(|r| CalibrationPolicyRecord {
            observation: CalibrationRecord {
                group: r.observation.group.clone(),
                raw_logits: transform_logits(&r.observation.raw_logits, weights, bias),
                correct_option: r.observation.correct_option,
            },
            base_candidate_mass: r.base_candidate_mass,
        })
        .collect()
}

fn identity_weights(dim: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut weights = vec![vec![0.0; dim]; dim];
    for (i, row) in weights.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    (weights, vec![0.0; dim])
}

/// Fit L2-regularized affine logit weights and bias initialized at `(I, 0)`.
/// If `bias_only` is true, `W` stays fixed at `I` and only prior shift `b` is learned.
fn optimize_affine(
    records: &[CalibrationRecord],
    dim: usize,
    l2_reg: f64,
    bias_only: bool,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = records.len() as f64;
    let (mut w, mut b) = identity_weights(dim);
    let loss_and_grad = |w: &[Vec<f64>], b: &[f64]| -> (f64, Vec<Vec<f64>>, Vec<f64>) {
        let mut loss = 0.0;
        let mut gw = vec![vec![0.0; dim]; dim];
        let mut gb = vec![0.0; dim];
        for r in records {
            let z = transform_logits(&r.raw_logits, w, b);
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            loss += (lse - z[r.correct_option]) / n;
            for k in 0..dim {
                let p_k = (z[k] - lse).exp();
                let diff = (p_k - f64::from(k == r.correct_option)) / n;
                gb[k] += diff;
                if !bias_only {
                    for (j, &x_j) in r.raw_logits.iter().enumerate() {
                        gw[k][j] += diff * x_j;
                    }
                }
            }
        }
        for k in 0..dim {
            loss += 0.5 * l2_reg * b[k] * b[k];
            gb[k] += l2_reg * b[k];
            if !bias_only {
                for j in 0..dim {
                    let target = if k == j { 1.0 } else { 0.0 };
                    let d = w[k][j] - target;
                    loss += 0.5 * l2_reg * d * d;
                    gw[k][j] += l2_reg * d;
                }
            }
        }
        (loss, gw, gb)
    };

    let mean_norm_sq = records
        .iter()
        .map(|r| r.raw_logits.iter().map(|z| z * z).sum::<f64>())
        .sum::<f64>()
        / n.max(1.0);
    let mut step = (1.0 / (1.0 + mean_norm_sq + l2_reg)).clamp(1e-3, 1.0);

    for _ in 0..250 {
        let (loss, gw, gb) = loss_and_grad(&w, &b);
        if !loss.is_finite() {
            return identity_weights(dim);
        }
        let grad_sq: f64 = gw.iter().flatten().chain(gb.iter()).map(|g| g * g).sum();
        if !grad_sq.is_finite() || grad_sq.sqrt() < 1e-7 {
            break;
        }
        let mut accepted = false;
        let mut alpha = step;
        for _ in 0..20 {
            let mut nw = w.clone();
            let mut nb = b.clone();
            for k in 0..dim {
                nb[k] -= alpha * gb[k];
                if !bias_only {
                    for j in 0..dim {
                        nw[k][j] -= alpha * gw[k][j];
                    }
                }
            }
            // Center bias to zero mean for canonical representation.
            let mean_b = nb.iter().sum::<f64>() / dim as f64;
            for v in &mut nb {
                *v -= mean_b;
            }
            let (new_loss, _, _) = loss_and_grad(&nw, &nb);
            if new_loss.is_finite() && new_loss <= loss - 1e-4 * alpha * grad_sq {
                w = nw;
                b = nb;
                step = (alpha * 1.2).min(2.0);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    (w, b)
}

/// Fit a `logit_affine` `OutputHead` over labeled candidate logits.
pub fn fit_logit_affine_head(
    id: String,
    model: &ModelIdentity,
    decision: &Decision,
    records: &[CalibrationRecord],
    held_out: &[CalibrationRecord],
    l2_regularization: Option<f64>,
) -> Result<OutputHead> {
    let options = decision.options();
    let dim = options.len();
    if !(2..=26).contains(&dim) || records.is_empty() {
        return Err(Error::Invalid(
            "logit_affine head requires 2..=26 options and nonempty records".into(),
        ));
    }
    let candidate_regs = if let Some(reg) = l2_regularization {
        vec![reg]
    } else {
        vec![1e-3, 1e-2, 1e-1, 0.5]
    };

    let eval_set = if !held_out.is_empty() {
        held_out
    } else {
        records
    };

    let base_cal = ScalarCalibration::fit(id.clone(), model, decision, records)?;
    let base_eval_metrics = calibration_metrics(eval_set, base_cal.temperature)?;
    let base_acc = raw_accuracy(eval_set);

    let mut best_w = identity_weights(dim).0;
    let mut best_b = vec![0.0; dim];
    let mut best_temp = base_cal.temperature;
    let mut best_score = (base_acc, -base_eval_metrics.nll);

    for &reg in &candidate_regs {
        for bias_only in [true, false] {
            let (w, b) = optimize_affine(records, dim, reg, bias_only);
            let transformed_fit: Vec<_> = records
                .iter()
                .map(|r| CalibrationRecord {
                    group: r.group.clone(),
                    raw_logits: transform_logits(&r.raw_logits, &w, &b),
                    correct_option: r.correct_option,
                })
                .collect();
            let Ok(cal) = ScalarCalibration::fit(id.clone(), model, decision, &transformed_fit)
            else {
                continue;
            };
            let transformed_eval: Vec<_> = eval_set
                .iter()
                .map(|r| CalibrationRecord {
                    group: r.group.clone(),
                    raw_logits: transform_logits(&r.raw_logits, &w, &b),
                    correct_option: r.correct_option,
                })
                .collect();
            let Ok(metrics) = calibration_metrics(&transformed_eval, cal.temperature) else {
                continue;
            };
            let acc = raw_accuracy(&transformed_eval);
            let score = (acc, -metrics.nll);
            if score.0 > best_score.0 + 1e-12
                || ((score.0 - best_score.0).abs() <= 1e-12 && score.1 > best_score.1 + 1e-9)
            {
                best_score = score;
                best_w = w;
                best_b = b;
                best_temp = cal.temperature;
            }
        }
    }

    let head = OutputHead {
        version: 1,
        id,
        feature_kind: "logit_affine".into(),
        model_sha256: model.weights_sha256.clone(),
        device: model.device.clone(),
        compute: model.compute,
        prompt_version: model.prompt_version.clone(),
        decision_id: decision.id.clone(),
        instruction: decision.instruction.clone(),
        options,
        weights: best_w,
        bias: best_b,
        temperature: best_temp,
    };
    head.validate(dim)?;
    Ok(head)
}

fn raw_accuracy(records: &[CalibrationRecord]) -> f64 {
    if records.is_empty() {
        return 0.0;
    }
    let correct = records
        .iter()
        .filter(|r| {
            let best = (1..r.raw_logits.len()).fold(0, |best, i| {
                if r.raw_logits[i] > r.raw_logits[best] {
                    i
                } else {
                    best
                }
            });
            best == r.correct_option
        })
        .count();
    correct as f64 / records.len() as f64
}

/// Run the accuracy tuning pipeline on examples that already have `raw_logits` populated.
pub fn tune_records(request: &TuneRequest, model: &ModelIdentity) -> Result<TuneResponse> {
    request.validate()?;
    let id = request.effective_id();
    let fit_policy_records = to_policy_records(&request.decision, &request.examples)?;
    let held_policy_records = if request.held_out.is_empty() {
        Vec::new()
    } else {
        to_policy_records(&request.decision, &request.held_out)?
    };
    let fit_obs: Vec<_> = fit_policy_records
        .iter()
        .map(|r| r.observation.clone())
        .collect();
    let held_obs: Vec<_> = held_policy_records
        .iter()
        .map(|r| r.observation.clone())
        .collect();

    let uncalibrated_fit =
        evaluate_split_metrics(&fit_policy_records, 1.0, &DecisionPolicy::default())?;
    let uncalibrated_held_out = (!held_policy_records.is_empty())
        .then(|| evaluate_split_metrics(&held_policy_records, 1.0, &DecisionPolicy::default()))
        .transpose()?;

    let scalar = ScalarCalibration::fit(id.clone(), model, &request.decision, &fit_obs)?;
    if !held_obs.is_empty() {
        scalar.evaluate_held_out(&held_obs)?;
    }

    let head = if matches!(request.mode, TuningMode::Auto | TuningMode::LogitAffine) {
        Some(fit_logit_affine_head(
            format!("{id}-affine"),
            model,
            &request.decision,
            &fit_obs,
            &held_obs,
            request.l2_regularization,
        )?)
    } else {
        None
    };

    // Determine whether `logit_affine_head` or `temperature_calibration` is active.
    let use_affine = match request.mode {
        TuningMode::LogitAffine => true,
        TuningMode::PolicyAndTemperature | TuningMode::StrategySweep => false,
        TuningMode::Auto => {
            if let Some(h) = &head {
                let eval_records = if !held_policy_records.is_empty() {
                    &held_policy_records
                } else {
                    &fit_policy_records
                };
                let affine_records = apply_transform_to_records(eval_records, &h.weights, &h.bias);
                let affine_metrics = evaluate_split_metrics(
                    &affine_records,
                    h.temperature,
                    &DecisionPolicy::default(),
                )?;
                let temp_metrics = evaluate_split_metrics(
                    eval_records,
                    scalar.temperature,
                    &DecisionPolicy::default(),
                )?;
                affine_metrics.raw_top_accuracy > temp_metrics.raw_top_accuracy + 1e-12
                    || ((affine_metrics.raw_top_accuracy - temp_metrics.raw_top_accuracy).abs()
                        <= 1e-12
                        && affine_metrics.nll + 1e-9 < temp_metrics.nll)
            } else {
                false
            }
        }
    };

    let (active_fit_records, active_held_records, active_temp, recommended_method) = if use_affine {
        let h = head.as_ref().expect("head fitted");
        (
            apply_transform_to_records(&fit_policy_records, &h.weights, &h.bias),
            apply_transform_to_records(&held_policy_records, &h.weights, &h.bias),
            h.temperature,
            "logit_affine_head".to_owned(),
        )
    } else {
        (
            fit_policy_records.clone(),
            held_policy_records.clone(),
            scalar.temperature,
            "temperature_calibration".to_owned(),
        )
    };

    let selection_records = if !active_held_records.is_empty() {
        &active_held_records
    } else {
        &active_fit_records
    };
    let (recommended_policy, target_met, policy_frontier) = optimize_policy(
        selection_records,
        active_temp,
        request.target_accepted_accuracy,
        request.min_coverage,
    )?;

    let tuned_fit = evaluate_split_metrics(&active_fit_records, active_temp, &recommended_policy)?;
    let tuned_held_out = (!active_held_records.is_empty())
        .then(|| evaluate_split_metrics(&active_held_records, active_temp, &recommended_policy))
        .transpose()?;

    Ok(TuneResponse {
        id,
        decision_id: request.decision.id.clone(),
        mode: request.mode,
        recommended_method,
        recommended_policy,
        target_met,
        applied: false,
        uncalibrated_fit,
        tuned_fit,
        uncalibrated_held_out,
        tuned_held_out,
        scalar_calibration: Some(scalar),
        output_head: head,
        policy_frontier,
        strategies: Vec::new(),
    })
}

fn populate_examples_with_backend<B: DecisionBackend>(
    backend: &mut B,
    decision: &Decision,
    default_shared: &Option<serde_json::Value>,
    examples: &[TuneExample],
) -> Result<(Vec<TuneExample>, Option<BackendInfo>)> {
    let mut populated = Vec::with_capacity(examples.len());
    let mut last_info = None;
    for example in examples {
        if example.raw_logits.is_some() && example.candidate_mass.is_some() {
            populated.push(example.clone());
            continue;
        }
        let req = DecisionRequest {
            shared: example.shared.clone().or_else(|| default_shared.clone()),
            state: example.state.clone(),
            decisions: vec![decision.clone()],
        };
        let response = backend.decide(&req)?;
        last_info = Some(response.backend);
        let result = response
            .results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Backend("backend returned no decision result".into()))?;
        let mut item = example.clone();
        item.raw_logits = Some(result.scores.iter().map(|s| s.raw_logit).collect());
        item.candidate_mass = Some(result.candidate_mass);
        populated.push(item);
    }
    Ok((populated, last_info))
}

pub fn synthetic_identity_from_info(info: Option<&BackendInfo>) -> ModelIdentity {
    let digest_str = |s: &str| crate::interoperability::digest(s.as_bytes());
    let model_path = info.map(|i| i.model_path.as_str()).unwrap_or("offline");
    let prompt_version = info
        .map(|i| i.prompt_version.clone())
        .unwrap_or_else(|| "v1".into());
    let prompt_profile = info
        .map(|i| i.prompt_profile.clone())
        .unwrap_or_else(|| "model".into());
    let runtime = info.map(|i| i.runtime.as_str()).unwrap_or("offline");
    let device = info
        .and_then(|i| i.offload_device.clone())
        .unwrap_or_else(|| "CPU".into());
    let compute = info.and_then(|i| i.compute).unwrap_or_default();
    let execution_mode = info
        .map(|i| i.execution_mode)
        .unwrap_or(ExecutionMode::Fresh);
    let parallel_width = info.map(|i| i.parallel_width).unwrap_or(1);
    ModelIdentity {
        evidence_transfer: info.map(|i| i.evidence_transfer).unwrap_or_default(),
        weights_sha256: digest_str(model_path),
        template_sha256: digest_str(&prompt_version),
        prompt_profile,
        prompt_version,
        runtime_build_sha256: option_env!("L2S1_RUNTIME_BUILD_SHA256")
            .map(str::to_owned)
            .unwrap_or_else(|| digest_str(env!("CARGO_PKG_VERSION"))),
        loaded_runtime_sha256: digest_str(runtime),
        adapter_sha256: info.and_then(|i| i.lora_path.as_deref()).map(digest_str),
        head_sha256: None,
        device,
        compute,
        execution_mode,
        parallel_width,
        parallel_context_dynamic: info.is_some_and(|i| i.parallel_context_dynamic),
        parallel_prefix_alignment: info
            .map(|i| i.parallel_prefix_alignment)
            .unwrap_or_default(),
        parallel_wave_order: info.map(|i| i.parallel_wave_order).unwrap_or_default(),
    }
}

/// Run accuracy tuning against any `DecisionBackend`.
pub fn tune_backend<B: DecisionBackend>(
    backend: &mut B,
    request: &TuneRequest,
    identity: Option<&ModelIdentity>,
) -> Result<TuneResponse> {
    request.validate()?;
    let (examples, info_fit) = populate_examples_with_backend(
        backend,
        &request.decision,
        &request.shared,
        &request.examples,
    )?;
    let (held_out, info_held) = populate_examples_with_backend(
        backend,
        &request.decision,
        &request.shared,
        &request.held_out,
    )?;
    let resolved_identity = match identity {
        Some(id) => id.clone(),
        None => synthetic_identity_from_info(info_fit.as_ref().or(info_held.as_ref())),
    };
    let populated_request = TuneRequest {
        examples,
        held_out,
        ..request.clone()
    };
    tune_records(&populated_request, &resolved_identity)
}

#[cfg(feature = "llama")]
pub fn tune_llama(
    backend: &mut crate::llama::LlamaBackend,
    request: &TuneRequest,
) -> Result<TuneResponse> {
    request.validate()?;
    // Temporarily save and clear active calibrations/heads/policy so we collect
    // clean base logits during tuning; restore on error or if `apply` is false.
    let saved_policy = backend.policy().clone();
    let saved_detail = backend.prompt_detail();
    let saved_rotation = backend.code_rotation();
    let saved_mode = backend.info().execution_mode;
    let saved_calibrations = backend.calibrations_snapshot();
    let saved_family_calibrations = backend.family_calibrations_snapshot();
    let saved_head = backend.output_head_snapshot();

    let outcome = (|| -> Result<(TuneResponse, PromptDetail)> {
        backend.clear_calibrations();
        backend.clear_output_head();
        backend.set_policy(DecisionPolicy::default())?;
        if request.mode == TuningMode::LogitAffine && request.apply {
            backend.set_execution_mode(ExecutionMode::Fresh);
        }

        let mut strategies = Vec::new();
        let mut winning_detail = saved_detail;

        if request.mode == TuningMode::StrategySweep {
            let max_rotations = request
                .rotation_passes
                .unwrap_or_else(|| request.decision.options().len().min(3))
                .clamp(1, request.decision.options().len());
            let mut best_strategy_score = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            let mut best_examples = None;
            let mut best_held_out = None;
            let mut best_method_override = None;

            for detail in [
                PromptDetail::Minimal,
                PromptDetail::Typed,
                PromptDetail::TypedExamples,
            ] {
                backend.set_prompt_detail(detail);
                let mut rotation_options: Vec<usize> = vec![1];
                if max_rotations >= 2 && !backend.reasoning().is_thinking() {
                    rotation_options.push(max_rotations);
                }
                for &passes in &rotation_options {
                    let started = std::time::Instant::now();
                    let eval_examples = collect_llama_examples_with_rotations(
                        backend,
                        &request.decision,
                        &request.shared,
                        &request.examples,
                        passes,
                    )?;
                    let eval_held = collect_llama_examples_with_rotations(
                        backend,
                        &request.decision,
                        &request.shared,
                        &request.held_out,
                        passes,
                    )?;
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    backend.set_code_rotation(saved_rotation)?;
                    let sub_req = TuneRequest {
                        mode: TuningMode::PolicyAndTemperature,
                        examples: eval_examples.clone(),
                        held_out: eval_held.clone(),
                        ..request.clone()
                    };
                    let sub_res = tune_records(&sub_req, &backend.identity())?;
                    let active_metrics = sub_res
                        .tuned_held_out
                        .as_ref()
                        .unwrap_or(&sub_res.tuned_fit);
                    let uncal_metrics = sub_res
                        .uncalibrated_held_out
                        .as_ref()
                        .unwrap_or(&sub_res.uncalibrated_fit);
                    let name = if passes == 1 {
                        format!("detail_{detail:?}").to_ascii_lowercase()
                    } else {
                        format!("detail_{detail:?}_mixture_{passes}").to_ascii_lowercase()
                    };
                    strategies.push(StrategyEvaluation {
                        name,
                        prompt_detail: detail,
                        rotation_passes: passes,
                        raw_top_accuracy: active_metrics.raw_top_accuracy,
                        uncalibrated_nll: uncal_metrics.nll,
                        calibrated_nll: active_metrics.nll,
                        calibrated_ece: active_metrics.ece,
                        temperature: sub_res
                            .scalar_calibration
                            .as_ref()
                            .map_or(1.0, |c| c.temperature),
                        recommended_policy: sub_res.recommended_policy.clone(),
                        accepted_accuracy: active_metrics.accepted_accuracy,
                        coverage: active_metrics.coverage,
                        elapsed_ms,
                    });
                    let score = (active_metrics.raw_top_accuracy, -active_metrics.nll);
                    if score.0 > best_strategy_score.0 + 1e-12
                        || ((score.0 - best_strategy_score.0).abs() <= 1e-12
                            && score.1 > best_strategy_score.1 + 1e-9)
                    {
                        best_strategy_score = score;
                        winning_detail = detail;
                        best_examples = Some(eval_examples);
                        best_held_out = Some(eval_held);
                        best_method_override = (passes > 1).then(|| "semantic_mixture".to_owned());
                    }
                }
            }
            backend.set_prompt_detail(winning_detail);
            backend.set_code_rotation(saved_rotation)?;
            let populated = TuneRequest {
                examples: best_examples.unwrap(),
                held_out: best_held_out.unwrap(),
                ..request.clone()
            };
            let mut response = tune_records(&populated, &backend.identity())?;
            if let Some(method) = best_method_override {
                response.recommended_method = method;
            }
            response.strategies = strategies;
            Ok((response, winning_detail))
        } else {
            let mut response = tune_backend(backend, request, Some(&backend.identity()))?;
            response.strategies = strategies;
            Ok((response, winning_detail))
        }
    })();

    match outcome {
        Ok((mut response, winning_detail)) => {
            if request.apply {
                backend.set_prompt_detail(winning_detail);
                backend.set_code_rotation(saved_rotation)?;
                backend.set_policy(response.recommended_policy.clone())?;
                if response.recommended_method == "logit_affine_head"
                    && let Some(mut head) = response.output_head.clone()
                {
                    if backend.info().execution_mode != ExecutionMode::Fresh {
                        backend.set_execution_mode(ExecutionMode::Fresh);
                    }
                    head.prompt_version = backend.info().prompt_version;
                    head.compute = backend.identity().compute;
                    head.device = backend.identity().device;
                    backend.register_output_head(head.clone())?;
                    response.output_head = Some(head);
                } else if let Some(cal) = &response.scalar_calibration
                    && !backend.reasoning().is_thinking()
                {
                    let mut updated = cal.clone();
                    updated.model_fingerprint = backend.identity().fingerprint();
                    backend.upsert_calibration(updated.clone())?;
                    response.scalar_calibration = Some(updated);
                }
                response.applied = true;
            } else {
                backend.set_execution_mode(saved_mode);
                backend.set_prompt_detail(saved_detail);
                let _ = backend.set_code_rotation(saved_rotation);
                let _ = backend.set_policy(saved_policy);
                backend.restore_artifacts(
                    saved_calibrations,
                    saved_family_calibrations,
                    saved_head,
                );
            }
            Ok(response)
        }
        Err(err) => {
            backend.set_execution_mode(saved_mode);
            backend.set_prompt_detail(saved_detail);
            let _ = backend.set_code_rotation(saved_rotation);
            let _ = backend.set_policy(saved_policy);
            backend.restore_artifacts(saved_calibrations, saved_family_calibrations, saved_head);
            Err(err)
        }
    }
}

#[cfg(feature = "llama")]
fn collect_llama_examples_with_rotations(
    backend: &mut crate::llama::LlamaBackend,
    decision: &Decision,
    default_shared: &Option<serde_json::Value>,
    examples: &[TuneExample],
    passes: usize,
) -> Result<Vec<TuneExample>> {
    if passes <= 1 {
        backend.set_code_rotation(0)?;
        return Ok(populate_examples_with_backend(backend, decision, default_shared, examples)?.0);
    }
    let mut populated = Vec::with_capacity(examples.len());
    for example in examples {
        if example.raw_logits.is_some() && example.candidate_mass.is_some() {
            populated.push(example.clone());
            continue;
        }
        let req = DecisionRequest {
            shared: example.shared.clone().or_else(|| default_shared.clone()),
            state: example.state.clone(),
            decisions: vec![decision.clone()],
        };
        let mut pass_results = Vec::with_capacity(passes);
        for rot in 0..passes {
            backend.set_code_rotation(rot)?;
            let resp = backend.decide(&req)?;
            pass_results.push(
                resp.results
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Backend("missing decision result".into()))?,
            );
        }
        let mixture = crate::consensus::score_semantic_mixture(
            decision,
            &pass_results,
            &DecisionPolicy::default(),
        )?;
        let mut item = example.clone();
        item.raw_logits = Some(mixture.scores.iter().map(|s| s.raw_logit).collect());
        item.candidate_mass = Some(mixture.candidate_mass);
        populated.push(item);
    }
    Ok(populated)
}

/// Generic HTTP/stdio tuning handler for any `HttpDecisionBackend`.
pub(crate) fn tune_http_backend<B: crate::http::HttpDecisionBackend + ?Sized>(
    backend: &mut B,
    body: &[u8],
) -> Result<serde_json::Value> {
    let caps = backend.capabilities();
    if caps.get("evidence").and_then(serde_json::Value::as_str) == Some("selection_only") {
        return Err(Error::Invalid(
            "this backend returns selection_only evidence without logits for accuracy tuning"
                .into(),
        ));
    }
    let request: TuneRequest = serde_json::from_slice(body)
        .map_err(|e| Error::Invalid(format!("invalid tune request: {e}")))?;
    request.validate()?;

    let mut examples = Vec::with_capacity(request.examples.len());
    let mut last_backend_details: Option<BackendInfo> = None;
    for example in &request.examples {
        let (populated, details) =
            populate_http_example(backend, &request.decision, &request.shared, example)?;
        if details.is_some() {
            last_backend_details = details;
        }
        examples.push(populated);
    }
    let mut held_out = Vec::with_capacity(request.held_out.len());
    for example in &request.held_out {
        let (populated, details) =
            populate_http_example(backend, &request.decision, &request.shared, example)?;
        if details.is_some() {
            last_backend_details = details;
        }
        held_out.push(populated);
    }

    let mut identity = synthetic_identity_from_info(last_backend_details.as_ref());
    if last_backend_details.is_none() {
        if let Some(model) = caps
            .pointer("/backend/model")
            .and_then(serde_json::Value::as_str)
        {
            identity.weights_sha256 = crate::interoperability::digest(model.as_bytes());
        }
        if let Some(runtime) = caps
            .pointer("/backend/runtime")
            .and_then(serde_json::Value::as_str)
        {
            identity.loaded_runtime_sha256 = crate::interoperability::digest(runtime.as_bytes());
        }
    }

    let populated = TuneRequest {
        examples,
        held_out,
        ..request
    };
    let response = tune_records(&populated, &identity)?;
    let mut value = serde_json::to_value(response).map_err(|e| Error::Backend(e.to_string()))?;
    if let serde_json::Value::Object(map) = &mut value {
        map.insert("api_version".into(), serde_json::json!(1));
    }
    Ok(value)
}

fn populate_http_example<B: crate::http::HttpDecisionBackend + ?Sized>(
    backend: &mut B,
    decision: &Decision,
    default_shared: &Option<serde_json::Value>,
    example: &TuneExample,
) -> Result<(TuneExample, Option<BackendInfo>)> {
    if example.raw_logits.is_some() && example.candidate_mass.is_some() {
        return Ok((example.clone(), None));
    }
    let req = DecisionRequest {
        shared: example.shared.clone().or_else(|| default_shared.clone()),
        state: example.state.clone(),
        decisions: vec![decision.clone()],
    };
    let output = backend.decide_json(&req, &[])?;
    let details: Option<BackendInfo> = output
        .pointer("/backend/details")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let result = output
        .pointer("/results/0")
        .ok_or_else(|| Error::Backend("backend returned no decision result".into()))?;
    let scores = result
        .pointer("/evidence/scores")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            Error::Invalid("backend returned no model_scored candidate logits for tuning".into())
        })?;
    let options = decision.options();
    if scores.len() != options.len() {
        return Err(Error::Backend("candidate score count mismatch".into()));
    }
    let mut raw_logits = Vec::with_capacity(options.len());
    for opt in &options {
        let logit = scores
            .iter()
            .find(|s| s.get("id").and_then(serde_json::Value::as_str) == Some(&opt.id))
            .and_then(|s| s.get("raw_logit").and_then(serde_json::Value::as_f64))
            .ok_or_else(|| Error::Backend("missing candidate raw_logit".into()))?;
        raw_logits.push(logit);
    }
    let mass = result
        .pointer("/evidence/candidate_mass")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    let mut populated = example.clone();
    populated.raw_logits = Some(raw_logits);
    populated.candidate_mass = Some(mass);
    Ok((populated, details))
}
