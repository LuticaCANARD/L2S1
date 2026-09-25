use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("inference failed: {0}")]
    Backend(String),
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("upstream provider failed: {0}")]
    Upstream(String),
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
    /// Construct an ordinal decision from typed levels. Request validation still
    /// checks IDs, criteria, and strictly increasing finite values.
    pub fn ordinal(
        id: impl Into<String>,
        instruction: impl Into<String>,
        levels: impl IntoIterator<Item = Level>,
    ) -> Self {
        Self {
            id: id.into(),
            instruction: instruction.into(),
            kind: DecisionKind::Ordinal {
                levels: levels.into_iter().collect(),
            },
        }
    }

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
            if options.len() < 2 {
                return Err(Error::Invalid(
                    "each decision requires at least 2 options".into(),
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
    /// Full continuation for sequence scoring; token_id is -1 for multiple tokens.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<i32>,
    /// A native logit for single-token scoring; joint log probability for sequences.
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
    /// Exact prefix tokens reused in this forward pass; evaluated = input - reused.
    #[serde(default)]
    pub reused_prefix_tokens: usize,
    /// Native prefix evaluations for complete multi-letter code likelihoods.
    #[serde(default, skip_serializing_if = "zero_rotation")]
    pub code_prefix_evaluations: usize,
    /// Tokens actually decoded across those evaluations, excluding KV reuse.
    #[serde(default, skip_serializing_if = "zero_rotation")]
    pub code_evaluated_tokens: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    #[serde(default, skip_serializing_if = "crate::PromptDetail::is_minimal")]
    pub prompt_detail: crate::PromptDetail,
    #[serde(default, skip_serializing_if = "zero_rotation")]
    pub code_rotation: usize,
    #[serde(default, skip_serializing_if = "crate::EvidenceTransfer::is_full")]
    pub evidence_transfer: crate::EvidenceTransfer,
    pub model_path: String,
    /// Projector used for direct image input, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_projector_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_projector_sha256: Option<String>,
    /// Explicit adapter at scale 1; absent for the unchanged base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,
    /// Task-scoped output head; inspect each result's scoring_method for usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_head_path: Option<String>,
    pub model_description: String,
    #[serde(default)]
    pub model_architecture: String,
    #[serde(default)]
    pub prompt_profile: String,
    #[serde(default)]
    pub prompt_layout: crate::PromptLayout,
    pub prompt_version: String,
    pub runtime: String,
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    /// Configured maximum questions per parallel wave; one for serial modes.
    #[serde(default = "serial_width")]
    pub parallel_width: usize,
    /// Opt-in context reservation based on the current parallel wave's token counts.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub parallel_context_dynamic: bool,
    /// Currently allocated padded KV context capacity in dynamic parallel mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_context_tokens: Option<u32>,
    /// Requested compute settings; absent in historical responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<ComputeOptions>,
    pub offload_requested: bool,
    pub offload_device: Option<String>,
}

fn zero_rotation(value: &usize) -> bool {
    *value == 0
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

fn serial_width() -> usize {
    1
}

pub(crate) fn bool_is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttention {
    #[default]
    Off,
    Auto,
    On,
}

/// Model-file loading strategy; separate from tensor placement and quantization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadMode {
    /// Preserve llama.cpp's device-dependent default.
    #[default]
    Auto,
    /// Read weights into backend buffers without mapping the entire GGUF.
    Read,
}

impl ModelLoadMode {
    fn is_auto(&self) -> bool {
        *self == Self::Auto
    }
}

/// Explicit opt-in compute tuning. FlashAttention records the requested mode;
/// consult llama.cpp logs for actual kernel support on the selected device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeOptions {
    pub context: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub threads: i32,
    pub flash_attention: FlashAttention,
    /// None preserves the device default; Some(n) puts at most n layers on the GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,
    /// Keep expert weights of the first n MoE layers in CPU RAM for GPU loading.
    #[serde(default, skip_serializing_if = "zero_u32")]
    pub cpu_moe_layers: u32,
    #[serde(default, skip_serializing_if = "ModelLoadMode::is_auto")]
    pub model_load_mode: ModelLoadMode,
}

fn zero_u32(value: &u32) -> bool {
    *value == 0
}

impl Default for ComputeOptions {
    fn default() -> Self {
        Self {
            context: 2048,
            batch: 256,
            ubatch: 256,
            threads: 4,
            flash_attention: FlashAttention::Off,
            gpu_layers: None,
            cpu_moe_layers: 0,
            model_load_mode: ModelLoadMode::Auto,
        }
    }
}

impl ComputeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.context == 0
            || self.context > i32::MAX as u32
            || self.batch == 0
            || self.batch > self.context
            || self.ubatch == 0
            || self.ubatch > self.batch
            || self.threads <= 0
            || self.gpu_layers.is_some_and(|n| n > i32::MAX as u32)
            || self.cpu_moe_layers > i32::MAX as u32
        {
            return Err(Error::Invalid(
                "require 0 < ubatch <= batch <= context <= i32::MAX, threads > 0, and placement counts <= i32::MAX".into(),
            ));
        }
        Ok(())
    }
    pub fn validate_device(&self, gpu: bool) -> Result<()> {
        self.validate()?;
        if !gpu && (self.gpu_layers.is_some_and(|n| n > 0) || self.cpu_moe_layers > 0) {
            return Err(Error::Invalid(
                "GPU layer placement and CPU MoE splitting require a GPU device".into(),
            ));
        }
        Ok(())
    }
}

/// Optimized modes are opt-in because batch shapes can affect model scores.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    #[default]
    Fresh,
    PrefixReuse,
    /// Experimental independent sequence batching with shared-prefix prefill.
    Parallel,
    /// Request-local whole-sequence snapshot restoration for a shared prefix.
    /// Supports recurrent and hybrid models when their sequence state can be saved.
    StateRestore,
}

/// Scores are conditional on the supplied candidate set. No calibration is implied.
pub fn score_logits(
    decision: &Decision,
    logits: &[f32],
    candidate_tokens: &[i32],
    input_tokens: usize,
    policy: &DecisionPolicy,
) -> Result<DecisionResult> {
    crate::ExactEvidence::from_logits(decision, logits, candidate_tokens)?.score(
        decision,
        input_tokens,
        policy,
    )
}

/// A learned head has no full-vocabulary mass. Retain the base LM mass as a
/// separate compatibility gate rather than fabricating one for the new head.
pub(crate) fn score_candidate_logits(
    decision: &Decision,
    selected_logits: &[f64],
    candidate_tokens: &[i32],
    input_tokens: usize,
    mass: f64,
    policy: &DecisionPolicy,
) -> Result<DecisionResult> {
    policy.validate()?;
    let options = decision.options();
    if options.len() != selected_logits.len()
        || options.len() != candidate_tokens.len()
        || options.len() < 2
        || selected_logits.iter().any(|x| !x.is_finite())
        || !mass.is_finite()
        || !(0.0..=1.0).contains(&mass)
    {
        return Err(Error::Backend(
            "invalid candidate scores or base mass".into(),
        ));
    }
    let max = selected_logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let candidate_lse = max
        + selected_logits
            .iter()
            .map(|v| (v - max).exp())
            .sum::<f64>()
            .ln();
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
                code: crate::option_code(i, options.len()).expect("validated option count"),
                token_id: candidate_tokens[i],
                token_ids: Vec::new(),
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
        reused_prefix_tokens: 0,
        code_prefix_evaluations: 0,
        code_evaluated_tokens: 0,
        truncated: false,
    })
}
