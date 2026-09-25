use crate::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
#[cfg(any(feature = "llama", feature = "wgpu"))]
pub(crate) fn file_digest(path: &std::path::Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| Error::Backend(e.to_string()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| Error::Backend(e.to_string()))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Verified at registration. GGUF checksum includes tokenizer and embedded template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    #[serde(default, skip_serializing_if = "EvidenceTransfer::is_full")]
    pub evidence_transfer: EvidenceTransfer,
    pub weights_sha256: String,
    pub template_sha256: String,
    pub prompt_profile: String,
    pub prompt_version: String,
    pub runtime_build_sha256: String,
    pub loaded_runtime_sha256: String,
    pub adapter_sha256: Option<String>,
    pub head_sha256: Option<String>,
    pub device: String,
    pub compute: ComputeOptions,
    pub execution_mode: ExecutionMode,
    pub parallel_width: usize,
}
impl ModelIdentity {
    pub fn fingerprint(&self) -> String {
        digest(&serde_json::to_vec(self).expect("serializable identity"))
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ModelCapabilities {
    pub architecture: String,
    pub vocabulary_size: usize,
    pub configured_context: usize,
    pub training_context: u32,
    pub recurrent_or_hybrid: bool,
    pub exact_full_vocabulary_logits: bool,
    pub execution_modes: Vec<ExecutionMode>,
    pub prefix_reuse_fallback: Option<String>,
    pub hidden_features: bool,
    pub snapshot_limit_bytes: usize,
    pub evidence_status: &'static str,
}
#[derive(Debug, Serialize)]
pub struct ModelInspection {
    pub identity: ModelIdentity,
    pub capabilities: ModelCapabilities,
}
#[derive(Debug, Serialize)]
pub struct PreparedDecisionReport {
    pub decision_id: String,
    pub input_tokens: usize,
    pub prompt_tokens_sha256: String,
    pub candidate_token_ids: Vec<i32>,
    /// Populated instead of scalar token IDs for multi-letter answer codes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidate_token_sequences: Vec<Vec<i32>>,
    pub option_ids: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct PreflightReport {
    pub model: ModelInspection,
    pub decisions: Vec<PreparedDecisionReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    InvalidRequest,
    UnsupportedCapability,
    IncompatibleArtifact,
    ContextExceeded,
    InvalidEvidence,
    BackendFailure,
}
#[derive(Debug, Serialize)]
pub struct DecisionFailure {
    pub kind: FailureKind,
    pub stage: String,
    pub decision_id: Option<String>,
    pub message: String,
}
impl DecisionFailure {
    pub(crate) fn new(
        kind: FailureKind,
        stage: &str,
        id: Option<&str>,
        error: impl std::fmt::Display,
    ) -> Self {
        Self {
            kind,
            stage: stage.into(),
            decision_id: id.map(str::to_owned),
            message: error.to_string(),
        }
    }
}
impl std::fmt::Display for DecisionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for DecisionFailure {}
#[derive(Debug, Default, Clone, Serialize)]
pub struct StateRestoreMetrics {
    pub snapshot_bytes: usize,
    pub save_ms: f64,
    pub restore_ms: f64,
    pub prefill_ms: f64,
    pub suffix_ms: f64,
    pub restores: usize,
    pub fallback_reason: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct ExecutionDiagnostic {
    pub decision_id: String,
    pub prompt_tokens_sha256: String,
    pub requested_mode: ExecutionMode,
    pub effective_mode: ExecutionMode,
    pub reused_prefix_tokens: usize,
    pub fallback_reason: Option<String>,
    pub evidence_kind: &'static str,
    pub calibration_id: Option<String>,
}
#[cfg(feature = "llama")]
#[derive(Debug, Serialize)]
pub struct DiagnosticResponse {
    pub response: DecisionResponse,
    pub model: ModelIdentity,
    pub decisions: Vec<ExecutionDiagnostic>,
    pub timings: crate::llama::InferenceTimings,
    pub state_restore: StateRestoreMetrics,
}

impl From<Error> for DecisionFailure {
    fn from(error: Error) -> Self {
        Self::new(
            match error {
                Error::Invalid(_) => FailureKind::InvalidRequest,
                Error::Backend(_) | Error::Upstream(_) => FailureKind::BackendFailure,
            },
            "prepare",
            None,
            error,
        )
    }
}
