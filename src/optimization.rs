use serde::{Deserialize, Serialize};

/// Full vocabulary computation is retained in both modes. Compact only avoids
/// transferring/copying the entire native host buffer into Rust.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTransfer {
    #[default]
    Full,
    Compact,
}
impl EvidenceTransfer {
    pub fn is_full(&self) -> bool {
        *self == Self::Full
    }
}

/// Opt-in preparation memoization. The byte budget is shared between complete
/// text prompt tokens (one half), vision prompt parts (one quarter), and
/// answer-boundary mappings (one quarter).
/// Limits bound retained cache entries, not process RSS or temporary allocations.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreparationCacheConfig {
    /// Maximum entries in each of the three caches.
    pub max_entries: usize,
    pub max_bytes: usize,
}

/// Rounding of exact shared prefixes in `parallel` waves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ParallelPrefixAlignment {
    /// Share only complete prefill batches, so shared KV is computed with the
    /// same decode boundaries as serial execution. Up to `batch - 1` common
    /// tokens per shared segment are evaluated again by each question.
    #[default]
    Batch,
    /// Share every common token. Faster for long common prefixes, but shared
    /// KV uses different decode boundaries and scores can differ slightly
    /// from `batch`; validate decisions before relying on it.
    Token,
}
impl ParallelPrefixAlignment {
    pub fn is_batch(&self) -> bool {
        *self == Self::Batch
    }
}

/// Assignment of questions to bounded `parallel` waves. Results always keep
/// request order; only wave membership changes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ParallelWaveOrder {
    /// Consecutive questions in request order (the behavior before 0.2.0).
    #[default]
    Request,
    /// Sort prepared prompts by exact token sequence before forming waves, so
    /// questions with the longest common prefixes share a wave.
    Prefix,
}
impl ParallelWaveOrder {
    pub fn is_request(&self) -> bool {
        *self == Self::Request
    }
}
