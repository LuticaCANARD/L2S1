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
