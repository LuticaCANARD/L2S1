use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    #[default]
    Direct,
    Thinking,
}

/// Request reasoning contract. Native batching supports direct scoring only;
/// unsupported reasoning modes must fail explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReasoningOptions {
    pub mode: ReasoningMode,
    pub max_tokens: usize,
}
impl Default for ReasoningOptions {
    fn default() -> Self {
        Self {
            mode: ReasoningMode::Direct,
            max_tokens: 128,
        }
    }
}
impl ReasoningOptions {
    pub fn validate(self) -> Result<()> {
        if !(1..=1024).contains(&self.max_tokens) {
            return Err(Error::Invalid(
                "max reasoning tokens must be 1..=1024".into(),
            ));
        }
        Ok(())
    }
    pub fn is_thinking(self) -> bool {
        self.mode == ReasoningMode::Thinking
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_direct_and_bounded_thinking_contract() {
        let default: ReasoningOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(default, ReasoningOptions::default());
        assert!(!default.is_thinking());
        for max_tokens in [0, 1025, usize::MAX] {
            assert!(
                ReasoningOptions {
                    mode: ReasoningMode::Thinking,
                    max_tokens
                }
                .validate()
                .is_err()
            );
        }
        for max_tokens in [1, 128, 1024] {
            assert!(
                ReasoningOptions {
                    mode: ReasoningMode::Thinking,
                    max_tokens
                }
                .validate()
                .is_ok()
            );
        }
        assert!(serde_json::from_str::<ReasoningOptions>("{\"mode\":\"auto\"}").is_err());
        assert!(serde_json::from_str::<ReasoningOptions>("{\"trace\":true}").is_err());
    }
}
