//! Typed decision contracts and scoring, independent of the inference backend.
mod decision;
mod prompt;
pub use decision::*;
pub use prompt::{MODEL_PROMPT_VERSION, PROMPT_VERSION, PromptPart, PromptProfile, compile_prompt};
#[cfg(feature = "llama")]
pub mod llama;
