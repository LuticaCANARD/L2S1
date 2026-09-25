//! L2S1 — LLM to System 1.
//!
//! Typed decision contracts and scoring, independent of the inference backend.
mod codes;
mod decision;
pub use codes::{option_code, option_code_width};
mod output_head;
mod prompt;
pub use decision::*;
pub use output_head::*;
pub use prompt::{
    GPT_OSS_FINAL_PROMPT_VERSION, MODEL_PROMPT_VERSION, PROMPT_VERSION, PromptDetail, PromptLayout,
    PromptPart, PromptProfile, STATE_FIRST_GPT_OSS_PROMPT_VERSION,
    STATE_FIRST_MODEL_PROMPT_VERSION, STATE_FIRST_PROMPT_VERSION, compile_prompt,
    compile_prompt_with_detail, compile_prompt_with_layout,
};
pub mod http;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "openrouter")]
pub mod openrouter;
mod vision;
#[cfg(feature = "wgpu")]
pub mod wgpu;
pub use vision::{MAX_IMAGE_BYTES, VisionDecisionBackend, validate_image};

mod calibration;
mod consensus;
mod evidence;
mod interoperability;
mod optimization;
mod worker;
pub use calibration::*;
pub use consensus::*;
pub use evidence::*;
pub use interoperability::*;
pub use optimization::*;
pub use worker::*;
