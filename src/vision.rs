//! Shared image request boundary for inference backends.

use crate::{DecisionBackend, DecisionRequest, DecisionResponse, Error, Result};

pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;

pub fn validate_image(image: &[u8]) -> Result<()> {
    if image.is_empty() || image.len() > MAX_IMAGE_BYTES {
        return Err(Error::Invalid("image must contain 1 byte to 8 MiB".into()));
    }
    Ok(())
}

/// Backends that score decisions directly from the supplied image bytes.
/// The HTTP layer depends on this contract rather than a concrete runtime.
pub trait VisionDecisionBackend: DecisionBackend {
    fn decide_vision(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse>;
}

#[cfg(feature = "llama")]
impl VisionDecisionBackend for crate::llama::LlamaBackend {
    fn decide_vision(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse> {
        crate::llama::LlamaBackend::decide_vision(self, request, image)
    }
}
