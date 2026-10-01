//! Shared image request boundary for inference backends.

use crate::{DecisionBackend, DecisionRequest, DecisionResponse, Error, Result};

pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
/// Maximum ordered images in one local llama vision decision.
pub const MAX_VISION_IMAGES: usize = 8;

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

    /// Score one request against an ordered image set. This is one shared
    /// visual context, unlike `decide_vision_batch`'s independent requests.
    /// Empty image sets use text inference; single-image backends retain their
    /// existing implementation and explicitly reject additional images.
    fn decide_vision_images(
        &mut self,
        request: &DecisionRequest,
        images: &[&[u8]],
    ) -> Result<DecisionResponse> {
        match images {
            [] => self.decide(request),
            [image] => self.decide_vision(request, image),
            _ => Err(Error::Invalid(
                "this backend accepts at most one image per decision".into(),
            )),
        }
    }

    /// Evaluate independent image requests. Backends may override this to use
    /// native batching; the default preserves serial execution and metadata.
    fn decide_vision_batch(
        &mut self,
        requests: &[DecisionRequest],
        images: &[&[u8]],
    ) -> Result<Vec<DecisionResponse>> {
        if requests.len() != images.len() {
            return Err(Error::Invalid(
                "one image is required per vision request".into(),
            ));
        }
        requests
            .iter()
            .zip(images)
            .map(|(request, image)| self.decide_vision(request, image))
            .collect()
    }
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

    fn decide_vision_images(
        &mut self,
        request: &DecisionRequest,
        images: &[&[u8]],
    ) -> Result<DecisionResponse> {
        if images.is_empty() {
            self.decide(request)
        } else {
            crate::llama::LlamaBackend::decide_vision_images(self, request, images)
        }
    }

    fn decide_vision_batch(
        &mut self,
        requests: &[DecisionRequest],
        images: &[&[u8]],
    ) -> Result<Vec<DecisionResponse>> {
        crate::llama::LlamaBackend::decide_vision_batch(self, requests, images)
    }
}
