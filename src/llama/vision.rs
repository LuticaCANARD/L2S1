//! Direct image input for one image and one request at a time.
use super::*;

const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;

impl LlamaBackend {
    /// Attach a vision projector compatible with this GGUF. Text decisions
    /// remain available on the same backend.
    pub fn load_vision_projector(&mut self, path: &Path) -> Result<()> {
        let path_text = path.to_string_lossy().into_owned();
        let path_c = CString::new(path_text.as_bytes())
            .map_err(|_| Error::Invalid("vision projector path contains NUL".into()))?;
        let hash = crate::interoperability::file_digest(path)?;
        let mut error = [0 as c_char; 1024];
        if !unsafe {
            sd_load_vision_projector(
                self.engine.as_ptr(),
                path_c.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        } {
            return Err(native_error(&error));
        }
        self.vision_projector_path = Some(path_text);
        self.vision_projector_sha256 = Some(hash);
        self.clear_preparation_cache();
        Ok(())
    }

    /// Score typed decisions from the original state and one still image.
    /// The image is encoded by libmtmd; no generated caption is substituted.
    pub fn decide_vision(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse> {
        unsafe { sd_clear(self.engine.as_ptr()) };
        let result = self.decide_vision_inner(request, image);
        unsafe { sd_clear(self.engine.as_ptr()) };
        result
    }

    fn decide_vision_inner(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse> {
        request.validate()?;
        if self.vision_projector_path.is_none() {
            return Err(Error::Invalid("vision projector is not loaded".into()));
        }
        if image.is_empty() || image.len() > MAX_IMAGE_BYTES {
            return Err(Error::Invalid("image must contain 1 byte to 8 MiB".into()));
        }
        if self.execution_mode != ExecutionMode::Fresh
            || self.evidence_transfer != EvidenceTransfer::Full
            || self.output_head.is_some()
            || !self.calibrations.is_empty()
            || self.collect_features
        {
            return Err(Error::Invalid(
                "vision decisions require fresh full-evidence execution without output heads, calibration or feature export".into(),
            ));
        }
        if request.decisions.iter().any(|d| d.options().len() > 26) {
            return Err(Error::Invalid(
                "vision decisions currently support at most 26 options per decision".into(),
            ));
        }
        let marker = unsafe { CStr::from_ptr(sd_vision_marker()) }
            .to_str()
            .map_err(|_| Error::Backend("invalid native vision marker".into()))?;
        if serde_json::to_string(request)
            .map_err(|e| Error::Invalid(e.to_string()))?
            .contains(marker)
        {
            return Err(Error::Invalid(
                "request text contains reserved vision marker".into(),
            ));
        }
        let mut results = Vec::with_capacity(request.decisions.len());
        for decision in &request.decisions {
            let (_, candidates) = self.prepare(&request.state, decision)?;
            let parts = match &self.chat_skeleton {
                Some(skeleton) => crate::prompt::compile_model_prompt_with_detail(
                    skeleton,
                    &request.state,
                    decision,
                    self.prompt_layout,
                    self.prompt_detail,
                    self.code_rotation,
                )?,
                None => compile_prompt_with_detail(
                    &request.state,
                    decision,
                    self.prompt_layout,
                    self.prompt_detail,
                    self.code_rotation,
                ),
            };
            // Keep media in the user-data segment; model control tokens remain
            // confined to the trusted prefix and suffix.
            let before = "Image:\n";
            let after = format!("\nDecision data:\n{}", parts[1].text);
            let size = unsafe { sd_vocab_size(self.engine.as_ptr()) };
            if size <= 0 {
                return Err(Error::Backend("invalid vocabulary size".into()));
            }
            self.logits_buffer.resize(size as usize, 0.0);
            let mut input_tokens = 0;
            let mut error = [0 as c_char; 1024];
            let start = Instant::now();
            let ok = unsafe {
                sd_forward_vision(
                    self.engine.as_ptr(),
                    parts[0].text.as_ptr().cast(),
                    parts[0].text.len(),
                    before.as_ptr().cast(),
                    before.len(),
                    image.as_ptr(),
                    image.len(),
                    after.as_ptr().cast(),
                    after.len(),
                    parts[2].text.as_ptr().cast(),
                    parts[2].text.len(),
                    self.logits_buffer.as_mut_ptr(),
                    self.logits_buffer.len(),
                    &mut input_tokens,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            self.timings.native_ms += start.elapsed().as_secs_f64() * 1000.0;
            if !ok {
                return Err(native_error(&error));
            }
            let mut scored = score_logits(
                decision,
                &self.logits_buffer,
                &candidates,
                input_tokens,
                &self.policy,
            )?;
            self.restore_code_metadata(&mut scored);
            results.push(scored);
            self.timings.decisions += 1;
        }
        let mut info = self.info_for_request(request);
        info.runtime = "local-libllama-mtmd".into();
        info.prompt_version.push_str("/vision-image-v1");
        Ok(DecisionResponse {
            backend: info,
            policy: self.policy.clone(),
            results,
        })
    }
}
