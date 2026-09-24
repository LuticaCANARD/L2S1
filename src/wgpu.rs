//! Native Gemma 4 decisions through Rust wgpu, using GGUF + mmproj files.
mod paired_gguf;

use crate::{
    BackendInfo, DecisionBackend, DecisionPolicy, DecisionRequest, DecisionResponse, Error,
    ExactEvidence, ExecutionMode, PromptDetail, PromptLayout, Result, VisionDecisionBackend,
    option_code, validate_image,
};
use image::imageops::FilterType;
use rullama_engine::api::{ChatMessage, ChatRole, Model};
use std::path::Path;

const IMAGE_ALIGN: u32 = 48;
const IMAGE_MAX_SIDE: u32 = 432;

pub struct WgpuBackend {
    model: Model,
    info: BackendInfo,
    policy: DecisionPolicy,
    layout: PromptLayout,
    detail: PromptDetail,
}

impl WgpuBackend {
    /// Both text and image decisions share this initialized model.
    pub fn load(model_path: &Path, projector_path: &Path, policy: DecisionPolicy) -> Result<Self> {
        Self::load_with_software_adapter(model_path, projector_path, policy, false)
    }

    /// Permit a CPU Vulkan adapter only for development and parity checks.
    pub fn load_with_software_adapter(
        model_path: &Path,
        projector_path: &Path,
        policy: DecisionPolicy,
        allow_software_adapter: bool,
    ) -> Result<Self> {
        policy.validate()?;
        let fetcher = pollster::block_on(paired_gguf::open(model_path, projector_path))?;
        let model = pollster::block_on(Model::load_streaming(fetcher)).map_err(backend_error)?;
        let adapter = model.forward().ctx().adapter.get_info();
        let software_adapter = adapter.device_type == wgpu::DeviceType::Cpu;
        if software_adapter && !allow_software_adapter {
            return Err(Error::Backend(format!(
                "wgpu selected the software adapter {}; a GPU adapter is required",
                adapter.name
            )));
        }
        if !model.has_vision_native() || model.image_sentinel_ids_native().is_none() {
            return Err(Error::Backend(
                "Gemma 4 vision tower or image tokens are missing".into(),
            ));
        }
        let info = BackendInfo {
            prompt_detail: PromptDetail::Minimal,
            code_rotation: 0,
            evidence_transfer: crate::EvidenceTransfer::Full,
            model_path: model_path.display().to_string(),
            vision_projector_path: Some(projector_path.display().to_string()),
            vision_projector_sha256: Some(crate::interoperability::file_digest(projector_path)?),
            lora_path: None,
            output_head_path: None,
            model_description: "Gemma 4 with paired vision projector".into(),
            model_architecture: "gemma4".into(),
            prompt_profile: "gemma4".into(),
            prompt_layout: PromptLayout::Legacy,
            prompt_version: "gemma4-wgpu-decision-v3".into(),
            runtime: "rullama-engine-wgpu".into(),
            execution_mode: ExecutionMode::Fresh,
            parallel_width: 1,
            compute: None,
            offload_requested: !software_adapter,
            offload_device: Some(adapter.name),
        };
        Ok(Self {
            model,
            info,
            policy,
            layout: PromptLayout::Legacy,
            detail: PromptDetail::Minimal,
        })
    }

    pub fn set_prompt_layout(&mut self, layout: PromptLayout) {
        self.layout = layout;
        self.info.prompt_layout = layout;
    }
    pub fn set_prompt_detail(&mut self, detail: PromptDetail) {
        self.detail = detail;
        self.info.prompt_detail = detail;
    }
    pub fn inspect(&self) -> &BackendInfo {
        &self.info
    }

    fn prompt(&self, state: &serde_json::Value, decision: &crate::Decision, image: bool) -> String {
        let data =
            crate::prompt::compile_prompt_with_detail(state, decision, self.layout, self.detail, 0)
                [1]
            .text
            .replace('<', "\\u003c");
        let user = if image {
            format!(
                "Image:\n<|image><image|>\nDecision data:\n{data}\nReply with only the matching option code."
            )
        } else {
            format!("Decision data:\n{data}\nReply with only the matching option code.")
        };
        self.model.render_chat_native(
            &[ChatMessage {
                role: ChatRole::User,
                content: format!("{}\n\n\n{user}", crate::prompt::decision_system(decision)),
            }],
            true,
        )
    }

    fn logits(&mut self, tokens: &[u32], soft: Option<&[f32]>) -> Result<Vec<f32>> {
        let embedding_width = self.model.forward().cfg().d_model as usize;
        let (image_begin, _) = self
            .model
            .image_sentinel_ids_native()
            .ok_or_else(|| Error::Backend("image sentinel IDs unavailable".into()))?;
        let soft_rows = soft.map(|v| v.len() / embedding_width).unwrap_or(0);
        if tokens.is_empty() || tokens.len() + soft_rows > self.model.max_context_native() as usize
        {
            return Err(Error::Invalid("prompt exceeds model context".into()));
        }
        if tokens
            .iter()
            .any(|&id| id >= self.model.vocab_size_native())
        {
            return Err(Error::Backend(
                "tokenizer produced out-of-vocabulary ID".into(),
            ));
        }
        if tokens.iter().filter(|&&id| id == image_begin).count() != usize::from(soft.is_some()) {
            return Err(Error::Backend(
                "image sentinel count differs from supplied image".into(),
            ));
        }
        self.model.reset_native();
        let mut logits = Vec::new();
        for &id in tokens {
            logits =
                pollster::block_on(self.model.forward_mut().step(id)).map_err(backend_error)?;
            if id == image_begin
                && let Some(soft) = soft
            {
                for row in soft.chunks_exact(embedding_width) {
                    logits = pollster::block_on(self.model.forward_mut().step_with_embedding(row))
                        .map_err(backend_error)?;
                }
            }
        }
        if logits.len() != self.model.vocab_size_native() as usize
            || logits.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Backend(
                "wgpu returned invalid vocabulary logits".into(),
            ));
        }
        Ok(logits)
    }

    fn decide_inner(
        &mut self,
        request: &DecisionRequest,
        image: Option<&[u8]>,
    ) -> Result<DecisionResponse> {
        request.validate()?;
        let soft = image.map(|bytes| self.encode_image(bytes)).transpose()?;
        let mut results = Vec::with_capacity(request.decisions.len());
        for decision in &request.decisions {
            let prompt = self.prompt(&request.state, decision, soft.is_some());
            let input = self.model.encode_tokens(&prompt);
            let options = decision.options();
            let mut paths = Vec::with_capacity(options.len());
            for i in 0..options.len() {
                let code = option_code(i, options.len())?;
                let combined = self.model.encode_tokens(&format!("{prompt}{code}"));
                if !combined.starts_with(&input) || combined.len() <= input.len() {
                    return Err(Error::Backend(format!(
                        "candidate {code} changes the answer token boundary"
                    )));
                }
                paths.push(
                    combined[input.len()..]
                        .iter()
                        .map(|&id| id as i32)
                        .collect::<Vec<_>>(),
                );
            }
            crate::codes::validate_code_paths(&paths)?;
            let longest = paths.iter().map(Vec::len).max().unwrap_or(0);
            let soft_rows = soft
                .as_ref()
                .map(|v| v.len() / self.model.forward().cfg().d_model as usize)
                .unwrap_or(0);
            let input_tokens = input.len() + soft_rows;
            if input_tokens + longest.saturating_sub(1) > self.model.max_context_native() as usize {
                return Err(Error::Invalid(
                    "prompt plus answer code exceeds model context".into(),
                ));
            }
            let result = if paths.iter().all(|path| path.len() == 1) {
                let logits = self.logits(&input, soft.as_deref())?;
                let ids: Vec<i32> = paths.iter().map(|path| path[0]).collect();
                ExactEvidence::from_logits(decision, &logits, &ids)?.score(
                    decision,
                    input_tokens,
                    &self.policy,
                )?
            } else {
                let mut evaluations = 0;
                let mut evaluated_tokens = 0;
                let scores = crate::codes::sequence_log_probabilities(&paths, |prefix| {
                    let mut tokens = input.clone();
                    tokens.extend(prefix.iter().map(|&id| id as u32));
                    evaluations += 1;
                    evaluated_tokens += tokens.len() + soft_rows;
                    self.logits(&tokens, soft.as_deref())
                })?;
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mass = max + scores.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
                if mass > 1e-8 {
                    return Err(Error::Backend(
                        "overlapping answer-code probability mass".into(),
                    ));
                }
                let representative: Vec<i32> = paths
                    .iter()
                    .map(|p| if p.len() == 1 { p[0] } else { -1 })
                    .collect();
                let mut result = crate::decision::score_candidate_logits(
                    decision,
                    &scores,
                    &representative,
                    input_tokens,
                    mass.exp().min(1.0),
                    &self.policy,
                )?;
                for (score, path) in result.scores.iter_mut().zip(paths) {
                    score.token_ids = path;
                }
                result.scoring_method = "code_sequence_conditional_softmax_v1".into();
                result.code_prefix_evaluations = evaluations;
                result.code_evaluated_tokens = evaluated_tokens;
                result
            };
            results.push(result);
        }
        Ok(DecisionResponse {
            backend: self.info.clone(),
            policy: self.policy.clone(),
            results,
        })
    }

    fn encode_image(&mut self, bytes: &[u8]) -> Result<Vec<f32>> {
        validate_image(bytes)?;
        let dimensions = image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| Error::Invalid(format!("invalid image: {e}")))?
            .into_dimensions()
            .map_err(|e| Error::Invalid(format!("invalid image: {e}")))?;
        if dimensions.0 == 0
            || dimensions.1 == 0
            || u64::from(dimensions.0) * u64::from(dimensions.1) > 25_000_000
        {
            return Err(Error::Invalid("image exceeds 25 megapixels".into()));
        }
        let image = image::load_from_memory(bytes)
            .map_err(|e| Error::Invalid(format!("invalid image: {e}")))?
            .to_rgb8();
        let (width, height) = image.dimensions();
        let max_side = width.max(height) as f64;
        let scale = f64::from(IMAGE_MAX_SIDE) / max_side;
        let scaled = |size: u32| {
            ((f64::from(size) * scale / f64::from(IMAGE_ALIGN)).round() as u32)
                .clamp(1, IMAGE_MAX_SIDE / IMAGE_ALIGN)
                * IMAGE_ALIGN
        };
        let (w, h) = (scaled(width), scaled(height));
        let resized = image::imageops::resize(&image, w, h, FilterType::Triangle);
        let plane = (w * h) as usize;
        let mut pixels = vec![0f32; plane * 3];
        for (i, pixel) in resized.pixels().enumerate() {
            for c in 0..3 {
                pixels[c * plane + i] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
        let soft = pollster::block_on(
            self.model
                .encode_image_native(&pixels, h as usize, w as usize, None),
        )
        .map_err(backend_error)?;
        let invalid = soft.iter().filter(|v| !v.is_finite()).count();
        if soft.is_empty()
            || soft.len() % self.model.forward().cfg().d_model as usize != 0
            || invalid != 0
        {
            return Err(Error::Backend(format!(
                "vision encoder returned invalid embeddings: len={}, nonfinite={invalid}, image={}x{}",
                soft.len(),
                w,
                h
            )));
        }
        Ok(soft)
    }
}

impl DecisionBackend for WgpuBackend {
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.decide_inner(request, None)
    }
}
impl VisionDecisionBackend for WgpuBackend {
    fn decide_vision(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse> {
        self.decide_inner(request, Some(image))
    }
}
fn backend_error(error: impl std::fmt::Display) -> Error {
    Error::Backend(error.to_string())
}
