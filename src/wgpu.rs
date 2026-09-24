//! Optional GGUF text inference through FlareLLM's native wgpu compute path.
use crate::{
    BackendInfo, DecisionBackend, DecisionPolicy, DecisionRequest, DecisionResponse, Error,
    ExactEvidence, ExecutionMode, PromptDetail, PromptLayout, Result, option_code,
};
use flare_core::model::Model;
use flare_gpu::WebGpuBackend;
use flare_loader::{
    gguf::{GgufFile, MetadataValue},
    weights::load_model_weights,
};
use std::{fs::File, io::BufReader, path::Path};
use tokenizers::Tokenizer;

pub struct WgpuBackend {
    model: Model,
    tokenizer: Tokenizer,
    info: BackendInfo,
    policy: DecisionPolicy,
    layout: PromptLayout,
    detail: PromptDetail,
}

impl WgpuBackend {
    /// Load a ChatML GGUF with the matching Hugging Face tokenizer.json.
    /// GPU initialization and weight upload are mandatory: there is no CPU fallback.
    pub fn load(model_path: &Path, tokenizer_path: &Path, policy: DecisionPolicy) -> Result<Self> {
        policy.validate()?;
        let mut reader = BufReader::new(File::open(model_path).map_err(backend_error)?);
        let gguf = GgufFile::parse_header(&mut reader).map_err(backend_error)?;
        let template = gguf
            .metadata
            .get("tokenizer.chat_template")
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| Error::Backend("GGUF has no chat template".into()))?;
        if !template.contains("<|im_start|>") || !template.contains("<|im_end|>") {
            return Err(Error::Backend(
                "wgpu backend currently requires a ChatML GGUF".into(),
            ));
        }
        let config = gguf.to_model_config().map_err(backend_error)?;
        let weights = load_model_weights(&gguf, &mut reader).map_err(backend_error)?;
        let mut model = Model::new(config.clone(), weights);

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(backend_error)?;
        if tokenizer.get_vocab_size(true) != config.vocab_size {
            return Err(Error::Backend(format!(
                "tokenizer vocabulary ({}) differs from GGUF model ({})",
                tokenizer.get_vocab_size(true),
                config.vocab_size
            )));
        }
        if let Some(MetadataValue::Array(tokens)) = gguf.metadata.get("tokenizer.ggml.tokens") {
            for (id, token) in tokens.iter().enumerate() {
                let Some(text) = token.as_str() else {
                    return Err(Error::Backend(
                        "GGUF vocabulary contains a non-string token".into(),
                    ));
                };
                if tokenizer.token_to_id(text) != Some(id as u32) {
                    return Err(Error::Backend(format!(
                        "tokenizer and GGUF disagree on token ID {id}"
                    )));
                }
            }
        }
        for special in ["<|im_start|>", "<|im_end|>"] {
            let tokenizer_id = tokenizer.token_to_id(special);
            let gguf_id =
                gguf.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|tokens| match tokens {
                        MetadataValue::Array(tokens) => tokens
                            .iter()
                            .position(|token| token.as_str() == Some(special))
                            .map(|i| i as u32),
                        _ => None,
                    });
            if tokenizer_id.is_none() || tokenizer_id != gguf_id {
                return Err(Error::Backend(format!(
                    "tokenizer and GGUF disagree on {special} token ID"
                )));
            }
        }

        let mut raw_reader = BufReader::new(File::open(model_path).map_err(backend_error)?);
        let raw_gguf = GgufFile::parse_header(&mut raw_reader).map_err(backend_error)?;
        let raw_layers = (0..config.num_layers)
            .map(|i| {
                raw_gguf
                    .load_raw_layer_weights(&mut raw_reader, i)
                    .map_err(backend_error)?
                    .ok_or_else(|| Error::Backend(format!("GPU raw weights missing for layer {i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        model.set_raw_weights(raw_layers);

        let gpu = pollster::block_on(WebGpuBackend::new()).map_err(backend_error)?;
        model.set_backend(Box::new(gpu));
        model.upload_weights_to_gpu();
        if !model.backend().has_gpu_weights() || !model.backend().has_gpu_kv_cache() {
            return Err(Error::Backend(
                "wgpu weights or KV cache were not uploaded".into(),
            ));
        }

        let info = BackendInfo {
            prompt_detail: PromptDetail::Minimal,
            code_rotation: 0,
            evidence_transfer: crate::EvidenceTransfer::Full,
            model_path: model_path.display().to_string(),
            vision_projector_path: None,
            vision_projector_sha256: None,
            lora_path: None,
            output_head_path: None,
            model_description: format!("{:?}, {} layers", config.architecture, config.num_layers),
            model_architecture: format!("{:?}", config.architecture).to_lowercase(),
            prompt_profile: "chatml".into(),
            prompt_layout: PromptLayout::Legacy,
            prompt_version: "chatml-wgpu-decision-v1".into(),
            runtime: "flarellm-wgpu-24".into(),
            execution_mode: ExecutionMode::Fresh,
            parallel_width: 1,
            compute: None,
            offload_requested: true,
            offload_device: Some("wgpu".into()),
        };
        Ok(Self {
            model,
            tokenizer,
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

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(backend_error)
    }

    fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() || tokens.len() > self.model.config().max_seq_len {
            return Err(Error::Invalid(
                "prompt exceeds model context; truncation is disabled".into(),
            ));
        }
        if tokens
            .iter()
            .any(|&id| id as usize >= self.model.config().vocab_size)
        {
            return Err(Error::Backend(
                "tokenizer emitted an out-of-vocabulary ID".into(),
            ));
        }
        self.model.reset();
        let logits = self.model.forward_prefill(tokens).data().to_vec();
        if logits.len() != self.model.config().vocab_size || logits.iter().any(|v| !v.is_finite()) {
            return Err(Error::Backend(
                "wgpu produced invalid vocabulary logits".into(),
            ));
        }
        Ok(logits)
    }
}

impl DecisionBackend for WgpuBackend {
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        request.validate()?;
        let mut results = Vec::with_capacity(request.decisions.len());
        for decision in &request.decisions {
            let prompt = crate::prompt::compile_chatml_prompt_with_detail(
                &request.state,
                decision,
                self.layout,
                self.detail,
                0,
            );
            let input = self.encode(&prompt)?;
            let options = decision.options();
            let mut paths = Vec::with_capacity(options.len());
            for i in 0..options.len() {
                let code = option_code(i, options.len())?;
                let combined = self.encode(&format!("{prompt}{code}"))?;
                if !combined.starts_with(&input) || combined.len() <= input.len() {
                    return Err(Error::Backend(format!(
                        "candidate {code} changes the assistant token boundary"
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
            let longest = paths.iter().map(|p| p.len()).max().unwrap_or(0);
            if input.len() + longest.saturating_sub(1) > self.model.config().max_seq_len {
                return Err(Error::Invalid(
                    "prompt plus answer-code prefix exceeds model context".into(),
                ));
            }

            let result = if paths.iter().all(|path| path.len() == 1) {
                let logits = self.logits(&input)?;
                let ids: Vec<i32> = paths.iter().map(|path| path[0]).collect();
                ExactEvidence::from_logits(decision, &logits, &ids)?.score(
                    decision,
                    input.len(),
                    &self.policy,
                )?
            } else {
                let mut evaluations = 0;
                let mut evaluated_tokens = 0;
                let scores = crate::codes::sequence_log_probabilities(&paths, |prefix| {
                    let mut tokens = input.clone();
                    tokens.extend(prefix.iter().map(|&id| id as u32));
                    evaluations += 1;
                    evaluated_tokens += tokens.len();
                    self.logits(&tokens)
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
                    input.len(),
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
}

fn backend_error(error: impl std::fmt::Display) -> Error {
    Error::Backend(error.to_string())
}
