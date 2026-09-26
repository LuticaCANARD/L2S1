//! HTTP wire contract and backend-specific evidence mapping.
#[cfg(any(feature = "llama", feature = "wgpu", feature = "openrouter", test))]
use super::{MAX_BODY, MAX_CONNECTIONS, MAX_INFLIGHT_BODY_BYTES, QUEUE_DEPTH};
use super::{MAX_DECISIONS, MAX_MEDIA};
#[cfg(any(feature = "llama", feature = "wgpu", test))]
use crate::VisionDecisionBackend;
use crate::{Decision, DecisionRequest, Error};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

/// One backend owns the inference call. The wire format is independent of its evidence type.
pub trait HttpDecisionBackend {
    fn capabilities(&self) -> Value;
    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> crate::Result<Value>;

    /// Cross-request native batching. No serial default or replay is permitted.
    fn decide_native_batch_json(
        &mut self,
        _requests: &[DecisionRequest],
        _images: &[Vec<&[u8]>],
    ) -> crate::Result<Vec<Value>> {
        Err(Error::Invalid(
            "batch_unsupported: this backend does not support native batching".into(),
        ))
    }

    /// Evaluate validated media groups. The default keeps backend calls serial;
    /// runtimes with native image batching can override this boundary.
    fn decide_json_batch(
        &mut self,
        requests: &[DecisionRequest],
        images: &[Vec<&[u8]>],
    ) -> crate::Result<Vec<Value>> {
        if requests.len() != images.len() {
            return Err(Error::Invalid(
                "media group count does not match requests".into(),
            ));
        }
        requests
            .iter()
            .zip(images)
            .map(|(request, images)| self.decide_json(request, images))
            .collect()
    }

    /// Optional request-local reasoning. Unsupported engines must refuse it.
    fn decide_json_batch_with_reasoning(
        &mut self,
        requests: &[DecisionRequest],
        images: &[Vec<&[u8]>],
        reasoning: Option<&crate::ReasoningOptions>,
    ) -> crate::Result<Vec<Value>> {
        if let Some(options) = reasoning {
            options.validate()?;
            if options.mode == crate::ReasoningMode::Thinking {
                return Err(Error::Invalid(
                    "thinking mode is not supported by this backend".into(),
                ));
            }
        }
        self.decide_json_batch(requests, images)
    }
}

#[cfg(any(feature = "llama", feature = "wgpu", test))]
fn local_capabilities(info: &crate::BackendInfo) -> Value {
    let (formats, format_support) = if info.runtime == "rullama-engine-wgpu" {
        (vec!["png", "jpeg", "webp"], "enumerated")
    } else {
        (vec![], "runtime_dependent")
    };
    json!({"api_version":1,"backend":{"runtime":info.runtime,"model":info.model_path},
        "decision_types":["binary","choice","ordinal"],"evidence":"model_scored",
        "media":{"image":{"supported":info.vision_projector_path.is_some(),"max_per_decision":1,
            "max_bytes_each":crate::MAX_IMAGE_BYTES,"formats":formats,"format_support":format_support}},
        "limits":{"max_body_bytes":MAX_BODY,"max_media":MAX_MEDIA,"max_decisions":MAX_DECISIONS,"max_connections":MAX_CONNECTIONS,"queued_inference_requests":QUEUE_DEPTH,"max_inflight_body_bytes":MAX_INFLIGHT_BODY_BYTES}})
}

#[cfg(any(feature = "llama", feature = "wgpu", test))]
fn local_decide_json<B: VisionDecisionBackend>(
    backend: &mut B,
    request: &DecisionRequest,
    images: &[&[u8]],
) -> crate::Result<Value> {
    let response = match images {
        [] => backend.decide(request)?,
        [image] => backend.decide_vision(request, image)?,
        _ => {
            return Err(Error::Invalid(
                "this backend accepts at most one image per decision".into(),
            ));
        }
    };
    local_response_json(response)
}

#[cfg(any(feature = "llama", feature = "wgpu", test))]
fn local_response_json(response: crate::DecisionResponse) -> crate::Result<Value> {
    let results = response.results.into_iter().map(|result| {
        let (value, estimate) = match result.value {
            crate::DecisionValue::Binary { value, p_true } => (json!({"type":"binary","value":value}), json!({"p_true":p_true})),
            crate::DecisionValue::Choice { selected } => (json!({"type":"choice","selected":selected}), json!({})),
            crate::DecisionValue::Ordinal { selected, expected_value } => (json!({"type":"ordinal","selected":selected}), json!({"expected_value":expected_value})),
        };
        let status = if result.abstention_reasons.is_empty() { "selected" } else { "abstained" };
        let evidence = json!({"type":"model_scored","scores":result.scores,
            "candidate_mass":result.candidate_mass,"top_option_probability":result.top_option_probability,
            "entropy_confidence":result.entropy_confidence,"scoring_method":result.scoring_method,
            "calibration_id":result.calibration_id,"truncated":result.truncated,
            "code_prefix_evaluations":result.code_prefix_evaluations,"code_evaluated_tokens":result.code_evaluated_tokens,
            "estimate":estimate});
        let output = json!({"id":result.id,"value":value,"status":status,"abstention_reasons":result.abstention_reasons,
            "evidence":evidence,"usage":{"input_tokens":result.input_tokens,"reused_prefix_tokens":result.reused_prefix_tokens}});
        output
    }).collect::<Vec<_>>();
    Ok(
        json!({"backend":{"runtime":response.backend.runtime,"model":response.backend.model_path,
        "details":response.backend},"policy":response.policy,"results":results}),
    )
}

#[cfg(feature = "llama")]
impl HttpDecisionBackend for crate::llama::LlamaBackend {
    fn capabilities(&self) -> Value {
        let info = self.info();
        let mut capabilities = local_capabilities(&info);
        capabilities["media"]["image"]["parallel"] = json!({
            "supported": info.vision_projector_path.is_some(),
            "enabled": info.execution_mode == crate::ExecutionMode::Parallel,
            "max_decisions_per_wave": info.parallel_width,
            "max_options_per_decision": 26,
            "isolated_sequences": true,
            "projector_encoding": if info.vision_projector_reuse {
                "independent_unique_chunks_with_reuse"
            } else {
                "batched_when_compatible"
            },
        });
        capabilities["reasoning"] = json!({"modes":["direct"],"thinking_supported":false});
        capabilities["request_policy"] = json!({"supported":true,
            "target_error_rate":"maps to a model-score threshold; not a guaranteed correctness error rate"});
        capabilities["batch"] = json!({"supported":true,"enabled":info.execution_mode == crate::ExecutionMode::Parallel,
            "execution":"native_parallel","max_requests":super::MAX_BATCH_REQUESTS,
            "max_decisions":MAX_DECISIONS,"max_decisions_per_wave":info.parallel_width,
            "text":true,"image":info.vision_projector_path.is_some(),"mixed_media":false,"reasoning_modes":["direct"]});
        capabilities
    }

    fn decide_native_batch_json(
        &mut self,
        requests: &[DecisionRequest],
        images: &[Vec<&[u8]>],
    ) -> crate::Result<Vec<Value>> {
        if self.info().execution_mode != crate::ExecutionMode::Parallel {
            return Err(Error::Invalid(
                "batch_not_enabled: load with execution_mode parallel for native batching".into(),
            ));
        }
        if requests.len() != images.len() {
            return Err(Error::Invalid(
                "media group count does not match requests".into(),
            ));
        }
        if images.iter().all(Vec::is_empty) {
            return self
                .decide_batch(requests)?
                .into_iter()
                .map(local_response_json)
                .collect();
        }
        if images.iter().all(|images| images.len() == 1) {
            // Existing native projector/decoder path, including execution counters.
            return self.decide_json_batch(requests, images);
        }
        Err(Error::Invalid("native batches require either all text or one image per decision; mixed media is unsupported".into()))
    }

    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> crate::Result<Value> {
        local_decide_json(self, request, images)
    }
    fn decide_json_batch(
        &mut self,
        requests: &[DecisionRequest],
        images: &[Vec<&[u8]>],
    ) -> crate::Result<Vec<Value>> {
        if requests.len() != images.len() {
            return Err(Error::Invalid(
                "media group count does not match requests".into(),
            ));
        }
        if self.info().execution_mode == crate::ExecutionMode::Parallel
            && !requests.is_empty()
            && images.iter().all(|images| images.len() == 1)
        {
            let images = images.iter().map(|images| images[0]).collect::<Vec<_>>();
            let responses = self.decide_vision_batch(requests, &images)?;
            let metrics = self.vision_batch_metrics()?;
            let preparation_cache = self.preparation_cache_stats();
            responses
                .into_iter()
                .map(|response| {
                    let mut output = local_response_json(response)?;
                    output["backend"]["details"]["vision_batch"] = json!({
                        "scope": "last_native_wave",
                        "projector_encode_calls": metrics.projector_encode_calls,
                        "projector_batch_max": metrics.projector_batch_max,
                        "decoder_calls": metrics.decoder_calls,
                        "decoder_batch_max_sequences": metrics.decoder_batch_max_sequences,
                        "projector_reused_chunks": metrics.projector_reused_chunks,
                        "kv_clear_calls": metrics.kv_clear_calls,
                        "kv_clear_skipped": metrics.kv_clear_skipped,
                        "kv_clear_scope": "since_last_native_vision_start",
                        "preparation_cache_scope": "backend_lifetime_since_cache_configuration",
                        "preparation_cache": preparation_cache,
                    });
                    Ok(output)
                })
                .collect()
        } else {
            let mut outputs = requests
                .iter()
                .zip(images)
                .map(|(request, images)| self.decide_json(request, images))
                .collect::<crate::Result<Vec<_>>>()?;
            if !requests.is_empty() && images.iter().all(|images| images.len() == 1) {
                // Attach one shared snapshot after all serial media groups, so
                // their backend metadata still agrees in the wire contract.
                let cache = self.preparation_cache_stats();
                let metrics = self.vision_batch_metrics()?;
                for output in &mut outputs {
                    output["backend"]["details"]["vision_preparation"] = json!({
                        "scope": "backend_lifetime_since_cache_configuration",
                        "cache": cache,
                    });
                    output["backend"]["details"]["vision_kv_clear"] = json!({
                        "scope": "since_last_native_vision_start",
                        "calls": metrics.kv_clear_calls,
                        "skipped": metrics.kv_clear_skipped,
                    });
                }
            }
            Ok(outputs)
        }
    }
}

#[cfg(feature = "wgpu")]
impl HttpDecisionBackend for crate::wgpu::WgpuBackend {
    fn capabilities(&self) -> Value {
        local_capabilities(self.inspect())
    }
    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> crate::Result<Value> {
        local_decide_json(self, request, images)
    }
}

#[cfg(feature = "openrouter")]
impl HttpDecisionBackend for crate::openrouter::OpenRouterBackend {
    fn capabilities(&self) -> Value {
        json!({"api_version":1,"backend":{"runtime":"openrouter-chat-completions","model":self.model()},
            "decision_types":["binary","choice","ordinal"],"evidence":"selection_only",
            "media":{"image":{"supported":true,"max_per_decision":MAX_MEDIA,
                "max_bytes_each":crate::MAX_IMAGE_BYTES,"formats":["png","jpeg","gif","webp"],
                "provider_support":"model_dependent","provider_limit":"model_dependent"}},
            "limits":{"max_body_bytes":MAX_BODY,"max_media":MAX_MEDIA,"max_decisions":MAX_DECISIONS,"max_connections":MAX_CONNECTIONS,"max_inflight_body_bytes":MAX_INFLIGHT_BODY_BYTES,
                "queued_inference_requests":QUEUE_DEPTH,"parallel_decisions":4}})
    }
    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> crate::Result<Value> {
        let response = self.decide_images(request, images)?;
        let results = response
            .results
            .into_iter()
            .map(|result| {
                let value = match result.value {
                    crate::openrouter::RemoteDecisionValue::Binary { value } => {
                        json!({"type":"binary","value":value})
                    }
                    crate::openrouter::RemoteDecisionValue::Choice { selected } => {
                        json!({"type":"choice","selected":selected})
                    }
                    crate::openrouter::RemoteDecisionValue::Ordinal {
                        selected,
                        level_value,
                    } => json!({"type":"ordinal","selected":selected,"level_value":level_value}),
                };
                let (status, reasons) = match result.status {
                    crate::openrouter::RemoteStatus::Selected => ("selected", vec![]),
                    crate::openrouter::RemoteStatus::AbstainedInvalidOutput => {
                        ("abstained", vec!["invalid_output"])
                    }
                    crate::openrouter::RemoteStatus::AbstainedIncomplete => {
                        ("abstained", vec!["incomplete_output"])
                    }
                };
                json!({"id":result.id,"value":value,"status":status,"abstention_reasons":reasons,
                "evidence":{"type":"selection_only","selected_code":result.selected_code,
                    "provider_model":result.provider_model},
                "usage":{"input_tokens":result.input_tokens,"output_tokens":result.output_tokens}})
            })
            .collect::<Vec<_>>();
        Ok(
            json!({"backend":{"runtime":response.backend.runtime,"model":response.backend.model,"details":null},
            "policy":null,"results":results}),
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRequest {
    state: Value,
    #[serde(default)]
    media: Vec<WireMedia>,
    decisions: Vec<WireDecision>,
    #[serde(default)]
    reasoning: Option<crate::ReasoningOptions>,
    #[serde(default, deserialize_with = "strict_request_policy")]
    policy: Option<crate::DecisionPolicy>,
    #[serde(default)]
    target_error_rate: Option<f64>,
    #[serde(default)]
    failure_reasons: HashMap<String, String>,
}

fn strict_request_policy<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<crate::DecisionPolicy>, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Policy {
        min_top_probability: f64,
        min_candidate_mass: f64,
    }
    Option::<Policy>::deserialize(deserializer).map(|policy| {
        policy.map(|policy| crate::DecisionPolicy {
            min_top_probability: policy.min_top_probability,
            min_candidate_mass: policy.min_candidate_mass,
        })
    })
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireMedia {
    Image { id: String, data_base64: String },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDecision {
    id: String,
    instruction: String,
    kind: crate::DecisionKind,
    /// Omitted means all request media; an empty list means text only.
    #[serde(default)]
    media_ids: Option<Vec<String>>,
}

struct PreparedRequest {
    request: DecisionRequest,
    groups: Vec<DecisionRequest>,
    media: HashMap<String, Vec<u8>>,
    group_images: Vec<Vec<String>>,
    reasoning: Option<crate::ReasoningOptions>,
    policy: Option<crate::DecisionPolicy>,
    target_error_rate: Option<f64>,
    failure_reasons: HashMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBatch {
    requests: Vec<Box<serde_json::value::RawValue>>,
}

pub(super) fn run_batch<B: HttpDecisionBackend>(
    backend: &mut B,
    body: &[u8],
    request_id: &str,
) -> crate::Result<Value> {
    let wire: WireBatch = serde_json::from_slice(body)
        .map_err(|error| Error::Invalid(format!("invalid decision batch: {error}")))?;
    if wire.requests.is_empty() || wire.requests.len() > super::MAX_BATCH_REQUESTS {
        return Err(Error::Invalid(format!(
            "native batch requires 1..{} requests",
            super::MAX_BATCH_REQUESTS
        )));
    }
    let capabilities = backend.capabilities();
    if capabilities
        .pointer("/batch/supported")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(Error::Invalid(
            "batch_unsupported: this backend does not support native batching".into(),
        ));
    }
    if capabilities
        .pointer("/batch/enabled")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(Error::Invalid(
            "batch_not_enabled: load with execution_mode parallel for native batching".into(),
        ));
    }
    // Validate every independent wire request before any model execution.
    let prepared = wire
        .requests
        .iter()
        .map(|item| prepare_request(backend, item.get().as_bytes()))
        .collect::<crate::Result<Vec<_>>>()?;
    let total_decisions: usize = prepared
        .iter()
        .map(|item| item.request.decisions.len())
        .sum();
    if total_decisions > MAX_DECISIONS {
        return Err(Error::Invalid(format!(
            "at most {MAX_DECISIONS} total decisions are supported in a native batch"
        )));
    }
    if prepared.iter().any(|item| {
        item.reasoning
            .as_ref()
            .is_some_and(|r| r.mode == crate::ReasoningMode::Thinking)
    }) {
        return Err(Error::Invalid(
            "native parallel batches support direct reasoning only".into(),
        ));
    }
    let groups = prepared
        .iter()
        .flat_map(|item| item.groups.iter().cloned())
        .collect::<Vec<_>>();
    let images = prepared
        .iter()
        .flat_map(PreparedRequest::images)
        .collect::<Vec<_>>();
    let outputs = backend.decide_native_batch_json(&groups, &images)?;
    if outputs.len() != groups.len() {
        return Err(Error::Backend(
            "native batch returned incorrect result count".into(),
        ));
    }
    let mut outputs = outputs.into_iter();
    let mut responses = Vec::with_capacity(prepared.len());
    for (index, item) in prepared.into_iter().enumerate() {
        let count = item.groups.len();
        let mut response = finish_request(item, outputs.by_ref().take(count).collect())?;
        response["request_id"] = json!(format!("{request_id}/{index}"));
        responses.push(response);
    }
    Ok(
        json!({"api_version":1,"request_id":request_id,"execution":"native_parallel","responses":responses}),
    )
}
impl PreparedRequest {
    fn images(&self) -> Vec<Vec<&[u8]>> {
        self.group_images
            .iter()
            .map(|ids| ids.iter().map(|id| self.media[id].as_slice()).collect())
            .collect()
    }
}

pub(super) fn run_request<B: HttpDecisionBackend>(
    backend: &mut B,
    body: &[u8],
) -> crate::Result<Value> {
    let prepared = prepare_request(backend, body)?;
    let outputs = backend.decide_json_batch_with_reasoning(
        &prepared.groups,
        &prepared.images(),
        prepared.reasoning.as_ref(),
    )?;
    finish_request(prepared, outputs)
}

fn prepare_request<B: HttpDecisionBackend>(
    backend: &B,
    body: &[u8],
) -> crate::Result<PreparedRequest> {
    let wire: WireRequest = serde_json::from_slice(body)
        .map_err(|e| Error::Invalid(format!("invalid decision request: {e}")))?;
    if let Some(policy) = &wire.policy {
        policy.validate()?;
    }
    if let Some(reasoning) = &wire.reasoning {
        reasoning.validate()?;
    }
    if wire
        .target_error_rate
        .is_some_and(|rate| !rate.is_finite() || !(0.0..=1.0).contains(&rate))
    {
        return Err(Error::Invalid(
            "target_error_rate must be a finite fraction in [0, 1]".into(),
        ));
    }
    validate_failure_reasons(&wire.failure_reasons)?;
    if (wire.policy.is_some() || wire.target_error_rate.is_some())
        && backend
            .capabilities()
            .get("evidence")
            .and_then(Value::as_str)
            == Some("selection_only")
    {
        return Err(Error::Invalid(
            "this backend returns no scores for a request acceptance policy".into(),
        ));
    }
    if wire.media.len() > MAX_MEDIA {
        return Err(Error::Invalid(format!(
            "at most {MAX_MEDIA} media items are supported"
        )));
    }
    if wire.decisions.len() > MAX_DECISIONS {
        return Err(Error::Invalid(format!(
            "at most {MAX_DECISIONS} decisions are supported"
        )));
    }
    let mut media = HashMap::new();
    let mut media_order = Vec::new();
    for item in wire.media {
        let WireMedia::Image { id, data_base64 } = item;
        if id.trim().is_empty() || media.contains_key(&id) {
            return Err(Error::Invalid(
                "media IDs must be nonempty and unique".into(),
            ));
        }
        if data_base64.len() > crate::MAX_IMAGE_BYTES.div_ceil(3) * 4 {
            return Err(Error::Invalid("image exceeds size limit".into()));
        }
        let image = base64::engine::general_purpose::STANDARD
            .decode(&data_base64)
            .map_err(|_| Error::Invalid("image is not valid standard base64".into()))?;
        crate::validate_image(&image)?;
        media_order.push(id.clone());
        media.insert(id, image);
    }
    let ids = media.keys().cloned().collect::<HashSet<_>>();
    let all_ids = wire
        .decisions
        .iter()
        .flat_map(|d| d.media_ids.as_ref().into_iter().flatten())
        .cloned()
        .collect::<HashSet<_>>();
    if !all_ids.is_subset(&ids) {
        return Err(Error::Invalid(
            "decision references unknown media ID".into(),
        ));
    }
    let request = DecisionRequest {
        state: wire.state.clone(),
        decisions: wire
            .decisions
            .iter()
            .map(|d| Decision {
                id: d.id.clone(),
                instruction: d.instruction.clone(),
                kind: d.kind.clone(),
            })
            .collect(),
    };
    request.validate()?;
    let capabilities = backend.capabilities();
    let max_images = capabilities
        .pointer("/media/image/max_per_decision")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let supported = capabilities
        .pointer("/media/image/supported")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let selections = wire
        .decisions
        .into_iter()
        .map(|wire_decision| {
            let selected_ids = wire_decision
                .media_ids
                .unwrap_or_else(|| media_order.clone());
            let mut seen = HashSet::new();
            if selected_ids.iter().any(|id| !seen.insert(id)) {
                return Err(Error::Invalid("duplicate media reference".into()));
            }
            if selected_ids.len() > max_images || (!selected_ids.is_empty() && !supported) {
                return Err(Error::Invalid(format!(
                    "backend accepts at most {max_images} images per decision"
                )));
            }
            Ok(selected_ids)
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let mut groups = Vec::new();
    let mut group_images = Vec::new();
    let mut index = 0;
    while index < request.decisions.len() {
        let selected_ids = &selections[index];
        let mut end = index + 1;
        while end < request.decisions.len() && selections[end] == *selected_ids {
            end += 1;
        }
        group_images.push(selected_ids.clone());
        groups.push(DecisionRequest {
            state: wire.state.clone(),
            decisions: request.decisions[index..end].to_vec(),
        });
        index = end;
    }
    Ok(PreparedRequest {
        request,
        groups,
        media,
        group_images,
        reasoning: wire.reasoning,
        policy: wire.policy,
        target_error_rate: wire.target_error_rate,
        failure_reasons: wire.failure_reasons,
    })
}

fn finish_request(prepared: PreparedRequest, mut outputs: Vec<Value>) -> crate::Result<Value> {
    let PreparedRequest {
        request,
        groups,
        policy,
        reasoning,
        target_error_rate,
        failure_reasons,
        ..
    } = prepared;
    for (group, output) in groups.iter().zip(&mut outputs) {
        if policy.is_some() || target_error_rate.is_some() {
            let mut policy = match &policy {
                Some(policy) => policy.clone(),
                None => serde_json::from_value(output["policy"].clone()).map_err(|_| {
                    Error::Invalid("backend has no scored acceptance policy".into())
                })?,
            };
            if let Some(rate) = target_error_rate {
                policy.min_top_probability = 1.0 - rate;
            }
            apply_request_policy(output, group, &policy)?;
        }
        if let Some(results) = output["results"].as_array_mut() {
            for result in results {
                let messages = result["abstention_reasons"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|reason| reason.as_str())
                    .filter_map(|code| {
                        failure_reasons.get(code).map(
                            |message| json!({"code":code,"message":message,"user_defined":true}),
                        )
                    })
                    .collect::<Vec<_>>();
                if !messages.is_empty() {
                    result["reason_messages"] = json!(messages);
                }
            }
        }
    }
    if outputs.len() != groups.len() {
        return Err(Error::Backend(
            "backend returned incorrect media group count".into(),
        ));
    }
    let mut results = Vec::with_capacity(request.decisions.len());
    let mut response_backend = None;
    let mut response_policy = None;
    for (batch, output) in groups.iter().zip(&outputs) {
        let backend_info = output
            .get("backend")
            .filter(|info| {
                info.get("runtime").and_then(Value::as_str).is_some()
                    && info.get("model").and_then(Value::as_str).is_some()
            })
            .ok_or_else(|| Error::Backend("backend returned invalid metadata".into()))?;
        let policy = output
            .get("policy")
            .ok_or_else(|| Error::Backend("backend returned no policy field".into()))?;
        if let Some(existing) = &response_backend {
            if existing != backend_info || response_policy.as_ref() != Some(policy) {
                return Err(Error::Backend(
                    "backend metadata changed within request".into(),
                ));
            }
        } else {
            response_backend = Some(backend_info.clone());
            response_policy = Some(policy.clone());
        }
        let batch_results = output
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Backend("backend returned no decision results".into()))?;
        if batch_results.len() != batch.decisions.len()
            || batch_results
                .iter()
                .zip(&batch.decisions)
                .any(|(result, decision)| {
                    let kind = match &decision.kind {
                        crate::DecisionKind::Binary { .. } => "binary",
                        crate::DecisionKind::Choice { .. } => "choice",
                        crate::DecisionKind::Ordinal { .. } => "ordinal",
                    };
                    result.get("id").and_then(Value::as_str) != Some(decision.id.as_str())
                        || result.pointer("/value/type").and_then(Value::as_str) != Some(kind)
                        || !matches!(
                            result.get("status").and_then(Value::as_str),
                            Some("selected" | "abstained")
                        )
                        || result
                            .get("abstention_reasons")
                            .and_then(Value::as_array)
                            .is_none()
                        || result
                            .pointer("/evidence/type")
                            .and_then(Value::as_str)
                            .is_none()
                        || result.get("usage").and_then(Value::as_object).is_none()
                })
        {
            return Err(Error::Backend(
                "backend results do not match the API contract".into(),
            ));
        }
        results.extend(batch_results.iter().cloned());
    }
    let mut output = json!({"api_version":1,"backend":response_backend,"policy":response_policy,"results":results});
    if let Some(rate) = target_error_rate {
        output["error_budget"] = json!({"requested_rate":rate,"guaranteed":false,
            "interpretation":"model_score_threshold","min_top_probability":1.0-rate});
    }
    if let Some(reasoning) = reasoning {
        output["reasoning"] = json!(reasoning);
    }
    Ok(output)
}

const FAILURE_REASON_CODES: &[&str] = &[
    "low_top_probability",
    "low_candidate_mass",
    "tied_candidates",
    "reasoning_limit",
    "native_failure",
];

fn validate_failure_reasons(reasons: &HashMap<String, String>) -> crate::Result<()> {
    for (code, message) in reasons {
        if !FAILURE_REASON_CODES.contains(&code.as_str())
            || message.trim().is_empty()
            || message.len() > 512
        {
            return Err(Error::Invalid("failure_reasons must use known codes and nonempty messages of at most 512 UTF-8 bytes".into()));
        }
    }
    Ok(())
}

pub(crate) fn user_failure_messages(body: &[u8]) -> HashMap<String, String> {
    // Unknown fields are consumed as IgnoredAny by serde rather than allocating
    // a second Value tree containing a potentially 44 MiB base64 image body.
    #[derive(Deserialize)]
    struct Messages {
        #[serde(default)]
        failure_reasons: HashMap<String, String>,
    }
    let Ok(messages) = serde_json::from_slice::<Messages>(body) else {
        return HashMap::new();
    };
    if validate_failure_reasons(&messages.failure_reasons).is_err() {
        return HashMap::new();
    }
    messages.failure_reasons
}

/// Change acceptance after scoring; model evidence and ordinal estimates are retained.
fn apply_request_policy(
    output: &mut Value,
    request: &DecisionRequest,
    policy: &crate::DecisionPolicy,
) -> crate::Result<()> {
    policy.validate()?;
    let results = output["results"]
        .as_array_mut()
        .ok_or_else(|| Error::Backend("missing scored results".into()))?;
    if results.len() != request.decisions.len() {
        return Err(Error::Backend("incorrect policy result count".into()));
    }
    for (result, decision) in results.iter_mut().zip(&request.decisions) {
        if result.pointer("/evidence/type").and_then(Value::as_str) != Some("model_scored") {
            return Err(Error::Invalid(
                "request policy requires model-scored evidence".into(),
            ));
        }
        let scores: Vec<crate::OptionScore> =
            serde_json::from_value(result["evidence"]["scores"].clone())
                .map_err(|_| Error::Backend("invalid candidate scores".into()))?;
        let expected_ids = decision
            .options()
            .into_iter()
            .map(|option| option.id)
            .collect::<Vec<_>>();
        let score_ids = scores
            .iter()
            .map(|score| score.id.clone())
            .collect::<Vec<_>>();
        let mass = result["evidence"]["candidate_mass"]
            .as_f64()
            .ok_or_else(|| Error::Backend("missing candidate mass".into()))?;
        if scores.is_empty()
            || scores.len() != expected_ids.len()
            || score_ids != expected_ids
            || !mass.is_finite()
            || !(0.0..=1.0).contains(&mass)
            || scores.iter().any(|s| {
                !s.option_probability.is_finite() || !(0.0..=1.0).contains(&s.option_probability)
            })
            || (scores
                .iter()
                .map(|score| score.option_probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                > 1e-6
        {
            return Err(Error::Backend("invalid scored policy evidence".into()));
        }
        let best = scores.iter().fold(&scores[0], |best, s| {
            if s.option_probability > best.option_probability {
                s
            } else {
                best
            }
        });
        if result["id"] != decision.id
            || result["value"]["type"]
                != match decision.kind {
                    crate::DecisionKind::Binary { .. } => "binary",
                    crate::DecisionKind::Choice { .. } => "choice",
                    crate::DecisionKind::Ordinal { .. } => "ordinal",
                }
            || result["evidence"]["top_option_probability"]
                .as_f64()
                .is_none_or(|probability| {
                    !probability.is_finite() || (probability - best.option_probability).abs() > 1e-6
                })
        {
            return Err(Error::Backend("inconsistent scored policy metadata".into()));
        }
        let mut reasons = Vec::new();
        if mass < policy.min_candidate_mass {
            reasons.push("low_candidate_mass");
        }
        if best.option_probability < policy.min_top_probability {
            reasons.push("low_top_probability");
        }
        if scores
            .iter()
            .filter(|s| (s.option_probability - best.option_probability).abs() < 1e-12)
            .count()
            > 1
        {
            reasons.push("tied_candidates");
        }
        let accepted = reasons.is_empty();
        result["value"] = match decision.kind {
            crate::DecisionKind::Binary { .. } => {
                json!({"type":"binary","value":accepted.then_some(best.id=="true")})
            }
            crate::DecisionKind::Choice { .. } => {
                json!({"type":"choice","selected":accepted.then_some(best.id.as_str())})
            }
            crate::DecisionKind::Ordinal { .. } => {
                json!({"type":"ordinal","selected":accepted.then_some(best.id.as_str())})
            }
        };
        result["status"] = json!(if accepted { "selected" } else { "abstained" });
        result["abstention_reasons"] = json!(reasons);
    }
    output["policy"] = json!(policy);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecisionBackend, DecisionResponse};

    struct LocalProbe;
    impl DecisionBackend for LocalProbe {
        fn decide(&mut self, _: &DecisionRequest) -> crate::Result<DecisionResponse> {
            Ok(serde_json::from_value(json!({
                "backend":{"model_path":"model.gguf","model_description":"test","prompt_version":"v1",
                    "runtime":"local-test","offload_requested":false,"offload_device":null},
                "policy":{"min_top_probability":0.8,"min_candidate_mass":0.05},
                "results":[{"id":"flag","value":{"type":"binary","p_true":0.9,"value":true},
                    "scores":[{"id":"false","code":"A","token_id":1,"raw_logit":0.0,"option_probability":0.1},
                        {"id":"true","code":"B","token_id":2,"raw_logit":2.0,"option_probability":0.9}],
                    "candidate_mass":0.7,"top_option_probability":0.9,"entropy_confidence":0.5,
                    "abstention_reasons":[],"scoring_method":"test","calibration_id":null,
                    "input_tokens":12,"truncated":false}]
            })).unwrap())
        }
    }
    impl VisionDecisionBackend for LocalProbe {
        fn decide_vision(
            &mut self,
            request: &DecisionRequest,
            _: &[u8],
        ) -> crate::Result<DecisionResponse> {
            self.decide(request)
        }
    }

    #[test]
    fn local_response_keeps_logits_only_in_evidence() {
        let request: DecisionRequest = serde_json::from_value(json!({"state":{},"decisions":[{
            "id":"flag","instruction":"Choose","kind":{"type":"binary","false_label":"no","true_label":"yes"}}]})).unwrap();
        let capabilities = local_capabilities(&LocalProbe.decide(&request).unwrap().backend);
        assert_eq!(capabilities["media"]["image"]["supported"], false);
        assert_eq!(capabilities["evidence"], "model_scored");
        let output = local_decide_json(&mut LocalProbe, &request, &[]).unwrap();
        assert_eq!(
            output["results"][0]["value"],
            json!({"type":"binary","value":true})
        );
        assert_eq!(output["results"][0]["evidence"]["type"], "model_scored");
        assert_eq!(output["results"][0]["evidence"]["estimate"]["p_true"], 0.9);
        assert_eq!(
            output["results"][0]["evidence"]["scores"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(output["results"][0]["usage"]["input_tokens"], 12);
    }

    #[derive(Default)]
    struct BatchProbe {
        calls: usize,
        observed: Vec<(Value, Vec<String>, Vec<Vec<u8>>)>,
    }

    impl HttpDecisionBackend for BatchProbe {
        fn capabilities(&self) -> Value {
            json!({"media":{"image":{"supported":true,"max_per_decision":1}}})
        }
        fn decide_json(&mut self, _: &DecisionRequest, _: &[&[u8]]) -> crate::Result<Value> {
            panic!("validated image groups should reach the batch boundary")
        }
        fn decide_json_batch(
            &mut self,
            requests: &[DecisionRequest],
            images: &[Vec<&[u8]>],
        ) -> crate::Result<Vec<Value>> {
            self.calls += 1;
            self.observed = requests
                .iter()
                .zip(images)
                .map(|(request, images)| {
                    (
                        request.state.clone(),
                        request
                            .decisions
                            .iter()
                            .map(|decision| decision.id.clone())
                            .collect(),
                        images.iter().map(|image| image.to_vec()).collect(),
                    )
                })
                .collect();
            Ok(requests
                .iter()
                .map(|request| {
                    json!({
                        "backend":{"runtime":"batch-probe","model":"test"}, "policy":null,
                        "results":request.decisions.iter().map(|decision| json!({
                            "id":decision.id,"value":{"type":"binary","value":true},
                            "status":"selected","abstention_reasons":[],
                            "evidence":{"type":"model_scored"},"usage":{},
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect())
        }
    }

    #[test]
    fn validated_media_groups_reach_one_batch_call_in_decision_order() {
        let mut backend = BatchProbe::default();
        let body = json!({
            "state":{"tenant":"alpha"},
            "media":[{"id":"a","type":"image","data_base64":"Zmlyc3Q="},
                {"id":"b","type":"image","data_base64":"c2Vjb25k"}],
            "decisions":([("one","a"),("two","b"),("three","a")].iter()
                .map(|(id, image)| json!({"id":id,"instruction":"Choose",
                    "kind":{"type":"binary","false_label":"no","true_label":"yes"},
                    "media_ids":[image]})).collect::<Vec<_>>()),
        });
        let output = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(backend.calls, 1);
        assert_eq!(backend.observed.len(), 3);
        assert_eq!(backend.observed[0].2, vec![b"first".to_vec()]);
        assert_eq!(backend.observed[1].2, vec![b"second".to_vec()]);
        assert_eq!(backend.observed[2].2, vec![b"first".to_vec()]);
        assert!(
            backend
                .observed
                .iter()
                .all(|(state, _, _)| state == &body["state"])
        );
        assert_eq!(
            output["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|result| result["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["one", "two", "three"]
        );
    }

    #[test]
    fn invalid_media_reference_fails_before_batch_dispatch() {
        let mut backend = BatchProbe::default();
        let body = json!({"state":{},"decisions":[{
            "id":"one","instruction":"Choose",
            "kind":{"type":"binary","false_label":"no","true_label":"yes"},
            "media_ids":["missing"],
        }]});
        assert!(run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).is_err());
        assert_eq!(backend.calls, 0);
    }

    struct ScoredProbe {
        calls: usize,
        output: Value,
        selection_only: bool,
    }
    impl HttpDecisionBackend for ScoredProbe {
        fn capabilities(&self) -> Value {
            json!({"evidence":if self.selection_only {"selection_only"} else {"model_scored"},
                "media":{"image":{"supported":false,"max_per_decision":0}}})
        }
        fn decide_json(&mut self, _: &DecisionRequest, _: &[&[u8]]) -> crate::Result<Value> {
            self.calls += 1;
            Ok(self.output.clone())
        }
    }
    fn policy_fixture() -> (ScoredProbe, Value) {
        let body = json!({"state":{},"decisions":[{"id":"flag","instruction":"Choose",
            "kind":{"type":"binary","false_label":"no","true_label":"yes"}}]});
        let request: DecisionRequest = serde_json::from_value(body.clone()).unwrap();
        let output = local_decide_json(&mut LocalProbe, &request, &[]).unwrap();
        (
            ScoredProbe {
                calls: 0,
                output,
                selection_only: false,
            },
            body,
        )
    }

    #[derive(Default)]
    struct NativeProbe {
        calls: usize,
        states: Vec<Value>,
        image_bytes: Vec<Vec<Vec<u8>>>,
        fail: bool,
        short_output: bool,
    }
    impl HttpDecisionBackend for NativeProbe {
        fn capabilities(&self) -> Value {
            json!({"batch":{"supported":true,"enabled":true},"evidence":"model_scored",
                "media":{"image":{"supported":true,"max_per_decision":1}}})
        }
        fn decide_json(&mut self, _: &DecisionRequest, _: &[&[u8]]) -> crate::Result<Value> {
            panic!("native request batches must never invoke a serial fallback")
        }
        fn decide_native_batch_json(
            &mut self,
            requests: &[DecisionRequest],
            images: &[Vec<&[u8]>],
        ) -> crate::Result<Vec<Value>> {
            self.calls += 1;
            self.states = requests.iter().map(|r| r.state.clone()).collect();
            self.image_bytes = images
                .iter()
                .map(|group| group.iter().map(|image| image.to_vec()).collect())
                .collect();
            if self.fail {
                return Err(Error::Backend("native test failure".into()));
            }
            let mut outputs = requests
                .iter()
                .zip(images)
                .map(|(r, images)| local_decide_json(&mut LocalProbe, r, images))
                .collect::<crate::Result<Vec<_>>>()?;
            if self.short_output {
                outputs.pop();
            }
            Ok(outputs)
        }
    }

    #[test]
    fn native_batch_preserves_independent_states_policies_and_duplicate_ids_across_requests() {
        let (_, mut first) = policy_fixture();
        first["state"] = json!({"tenant":"first"});
        first["policy"] = json!({"min_top_probability":0.95,"min_candidate_mass":0.05});
        first["failure_reasons"] = json!({"low_top_probability":"review first"});
        let (_, mut second) = policy_fixture();
        second["state"] = json!({"tenant":"second"});
        second["policy"] = json!({"min_top_probability":0.5,"min_candidate_mass":0.05});
        let mut backend = NativeProbe::default();
        let output = run_batch(
            &mut backend,
            &serde_json::to_vec(&json!({"requests":[first,second]})).unwrap(),
            "batch-1",
        )
        .unwrap();
        assert_eq!(backend.calls, 1);
        assert_eq!(
            backend.states,
            vec![json!({"tenant":"first"}), json!({"tenant":"second"})]
        );
        let responses = &output["responses"];
        assert_eq!(output["execution"], "native_parallel");
        assert_eq!(responses[0]["request_id"], "batch-1/0");
        assert_eq!(responses[1]["request_id"], "batch-1/1");
        assert_eq!(responses[0]["results"][0]["id"], "flag");
        assert_eq!(responses[1]["results"][0]["id"], "flag");
        assert_eq!(responses[0]["results"][0]["value"]["value"], Value::Null);
        assert_eq!(responses[1]["results"][0]["value"]["value"], true);
        assert_eq!(
            responses[0]["results"][0]["reason_messages"][0]["message"],
            "review first"
        );
        assert_eq!(
            responses[0]["results"][0]["evidence"],
            responses[1]["results"][0]["evidence"]
        );
    }

    #[test]
    fn native_batch_validates_all_requests_and_limits_before_dispatch() {
        let (_, body) = policy_fixture();
        let mut backend = NativeProbe::default();
        let mut invalid = body.clone();
        invalid["decisions"] = json!([body["decisions"][0], body["decisions"][0]]);
        let mut missing = body.clone();
        missing["decisions"][0]["media_ids"] = json!(["missing"]);
        let mut thinking = body.clone();
        thinking["reasoning"] = json!({"mode":"thinking"});
        for last in [invalid, missing, thinking] {
            assert!(
                run_batch(
                    &mut backend,
                    &serde_json::to_vec(&json!({"requests":[body,last]})).unwrap(),
                    "test"
                )
                .is_err()
            );
        }
        for requests in [vec![], vec![body.clone(); 129]] {
            assert!(
                run_batch(
                    &mut backend,
                    &serde_json::to_vec(&json!({"requests":requests})).unwrap(),
                    "test"
                )
                .is_err()
            );
        }
        let mut many = body.clone();
        many["decisions"] = json!(
            (0..65)
                .map(|i| {
                    let mut decision = body["decisions"][0].clone();
                    decision["id"] = json!(format!("d-{i}"));
                    decision
                })
                .collect::<Vec<_>>()
        );
        assert!(
            run_batch(
                &mut backend,
                &serde_json::to_vec(&json!({"requests":[many,many]})).unwrap(),
                "test"
            )
            .is_err()
        );
        assert_eq!(backend.calls, 0);
        let (mut unsupported, _) = policy_fixture();
        assert!(
            run_batch(
                &mut unsupported,
                &serde_json::to_vec(&json!({"requests":[body]})).unwrap(),
                "test"
            )
            .unwrap_err()
            .to_string()
            .contains("batch_unsupported")
        );
        assert_eq!(unsupported.calls, 0);
    }

    #[test]
    fn native_batch_keeps_request_local_media_and_never_replays_failures() {
        let (_, mut first) = policy_fixture();
        first["media"] = json!([{"id":"same","type":"image","data_base64":"Zmlyc3Q="}]);
        let mut second = first.clone();
        second["media"][0]["data_base64"] = json!("c2Vjb25k");
        let body = serde_json::to_vec(&json!({"requests":[first,second]})).unwrap();
        let mut backend = NativeProbe::default();
        run_batch(&mut backend, &body, "test").unwrap();
        assert_eq!(
            backend.image_bytes,
            vec![vec![b"first".to_vec()], vec![b"second".to_vec()]]
        );
        backend.fail = true;
        assert!(run_batch(&mut backend, &body, "test").is_err());
        assert_eq!(backend.calls, 2);
        backend.fail = false;
        backend.short_output = true;
        assert!(
            run_batch(&mut backend, &body, "test")
                .unwrap_err()
                .to_string()
                .contains("incorrect result count")
        );
        assert_eq!(backend.calls, 3);
    }

    #[test]
    fn compiled_stdio_dispatches_one_native_batch_without_a_listener_and_survives_request_error() {
        let (_, body) = policy_fixture();
        let calls = [
            json!({"id":"bad","op":"decide","body":{}}),
            json!({"id":"local","op":"decide_batch","body":{"requests":[body,body]}}),
            json!({"id":"health","op":"health"}),
        ];
        let input = calls.iter().map(|v| format!("{v}\n")).collect::<String>();
        let mut output = Vec::new();
        let mut backend = NativeProbe::default();
        crate::stdio::serve_stream(&mut backend, std::io::Cursor::new(input), &mut output).unwrap();
        let envelopes = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(backend.calls, 1);
        assert_eq!(envelopes[0]["error"]["code"], "invalid_request");
        assert_eq!(envelopes[1]["result"]["execution"], "native_parallel");
        assert_eq!(
            envelopes[1]["result"]["responses"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(envelopes[2]["result"]["status"], "ok");
    }
    #[test]
    fn request_thresholds_preserve_raw_evidence_and_do_not_persist() {
        let (mut backend, mut body) = policy_fixture();
        let original = backend.output["results"][0].clone();
        body["policy"] = json!({"min_top_probability":0.95,"min_candidate_mass":0.05});
        body["failure_reasons"] = json!({"low_top_probability":"검토 담당자에게 전달하세요."});
        let output = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap();
        let result = &output["results"][0];
        assert_eq!(result["status"], "abstained");
        assert_eq!(result["value"]["value"], Value::Null);
        assert_eq!(result["abstention_reasons"], json!(["low_top_probability"]));
        assert_eq!(
            result["reason_messages"][0],
            json!({"code":"low_top_probability","message":"검토 담당자에게 전달하세요.","user_defined":true})
        );
        assert_eq!(result["evidence"], original["evidence"]);
        assert_eq!(result["usage"], original["usage"]);
        assert_eq!(output["policy"]["min_top_probability"], 0.95);
        body["policy"] = json!({"min_top_probability":0.5,"min_candidate_mass":0.05});
        let lower = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(lower["results"][0]["value"]["value"], true);
        assert_eq!(lower["results"][0]["evidence"], original["evidence"]);
        body.as_object_mut().unwrap().remove("policy");
        body.as_object_mut().unwrap().remove("failure_reasons");
        let restored = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(restored["results"][0], original);
        assert_eq!(restored["policy"]["min_top_probability"], 0.8);
    }
    #[test]
    fn target_error_rate_maps_to_a_threshold_without_correctness_guarantee() {
        let (mut backend, mut body) = policy_fixture();
        body["policy"] = json!({"min_top_probability":0.3,"min_candidate_mass":0.8});
        body["target_error_rate"] = json!(0.03);
        let output = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(output["policy"]["min_top_probability"], 0.97);
        assert_eq!(output["policy"]["min_candidate_mass"], 0.8);
        assert_eq!(output["error_budget"]["guaranteed"], false);
        assert_eq!(
            output["error_budget"]["interpretation"],
            "model_score_threshold"
        );
        assert_eq!(
            output["results"][0]["abstention_reasons"],
            json!(["low_candidate_mass", "low_top_probability"])
        );
        assert_eq!(
            output["results"][0]["evidence"],
            backend.output["results"][0]["evidence"]
        );
    }
    #[test]
    fn invalid_request_controls_are_rejected_before_inference() {
        let (mut backend, body) = policy_fixture();
        for (field, value) in [
            (
                "policy",
                json!({"min_top_probability":-0.1,"min_candidate_mass":0.05}),
            ),
            (
                "policy",
                json!({"min_top_probability":0.9,"min_candidate_mass":1.1}),
            ),
            (
                "policy",
                json!({"min_top_probability":0.9,"min_candidate_mass":0.05,"misspelled_threshold":1}),
            ),
            ("target_error_rate", json!(-0.01)),
            ("target_error_rate", json!(1.01)),
            ("failure_reasons", json!({"unknown":"x"})),
            ("failure_reasons", json!({"low_top_probability":"  "})),
            (
                "failure_reasons",
                json!({"low_top_probability":"가".repeat(171)}),
            ),
            ("reasoning", json!({"mode":"thinking","max_tokens":0})),
            ("reasoning", json!({"mode":"thinking","max_tokens":1025})),
        ] {
            let mut invalid = body.clone();
            invalid[field] = value;
            assert!(
                matches!(
                    run_request(&mut backend, &serde_json::to_vec(&invalid).unwrap()),
                    Err(Error::Invalid(_))
                ),
                "{invalid}"
            );
        }
        assert_eq!(backend.calls, 0);
        backend.selection_only = true;
        let mut unsupported = body;
        unsupported["target_error_rate"] = json!(0.1);
        assert!(run_request(&mut backend, &serde_json::to_vec(&unsupported).unwrap()).is_err());
        assert_eq!(backend.calls, 0);
    }
    #[test]
    fn unsupported_backend_refuses_thinking_before_inference() {
        let (mut backend, mut body) = policy_fixture();
        body["reasoning"] = json!({"mode":"thinking","max_tokens":128});
        let error = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap_err();
        assert!(error.to_string().contains("thinking mode is not supported"));
        assert_eq!(backend.calls, 0);
        body["reasoning"] = json!({"mode":"direct","max_tokens":128});
        let output = run_request(&mut backend, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(output["reasoning"]["mode"], "direct");
        assert_eq!(backend.calls, 1);
    }
    #[test]
    fn policy_overlay_rejects_malformed_probabilities_and_preserves_estimates() {
        let (backend, body) = policy_fixture();
        let request: DecisionRequest = serde_json::from_value(body).unwrap();
        let policy = crate::DecisionPolicy::default();
        for mutation in 0..5 {
            let mut output = backend.output.clone();
            match mutation {
                0 => {
                    output["results"][0]["evidence"]["scores"][0]["option_probability"] =
                        json!(0.8);
                }
                1 => {
                    output["results"][0]["evidence"]["scores"]
                        .as_array_mut()
                        .unwrap()
                        .reverse();
                }
                2 => {
                    output["results"][0]["evidence"]["top_option_probability"] = json!(0.5);
                }
                3 => {
                    output["results"][0]["value"]["type"] = json!("choice");
                }
                _ => {
                    output["results"][0]["evidence"]["candidate_mass"] = json!(1.1);
                }
            }
            assert!(apply_request_policy(&mut output, &request, &policy).is_err());
        }
        let ordinal: DecisionRequest=serde_json::from_value(json!({"state":{},"decisions":[{
            "id":"flag","instruction":"Rate","kind":{"type":"ordinal","levels":[
                {"id":"low","criterion":"Low","value":10},{"id":"high","criterion":"High","value":20}]}}]})).unwrap();
        let mut output = backend.output.clone();
        output["results"][0]["value"] = json!({"type":"ordinal","selected":"high"});
        output["results"][0]["evidence"]["scores"][0]["id"] = json!("low");
        output["results"][0]["evidence"]["scores"][1]["id"] = json!("high");
        output["results"][0]["evidence"]["estimate"] = json!({"expected_value":19.0});
        let evidence = output["results"][0]["evidence"].clone();
        apply_request_policy(
            &mut output,
            &ordinal,
            &crate::DecisionPolicy {
                min_top_probability: 0.95,
                min_candidate_mass: 0.05,
            },
        )
        .unwrap();
        assert_eq!(output["results"][0]["value"]["selected"], Value::Null);
        assert_eq!(output["results"][0]["evidence"], evidence);
    }
    #[test]
    fn failure_message_extraction_validates_codes_and_skips_large_unused_data() {
        let body = json!({"media":[{"data_base64":"a".repeat(1024*1024)}],"state":{"unused":[1,2,3]},
            "failure_reasons":{"reasoning_limit":"추론 한도를 늘리세요.","native_failure":"다시 시도하세요."}});
        let messages = user_failure_messages(&serde_json::to_vec(&body).unwrap());
        assert_eq!(messages["reasoning_limit"], "추론 한도를 늘리세요.");
        assert_eq!(messages.len(), 2);
        for body in [
            b"{".as_slice(),
            b"{\"failure_reasons\":{\"unknown\":\"x\"}}".as_slice(),
            b"{}".as_slice(),
        ] {
            assert!(user_failure_messages(body).is_empty());
        }
    }
}
