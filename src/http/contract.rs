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
        json!({"id":result.id,"value":value,"status":status,"abstention_reasons":result.abstention_reasons,
            "evidence":evidence,"usage":{"input_tokens":result.input_tokens,"reused_prefix_tokens":result.reused_prefix_tokens}})
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
            "projector_encoding": "batched_when_compatible",
        });
        capabilities
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
            requests
                .iter()
                .zip(images)
                .map(|(request, images)| self.decide_json(request, images))
                .collect()
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

pub(super) fn run_request<B: HttpDecisionBackend>(
    backend: &mut B,
    body: &[u8],
) -> crate::Result<Value> {
    let wire: WireRequest = serde_json::from_slice(body)
        .map_err(|e| Error::Invalid(format!("invalid decision request: {e}")))?;
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
        group_images.push(
            selected_ids
                .iter()
                .map(|id| media.get(id).expect("validated media ID").as_slice())
                .collect::<Vec<_>>(),
        );
        groups.push(DecisionRequest {
            state: wire.state.clone(),
            decisions: request.decisions[index..end].to_vec(),
        });
        index = end;
    }
    let outputs = backend.decide_json_batch(&groups, &group_images)?;
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
    Ok(
        json!({"api_version":1,"backend":response_backend,"policy":response_policy,"results":results}),
    )
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
}
