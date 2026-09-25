//! Selection-only OpenRouter adapter. Remote chat completions do not provide
//! the full-vocabulary evidence required by `DecisionResponse`.
use crate::{DecisionKind, DecisionRequest, Error, Result, option_code};
use base64::Engine;
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};
use std::{io::Read, time::Duration};

const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;

pub struct OpenRouterBackend {
    client: Client,
    endpoint: String,
    model: String,
    api_key: String,
    max_tokens: u32,
    reasoning_effort: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RemoteDecisionResponse {
    pub backend: RemoteBackendInfo,
    /// Remote generation supplies a selected code, not complete candidate scores.
    pub evidence: &'static str,
    pub results: Vec<RemoteDecisionResult>,
}

#[derive(Debug, Serialize)]
pub struct RemoteBackendInfo {
    pub runtime: &'static str,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct RemoteDecisionResult {
    pub id: String,
    pub value: RemoteDecisionValue,
    pub selected_code: Option<String>,
    pub status: RemoteStatus,
    pub provider_model: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteDecisionValue {
    Binary {
        value: Option<bool>,
    },
    Choice {
        selected: Option<String>,
    },
    Ordinal {
        selected: Option<String>,
        level_value: Option<f64>,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteStatus {
    Selected,
    AbstainedInvalidOutput,
    AbstainedIncomplete,
}

impl OpenRouterBackend {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let model = model.into();
        let api_key = api_key.into();
        if model.trim().is_empty() || api_key.trim().is_empty() {
            return Err(Error::Invalid(
                "OpenRouter model and API key are required".into(),
            ));
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Backend(format!("HTTP client initialization failed: {e}")))?;
        Ok(Self {
            client,
            endpoint: ENDPOINT.into(),
            model,
            api_key,
            max_tokens: 1024,
            reasoning_effort: None,
        })
    }

    pub fn set_max_tokens(&mut self, max_tokens: u32) -> Result<()> {
        if !(1..=4096).contains(&max_tokens) {
            return Err(Error::Invalid(
                "OpenRouter max_tokens must be in 1..=4096".into(),
            ));
        }
        self.max_tokens = max_tokens;
        Ok(())
    }

    pub fn set_reasoning_effort(&mut self, effort: &str) -> Result<()> {
        if !matches!(
            effort,
            "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
        ) {
            return Err(Error::Invalid("invalid OpenRouter reasoning effort".into()));
        }
        self.reasoning_effort = Some(effort.into());
        Ok(())
    }

    pub fn decide(
        &mut self,
        request: &DecisionRequest,
        image: Option<&[u8]>,
    ) -> Result<RemoteDecisionResponse> {
        request.validate()?;
        let image_url = image.map(image_data_url).transpose()?;
        let mut results = Vec::with_capacity(request.decisions.len());
        for decision in &request.decisions {
            let user_content = if let Some(image_url) = &image_url {
                json!([
                    {"type":"text","text":crate::prompt::decision_data(&request.state, decision, crate::PromptLayout::Legacy)},
                    {"type":"image_url","image_url":{"url":image_url}}
                ])
            } else {
                json!(crate::prompt::decision_data(
                    &request.state,
                    decision,
                    crate::PromptLayout::Legacy
                ))
            };
            let mut payload = json!({
                "model": self.model,
                "messages": [
                    {"role":"system","content":crate::prompt::decision_system(decision)},
                    {"role":"user","content":user_content}
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "stream": false
            });
            if let Some(effort) = &self.reasoning_effort {
                payload["reasoning"] = json!({"effort": effort});
            }
            let response = self
                .client
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&payload)
                .send()
                .map_err(|e| Error::Upstream(format!("OpenRouter request failed: {e}")))?;
            let status = response.status();
            if !status.is_success() {
                return Err(Error::Upstream(format!(
                    "OpenRouter returned HTTP {status}"
                )));
            }
            let mut bytes = Vec::new();
            response
                .take(MAX_RESPONSE_BYTES + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| Error::Upstream(format!("OpenRouter response read failed: {e}")))?;
            if bytes.len() as u64 > MAX_RESPONSE_BYTES {
                return Err(Error::Upstream(
                    "OpenRouter response exceeds size limit".into(),
                ));
            }
            let body: Value = serde_json::from_slice(&bytes)
                .map_err(|_| Error::Upstream("OpenRouter returned invalid JSON".into()))?;
            results.push(parse_result(decision, &body)?);
        }
        Ok(RemoteDecisionResponse {
            backend: RemoteBackendInfo {
                runtime: "openrouter-chat-completions",
                model: self.model.clone(),
            },
            evidence: "selection_only",
            results,
        })
    }
}

fn image_data_url(image: &[u8]) -> Result<String> {
    crate::validate_image(image)?;
    let mime = if image.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if image.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if image.starts_with(b"GIF87a") || image.starts_with(b"GIF89a") {
        "image/gif"
    } else if image.len() >= 12 && image.starts_with(b"RIFF") && &image[8..12] == b"WEBP" {
        "image/webp"
    } else {
        return Err(Error::Invalid(
            "image must be PNG, JPEG, GIF, or WebP".into(),
        ));
    };
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(image)
    ))
}

fn parse_result(decision: &crate::Decision, body: &Value) -> Result<RemoteDecisionResult> {
    let choice = body
        .pointer("/choices/0")
        .ok_or_else(|| Error::Upstream("OpenRouter response has no choice".into()))?;
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    let content = choice
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or("");
    let options = decision.options();
    let code = content.trim();
    let selected_index = (finish == Some("stop"))
        .then(|| crate::codes::option_code_index(code, options.len()))
        .flatten();
    let selected_code = selected_index
        .map(|index| option_code(index, options.len()).expect("validated option count"));
    let status = if finish != Some("stop") {
        RemoteStatus::AbstainedIncomplete
    } else if selected_index.is_some() {
        RemoteStatus::Selected
    } else {
        RemoteStatus::AbstainedInvalidOutput
    };
    let value = match &decision.kind {
        DecisionKind::Binary { .. } => RemoteDecisionValue::Binary {
            value: selected_index.map(|index| index == 1),
        },
        DecisionKind::Choice { .. } => RemoteDecisionValue::Choice {
            selected: selected_index.map(|index| options[index].id.clone()),
        },
        DecisionKind::Ordinal { levels } => RemoteDecisionValue::Ordinal {
            selected: selected_index.map(|index| levels[index].id.clone()),
            level_value: selected_index.map(|index| levels[index].value),
        },
    };
    Ok(RemoteDecisionResult {
        id: decision.id.clone(),
        value,
        selected_code,
        status,
        provider_model: body.get("model").and_then(Value::as_str).map(str::to_owned),
        input_tokens: body.pointer("/usage/prompt_tokens").and_then(Value::as_u64),
        output_tokens: body
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };

    fn request() -> DecisionRequest {
        serde_json::from_value(json!({
            "state":{"color":"red"},
            "decisions":[{"id":"color","instruction":"Choose the color.","kind":{"type":"choice","options":[
                {"id":"red","criterion":"red"},{"id":"blue","criterion":"blue"}
            ]}}]
        })).unwrap()
    }

    #[test]
    fn strict_code_and_incomplete_output_abstain_without_scores() {
        let decision = &request().decisions[0];
        let selected = parse_result(
            decision,
            &json!({"choices":[{"finish_reason":"stop","message":{"content":" A\n"}}]}),
        )
        .unwrap();
        assert!(matches!(selected.status, RemoteStatus::Selected));
        assert_eq!(selected.selected_code.as_deref(), Some("A"));
        for (finish, content, incomplete) in [("stop", "A.", false), ("length", "A", true)] {
            let result = parse_result(
                decision,
                &json!({"choices":[{"finish_reason":finish,"message":{"content":content}}]}),
            )
            .unwrap();
            assert!(result.selected_code.is_none());
            assert!(matches!(
                result.value,
                RemoteDecisionValue::Choice { selected: None }
            ));
            assert_eq!(
                matches!(result.status, RemoteStatus::AbstainedIncomplete),
                incomplete
            );
        }
        let json = serde_json::to_value(selected).unwrap();
        assert!(json.get("candidate_mass").is_none());
        assert!(json.get("scores").is_none());
    }

    #[test]
    fn wide_codes_and_ordinal_values_keep_semantic_mapping() {
        let options = (0..27)
            .map(|i| crate::OptionSpec {
                id: format!("option_{i}"),
                criterion: format!("criterion {i}"),
            })
            .collect();
        let decision = crate::Decision {
            id: "wide".into(),
            instruction: "Choose one".into(),
            kind: DecisionKind::Choice { options },
        };
        let result = parse_result(
            &decision,
            &json!({"choices":[{"finish_reason":"stop","message":{"content":"BA"}}]}),
        )
        .unwrap();
        assert_eq!(result.selected_code.as_deref(), Some("BA"));
        assert!(
            matches!(result.value, RemoteDecisionValue::Choice { selected: Some(ref id) } if id == "option_26")
        );
        let invalid = parse_result(
            &decision,
            &json!({"choices":[{"finish_reason":"stop","message":{"content":"A"}}]}),
        )
        .unwrap();
        assert!(matches!(
            invalid.status,
            RemoteStatus::AbstainedInvalidOutput
        ));

        let ordinal = crate::Decision {
            id: "level".into(),
            instruction: "Choose level".into(),
            kind: DecisionKind::Ordinal {
                levels: vec![
                    crate::Level {
                        id: "low".into(),
                        criterion: "low".into(),
                        value: 1.0,
                    },
                    crate::Level {
                        id: "high".into(),
                        criterion: "high".into(),
                        value: 5.0,
                    },
                ],
            },
        };
        let result = parse_result(
            &ordinal,
            &json!({"choices":[{"finish_reason":"stop","message":{"content":"B"}}]}),
        )
        .unwrap();
        assert!(
            matches!(result.value, RemoteDecisionValue::Ordinal { selected: Some(ref id), level_value: Some(5.0) } if id == "high")
        );

        let binary = crate::Decision {
            id: "flag".into(),
            instruction: "Choose".into(),
            kind: DecisionKind::Binary {
                false_label: "no".into(),
                true_label: "yes".into(),
            },
        };
        let result = parse_result(
            &binary,
            &json!({"choices":[{"finish_reason":"stop","message":{"content":"B"}}]}),
        )
        .unwrap();
        assert!(matches!(
            result.value,
            RemoteDecisionValue::Binary { value: Some(true) }
        ));
    }

    #[test]
    fn sends_text_and_image_to_mock_openrouter() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!(
            "http://{}/api/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let server = std::thread::spawn(move || {
            for image_expected in [false, true] {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let header_end = loop {
                    let mut buffer = [0; 4096];
                    let n = stream.read(&mut buffer).unwrap();
                    bytes.extend_from_slice(&buffer[..n]);
                    if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&bytes[..header_end]).to_ascii_lowercase();
                assert!(headers.contains("authorization: bearer test-key"));
                let length: usize = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length: "))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                while bytes.len() - header_end < length {
                    let mut buffer = [0; 4096];
                    let n = stream.read(&mut buffer).unwrap();
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let body: Value =
                    serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
                assert_eq!(body["model"], "example/model");
                assert_eq!(body["messages"][0]["role"], "system");
                if image_expected {
                    assert_eq!(body["reasoning"]["effort"], "none");
                    assert!(
                        body["messages"][1]["content"][1]["image_url"]["url"]
                            .as_str()
                            .unwrap()
                            .starts_with("data:image/png;base64,")
                    );
                } else {
                    assert!(body.get("reasoning").is_none());
                    assert!(body["messages"][1]["content"].is_string());
                }
                let response = br#"{"choices":[{"finish_reason":"stop","message":{"content":"A"}}],"model":"example/model","usage":{"prompt_tokens":42,"completion_tokens":1}}"#;
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len()).unwrap();
                stream.write_all(response).unwrap();
            }
        });
        let mut backend = OpenRouterBackend::new("example/model", "test-key").unwrap();
        backend.endpoint = endpoint;
        let text = backend.decide(&request(), None).unwrap();
        assert_eq!(text.results[0].selected_code.as_deref(), Some("A"));
        backend.set_reasoning_effort("none").unwrap();
        let image = backend
            .decide(
                &request(),
                Some(include_bytes!("../tests/fixtures/vision_red_64.png")),
            )
            .unwrap();
        assert_eq!(image.results[0].selected_code.as_deref(), Some("A"));
        server.join().unwrap();
    }
}
