//! TypeSafe-compatible `/v1/systemone` and `/v1/models`, so TypeSafe SDK
//! clients can call L2S1 by changing only their base URL. Questions are mapped to
//! native decisions (`noul` → binary, `choice` → choice, `score` → ordinal with
//! levels 0..n-1) and answers use TypeSafe's shapes, rounded to four decimals.
//! Native abstention policy is not applied here: TypeSafe answers carry
//! probabilities only. Criteria order is preserved because it fixes answer codes.
use crate::Error;
use serde::{Deserialize, Serialize, de};
use serde_json::{Value, json};
use std::fmt;

const MAX_QUESTIONS: usize = super::MAX_DECISIONS;

/// A JSON object whose key order is kept (serde_json here sorts keys).
struct Ordered<T = Value>(Vec<(String, T)>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Ordered<T> {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<T>(std::marker::PhantomData<T>);
        impl<'de, T: Deserialize<'de>> de::Visitor<'de> for V<T> {
            type Value = Ordered<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Ordered<T>, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Ordered(entries))
            }
        }
        d.deserialize_map(V(std::marker::PhantomData))
    }
}

impl<T: Serialize> Serialize for Ordered<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_map(self.0.iter().map(|(k, v)| (k, v)))
    }
}

type Raw = Box<serde_json::value::RawValue>;

/// An object with key order kept, or `None` for any other JSON value.
fn object<T: for<'de> Deserialize<'de>>(raw: &Raw) -> Option<Ordered<T>> {
    raw.get()
        .trim_start()
        .starts_with('{')
        .then(|| serde_json::from_str(raw.get()).ok())
        .flatten()
}

fn value(raw: &Raw) -> Value {
    serde_json::from_str(raw.get()).unwrap_or(Value::Null)
}

#[derive(Deserialize)]
struct Body {
    #[serde(default)]
    model: Option<Value>,
    #[serde(default)]
    state: Option<Value>,
    #[serde(default)]
    questions: Option<Raw>,
}

pub(super) enum Kind {
    Noul,
    Choice(Vec<String>),
    Score(Vec<Value>),
}

pub(super) struct Translated {
    pub body: Vec<u8>,
    pub model: String,
    questions: Vec<(String, Kind)>,
}

pub(super) struct Failure {
    pub status: u16,
    pub code: &'static str,
    pub error: String,
    pub detail: Option<Vec<Value>>,
}

impl Failure {
    pub fn new(status: u16, code: &'static str, error: impl Into<String>) -> Self {
        Self {
            status,
            code,
            error: error.into(),
            detail: None,
        }
    }
    fn issues(code: &'static str, detail: Vec<Value>) -> Self {
        // As the TypeSafe SDK's extract_message renders a FastAPI detail list.
        let error = detail
            .iter()
            .map(|issue| {
                let loc = issue["loc"].as_array().into_iter().flatten().skip(1);
                let loc = loc
                    .map(|p| p.as_str().map_or_else(|| p.to_string(), str::to_owned))
                    .collect::<Vec<_>>()
                    .join(".");
                format!("{loc}: {}", issue["msg"].as_str().unwrap_or_default())
            })
            .collect::<Vec<_>>()
            .join("; ");
        Self {
            status: 422,
            code,
            error,
            detail: Some(detail),
        }
    }
    pub fn body(&self) -> Value {
        let mut body = json!({"error": self.error, "code": self.code});
        if let Some(detail) = &self.detail {
            body["detail"] = json!(detail);
        }
        body
    }
    /// Native runtime errors in TypeSafe's status and code vocabulary.
    pub fn from_native(error: &Error) -> Self {
        match error {
            Error::Invalid(m) if m.contains("input was not truncated") => Self::issues(
                "STATE_TRUNCATED",
                vec![issue(&["state"], m, "state_truncated", None)],
            ),
            Error::Invalid(m) => {
                Self::issues("INVALID_REQUEST", vec![issue(&[], m, "value_error", None)])
            }
            Error::Backend(m) if m.starts_with("deadline_exceeded:") => {
                Self::new(504, "DEADLINE_EXCEEDED", m.as_str())
            }
            Error::ModelLoad(m) => Self::new(500, "MODEL_LOAD_FAILED", m.as_str()),
            Error::Upstream(m) => Self::new(502, "UPSTREAM_ERROR", m.as_str()),
            other => Self::new(500, "INFERENCE_FAILED", other.to_string()),
        }
    }
}

fn issue(loc: &[&str], msg: &str, kind: &str, ctx: Option<Value>) -> Value {
    let mut path = vec![json!("body")];
    path.extend(loc.iter().map(|p| json!(p)));
    let mut issue = json!({"loc": path, "msg": msg, "type": kind});
    if let Some(ctx) = ctx {
        issue["ctx"] = ctx;
    }
    issue
}

/// Strings verbatim; objects and arrays as compact JSON; `null`/empty uses `fallback`.
fn text(value: Option<&Value>, fallback: &str) -> Option<String> {
    match value {
        None | Some(Value::Null) => Some(fallback.into()),
        Some(Value::String(s)) if s.trim().is_empty() => Some(fallback.into()),
        Some(Value::String(s)) => Some(s.clone()),
        Some(v @ (Value::Object(_) | Value::Array(_))) => Some(v.to_string()),
        Some(_) => None,
    }
}

/// The model name this server answers to: a GGUF file stem, else the backend model.
pub(super) fn served_name(capabilities: &Value) -> String {
    let model = capabilities
        .pointer("/backend/model")
        .and_then(Value::as_str)
        .unwrap_or("l2s1");
    let path = std::path::Path::new(model);
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("gguf") => path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(model)
            .into(),
        _ => model.into(),
    }
}

pub(super) fn models(capabilities: &Value) -> Value {
    let model = capabilities
        .pointer("/backend/model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let modified = std::fs::metadata(model)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs() / 86_400);
    let runtime = capabilities
        .pointer("/backend/runtime")
        .and_then(Value::as_str)
        .unwrap_or("l2s1");
    json!({"models":[{"name": served_name(capabilities),
        "description": format!("L2S1 {runtime}: {model}"),
        "release_date": civil_date(modified as i64)}]})
}

/// Days since 1970-01-01 to `YYYY-MM-DD` (Howard Hinnant's civil_from_days).
fn civil_date(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Validate a TypeSafe body (all issues at once) and build the native request.
pub(super) fn translate(body: &[u8], served: &str) -> Result<Translated, Failure> {
    let body: Body = serde_json::from_slice(body)
        .map_err(|_| Failure::new(400, "INVALID_JSON", "body is not a valid JSON object"))?;
    let mut issues = Vec::new();
    let model = match &body.model {
        None => {
            issues.push(issue(&["model"], "Field required", "missing", None));
            String::new()
        }
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::String(_)) => {
            issues.push(issue(
                &["model"],
                "String should have at least 1 character",
                "string_too_short",
                Some(json!({"min_length":1})),
            ));
            String::new()
        }
        Some(_) => {
            issues.push(issue(
                &["model"],
                "Input should be a valid string",
                "string_type",
                None,
            ));
            String::new()
        }
    };
    match &body.state {
        None => issues.push(issue(&["state"], "Field required", "missing", None)),
        Some(Value::String(_) | Value::Object(_) | Value::Array(_)) => {}
        Some(_) => issues.push(issue(
            &["state"],
            "state must be a string, object or array",
            "state_type",
            None,
        )),
    }
    let questions: Vec<(String, Raw)> = match &body.questions {
        None => {
            issues.push(issue(&["questions"], "Field required", "missing", None));
            vec![]
        }
        Some(raw) => match object(raw) {
            Some(Ordered(q)) => q,
            None => {
                issues.push(issue(
                    &["questions"],
                    "Input should be a valid dictionary",
                    "dict_type",
                    None,
                ));
                vec![]
            }
        },
    };
    if body.questions.is_some()
        && !(1..=MAX_QUESTIONS).contains(&questions.len())
        && issues.iter().all(|i| i["loc"][1] != "questions")
    {
        let (kind, msg, ctx) = if questions.is_empty() {
            (
                "too_short",
                "Dictionary should have at least 1 item after validation, not 0".to_string(),
                json!({"field_type":"Dictionary","min_length":1,"actual_length":0}),
            )
        } else {
            (
                "too_long",
                format!(
                    "Dictionary should have at most {MAX_QUESTIONS} items after validation, not {}",
                    questions.len()
                ),
                json!({"field_type":"Dictionary","max_length":MAX_QUESTIONS,"actual_length":questions.len()}),
            )
        };
        issues.push(issue(&["questions"], &msg, kind, Some(ctx)));
    }
    let mut decisions = Vec::new();
    let mut kinds = Vec::new();
    for (id, question) in &questions {
        let Some(Ordered::<Raw>(fields)) = object(question) else {
            issues.push(issue(
                &["questions", id],
                "Input should be a valid dictionary",
                "dict_type",
                None,
            ));
            continue;
        };
        let raw = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        let get = |key: &str| raw(key).map(value);
        let Some(Value::String(qtype)) = get("type") else {
            issues.push(issue(
                &["questions", id],
                "Unable to extract tag using discriminator 'type'",
                "union_tag_not_found",
                Some(json!({"discriminator":"'type'"})),
            ));
            continue;
        };
        let loc = |field: &'static str| {
            [
                String::from("questions"),
                id.clone(),
                qtype.clone(),
                field.into(),
            ]
        };
        let push = |issues: &mut Vec<Value>,
                    field: &'static str,
                    msg: &str,
                    kind: &str,
                    ctx: Option<Value>| {
            let loc = loc(field);
            issues.push(issue(
                &loc.iter()
                    .map(String::as_str)
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
                msg,
                kind,
                ctx,
            ));
        };
        let Some(instruction) = text(get("instructions").as_ref(), id) else {
            push(
                &mut issues,
                "instructions",
                "Input should be a valid string, object or array",
                "json_type",
                None,
            );
            continue;
        };
        let criteria = get("criteria");
        let criteria = criteria.as_ref();
        let (kind, native) = match qtype.as_str() {
            "noul" => {
                let pick = |key: &str, fallback: &str| match criteria {
                    Some(Value::Object(m)) => text(m.get(key), fallback),
                    _ => Some(fallback.into()),
                };
                if !matches!(criteria, None | Some(Value::Null | Value::Object(_))) {
                    push(
                        &mut issues,
                        "criteria",
                        "Input should be a valid dictionary",
                        "dict_type",
                        None,
                    );
                    continue;
                }
                let (Some(f), Some(t)) = (pick("false", "No"), pick("true", "Yes")) else {
                    push(
                        &mut issues,
                        "criteria",
                        "Input should be a valid string, object or array",
                        "json_type",
                        None,
                    );
                    continue;
                };
                (
                    Kind::Noul,
                    json!({"type":"binary","false_label":f,"true_label":t}),
                )
            }
            "choice" => {
                let labels: Vec<(String, Option<String>)> = match criteria {
                    Some(Value::Object(_)) => {
                        let Some(Ordered::<Value>(m)) = raw("criteria").and_then(object) else {
                            continue;
                        };
                        m.into_iter()
                            .map(|(k, v)| {
                                let t = text(Some(&v), &k);
                                (k, t)
                            })
                            .collect()
                    }
                    // Extension: an array of labels; duplicates collapse onto the first.
                    Some(Value::Array(a)) if a.iter().all(Value::is_string) => {
                        let mut seen = Vec::new();
                        for l in a.iter().filter_map(Value::as_str) {
                            if !seen.iter().any(|(k, _): &(String, _)| k == l) {
                                seen.push((l.to_string(), Some(l.to_string())));
                            }
                        }
                        seen
                    }
                    None => {
                        push(&mut issues, "criteria", "Field required", "missing", None);
                        continue;
                    }
                    Some(_) => {
                        push(
                            &mut issues,
                            "criteria",
                            "Input should be a valid dictionary",
                            "dict_type",
                            None,
                        );
                        continue;
                    }
                };
                if labels.iter().any(|(_, t)| t.is_none()) {
                    push(
                        &mut issues,
                        "criteria",
                        "Input should be a valid string, object or array",
                        "json_type",
                        None,
                    );
                    continue;
                }
                if !(2..=255).contains(&labels.len()) {
                    let (kind, bound) = if labels.len() < 2 {
                        ("too_short", json!({"min_length":2}))
                    } else {
                        ("too_long", json!({"max_length":255}))
                    };
                    let mut ctx = bound;
                    ctx["field_type"] = json!("Dictionary");
                    ctx["actual_length"] = json!(labels.len());
                    push(
                        &mut issues,
                        "criteria",
                        &format!(
                            "Dictionary should have between 2 and 255 items after validation, not {}",
                            labels.len()
                        ),
                        kind,
                        Some(ctx),
                    );
                    continue;
                }
                let options = labels
                    .iter()
                    .map(|(k, t)| json!({"id":k,"criterion":t}))
                    .collect::<Vec<_>>();
                (
                    Kind::Choice(labels.into_iter().map(|(k, _)| k).collect()),
                    json!({"type":"choice","options":options}),
                )
            }
            "score" => {
                let Some(Value::Array(levels)) = criteria else {
                    let (msg, kind) = if criteria.is_none() {
                        ("Field required", "missing")
                    } else {
                        ("Input should be a valid list", "list_type")
                    };
                    push(&mut issues, "criteria", msg, kind, None);
                    continue;
                };
                if !(2..=10).contains(&levels.len()) {
                    let (kind, bound) = if levels.len() < 2 {
                        ("too_short", json!({"min_length":2}))
                    } else {
                        ("too_long", json!({"max_length":10}))
                    };
                    let mut ctx = bound;
                    ctx["field_type"] = json!("List");
                    ctx["actual_length"] = json!(levels.len());
                    let msg = if levels.len() < 2 {
                        format!(
                            "List should have at least 2 items after validation, not {}",
                            levels.len()
                        )
                    } else {
                        format!(
                            "List should have at most 10 items after validation, not {}",
                            levels.len()
                        )
                    };
                    push(&mut issues, "criteria", &msg, kind, Some(ctx));
                    continue;
                }
                let mut native = Vec::new();
                for (i, level) in levels.iter().enumerate() {
                    match text(Some(level).filter(|l| !l.is_null()), "") {
                        Some(t) if !t.is_empty() => {
                            native.push(json!({"id":i.to_string(),"criterion":t,"value":i}))
                        }
                        _ => {
                            push(
                                &mut issues,
                                "criteria",
                                "Input should be a valid string, object or array",
                                "json_type",
                                None,
                            );
                            break;
                        }
                    }
                }
                if native.len() != levels.len() {
                    continue;
                }
                (
                    Kind::Score(levels.clone()),
                    json!({"type":"ordinal","levels":native}),
                )
            }
            other => {
                issues.push(issue(&["questions", id], &format!("Input tag '{other}' found using 'type' does not match any of the expected tags: 'choice', 'score', 'noul'"), "union_tag_invalid", Some(json!({"discriminator":"'type'","tag":other,"expected_tags":"'choice', 'score', 'noul'"}))));
                continue;
            }
        };
        decisions.push(json!({"id":id,"instruction":instruction,"kind":native}));
        kinds.push((id.clone(), kind));
    }
    if !issues.is_empty() {
        return Err(Failure::issues("INVALID_REQUEST", issues));
    }
    if model != served {
        return Err(Failure::new(
            404,
            "MODEL_NOT_FOUND",
            format!("model \"{model}\" not found; this server answers as \"{served}\""),
        ));
    }
    let native = json!({"state": body.state, "decisions": decisions});
    Ok(Translated {
        body: serde_json::to_vec(&native).expect("serializable request"),
        model,
        questions: kinds,
    })
}

fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Answer {
    Choice {
        choice: String,
        confidence: f64,
        probabilities: Ordered,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: Ordered,
        probabilities: Ordered,
    },
    Noul {
        noul: f64,
    },
}

/// Native response → TypeSafe `{model, answers, usage}` JSON bytes, keys in request order.
pub(super) fn answers(request: &Translated, output: &Value) -> crate::Result<Vec<u8>> {
    let results = output["results"]
        .as_array()
        .ok_or_else(|| Error::Backend("backend returned no decision results".into()))?;
    let mut answers = Vec::new();
    let mut input_tokens = 0;
    for (id, kind) in &request.questions {
        let result = results
            .iter()
            .find(|r| r["id"] == **id)
            .ok_or_else(|| Error::Backend(format!("backend returned no result for {id}")))?;
        let ids: Vec<String> = match kind {
            Kind::Noul => vec!["false".into(), "true".into()],
            Kind::Choice(labels) => labels.clone(),
            Kind::Score(levels) => (0..levels.len()).map(|i| i.to_string()).collect(),
        };
        let p = ids
            .iter()
            .map(|option| {
                result["evidence"]["scores"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|s| s["id"] == **option)
                    .and_then(|s| s["option_probability"].as_f64())
                    .filter(|p| p.is_finite() && *p >= 0.0)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::Backend(format!("backend returned no option probabilities for {id}"))
            })?;
        let total: f64 = p.iter().sum();
        if total <= 0.0 {
            return Err(Error::Backend(format!("zero probability mass for {id}")));
        }
        let p = p.iter().map(|v| v / total).collect::<Vec<_>>();
        let k = p.len() as f64;
        let top = p
            .iter()
            .enumerate()
            .fold(0, |best, (i, v)| if *v > p[best] { i } else { best });
        let confidence = round4(((k * p[top] - 1.0) / (k - 1.0)).clamp(0.0, 1.0));
        let probabilities = || {
            Ordered(
                ids.iter()
                    .cloned()
                    .zip(p.iter().map(|v| json!(round4(*v))))
                    .collect(),
            )
        };
        input_tokens += result["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let answer = match kind {
            Kind::Noul => Answer::Noul { noul: round4(p[1]) },
            Kind::Choice(labels) => Answer::Choice {
                choice: labels[top].clone(),
                confidence,
                probabilities: probabilities(),
            },
            Kind::Score(levels) => Answer::Score {
                score: round4(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()),
                confidence,
                legend: Ordered(ids.iter().cloned().zip(levels.iter().cloned()).collect()),
                probabilities: probabilities(),
            },
        };
        answers.push((id.clone(), answer));
    }
    #[derive(Serialize)]
    struct Response<'a> {
        model: &'a str,
        answers: Ordered<Answer>,
        usage: Value,
    }
    Ok(serde_json::to_vec(&Response {
        model: &request.model,
        answers: Ordered(answers),
        usage: json!({"input_tokens": input_tokens, "output_tokens": 0}),
    })
    .expect("serializable response"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &str = r#"{"model":"gemma","state":"I was charged twice",
        "questions":{"zeta":{"type":"choice","criteria":{"refund":"Wants money back","invoice":{"needs":"receipt"},"other":null}},
        "billing":{"type":"noul","instructions":"Is this about billing?"},
        "urgency":{"type":"score","instructions":"How urgent?","criteria":["Can wait","Today","Now"]}}}"#;

    #[test]
    fn questions_map_to_native_decisions_in_request_order() {
        let t = translate(BODY.as_bytes(), "gemma").ok().unwrap();
        let native: Value = serde_json::from_slice(&t.body).unwrap();
        let d = &native["decisions"];
        assert_eq!(d[0]["id"], "zeta");
        assert_eq!(
            d[0]["instruction"], "zeta",
            "missing instructions read the id"
        );
        let ids: Vec<_> = d[0]["kind"]["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            ["refund", "invoice", "other"],
            "criteria order fixes answer codes"
        );
        assert_eq!(
            d[0]["kind"]["options"][1]["criterion"],
            r#"{"needs":"receipt"}"#
        );
        assert_eq!(d[0]["kind"]["options"][2]["criterion"], "other");
        assert_eq!(
            d[1]["kind"],
            json!({"type":"binary","false_label":"No","true_label":"Yes"})
        );
        assert_eq!(
            d[2]["kind"]["levels"][2],
            json!({"id":"2","criterion":"Now","value":2})
        );
        assert_eq!(native["state"], "I was charged twice");
    }

    #[test]
    fn answers_use_typesafe_shapes_and_order() {
        let t = translate(BODY.as_bytes(), "gemma").ok().unwrap();
        let score = |id: &str, p: f64| json!({"id": id, "option_probability": p});
        let output = json!({"results":[
            {"id":"urgency","usage":{"input_tokens":5},"evidence":{"scores":[score("0",0.2),score("1",0.3),score("2",0.5)]}},
            {"id":"zeta","usage":{"input_tokens":7},"evidence":{"scores":[score("refund",0.6),score("invoice",0.3),score("other",0.1)]}},
            {"id":"billing","usage":{"input_tokens":3},"evidence":{"scores":[score("false",0.25),score("true",0.75)]}}]});
        let body = String::from_utf8(answers(&t, &output).unwrap()).unwrap();
        assert_eq!(
            body,
            concat!(
                r#"{"model":"gemma","answers":{"#,
                r#""zeta":{"type":"choice","choice":"refund","confidence":0.4,"probabilities":{"refund":0.6,"invoice":0.3,"other":0.1}},"#,
                r#""billing":{"type":"noul","noul":0.75},"#,
                r#""urgency":{"type":"score","score":1.3,"confidence":0.25,"legend":{"0":"Can wait","1":"Today","2":"Now"},"probabilities":{"0":0.2,"1":0.3,"2":0.5}}},"#,
                r#""usage":{"input_tokens":15,"output_tokens":0}}"#
            )
        );
    }

    #[test]
    fn every_issue_is_reported_and_errors_map_to_typesafe_codes() {
        let bad = r#"{"model":"gemma","questions":{"u":{"type":"score","criteria":["Can wait"]},"k":{"type":"maybe"}}}"#;
        let failure = translate(bad.as_bytes(), "gemma").err().unwrap();
        assert_eq!((failure.status, failure.code), (422, "INVALID_REQUEST"));
        let types: Vec<_> = failure
            .detail
            .unwrap()
            .iter()
            .map(|i| i["type"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(types, ["missing", "too_short", "union_tag_invalid"]);
        assert!(failure.error.starts_with(
            "state: Field required; questions.u.score.criteria: List should have at least 2"
        ));
        assert_eq!(
            translate(b"[1", "gemma").err().unwrap().code,
            "INVALID_JSON"
        );
        assert_eq!(
            translate(BODY.as_bytes(), "other").err().unwrap().code,
            "MODEL_NOT_FOUND"
        );
        let overflow =
            Error::Invalid("q has 9 input tokens, context limit 4; input was not truncated".into());
        assert_eq!(Failure::from_native(&overflow).code, "STATE_TRUNCATED");
        assert_eq!(
            served_name(&json!({"backend":{"model":"/m/gemma-4.Q8_0.gguf"}})),
            "gemma-4.Q8_0"
        );
        assert_eq!(civil_date(20_727), "2026-10-01");
    }
}
