//! Named decision sets loaded at startup. A request may send `"preset": "name"`
//! in place of `decisions`; the preset's decisions are inserted verbatim before
//! the ordinary request validation, so results are identical to sending them.
// ponytail: one process-global registry (one server per process), loaded once
// from files; add runtime create/delete only if a client needs it.
use crate::{Decision, DecisionRequest, Error, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{borrow::Cow, collections::BTreeMap, path::Path, sync::RwLock};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Preset {
    name: String,
    #[serde(default)]
    description: String,
    decisions: Vec<Decision>,
}

static PRESETS: RwLock<BTreeMap<String, Preset>> = RwLock::new(BTreeMap::new());

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Register a preset from JSON bytes; `source` names it in errors.
pub fn register(bytes: &[u8], source: &str) -> Result<()> {
    let preset: Preset = serde_json::from_slice(bytes)
        .map_err(|e| Error::Invalid(format!("preset {source}: {e}")))?;
    if !valid_name(&preset.name) {
        return Err(Error::Invalid(format!(
            "preset {source}: name must be 1-64 of a-z, 0-9, '-' or '_', starting with a letter or digit"
        )));
    }
    DecisionRequest {
        shared: None,
        state: json!({}),
        decisions: preset.decisions.clone(),
    }
    .validate()
    .map_err(|e| Error::Invalid(format!("preset {source}: {e}")))?;
    let mut presets = PRESETS.write().expect("preset registry");
    if presets.contains_key(&preset.name) {
        return Err(Error::Invalid(format!(
            "preset {source}: duplicate name {}",
            preset.name
        )));
    }
    presets.insert(preset.name.clone(), preset);
    Ok(())
}

pub fn register_file(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Invalid(format!("preset {}: {e}", path.display())))?;
    register(&bytes, &path.display().to_string())
}

/// Registered presets for capability listings: name, description and decision IDs.
pub fn list() -> Value {
    let presets = PRESETS.read().expect("preset registry");
    presets
        .values()
        .map(|p| {
            json!({"name":p.name,"description":p.description,
                "decisions":p.decisions.iter().map(|d| &d.id).collect::<Vec<_>>()})
        })
        .collect()
}

fn expand_request(request: &mut Value) -> Result<()> {
    let Some(name) = request.as_object_mut().and_then(|r| r.remove("preset")) else {
        return Ok(());
    };
    let Value::String(name) = name else {
        return Err(Error::Invalid("preset must be a string".into()));
    };
    if request.get("decisions").is_some() {
        return Err(Error::Invalid(
            "send either preset or decisions, not both".into(),
        ));
    }
    let presets = PRESETS.read().expect("preset registry");
    let preset = presets
        .get(&name)
        .ok_or_else(|| Error::Invalid(format!("unknown preset {name:?}")))?;
    request["decisions"] = serde_json::to_value(&preset.decisions).expect("decisions");
    Ok(())
}

/// Replace `preset` with its decisions in a request (or each batch request).
/// Bodies without a `"preset"` key are returned unchanged and unparsed.
pub fn expand(body: &[u8], batch: bool) -> Result<Cow<'_, [u8]>> {
    if !body.windows(8).any(|w| w == b"\"preset\"") {
        return Ok(Cow::Borrowed(body));
    }
    let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
        // Let the normal request parser report the syntax error.
        return Ok(Cow::Borrowed(body));
    };
    if batch {
        for request in value
            .get_mut("requests")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            expand_request(request)?;
        }
    } else {
        expand_request(&mut value)?;
    }
    Ok(Cow::Owned(serde_json::to_vec(&value).expect("request")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_expand_validate_and_list() {
        let preset = br#"{"name":"spam-check","description":"Spam?","decisions":[
            {"id":"spam","instruction":"Is this spam?","kind":{"type":"binary","false_label":"No","true_label":"Yes"}}]}"#;
        register(preset, "test").unwrap();
        assert!(
            register(preset, "again")
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        let renamed = String::from_utf8(preset.to_vec())
            .unwrap()
            .replace("spam-check", "Bad Name");
        assert!(register(renamed.as_bytes(), "x").is_err());
        let one_option = br#"{"name":"bad","decisions":[{"id":"a","instruction":"?","kind":{"type":"choice","options":[{"id":"x","criterion":"x"}]}}]}"#;
        assert!(register(one_option, "x").is_err());

        let plain = br#"{"state":{},"decisions":[]}"#;
        assert!(matches!(expand(plain, false).unwrap(), Cow::Borrowed(_)));
        let body = expand(br#"{"state":"buy now","preset":"spam-check"}"#, false).unwrap();
        let request: DecisionRequest = serde_json::from_slice(&body).unwrap();
        assert_eq!(request.decisions[0].id, "spam");
        let batch = expand(
            br#"{"requests":[{"state":"a","preset":"spam-check"},{"state":"b","decisions":[]}]}"#,
            true,
        )
        .unwrap();
        let batch: Value = serde_json::from_slice(&batch).unwrap();
        assert_eq!(batch["requests"][0]["decisions"][0]["id"], "spam");
        assert!(batch["requests"][0].get("preset").is_none());
        for bad in [
            &br#"{"state":{},"preset":"missing"}"#[..],
            br#"{"state":{},"preset":"spam-check","decisions":[]}"#,
            br#"{"state":{},"preset":1}"#,
        ] {
            assert!(matches!(expand(bad, false), Err(Error::Invalid(_))));
        }
        assert_eq!(list()[0]["decisions"], json!(["spam"]));
    }
}
