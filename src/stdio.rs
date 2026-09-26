//! Resident compiled-engine RPC over stdin/stdout. No socket or HTTP server.
use crate::http::HttpDecisionBackend;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, Read, Write};

const MAX_LINE: u64 = 44 * 1024 * 1024 + 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    id: String,
    op: String,
    body: Option<Box<serde_json::value::RawValue>>,
}

pub fn serve<B: HttpDecisionBackend>(backend: &mut B) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("l2s1 stdio ready");
    serve_stream(backend, std::io::stdin().lock(), std::io::stdout().lock())
}

pub fn serve_stream<B: HttpDecisionBackend, R: BufRead, W: Write>(
    backend: &mut B,
    mut input: R,
    mut output: W,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let mut line = Vec::new();
        if input
            .by_ref()
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut line)?
            == 0
        {
            return Ok(());
        }
        if line.len() as u64 > MAX_LINE {
            return Err("stdio request exceeds 44 MiB envelope limit".into());
        }
        let call: Call = serde_json::from_slice(&line)?;
        let body = call
            .body
            .as_ref()
            .map(|value| value.get().as_bytes())
            .unwrap_or(b"null");
        let result = match call.op.as_str() {
            "health" => Ok(json!({"status":"ok"})),
            "capabilities" => Ok(backend.capabilities()),
            "decide" => crate::http::dispatch_wire(backend, body, false, &call.id),
            "decide_batch" => crate::http::dispatch_wire(backend, body, true, &call.id),
            _ => Err(crate::Error::Invalid("unknown stdio operation".into())),
        };
        let response = match result {
            Ok(value) => json!({"id":call.id,"result":value}),
            Err(error) => {
                let code = match &error {
                    crate::Error::Invalid(message) if message.starts_with("batch_unsupported:") => {
                        "batch_unsupported"
                    }
                    crate::Error::Invalid(message) if message.starts_with("batch_not_enabled:") => {
                        "batch_not_enabled"
                    }
                    crate::Error::Invalid(_) => "invalid_request",
                    crate::Error::Backend(message) if message.starts_with("reasoning_limit:") => {
                        "reasoning_limit"
                    }
                    crate::Error::Backend(_) => "backend_error",
                    crate::Error::ModelLoad(_) => "model_load_failed",
                    crate::Error::Upstream(_) => "upstream_error",
                };
                let mut envelope = json!({"id":call.id,"error":{"code":code,"message":error.to_string(),"request_id":call.id}});
                let messages = crate::http::user_failure_messages(body);
                let reason = match code {
                    "reasoning_limit" => messages.get("reasoning_limit"),
                    "backend_error" => messages.get("native_failure"),
                    _ => None,
                };
                if let Some(reason) = reason {
                    envelope["error"]["user_reason"] = json!(reason);
                }
                envelope
            }
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
}
