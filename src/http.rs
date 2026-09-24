//! Small synchronous JSON API. One owned backend serializes GPU access.
use base64::Engine;
use l2s1::{DecisionBackend, DecisionRequest, Error, llama::LlamaBackend};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    time::Duration,
};

const MAX_HEADER: usize = 16 * 1024;
const MAX_BODY: usize = 12 * 1024 * 1024;
const MAX_IMAGE_BASE64: usize = 11 * 1024 * 1024;

pub fn serve(address: &str, backend: &mut LlamaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(address)?;
    eprintln!("l2s1 HTTP listening on {}", listener.local_addr()?);
    for connection in listener.incoming() {
        match connection {
            Ok(mut stream) => {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
                if let Err(error) = handle(&mut stream, backend) {
                    eprintln!("HTTP connection error: {error}");
                }
            }
            Err(error) => eprintln!("HTTP accept error: {error}"),
        }
    }
    Ok(())
}

fn handle(stream: &mut TcpStream, backend: &mut LlamaBackend) -> std::io::Result<()> {
    let request = match read_request(stream) {
        Ok(request) => request,
        Err((status, message)) => {
            return respond(stream, status, &serde_json::json!({"error": message}));
        }
    };
    if request.method == "GET" && request.path == "/healthz" {
        return respond(stream, 200, &serde_json::json!({"status": "ok"}));
    }
    if request.method != "POST" || request.path != "/v1/decisions" {
        return respond(
            stream,
            404,
            &serde_json::json!({"error": "route not found"}),
        );
    }
    let content_type = request.content_type.to_ascii_lowercase();
    if content_type != "application/json" && !content_type.starts_with("application/json;") {
        return respond(
            stream,
            415,
            &serde_json::json!({"error": "Content-Type must be application/json"}),
        );
    }
    let result = run_request(backend, &request.body);
    match result {
        Ok(output) => respond(stream, 200, &output),
        Err(Error::Invalid(message)) => {
            respond(stream, 400, &serde_json::json!({"error": message}))
        }
        Err(Error::Backend(message)) => {
            respond(stream, 422, &serde_json::json!({"error": message}))
        }
    }
}

fn run_request(backend: &mut LlamaBackend, body: &[u8]) -> l2s1::Result<serde_json::Value> {
    let mut value: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| Error::Invalid(format!("invalid JSON: {e}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| Error::Invalid("request must be a JSON object".into()))?;
    let image = object.remove("image_base64");
    let request: DecisionRequest = serde_json::from_value(value)
        .map_err(|e| Error::Invalid(format!("invalid decision request: {e}")))?;
    let response = if let Some(image) = image {
        let encoded = image
            .as_str()
            .ok_or_else(|| Error::Invalid("image_base64 must be a string".into()))?;
        if encoded.len() > MAX_IMAGE_BASE64 {
            return Err(Error::Invalid("image_base64 exceeds size limit".into()));
        }
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Error::Invalid("image_base64 is not valid standard base64".into()))?;
        backend.decide_vision(&request, &decoded)?
    } else {
        backend.decide(&request)?
    };
    serde_json::to_value(response).map_err(|e| Error::Backend(e.to_string()))
}

struct HttpRequest {
    method: String,
    path: String,
    content_type: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, (u16, &'static str)> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    let header_end = loop {
        let count = stream
            .read(&mut buffer)
            .map_err(|_| (400, "request read failed"))?;
        if count == 0 {
            return Err((400, "incomplete request"));
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            if index > MAX_HEADER {
                return Err((431, "request headers too large"));
            }
            break index + 4;
        }
        if bytes.len() > MAX_HEADER {
            return Err((431, "request headers too large"));
        }
    };
    let header =
        std::str::from_utf8(&bytes[..header_end]).map_err(|_| (400, "invalid request headers"))?;
    let mut lines = header.split("\r\n");
    let first = lines.next().ok_or((400, "missing request line"))?;
    let mut words = first.split_ascii_whitespace();
    let method = words.next().ok_or((400, "missing method"))?;
    let path = words.next().ok_or((400, "missing path"))?;
    let version = words.next().ok_or((400, "missing HTTP version"))?;
    if words.next().is_some() || !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err((400, "invalid request line"));
    }
    let mut content_length = None;
    let mut content_type = "";
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line.split_once(':').ok_or((400, "invalid header"))?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return Err((400, "duplicate Content-Length"));
            }
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| (400, "invalid Content-Length"))?,
            );
        } else if name.eq_ignore_ascii_case("content-type") {
            content_type = value;
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err((400, "Transfer-Encoding is unsupported"));
        }
    }
    let length = if method == "GET" {
        content_length.unwrap_or(0)
    } else {
        content_length.ok_or((411, "Content-Length required"))?
    };
    if length > MAX_BODY {
        return Err((413, "request body too large"));
    }
    let mut body = bytes[header_end..].to_vec();
    if body.len() > length {
        body.truncate(length);
    }
    if body.len() < length {
        body.resize(length, 0);
        stream
            .read_exact(&mut body[bytes.len().saturating_sub(header_end)..])
            .map_err(|_| (400, "incomplete request body"))?;
    }
    Ok(HttpRequest {
        method: method.into(),
        path: path.into(),
        content_type: content_type.into(),
        body,
    })
}

fn respond(stream: &mut TcpStream, status: u16, body: &serde_json::Value) -> std::io::Result<()> {
    let json = serde_json::to_vec(body)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Content",
        431 => "Request Header Fields Too Large",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        json.len()
    )?;
    stream.write_all(&json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_request_is_rejected_before_inference() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.write_all(b"POST /v1/decisions HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n").unwrap();
        });
        let (mut server, _) = listener.accept().unwrap();
        assert!(matches!(
            read_request(&mut server),
            Err((400, "duplicate Content-Length"))
        ));
        client.join().unwrap();
    }
}
