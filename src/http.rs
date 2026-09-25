//! Versioned HTTP decision API. A local model stays on its owning thread.
use crate::Error;
use serde_json::{Value, json};
#[cfg(feature = "openrouter")]
use std::sync::Mutex;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread,
    time::Duration,
};

const MAX_HEADER: usize = 16 * 1024;
const MAX_BODY: usize = 44 * 1024 * 1024;
const MAX_MEDIA: usize = 4;
const MAX_DECISIONS: usize = 128;
const MAX_CONNECTIONS: usize = 32;
const QUEUE_DEPTH: usize = 16;
const MAX_INFLIGHT_BODY_BYTES: usize = 192 * 1024 * 1024;
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static INFLIGHT_BODY_BYTES: AtomicUsize = AtomicUsize::new(0);

mod contract;
pub use contract::HttpDecisionBackend;
use contract::run_request;

struct Job {
    body: Vec<u8>,
    request_id: String,
    reply: mpsc::Sender<Result<Value, Error>>,
}

/// The backend remains on the caller thread. Connection parsing and health checks continue independently.
pub fn serve<B: HttpDecisionBackend>(
    address: &str,
    backend: &mut B,
) -> Result<(), Box<dyn std::error::Error>> {
    let (sender, receiver) = mpsc::sync_channel(QUEUE_DEPTH);
    let capabilities = backend.capabilities();
    let _listener = start_listener(address, capabilities, sender)?;
    for job in receiver {
        let result = run_request(backend, &job.body);
        let _ = job.reply.send(result.map(|mut value| {
            value["request_id"] = json!(job.request_id);
            value
        }));
    }
    Ok(())
}

/// Remote adapters may process separate HTTP requests on independent owned workers.
#[cfg(feature = "openrouter")]
pub fn serve_openrouter(
    address: &str,
    backend: crate::openrouter::OpenRouterBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let (sender, receiver) = mpsc::sync_channel(QUEUE_DEPTH);
    let receiver = Arc::new(Mutex::new(receiver));
    let listener = start_listener(address, backend.capabilities(), sender)?;
    for _ in 0..4 {
        let receiver = Arc::clone(&receiver);
        let mut backend = backend.clone();
        thread::spawn(move || {
            loop {
                let job = { receiver.lock().expect("queue lock").recv() };
                let Ok(job) = job else { break };
                let result = run_request(&mut backend, &job.body);
                let _ = job.reply.send(result.map(|mut value| {
                    value["request_id"] = json!(job.request_id);
                    value
                }));
            }
        });
    }
    listener
        .join()
        .map_err(|_| std::io::Error::other("HTTP listener panicked"))?;
    Ok(())
}

fn start_listener(
    address: &str,
    capabilities: Value,
    sender: SyncSender<Job>,
) -> std::io::Result<thread::JoinHandle<()>> {
    let listener = TcpListener::bind(address)?;
    eprintln!("l2s1 HTTP listening on {}", listener.local_addr()?);
    Ok(thread::spawn(move || {
        let active = Arc::new(AtomicUsize::new(0));
        for connection in listener.incoming() {
            match connection {
                Ok(mut stream) => {
                    let request_id =
                        format!("req-{}", NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed));
                    if active.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                        active.fetch_sub(1, Ordering::AcqRel);
                        let _ = respond_error(
                            &mut stream,
                            503,
                            "busy",
                            "too many connections",
                            &request_id,
                        );
                        continue;
                    }
                    let active = Arc::clone(&active);
                    let sender = sender.clone();
                    let capabilities = capabilities.clone();
                    thread::spawn(move || {
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(180)));
                        if let Err(error) = handle(&mut stream, &sender, &capabilities, &request_id)
                        {
                            eprintln!("HTTP connection error: {error}");
                        }
                        active.fetch_sub(1, Ordering::AcqRel);
                    });
                }
                Err(error) => eprintln!("HTTP accept error: {error}"),
            }
        }
    }))
}

fn handle(
    stream: &mut TcpStream,
    sender: &SyncSender<Job>,
    capabilities: &Value,
    request_id: &str,
) -> std::io::Result<()> {
    let mut request = match read_request(stream) {
        Ok(request) => request,
        Err((status, message)) => {
            let code = if status == 503 {
                "busy"
            } else {
                "invalid_http_request"
            };
            return respond_error(stream, status, code, message, request_id);
        }
    };
    if request.method == "GET" && request.path == "/healthz" {
        return respond(stream, 200, &json!({"status":"ok"}));
    }
    if request.method == "GET" && request.path == "/v1/capabilities" {
        return respond(stream, 200, capabilities);
    }
    if request.method != "POST" || request.path != "/v1/decisions" {
        return respond_error(
            stream,
            404,
            "route_not_found",
            "route not found",
            request_id,
        );
    }
    let content_type = request.content_type.to_ascii_lowercase();
    if content_type != "application/json" && !content_type.starts_with("application/json;") {
        return respond_error(
            stream,
            415,
            "unsupported_media_type",
            "Content-Type must be application/json",
            request_id,
        );
    }
    let (reply, receiver) = mpsc::channel();
    match sender.try_send(Job {
        body: std::mem::take(&mut request.body),
        request_id: request_id.into(),
        reply,
    }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            return respond_error(stream, 503, "busy", "inference queue is full", request_id);
        }
        Err(TrySendError::Disconnected(_)) => {
            return respond_error(
                stream,
                503,
                "unavailable",
                "inference worker unavailable",
                request_id,
            );
        }
    }
    match receiver.recv() {
        Ok(Ok(output)) => respond(stream, 200, &output),
        Ok(Err(Error::Invalid(message))) => {
            respond_error(stream, 400, "invalid_request", &message, request_id)
        }
        Ok(Err(Error::Backend(message))) => {
            respond_error(stream, 500, "backend_error", &message, request_id)
        }
        Ok(Err(Error::ModelLoad(message))) => {
            respond_error(stream, 500, "model_load_failed", &message, request_id)
        }
        Ok(Err(Error::Upstream(message))) => {
            respond_error(stream, 502, "upstream_error", &message, request_id)
        }
        Err(_) => respond_error(
            stream,
            503,
            "unavailable",
            "inference worker unavailable",
            request_id,
        ),
    }
}

fn respond_error(
    stream: &mut TcpStream,
    status: u16,
    code: &str,
    message: &str,
    request_id: &str,
) -> std::io::Result<()> {
    respond(
        stream,
        status,
        &json!({"error":{"code":code,"message":message,"request_id":request_id}}),
    )
}

struct HttpRequest {
    method: String,
    path: String,
    content_type: String,
    body: Vec<u8>,
    _reservation: BodyReservation,
}

struct BodyReservation(usize);
impl BodyReservation {
    fn acquire(bytes: usize) -> Result<Self, (u16, &'static str)> {
        INFLIGHT_BODY_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= MAX_INFLIGHT_BODY_BYTES)
            })
            .map_err(|_| (503, "request memory budget exhausted"))?;
        Ok(Self(bytes))
    }
}
impl Drop for BodyReservation {
    fn drop(&mut self) {
        INFLIGHT_BODY_BYTES.fetch_sub(self.0, Ordering::AcqRel);
    }
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
    let reservation = BodyReservation::acquire(length)?;
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
        _reservation: reservation,
    })
}

fn respond(stream: &mut TcpStream, status: u16, body: &Value) -> std::io::Result<()> {
    let json = serde_json::to_vec(body)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        500 => "Internal Server Error",
        431 => "Request Header Fields Too Large",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
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
    use crate::DecisionRequest;

    #[derive(Default)]
    struct Probe {
        calls: Vec<(Vec<String>, Vec<Vec<u8>>)>,
        max_images: usize,
    }
    impl HttpDecisionBackend for Probe {
        fn capabilities(&self) -> Value {
            json!({"media":{"image":{"supported":true,"max_per_decision":self.max_images}}})
        }
        fn decide_json(
            &mut self,
            request: &DecisionRequest,
            images: &[&[u8]],
        ) -> crate::Result<Value> {
            self.calls.push((
                request.decisions.iter().map(|d| d.id.clone()).collect(),
                images.iter().map(|image| image.to_vec()).collect(),
            ));
            Ok(
                json!({"backend":{"runtime":"probe","model":"probe"},"policy":null,
                "results":request.decisions.iter().map(|d| json!({"id":d.id,"value":{"type":"binary","value":true},
                    "status":"selected","abstention_reasons":[],"evidence":{"type":"selection_only"},"usage":{}})).collect::<Vec<_>>() }),
            )
        }
    }
    fn decision(id: &str, media_ids: Option<Value>) -> Value {
        let mut d = json!({"id":id,"instruction":"Choose","kind":{"type":"binary","false_label":"no","true_label":"yes"}});
        if let Some(ids) = media_ids {
            d["media_ids"] = ids;
        }
        d
    }

    #[test]
    fn media_references_preserve_order_and_batch_matching_decisions() {
        let mut probe = Probe {
            max_images: 2,
            ..Default::default()
        };
        let request = json!({"state":{},"media":[
            {"id":"front","type":"image","data_base64":"AQID"},
            {"id":"side","type":"image","data_base64":"BAUG"}],
            "decisions":[decision("a",None), decision("b",None),
                decision("c",Some(json!(["side"]))),decision("d",Some(json!([])))]});
        let result = run_request(&mut probe, request.to_string().as_bytes()).unwrap();
        assert_eq!(result["results"].as_array().unwrap().len(), 4);
        assert_eq!(
            probe.calls[0],
            (
                vec!["a".into(), "b".into()],
                vec![vec![1, 2, 3], vec![4, 5, 6]]
            )
        );
        assert_eq!(probe.calls[1], (vec!["c".into()], vec![vec![4, 5, 6]]));
        assert_eq!(probe.calls[2], (vec!["d".into()], vec![]));
    }

    #[test]
    fn invalid_media_is_rejected_before_any_inference() {
        let mut probe = Probe {
            max_images: 1,
            ..Default::default()
        };
        let request = json!({"state":{},"media":[
            {"id":"front","type":"image","data_base64":"AQID"},
            {"id":"side","type":"image","data_base64":"BAUG"}],
            "decisions":[decision("a",Some(json!([]))),decision("b",None)]});
        assert!(matches!(
            run_request(&mut probe, request.to_string().as_bytes()),
            Err(Error::Invalid(_))
        ));
        assert!(probe.calls.is_empty());
    }

    #[test]
    fn capability_and_health_routes_respond_without_backend() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, _receiver) = mpsc::sync_channel(1);
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                handle(&mut stream, &sender, &json!({"api_version":1}), "req-test").unwrap();
            }
        });
        for (path, expected) in [
            ("/healthz", "\"ok\""),
            ("/v1/capabilities", "\"api_version\":1"),
        ] {
            let mut stream = TcpStream::connect(address).unwrap();
            write!(stream, "GET {path} HTTP/1.1\r\n\r\n").unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            assert!(response.contains(expected), "{response}");
        }
        server.join().unwrap();
    }

    #[test]
    fn health_responds_while_decision_waits_for_backend() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        let (accepted, first_ready) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let first_sender = sender.clone();
            let first_handle = thread::spawn(move || {
                handle(&mut first, &first_sender, &json!({}), "req-first").unwrap();
            });
            accepted.send(()).unwrap();
            let (mut second, _) = listener.accept().unwrap();
            handle(&mut second, &sender, &json!({}), "req-second").unwrap();
            drop(second);
            first_handle.join().unwrap();
        });
        let body = json!({"state":{},"decisions":[decision("flag",None)]}).to_string();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            write!(stream, "POST /v1/decisions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            response
        });
        first_ready.recv().unwrap();
        let mut health = TcpStream::connect(address).unwrap();
        health.write_all(b"GET /healthz HTTP/1.1\r\n\r\n").unwrap();
        let mut health_response = String::new();
        health.read_to_string(&mut health_response).unwrap();
        assert!(health_response.contains("\"status\":\"ok\""));
        let job = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        job.reply
            .send(Ok(json!({"api_version":1,"results":[]})))
            .unwrap();
        assert!(client.join().unwrap().starts_with("HTTP/1.1 200"));
        server.join().unwrap();
    }

    #[test]
    fn malformed_json_has_machine_readable_error_and_request_id() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            handle(&mut stream, &sender, &json!({}), "req-invalid").unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(b"POST /v1/decisions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n{").unwrap();
        let job = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        let mut backend = Probe {
            max_images: 1,
            ..Default::default()
        };
        job.reply
            .send(run_request(&mut backend, &job.body))
            .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(response.contains("\"code\":\"invalid_request\""));
        assert!(response.contains("\"request_id\":\"req-invalid\""));
        server.join().unwrap();
    }
}
