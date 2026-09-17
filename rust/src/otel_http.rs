use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};
use tracing::{field::Empty, Span};

const CAPTURE_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Default)]
struct Capture {
    bytes: Vec<u8>,
    total: usize,
}

impl Capture {
    fn append(&mut self, bytes: &[u8]) {
        self.total += bytes.len();
        let remaining = CAPTURE_LIMIT.saturating_sub(self.bytes.len());
        self.bytes
            .extend_from_slice(&bytes[..bytes.len().min(remaining)]);
    }

    fn value(&self) -> Value {
        let value = serde_json::from_slice::<Value>(&self.bytes).unwrap_or_else(|_| {
            match std::str::from_utf8(&self.bytes) {
                Ok(text) => Value::String(text.to_owned()),
                Err(_) => json!({"encoding": "base64", "data": STANDARD.encode(&self.bytes)}),
            }
        });
        if self.total > self.bytes.len() {
            json!({"captured": value, "truncated": true, "total_bytes": self.total})
        } else {
            value
        }
    }
}

struct RequestTrace {
    span: Span,
    input: Arc<Mutex<Capture>>,
    output: Capture,
    complete: bool,
    started: Instant,
}

impl Drop for RequestTrace {
    fn drop(&mut self) {
        let input = self.input.lock().unwrap_or_else(|e| e.into_inner());
        let request = input.value();
        if let Some(model) = request.get("model").and_then(Value::as_str) {
            self.span.record("gen_ai.request.model", model);
        }
        let response = self.output.value();
        let usage = match &response {
            Value::String(text) => text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter_map(|value| value.get("usage").cloned())
                .next_back(),
            value => value.get("usage").cloned(),
        };
        if let Some(usage) = usage {
            self.span.record("langfuse.observation.usage_details", json!({
                "input": usage.get("prompt_tokens").or_else(|| usage.get("input_tokens")).and_then(Value::as_u64).unwrap_or(0),
                "output": usage.get("completion_tokens").or_else(|| usage.get("output_tokens")).and_then(Value::as_u64).unwrap_or(0)
            }).to_string());
        }
        self.span
            .record("langfuse.observation.input", request.to_string());
        self.span.record(
            "langfuse.observation.output",
            self.output.value().to_string(),
        );
        self.span.record("response.complete", self.complete);
        self.span.record("input.bytes", input.total);
        self.span.record("output.bytes", self.output.total);
        self.span.record(
            "capture.truncated",
            input.total > CAPTURE_LIMIT || self.output.total > CAPTURE_LIMIT,
        );
    }
}

pub async fn trace_request(request: Request, next: Next) -> Response {
    if !crate::otel::is_enabled() || !request.uri().path().starts_with("/v1/") {
        return next.run(request).await;
    }
    trace_enabled_request(request, next).await
}

async fn trace_enabled_request(request: Request, next: Next) -> Response {
    let span = tracing::info_span!(
        "speaches.request",
        "langfuse.observation.type" = "GENERATION",
        "langfuse.environment" = std::env::var("LANGFUSE_TRACING_ENVIRONMENT").unwrap_or_else(|_| "production".into()),
        "http.request.method" = %request.method(),
        "url.path" = request.uri().path(),
        "http.response.status_code" = Empty,
        "langfuse.observation.input" = Empty,
        "langfuse.observation.output" = Empty,
        "gen_ai.request.model" = Empty,
        "langfuse.observation.usage_details" = Empty,
        "langfuse.observation.level" = Empty,
        "response.complete" = Empty,
        "input.bytes" = Empty,
        "output.bytes" = Empty,
        "capture.truncated" = Empty,
        "response.first_byte_ms" = Empty,
    );
    let input = Arc::new(Mutex::new(Capture::default()));
    let input_sink = input.clone();
    let (parts, body) = request.into_parts();
    let body = Body::from_stream(body.into_data_stream().map(move |chunk| {
        if let Ok(bytes) = &chunk {
            input_sink
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .append(bytes);
        }
        chunk
    }));
    let trace = RequestTrace {
        span,
        input,
        output: Capture::default(),
        complete: false,
        started: Instant::now(),
    };
    let response = next.run(Request::from_parts(parts, body)).await;
    trace
        .span
        .record("http.response.status_code", response.status().as_u16());
    if response.status().is_client_error() || response.status().is_server_error() {
        trace.span.record("langfuse.observation.level", "ERROR");
    }
    let (parts, body) = response.into_parts();
    let stream = stream::unfold(
        (body.into_data_stream(), trace),
        |(mut source, mut trace)| async move {
            match source.next().await {
                Some(chunk) => {
                    if let Ok(bytes) = &chunk {
                        if trace.output.total == 0 {
                            trace.span.record(
                                "response.first_byte_ms",
                                trace.started.elapsed().as_secs_f64() * 1000.0,
                            );
                        }
                        trace.output.append(bytes);
                    } else {
                        trace.span.record("langfuse.observation.level", "ERROR");
                    }
                    Some((chunk, (source, trace)))
                }
                None => {
                    trace.complete = true;
                    drop(trace);
                    None
                }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

pub fn trace_realtime_event(session_id: &str, direction: &str, text: &str) {
    if !crate::otel::is_enabled() {
        return;
    }
    let span = tracing::info_span!(
        "speaches.realtime.event",
        "langfuse.observation.type" = "EVENT",
        "session.id" = session_id,
        "event.direction" = direction,
        "langfuse.observation.input" = Empty,
        "langfuse.observation.output" = Empty,
    );
    let field = if direction == "input" {
        "langfuse.observation.input"
    } else {
        "langfuse.observation.output"
    };
    span.record(field, text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn preserves_request_body_and_error_response() {
        use axum::{http::StatusCode, middleware, routing::post, Router};
        use tower::ServiceExt;
        let app = Router::new()
            .route(
                "/v1/echo",
                post(|body: bytes::Bytes| async move { (StatusCode::IM_A_TEAPOT, body) }),
            )
            .layer(middleware::from_fn(trace_enabled_request));
        let body = "{\"model\":\"test\",\"messages\":[\"蜂\"]}";
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/echo")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::IM_A_TEAPOT);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), body.as_bytes());
    }

    #[tokio::test]
    async fn streams_first_chunk_without_waiting_for_completion() {
        use axum::{middleware, routing::get, Router};
        use tower::ServiceExt;
        let app = Router::new()
            .route(
                "/v1/stream",
                get(|| async {
                    Body::from_stream(
                        stream::iter([Ok::<_, std::io::Error>("data: first\n\n")])
                            .chain(stream::pending()),
                    )
                }),
            )
            .layer(middleware::from_fn(trace_enabled_request));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        let first = tokio::time::timeout(std::time::Duration::from_secs(1), body.next())
            .await
            .expect("tracing must not buffer an unfinished response")
            .unwrap()
            .unwrap();
        assert_eq!(first.as_ref(), b"data: first\n\n");
        drop(body);
    }

    #[test]
    fn preserves_split_utf8_and_json() {
        let mut c = Capture::default();
        for byte in "{\"prompt\":\"蜂\"}".as_bytes() {
            c.append(&[*byte]);
        }
        assert_eq!(c.value(), json!({"prompt": "蜂"}));
    }

    #[test]
    fn preserves_sse_and_binary() {
        let mut c = Capture::default();
        c.append(b"data: {\"text\":\"hello\"}\n\n");
        c.append(b"data: [DONE]\n\n");
        assert_eq!(
            c.value(),
            json!("data: {\"text\":\"hello\"}\n\ndata: [DONE]\n\n")
        );
        let mut binary = Capture::default();
        binary.append(&[0xff, 0x00]);
        assert_eq!(binary.value(), json!({"encoding":"base64", "data":"/wA="}));
    }
}
