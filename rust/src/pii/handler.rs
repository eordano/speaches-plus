use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::classifier::PiiClassifier;
use super::spans::PiiSpan;
use crate::oapi;
use crate::oapi::chat::json_ext::OaiJson;

const MAX_BATCH: usize = 32;

#[derive(Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ClassifyRequest {
    pub text: String,
}

#[derive(Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ClassifyBatchRequest {
    pub texts: Vec<String>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct SpanOut {
    pub start: usize,
    #[serde(rename = "endExclusive")]
    pub end_exclusive: usize,
    pub label: String,
}

#[derive(Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ClassifyResponse {
    pub spans: Vec<SpanOut>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ClassifyBatchResponse {
    pub results: Vec<ClassifyResponse>,
}

impl From<PiiSpan> for SpanOut {
    fn from(s: PiiSpan) -> Self {
        Self {
            start: s.start,
            end_exclusive: s.end_exclusive,
            label: s.label,
        }
    }
}

pub async fn classify_post(
    State(classifier): State<Arc<PiiClassifier>>,
    OaiJson(req): OaiJson<ClassifyRequest>,
) -> impl IntoResponse {
    let text = req.text;
    let result = tokio::task::spawn_blocking(move || classifier.classify_one(&text)).await;
    match result {
        Ok(Ok(spans)) => {
            let resp = ClassifyResponse {
                spans: spans.into_iter().map(SpanOut::from).collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Ok(Err(err)) => classify_failed(err),
        Err(err) => join_failed(err),
    }
}

pub async fn classify_batch_post(
    State(classifier): State<Arc<PiiClassifier>>,
    OaiJson(req): OaiJson<ClassifyBatchRequest>,
) -> impl IntoResponse {
    if req.texts.len() > MAX_BATCH {
        return batch_too_large(req.texts.len());
    }

    let texts = req.texts;
    let result = tokio::task::spawn_blocking(move || classifier.classify_batch(&texts)).await;
    match result {
        Ok(Ok(batch_spans)) => {
            let resp = ClassifyBatchResponse {
                results: batch_spans
                    .into_iter()
                    .map(|spans| ClassifyResponse {
                        spans: spans.into_iter().map(SpanOut::from).collect(),
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Ok(Err(err)) => classify_failed(err),
        Err(err) => join_failed(err),
    }
}

fn classify_failed(err: anyhow::Error) -> Response {
    oapi::openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("{err:#}"),
        oapi::kind::SERVER,
        None,
        Some("classify_failed"),
    )
}

fn join_failed(err: tokio::task::JoinError) -> Response {
    oapi::openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("join: {err}"),
        oapi::kind::SERVER,
        None,
        Some("join_error"),
    )
}

fn batch_too_large(len: usize) -> Response {
    oapi::openai_error(
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("batch size {len} exceeds max {MAX_BATCH}"),
        oapi::kind::INVALID_REQUEST,
        Some("texts"),
        Some("batch_too_large"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn envelope(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn batch_cap_is_413_with_openai_envelope() {
        let resp = batch_too_large(MAX_BATCH + 1);
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "texts");
        assert_eq!(v["error"]["code"], "batch_too_large");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("33") && msg.contains("32"), "{msg}");
    }

    #[tokio::test]
    async fn malformed_json_is_400_with_openai_envelope() {
        use axum::body::Body;
        use axum::extract::{FromRequest, Request};
        let req = Request::builder()
            .method("POST")
            .uri("/v1/pii/classify")
            .header("content-type", "application/json")
            .body(Body::from("{\"text\": "))
            .unwrap();
        let resp = OaiJson::<ClassifyRequest>::from_request(req, &())
            .await
            .err()
            .expect("malformed JSON must be rejected");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "invalid_json");
        assert!(!v["error"]["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn batch_missing_content_type_is_415_with_openai_envelope() {
        use axum::body::Body;
        use axum::extract::{FromRequest, Request};
        let req = Request::builder()
            .method("POST")
            .uri("/v1/pii/classify/batch")
            .body(Body::from("{\"texts\":[\"a\"]}"))
            .unwrap();
        let resp = OaiJson::<ClassifyBatchRequest>::from_request(req, &())
            .await
            .err()
            .expect("a missing JSON content-type must be rejected");
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "unsupported_content_type");
    }

    #[tokio::test]
    async fn classify_failure_is_500_with_openai_envelope() {
        let resp = classify_failed(anyhow::anyhow!("boom"));
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["type"], "internal_server_error");
        assert_eq!(v["error"]["code"], "classify_failed");
        assert_eq!(v["error"]["message"], "boom");
        assert!(v["error"]["param"].is_null());
    }
}
