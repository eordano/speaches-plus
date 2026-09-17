use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::oapi::chat::ChatAppState;

pub const LAYER_CAPTURE_DEFAULT_CONTEXT: usize = 2048;

#[derive(Clone, Debug, Deserialize)]
pub struct LayerCaptureRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub layer: usize,
    #[serde(default)]
    pub texts: Vec<String>,
    #[serde(default)]
    pub inputs: Vec<Vec<u32>>,
    #[serde(default)]
    pub tokens: bool,
    #[serde(default)]
    pub max_context: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaptureRecord {
    pub tokens: usize,
    pub chunks: usize,
    pub input_token_ids: Vec<u32>,
    pub mean: Vec<f32>,
    pub last: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_states: Option<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerCaptureResponse {
    pub model: String,
    pub layer: usize,
    pub hidden_size: usize,
    pub records: Vec<CaptureRecord>,
    pub seconds: f64,
}

pub async fn handle(
    State(state): State<ChatAppState>,
    Json(req): Json<LayerCaptureRequest>,
) -> Response {
    let Some(engine) = state.registry.resolve(req.model.as_deref()) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"type": "not_found", "message": "no such chat model"}})),
        )
            .into_response();
    };
    if req.texts.is_empty() && req.inputs.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": {"type": "invalid_request_error", "message": "texts or inputs required"}})),
        )
            .into_response();
    }
    match engine.capture_layer(req).await {
        Ok(out) => Json(out).into_response(),
        Err(err) => {
            let msg = format!("{err:#}");
            let status = if msg.contains("not available") {
                StatusCode::NOT_IMPLEMENTED
            } else if msg.contains("retired") || msg.contains("busy") {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(serde_json::json!({"error": {"type": "capture_error", "message": msg}})),
            )
                .into_response()
        }
    }
}
