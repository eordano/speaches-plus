use axum::extract::{Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::audio::decode_any_to_16k_mono;
use crate::oapi;
use crate::AppState;

use super::http::decode_data_url_with_mime;

pub async fn audio_embeddings_post(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Response {
    let mut audio_inputs: Vec<(Vec<u8>, Option<String>)> = Vec::new();
    let mut requested_model: Option<String> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let ct = field.content_type().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(b) => audio_inputs.push((b.to_vec(), ct)),
                    Err(err) => {
                        return oapi::openai_error(
                            StatusCode::BAD_REQUEST,
                            format!("file read: {err}"),
                            oapi::kind::INVALID_REQUEST,
                            Some("file"),
                            Some("multipart_read_error"),
                        );
                    }
                }
            }
            "audio" => {
                if let Ok(data_url) = field.text().await {
                    match decode_data_url_with_mime(&data_url) {
                        Ok((bytes, mime)) => audio_inputs.push((bytes, mime)),
                        Err(err) => {
                            return oapi::openai_error(
                                StatusCode::BAD_REQUEST,
                                format!("audio data URL: {err}"),
                                oapi::kind::INVALID_REQUEST,
                                Some("audio"),
                                Some("data_url_decode_error"),
                            );
                        }
                    }
                }
            }
            "model" => {
                if let Ok(v) = field.text().await {
                    requested_model = Some(v);
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    if audio_inputs.is_empty() {
        return oapi::fastapi_validation_error(vec![oapi::missing_field(&["body", "file"])]);
    }

    let Some(emb) = state.models.diar_embedding.clone() else {
        return oapi::openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "embedding model not loaded; run scripts/fetch-models.sh",
            oapi::kind::SERVICE_UNAVAIL,
            None,
            Some("model_not_loaded"),
        );
    };

    let mut data_items: Vec<serde_json::Value> = Vec::with_capacity(audio_inputs.len());
    let mut total_seconds = 0.0f64;
    for (idx, (bytes, mime)) in audio_inputs.into_iter().enumerate() {
        let samples = match decode_any_to_16k_mono(&bytes, mime.as_deref()) {
            Ok(s) => s,
            Err(err) => {
                return oapi::openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("audio decode (file index {idx}): {err}"),
                    oapi::kind::INVALID_REQUEST,
                    Some("file"),
                    Some("audio_decode_error"),
                );
            }
        };
        if samples.len() < emb.min_input_samples() {
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "input audio too short (file index {idx}, {} samples; need >={})",
                    samples.len(),
                    emb.min_input_samples()
                ),
                oapi::kind::INVALID_REQUEST,
                Some("file"),
                Some("audio_too_short"),
            );
        }
        total_seconds += samples.len() as f64 / 16_000.0;

        let emb_clone = emb.clone();
        let vector = match tokio::task::spawn_blocking(move || emb_clone.embed(&samples)).await {
            Ok(Ok(v)) => v,
            Ok(Err(err)) => {
                return oapi::openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embed (file index {idx}): {err}"),
                    oapi::kind::SERVER,
                    None,
                    Some("embed_failed"),
                );
            }
            Err(err) => {
                return oapi::openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("join: {err}"),
                    oapi::kind::SERVER,
                    None,
                    Some("join_error"),
                );
            }
        };

        data_items.push(serde_json::json!({
            "object": "embedding",
            "index": idx,
            "embedding": vector,
        }));
    }

    let body = serde_json::json!({
        "object": "list",
        "data": data_items,
        "model": requested_model.unwrap_or_else(|| "wespeaker-resnet293-LM".to_string()),
        "usage": { "audio_seconds": total_seconds },
    });

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}
