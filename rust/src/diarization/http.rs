use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use tracing::warn;

use crate::audio::decode_any_to_16k_mono;
use crate::oapi;
use crate::AppState;

use super::{DiarConfig, DiarSegment, Diarizer, EmbeddingModel};

pub async fn diarization_post(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut audio_filename: Option<String> = None;
    let mut audio_content_type: Option<String> = None;
    let mut response_format = "json".to_string();
    let mut known_names: Vec<String> = Vec::new();
    let mut known_refs: Vec<String> = Vec::new();
    let mut _model: Option<String> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                audio_filename = field.file_name().map(|s| s.to_string());
                audio_content_type = field.content_type().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(b) => audio_bytes = Some(b.to_vec()),
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
            "model" => {
                if let Ok(v) = field.text().await {
                    _model = Some(v);
                }
            }
            "response_format" => {
                if let Ok(v) = field.text().await {
                    response_format = v;
                }
            }
            "known_speaker_names[]" | "known_speaker_names" => {
                if let Ok(v) = field.text().await {
                    known_names.push(v);
                }
            }
            "known_speaker_references[]" | "known_speaker_references" => {
                if let Ok(v) = field.text().await {
                    known_refs.push(v);
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let Some(bytes) = audio_bytes else {
        return oapi::fastapi_validation_error(vec![oapi::missing_field(&["body", "file"])]);
    };

    if known_names.len() != known_refs.len() {
        return oapi::openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "known_speaker_names and known_speaker_references must be sent in equal \
                 numbers; got {} name(s) and {} reference(s)",
                known_names.len(),
                known_refs.len()
            ),
            oapi::kind::INVALID_REQUEST,
            Some("known_speaker_names"),
            Some("known_speaker_arity_mismatch"),
        );
    }

    let (Some(seg), Some(emb)) = (
        state.models.diar_segmentation.clone(),
        state.models.diar_embedding.clone(),
    ) else {
        return oapi::openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "diarization model not loaded; run rust/scripts/fetch-models.sh and \
             rust/scripts/export-diarizen-onnx.py so that diarizen-segmentation.onnx and \
             wespeaker-resnet293-LM.onnx exist under SPEACHES_PLUS_MODELS",
            oapi::kind::SERVICE_UNAVAIL,
            None,
            Some("model_not_loaded"),
        );
    };

    let samples = match decode_any_to_16k_mono(&bytes, audio_content_type.as_deref()) {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "audio decode failed");
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                format!("audio decode: {err}"),
                oapi::kind::INVALID_REQUEST,
                Some("file"),
                Some("audio_decode_error"),
            );
        }
    };
    let duration_s = samples.len() as f64 / 16_000.0;

    let known_embeddings: Vec<(String, Vec<f32>)> = if !known_names.is_empty() {
        let mut out = Vec::with_capacity(known_names.len());
        for (name, data_url) in known_names.iter().zip(known_refs.iter()) {
            let (ref_bytes, ref_mime) = match decode_data_url_with_mime(data_url) {
                Ok(b) => b,
                Err(err) => {
                    return oapi::openai_error(
                        StatusCode::BAD_REQUEST,
                        format!("known_speaker_references[{name}]: {err}"),
                        oapi::kind::INVALID_REQUEST,
                        Some("known_speaker_references"),
                        Some("data_url_decode_error"),
                    );
                }
            };
            let ref_samples = match decode_any_to_16k_mono(&ref_bytes, ref_mime.as_deref()) {
                Ok(s) => s,
                Err(err) => {
                    return oapi::openai_error(
                        StatusCode::BAD_REQUEST,
                        format!("known_speaker_references[{name}] decode: {err}"),
                        oapi::kind::INVALID_REQUEST,
                        Some("known_speaker_references"),
                        Some("audio_decode_error"),
                    );
                }
            };
            let emb_clone = emb.clone();
            let v = match tokio::task::spawn_blocking(move || emb_clone.embed(&ref_samples)).await {
                Ok(Ok(v)) => v,
                Ok(Err(err)) => {
                    return oapi::openai_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("embed reference {name}: {err}"),
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
            out.push((name.clone(), v));
        }
        out
    } else {
        Vec::new()
    };

    let mut diarizer = Diarizer::new(seg, emb.clone(), DiarConfig::default());
    let samples_for_diar = samples.clone();
    let segments =
        match tokio::task::spawn_blocking(move || diarizer.diarize_utterance(&samples_for_diar, 0))
            .await
        {
            Ok(Ok(segs)) => segs,
            Ok(Err(err)) => {
                return oapi::openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("diarize: {err}"),
                    oapi::kind::SERVER,
                    None,
                    Some("diarize_failed"),
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

    let label_for = build_speaker_label_map(&segments, &samples, &emb, &known_embeddings).await;

    if response_format == "rttm" {
        let file_id = audio_filename
            .as_deref()
            .map(|n| {
                std::path::Path::new(n)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("audio")
                    .to_string()
            })
            .unwrap_or_else(|| "audio".to_string());
        let body: String = segments
            .iter()
            .map(|s| {
                let dur = (s.t_end_ms.saturating_sub(s.t_start_ms)) as f64 / 1000.0;
                let start = s.t_start_ms as f64 / 1000.0;
                let label = label_for(s.speaker);
                format!(
                    "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
                    file_id, start, dur, label
                )
            })
            .collect();
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            body,
        )
            .into_response()
    } else {
        let body = serde_json::json!({
            "duration": duration_s,
            "segments": segments
                .iter()
                .map(|s| serde_json::json!({
                    "start": s.t_start_ms as f64 / 1000.0,
                    "end": s.t_end_ms as f64 / 1000.0,
                    "speaker": label_for(s.speaker),
                }))
                .collect::<Vec<_>>(),
        });
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response()
    }
}

pub async fn build_speaker_label_map(
    segments: &[DiarSegment],
    audio: &[f32],
    emb: &Arc<EmbeddingModel>,
    known: &[(String, Vec<f32>)],
) -> impl Fn(u32) -> String {
    use super::embedding::cosine_sim;

    let mut per_cluster: std::collections::HashMap<u32, Vec<f32>> =
        std::collections::HashMap::new();
    for s in segments {
        let start_idx = ((s.t_start_ms as usize) * 16_000) / 1000;
        let end_idx = ((s.t_end_ms as usize) * 16_000) / 1000;
        let end_idx = end_idx.min(audio.len());
        if end_idx <= start_idx {
            continue;
        }
        per_cluster
            .entry(s.speaker)
            .or_insert_with(|| Vec::with_capacity(end_idx - start_idx))
            .extend_from_slice(&audio[start_idx..end_idx]);
    }

    let mut cluster_to_known: std::collections::HashMap<u32, String> =
        std::collections::HashMap::new();
    if !known.is_empty() {
        for (cid, samples) in &per_cluster {
            if samples.len() < 16_000 {
                continue;
            }
            let emb_clone = emb.clone();
            let samples_owned = samples.clone();
            let cluster_emb =
                match tokio::task::spawn_blocking(move || emb_clone.embed(&samples_owned)).await {
                    Ok(Ok(v)) => v,
                    _ => continue,
                };

            let mut best: Option<(String, f32)> = None;
            for (name, kv) in known {
                let s = cosine_sim(&cluster_emb, kv);
                match &best {
                    None => best = Some((name.clone(), s)),
                    Some((_, bs)) if s > *bs => best = Some((name.clone(), s)),
                    _ => {}
                }
            }
            if let Some((name, _)) = best {
                cluster_to_known.insert(*cid, name);
            }
        }
    }

    move |cid: u32| -> String {
        cluster_to_known
            .get(&cid)
            .cloned()
            .unwrap_or_else(|| format!("SPEAKER_{:02}", cid))
    }
}

#[allow(dead_code)]
pub fn decode_data_url(s: &str) -> anyhow::Result<Vec<u8>> {
    let (bytes, _mime) = decode_data_url_with_mime(s)?;
    Ok(bytes)
}

pub fn decode_data_url_with_mime(s: &str) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    let s = s.trim();
    let rest = s
        .strip_prefix("data:")
        .ok_or_else(|| anyhow::anyhow!("not a data URL"))?;
    let comma = rest
        .find(',')
        .ok_or_else(|| anyhow::anyhow!("missing comma"))?;
    let header = &rest[..comma];
    let body = &rest[comma + 1..];
    let mut mime: Option<String> = None;
    let mut is_b64 = false;
    for (i, p) in header.split(';').enumerate() {
        if i == 0 && !p.is_empty() {
            mime = Some(p.to_string());
            continue;
        }
        if p.eq_ignore_ascii_case("base64") {
            is_b64 = true;
        }
    }
    if !is_b64 {
        anyhow::bail!("only base64 data URLs are supported");
    }
    let bytes = B64
        .decode(body)
        .map_err(|e| anyhow::anyhow!("base64: {e}"))?;
    Ok((bytes, mime))
}
