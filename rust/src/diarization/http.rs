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

    let speakers = match speaker_summaries(&segments, &samples, &emb).await {
        Ok(v) => v,
        Err(err) => {
            return oapi::openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("speaker embeddings: {err}"),
                oapi::kind::SERVER,
                None,
                Some("embed_failed"),
            )
        }
    };
    let label_for = speaker_labels(&speakers, &known_embeddings);

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
        let body = diarization_json(duration_s, &segments, &speakers, &label_for);
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response()
    }
}

fn diarization_json(
    duration_s: f64,
    segments: &[DiarSegment],
    speakers: &[SpeakerSummary],
    label_for: &impl Fn(u32) -> String,
) -> serde_json::Value {
    serde_json::json!({
            "duration": duration_s,
            "speakers": speakers.iter().map(|s| serde_json::json!({
                "speaker": label_for(s.id), "seconds": s.seconds,
                "turns": s.turns, "embedding": s.embedding,
            })).collect::<Vec<_>>(),
            "segments": segments
                .iter()
                .map(|s| serde_json::json!({
                    "start": s.t_start_ms as f64 / 1000.0,
                    "end": s.t_end_ms as f64 / 1000.0,
                    "speaker": label_for(s.speaker),
                    "confidence": s.confidence,
                }))
                .collect::<Vec<_>>(),
    })
}

#[derive(Debug)]
struct SpeakerSummary {
    id: u32,
    seconds: f64,
    turns: usize,
    embedding: Vec<f32>,
}

async fn speaker_summaries(
    segments: &[DiarSegment],
    audio: &[f32],
    emb: &Arc<EmbeddingModel>,
) -> anyhow::Result<Vec<SpeakerSummary>> {
    let mut clusters: std::collections::BTreeMap<u32, (Vec<f32>, f64, usize)> = Default::default();
    for s in segments {
        let start = ((s.t_start_ms as usize) * 16).min(audio.len());
        let end = ((s.t_end_ms as usize) * 16).min(audio.len());
        if end <= start {
            continue;
        }
        let c = clusters.entry(s.speaker).or_default();
        c.1 += (end - start) as f64 / 16_000.0;
        c.2 += 1;
        let remaining = (30 * 16_000usize).saturating_sub(c.0.len());
        c.0.extend_from_slice(&audio[start..end.min(start + remaining)]);
    }
    let emb = emb.clone();
    tokio::task::spawn_blocking(move || {
        clusters
            .into_iter()
            .map(|(id, (samples, seconds, turns))| {
                let embedding = if samples.len() >= 16_000 {
                    emb.embed(&samples)?
                } else {
                    Vec::new()
                };
                Ok(SpeakerSummary {
                    id,
                    seconds,
                    turns,
                    embedding,
                })
            })
            .collect()
    })
    .await?
}

fn speaker_labels(
    speakers: &[SpeakerSummary],
    known: &[(String, Vec<f32>)],
) -> impl Fn(u32) -> String {
    let mut labels = std::collections::HashMap::new();
    for s in speakers {
        if s.embedding.is_empty() {
            continue;
        }
        if let Some((name, _)) = known
            .iter()
            .map(|(name, e)| (name, super::embedding::cosine_sim(&s.embedding, e)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        {
            labels.insert(s.id, name.clone());
        }
    }
    move |id| {
        labels
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("SPEAKER_{id:02}"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voices::{stages, Workspace};

    #[tokio::test]
    #[ignore = "requires DiariZen and WeSpeaker ONNX model paths"]
    async fn voices_real_diarization_enters_registry() -> anyhow::Result<()> {
        let seg = Arc::new(super::super::SegmentationModel::load(std::env::var(
            "NV_TEST_DIAR_MODEL",
        )?)?);
        let emb = Arc::new(EmbeddingModel::load(std::env::var(
            "NV_TEST_EMBEDDING_MODEL",
        )?)?);
        let audio = decode_any_to_16k_mono(
            include_bytes!("../../tests/data/parakeet_fox_16k.wav"),
            Some("audio/wav"),
        )?;
        let mut diarizer = Diarizer::new(seg, emb.clone(), DiarConfig::default());
        let segments = diarizer.diarize_utterance(&audio, 0)?;
        assert!(
            !segments.is_empty(),
            "real speech must produce diarized segments"
        );
        let speakers = speaker_summaries(&segments, &audio, &emb).await?;
        assert!(speakers
            .iter()
            .any(|s| s.seconds >= 1. && s.embedding.len() == 256));
        let body = diarization_json(
            audio.len() as f64 / 16_000.,
            &segments,
            &speakers,
            &speaker_labels(&speakers, &[]),
        );
        let td = stages::TempDir::new("real-diar")?;
        let ws = Workspace::new(&td.0);
        ws.init()?;
        std::fs::write(
            ws.recordings_dir().join("real.wav"),
            include_bytes!("../../tests/data/parakeet_fox_16k.wav"),
        )?;
        let mut manifest = ws.load()?;
        ws.scan(&mut manifest, false)?;
        let sha = manifest.rows.keys().next().unwrap().clone();
        ws.put_artifact(&mut manifest, &sha, "diar", &body)?;
        let registry = stages::identify(&ws, &manifest, 1., &mut |_| {})?;
        assert!(!registry.speakers.is_empty());
        assert!(registry.speakers.iter().all(|s| !s.sources.is_empty()));
        Ok(())
    }

    #[test]
    fn voices_identify_consumes_the_engine_diarization_response() -> anyhow::Result<()> {
        let td = stages::TempDir::new("diar-contract")?;
        let ws = Workspace::new(&td.0);
        ws.init()?;
        std::fs::write(ws.recordings_dir().join("episode.wav"), b"fixture")?;
        let mut manifest = ws.load()?;
        ws.scan(&mut manifest, false)?;
        let sha = manifest.rows.keys().next().unwrap().clone();
        let segments = vec![DiarSegment {
            speaker: 0,
            t_start_ms: 0,
            t_end_ms: 90_000,
            confidence: 0.9,
        }];
        let speakers = vec![SpeakerSummary {
            id: 0,
            seconds: 90.,
            turns: 1,
            embedding: vec![1., 0., 0.],
        }];
        let labels = speaker_labels(&speakers, &[]);
        let body = diarization_json(90., &segments, &speakers, &labels);
        assert_eq!(
            body["segments"][0]["speaker"],
            body["speakers"][0]["speaker"]
        );
        ws.put_artifact(&mut manifest, &sha, "diar", &body)?;
        let registry = stages::identify(&ws, &manifest, 60., &mut |_| {})?;
        assert_eq!(registry.speakers.len(), 1);
        assert_eq!(registry.speakers[0].seconds, 90.);
        assert_eq!(registry.speakers[0].sources[0].label, "SPEAKER_00");
        Ok(())
    }
}
