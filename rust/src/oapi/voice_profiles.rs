use std::sync::Arc;

use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::json;
use tracing::warn;

use crate::audio::decode_any_to_16k_mono;
use crate::oapi::{fastapi_validation_error, kind, missing_field, openai_error};
use nv_tts::{SpeakerEncoder, VoiceProfile, VoiceProfileStore};

pub const VOICE_PROFILE_SCHEMA_VERSION: u32 = 2;

pub fn voice_profile_root() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("SPEACHES_PLUS_VOICE_PROFILES_DIR") {
        return std::path::PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var(crate::defaults::env::SPEACHES_PLUS_MODELS) {
        return std::path::PathBuf::from(dir).join("voice-profiles");
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("voice-profiles")
}

#[derive(Clone)]
pub struct VoiceProfilesAppState {
    pub store: Arc<VoiceProfileStore>,

    pub encoder: Option<Arc<SpeakerEncoder>>,
}

impl VoiceProfilesAppState {
    pub fn new(store: VoiceProfileStore) -> Self {
        Self {
            store: Arc::new(store),
            encoder: None,
        }
    }

    pub fn with_encoder(mut self, encoder: Option<Arc<SpeakerEncoder>>) -> Self {
        self.encoder = encoder;
        self
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct VoiceProfileResponse {
    pub name: String,
    pub schema_version: u32,
    pub embedding_dim: usize,

    #[cfg_attr(
        feature = "ts-bindings",
        ts(type = "\"encoded\" | \"no_encoder\"")
    )]
    pub embedding_state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub design_params: Option<serde_json::Value>,
}

impl From<&VoiceProfile> for VoiceProfileResponse {
    fn from(p: &VoiceProfile) -> Self {
        let state = if p.embedding.iter().all(|x| *x == 0.0) {
            "no_encoder"
        } else {
            "encoded"
        };
        Self {
            name: p.name.clone(),
            schema_version: p.schema_version,
            embedding_dim: p.embedding.len(),
            embedding_state: state,
            design_params: p.design_params.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ListVoiceProfilesResponse {
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"list\""))]
    pub object: &'static str,
    pub data: Vec<VoiceProfileResponse>,
}

pub async fn handle_create(
    State(state): State<VoiceProfilesAppState>,
    mut multipart: Multipart,
) -> Response {
    let mut name: Option<String> = None;
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut audio_ct: Option<String> = None;
    let mut design_params: Option<serde_json::Value> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let fname = field.name().unwrap_or("").to_string();
        match fname.as_str() {
            "name" => {
                if let Ok(v) = field.text().await {
                    name = Some(v);
                }
            }
            "file" | "audio" => {
                audio_ct = field.content_type().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(b) => audio_bytes = Some(b.to_vec()),
                    Err(err) => {
                        return openai_error(
                            StatusCode::BAD_REQUEST,
                            format!("file read: {err}"),
                            kind::INVALID_REQUEST,
                            Some("file"),
                            Some("multipart_read_error"),
                        );
                    }
                }
            }
            "design_params" => {
                if let Ok(v) = field.text().await {
                    match serde_json::from_str::<serde_json::Value>(&v) {
                        Ok(j) => design_params = Some(j),
                        Err(err) => {
                            return openai_error(
                                StatusCode::BAD_REQUEST,
                                format!("design_params JSON: {err}"),
                                kind::INVALID_REQUEST,
                                Some("design_params"),
                                Some("invalid_json"),
                            );
                        }
                    }
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let Some(name) = name else {
        return fastapi_validation_error(vec![missing_field(&["body", "name"])]);
    };
    if name.is_empty() || !is_safe_name(&name) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "name must be non-empty and contain only [A-Za-z0-9_.-]",
            kind::INVALID_REQUEST,
            Some("name"),
            Some("invalid_name"),
        );
    }

    let Some(bytes) = audio_bytes else {
        return fastapi_validation_error(vec![missing_field(&["body", "file"])]);
    };
    let samples = match decode_any_to_16k_mono(&bytes, audio_ct.as_deref()) {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "voice-profile audio decode failed");
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("audio decode: {err}"),
                kind::INVALID_REQUEST,
                Some("file"),
                Some("audio_decode_error"),
            );
        }
    };

    let Some(encoder) = state.encoder.as_ref() else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "voice-profile enrollment is unavailable: the loaded TTS checkpoint has no \
             speaker encoder (CustomVoice/VoiceDesign checkpoints cannot embed reference \
             audio); load a Qwen3-TTS Base checkpoint",
            kind::SERVICE_UNAVAIL,
            None,
            Some("no_speaker_encoder"),
        );
    };
    let embedding = match extract_embedding(encoder, &samples) {
        Ok(v) => v,
        Err(err) => {
            warn!(error = %err, "speaker encoder failed");
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("speaker embedding: {err}"),
                kind::INVALID_REQUEST,
                Some("file"),
                Some("speaker_embedding_failed"),
            );
        }
    };

    let profile = VoiceProfile {
        schema_version: VOICE_PROFILE_SCHEMA_VERSION,
        name: name.clone(),
        embedding,
        design_params,
    };

    if let Err(err) = state.store.put(&profile) {
        warn!(error = %err, name=%name, "voice-profile store put failed");
        return openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "voice profile store write failed",
            kind::SERVER,
            None,
            Some("store_write_error"),
        );
    }

    (
        StatusCode::CREATED,
        Json(VoiceProfileResponse::from(&profile)),
    )
        .into_response()
}

pub async fn handle_list(State(state): State<VoiceProfilesAppState>) -> Response {
    let names = match state.store.list() {
        Ok(v) => v,
        Err(err) => {
            warn!(error = %err, "voice-profile list failed");
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "voice profile store read failed",
                kind::SERVER,
                None,
                Some("store_read_error"),
            );
        }
    };
    let mut data = Vec::with_capacity(names.len());
    for name in names {
        match state.store.get(&name) {
            Ok(p) => data.push(VoiceProfileResponse::from(&p)),
            Err(err) => {
                warn!(error = %err, name=%name, "voice-profile read during list failed");
            }
        }
    }
    Json(ListVoiceProfilesResponse {
        object: "list",
        data,
    })
    .into_response()
}

pub async fn handle_get(
    State(state): State<VoiceProfilesAppState>,
    Path(name): Path<String>,
) -> Response {
    if !is_safe_name(&name) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "name must contain only [A-Za-z0-9_.-]",
            kind::INVALID_REQUEST,
            Some("name"),
            Some("invalid_name"),
        );
    }
    match state.store.get(&name) {
        Ok(p) => Json(VoiceProfileResponse::from(&p)).into_response(),
        Err(err) => {
            if is_not_found(&err) {
                openai_error(
                    StatusCode::NOT_FOUND,
                    format!("voice profile not found: {name}"),
                    kind::NOT_FOUND,
                    Some("name"),
                    Some("voice_profile_not_found"),
                )
            } else {
                warn!(error = %err, name=%name, "voice-profile get failed");
                openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "voice profile store read failed",
                    kind::SERVER,
                    None,
                    Some("store_read_error"),
                )
            }
        }
    }
}

fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
    })
}

pub async fn handle_delete(
    State(state): State<VoiceProfilesAppState>,
    Path(name): Path<String>,
) -> Response {
    if !is_safe_name(&name) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "name must contain only [A-Za-z0-9_.-]",
            kind::INVALID_REQUEST,
            Some("name"),
            Some("invalid_name"),
        );
    }

    let path = state.store.path_for(&name);
    if !path.exists() {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("voice profile not found: {name}"),
            kind::NOT_FOUND,
            Some("name"),
            Some("voice_profile_not_found"),
        );
    }
    if let Err(err) = state.store.delete(&name) {
        warn!(error = %err, name=%name, "voice-profile delete failed");
        return openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "voice profile store delete failed",
            kind::SERVER,
            None,
            Some("store_delete_error"),
        );
    }
    (StatusCode::OK, Json(json!({"deleted": name}))).into_response()
}

const MIN_ENROLL_SECONDS_16K: usize = 8_000;

fn extract_embedding(
    encoder: &Arc<SpeakerEncoder>,
    samples_16k: &[f32],
) -> anyhow::Result<Vec<f32>> {
    use crate::audio::downmix_and_resample_f32;

    if samples_16k.len() < MIN_ENROLL_SECONDS_16K {
        anyhow::bail!(
            "reference audio too short: got {:.2}s, need >= 0.5s",
            samples_16k.len() as f32 / 16_000.0
        );
    }
    let samples_24k = downmix_and_resample_f32(samples_16k, 1, 16_000, 24_000);
    encoder
        .encode_pcm_24k(&samples_24k)
        .map_err(|e| anyhow::anyhow!("encoder.encode_pcm_24k: {e}"))
}

fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !name.starts_with('.')
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export)]
    struct VoiceProfileDeleteAck {
        deleted: String,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_name_accepts_alnum_dash_underscore_dot() {
        assert!(is_safe_name("alice"));
        assert!(is_safe_name("v1.0-spk_42"));
    }

    #[test]
    fn safe_name_rejects_path_traversal_and_leading_dot() {
        assert!(!is_safe_name(""));
        assert!(!is_safe_name(".hidden"));
        assert!(!is_safe_name("a/b"));
        assert!(!is_safe_name("../etc/passwd"));
        assert!(!is_safe_name("name with space"));
    }

    #[test]
    fn not_found_is_detected_by_io_error_kind() {
        let raw = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(is_not_found(&raw));
        let wrapped = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound))
            .context("read voice profile");
        assert!(is_not_found(&wrapped));
    }

    #[test]
    fn other_io_kinds_and_message_lookalikes_are_not_not_found() {
        let denied = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert!(!is_not_found(&denied));
        let lookalike = anyhow::anyhow!("No such file or directory (os error 2)");
        assert!(!is_not_found(&lookalike));
    }

    #[test]
    fn response_marks_zero_embedding_as_no_encoder() {
        let p = VoiceProfile {
            schema_version: 1,
            name: "n".into(),
            embedding: vec![0.0; 8],
            design_params: None,
        };
        let r = VoiceProfileResponse::from(&p);
        assert_eq!(r.embedding_state, "no_encoder");
        assert_eq!(r.embedding_dim, 8);
    }

    #[test]
    fn response_marks_nonzero_embedding_as_encoded() {
        let p = VoiceProfile {
            schema_version: 1,
            name: "n".into(),
            embedding: vec![0.0, 0.1, 0.0],
            design_params: None,
        };
        let r = VoiceProfileResponse::from(&p);
        assert_eq!(r.embedding_state, "encoded");
    }
}
