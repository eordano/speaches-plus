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
use nv_tts::{SpeakerEncoder, VoiceProfile, VoiceProfileStore, VoiceStoreHandle};

pub const VOICE_PROFILE_SCHEMA_VERSION: u32 = 2;

pub fn voice_profile_root() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("NV_VOICES_DATA_DIR") {
        return std::path::PathBuf::from(dir).join("voices");
    }
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
    pub store: Arc<VoiceStoreHandle>,

    pub encoder: Option<Arc<SpeakerEncoder>>,
}

impl VoiceProfilesAppState {
    pub fn new(store: Arc<VoiceStoreHandle>) -> Self {
        Self {
            store,
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

    #[cfg_attr(feature = "ts-bindings", ts(type = "\"encoded\" | \"no_encoder\""))]
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

async fn with_store<T: Send + 'static>(
    state: &VoiceProfilesAppState,
    op: impl FnOnce(&VoiceProfileStore) -> T + Send + 'static,
) -> Result<T, Response> {
    let handle = state.store.clone();
    let opened = tokio::task::spawn_blocking(move || handle.get().map(|store| op(&store)))
        .await
        .map_err(|err| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("voice profile store task: {err}"),
                kind::SERVER,
                None,
                Some("store_task_error"),
            )
        })?;
    opened.map_err(|err| {
        warn!(error = %err, "voice-profile store unavailable");
        openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("{err:#}"),
            kind::SERVICE_UNAVAIL,
            None,
            Some("voice_store_unavailable"),
        )
    })
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
    if name != crate::oapi::audio_speech_nvtts::DEFAULT_PROFILE
        && crate::oapi::audio_speech_nvtts::is_default_voice_alias(&name)
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "name {name:?} is an OpenAI voice alias and would never be selected; the \
                 aliases resolve to the profile named {:?}",
                crate::oapi::audio_speech_nvtts::DEFAULT_PROFILE
            ),
            kind::INVALID_REQUEST,
            Some("name"),
            Some("reserved_name"),
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

    let put = {
        let profile = profile.clone();
        with_store(&state, move |store| store.put(&profile)).await
    };
    let put = match put {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    if let Err(err) = put {
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
    let listed = with_store(&state, |store| {
        let names = store.list()?;
        let mut data = Vec::with_capacity(names.len());
        for name in names {
            match store.get(&name) {
                Ok(p) => data.push(VoiceProfileResponse::from(&p)),
                Err(err) => {
                    warn!(error = %err, name=%name, "voice-profile read during list failed");
                }
            }
        }
        anyhow::Ok(data)
    })
    .await;
    let data = match listed {
        Ok(Ok(data)) => data,
        Ok(Err(err)) => {
            warn!(error = %err, "voice-profile list failed");
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "voice profile store read failed",
                kind::SERVER,
                None,
                Some("store_read_error"),
            );
        }
        Err(resp) => return resp,
    };
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
    let got = {
        let name = name.clone();
        with_store(&state, move |store| store.get(&name)).await
    };
    match got {
        Err(resp) => resp,
        Ok(Ok(p)) => Json(VoiceProfileResponse::from(&p)).into_response(),
        Ok(Err(err)) => {
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

    let deleted = {
        let name = name.clone();
        with_store(&state, move |store| {
            if !store.path_for(&name).exists() {
                return Ok(false);
            }
            store.delete(&name).map(|()| true)
        })
        .await
    };
    let deleted = match deleted {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    if let Ok(false) = deleted {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("voice profile not found: {name}"),
            kind::NOT_FOUND,
            Some("name"),
            Some("voice_profile_not_found"),
        );
    }
    if let Err(err) = deleted {
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

pub fn extract_embedding(
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

pub(crate) fn is_safe_name(name: &str) -> bool {
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
    fn safe_name_accepts_alnum_dash_underscore_dot_and_rejects_traversal() {
        assert!(is_safe_name("alice"));
        assert!(is_safe_name("v1.0-spk_42"));
        {
            assert!(!is_safe_name(""));
            assert!(!is_safe_name(".hidden"));
            assert!(!is_safe_name("a/b"));
            assert!(!is_safe_name("../etc/passwd"));
            assert!(!is_safe_name("name with space"));
        }
    }

    #[test]
    fn not_found_is_detected_by_io_error_kind_only() {
        let raw = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(is_not_found(&raw));
        let wrapped = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound))
            .context("read voice profile");
        assert!(is_not_found(&wrapped));
        {
            let denied = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
            assert!(!is_not_found(&denied));
            let lookalike = anyhow::anyhow!("No such file or directory (os error 2)");
            assert!(!is_not_found(&lookalike));
        }
    }

    #[test]
    fn response_marks_embedding_state_by_whether_it_is_all_zero() {
        let p = VoiceProfile {
            schema_version: 1,
            name: "n".into(),
            embedding: vec![0.0; 8],
            design_params: None,
        };
        let r = VoiceProfileResponse::from(&p);
        assert_eq!(r.embedding_state, "no_encoder");
        assert_eq!(r.embedding_dim, 8);
        {
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

}
