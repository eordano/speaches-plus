use std::path::Path;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map};

use super::{
    kind, openai_error, task, ListModelsResponse, Model, KOKORO_LANGUAGES, WHISPER_LANGUAGES,
};
use crate::models::Models;

#[derive(Debug, Default, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ModelsQuery {
    pub task: Option<String>,
}

pub async fn handle_list_models(
    State(state): State<crate::AppState>,
    Query(q): Query<ModelsQuery>,
) -> Response {
    let mut data: Vec<Model> = build_models(&state.models);
    if let Some(talker) = &state.tts_talker {
        if let Some(id) = talker.model_id() {
            let owner = id.split('/').next().unwrap_or("speaches-plus").to_string();
            let mut extras = Map::new();
            extras.insert("sample_rate".into(), json!(talker.sample_rate()));
            data.push(Model {
                id,
                created: 1,
                owned_by: owner,
                languages: None,
                task: task::TTS.into(),
                max_model_len: None,
                extras,
            });
        }
    }
    if state.speaker_encoder.is_some() {
        data.push(Model {
            id: "nv-tts/speaker-encoder".into(),
            created: 1,
            owned_by: "speaches-plus".into(),
            languages: None,
            task: task::SPEAKER_EMBEDDING.into(),
            max_model_len: None,
            extras: Map::new(),
        });
    }
    if state.models.diar_segmentation.is_some() && state.models.diar_embedding.is_some() {
        data.push(Model {
            id: "diarizen-segmentation+wespeaker".into(),
            created: 1,
            owned_by: "speaches-plus".into(),
            languages: None,
            task: task::DIARIZATION.into(),
            max_model_len: None,
            extras: Map::new(),
        });
    }
    if let Some(reg) = &state.chat_registry {
        for id in reg.model_ids() {
            let owner = id.split('/').next().unwrap_or("speaches-plus").to_string();
            let mut extras = Map::new();
            let spec = reg
                .resolve(Some(id))
                .map(|e| super::chat::spec_decode_header_value(e.spec_decode_status()))
                .unwrap_or("unknown");
            extras.insert("spec_decode".into(), json!(spec));
            data.push(Model {
                id: id.clone(),
                created: 1,
                owned_by: owner,
                languages: None,
                task: "chat".into(),
                max_model_len: None,
                extras,
            });
        }
    }

    for m in super::lora::model_rows() {
        match data.iter_mut().find(|x| x.id == m.id) {
            Some(existing) => {
                existing.owned_by = m.owned_by;
                for (k, v) in m.extras {
                    existing.extras.entry(k).or_insert(v);
                }
            }
            None => data.push(m),
        }
    }
    if let Some(filter) = q.task.as_deref() {
        data.retain(|m| m.task == filter);
    }
    Json(ListModelsResponse::new(data)).into_response()
}

#[allow(dead_code)]
pub fn models_unavailable() -> Response {
    openai_error(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "models registry not initialised",
        kind::SERVICE_UNAVAIL,
        None,
        Some("models_not_loaded"),
    )
}

fn build_models(models: &Arc<Models>) -> Vec<Model> {
    let mut out: Vec<Model> = Vec::new();

    if let Some(w) = models.whisper_opt() {
        let whisper_id = w.model_id().to_string();
        let (whisper_id, whisper_owner) = whisper_id_and_owner(&whisper_id);
        out.push(Model {
            id: whisper_id,
            created: 1,
            owned_by: whisper_owner,
            languages: Some(WHISPER_LANGUAGES.iter().map(|s| s.to_string()).collect()),
            task: task::ASR.into(),
            max_model_len: None,
            extras: Map::new(),
        });
    }

    if let Some(kokoro) = &models.kokoro {
        let voices: Vec<String> = kokoro.voice_names();
        let mut extras = Map::new();
        extras.insert("sample_rate".into(), json!(24_000));
        extras.insert("voices".into(), json!(voices));
        out.push(Model {
            id: "speaches-ai/Kokoro-82M-v1.0-ONNX".into(),
            created: 1,
            owned_by: "speaches-ai".into(),
            languages: Some(KOKORO_LANGUAGES.iter().map(|s| s.to_string()).collect()),
            task: task::TTS.into(),
            max_model_len: None,
            extras,
        });
    }

    if models.vad().is_ok() {
        out.push(Model {
            id: "silero_vad_v6".into(),
            created: 1,
            owned_by: "snakers4".into(),
            languages: None,
            task: task::VAD.into(),
            max_model_len: None,
            extras: Map::new(),
        });
    }

    out.extend(embeddings_model_entry(
        super::text_embeddings::loaded_embedding_model_id(),
    ));
    out.extend(pii_model_entry(loaded_pii_model_id()));

    out
}

fn embeddings_model_entry(loaded_id: Option<String>) -> Option<Model> {
    let embed_id = loaded_id?;
    let owner = embed_id
        .split('/')
        .next()
        .unwrap_or("speaches-plus")
        .to_string();
    Some(Model {
        id: embed_id,
        created: 1,
        owned_by: owner,
        languages: None,
        task: task::EMBEDDINGS.into(),
        max_model_len: None,
        extras: Map::new(),
    })
}

fn pii_model_entry(loaded_dir: Option<String>) -> Option<Model> {
    let pii_id = loaded_dir?;
    let name = pii_id.split('/').next_back().unwrap_or(&pii_id).to_string();
    let owner = if pii_id.contains('/') {
        pii_id.split('/').next().unwrap_or("openai").to_string()
    } else {
        "openai".into()
    };
    Some(Model {
        id: name,
        created: 1,
        owned_by: owner,
        languages: None,
        task: task::TOKEN_CLASSIFICATION.into(),
        max_model_len: None,
        extras: Map::new(),
    })
}

static PII_LOADED_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

pub fn note_pii_loaded(dir: &std::path::Path) {
    let _ = PII_LOADED_DIR.set(Some(dir.display().to_string()));
}

fn loaded_pii_model_id() -> Option<String> {
    PII_LOADED_DIR.get().cloned().flatten()
}

fn whisper_id_and_owner(input: &str) -> (String, String) {
    let has_ext = Path::new(input).extension().is_some();
    let looks_like_hf =
        input.contains('/') && !input.starts_with('/') && !input.starts_with('.') && !has_ext;
    if looks_like_hf {
        let owner = input.split('/').next().unwrap_or("unknown").to_string();
        return (input.to_string(), owner);
    }
    let base = Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(input)
        .to_string();
    (base, "speaches-plus".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_and_pii_rows_follow_load_state_in_both_directions() {
        assert!(
            embeddings_model_entry(None).is_none(),
            "/v1/models must not advertise an embeddings model that never loaded"
        );
        assert!(
            pii_model_entry(None).is_none(),
            "/v1/models must not advertise a PII model that never loaded"
        );

        let e = embeddings_model_entry(Some("Qwen/Qwen3-Embedding-0.6B".into()))
            .expect("a loaded embedder must be advertised");
        assert_eq!(e.id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(e.owned_by, "Qwen");
        assert_eq!(e.task, task::EMBEDDINGS);

        let p = pii_model_entry(Some("/models/piiranha-v1".into()))
            .expect("a loaded PII classifier must be advertised");
        assert_eq!(p.id, "piiranha-v1");
        assert_eq!(p.task, task::TOKEN_CLASSIFICATION);
    }

    #[test]
    fn pii_is_not_advertised_from_the_env_var_alone() {
        std::env::set_var("REDACT_MODEL_DIR", "/no/such/pii/dir");
        assert!(
            loaded_pii_model_id().is_none(),
            "REDACT_MODEL_DIR alone must not put a model in /v1/models"
        );
        std::env::remove_var("REDACT_MODEL_DIR");
    }

    #[test]
    fn whisper_id_and_owner_handles_hf_ids() {
        let (id, owner) = whisper_id_and_owner("deepdml/faster-whisper-large-v3-turbo-ct2");
        assert_eq!(id, "deepdml/faster-whisper-large-v3-turbo-ct2");
        assert_eq!(owner, "deepdml");
    }

    #[test]
    fn whisper_id_and_owner_handles_local_paths() {
        let (id, owner) = whisper_id_and_owner("models/ggml-tiny.en.bin");
        assert_eq!(id, "ggml-tiny.en");
        assert_eq!(owner, "speaches-plus");
    }

    #[test]
    fn whisper_id_and_owner_handles_bare_dir_name() {
        let (id, owner) = whisper_id_and_owner("ct2-large-v3-turbo");
        assert_eq!(id, "ct2-large-v3-turbo");
        assert_eq!(owner, "speaches-plus");
    }
}
