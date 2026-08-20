#![allow(dead_code)]

pub static SELF_ADDR: std::sync::OnceLock<std::net::SocketAddr> = std::sync::OnceLock::new();

pub mod admission;
pub mod audio_speech;
pub mod audio_speech_nvtts;
pub mod backend_select;
pub mod batch_chat;
pub mod chat;
pub mod chat_engine;
#[cfg(feature = "wgpu")]
pub mod chat_engine_wgpu;
pub mod chat_multimodal;
pub mod chat_multimodal_qwen3;
pub mod chat_template;
pub mod completions;
pub mod deadline;
pub mod fine_tuning;
pub mod gate;
pub mod lora;
pub mod messages;
pub mod model_ids;
pub mod responses;
pub mod models_handler;
pub mod ocr;
pub mod ocr_batch;
pub mod ocr_orient;
pub mod ocr_suspect;
#[cfg(feature = "cuda")]
pub mod ocr_batch_n;
pub mod text_embeddings;
pub mod tool_parse;
pub mod transcriptions;
pub mod voice_profiles;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::json;

pub mod kind {
    pub const INVALID_REQUEST: &str = "invalid_request_error";
    pub const AUTH: &str = "authentication_error";
    pub const NOT_FOUND: &str = "not_found_error";
    pub const SERVER: &str = "internal_server_error";
    pub const SERVICE_UNAVAIL: &str = "service_unavailable_error";
    pub const RATE_LIMIT: &str = "rate_limit_error";
}

pub fn constant_time_eq(candidate: &[u8], secret: &[u8]) -> bool {
    let mut diff = (candidate.len() ^ secret.len()) as u64;
    for (i, &s) in secret.iter().enumerate() {
        diff |= (candidate.get(i).copied().unwrap_or(0) ^ s) as u64;
    }
    diff == 0
}

pub fn request_key_ok(headers: &axum::http::HeaderMap, key: &str) -> bool {
    let bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|h| {
            constant_time_eq(
                h.strip_prefix("Bearer ").unwrap_or(h).as_bytes(),
                key.as_bytes(),
            )
        })
        .unwrap_or(false);
    let x_api_key = headers
        .get("x-api-key")
        .map(|v| constant_time_eq(v.as_bytes(), key.as_bytes()))
        .unwrap_or(false);
    bearer || x_api_key
}

pub fn openai_error(
    status: StatusCode,
    message: impl Into<String>,
    err_type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    let body = json!({
        "error": {
            "message": message.into(),
            "type": err_type,
            "param": param,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}

pub fn fastapi_validation_error(entries: Vec<serde_json::Value>) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({ "detail": entries })),
    )
        .into_response()
}

pub fn missing_field(loc: &[&str]) -> serde_json::Value {
    json!({
        "type": "missing",
        "loc": loc,
        "msg": "Field required",
    })
}

#[derive(Clone, Debug)]
pub struct Model {
    pub id: String,
    pub created: i64,
    pub owned_by: String,
    pub languages: Option<Vec<String>>,
    pub task: String,

    pub max_model_len: Option<u64>,
    pub extras: serde_json::Map<String, serde_json::Value>,
}

impl Serialize for Model {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let total = 7 + self.max_model_len.is_some() as usize + self.extras.len();
        let mut map = serializer.serialize_map(Some(total))?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("object", "model")?;
        map.serialize_entry("created", &self.created)?;
        map.serialize_entry("owned_by", &self.owned_by)?;

        map.serialize_entry("root", &self.id)?;
        if let Some(len) = self.max_model_len {
            map.serialize_entry("max_model_len", &len)?;
        }
        map.serialize_entry("language", &self.languages)?;
        map.serialize_entry("task", &self.task)?;
        for (k, v) in &self.extras {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ListModelsResponse {
    pub object: &'static str,
    pub data: Vec<Model>,
}

impl ListModelsResponse {
    pub fn new(data: Vec<Model>) -> Self {
        Self {
            object: "list",
            data,
        }
    }
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export, rename = "Model")]
    struct ModelWire {
        id: String,
        #[ts(type = "\"model\"")]
        object: (),
        created: i64,
        owned_by: String,
        root: String,
        #[ts(optional)]
        max_model_len: Option<u64>,
        language: Option<Vec<String>>,
        task: String,
        #[ts(optional)]
        sample_rate: Option<u32>,
        #[ts(optional)]
        voices: Option<Vec<String>>,
        #[ts(optional)]
        spec_decode: Option<String>,
        #[ts(optional)]
        parent: Option<String>,
        #[ts(
            optional,
            type = "{ rank: number, alpha: number, scaling: number, target_modules: Array<string>, path: string }"
        )]
        lora: Option<()>,
    }

    #[derive(TS)]
    #[ts(export, rename = "ListModelsResponse")]
    struct ListModelsResponseWire {
        #[ts(type = "\"list\"")]
        object: (),
        data: Vec<ModelWire>,
    }
}

pub mod task {
    pub const ASR: &str = "automatic-speech-recognition";
    pub const TTS: &str = "text-to-speech";
    pub const VAD: &str = "voice-activity-detection";
    pub const EMBEDDINGS: &str = "embeddings";
    pub const TOKEN_CLASSIFICATION: &str = "token-classification";
    pub const DIARIZATION: &str = "speaker-diarization";
    pub const SPEAKER_EMBEDDING: &str = "speaker-embedding";
}

pub const WHISPER_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

pub const KOKORO_LANGUAGES: &[&str] = &["en", "es", "fr", "hi", "it", "ja", "pt", "zh"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_key_ok_accepts_bearer_and_x_api_key() {
        let empty = axum::http::HeaderMap::new();
        assert!(!request_key_ok(&empty, "k"));

        let mut bearer = axum::http::HeaderMap::new();
        bearer.insert(axum::http::header::AUTHORIZATION, "Bearer k".parse().unwrap());
        assert!(request_key_ok(&bearer, "k"));

        let mut xkey = axum::http::HeaderMap::new();
        xkey.insert("x-api-key", "k".parse().unwrap());
        assert!(request_key_ok(&xkey, "k"));

        let mut wrong = axum::http::HeaderMap::new();
        wrong.insert(axum::http::header::AUTHORIZATION, "Bearer nope".parse().unwrap());
        wrong.insert("x-api-key", "nope".parse().unwrap());
        assert!(!request_key_ok(&wrong, "k"));
    }

    #[test]
    fn model_serializes_with_extras() {
        let m = Model {
            id: "x".into(),
            created: 1,
            owned_by: "owner".into(),
            languages: Some(vec!["en".into()]),
            task: task::TTS.into(),
            max_model_len: None,
            extras: {
                let mut e = serde_json::Map::new();
                e.insert("voices".into(), json!(["a", "b"]));
                e.insert("sample_rate".into(), json!(24000));
                e
            },
        };
        let s = serde_json::to_value(&m).unwrap();
        assert_eq!(s["id"], "x");
        assert_eq!(s["object"], "model");
        assert_eq!(s["root"], "x");
        assert_eq!(s["task"], task::TTS);
        assert_eq!(s["voices"], json!(["a", "b"]));
        assert_eq!(s["sample_rate"], 24000);
    }

    #[test]
    fn list_response_has_object_list() {
        let r = ListModelsResponse::new(vec![]);
        let s = serde_json::to_value(&r).unwrap();
        assert_eq!(s["object"], "list");
        assert_eq!(s["data"], json!([]));
    }
}
