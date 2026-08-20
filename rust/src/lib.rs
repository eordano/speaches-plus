#![allow(dead_code)]

pub mod audio;
pub mod conversation;
pub mod defaults;
pub mod diarization;
pub mod eou;
pub mod errors;
pub mod ids;
pub mod inspect;
pub(crate) mod mel_scale;
pub mod models;
pub mod oapi;
pub mod otel;
pub mod pii;
pub mod realtime;
pub mod soak;
pub mod stt;
pub mod trace;
pub mod tts;
pub mod types;
pub mod vad;

use std::sync::Arc;

use serde::Deserialize;

#[derive(Clone)]
pub struct AppState {
    pub models: Arc<models::Models>,

    pub chat_engine: Option<Arc<dyn oapi::chat::ChatEngine>>,

    pub chat_registry: Option<oapi::chat_engine::ChatRegistry>,

    pub tts_talker: Option<Arc<dyn oapi::audio_speech::AudioSpeech + Send + Sync>>,

    pub speaker_encoder: Option<Arc<nv_tts::SpeakerEncoder>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RealtimeQuery {
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub transcription_model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub speech_model: Option<String>,
}
