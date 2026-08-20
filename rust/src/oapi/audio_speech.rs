use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::Json;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, warn};

use crate::oapi::{fastapi_validation_error, kind, missing_field, openai_error};
use crate::AppState;

pub const NV_TTS_SAMPLE_RATE: u32 = 24_000;

#[derive(Debug)]
pub struct InvalidVoiceRequest(pub String);

impl std::fmt::Display for InvalidVoiceRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for InvalidVoiceRequest {}

#[derive(Debug)]
pub struct UnknownVoice(pub String);

impl std::fmt::Display for UnknownVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UnknownVoice {}

#[derive(Debug)]
pub struct SilentVocoder(pub String);

impl std::fmt::Display for SilentVocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SilentVocoder {}

#[async_trait::async_trait]
pub trait AudioSpeech {
    async fn synthesize(&self, text: &str, voice: &str)
        -> anyhow::Result<mpsc::Receiver<Vec<f32>>>;

    fn sample_rate(&self) -> u32 {
        NV_TTS_SAMPLE_RATE
    }

    fn model_id(&self) -> Option<String> {
        None
    }
}

#[derive(Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct AudioSpeechRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: Option<String>,
    pub voice: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub stream_format: Option<String>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

fn is_kokoro_model(model: Option<&str>) -> bool {
    model.unwrap_or("").to_ascii_lowercase().contains("kokoro")
}

pub const KOKORO_MODEL_ID: &str = "speaches-ai/Kokoro-82M-v1.0-ONNX";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechRoute {
    NvTts,
    Kokoro,
}

fn normalize_model_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn model_id_matches(requested: &str, loaded: &str) -> bool {
    let r = normalize_model_id(requested);
    if r.len() < 4 {
        return false;
    }
    let full = normalize_model_id(loaded);
    let base = normalize_model_id(loaded.rsplit('/').next().unwrap_or(loaded));
    full.starts_with(&r) || base.starts_with(&r)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechRouteError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

fn no_tts_loaded() -> SpeechRouteError {
    let talker_env = crate::oapi::audio_speech_nvtts::ENV_TALKER_DIR;
    SpeechRouteError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code: "tts_not_configured",
        message: format!(
            "no text-to-speech model is loaded: set {talker_env} to a Qwen3-TTS checkpoint, or \
             install kokoro-v1.0.onnx + voices.bin under the model directory"
        ),
    }
}

fn unknown_tts_model(message: String) -> SpeechRouteError {
    SpeechRouteError {
        status: StatusCode::NOT_FOUND,
        code: "model_not_found",
        message,
    }
}

fn talker_unavailable(detail: &str) -> SpeechRouteError {
    SpeechRouteError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code: "tts_talker_unavailable",
        message: detail.to_string(),
    }
}

pub fn resolve_speech_route(
    requested: Option<&str>,
    talker_model_id: Option<&str>,
    kokoro_loaded: bool,
    talker_failure: Option<&str>,
) -> Result<SpeechRoute, SpeechRouteError> {
    let talker_env = crate::oapi::audio_speech_nvtts::ENV_TALKER_DIR;
    let requested = requested.map(str::trim).filter(|s| !s.is_empty());
    let Some(req) = requested else {
        return match (talker_model_id.is_some(), talker_failure, kokoro_loaded) {
            (true, _, _) => Ok(SpeechRoute::NvTts),
            (false, Some(detail), _) => Err(talker_unavailable(detail)),
            (false, None, true) => Ok(SpeechRoute::Kokoro),
            (false, None, false) => Err(no_tts_loaded()),
        };
    };
    if is_kokoro_model(Some(req)) {
        return Ok(SpeechRoute::Kokoro);
    }
    match talker_model_id {
        Some(id) if model_id_matches(req, id) => Ok(SpeechRoute::NvTts),
        Some(id) => Err(unknown_tts_model(format!(
            "unknown text-to-speech model {req:?}: this server serves {id:?} (nv-tts, \
             {talker_env}) and {KOKORO_MODEL_ID} (kokoro). The request was refused rather than \
             rendered by whichever engine happened to be loaded -- see GET \
             /v1/models?task=text-to-speech for the served ids"
        ))),
        None if talker_failure.is_some() => {
            Err(talker_unavailable(talker_failure.expect("checked is_some")))
        }
        None if kokoro_loaded => Err(unknown_tts_model(format!(
            "unknown text-to-speech model {req:?}: no nv-tts talker is loaded ({talker_env} is \
             unset, or the checkpoint failed to load), so the only model this server can serve \
             is {KOKORO_MODEL_ID}. Earlier builds answered 200 with Kokoro audio under the \
             requested name; that silent substitution is what this refusal replaces"
        ))),
        None => Err(no_tts_loaded()),
    }
}

pub async fn handle(
    State(state): State<AppState>,
    Json(req): Json<AudioSpeechRequest>,
) -> Response {
    let talker_opt = state.tts_talker.clone();
    let talker_id = talker_opt.as_ref().and_then(|t| t.model_id());
    let route = match resolve_speech_route(
        req.model.as_deref(),
        talker_id.as_deref(),
        state.models.kokoro.is_some(),
        crate::oapi::audio_speech_nvtts::bootstrap_failure(),
    ) {
        Ok(r) => r,
        Err(err) => {
            warn!(
                model = req.model.as_deref().unwrap_or(""),
                served = talker_id.as_deref().unwrap_or("<no nv-tts talker>"),
                reason = %err.message,
                "tts model id rejected"
            );
            let err_kind = if err.status == StatusCode::NOT_FOUND {
                kind::NOT_FOUND
            } else {
                kind::SERVICE_UNAVAIL
            };
            return openai_error(
                err.status,
                err.message,
                err_kind,
                Some("model"),
                Some(err.code),
            );
        }
    };
    let talker = match (route, talker_opt) {
        (SpeechRoute::NvTts, Some(t)) => t,
        _ => return route_to_kokoro(state, req).await,
    };

    let mut entries: Vec<serde_json::Value> = Vec::new();
    if req.input.is_none() {
        entries.push(missing_field(&["body", "input"]));
    }
    if req.voice.is_none() {
        entries.push(missing_field(&["body", "voice"]));
    }
    if !entries.is_empty() {
        return fastapi_validation_error(entries);
    }
    let text = req.input.as_deref().unwrap_or_default();
    let voice = req.voice.as_deref().unwrap_or("alloy");
    let response_format = req
        .response_format
        .clone()
        .unwrap_or_else(|| "wav".to_string())
        .to_lowercase();

    if text.trim().is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "input must not be empty",
            kind::INVALID_REQUEST,
            Some("input"),
            Some("empty_input"),
        );
    }

    let sr = talker.sample_rate();

    if let Some(err) = unsupported_nvtts_params(&req, sr) {
        return err;
    }
    let rx = match talker.synthesize(text, voice).await {
        Ok(rx) => rx,
        Err(err) => {
            if let Some(unknown) = err.downcast_ref::<UnknownVoice>() {
                warn!(error = %unknown, voice, "nv-tts voice not found");
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    unknown.to_string(),
                    kind::INVALID_REQUEST,
                    Some("voice"),
                    Some("invalid_voice"),
                );
            }
            if let Some(bad_voice) = err.downcast_ref::<InvalidVoiceRequest>() {
                warn!(error = %bad_voice, voice, "nv-tts voice rejected");
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("tts: {bad_voice}"),
                    kind::INVALID_REQUEST,
                    Some("voice"),
                    Some("voice_profile_unsupported"),
                );
            }
            if let Some(silent) = err.downcast_ref::<SilentVocoder>() {
                tracing::error!(error = %silent, "nv-tts refused: zero-init vocoder");
                return openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    silent.to_string(),
                    kind::SERVICE_UNAVAIL,
                    None,
                    Some("tts_vocoder_zero_init"),
                );
            }
            warn!(error = %err, "nv-tts synthesize failed");
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("tts: {err}"),
                kind::SERVER,
                None,
                Some("synthesize_failed"),
            );
        }
    };

    if response_format.as_str() == "pcm" {
        debug!(sr, %response_format, "nv-tts streaming pcm");
        return pcm_streaming_response(rx);
    }

    let samples = drain_receiver(rx).await;
    debug!(samples = samples.len(), sr, %response_format, "nv-tts synth complete");

    match response_format.as_str() {
        "wav" => wav_response(&samples, sr),
        "pcm" => unreachable!(),
        "mp3" => openai_error(
            StatusCode::NOT_IMPLEMENTED,
            "response_format=mp3 not yet supported by nv-tts path",
            kind::SERVICE_UNAVAIL,
            Some("response_format"),
            Some("not_implemented"),
        ),
        other => openai_error(
            StatusCode::BAD_REQUEST,
            format!("unsupported response_format: {other:?} (supported: wav, pcm)"),
            kind::INVALID_REQUEST,
            Some("response_format"),
            Some("unsupported_value"),
        ),
    }
}

fn unsupported_nvtts_params(req: &AudioSpeechRequest, sr: u32) -> Option<Response> {
    if let Some(speed) = req.speed {
        if (speed - 1.0).abs() > 1e-3 {
            return Some(openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "speed={speed} is not supported by the nv-tts (Qwen3-TTS) path: it has no \
                     duration control, so the parameter would have been ignored and the audio \
                     returned at speed 1.0. Omit speed, or use model={KOKORO_MODEL_ID}, which \
                     implements speed over 0.5..=2.0"
                ),
                kind::INVALID_REQUEST,
                Some("speed"),
                Some("speed_unsupported"),
            ));
        }
    }
    if let Some(rate) = req.sample_rate {
        if rate != sr {
            return Some(openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "sample_rate={rate} is not supported by the nv-tts (Qwen3-TTS) path: its \
                     vocoder emits {sr} Hz and no resampler is wired in, so the parameter would \
                     have been ignored and {sr} Hz returned. Omit sample_rate, or use \
                     model={KOKORO_MODEL_ID}, which resamples"
                ),
                kind::INVALID_REQUEST,
                Some("sample_rate"),
                Some("sample_rate_unsupported"),
            ));
        }
    }
    if let Some(fmt) = req.stream_format.as_deref() {
        let fmt = fmt.trim().to_lowercase();
        if !fmt.is_empty() && fmt != "audio" {
            return Some(openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "stream_format={fmt:?} is not supported by the nv-tts (Qwen3-TTS) path: it \
                     only emits a raw audio body, so the parameter would have been ignored and \
                     binary audio returned to a client expecting an event stream. Use \
                     response_format=pcm for incremental audio, or model={KOKORO_MODEL_ID}, \
                     which implements stream_format=sse"
                ),
                kind::INVALID_REQUEST,
                Some("stream_format"),
                Some("stream_format_unsupported"),
            ));
        }
    }
    None
}

async fn drain_receiver(mut rx: mpsc::Receiver<Vec<f32>>) -> Vec<f32> {
    let mut all = Vec::new();
    while let Some(chunk) = rx.recv().await {
        all.extend_from_slice(&chunk);
    }
    all
}

fn wav_response(samples: &[f32], sample_rate: u32) -> Response {
    let pcm16: Vec<u8> = samples
        .iter()
        .flat_map(|s| {
            let clipped = (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            clipped.to_le_bytes()
        })
        .collect();
    let mut wav = Vec::with_capacity(44 + pcm16.len());
    write_wav_header(&mut wav, sample_rate, 1, pcm16.len() as u32);
    wav.extend_from_slice(&pcm16);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from(wav))
        .unwrap()
}

fn pcm_streaming_response(rx: mpsc::Receiver<Vec<f32>>) -> Response {
    let stream = ReceiverStream::new(rx).map(|chunk| {
        let pcm16: Vec<u8> = chunk
            .iter()
            .flat_map(|s| {
                let clipped = (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
                clipped.to_le_bytes()
            })
            .collect();
        Ok::<_, std::convert::Infallible>(pcm16)
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn write_wav_header(buf: &mut Vec<u8>, sample_rate: u32, channels: u16, pcm_bytes: u32) {
    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align = channels * 2;
    let chunk_size = 36u32.saturating_add(pcm_bytes);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&pcm_bytes.to_le_bytes());
}

async fn route_to_kokoro(state: AppState, req: AudioSpeechRequest) -> Response {
    let body = json!({
        "model": req.model.clone(),
        "input": req.input.clone(),
        "voice": req.voice.clone(),
        "response_format": req.response_format.clone(),
        "speed": req.speed,
        "stream_format": req.stream_format.clone(),
        "sample_rate": req.sample_rate,
    });

    let kokoro = state.models.kokoro.clone();
    let speech_state = crate::tts::http::SpeechAppState::new(kokoro);

    let kokoro_req: crate::tts::http::CreateSpeechRequestBody = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(err) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("invalid request: {err}"),
                kind::INVALID_REQUEST,
                None,
                Some("invalid_body"),
            );
        }
    };

    crate::tts::http::handle_create_speech(State(speech_state), Json(kokoro_req)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_is_44_bytes_for_zero_pcm() {
        let mut buf = Vec::new();
        write_wav_header(&mut buf, 24_000, 1, 0);
        assert_eq!(buf.len(), 44);
        assert_eq!(&buf[..4], b"RIFF");
        assert_eq!(&buf[8..12], b"WAVE");
        assert_eq!(&buf[12..16], b"fmt ");
        assert_eq!(&buf[36..40], b"data");
    }

    #[test]
    fn wav_header_reports_data_size() {
        let mut buf = Vec::new();
        write_wav_header(&mut buf, 16_000, 1, 1_000);
        let data_len = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(data_len, 1_000);
        let chunk_size = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(chunk_size, 36 + 1_000);
    }

    #[test]
    fn kokoro_dispatch_recognises_known_names() {
        assert!(is_kokoro_model(Some("kokoro")));
        assert!(is_kokoro_model(Some("Kokoro")));
        assert!(is_kokoro_model(Some("speaches-ai/Kokoro-82M-v1.0-ONNX")));
        assert!(!is_kokoro_model(Some(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )));
        assert!(!is_kokoro_model(Some("")));
        assert!(!is_kokoro_model(None));
    }

    const TALKER_ID: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";

    #[test]
    fn route_defaults_when_no_model_is_named() {
        assert_eq!(
            resolve_speech_route(None, Some(TALKER_ID), true, None).unwrap(),
            SpeechRoute::NvTts
        );
        assert_eq!(
            resolve_speech_route(Some("  "), Some(TALKER_ID), true, None).unwrap(),
            SpeechRoute::NvTts
        );
        assert_eq!(
            resolve_speech_route(None, None, true, None).unwrap(),
            SpeechRoute::Kokoro
        );
        let err = resolve_speech_route(None, None, false, None).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "tts_not_configured");
    }

    #[test]
    fn route_accepts_the_loaded_talker_id_and_its_prefixes() {
        for id in [
            TALKER_ID,
            "qwen/qwen3-tts-12hz-0.6b-customvoice",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "qwen3-tts",
            "qwen3_tts",
        ] {
            assert_eq!(
                resolve_speech_route(Some(id), Some(TALKER_ID), true, None).unwrap(),
                SpeechRoute::NvTts,
                "{id} should route to nv-tts"
            );
        }
    }

    #[test]
    fn route_sends_kokoro_ids_to_kokoro_even_with_a_talker_loaded() {
        assert_eq!(
            resolve_speech_route(Some(KOKORO_MODEL_ID), Some(TALKER_ID), true, None).unwrap(),
            SpeechRoute::Kokoro
        );
        assert_eq!(
            resolve_speech_route(Some("kokoro"), None, true, None).unwrap(),
            SpeechRoute::Kokoro
        );
    }

    #[test]
    fn route_rejects_an_absent_talker_instead_of_serving_kokoro() {
        let err = resolve_speech_route(Some(TALKER_ID), None, true, None).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.code, "model_not_found");
        assert!(err.message.contains(TALKER_ID), "{err:?}");
        assert!(err.message.contains("NV_TTS_TALKER_DIR"), "{err:?}");
        assert!(err.message.contains(KOKORO_MODEL_ID), "{err:?}");
    }

    #[test]
    fn route_rejects_a_model_that_is_neither_loaded_engine() {
        for bad in ["tts-1", "gpt-4o-mini-tts", "piper", "xtts-v2"] {
            let err = resolve_speech_route(Some(bad), Some(TALKER_ID), true, None).unwrap_err();
            assert_eq!(err.status, StatusCode::NOT_FOUND);
            assert!(err.message.contains(bad), "{err:?}");
            let err_no_talker = resolve_speech_route(Some(bad), None, true, None).unwrap_err();
            assert_eq!(err_no_talker.status, StatusCode::NOT_FOUND);
            assert!(err_no_talker.message.contains(bad), "{err_no_talker:?}");
        }
    }

    const BOOT_FAIL: &str = "nv-tts bootstrap failed for /bad: missing vocab.json";

    #[test]
    fn a_failed_talker_bootstrap_refuses_instead_of_falling_back_to_kokoro() {
        let err = resolve_speech_route(None, None, true, Some(BOOT_FAIL)).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "tts_talker_unavailable");
        assert!(err.message.contains("missing vocab.json"), "{err:?}");

        let err = resolve_speech_route(Some(TALKER_ID), None, true, Some(BOOT_FAIL)).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "tts_talker_unavailable");

        let err = resolve_speech_route(None, None, false, Some(BOOT_FAIL)).unwrap_err();
        assert_eq!(err.code, "tts_talker_unavailable");
    }

    #[test]
    fn a_failed_talker_bootstrap_still_serves_an_explicit_kokoro_request() {
        assert_eq!(
            resolve_speech_route(Some("kokoro"), None, true, Some(BOOT_FAIL)).unwrap(),
            SpeechRoute::Kokoro
        );
        assert_eq!(
            resolve_speech_route(Some(KOKORO_MODEL_ID), None, true, Some(BOOT_FAIL)).unwrap(),
            SpeechRoute::Kokoro
        );
    }

    fn req_with(
        speed: Option<f32>,
        sample_rate: Option<u32>,
        stream_format: Option<&str>,
    ) -> AudioSpeechRequest {
        AudioSpeechRequest {
            model: None,
            input: Some("hi".into()),
            voice: Some("serena".into()),
            response_format: None,
            speed,
            sample_rate,
            stream_format: stream_format.map(str::to_string),
        }
    }

    #[test]
    fn nvtts_accepts_the_parameter_values_it_actually_honours() {
        assert!(unsupported_nvtts_params(&req_with(None, None, None), 24_000).is_none());
        assert!(unsupported_nvtts_params(
            &req_with(Some(1.0), Some(24_000), Some("audio")),
            24_000
        )
        .is_none());
    }

    #[test]
    fn nvtts_refuses_parameters_it_would_silently_ignore() {
        for (r, param) in [
            (req_with(Some(2.0), None, None), "speed"),
            (req_with(Some(0.5), None, None), "speed"),
            (req_with(None, Some(16_000), None), "sample_rate"),
            (req_with(None, None, Some("sse")), "stream_format"),
        ] {
            let resp = unsupported_nvtts_params(&r, 24_000)
                .unwrap_or_else(|| panic!("{param} must be refused, not ignored"));
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{param}");
        }
    }

    #[test]
    fn model_id_match_rejects_short_and_unrelated_ids() {
        assert!(!model_id_matches("q", TALKER_ID));
        assert!(!model_id_matches("tts", TALKER_ID));
        assert!(!model_id_matches("tts-1", TALKER_ID));
        assert!(model_id_matches("qwen3", TALKER_ID));
    }

    #[test]
    fn wav_response_round_trip_with_silence() {
        let samples = vec![0.0_f32; 8];
        let resp = wav_response(&samples, 24_000);
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(ct, "audio/wav");
    }
}
