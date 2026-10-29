use std::process::Stdio;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::Json;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, warn};

use crate::oapi::{fastapi_validation_error, kind, missing_field, openai_error};
use crate::tts::text::{f32_to_s16le, ResponseFormat};
use crate::tts::voxtral_tts::{
    VOXTRAL_TTS_ALIAS, VOXTRAL_TTS_MODELS_DIR_LINK_NAME, VOXTRAL_TTS_MODEL_DIR_ENV,
    VOXTRAL_TTS_MODEL_ID,
};
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
pub struct InvalidSpeechParam {
    pub param: &'static str,
    pub code: &'static str,
    pub message: String,
}

impl std::fmt::Display for InvalidSpeechParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for InvalidSpeechParam {}

#[derive(Clone, Debug, Default)]
pub struct SpeechOptions {
    pub instructions: Option<String>,
    pub language: Option<String>,
    pub seed: Option<u64>,
}

impl SpeechOptions {
    pub fn instructions(&self) -> Option<&str> {
        self.instructions
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
}

#[async_trait::async_trait]
pub trait AudioSpeech {
    async fn synthesize(
        &self,
        text: &str,
        voice: &str,
        opts: &SpeechOptions,
    ) -> anyhow::Result<mpsc::Receiver<anyhow::Result<Vec<f32>>>>;

    fn sample_rate(&self) -> u32 {
        NV_TTS_SAMPLE_RATE
    }

    fn model_id(&self) -> Option<String> {
        None
    }

    fn voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn engine(&self) -> &'static str {
        "nv-tts (Qwen3-TTS)"
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
    pub language: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub stream_format: Option<String>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn is_kokoro_model(model: Option<&str>) -> bool {
    model.unwrap_or("").to_ascii_lowercase().contains("kokoro")
}

pub const KOKORO_MODEL_ID: &str = "speaches-ai/Kokoro-82M-v1.0-ONNX";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechRoute {
    NvTts,
    Kokoro,
    VoxtralTts,
}

pub fn normalize_model_id(s: &str) -> String {
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

pub fn is_voxtral_tts_model(requested: &str) -> bool {
    normalize_model_id(requested) == normalize_model_id(VOXTRAL_TTS_ALIAS)
        || model_id_matches(requested, VOXTRAL_TTS_MODEL_ID)
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
            "no text-to-speech model is loaded: set {talker_env} to a Qwen3-TTS checkpoint, \
             install kokoro-v1.0.onnx + voices.bin under the model directory, or link a \
             {VOXTRAL_TTS_MODEL_ID} snapshot as <models>/{VOXTRAL_TTS_MODELS_DIR_LINK_NAME} \
             (or set {VOXTRAL_TTS_MODEL_DIR_ENV})"
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

fn served_ids(talker_model_id: Option<&str>, kokoro_loaded: bool, voxtral_loaded: bool) -> String {
    let talker_env = crate::oapi::audio_speech_nvtts::ENV_TALKER_DIR;
    let mut served = Vec::new();
    if let Some(id) = talker_model_id {
        served.push(format!("{id:?} (nv-tts, {talker_env})"));
    }
    if kokoro_loaded {
        served.push(format!("{KOKORO_MODEL_ID} (kokoro)"));
    }
    if voxtral_loaded {
        served.push(format!("{VOXTRAL_TTS_MODEL_ID} (alias {VOXTRAL_TTS_ALIAS})"));
    }
    if served.is_empty() {
        "no text-to-speech model".to_string()
    } else {
        served.join(" and ")
    }
}

pub fn resolve_speech_route(
    requested: Option<&str>,
    talker_model_id: Option<&str>,
    kokoro_loaded: bool,
    talker_failure: Option<&str>,
    voxtral_loaded: bool,
) -> Result<SpeechRoute, SpeechRouteError> {
    let talker_env = crate::oapi::audio_speech_nvtts::ENV_TALKER_DIR;
    let requested = requested.map(str::trim).filter(|s| !s.is_empty());
    let Some(req) = requested else {
        return match (
            talker_model_id.is_some(),
            talker_failure,
            kokoro_loaded,
            voxtral_loaded,
        ) {
            (true, _, _, _) => Ok(SpeechRoute::NvTts),
            (false, Some(detail), _, _) => Err(talker_unavailable(detail)),
            (false, None, true, _) => Ok(SpeechRoute::Kokoro),
            (false, None, false, true) => Ok(SpeechRoute::VoxtralTts),
            (false, None, false, false) => Err(no_tts_loaded()),
        };
    };
    if is_kokoro_model(Some(req)) {
        return Ok(SpeechRoute::Kokoro);
    }
    let served = served_ids(talker_model_id, kokoro_loaded, voxtral_loaded);
    if is_voxtral_tts_model(req) {
        return if voxtral_loaded {
            Ok(SpeechRoute::VoxtralTts)
        } else {
            Err(unknown_tts_model(format!(
                "unknown text-to-speech model {req:?}: {VOXTRAL_TTS_MODEL_ID} is not loaded (no \
                 <models>/{VOXTRAL_TTS_MODELS_DIR_LINK_NAME} link and {VOXTRAL_TTS_MODEL_DIR_ENV} \
                 unset, or the checkpoint failed to load -- see the boot log); this server serves \
                 {served}"
            )))
        };
    }
    match talker_model_id {
        Some(id) if model_id_matches(req, id) => Ok(SpeechRoute::NvTts),
        Some(_) => Err(unknown_tts_model(format!(
            "unknown text-to-speech model {req:?}: this server serves {served}. The request was \
             refused rather than rendered by whichever engine happened to be loaded -- see GET \
             /v1/models?task=text-to-speech for the served ids"
        ))),
        None if talker_failure.is_some() => {
            Err(talker_unavailable(talker_failure.expect("checked is_some")))
        }
        None if kokoro_loaded || voxtral_loaded => Err(unknown_tts_model(format!(
            "unknown text-to-speech model {req:?}: no nv-tts talker is loaded ({talker_env} is \
             unset, or the checkpoint failed to load), so this server serves {served}. Earlier \
             builds answered 200 with Kokoro audio under the requested name; that silent \
             substitution is what this refusal replaces"
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
    let voice_design = crate::oapi::audio_speech_nvtts::voice_design()
        .filter(|vd| vd.serves(req.model.as_deref(), talker_id.as_deref()));
    let route = match voice_design
        .map(|_| Ok(SpeechRoute::NvTts))
        .unwrap_or_else(|| {
            resolve_speech_route(
                req.model.as_deref(),
                talker_id.as_deref(),
                state.models.kokoro.is_some(),
                crate::oapi::audio_speech_nvtts::bootstrap_failure(),
                state.models.voxtral_tts.is_some(),
            )
        }) {
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
    let voxtral = state.models.voxtral_tts.clone();
    let talker: Option<Arc<dyn AudioSpeech + Send + Sync>> =
        match (route, talker_opt, voxtral, voice_design) {
            (_, _, _, Some(_)) => None,
            (SpeechRoute::NvTts, Some(t), _, None) => Some(t),
            (SpeechRoute::VoxtralTts, _, Some(v), None) => Some(v),
            _ => return route_to_kokoro(state, req).await,
        };

    if req.input.is_none() {
        return fastapi_validation_error(vec![missing_field(&["body", "input"])]);
    }
    let text = req.input.as_deref().unwrap_or_default();
    let voice = req.voice.as_deref().map(str::trim).unwrap_or("");
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

    let mut opts = SpeechOptions {
        instructions: req.instructions.clone(),
        language: None,
        seed: req.seed,
    };
    if let Some(vd) = voice_design {
        opts.language = req.language.clone();
        if let Err(err) = vd.validate(&opts) {
            return invalid_param(err);
        }
    } else if opts.instructions().is_some() {
        return invalid_param(InvalidSpeechParam {
            param: "instructions",
            code: "instructions_unsupported",
            message: format!(
                "instructions are not supported by {}: it has no voice-design conditioning, so \
                 the description would have been ignored. Send it to the VoiceDesign model ({}) \
                 instead, or omit instructions",
                talker
                    .as_ref()
                    .and_then(|t| t.model_id())
                    .unwrap_or_default(),
                crate::oapi::audio_speech_nvtts::VOICE_DESIGN_SERVED_NAME
            ),
        });
    }
    let (sr, engine) = match &talker {
        Some(t) => (t.sample_rate(), t.engine()),
        None => (
            NV_TTS_SAMPLE_RATE,
            crate::oapi::audio_speech_nvtts::VOICE_DESIGN_ENGINE,
        ),
    };

    if let Some(err) = unsupported_nvtts_params(&req, sr, engine) {
        return err;
    }
    let format: ResponseFormat = match serde_json::from_value(json!(response_format)) {
        Ok(f) => f,
        Err(_) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format: {response_format:?} (supported: wav, pcm, mp3, \
                     opus, flac, aac)"
                ),
                kind::INVALID_REQUEST,
                Some("response_format"),
                Some("unsupported_value"),
            )
        }
    };
    let talker = match (talker, voice_design) {
        (Some(t), _) => t,
        (None, Some(vd)) => match vd.resident().await {
            Some(engine) => engine,
            None => {
                let message = vd.unavailable_message();
                warn!(model = %vd.model_id, reason = %message, "voice design unavailable");
                return crate::oapi::chat::retryable_response(openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    message,
                    kind::SERVICE_UNAVAIL,
                    Some("model"),
                    Some("voice_design_unavailable"),
                ));
            }
        },
        (None, None) => unreachable!("routed to kokoro above"),
    };
    match talker.synthesize(text, voice, &opts).await {
        Ok(rx) => {
            debug!(sr, ?format, engine, "tts streaming");
            audio_response(rx, sr, format)
        }
        Err(err) => synthesis_error(err, voice, engine),
    }
}

fn synthesis_error(err: anyhow::Error, voice: &str, engine: &'static str) -> Response {
    if let Some(bad) = err.downcast_ref::<InvalidSpeechParam>() {
        return invalid_param(InvalidSpeechParam {
            param: bad.param,
            code: bad.code,
            message: bad.message.clone(),
        });
    }
    if let Some(unknown) = err.downcast_ref::<UnknownVoice>() {
        warn!(error = %unknown, voice, engine, "tts voice not found");
        return openai_error(
            StatusCode::BAD_REQUEST,
            unknown.to_string(),
            kind::INVALID_REQUEST,
            Some("voice"),
            Some("invalid_voice"),
        );
    }
    if let Some(bad_voice) = err.downcast_ref::<InvalidVoiceRequest>() {
        warn!(error = %bad_voice, voice, engine, "tts voice rejected");
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("tts: {bad_voice}"),
            kind::INVALID_REQUEST,
            Some("voice"),
            Some("voice_profile_unsupported"),
        );
    }
    if nv_tts::is_store_unavailable(&err) {
        warn!(error = %err, voice, engine, "tts voice store unavailable");
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("tts: {err:#}"),
            kind::SERVICE_UNAVAIL,
            Some("voice"),
            Some("voice_store_unavailable"),
        );
    }
    let detail = format!("{err:#}");
    if crate::oapi::chat::is_busy_shed(&detail) {
        warn!(error = %detail, engine, "tts synthesis ran out of device memory");
        return crate::oapi::chat::retryable_response(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("tts: the device ran out of memory during synthesis: {detail}"),
            kind::SERVICE_UNAVAIL,
            None,
            Some("insufficient_vram"),
        ));
    }
    warn!(error = %detail, engine, "tts synthesize failed");
    openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("tts: {detail}"),
        kind::SERVER,
        None,
        Some("synthesize_failed"),
    )
}

fn invalid_param(err: InvalidSpeechParam) -> Response {
    warn!(param = err.param, error = %err.message, "tts request rejected");
    openai_error(
        StatusCode::BAD_REQUEST,
        err.message,
        kind::INVALID_REQUEST,
        Some(err.param),
        Some(err.code),
    )
}

fn unsupported_nvtts_params(
    req: &AudioSpeechRequest,
    sr: u32,
    engine: &'static str,
) -> Option<Response> {
    if let Some(speed) = req.speed {
        if (speed - 1.0).abs() > 1e-3 {
            return Some(openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "speed={speed} is not supported by the {engine} path: it has no duration \
                     control, so the parameter would have been ignored and the audio returned at \
                     speed 1.0. Omit speed, or use model={KOKORO_MODEL_ID}, which implements \
                     speed over 0.5..=2.0"
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
                    "sample_rate={rate} is not supported by the {engine} path: its vocoder emits \
                     {sr} Hz and no resampler is wired in, so the parameter would have been \
                     ignored and {sr} Hz returned. Omit sample_rate, or use \
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
                    "stream_format={fmt:?} is not supported by the {engine} path: it only emits \
                     a raw audio body, so the parameter would have been ignored and binary audio \
                     returned to a client expecting an event stream. Use response_format=pcm for \
                     incremental audio, or model={KOKORO_MODEL_ID}, which implements \
                     stream_format=sse"
                ),
                kind::INVALID_REQUEST,
                Some("stream_format"),
                Some("stream_format_unsupported"),
            ));
        }
    }
    None
}

pub const STREAMED_WAV_CARRIES_THE_FFMPEG_PLACEHOLDER_LENGTHS_THE_WAV_DECODER_PATCHES: u32 = u32::MAX;

fn s16le_chunk(chunk: anyhow::Result<Vec<f32>>, format: ResponseFormat) -> std::io::Result<Vec<u8>> {
    chunk.map(|pcm| f32_to_s16le(&pcm)).map_err(|err| {
        warn!(error = %format!("{err:#}"), ?format, "tts stream aborted mid-transfer");
        std::io::Error::other(format!("{err:#}"))
    })
}

fn audio_response(
    rx: mpsc::Receiver<anyhow::Result<Vec<f32>>>,
    sr: u32,
    format: ResponseFormat,
) -> Response {
    let (content_type, head) = match format {
        ResponseFormat::Pcm => ("application/octet-stream", None),
        ResponseFormat::Wav => {
            let mut head = Vec::with_capacity(44);
            write_wav_header(
                &mut head,
                sr,
                1,
                STREAMED_WAV_CARRIES_THE_FFMPEG_PLACEHOLDER_LENGTHS_THE_WAV_DECODER_PATCHES,
            );
            (format.mime_type(), Some(head))
        }
        encoded => return ffmpeg_response(rx, sr, encoded),
    };
    let stream = futures_util::stream::iter(head.map(Ok::<_, std::io::Error>))
        .chain(ReceiverStream::new(rx).map(move |chunk| s16le_chunk(chunk, format)));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from_stream(stream))
        .unwrap()
}

fn ffmpeg_response(
    mut rx: mpsc::Receiver<anyhow::Result<Vec<f32>>>,
    sr: u32,
    format: ResponseFormat,
) -> Response {
    let (muxer, codec) = match format {
        ResponseFormat::Mp3 => ("mp3", Some("libmp3lame")),
        ResponseFormat::Opus => ("opus", Some("libopus")),
        ResponseFormat::Flac => ("flac", None),
        ResponseFormat::Aac => ("adts", Some("aac")),
        ResponseFormat::Pcm | ResponseFormat::Wav => unreachable!("raw formats bypass ffmpeg"),
    };
    let mut cmd = tokio::process::Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar"])
        .arg(sr.to_string())
        .args(["-ac", "1", "-i", "pipe:0", "-f", muxer]);
    if let Some(codec) = codec {
        cmd.args(["-codec:a", codec]);
    }
    cmd.arg("pipe:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            return openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("response_format={muxer} needs ffmpeg on PATH: {err}"),
                kind::SERVICE_UNAVAIL,
                Some("response_format"),
                Some("encoder_unavailable"),
            )
        }
    };
    let mut stdin = child.stdin.take().expect("piped stdin");
    let mut stdout = child.stdout.take().expect("piped stdout");
    let (tx, body) = mpsc::channel::<std::io::Result<Vec<u8>>>(8);
    let feeder_tx = tx.clone();
    tokio::spawn(async move {
        while let Some(chunk) = rx.recv().await {
            let bytes = match s16le_chunk(chunk, format) {
                Ok(bytes) => bytes,
                Err(err) => {
                    let _ = feeder_tx.send(Err(err)).await;
                    break;
                }
            };
            if stdin.write_all(&bytes).await.is_err() {
                break;
            }
        }
    });
    tokio::spawn(async move {
        let mut buf = vec![0u8; 16 * 1024];
        let drained = loop {
            match stdout.read(&mut buf).await {
                Ok(0) => break true,
                Ok(n) => {
                    if tx.send(Ok(buf[..n].to_vec())).await.is_err() {
                        break false;
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                    break false;
                }
            }
        };
        if !drained {
            let _ = child.start_kill();
        }
        if let Ok(status) = child.wait().await {
            if drained && !status.success() {
                warn!(%status, muxer, "ffmpeg encoder failed; aborting the response");
                let _ = tx
                    .send(Err(std::io::Error::other(format!("ffmpeg {muxer} encoder exited with {status}"))))
                    .await;
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.mime_type())
        .body(Body::from_stream(ReceiverStream::new(body)))
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

fn kokoro_request_body(req: &AudioSpeechRequest) -> serde_json::Value {
    json!({
        "model": req.model.clone(),
        "input": req.input.clone(),
        "voice": req.voice.clone(),
        "language": req.language.clone(),
        "response_format": req.response_format.clone(),
        "speed": req.speed,
        "stream_format": req.stream_format.clone(),
        "sample_rate": req.sample_rate,
    })
}

async fn route_to_kokoro(state: AppState, req: AudioSpeechRequest) -> Response {
    let body = kokoro_request_body(&req);
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
    fn wav_header_carries_riff_framing_the_data_size_and_the_streaming_placeholder() {
        let mut buf = Vec::new();
        write_wav_header(&mut buf, 24_000, 1, 0);
        assert_eq!(buf.len(), 44);
        assert_eq!(&buf[..4], b"RIFF");
        assert_eq!(&buf[8..12], b"WAVE");
        assert_eq!(&buf[12..16], b"fmt ");
        assert_eq!(&buf[36..40], b"data");
        {
            let mut buf = Vec::new();
            write_wav_header(&mut buf, 16_000, 1, 1_000);
            let data_len = u32::from_le_bytes(buf[40..44].try_into().unwrap());
            assert_eq!(data_len, 1_000);
            let chunk_size = u32::from_le_bytes(buf[4..8].try_into().unwrap());
            assert_eq!(chunk_size, 36 + 1_000);
        }
        {
            let mut buf = Vec::new();
            write_wav_header(
                &mut buf,
                24_000,
                1,
                STREAMED_WAV_CARRIES_THE_FFMPEG_PLACEHOLDER_LENGTHS_THE_WAV_DECODER_PATCHES,
            );
            assert_eq!(buf[4..8], [0xFF; 4], "riff size placeholder");
            assert_eq!(buf[40..44], [0xFF; 4], "data size placeholder");
        }
    }

    #[tokio::test]
    async fn streamed_wav_and_pcm_bodies_decode_through_the_wav_decoder_and_carry_their_types() {
        for (format, content_type, header_bytes) in [
            (ResponseFormat::Wav, "audio/wav", 44usize),
            (ResponseFormat::Pcm, "application/octet-stream", 0),
        ] {
            let (tx, rx) = mpsc::channel(4);
            tx.send(Ok(vec![0.25f32; 2_400])).await.unwrap();
            tx.send(Ok(vec![-0.25f32; 2_400])).await.unwrap();
            drop(tx);
            let resp = audio_response(rx, 24_000, format);
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                resp.headers().get(header::CONTENT_TYPE).unwrap(),
                content_type,
                "{format:?}"
            );
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(body.len(), header_bytes + 4_800 * 2, "{format:?}");
            if format == ResponseFormat::Wav {
                let decoded = crate::audio::wav::decode_wav_to_16k_mono(&body).unwrap();
                assert_eq!(decoded.len(), 3_200, "0.2 s at 16 kHz");
            }
        }
    }

    #[test]
    fn kokoro_dispatch_and_model_id_match_recognise_known_names_only() {
        assert!(is_kokoro_model(Some("kokoro")));
        assert!(is_kokoro_model(Some("Kokoro")));
        assert!(is_kokoro_model(Some("speaches-ai/Kokoro-82M-v1.0-ONNX")));
        assert!(!is_kokoro_model(Some(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )));
        assert!(!is_kokoro_model(Some("")));
        assert!(!is_kokoro_model(None));
        {
            assert!(!model_id_matches("q", TALKER_ID));
            assert!(!model_id_matches("tts", TALKER_ID));
            assert!(!model_id_matches("tts-1", TALKER_ID));
            assert!(model_id_matches("qwen3", TALKER_ID));
        }
        {
            for id in [
                VOXTRAL_TTS_MODEL_ID,
                "mistralai/voxtral-4b-tts-2603",
                "Voxtral-4B-TTS-2603",
                "voxtral-tts",
                "voxtral_tts",
                "voxtral",
            ] {
                assert!(is_voxtral_tts_model(id), "{id}");
            }
            for id in ["vox", "tts-1", TALKER_ID, KOKORO_MODEL_ID] {
                assert!(!is_voxtral_tts_model(id), "{id}");
            }
        }
    }

    const TALKER_ID: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";

    #[test]
    fn route_defaults_to_the_talker_and_accepts_talker_and_kokoro_ids() {
        assert_eq!(
            resolve_speech_route(None, Some(TALKER_ID), true, None, false).unwrap(),
            SpeechRoute::NvTts
        );
        assert_eq!(
            resolve_speech_route(Some("  "), Some(TALKER_ID), true, None, false).unwrap(),
            SpeechRoute::NvTts
        );
        assert_eq!(
            resolve_speech_route(None, None, true, None, false).unwrap(),
            SpeechRoute::Kokoro
        );
        let err = resolve_speech_route(None, None, false, None, false).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "tts_not_configured");
        {
            for id in [
                TALKER_ID,
                "qwen/qwen3-tts-12hz-0.6b-customvoice",
                "Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "qwen3-tts",
                "qwen3_tts",
            ] {
                assert_eq!(
                    resolve_speech_route(Some(id), Some(TALKER_ID), true, None, false).unwrap(),
                    SpeechRoute::NvTts,
                    "{id} should route to nv-tts"
                );
            }
        }
        {
            assert_eq!(
                resolve_speech_route(Some(KOKORO_MODEL_ID), Some(TALKER_ID), true, None, false)
                    .unwrap(),
                SpeechRoute::Kokoro
            );
            assert_eq!(
                resolve_speech_route(Some("kokoro"), None, true, None, false).unwrap(),
                SpeechRoute::Kokoro
            );
        }
    }

    #[test]
    fn voxtral_tts_routes_by_id_or_alias_defaults_only_when_alone_and_404s_when_not_loaded() {
        for id in [VOXTRAL_TTS_MODEL_ID, VOXTRAL_TTS_ALIAS, "voxtral"] {
            assert_eq!(
                resolve_speech_route(Some(id), Some(TALKER_ID), true, None, true).unwrap(),
                SpeechRoute::VoxtralTts,
                "{id}"
            );
            let err = resolve_speech_route(Some(id), Some(TALKER_ID), true, None, false).unwrap_err();
            assert_eq!(err.status, StatusCode::NOT_FOUND, "{id}");
            assert_eq!(err.code, "model_not_found");
            assert!(err.message.contains(VOXTRAL_TTS_MODELS_DIR_LINK_NAME), "{err:?}");
            assert!(err.message.contains(TALKER_ID), "{err:?}");
        }
        {
            assert_eq!(
                resolve_speech_route(None, Some(TALKER_ID), true, None, true).unwrap(),
                SpeechRoute::NvTts
            );
            assert_eq!(
                resolve_speech_route(None, None, true, None, true).unwrap(),
                SpeechRoute::Kokoro
            );
            assert_eq!(
                resolve_speech_route(None, None, false, None, true).unwrap(),
                SpeechRoute::VoxtralTts
            );
        }
        {
            let err = resolve_speech_route(Some("tts-1"), None, false, None, true).unwrap_err();
            assert_eq!(err.status, StatusCode::NOT_FOUND);
            assert!(err.message.contains(VOXTRAL_TTS_MODEL_ID), "{err:?}");
            assert!(err.message.contains(VOXTRAL_TTS_ALIAS), "{err:?}");
            let err = resolve_speech_route(None, None, false, None, false).unwrap_err();
            assert!(err.message.contains(VOXTRAL_TTS_MODELS_DIR_LINK_NAME), "{err:?}");
        }
    }

    #[test]
    fn route_rejects_an_absent_talker_and_unknown_models_with_404() {
        let err = resolve_speech_route(Some(TALKER_ID), None, true, None, false).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.code, "model_not_found");
        assert!(err.message.contains(TALKER_ID), "{err:?}");
        assert!(err.message.contains("NV_TTS_TALKER_DIR"), "{err:?}");
        assert!(err.message.contains(KOKORO_MODEL_ID), "{err:?}");
        {
            for bad in ["tts-1", "gpt-4o-mini-tts", "piper", "xtts-v2"] {
                let err =
                    resolve_speech_route(Some(bad), Some(TALKER_ID), true, None, false).unwrap_err();
                assert_eq!(err.status, StatusCode::NOT_FOUND);
                assert!(err.message.contains(bad), "{err:?}");
                let err_no_talker =
                    resolve_speech_route(Some(bad), None, true, None, false).unwrap_err();
                assert_eq!(err_no_talker.status, StatusCode::NOT_FOUND);
                assert!(err_no_talker.message.contains(bad), "{err_no_talker:?}");
            }
        }
    }

    const BOOT_FAIL: &str = "nv-tts bootstrap failed for /bad: missing vocab.json";

    #[test]
    fn a_failed_talker_bootstrap_refuses_defaults_but_still_serves_explicit_kokoro() {
        let err = resolve_speech_route(None, None, true, Some(BOOT_FAIL), false).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "tts_talker_unavailable");
        assert!(err.message.contains("missing vocab.json"), "{err:?}");

        let err =
            resolve_speech_route(Some(TALKER_ID), None, true, Some(BOOT_FAIL), false).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "tts_talker_unavailable");

        let err = resolve_speech_route(None, None, false, Some(BOOT_FAIL), false).unwrap_err();
        assert_eq!(err.code, "tts_talker_unavailable");
        {
            assert_eq!(
                resolve_speech_route(Some("kokoro"), None, true, Some(BOOT_FAIL), false).unwrap(),
                SpeechRoute::Kokoro
            );
            assert_eq!(
                resolve_speech_route(Some(KOKORO_MODEL_ID), None, true, Some(BOOT_FAIL), false)
                    .unwrap(),
                SpeechRoute::Kokoro
            );
            assert_eq!(
                resolve_speech_route(Some(VOXTRAL_TTS_ALIAS), None, true, Some(BOOT_FAIL), true)
                    .unwrap(),
                SpeechRoute::VoxtralTts
            );
        }
    }

    fn req_with(
        speed: Option<f32>,
        sample_rate: Option<u32>,
        stream_format: Option<&str>,
    ) -> AudioSpeechRequest {
        AudioSpeechRequest {
            language: None,
            model: None,
            input: Some("hi".into()),
            voice: Some("serena".into()),
            response_format: None,
            speed,
            sample_rate,
            stream_format: stream_format.map(str::to_string),
            instructions: None,
            seed: None,
        }
    }

    #[test]
    fn public_speech_route_preserves_explicit_spanish_language_for_kokoro() {
        let request: AudioSpeechRequest = serde_json::from_value(json!({
            "model":"kokoro", "input":"Buenas tardes", "voice":"em_alex", "language":"es"
        }))
        .unwrap();
        let forwarded: crate::tts::http::CreateSpeechRequestBody =
            serde_json::from_value(kokoro_request_body(&request)).unwrap();
        assert_eq!(forwarded.language.as_deref(), Some("es"));
        assert_eq!(forwarded.voice.as_deref(), Some("em_alex"));
    }

    fn producer(chunks: usize, tail: Option<&str>) -> mpsc::Receiver<anyhow::Result<Vec<f32>>> {
        let (tx, rx) = mpsc::channel(chunks + 1);
        for _ in 0..chunks {
            tx.try_send(Ok(vec![0.25f32; 24_000])).unwrap();
        }
        if let Some(msg) = tail {
            tx.try_send(Err(anyhow::anyhow!(msg.to_string()))).unwrap();
        }
        rx
    }

    async fn body_len(resp: Response) -> Result<usize, axum::Error> {
        Ok(http_body_util::BodyExt::collect(resp.into_body())
            .await?
            .to_bytes()
            .len())
    }

    const OOM: &str = "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")";
    const PANIC: &str = "vocoder decode task panicked";

    #[tokio::test]
    async fn a_synthesis_that_fails_after_it_started_never_answers_a_short_200() {
        let whole = audio_response(producer(3, None), 24_000, ResponseFormat::Wav);
        assert_eq!(whole.status(), StatusCode::OK);
        assert_eq!(body_len(whole).await.unwrap(), 44 + 3 * 24_000 * 2);
        for format in [ResponseFormat::Wav, ResponseFormat::Pcm, ResponseFormat::Flac] {
            for tail in [PANIC, OOM] {
                let cut = audio_response(producer(3, Some(tail)), 24_000, format);
                if cut.status() == StatusCode::SERVICE_UNAVAILABLE {
                    eprintln!("skip {format:?}: ffmpeg is not on PATH");
                    continue;
                }
                assert_eq!(cut.status(), StatusCode::OK, "{format:?} streams before the failure is known");
                assert!(
                    body_len(cut).await.is_err(),
                    "a {format:?} stream that fails mid-transfer must abort the body, not end it cleanly"
                );
            }
        }
        let clean = audio_response(producer(2, None), 24_000, ResponseFormat::Pcm);
        assert_eq!(body_len(clean).await.unwrap(), 2 * 24_000 * 2);
    }

    #[test]
    fn a_synthesis_that_fails_before_its_first_chunk_sheds_an_oom_as_a_retryable_503() {
        let oom = synthesis_error(anyhow::anyhow!(OOM), "", "voxtral-tts");
        assert_eq!(oom.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(oom.headers().get(header::RETRY_AFTER).unwrap(), "1");
        let other = synthesis_error(anyhow::anyhow!(PANIC), "", "voxtral-tts");
        assert_eq!(other.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(other.headers().get(header::RETRY_AFTER).is_none());
        let voice = synthesis_error(UnknownVoice("no such voice".into()).into(), "x", "voxtral-tts");
        assert_eq!(voice.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn nvtts_accepts_the_parameters_it_honours_and_refuses_the_ones_it_would_ignore() {
        const ENGINE: &str = "voxtral-tts";
        assert!(unsupported_nvtts_params(&req_with(None, None, None), 24_000, ENGINE).is_none());
        assert!(unsupported_nvtts_params(
            &req_with(Some(1.0), Some(24_000), Some("audio")),
            24_000,
            ENGINE
        )
        .is_none());
        {
            for (r, param) in [
                (req_with(Some(2.0), None, None), "speed"),
                (req_with(Some(0.5), None, None), "speed"),
                (req_with(None, Some(16_000), None), "sample_rate"),
                (req_with(None, None, Some("sse")), "stream_format"),
            ] {
                let resp = unsupported_nvtts_params(&r, 24_000, ENGINE)
                    .unwrap_or_else(|| panic!("{param} must be refused, not ignored"));
                assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{param}");
            }
        }
    }
}
