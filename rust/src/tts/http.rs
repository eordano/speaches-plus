use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, warn};

use super::text::{
    self as speech, f32_to_s16le, is_openai_voice_alias, normalize_for_tts, strip_emojis,
    strip_markdown_emphasis, ResponseFormat, StreamFormat, KOKORO_SAMPLE_RATE, MAX_CHUNK_CHARS,
    MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, SPEED_MAX, SPEED_MIN,
};
use super::KokoroHandle;
use crate::defaults;

#[derive(Debug, Deserialize)]
pub struct CreateSpeechRequestBody {
    pub model: Option<String>,
    pub input: Option<String>,
    pub voice: Option<String>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub stream_format: Option<StreamFormat>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

#[derive(Clone)]
pub struct SpeechAppState {
    pub kokoro: Option<KokoroHandle>,
    pub language: Arc<str>,
}

impl SpeechAppState {
    pub fn new(kokoro: Option<KokoroHandle>) -> Self {
        Self {
            kokoro,
            language: Arc::from(speech::DEFAULT_LANGUAGE),
        }
    }
}

use crate::oapi::{fastapi_validation_error, kind, missing_field as missing, openai_error};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn max_input_chars() -> usize {
    env_usize(
        defaults::env::KOKORO_MAX_INPUT_CHARS,
        defaults::kokoro::MAX_INPUT_CHARS,
    )
}

fn chunk_chars() -> usize {
    env_usize(
        defaults::env::KOKORO_CHUNK_CHARS,
        defaults::kokoro::CHUNK_CHARS,
    )
    .min(MAX_CHUNK_CHARS)
}

fn join_silence_prefixed(samples: Vec<f32>) -> Vec<f32> {
    let n = KOKORO_SAMPLE_RATE as usize * defaults::kokoro::JOIN_SILENCE_MS / 1000;
    let mut out = vec![0.0f32; n];
    out.extend_from_slice(&samples);
    out
}

fn queue_wait() -> Duration {
    Duration::from_secs(env_u64(
        defaults::env::KOKORO_QUEUE_WAIT_S,
        defaults::kokoro::QUEUE_WAIT_S,
    ))
}

fn synth_budget(n_chars: usize) -> Duration {
    let base = env_u64(
        defaults::env::KOKORO_SYNTH_BUDGET_S,
        defaults::kokoro::SYNTH_BUDGET_S,
    );
    let units = (n_chars / defaults::kokoro::SYNTH_BUDGET_UNIT_CHARS) as u64 + 1;
    Duration::from_secs(base.saturating_mul(units))
}

fn ffmpeg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

#[derive(Debug)]
enum SynthError {
    Busy,
    Failed(String),
}

impl SynthError {
    fn message(&self) -> String {
        match self {
            SynthError::Busy => "TTS queue overloaded; retry shortly".to_string(),
            SynthError::Failed(m) => m.clone(),
        }
    }
}

async fn synth_chunk(
    kokoro: KokoroHandle,
    chunk: String,
    voice: String,
    lang: Arc<str>,
    speed: f32,
) -> Result<Vec<f32>, SynthError> {
    let permit = match tokio::time::timeout(queue_wait(), kokoro.queue().acquire_owned()).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return Err(SynthError::Failed(format!("kokoro queue closed: {e}"))),
        Err(_) => return Err(SynthError::Busy),
    };
    match tokio::task::spawn_blocking(move || {
        let out = kokoro.synthesize(&chunk, Some(&voice), Some(&lang), speed);
        drop(permit);
        out
    })
    .await
    {
        Ok(Ok(audio)) => Ok(audio.into_vec()),
        Ok(Err(e)) => Err(SynthError::Failed(format!("{e:#}"))),
        Err(e) => Err(SynthError::Failed(format!("synth task join: {e}"))),
    }
}

fn is_input_fault(msg: &str) -> bool {
    msg.contains("exceeds MAX_PHONEME_LENGTH")
        || msg.contains("phonemize produced empty")
        || msg.contains("tokenize empty")
}

fn is_unknown_voice_fault(msg: &str) -> bool {
    msg.contains("not found in voices.bin")
}

fn invalid_voice_response(voice: &str, mut names: Vec<String>) -> Response {
    names.sort();
    openai_error(
        StatusCode::BAD_REQUEST,
        format!(
            "voice {voice:?} not found; the {} valid voices are: {}",
            names.len(),
            names.join(", ")
        ),
        kind::INVALID_REQUEST,
        Some("voice"),
        Some("invalid_voice"),
    )
}

fn first_chunk_error_response(err: SynthError) -> Response {
    match err {
        SynthError::Busy => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "TTS is busy: synthesis queue wait exceeded; retry shortly",
            kind::SERVICE_UNAVAIL,
            None,
            Some("tts_overloaded"),
        ),
        SynthError::Failed(msg) => {
            if is_unknown_voice_fault(&msg) {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("synthesis rejected: {msg}"),
                    kind::INVALID_REQUEST,
                    Some("voice"),
                    Some("invalid_voice"),
                );
            }
            if is_input_fault(&msg) {
                openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("synthesis rejected: {msg}"),
                    kind::INVALID_REQUEST,
                    Some("input"),
                    Some("unsynthesizable_input"),
                )
            } else {
                openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("synthesis failed: {msg}"),
                    kind::SERVER,
                    None,
                    Some("synthesize_failed"),
                )
            }
        }
    }
}

pub async fn handle_create_speech(
    State(state): State<SpeechAppState>,
    Json(req): Json<CreateSpeechRequestBody>,
) -> Response {
    let Some(kokoro) = state.kokoro.clone() else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "TTS not configured: kokoro is not loaded (kokoro-v1.0.onnx + voices.bin missing \
             from the model directory, or they failed to load -- the boot log carries the \
             reason)",
            kind::SERVICE_UNAVAIL,
            Some("model"),
            Some("tts_not_configured"),
        );
    };

    let mut entries: Vec<serde_json::Value> = Vec::new();
    if req.model.as_deref().unwrap_or("").is_empty() {
        entries.push(missing(&["body", "model"]));
    }
    if req.input.is_none() {
        entries.push(missing(&["body", "input"]));
    }
    if req.voice.is_none() {
        entries.push(missing(&["body", "voice"]));
    }

    if let Some(sr) = req.sample_rate {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sr) {
            entries.push(json!({
                "type": "less_than_equal",
                "loc": ["body", "sample_rate"],
                "msg": format!("Input should be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}"),
                "input": sr,
            }));
        }
    }

    if !entries.is_empty() {
        return fastapi_validation_error(entries);
    }

    let speed = req.speed.unwrap_or(1.0);
    let response_format = req.response_format.unwrap_or_default();
    let stream_format = req.stream_format.unwrap_or_default();

    if !(SPEED_MIN..=SPEED_MAX).contains(&speed) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("speed must be between {SPEED_MIN:.1} and {SPEED_MAX:.1}, got {speed}"),
            kind::INVALID_REQUEST,
            Some("speed"),
            Some("out_of_range"),
        );
    }

    let mut voice = req.voice.unwrap();
    if !kokoro.has_voice(&voice) && is_openai_voice_alias(&voice) {
        warn!(requested = %voice, fallback = speech::DEFAULT_VOICE,
            "openai voice alias falling back to default");
        voice = speech::DEFAULT_VOICE.to_string();
    }
    if !kokoro.has_voice(&voice) {
        return invalid_voice_response(&voice, kokoro.voice_names());
    }

    let input = req.input.unwrap();
    if input.trim().is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "input must not be empty",
            kind::INVALID_REQUEST,
            Some("input"),
            Some("empty_input"),
        );
    }
    let n_chars = input.chars().count();
    let max_chars = max_input_chars();
    if n_chars > max_chars {
        return openai_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "input is {n_chars} characters; maximum is {max_chars} \
                 (set {} to raise)",
                defaults::env::KOKORO_MAX_INPUT_CHARS
            ),
            kind::INVALID_REQUEST,
            Some("input"),
            Some("input_too_long"),
        );
    }

    let cleaned = strip_markdown_emphasis(&strip_emojis(&input));
    let cleaned = normalize_for_tts(&cleaned);
    let plan = super::chunk::plan(&cleaned, chunk_chars());
    if plan.chunks.len() > 1 {
        let boundary_tails: Vec<String> = plan
            .boundaries
            .iter()
            .map(|&b| {
                if b == super::chunk::INTRA_SENTENCE_SPLIT {
                    "<intra-sentence>".to_string()
                } else {
                    let tail_start = cleaned[..b]
                        .char_indices()
                        .rev()
                        .nth(15)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    format!("{}@{}", &cleaned[tail_start..b], b)
                }
            })
            .collect();
        tracing::info!(
            n_chunks = plan.chunks.len(),
            oversize_splits = plan.oversize_splits,
            boundaries = ?boundary_tails,
            "punkt chunk plan"
        );
    }
    let chunks = plan.chunks;
    let Some((first_chunk, rest)) = chunks.split_first() else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "input contains no speakable text after normalization",
            kind::INVALID_REQUEST,
            Some("input"),
            Some("empty_input"),
        );
    };
    let rest: Vec<String> = rest.to_vec();

    if matches!(stream_format, StreamFormat::Audio)
        && !matches!(response_format, ResponseFormat::Pcm)
        && !ffmpeg_available()
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "response_format {:?} requires an ffmpeg binary on PATH, which is not available; \
                 use response_format=pcm",
                response_format
            ),
            kind::INVALID_REQUEST,
            Some("response_format"),
            Some("encoder_unavailable"),
        );
    }

    let deadline = Instant::now() + synth_budget(n_chars);
    let mut first_guard = FirstChunkCancelLog { armed: true };
    let first = match synth_chunk(
        kokoro.clone(),
        first_chunk.clone(),
        voice.clone(),
        state.language.clone(),
        speed,
    )
    .await
    {
        Ok(samples) => samples,
        Err(err) => {
            first_guard.armed = false;
            return first_chunk_error_response(err);
        }
    };
    first_guard.armed = false;

    match stream_format {
        StreamFormat::Sse => stream_sse(
            kokoro,
            first,
            rest,
            voice,
            state.language.clone(),
            speed,
            deadline,
        ),
        StreamFormat::Audio => stream_audio(
            kokoro,
            first,
            rest,
            voice,
            state.language.clone(),
            speed,
            response_format,
            req.sample_rate,
            deadline,
        )
        .await
        .unwrap_or_else(|err| {
            error!(?err, "speech audio streaming failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("encode: {err}")).into_response()
        }),
    }
}

fn io_err(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::other(msg.into())
}

struct FirstChunkCancelLog {
    armed: bool,
}

impl Drop for FirstChunkCancelLog {
    fn drop(&mut self) {
        if self.armed {
            tracing::info!("client disconnected; cancelling tts synthesis (first chunk in flight)");
        }
    }
}

fn stream_sse(
    kokoro: KokoroHandle,
    first: Vec<f32>,
    rest: Vec<String>,
    voice: String,
    lang: Arc<str>,
    speed: f32,
    deadline: Instant,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    tokio::spawn(async move {
        let send_event = |body: serde_json::Value| {
            let line = format!("data: {}\n\n", body);
            Bytes::from(line.into_bytes())
        };
        let delta = |samples: &[f32]| {
            json!({
                "type": "speech.audio.delta",
                "audio": B64.encode(f32_to_s16le(samples)),
            })
        };
        if tx.send(Ok(send_event(delta(&first)))).await.is_err() {
            return;
        }
        drop(first);
        for (chunk_idx, chunk) in rest.into_iter().enumerate() {
            if Instant::now() > deadline {
                let ev = json!({
                    "type": "error",
                    "error": {"message": "synthesis time budget exceeded", "type": kind::SERVER},
                });
                let _ = tx.send(Ok(send_event(ev))).await;
                return;
            }
            let synth = synth_chunk(kokoro.clone(), chunk, voice.clone(), lang.clone(), speed);
            let result = tokio::select! {
                _ = tx.closed() => {
                    tracing::info!(
                        chunks_done = chunk_idx + 1,
                        "client disconnected; cancelling tts synthesis (sse)"
                    );
                    return;
                }
                res = synth => res,
            };
            match result {
                Ok(samples) => {
                    let samples = join_silence_prefixed(samples);
                    if tx.send(Ok(send_event(delta(&samples)))).await.is_err() {
                        tracing::info!(
                            chunks_done = chunk_idx + 2,
                            "client disconnected; cancelling tts synthesis (sse)"
                        );
                        return;
                    }
                }
                Err(err) => {
                    warn!(?err, "kokoro synth failed mid-stream (sse)");
                    let ev = json!({
                        "type": "error",
                        "error": {"message": err.message(), "type": kind::SERVER},
                    });
                    let _ = tx.send(Ok(send_event(ev))).await;
                    return;
                }
            }
        }
        let done = json!({
            "type": "speech.audio.done",
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        });
        let _ = tx.send(Ok(send_event(done))).await;
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap_or_else(|e| {
            error!(?e, "sse response build failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}

#[allow(clippy::too_many_arguments)]
async fn stream_audio(
    kokoro: KokoroHandle,
    first: Vec<f32>,
    rest: Vec<String>,
    voice: String,
    lang: Arc<str>,
    speed: f32,
    format: ResponseFormat,
    target_sample_rate: Option<u32>,
    deadline: Instant,
) -> std::io::Result<Response> {
    let target_sr = target_sample_rate.unwrap_or(KOKORO_SAMPLE_RATE);

    if matches!(format, ResponseFormat::Pcm) {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
        tokio::spawn(async move {
            if tx
                .send(Ok(Bytes::from(f32_to_s16le(&first))))
                .await
                .is_err()
            {
                return;
            }
            drop(first);
            for (chunk_idx, chunk) in rest.into_iter().enumerate() {
                if Instant::now() > deadline {
                    let _ = tx.send(Err(io_err("synthesis time budget exceeded"))).await;
                    return;
                }
                let synth = synth_chunk(kokoro.clone(), chunk, voice.clone(), lang.clone(), speed);
                let result = tokio::select! {
                    _ = tx.closed() => {
                        tracing::info!(
                            chunks_done = chunk_idx + 1,
                            "client disconnected; cancelling tts synthesis (pcm)"
                        );
                        return;
                    }
                    res = synth => res,
                };
                match result {
                    Ok(samples) => {
                        let samples = join_silence_prefixed(samples);
                        if tx
                            .send(Ok(Bytes::from(f32_to_s16le(&samples))))
                            .await
                            .is_err()
                        {
                            tracing::info!(
                                chunks_done = chunk_idx + 2,
                                "client disconnected; cancelling tts synthesis (pcm)"
                            );
                            return;
                        }
                    }
                    Err(err) => {
                        warn!(?err, "kokoro synth failed mid-stream (pcm)");
                        let _ = tx.send(Err(io_err(err.message()))).await;
                        return;
                    }
                }
            }
        });
        let stream = ReceiverStream::new(rx);
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, format.mime_type())
            .body(Body::from_stream(stream))
            .map_err(|e| io_err(format!("response build: {e}")));
    }

    let mut child = spawn_ffmpeg(format, KOKORO_SAMPLE_RATE, target_sr)
        .map_err(|e| io_err(format!("spawn ffmpeg: {e}")))?;
    tracing::debug!(?format, target_sr, "ffmpeg spawned");
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| io_err("ffmpeg stdin missing"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| io_err("ffmpeg stdout missing"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| io_err("ffmpeg stderr missing"))?;

    tokio::spawn(async move {
        let mut buf = vec![0u8; 4096];
        let mut collected = Vec::new();
        loop {
            match stderr.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    collected.extend_from_slice(&buf[..n]);
                    if collected.len() > 4096 {
                        let line = String::from_utf8_lossy(&collected).into_owned();
                        warn!(target: "speaches_plus::tts", ffmpeg_stderr = %line, "ffmpeg stderr");
                        collected.clear();
                    }
                }
                Err(_) => break,
            }
        }
        if !collected.is_empty() {
            let line = String::from_utf8_lossy(&collected).into_owned();
            warn!(target: "speaches_plus::tts", ffmpeg_stderr = %line, "ffmpeg stderr (final)");
        }
    });

    let (abort_tx, abort_rx) = oneshot::channel::<String>();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cancel_writer = cancel.clone();
    tokio::spawn(async move {
        let mut chunk_idx = 0usize;
        let mut total_bytes = 0usize;
        let pcm = f32_to_s16le(&first);
        drop(first);
        if let Err(e) = stdin.write_all(&pcm).await {
            warn!(?e, chunk_idx, "ffmpeg stdin write failed");
            let _ = stdin.shutdown().await;
            return;
        }
        total_bytes += pcm.len();
        drop(pcm);
        for chunk in rest {
            chunk_idx += 1;
            if cancel_writer.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    chunks_done = chunk_idx,
                    "client disconnected; cancelling tts synthesis (ffmpeg)"
                );
                let _ = stdin.shutdown().await;
                return;
            }
            if Instant::now() > deadline {
                let _ = abort_tx.send("synthesis time budget exceeded".to_string());
                let _ = stdin.shutdown().await;
                return;
            }
            let samples = match synth_chunk(
                kokoro.clone(),
                chunk,
                voice.clone(),
                lang.clone(),
                speed,
            )
            .await
            {
                Ok(s) => join_silence_prefixed(s),
                Err(err) => {
                    warn!(?err, chunk_idx, "kokoro synth failed mid-stream (ffmpeg)");
                    let _ = abort_tx.send(err.message());
                    let _ = stdin.shutdown().await;
                    return;
                }
            };
            let pcm = f32_to_s16le(&samples);
            tracing::debug!(chunk_idx, pcm_bytes = pcm.len(), "writing to ffmpeg stdin");
            if let Err(e) = stdin.write_all(&pcm).await {
                warn!(?e, chunk_idx, "ffmpeg stdin write failed");
                break;
            }
            total_bytes += pcm.len();
        }
        tracing::debug!(total_bytes, "closing ffmpeg stdin");
        let _ = stdin.shutdown().await;
    });

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    tokio::spawn(async move {
        let mut buf = vec![0u8; 4096];
        let mut total_out = 0usize;
        let mut abort_rx = Some(abort_rx);
        loop {
            let abort_wait = async {
                match abort_rx.as_mut() {
                    Some(rx) => rx.await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                aborted = abort_wait => {
                    match aborted {
                        Ok(msg) => {
                            warn!(%msg, total_out, "aborting ffmpeg stream after synth failure");
                            let _ = tx.send(Err(io_err(msg))).await;
                            let _ = child.start_kill();
                            break;
                        }
                        Err(_) => {
                            abort_rx = None;
                        }
                    }
                }
                read = stdout.read(&mut buf) => {
                    match read {
                        Ok(0) => {
                            tracing::debug!(total_out, "ffmpeg stdout EOF");
                            break;
                        }
                        Ok(n) => {
                            total_out += n;
                            if tx
                                .send(Ok(Bytes::copy_from_slice(&buf[..n])))
                                .await
                                .is_err()
                            {
                                tracing::info!(total_out, "client disconnected; stopping ffmpeg stream");
                                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                                let _ = child.start_kill();
                                break;
                            }
                        }
                        Err(e) => {
                            warn!(?e, total_out, "ffmpeg stdout read failed");
                            let _ = tx.send(Err(e)).await;
                            break;
                        }
                    }
                }
            }
        }
        match child.wait().await {
            Ok(status) => tracing::debug!(?status, total_out, "ffmpeg exited"),
            Err(e) => warn!(?e, "ffmpeg wait failed"),
        }
    });

    let stream = ReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.mime_type())
        .body(Body::from_stream(stream))
        .map_err(|e| io_err(format!("response build: {e}")))
}

fn spawn_ffmpeg(format: ResponseFormat, source_sr: u32, target_sr: u32) -> std::io::Result<Child> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-f")
        .arg("s16le")
        .arg("-ar")
        .arg(source_sr.to_string())
        .arg("-ac")
        .arg("1")
        .arg("-i")
        .arg("pipe:0")
        .arg("-ar")
        .arg(target_sr.to_string());
    match format {
        ResponseFormat::Mp3 => {
            cmd.arg("-f").arg("mp3").arg("-codec:a").arg("libmp3lame");
        }
        ResponseFormat::Wav => {
            cmd.arg("-f").arg("wav");
        }
        ResponseFormat::Flac => {
            cmd.arg("-f").arg("flac");
        }
        ResponseFormat::Opus => {
            cmd.arg("-f").arg("opus").arg("-codec:a").arg("libopus");
        }
        ResponseFormat::Aac => {
            cmd.arg("-f").arg("adts").arg("-codec:a").arg("aac");
        }
        ResponseFormat::Pcm => unreachable!("pcm should bypass ffmpeg"),
    }
    cmd.arg("pipe:1")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(input: &str) -> CreateSpeechRequestBody {
        CreateSpeechRequestBody {
            model: Some("speaches-ai/Kokoro-82M-v1.0-ONNX".to_string()),
            input: Some(input.to_string()),
            voice: Some("af_heart".to_string()),
            response_format: None,
            speed: None,
            stream_format: None,
            sample_rate: None,
        }
    }

    #[tokio::test]
    async fn speech_route_is_unavailable_not_panicking_when_kokoro_disabled() {
        let state = SpeechAppState::new(None);
        let res = handle_create_speech(State(state), Json(req("hello"))).await;
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    async fn error_body(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("error body under 64k");
        serde_json::from_slice(&bytes).expect("error body is json")
    }

    #[tokio::test]
    async fn unknown_voice_is_a_400_naming_the_param_and_listing_voices() {
        let resp = invalid_voice_response("default", vec!["am_adam".into(), "af_heart".into()]);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = error_body(resp).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "voice");
        assert_eq!(body["error"]["code"], "invalid_voice");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("\"default\""), "{msg}");
        assert!(msg.contains("the 2 valid voices are: af_heart, am_adam"), "{msg}");
    }

    #[tokio::test]
    async fn synth_layer_unknown_voice_maps_to_400_not_500() {
        let resp = first_chunk_error_response(SynthError::Failed(
            "voice \"default\" not found in voices.bin (54 voices loaded)".to_string(),
        ));
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = error_body(resp).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "voice");
        assert_eq!(body["error"]["code"], "invalid_voice");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("54 voices loaded"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn non_voice_synth_failures_still_map_to_500() {
        let resp =
            first_chunk_error_response(SynthError::Failed("onnx session exploded".to_string()));
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn disabled_kokoro_short_circuits_before_request_validation() {
        let state = SpeechAppState::new(None);
        let mut body = req("hello");
        body.model = None;
        body.voice = None;
        let res = handle_create_speech(State(state), Json(body)).await;
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
