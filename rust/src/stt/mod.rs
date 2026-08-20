use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use ct2rs::sys::{
    Config as Ct2Config, StorageView as Ct2StorageView, Whisper as Ct2Whisper,
    WhisperOptions as Ct2WhisperOptions,
};
use ct2rs::tokenizers::hf::Tokenizer as Ct2Tokenizer;
use ct2rs::Tokenizer as Ct2TokenizerTrait;
#[cfg(feature = "cuda")]
use ct2rs::{ComputeType as Ct2ComputeType, Device as Ct2Device};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext as CppContext, WhisperContextParameters,
};

pub mod mel;
pub mod parakeet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Ct2,
    WhisperCpp,
    Parakeet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WhisperTask {
    #[default]
    Transcribe,
    Translate,
}

impl WhisperTask {
    pub fn as_str(self) -> &'static str {
        match self {
            WhisperTask::Transcribe => "transcribe",
            WhisperTask::Translate => "translate",
        }
    }
}

pub const ALLOW_UNSUPPORTED_TRANSLATE_ENV: &str = "STT_ALLOW_UNSUPPORTED_TRANSLATE";

#[derive(Debug, Clone)]
pub struct TranslateUnsupported {
    pub model_id: String,
}

impl std::fmt::Display for TranslateUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "speech model {:?} is a transcribe-only checkpoint and cannot perform \
             translation; load a translation-capable model or set {}=1 to override",
            self.model_id, ALLOW_UNSUPPORTED_TRANSLATE_ENV
        )
    }
}

impl std::error::Error for TranslateUnsupported {}

pub fn is_translate_unsupported(err: &anyhow::Error) -> bool {
    err.downcast_ref::<TranslateUnsupported>().is_some()
}

fn checkpoint_is_english_only(lower_id: &str) -> bool {
    let bytes = lower_id.as_bytes();
    let mut from = 0;
    while let Some(pos) = lower_id[from..].find(".en") {
        let at = from + pos;
        let after = at + 3;
        if bytes.get(after).is_none_or(|c| !c.is_ascii_alphanumeric()) {
            return true;
        }
        from = after;
    }
    false
}

fn checkpoint_is_transcribe_only(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    lower.contains("turbo") || checkpoint_is_english_only(&lower)
}

fn checkpoint_tail(path: &Path) -> String {
    let comps: Vec<String> = path
        .components()
        .rev()
        .take(3)
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    comps.join("/")
}

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    }
}

impl Backend {
    pub fn from_env() -> Self {
        match std::env::var(super::defaults::env::STT_BACKEND)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "ct2" | "ctranslate2" | "faster-whisper" => Backend::Ct2,
            "parakeet" | "parakeet-tdt" | "tdt" => Backend::Parakeet,
            "" | "whisper-cpp" | "whisper_cpp" | "whispercpp" | "cpp" | "ggml" => {
                Backend::WhisperCpp
            }
            other => {
                tracing::warn!(
                    value = other,
                    var = super::defaults::env::STT_BACKEND,
                    "unrecognized STT backend; falling back to whisper.cpp"
                );
                Backend::WhisperCpp
            }
        }
    }
}

pub struct WhisperEngine {
    handle: WhisperHandle,
}

pub struct Ct2State {
    pub model: Ct2Whisper,
    pub tokenizer: Ct2Tokenizer,
    pub mel: mel::WhisperMel,

    pub timestamp_begin_id: Option<u32>,
    pub supports_translate: bool,
}

impl WhisperEngine {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let backend = Backend::from_env();
        tracing::info!(?backend, "STT backend selected");
        let handle = match backend {
            Backend::Ct2 => {
                let dir = ct2_model_dir(model_dir)?;
                let id = dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ct2-model")
                    .to_string();
                let model = Ct2Whisper::new(&dir, ct2_config())
                    .with_context(|| format!("load whisper-ct2 model: {}", dir.display()))?;
                let tokenizer = Ct2Tokenizer::new(&dir)
                    .with_context(|| format!("load ct2 tokenizer.json from {}", dir.display()))?;
                let n_mels = model.n_mels();

                let timestamp_begin_id = tokenizer.token_to_id("<|0.00|>");
                if timestamp_begin_id.is_none() {
                    tracing::warn!(
                        "ct2 tokenizer does not expose `<|0.00|>` as an added token; falling back to string-based timestamp parsing"
                    );
                }
                let resolved = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
                let supports_translate = tokenizer.token_to_id("<|translate|>").is_some()
                    && !checkpoint_is_transcribe_only(&checkpoint_tail(&resolved))
                    && !checkpoint_is_transcribe_only(&id);
                if !supports_translate {
                    tracing::warn!(
                        model = %resolved.display(),
                        "ct2 checkpoint appears transcribe-only; /v1/audio/translations will be refused"
                    );
                }
                let state = Ct2State {
                    model,
                    tokenizer,
                    mel: mel::WhisperMel::new(n_mels),
                    timestamp_begin_id,
                    supports_translate,
                };
                WhisperHandle::Ct2 {
                    inner: Arc::new(state),
                    id,
                }
            }
            Backend::Parakeet => {
                let dir = parakeet::parakeet_dir()?;
                let id = dir
                    .parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .map(|s| s.trim_start_matches("models--").replace("--", "/"))
                    .unwrap_or_else(|| "parakeet-tdt-0.6b-v2".to_string());
                let model = parakeet::ParakeetTdt::load(&dir)
                    .with_context(|| format!("load parakeet-tdt from {}", dir.display()))?;
                tracing::info!(dir = %dir.display(), id, "parakeet-tdt backend loaded");
                WhisperHandle::Parakeet {
                    inner: Arc::new(model),
                    id,
                }
            }
            Backend::WhisperCpp => {
                let path = whisper_cpp_model_path(model_dir)?;
                let path_str = path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-UTF-8 model path: {:?}", path))?;
                let id = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ggml")
                    .to_string();
                let ctx =
                    CppContext::new_with_params(path_str, WhisperContextParameters::default())
                        .with_context(|| format!("load whisper.cpp model: {path_str}"))?;
                WhisperHandle::WhisperCpp {
                    inner: Arc::new(ctx),
                    id,
                }
            }
        };
        Ok(Self { handle })
    }

    pub fn handle(&self) -> WhisperHandle {
        self.handle.clone()
    }
}

#[derive(Clone)]
pub enum WhisperHandle {
    Ct2 { inner: Arc<Ct2State>, id: String },
    WhisperCpp { inner: Arc<CppContext>, id: String },
    Parakeet { inner: Arc<parakeet::ParakeetTdt>, id: String },
}

impl WhisperHandle {
    pub fn model_id(&self) -> &str {
        match self {
            WhisperHandle::Ct2 { id, .. } => id,
            WhisperHandle::WhisperCpp { id, .. } => id,
            WhisperHandle::Parakeet { id, .. } => id,
        }
    }

    pub fn supports_translate(&self) -> bool {
        if env_flag_enabled(ALLOW_UNSUPPORTED_TRANSLATE_ENV) {
            return true;
        }
        match self {
            WhisperHandle::Ct2 { inner, .. } => inner.supports_translate,
            WhisperHandle::WhisperCpp { id, .. } => !checkpoint_is_transcribe_only(id),
            WhisperHandle::Parakeet { .. } => false,
        }
    }
}

const SILENCE_PEAK_THRESHOLD: f32 = super::defaults::stt::SILENCE_PEAK_THRESHOLD;

pub mod noise_gate;

#[derive(Debug, Clone, Default)]
pub struct TranscriptionResult {
    pub text: String,
    pub avg_logprob: Option<f32>,
    pub no_speech_prob: Option<f32>,
    pub compression_ratio: Option<f32>,

    pub segments: Vec<TimedSegment>,
    pub language: Option<String>,
    pub duration_s: Option<f32>,
    pub task: WhisperTask,
}

#[derive(Debug, Clone, Default)]
pub struct TimedSegment {
    pub t_start_ms: u32,
    pub t_end_ms: u32,
    pub text: String,
    pub avg_logprob: Option<f32>,
    pub no_speech_prob: Option<f32>,
    pub words: Vec<TimedWord>,
}

#[derive(Debug, Clone, Default)]
pub struct TimedWord {
    pub word: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

pub fn proportional_word_timings(text: &str, seg_start_ms: u32, seg_end_ms: u32) -> Vec<TimedWord> {
    let trimmed = text.trim();
    if trimmed.is_empty() || seg_end_ms <= seg_start_ms {
        return Vec::new();
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();
    if total_chars == 0 {
        return Vec::new();
    }
    let span_ms = (seg_end_ms - seg_start_ms) as f64;
    let mut out = Vec::with_capacity(words.len());
    let mut acc_chars: usize = 0;
    for w in words {
        let w_chars = w.chars().count();
        let start_ms = seg_start_ms as f64 + span_ms * (acc_chars as f64) / (total_chars as f64);
        acc_chars += w_chars;
        let end_ms = seg_start_ms as f64 + span_ms * (acc_chars as f64) / (total_chars as f64);
        out.push(TimedWord {
            word: w.to_string(),
            start_ms: start_ms.round() as u32,
            end_ms: end_ms.round() as u32,
        });
    }
    out
}

impl TranscriptionResult {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn from_text<S: Into<String>>(s: S) -> Self {
        Self {
            text: s.into(),
            ..Self::default()
        }
    }
}

impl WhisperHandle {
    pub fn transcribe(&self, audio_16k_mono: &[f32]) -> Result<String> {
        Ok(self.transcribe_full(audio_16k_mono)?.text)
    }

    pub fn transcribe_full(&self, audio_16k_mono: &[f32]) -> Result<TranscriptionResult> {
        self.transcribe_full_with_task(audio_16k_mono, WhisperTask::Transcribe)
    }

    pub fn translate_full(&self, audio_16k_mono: &[f32]) -> Result<TranscriptionResult> {
        self.transcribe_full_with_task(audio_16k_mono, WhisperTask::Translate)
    }

    pub fn transcribe_full_with_task(
        &self,
        audio_16k_mono: &[f32],
        task: WhisperTask,
    ) -> Result<TranscriptionResult> {
        if task == WhisperTask::Translate && !self.supports_translate() {
            return Err(anyhow::Error::new(TranslateUnsupported {
                model_id: self.model_id().to_string(),
            }));
        }
        if peak_amplitude(audio_16k_mono) < SILENCE_PEAK_THRESHOLD {
            tracing::debug!(
                samples = audio_16k_mono.len(),
                "silence pre-gate fired; skipping STT"
            );
            return Ok(TranscriptionResult {
                duration_s: Some(audio_16k_mono.len() as f32 / 16_000.0),
                task,
                ..TranscriptionResult::empty()
            });
        }
        match self {
            WhisperHandle::Ct2 { inner, .. } => transcribe_ct2_long(inner, audio_16k_mono, task),
            WhisperHandle::WhisperCpp { inner, .. } => {
                transcribe_whisper_cpp(inner, audio_16k_mono, task)
            }
            WhisperHandle::Parakeet { inner, .. } => {
                transcribe_parakeet(inner, audio_16k_mono, task)
            }
        }
    }
}

fn transcribe_parakeet(
    model: &parakeet::ParakeetTdt,
    audio_16k_mono: &[f32],
    task: WhisperTask,
) -> Result<TranscriptionResult> {
    let text = model.transcribe(audio_16k_mono)?;
    let duration_ms = (audio_16k_mono.len() as f64 / 16.0).round() as u32;
    let segments = if text.is_empty() {
        Vec::new()
    } else {
        vec![TimedSegment {
            t_start_ms: 0,
            t_end_ms: duration_ms,
            text: text.clone(),
            avg_logprob: None,
            no_speech_prob: None,
            words: proportional_word_timings(&text, 0, duration_ms),
        }]
    };
    Ok(TranscriptionResult {
        text,
        avg_logprob: None,
        no_speech_prob: None,
        compression_ratio: None,
        segments,
        language: Some("en".to_string()),
        duration_s: Some(audio_16k_mono.len() as f32 / 16_000.0),
        task,
    })
}

fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()))
}

const CHUNK_SAMPLES: usize = 30 * 16_000;

fn stt_beam_size() -> usize {
    std::env::var(super::defaults::stt::BEAM_SIZE_ENV_RESTORES_THE_GREEDY_DECODE)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(super::defaults::stt::BEAM_SIZE)
}

fn transcribe_ct2_long(
    state: &Ct2State,
    audio: &[f32],
    task: WhisperTask,
) -> Result<TranscriptionResult> {
    if audio.len() <= CHUNK_SAMPLES {
        return transcribe_ct2(state, audio, task);
    }
    let mut all_segments: Vec<TimedSegment> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut lp_sum = 0.0_f64;
    let mut lp_weight = 0.0_f64;
    let mut nsp_sum = 0.0_f64;
    let mut nsp_weight = 0.0_f64;
    let mut language: Option<String> = None;

    let mut pos = 0usize;
    while pos < audio.len() {
        let end = (pos + CHUNK_SAMPLES).min(audio.len());
        let chunk = &audio[pos..end];
        let offset_ms = (pos as u64 * 1000 / 16_000) as u32;

        if peak_amplitude(chunk) < SILENCE_PEAK_THRESHOLD {
            pos = end;
            continue;
        }

        let res = transcribe_ct2(state, chunk, task)?;
        let chunk_dur_ms = (chunk.len() as f64 * 1000.0) / 16_000.0;

        let trimmed = res.text.trim().to_string();
        if trimmed.is_empty() && res.segments.is_empty() {
            pos = end;
            continue;
        }
        if !trimmed.is_empty() {
            texts.push(trimmed);
        }
        if language.is_none() {
            language = res.language.clone();
        }
        if let Some(lp) = res.avg_logprob {
            if lp.is_finite() {
                lp_sum += lp as f64 * chunk_dur_ms;
                lp_weight += chunk_dur_ms;
            }
        }
        if let Some(nsp) = res.no_speech_prob {
            if nsp.is_finite() {
                nsp_sum += nsp as f64 * chunk_dur_ms;
                nsp_weight += chunk_dur_ms;
            }
        }
        for mut seg in res.segments {
            seg.t_start_ms += offset_ms;
            seg.t_end_ms += offset_ms;
            for w in seg.words.iter_mut() {
                w.start_ms += offset_ms;
                w.end_ms += offset_ms;
            }
            all_segments.push(seg);
        }
        pos = end;
    }

    Ok(TranscriptionResult {
        text: join_segments(texts.iter().map(|s| s.as_str())),
        avg_logprob: if lp_weight > 0.0 {
            Some((lp_sum / lp_weight) as f32)
        } else {
            None
        },
        no_speech_prob: if nsp_weight > 0.0 {
            Some((nsp_sum / nsp_weight) as f32)
        } else {
            None
        },
        compression_ratio: None,
        segments: all_segments,
        language,
        duration_s: Some(audio.len() as f32 / 16_000.0),
        task,
    })
}

fn transcribe_ct2(
    state: &Ct2State,
    audio: &[f32],
    task: WhisperTask,
) -> Result<TranscriptionResult> {
    let padded = mel::pad_or_truncate_to_30s(audio);
    let mut mel_data = state.mel.log_mel(&padded);
    let n_mels = state.mel.n_mels;

    let mel_view = Ct2StorageView::new(
        &[1usize, n_mels, mel::N_FRAMES],
        mel_data.as_mut_slice(),
        ct2rs::Device::CPU,
    )
    .context("ct2 storage view for mel features")?;
    let encoded = state
        .model
        .encode(&mel_view, false)
        .context("ct2 whisper encode")?;
    drop(mel_view);

    let task_token = match task {
        WhisperTask::Transcribe => "<|transcribe|>",
        WhisperTask::Translate => "<|translate|>",
    };
    let detected = state
        .model
        .detect_language(&encoded)
        .context("ct2 whisper detect_language")?;
    let language_token = detected
        .into_iter()
        .next()
        .and_then(|batch| batch.into_iter().next())
        .map(|d| {
            tracing::debug!(language = %d.language, probability = d.probability, "ct2 language detection");
            d.language
        })
        .unwrap_or_else(|| "<|en|>".to_string());
    let prompts: Vec<Vec<&str>> = vec![vec![
        "<|startoftranscript|>",
        language_token.as_str(),
        task_token,
    ]];
    let mut options = Ct2WhisperOptions {
        beam_size: stt_beam_size(),
        ..Default::default()
    };
    options.return_scores = true;
    options.return_no_speech_prob = true;

    let mut results = state
        .model
        .generate(&encoded, &prompts, &options)
        .context("ct2 whisper generate")?;
    let r = results
        .pop()
        .ok_or_else(|| anyhow!("ct2 whisper returned no generation results"))?;
    let no_speech_prob = Some(r.no_speech_prob);
    let avg_logprob = r.scores.first().copied();
    let tokens = r.sequences.into_iter().next().unwrap_or_default();
    let token_ids = r.sequences_ids.into_iter().next().unwrap_or_default();

    let real_audio_ms = (audio.len() as u64 * 1000 / 16_000).min(u32::MAX as u64) as u32;
    let segments = split_ct2_segments(&tokens, &token_ids, state.timestamp_begin_id, real_audio_ms);
    let text_segments: Vec<String> = segments
        .iter()
        .map(|seg| {
            if seg.text_tokens.is_empty() {
                Ok::<String, anyhow::Error>(String::new())
            } else {
                let decoded = state
                    .tokenizer
                    .decode(seg.text_tokens.clone())
                    .context("ct2 tokenizer decode")?;
                Ok(strip_special_tokens(&decoded).trim().to_string())
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let timed: Vec<TimedSegment> = segments
        .iter()
        .zip(text_segments.iter())
        .filter_map(|(seg, text)| {
            if text.is_empty() {
                return None;
            }
            let words = proportional_word_timings(text, seg.t_start_ms, seg.t_end_ms);
            Some(TimedSegment {
                t_start_ms: seg.t_start_ms,
                t_end_ms: seg.t_end_ms,
                text: text.clone(),

                avg_logprob: None,
                no_speech_prob: None,
                words,
            })
        })
        .collect();
    let (text, timed) = if timed.is_empty() {
        let repaired = if tokens.is_empty() {
            String::new()
        } else {
            let decoded = state
                .tokenizer
                .decode(tokens)
                .context("ct2 tokenizer decode")?;
            strip_special_tokens(&decoded).trim().to_string()
        };
        let segs = whole_clip_segment(&repaired, real_audio_ms, avg_logprob, no_speech_prob);
        (repaired, segs)
    } else {
        (
            join_segments(text_segments.iter().map(|s| s.as_str())),
            timed,
        )
    };
    Ok(TranscriptionResult {
        text,
        avg_logprob,
        no_speech_prob,
        compression_ratio: None,
        segments: timed,
        language: language_code(&language_token),
        duration_s: Some(real_audio_ms as f32 / 1000.0),
        task,
    })
}

fn whole_clip_segment(
    text: &str,
    audio_ms: u32,
    avg_logprob: Option<f32>,
    no_speech_prob: Option<f32>,
) -> Vec<TimedSegment> {
    let trimmed = text.trim();
    if trimmed.is_empty() || audio_ms == 0 {
        return Vec::new();
    }
    vec![TimedSegment {
        t_start_ms: 0,
        t_end_ms: audio_ms,
        text: trimmed.to_string(),
        avg_logprob,
        no_speech_prob,
        words: proportional_word_timings(trimmed, 0, audio_ms),
    }]
}

fn language_code(tok: &str) -> Option<String> {
    let inner = tok.strip_prefix("<|").unwrap_or(tok);
    let inner = inner.strip_suffix("|>").unwrap_or(inner).trim();
    if inner.is_empty() {
        None
    } else {
        Some(inner.to_ascii_lowercase())
    }
}

#[derive(Debug)]
struct Ct2Segment {
    t_start_ms: u32,
    t_end_ms: u32,
    text_tokens: Vec<String>,
}

fn split_ct2_segments(
    tokens: &[String],
    token_ids: &[usize],
    ts_begin_id: Option<u32>,
    audio_ms: u32,
) -> Vec<Ct2Segment> {
    let mut segments = Vec::new();
    let mut current_start: Option<u32> = None;
    let mut current_tokens: Vec<String> = Vec::new();

    for (i, tok) in tokens.iter().enumerate() {
        let ts = classify_timestamp(tok, token_ids.get(i).copied(), ts_begin_id);
        match ts {
            Some(ts_ms) => {
                let ts_ms = ts_ms.min(audio_ms);
                match current_start {
                    None => current_start = Some(ts_ms),
                    Some(start) => {
                        let valid_segment = ts_ms > start && !current_tokens.is_empty();
                        if valid_segment {
                            segments.push(Ct2Segment {
                                t_start_ms: start.min(audio_ms),
                                t_end_ms: ts_ms,
                                text_tokens: std::mem::take(&mut current_tokens),
                            });
                        } else if !current_tokens.is_empty() {
                            current_tokens.clear();
                        }
                        current_start = Some(ts_ms);
                    }
                }
            }
            None => {
                if current_start.is_some() {
                    current_tokens.push(tok.clone());
                }
            }
        }
    }
    segments
}

fn classify_timestamp(tok: &str, id: Option<usize>, ts_begin_id: Option<u32>) -> Option<u32> {
    if let (Some(id), Some(begin)) = (id, ts_begin_id) {
        let id = id as u64;
        let begin = begin as u64;

        if id >= begin && id < begin + 1501 {
            let step_ms = (id - begin) * 20;
            return Some(step_ms.min(u32::MAX as u64) as u32);
        }

        return None;
    }
    parse_timestamp_token(tok)
}

fn parse_timestamp_token(tok: &str) -> Option<u32> {
    let inner = tok.strip_prefix("<|")?.strip_suffix("|>")?;
    let dot = inner.find('.')?;
    let (whole, frac_with_dot) = inner.split_at(dot);
    let frac = &frac_with_dot[1..];
    if whole.is_empty() || frac.is_empty() {
        return None;
    }
    if !whole.chars().all(|c| c.is_ascii_digit()) || !frac.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let secs: u32 = whole.parse().ok()?;
    let frac_val: u32 = frac.parse().ok()?;
    let frac_ms = match frac.len() {
        1 => frac_val * 100,
        2 => frac_val * 10,
        3 => frac_val,
        n if n > 3 => frac_val / 10u32.pow((n - 3) as u32),
        _ => 0,
    };
    Some(secs.saturating_mul(1000).saturating_add(frac_ms))
}

fn transcribe_whisper_cpp(
    ctx: &CppContext,
    audio: &[f32],
    task: WhisperTask,
) -> Result<TranscriptionResult> {
    let mut state = ctx.create_state().context("create whisper state")?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    match task {
        WhisperTask::Transcribe => {
            params.set_language(Some("auto"));
            params.set_translate(false);
        }
        WhisperTask::Translate => {
            params.set_language(Some("auto"));
            params.set_translate(true);
        }
    }
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);
    params.set_suppress_blank(true);
    params.set_no_context(true);
    params.set_token_timestamps(true);
    params.set_n_threads(num_cpus());

    state.full(params, audio).context("whisper full")?;

    let real_audio_ms = (audio.len() as u64 * 1000 / 16_000).min(u32::MAX as u64) as u32;
    let language =
        whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(|s| s.to_ascii_lowercase());
    let n = state.full_n_segments();
    let mut timed: Vec<TimedSegment> = Vec::with_capacity(n as usize);
    let mut nsp_sum = 0.0_f64;
    let mut nsp_count: usize = 0;
    let mut log_sum = 0.0_f64;
    let mut tok_count: usize = 0;
    for i in 0..n {
        let seg = state
            .get_segment(i)
            .ok_or_else(|| anyhow!("segment {i} oob"))?;
        let seg_text = seg.to_str_lossy().context("segment text")?.into_owned();
        let seg_nsp = seg.no_speech_probability() as f64;
        nsp_sum += seg_nsp;
        nsp_count += 1;

        let n_tokens = seg.n_tokens();
        let mut seg_log_sum = 0.0_f64;
        let mut seg_tok_count: usize = 0;
        let mut tok_texts: Vec<(String, i64, i64)> = Vec::with_capacity(n_tokens as usize);
        for t in 0..n_tokens {
            if let Some(tok) = seg.get_token(t) {
                let text = tok
                    .to_str_lossy()
                    .map(|c| c.into_owned())
                    .unwrap_or_default();
                if is_whisper_pseudo_token(&text) {
                    continue;
                }
                let p = tok.token_probability().clamp(f32::MIN_POSITIVE, 1.0);
                let lp = (p as f64).ln();
                log_sum += lp;
                tok_count += 1;
                seg_log_sum += lp;
                seg_tok_count += 1;
                let td = tok.token_data();
                tok_texts.push((text, td.t0, td.t1));
            }
        }

        let t0 = clamp_centisecond_ts(seg.start_timestamp(), real_audio_ms);
        let t1 = clamp_centisecond_ts(seg.end_timestamp(), real_audio_ms).max(t0);
        let is_first = timed.is_empty();
        let trimmed = seg_text.trim();
        let trimmed = if is_first {
            strip_leading_colon(trimmed)
        } else {
            trimmed
        };
        if !trimmed.is_empty() {
            let mut words = group_whisper_tokens_into_words(&tok_texts, t0, t1, trimmed);
            if is_first {
                while words.first().is_some_and(|w| w.word.trim() == ":") {
                    words.remove(0);
                }
            }
            for w in words.iter_mut() {
                w.start_ms = w.start_ms.min(real_audio_ms);
                w.end_ms = w.end_ms.min(real_audio_ms).max(w.start_ms);
            }
            timed.push(TimedSegment {
                t_start_ms: t0,
                t_end_ms: t1,
                text: trimmed.to_string(),
                avg_logprob: if seg_tok_count > 0 {
                    Some((seg_log_sum / seg_tok_count as f64) as f32)
                } else {
                    None
                },
                no_speech_prob: Some(seg_nsp as f32),
                words,
            });
        }
    }
    let no_speech_prob = if nsp_count > 0 {
        Some((nsp_sum / nsp_count as f64) as f32)
    } else {
        None
    };
    let avg_logprob = if tok_count > 0 {
        Some((log_sum / tok_count as f64) as f32)
    } else {
        None
    };
    Ok(TranscriptionResult {
        text: join_segments(timed.iter().map(|s| s.text.as_str())),
        avg_logprob,
        no_speech_prob,
        compression_ratio: None,
        segments: timed,
        language,
        duration_s: Some(real_audio_ms as f32 / 1000.0),
        task,
    })
}

fn is_whisper_pseudo_token(text: &str) -> bool {
    let t = text.trim();
    (t.starts_with('<') && t.ends_with('>') && t.len() >= 2)
        || (t.starts_with('[') && t.ends_with(']') && t.len() >= 2)
}

fn clamp_centisecond_ts(ts: i64, audio_ms: u32) -> u32 {
    let ms = (ts.max(0) as u64).saturating_mul(10).min(u32::MAX as u64) as u32;
    ms.min(audio_ms)
}

fn strip_leading_colon(s: &str) -> &str {
    match s.strip_prefix(':') {
        Some(rest) => rest.trim_start(),
        None => s,
    }
}

fn group_whisper_tokens_into_words(
    tokens: &[(String, i64, i64)],
    seg_start_ms: u32,
    seg_end_ms: u32,
    fallback_text: &str,
) -> Vec<TimedWord> {
    if tokens.is_empty() {
        return proportional_word_timings(fallback_text, seg_start_ms, seg_end_ms);
    }
    let mut words: Vec<TimedWord> = Vec::new();
    let mut cur_text = String::new();
    let mut cur_t0: Option<i64> = None;
    let mut cur_t1: Option<i64> = None;
    for (text, t0, t1) in tokens {
        let starts_word = text.starts_with(' ');
        if starts_word && !cur_text.trim().is_empty() {
            let s_ms = (cur_t0.unwrap_or(0).max(0) * 10) as u32;
            let e_ms = (cur_t1.unwrap_or(0).max(0) * 10) as u32;
            let w = cur_text.trim().to_string();
            if !w.is_empty() {
                words.push(TimedWord {
                    word: w,
                    start_ms: s_ms,
                    end_ms: e_ms,
                });
            }
            cur_text.clear();
            cur_t0 = None;
            let _ = cur_t1.take();
        }
        if cur_t0.is_none() {
            cur_t0 = Some(*t0);
        }
        cur_t1 = Some(*t1);
        cur_text.push_str(text);
    }
    let trimmed = cur_text.trim().to_string();
    if !trimmed.is_empty() {
        let s_ms = (cur_t0.unwrap_or(0).max(0) * 10) as u32;
        let e_ms = (cur_t1.unwrap_or(0).max(0) * 10) as u32;
        words.push(TimedWord {
            word: trimmed,
            start_ms: s_ms,
            end_ms: e_ms,
        });
    }
    if words.is_empty() || words.iter().all(|w| w.end_ms == 0 && w.start_ms == 0) {
        return proportional_word_timings(fallback_text, seg_start_ms, seg_end_ms);
    }
    words
}

fn join_segments<'a>(segments: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for seg in segments {
        let trimmed = seg.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(trimmed);
    }
    out
}

fn ct2_config() -> Ct2Config {
    #[cfg(feature = "cuda")]
    {
        let mut cfg = Ct2Config::default();
        cfg.device = Ct2Device::CUDA;
        cfg.compute_type = Ct2ComputeType::FLOAT16;
        cfg
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ct2Config::default()
    }
}

fn ct2_model_dir(model_dir: &Path) -> Result<PathBuf> {
    let dir = model_dir.join("whisper-ct2");
    if !dir.join("model.bin").exists() {
        anyhow::bail!(
            "{}/model.bin not found -- run `./scripts/fetch-models.sh` from rust/",
            dir.display()
        );
    }
    Ok(dir)
}

fn whisper_cpp_model_path(model_dir: &Path) -> Result<PathBuf> {
    for candidate in [
        "ggml-large-v3-turbo.bin",
        "ggml-large-v3.bin",
        "ggml-tiny.en.bin",
    ] {
        let p = model_dir.join(candidate);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "no GGML whisper model in {} -- run `./scripts/fetch-models.sh` from rust/",
        model_dir.display()
    )
}

fn strip_special_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

fn num_cpus() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(super::defaults::stt::CT2_THREADS_DEFAULT as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_whisper_specials() {
        let raw =
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> hello world<|endoftext|>";
        assert_eq!(strip_special_tokens(raw), " hello world");
    }

    #[test]
    fn join_segments_inserts_spaces() {
        assert_eq!(join_segments(vec!["hello", "world"]), "hello world");
        assert_eq!(join_segments(vec![" hello ", " world"]), "hello world");
        assert_eq!(join_segments(vec!["hello", "", "world"]), "hello world");
        assert_eq!(join_segments(Vec::<&str>::new()), "");
    }

    #[test]
    fn parses_whisper_timestamp_tokens() {
        assert_eq!(parse_timestamp_token("<|0.00|>"), Some(0));
        assert_eq!(parse_timestamp_token("<|1.20|>"), Some(1200));
        assert_eq!(parse_timestamp_token("<|29.98|>"), Some(29_980));
        assert_eq!(parse_timestamp_token("<|0.5|>"), Some(500));
        assert_eq!(parse_timestamp_token("<|0.500|>"), Some(500));
        assert_eq!(parse_timestamp_token("<|sot|>"), None);
        assert_eq!(parse_timestamp_token("hello"), None);
        assert_eq!(parse_timestamp_token("<||>"), None);
    }

    #[test]
    fn splits_ct2_segments_on_timestamp_pairs() {
        let toks: Vec<String> = vec![
            "<|0.00|>", " hel", "lo", "<|1.20|>", "<|1.20|>", " wor", "ld", "<|2.50|>",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let segs = split_ct2_segments(&toks, &[], None, u32::MAX);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].t_start_ms, 0);
        assert_eq!(segs[0].t_end_ms, 1200);
        assert_eq!(segs[0].text_tokens, vec![" hel", "lo"]);
        assert_eq!(segs[1].t_start_ms, 1200);
        assert_eq!(segs[1].t_end_ms, 2500);
        assert_eq!(segs[1].text_tokens, vec![" wor", "ld"]);
    }

    #[test]
    fn splits_ct2_segments_skips_leading_text() {
        let toks: Vec<String> = vec!["<|sot|>", "<|en|>", "<|0.00|>", "hi", "<|1.00|>"]
            .into_iter()
            .map(String::from)
            .collect();
        let segs = split_ct2_segments(&toks, &[], None, u32::MAX);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text_tokens, vec!["hi"]);
    }

    #[test]
    fn splits_ct2_segments_by_id() {
        let toks: Vec<String> = vec!["<|0.00|>", "hi", "<|1.20|>"]
            .into_iter()
            .map(String::from)
            .collect();
        let ids: Vec<usize> = vec![50364, 12345, 50424];
        let segs = split_ct2_segments(&toks, &ids, Some(50364), u32::MAX);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].t_start_ms, 0);
        assert_eq!(segs[0].t_end_ms, 1200);
    }

    #[test]
    fn splits_ct2_segments_clamps_to_audio_ms() {
        let toks: Vec<String> = vec!["<|0.00|>", "hi", "<|29.84|>"]
            .into_iter()
            .map(String::from)
            .collect();
        let segs = split_ct2_segments(&toks, &[], None, 3_000);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].t_start_ms, 0);
        assert_eq!(segs[0].t_end_ms, 3_000);
    }

    #[test]
    fn splits_ct2_segments_drops_inverted_pairs() {
        let toks: Vec<String> = vec!["<|2.00|>", "hi", "<|1.00|>"]
            .into_iter()
            .map(String::from)
            .collect();
        let segs = split_ct2_segments(&toks, &[], None, u32::MAX);
        assert!(segs.is_empty(), "{segs:?}");
    }

    #[test]
    fn splits_ct2_segments_handles_no_timestamps() {
        let toks: Vec<String> = vec!["hi", "there"].into_iter().map(String::from).collect();
        let segs = split_ct2_segments(&toks, &[], None, u32::MAX);
        assert!(segs.is_empty(), "{segs:?}");
    }

    #[test]
    fn splits_ct2_segments_handles_trailing_only_timestamp() {
        let toks: Vec<String> = vec!["<|0.00|>", "hi"]
            .into_iter()
            .map(String::from)
            .collect();
        let segs = split_ct2_segments(&toks, &[], None, u32::MAX);
        assert!(segs.is_empty(), "{segs:?}");
    }

    #[test]
    fn proportional_word_timings_basic_three_word_uniform_chars() {
        let words = proportional_word_timings("aaa bbb ccc", 0, 900);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].word, "aaa");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 300);
        assert_eq!(words[1].start_ms, 300);
        assert_eq!(words[1].end_ms, 600);
        assert_eq!(words[2].start_ms, 600);
        assert_eq!(words[2].end_ms, 900);
    }

    #[test]
    fn proportional_word_timings_handles_unicode_char_counts() {
        let words = proportional_word_timings("café latte", 0, 2000);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "café");
        assert!(words[0].end_ms == words[1].start_ms);
    }

    #[test]
    fn group_whisper_tokens_basic_leading_space_boundary() {
        let toks: Vec<(String, i64, i64)> = vec![
            (" hel".into(), 10, 20),
            ("lo".into(), 20, 35),
            (" wor".into(), 50, 60),
            ("ld".into(), 60, 75),
        ];
        let words = group_whisper_tokens_into_words(&toks, 0, 800, "hello world");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start_ms, 100);
        assert_eq!(words[0].end_ms, 350);
        assert_eq!(words[1].word, "world");
        assert_eq!(words[1].start_ms, 500);
        assert_eq!(words[1].end_ms, 750);
    }

    #[test]
    fn group_whisper_tokens_falls_back_when_empty() {
        let words = group_whisper_tokens_into_words(&[], 0, 1000, "fall back text");
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].word, "fall");
    }

    #[test]
    fn silence_gate_returns_empty() {
        let zeros = vec![0.0f32; 16_000];
        assert!(peak_amplitude(&zeros) < SILENCE_PEAK_THRESHOLD);
        let signal: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.001).sin()).collect();
        assert!(peak_amplitude(&signal) > SILENCE_PEAK_THRESHOLD);
    }

    #[test]
    fn pseudo_token_filter_drops_bracketed_and_angled() {
        assert!(is_whisper_pseudo_token("[_BEG_]"));
        assert!(is_whisper_pseudo_token("[_TT_110]"));
        assert!(is_whisper_pseudo_token("[BLANK_AUDIO]"));
        assert!(is_whisper_pseudo_token("<|endoftext|>"));
        assert!(is_whisper_pseudo_token(" <|0.00|> "));
        assert!(!is_whisper_pseudo_token(" hello"));
        assert!(!is_whisper_pseudo_token("["));
        assert!(!is_whisper_pseudo_token("]bracket"));
    }

    #[test]
    fn pseudo_tokens_never_reach_word_grouping() {
        let toks: Vec<(String, i64, i64)> = vec![(" hel".into(), 10, 20), ("lo".into(), 20, 35)];
        let words = group_whisper_tokens_into_words(&toks, 0, 800, "hello");
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hello");
        assert!(!words.iter().any(|w| is_whisper_pseudo_token(&w.word)));
    }

    #[test]
    fn clamps_centisecond_timestamps_to_real_audio() {
        assert_eq!(clamp_centisecond_ts(0, 8_420), 0);
        assert_eq!(clamp_centisecond_ts(300, 8_420), 3_000);
        assert_eq!(clamp_centisecond_ts(3_000, 8_420), 8_420);
        assert_eq!(clamp_centisecond_ts(-5, 8_420), 0);
    }

    #[test]
    fn strips_stray_leading_colon() {
        assert_eq!(strip_leading_colon(": hello"), "hello");
        assert_eq!(strip_leading_colon(":hello"), "hello");
        assert_eq!(strip_leading_colon("hello: world"), "hello: world");
        assert_eq!(strip_leading_colon(""), "");
    }

    #[test]
    fn whole_clip_segment_repairs_empty_segment_list() {
        let toks: Vec<String> = vec!["hi", "there"].into_iter().map(String::from).collect();
        let segs = split_ct2_segments(&toks, &[], None, 8_420);
        assert!(segs.is_empty(), "precondition: no timestamp tokens");

        let repaired = whole_clip_segment("hi there", 8_420, Some(-0.3), Some(0.02));
        assert_eq!(repaired.len(), 1);
        assert_eq!(repaired[0].t_start_ms, 0);
        assert_eq!(repaired[0].t_end_ms, 8_420);
        assert_eq!(repaired[0].text, "hi there");
        assert_eq!(repaired[0].avg_logprob, Some(-0.3));
        assert_eq!(repaired[0].words.len(), 2);
        assert!(repaired[0].words.iter().all(|w| w.end_ms <= 8_420));
    }

    #[test]
    fn whole_clip_segment_stays_empty_for_empty_text() {
        assert!(whole_clip_segment("", 8_420, None, None).is_empty());
        assert!(whole_clip_segment("   ", 8_420, None, None).is_empty());
        assert!(whole_clip_segment("hi", 0, None, None).is_empty());
    }

    #[test]
    fn language_code_strips_whisper_token_delimiters() {
        assert_eq!(language_code("<|en|>"), Some("en".to_string()));
        assert_eq!(language_code("<|ZH|>"), Some("zh".to_string()));
        assert_eq!(language_code("de"), Some("de".to_string()));
        assert_eq!(language_code("<||>"), None);
    }

    #[test]
    fn turbo_checkpoints_are_transcribe_only() {
        assert!(checkpoint_is_transcribe_only("ggml-large-v3-turbo"));
        assert!(checkpoint_is_transcribe_only(
            "faster-whisper-large-v3-turbo-ct2"
        ));
        assert!(!checkpoint_is_transcribe_only("ggml-large-v3"));
    }

    #[test]
    fn english_only_checkpoints_are_transcribe_only() {
        assert!(checkpoint_is_transcribe_only("ggml-tiny.en"));
        assert!(checkpoint_is_transcribe_only("ggml-base.en.bin"));
        assert!(checkpoint_is_transcribe_only("faster-whisper-medium.en"));
        assert!(checkpoint_is_transcribe_only(&checkpoint_tail(Path::new(
            "/hub/models--Systran--faster-whisper-tiny.en/snapshots/abc"
        ))));
        assert!(!checkpoint_is_transcribe_only("ggml-large-v3"));
        assert!(!checkpoint_is_transcribe_only("whisper-large-v3"));
        assert!(!checkpoint_is_transcribe_only("checkpoints.enc"));
        assert!(!checkpoint_is_transcribe_only("model.env"));
    }

    #[test]
    fn checkpoint_tail_sees_hf_repo_name_but_not_ancestors() {
        let hf = Path::new(
            "/turbo-cache/models--deepdml--faster-whisper-large-v3-turbo-ct2/snapshots/abc",
        );
        assert!(checkpoint_is_transcribe_only(&checkpoint_tail(hf)));
        let plain = Path::new("/turbo-cache/models/whisper-large-v3/snapshots/abc");
        assert!(!checkpoint_is_transcribe_only(&checkpoint_tail(plain)));
    }

    #[test]
    fn translate_unsupported_is_downcastable() {
        let err = anyhow::Error::new(TranslateUnsupported {
            model_id: "ggml-large-v3-turbo".into(),
        });
        assert!(is_translate_unsupported(&err));
        assert!(err.to_string().contains("ggml-large-v3-turbo"));
        assert!(!is_translate_unsupported(&anyhow!("some other failure")));
    }

    #[test]
    fn backend_selection_via_env() {
        let saved = std::env::var("STT_BACKEND").ok();
        std::env::remove_var("STT_BACKEND");
        assert_eq!(Backend::from_env(), Backend::WhisperCpp);
        std::env::set_var("STT_BACKEND", "ct2");
        assert_eq!(Backend::from_env(), Backend::Ct2);
        std::env::set_var("STT_BACKEND", "parakeet");
        assert_eq!(Backend::from_env(), Backend::Parakeet);
        std::env::set_var("STT_BACKEND", "whisper-cpp");
        assert_eq!(Backend::from_env(), Backend::WhisperCpp);
        match saved {
            Some(v) => std::env::set_var("STT_BACKEND", v),
            None => std::env::remove_var("STT_BACKEND"),
        }
    }

    fn transcribe_ct2_legacy(state: &Ct2State, audio: &[f32]) -> Result<String> {
        let padded = mel::pad_or_truncate_to_30s(audio);
        let mut mel_data = state.mel.log_mel(&padded);
        let n_mels = state.mel.n_mels;
        let features = Ct2StorageView::new(
            &[1usize, n_mels, mel::N_FRAMES],
            mel_data.as_mut_slice(),
            ct2rs::Device::CPU,
        )
        .context("ct2 storage view for mel features")?;
        let prompts: Vec<Vec<&str>> = vec![vec![
            "<|startoftranscript|>",
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ]];
        let options = Ct2WhisperOptions {
            beam_size: stt_beam_size(),
            ..Default::default()
        };
        let mut results = state
            .model
            .generate(&features, &prompts, &options)
            .context("ct2 whisper generate")?;
        let r = results
            .pop()
            .ok_or_else(|| anyhow!("ct2 whisper returned no generation results"))?;
        let tokens = r.sequences.into_iter().next().unwrap_or_default();
        if tokens.is_empty() {
            return Ok(String::new());
        }
        let decoded = state
            .tokenizer
            .decode(tokens)
            .context("ct2 tokenizer decode")?;
        Ok(strip_special_tokens(&decoded).trim().to_string())
    }

    #[test]
    fn ct2_timestamps_perf_cost() {
        if std::env::var("CT2_BENCH").ok().as_deref() != Some("1") {
            eprintln!("ct2 bench: skipping -- set CT2_BENCH=1 to run");
            return;
        }
        use std::path::PathBuf;
        let audio_path =
            std::env::var("CT2_DIFF_AUDIO").unwrap_or_else(|_| "/tmp/audio16k_ref.bin".to_string());
        let model_dir_root = std::env::var("CT2_DIFF_MODEL_DIR").unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .to_string_lossy()
                .into_owned()
        });
        if !std::path::Path::new(&audio_path).exists() {
            eprintln!("ct2 bench: skipping -- no {audio_path}");
            return;
        }
        let dir = match ct2_model_dir(std::path::Path::new(&model_dir_root)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ct2 bench: skipping -- no ct2 model: {e:#}");
                return;
            }
        };

        let bytes = std::fs::read(&audio_path).expect("read audio bin");
        let audio: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let model = Ct2Whisper::new(&dir, ct2_config()).expect("load ct2 model");
        let tokenizer = Ct2Tokenizer::new(&dir).expect("load tokenizer");
        let n_mels = model.n_mels();
        let timestamp_begin_id = tokenizer.token_to_id("<|0.00|>");
        let state = Ct2State {
            model,
            tokenizer,
            mel: mel::WhisperMel::new(n_mels),
            timestamp_begin_id,
            supports_translate: false,
        };

        let runs: usize = std::env::var("CT2_BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);

        let bench = |label: &str, f: &dyn Fn()| -> u128 {
            f();
            let mut times: Vec<u128> = Vec::with_capacity(runs);
            for _ in 0..runs {
                let t = std::time::Instant::now();
                f();
                times.push(t.elapsed().as_micros());
            }
            times.sort();
            let median = times[times.len() / 2];
            let min = times[0];
            let max = times[times.len() - 1];
            eprintln!(
                "{label:>16}: median {:>6.1} ms  (min {:>6.1}, max {:>6.1}, n={runs})",
                median as f64 / 1000.0,
                min as f64 / 1000.0,
                max as f64 / 1000.0,
            );
            median
        };

        let m_legacy = bench("legacy (no-ts)", &|| {
            let _ = transcribe_ct2_legacy(&state, &audio).expect("legacy");
        });
        let m_new = bench("new (ts+split)", &|| {
            let _ = transcribe_ct2(&state, &audio, WhisperTask::Transcribe).expect("new");
        });

        let delta_pct = (m_new as f64 - m_legacy as f64) / m_legacy as f64 * 100.0;
        eprintln!(
            "delta: {:+.2}% ({:+.1} ms median)",
            delta_pct,
            (m_new as f64 - m_legacy as f64) / 1000.0
        );
    }

    #[test]
    fn ct2_per_segment_join_matches_legacy_on_real_sample() {
        use std::path::PathBuf;
        let audio_path =
            std::env::var("CT2_DIFF_AUDIO").unwrap_or_else(|_| "/tmp/audio16k_ref.bin".to_string());
        let model_dir_root = std::env::var("CT2_DIFF_MODEL_DIR").unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .to_string_lossy()
                .into_owned()
        });

        if !std::path::Path::new(&audio_path).exists() {
            eprintln!("ct2 diff: skipping -- no {audio_path}");
            return;
        }
        let dir = match ct2_model_dir(std::path::Path::new(&model_dir_root)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ct2 diff: skipping -- no ct2 model: {e:#}");
                return;
            }
        };

        let bytes = std::fs::read(&audio_path).expect("read audio bin");
        if !bytes.len().is_multiple_of(4) {
            panic!("audio file size {} not a multiple of 4 bytes", bytes.len());
        }
        let audio: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let model = Ct2Whisper::new(&dir, ct2_config()).expect("load ct2 model");
        let tokenizer = Ct2Tokenizer::new(&dir).expect("load tokenizer");
        let n_mels = model.n_mels();
        let timestamp_begin_id = tokenizer.token_to_id("<|0.00|>");
        let state = Ct2State {
            model,
            tokenizer,
            mel: mel::WhisperMel::new(n_mels),
            timestamp_begin_id,
            supports_translate: false,
        };

        let legacy = transcribe_ct2_legacy(&state, &audio).expect("legacy decode");
        let new = transcribe_ct2(&state, &audio, WhisperTask::Transcribe).expect("new decode");

        eprintln!("legacy  : {:?}", legacy);
        eprintln!("new.text: {:?}", new.text);
        eprintln!("new.segments ({}):", new.segments.len());
        for (i, s) in new.segments.iter().enumerate() {
            eprintln!("  [{i}] {}-{} ms  {:?}", s.t_start_ms, s.t_end_ms, s.text);
        }

        let words = |s: &str| -> Vec<String> {
            s.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(String::from)
                .collect()
        };
        let lw = words(&legacy);
        let nw = words(&new.text);
        assert!(
            !lw.is_empty(),
            "legacy decode produced no words; sample likely too quiet"
        );
        assert_eq!(
            lw, nw,
            "ct2 BPE-per-segment join drifted from legacy whole-sequence decode\n\
             legacy words: {:?}\n   new words: {:?}",
            lw, nw
        );

        let all_seg_words: Vec<String> = new.segments.iter().flat_map(|s| words(&s.text)).collect();
        assert_eq!(
            all_seg_words, nw,
            "joined text doesn't equal concatenation of segment texts"
        );
    }
}
