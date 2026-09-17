use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use tokio::sync::mpsc;
use tracing::{info, warn};

use nv_omni::vocoder::{Vocoder, NUM_CODEBOOKS, STREAM_CHUNK_FRAMES, STREAM_LEFT_CONTEXT_FRAMES};
use nv_tts::{
    Qwen3TtsCodecDecoder, Qwen3TtsTalker, Qwen3TtsTokenizer, Sampler, SamplerConfig,
    VocoderInventory, VocoderLoadReport, VoiceStoreHandle,
};

use crate::oapi::audio_speech::{
    AudioSpeech, InvalidVoiceRequest, SilentVocoder, UnknownVoice, NV_TTS_SAMPLE_RATE,
};
use crate::oapi::voice_profiles::voice_profile_root;

pub const ENV_TALKER_DIR: &str = "NV_TTS_TALKER_DIR";

pub const ENV_VOCODER_DIR: &str = "NV_TTS_VOCODER_DIR";

pub const ENV_ALLOW_SILENT_VOCODER: &str = "NV_TTS_ALLOW_SILENT_VOCODER";

pub const MIN_SILENCE_SECONDS: f32 = 0.20;

pub const SAMPLES_PER_TOKEN: usize = 1920 * 2;

pub const MAX_CODEC_STEPS: usize = 2048;

pub const MIN_CODEC_STEPS: usize = 2;

pub const ENV_STREAM_CHUNK_FRAMES: &str = "NV_TTS_STREAM_CHUNK_FRAMES";

pub const ENV_STREAM_LEFT_CONTEXT: &str = "NV_TTS_STREAM_LEFT_CONTEXT";

pub const ENV_LANGUAGE: &str = "NV_TTS_LANG";

pub const ENV_SEED: &str = "NV_TTS_SEED";

pub const ENV_GREEDY: &str = "NV_TTS_GREEDY";

pub const DEFAULT_SPEAKER: &str = "serena";

pub const DEFAULT_PROFILE: &str = "default";

pub struct Qwen3TtsAudioSpeech {
    pub talker_dir: PathBuf,
    pub vocoder_dir: PathBuf,
    pub tokenizer: Arc<Qwen3TtsTokenizer>,

    pub talker: Option<Arc<Qwen3TtsTalker>>,
    pub code_predictor: Option<Arc<Qwen3TtsCodecDecoder>>,

    pub vocoder: Arc<Vocoder>,
    pub vocoder_inventory: VocoderInventory,
    pub vocoder_report: VocoderLoadReport,

    pub voice_profiles: Arc<VoiceStoreHandle>,
    pub profiles_supported: bool,
    pub default_voice: Arc<DefaultVoiceCache>,

    pub sample_rate: u32,

    pub allow_silent_vocoder: bool,
}

pub fn allow_silent_vocoder(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some(v) => {
            let v = v.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        }
    }
}

fn allow_silent_vocoder_from_env() -> bool {
    allow_silent_vocoder(std::env::var(ENV_ALLOW_SILENT_VOCODER).ok().as_deref())
}

pub fn zero_init_vocoder_message(vocoder_dir: &Path, source: &str, reason: Option<&str>) -> String {
    format!(
        "refusing to serve nv-tts with a zero-initialised vocoder: loaded from {} ({source}); \
         {}. A zero-init vocoder decodes every frame to silence, so /v1/audio/speech would \
         answer 200 with a correctly-sized WAV of PURE SILENCE -- status, length and \
         content-type all indistinguishable from success. Point {} at a Qwen3-TTS checkpoint \
         whose speech_tokenizer/model.safetensors carries the decoder.* tensors, or set {}=1 \
         to serve silence deliberately.",
        vocoder_dir.display(),
        reason.unwrap_or("no reason reported by the loader"),
        ENV_VOCODER_DIR,
        ENV_ALLOW_SILENT_VOCODER,
    )
}

static BOOTSTRAP_FAILURE: OnceLock<String> = OnceLock::new();

pub fn bootstrap_failure() -> Option<&'static str> {
    BOOTSTRAP_FAILURE.get().map(String::as_str)
}

pub fn bootstrap_failure_message(talker_dir: &Path, err: &str) -> String {
    format!(
        "nv-tts bootstrap failed for {}: {err}. {} is set, so /v1/audio/speech refuses \
         nv-tts requests with 503 rather than silently answering them with Kokoro. Fix the \
         load error or unset {} to disable nv-tts.",
        talker_dir.display(),
        ENV_TALKER_DIR,
        ENV_TALKER_DIR,
    )
}

impl Qwen3TtsAudioSpeech {
    pub fn from_env() -> Option<Arc<Self>> {
        let talker_dir = std::env::var_os(ENV_TALKER_DIR)?;
        let talker_dir = PathBuf::from(talker_dir);
        match Self::from_dirs(&talker_dir, None) {
            Ok(s) => Some(Arc::new(s)),
            Err(err) => {
                let detail = bootstrap_failure_message(&talker_dir, &format!("{err:#}"));
                tracing::error!(
                    talker_dir = %talker_dir.display(),
                    error = %format!("{err:#}"),
                    "{}",
                    detail
                );
                let _ = BOOTSTRAP_FAILURE.set(detail);
                None
            }
        }
    }

    pub fn from_dirs(talker_dir: &Path, vocoder_dir_override: Option<&Path>) -> Result<Self> {
        Self::from_dirs_with(
            talker_dir,
            vocoder_dir_override,
            allow_silent_vocoder_from_env(),
        )
    }

    pub fn from_dirs_with(
        talker_dir: &Path,
        vocoder_dir_override: Option<&Path>,
        allow_silent_vocoder: bool,
    ) -> Result<Self> {
        if !talker_dir.is_dir() {
            anyhow::bail!(
                "{}: not a directory: {}",
                ENV_TALKER_DIR,
                talker_dir.display()
            );
        }
        let tokenizer = Qwen3TtsTokenizer::from_dir(talker_dir)
            .with_context(|| format!("load tokenizer from {}", talker_dir.display()))?;
        let (vocoder_dir_owned, vocoder_dir_source): (PathBuf, &str) = match vocoder_dir_override {
            Some(p) => (p.to_path_buf(), "explicit override"),
            None => match std::env::var_os(ENV_VOCODER_DIR) {
                Some(s) => (PathBuf::from(s), ENV_VOCODER_DIR),
                None => (
                    talker_dir.to_path_buf(),
                    "unset NV_TTS_VOCODER_DIR, defaulted to NV_TTS_TALKER_DIR",
                ),
            },
        };
        let device = Device::Cpu;
        let (voc, report) =
            nv_tts::load_vocoder_from_qwen3_tts(&vocoder_dir_owned, &device, DType::F32)
                .with_context(|| {
                    format!("load vocoder weights from {}", vocoder_dir_owned.display())
                })?;
        if report.zero_init_fallback {
            if !allow_silent_vocoder {
                anyhow::bail!(
                    "{}",
                    zero_init_vocoder_message(
                        &vocoder_dir_owned,
                        vocoder_dir_source,
                        report.fallback_reason.as_deref(),
                    )
                );
            }
            tracing::error!(
                vocoder_dir = %vocoder_dir_owned.display(),
                source = vocoder_dir_source,
                reason = report.fallback_reason.as_deref().unwrap_or("unknown"),
                "{}=1: serving nv-tts with a ZERO-INIT vocoder -- every /v1/audio/speech \
                 response will be a correctly-sized WAV of pure silence",
                ENV_ALLOW_SILENT_VOCODER,
            );
        } else if let Some(reason) = &report.fallback_reason {
            warn!(reason = %reason, "vocoder load reported a fallback");
        }

        let (talker, code_predictor) = load_talker_and_code_predictor(talker_dir, &device);
        let has_talker = talker.is_some();
        let has_cp = code_predictor.is_some();

        info!(
            talker_dir = %talker_dir.display(),
            vocoder_dir = %vocoder_dir_owned.display(),
            decoder_keys = report.inventory.decoder_key_count,
            sample_rate = report.inventory.sample_rate,
            has_talker,
            has_code_predictor = has_cp,
            "nv-tts AudioSpeech ready"
        );
        let inventory = report.inventory.clone();
        let voice_profiles = Arc::new(VoiceStoreHandle::new(voice_profile_root()));
        let profiles_supported = tts_model_type(talker_dir).as_deref() == Some("base");

        Ok(Self {
            talker_dir: talker_dir.to_path_buf(),
            vocoder_dir: vocoder_dir_owned,
            tokenizer: Arc::new(tokenizer),
            talker: talker.map(Arc::new),
            code_predictor: code_predictor.map(Arc::new),
            vocoder: Arc::new(voc),
            vocoder_inventory: inventory,
            vocoder_report: report,
            voice_profiles,
            profiles_supported,
            default_voice: Arc::new(DefaultVoiceCache::default()),
            sample_rate: NV_TTS_SAMPLE_RATE,
            allow_silent_vocoder,
        })
    }

    pub fn talker_model_id(&self) -> String {
        crate::oapi::model_ids::model_id_for_dir(&self.talker_dir)
    }
}

pub fn is_default_voice_alias(name: &str) -> bool {
    let n = name.trim().to_ascii_lowercase();
    n.is_empty() || n == "default" || crate::tts::text::is_openai_voice_alias(&n)
}

fn alias_without_speakers_message(voice: &str, profiles_supported: bool, hidden: usize) -> String {
    if profiles_supported {
        format!(
            "voice {voice:?} is an OpenAI alias, but this Base checkpoint ships no built-in \
             speakers (spk_id is empty), no voice profile named {DEFAULT_PROFILE:?} is \
             enrolled, and the engine's built-in default voice is not {hidden}-d, so there is \
             nothing to condition on; enroll a profile via POST /v1/voice-profiles and pass \
             its name as `voice`, or enroll one named {DEFAULT_PROFILE:?} to give the OpenAI \
             aliases a voice"
        )
    } else {
        format!(
            "voice {voice:?} is an OpenAI alias, but this checkpoint ships no built-in speakers \
             (spk_id is empty) and its tts_model_type != \"base\" cannot use voice profiles \
             either; load a CustomVoice checkpoint for named speakers or a Base checkpoint for \
             profiles"
        )
    }
}

fn unknown_voice_message(
    voice: &str,
    spk_map: &[(String, u32)],
    profiles_supported: bool,
) -> String {
    let mut names: Vec<&str> = spk_map.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let profiles = if profiles_supported {
        ", or any voice profile enrolled via POST /v1/voice-profiles"
    } else {
        " (this checkpoint has tts_model_type != \"base\", so enrolled voice profiles are not \
         usable with it)"
    };
    format!(
        "voice {voice:?} not found; valid voices: {}{profiles}. Earlier builds answered 200 \
         with the default speaker under the requested name; that silent substitution is what \
         this refusal replaces",
        names.join(", ")
    )
}

fn tts_model_type(dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("tts_model_type")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

#[async_trait::async_trait]
impl AudioSpeech for Qwen3TtsAudioSpeech {
    async fn synthesize(&self, text: &str, voice: &str) -> Result<mpsc::Receiver<Vec<f32>>> {
        let token_ids = self.tokenizer.encode_text(text).unwrap_or_else(|err| {
            warn!(error = %err, "tokenizer encode failed; emitting min-length silence");
            Vec::new()
        });
        let token_count = token_ids.len();

        let (tx, rx) = mpsc::channel::<Vec<f32>>(4);

        if self.vocoder.is_zero_init() {
            if !self.allow_silent_vocoder {
                return Err(anyhow::Error::new(SilentVocoder(
                    zero_init_vocoder_message(
                        &self.vocoder_dir,
                        "loaded engine",
                        self.vocoder_report.fallback_reason.as_deref(),
                    ),
                )));
            }
            warn!(
                vocoder_dir = %self.vocoder_dir.display(),
                "{}=1: emitting a WAV of pure silence for this request (zero-init vocoder)",
                ENV_ALLOW_SILENT_VOCODER,
            );
            let total_samples = (token_count.saturating_mul(SAMPLES_PER_TOKEN))
                .max((self.sample_rate as f32 * MIN_SILENCE_SECONDS) as usize);
            let chunk = 4_096usize;
            let mut remaining = total_samples;
            let send_tx = tx.clone();
            tokio::spawn(async move {
                while remaining > 0 {
                    let n = remaining.min(chunk);
                    let buf = vec![0.0f32; n];
                    if send_tx.send(buf).await.is_err() {
                        break;
                    }
                    remaining -= n;
                }
                drop(send_tx);
            });
            drop(tx);
            return Ok(rx);
        }

        if let (Some(talker), Some(cp)) = (self.talker.clone(), self.code_predictor.clone()) {
            if talker.has_text_embedding() && !token_ids.is_empty() {
                let vocoder = self.vocoder.clone();
                let role_ids = self.role_prefix_ids()?;
                let (speaker_embed, speaker_name) = match self.resolver() {
                    Some(r) => {
                        let voice = voice.to_string();
                        tokio::task::spawn_blocking(move || r.resolve(&voice))
                            .await
                            .context("speaker resolution task")??
                    }
                    None => (None, "none".to_string()),
                };
                let language_id = self.resolve_language_id();
                let tk = self.tokenizer.clone();
                let (chunk_frames, left_context) = stream_geometry();

                let gen_talker = talker.clone();
                let gen_cp = cp.clone();
                let codebook = cp.config().codebook_vocab_size as u32;
                let started = std::time::Instant::now();

                let (frame_tx, mut frame_rx) = mpsc::channel::<[u32; NUM_CODEBOOKS]>(64);
                let (pcm_tx, mut pcm_rx) = mpsc::channel::<Vec<f32>>(4);

                let gen_task = tokio::task::spawn_blocking(move || -> Result<usize> {
                    let (prefill, tts_pad) = gen_talker.build_nonstreaming_prefill(
                        &role_ids,
                        &token_ids,
                        speaker_embed.as_ref(),
                        language_id,
                        tk.bos_id(),
                        tk.eos_id(),
                        tk.special_id(nv_tts::tokenizer::SPECIAL_TTS_PAD)
                            .unwrap_or(nv_tts::tokenizer::SPECIAL_ID_TTS_PAD),
                    )?;
                    let (mut talker_sampler, mut sub_sampler) = build_samplers();
                    let mut produced = 0usize;
                    let n = gen_talker.generate_frames_streaming(
                        &prefill,
                        &tts_pad,
                        &gen_cp,
                        MAX_CODEC_STEPS,
                        MIN_CODEC_STEPS,
                        &mut talker_sampler,
                        &mut sub_sampler,
                        &mut |frame| {
                            for (k, &tok) in frame.iter().enumerate() {
                                if tok >= codebook {
                                    anyhow::bail!(
                                        "qwen3-tts generated out-of-range codec id: frame {produced} \
                                         codebook {k} id {tok} >= {codebook}; refusing to vocode garbage"
                                    );
                                }
                            }
                            produced += 1;
                            Ok(frame_tx.blocking_send(frame).is_ok())
                        },
                    )?;
                    if n == 0 {
                        anyhow::bail!("qwen3-tts produced no frames before EOS");
                    }
                    info!(
                        frames = n,
                        text_tokens = token_count,
                        speaker = %speaker_name,
                        audio_secs = n as f32 / 12.5,
                        gen_secs = started.elapsed().as_secs_f32(),
                        "qwen3-tts talker generation complete"
                    );
                    Ok(n)
                });

                let dec_task = tokio::task::spawn_blocking(move || -> Result<()> {
                    let mut streamer = vocoder.streamer(chunk_frames, left_context)?;
                    let send = |pcm: Vec<f32>| -> bool {
                        for chunk in pcm.chunks(24_000) {
                            if pcm_tx.blocking_send(chunk.to_vec()).is_err() {
                                return false;
                            }
                        }
                        true
                    };
                    while let Some(frame) = frame_rx.blocking_recv() {
                        if let Some(pcm) = streamer.push(frame)? {
                            if !send(pcm) {
                                return Ok(());
                            }
                        }
                    }
                    let tail = streamer.finish()?;
                    if !tail.is_empty() {
                        send(tail);
                    }
                    Ok(())
                });

                let Some(first) = pcm_rx.recv().await else {
                    let gen_res = gen_task.await.context("talker generation task panicked")?;
                    let dec_res = dec_task.await.context("vocoder decode task panicked")?;
                    gen_res?;
                    dec_res?;
                    anyhow::bail!("qwen3-tts pipeline closed without emitting audio");
                };
                info!(
                    ttfa_secs = started.elapsed().as_secs_f32(),
                    chunk_frames, left_context, "qwen3-tts first audio chunk ready"
                );

                let send_tx = tx.clone();
                tokio::spawn(async move {
                    let mut ok = send_tx.send(first).await.is_ok();
                    while ok {
                        match pcm_rx.recv().await {
                            Some(chunk) => ok = send_tx.send(chunk).await.is_ok(),
                            None => break,
                        }
                    }
                    drop(pcm_rx);
                    drop(send_tx);
                    match gen_task.await {
                        Ok(Ok(_)) => {}
                        Ok(Err(err)) => warn!(
                            error = %format!("{err:#}"),
                            "qwen3-tts generation failed mid-stream; audio truncated"
                        ),
                        Err(err) => warn!(
                            error = %err,
                            "talker generation task panicked mid-stream"
                        ),
                    }
                    match dec_task.await {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => warn!(
                            error = %format!("{err:#}"),
                            "qwen3-tts streaming decode failed mid-stream; audio truncated"
                        ),
                        Err(err) => warn!(
                            error = %err,
                            "vocoder decode task panicked mid-stream"
                        ),
                    }
                });
                drop(tx);
                return Ok(rx);
            }
        }

        anyhow::bail!(
            "Qwen3-TTS pipeline incomplete: talker={}, code_predictor={}, text_embedding={}. \
             Refusing to emit random-LCG fake audio. Ensure talker_dir contains \
             model.safetensors with both the talker and code_predictor weights.",
            self.talker.is_some(),
            self.code_predictor.is_some(),
            self.talker
                .as_ref()
                .map(|t| t.has_text_embedding())
                .unwrap_or(false),
        );
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn model_id(&self) -> Option<String> {
        Some(self.talker_model_id())
    }

    fn voices(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .talker
            .as_ref()
            .map(|t| t.config().spk_id.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default();
        names.sort_unstable();
        names
    }
}

const BUILTIN_DEFAULT_XVECTOR: &[u8] =
    include_bytes!("../../voices/qwen3-tts-0_6b-default.xvector.json");

fn builtin_default_profile() -> Option<&'static nv_tts::VoiceProfile> {
    static PARSED: std::sync::OnceLock<Option<nv_tts::VoiceProfile>> = std::sync::OnceLock::new();
    PARSED
        .get_or_init(|| match serde_json::from_slice(BUILTIN_DEFAULT_XVECTOR) {
            Ok(p) => Some(p),
            Err(err) => {
                tracing::error!(error = %err, "built-in default x-vector does not parse");
                None
            }
        })
        .as_ref()
}

pub fn builtin_default_xvector(hidden: usize) -> Option<&'static [f32]> {
    let p = builtin_default_profile()?;
    (p.embedding.len() == hidden).then_some(p.embedding.as_slice())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DefaultVoiceSource {
    Profile(Option<SystemTime>),
    Builtin,
}

#[derive(Default)]
pub struct DefaultVoiceCache {
    slot: std::sync::Mutex<Option<(DefaultVoiceSource, candle_core::Tensor)>>,
}

impl DefaultVoiceCache {
    fn get_or_build(
        &self,
        source: DefaultVoiceSource,
        build: impl FnOnce() -> Result<candle_core::Tensor>,
    ) -> Result<candle_core::Tensor> {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((cached_source, tensor)) = slot.as_ref() {
            if *cached_source == source {
                return Ok(tensor.clone());
            }
        }
        let tensor = build()?;
        *slot = Some((source, tensor.clone()));
        Ok(tensor)
    }
}

pub struct SpeakerResolver {
    pub talker: Arc<Qwen3TtsTalker>,
    pub store: Arc<VoiceStoreHandle>,
    pub profiles_supported: bool,
    pub cache: Arc<DefaultVoiceCache>,
}

fn profile_tensor(profile: nv_tts::VoiceProfile, hidden: usize) -> Result<candle_core::Tensor> {
    let name = profile.name;
    if profile.embedding.len() != hidden {
        return Err(anyhow::Error::new(InvalidVoiceRequest(format!(
            "voice profile {name:?} has a {}-d embedding but this talker expects {hidden}-d; \
             re-enroll the profile against the current checkpoint",
            profile.embedding.len()
        ))));
    }
    if profile.embedding.iter().all(|x| *x == 0.0) {
        return Err(anyhow::Error::new(InvalidVoiceRequest(format!(
            "voice profile {name:?} has an all-zero embedding (enrolled without a \
             speaker encoder); re-enroll it against a Base checkpoint"
        ))));
    }
    candle_core::Tensor::from_vec(profile.embedding, (1usize, 1usize, hidden), &Device::Cpu)
        .map_err(|e| anyhow::anyhow!("voice profile tensor: {e}"))
}

impl SpeakerResolver {
    fn codec_row(&self, name: &str, id: u32) -> Result<(Option<candle_core::Tensor>, String)> {
        match self.talker.codec_embed_rows(&[id]) {
            Ok(emb) => Ok((Some(emb), name.to_string())),
            Err(err) => {
                warn!(error = %err, speaker = %name, "speaker codec embed failed");
                Ok((None, "none".to_string()))
            }
        }
    }

    fn named_profile(&self, voice: &str) -> Result<(Option<candle_core::Tensor>, String)> {
        let spk_map = &self.talker.config().spk_id;
        if !self.profiles_supported {
            return Err(anyhow::Error::new(UnknownVoice(unknown_voice_message(
                voice, spk_map, false,
            ))));
        }
        let store = self.store.get()?;
        if !store.exists(voice) {
            return Err(anyhow::Error::new(UnknownVoice(unknown_voice_message(
                voice, spk_map, true,
            ))));
        }
        let profile = store
            .get(voice)
            .map_err(|e| InvalidVoiceRequest(format!("voice profile {voice:?}: {e:#}")))?;
        let tensor = profile_tensor(profile, self.talker.config().hidden_size)?;
        Ok((Some(tensor), format!("profile:{voice}")))
    }

    fn default_voice(&self, requested: &str) -> Result<(Option<candle_core::Tensor>, String)> {
        let hidden = self.talker.config().hidden_size;
        if !self.profiles_supported {
            return Err(anyhow::Error::new(UnknownVoice(
                alias_without_speakers_message(requested, false, hidden),
            )));
        }
        let store = match self.store.get() {
            Ok(store) => Some(store),
            Err(err) => {
                tracing::debug!(error = %err, "voice store unavailable; using the built-in default voice");
                None
            }
        };
        if let Some(store) = store.filter(|s| s.exists(DEFAULT_PROFILE)) {
            let source = DefaultVoiceSource::Profile(store.modified(DEFAULT_PROFILE));
            let tensor = self.cache.get_or_build(source, || {
                let profile = store.get(DEFAULT_PROFILE).map_err(|e| {
                    InvalidVoiceRequest(format!("voice profile {DEFAULT_PROFILE:?}: {e:#}"))
                })?;
                profile_tensor(profile, hidden)
            })?;
            return Ok((Some(tensor), format!("profile:{DEFAULT_PROFILE}")));
        }
        if let Some(xvec) = builtin_default_xvector(hidden) {
            let tensor = self.cache.get_or_build(DefaultVoiceSource::Builtin, || {
                candle_core::Tensor::from_vec(xvec.to_vec(), (1usize, 1usize, hidden), &Device::Cpu)
                    .map_err(|e| anyhow::anyhow!("built-in default voice tensor: {e}"))
            })?;
            return Ok((Some(tensor), "builtin:default".to_string()));
        }
        Err(anyhow::Error::new(UnknownVoice(
            alias_without_speakers_message(requested, self.profiles_supported, hidden),
        )))
    }

    pub fn resolve(&self, voice: &str) -> Result<(Option<candle_core::Tensor>, String)> {
        let spk_map = &self.talker.config().spk_id;
        let requested = voice.trim().to_lowercase();
        if let Some((name, id)) = spk_map.iter().find(|(name, _)| *name == requested) {
            return self.codec_row(name, *id);
        }
        if is_default_voice_alias(&requested) {
            if spk_map.is_empty() {
                return self.default_voice(voice);
            }
            let (name, id) = spk_map
                .iter()
                .find(|(name, _)| name == DEFAULT_SPEAKER)
                .or_else(|| spk_map.first())
                .cloned()
                .expect("non-empty spk_map");
            return self.codec_row(&name, id);
        }
        self.named_profile(voice)
    }
}

impl Qwen3TtsAudioSpeech {
    fn role_prefix_ids(&self) -> Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode_text("<|im_start|>assistant\n")
            .context("encode role prefix")?;
        if ids.len() != 3 {
            anyhow::bail!(
                "role prefix tokenized to {} ids ({:?}), expected 3",
                ids.len(),
                ids
            );
        }
        Ok(ids)
    }

    pub fn resolver(&self) -> Option<SpeakerResolver> {
        Some(SpeakerResolver {
            talker: self.talker.clone()?,
            store: self.voice_profiles.clone(),
            profiles_supported: self.profiles_supported,
            cache: self.default_voice.clone(),
        })
    }

    fn resolve_speaker_embed(&self, voice: &str) -> Result<(Option<candle_core::Tensor>, String)> {
        match self.resolver() {
            Some(r) => r.resolve(voice),
            None => Ok((None, "none".to_string())),
        }
    }

    fn resolve_language_id(&self) -> Option<u32> {
        let talker = self.talker.as_ref()?;
        let lang = std::env::var(ENV_LANGUAGE).ok()?.to_lowercase();
        if lang.is_empty() || lang == "auto" {
            return None;
        }
        talker
            .config()
            .language_id
            .iter()
            .find(|(name, _)| *name == lang)
            .map(|(_, id)| *id)
    }
}

fn stream_geometry() -> (usize, usize) {
    let chunk = std::env::var(ENV_STREAM_CHUNK_FRAMES)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 2)
        .unwrap_or(STREAM_CHUNK_FRAMES);
    let ctx = std::env::var(ENV_STREAM_LEFT_CONTEXT)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(STREAM_LEFT_CONTEXT_FRAMES);
    (chunk, ctx)
}

fn build_samplers() -> (Sampler, Sampler) {
    let greedy = std::env::var(ENV_GREEDY).map(|v| v == "1").unwrap_or(false);
    let seed = std::env::var(ENV_SEED)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        });
    let talker_sampler = Sampler::new(SamplerConfig {
        do_sample: !greedy,
        temperature: 0.9,
        top_k: 50,
        top_p: 1.0,
        repetition_penalty: 1.05,
        seed,
    });
    let sub_sampler = Sampler::new(SamplerConfig {
        do_sample: !greedy,
        temperature: 0.9,
        top_k: 50,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: seed.rotate_left(17) ^ 0xD1B5_4A32_D192_ED03,
    });
    (talker_sampler, sub_sampler)
}

fn load_talker_and_code_predictor(
    dir: &Path,
    device: &Device,
) -> (Option<Qwen3TtsTalker>, Option<Qwen3TtsCodecDecoder>) {
    let shard = dir.join("model.safetensors");
    if !shard.is_file() {
        info!(path = %shard.display(), "no model.safetensors; talker/code_predictor disabled");
        return (None, None);
    }
    let weights = match nv_weights::WeightLoader::open_file(&shard, device) {
        Ok(w) => w,
        Err(err) => {
            warn!(error = %err, path = %shard.display(), "open model.safetensors failed");
            return (None, None);
        }
    };

    let config_path = dir.join("config.json");
    let talker_cfg = match nv_tts::Qwen3TtsTalkerConfig::from_hf_config_file(&config_path) {
        Ok(mut c) => {
            c.dtype = DType::F32;
            c
        }
        Err(err) => {
            warn!(error = %err, "parse talker_config from config.json failed");
            return (None, None);
        }
    };

    let mut talker = match Qwen3TtsTalker::new(talker_cfg, device) {
        Ok(t) => t,
        Err(err) => {
            warn!(error = %err, "construct talker failed");
            return (None, None);
        }
    };
    if let Err(err) = talker.load_weights(&weights) {
        warn!(error = %err, "talker load_weights failed");
        return (None, None);
    }
    if !talker.has_text_embedding() {
        warn!("talker loaded but text_embedding absent; generation will use LCG fallback");
    }

    let cp_cfg = match nv_tts::CodecDecoderConfig::from_hf_config_file(&config_path) {
        Ok(mut c) => {
            c.dtype = DType::F32;
            c
        }
        Err(err) => {
            warn!(error = %err, "parse code_predictor_config failed; talker-only mode");
            return (Some(talker), None);
        }
    };
    let mut cp = match Qwen3TtsCodecDecoder::new(cp_cfg, device) {
        Ok(c) => c,
        Err(err) => {
            warn!(error = %err, "construct code_predictor failed");
            return (Some(talker), None);
        }
    };
    if let Err(err) = cp.load_weights(&weights) {
        warn!(error = %err, "code_predictor load_weights failed");
        return (Some(talker), None);
    }
    (Some(talker), Some(cp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_tts::qwen3_tts_cache_dir;

    fn cached_or_skip() -> Option<PathBuf> {
        qwen3_tts_cache_dir()
    }

    #[test]
    fn from_dirs_loads_real_tokenizer_and_real_vocoder() {
        let Some(dir) = cached_or_skip() else {
            eprintln!("skip from_dirs_loads_real_tokenizer_and_real_vocoder: cache absent");
            return;
        };
        let svc = Qwen3TtsAudioSpeech::from_dirs(&dir, None)
            .expect("nv-tts bootstrap should succeed against cached snapshot");
        assert_eq!(svc.sample_rate, NV_TTS_SAMPLE_RATE);
        assert!(svc.vocoder_inventory.is_real_qwen3_decoder());
        assert!(
            !svc.vocoder_report.zero_init_fallback,
            "real Qwen3-TTS vocoder weights should load without fallback: {:?}",
            svc.vocoder_report.fallback_reason
        );
        assert!(!svc.vocoder.is_zero_init());
        assert!(
            svc.talker.is_some(),
            "talker should be loaded from model.safetensors"
        );
        assert!(
            svc.code_predictor.is_some(),
            "code_predictor should be loaded from model.safetensors"
        );
        let talker = svc.talker.as_ref().unwrap();
        assert!(
            talker.has_text_embedding(),
            "talker.model.text_embedding.weight should be loaded"
        );
    }

    #[test]
    fn synthesize_emits_non_silent_pcm_when_vocoder_is_real() {
        let Some(dir) = cached_or_skip() else {
            eprintln!("skip synthesize_emits_non_silent_pcm_when_vocoder_is_real: cache absent");
            return;
        };
        let svc = Qwen3TtsAudioSpeech::from_dirs(&dir, None).expect("bootstrap");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let collected = rt.block_on(async {
            let mut rx = svc
                .synthesize("Hello world from the test harness", "default")
                .await
                .expect("synthesize");
            let mut all = Vec::new();
            while let Some(chunk) = rx.recv().await {
                all.extend_from_slice(&chunk);
            }
            all
        });

        assert!(
            !collected.is_empty(),
            "must emit at least the min-silence buffer"
        );

        let min_samples = (NV_TTS_SAMPLE_RATE as f32 * MIN_SILENCE_SECONDS) as usize;
        assert!(
            collected.len() >= min_samples,
            "got {} < {}",
            collected.len(),
            min_samples
        );

        let max_abs = collected.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max_abs > 0.0,
            "real-vocoder synthesize must produce non-silent PCM; got max_abs={max_abs}"
        );
    }

    #[test]
    fn profile_resolution_rules_on_real_checkpoint() {
        use nv_tts::VoiceProfile;

        let Some(dir) = cached_or_skip() else {
            eprintln!("skip profile_resolution_rules_on_real_checkpoint: cache absent");
            return;
        };
        let mut svc = Qwen3TtsAudioSpeech::from_dirs(&dir, None).expect("bootstrap");
        let talker = svc.talker.clone().expect("talker");
        let hidden = talker.config().hidden_size;
        let has_speakers = !talker.config().spk_id.is_empty();
        assert_eq!(
            svc.profiles_supported, !has_speakers,
            "Base ships no speakers and supports profiles; CustomVoice the opposite"
        );

        let (tmp, store) = temp_profile_store("vp_rules");
        for (name, schema, emb) in [
            ("p_good", 2, vec![0.1; hidden]),
            ("p_short", 2, vec![0.1; 8]),
            ("p_zero", 1, vec![0.0; hidden]),
        ] {
            store
                .put(&VoiceProfile {
                    schema_version: schema,
                    name: name.into(),
                    embedding: emb,
                    design_params: None,
                })
                .unwrap();
        }
        svc.voice_profiles = Arc::new(VoiceStoreHandle::new(&tmp));

        let err = svc
            .resolve_speaker_embed("no_such_voice")
            .expect_err("unknown voice must be refused, not silently substituted");
        let msg = format!("{err:#}");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{msg}");
        assert!(msg.contains("no_such_voice"), "{msg}");

        if has_speakers {
            let err = svc
                .resolve_speaker_embed("p_good")
                .expect_err("profiles are not usable on a CustomVoice checkpoint");
            assert!(format!("{err:#}").contains("tts_model_type"), "{err:#}");
            let (emb, name) = svc.resolve_speaker_embed("serena").expect("named speaker");
            assert_eq!(name, "serena");
            assert!(emb.is_some());
            for alias in ["default", "alloy", ""] {
                let (_, name) = svc.resolve_speaker_embed(alias).expect("alias");
                assert_eq!(name, DEFAULT_SPEAKER, "{alias:?}");
            }
        } else {
            let (emb, name) = svc.resolve_speaker_embed("p_good").expect("profile embed");
            assert_eq!(name, "profile:p_good");
            assert_eq!(emb.expect("tensor").dims(), &[1usize, 1, hidden]);
            for bad in ["p_short", "p_zero"] {
                let err = svc.resolve_speaker_embed(bad).expect_err("must reject");
                assert!(
                    err.downcast_ref::<InvalidVoiceRequest>().is_some(),
                    "{bad}: expected InvalidVoiceRequest, got: {err:#}"
                );
            }
            for alias in ["default", "alloy", ""] {
                let (_, name) = svc.resolve_speaker_embed(alias).expect("alias");
                assert_eq!(name, "builtin:default", "{alias:?}");
            }
        }

        drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn fake_talker(hidden: usize, spk_id: Vec<(String, u32)>) -> Arc<Qwen3TtsTalker> {
        let cfg = nv_tts::Qwen3TtsTalkerConfig {
            hidden_size: hidden,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            speech_vocab_size: 16,
            text_vocab_size: 64,
            text_hidden_size: 16,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            mrope_section: vec![2, 1, 1],
            dtype: DType::F32,
            spk_id,
            language_id: Vec::new(),
        };
        Arc::new(Qwen3TtsTalker::new(cfg, &Device::Cpu).expect("build tiny talker"))
    }

    fn temp_profile_store(tag: &str) -> (PathBuf, nv_tts::VoiceProfileStore) {
        let tmp = std::env::temp_dir().join(format!(
            "nvtts_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = nv_tts::VoiceProfileStore::open(&tmp).expect("open temp store");
        (tmp, store)
    }

    fn resolver(
        talker: &Arc<Qwen3TtsTalker>,
        root: &std::path::Path,
        profiles_supported: bool,
    ) -> SpeakerResolver {
        SpeakerResolver {
            talker: talker.clone(),
            store: Arc::new(VoiceStoreHandle::new(root)),
            profiles_supported,
            cache: Arc::new(DefaultVoiceCache::default()),
        }
    }

    fn row(emb: Option<candle_core::Tensor>) -> Vec<f32> {
        emb.expect("embed")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    #[test]
    fn builtin_default_xvector_matches_the_0_6b_talker_width() {
        let xvec = builtin_default_xvector(1024).expect("shipped for the 0.6B family");
        assert_eq!(xvec.len(), 1024);
        assert!(xvec.iter().any(|x| *x != 0.0));
        assert!(builtin_default_xvector(2048).is_none());
        assert!(builtin_default_xvector(32).is_none());
    }

    #[test]
    fn base_talker_aliases_use_the_default_profile_then_the_builtin_voice() {
        use nv_tts::VoiceProfile;

        let talker = fake_talker(1024, Vec::new());
        let (tmp, store) = temp_profile_store("vp_alias");
        let r = resolver(&talker, &tmp, true);
        let builtin = builtin_default_xvector(1024).unwrap();

        for alias in ["alloy", "shimmer", "default", "", "Echo"] {
            let (emb, name) = r
                .resolve(alias)
                .unwrap_or_else(|e| panic!("{alias:?} must resolve without any enrollment: {e:#}"));
            assert_eq!(name, "builtin:default", "{alias:?}");
            assert_eq!(row(emb), builtin, "{alias:?}");
        }

        let err = r
            .resolve("nope")
            .expect_err("unknown non-alias voice is refused");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{err:#}");
        assert!(format!("{err:#}").contains("nope"));

        store
            .put(&VoiceProfile {
                schema_version: 2,
                name: "mine".into(),
                embedding: vec![0.25; 1024],
                design_params: None,
            })
            .unwrap();
        let (emb, name) = r.resolve("mine").expect("profile");
        assert_eq!(name, "profile:mine");
        assert_eq!(emb.expect("embed").dims(), &[1usize, 1, 1024]);
        let (_, name) = r.resolve("alloy").expect("alias");
        assert_eq!(
            name, "builtin:default",
            "a profile that is not named default does not capture the aliases"
        );

        store
            .put(&VoiceProfile {
                schema_version: 2,
                name: DEFAULT_PROFILE.into(),
                embedding: vec![0.5; 1024],
                design_params: None,
            })
            .unwrap();
        for alias in ["alloy", "echo", "default", ""] {
            let (emb, name) = r.resolve(alias).expect("routed");
            assert_eq!(name, format!("profile:{DEFAULT_PROFILE}"), "{alias:?}");
            assert!(
                row(emb).iter().all(|x| (*x - 0.5).abs() < 1e-6),
                "{alias:?}"
            );
        }
        let (_, name) = r.resolve("mine").expect("profile");
        assert_eq!(
            name, "profile:mine",
            "a named profile still wins over the default"
        );

        store.delete(DEFAULT_PROFILE).unwrap();
        let (emb, name) = r.resolve("alloy").expect("alias");
        assert_eq!(
            name, "builtin:default",
            "deleting the profile falls back again"
        );
        assert_eq!(row(emb), builtin);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn base_talker_without_a_matching_builtin_refuses_aliases() {
        let talker = fake_talker(32, Vec::new());
        let (tmp, _store) = temp_profile_store("vp_narrow");
        let r = resolver(&talker, &tmp, true);
        let err = r.resolve("alloy").expect_err("no 32-d default exists");
        let msg = format!("{err:#}");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{msg}");
        assert!(
            msg.contains(DEFAULT_PROFILE) && msg.contains("32-d"),
            "{msg}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn non_base_talker_without_speakers_refuses_aliases_and_profiles() {
        let talker = fake_talker(1024, Vec::new());
        let (tmp, _store) = temp_profile_store("vp_design");
        let r = resolver(&talker, &tmp, false);
        let err = r.resolve("alloy").expect_err("no conditioning pathway");
        let msg = format!("{err:#}");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{msg}");
        assert!(msg.contains("tts_model_type"), "{msg}");
        let err = r.resolve("mine").expect_err("profiles are not usable");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{err:#}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn custom_voice_talker_never_touches_the_store() {
        use nv_tts::VoiceProfile;

        let talker = fake_talker(32, vec![("serena".to_string(), 0), ("ryan".to_string(), 1)]);
        let (tmp, store) = temp_profile_store("vp_custom");
        store
            .put(&VoiceProfile {
                schema_version: 2,
                name: DEFAULT_PROFILE.into(),
                embedding: vec![0.5; 32],
                design_params: None,
            })
            .unwrap();
        let r = resolver(&talker, &tmp, false);
        for alias in ["default", "alloy", ""] {
            let (emb, name) = r.resolve(alias).expect("alias");
            assert_eq!(name, "serena", "{alias:?}");
            assert!(emb.is_some());
        }
        let (_, name) = r.resolve("Ryan").expect("named");
        assert_eq!(name, "ryan");
        let err = r.resolve("mine").expect_err("no profiles on CustomVoice");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{err:#}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn unreachable_store_degrades_to_the_builtin_voice_and_503s_named_profiles() {
        let talker = fake_talker(1024, Vec::new());
        let blocker = std::env::temp_dir().join(format!(
            "nvtts_vp_blocked_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&blocker, b"not a directory").unwrap();
        let root = blocker.join("voices");
        let r = SpeakerResolver {
            store: Arc::new(
                VoiceStoreHandle::new(&root).with_reopen_backoff(std::time::Duration::ZERO),
            ),
            ..resolver(&talker, &root, true)
        };
        assert!(!r.store.is_live());

        let (emb, name) = r.resolve("alloy").expect("aliases keep working");
        assert_eq!(name, "builtin:default");
        assert_eq!(row(emb), builtin_default_xvector(1024).unwrap());

        let err = r
            .resolve("mine")
            .expect_err("a named profile needs the store");
        assert!(nv_tts::is_store_unavailable(&err), "{err:#}");
        assert!(err.downcast_ref::<UnknownVoice>().is_none());

        std::fs::remove_file(&blocker).unwrap();
        assert!(
            r.store.is_live(),
            "the handle reopens once the root is creatable"
        );
        let err = r.resolve("mine").expect_err("now a plain unknown voice");
        assert!(err.downcast_ref::<UnknownVoice>().is_some(), "{err:#}");
        let _ = std::fs::remove_dir_all(&blocker);
    }

    #[test]
    #[ignore = "needs the real Qwen3-TTS Base checkpoint (NV_TTS_BASE_TALKER_DIR or the HF cache)"]
    fn real_base_checkpoint_serves_aliases_from_the_builtin_then_the_default_profile() {
        use nv_tts::VoiceProfile;

        let dir = nv_tts::qwen3_tts_base_dir().expect("NV_TTS_BASE_TALKER_DIR or the HF cache");
        let mut svc = Qwen3TtsAudioSpeech::from_dirs(&dir, None).expect("bootstrap on Base");
        assert!(
            svc.profiles_supported,
            "Base must report profiles_supported"
        );
        let talker = svc.talker.as_ref().expect("talker");
        assert!(
            talker.config().spk_id.is_empty(),
            "Base ships spk_id = {{}}"
        );
        assert!(svc.voices().is_empty());
        let hidden = talker.config().hidden_size;
        assert!(
            builtin_default_xvector(hidden).is_some(),
            "the shipped default x-vector must match the Base talker width {hidden}"
        );

        let (tmp, store) = temp_profile_store("vp_base_alias");
        svc.voice_profiles = Arc::new(VoiceStoreHandle::new(&tmp));
        let (emb, name) = svc.resolve_speaker_embed("alloy").expect("builtin default");
        assert_eq!(name, "builtin:default");
        assert_eq!(emb.expect("embed").dims(), &[1usize, 1, hidden]);

        store
            .put(&VoiceProfile {
                schema_version: 2,
                name: DEFAULT_PROFILE.into(),
                embedding: vec![0.01; hidden],
                design_params: None,
            })
            .unwrap();
        let (emb, name) = svc.resolve_speaker_embed("alloy").expect("routed");
        assert_eq!(name, format!("profile:{DEFAULT_PROFILE}"));
        assert_eq!(emb.expect("embed").dims(), &[1usize, 1, hidden]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn write_decoderless_vocoder_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nvtts_zeroinit_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("speech_tokenizer")).expect("mkdir");
        let header = br#"{"junk":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        std::fs::write(dir.join("speech_tokenizer/model.safetensors"), blob).expect("write shard");
        dir
    }

    #[test]
    fn zero_init_vocoder_is_refused_by_default_and_served_only_on_opt_in() {
        let Some(talker) = cached_or_skip() else {
            eprintln!("skip zero_init_vocoder_is_refused_by_default_and_served_only_on_opt_in: cache absent");
            return;
        };
        let fake = write_decoderless_vocoder_dir();

        let err = match Qwen3TtsAudioSpeech::from_dirs_with(&talker, Some(&fake), false) {
            Ok(_) => panic!("a zero-init vocoder must refuse to boot by default"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains(ENV_VOCODER_DIR), "{msg}");
        assert!(msg.contains(ENV_ALLOW_SILENT_VOCODER), "{msg}");
        assert!(msg.contains("SILENCE"), "{msg}");

        let mut opted_in = Qwen3TtsAudioSpeech::from_dirs_with(&talker, Some(&fake), true)
            .expect("the opt-in must still boot");
        assert!(opted_in.vocoder.is_zero_init());
        assert!(opted_in.vocoder_report.zero_init_fallback);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let silent = rt.block_on(async {
            let mut rx = opted_in
                .synthesize("hello", "default")
                .await
                .expect("opted-in synthesize still emits silence");
            let mut all = Vec::new();
            while let Some(c) = rx.recv().await {
                all.extend_from_slice(&c);
            }
            all
        });
        assert!(!silent.is_empty());
        assert!(silent.iter().all(|s| *s == 0.0), "opt-in path is silence");

        opted_in.allow_silent_vocoder = false;
        let err = rt
            .block_on(opted_in.synthesize("hello", "default"))
            .expect_err("a zero-init engine must refuse the request");
        assert!(
            err.downcast_ref::<crate::oapi::audio_speech::SilentVocoder>()
                .is_some(),
            "expected SilentVocoder, got {err:#}"
        );

        let _ = std::fs::remove_dir_all(&fake);
    }

    #[test]
    fn real_vocoder_does_not_trip_the_zero_init_gate() {
        let Some(talker) = cached_or_skip() else {
            eprintln!("skip real_vocoder_does_not_trip_the_zero_init_gate: cache absent");
            return;
        };
        let svc = Qwen3TtsAudioSpeech::from_dirs_with(&talker, None, false)
            .expect("real weights must boot with the gate armed");
        assert!(!svc.vocoder.is_zero_init());
        assert!(!svc.vocoder_report.zero_init_fallback);
        assert_eq!(
            svc.model_id().as_deref(),
            Some(svc.talker_model_id().as_str())
        );
    }

    #[test]
    fn allow_silent_vocoder_parses_like_the_other_required_flags() {
        assert!(!allow_silent_vocoder(None));
        assert!(!allow_silent_vocoder(Some("")));
        assert!(!allow_silent_vocoder(Some("0")));
        assert!(!allow_silent_vocoder(Some("false")));
        assert!(allow_silent_vocoder(Some("1")));
        assert!(allow_silent_vocoder(Some("yes")));
    }

    #[test]
    fn from_dirs_errors_when_talker_dir_missing() {
        let bogus = PathBuf::from("/tmp/__no_such_qwen3_tts_dir__");
        let err = Qwen3TtsAudioSpeech::from_dirs(&bogus, None)
            .err()
            .expect("must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("NV_TTS_TALKER_DIR") || msg.contains("not a directory"),
            "{msg}"
        );
    }
}
