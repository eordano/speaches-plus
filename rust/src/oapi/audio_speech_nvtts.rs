use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use tokio::sync::mpsc;
use tracing::{info, warn};

use nv_omni::vocoder::{Vocoder, NUM_CODEBOOKS, STREAM_CHUNK_FRAMES, STREAM_LEFT_CONTEXT_FRAMES};
use nv_tts::{LinearQuant, TalkerPrecision};
use nv_tts::{
    Qwen3TtsCodecDecoder, Qwen3TtsTalker, Qwen3TtsTokenizer, Sampler, SamplerConfig,
    VocoderInventory, VocoderLoadReport, VoiceStoreHandle,
};

use crate::oapi::audio_speech::{
    AudioSpeech, InvalidSpeechParam, InvalidVoiceRequest, SpeechOptions, UnknownVoice,
    NV_TTS_SAMPLE_RATE,
};
use crate::oapi::cold_tier::ColdSlot;
use crate::oapi::voice_profiles::voice_profile_root;

pub const ENV_TALKER_DIR: &str = "NV_TTS_TALKER_DIR";

pub const ENV_VOCODER_DIR: &str = "NV_TTS_VOCODER_DIR";

pub const MIN_SILENCE_SECONDS: f32 = 0.20;

pub const MAX_CODEC_STEPS: usize = 2048;

pub const MIN_CODEC_STEPS: usize = 2;

pub const ENV_STREAM_CHUNK_FRAMES: &str = "NV_TTS_STREAM_CHUNK_FRAMES";

pub const ENV_STREAM_LEFT_CONTEXT: &str = "NV_TTS_STREAM_LEFT_CONTEXT";

pub const ENV_LANGUAGE: &str = "NV_TTS_LANG";

pub const ENV_SEED: &str = "NV_TTS_SEED";

pub const ENV_GREEDY: &str = "NV_TTS_GREEDY";

pub const DEFAULT_SPEAKER: &str = "serena";

pub const DEFAULT_PROFILE: &str = "default";

pub const ENV_VOICE_DESIGN_DIR: &str = "NV_TTS_VOICE_DESIGN_DIR";

pub const VOICE_DESIGN_SERVED_NAME: &str = "qwen3-tts-voicedesign";

pub const VOICE_DESIGN_MODEL_TYPE: &str = "voice_design";

pub const VOICE_DESIGN_ENGINE: &str = "nv-tts (Qwen3-TTS VoiceDesign)";

pub const VOICE_DESIGN_CUDA_PRECISION_BF16_HALVES_VRAM_AT_EQUAL_WER_F0_AND_CLONE_SIMILARITY:
    TalkerPrecision = TalkerPrecision::Bf16;

pub const BASE_TALKER_CUDA_PRECISION_FP8_HOLDS_F32S_WER_PITCH_AND_CLONE_LIKENESS_AT_THREE_FIFTHS_OF_ITS_PEAK:
    TalkerPrecision = TalkerPrecision::Fp8E4m3;

pub const VOICE_DESIGN_RESIDENT_BYTES: u64 =
    crate::oapi::vram_plan::AUX_QWEN3_TTS_1P7B_VOICEDESIGN_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_ONE_STREAM
        * crate::oapi::vram_plan::MIB;

pub const ISO_639_1_TO_QWEN3_TTS_LANGUAGE: [(&str, &str); 10] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("fr", "french"),
    ("it", "italian"),
    ("ja", "japanese"),
    ("ko", "korean"),
    ("pt", "portuguese"),
    ("ru", "russian"),
];

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
    pub voice_design: bool,
    pub precision: TalkerPrecision,
    pub default_voice: Arc<DefaultVoiceCache>,

    pub sample_rate: u32,
}

pub fn zero_init_vocoder_message(vocoder_dir: &Path, source: &str, reason: Option<&str>) -> String {
    format!(
        "refusing to serve nv-tts with a zero-initialised vocoder: loaded from {} ({source}); \
         {}. A zero-init vocoder decodes every frame to silence, so /v1/audio/speech would \
         answer 200 with a correctly-sized WAV of PURE SILENCE -- status, length and \
         content-type all indistinguishable from success. Point {} at a Qwen3-TTS checkpoint \
         whose speech_tokenizer/model.safetensors carries the decoder.* tensors.",
        vocoder_dir.display(),
        reason.unwrap_or("no reason reported by the loader"),
        ENV_VOCODER_DIR,
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
        Self::from_dirs_with_precision(talker_dir, vocoder_dir_override, None)
    }

    pub fn from_dirs_with_precision(
        talker_dir: &Path,
        vocoder_dir_override: Option<&Path>,
        requested: Option<TalkerPrecision>,
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
        let device = qwen3_tts_device();
        let (voc, report) =
            nv_tts::load_vocoder_from_qwen3_tts(&vocoder_dir_owned, &device, DType::F32)
                .with_context(|| {
                    format!("load vocoder weights from {}", vocoder_dir_owned.display())
                })?;
        if report.zero_init_fallback {
            anyhow::bail!(
                "{}",
                zero_init_vocoder_message(
                    &vocoder_dir_owned,
                    vocoder_dir_source,
                    report.fallback_reason.as_deref(),
                )
            );
        } else if let Some(reason) = &report.fallback_reason {
            warn!(reason = %reason, "vocoder load reported a fallback");
        }

        let model_type = tts_model_type(talker_dir);
        let voice_design = model_type.as_deref() == Some(VOICE_DESIGN_MODEL_TYPE);
        let precision = resolve_precision(voice_design, &device, requested)?;
        let (talker, code_predictor) =
            load_talker_and_code_predictor(talker_dir, &device, precision);
        let has_talker = talker.is_some();
        let has_cp = code_predictor.is_some();

        info!(
            talker_dir = %talker_dir.display(),
            vocoder_dir = %vocoder_dir_owned.display(),
            decoder_keys = report.inventory.decoder_key_count,
            sample_rate = report.inventory.sample_rate,
            has_talker,
            has_code_predictor = has_cp,
            precision = precision.label(),
            "nv-tts AudioSpeech ready"
        );
        let inventory = report.inventory.clone();
        let voice_profiles = Arc::new(VoiceStoreHandle::new(voice_profile_root()));
        let profiles_supported = model_type.as_deref() == Some("base");

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
            voice_design,
            precision,
            default_voice: Arc::new(DefaultVoiceCache::default()),
            sample_rate: NV_TTS_SAMPLE_RATE,
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

pub fn qwen3_tts_device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

fn resolve_precision(
    voice_design: bool,
    device: &Device,
    requested: Option<TalkerPrecision>,
) -> Result<TalkerPrecision> {
    if voice_design {
        return Ok(if device.is_cuda() {
            VOICE_DESIGN_CUDA_PRECISION_BF16_HALVES_VRAM_AT_EQUAL_WER_F0_AND_CLONE_SIMILARITY
        } else {
            TalkerPrecision::F32
        });
    }
    if !device.is_cuda() {
        if let Some(p) = requested.filter(|p| p.needs_cuda()) {
            warn!(requested = p.label(), "needs CUDA; the CPU talker stays f32");
        }
        return Ok(TalkerPrecision::F32);
    }
    Ok(requested.unwrap_or(
        BASE_TALKER_CUDA_PRECISION_FP8_HOLDS_F32S_WER_PITCH_AND_CLONE_LIKENESS_AT_THREE_FIFTHS_OF_ITS_PEAK,
    ))
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
    async fn synthesize(
        &self,
        text: &str,
        voice: &str,
        opts: &SpeechOptions,
    ) -> Result<mpsc::Receiver<Result<Vec<f32>>>> {
        let token_ids = self
            .tokenizer
            .encode_text(text)
            .context("qwen3-tts tokenizer could not encode the input")?;
        if token_ids.is_empty() {
            anyhow::bail!("qwen3-tts tokenizer produced no tokens for a non-empty input");
        }
        let token_count = token_ids.len();

        let (tx, rx) = mpsc::channel::<Result<Vec<f32>>>(4);

        if let (Some(talker), Some(cp)) = (self.talker.clone(), self.code_predictor.clone()) {
            if talker.has_text_embedding() {
                let vocoder = self.vocoder.clone();
                let role_ids = self.role_prefix_ids()?;
                let (speaker_embed, speaker_name, instruct_ids) = if self.voice_design {
                    let instruct = voice_design_instructions(opts)?;
                    let ids = self
                        .tokenizer
                        .encode_text(&voice_design_prompt(instruct))
                        .context("encode voice design instructions")?;
                    (None, "designed".to_string(), ids)
                } else {
                    let (embed, name) = match self.resolver() {
                        Some(r) => {
                            let voice = voice.to_string();
                            tokio::task::spawn_blocking(move || r.resolve(&voice))
                                .await
                                .context("speaker resolution task")??
                        }
                        None => (None, "none".to_string()),
                    };
                    (embed, name, Vec::new())
                };
                let language_id = match opts.language.as_deref() {
                    Some(requested) => language_id(&talker.config().language_id, requested)?,
                    None => self.resolve_language_id(),
                };
                let seed = request_seed(self.voice_design, opts, voice);
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
                        &instruct_ids,
                        &role_ids,
                        &token_ids,
                        speaker_embed.as_ref(),
                        language_id,
                        tk.bos_id(),
                        tk.eos_id(),
                        tk.special_id(nv_tts::tokenizer::SPECIAL_TTS_PAD)
                            .unwrap_or(nv_tts::tokenizer::SPECIAL_ID_TTS_PAD),
                    )?;
                    let (mut talker_sampler, mut sub_sampler) = build_samplers(seed);
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
                    let mut ok = send_tx.send(Ok(first)).await.is_ok();
                    while ok {
                        match pcm_rx.recv().await {
                            Some(chunk) => ok = send_tx.send(Ok(chunk)).await.is_ok(),
                            None => break,
                        }
                    }
                    drop(pcm_rx);
                    let gen_res = match gen_task.await {
                        Ok(res) => res.map(|_| ()),
                        Err(err) => Err(anyhow::anyhow!("talker generation task panicked: {err}")),
                    };
                    let dec_res = match dec_task.await {
                        Ok(res) => res,
                        Err(err) => Err(anyhow::anyhow!("vocoder decode task panicked: {err}")),
                    };
                    if let Err(err) = gen_res.and(dec_res) {
                        warn!(
                            error = %format!("{err:#}"),
                            "qwen3-tts synthesis failed mid-stream; the response fails instead of ending short"
                        );
                        let _ = send_tx.send(Err(err)).await;
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

    fn engine(&self) -> &'static str {
        if self.voice_design {
            VOICE_DESIGN_ENGINE
        } else {
            "nv-tts (Qwen3-TTS)"
        }
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

fn profile_tensor(
    profile: nv_tts::VoiceProfile,
    hidden: usize,
    device: &Device,
) -> Result<candle_core::Tensor> {
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
    candle_core::Tensor::from_vec(profile.embedding, (1usize, 1usize, hidden), device)
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
        let tensor = profile_tensor(
            profile,
            self.talker.config().hidden_size,
            self.talker.device(),
        )?;
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
                profile_tensor(profile, hidden, self.talker.device())
            })?;
            return Ok((Some(tensor), format!("profile:{DEFAULT_PROFILE}")));
        }
        if let Some(xvec) = builtin_default_xvector(hidden) {
            let tensor = self.cache.get_or_build(DefaultVoiceSource::Builtin, || {
                candle_core::Tensor::from_vec(
                    xvec.to_vec(),
                    (1usize, 1usize, hidden),
                    self.talker.device(),
                )
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

    #[cfg(test)]
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

pub fn voice_design_prompt(instructions: &str) -> String {
    format!("<|im_start|>user\n{instructions}<|im_end|>\n")
}

fn voice_design_instructions(opts: &SpeechOptions) -> Result<&str, InvalidSpeechParam> {
    opts.instructions().ok_or_else(|| InvalidSpeechParam {
        param: "instructions",
        code: "missing_instructions",
        message: format!(
            "the VoiceDesign talker ({VOICE_DESIGN_SERVED_NAME}) builds its voice from \
             `instructions`, a description in words such as \"a warm, mature male narrator\"; \
             the request has none, and the checkpoint has no built-in speakers to fall back to"
        ),
    })
}

pub fn request_seed(voice_design: bool, opts: &SpeechOptions, voice: &str) -> Option<u64> {
    let voice_names_the_draw = || voice.trim().parse::<u64>().ok();
    opts.seed
        .or_else(|| voice_design.then(voice_names_the_draw).flatten())
}

pub fn language_id(
    languages: &[(String, u32)],
    requested: &str,
) -> Result<Option<u32>, InvalidSpeechParam> {
    let lower = requested.trim().to_lowercase();
    if lower.is_empty() || lower == "auto" {
        return Ok(None);
    }
    let primary = lower.split(['-', '_']).next().unwrap_or_default();
    let name = ISO_639_1_TO_QWEN3_TTS_LANGUAGE
        .iter()
        .find(|(iso, _)| *iso == primary)
        .map_or(lower.as_str(), |(_, name)| name);
    match languages.iter().find(|(n, _)| n == name) {
        Some((_, id)) => Ok(Some(*id)),
        None => Err(InvalidSpeechParam {
            param: "language",
            code: "unsupported_language",
            message: format!(
                "language {requested:?} is not one this talker knows; send auto, an ISO 639-1 \
                 code, or one of: {}",
                languages
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}

pub fn routes_to_voice_design(
    requested: Option<&str>,
    voice_design_id: &str,
    base_talker_id: Option<&str>,
) -> bool {
    use crate::oapi::audio_speech::{model_id_matches, normalize_model_id};
    let Some(req) = requested.map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };
    if base_talker_id.is_some_and(|id| model_id_matches(req, id)) {
        return false;
    }
    model_id_matches(req, voice_design_id)
        || normalize_model_id(req) == normalize_model_id(VOICE_DESIGN_SERVED_NAME)
}

pub struct VoiceDesign {
    pub model_id: String,
    languages: Vec<(String, u32)>,
    slot: Arc<ColdSlot<Qwen3TtsAudioSpeech>>,
}

static VOICE_DESIGN: OnceLock<Option<VoiceDesign>> = OnceLock::new();

pub fn voice_design() -> Option<&'static VoiceDesign> {
    VOICE_DESIGN.get_or_init(VoiceDesign::from_env).as_ref()
}

impl VoiceDesign {
    fn from_env() -> Option<Self> {
        let dir = PathBuf::from(std::env::var_os(ENV_VOICE_DESIGN_DIR)?);
        let model_type = tts_model_type(&dir);
        let languages = match nv_tts::Qwen3TtsTalkerConfig::from_hf_config_file(
            &dir.join("config.json"),
        ) {
            Ok(cfg) if model_type.as_deref() == Some(VOICE_DESIGN_MODEL_TYPE) => cfg.language_id,
            other => {
                tracing::error!(
                    dir = %dir.display(),
                    tts_model_type = ?model_type,
                    config_error = ?other.err().map(|e| format!("{e:#}")),
                    "{ENV_VOICE_DESIGN_DIR} is not a readable Qwen3-TTS VoiceDesign checkpoint; \
                     voice design is disabled"
                );
                return None;
            }
        };
        let model_id = crate::oapi::model_ids::model_id_for_dir(&dir);
        info!(
            model = %model_id,
            dir = %dir.display(),
            "Qwen3-TTS VoiceDesign is a cold-tier resident: it loads on the first request that names it"
        );
        let slot = ColdSlot::demand(
            VOICE_DESIGN_SERVED_NAME,
            VOICE_DESIGN_RESIDENT_BYTES,
            Arc::new(move || load_voice_design(&dir)),
        );
        Some(Self {
            model_id,
            languages,
            slot,
        })
    }

    pub fn serves(&self, requested: Option<&str>, base_talker_id: Option<&str>) -> bool {
        routes_to_voice_design(requested, &self.model_id, base_talker_id)
    }

    pub fn validate(&self, opts: &SpeechOptions) -> Result<(), InvalidSpeechParam> {
        voice_design_instructions(opts)?;
        if let Some(requested) = opts.language.as_deref() {
            language_id(&self.languages, requested)?;
        }
        Ok(())
    }

    pub async fn resident(&self) -> Option<Arc<dyn AudioSpeech + Send + Sync>> {
        let engine = self.slot.get().await?;
        Some(engine)
    }

    pub fn unavailable_message(&self) -> String {
        const GIB: f64 = (1u64 << 30) as f64;
        let free = crate::oapi::cold_tier::device_free_bytes().map_or_else(
            || "an unknown amount".to_string(),
            |f| format!("{:.2} GiB", f as f64 / GIB),
        );
        format!(
            "the VoiceDesign talker {} is a cold-tier resident and cannot load here now: it needs \
             {:.2} GiB of free VRAM on top of chat's {:.2} GiB floor, and the device has {free} \
             free after evicting idle residents; retry later or on another gateway",
            self.model_id,
            VOICE_DESIGN_RESIDENT_BYTES as f64 / GIB,
            crate::oapi::cold_tier::chat_min_free_bytes() as f64 / GIB,
        )
    }
}

fn load_voice_design(dir: &Path) -> Option<Arc<Qwen3TtsAudioSpeech>> {
    use crate::oapi::cold_tier::{chat_min_free_bytes, device_free_bytes, ensure_free};
    let need = VOICE_DESIGN_RESIDENT_BYTES + chat_min_free_bytes();
    let report = ensure_free(
        need,
        VOICE_DESIGN_SERVED_NAME,
        Some(VOICE_DESIGN_SERVED_NAME),
    );
    if let Some(free) = report.free_after.or_else(device_free_bytes) {
        if free < need {
            warn!(
                free_mib = free / crate::oapi::vram_plan::MIB,
                need_mib = need / crate::oapi::vram_plan::MIB,
                "VoiceDesign talker not loaded: the device lacks its resident size on top of chat's floor"
            );
            return None;
        }
    }
    match Qwen3TtsAudioSpeech::from_dirs(dir, None) {
        Ok(engine) if engine.talker.is_some() && engine.code_predictor.is_some() => {
            Some(Arc::new(engine))
        }
        Ok(_) => {
            warn!(
                dir = %dir.display(),
                "VoiceDesign checkpoint loaded without its talker or code predictor; see the load \
                 warnings above"
            );
            None
        }
        Err(err) => {
            warn!(
                dir = %dir.display(),
                error = %format!("{err:#}"),
                "VoiceDesign talker failed to load"
            );
            None
        }
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

fn build_samplers(request_seed: Option<u64>) -> (Sampler, Sampler) {
    let greedy = std::env::var(ENV_GREEDY).map(|v| v == "1").unwrap_or(false);
    let seed = request_seed
        .or_else(|| {
            std::env::var(ENV_SEED)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
        })
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
    precision: TalkerPrecision,
) -> (Option<Qwen3TtsTalker>, Option<Qwen3TtsCodecDecoder>) {
    let shard = dir.join("model.safetensors");
    if !shard.is_file() {
        info!(path = %shard.display(), "no model.safetensors; talker/code_predictor disabled");
        return (None, None);
    }
    let dtype = precision.activation_dtype();
    let quant = match precision {
        TalkerPrecision::Fp8E4m3 => match LinearQuant::fp8_for(device) {
            Ok(q) => q,
            Err(err) => {
                warn!(error = %err, "fp8 talker setup failed; talker/code_predictor disabled");
                return (None, None);
            }
        },
        _ => LinearQuant::Dense,
    };
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
            c.dtype = dtype;
            c.quant = quant.clone();
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
            c.dtype = dtype;
            c.quant = quant;
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
    fn the_cached_checkpoint_boots_real_weights_synthesizes_audible_pcm_and_refuses_a_zero_init_vocoder() {
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
        {
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
                    .synthesize(
                        "Hello world from the test harness",
                        "default",
                        &SpeechOptions::default(),
                    )
                    .await
                    .expect("synthesize");
                let mut all = Vec::new();
                while let Some(chunk) = rx.recv().await {
                    all.extend_from_slice(&chunk.expect("no mid-stream failure"));
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
        {
            let Some(talker) = cached_or_skip() else {
                eprintln!("skip real_vocoder_does_not_trip_the_zero_init_gate: cache absent");
                return;
            };
            let svc = Qwen3TtsAudioSpeech::from_dirs(&talker, None)
                .expect("real weights must boot with the gate armed");
            assert!(!svc.vocoder.is_zero_init());
            assert!(!svc.vocoder_report.zero_init_fallback);
            assert_eq!(
                svc.model_id().as_deref(),
                Some(svc.talker_model_id().as_str())
            );
        }
        {
            let Some(talker) = cached_or_skip() else {
                eprintln!("skip zero_init_vocoder_is_refused_at_boot: cache absent");
                return;
            };
            let fake = write_decoderless_vocoder_dir();
            let err = match Qwen3TtsAudioSpeech::from_dirs(&talker, Some(&fake)) {
                Ok(_) => panic!("a zero-init vocoder must refuse to boot"),
                Err(e) => e,
            };
            let msg = format!("{err:#}");
            assert!(msg.contains(ENV_VOCODER_DIR), "{msg}");
            assert!(msg.contains("SILENCE"), "{msg}");
            let _ = std::fs::remove_dir_all(&fake);
        }
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
            quant: Default::default(),
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
    fn base_talker_aliases_use_the_default_profile_then_the_builtin_voice_of_the_talker_width() {
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
        {
            let xvec = builtin_default_xvector(1024).expect("shipped for the 0.6B family");
            assert_eq!(xvec.len(), 1024);
            assert!(xvec.iter().any(|x| *x != 0.0));
            assert!(builtin_default_xvector(2048).is_none());
            assert!(builtin_default_xvector(32).is_none());
        }
    }

    #[test]
    fn talkers_without_a_builtin_speakers_or_profiles_refuse_what_they_cannot_condition() {
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
        {
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
        {
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
    fn voice_design_routing_languages_and_the_instructions_requirement() {
        const VD: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
        const BASE: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
        for (req, base, want) in [
            (Some(VD), Some(BASE), true),
            (Some("Qwen3-TTS-12Hz-1.7B-VoiceDesign"), Some(BASE), true),
            (Some(VOICE_DESIGN_SERVED_NAME), Some(BASE), true),
            (Some("Qwen3_TTS_VoiceDesign"), None, true),
            (Some("qwen3-tts"), Some(BASE), false),
            (Some("qwen3-tts"), None, true),
            (Some(BASE), Some(BASE), false),
            (Some("kokoro"), Some(BASE), false),
            (Some("  "), Some(BASE), false),
            (None, None, false),
        ] {
            assert_eq!(
                routes_to_voice_design(req, VD, base),
                want,
                "{req:?} with {base:?}"
            );
        }
        let langs: Vec<(String, u32)> = vec![("english".into(), 2050), ("german".into(), 2053)];
        for (req, want) in [
            ("en", Some(2050)),
            ("en-US", Some(2050)),
            ("English", Some(2050)),
            ("de_DE", Some(2053)),
            ("auto", None),
            ("", None),
        ] {
            assert_eq!(language_id(&langs, req).unwrap(), want, "{req:?}");
        }
        let err = language_id(&langs, "fr").unwrap_err();
        assert_eq!(err.param, "language");
        assert!(err.message.contains("english, german"), "{}", err.message);
        for missing in [None, Some("   ".to_string())] {
            let opts = SpeechOptions {
                instructions: missing,
                ..SpeechOptions::default()
            };
            let err = voice_design_instructions(&opts).unwrap_err();
            assert_eq!(
                (err.param, err.code),
                ("instructions", "missing_instructions")
            );
        }
        let opts = SpeechOptions {
            instructions: Some(" a warm narrator ".into()),
            ..SpeechOptions::default()
        };
        assert_eq!(voice_design_instructions(&opts).unwrap(), "a warm narrator");
        let seeded = SpeechOptions {
            seed: Some(3),
            ..SpeechOptions::default()
        };
        for (vd, opts, voice, want) in [
            (true, &seeded, "7", Some(3)),
            (true, &opts, " 7 ", Some(7)),
            (true, &opts, "alloy", None),
            (false, &opts, "7", None),
            (false, &seeded, "alloy", Some(3)),
        ] {
            assert_eq!(request_seed(vd, opts, voice), want, "{vd} {voice:?}");
        }
    }

    const DESIGN_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../conformance/fixtures/030-voice-clone-design-warm/fixture.json"
    );

    fn wav_16bit(pcm: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: NV_TTS_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cur = std::io::Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
        for s in pcm {
            w.write_sample((s.clamp(-1.0, 1.0) * 32_767.0).round() as i16)
                .unwrap();
        }
        w.finalize().unwrap();
        cur.into_inner()
    }

    fn median_f0_hz(pcm: &[f32]) -> f32 {
        let sr = NV_TTS_SAMPLE_RATE as f32;
        let frame = (0.04 * sr) as usize;
        let (lo, hi) = ((sr / 400.0) as usize, (sr / 60.0) as usize);
        let mut f0s = Vec::new();
        for start in (0..pcm.len().saturating_sub(frame + hi)).step_by(frame / 2) {
            let x = &pcm[start..start + frame + hi];
            let e0: f32 = x[..frame].iter().map(|v| v * v).sum();
            if e0 / (frame as f32) < 1e-4 {
                continue;
            }
            let r: Vec<f32> = (lo..=hi)
                .map(|lag| {
                    let (mut num, mut el) = (0.0f32, 0.0f32);
                    for i in 0..frame {
                        num += x[i] * x[i + lag];
                        el += x[i + lag] * x[i + lag];
                    }
                    num / (e0 * el).sqrt().max(1e-9)
                })
                .collect();
            let best = r.iter().cloned().fold(f32::MIN, f32::max);
            if best < 0.6 {
                continue;
            }
            let lag = lo + r.iter().position(|v| *v >= 0.9 * best).unwrap();
            f0s.push(sr / lag as f32);
        }
        assert!(!f0s.is_empty(), "no voiced frame in {} samples", pcm.len());
        f0s.sort_by(f32::total_cmp);
        f0s[f0s.len() / 2]
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b)).max(1e-12)
    }

    async fn transcribe(wav: Vec<u8>) -> String {
        let url = std::env::var("NV_TTS_STT_URL")
            .expect("NV_TTS_STT_URL: an OpenAI /v1/audio/transcriptions endpoint (the fleet STT)");
        let model = std::env::var("NV_TTS_STT_MODEL").unwrap_or_else(|_| "voxtral".into());
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("sample.wav")
            .mime_str("audio/wav")
            .unwrap();
        let form = reqwest::multipart::Form::new()
            .text("model", model)
            .part("file", part);
        let mut req = reqwest::Client::new().post(&url).multipart(form);
        if let Ok(key) = std::env::var("NV_TTS_STT_KEY") {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.expect("STT endpoint reachable");
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.expect("STT answers JSON");
        assert!(status.is_success(), "STT {status}: {body}");
        body["text"].as_str().expect("STT text").to_string()
    }

    #[test]
    #[ignore = "real Qwen3-TTS VoiceDesign and Base weights plus the fleet STT: \
                NV_TTS_VOICE_DESIGN_TEST=1, NV_TTS_VOICE_DESIGN_DIR, NV_TTS_STT_URL"]
    fn real_voice_design_follows_the_description_repeats_by_seed_and_clones_onto_base() {
        if std::env::var("NV_TTS_VOICE_DESIGN_TEST").as_deref() != Ok("1") {
            eprintln!("skip real_voice_design: NV_TTS_VOICE_DESIGN_TEST != 1");
            return;
        }
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(DESIGN_FIXTURE).unwrap()).unwrap();
        let design = &fixture["steps"][0]["request_body"];
        let enroll = &fixture["steps"][1]["request_multipart"];
        let clone = &fixture["steps"][2]["request_body"];
        let contrast = &fixture["contrast"];
        let gate = |k: &str| fixture["gates"][k].as_f64().expect(k) as f32;
        let text = |v: &serde_json::Value, k: &str| v[k].as_str().expect(k).to_string();
        let draw = text(design, "voice");
        let seed = clone["seed"].as_u64().unwrap();

        let vd_dir =
            PathBuf::from(std::env::var_os(ENV_VOICE_DESIGN_DIR).expect("NV_TTS_VOICE_DESIGN_DIR"));
        let base_dir = nv_tts::qwen3_tts_base_dir().expect("NV_TTS_BASE_TALKER_DIR");
        let started = std::time::Instant::now();
        let vd = Qwen3TtsAudioSpeech::from_dirs(&vd_dir, None).expect("VoiceDesign boots");
        eprintln!(
            "voice design loaded in {:.1}s",
            started.elapsed().as_secs_f32()
        );
        assert!(vd.voice_design && !vd.profiles_supported && vd.voices().is_empty());
        assert!(routes_to_voice_design(
            design["model"].as_str(),
            &vd.talker_model_id(),
            clone["model"].as_str()
        ));
        let reference = &fixture["python_reference"];
        let ids =
            |v: &serde_json::Value| -> Vec<u32> { serde_json::from_value(v.clone()).unwrap() };
        assert_eq!(
            vd.tokenizer.encode_text(&text(design, "input")).unwrap(),
            ids(&reference["body_ids"]),
            "the text tokens match the Python reference processor"
        );
        assert_eq!(
            vd.tokenizer
                .encode_text(&voice_design_prompt(&text(design, "instructions")))
                .unwrap(),
            ids(&reference["instruct_ids"]),
            "the instruction prompt matches the Python reference's _build_instruct_text tokens"
        );
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let synth =
            |svc: &Qwen3TtsAudioSpeech, input: &str, voice: &str, instructions: Option<String>| {
                let started = std::time::Instant::now();
                let opts = SpeechOptions {
                    instructions,
                    language: None,
                    seed: (!svc.voice_design).then_some(seed),
                };
                let pcm = rt.block_on(async {
                    let mut rx = svc
                        .synthesize(input, voice, &opts)
                        .await
                        .expect("synthesize");
                    let mut all = Vec::new();
                    while let Some(chunk) = rx.recv().await {
                        all.extend_from_slice(&chunk.expect("no mid-stream failure"));
                    }
                    all
                });
                eprintln!(
                    "{voice:?} {:.2}s of audio in {:.1}s",
                    pcm.len() as f32 / NV_TTS_SAMPLE_RATE as f32,
                    started.elapsed().as_secs_f32()
                );
                pcm
            };
        let err = rt
            .block_on(vd.synthesize("Hello there.", "", &SpeechOptions::default()))
            .expect_err("voice design without instructions");
        assert_eq!(
            err.downcast_ref::<InvalidSpeechParam>().map(|e| e.param),
            Some("instructions")
        );

        let warm = synth(
            &vd,
            &text(design, "input"),
            &draw,
            Some(text(design, "instructions")),
        );
        assert_eq!(
            synth(
                &vd,
                &text(design, "input"),
                &draw,
                Some(text(design, "instructions"))
            ),
            warm,
            "the same draw and description must give the same audio"
        );
        let low = synth(
            &vd,
            &text(contrast, "input"),
            &draw,
            Some(text(contrast, "low")),
        );
        let high = synth(
            &vd,
            &text(contrast, "input"),
            &draw,
            Some(text(contrast, "high")),
        );

        let encoder = Arc::new(
            nv_tts::SpeakerEncoderSource::Checkpoint(base_dir.clone())
                .load(&Device::Cpu)
                .expect("Base speaker encoder"),
        );
        let mut base = Qwen3TtsAudioSpeech::from_dirs(&base_dir, None).expect("Base boots");
        let (tmp, store) = temp_profile_store("vd_clone");
        base.voice_profiles = Arc::new(VoiceStoreHandle::new(&tmp));
        let name = text(enroll, "name");
        let samples =
            crate::audio::decode_any_to_16k_mono(&wav_16bit(&warm), Some("audio/wav")).unwrap();
        store
            .put(&nv_tts::VoiceProfile {
                schema_version: 2,
                name: name.clone(),
                embedding: crate::oapi::voice_profiles::extract_embedding(&encoder, &samples)
                    .expect("enrollment embedding"),
                design_params: Some(enroll["design_params"].clone()),
            })
            .unwrap();
        let paragraph = text(clone, "input");
        let cloned = synth(&base, &paragraph, &name, None);
        let default_voice = synth(&base, &paragraph, "default", None);

        let wer = |pcm: &[f32], reference: &str| {
            let heard = rt.block_on(transcribe(wav_16bit(pcm)));
            let (errors, words) = nv_models::voxtral_train::word_error_rate(&heard, reference);
            eprintln!("heard {heard:?}");
            errors as f32 / words as f32
        };
        let voices = [
            ("warm", &warm, text(design, "input")),
            ("low", &low, text(contrast, "input")),
            ("high", &high, text(contrast, "input")),
            ("clone", &cloned, paragraph.clone()),
            ("default", &default_voice, paragraph.clone()),
        ];
        let measured: Vec<(&str, f32, f32, Vec<f32>)> = voices
            .iter()
            .map(|(label, pcm, reference)| {
                (
                    *label,
                    wer(pcm, reference),
                    median_f0_hz(pcm),
                    encoder.encode_pcm_24k(pcm).expect("x-vector"),
                )
            })
            .collect();
        let cos = |a: &str, b: &str| {
            let x = |l: &str| &measured.iter().find(|m| m.0 == l).unwrap().3;
            cosine(x(a), x(b))
        };
        for (label, wer, f0, _) in &measured {
            eprintln!("{label}: WER {wer:.3}, median F0 {f0:.1} Hz");
        }
        for (a, b) in [
            ("warm", "low"),
            ("warm", "high"),
            ("low", "high"),
            ("warm", "clone"),
            ("warm", "default"),
            ("clone", "default"),
            ("high", "clone"),
        ] {
            eprintln!("ECAPA cosine {a} vs {b}: {:.3}", cos(a, b));
        }
        let f0 = |l: &str| measured.iter().find(|m| m.0 == l).unwrap().2;
        assert!(
            measured.iter().all(|m| m.1 <= gate("max_word_error_rate")),
            "WER gate"
        );
        assert!(f0("low") <= gate("low_max_median_f0_hz"), "low F0");
        assert!(f0("high") >= gate("high_min_median_f0_hz"), "high F0");
        assert!(
            cos("warm", "low") - cos("low", "high") >= gate("contrast_min_ecapa_margin"),
            "two deep male descriptions must sit closer than a deep male and a high female"
        );
        assert!(
            cos("warm", "clone") - cos("warm", "default")
                >= gate("clone_min_ecapa_margin_over_default_voice"),
            "the clone must sound like the designed sample, not like the Base default voice"
        );
        assert!(
            (f0("clone") / f0("warm") - 1.0).abs() <= gate("clone_max_median_f0_deviation"),
            "the clone keeps the designed voice's pitch"
        );
        drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(feature = "cuda")]
    const BASE_PRECISION_AUDIO_LENGTH_MAX_DEVIATION_FROM_THE_ANCHOR_0P25_BEYOND_IT_A_READ_RAN_AWAY_OR_WAS_CUT:
        f32 = 0.25;
    #[cfg(feature = "cuda")]
    const CLONE_MIN_ECAPA_MARGIN_OVER_THE_OTHER_VOICES_CLONE_0P01_BOTH_REFERENCES_ARE_CALM_MALE_NARRATORS:
        f32 = 0.01;
    #[cfg(feature = "cuda")]
    const PRECISION_MIN_ECAPA_COSINE_TO_THE_ANCHORS_CLONE_0P99: f32 = 0.99;

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "real Qwen3-TTS Base weights, reference clips and the fleet STT: \
                NV_TTS_BASE_PRECISION_TEST=1, NV_TTS_BASE_TALKER_DIR, NV_TTS_CLONE_REFS, \
                NV_TTS_STT_URL"]
    fn real_base_talker_precisions_keep_the_anchors_words_pitch_and_clone_likeness() {
        if std::env::var("NV_TTS_BASE_PRECISION_TEST").as_deref() != Ok("1") {
            eprintln!("skip real_base_talker_precisions: NV_TTS_BASE_PRECISION_TEST != 1");
            return;
        }
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_writer(std::io::stderr)
            .try_init();
        let base_dir = nv_tts::qwen3_tts_base_dir().expect("NV_TTS_BASE_TALKER_DIR");
        let refs: Vec<(String, PathBuf)> = std::env::var("NV_TTS_CLONE_REFS")
            .expect("NV_TTS_CLONE_REFS: label=clip.wav[,label=clip.wav]")
            .split(',')
            .map(|kv| {
                let (label, path) = kv.split_once('=').expect("label=path");
                (label.trim().to_string(), PathBuf::from(path.trim()))
            })
            .collect();
        let precisions: Vec<TalkerPrecision> = std::env::var("NV_TTS_PRECISIONS")
            .unwrap_or_else(|_| "f32,bf16,fp8".into())
            .split(',')
            .map(|s| TalkerPrecision::parse(s).expect("NV_TTS_PRECISIONS"))
            .collect();
        let anchor = *precisions.first().expect("NV_TTS_PRECISIONS names the anchor first");
        assert!(
            refs.len() >= 2,
            "NV_TTS_CLONE_REFS needs two voices: each clone is judged against the other's"
        );
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(DESIGN_FIXTURE).unwrap()).unwrap();
        let clone_step = &fixture["steps"][2]["request_body"];
        let paragraph = clone_step["input"].as_str().unwrap().to_string();
        let seed = clone_step["seed"].as_u64().unwrap();
        let gate = |k: &str| fixture["gates"][k].as_f64().expect(k) as f32;

        let encoder = Arc::new(
            nv_tts::SpeakerEncoderSource::Checkpoint(base_dir.clone())
                .load(&Device::Cpu)
                .expect("Base speaker encoder"),
        );
        let (tmp, store) = temp_profile_store("precision");
        let mut ref_xvec: Vec<(String, Vec<f32>)> = Vec::new();
        for (label, path) in &refs {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let samples = crate::audio::decode_any_to_16k_mono(&bytes, Some("audio/wav"))
                .expect("decode reference");
            let embedding = crate::oapi::voice_profiles::extract_embedding(&encoder, &samples)
                .expect("reference embedding");
            store
                .put(&nv_tts::VoiceProfile {
                    schema_version: 2,
                    name: label.clone(),
                    embedding: embedding.clone(),
                    design_params: None,
                })
                .unwrap();
            ref_xvec.push((label.clone(), embedding));
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let free = || crate::oapi::cold_tier::device_free_bytes().unwrap_or(0);
        let gib = |b: u64| b as f32 / (1u64 << 30) as f32;

        struct Row {
            precision: TalkerPrecision,
            voice: String,
            audio_s: f32,
            gen_s: f32,
            wer: f32,
            f0: f32,
            xvec: Vec<f32>,
        }
        let mut rows: Vec<Row> = Vec::new();
        for &precision in &precisions {
            let idle = free();
            let started = std::time::Instant::now();
            let mut svc =
                Qwen3TtsAudioSpeech::from_dirs_with_precision(&base_dir, None, Some(precision))
                    .expect("Base boots");
            assert_eq!(svc.precision, precision);
            assert!(
                svc.talker.is_some() && svc.code_predictor.is_some(),
                "{} loaded the talker and code predictor",
                precision.label()
            );
            svc.voice_profiles = Arc::new(VoiceStoreHandle::new(&tmp));
            let resident = idle.saturating_sub(free());
            let mut peak = resident;
            eprintln!(
                "{}: loaded in {:.1}s, resident {:.2} GiB",
                precision.label(),
                started.elapsed().as_secs_f32(),
                gib(resident)
            );
            let voices: Vec<String> = refs.iter().map(|(l, _)| l.clone()).collect();
            for voice in voices {
                let started = std::time::Instant::now();
                let opts = SpeechOptions {
                    instructions: None,
                    language: None,
                    seed: Some(seed),
                };
                let pcm: Vec<f32> = rt.block_on(async {
                    let mut rx = svc
                        .synthesize(&paragraph, &voice, &opts)
                        .await
                        .expect("synthesize");
                    let mut all = Vec::new();
                    while let Some(chunk) = rx.recv().await {
                        all.extend_from_slice(&chunk.expect("no mid-stream failure"));
                    }
                    all
                });
                let gen_s = started.elapsed().as_secs_f32();
                peak = peak.max(idle.saturating_sub(free()));
                let audio_s = pcm.len() as f32 / NV_TTS_SAMPLE_RATE as f32;
                let heard = rt.block_on(transcribe(wav_16bit(&pcm)));
                let (errors, words) =
                    nv_models::voxtral_train::word_error_rate(&heard, &paragraph);
                let row = Row {
                    precision,
                    voice: voice.clone(),
                    audio_s,
                    gen_s,
                    wer: errors as f32 / words as f32,
                    f0: median_f0_hz(&pcm),
                    xvec: encoder.encode_pcm_24k(&pcm).expect("x-vector"),
                };
                eprintln!(
                    "{} {voice:?}: {:.2}s audio in {:.1}s, WER {:.3}, median F0 {:.1} Hz, \
                     heard {heard:?}",
                    precision.label(),
                    row.audio_s,
                    row.gen_s,
                    row.wer,
                    row.f0
                );
                rows.push(row);
            }
            eprintln!("{}: peak {:.2} GiB over idle", precision.label(), gib(peak));
            drop(svc);
        }
        let find = |p: TalkerPrecision, v: &str| {
            rows.iter()
                .find(|r| r.precision == p && r.voice == v)
                .unwrap()
        };
        let nearest_other = |p: TalkerPrecision, label: &str, xref: &[f32]| {
            ref_xvec
                .iter()
                .filter(|(other, _)| other != label)
                .map(|(other, _)| cosine(&find(p, other).xvec, xref))
                .fold(f32::MIN, f32::max)
        };
        for &p in &precisions {
            for (label, xref) in &ref_xvec {
                let (clone, twin) = (find(p, label), find(anchor, label));
                eprintln!(
                    "{} {label}: cos(clone, reference) {:.3}, cos(other clone, reference) {:.3}, \
                     cos(clone, {} clone) {:.3}",
                    p.label(),
                    cosine(&clone.xvec, xref),
                    nearest_other(p, label, xref),
                    anchor.label(),
                    cosine(&clone.xvec, &twin.xvec)
                );
            }
        }
        for r in &rows {
            assert!(
                r.wer <= gate("max_word_error_rate"),
                "{} {:?}: WER {:.3}",
                r.precision.label(),
                r.voice,
                r.wer
            );
        }
        for &p in &precisions {
            for (label, xref) in &ref_xvec {
                assert!(
                    cosine(&find(p, label).xvec, xref) - nearest_other(p, label, xref)
                        >= CLONE_MIN_ECAPA_MARGIN_OVER_THE_OTHER_VOICES_CLONE_0P01_BOTH_REFERENCES_ARE_CALM_MALE_NARRATORS,
                    "{} {label}: the clone must sit nearer its reference than the other voice's clone",
                    p.label()
                );
                assert!(
                    cosine(&find(p, label).xvec, &find(anchor, label).xvec)
                        >= PRECISION_MIN_ECAPA_COSINE_TO_THE_ANCHORS_CLONE_0P99,
                    "{} {label}: the clone must stay the {} clone's voice",
                    p.label(),
                    anchor.label()
                );
            }
            if p == anchor {
                continue;
            }
            for r in rows.iter().filter(|r| r.precision == p) {
                let twin = find(anchor, &r.voice);
                assert!(
                    (r.f0 / twin.f0 - 1.0).abs() <= gate("clone_max_median_f0_deviation"),
                    "{} {:?}: median F0 {:.1} Hz against {}'s {:.1} Hz",
                    p.label(),
                    r.voice,
                    r.f0,
                    anchor.label(),
                    twin.f0
                );
                assert!(
                    (r.audio_s / twin.audio_s - 1.0).abs()
                        <= BASE_PRECISION_AUDIO_LENGTH_MAX_DEVIATION_FROM_THE_ANCHOR_0P25_BEYOND_IT_A_READ_RAN_AWAY_OR_WAS_CUT,
                    "{} {:?}: {:.2}s of audio against {}'s {:.2}s",
                    p.label(),
                    r.voice,
                    r.audio_s,
                    anchor.label(),
                    twin.audio_s
                );
            }
        }
        drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
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
