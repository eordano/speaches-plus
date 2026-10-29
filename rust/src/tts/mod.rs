pub mod builtin_voices;
pub mod chunk;
pub mod fit;
pub mod http;
pub mod npz;
pub mod phonemize;
pub mod text;
pub mod vocab;
pub mod voxtral_tts;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::{anyhow, Context, Result};
use ort::ep::{self, ExecutionProvider};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use tracing::debug;

use self::npz::Voice;
use super::vad::ort_err;
use nv_tts::kokoro::KokoroCandle;
use nv_tts::{VoiceKind, VoiceStoreHandle, ADAPTER_FILE, PACK_FILE};

pub fn load_candle_from_env() -> Option<Arc<KokoroCandle>> {
    let weights = std::env::var_os("KOKORO_CANDLE_WEIGHTS")?;
    let weights = PathBuf::from(weights);
    let enc = match (
        std::env::var_os("NV_QWEN_VOICE_ENCODER_DIR"),
        std::env::var_os("NV_WESPEAKER_CANDLE_DIR"),
    ) {
        (Some(q), Some(w)) => Some((PathBuf::from(q), PathBuf::from(w))),
        _ => None,
    };
    let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
    match KokoroCandle::load(
        &weights,
        enc.as_ref().map(|(q, w)| (q.as_path(), w.as_path())),
        &device,
    ) {
        Ok(h) => {
            debug!(path = %weights.display(), fit = enc.is_some(), "kokoro candle model loaded");
            Some(Arc::new(h))
        }
        Err(e) => {
            debug!(?e, path = %weights.display(), "kokoro candle model failed to load");
            None
        }
    }
}

pub fn load_adapter_map(
    directory: &std::path::Path,
    device: &candle_core::Device,
) -> Result<HashMap<String, candle_core::Tensor>> {
    let st = directory.join("adapter.safetensors");
    if st.is_file() {
        return Ok(candle_core::safetensors::load(&st, device)?);
    }
    let voices = npz::load_voices(&directory.join(ADAPTER_FILE)).context("read adapter.npz")?;
    let mut out = HashMap::with_capacity(voices.len());
    for (k, v) in voices {
        out.insert(
            k,
            candle_core::Tensor::from_vec(v.data, v.shape.as_slice(), device)?,
        );
    }
    Ok(out)
}

#[derive(Clone)]
struct StoreVoice {
    directory: PathBuf,
    modified: Option<SystemTime>,
    pack: Voice,
    session: Option<Arc<Mutex<Session>>>,

    adapter: Option<Arc<HashMap<String, candle_core::Tensor>>>,
}

#[allow(dead_code)]
pub const TTS_SAMPLE_RATE: usize = super::defaults::audio::TTS_SAMPLE_RATE;

#[derive(Clone)]
pub struct KokoroHandle {
    session: Arc<Mutex<Session>>,
    voices: Arc<HashMap<String, Voice>>,
    queue: Arc<tokio::sync::Semaphore>,
    provider: &'static str,
    model_path: Arc<PathBuf>,
    threads: usize,
    store: Arc<VoiceStoreHandle>,
    store_cache: Arc<Mutex<HashMap<String, StoreVoice>>>,
    candle: Option<Arc<KokoroCandle>>,
}

impl KokoroHandle {
    pub fn provider(&self) -> &'static str {
        self.provider
    }

    pub fn candle(&self) -> Option<Arc<KokoroCandle>> {
        self.candle.clone()
    }

    pub fn has_voice(&self, name: &str) -> bool {
        self.voices.contains_key(name) || self.store_has(name)
    }

    pub fn voice_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.voices.keys().cloned().collect();
        if let Ok(store) = self.store.get() {
            if let Ok(metas) = store.list_kokoro() {
                names.extend(metas.into_iter().map(|m| m.name));
            }
        }
        names.sort();
        names.dedup();
        names
    }

    fn store_has(&self, name: &str) -> bool {
        self.store
            .get()
            .ok()
            .and_then(|s| s.get_meta(name).ok())
            .map(|m| m.kind.is_kokoro())
            .unwrap_or(false)
    }

    fn store_voice(&self, name: &str) -> Result<Option<StoreVoice>> {
        let Ok(store) = self.store.get() else {
            return Ok(None);
        };
        if !store.exists(name) {
            return Ok(None);
        }
        let directory = store.dir_for(name);
        let modified = std::fs::metadata(directory.join(nv_tts::META_FILE))
            .and_then(|m| m.modified())
            .ok();
        {
            let cache = self
                .store_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(sv) = cache.get(name) {
                if sv.modified == modified && sv.directory == directory {
                    return Ok(Some(sv.clone()));
                }
            }
        }
        let meta: nv_tts::VoiceMeta =
            serde_json::from_slice(&std::fs::read(directory.join(nv_tts::META_FILE))?)?;
        if !meta.kind.is_kokoro() {
            return Ok(None);
        }
        let pack = npz::parse_npy(&std::fs::read(directory.join(PACK_FILE))?)
            .with_context(|| format!("voice {name:?}: {PACK_FILE}"))?;
        let adapter = match (&meta.kind, &self.candle) {
            (VoiceKind::KokoroAdapter, Some(candle)) => Some(Arc::new(
                load_adapter_map(&directory, &candle.device)
                    .with_context(|| format!("voice {name:?}: adapter"))?,
            )),
            _ => None,
        };
        let session = if meta.kind == VoiceKind::KokoroAdapter && adapter.is_none() {
            let t0 = std::time::Instant::now();
            let session = self
                .build_adapted_session(&directory.join(ADAPTER_FILE))
                .with_context(|| format!("voice {name:?}: {ADAPTER_FILE}"))?;
            debug!(
                voice = name,
                ms = t0.elapsed().as_millis() as u64,
                "kokoro adapted session ready"
            );
            Some(Arc::new(Mutex::new(session)))
        } else {
            None
        };
        let sv = StoreVoice {
            directory,
            modified,
            pack,
            session,
            adapter,
        };
        let mut cache = self
            .store_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if sv.session.is_some() {
            let limit = adapted_sessions_limit();
            while cache.values().filter(|v| v.session.is_some()).count() >= limit {
                let Some(victim) = cache
                    .iter()
                    .find(|(k, v)| v.session.is_some() && k.as_str() != name)
                    .map(|(k, _)| k.clone())
                else {
                    break;
                };
                cache.remove(&victim);
            }
        }
        cache.insert(name.to_string(), sv.clone());
        Ok(Some(sv))
    }

    fn build_adapted_session(&self, adapter: &Path) -> Result<Session> {
        let overrides = npz::load_voices(adapter).context("read adapter")?;
        if overrides.is_empty() {
            return Err(anyhow!("adapter has no initializer entries"));
        }
        let builder = session_builder(self.threads)?;
        let (mut builder, _provider) = with_requested_provider(builder)?;
        for (init_name, value) in overrides {
            let tensor = Tensor::<f32>::from_array((value.shape.clone(), value.data))
                .map_err(ort_err)
                .with_context(|| format!("initializer {init_name}"))?;
            builder = builder
                .with_external_initializer(&init_name, Arc::new(tensor.into_dyn()))
                .map_err(ort_err)
                .with_context(|| format!("override initializer {init_name}"))?;
        }
        builder
            .commit_from_file(self.model_path.as_path())
            .map_err(ort_err)
            .with_context(|| format!("load kokoro {}", self.model_path.display()))
    }

    fn resolve_voice(&self, voice_name: &str) -> Result<(Voice, Arc<Mutex<Session>>)> {
        if let Some(sv) = self.store_voice(voice_name)? {
            return Ok((sv.pack, sv.session.unwrap_or_else(|| self.session.clone())));
        }
        let pack = self.voices.get(voice_name).ok_or_else(|| {
            anyhow!(
                "voice {voice_name:?} not found in voices.bin ({} voices loaded)",
                self.voices.len()
            )
        })?;
        Ok((pack.clone(), self.session.clone()))
    }

    pub fn queue(&self) -> Arc<tokio::sync::Semaphore> {
        self.queue.clone()
    }

    pub fn synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        lang: Option<&str>,
        speed: f32,
    ) -> Result<super::types::MonoF32At24k> {
        if !(0.5..=2.0).contains(&speed) {
            return Err(anyhow!("speed out of range: {speed}"));
        }
        let voice_name = voice.unwrap_or("af_heart");
        let lang = lang.unwrap_or("en-us");
        let (cleaned, tokens) = self.tokens_for(text, lang)?;
        let n = tokens.len();
        if let (Some(candle), Some(sv)) = (&self.candle, self.store_voice(voice_name)?) {
            if let Some(adapter) = &sv.adapter {
                let rows = sv.pack.shape.first().copied().unwrap_or(0);
                if rows == 0 {
                    return Err(anyhow!("voice {voice_name:?} has no style rows"));
                }
                let style_vec = sv.pack.row(n.min(rows - 1)).context("style row")?;
                let mut ids: Vec<u32> = Vec::with_capacity(n + 2);
                ids.push(0);
                ids.extend(tokens.iter().map(|t| *t as u32));
                ids.push(0);
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(1);
                let audio_vec = candle.synthesize(
                    &ids,
                    &style_vec,
                    speed as f64,
                    Some(adapter.as_ref()),
                    seed,
                )?;
                debug!(
                    phonemes = cleaned.as_str(),
                    n_tokens = n,
                    samples = audio_vec.len(),
                    "kokoro candle synth complete (adapter)"
                );
                return Ok(super::types::MonoF32At24k::new(audio_vec));
            }
        }
        let (voice_pack, session) = self.resolve_voice(voice_name)?;
        let rows = voice_pack.shape.first().copied().unwrap_or(0);
        if rows == 0 {
            return Err(anyhow!("voice {voice_name:?} has no style rows"));
        }
        let style_vec = voice_pack.row(n.min(rows - 1)).context("style row")?;
        let audio_vec = run_kokoro(&session, &tokens, &style_vec, speed)?;
        debug!(
            phonemes = cleaned.as_str(),
            n_tokens = n,
            samples = audio_vec.len(),
            "kokoro synth complete"
        );
        Ok(super::types::MonoF32At24k::new(audio_vec))
    }

    pub fn tokens_for(&self, text: &str, lang: &str) -> Result<(String, Vec<i64>)> {
        let raw =
            phonemize::phonemize(text, lang).with_context(|| format!("phonemize {text:?}"))?;
        let cleaned = vocab::clean_phonemes(&raw, lang);
        if cleaned.is_empty() {
            return Err(anyhow!("phonemize produced empty output for {text:?}"));
        }
        let tokens = vocab::tokenize(&cleaned);
        if tokens.is_empty() {
            return Err(anyhow!("tokenize empty for cleaned phonemes {cleaned:?}"));
        }
        if tokens.len() > vocab::MAX_PHONEME_LENGTH {
            return Err(anyhow!(
                "phoneme token count {} exceeds MAX_PHONEME_LENGTH ({})",
                tokens.len(),
                vocab::MAX_PHONEME_LENGTH
            ));
        }
        Ok((cleaned, tokens))
    }

    pub fn synth_tokens_with_style(
        &self,
        tokens: &[i64],
        style: &[f32],
        speed: f32,
    ) -> Result<Vec<f32>> {
        run_kokoro(&self.session, tokens, style, speed)
    }

    pub fn stock_style_rows(&self) -> Vec<(String, Vec<f32>)> {
        let mut out: Vec<(String, Vec<f32>)> = self
            .voices
            .iter()
            .filter_map(|(name, v)| {
                let rows = v.shape.first().copied().unwrap_or(0);
                if rows == 0 {
                    return None;
                }
                v.row(100.min(rows - 1)).ok().map(|r| (name.clone(), r))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

pub fn intra_threads() -> usize {
    if let Some(n) = std::env::var(super::defaults::env::KOKORO_INTRA_THREADS)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        return n.max(super::defaults::kokoro::INTRA_THREADS_MIN);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(super::defaults::kokoro::INTRA_THREADS_MIN)
        .clamp(
            super::defaults::kokoro::INTRA_THREADS_MIN,
            super::defaults::kokoro::INTRA_THREADS_MAX,
        )
}

const FORCE_ONLY_BECAUSE_REGISTER_SUCCEEDS_WITHOUT_A_USABLE_DEVICE: [&str; 2] =
    ["openvino", "migraphx"];

fn gpu_eps() -> Vec<(&'static str, Box<dyn ExecutionProvider>)> {
    vec![
        ("cuda", Box::new(ep::CUDA::default())),
        ("rocm", Box::new(ep::ROCm::default())),
        ("migraphx", Box::new(ep::MIGraphX::default())),
        ("coreml", Box::new(ep::CoreML::default())),
        ("webgpu", Box::new(ep::WebGPU::default())),
        ("dml", Box::new(ep::DirectML::default())),
        ("openvino", Box::new(ep::OpenVINO::default())),
    ]
}

fn with_requested_provider(mut builder: SessionBuilder) -> Result<(SessionBuilder, &'static str)> {
    let requested = std::env::var(super::defaults::env::KOKORO_ONNX_PROVIDER)
        .map(|v| {
            v.trim()
                .to_ascii_lowercase()
                .replace("executionprovider", "")
        })
        .unwrap_or_default();
    if requested == "cpu" {
        return Ok((builder, "cpu"));
    }
    if !requested.is_empty() {
        let (name, provider) = gpu_eps()
            .into_iter()
            .find(|(name, _)| *name == requested || requested == "directml" && *name == "dml")
            .ok_or_else(|| {
                anyhow!(
                    "{} must be one of cpu, cuda, rocm, migraphx, coreml, webgpu, dml, openvino; got {requested:?}",
                    super::defaults::env::KOKORO_ONNX_PROVIDER
                )
            })?;
        provider
            .register(&mut builder)
            .map_err(|e| anyhow!("{name} execution provider failed to register: {e}"))?;
        return Ok((builder, name));
    }
    for (name, provider) in gpu_eps() {
        if FORCE_ONLY_BECAUSE_REGISTER_SUCCEEDS_WITHOUT_A_USABLE_DEVICE.contains(&name) {
            continue;
        }
        if !provider.is_available().unwrap_or(false) {
            continue;
        }
        match provider.register(&mut builder) {
            Ok(()) => return Ok((builder, name)),
            Err(e) => debug!(
                name,
                ?e,
                "kokoro gpu provider available but failed to register"
            ),
        }
    }
    Ok((builder, "cpu"))
}

pub fn prepared_handle(model_dir: &Path) -> Result<Option<KokoroHandle>> {
    let model_path = model_dir.join("kokoro-v1.0.onnx");
    let voices_path = first_existing(&[
        model_dir.join("kokoro-voices.bin"),
        model_dir.join("voices.bin"),
        kokoro_voices_alongside(&model_path),
    ]);
    if !model_path.exists() {
        return Ok(None);
    }
    let Some(voices_path) = voices_path else {
        debug!("Kokoro voices.bin not found alongside model -- TTS disabled");
        return Ok(None);
    };

    let mut voices = npz::load_voices(&voices_path).context("load voices")?;
    for (name, voice) in builtin_voices::load().context("builtin voices")? {
        voices.entry(name).or_insert(voice);
    }
    debug!(count = voices.len(), "loaded Kokoro voices");

    let espeak_data = espeak_data_dir();
    phonemize::init(espeak_data.as_deref()).context("init phonemizer")?;

    let threads = intra_threads();
    let builder = session_builder(threads)?;
    let (mut builder, provider) = with_requested_provider(builder)?;
    let session = builder
        .commit_from_file(&model_path)
        .map_err(ort_err)
        .with_context(|| format!("load kokoro {}", model_path.display()))?;
    debug!(
        intra_threads = threads,
        provider,
        path = %model_path.display(),
        "kokoro session ready"
    );

    let store = Arc::new(VoiceStoreHandle::new(
        crate::oapi::voice_profiles::voice_profile_root(),
    ));
    let handle = KokoroHandle {
        session: Arc::new(Mutex::new(session)),
        voices: Arc::new(voices),
        queue: Arc::new(tokio::sync::Semaphore::new(1)),
        provider,
        model_path: Arc::new(model_path.clone()),
        threads,
        store,
        store_cache: Arc::new(Mutex::new(HashMap::new())),
        candle: load_candle_from_env(),
    };

    let h = handle.clone();
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        match h.synthesize("Warm up.", None, None, 1.0) {
            Ok(audio) => debug!(
                ms = t0.elapsed().as_millis() as u64,
                samples = audio.len(),
                "kokoro warmup complete"
            ),
            Err(e) => debug!(?e, "kokoro warmup failed"),
        }
    });

    Ok(Some(handle))
}

fn session_builder(threads: usize) -> Result<SessionBuilder> {
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(threads)
        .map_err(ort_err)?
        .with_inter_threads(1)
        .map_err(ort_err)
}

fn adapted_sessions_limit() -> usize {
    std::env::var("KOKORO_ADAPTED_SESSIONS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2)
        .max(1)
}

fn run_kokoro(
    session: &Arc<Mutex<Session>>,
    tokens: &[i64],
    style: &[f32],
    speed: f32,
) -> Result<Vec<f32>> {
    let n = tokens.len();
    let mut padded = Vec::with_capacity(n + 2);
    padded.push(0i64);
    padded.extend_from_slice(tokens);
    padded.push(0i64);
    let token_tensor =
        Tensor::<i64>::from_array(([1usize, n + 2], padded.into_boxed_slice())).map_err(ort_err)?;
    let style_tensor =
        Tensor::<f32>::from_array(([1usize, style.len()], style.to_vec().into_boxed_slice()))
            .map_err(ort_err)?;
    let speed_tensor =
        Tensor::<f32>::from_array(([1usize], vec![speed].into_boxed_slice())).map_err(ort_err)?;
    let mut session = session
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session
        .run(ort::inputs![
            "tokens" => token_tensor,
            "style" => style_tensor,
            "speed" => speed_tensor,
        ])
        .map_err(ort_err)?;
    let (_, audio) = outputs["audio"]
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    Ok(audio.to_vec())
}

fn first_existing(candidates: &[std::path::PathBuf]) -> Option<std::path::PathBuf> {
    for c in candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }
    None
}

fn kokoro_voices_alongside(model_path: &Path) -> std::path::PathBuf {
    if let Ok(target) = std::fs::read_link(model_path) {
        if let Some(parent) = target.parent() {
            return parent.join("voices.bin");
        }
    }
    model_path.with_file_name("voices.bin")
}

fn espeak_data_dir() -> Option<String> {
    if let Ok(p) = std::env::var("ESPEAK_DATA_PATH") {
        return Some(p);
    }
    if let Ok(home) = std::env::var("HOME") {
        let cand = format!("{home}/.nix-profile/share/espeak-ng-data");
        if std::path::Path::new(&cand).exists() {
            return Some(cand);
        }
    }
    None
}

#[cfg(test)]
mod adapter_tests {
    use super::*;

    fn check_adapter_formats(device: &candle_core::Device) -> Result<()> {
        let td = crate::voices::stages::TempDir::new("adapter-device")?;
        let path = td.0.join("adapter.safetensors");
        let tensor = candle_core::Tensor::from_vec(
            vec![1f32, 2., 3., 4.],
            (2, 2),
            &candle_core::Device::Cpu,
        )?;
        let map = HashMap::from([("decoder.test.weight".to_string(), tensor)]);
        candle_core::safetensors::save(&map, &path)?;
        for safetensors in [true, false] {
            if !safetensors {
                std::fs::remove_file(&path)?;
                npz::write_npz(
                    &td.0.join(ADAPTER_FILE),
                    &[(
                        "decoder.test.weight".into(),
                        vec![2, 2],
                        vec![1., 2., 3., 4.],
                    )],
                )?;
            }
            let loaded = load_adapter_map(&td.0, device)?;
            let w = &loaded["decoder.test.weight"];
            assert!(w.device().same_device(device));
            let result = w.matmul(&candle_core::Tensor::ones(
                (2, 1),
                candle_core::DType::F32,
                device,
            )?)?;
            assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, vec![3., 7.]);
        }
        Ok(())
    }

    #[test]
    fn voices_adapter_formats_load_on_requested_cpu() -> Result<()> {
        check_adapter_formats(&candle_core::Device::Cpu)
    }
}
