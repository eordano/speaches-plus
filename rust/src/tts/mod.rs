pub mod chunk;
pub mod http;
pub mod npz;
pub mod phonemize;
pub mod text;
pub mod vocab;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use ort::ep::{self, ExecutionProvider};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use tracing::debug;

use self::npz::Voice;
use super::vad::ort_err;

#[allow(dead_code)]
pub const TTS_SAMPLE_RATE: usize = super::defaults::audio::TTS_SAMPLE_RATE;

#[derive(Clone)]
pub struct KokoroHandle {
    session: Arc<Mutex<Session>>,
    voices: Arc<HashMap<String, Voice>>,
    queue: Arc<tokio::sync::Semaphore>,
}

impl KokoroHandle {
    pub fn has_voice(&self, name: &str) -> bool {
        self.voices.contains_key(name)
    }

    pub fn voice_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.voices.keys().cloned().collect();
        names.sort();
        names
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
        let voice_pack = self.voices.get(voice_name).ok_or_else(|| {
            anyhow!(
                "voice {voice_name:?} not found in voices.bin ({} voices loaded)",
                self.voices.len()
            )
        })?;

        let raw =
            phonemize::phonemize(text, lang).with_context(|| format!("phonemize {text:?}"))?;
        let cleaned = vocab::clean_phonemes(&raw);
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
        let n = tokens.len();

        let mut padded = Vec::with_capacity(n + 2);
        padded.push(0i64);
        padded.extend_from_slice(&tokens);
        padded.push(0i64);

        if voice_pack.shape.is_empty() || n >= voice_pack.shape[0] {
            return Err(anyhow!(
                "token count {n} >= style rows {}",
                voice_pack.shape.first().copied().unwrap_or(0)
            ));
        }
        let style_vec = voice_pack.row(n).context("style row")?;

        let token_tensor = Tensor::<i64>::from_array(([1usize, n + 2], padded.into_boxed_slice()))
            .map_err(ort_err)?;
        let style_tensor =
            Tensor::<f32>::from_array(([1usize, style_vec.len()], style_vec.into_boxed_slice()))
                .map_err(ort_err)?;
        let speed_tensor = Tensor::<f32>::from_array(([1usize], vec![speed].into_boxed_slice()))
            .map_err(ort_err)?;

        let audio_vec: Vec<f32> = {
            let mut session = self
                .session
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
            audio.to_vec()
        };
        debug!(
            phonemes = cleaned.as_str(),
            n_tokens = n,
            samples = audio_vec.len(),
            "kokoro synth complete"
        );
        Ok(super::types::MonoF32At24k::new(audio_vec))
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
        .map(|v| v.trim().to_ascii_lowercase().replace("executionprovider", ""))
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
            Err(e) => debug!(name, ?e, "kokoro gpu provider available but failed to register"),
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

    let voices = npz::load_voices(&voices_path).context("load voices")?;
    debug!(count = voices.len(), "loaded Kokoro voices");

    let espeak_data = espeak_data_dir();
    phonemize::init(espeak_data.as_deref()).context("init phonemizer")?;

    let threads = intra_threads();
    let builder = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(threads)
        .map_err(ort_err)?
        .with_inter_threads(1)
        .map_err(ort_err)?;
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

    let handle = KokoroHandle {
        session: Arc::new(Mutex::new(session)),
        voices: Arc::new(voices),
        queue: Arc::new(tokio::sync::Semaphore::new(1)),
    };

    let warmup_enabled = std::env::var(super::defaults::env::KOKORO_WARMUP)
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    if warmup_enabled {
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
    }

    Ok(Some(handle))
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
