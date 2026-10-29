use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};

use super::diarization::{EmbeddingModel, SegmentationModel};
use super::stt::{WhisperEngine, WhisperHandle};
use super::tts::KokoroHandle;

pub struct Models {
    vad_session: Option<Arc<Mutex<ort::session::Session>>>,
    whisper: Option<WhisperHandle>,
    vad_unavailable: Option<String>,
    whisper_unavailable: Option<String>,
    pub kokoro: Option<KokoroHandle>,

    pub voxtral_tts: Option<Arc<super::tts::voxtral_tts::VoxtralTtsSpeech>>,

    pub diar_segmentation: Option<Arc<SegmentationModel>>,

    pub diar_embedding: Option<Arc<EmbeddingModel>>,
    #[allow(dead_code)]
    pub model_dir: PathBuf,
}

static MODELS: OnceLock<Arc<Models>> = OnceLock::new();

impl Models {
    pub fn get_or_init() -> Result<Arc<Self>> {
        if let Some(models) = MODELS.get() {
            return Ok(models.clone());
        }
        let loaded = Arc::new(Self::load()?);
        let _ = MODELS.set(loaded);
        Ok(MODELS.get().expect("just set").clone())
    }

    fn load() -> Result<Self> {
        let model_dir = model_dir();
        tracing::info!(path = %model_dir.display(), "loading models");

        let t = std::time::Instant::now();
        let vad_path = model_dir.join("silero_vad.onnx");
        let mut vad_unavailable: Option<String> = None;
        let vad_session = if vad_path.exists() {
            match build_vad_session_with_timeout(&vad_path) {
                Ok(s) => {
                    tracing::info!(elapsed_ms = t.elapsed().as_millis() as u64, "VAD loaded");
                    Some(s)
                }
                Err(e) => {
                    vad_unavailable = Some(format!("VAD failed to load: {e:#}"));
                    None
                }
            }
        } else {
            vad_unavailable = Some(format!(
                "{} not found -- run `./scripts/fetch-models.sh` from rust/",
                vad_path.display()
            ));
            None
        };
        if let Some(r) = &vad_unavailable {
            tracing::warn!(reason = %r, "VAD unavailable; audio endpoints will report 503 and the server still serves chat");
        }

        let t = std::time::Instant::now();
        let mut whisper_unavailable: Option<String> = None;
        let whisper = match WhisperEngine::load(&model_dir) {
            Ok(engine) => {
                let handle = engine.handle();
                let stt_backend = match &handle {
                    WhisperHandle::Ct2 { .. } => "ct2",
                    WhisperHandle::WhisperCpp { .. } => "whisper.cpp",
                    WhisperHandle::Parakeet { .. } => "parakeet-tdt",
                    WhisperHandle::Voxtral { .. } => "voxtral-realtime",
                };
                tracing::info!(
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    backend = stt_backend,
                    ep = handle.ep().unwrap_or("whisper.cpp-internal"),
                    "Whisper loaded"
                );
                Some(engine)
            }
            Err(e) => {
                let r = format!("load whisper from {}: {e:#}", model_dir.display());
                tracing::warn!(reason = %r, "Whisper unavailable; transcription endpoints will report 503 and the server still serves chat");
                whisper_unavailable = Some(r);
                None
            }
        };

        let t = std::time::Instant::now();
        let kokoro = load_kokoro(&model_dir);
        if kokoro.is_some() {
            tracing::info!(
                elapsed_ms = t.elapsed().as_millis() as u64,
                "Kokoro loaded; kokoro TTS ready"
            );
        } else {
            tracing::warn!(
                "kokoro-v1.0.onnx/voices.bin missing or unloadable under {} -- kokoro voices on \
                 /v1/audio/speech disabled",
                model_dir.display()
            );
        }

        let t = std::time::Instant::now();
        let voxtral_tts = match super::tts::voxtral_tts::VoxtralTtsSpeech::prepared(&model_dir) {
            Ok(Some(v)) => {
                tracing::info!(
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    device = v.device_name(),
                    codec = v.codec_device(),
                    "Voxtral TTS loaded; voxtral-tts on /v1/audio/speech ready"
                );
                Some(Arc::new(v))
            }
            Ok(None) => {
                tracing::info!(
                    "no {}/{} link and {} unset -- voxtral-tts on /v1/audio/speech disabled",
                    model_dir.display(),
                    super::tts::voxtral_tts::VOXTRAL_TTS_MODELS_DIR_LINK_NAME,
                    super::tts::voxtral_tts::VOXTRAL_TTS_MODEL_DIR_ENV
                );
                None
            }
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "failed to load Voxtral TTS; voxtral-tts disabled");
                None
            }
        };

        let t = std::time::Instant::now();
        let diar_segmentation = {
            let p = model_dir.join("diarizen-segmentation.onnx");
            if p.exists() {
                match SegmentationModel::load(&p) {
                    Ok(m) => {
                        tracing::info!(
                            elapsed_ms = t.elapsed().as_millis() as u64,
                            "diarization segmentation loaded"
                        );
                        Some(Arc::new(m))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to load diarization segmentation; disabled");
                        None
                    }
                }
            } else {
                tracing::warn!(
                    "diarizen-segmentation.onnx not found at {} -- /v1/audio/diarization and \
                     diarized_json disabled (nix build .#diarizen-segmentation-onnx, or enter \
                     the python dev shell to symlink rust/models)",
                    p.display()
                );
                None
            }
        };

        let t = std::time::Instant::now();
        let diar_embedding = {
            let p = model_dir.join("wespeaker-resnet293-LM.onnx");
            if p.exists() {
                match EmbeddingModel::load(&p) {
                    Ok(m) => {
                        tracing::info!(
                            elapsed_ms = t.elapsed().as_millis() as u64,
                            "speaker embedding loaded"
                        );
                        Some(Arc::new(m))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to load speaker embedding; disabled");
                        None
                    }
                }
            } else {
                tracing::warn!(
                    "wespeaker-resnet293-LM.onnx not found at {} -- speaker embeddings, \
                     diarization speaker labels and /v1/audio/embeddings disabled (run \
                     scripts/fetch-models.sh to enable)",
                    p.display()
                );
                None
            }
        };

        Ok(Self {
            vad_session: vad_session.map(|s| Arc::new(Mutex::new(s))),
            whisper: whisper.map(|e| e.handle()),
            vad_unavailable,
            whisper_unavailable,
            kokoro,
            voxtral_tts,
            diar_segmentation,
            diar_embedding,
            model_dir,
        })
    }

    pub fn whisper(&self) -> Result<&WhisperHandle> {
        self.whisper.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "speech-to-text is unavailable: {}",
                self.whisper_unavailable
                    .as_deref()
                    .unwrap_or("no whisper model loaded")
            )
        })
    }

    pub fn vad(&self) -> Result<Arc<Mutex<ort::session::Session>>> {
        self.vad_session.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "voice activity detection is unavailable: {}",
                self.vad_unavailable
                    .as_deref()
                    .unwrap_or("no VAD model loaded")
            )
        })
    }

    pub fn whisper_opt(&self) -> Option<&WhisperHandle> {
        self.whisper.as_ref()
    }
}

pub fn get_or_init() -> Result<Arc<Models>> {
    Models::get_or_init()
}

const ORT_INIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

fn build_vad_session(vad_path: &std::path::Path) -> Result<ort::session::Session> {
    ort::session::Session::builder()
        .map_err(crate::vad::ort_err)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(crate::vad::ort_err)?
        .with_intra_threads(1)
        .map_err(crate::vad::ort_err)?
        .commit_from_file(vad_path)
        .map_err(crate::vad::ort_err)
        .with_context(|| format!("load {}", vad_path.display()))
}

fn build_vad_session_with_timeout(vad_path: &std::path::Path) -> Result<ort::session::Session> {
    let (tx, rx) = std::sync::mpsc::channel();
    let path = vad_path.to_path_buf();
    std::thread::Builder::new()
        .name("ort-init".into())
        .spawn(move || {
            let _ = tx.send(build_vad_session(&path));
        })
        .context("spawn ort init thread")?;
    match rx.recv_timeout(ORT_INIT_TIMEOUT) {
        Ok(result) => result.with_context(|| {
            format!(
                "onnxruntime failed to load (ORT_DYLIB_PATH={})",
                std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into())
            )
        }),
        Err(_) => anyhow::bail!(
            "onnxruntime init did not complete within {}s -- libonnxruntime could not be \
             loaded or hung during dlopen; set ORT_DYLIB_PATH to a valid onnxruntime dylib \
             (currently {}) or launch from the speaches-plus dev shell",
            ORT_INIT_TIMEOUT.as_secs(),
            std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into())
        ),
    }
}

fn load_kokoro(model_dir: &std::path::Path) -> Option<KokoroHandle> {
    match super::tts::prepared_handle(model_dir) {
        Ok(handle) => handle,
        Err(e) => {
            tracing::warn!(error = ?e, "failed to load kokoro TTS; disabled");
            None
        }
    }
}

fn model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(super::defaults::env::SPEACHES_PLUS_MODELS) {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("speaches-models-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create scratch dir");
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_corrupt_missing_or_voiceless_kokoro_disables_kokoro_instead_of_aborting_boot() {
        let s = Scratch::new("corrupt-voices");
        std::fs::write(s.0.join("kokoro-v1.0.onnx"), b"placeholder onnx").unwrap();
        std::fs::write(s.0.join("voices.bin"), b"not a zip archive").unwrap();

        let err = match crate::tts::prepared_handle(&s.0) {
            Ok(_) => panic!("corrupt voices.bin must error"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("zip"),
            "unexpected error: {err:#}"
        );

        assert!(load_kokoro(&s.0).is_none());
        {
            let s = Scratch::new("no-kokoro");
            assert!(crate::tts::prepared_handle(&s.0).unwrap().is_none());
            assert!(load_kokoro(&s.0).is_none());
        }
        {
            let s = Scratch::new("no-voices");
            std::fs::write(s.0.join("kokoro-v1.0.onnx"), b"placeholder onnx").unwrap();
            assert!(crate::tts::prepared_handle(&s.0).unwrap().is_none());
            assert!(load_kokoro(&s.0).is_none());
        }
    }

}
