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
                let stt_backend = match engine.handle() {
                    WhisperHandle::Ct2 { .. } => "ct2",
                    WhisperHandle::WhisperCpp { .. } => "whisper.cpp",
                    WhisperHandle::Parakeet { .. } => "parakeet-tdt",
                };
                tracing::info!(
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    backend = stt_backend,
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
            tracing::info!("kokoro TTS disabled (missing or unloadable model/voices.bin)");
        }

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
                tracing::info!(
                    "diarization segmentation not present at {} (run scripts/export-diarizen-onnx.py to enable)",
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
                tracing::info!(
                    "speaker embedding not present at {} (run scripts/fetch-models.sh to enable)",
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

    pub fn audio_disabled_reason(&self) -> Option<&str> {
        self.vad_unavailable
            .as_deref()
            .or(self.whisper_unavailable.as_deref())
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
    fn corrupt_voices_bin_disables_kokoro_instead_of_aborting_boot() {
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
    }

    #[test]
    fn missing_kokoro_model_disables_kokoro() {
        let s = Scratch::new("no-kokoro");
        assert!(crate::tts::prepared_handle(&s.0).unwrap().is_none());
        assert!(load_kokoro(&s.0).is_none());
    }

    #[test]
    fn kokoro_model_without_voices_disables_kokoro() {
        let s = Scratch::new("no-voices");
        std::fs::write(s.0.join("kokoro-v1.0.onnx"), b"placeholder onnx").unwrap();
        assert!(crate::tts::prepared_handle(&s.0).unwrap().is_none());
        assert!(load_kokoro(&s.0).is_none());
    }
}
