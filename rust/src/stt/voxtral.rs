use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use nv_models::voxtral::{VoxtralRealtime, SAMPLES_PER_TOKEN, SAMPLE_RATE};

pub const VOXTRAL_MODEL_DIR_ENV: &str = "VOXTRAL_MODEL_DIR";
pub const VOXTRAL_MODELS_DIR_LINK_NAME: &str = "voxtral-realtime";
pub const VOXTRAL_HUB_REPO: &str = "models--mistralai--Voxtral-Mini-4B-Realtime-2602";
pub const VOXTRAL_MODEL_ID: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602";
pub const VOXTRAL_LORA_DIR_ENV: &str = "NV_VOXTRAL_LORA_DIR";

pub const CHUNK_SECONDS_KEEP_THE_DECODER_UNDER_ITS_SLIDING_WINDOW: usize = 480;

pub const A_NON_BLOCKING_DEVICE_STREAM_IS_WHAT_THE_GRAPHED_DECODE_STEP_CAPTURES: &str =
    "cuStreamBeginCapture refuses the legacy default stream that Device::new_cuda hands candle; \
     new_cuda_with_stream puts every candle op on a CU_STREAM_NON_BLOCKING stream, the same one \
     nv-layers' current_stream returns, so the whole step lands in one capture";

pub fn model_dir(models_dir: &Path, override_dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(d) = override_dir {
        if d.join(nv_models::voxtral::HF_CONFIG_FILE).is_file() {
            return Ok(d);
        }
        return Err(anyhow!(
            "{VOXTRAL_MODEL_DIR_ENV}={} holds no {}",
            d.display(),
            nv_models::voxtral::HF_CONFIG_FILE
        ));
    }
    let linked = models_dir.join(VOXTRAL_MODELS_DIR_LINK_NAME);
    if linked.join(nv_models::voxtral::HF_CONFIG_FILE).is_file() {
        return Ok(linked);
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(VOXTRAL_HUB_REPO)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    entries.sort();
    entries
        .into_iter()
        .find(|p| p.join(nv_models::voxtral::HF_WEIGHTS_FILE).is_file())
        .ok_or_else(|| {
            anyhow!(
                "no Voxtral checkpoint: set {VOXTRAL_MODEL_DIR_ENV} to a directory holding \
                 config.json + tekken.json + model.safetensors, link it as {}/{}, or fetch \
                 {VOXTRAL_MODEL_ID} into the hub cache ({})",
                models_dir.display(),
                VOXTRAL_MODELS_DIR_LINK_NAME,
                snaps.display()
            )
        })
}

pub fn serving_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda_with_stream(0) {
        let _ = A_NON_BLOCKING_DEVICE_STREAM_IS_WHAT_THE_GRAPHED_DECODE_STEP_CAPTURES;
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

pub const SERVE_BACKEND_ENV: &str = "NV_SERVE_BACKEND";

pub const WGPU_SERVES_WHEN_ASKED_FOR_OR_WHEN_NO_CUDA_DEVICE_EXISTS: &str =
    "NV_SERVE_BACKEND=wgpu selects the Vulkan twin outright and a missing wgpu adapter is then \
     an error; without the variable the CUDA port serves wherever a CUDA device exists, and a \
     box without one tries the wgpu twin before falling back to the candle CPU path";

pub fn wgpu_backend_requested() -> bool {
    std::env::var(SERVE_BACKEND_ENV).ok().as_deref() == Some("wgpu")
}

enum Engine {
    Candle(VoxtralRealtime),
    #[cfg(feature = "wgpu")]
    Wgpu(nv_models::voxtral_wgpu::VoxtralWgpu),
}

pub struct VoxtralStt {
    model: Mutex<Engine>,
    dir: PathBuf,
    device_name: &'static str,
}

impl VoxtralStt {
    pub fn load(dir: &Path) -> Result<Self> {
        let _ = WGPU_SERVES_WHEN_ASKED_FOR_OR_WHEN_NO_CUDA_DEVICE_EXISTS;
        let device = serving_device();
        let adapter = std::env::var_os(VOXTRAL_LORA_DIR_ENV).map(PathBuf::from);
        #[cfg(feature = "wgpu")]
        if wgpu_backend_requested() || !device.is_cuda() {
            match nv_models::voxtral_wgpu::VoxtralWgpu::load_dir_with_adapter(
                dir,
                adapter.as_deref(),
            ) {
                Ok(model) => {
                    if let Some(a) = &adapter {
                        tracing::info!(adapter = %a.display(), "voxtral LoRA adapter merged into the served weights");
                    }
                    return Ok(Self {
                        model: Mutex::new(Engine::Wgpu(model)),
                        dir: dir.to_path_buf(),
                        device_name: "wgpu",
                    });
                }
                Err(e) if wgpu_backend_requested() => {
                    return Err(e.context(format!(
                        "load Voxtral realtime on wgpu ({SERVE_BACKEND_ENV}=wgpu) from {}",
                        dir.display()
                    )))
                }
                Err(e) => {
                    tracing::warn!(error = %format!("{e:#}"), "voxtral wgpu twin unavailable; serving on candle");
                }
            }
        }
        let device_name = if device.is_cuda() {
            "cuda"
        } else if device.is_metal() {
            "metal"
        } else {
            "cpu"
        };
        let model = VoxtralRealtime::load_dir_with_adapter(dir, &device, adapter.as_deref())
            .with_context(|| format!("load Voxtral realtime from {}", dir.display()))?;
        if let Some(a) = &adapter {
            tracing::info!(adapter = %a.display(), "voxtral LoRA adapter merged into the served weights");
        }
        Ok(Self {
            model: Mutex::new(Engine::Candle(model)),
            dir: dir.to_path_buf(),
            device_name,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    pub fn transcribe(&self, audio_16k_mono: &[f32]) -> Result<(String, Vec<u32>)> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow!("voxtral model mutex poisoned"))?;
        let chunk = CHUNK_SECONDS_KEEP_THE_DECODER_UNDER_ITS_SLIDING_WINDOW * SAMPLE_RATE;
        let mut texts = Vec::new();
        let mut tokens = Vec::new();
        for piece in audio_16k_mono.chunks(chunk) {
            if piece.len() < SAMPLES_PER_TOKEN {
                continue;
            }
            let (text, toks) = match &mut *model {
                Engine::Candle(m) => m.transcribe(piece)?,
                #[cfg(feature = "wgpu")]
                Engine::Wgpu(m) => m.transcribe(piece)?,
            };
            let text = text.trim().to_string();
            if !text.is_empty() {
                texts.push(text);
            }
            tokens.extend(toks);
        }
        Ok((texts.join(" "), tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Root(PathBuf);

    impl Root {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Root {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn model_dir_prefers_the_override_then_the_models_dir_link() {
        let root = std::env::temp_dir().join(format!(
            "voxtral-model-dir-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let dir = Root(root);
        let link = dir.path().join(VOXTRAL_MODELS_DIR_LINK_NAME);
        std::fs::create_dir(&link).unwrap();
        std::fs::write(link.join("config.json"), b"{}").unwrap();
        assert_eq!(model_dir(dir.path(), None).unwrap(), link);
        let other = dir.path().join("other");
        std::fs::create_dir(&other).unwrap();
        assert!(model_dir(dir.path(), Some(other.clone())).is_err());
        std::fs::write(other.join("config.json"), b"{}").unwrap();
        assert_eq!(model_dir(dir.path(), Some(other.clone())).unwrap(), other);
    }
}
