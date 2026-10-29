use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use nv_models::voxtral_tts::{SeededNoise, VoxtralTts, END_AUDIO, MAX_FRAMES_DEFAULT, PARAMS_FILE};
use nv_models::voxtral_tts_codec::{CodecState, VoxtralTtsCodec, CODEC_SAMPLE_RATE};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::oapi::audio_speech::{AudioSpeech, SpeechOptions, UnknownVoice};
use crate::oapi::audio_speech_nvtts::is_default_voice_alias;
use crate::stt::voxtral::serving_device;

pub const VOXTRAL_TTS_MODEL_ID: &str = "mistralai/Voxtral-4B-TTS-2603";
pub const VOXTRAL_TTS_ALIAS: &str = "voxtral-tts";
pub const VOXTRAL_TTS_MODELS_DIR_LINK_NAME: &str = "voxtral-tts";
pub const VOXTRAL_TTS_MODEL_DIR_ENV: &str = nv_models::voxtral_tts::MODEL_DIR_ENV;
pub const DEFAULT_VOICE: &str = "neutral_female";
pub const SEGMENT_CHARS_GROUP_SENTENCES_UNDER_ONE_PREFILL: usize = 160;
pub const FIXED_SEED_UNTIL_THE_ROUTE_CARRIES_ONE: u64 = 0x5EED_2603;

pub fn model_dir(models_dir: &Path, override_dir: Option<PathBuf>) -> Result<Option<PathBuf>> {
    if let Some(d) = override_dir {
        ensure!(
            d.join(PARAMS_FILE).is_file(),
            "{VOXTRAL_TTS_MODEL_DIR_ENV}={} holds no {PARAMS_FILE}",
            d.display()
        );
        return Ok(Some(d));
    }
    let linked = models_dir.join(VOXTRAL_TTS_MODELS_DIR_LINK_NAME);
    Ok(linked.join(PARAMS_FILE).is_file().then_some(linked))
}

fn device_name(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

enum Codec {
    Candle(VoxtralTtsCodec),
    #[cfg(feature = "wgpu")]
    Wgpu(nv_models::voxtral_tts_codec_wgpu::VoxtralTtsCodecWgpu),
}

enum CodecStream {
    Candle(CodecState),
    #[cfg(feature = "wgpu")]
    Wgpu(nv_models::voxtral_tts_codec_wgpu::CodecState),
}

impl Codec {
    fn load(dir: &Path, device: &Device) -> Result<(Self, &'static str)> {
        #[cfg(feature = "wgpu")]
        if crate::stt::voxtral::wgpu_backend_requested() {
            let codec = nv_models::voxtral_tts_codec_wgpu::VoxtralTtsCodecWgpu::load(dir)
                .with_context(|| {
                    format!(
                        "load the Voxtral TTS codec on wgpu ({}=wgpu) from {}",
                        crate::stt::voxtral::SERVE_BACKEND_ENV,
                        dir.display()
                    )
                })?;
            return Ok((Self::Wgpu(codec), "wgpu"));
        }
        let codec = VoxtralTtsCodec::load(dir, device)
            .with_context(|| format!("load the Voxtral TTS codec from {}", dir.display()))?;
        Ok((Self::Candle(codec), device_name(device)))
    }

    fn stream(&self) -> CodecStream {
        match self {
            Self::Candle(_) => CodecStream::Candle(CodecState::default()),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => CodecStream::Wgpu(Default::default()),
        }
    }

    fn decode(&self, stream: &mut CodecStream, frames: &[i32]) -> Result<Vec<f32>> {
        match (self, stream) {
            (Self::Candle(c), CodecStream::Candle(s)) => c.decode_frames(s, frames),
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(c), CodecStream::Wgpu(s)) => c.decode_frames(s, frames),
            #[cfg(feature = "wgpu")]
            _ => Err(anyhow::anyhow!("codec stream state belongs to the other backend")),
        }
    }
}

struct Engine {
    model: VoxtralTts,
    codec: Codec,
    voices: Vec<String>,
    dir: PathBuf,
    device_name: &'static str,
    codec_device: &'static str,
    gate: Mutex<()>,
}

impl Engine {
    fn resolve_voice(&self, voice: &str) -> Result<String> {
        let v = voice.trim();
        if is_default_voice_alias(v) {
            return Ok(DEFAULT_VOICE.to_string());
        }
        if self.voices.iter().any(|n| n == v) {
            return Ok(v.to_string());
        }
        Err(UnknownVoice(format!(
            "voice {v:?} is not a Voxtral TTS preset; {VOXTRAL_TTS_MODEL_ID} ships {} presets \
             ({DEFAULT_VOICE} when voice is omitted): {}",
            self.voices.len(),
            self.voices.join(", ")
        ))
        .into())
    }

    fn decode_the_frames_waiting_as_one_chunk(
        &self,
        frames: std::sync::mpsc::Receiver<Vec<i32>>,
        tx: &mpsc::Sender<Result<Vec<f32>>>,
        first_audio: &mut Option<Instant>,
    ) -> Result<bool> {
        let mut state = self.codec.stream();
        while let Ok(mut batch) = frames.recv() {
            while let Ok(more) = frames.try_recv() {
                batch.extend(more);
            }
            let pcm = self.codec.decode(&mut state, &batch)?;
            first_audio.get_or_insert_with(Instant::now);
            if tx.blocking_send(Ok(pcm)).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn run(&self, text: &str, voice: &str, tx: &mpsc::Sender<Result<Vec<f32>>>) -> Result<()> {
        let _gate = self.gate.lock().unwrap_or_else(PoisonError::into_inner);
        let started = Instant::now();
        let n_acoustic = self.model.codebooks().n_acoustic;
        let plan = crate::tts::chunk::plan(text, SEGMENT_CHARS_GROUP_SENTENCES_UNDER_ONE_PREFILL);
        let mut total_frames = 0usize;
        let mut first_audio = None;
        for (i, segment) in plan.chunks.iter().enumerate() {
            let mut noise = SeededNoise::new(FIXED_SEED_UNTIL_THE_ROUTE_CARRIES_ONE ^ i as u64);
            let t = Instant::now();
            let (ftx, frx) = std::sync::mpsc::channel::<Vec<i32>>();
            let (syn, open) = std::thread::scope(|s| {
                let decoder = s.spawn(|| self.decode_the_frames_waiting_as_one_chunk(frx, tx, &mut first_audio));
                let syn = self.model.synthesize_streaming(
                    segment,
                    voice,
                    &mut |_| noise.frame(n_acoustic),
                    MAX_FRAMES_DEFAULT,
                    &mut |f| Ok(f[0] == END_AUDIO || ftx.send(f.iter().map(|&c| c as i32).collect()).is_ok()),
                );
                drop(ftx);
                (syn, decoder.join())
            });
            let open = open.map_err(|_| anyhow::anyhow!("voxtral tts codec thread panicked"))??;
            let syn = syn.with_context(|| format!("voxtral tts segment {i} ({} chars)", segment.len()))?;
            if !open {
                return Ok(());
            }
            let frames = syn.frames.len() - usize::from(syn.ended);
            if !syn.ended {
                warn!(segment = i, frames, "voxtral tts hit MAX_FRAMES before [END_AUDIO]");
            }
            total_frames += frames;
            info!(
                segment = i,
                segments = plan.chunks.len(),
                frames,
                audio_secs = frames as f32 * 0.08,
                gen_secs = t.elapsed().as_secs_f32(),
                "voxtral tts segment synthesized"
            );
        }
        info!(
            voice,
            frames = total_frames,
            ttfa_secs = first_audio.map(|f| (f - started).as_secs_f32()),
            total_secs = started.elapsed().as_secs_f32(),
            "voxtral tts synthesis complete"
        );
        Ok(())
    }
}

#[derive(Clone)]
pub struct VoxtralTtsSpeech(Arc<Engine>);

impl VoxtralTtsSpeech {
    pub fn load(dir: &Path) -> Result<Self> {
        let device = serving_device();
        let t = Instant::now();
        let model = VoxtralTts::load_dir(dir, &device)
            .with_context(|| format!("load Voxtral TTS from {}", dir.display()))?;
        let (codec, codec_device) = Codec::load(dir, &device)?;
        let voices: Vec<String> = model.voices().map(|(n, _)| n.to_string()).collect();
        ensure!(
            voices.iter().any(|v| v == DEFAULT_VOICE),
            "{} lists no {DEFAULT_VOICE} preset among {voices:?}",
            dir.display()
        );
        let engine = Arc::new(Engine {
            model,
            codec,
            voices,
            dir: dir.to_path_buf(),
            device_name: device_name(&device),
            codec_device,
            gate: Mutex::new(()),
        });
        info!(
            dir = %dir.display(),
            device = engine.device_name,
            codec = engine.codec_device,
            voices = engine.voices.len(),
            load_secs = t.elapsed().as_secs_f32(),
            "voxtral tts ready"
        );
        Ok(Self(engine))
    }

    pub fn prepared(models_dir: &Path) -> Result<Option<Self>> {
        let override_dir = std::env::var(VOXTRAL_TTS_MODEL_DIR_ENV)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let Some(dir) = model_dir(models_dir, override_dir)? else {
            return Ok(None);
        };
        let s = Self::load(&dir)?;
        s.warm_up();
        Ok(Some(s))
    }

    fn warm_up(&self) {
        let engine = self.0.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            let (tx, mut rx) = mpsc::channel::<Result<Vec<f32>>>(64);
            match engine.run("Warm up.", DEFAULT_VOICE, &tx) {
                Ok(()) => {
                    drop(tx);
                    let mut samples = 0usize;
                    while let Some(c) = rx.blocking_recv() {
                        samples += c.map_or(0, |c| c.len());
                    }
                    info!(secs = t.elapsed().as_secs_f32(), samples, "voxtral tts warmup complete");
                }
                Err(e) => warn!(error = %format!("{e:#}"), "voxtral tts warmup failed"),
            }
        });
    }

    pub fn dir(&self) -> &Path {
        &self.0.dir
    }

    pub fn device_name(&self) -> &'static str {
        self.0.device_name
    }

    pub fn codec_device(&self) -> &'static str {
        self.0.codec_device
    }
}

#[async_trait::async_trait]
impl AudioSpeech for VoxtralTtsSpeech {
    async fn synthesize(
        &self,
        text: &str,
        voice: &str,
        _opts: &SpeechOptions,
    ) -> Result<mpsc::Receiver<Result<Vec<f32>>>> {
        let voice = self.0.resolve_voice(voice)?;
        let engine = self.0.clone();
        let text = text.to_string();
        let (tx, mut rx) = mpsc::channel::<Result<Vec<f32>>>(4);
        let task = tokio::task::spawn_blocking(move || engine.run(&text, &voice, &tx));
        let Some(first) = rx.recv().await else {
            task.await.context("voxtral tts task panicked")??;
            anyhow::bail!("voxtral tts produced no audio frames before [END_AUDIO]");
        };
        let (out_tx, out_rx) = mpsc::channel::<Result<Vec<f32>>>(4);
        tokio::spawn(async move {
            let mut open = out_tx.send(first).await.is_ok();
            while open {
                match rx.recv().await {
                    Some(c) => open = out_tx.send(c).await.is_ok(),
                    None => break,
                }
            }
            drop(rx);
            let res = match task.await {
                Ok(res) => res,
                Err(e) => Err(anyhow::anyhow!("voxtral tts task panicked: {e}")),
            };
            if let Err(e) = res {
                warn!(error = %format!("{e:#}"), "voxtral tts failed mid-stream; the response fails instead of ending short");
                let _ = out_tx.send(Err(e)).await;
            }
        });
        Ok(out_rx)
    }

    fn sample_rate(&self) -> u32 {
        CODEC_SAMPLE_RATE as u32
    }

    fn model_id(&self) -> Option<String> {
        Some(VOXTRAL_TTS_MODEL_ID.to_string())
    }

    fn voices(&self) -> Vec<String> {
        self.0.voices.clone()
    }

    fn engine(&self) -> &'static str {
        "voxtral-tts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dir_prefers_the_override_then_the_models_dir_link_and_is_absent_otherwise() {
        let root = std::env::temp_dir().join(format!("voxtral-tts-model-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(model_dir(&root, None).unwrap(), None);
        let link = root.join(VOXTRAL_TTS_MODELS_DIR_LINK_NAME);
        std::fs::create_dir(&link).unwrap();
        std::fs::write(link.join(PARAMS_FILE), b"{}").unwrap();
        assert_eq!(model_dir(&root, None).unwrap(), Some(link));
        let other = root.join("other");
        std::fs::create_dir(&other).unwrap();
        assert!(model_dir(&root, Some(other.clone())).is_err());
        std::fs::write(other.join(PARAMS_FILE), b"{}").unwrap();
        assert_eq!(model_dir(&root, Some(other.clone())).unwrap(), Some(other));
        let _ = std::fs::remove_dir_all(&root);
    }
}
