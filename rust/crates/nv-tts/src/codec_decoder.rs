use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use nv_omni::vocoder::{Vocoder, NUM_CODEBOOKS, SAMPLES_PER_FRAME, SAMPLE_RATE_HZ};
use nv_weights::WeightLoader;

pub const CODEC_NUM_QUANTIZERS: usize = NUM_CODEBOOKS;
pub const CODEC_CODEBOOK_SIZE: usize = 2048;
pub const CODEC_SAMPLES_PER_FRAME: usize = SAMPLES_PER_FRAME;
pub const CODEC_SAMPLE_RATE: u32 = SAMPLE_RATE_HZ as u32;

pub struct Qwen3TtsCodecVocoder {
    inner: Vocoder,
}

impl Qwen3TtsCodecVocoder {
    pub fn from_speech_tokenizer_shard(shard: &Path, device: &Device) -> Result<Self> {
        let loader = WeightLoader::open_file(shard, device)
            .map_err(|e| anyhow!("open {}: {e}", shard.display()))?;
        let inner = Vocoder::from_qwen3_weights(&loader, device, DType::F32)?;
        Ok(Self { inner })
    }

    pub fn from_model_dir(dir: &Path, device: &Device) -> Result<Self> {
        let shard = dir.join("speech_tokenizer").join("model.safetensors");
        if !shard.is_file() {
            anyhow::bail!(
                "no speech_tokenizer/model.safetensors under {}",
                dir.display()
            );
        }
        Self::from_speech_tokenizer_shard(&shard, device)
    }

    pub fn sample_rate(&self) -> u32 {
        CODEC_SAMPLE_RATE
    }

    pub fn samples_per_frame(&self) -> usize {
        CODEC_SAMPLES_PER_FRAME
    }

    pub fn decode(&self, frames: &[[u32; CODEC_NUM_QUANTIZERS]]) -> Result<Vec<f32>> {
        self.inner.decode(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_decodes_when_cached() {
        let Some(dir) = crate::model_gate::require("codec_decoder::loads_and_decodes_when_cached")
        else {
            return;
        };
        let voc = Qwen3TtsCodecVocoder::from_model_dir(&dir, &Device::Cpu).expect("load");
        let frames = vec![[100u32; CODEC_NUM_QUANTIZERS]; 4];
        let pcm = voc.decode(&frames).expect("decode");
        assert_eq!(pcm.len(), 4 * CODEC_SAMPLES_PER_FRAME);
        let peak = pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak <= 1.0);
    }

    #[test]
    fn rejects_out_of_range_codes() {
        let Some(dir) = crate::model_gate::require("codec_decoder::rejects_out_of_range_codes")
        else {
            return;
        };
        let voc = Qwen3TtsCodecVocoder::from_model_dir(&dir, &Device::Cpu).expect("load");
        let mut bad = [0u32; CODEC_NUM_QUANTIZERS];
        bad[3] = CODEC_CODEBOOK_SIZE as u32;
        let err = voc
            .decode(&[[0u32; CODEC_NUM_QUANTIZERS], bad])
            .expect_err("must reject");
        assert!(format!("{err}").contains("codebook_size"), "{err}");
    }

    #[test]
    fn from_model_dir_errors_without_speech_tokenizer() {
        let err = Qwen3TtsCodecVocoder::from_model_dir(
            Path::new("/tmp/__no_such_qwen3_tts_dir__"),
            &Device::Cpu,
        )
        .err()
        .expect("must fail");
        assert!(format!("{err}").contains("speech_tokenizer"), "{err}");
    }
}
