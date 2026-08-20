use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};

use nv_omni::vocoder::{Vocoder, VocoderConfig};
use nv_weights::WeightLoader;

pub const SPEECH_TOKENIZER_SHARD: &str = "speech_tokenizer/model.safetensors";

pub const DECODER_PREFIX: &str = "decoder.";

#[derive(Debug, Clone)]
pub struct VocoderInventory {
    pub shard_path: PathBuf,
    pub decoder_key_count: usize,

    pub pre_transformer_keys: usize,
    pub pre_conv_keys: usize,
    pub quantizer_keys: usize,
    pub upsample_keys: usize,
    pub main_decoder_keys: usize,

    pub upsample_factor: usize,
    pub sample_rate: u32,
    pub frame_rate_hz: f32,
}

impl VocoderInventory {
    pub fn is_real_qwen3_decoder(&self) -> bool {
        self.pre_transformer_keys > 0
            && self.pre_conv_keys > 0
            && self.quantizer_keys > 0
            && self.main_decoder_keys > 0
    }
}

pub fn vocoder_shard_path(model_dir: &Path) -> PathBuf {
    model_dir.join(SPEECH_TOKENIZER_SHARD)
}

pub fn scan_vocoder_dir(model_dir: &Path) -> Result<VocoderInventory> {
    let shard = vocoder_shard_path(model_dir);
    if !shard.is_file() {
        anyhow::bail!(
            "vocoder shard not found at {} (expected file)",
            shard.display()
        );
    }
    let device = Device::Cpu;
    let weights = WeightLoader::open_file(&shard, &device)
        .with_context(|| format!("open vocoder shard {}", shard.display()))?;

    let names = weights.names();
    let decoder_keys: Vec<String> = names
        .into_iter()
        .filter(|n| n.starts_with(DECODER_PREFIX))
        .collect();

    let mut pre_transformer_keys = 0usize;
    let mut pre_conv_keys = 0usize;
    let mut quantizer_keys = 0usize;
    let mut upsample_keys = 0usize;
    let mut main_decoder_keys = 0usize;
    for k in &decoder_keys {
        if k.starts_with("decoder.pre_transformer.") {
            pre_transformer_keys += 1;
        } else if k.starts_with("decoder.pre_conv.") {
            pre_conv_keys += 1;
        } else if k.starts_with("decoder.quantizer.") {
            quantizer_keys += 1;
        } else if k.starts_with("decoder.upsample.") {
            upsample_keys += 1;
        } else if k.starts_with("decoder.decoder.") {
            main_decoder_keys += 1;
        }
    }

    Ok(VocoderInventory {
        shard_path: shard,
        decoder_key_count: decoder_keys.len(),
        pre_transformer_keys,
        pre_conv_keys,
        quantizer_keys,
        upsample_keys,
        main_decoder_keys,
        upsample_factor: 1920,
        sample_rate: 24_000,
        frame_rate_hz: 12.5,
    })
}

#[derive(Debug, Clone)]
pub struct VocoderLoadReport {
    pub inventory: VocoderInventory,
    pub zero_init_fallback: bool,
    pub fallback_reason: Option<String>,
}

pub fn load_from_qwen3_tts(
    model_dir: &Path,
    device: &Device,
    dtype: DType,
) -> Result<(Vocoder, VocoderLoadReport)> {
    let inventory = scan_vocoder_dir(model_dir).with_context(|| {
        format!(
            "scan vocoder weights in {} (looking for {})",
            model_dir.display(),
            SPEECH_TOKENIZER_SHARD
        )
    })?;

    if inventory.is_real_qwen3_decoder() {
        let loader = WeightLoader::open_file(&inventory.shard_path, device)
            .with_context(|| format!("open vocoder shard {}", inventory.shard_path.display()))?;
        match Vocoder::from_qwen3_weights(&loader, device, dtype) {
            Ok(voc) => {
                return Ok((
                    voc,
                    VocoderLoadReport {
                        inventory,
                        zero_init_fallback: false,
                        fallback_reason: None,
                    },
                ));
            }
            Err(err) => {
                let cfg = VocoderConfig {
                    dtype,
                    ..VocoderConfig::default()
                };
                let voc = Vocoder::new(cfg, device)
                    .map_err(|e| anyhow!("zero-init fallback build: {e}"))?;
                return Ok((
                    voc,
                    VocoderLoadReport {
                        inventory,
                        zero_init_fallback: true,
                        fallback_reason: Some(format!(
                            "real-weight load failed, falling back to zero-init: {err}"
                        )),
                    },
                ));
            }
        }
    }

    let cfg = VocoderConfig {
        dtype,
        ..VocoderConfig::default()
    };
    let voc = Vocoder::new(cfg, device).map_err(|e| anyhow!("build zero-init vocoder: {e}"))?;
    let reason = format!(
        "no recognised Qwen3-TTS decoder structure in {} (decoder_key_count={})",
        model_dir.display(),
        inventory.decoder_key_count
    );
    Ok((
        voc,
        VocoderLoadReport {
            inventory,
            zero_init_fallback: true,
            fallback_reason: Some(reason),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cached_snapshot_dir() -> Option<PathBuf> {
        crate::tokenizer::qwen3_tts_cache_dir()
    }

    #[test]
    fn shard_path_is_relative_join() {
        let dir = PathBuf::from("/tmp/fake-snapshot");
        let s = vocoder_shard_path(&dir);
        assert_eq!(s, dir.join("speech_tokenizer/model.safetensors"));
    }

    #[test]
    fn scan_errors_when_shard_missing() {
        let dir = PathBuf::from("/tmp/__definitely_does_not_exist_qwen3");
        let err = scan_vocoder_dir(&dir).expect_err("must fail on missing shard");
        let msg = format!("{err}");
        assert!(msg.contains("vocoder shard not found"), "{msg}");
    }

    #[test]
    #[ignore = "inspection-only: prints pre_transformer.layers.0 keys + shapes"]
    fn inspect_pre_transformer_layer0() {
        let Some(dir) = cached_snapshot_dir() else {
            eprintln!("skip inspect: cache absent");
            return;
        };
        let shard = vocoder_shard_path(&dir);
        let device = Device::Cpu;
        let weights = WeightLoader::open_file(&shard, &device).expect("open");
        let mut names = weights.names();
        names.sort();
        for n in &names {
            if n.starts_with("decoder.pre_transformer.") {
                let shape = weights.shape_of(n).unwrap_or_default();
                let dt = weights.dtype_of(n);
                eprintln!("{n} {shape:?} {dt:?}");
            }
        }
    }

    #[test]
    fn scan_real_cache_detects_decoder_structure() {
        let Some(dir) =
            crate::model_gate::require("vocoder_loader::scan_real_cache_detects_decoder_structure")
        else {
            return;
        };
        let inv = scan_vocoder_dir(&dir).expect("scan cached vocoder shard");
        assert!(inv.is_real_qwen3_decoder(), "{inv:?}");

        assert_eq!(inv.pre_transformer_keys, 93, "{inv:?}");
        assert_eq!(inv.pre_conv_keys, 2, "{inv:?}");
        assert_eq!(inv.quantizer_keys, 36, "{inv:?}");
        assert_eq!(inv.upsample_keys, 22, "{inv:?}");
        assert_eq!(inv.main_decoder_keys, 118, "{inv:?}");
        assert_eq!(inv.decoder_key_count, 271, "{inv:?}");
        assert_eq!(inv.upsample_factor, 1920);
        assert_eq!(inv.sample_rate, 24_000);
    }

    #[test]
    fn load_real_weights_produces_non_silent_pcm() {
        let Some(dir) =
            crate::model_gate::require("vocoder_loader::load_real_weights_produces_non_silent_pcm")
        else {
            return;
        };
        let device = Device::Cpu;
        let (voc, report) = load_from_qwen3_tts(&dir, &device, DType::F32).expect("build vocoder");
        assert!(!report.zero_init_fallback, "report: {:?}", report);
        assert!(report.fallback_reason.is_none());
        assert!(!voc.is_zero_init());

        let frames: Vec<[u32; nv_omni::vocoder::NUM_CODEBOOKS]> = vec![
            [0u32; nv_omni::vocoder::NUM_CODEBOOKS],
            [0u32; nv_omni::vocoder::NUM_CODEBOOKS],
        ];
        let pcm = voc.decode(&frames).expect("decode 2 frames");
        assert_eq!(pcm.len(), 2 * voc.config().upsample_factor());

        let max_abs = pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms = (pcm.iter().map(|s| (s * s) as f64).sum::<f64>() / pcm.len() as f64).sqrt();
        eprintln!(
            "real-vocoder 2-frame decode: samples={} peak={:.4} rms={:.4e}",
            pcm.len(),
            max_abs,
            rms
        );
        assert!(
            max_abs > 0.0,
            "real-weight vocoder must produce non-silent PCM, got max_abs={max_abs}"
        );
    }

    #[test]
    fn load_real_weights_with_random_tokens_produces_audible_pcm() {
        let Some(dir) = crate::model_gate::require(
            "vocoder_loader::load_real_weights_with_random_tokens_produces_audible_pcm",
        ) else {
            return;
        };
        let device = Device::Cpu;
        let (voc, report) = load_from_qwen3_tts(&dir, &device, DType::F32).expect("build vocoder");
        assert!(!report.zero_init_fallback);

        let cb_size = voc.config().codebook_size as u32;
        let t = 32usize;
        let mut frames: Vec<[u32; nv_omni::vocoder::NUM_CODEBOOKS]> = Vec::with_capacity(t);
        let mut seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
        for _ in 0..t {
            let mut row = [0u32; nv_omni::vocoder::NUM_CODEBOOKS];
            for slot in row.iter_mut() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *slot = (seed as u32) % cb_size;
            }
            frames.push(row);
        }
        let pcm = voc.decode(&frames).expect("decode random frames");
        assert_eq!(pcm.len(), t * voc.config().upsample_factor());
        let peak = pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let rms = (pcm.iter().map(|s| (s * s) as f64).sum::<f64>() / pcm.len() as f64).sqrt();
        eprintln!(
            "real-vocoder 32-frame random decode: samples={} ({:.2} s) peak={:.4} rms={:.4e}",
            pcm.len(),
            pcm.len() as f32 / 24_000.0,
            peak,
            rms
        );
        assert!(peak > 0.0, "expected non-silent peak");
        assert!(rms > 0.0, "expected non-zero rms");

        if let Ok(out_path) = std::env::var("NV_TTS_VOCODER_WAV_DUMP") {
            if let Err(e) = dump_wav(&out_path, &pcm, 24_000) {
                eprintln!("WAV dump failed: {e}");
            } else {
                eprintln!("dumped WAV to {out_path}");
            }
        }
    }

    fn dump_wav(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
        use std::io::Write;
        let pcm16: Vec<i16> = samples
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16)
            .collect();
        let data_bytes: u32 = (pcm16.len() as u32).saturating_mul(2);
        let mut buf: Vec<u8> = Vec::with_capacity(44 + data_bytes as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        for s in &pcm16 {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(&buf)
    }
}
