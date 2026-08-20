use anyhow::{anyhow, Context};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;

use crate::audio::g711;

const TARGET_HZ: usize = 16_000;
const SCALE_I16: f32 = 32_768.0;

#[derive(Debug)]
pub enum IngestError {
    Unsupported(String),
    Decode(anyhow::Error),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::Unsupported(fmt) => write!(f, "unsupported input_audio_format: {fmt}"),
            IngestError::Decode(err) => write!(f, "ingest decode: {err:#}"),
        }
    }
}

impl std::error::Error for IngestError {}

#[derive(Debug, Clone, Copy)]
enum WireCodec {
    Pcm16Le,
    Ulaw,
    Alaw,
}

#[derive(Debug)]
pub struct WsAudioIngest {
    codec: WireCodec,
    src_hz: usize,

    src_position: f64,
    last_sample: f32,
}

impl WsAudioIngest {
    pub fn new(format: &str) -> Result<Self, IngestError> {
        let (codec, src_hz) = match format {
            "pcm16" | "" => (WireCodec::Pcm16Le, 24_000),
            "pcm16_8k" => (WireCodec::Pcm16Le, 8_000),
            "pcm16_16k" => (WireCodec::Pcm16Le, 16_000),
            "pcm16_24k" => (WireCodec::Pcm16Le, 24_000),
            "pcm16_44k1" => (WireCodec::Pcm16Le, 44_100),
            "pcm16_48k" => (WireCodec::Pcm16Le, 48_000),
            "g711_ulaw" => (WireCodec::Ulaw, 8_000),
            "g711_alaw" => (WireCodec::Alaw, 8_000),
            other => return Err(IngestError::Unsupported(other.to_string())),
        };
        Ok(Self {
            codec,
            src_hz,
            src_position: 0.0,
            last_sample: 0.0,
        })
    }

    pub fn ingest_b64(&mut self, b64: &str) -> Result<Vec<f32>, IngestError> {
        let bytes = B64
            .decode(b64.trim())
            .context("base64 decode")
            .map_err(IngestError::Decode)?;
        let f32_in = self.decode_bytes(&bytes)?;
        if self.src_hz == TARGET_HZ {
            self.last_sample = *f32_in.last().unwrap_or(&self.last_sample);
            return Ok(f32_in);
        }
        Ok(self.linear_resample(&f32_in))
    }

    fn decode_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>, IngestError> {
        match self.codec {
            WireCodec::Pcm16Le => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(IngestError::Decode(anyhow!(
                        "PCM16 payload has odd byte count: {}",
                        bytes.len()
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / SCALE_I16)
                    .collect())
            }
            WireCodec::Ulaw => Ok(bytes
                .iter()
                .map(|&b| g711::ulaw_decode_byte(b) as f32 / SCALE_I16)
                .collect()),
            WireCodec::Alaw => Ok(bytes
                .iter()
                .map(|&b| g711::alaw_decode_byte(b) as f32 / SCALE_I16)
                .collect()),
        }
    }

    fn linear_resample(&mut self, src: &[f32]) -> Vec<f32> {
        if src.is_empty() {
            return Vec::new();
        }
        let ratio = TARGET_HZ as f64 / self.src_hz as f64;
        let out_cap = ((src.len() as f64) * ratio).ceil() as usize + 1;
        let mut out = Vec::with_capacity(out_cap);
        let mut pos = self.src_position;
        let step = 1.0 / ratio;
        let mut last = self.last_sample;
        while pos < src.len() as f64 {
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(src.len() - 1);
            let frac = (pos - lo as f64) as f32;
            let s = if lo >= src.len() {
                last
            } else {
                let a = src[lo];
                let b = src[hi];
                a + (b - a) * frac
            };
            out.push(s);
            last = s;
            pos += step;
        }
        self.src_position = pos - src.len() as f64;
        self.last_sample = last;
        out
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;
    use crate::realtime::fuzz::Lcg;

    const FORMATS: &[&str] = &[
        "pcm16",
        "pcm16_8k",
        "pcm16_16k",
        "pcm16_24k",
        "pcm16_44k1",
        "pcm16_48k",
        "g711_ulaw",
        "g711_alaw",
        "",
        "opus",
        "wav",
        "pcm32",
        "../../etc/passwd",
        "💀",
    ];

    fn rand_bytes(rng: &mut Lcg, len: usize) -> Vec<u8> {
        (0..len).map(|_| (rng.next() & 0xFF) as u8).collect()
    }

    fn fuzz_one(seed: u64, steps: usize) -> Result<(), String> {
        let mut rng = Lcg::new(seed);
        let format = FORMATS[(rng.next() as usize) % FORMATS.len()];
        let mut ing = match WsAudioIngest::new(format) {
            Ok(i) => i,
            Err(_) => return Ok(()),
        };
        for step in 0..steps {
            let len = (rng.next() as usize) % 4096;
            let bytes = rand_bytes(&mut rng, len);
            let b64 = B64.encode(&bytes);

            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ing.ingest_b64(&b64)));
            if result.is_err() {
                return Err(format!("panic at seed={seed} step={step} format={format}"));
            }

            if !ing.last_sample.is_finite() {
                return Err(format!(
                    "non-finite carry sample at seed={seed} step={step} format={format}"
                ));
            }
            if !ing.src_position.is_finite() {
                return Err(format!(
                    "non-finite src_position at seed={seed} step={step} format={format}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn ws_audio_in_fuzz_5000_steps_seed_0() {
        fuzz_one(0, 5000).unwrap();
    }

    #[test]
    fn ws_audio_in_fuzz_seed_diversity() {
        for seed in [1u64, 7, 42, 99, 2024, 0xDEAD_BEEF, 0xC0FFEE] {
            fuzz_one(seed, 1000).unwrap_or_else(|err| panic!("seed={seed}: {err}"));
        }
    }

    #[test]
    fn ws_audio_in_fuzz_truncated_base64() {
        let mut ing = WsAudioIngest::new("pcm16").unwrap();
        for s in ["A", "AB", "ABC", "ABCD=", "===", "@@@@", "AB CD"] {
            let _ = ing.ingest_b64(s);
        }
    }

    #[test]
    fn ws_audio_in_fuzz_extreme_payloads() {
        let mut ing = WsAudioIngest::new("pcm16").unwrap();
        let bytes: Vec<u8> = (0..1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
        let b64 = B64.encode(&bytes);
        let out = ing.ingest_b64(&b64).unwrap();

        let in_samples = bytes.len() / 2;
        let expected = (in_samples as f64 * 16_000.0 / 24_000.0).round() as isize;
        assert!((out.len() as isize - expected).abs() < 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm16_b64(samples: &[i16]) -> String {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        B64.encode(bytes)
    }

    #[test]
    fn passthrough_at_16k_uses_no_resampler() {
        let mut ing = WsAudioIngest::new("pcm16_16k").unwrap();
        let in_samples: Vec<i16> = (0..16).collect();
        let out = ing.ingest_b64(&pcm16_b64(&in_samples)).unwrap();
        assert_eq!(out.len(), in_samples.len());
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[15] - (15.0 / SCALE_I16)).abs() < 1e-6);
    }

    #[test]
    fn resample_24k_to_16k_reduces_sample_count() {
        let mut ing = WsAudioIngest::new("pcm16").unwrap();
        let in_samples: Vec<i16> = (0..240).map(|i| (i * 100) as i16).collect();
        let out = ing.ingest_b64(&pcm16_b64(&in_samples)).unwrap();
        let expected = (240.0_f64 * (16_000.0 / 24_000.0)).round() as usize;
        assert!(
            out.len() as isize - expected as isize <= 2,
            "got {} samples, expected ~{expected}",
            out.len()
        );
    }

    #[test]
    fn resample_8k_to_16k_doubles_sample_count() {
        let mut ing = WsAudioIngest::new("pcm16_8k").unwrap();
        let in_samples: Vec<i16> = (0..80).map(|i| (i * 100) as i16).collect();
        let out = ing.ingest_b64(&pcm16_b64(&in_samples)).unwrap();
        assert!(
            (out.len() as isize - 160).abs() <= 2,
            "got {}, expected ~160",
            out.len()
        );
    }

    #[test]
    fn ulaw_decodes_and_upsamples() {
        let mut ing = WsAudioIngest::new("g711_ulaw").unwrap();

        let bytes = vec![0xFFu8; 80];
        let out = ing.ingest_b64(&B64.encode(&bytes)).unwrap();
        assert!(
            (out.len() as isize - 160).abs() <= 2,
            "got {}, expected ~160",
            out.len()
        );
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn alaw_decodes_and_upsamples() {
        let mut ing = WsAudioIngest::new("g711_alaw").unwrap();

        let bytes = vec![0xD5u8; 80];
        let out = ing.ingest_b64(&B64.encode(&bytes)).unwrap();
        assert!(
            (out.len() as isize - 160).abs() <= 2,
            "got {}, expected ~160",
            out.len()
        );
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn unsupported_format_is_reported() {
        let err = WsAudioIngest::new("opus").unwrap_err();
        match err {
            IngestError::Unsupported(s) => assert!(s.contains("opus")),
            other => panic!("expected Unsupported, got {other}"),
        }
    }

    #[test]
    fn odd_byte_count_is_reported_for_pcm16() {
        let mut ing = WsAudioIngest::new("pcm16").unwrap();
        let bad = B64.encode([0xAA, 0xBB, 0xCC]);
        let err = ing.ingest_b64(&bad).unwrap_err();
        assert!(matches!(err, IngestError::Decode(_)));
    }
}
