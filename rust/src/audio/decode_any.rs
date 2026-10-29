use std::sync::OnceLock;

use super::avdecode::decode_via_symphonia;
use super::ogg_opus;
use super::types::{
    BYTES_PER_S16, DEFAULT_MAX_DECODE_SECONDS, ENV_MAX_DECODE_SECONDS, MIME_RAW, MIME_RAW_PCM,
    S16_SCALE, TARGET_SAMPLE_RATE,
};
use super::wav::decode_wav_to_16k_mono;
use super::webm_opus;

pub fn max_decode_samples_from(raw: Option<&str>) -> usize {
    let secs = match raw.map(str::trim) {
        None | Some("") => DEFAULT_MAX_DECODE_SECONDS,
        Some(v) if v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("unlimited") => 0.0,
        Some(v) => match v.parse::<f64>() {
            Ok(s) if s.is_finite() && s >= 0.0 => s,
            _ => DEFAULT_MAX_DECODE_SECONDS,
        },
    };
    if secs <= 0.0 {
        return usize::MAX;
    }
    let n = secs * TARGET_SAMPLE_RATE as f64;
    if n >= usize::MAX as f64 {
        usize::MAX
    } else {
        n as usize
    }
}

fn max_decode_samples() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        max_decode_samples_from(std::env::var(ENV_MAX_DECODE_SECONDS).ok().as_deref())
    })
}

pub fn enforce_max_decode_samples(n_samples: usize, cap: usize) -> anyhow::Result<()> {
    if n_samples > cap {
        anyhow::bail!(
            "decoded audio is {:.1}s, over the {:.1}s limit (raise {} or send a shorter clip)",
            n_samples as f64 / TARGET_SAMPLE_RATE as f64,
            cap as f64 / TARGET_SAMPLE_RATE as f64,
            ENV_MAX_DECODE_SECONDS
        );
    }
    Ok(())
}

pub fn decode_any_to_16k_mono(bytes: &[u8], mime: Option<&str>) -> anyhow::Result<Vec<f32>> {
    let cap = max_decode_samples();
    let mime_lc = mime.map(|s| s.trim().to_ascii_lowercase());
    if matches!(mime_lc.as_deref(), Some(MIME_RAW_PCM) | Some(MIME_RAW)) {
        enforce_max_decode_samples(bytes.len() / BYTES_PER_S16, cap)?;
        let mut out = Vec::with_capacity(bytes.len() / BYTES_PER_S16);
        for ch in bytes.chunks_exact(BYTES_PER_S16) {
            let s = i16::from_le_bytes([ch[0], ch[1]]);
            out.push(s as f32 / S16_SCALE);
        }
        return Ok(out);
    }
    let samples = if let Ok(samples) = decode_wav_to_16k_mono(bytes) {
        samples
    } else if ogg_opus::is_ogg_opus(bytes) {
        ogg_opus::decode_ogg_opus_to_16k_mono(bytes)?
    } else if webm_opus::is_webm(bytes) {
        webm_opus::decode_webm_opus_to_16k_mono(bytes).or_else(|opus_error| {
            decode_via_symphonia(bytes).map_err(|error| {
                anyhow::anyhow!(
                    "WebM/Matroska audio decode failed: {error:#}; Opus decoder: {opus_error:#}"
                )
            })
        })?
    } else {
        decode_via_symphonia(bytes)?
    };
    enforce_max_decode_samples(samples.len(), cap)?;
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tone(samples: &[f32]) {
        assert!(
            (7_000..=9_000).contains(&samples.len()),
            "half-second tone: {} samples",
            samples.len()
        );
        assert!(samples.iter().all(|s| s.is_finite()));
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        assert!((0.05..0.15).contains(&rms), "decoded sine RMS: {rms}");
    }

    #[test]
    fn webm_vorbis_and_opus_both_decode() {
        let bytes = include_bytes!("../../tests/data/tone-vorbis.webm");
        assert!(webm_opus::is_webm(bytes));
        assert!(webm_opus::decode_webm_opus_to_16k_mono(bytes).is_err());
        assert_tone(&decode_any_to_16k_mono(bytes, Some("audio/webm")).unwrap());
        {
            let bytes = include_bytes!("../../tests/data/tone-opus.webm");
            assert_tone(&decode_any_to_16k_mono(bytes, Some("video/webm")).unwrap());
        }
    }

    #[test]
    fn webm_without_audio_reports_general_decoder_failure() {
        let bytes = include_bytes!("../../tests/data/video-only.webm");
        let error = decode_any_to_16k_mono(bytes, None).unwrap_err().to_string();
        assert!(error.contains("no audio track"), "{error}");
    }

    #[test]
    fn decode_cap_defaults_parses_seconds_disables_on_zero_and_falls_back_on_garbage() {
        assert_eq!(max_decode_samples_from(None), 1_800 * 16_000);
        assert_eq!(max_decode_samples_from(Some("  ")), 1_800 * 16_000);
        {
            assert_eq!(max_decode_samples_from(Some("60")), 60 * 16_000);
            assert_eq!(max_decode_samples_from(Some("0.5")), 8_000);
        }
        {
            assert_eq!(max_decode_samples_from(Some("0")), usize::MAX);
            assert_eq!(max_decode_samples_from(Some("off")), usize::MAX);
            assert_eq!(max_decode_samples_from(Some("unlimited")), usize::MAX);
        }
        {
            assert_eq!(max_decode_samples_from(Some("abc")), 1_800 * 16_000);
            assert_eq!(max_decode_samples_from(Some("-5")), 1_800 * 16_000);
            assert_eq!(max_decode_samples_from(Some("inf")), 1_800 * 16_000);
        }
    }

    #[test]
    fn enforce_rejects_only_over_cap_including_raw_pcm_before_decode() {
        assert!(enforce_max_decode_samples(16_000, 16_000).is_ok());
        let err = enforce_max_decode_samples(16_001, 16_000).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("over the"), "{msg}");
        assert!(msg.contains(ENV_MAX_DECODE_SECONDS), "{msg}");
        {
            let bytes = vec![0u8; 4 * BYTES_PER_S16];
            assert!(decode_any_to_16k_mono(&bytes, Some(MIME_RAW_PCM)).is_ok());
            assert!(enforce_max_decode_samples(bytes.len() / BYTES_PER_S16, 2).is_err());
        }
    }

}
