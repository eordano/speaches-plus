use anyhow::{bail, Context};

use super::resample::downmix_and_resample_f32;
use super::types::{S16_SCALE, S24_SCALE, S32_SCALE, TARGET_SAMPLE_RATE};

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
const WAVE_FORMAT_EXTENSIBLE_MIN_FMT_BYTES: usize = 40;
const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] = [
    0x01, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71,
];
const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: [u8; 16] = [
    0x03, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71,
];
const STREAMING_UNBOUNDED_CHUNK_SIZE: u32 = u32::MAX;

fn chunk_body(buf: &[u8], idx: usize) -> &[u8] {
    let size = u32::from_le_bytes([buf[idx + 4], buf[idx + 5], buf[idx + 6], buf[idx + 7]]);
    let start = idx + 8;
    &buf[start..start.saturating_add(size as usize).min(buf.len())]
}

pub fn decode_wav_to_16k_mono(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    if bytes.len() < 12 {
        bail!("wav: too short ({} bytes)", bytes.len());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("wav: not a RIFF/WAVE file");
    }
    let fmt = chunk_body(bytes, find_chunk(bytes, b"fmt ").context("wav: missing fmt chunk")?);
    if fmt.len() < 16 {
        bail!("wav: fmt chunk too small ({})", fmt.len());
    }
    let mut format_tag = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]) as usize;
    let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]) as usize;
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    if format_tag == WAVE_FORMAT_EXTENSIBLE && fmt.len() >= WAVE_FORMAT_EXTENSIBLE_MIN_FMT_BYTES {
        if fmt[24..40] == KSDATAFORMAT_SUBTYPE_PCM {
            format_tag = WAVE_FORMAT_PCM;
        } else if fmt[24..40] == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {
            format_tag = WAVE_FORMAT_IEEE_FLOAT;
        }
    }
    if channels == 0 {
        bail!("wav: zero channels");
    }
    let data = chunk_body(bytes, find_chunk(bytes, b"data").context("wav: missing data chunk")?);
    let samples: Vec<f32> = match (format_tag, bits) {
        (WAVE_FORMAT_PCM, 16) => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / S16_SCALE)
            .collect(),
        (WAVE_FORMAT_PCM, 24) => data
            .chunks_exact(3)
            .map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / S24_SCALE)
            .collect(),
        (WAVE_FORMAT_PCM, 32) => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / S32_SCALE)
            .collect(),
        (WAVE_FORMAT_IEEE_FLOAT, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (WAVE_FORMAT_IEEE_FLOAT, 64) => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        (WAVE_FORMAT_PCM, b) => bail!("wav: unsupported bits_per_sample: {b}"),
        (WAVE_FORMAT_IEEE_FLOAT, b) => bail!("wav: unsupported float bits_per_sample: {b}"),
        (t, _) => bail!("wav: unsupported format tag {t}"),
    };
    Ok(downmix_and_resample_f32(
        &samples,
        channels,
        sample_rate,
        TARGET_SAMPLE_RATE as usize,
    ))
}

pub fn find_chunk(buf: &[u8], tag: &[u8; 4]) -> Option<usize> {
    let mut i = 12;
    while i + 8 <= buf.len() {
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]);
        if &buf[i..i + 4] == tag {
            return Some(i);
        }
        if size == STREAMING_UNBOUNDED_CHUNK_SIZE {
            return None;
        }
        i = i + 8 + size as usize + (size as usize & 1);
    }
    None
}

fn pcm16_round_half_away_from_zero(s: f32) -> i16 {
    if s.is_nan() {
        0
    } else {
        (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16
    }
}

pub fn encode_wav_mono16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_size = (samples.len() as u32).wrapping_mul(2);
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&data_size.wrapping_add(36).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&sample_rate.wrapping_mul(2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&pcm16_round_half_away_from_zero(s).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt_body(tag: u16, channels: u16, rate: u32, bits: u16, subtype: Option<[u8; 16]>) -> Vec<u8> {
        let align = channels * bits / 8;
        let mut f = Vec::new();
        f.extend_from_slice(&tag.to_le_bytes());
        f.extend_from_slice(&channels.to_le_bytes());
        f.extend_from_slice(&rate.to_le_bytes());
        f.extend_from_slice(&(rate * align as u32).to_le_bytes());
        f.extend_from_slice(&align.to_le_bytes());
        f.extend_from_slice(&bits.to_le_bytes());
        if let Some(sub) = subtype {
            f.extend_from_slice(&22u16.to_le_bytes());
            f.extend_from_slice(&bits.to_le_bytes());
            f.extend_from_slice(&0u32.to_le_bytes());
            f.extend_from_slice(&sub);
        }
        f
    }

    fn riff(riff_size: Option<u32>, chunks: &[(&[u8; 4], Option<u32>, &[u8])]) -> Vec<u8> {
        let mut body = b"WAVE".to_vec();
        for (tag, size, payload) in chunks {
            body.extend_from_slice(*tag);
            body.extend_from_slice(&size.unwrap_or(payload.len() as u32).to_le_bytes());
            body.extend_from_slice(payload);
            if payload.len() % 2 == 1 && size.is_none() {
                body.push(0);
            }
        }
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&riff_size.unwrap_or(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn le<T: Copy, const N: usize>(xs: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
        xs.iter().flat_map(|&x| f(x)).collect()
    }

    #[test]
    fn find_chunk_stops_at_an_unbounded_chunk_that_is_not_the_tag() {
        let buf = riff(None, &[(b"junk", Some(u32::MAX), &[1, 2, 3, 4]), (b"data", None, &[0, 0])]);
        assert_eq!(find_chunk(&buf, b"data"), None);
        assert_eq!(find_chunk(&buf, b"junk"), Some(12));
        let fmt = fmt_body(1, 1, 16_000, 16, None);
        let buf = riff(None, &[(b"fmt ", None, &fmt), (b"junk", Some(u32::MAX), &[0; 4]), (b"data", None, &[0, 0])]);
        assert!(decode_wav_to_16k_mono(&buf).unwrap_err().to_string().contains("missing data chunk"));
        assert_eq!(find_chunk(&buf[..12], b"data"), None);
    }

    #[test]
    fn streaming_placeholder_sizes_decode_to_the_end_of_the_buffer() {
        let fmt = fmt_body(1, 1, 16_000, 16, None);
        let pcm = le(&[16_384i16, -16_384, 32_767], i16::to_le_bytes);
        let want = [0.5f32, -0.5, 32_767.0 / 32_768.0];
        for (riff_size, data_size) in [(Some(u32::MAX), Some(u32::MAX)), (None, Some(u32::MAX)), (Some(u32::MAX), None), (Some(7), Some(u32::MAX))] {
            let buf = riff(riff_size, &[(b"fmt ", None, &fmt), (b"LIST", None, b"odd"), (b"data", data_size, &pcm)]);
            assert_eq!(decode_wav_to_16k_mono(&buf).unwrap(), want, "{riff_size:?} {data_size:?}");
        }
    }

    #[test]
    fn float_extensible_and_wide_pcm_decode_with_float32_downmix() {
        let stereo16 = fmt_body(1, 2, 16_000, 16, None);
        let pcm16 = le(&[16_384i16, -16_384, 32_767, 32_767], i16::to_le_bytes);
        assert_eq!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &stereo16), (b"data", None, &pcm16)])).unwrap(), [0.0, 32_767.0 / 32_768.0]);
        let f32s = le(&[0.25f32, 0.75, -1.0, 0.5], f32::to_le_bytes);
        for fmt in [fmt_body(3, 2, 16_000, 32, None), fmt_body(0xFFFE, 2, 16_000, 32, Some(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT))] {
            assert_eq!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &fmt), (b"data", None, &f32s)])).unwrap(), [0.5, -0.25]);
        }
        let f64s = le(&[0.125f64, -0.5, 1.0 / 3.0], f64::to_le_bytes);
        let fmt = fmt_body(3, 1, 16_000, 64, None);
        assert_eq!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &fmt), (b"data", None, &f64s)])).unwrap(), [0.125, -0.5, (1.0f64 / 3.0) as f32]);
        let pcm24 = [0x00, 0x00, 0x40, 0x00, 0x00, 0xC0, 0xFF, 0xFF, 0x7F];
        let fmt = fmt_body(0xFFFE, 1, 16_000, 24, Some(KSDATAFORMAT_SUBTYPE_PCM));
        assert_eq!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &fmt), (b"data", None, &pcm24)])).unwrap(), [0.5, -0.5, 8_388_607.0 / 8_388_608.0]);
        let pcm32 = le(&[i32::MIN, 1 << 30], i32::to_le_bytes);
        let fmt = fmt_body(1, 1, 16_000, 32, None);
        assert_eq!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &fmt), (b"data", None, &pcm32)])).unwrap(), [-1.0, 0.5]);
        let fmt = fmt_body(1, 1, 500, 16, None);
        assert!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &fmt), (b"data", None, &pcm16)])).unwrap().is_empty());
        let fmt = fmt_body(0xFFFE, 1, 16_000, 16, Some([7; 16]));
        assert!(decode_wav_to_16k_mono(&riff(None, &[(b"fmt ", None, &fmt), (b"data", None, &pcm16)])).is_err());
    }

    #[test]
    fn encode_wav_mono16_matches_the_shared_byte_exact_fixture() {
        let samples = [
            0.0f32, -0.0, 0.25, 0.5, -0.5,
            f32::from_bits(0x38a0_0140), f32::from_bits(0xb8a0_0140), f32::from_bits(0xbb49_0192),
            1.0, -1.0, 1.5, -1.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
        ];
        let got: String = encode_wav_mono16(&samples, 16_000).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, ENCODE_WAV_MONO16_FIXTURE_HEX);
    }

    const ENCODE_WAV_MONO16_FIXTURE_HEX: &str = include_str!("../../../conformance/encode_wav_mono16.hex");
}
