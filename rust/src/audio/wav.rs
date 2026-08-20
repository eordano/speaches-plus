use anyhow::Context;

use super::resample::downmix_and_resample_f32;
use super::types::{S16_SCALE, S24_SCALE, S32_SCALE, TARGET_SAMPLE_RATE};

pub fn decode_wav_to_16k_mono(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let mut buf;
    let bytes: &[u8] = if bytes.len() >= 12
        && &bytes[0..4] == b"RIFF"
        && &bytes[8..12] == b"WAVE"
        && bytes[4..8] == [0xFF; 4]
    {
        buf = bytes.to_vec();
        let riff_size = (buf.len() - 8) as u32;
        buf[4..8].copy_from_slice(&riff_size.to_le_bytes());
        if let Some(data_idx) = find_chunk(&buf, b"data") {
            if data_idx + 8 <= buf.len() && buf[data_idx + 4..data_idx + 8] == [0xFF; 4] {
                let data_size = (buf.len() - data_idx - 8) as u32;
                buf[data_idx + 4..data_idx + 8].copy_from_slice(&data_size.to_le_bytes());
            }
        }
        &buf
    } else {
        bytes
    };
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor).context("hound::WavReader")?;
    let spec = reader.spec();
    let samples_i: Vec<i32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .context("read i32 samples")?,
        hound::SampleFormat::Float => {
            let f: Vec<f32> = reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .context("read f32 samples")?;
            return Ok(downmix_and_resample_f32(
                &f,
                spec.channels as usize,
                spec.sample_rate as usize,
                TARGET_SAMPLE_RATE as usize,
            ));
        }
    };
    let scale = match spec.bits_per_sample {
        16 => S16_SCALE,
        24 => S24_SCALE,
        32 => S32_SCALE,
        b => anyhow::bail!("unsupported bits_per_sample: {b}"),
    };
    let f: Vec<f32> = samples_i.iter().map(|&s| s as f32 / scale).collect();
    Ok(downmix_and_resample_f32(
        &f,
        spec.channels as usize,
        spec.sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
    ))
}

pub fn find_chunk(buf: &[u8], tag: &[u8; 4]) -> Option<usize> {
    let mut i = 12;
    while i + 8 <= buf.len() {
        let chunk_tag = &buf[i..i + 4];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]);
        if chunk_tag == tag {
            return Some(i);
        }
        if size == 0xFFFFFFFF {
            return Some(i);
        }
        let next = i + 8 + size as usize + (size as usize & 1);
        if next <= i {
            return None;
        }
        i = next;
    }
    None
}
