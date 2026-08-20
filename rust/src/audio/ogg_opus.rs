use anyhow::{anyhow, Context};
use std::io::Cursor;

use super::resample::downmix_and_resample_f32;
use super::types::TARGET_SAMPLE_RATE;

const OPUS_HEAD_MAGIC: &[u8] = b"OpusHead";
const OGG_MAGIC: &[u8] = b"OggS";

pub fn is_ogg_opus(bytes: &[u8]) -> bool {
    if bytes.len() < 36 {
        return false;
    }
    if &bytes[0..4] != OGG_MAGIC {
        return false;
    }
    for window in bytes[..bytes.len().min(200)].windows(OPUS_HEAD_MAGIC.len()) {
        if window == OPUS_HEAD_MAGIC {
            return true;
        }
    }
    false
}

pub fn decode_ogg_opus_to_16k_mono(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let cursor = Cursor::new(bytes);
    let mut reader = ogg::PacketReader::new(cursor);

    let head_packet = reader
        .read_packet()
        .context("ogg read OpusHead")?
        .ok_or_else(|| anyhow!("empty ogg stream"))?;

    let head_data = &head_packet.data;
    if head_data.len() < 19 || &head_data[0..8] != OPUS_HEAD_MAGIC {
        return Err(anyhow!("not an OpusHead packet"));
    }
    let channels = head_data[9] as u32;
    let pre_skip = u16::from_le_bytes([head_data[10], head_data[11]]) as usize;
    let input_rate =
        u32::from_le_bytes([head_data[12], head_data[13], head_data[14], head_data[15]]);
    let _ = input_rate;

    let _comment_packet = reader.read_packet().context("ogg read OpusTags")?;

    let decode_rate = 48_000u32;
    let mut decoder = opus::Decoder::new(
        decode_rate,
        match channels {
            1 => opus::Channels::Mono,
            _ => opus::Channels::Stereo,
        },
    )
    .context("opus decoder create")?;

    let out_channels = if channels > 1 { 2usize } else { 1usize };
    let max_frame = 5760 * out_channels;
    let mut pcm_buf = vec![0f32; max_frame];
    let mut all_samples: Vec<f32> = Vec::new();
    let mut total_samples: usize = 0;

    while let Some(packet) = reader.read_packet().context("ogg read packet")? {
        let n = decoder
            .decode_float(&packet.data, &mut pcm_buf, false)
            .context("opus decode")?;
        let sample_count = n * out_channels;
        all_samples.extend_from_slice(&pcm_buf[..sample_count]);
        total_samples += n;
    }

    if total_samples > pre_skip {
        all_samples = all_samples[(pre_skip * out_channels)..].to_vec();
    }

    let mono = if out_channels == 1 {
        all_samples
    } else {
        let frames = all_samples.len() / out_channels;
        let mut m = Vec::with_capacity(frames);
        for i in 0..frames {
            let l = all_samples[i * 2];
            let r = all_samples[i * 2 + 1];
            m.push((l + r) * 0.5);
        }
        m
    };

    if decode_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }

    Ok(downmix_and_resample_f32(
        &mono,
        1,
        decode_rate as usize,
        TARGET_SAMPLE_RATE as usize,
    ))
}
