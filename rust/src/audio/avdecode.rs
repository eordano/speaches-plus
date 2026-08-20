use anyhow::Context;

use super::resample::downmix_and_resample_f32;
use super::types::TARGET_SAMPLE_RATE;

pub fn decode_via_symphonia(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    use std::io::Cursor;
    use symphonia::core::audio::{AudioBuffer, Signal as _};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("symphonia probe")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("no audio track")?
        .clone();
    let track_id = track.id;
    let in_rate = track.codec_params.sample_rate.unwrap_or(16_000);
    let in_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("symphonia decoder")?;

    let mut native: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e).context("symphonia next_packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("symphonia decode"),
        };

        let spec = *decoded.spec();
        let frames = decoded.frames();
        let mut f32_buf = AudioBuffer::<f32>::new(frames as u64, spec);
        decoded.convert(&mut f32_buf);
        for f in 0..frames {
            for c in 0..in_channels {
                native.push(f32_buf.chan(c)[f]);
            }
        }
    }

    let mut mono: Vec<f32> = if in_channels == 1 {
        native
    } else {
        let frames = native.len() / in_channels;
        let mut m = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut s = 0.0f32;
            for c in 0..in_channels {
                s += native[i * in_channels + c];
            }
            m.push(s / in_channels as f32);
        }
        m
    };

    if in_rate != TARGET_SAMPLE_RATE {
        mono = downmix_and_resample_f32(&mono, 1, in_rate as usize, TARGET_SAMPLE_RATE as usize);
    }
    Ok(mono)
}
