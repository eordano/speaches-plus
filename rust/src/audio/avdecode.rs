use anyhow::Context;

use super::resample::downmix_and_resample_f32;
use super::types::TARGET_SAMPLE_RATE;

pub fn decode_via_symphonia(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    use std::io::Cursor;
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("symphonia probe")?;

    let track = format
        .default_track(TrackType::Audio)
        .context("no audio track")?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .context("no audio codec parameters")?
        .clone();
    let mut in_rate = params.sample_rate.unwrap_or(16_000);
    let mut in_channels = params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .context("symphonia decoder")?;

    let mut native: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e).context("symphonia next_packet"),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("symphonia decode"),
        };

        in_rate = decoded.spec().rate();
        in_channels = decoded.spec().channels().count().max(1);
        let start = native.len();
        native.resize(start + decoded.samples_interleaved(), 0.0);
        decoded.copy_to_slice_interleaved(&mut native[start..]);
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
