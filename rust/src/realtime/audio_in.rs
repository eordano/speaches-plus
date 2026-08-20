use anyhow::{Context, Result};
use opus::{Channels, Decoder};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};

use crate::defaults;

const MAX_DECODE_FRAMES: usize = defaults::audio::MAX_DECODE_FRAMES;
const INPUT_HZ: f64 = defaults::audio::INPUT_HZ;
const OUTPUT_HZ: f64 = defaults::audio::OUTPUT_HZ;
const OPUS_SAMPLE_RATE_HZ: u32 = defaults::audio::OPUS_SAMPLE_RATE_HZ;
const OPUS_FRAME_SAMPLES: usize = defaults::audio::FRAME_SAMPLES;

pub struct AudioIngest {
    decoder: Decoder,
    channels: usize,
    out: Vec<f32>,
    resampler: FastFixedIn<f32>,
    decode_workspace: Vec<i16>,
    mono_48k_workspace: Vec<f32>,
    resampler_in: Vec<Vec<f32>>,
    resampler_out: Vec<Vec<f32>>,
    chunk_in_frames: usize,
    leftover_48k: Vec<f32>,
}

impl AudioIngest {
    pub fn new(channels: usize) -> Result<Self> {
        let opus_channels = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => anyhow::bail!("unsupported opus channel count: {channels}"),
        };
        let decoder =
            Decoder::new(OPUS_SAMPLE_RATE_HZ, opus_channels).context("create opus decoder")?;

        let chunk_in_frames = OPUS_FRAME_SAMPLES;
        let resampler = FastFixedIn::<f32>::new(
            OUTPUT_HZ / INPUT_HZ,
            1.0,
            PolynomialDegree::Septic,
            chunk_in_frames,
            1,
        )
        .context("create rubato resampler")?;

        let chunk_out_frames = resampler.output_frames_max();
        Ok(Self {
            decoder,
            channels,
            out: Vec::with_capacity(8 * chunk_out_frames),
            resampler,
            decode_workspace: vec![0i16; MAX_DECODE_FRAMES * channels],
            mono_48k_workspace: Vec::with_capacity(MAX_DECODE_FRAMES),
            resampler_in: vec![vec![0.0; chunk_in_frames]],
            resampler_out: vec![vec![0.0; chunk_out_frames]],
            chunk_in_frames,
            leftover_48k: Vec::with_capacity(2 * chunk_in_frames),
        })
    }

    pub fn process(&mut self, opus_payload: &[u8]) -> Result<()> {
        let frames_per_channel = self
            .decoder
            .decode(opus_payload, &mut self.decode_workspace, false)
            .context("opus_decode")?;

        self.mono_48k_workspace.clear();
        self.mono_48k_workspace.reserve(frames_per_channel);
        match self.channels {
            1 => {
                for &s in &self.decode_workspace[..frames_per_channel] {
                    self.mono_48k_workspace.push(s as f32 / 32768.0);
                }
            }
            2 => {
                for chunk in self.decode_workspace[..frames_per_channel * 2].chunks_exact(2) {
                    let avg = (chunk[0] as i32 + chunk[1] as i32) as f32 / 2.0 / 32768.0;
                    self.mono_48k_workspace.push(avg);
                }
            }
            _ => unreachable!(),
        }

        self.leftover_48k
            .extend_from_slice(&self.mono_48k_workspace);
        let mut consumed = 0usize;
        while self.leftover_48k.len() - consumed >= self.chunk_in_frames {
            let chunk = &self.leftover_48k[consumed..consumed + self.chunk_in_frames];
            self.resampler_in[0].clear();
            self.resampler_in[0].extend_from_slice(chunk);
            let (in_used, out_produced) = self
                .resampler
                .process_into_buffer(&self.resampler_in, &mut self.resampler_out, None)
                .context("rubato process")?;
            debug_assert_eq!(in_used, self.chunk_in_frames);
            self.out
                .extend_from_slice(&self.resampler_out[0][..out_produced]);
            consumed += in_used;
        }
        if consumed > 0 {
            self.leftover_48k.drain(..consumed);
        }
        Ok(())
    }

    pub fn take(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus::{Application, Encoder};

    #[test]
    fn round_trip_decode_resample_silence() -> Result<()> {
        let mut enc = Encoder::new(OPUS_SAMPLE_RATE_HZ, Channels::Stereo, Application::Voip)?;
        let silence = vec![0i16; OPUS_FRAME_SAMPLES * 2];
        let mut payload = vec![0u8; defaults::audio::OPUS_ENCODE_BUFFER_BYTES];
        let n = enc.encode(&silence, &mut payload)?;

        let mut ingest = AudioIngest::new(2)?;
        ingest.process(&payload[..n])?;
        let samples = ingest.take();
        assert!(!samples.is_empty(), "no samples produced");
        assert!(
            samples.iter().all(|&s| s.abs() < 0.001),
            "expected silence, got non-zero peak"
        );
        Ok(())
    }
}
