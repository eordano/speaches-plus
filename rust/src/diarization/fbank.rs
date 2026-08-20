use std::sync::Arc;

use anyhow::{anyhow, Result};
use realfft::{RealFftPlanner, RealToComplex};

const PRE_EMPHASIS: f32 = 0.97;
const SAMPLE_RATE: f32 = 16_000.0;
const LOW_FREQ_HZ: f32 = 20.0;
const HIGH_FREQ_HZ: f32 = 7600.0;
const LOG_FLOOR: f32 = 1e-10;

pub struct FBank {
    num_mels: usize,
    frame_length: usize,
    frame_shift: usize,
    n_fft: usize,
    window: Vec<f32>,
    mel_filters: Vec<Vec<(usize, f32)>>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl FBank {
    pub fn new(num_mels: usize, frame_length: usize, frame_shift: usize) -> Self {
        let n_fft = next_power_of_two(frame_length);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window = povey_window(frame_length);
        let mel_filters =
            build_mel_filters(num_mels, n_fft, SAMPLE_RATE, LOW_FREQ_HZ, HIGH_FREQ_HZ);

        Self {
            num_mels,
            frame_length,
            frame_shift,
            n_fft,
            window,
            mel_filters,
            fft,
        }
    }

    pub fn num_mels(&self) -> usize {
        self.num_mels
    }

    pub fn compute(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.len() < self.frame_length {
            return Err(anyhow!(
                "fbank: audio too short ({} < {})",
                audio.len(),
                self.frame_length
            ));
        }

        let num_frames = 1 + (audio.len() - self.frame_length) / self.frame_shift;
        let mut out = vec![0.0f32; num_frames * self.num_mels];
        let mut frame_buf = vec![0.0f32; self.n_fft];
        let mut spectrum = self.fft.make_output_vec();
        let mut power = vec![0.0f32; spectrum.len()];
        let mut scratch = self.fft.make_scratch_vec();

        for frame_i in 0..num_frames {
            let start = frame_i * self.frame_shift;

            frame_buf.fill(0.0);
            let prev0 = if start == 0 {
                audio[0]
            } else {
                audio[start - 1]
            };
            frame_buf[0] = audio[start] - PRE_EMPHASIS * prev0;
            for i in 1..self.frame_length {
                frame_buf[i] = audio[start + i] - PRE_EMPHASIS * audio[start + i - 1];
            }

            for (b, w) in frame_buf.iter_mut().zip(self.window.iter()) {
                *b *= w;
            }

            self.fft
                .process_with_scratch(&mut frame_buf, &mut spectrum, &mut scratch)
                .map_err(|e| anyhow!("fbank fft: {}", e))?;

            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }

            let row = &mut out[frame_i * self.num_mels..(frame_i + 1) * self.num_mels];
            for (m, taps) in self.mel_filters.iter().enumerate() {
                let mut acc = 0.0f32;
                for &(bin, w) in taps {
                    acc += power[bin] * w;
                }
                row[m] = acc.max(LOG_FLOOR).ln();
            }
        }

        cmn_in_place(&mut out, self.num_mels);

        Ok(out)
    }
}

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn povey_window(n: usize) -> Vec<f32> {
    const POVEY_EXP: f32 = 0.85;
    let denom = (n as f32 - 1.0).max(1.0);
    (0..n)
        .map(|i| {
            let raised = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos();
            raised.max(0.0).powf(POVEY_EXP)
        })
        .collect()
}

#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

fn build_mel_filters(
    num_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    low_hz: f32,
    high_hz: f32,
) -> Vec<Vec<(usize, f32)>> {
    let num_bins = n_fft / 2 + 1;
    let low_mel = hz_to_mel(low_hz);
    let high_mel = hz_to_mel(high_hz);

    let mut mel_points = Vec::with_capacity(num_mels + 2);
    for i in 0..(num_mels + 2) {
        let m = low_mel + (high_mel - low_mel) * (i as f32) / ((num_mels + 1) as f32);
        mel_points.push(mel_to_hz(m));
    }

    let bins: Vec<f32> = mel_points
        .iter()
        .map(|&hz| hz * (n_fft as f32) / sample_rate)
        .collect();

    let mut filters = vec![Vec::<(usize, f32)>::new(); num_mels];
    for m in 0..num_mels {
        let left = bins[m];
        let center = bins[m + 1];
        let right = bins[m + 2];
        let lo = left.floor() as i64;
        let hi = right.ceil() as i64;
        for k in lo..=hi {
            if k < 0 || (k as usize) >= num_bins {
                continue;
            }
            let kf = k as f32;
            let w = if kf < center {
                if center > left {
                    (kf - left) / (center - left)
                } else {
                    0.0
                }
            } else if kf <= right {
                if right > center {
                    (right - kf) / (right - center)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            if w > 0.0 {
                filters[m].push((k as usize, w));
            }
        }
    }
    filters
}

fn cmn_in_place(feats: &mut [f32], num_mels: usize) {
    if feats.is_empty() || num_mels == 0 {
        return;
    }
    let num_frames = feats.len() / num_mels;
    let mut mean = vec![0.0f32; num_mels];
    for f in 0..num_frames {
        for m in 0..num_mels {
            mean[m] += feats[f * num_mels + m];
        }
    }
    for m in mean.iter_mut() {
        *m /= num_frames as f32;
    }
    for f in 0..num_frames {
        for m in 0..num_mels {
            feats[f * num_mels + m] -= mean[m];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbank_silence_gives_floor() {
        let fb = FBank::new(80, 400, 160);
        let audio = vec![0.0f32; 16_000];
        let feats = fb.compute(&audio).unwrap();
        let frames = feats.len() / 80;
        assert!(frames >= 90, "expected >=90 frames for 1s @ 10ms hop");

        for &v in feats.iter() {
            assert!(
                v.abs() < 1e-3,
                "post-CMN silence should be near zero, got {}",
                v
            );
        }
    }

    #[test]
    fn fbank_frame_count_matches_kaldi_formula() {
        let fb = FBank::new(80, 400, 160);
        let audio = vec![0.1f32; 16_000];
        let feats = fb.compute(&audio).unwrap();

        let expected = 1 + (16_000 - 400) / 160;
        assert_eq!(feats.len() / 80, expected);
    }

    #[test]
    fn mel_filters_cover_band() {
        let filters = build_mel_filters(80, 512, 16_000.0, 20.0, 7600.0);
        assert_eq!(filters.len(), 80);

        for (m, f) in filters.iter().enumerate() {
            assert!(!f.is_empty(), "mel {} has no taps", m);
        }
    }

    #[test]
    fn hz_mel_round_trip() {
        for hz in [20.0f32, 1000.0, 4000.0, 7600.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 0.5, "hz {} -> {}", hz, back);
        }
    }
}
