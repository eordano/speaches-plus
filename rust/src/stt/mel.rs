const SAMPLE_RATE: usize = 16_000;

pub const N_FRAMES: usize = 3_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const TARGET_SAMPLES: usize = 30 * SAMPLE_RATE;

pub fn pad_or_truncate_to_30s(audio: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; TARGET_SAMPLES];
    let n = audio.len().min(TARGET_SAMPLES);
    out[..n].copy_from_slice(&audio[..n]);
    out
}

pub struct WhisperMel {
    pub n_mels: usize,
    filters: Vec<f32>,
    hann: Vec<f32>,

    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

impl WhisperMel {
    pub fn new(n_mels: usize) -> Self {
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self {
            n_mels,
            filters: crate::mel_scale::build_mel_filters(n_mels, N_FFT, SAMPLE_RATE),
            hann: crate::mel_scale::hann_window(N_FFT),
            fft,
        }
    }

    pub fn log_mel(&self, audio_30s: &[f32]) -> Vec<f32> {
        assert_eq!(audio_30s.len(), TARGET_SAMPLES, "expected 30s of audio");
        let n_bins = N_FFT / 2 + 1;

        let mut input_buf = self.fft.make_input_vec();
        let mut output_buf = self.fft.make_output_vec();

        let pad = N_FFT / 2;
        let mut padded = vec![0.0_f32; audio_30s.len() + N_FFT];
        for i in 0..pad {
            padded[i] = audio_30s[pad - i];
        }
        padded[pad..pad + audio_30s.len()].copy_from_slice(audio_30s);
        for i in 0..pad {
            let src = audio_30s.len().saturating_sub(2 + i);
            padded[pad + audio_30s.len() + i] = audio_30s[src];
        }

        let mut mel = vec![0.0_f32; self.n_mels * N_FRAMES];
        let mut power = vec![0.0_f32; n_bins];
        for frame in 0..N_FRAMES {
            let start = frame * HOP_LENGTH;
            for i in 0..N_FFT {
                input_buf[i] = padded[start + i] * self.hann[i];
            }
            self.fft
                .process(&mut input_buf, &mut output_buf)
                .expect("FFT process");
            for (k, c) in output_buf.iter().enumerate() {
                power[k] = c.re * c.re + c.im * c.im;
            }
            for m in 0..self.n_mels {
                let row = &self.filters[m * n_bins..(m + 1) * n_bins];
                let mut sum = 0.0_f32;
                for k in 0..n_bins {
                    sum += row[k] * power[k];
                }
                mel[m * N_FRAMES + frame] = sum;
            }
        }

        let eps = 1e-10_f32;
        let mut max_val = f32::NEG_INFINITY;
        for v in mel.iter_mut() {
            *v = v.max(eps).log10();
            if *v > max_val {
                max_val = *v;
            }
        }
        let floor = max_val - 8.0;
        for v in mel.iter_mut() {
            if *v < floor {
                *v = floor;
            }
            *v = (*v + 4.0) / 4.0;
        }
        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_short_audio() {
        let p = pad_or_truncate_to_30s(&[1.0, 2.0, 3.0]);
        assert_eq!(p.len(), TARGET_SAMPLES);
        assert_eq!(&p[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(p[3], 0.0);
    }

    #[test]
    fn truncate_long_audio() {
        let big = vec![1.0_f32; TARGET_SAMPLES + 100];
        let p = pad_or_truncate_to_30s(&big);
        assert_eq!(p.len(), TARGET_SAMPLES);
    }

    #[test]
    fn mel_shape_for_80() {
        let m = WhisperMel::new(80);
        let audio = vec![0.1_f32; TARGET_SAMPLES];
        let mel = m.log_mel(&audio);
        assert_eq!(mel.len(), 80 * N_FRAMES);
        for v in &mel {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn mel_shape_for_128() {
        let m = WhisperMel::new(128);
        let audio = vec![0.0_f32; TARGET_SAMPLES];
        let mel = m.log_mel(&audio);
        assert_eq!(mel.len(), 128 * N_FRAMES);
    }
}
