use anyhow::Result;

pub const SPK_MEL_SAMPLE_RATE: usize = 24_000;
pub const SPK_MEL_N_FFT: usize = 1024;
pub const SPK_MEL_HOP: usize = 256;
pub const SPK_MEL_WIN: usize = 1024;
pub const SPK_MEL_N_MELS: usize = 128;
pub const SPK_MEL_FMIN: f64 = 0.0;
pub const SPK_MEL_FMAX: f64 = 12_000.0;
pub const SPK_MEL_MIN_SAMPLES: usize = SPK_MEL_N_FFT;

fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f < min_log_hz {
        f / f_sp
    } else {
        min_log_mel + (f / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m < min_log_mel {
        m * f_sp
    } else {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    }
}

pub fn mel_filterbank_slaney(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    let mel_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let mut weights = vec![0.0f64; n_mels * n_bins];
    for m in 0..n_mels {
        let (f_lo, f_c, f_hi) = (mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]);
        let enorm = 2.0 / (f_hi - f_lo);
        for (k, &fk) in fft_freqs.iter().enumerate() {
            let lower = (fk - f_lo) / (f_c - f_lo);
            let upper = (f_hi - fk) / (f_hi - f_c);
            let w = lower.min(upper).max(0.0);
            weights[m * n_bins + k] = w * enorm;
        }
    }
    weights
}

fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cur_r, mut cur_i) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cur_r - im[i + k + len / 2] * cur_i,
                    re[i + k + len / 2] * cur_i + im[i + k + len / 2] * cur_r,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
            i += len;
        }
        len <<= 1;
    }
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(samples[i.min(n - 1)]);
    }
    out.extend_from_slice(samples);
    for i in 1..=pad {
        out.push(samples[n - 1 - i.min(n - 1)]);
    }
    out
}

pub fn log_mel_24k(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    if samples.len() < SPK_MEL_MIN_SAMPLES {
        anyhow::bail!(
            "log_mel_24k: need at least {} samples at 24 kHz, got {}",
            SPK_MEL_MIN_SAMPLES,
            samples.len()
        );
    }
    let pad = (SPK_MEL_N_FFT - SPK_MEL_HOP) / 2;
    let padded = reflect_pad(samples, pad);
    let n_frames = (padded.len() - SPK_MEL_N_FFT) / SPK_MEL_HOP + 1;
    if n_frames == 0 {
        anyhow::bail!("log_mel_24k: zero frames from {} samples", samples.len());
    }
    let window: Vec<f64> = (0..SPK_MEL_WIN)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / SPK_MEL_WIN as f64).cos())
        .collect();
    let n_bins = SPK_MEL_N_FFT / 2 + 1;
    let fb = mel_filterbank_slaney(
        SPK_MEL_N_MELS,
        SPK_MEL_N_FFT,
        SPK_MEL_SAMPLE_RATE,
        SPK_MEL_FMIN,
        SPK_MEL_FMAX,
    );
    let mut mags = vec![0.0f64; n_bins * n_frames];
    let mut re = vec![0.0f64; SPK_MEL_N_FFT];
    let mut im = vec![0.0f64; SPK_MEL_N_FFT];
    for t in 0..n_frames {
        let start = t * SPK_MEL_HOP;
        for i in 0..SPK_MEL_N_FFT {
            re[i] = padded[start + i] as f64 * window[i];
            im[i] = 0.0;
        }
        fft_in_place(&mut re, &mut im);
        for k in 0..n_bins {
            mags[k * n_frames + t] = (re[k] * re[k] + im[k] * im[k] + 1e-9).sqrt();
        }
    }
    let mut out = vec![0.0f32; SPK_MEL_N_MELS * n_frames];
    for m in 0..SPK_MEL_N_MELS {
        for t in 0..n_frames {
            let mut acc = 0.0f64;
            for k in 0..n_bins {
                acc += fb[m * n_bins + k] * mags[k * n_frames + t];
            }
            out[m * n_frames + t] = (acc.max(1e-5)).ln() as f32;
        }
    }
    Ok((out, n_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_too_short_input() {
        let s = vec![0.0f32; SPK_MEL_MIN_SAMPLES - 1];
        assert!(log_mel_24k(&s).is_err());
    }

    #[test]
    fn frame_count_matches_stft_formula() {
        let n = 24_000usize;
        let s = vec![0.0f32; n];
        let (mel, frames) = log_mel_24k(&s).unwrap();
        let pad = (SPK_MEL_N_FFT - SPK_MEL_HOP) / 2;
        let expected = (n + 2 * pad - SPK_MEL_N_FFT) / SPK_MEL_HOP + 1;
        assert_eq!(frames, expected);
        assert_eq!(mel.len(), SPK_MEL_N_MELS * frames);
    }

    #[test]
    fn silence_hits_log_floor() {
        let s = vec![0.0f32; 24_000];
        let (mel, _) = log_mel_24k(&s).unwrap();
        let floor = (1e-5f64).ln() as f32;
        for &v in &mel {
            assert!(
                (v - floor).abs() < 1e-3,
                "expected log floor {floor}, got {v}"
            );
        }
    }

    #[test]
    fn pure_tone_peaks_in_expected_mel_band() {
        let sr = SPK_MEL_SAMPLE_RATE as f64;
        let hz = 440.0f64;
        let s: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / sr).sin() as f32 * 0.5)
            .collect();
        let (mel, frames) = log_mel_24k(&s).unwrap();
        let mid = frames / 2;
        let col: Vec<f32> = (0..SPK_MEL_N_MELS).map(|m| mel[m * frames + mid]).collect();
        let peak = col
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let mel_pos = hz_to_mel_slaney(hz) / hz_to_mel_slaney(SPK_MEL_FMAX);
        let expected = (mel_pos * (SPK_MEL_N_MELS + 1) as f64) as usize;
        assert!(
            peak.abs_diff(expected) <= 2,
            "440 Hz peak at mel bin {peak}, expected near {expected}"
        );
    }

    #[test]
    fn filterbank_rows_have_positive_mass_and_local_support() {
        let n_bins = SPK_MEL_N_FFT / 2 + 1;
        let fb = mel_filterbank_slaney(
            SPK_MEL_N_MELS,
            SPK_MEL_N_FFT,
            SPK_MEL_SAMPLE_RATE,
            SPK_MEL_FMIN,
            SPK_MEL_FMAX,
        );
        for m in 0..SPK_MEL_N_MELS {
            let row = &fb[m * n_bins..(m + 1) * n_bins];
            let mass: f64 = row.iter().sum();
            assert!(mass > 0.0, "mel row {m} has zero mass");
            let nz = row.iter().filter(|w| **w > 0.0).count();
            assert!(nz < n_bins / 2, "mel row {m} support too wide ({nz} bins)");
        }
    }

    #[test]
    fn tone_amplitude_moves_mel_energy() {
        let sr = SPK_MEL_SAMPLE_RATE as f64;
        let mk = |amp: f32| -> Vec<f32> {
            (0..24_000)
                .map(|i| (2.0 * std::f64::consts::PI * 300.0 * i as f64 / sr).sin() as f32 * amp)
                .collect()
        };
        let (quiet, frames) = log_mel_24k(&mk(0.01)).unwrap();
        let (loud, _) = log_mel_24k(&mk(0.8)).unwrap();
        let mid = frames / 2;
        let q: f32 = (0..SPK_MEL_N_MELS).map(|m| quiet[m * frames + mid]).sum();
        let l: f32 = (0..SPK_MEL_N_MELS).map(|m| loud[m * frames + mid]).sum();
        assert!(l > q, "louder tone must raise summed log-mel: {l} <= {q}");
    }
}
