pub(crate) const MEL_F_SP: f32 = 200.0 / 3.0;
pub(crate) const MEL_MIN_LOG_HZ: f32 = 1000.0;

pub(crate) fn mel_min_log_mel() -> f32 {
    MEL_MIN_LOG_HZ / MEL_F_SP
}

pub(crate) fn mel_logstep() -> f32 {
    (6.4f32).ln() / 27.0
}

pub(crate) fn hz_to_mel(f: f32) -> f32 {
    if f >= MEL_MIN_LOG_HZ {
        mel_min_log_mel() + (f / MEL_MIN_LOG_HZ).ln() / mel_logstep()
    } else {
        f / MEL_F_SP
    }
}

pub(crate) fn mel_to_hz(m: f32) -> f32 {
    if m >= mel_min_log_mel() {
        MEL_MIN_LOG_HZ * ((m - mel_min_log_mel()) * mel_logstep()).exp()
    } else {
        MEL_F_SP * m
    }
}

pub(crate) fn hann_window(n_fft: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n_fft];
    for (i, slot) in w.iter_mut().enumerate() {
        let phase = 2.0 * std::f32::consts::PI * (i as f32) / (n_fft as f32);
        *slot = 0.5 - 0.5 * phase.cos();
    }
    w
}

pub(crate) fn build_mel_filters(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_min = 0.0f32;
    let f_max = sample_rate as f32 / 2.0;
    let m_min = hz_to_mel(f_min);
    let m_max = hz_to_mel(f_max);

    let mut mel_points = vec![0.0f32; n_mels + 2];
    for (i, slot) in mel_points.iter_mut().enumerate() {
        let frac = i as f32 / (n_mels as f32 + 1.0);
        *slot = m_min + (m_max - m_min) * frac;
    }
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let mut fft_freqs = vec![0.0f32; n_bins];
    for (i, slot) in fft_freqs.iter_mut().enumerate() {
        *slot = i as f32 * sample_rate as f32 / n_fft as f32;
    }

    let mut filters = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        let lower_slope = (center - lower).max(f32::EPSILON);
        let upper_slope = (upper - center).max(f32::EPSILON);
        let enorm = 2.0 / (upper - lower).max(f32::EPSILON);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let mut weight = 0.0f32;
            if freq >= lower && freq <= center {
                weight = (freq - lower) / lower_slope;
            } else if freq > center && freq <= upper {
                weight = (upper - freq) / upper_slope;
            }
            filters[m * n_bins + k] = weight * enorm;
        }
    }
    filters
}
