use super::types::{MAX_DECODE_SAMPLE_RATE, MIN_DECODE_SAMPLE_RATE};

pub fn downmix_and_resample_f32(
    interleaved: &[f32],
    channels: usize,
    sr_in: usize,
    sr_out: usize,
) -> Vec<f32> {
    if !(MIN_DECODE_SAMPLE_RATE..=MAX_DECODE_SAMPLE_RATE).contains(&sr_in)
        || !(MIN_DECODE_SAMPLE_RATE..=MAX_DECODE_SAMPLE_RATE).contains(&sr_out)
        || channels == 0
    {
        return Vec::new();
    }
    let mono: Vec<f32> = if channels == 1 {
        interleaved.to_vec()
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    if sr_in == sr_out {
        return mono;
    }
    let n_out = (mono.len() as u128 * sr_out as u128 / sr_in as u128) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f64 * sr_in as f64 / sr_out as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(mono.len() - 1);
        let t = pos - lo as f64;
        let v = mono[lo] as f64 * (1.0 - t) + mono[hi] as f64 * t;
        out.push(v as f32);
    }
    out
}
