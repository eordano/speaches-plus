use anyhow::Result;
use candle_core::{Device, Tensor};

pub const SR: f64 = 16_000.0;
pub const FRAME_LEN: usize = 400;
pub const FRAME_SHIFT: usize = 160;
pub const N_FFT: usize = 512;
pub const N_MELS: usize = 80;
const PRE_EMPHASIS: f64 = 0.97;
const LOW_HZ: f64 = 20.0;
const HIGH_HZ: f64 = 7600.0;

fn hz_to_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

pub struct Fbank {
    dft_cos: Tensor,
    dft_sin: Tensor,
    mel: Tensor,
}

impl Fbank {
    pub fn new(device: &Device) -> Result<Self> {
        let bins = N_FFT / 2 + 1;
        let denom = (FRAME_LEN - 1) as f64;
        let window: Vec<f64> = (0..FRAME_LEN)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos())
                    .max(0.0)
                    .powf(0.85)
            })
            .collect();
        let mut c = vec![0f32; FRAME_LEN * bins];
        let mut s = vec![0f32; FRAME_LEN * bins];
        for i in 0..FRAME_LEN {
            for k in 0..bins {
                let a = 2.0 * std::f64::consts::PI * (i * k) as f64 / N_FFT as f64;
                c[i * bins + k] = (window[i] * a.cos()) as f32;
                s[i * bins + k] = (-window[i] * a.sin()) as f32;
            }
        }
        let (lo, hi) = (hz_to_mel(LOW_HZ), hz_to_mel(HIGH_HZ));
        let pts: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64) * N_FFT as f64 / SR)
            .collect();
        let mut fb = vec![0f32; N_MELS * bins];
        for m in 0..N_MELS {
            let (l, cen, r) = (pts[m], pts[m + 1], pts[m + 2]);
            for k in 0..bins {
                let kf = k as f64;
                let w = if kf < l {
                    0.0
                } else if kf < cen {
                    if cen > l {
                        (kf - l) / (cen - l)
                    } else {
                        0.0
                    }
                } else if kf <= r {
                    if r > cen {
                        (r - kf) / (r - cen)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                if w > 0.0 {
                    fb[m * bins + k] = w as f32;
                }
            }
        }
        Ok(Self {
            dft_cos: Tensor::from_vec(c, (FRAME_LEN, bins), device)?,
            dft_sin: Tensor::from_vec(s, (FRAME_LEN, bins), device)?,
            mel: Tensor::from_vec(fb, (N_MELS, bins), device)?,
        })
    }

    pub fn forward(&self, wav: &Tensor) -> Result<Tensor> {
        let l = wav.dim(0)?;
        let frames = 1 + (l - FRAME_LEN) / FRAME_SHIFT;
        let mut cur: Vec<u32> = Vec::with_capacity(frames * FRAME_LEN);
        let mut prev: Vec<u32> = Vec::with_capacity(frames * FRAME_LEN);
        for t in 0..frames {
            let start = t * FRAME_SHIFT;
            for i in 0..FRAME_LEN {
                cur.push((start + i) as u32);
                prev.push(if start + i == 0 {
                    0
                } else {
                    (start + i - 1) as u32
                });
            }
        }
        let dev = wav.device();
        let x = wav
            .index_select(&Tensor::new(cur, dev)?, 0)?
            .reshape((frames, FRAME_LEN))?;
        let p = wav
            .index_select(&Tensor::new(prev, dev)?, 0)?
            .reshape((frames, FRAME_LEN))?;
        let emph = (x - (p * PRE_EMPHASIS)?)?;
        let re = emph.matmul(&self.dft_cos)?;
        let im = emph.matmul(&self.dft_sin)?;
        let power = (re.sqr()? + im.sqr()?)?;
        let mel = power.matmul(&self.mel.t()?.contiguous()?)?;
        let logmel = mel.clamp(1e-10f32, f32::MAX)?.log()?;
        let mean = logmel.mean_keepdim(0)?;
        Ok(logmel.broadcast_sub(&mean)?)
    }
}
