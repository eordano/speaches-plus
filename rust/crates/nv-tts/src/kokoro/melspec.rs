use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::spk_mel::{
    mel_filterbank_slaney, SPK_MEL_FMAX, SPK_MEL_FMIN, SPK_MEL_HOP, SPK_MEL_N_FFT, SPK_MEL_N_MELS,
    SPK_MEL_SAMPLE_RATE,
};

pub struct LogMel24k {
    dft_cos: Tensor,
    dft_sin: Tensor,
    mel: Tensor,
}

impl LogMel24k {
    pub fn new(device: &Device) -> Result<Self> {
        let n = SPK_MEL_N_FFT;
        let bins = n / 2 + 1;
        let window: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();
        let mut c = vec![0f32; n * bins];
        let mut s = vec![0f32; n * bins];
        for i in 0..n {
            for k in 0..bins {
                let a = 2.0 * std::f64::consts::PI * (i * k) as f64 / n as f64;
                c[i * bins + k] = (window[i] * a.cos()) as f32;
                s[i * bins + k] = (-window[i] * a.sin()) as f32;
            }
        }
        let fb = mel_filterbank_slaney(
            SPK_MEL_N_MELS,
            n,
            SPK_MEL_SAMPLE_RATE,
            SPK_MEL_FMIN,
            SPK_MEL_FMAX,
        );
        let fb: Vec<f32> = fb.iter().map(|x| *x as f32).collect();
        Ok(Self {
            dft_cos: Tensor::from_vec(c, (n, bins), device)?,
            dft_sin: Tensor::from_vec(s, (n, bins), device)?,
            mel: Tensor::from_vec(fb, (SPK_MEL_N_MELS, bins), device)?,
        })
    }

    pub fn forward(&self, wav: &Tensor) -> Result<Tensor> {
        let n = SPK_MEL_N_FFT;
        let pad = (n - SPK_MEL_HOP) / 2;
        let l = wav.dim(0)?;

        let mut idx: Vec<u32> = Vec::with_capacity(l + 2 * pad);
        for i in 0..pad {
            idx.push((pad - i) as u32);
        }
        idx.extend(0..l as u32);
        for i in 0..pad {
            idx.push((l - 2 - i) as u32);
        }
        let padded = wav.index_select(&Tensor::new(idx, wav.device())?, 0)?;
        let frames_n = (l + 2 * pad - n) / SPK_MEL_HOP + 1;
        let mut fidx: Vec<u32> = Vec::with_capacity(frames_n * n);
        for t in 0..frames_n {
            let start = (t * SPK_MEL_HOP) as u32;
            fidx.extend(start..start + n as u32);
        }
        let frames = padded
            .index_select(&Tensor::new(fidx, wav.device())?, 0)?
            .reshape((frames_n, n))?;
        let re = frames.matmul(&self.dft_cos)?;
        let im = frames.matmul(&self.dft_sin)?;
        let mag = ((re.sqr()? + im.sqr()?)? + 1e-9)?.sqrt()?;
        let mel = self.mel.matmul(&mag.t()?.contiguous()?)?;
        let logmel = mel.clamp(1e-5f32, f32::MAX)?.log()?;
        Ok(logmel.unsqueeze(0)?.to_dtype(DType::F32)?)
    }
}
