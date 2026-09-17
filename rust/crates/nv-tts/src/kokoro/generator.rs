use anyhow::{anyhow, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

use super::layers::{
    conv1d_load, conv_transpose1d_diff, differentiable, leaky_relu, same_conv, snake,
    transpose_kernel_to_conv, AdaIn1d, ConvT1d,
};

pub const SAMPLE_RATE: f32 = 24_000.0;
pub const HARMONICS: usize = 8;
pub const SINE_AMP: f32 = 0.1;
pub const NOISE_STD: f32 = 0.003;
pub const VOICED_THRESHOLD: f32 = 10.0;

#[derive(Clone, Debug)]
pub struct GeneratorConfig {
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub n_fft: usize,
    pub hop: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            resblock_kernel_sizes: vec![3, 7, 11],
            upsample_rates: vec![10, 6],
            upsample_initial_channel: 512,
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_kernel_sizes: vec![20, 12],
            n_fft: 20,
            hop: 5,
        }
    }
}

impl GeneratorConfig {
    pub fn upsample_scale(&self) -> usize {
        self.upsample_rates.iter().product::<usize>() * self.hop
    }
}

pub struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
}

impl AdaInResBlock1 {
    pub fn load(vb: VarBuilder, channels: usize, k: usize, dilations: &[usize]) -> Result<Self> {
        let mut s = Self {
            convs1: vec![],
            convs2: vec![],
            adain1: vec![],
            adain2: vec![],
            alpha1: vec![],
            alpha2: vec![],
        };
        for (j, d) in dilations.iter().enumerate() {
            s.convs1.push(conv1d_load(
                vb.pp(format!("convs1.{j}")),
                channels,
                channels,
                k,
                same_conv(k, *d),
                true,
            )?);
            s.convs2.push(conv1d_load(
                vb.pp(format!("convs2.{j}")),
                channels,
                channels,
                k,
                same_conv(k, 1),
                true,
            )?);
            s.adain1
                .push(AdaIn1d::load(vb.pp(format!("adain1.{j}")), channels)?);
            s.adain2
                .push(AdaIn1d::load(vb.pp(format!("adain2.{j}")), channels)?);
            s.alpha1
                .push(vb.get((1, channels, 1), &format!("alpha1.{j}"))?);
            s.alpha2
                .push(vb.get((1, channels, 1), &format!("alpha2.{j}"))?);
        }
        Ok(s)
    }
    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for j in 0..self.convs1.len() {
            let mut xt = snake(&self.adain1[j].forward(&x, style)?, &self.alpha1[j])?;
            xt = self.convs1[j].forward(&xt)?;
            xt = snake(&self.adain2[j].forward(&xt, style)?, &self.alpha2[j])?;
            xt = self.convs2[j].forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
    pub fn adains_mut(&mut self) -> Vec<&mut AdaIn1d> {
        self.adain1
            .iter_mut()
            .chain(self.adain2.iter_mut())
            .collect()
    }
}

pub struct DftStft {
    fwd_real: Tensor,
    fwd_imag: Tensor,
    bwd_real: Tensor,
    bwd_imag: Tensor,
    bwd_real_conv: Tensor,
    bwd_imag_conv: Tensor,
    n_fft: usize,
    hop: usize,
}

impl DftStft {
    pub fn load(vb: VarBuilder, n_fft: usize, hop: usize) -> Result<Self> {
        let bins = n_fft / 2 + 1;
        let bwd_real = vb.get((bins, 1, n_fft), "weight_backward_real")?;
        let bwd_imag = vb.get((bins, 1, n_fft), "weight_backward_imag")?;
        Ok(Self {
            fwd_real: vb.get((bins, 1, n_fft), "weight_forward_real")?,
            fwd_imag: vb.get((bins, 1, n_fft), "weight_forward_imag")?,
            bwd_real_conv: transpose_kernel_to_conv(&bwd_real, 1)?,
            bwd_imag_conv: transpose_kernel_to_conv(&bwd_imag, 1)?,
            bwd_real,
            bwd_imag,
            n_fft,
            hop,
        })
    }

    pub fn transform(&self, wav: &Tensor) -> Result<(Tensor, Tensor)> {
        let pad = self.n_fft / 2;
        let x = wav.pad_with_same(1, pad, pad)?.unsqueeze(1)?;
        let re = x.conv1d(&self.fwd_real, 0, self.hop, 1, 1)?;
        let im = x.conv1d(&self.fwd_imag, 0, self.hop, 1, 1)?;
        let mag = ((re.sqr()? + im.sqr()?)? + 1e-14)?.sqrt()?;
        let (b, bins, frames) = re.dims3()?;
        let rv: Vec<f32> = re.flatten_all()?.to_vec1()?;
        let iv: Vec<f32> = im.flatten_all()?.to_vec1()?;
        let ph: Vec<f32> = rv
            .iter()
            .zip(&iv)
            .map(|(r, i)| {
                if *i == 0.0 && *r < 0.0 {
                    std::f32::consts::PI
                } else {
                    i.atan2(*r)
                }
            })
            .collect();
        Ok((mag, Tensor::from_vec(ph, (b, bins, frames), wav.device())?))
    }

    pub fn inverse(&self, mag: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let re = mag.mul(&phase.cos()?)?;
        let im = mag.mul(&phase.sin()?)?;
        let (wr, wi) = if differentiable() {
            (
                conv_transpose1d_diff(&re, &self.bwd_real_conv, self.n_fft, self.hop, 0, 0, 1)?,
                conv_transpose1d_diff(&im, &self.bwd_imag_conv, self.n_fft, self.hop, 0, 0, 1)?,
            )
        } else {
            (
                re.conv_transpose1d(&self.bwd_real, 0, 0, self.hop, 1, 1)?,
                im.conv_transpose1d(&self.bwd_imag, 0, 0, self.hop, 1, 1)?,
            )
        };
        let w = (wr - wi)?.squeeze(1)?;
        let pad = self.n_fft / 2;
        let len = w.dim(1)?;
        Ok(w.narrow(1, pad, len - 2 * pad)?)
    }
}

pub struct Rng(pub u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn interp_linear(x: &[f32], out_len: usize) -> Vec<f32> {
    let n = x.len();
    let scale = n as f64 / out_len as f64;
    (0..out_len)
        .map(|j| {
            let src = ((j as f64 + 0.5) * scale - 0.5).max(0.0);
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(n - 1);
            let t = (src - i0 as f64) as f32;
            x[i0.min(n - 1)] * (1.0 - t) + x[i1] * t
        })
        .collect()
}

pub struct HarmonicSource {
    linear: Linear,
    scale: usize,
}

impl HarmonicSource {
    pub fn load(vb: VarBuilder, scale: usize) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(HARMONICS + 1, 1, vb.pp("l_linear"))?,
            scale,
        })
    }

    pub fn forward(
        &self,
        f0: &[f32],
        device: &Device,
        deterministic: bool,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        let t2 = f0.len();
        let l = t2 * self.scale;
        let dim = HARMONICS + 1;

        let f0u: Vec<f32> = (0..l).map(|i| f0[i / self.scale]).collect();
        let uv: Vec<f32> = f0u
            .iter()
            .map(|f| if *f > VOICED_THRESHOLD { 1.0 } else { 0.0 })
            .collect();
        let mut sines = vec![0f32; l * dim];
        for h in 0..dim {
            let mut rad: Vec<f32> = f0u
                .iter()
                .map(|f| (f * (h + 1) as f32 / SAMPLE_RATE).rem_euclid(1.0))
                .collect();
            if !deterministic && h > 0 {
                rad[0] += rng.uniform();
            }
            let down = interp_linear(&rad, t2);
            let mut phase = Vec::with_capacity(t2);
            let mut acc = 0f32;
            for r in down {
                acc += r;
                phase.push(acc * 2.0 * std::f32::consts::PI * self.scale as f32);
            }
            let up = interp_linear(&phase, l);
            for (i, p) in up.iter().enumerate() {
                sines[i * dim + h] = p.sin() * SINE_AMP;
            }
        }
        for i in 0..l {
            let amp = uv[i] * NOISE_STD + (1.0 - uv[i]) * SINE_AMP / 3.0;
            for h in 0..dim {
                let noise = if deterministic {
                    0.0
                } else {
                    amp * rng.normal()
                };
                sines[i * dim + h] = sines[i * dim + h] * uv[i] + noise;
            }
        }
        let s = Tensor::from_vec(sines, (1, l, dim), device)?;
        Ok(self.linear.forward(&s)?.tanh()?.squeeze(2)?)
    }
}

pub struct Generator {
    cfg: GeneratorConfig,
    pub source: HarmonicSource,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    ups: Vec<ConvT1d>,
    resblocks: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    stft: DftStft,
}

impl Generator {
    pub fn load(vb: VarBuilder, cfg: GeneratorConfig) -> Result<Self> {
        let n_up = cfg.upsample_rates.len();
        let mut ups = Vec::with_capacity(n_up);
        let mut noise_convs = Vec::with_capacity(n_up);
        let mut noise_res = Vec::with_capacity(n_up);
        let mut resblocks = Vec::with_capacity(n_up * cfg.resblock_kernel_sizes.len());
        let mut ch = cfg.upsample_initial_channel;
        for i in 0..n_up {
            let (u, k) = (cfg.upsample_rates[i], cfg.upsample_kernel_sizes[i]);
            let cin = cfg.upsample_initial_channel >> i;
            ch = cfg.upsample_initial_channel >> (i + 1);
            ups.push(ConvT1d::load(
                vb.pp(format!("ups.{i}")),
                cin,
                ch,
                k,
                u,
                (k - u) / 2,
                0,
                1,
            )?);
            for (j, (rk, rd)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
                .enumerate()
            {
                resblocks.push(AdaInResBlock1::load(
                    vb.pp(format!(
                        "resblocks.{}",
                        i * cfg.resblock_kernel_sizes.len() + j
                    )),
                    ch,
                    *rk,
                    rd,
                )?);
            }
            if i + 1 < n_up {
                let stride_f0: usize = cfg.upsample_rates[i + 1..].iter().product();
                noise_convs.push(conv1d_load(
                    vb.pp(format!("noise_convs.{i}")),
                    cfg.n_fft + 2,
                    ch,
                    stride_f0 * 2,
                    Conv1dConfig {
                        padding: stride_f0.div_ceil(2),
                        stride: stride_f0,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                    true,
                )?);
                noise_res.push(AdaInResBlock1::load(
                    vb.pp(format!("noise_res.{i}")),
                    ch,
                    7,
                    &[1, 3, 5],
                )?);
            } else {
                noise_convs.push(conv1d_load(
                    vb.pp(format!("noise_convs.{i}")),
                    cfg.n_fft + 2,
                    ch,
                    1,
                    Conv1dConfig::default(),
                    true,
                )?);
                noise_res.push(AdaInResBlock1::load(
                    vb.pp(format!("noise_res.{i}")),
                    ch,
                    11,
                    &[1, 3, 5],
                )?);
            }
        }
        Ok(Self {
            source: HarmonicSource::load(vb.pp("m_source"), cfg.upsample_scale())?,
            conv_post: conv1d_load(
                vb.pp("conv_post"),
                ch,
                cfg.n_fft + 2,
                7,
                same_conv(7, 1),
                true,
            )?,
            stft: DftStft::load(vb.pp("stft"), cfg.n_fft, cfg.hop)?,
            cfg,
            noise_convs,
            noise_res,
            ups,
            resblocks,
        })
    }

    pub fn harmonic_spectrogram(
        &self,
        f0: &Tensor,
        deterministic: bool,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1()?;
        let src = self.source.forward(&f0v, f0.device(), deterministic, rng)?;
        let (mag, ph) = self.stft.transform(&src)?;
        Ok(Tensor::cat(&[mag, ph], 1)?)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        style: &Tensor,
        f0: &Tensor,
        deterministic: bool,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        let har = self.harmonic_spectrogram(f0, deterministic, rng)?;
        self.forward_with_har(x, style, &har)
    }

    pub fn forward_with_har(&self, x: &Tensor, style: &Tensor, har: &Tensor) -> Result<Tensor> {
        let n_up = self.ups.len();
        let nk = self.cfg.resblock_kernel_sizes.len();
        let mut x = x.clone();
        for i in 0..n_up {
            x = leaky_relu(&x, 0.1)?;
            let x_source = self.noise_res[i].forward(&self.noise_convs[i].forward(har)?, style)?;
            x = self.ups[i].forward(&x)?;
            if i == n_up - 1 {
                x = Tensor::cat(&[x.narrow(2, 1, 1)?, x], 2)?;
            }
            if x.dim(2)? != x_source.dim(2)? {
                return Err(anyhow!(
                    "generator stage {i}: {} frames vs source {}",
                    x.dim(2)?,
                    x_source.dim(2)?
                ));
            }
            x = (x + x_source)?;
            let mut xs: Option<Tensor> = None;
            for j in 0..nk {
                let r = self.resblocks[i * nk + j].forward(&x, style)?;
                xs = Some(match xs {
                    None => r,
                    Some(acc) => (acc + r)?,
                });
            }
            x = (xs.expect("nk >= 1") / nk as f64)?;
        }
        x = leaky_relu(&x, 0.01)?;
        x = self.conv_post.forward(&x)?;
        let bins = self.cfg.n_fft / 2 + 1;
        let spec = x.narrow(1, 0, bins)?.exp()?;
        let phase = x.narrow(1, bins, bins)?.sin()?;
        Ok(self.stft.inverse(&spec, &phase)?.squeeze(0)?)
    }

    pub fn adains_mut(&mut self) -> Vec<&mut AdaIn1d> {
        self.noise_res
            .iter_mut()
            .chain(self.resblocks.iter_mut())
            .flat_map(|b| b.adains_mut())
            .collect()
    }
}
