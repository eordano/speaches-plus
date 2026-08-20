use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Module};

use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::conv::Conv2d;
use nv_layers::linear::Linear;

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const SAMPLE_RATE: usize = 16_000;
const MAX_SECONDS: usize = 300;
const MEL_F_SP: f32 = 200.0 / 3.0;
const MEL_MIN_LOG_HZ: f32 = 1_000.0;

#[derive(Clone, Debug)]
pub struct AuTConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub num_mel_bins: usize,
    pub downsample_hidden_size: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub conv_chunksize: usize,
    pub max_source_positions: usize,
    pub output_dim: usize,
    pub dtype: DType,
}

impl Default for AuTConfig {
    fn default() -> Self {
        Self {
            d_model: 1280,
            n_heads: 20,
            n_layers: 32,
            ffn_dim: 5120,
            num_mel_bins: 128,
            downsample_hidden_size: 480,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            max_source_positions: 1500,
            output_dim: 2048,
            dtype: DType::BF16,
        }
    }
}

impl AuTConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn from_hf_config_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = path.as_ref();
        let raw = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?;
        let ac = v
            .get("thinker_config")
            .and_then(|t| t.get("audio_config"))
            .ok_or_else(|| anyhow::anyhow!("config.json: missing thinker_config.audio_config"))?;
        let obj = ac
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("audio_config must be an object"))?;
        let geti = |k: &str, d: usize| -> usize {
            obj.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d)
        };
        Ok(Self {
            d_model: geti("d_model", 1280),
            n_heads: geti("encoder_attention_heads", 20),
            n_layers: geti("encoder_layers", 32),
            ffn_dim: geti("encoder_ffn_dim", 5120),
            num_mel_bins: geti("num_mel_bins", 128),
            downsample_hidden_size: geti("downsample_hidden_size", 480),
            n_window: geti("n_window", 50),
            n_window_infer: geti("n_window_infer", 800),
            conv_chunksize: geti("conv_chunksize", 500),
            max_source_positions: geti("max_source_positions", 1500),
            output_dim: geti("output_dim", 2048),
            dtype: DType::BF16,
        })
    }
}

fn fdiv(a: i64, b: i64) -> i64 {
    (a as f64 / b as f64).floor() as i64
}

pub fn audio_tokens_for_mel_frames(frames: usize) -> usize {
    let n_window = 50i64;
    let chunk_len = n_window * 2;
    let f = frames as i64;
    let leave = f % chunk_len;
    let feat = fdiv(leave - 1, 2) + 1;
    let x = fdiv(fdiv(feat - 1, 2) + 1 - 1, 2) + 1 + (f / chunk_len) * 13;
    x.max(0) as usize
}

struct AudioLayer {
    attn_ln: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    ffn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl AudioLayer {
    fn new(cfg: &AuTConfig, device: &Device) -> Result<Self> {
        let d = cfg.d_model;
        let ffn = cfg.ffn_dim;
        let dtype = cfg.dtype;
        let ln = |dim: usize| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                Tensor::ones(dim, dtype, device)?,
                Tensor::zeros(dim, dtype, device)?,
                1e-5,
            ))
        };
        let lin = |o: usize, i: usize| -> Result<Linear> {
            Linear::new(Tensor::zeros((o, i), dtype, device)?, Some(Tensor::zeros(o, dtype, device)?))
        };
        Ok(Self {
            attn_ln: ln(d)?,
            q_proj: lin(d, d)?,
            k_proj: lin(d, d)?,
            v_proj: lin(d, d)?,
            out_proj: lin(d, d)?,
            ffn_ln: ln(d)?,
            fc1: lin(ffn, d)?,
            fc2: lin(d, ffn)?,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
        let h = self.n_heads;
        let hd = self.head_dim;

        let normed = self.attn_ln.forward(x)?;
        let q = self.q_proj.forward(&normed)?.reshape((b, t, h, hd))?;
        let k = self.k_proj.forward(&normed)?.reshape((b, t, h, hd))?;
        let v = self.v_proj.forward(&normed)?.reshape((b, t, h, hd))?;
        let attn_cfg = AttnConfig {
            num_heads: h,
            num_kv_heads: h,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: false,
        };
        let attn = sdpa(&q, &k, &v, &attn_cfg)?.reshape((b, t, h * hd))?;
        let x = (x + self.out_proj.forward(&attn)?).map_err(|e| anyhow::anyhow!(e))?;

        let normed = self.ffn_ln.forward(&x)?;
        let ff = self.fc2.forward(&self.fc1.forward(&normed)?.gelu_erf()?)?;
        (x + ff).map_err(|e| anyhow::anyhow!(e))
    }
}

pub struct AudioEncoder {
    cfg: AuTConfig,
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    positional: Tensor,
    layers: Vec<AudioLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    device: Device,
}

impl AudioEncoder {
    pub fn new(cfg: &AuTConfig, device: &Device) -> Result<Self> {
        if !cfg.d_model.is_multiple_of(cfg.n_heads) {
            anyhow::bail!("AuTConfig: d_model {} not divisible by n_heads {}", cfg.d_model, cfg.n_heads);
        }
        let dtype = cfg.dtype;
        let ds = cfg.downsample_hidden_size;
        let conv = |ic: usize, oc: usize| -> Result<Conv2d> {
            Conv2d::new(
                Tensor::zeros((oc, ic, 3, 3), dtype, device)?,
                Some(Tensor::zeros(oc, dtype, device)?),
                2,
                1,
            )
        };
        let conv_out_in = ds * conv_freq_out(cfg.num_mel_bins);
        let positional = whisper_sinusoids(cfg.max_source_positions, cfg.d_model, dtype, device)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(AudioLayer::new(cfg, device)?);
        }
        let ln = |dim: usize| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                Tensor::ones(dim, dtype, device)?,
                Tensor::zeros(dim, dtype, device)?,
                1e-5,
            ))
        };
        Ok(Self {
            cfg: cfg.clone(),
            conv2d1: conv(1, ds)?,
            conv2d2: conv(ds, ds)?,
            conv2d3: conv(ds, ds)?,
            conv_out: Linear::new(Tensor::zeros((cfg.d_model, conv_out_in), dtype, device)?, None)?,
            positional,
            layers,
            ln_post: ln(cfg.d_model)?,
            proj1: Linear::new(
                Tensor::zeros((cfg.d_model, cfg.d_model), dtype, device)?,
                Some(Tensor::zeros(cfg.d_model, dtype, device)?),
            )?,
            proj2: Linear::new(
                Tensor::zeros((cfg.output_dim, cfg.d_model), dtype, device)?,
                Some(Tensor::zeros(cfg.output_dim, dtype, device)?),
            )?,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &AuTConfig {
        &self.cfg
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (mel_c, frames) = mel.dims2().map_err(|e| anyhow::anyhow!(e))?;
        if mel_c != self.cfg.num_mel_bins {
            anyhow::bail!("AudioEncoder::forward: expected {} mel bins, got {}", self.cfg.num_mel_bins, mel_c);
        }
        let chunk = (self.cfg.n_window * 2) as usize;
        let mut chunk_lengths: Vec<usize> = Vec::new();
        let mut off = 0usize;
        while off < frames {
            let l = (frames - off).min(chunk);
            chunk_lengths.push(l);
            off += chunk;
        }
        if chunk_lengths.is_empty() {
            anyhow::bail!("AudioEncoder::forward: empty audio (0 frames)");
        }
        let max_chunk_len = *chunk_lengths.iter().max().unwrap();

        let melf = mel.to_dtype(self.cfg.dtype)?;
        let mut padded_chunks: Vec<Tensor> = Vec::with_capacity(chunk_lengths.len());
        let mut start = 0usize;
        for &l in &chunk_lengths {
            let piece = melf.narrow(1, start, l)?;
            let piece = if l < max_chunk_len {
                let pad = Tensor::zeros((mel_c, max_chunk_len - l), self.cfg.dtype, &self.device)?;
                Tensor::cat(&[&piece, &pad], 1)?
            } else {
                piece
            };
            padded_chunks.push(piece.unsqueeze(0)?.unsqueeze(0)?);
            start += l;
        }
        let refs: Vec<&Tensor> = padded_chunks.iter().collect();
        let padded = Tensor::cat(&refs, 0)?;

        let nc = chunk_lengths.len();
        let mut embeds: Vec<Tensor> = Vec::new();
        let mut b0 = 0usize;
        while b0 < nc {
            let bn = (nc - b0).min(self.cfg.conv_chunksize);
            let batch = padded.narrow(0, b0, bn)?;
            let y = self.conv2d1.forward(&batch)?.gelu_erf()?;
            let y = self.conv2d2.forward(&y)?.gelu_erf()?;
            let y = self.conv2d3.forward(&y)?.gelu_erf()?;
            embeds.push(y);
            b0 += bn;
        }
        let refs2: Vec<&Tensor> = embeds.iter().collect();
        let conv = Tensor::cat(&refs2, 0)?;

        let (bb, cc, ff, tt) = conv.dims4().map_err(|e| anyhow::anyhow!(e))?;
        let hidden = conv
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .reshape((bb, tt, cc * ff))?;
        let hidden = self.conv_out.forward(&hidden)?;
        let pos = self.positional.narrow(0, 0, tt)?.unsqueeze(0)?;
        let hidden = hidden.broadcast_add(&pos)?;

        let mut valid_rows: Vec<Tensor> = Vec::with_capacity(nc);
        for (i, &l) in chunk_lengths.iter().enumerate() {
            let vi = audio_tokens_for_mel_frames(l);
            valid_rows.push(hidden.i(i)?.narrow(0, 0, vi)?);
        }
        let refs3: Vec<&Tensor> = valid_rows.iter().collect();
        let tokens = Tensor::cat(&refs3, 0)?;
        let total = tokens.dim(0)?;
        let expect = audio_tokens_for_mel_frames(frames);
        if total != expect {
            anyhow::bail!("AudioEncoder::forward: produced {total} tokens, expected {expect}");
        }

        let window = max_len_after_cnn_full(max_chunk_len) * (self.cfg.n_window_infer / (self.cfg.n_window * 2));
        let window = window.max(1);
        let mut outs: Vec<Tensor> = Vec::new();
        let mut w0 = 0usize;
        while w0 < total {
            let wn = (total - w0).min(window);
            let mut h = tokens.narrow(0, w0, wn)?.unsqueeze(0)?;
            for layer in &self.layers {
                h = layer.forward(&h)?;
            }
            outs.push(h.squeeze(0)?);
            w0 += wn;
        }
        let refs4: Vec<&Tensor> = outs.iter().collect();
        let hs = Tensor::cat(&refs4, 0)?;

        let hs = self.ln_post.forward(&hs)?;
        let hs = self.proj1.forward(&hs)?.gelu_erf()?;
        let hs = self.proj2.forward(&hs)?;
        Ok(hs)
    }

    pub fn load_weights(&mut self, weights: &nv_weights::WeightLoader) -> Result<usize> {
        let dtype = self.cfg.dtype;
        let d = self.cfg.d_model;
        let ffn = self.cfg.ffn_dim;
        let ds = self.cfg.downsample_hidden_size;
        let out = self.cfg.output_dim;
        let conv_out_in = ds * conv_freq_out(self.cfg.num_mel_bins);
        let mut count = 0usize;

        let conv = |name_w: &str, name_b: &str, ic: usize, oc: usize| -> Result<Conv2d> {
            let w = load_nd(weights, name_w, &[oc, ic, 3, 3], dtype)?;
            let b = load_1d(weights, name_b, oc, dtype)?;
            Conv2d::new(w, Some(b), 2, 1)
        };
        self.conv2d1 = conv("thinker.audio_tower.conv2d1.weight", "thinker.audio_tower.conv2d1.bias", 1, ds)?;
        self.conv2d2 = conv("thinker.audio_tower.conv2d2.weight", "thinker.audio_tower.conv2d2.bias", ds, ds)?;
        self.conv2d3 = conv("thinker.audio_tower.conv2d3.weight", "thinker.audio_tower.conv2d3.bias", ds, ds)?;
        self.conv_out =
            Linear::new(load_2d(weights, "thinker.audio_tower.conv_out.weight", (d, conv_out_in), dtype)?, None)?;
        count += 7;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("thinker.audio_tower.layers.{i}");
            layer.attn_ln = load_ln(weights, &format!("{p}.self_attn_layer_norm"), d, dtype)?;
            layer.ffn_ln = load_ln(weights, &format!("{p}.final_layer_norm"), d, dtype)?;
            layer.q_proj = load_lin_bias(weights, &format!("{p}.self_attn.q_proj"), d, d, dtype)?;
            layer.k_proj = load_lin_bias(weights, &format!("{p}.self_attn.k_proj"), d, d, dtype)?;
            layer.v_proj = load_lin_bias(weights, &format!("{p}.self_attn.v_proj"), d, d, dtype)?;
            layer.out_proj = load_lin_bias(weights, &format!("{p}.self_attn.out_proj"), d, d, dtype)?;
            layer.fc1 = load_lin_bias(weights, &format!("{p}.fc1"), ffn, d, dtype)?;
            layer.fc2 = load_lin_bias(weights, &format!("{p}.fc2"), d, ffn, dtype)?;
            count += 16;
        }

        self.ln_post = load_ln(weights, "thinker.audio_tower.ln_post", d, dtype)?;
        self.proj1 = load_lin_bias(weights, "thinker.audio_tower.proj1", d, d, dtype)?;
        self.proj2 = load_lin_bias(weights, "thinker.audio_tower.proj2", out, d, dtype)?;
        count += 6;
        Ok(count)
    }
}

pub fn whisper_log_mel_128(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let capped = samples.len().min(MAX_SECONDS * SAMPLE_RATE);
        let samples = &samples[..capped];
        let frames = samples.len() / HOP_LENGTH;
        if frames == 0 {
            anyhow::bail!("whisper_log_mel_128: audio too short ({} samples)", samples.len());
        }
        let n_mels = 128usize;
        let n_bins = N_FFT / 2 + 1;
        let hann = build_hann_window();
        let filters = build_mel_filters(n_mels);

        let pad = N_FFT / 2;
        let mut padded = vec![0f32; samples.len() + N_FFT];
        for i in 0..pad {
            padded[i] = samples[pad - i];
        }
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        for i in 0..pad {
            let src = samples.len().saturating_sub(2 + i);
            padded[pad + samples.len() + i] = samples[src];
        }

        let mut mel = vec![0f32; n_mels * frames];
        let (cos_tab, sin_tab) = dft_tables();
        let mut power = vec![0f32; n_bins];
        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            for (k, pw) in power.iter_mut().enumerate() {
                let mut re = 0f32;
                let mut im = 0f32;
                for n in 0..N_FFT {
                    let x = padded[start + n] * hann[n];
                    re += x * cos_tab[k * N_FFT + n];
                    im -= x * sin_tab[k * N_FFT + n];
                }
                *pw = re * re + im * im;
            }
            for m in 0..n_mels {
                let row = &filters[m * n_bins..(m + 1) * n_bins];
                let mut sum = 0f32;
                for k in 0..n_bins {
                    sum += row[k] * power[k];
                }
                mel[m * frames + frame] = sum;
            }
        }

        let eps = 1e-10f32;
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
        Ok((mel, frames))
}

fn conv_freq_out(mel_bins: usize) -> usize {
    let mut n = mel_bins as i64;
    for _ in 0..3 {
        n = fdiv(n - 1, 2) + 1;
    }
    n.max(0) as usize
}

fn max_len_after_cnn_full(max_chunk_len: usize) -> usize {
    audio_tokens_for_mel_frames(max_chunk_len).max(1)
}

fn whisper_sinusoids(length: usize, channels: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    if !channels.is_multiple_of(2) {
        anyhow::bail!("whisper_sinusoids needs even channels, got {channels}");
    }
    let half = channels / 2;
    let log_inc = (10_000f32).ln() / (half as f32 - 1.0);
    let mut data = vec![0f32; length * channels];
    for pos in 0..length {
        for i in 0..half {
            let inv = (-log_inc * i as f32).exp();
            let angle = pos as f32 * inv;
            data[pos * channels + i] = angle.sin();
            data[pos * channels + half + i] = angle.cos();
        }
    }
    Ok(Tensor::from_vec(data, (length, channels), device)?.to_dtype(dtype)?)
}

fn dft_tables() -> (Vec<f32>, Vec<f32>) {
    let n_bins = N_FFT / 2 + 1;
    let mut cos_tab = vec![0f32; n_bins * N_FFT];
    let mut sin_tab = vec![0f32; n_bins * N_FFT];
    for k in 0..n_bins {
        for n in 0..N_FFT {
            let ang = 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (N_FFT as f32);
            cos_tab[k * N_FFT + n] = ang.cos();
            sin_tab[k * N_FFT + n] = ang.sin();
        }
    }
    (cos_tab, sin_tab)
}

fn build_hann_window() -> Vec<f32> {
    let mut w = vec![0f32; N_FFT];
    for (i, slot) in w.iter_mut().enumerate() {
        let phase = 2.0 * std::f32::consts::PI * (i as f32) / (N_FFT as f32);
        *slot = 0.5 - 0.5 * phase.cos();
    }
    w
}

fn build_mel_filters(n_mels: usize) -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let m_min = hz_to_mel(0.0);
    let m_max = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let mut mel_points = vec![0f32; n_mels + 2];
    for (i, slot) in mel_points.iter_mut().enumerate() {
        let frac = i as f32 / (n_mels as f32 + 1.0);
        *slot = m_min + (m_max - m_min) * frac;
    }
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let mut fft_freqs = vec![0f32; n_bins];
    for (i, slot) in fft_freqs.iter_mut().enumerate() {
        *slot = i as f32 * SAMPLE_RATE as f32 / N_FFT as f32;
    }
    let mut filters = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        let lower_slope = (center - lower).max(f32::EPSILON);
        let upper_slope = (upper - center).max(f32::EPSILON);
        let enorm = 2.0 / (upper - lower).max(f32::EPSILON);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let mut weight = 0f32;
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

fn hz_to_mel(f: f32) -> f32 {
    let m_min_log = MEL_MIN_LOG_HZ / MEL_F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if f >= MEL_MIN_LOG_HZ {
        m_min_log + (f / MEL_MIN_LOG_HZ).ln() / logstep
    } else {
        f / MEL_F_SP
    }
}

fn mel_to_hz(m: f32) -> f32 {
    let m_min_log = MEL_MIN_LOG_HZ / MEL_F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if m >= m_min_log {
        MEL_MIN_LOG_HZ * ((m - m_min_log) * logstep).exp()
    } else {
        MEL_F_SP * m
    }
}

fn load_1d(weights: &nv_weights::WeightLoader, name: &str, dim: usize, dtype: DType) -> Result<Tensor> {
    let w = weights.get(name, dtype).with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("{name}: expected [{}], got {:?}", dim, d);
    }
    Ok(w)
}

fn load_2d(weights: &nv_weights::WeightLoader, name: &str, shape: (usize, usize), dtype: DType) -> Result<Tensor> {
    let w = weights.get(name, dtype).with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != shape.0 || d[1] != shape.1 {
        anyhow::bail!("{name}: expected [{}, {}], got {:?}", shape.0, shape.1, d);
    }
    Ok(w)
}

fn load_nd(weights: &nv_weights::WeightLoader, name: &str, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let w = weights.get(name, dtype).with_context(|| format!("load {name}"))?;
    if w.dims() != shape {
        anyhow::bail!("{name}: expected {:?}, got {:?}", shape, w.dims());
    }
    Ok(w)
}

fn load_ln(weights: &nv_weights::WeightLoader, prefix: &str, dim: usize, dtype: DType) -> Result<LayerNorm> {
    let w = load_1d(weights, &format!("{prefix}.weight"), dim, dtype)?;
    let b = load_1d(weights, &format!("{prefix}.bias"), dim, dtype)?;
    Ok(LayerNorm::new(w, b, 1e-5))
}

fn load_lin_bias(
    weights: &nv_weights::WeightLoader,
    prefix: &str,
    out_f: usize,
    in_f: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = load_2d(weights, &format!("{prefix}.weight"), (out_f, in_f), dtype)?;
    let b = load_1d(weights, &format!("{prefix}.bias"), out_f, dtype)?;
    Linear::new(w, Some(b))
}

use candle_core::IndexOp;

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> AuTConfig {
        AuTConfig {
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 64,
            num_mel_bins: 128,
            downsample_hidden_size: 8,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            max_source_positions: 128,
            output_dim: 48,
            dtype: DType::F32,
        }
    }

    #[test]
    fn output_length_law_matches_conv_stack_over_range() {
        for frames in 1..=350usize {
            let got = audio_tokens_for_mel_frames(frames);
            let chunk = 100usize;
            let mut off = 0;
            let mut total = 0usize;
            while off < frames {
                let l = (frames - off).min(chunk);
                let mut n = l as i64;
                for _ in 0..3 {
                    n = fdiv(n - 1, 2) + 1;
                }
                total += n.max(0) as usize;
                off += chunk;
            }
            assert_eq!(got, total, "frames {frames}: law {got} != per-chunk conv {total}");
        }
        assert_eq!(audio_tokens_for_mel_frames(300), 39);
        assert_eq!(audio_tokens_for_mel_frames(100), 13);
    }

    #[test]
    fn conv_freq_out_is_16_for_128_bins() {
        assert_eq!(conv_freq_out(128), 16);
    }

    #[test]
    fn whisper_sinusoid_spot_values() {
        let t = whisper_sinusoids(4, 8, DType::F32, &Device::Cpu).unwrap();
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        let half = 4usize;
        let log_inc = (10_000f32).ln() / (half as f32 - 1.0);
        for pos in 0..4usize {
            for i in 0..half {
                let inv = (-log_inc * i as f32).exp();
                let ang = pos as f32 * inv;
                assert!((v[pos * 8 + i] - ang.sin()).abs() < 1e-5);
                assert!((v[pos * 8 + half + i] - ang.cos()).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn forward_shape_matches_output_length_law() {
        let cfg = tiny_cfg();
        let enc = AudioEncoder::new(&cfg, &Device::Cpu).unwrap();
        let frames = 250usize;
        let mel = Tensor::zeros((128, frames), DType::F32, &Device::Cpu).unwrap();
        let out = enc.forward(&mel).unwrap();
        assert_eq!(out.dims(), &[audio_tokens_for_mel_frames(frames), cfg.output_dim]);
    }

    #[test]
    fn mel_shape_is_128_by_frames() {
        let samples = vec![0.01f32; 16_000];
        let (mel, frames) = whisper_log_mel_128(&samples).unwrap();
        assert_eq!(frames, 100);
        assert_eq!(mel.len(), 128 * frames);
        assert!(mel.iter().all(|x| x.is_finite()));
    }
}
