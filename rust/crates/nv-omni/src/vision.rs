use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{LayerNorm, Module};

use nv_layers::attn::{attention, AttnConfig};
use nv_layers::linear::Linear;

use crate::qwen3_vision::Qwen3VisionConfig;

const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const ROPE_THETA_VISION: f32 = 10_000.0;
const SMART_RESIZE_MIN_PIXELS: usize = 3136;
const SMART_RESIZE_MAX_PIXELS: usize = 12_845_056;

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionBlock {
    fn new(cfg: &Qwen3VisionConfig, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let dtype = cfg.dtype;
        Ok(Self {
            norm1: ln(h, dtype, device, cfg.layer_norm_eps)?,
            norm2: ln(h, dtype, device, cfg.layer_norm_eps)?,
            qkv: lin_bias_zeros(3 * h, h, dtype, device)?,
            proj: lin_bias_zeros(h, h, dtype, device)?,
            fc1: lin_bias_zeros(inter, h, dtype, device)?,
            fc2: lin_bias_zeros(h, inter, dtype, device)?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (seq, _h) = x.dims2().map_err(|e| anyhow::anyhow!(e))?;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let half = hd / 2;

        let normed = self.norm1.forward(x)?;
        let qkv = self.qkv.forward(&normed)?.reshape((seq, 3, nh, hd))?;
        let q = qkv.i((.., 0, .., ..))?.contiguous()?;
        let k = qkv.i((.., 1, .., ..))?.contiguous()?;
        let v = qkv.i((.., 2, .., ..))?.contiguous()?;
        let q = rotate(&q, cos, sin, half)?;
        let k = rotate(&k, cos, sin, half)?;

        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let v = v.unsqueeze(0)?;
        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nh,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: false,
        };
        let attn = attention(&q, &k, &v, &attn_cfg)?.reshape((seq, nh * hd))?;
        let x = (x + self.proj.forward(&attn)?).map_err(|e| anyhow::anyhow!(e))?;

        let normed = self.norm2.forward(&x)?;
        let ff = self.fc2.forward(&gelu_pytorch_tanh(&self.fc1.forward(&normed)?)?)?;
        (x + ff).map_err(|e| anyhow::anyhow!(e))
    }
}

struct Merger {
    ln_q: LayerNorm,
    mlp0: Linear,
    mlp2: Linear,
    postshuffle: bool,
    merged: usize,
}

impl Merger {
    fn new(cfg: &Qwen3VisionConfig, postshuffle: bool, device: &Device) -> Result<Self> {
        let merged = cfg.merger_hidden();
        let out = cfg.out_hidden_size;
        let dtype = cfg.dtype;
        let ln_dim = if postshuffle { merged } else { cfg.hidden_size };
        Ok(Self {
            ln_q: ln(ln_dim, dtype, device, 1e-6)?,
            mlp0: lin_bias_zeros(merged, merged, dtype, device)?,
            mlp2: lin_bias_zeros(out, merged, dtype, device)?,
            postshuffle,
            merged,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, h) = x.dims2().map_err(|e| anyhow::anyhow!(e))?;
        let group = self.merged / h;
        if seq % group != 0 {
            anyhow::bail!("Merger: seq {seq} not divisible by group {group}");
        }
        let n = seq / group;
        let hidden = if self.postshuffle {
            let regrouped = x.reshape((n, self.merged))?;
            self.ln_q.forward(&regrouped)?
        } else {
            let normed = self.ln_q.forward(x)?;
            normed.reshape((n, self.merged))?
        };
        let y = self.mlp0.forward(&hidden)?;
        let y = y.gelu_erf()?;
        Ok(self.mlp2.forward(&y)?)
    }
}

pub struct OmniVisionEncoder {
    cfg: Qwen3VisionConfig,
    deepstack_indexes: Vec<usize>,
    patch_proj: Linear,
    pos_embed: Tensor,
    rot_inv_freq: Vec<f32>,
    blocks: Vec<VisionBlock>,
    merger: Merger,
    merger_list: Vec<Merger>,
    num_grid_per_side: usize,
    device: Device,
}

impl OmniVisionEncoder {
    pub fn new(cfg: Qwen3VisionConfig, deepstack_indexes: Vec<usize>, device: &Device) -> Result<Self> {
        if !cfg.hidden_size.is_multiple_of(cfg.num_heads) {
            anyhow::bail!("vision hidden {} not divisible by heads {}", cfg.hidden_size, cfg.num_heads);
        }
        let dtype = cfg.dtype;
        let c = cfg.in_channels;
        let tp = cfg.temporal_patch_size;
        let p = cfg.patch_size;
        let patch_in = c * tp * p * p;
        let head_dim = cfg.head_dim();
        let rot_dim = head_dim / 2;
        let rot_inv_freq: Vec<f32> = (0..rot_dim / 2)
            .map(|i| 1.0 / ROPE_THETA_VISION.powf((2 * i) as f32 / rot_dim as f32))
            .collect();
        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            blocks.push(VisionBlock::new(&cfg, device)?);
        }
        let mut merger_list = Vec::with_capacity(deepstack_indexes.len());
        for _ in 0..deepstack_indexes.len() {
            merger_list.push(Merger::new(&cfg, true, device)?);
        }
        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        Ok(Self {
            patch_proj: Linear::new(
                Tensor::zeros((cfg.hidden_size, patch_in), dtype, device)?,
                Some(Tensor::zeros(cfg.hidden_size, dtype, device)?),
            )?,
            pos_embed: Tensor::zeros((cfg.num_position_embeddings, cfg.hidden_size), dtype, device)?,
            rot_inv_freq,
            blocks,
            merger: Merger::new(&cfg, false, device)?,
            merger_list,
            deepstack_indexes,
            num_grid_per_side,
            device: device.clone(),
            cfg,
        })
    }

    pub fn from_hf_config_json(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let p = path.as_ref();
        let raw = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let mut vc = v
            .get("thinker_config")
            .and_then(|t| t.get("vision_config"))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("config.json: missing thinker_config.vision_config"))?;
        if vc.get("num_position_embeddings").and_then(|x| x.as_u64()).is_none() {
            let image_size = vc
                .get("image_size")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow::anyhow!("vision_config: no num_position_embeddings and no image_size to derive it"))?;
            let patch_size = vc
                .get("patch_size")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow::anyhow!("vision_config: no num_position_embeddings and no patch_size to derive it"))?;
            let side = image_size / patch_size;
            vc.as_object_mut()
                .expect("vision_config is an object")
                .insert("num_position_embeddings".to_string(), serde_json::json!(side * side));
        }
        let deepstack: Vec<usize> = vc
            .get("deepstack_visual_indexes")
            .and_then(|d| d.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_else(|| vec![8, 16, 24]);
        let cfg = Qwen3VisionConfig::from_hf_value(&vc)?;
        Self::new(cfg, deepstack, device)
    }

    pub fn config(&self) -> &Qwen3VisionConfig {
        &self.cfg
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        smart_resize(height, width, 32, SMART_RESIZE_MIN_PIXELS, SMART_RESIZE_MAX_PIXELS)
    }

    pub fn patchify_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<(Tensor, (usize, usize, usize))> {
        let p = self.cfg.patch_size;
        let m = self.cfg.spatial_merge_size;
        let c = self.cfg.in_channels;
        let tp = self.cfg.temporal_patch_size;
        if width % (p * m) != 0 || height % (p * m) != 0 {
            anyhow::bail!(
                "patchify_rgb: {width}x{height} must be multiples of patch*merge={}",
                p * m
            );
        }
        if rgb.len() != width * height * c {
            anyhow::bail!("patchify_rgb: rgb len {} != {}*{}*{}", rgb.len(), width, height, c);
        }
        let gh = height / p;
        let gw = width / p;
        let seq = gh * gw;
        let feat = c * tp * p * p;
        let mut data = vec![0f32; seq * feat];
        let mut row = 0usize;
        for hb in 0..gh / m {
            for wb in 0..gw / m {
                for hi in 0..m {
                    for wi in 0..m {
                        let patch_row = hb * m + hi;
                        let patch_col = wb * m + wi;
                        let base = row * feat;
                        for ci in 0..c {
                            for tpi in 0..tp {
                                for ph in 0..p {
                                    for pw in 0..p {
                                        let py = patch_row * p + ph;
                                        let px = patch_col * p + pw;
                                        let pix = rgb[(py * width + px) * c + ci] as f32;
                                        let norm = (pix / 255.0 - CLIP_MEAN[ci]) / CLIP_STD[ci];
                                        let fi = ci * (tp * p * p) + tpi * (p * p) + ph * p + pw;
                                        data[base + fi] = norm;
                                    }
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
        let t = Tensor::from_vec(data, (seq, feat), device)?;
        Ok((t, (1, gh, gw)))
    }

    pub fn forward(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<(Tensor, Vec<Tensor>)> {
        let (t, gh, gw) = grid;
        let (seq, _feat) = patches.dims2().map_err(|e| anyhow::anyhow!(e))?;
        if seq != t * gh * gw {
            anyhow::bail!("vision forward: seq {seq} != t*gh*gw {}", t * gh * gw);
        }
        let dtype = self.cfg.dtype;
        let mut x = self.patch_proj.forward(&patches.to_dtype(dtype)?)?;

        let pos = self.bilinear_pos_embed(grid)?;
        x = x.broadcast_add(&pos.to_dtype(dtype)?)?;

        let (cos, sin) = self.rotary_cos_sin(grid)?;

        let mut deepstack: Vec<Tensor> = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &cos, &sin)?;
            if let Some(k) = self.deepstack_indexes.iter().position(|&d| d == i) {
                deepstack.push(self.merger_list[k].forward(&x)?);
            }
        }
        let merged = self.merger.forward(&x)?;
        Ok((merged, deepstack))
    }

    fn bilinear_pos_embed(&self, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, gh, gw) = grid;
        let m = self.cfg.spatial_merge_size;
        let side = self.num_grid_per_side;
        let seq = t * gh * gw;

        let lin = |n: usize| -> Vec<f32> {
            if n <= 1 {
                vec![0.0; n]
            } else {
                (0..n).map(|i| i as f32 * (side as f32 - 1.0) / (n as f32 - 1.0)).collect()
            }
        };
        let h_grid = lin(gh);
        let w_grid = lin(gw);

        let mut idx: Vec<Vec<u32>> = vec![Vec::with_capacity(seq); 4];
        let mut wgt: Vec<Vec<f32>> = vec![Vec::with_capacity(seq); 4];
        for _t in 0..t {
            for hb in 0..gh / m {
                for wb in 0..gw / m {
                    for hi in 0..m {
                        for wi in 0..m {
                            let hh = hb * m + hi;
                            let ww = wb * m + wi;
                            let hg = h_grid[hh];
                            let wg = w_grid[ww];
                            let hf = hg as usize;
                            let wf = wg as usize;
                            let hc = (hf + 1).min(side - 1);
                            let wc = (wf + 1).min(side - 1);
                            let hfrac = hg - hf as f32;
                            let wfrac = wg - wf as f32;
                            let ho = hf * side;
                            let hco = hc * side;
                            idx[0].push((ho + wf) as u32);
                            idx[1].push((ho + wc) as u32);
                            idx[2].push((hco + wf) as u32);
                            idx[3].push((hco + wc) as u32);
                            wgt[0].push((1.0 - hfrac) * (1.0 - wfrac));
                            wgt[1].push((1.0 - hfrac) * wfrac);
                            wgt[2].push(hfrac * (1.0 - wfrac));
                            wgt[3].push(hfrac * wfrac);
                        }
                    }
                }
            }
        }
        let pe = self.pos_embed.to_dtype(DType::F32)?;
        let mut acc: Option<Tensor> = None;
        for i in 0..4 {
            let it = Tensor::from_vec(idx[i].clone(), seq, &self.device)?;
            let wt = Tensor::from_vec(wgt[i].clone(), (seq, 1), &self.device)?;
            let gathered = pe.index_select(&it, 0)?;
            let term = gathered.broadcast_mul(&wt)?;
            acc = Some(match acc {
                Some(a) => a.add(&term)?,
                None => term,
            });
        }
        Ok(acc.unwrap())
    }

    fn rotary_cos_sin(&self, grid: (usize, usize, usize)) -> Result<(Tensor, Tensor)> {
        let (t, gh, gw) = grid;
        let m = self.cfg.spatial_merge_size;
        let seq = t * gh * gw;
        let n_inv = self.rot_inv_freq.len();
        let half = n_inv * 2;
        let mut freqs = vec![0f32; seq * half];
        let mut row = 0usize;
        for _t in 0..t {
            for hb in 0..gh / m {
                for wb in 0..gw / m {
                    for hi in 0..m {
                        for wi in 0..m {
                            let hpos = (hb * m + hi) as f32;
                            let wpos = (wb * m + wi) as f32;
                            let base = row * half;
                            for i in 0..n_inv {
                                freqs[base + i] = hpos * self.rot_inv_freq[i];
                                freqs[base + n_inv + i] = wpos * self.rot_inv_freq[i];
                            }
                            row += 1;
                        }
                    }
                }
            }
        }
        let f = Tensor::from_vec(freqs, (seq, half), &self.device)?;
        Ok((f.cos()?, f.sin()?))
    }

    pub fn load_weights(&mut self, weights: &nv_weights::WeightLoader) -> Result<usize> {
        let dtype = self.cfg.dtype;
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let c = self.cfg.in_channels;
        let tp = self.cfg.temporal_patch_size;
        let p = self.cfg.patch_size;
        let out = self.cfg.out_hidden_size;
        let merged = self.cfg.merger_hidden();
        let npos = self.cfg.num_position_embeddings;
        let patch_in = c * tp * p * p;
        let mut count = 0usize;

        let pw = weights
            .get("thinker.visual.patch_embed.proj.weight", dtype)
            .context("load patch_embed.proj.weight")?;
        if pw.elem_count() != h * patch_in {
            anyhow::bail!("patch_embed.proj.weight has {} elems, expected {}", pw.elem_count(), h * patch_in);
        }
        let pw = pw.reshape((h, patch_in))?;
        let pb = load_1d(weights, "thinker.visual.patch_embed.proj.bias", h, dtype)?;
        self.patch_proj = Linear::new(pw, Some(pb))?;
        self.pos_embed = load_2d(weights, "thinker.visual.pos_embed.weight", (npos, h), dtype)?;
        count += 3;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let bp = format!("thinker.visual.blocks.{i}");
            block.norm1 = load_ln(weights, &format!("{bp}.norm1"), h, dtype, self.cfg.layer_norm_eps)?;
            block.norm2 = load_ln(weights, &format!("{bp}.norm2"), h, dtype, self.cfg.layer_norm_eps)?;
            block.qkv = load_lin_bias(weights, &format!("{bp}.attn.qkv"), 3 * h, h, dtype)?;
            block.proj = load_lin_bias(weights, &format!("{bp}.attn.proj"), h, h, dtype)?;
            block.fc1 = load_lin_bias(weights, &format!("{bp}.mlp.linear_fc1"), inter, h, dtype)?;
            block.fc2 = load_lin_bias(weights, &format!("{bp}.mlp.linear_fc2"), h, inter, dtype)?;
            count += 12;
        }

        self.merger.ln_q = load_ln(weights, "thinker.visual.merger.ln_q", h, dtype, 1e-6)?;
        self.merger.mlp0 = load_lin_bias(weights, "thinker.visual.merger.mlp.0", merged, merged, dtype)?;
        self.merger.mlp2 = load_lin_bias(weights, "thinker.visual.merger.mlp.2", out, merged, dtype)?;
        count += 6;

        for (k, mg) in self.merger_list.iter_mut().enumerate() {
            let mp = format!("thinker.visual.merger_list.{k}");
            mg.ln_q = load_ln(weights, &format!("{mp}.ln_q"), merged, dtype, 1e-6)?;
            mg.mlp0 = load_lin_bias(weights, &format!("{mp}.mlp.0"), merged, merged, dtype)?;
            mg.mlp2 = load_lin_bias(weights, &format!("{mp}.mlp.2"), out, merged, dtype)?;
            count += 6;
        }
        Ok(count)
    }
}

pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let hf = height as f64;
    let wf = width as f64;
    let ff = factor as f64;
    let round_to = |x: f64| -> usize { ((x / ff).round() as usize).max(1) * factor };
    let mut h_bar = round_to(hf);
    let mut w_bar = round_to(wf);
    if h_bar * w_bar > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = ((hf / beta / ff).floor() as usize).max(1) * factor;
        w_bar = ((wf / beta / ff).floor() as usize).max(1) * factor;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ((hf * beta / ff).ceil() as usize).max(1) * factor;
        w_bar = ((wf * beta / ff).ceil() as usize).max(1) * factor;
    }
    if h_bar * w_bar > max_pixels {
        if h_bar <= w_bar {
            w_bar = (max_pixels / h_bar / factor).max(1) * factor;
        } else {
            h_bar = (max_pixels / w_bar / factor).max(1) * factor;
        }
    }
    (h_bar, w_bar)
}

fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor, half: usize) -> Result<Tensor> {
    let dtype = x.dtype();
    let dims = x.dims().to_vec();
    let head_dim = *dims.last().unwrap();
    let n_heads = dims[dims.len() - 2];
    let tokens: usize = dims[..dims.len() - 2].iter().product();
    let xf = x.to_dtype(DType::F32)?.reshape((tokens, n_heads, head_dim))?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let lo = xf.narrow(2, 0, half)?;
    let hi = xf.narrow(2, half, half)?;
    let out_lo = lo.broadcast_mul(&cos)?.sub(&hi.broadcast_mul(&sin)?)?;
    let out_hi = lo.broadcast_mul(&sin)?.add(&hi.broadcast_mul(&cos)?)?;
    let out = Tensor::cat(&[&out_lo, &out_hi], 2)?;
    Ok(out.reshape(dims)?.to_dtype(dtype)?)
}

fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let x3 = xf.powf(3.0)?;
    let inner = ((&xf + (x3 * 0.044715f64)?)? * (k as f64))?;
    let t = inner.tanh()?;
    let y = (&xf * 0.5f64)?.mul(&(t + 1.0f64)?)?;
    Ok(y.to_dtype(dtype)?)
}

fn ln(dim: usize, dtype: DType, device: &Device, eps: f64) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        Tensor::ones(dim, dtype, device)?,
        Tensor::zeros(dim, dtype, device)?,
        eps,
    ))
}

fn lin_bias_zeros(out_f: usize, in_f: usize, dtype: DType, device: &Device) -> Result<Linear> {
    Linear::new(
        Tensor::zeros((out_f, in_f), dtype, device)?,
        Some(Tensor::zeros(out_f, dtype, device)?),
    )
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

fn load_ln(
    weights: &nv_weights::WeightLoader,
    prefix: &str,
    dim: usize,
    dtype: DType,
    eps: f64,
) -> Result<LayerNorm> {
    let w = load_1d(weights, &format!("{prefix}.weight"), dim, dtype)?;
    let b = load_1d(weights, &format!("{prefix}.bias"), dim, dtype)?;
    Ok(LayerNorm::new(w, b, eps))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen3VisionConfig {
        Qwen3VisionConfig {
            depth: 3,
            hidden_size: 32,
            num_heads: 4,
            intermediate_size: 64,
            in_channels: 3,
            patch_size: 4,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 64,
            out_hidden_size: 48,
            layer_norm_eps: 1e-6,
            dtype: DType::F32,
        }
    }

    #[test]
    fn smart_resize_multiple_of_factor() {
        let (h, w) = smart_resize(100, 200, 32, SMART_RESIZE_MIN_PIXELS, SMART_RESIZE_MAX_PIXELS);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h * w >= SMART_RESIZE_MIN_PIXELS);
    }

    #[test]
    fn smart_resize_never_exceeds_the_pixel_budget_even_for_degenerate_aspects() {
        for (h, w) in [
            (1usize, 1_000_000usize),
            (1_000_000, 1),
            (1, 1),
            (200, 30_000),
            (30_000, 200),
            (12_000, 12_000),
        ] {
            let (hb, wb) = smart_resize(h, w, 32, SMART_RESIZE_MIN_PIXELS, SMART_RESIZE_MAX_PIXELS);
            assert!(
                hb * wb <= SMART_RESIZE_MAX_PIXELS,
                "{h}x{w} resized to {hb}x{wb}, over the {SMART_RESIZE_MAX_PIXELS}-pixel budget"
            );
            assert_eq!(hb % 32, 0);
            assert_eq!(wb % 32, 0);
        }
    }

    #[test]
    fn patchify_shape_and_grid() {
        let enc = OmniVisionEncoder::new(tiny_cfg(), vec![1], &Device::Cpu).unwrap();
        let (w, h) = (8usize, 8usize);
        let rgb = vec![128u8; w * h * 3];
        let (patches, grid) = enc.patchify_rgb(&rgb, w, h, &Device::Cpu).unwrap();
        assert_eq!(grid, (1, 2, 2));
        assert_eq!(patches.dims(), &[4, 3 * 2 * 4 * 4]);
    }

    #[test]
    fn forward_shapes_and_deepstack() {
        let cfg = tiny_cfg();
        let enc = OmniVisionEncoder::new(cfg.clone(), vec![0, 1], &Device::Cpu).unwrap();
        let (w, h) = (16usize, 16usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 255) as u8).collect();
        let (patches, grid) = enc.patchify_rgb(&rgb, w, h, &Device::Cpu).unwrap();
        let (main, deep) = enc.forward(&patches, grid).unwrap();
        let seq = grid.1 * grid.2;
        assert_eq!(main.dims(), &[seq / 4, cfg.out_hidden_size]);
        assert_eq!(deep.len(), 2);
        for d in &deep {
            assert_eq!(d.dims(), &[seq / 4, cfg.out_hidden_size]);
        }
        let v: Vec<f32> = main.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
