use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use nv_layers::attn::AttnConfig;
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_weights::WeightLoader;

use super::preprocess::{position_ids, PreparedImage, MERGE_SIZE, PATCH_DIM};
use crate::deepseek_ocr::sam::layer_norm;

pub const VISION_ROPE_THETA: f64 = 10000.0;

#[derive(Clone, Debug)]
pub struct DotsVisionConfig {
    pub embed_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub rms_norm_eps: f64,
    pub post_norm: bool,
}

impl Default for DotsVisionConfig {
    fn default() -> Self {
        Self {
            embed_dim: 1536,
            hidden_size: 1536,
            intermediate_size: 4224,
            num_hidden_layers: 42,
            num_attention_heads: 12,
            patch_size: 14,
            spatial_merge_size: 2,
            rms_norm_eps: 1e-5,
            post_norm: true,
        }
    }
}

impl DotsVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_attention_heads
    }
}

struct VisionBlock {
    norm1: RmsNorm,
    qkv: Linear,
    proj: Linear,
    norm2: RmsNorm,
    mlp: Mlp,
}

pub struct PatchMerger {
    ln_w: Tensor,
    ln_b: Tensor,
    fc0: Linear,
    fc2: Linear,
    context_dim: usize,
    merge_units: usize,
}

pub struct DotsVisionTower {
    cfg: DotsVisionConfig,
    patch_proj: Linear,
    patch_norm: RmsNorm,
    blocks: Vec<VisionBlock>,
    post_trunk_norm: Option<RmsNorm>,
    merger: PatchMerger,
    device: Device,
    dtype: DType,
}

fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    bias: Option<&str>,
    dtype: DType,
) -> Result<Linear> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let w = if w.dims().len() > 2 {
        let rows = w.dim(0)?;
        let cols: usize = w.dims()[1..].iter().product();
        w.reshape((rows, cols))?
    } else {
        w
    };
    let d = w.dims2()?;
    anyhow::ensure!(
        d == (out_dim, in_dim),
        "{name}: expected [{out_dim}, {in_dim}], got {:?}",
        w.dims()
    );
    let b = match bias {
        Some(bn) => Some(
            weights
                .get(bn, dtype)
                .with_context(|| format!("load {bn}"))?,
        ),
        None => None,
    };
    Linear::new(w.contiguous()?, b)
}

fn load_rmsnorm(weights: &WeightLoader, name: &str, dim: usize, eps: f64) -> Result<RmsNorm> {
    let w = weights
        .get(name, DType::F32)
        .with_context(|| format!("load {name}"))?;
    anyhow::ensure!(
        w.dims() == [dim],
        "{name}: expected [{dim}], got {:?}",
        w.dims()
    );
    Ok(RmsNorm::new(w, eps))
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

pub fn vision_rope_tables(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let quarter = head_dim / 4;
    let inv: Vec<f64> = (0..quarter)
        .map(|i| 1.0 / VISION_ROPE_THETA.powf((2 * i) as f64 / (head_dim / 2) as f64))
        .collect();
    let ids = position_ids(grid_h, grid_w);
    let n = ids.len();
    let mut cos = vec![0f32; n * head_dim];
    let mut sin = vec![0f32; n * head_dim];
    for (row, (hy, wx)) in ids.iter().enumerate() {
        let base = row * head_dim;
        for i in 0..quarter {
            let fh = *hy as f64 * inv[i];
            let fw = *wx as f64 * inv[i];
            let (ch, sh) = (fh.cos() as f32, fh.sin() as f32);
            let (cw, sw) = (fw.cos() as f32, fw.sin() as f32);
            cos[base + i] = ch;
            cos[base + quarter + i] = cw;
            cos[base + head_dim / 2 + i] = ch;
            cos[base + head_dim / 2 + quarter + i] = cw;
            sin[base + i] = sh;
            sin[base + quarter + i] = sw;
            sin[base + head_dim / 2 + i] = sh;
            sin[base + head_dim / 2 + quarter + i] = sw;
        }
    }
    let cos = Tensor::from_vec(cos, (1, n, 1, head_dim), device)?;
    let sin = Tensor::from_vec(sin, (1, n, 1, head_dim), device)?;
    Ok((cos, sin))
}

fn apply_vision_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let out = x32
        .broadcast_mul(cos)?
        .add(&rotate_half(&x32)?.broadcast_mul(sin)?)?;
    Ok(out.to_dtype(dtype)?)
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    nv_layers::attn::attention(q, k, v, cfg)
}

impl PatchMerger {
    fn from_loader(
        weights: &WeightLoader,
        prefix: &str,
        context_dim: usize,
        out_dim: usize,
        merge_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let merge_units = merge_size * merge_size;
        let hidden = context_dim * merge_units;
        let ln_w = weights
            .get(&format!("{prefix}ln_q.weight"), DType::F32)
            .context("load merger ln_q.weight")?;
        let ln_b = weights
            .get(&format!("{prefix}ln_q.bias"), DType::F32)
            .context("load merger ln_q.bias")?;
        let fc0 = load_linear(
            weights,
            &format!("{prefix}mlp.0.weight"),
            hidden,
            hidden,
            Some(&format!("{prefix}mlp.0.bias")),
            dtype,
        )?;
        let fc2 = load_linear(
            weights,
            &format!("{prefix}mlp.2.weight"),
            out_dim,
            hidden,
            Some(&format!("{prefix}mlp.2.bias")),
            dtype,
        )?;
        Ok(Self {
            ln_w,
            ln_b,
            fc0,
            fc2,
            context_dim,
            merge_units,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        anyhow::ensure!(
            n % self.merge_units == 0,
            "PatchMerger: {n} patches not divisible by {}",
            self.merge_units
        );
        let normed = layer_norm(x, &self.ln_w, &self.ln_b, 1e-6)?;
        let grouped =
            normed.reshape((n / self.merge_units, self.context_dim * self.merge_units))?;
        let h = self.fc0.forward(&grouped)?;
        let h = h.gelu_erf()?;
        self.fc2.forward(&h)
    }
}

impl DotsVisionTower {
    pub fn from_loader(
        weights: &WeightLoader,
        prefix: &str,
        cfg: DotsVisionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let dim = cfg.embed_dim;
        let patch_proj = load_linear(
            weights,
            &format!("{prefix}patch_embed.patchifier.proj.weight"),
            dim,
            PATCH_DIM,
            Some(&format!("{prefix}patch_embed.patchifier.proj.bias")),
            dtype,
        )?;
        let patch_norm = load_rmsnorm(
            weights,
            &format!("{prefix}patch_embed.patchifier.norm.weight"),
            dim,
            cfg.rms_norm_eps,
        )?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let bp = format!("{prefix}blocks.{i}.");
            let norm1 = load_rmsnorm(weights, &format!("{bp}norm1.weight"), dim, cfg.rms_norm_eps)?;
            let norm2 = load_rmsnorm(weights, &format!("{bp}norm2.weight"), dim, cfg.rms_norm_eps)?;
            let qkv = load_linear(
                weights,
                &format!("{bp}attn.qkv.weight"),
                dim * 3,
                dim,
                None,
                dtype,
            )?;
            let proj = load_linear(
                weights,
                &format!("{bp}attn.proj.weight"),
                dim,
                dim,
                None,
                dtype,
            )?;
            let fc1 = load_linear(
                weights,
                &format!("{bp}mlp.fc1.weight"),
                cfg.intermediate_size,
                dim,
                None,
                dtype,
            )?;
            let fc3 = load_linear(
                weights,
                &format!("{bp}mlp.fc3.weight"),
                cfg.intermediate_size,
                dim,
                None,
                dtype,
            )?;
            let fc2 = load_linear(
                weights,
                &format!("{bp}mlp.fc2.weight"),
                dim,
                cfg.intermediate_size,
                None,
                dtype,
            )?;
            let mlp = Mlp::new(fc1, fc3, fc2)?;
            blocks.push(VisionBlock {
                norm1,
                qkv,
                proj,
                norm2,
                mlp,
            });
        }
        let post_trunk_norm = if cfg.post_norm {
            Some(load_rmsnorm(
                weights,
                &format!("{prefix}post_trunk_norm.weight"),
                dim,
                cfg.rms_norm_eps,
            )?)
        } else {
            None
        };
        let merger = PatchMerger::from_loader(
            weights,
            &format!("{prefix}merger."),
            dim,
            cfg.hidden_size,
            cfg.spatial_merge_size,
            dtype,
        )?;
        Ok(Self {
            cfg,
            patch_proj,
            patch_norm,
            blocks,
            post_trunk_norm,
            merger,
            device: device.clone(),
            dtype,
        })
    }

    pub fn config(&self) -> &DotsVisionConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn encode(&self, prep: &PreparedImage) -> Result<Tensor> {
        let n = prep.num_patches();
        let patches = Tensor::from_slice(&prep.patches, (n, PATCH_DIM), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        self.forward_patches(&patches, prep.grid_h, prep.grid_w)
    }

    pub fn forward_patches(
        &self,
        patches: &Tensor,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let n = patches.dim(0)?;
        anyhow::ensure!(
            n == grid_h * grid_w,
            "forward_patches: {n} rows != grid {grid_h}x{grid_w}"
        );
        let head_dim = self.cfg.head_dim();
        let heads = self.cfg.num_attention_heads;
        let dim = self.cfg.embed_dim;

        let mut x = self.patch_proj.forward(patches)?;
        x = self.patch_norm.forward(&x)?.to_dtype(self.dtype)?;

        let (cos, sin) = vision_rope_tables(grid_h, grid_w, head_dim, &self.device)?;
        let attn_cfg = AttnConfig {
            num_heads: heads,
            num_kv_heads: heads,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
        };

        for blk in &self.blocks {
            let normed = blk.norm1.forward(&x)?.to_dtype(self.dtype)?;
            let qkv = blk.qkv.forward(&normed)?.reshape((n, 3, heads, head_dim))?;
            let q = qkv
                .i((.., 0))?
                .contiguous()?
                .reshape((1, n, heads, head_dim))?;
            let k = qkv
                .i((.., 1))?
                .contiguous()?
                .reshape((1, n, heads, head_dim))?;
            let v = qkv
                .i((.., 2))?
                .contiguous()?
                .reshape((1, n, heads, head_dim))?;
            let q = apply_vision_rope(&q, &cos, &sin)?;
            let k = apply_vision_rope(&k, &cos, &sin)?;
            let attn = attention(&q, &k, &v, &attn_cfg)?;
            let attn = attn.reshape((n, heads * head_dim))?;
            let attn = blk.proj.forward(&attn)?;
            x = x.add(&attn)?;

            let normed2 = blk.norm2.forward(&x)?.to_dtype(self.dtype)?;
            let mlp_out = blk.mlp.forward(&normed2)?;
            x = x.add(&mlp_out)?;
            anyhow::ensure!(x.dims() == [n, dim], "vision block shape {:?}", x.dims());
        }

        if let Some(norm) = &self.post_trunk_norm {
            x = norm.forward(&x)?.to_dtype(self.dtype)?;
        }
        let merged = self.merger.forward(&x)?;
        anyhow::ensure!(
            merged.dim(0)? == n / (MERGE_SIZE * MERGE_SIZE),
            "merger produced {} rows for {n} patches",
            merged.dim(0)?
        );
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_are_axial_and_duplicated() {
        let dev = Device::Cpu;
        let head_dim = 128usize;
        let (cos, sin) = vision_rope_tables(2, 4, head_dim, &dev).unwrap();
        assert_eq!(cos.dims(), &[1, 8, 1, head_dim]);
        let c: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        let s: Vec<f32> = sin.flatten_all().unwrap().to_vec1().unwrap();
        for row in 0..8 {
            let b = row * head_dim;
            for i in 0..head_dim / 2 {
                assert_eq!(
                    c[b + i],
                    c[b + head_dim / 2 + i],
                    "cos dup row {row} lane {i}"
                );
                assert_eq!(
                    s[b + i],
                    s[b + head_dim / 2 + i],
                    "sin dup row {row} lane {i}"
                );
            }
        }
        for i in 0..head_dim {
            assert_eq!(c[i], 1.0);
            assert_eq!(s[i], 0.0);
        }
        let quarter = head_dim / 4;
        let row3 = 3 * head_dim;
        let expect_h = (1.0f64).cos() as f32;
        assert!((c[row3] - expect_h).abs() < 1e-6);
        assert!((c[row3 + quarter] - expect_h).abs() < 1e-6);
    }

    #[test]
    fn rotate_half_swaps_and_negates() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &dev).unwrap();
        let r = rotate_half(&x).unwrap();
        let v: Vec<f32> = r.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![-3.0, -4.0, 1.0, 2.0]);
    }
}
