use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use nv_layers::attn::{attention, AttnConfig};
use nv_layers::linear::Linear;

#[derive(Clone, Debug)]
pub struct Qwen3VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub layer_norm_eps: f64,
    pub dtype: DType,
}

impl Default for Qwen3VisionConfig {
    fn default() -> Self {
        Self {
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            in_channels: 3,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            out_hidden_size: 2048,
            layer_norm_eps: 1e-6,
            dtype: DType::BF16,
        }
    }
}

impl Qwen3VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn merger_hidden(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }

    pub fn from_hf_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("vision_config must be an object"))?;
        let geti = |k: &str| -> Result<usize> {
            obj.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("vision_config: missing or non-int {k}"))
        };
        let geti_or = |k: &str, default: usize| -> usize {
            obj.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };

        Ok(Self {
            depth: geti("depth")?,
            hidden_size: geti("hidden_size")?,
            num_heads: geti("num_heads")?,
            intermediate_size: geti("intermediate_size")?,
            in_channels: geti_or("in_channels", 3),
            patch_size: geti("patch_size")?,
            temporal_patch_size: geti_or("temporal_patch_size", 2),
            spatial_merge_size: geti_or("spatial_merge_size", 2),
            num_position_embeddings: geti("num_position_embeddings")?,
            out_hidden_size: geti("out_hidden_size")?,
            layer_norm_eps: obj
                .get("layer_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6),
            dtype: DType::BF16,
        })
    }

    pub fn expected_checkpoint_tensor_names_with_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden_size;
        let inter = self.intermediate_size;
        let merged = self.merger_hidden();
        let out = self.out_hidden_size;
        let mut names: Vec<(String, Vec<usize>)> = vec![
            (
                "model.visual.patch_embed.proj.weight".into(),
                vec![
                    h,
                    self.in_channels,
                    self.temporal_patch_size,
                    self.patch_size,
                    self.patch_size,
                ],
            ),
            ("model.visual.patch_embed.proj.bias".into(), vec![h]),
            (
                "model.visual.pos_embed.weight".into(),
                vec![self.num_position_embeddings, h],
            ),
            ("model.visual.merger.norm.weight".into(), vec![h]),
            ("model.visual.merger.norm.bias".into(), vec![h]),
            (
                "model.visual.merger.linear_fc1.weight".into(),
                vec![merged, merged],
            ),
            ("model.visual.merger.linear_fc1.bias".into(), vec![merged]),
            (
                "model.visual.merger.linear_fc2.weight".into(),
                vec![out, merged],
            ),
            ("model.visual.merger.linear_fc2.bias".into(), vec![out]),
        ];
        for i in 0..self.depth {
            let p = format!("model.visual.blocks.{i}");
            names.push((format!("{p}.norm1.weight"), vec![h]));
            names.push((format!("{p}.norm1.bias"), vec![h]));
            names.push((format!("{p}.norm2.weight"), vec![h]));
            names.push((format!("{p}.norm2.bias"), vec![h]));
            names.push((format!("{p}.attn.qkv.weight"), vec![3 * h, h]));
            names.push((format!("{p}.attn.qkv.bias"), vec![3 * h]));
            names.push((format!("{p}.attn.proj.weight"), vec![h, h]));
            names.push((format!("{p}.attn.proj.bias"), vec![h]));
            names.push((format!("{p}.mlp.linear_fc1.weight"), vec![inter, h]));
            names.push((format!("{p}.mlp.linear_fc1.bias"), vec![inter]));
            names.push((format!("{p}.mlp.linear_fc2.weight"), vec![h, inter]));
            names.push((format!("{p}.mlp.linear_fc2.bias"), vec![h]));
        }
        names
    }

    pub fn from_hf_config_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = path.as_ref();
        let raw = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?;
        let vc = v
            .get("vision_config")
            .ok_or_else(|| anyhow::anyhow!("config.json: missing vision_config"))?;
        Self::from_hf_value(vc)
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new_zeros(dim: usize, dtype: DType, device: &Device, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(dim, dtype, device)?,
            bias: Tensor::zeros(dim, dtype, device)?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let mean = xf.mean_keepdim(candle_core::D::Minus1)?;
        let xc = xf.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let denom = (var + self.eps)?.sqrt()?;
        let normed = xc.broadcast_div(&denom)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let b = self.bias.to_dtype(DType::F32)?;
        let y = normed.broadcast_mul(&w)?.broadcast_add(&b)?;
        Ok(y.to_dtype(dtype)?)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionBlock {
    fn new_empty(cfg: &Qwen3VisionConfig, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let dtype = cfg.dtype;

        let norm1 = LayerNorm::new_zeros(h, dtype, device, cfg.layer_norm_eps)?;
        let norm2 = LayerNorm::new_zeros(h, dtype, device, cfg.layer_norm_eps)?;

        let qkv = Linear::new(
            Tensor::zeros((3 * h, h), dtype, device)?,
            Some(Tensor::zeros(3 * h, dtype, device)?),
        )?;
        let proj = Linear::new(
            Tensor::zeros((h, h), dtype, device)?,
            Some(Tensor::zeros(h, dtype, device)?),
        )?;
        let fc1 = Linear::new(
            Tensor::zeros((inter, h), dtype, device)?,
            Some(Tensor::zeros(inter, dtype, device)?),
        )?;
        let fc2 = Linear::new(
            Tensor::zeros((h, inter), dtype, device)?,
            Some(Tensor::zeros(h, dtype, device)?),
        )?;

        Ok(Self {
            norm1,
            qkv,
            proj,
            norm2,
            fc1,
            fc2,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, rope_cos: &Tensor, rope_sin: &Tensor) -> Result<Tensor> {
        let (b, t, h) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        let normed = self.norm1.forward(x)?;
        let qkv = self.qkv.forward(&normed)?;
        let qkv = qkv
            .reshape((b, t, 3, nh, hd))
            .map_err(|e| anyhow::anyhow!(e))?;
        let q = qkv.i((.., .., 0, .., ..))?.contiguous()?;
        let k = qkv.i((.., .., 1, .., ..))?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.contiguous()?;
        let q = apply_vision_rope_neox(&q, rope_cos, rope_sin)?;
        let k = apply_vision_rope_neox(&k, rope_cos, rope_sin)?;

        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nh,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: false,
        };
        let attn_out = attention(&q, &k, &v, &attn_cfg)?;
        let attn_out = attn_out
            .reshape((b, t, nh * hd))
            .map_err(|e| anyhow::anyhow!(e))?;
        let attn_out = self.proj.forward(&attn_out)?;
        let _ = h;
        let x = (x + attn_out).map_err(|e| anyhow::anyhow!(e))?;

        let normed = self.norm2.forward(&x)?;
        let h1 = self.fc1.forward(&normed)?;
        let h1 = gelu_pytorch_tanh(&h1)?;
        let h2 = self.fc2.forward(&h1)?;
        (x + h2).map_err(|e| anyhow::anyhow!(e))
    }
}

use candle_core::IndexOp;

const VISION_ROPE_THETA_10000_FIXED_BY_QWEN3_5VISIONROTARYEMBEDDING_DEFAULT: f32 = 10_000.0;

pub fn vision_rope_rows_raster_order_matching_hf_rot_pos_emb(
    gh: usize,
    gw: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        head_dim % 4 == 0,
        "vision rope splits head_dim {head_dim} into row/col quarters; it must divide by 4"
    );
    let half = head_dim / 2;
    let quarter = head_dim / 4;
    let theta = VISION_ROPE_THETA_10000_FIXED_BY_QWEN3_5VISIONROTARYEMBEDDING_DEFAULT;
    let inv: Vec<f32> = (0..quarter)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / half as f32))
        .collect();
    let n = gh * gw;
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    for r in 0..gh {
        for c in 0..gw {
            let base = (r * gw + c) * half;
            for j in 0..quarter {
                let fr = r as f32 * inv[j];
                let fc = c as f32 * inv[j];
                cos[base + j] = fr.cos();
                sin[base + j] = fr.sin();
                cos[base + quarter + j] = fc.cos();
                sin[base + quarter + j] = fc.sin();
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (n, 1, half), device)?,
        Tensor::from_vec(sin, (n, 1, half), device)?,
    ))
}

fn apply_vision_rope_neox(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let dims = x.dims().to_vec();
    let head_dim = *dims.last().unwrap();
    let half = head_dim / 2;
    let xf = x.to_dtype(DType::F32)?;
    let lo = xf.narrow(3, 0, half)?;
    let hi = xf.narrow(3, half, half)?;
    let out_lo = lo
        .broadcast_mul(cos)?
        .sub(&hi.broadcast_mul(sin)?)
        .map_err(|e| anyhow::anyhow!(e))?;
    let out_hi = lo
        .broadcast_mul(sin)?
        .add(&hi.broadcast_mul(cos)?)
        .map_err(|e| anyhow::anyhow!(e))?;
    let out = Tensor::cat(&[&out_lo, &out_hi], 3)?;
    Ok(out.to_dtype(dtype)?.contiguous()?)
}

struct Merger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Merger {
    fn new_empty(cfg: &Qwen3VisionConfig, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let merged = cfg.merger_hidden();
        let out = cfg.out_hidden_size;
        let dtype = cfg.dtype;

        let norm = LayerNorm::new_zeros(h, dtype, device, cfg.layer_norm_eps)?;
        let fc1 = Linear::new(
            Tensor::zeros((merged, merged), dtype, device)?,
            Some(Tensor::zeros(merged, dtype, device)?),
        )?;
        let fc2 = Linear::new(
            Tensor::zeros((out, merged), dtype, device)?,
            Some(Tensor::zeros(out, dtype, device)?),
        )?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor, spatial_merge_size: usize) -> Result<Tensor> {
        let (b, n, h) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
        let group = spatial_merge_size * spatial_merge_size;
        if n % group != 0 {
            anyhow::bail!("Merger::forward: N={n} not divisible by spatial_merge_size^2={group}");
        }
        let n_merged = n / group;
        let normed = self.norm.forward(x)?;
        let regrouped = normed
            .reshape((b, n_merged, group * h))
            .map_err(|e| anyhow::anyhow!(e))?;
        let y = self.fc1.forward(&regrouped)?;
        let y = gelu_pytorch_tanh(&y)?;
        let y = self.fc2.forward(&y)?;
        Ok(y)
    }
}

pub struct Qwen3VisionTower {
    cfg: Qwen3VisionConfig,
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    pos_embed: Tensor,
    blocks: Vec<VisionBlock>,
    merger: Merger,
    device: Device,
}

impl Qwen3VisionTower {
    pub fn new_empty(cfg: Qwen3VisionConfig, device: &Device) -> Result<Self> {
        if !cfg.hidden_size.is_multiple_of(cfg.num_heads) {
            anyhow::bail!(
                "Qwen3VisionConfig: hidden_size {} not divisible by num_heads {}",
                cfg.hidden_size,
                cfg.num_heads
            );
        }
        let p = cfg.patch_size;
        let tp = cfg.temporal_patch_size;
        let c = cfg.in_channels;
        let h = cfg.hidden_size;
        let dtype = cfg.dtype;

        let patch_embed_weight = Tensor::zeros((h, c, tp, p, p), dtype, device)?;
        let patch_embed_bias = Tensor::zeros(h, dtype, device)?;
        let pos_embed = Tensor::zeros(
            (cfg.num_position_embeddings, cfg.hidden_size),
            dtype,
            device,
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            blocks.push(VisionBlock::new_empty(&cfg, device)?);
        }
        let merger = Merger::new_empty(&cfg, device)?;

        Ok(Self {
            cfg,
            patch_embed_weight,
            patch_embed_bias,
            pos_embed,
            blocks,
            merger,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &Qwen3VisionConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn try_load(model_dir: &std::path::Path, device: &Device) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let cfg = Qwen3VisionConfig::from_hf_config_json(&config_path)?;

        let visual_path = model_dir.join("model_visual.safetensors");
        if !visual_path.is_file() {
            anyhow::bail!(
                "Qwen3VisionTower::try_load: missing {}",
                visual_path.display()
            );
        }
        let weights = nv_weights::WeightLoader::open_file(&visual_path, device)?;

        let mut tower = Self::new_empty(cfg, device)?;
        tower.load_weights(&weights)?;
        Ok(tower)
    }

    pub fn load_weights(&mut self, weights: &nv_weights::WeightLoader) -> Result<()> {
        let dtype = self.cfg.dtype;
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let c = self.cfg.in_channels;
        let p = self.cfg.patch_size;
        let tp = self.cfg.temporal_patch_size;
        let out = self.cfg.out_hidden_size;
        let merged = self.cfg.merger_hidden();
        let npos = self.cfg.num_position_embeddings;

        self.patch_embed_weight = load_5d(
            weights,
            "model.visual.patch_embed.proj.weight",
            (h, c, tp, p, p),
            dtype,
        )?;
        self.patch_embed_bias = load_1d(weights, "model.visual.patch_embed.proj.bias", h, dtype)?;
        self.pos_embed = load_2d(weights, "model.visual.pos_embed.weight", (npos, h), dtype)?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("model.visual.blocks.{i}");
            block.norm1 = LayerNorm {
                weight: load_1d(weights, &format!("{prefix}.norm1.weight"), h, dtype)?,
                bias: load_1d(weights, &format!("{prefix}.norm1.bias"), h, dtype)?,
                eps: self.cfg.layer_norm_eps,
            };
            block.norm2 = LayerNorm {
                weight: load_1d(weights, &format!("{prefix}.norm2.weight"), h, dtype)?,
                bias: load_1d(weights, &format!("{prefix}.norm2.bias"), h, dtype)?,
                eps: self.cfg.layer_norm_eps,
            };
            block.qkv = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.attn.qkv.weight"),
                    (3 * h, h),
                    dtype,
                )?,
                Some(load_1d(
                    weights,
                    &format!("{prefix}.attn.qkv.bias"),
                    3 * h,
                    dtype,
                )?),
            )?;
            block.proj = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.attn.proj.weight"),
                    (h, h),
                    dtype,
                )?,
                Some(load_1d(
                    weights,
                    &format!("{prefix}.attn.proj.bias"),
                    h,
                    dtype,
                )?),
            )?;
            block.fc1 = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.mlp.linear_fc1.weight"),
                    (inter, h),
                    dtype,
                )?,
                Some(load_1d(
                    weights,
                    &format!("{prefix}.mlp.linear_fc1.bias"),
                    inter,
                    dtype,
                )?),
            )?;
            block.fc2 = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.mlp.linear_fc2.weight"),
                    (h, inter),
                    dtype,
                )?,
                Some(load_1d(
                    weights,
                    &format!("{prefix}.mlp.linear_fc2.bias"),
                    h,
                    dtype,
                )?),
            )?;
        }

        self.merger.norm = LayerNorm {
            weight: load_1d(weights, "model.visual.merger.norm.weight", h, dtype)?,
            bias: load_1d(weights, "model.visual.merger.norm.bias", h, dtype)?,
            eps: self.cfg.layer_norm_eps,
        };
        self.merger.fc1 = Linear::new(
            load_2d(
                weights,
                "model.visual.merger.linear_fc1.weight",
                (merged, merged),
                dtype,
            )?,
            Some(load_1d(
                weights,
                "model.visual.merger.linear_fc1.bias",
                merged,
                dtype,
            )?),
        )?;
        self.merger.fc2 = Linear::new(
            load_2d(
                weights,
                "model.visual.merger.linear_fc2.weight",
                (out, merged),
                dtype,
            )?,
            Some(load_1d(
                weights,
                "model.visual.merger.linear_fc2.bias",
                out,
                dtype,
            )?),
        )?;
        Ok(())
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, c, hh, ww) = pixel_values.dims4().map_err(|e| anyhow::anyhow!(e))?;
        if c != self.cfg.in_channels {
            anyhow::bail!(
                "Qwen3VisionTower::forward: expected in_channels={}, got {}",
                self.cfg.in_channels,
                c
            );
        }
        let p = self.cfg.patch_size;
        if hh % p != 0 || ww % p != 0 {
            anyhow::bail!(
                "Qwen3VisionTower::forward: image H,W ({hh}, {ww}) not divisible by patch_size {p}"
            );
        }
        let tp = self.cfg.temporal_patch_size;

        let x = pixel_values.to_dtype(self.cfg.dtype)?;

        let stacked = if tp == 1 {
            x
        } else {
            let mut copies = Vec::with_capacity(tp);
            for _ in 0..tp {
                copies.push(x.clone());
            }

            let mut per_c = Vec::with_capacity(c);
            for ci in 0..c {
                let single = x.i((.., ci..ci + 1, .., ..))?;
                for _ in 0..tp {
                    per_c.push(single.clone());
                }
            }
            let refs: Vec<&Tensor> = per_c.iter().collect();
            Tensor::cat(&refs, 1)?
        };

        let w = self
            .patch_embed_weight
            .reshape((self.cfg.hidden_size, self.cfg.in_channels * tp, p, p))
            .map_err(|e| anyhow::anyhow!(e))?;
        let y = stacked
            .conv2d(&w, 0, p, 1, 1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let bias = self
            .patch_embed_bias
            .reshape((1, self.cfg.hidden_size, 1, 1))
            .map_err(|e| anyhow::anyhow!(e))?;
        let y = y.broadcast_add(&bias).map_err(|e| anyhow::anyhow!(e))?;

        let (_b2, hidden, gh, gw) = y.dims4().map_err(|e| anyhow::anyhow!(e))?;
        debug_assert_eq!(_b2, b);
        debug_assert_eq!(hidden, self.cfg.hidden_size);
        let num_patches = gh * gw;
        let mut x = y
            .reshape((b, hidden, num_patches))
            .map_err(|e| anyhow::anyhow!(e))?
            .transpose(1, 2)
            .map_err(|e| anyhow::anyhow!(e))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!(e))?;

        let pos = self
            .interpolated_pos_embed(gh, gw)?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!(e))?;
        x = x.broadcast_add(&pos).map_err(|e| anyhow::anyhow!(e))?;

        let (rope_cos, rope_sin) = vision_rope_rows_raster_order_matching_hf_rot_pos_emb(
            gh,
            gw,
            self.cfg.head_dim(),
            &self.device,
        )?;
        for block in &self.blocks {
            x = block.forward(&x, &rope_cos, &rope_sin)?;
        }

        let merge = self.cfg.spatial_merge_size;
        anyhow::ensure!(
            gh % merge == 0 && gw % merge == 0,
            "vision grid {gh}x{gw} not divisible by spatial_merge_size {merge}; the smart-resize \
             factor patch_size*spatial_merge_size must divide the resized image"
        );
        let order = merge_order_indices(gh, gw, merge);
        let idx = Tensor::from_vec(order, num_patches, &self.device)?;
        let x = x.index_select(&idx, 1).map_err(|e| anyhow::anyhow!(e))?;

        let merged = self.merger.forward(&x, merge)?;

        let (b2, n_merged, out_h) = merged.dims3().map_err(|e| anyhow::anyhow!(e))?;
        let flat = merged
            .reshape((b2 * n_merged, out_h))
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(flat)
    }

    fn interpolated_pos_embed(&self, gh: usize, gw: usize) -> Result<Tensor> {
        let npos = self.cfg.num_position_embeddings;
        let side = (npos as f64).sqrt().round() as usize;
        anyhow::ensure!(
            side * side == npos,
            "pos_embed grid must be square: num_position_embeddings {npos} is not a perfect square"
        );
        if gh == side && gw == side {
            return Ok(self.pos_embed.clone());
        }
        let table = self.pos_embed.to_dtype(DType::F32)?;
        let n = gh * gw;
        let ys = linspace_align(side, gh);
        let xs = linspace_align(side, gw);
        let mut i00 = vec![0u32; n];
        let mut i01 = vec![0u32; n];
        let mut i10 = vec![0u32; n];
        let mut i11 = vec![0u32; n];
        let mut w00 = vec![0f32; n];
        let mut w01 = vec![0f32; n];
        let mut w10 = vec![0f32; n];
        let mut w11 = vec![0f32; n];
        for (r, &fy) in ys.iter().enumerate() {
            let y0 = fy.floor();
            let y0i = y0 as usize;
            let y1i = (y0i + 1).min(side - 1);
            let ay = fy - y0;
            for (c, &fx) in xs.iter().enumerate() {
                let x0 = fx.floor();
                let x0i = x0 as usize;
                let x1i = (x0i + 1).min(side - 1);
                let ax = fx - x0;
                let p = r * gw + c;
                i00[p] = (y0i * side + x0i) as u32;
                i01[p] = (y0i * side + x1i) as u32;
                i10[p] = (y1i * side + x0i) as u32;
                i11[p] = (y1i * side + x1i) as u32;
                w00[p] = (1.0 - ay) * (1.0 - ax);
                w01[p] = (1.0 - ay) * ax;
                w10[p] = ay * (1.0 - ax);
                w11[p] = ay * ax;
            }
        }
        let sel = |idx: Vec<u32>, w: Vec<f32>| -> Result<Tensor> {
            let it = Tensor::from_vec(idx, n, &self.device)?;
            let rows = table.index_select(&it, 0)?;
            let wt = Tensor::from_vec(w, (n, 1), &self.device)?;
            Ok(rows.broadcast_mul(&wt)?)
        };
        let acc = sel(i00, w00)?;
        let acc = (acc + sel(i01, w01)?)?;
        let acc = (acc + sel(i10, w10)?)?;
        let acc = (acc + sel(i11, w11)?)?;
        Ok(acc.to_dtype(self.cfg.dtype)?)
    }

    pub fn build_splices(
        &self,
        token_ids: &[u32],
        image_token_id: u32,
        image_embeddings: &[Tensor],
    ) -> Result<Vec<crate::ModalitySplice>> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut i = 0usize;
        while i < token_ids.len() {
            if token_ids[i] == image_token_id {
                let start = i;
                while i < token_ids.len() && token_ids[i] == image_token_id {
                    i += 1;
                }
                runs.push((start, i - start));
            } else {
                i += 1;
            }
        }
        if runs.len() != image_embeddings.len() {
            anyhow::bail!(
                "Qwen3VisionTower::build_splices: token sequence has {} image runs but got {} image embeddings",
                runs.len(),
                image_embeddings.len()
            );
        }
        let mut out = Vec::with_capacity(runs.len());
        for ((pos, n_slots), emb) in runs.into_iter().zip(image_embeddings.iter()) {
            let d = emb.dims();
            if d.len() != 2 || d[1] != self.cfg.out_hidden_size {
                anyhow::bail!(
                    "image embedding must be [N, {}], got {:?}",
                    self.cfg.out_hidden_size,
                    d
                );
            }
            if d[0] != n_slots {
                anyhow::bail!(
                    "image embedding has {} tokens but token sequence reserves {} slots at position {}",
                    d[0],
                    n_slots,
                    pos
                );
            }
            out.push(crate::ModalitySplice {
                position: pos,
                embedding: emb.clone(),
            });
        }
        Ok(out)
    }
}

fn linspace_align(side: usize, g: usize) -> Vec<f32> {
    if g <= 1 {
        return vec![0.0];
    }
    let step = (side as f32 - 1.0) / (g as f32 - 1.0);
    (0..g).map(|i| i as f32 * step).collect()
}

fn merge_order_indices(gh: usize, gw: usize, merge: usize) -> Vec<u32> {
    let bh = gh / merge;
    let bw = gw / merge;
    let group = merge * merge;
    let mut idx = vec![0u32; gh * gw];
    for br in 0..bh {
        for bc in 0..bw {
            for dr in 0..merge {
                for dc in 0..merge {
                    let out = (br * bw + bc) * group + dr * merge + dc;
                    let src = (merge * br + dr) * gw + (merge * bc + dc);
                    idx[out] = src as u32;
                }
            }
        }
    }
    idx
}

fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let x3 = xf.powf(3.0)?;
    let inner = ((&xf + (x3 * 0.044715f64)?)? * (k as f64))?;
    let t = inner.tanh()?;
    let one_plus_t = (t + 1.0f64)?;
    let half_x = (&xf * 0.5f64)?;
    let y = half_x.mul(&one_plus_t)?;
    Ok(y.to_dtype(dtype)?)
}

fn load_1d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    dim: usize,
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("{name}: expected [{}], got {:?}", dim, d);
    }
    Ok(w)
}

fn load_2d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    shape: (usize, usize),
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != shape.0 || d[1] != shape.1 {
        anyhow::bail!("{name}: expected [{}, {}], got {:?}", shape.0, shape.1, d);
    }
    Ok(w)
}

fn load_5d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    shape: (usize, usize, usize, usize, usize),
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 5
        || d[0] != shape.0
        || d[1] != shape.1
        || d[2] != shape.2
        || d[3] != shape.3
        || d[4] != shape.4
    {
        anyhow::bail!(
            "{name}: expected [{}, {}, {}, {}, {}], got {:?}",
            shape.0,
            shape.1,
            shape.2,
            shape.3,
            shape.4,
            d
        );
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen3VisionConfig {
        Qwen3VisionConfig {
            depth: 2,
            hidden_size: 32,
            num_heads: 4,
            intermediate_size: 64,
            in_channels: 3,
            patch_size: 4,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            num_position_embeddings: 64,
            out_hidden_size: 48,
            layer_norm_eps: 1e-6,
            dtype: DType::F32,
        }
    }

    #[test]
    fn builds_on_cpu_with_defaults() {
        let cfg = tiny_cfg();
        let t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).expect("new_empty");
        assert_eq!(t.config().depth, cfg.depth);
        assert_eq!(t.blocks.len(), cfg.depth);

        let default = Qwen3VisionConfig::default();
        assert_eq!(default.hidden_size % default.num_heads, 0);
        assert_eq!(default.merger_hidden(), 1152 * 4);
        assert_eq!(default.out_hidden_size, 2048);
        assert_eq!(default.head_dim(), 72);
    }

    #[test]
    fn forward_runs_on_cpu_with_zero_weights() {
        let cfg = tiny_cfg();
        let t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).unwrap();

        let img = Tensor::zeros((1, cfg.in_channels, 16, 16), DType::F32, &Device::Cpu).unwrap();
        let out = t.forward(&img).expect("forward");
        let dims = out.dims();
        assert_eq!(dims.len(), 2);
        assert_eq!(dims[0], 4);
        assert_eq!(dims[1], cfg.out_hidden_size);
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_rejects_wrong_channels() {
        let cfg = tiny_cfg();
        let t = Qwen3VisionTower::new_empty(cfg, &Device::Cpu).unwrap();
        let bad = Tensor::zeros((1, 4, 16, 16), DType::F32, &Device::Cpu).unwrap();
        let err = t.forward(&bad).unwrap_err();
        assert!(err.to_string().contains("in_channels"));
    }

    #[test]
    fn forward_with_temporal_patch_size_2() {
        let mut cfg = tiny_cfg();
        cfg.temporal_patch_size = 2;
        let t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).unwrap();
        let img = Tensor::zeros((1, cfg.in_channels, 16, 16), DType::F32, &Device::Cpu).unwrap();
        let out = t.forward(&img).expect("forward t=2");
        let dims = out.dims();
        assert_eq!(dims, &[4, cfg.out_hidden_size]);
    }

    #[test]
    fn merge_order_indices_groups_2x2_blocks_not_1x4_strips() {
        let (gh, gw, merge) = (4, 4, 2);
        let idx = merge_order_indices(gh, gw, merge);
        assert_eq!(
            idx,
            vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
        let row_major: Vec<u32> = (0..(gh * gw) as u32).collect();
        assert_ne!(idx, row_major, "2x2 grouping must reorder the row-major strip");
    }

    #[test]
    fn interpolated_pos_embed_is_identity_on_native_grid() {
        let mut cfg = tiny_cfg();
        cfg.num_position_embeddings = 64;
        let mut t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).unwrap();
        let mut r = 0u64;
        let vals: Vec<f32> = (0..64 * cfg.hidden_size)
            .map(|_| {
                r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((r >> 33) as f32) / (1u64 << 31) as f32 - 1.0
            })
            .collect();
        t.pos_embed =
            Tensor::from_vec(vals, (64, cfg.hidden_size), &Device::Cpu).unwrap();
        let side = 8;
        let interp = t.interpolated_pos_embed(side, side).unwrap();
        let a: Vec<f32> = interp.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = t.pos_embed.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn build_splices_locates_image_token_runs() {
        let cfg = tiny_cfg();
        let t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).unwrap();
        const IMG: u32 = 248056;
        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, 9, 10, IMG, IMG, 99];
        let emb_a = Tensor::ones((3, cfg.out_hidden_size), DType::F32, &Device::Cpu).unwrap();
        let emb_b = Tensor::ones((2, cfg.out_hidden_size), DType::F32, &Device::Cpu).unwrap();
        let splices = t
            .build_splices(&tokens, IMG, &[emb_a, emb_b])
            .expect("build_splices");
        assert_eq!(splices.len(), 2);
        assert_eq!(splices[0].position, 2);
        assert_eq!(splices[0].embedding.dims(), &[3, cfg.out_hidden_size]);
        assert_eq!(splices[1].position, 7);
        assert_eq!(splices[1].embedding.dims(), &[2, cfg.out_hidden_size]);
    }

    #[test]
    fn build_splices_rejects_count_mismatch() {
        let cfg = tiny_cfg();
        let t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).unwrap();
        const IMG: u32 = 248056;
        let tokens: Vec<u32> = vec![1, IMG, IMG, 2];
        let emb = Tensor::ones((2, cfg.out_hidden_size), DType::F32, &Device::Cpu).unwrap();
        let err = t
            .build_splices(&tokens, IMG, &[emb.clone(), emb])
            .unwrap_err();
        assert!(err.to_string().contains("image runs"));
    }

    #[test]
    fn build_splices_rejects_slot_mismatch() {
        let cfg = tiny_cfg();
        let t = Qwen3VisionTower::new_empty(cfg.clone(), &Device::Cpu).unwrap();
        const IMG: u32 = 248056;
        let tokens: Vec<u32> = vec![1, IMG, IMG, IMG, 2];
        let wrong = Tensor::ones((2, cfg.out_hidden_size), DType::F32, &Device::Cpu).unwrap();
        let err = t.build_splices(&tokens, IMG, &[wrong]).unwrap_err();
        assert!(err.to_string().contains("reserves"));
    }
}
