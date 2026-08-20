use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_weights::WeightLoader;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize)]
pub struct VisionRopeParams {
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_params() -> VisionRopeParams {
    VisionRopeParams {
        rope_theta: 100.0,
        rope_type: None,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Gemma4VisionConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub default_output_length: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    pub hidden_activation: String,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default = "default_rope_params")]
    pub rope_parameters: VisionRopeParams,
    #[serde(default)]
    pub vision_soft_tokens_per_image: Option<usize>,
}

impl Gemma4VisionConfig {
    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).context("parse gemma4 vision config json")?;
        let root = v
            .as_object()
            .context("gemma4 vision config not an object")?;
        let (mut vis_obj, soft) = match root.get("vision_config") {
            Some(serde_json::Value::Object(o)) => (
                o.clone(),
                root.get("vision_soft_tokens_per_image")
                    .and_then(|x| x.as_u64()),
            ),
            Some(_) => anyhow::bail!("vision_config must be an object"),
            None => (root.clone(), None),
        };
        if let Some(n) = soft {
            vis_obj.insert(
                "vision_soft_tokens_per_image".into(),
                serde_json::Value::from(n),
            );
        }
        let cfg: Gemma4VisionConfig = serde_json::from_value(serde_json::Value::Object(vis_obj))
            .context("deserialize gemma4 vision_config")?;
        if let Some(mt) = &cfg.model_type {
            if mt != "gemma4_vision" {
                anyhow::bail!("expected model_type gemma4_vision, got {mt}");
            }
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            anyhow::bail!(
                "gemma4 vision: heads {} * head_dim {} != hidden {}",
                cfg.num_attention_heads,
                cfg.head_dim,
                cfg.hidden_size
            );
        }
        if !cfg.head_dim.is_multiple_of(4) {
            anyhow::bail!(
                "gemma4 vision: head_dim {} not divisible by 4",
                cfg.head_dim
            );
        }
        Ok(cfg)
    }

    pub fn patch_pixels(&self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    pub fn unit(&self) -> usize {
        self.patch_size * self.pooling_kernel_size
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }

    pub fn target_resolution(
        &self,
        image_width: usize,
        image_height: usize,
        max_soft_tokens: Option<usize>,
    ) -> (usize, usize) {
        let unit = self.unit() as f64;
        let patch = self.patch_size as f64;
        let pk2 = (self.pooling_kernel_size * self.pooling_kernel_size) as f64;
        let max_soft = max_soft_tokens.unwrap_or(self.default_output_length) as f64;
        let max_patches = max_soft * pk2;
        let orig = (image_height as f64 / patch) * (image_width as f64 / patch);
        let scale = (max_patches / orig).sqrt();
        let mut th = (((image_height as f64 * scale / unit).floor() * unit) as usize).max(self.unit());
        let mut tw = (((image_width as f64 * scale / unit).floor() * unit) as usize).max(self.unit());
        let unit_px = self.unit();
        let budget_px = max_patches as usize * self.patch_size * self.patch_size;
        if tw * th > budget_px {
            if th <= tw {
                tw = (budget_px / th / unit_px).max(1) * unit_px;
            } else {
                th = (budget_px / tw / unit_px).max(1) * unit_px;
            }
        }
        (tw, th)
    }

    pub fn compute_num_soft_tokens(
        &self,
        image_width: usize,
        image_height: usize,
        max_soft_tokens: Option<usize>,
    ) -> usize {
        let (tw, th) = self.target_resolution(image_width, image_height, max_soft_tokens);
        let pk2 = self.pooling_kernel_size * self.pooling_kernel_size;
        let num_patches = (th / self.patch_size) * (tw / self.patch_size);
        (num_patches / pk2).min(max_soft_tokens.unwrap_or(self.default_output_length))
    }
}

pub(crate) enum VisLinear {
    Dense(Linear),
    Int8Row { w: Tensor, scale: Tensor },
}

impl VisLinear {
    pub(crate) fn dense_weight(&self) -> Option<&Tensor> {
        match self {
            VisLinear::Dense(l) => l.weight(),
            VisLinear::Int8Row { .. } => None,
        }
    }

    fn quantize_int8_rows(w: Tensor) -> Result<Self> {
        let wf = w.to_dtype(DType::F32)?;
        let scale = (wf.abs()?.max_keepdim(1)? / 127.0)?.clamp(1e-12, 1e12)?;
        let q = wf
            .broadcast_div(&scale)?
            .round()?
            .affine(1.0, 128.0)?
            .clamp(0.0, 255.0)?
            .to_dtype(DType::U8)?;
        Ok(VisLinear::Int8Row { w: q, scale })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            VisLinear::Dense(l) => l.forward(x),
            VisLinear::Int8Row { w, scale } => {
                let wt = w
                    .to_dtype(DType::F32)?
                    .affine(1.0, -128.0)?
                    .broadcast_mul(scale)?
                    .to_dtype(x.dtype())?
                    .t()?
                    .contiguous()?;
                let dims = x.dims().to_vec();
                let leading: usize = dims[..dims.len() - 1].iter().product();
                let x2 = x.reshape((leading, *dims.last().unwrap()))?;
                let y = x2.matmul(&wt)?;
                let mut od = dims;
                *od.last_mut().unwrap() = wt.dim(1)?;
                y.reshape(od).map_err(Into::into)
            }
        }
    }
}

fn act_dtype(dtype: DType) -> DType {
    if dtype == DType::U8 {
        DType::BF16
    } else {
        dtype
    }
}

pub struct ClippedLinear {
    pub(crate) linear: VisLinear,
    input_clip: Option<(f64, f64)>,
    output_clip: Option<(f64, f64)>,
}

impl ClippedLinear {
    pub(crate) fn dense_weight(&self) -> Option<&Tensor> {
        self.linear.dense_weight()
    }

    fn plain(linear: VisLinear) -> Self {
        Self {
            linear,
            input_clip: None,
            output_clip: None,
        }
    }

    pub fn input_clip(&self) -> Option<(f64, f64)> {
        self.input_clip
    }

    pub fn output_clip(&self) -> Option<(f64, f64)> {
        self.output_clip
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = match self.input_clip {
            Some((lo, hi)) => x.clamp(lo, hi)?,
            None => x.clone(),
        };
        let y = self.linear.forward(&x)?;
        Ok(match self.output_clip {
            Some((lo, hi)) => y.clamp(lo, hi)?,
            None => y,
        })
    }
}

struct Rope2d {
    cos_x: Tensor,
    sin_x: Tensor,
    cos_y: Tensor,
    sin_y: Tensor,
    axis_dim: usize,
}

impl Rope2d {
    fn build(
        positions: &[(i64, i64)],
        head_dim: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        let axis_dim = head_dim / 2;
        let half = axis_dim / 2;
        let n = positions.len();
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| theta.powf(-((2 * j) as f32) / axis_dim as f32))
            .collect();
        let mut cos_x = vec![0f32; n * axis_dim];
        let mut sin_x = vec![0f32; n * axis_dim];
        let mut cos_y = vec![0f32; n * axis_dim];
        let mut sin_y = vec![0f32; n * axis_dim];
        for (i, &(x, y)) in positions.iter().enumerate() {
            let px = x.max(0) as f32;
            let py = y.max(0) as f32;
            for j in 0..half {
                let ax = px * inv_freq[j];
                let ay = py * inv_freq[j];
                cos_x[i * axis_dim + j] = ax.cos();
                cos_x[i * axis_dim + half + j] = ax.cos();
                sin_x[i * axis_dim + j] = ax.sin();
                sin_x[i * axis_dim + half + j] = ax.sin();
                cos_y[i * axis_dim + j] = ay.cos();
                cos_y[i * axis_dim + half + j] = ay.cos();
                sin_y[i * axis_dim + j] = ay.sin();
                sin_y[i * axis_dim + half + j] = ay.sin();
            }
        }
        Ok(Self {
            cos_x: Tensor::from_vec(cos_x, (n, 1, axis_dim), device)?,
            sin_x: Tensor::from_vec(sin_x, (n, 1, axis_dim), device)?,
            cos_y: Tensor::from_vec(cos_y, (n, 1, axis_dim), device)?,
            sin_y: Tensor::from_vec(sin_y, (n, 1, axis_dim), device)?,
            axis_dim,
        })
    }

    fn apply_axis(&self, t: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = self.axis_dim / 2;
        let t1 = t.narrow(2, 0, half)?;
        let t2 = t.narrow(2, half, half)?;
        let rot = Tensor::cat(&[&t2.neg()?, &t1], 2)?;
        let out = t
            .broadcast_mul(cos)?
            .add(&rot.broadcast_mul(sin)?)?;
        Ok(out)
    }

    fn apply(&self, t: &Tensor) -> Result<Tensor> {
        let tx = t.narrow(2, 0, self.axis_dim)?.contiguous()?;
        let ty = t.narrow(2, self.axis_dim, self.axis_dim)?.contiguous()?;
        let rx = self.apply_axis(&tx, &self.cos_x, &self.sin_x)?;
        let ry = self.apply_axis(&ty, &self.cos_y, &self.sin_y)?;
        Ok(Tensor::cat(&[&rx, &ry], 2)?)
    }
}

fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, scale: f64) -> Result<Tensor> {
    let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?.contiguous()?,
        None => scores,
    };
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(probs.matmul(v)?)
}

pub fn vision_prof_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_PROF_VISION").ok().as_deref() == Some("1"))
}

pub struct VisionWall {
    device: Device,
    last: std::time::Instant,
    pub patch_embed_ms: f64,
    pub rope_ms: f64,
    pub attn_ms: f64,
    pub mlp_ms: f64,
    pub pool_ms: f64,
    pub proj_ms: f64,
}

impl VisionWall {
    fn new(device: &Device) -> Option<Self> {
        if !vision_prof_enabled() {
            return None;
        }
        Some(Self {
            device: device.clone(),
            last: std::time::Instant::now(),
            patch_embed_ms: 0.0,
            rope_ms: 0.0,
            attn_ms: 0.0,
            mlp_ms: 0.0,
            pool_ms: 0.0,
            proj_ms: 0.0,
        })
    }

    fn lap(&mut self) -> f64 {
        let _ = self.device.synchronize();
        let ms = self.last.elapsed().as_secs_f64() * 1e3;
        self.last = std::time::Instant::now();
        ms
    }

    fn report(&self, n: usize, route: &str) {
        let total = self.patch_embed_ms
            + self.rope_ms
            + self.attn_ms
            + self.mlp_ms
            + self.pool_ms
            + self.proj_ms;
        eprintln!(
            "[vision_prof] route={route} patches={n} total={total:.2}ms patch_embed={:.2}ms \
             rope={:.2}ms attn={:.2}ms mlp={:.2}ms pool={:.2}ms proj={:.2}ms",
            self.patch_embed_ms, self.rope_ms, self.attn_ms, self.mlp_ms, self.pool_ms,
            self.proj_ms
        );
    }
}

fn lap_into(wall: &mut Option<VisionWall>, sink: fn(&mut VisionWall) -> &mut f64) {
    if let Some(w) = wall {
        let ms = w.lap();
        *sink(w) += ms;
    }
}

pub struct VisionLayer {
    pub(crate) input_layernorm: RmsNorm,
    pub(crate) post_attention_layernorm: RmsNorm,
    pub(crate) pre_feedforward_layernorm: RmsNorm,
    pub(crate) post_feedforward_layernorm: RmsNorm,
    pub(crate) q_proj: ClippedLinear,
    pub(crate) k_proj: ClippedLinear,
    pub(crate) v_proj: ClippedLinear,
    pub(crate) o_proj: ClippedLinear,
    pub(crate) q_norm: RmsNorm,
    pub(crate) k_norm: RmsNorm,
    pub(crate) gate_proj: ClippedLinear,
    pub(crate) up_proj: ClippedLinear,
    pub(crate) down_proj: ClippedLinear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionLayer {
    fn forward(
        &self,
        x: &Tensor,
        rope: &Rope2d,
        mask: Option<&Tensor>,
        wall: &mut Option<VisionWall>,
    ) -> Result<Tensor> {
        let (n, _h) = x.dims2()?;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let dtype = x.dtype();

        let normed = self.input_layernorm.forward(x)?;
        let q = self.q_proj.forward(&normed)?.reshape((n, nh, hd))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_proj.forward(&normed)?.reshape((n, nh, hd))?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_proj.forward(&normed)?.reshape((n, nh, hd))?;

        let q = rope.apply(&q.to_dtype(DType::F32)?)?;
        let k = rope.apply(&k.to_dtype(DType::F32)?)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let use_flash = mask.is_none()
            && matches!(x.device(), Device::Cuda(_))
            && matches!(dtype, DType::BF16 | DType::F16);
        let attn = if use_flash {
            let cfg = nv_layers::attn::AttnConfig {
                num_heads: nh,
                num_kv_heads: nh,
                head_dim: hd,
                softmax_scale: scale as f32,
                causal: false,
            };
            nv_layers::attn::flash_attn(
                &q.to_dtype(dtype)?.unsqueeze(0)?.contiguous()?,
                &k.to_dtype(dtype)?.unsqueeze(0)?.contiguous()?,
                &v.unsqueeze(0)?.contiguous()?,
                &cfg,
            )?
            .squeeze(0)?
        } else {
            let q = q.permute((1, 0, 2))?.contiguous()?;
            let k = k.permute((1, 0, 2))?.contiguous()?;
            let v = v.permute((1, 0, 2))?.contiguous()?.to_dtype(DType::F32)?;
            attend(&q, &k, &v, mask, scale)?.permute((1, 0, 2))?
        };
        let attn = attn.reshape((n, nh * hd))?.to_dtype(dtype)?;
        let attn = self.o_proj.forward(&attn)?;
        let attn = self.post_attention_layernorm.forward(&attn)?;
        let x = (x + attn)?;
        lap_into(wall, |w| &mut w.attn_ms);

        let normed = self.pre_feedforward_layernorm.forward(&x)?;
        let gate = self.gate_proj.forward(&normed)?.gelu()?;
        let up = self.up_proj.forward(&normed)?;
        let mlp = self.down_proj.forward(&(gate * up)?)?;
        let mlp = self.post_feedforward_layernorm.forward(&mlp)?;
        let out = (x + mlp)?;
        lap_into(wall, |w| &mut w.mlp_ms);
        Ok(out)
    }
}

pub(crate) struct PatchEmbedder {
    pub(crate) input_proj: VisLinear,
    pub(crate) table_x: Tensor,
    pub(crate) table_y: Tensor,
    position_embedding_size: usize,
}

fn position_id_tensors(
    positions: &[(i64, i64)],
    position_embedding_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let n = positions.len();
    let cap = (position_embedding_size - 1) as i64;
    let mut x_ids = vec![0u32; n];
    let mut y_ids = vec![0u32; n];
    for (i, &(x, y)) in positions.iter().enumerate() {
        x_ids[i] = x.clamp(0, cap) as u32;
        y_ids[i] = y.clamp(0, cap) as u32;
    }
    Ok((
        Tensor::from_vec(x_ids, n, device)?,
        Tensor::from_vec(y_ids, n, device)?,
    ))
}

impl PatchEmbedder {
    fn forward(
        &self,
        pixel_values: &Tensor,
        positions: &[(i64, i64)],
        valid: &[bool],
    ) -> Result<Tensor> {
        let n = positions.len();
        let device = pixel_values.device();
        let (x_ids, y_ids) =
            position_id_tensors(positions, self.position_embedding_size, device)?;
        let emb = self.forward_with_ids(pixel_values, &x_ids, &y_ids)?;
        if valid.iter().all(|&b| b) {
            return Ok(emb);
        }
        let mask: Vec<f32> = valid.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, (n, 1), device)?.to_dtype(emb.dtype())?;
        Ok(emb.broadcast_mul(&mask)?)
    }

    pub(crate) fn position_sum_for(
        &self,
        positions: &[(i64, i64)],
        device: &Device,
    ) -> Result<Tensor> {
        let (x_ids, y_ids) =
            position_id_tensors(positions, self.position_embedding_size, device)?;
        let px = self.table_x.index_select(&x_ids, 0)?;
        let py = self.table_y.index_select(&y_ids, 0)?;
        Ok(px.add(&py)?)
    }

    fn forward_with_ids(
        &self,
        pixel_values: &Tensor,
        x_ids: &Tensor,
        y_ids: &Tensor,
    ) -> Result<Tensor> {
        let proj = self.input_proj.forward(pixel_values)?;
        let pos_x = self
            .table_x
            .index_select(x_ids, 0)?
            .to_dtype(proj.dtype())?;
        let pos_y = self
            .table_y
            .index_select(y_ids, 0)?
            .to_dtype(proj.dtype())?;
        Ok(proj.add(&pos_x)?.add(&pos_y)?)
    }
}

pub struct FullGridPlan {
    grid_w: usize,
    grid_h: usize,
    x_ids: Tensor,
    y_ids: Tensor,
    rope: Rope2d,
}

impl FullGridPlan {
    pub fn grid(&self) -> (usize, usize) {
        (self.grid_w, self.grid_h)
    }

    pub fn num_patches(&self) -> usize {
        self.grid_w * self.grid_h
    }
}

pub fn full_grid_positions(grid_w: usize, grid_h: usize) -> Vec<(i64, i64)> {
    let mut positions = Vec::with_capacity(grid_w * grid_h);
    for y in 0..grid_h {
        for x in 0..grid_w {
            positions.push((x as i64, y as i64));
        }
    }
    positions
}

fn detect_row_major_full_grid(positions: &[(i64, i64)], valid: &[bool]) -> Option<(usize, usize)> {
    if positions.is_empty() || !valid.iter().all(|&b| b) {
        return None;
    }
    let grid_w = positions
        .iter()
        .position(|&(_, y)| y != 0)
        .unwrap_or(positions.len());
    if grid_w == 0 || !positions.len().is_multiple_of(grid_w) {
        return None;
    }
    let grid_h = positions.len() / grid_w;
    for (i, &(x, y)) in positions.iter().enumerate() {
        if x != (i % grid_w) as i64 || y != (i / grid_w) as i64 {
            return None;
        }
    }
    Some((grid_w, grid_h))
}

pub struct Gemma4VisionTower {
    cfg: Gemma4VisionConfig,
    pub(crate) patch_embedder: PatchEmbedder,
    pub(crate) layers: Vec<VisionLayer>,
    pub(crate) embed_pre_projection_norm: RmsNorm,
    pub(crate) embedding_projection: VisLinear,
    text_hidden_size: usize,
    device: Device,
    pub(crate) dtype: DType,
}

fn synth_tensor(
    shape: &[usize],
    seed: u64,
    scale: f32,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let count: usize = shape.iter().product();
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF1);
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) as f32 / u32::MAX as f32;
        v.push((u * 2.0 - 1.0) * scale);
    }
    Ok(Tensor::from_vec(v, shape, device)?.to_dtype(dtype)?)
}

fn synth_linear(
    out_dim: usize,
    in_dim: usize,
    seed: u64,
    dtype: DType,
    device: &Device,
) -> Result<VisLinear> {
    let w = synth_tensor(&[out_dim, in_dim], seed, 0.02, dtype, device)?;
    Ok(VisLinear::Dense(Linear::new(w, None)?))
}

fn ones_norm(dim: usize, eps: f64, dtype: DType, device: &Device) -> Result<RmsNorm> {
    Ok(RmsNorm::new(Tensor::ones(dim, act_dtype(dtype), device)?, eps))
}

fn load_linear(
    weights: &WeightLoader,
    name: &str,
    shape: (usize, usize),
    dtype: DType,
) -> Result<VisLinear> {
    let w = weights
        .get(name, act_dtype(dtype))
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != shape.0 || d[1] != shape.1 {
        anyhow::bail!("{name}: expected [{}, {}], got {:?}", shape.0, shape.1, d);
    }
    if dtype == DType::U8 {
        VisLinear::quantize_int8_rows(w)
    } else {
        Ok(VisLinear::Dense(Linear::new(w, None)?))
    }
}

fn load_scalar(weights: &WeightLoader, name: &str) -> Result<f64> {
    let t = weights
        .get(name, DType::F32)
        .with_context(|| format!("load {name}"))?;
    let v = t.reshape(1)?.to_vec1::<f32>()?[0];
    Ok(v as f64)
}

fn load_clipped(
    weights: &WeightLoader,
    base: &str,
    shape: (usize, usize),
    dtype: DType,
    use_clips: bool,
) -> Result<ClippedLinear> {
    let linear = load_linear(weights, &format!("{base}.linear.weight"), shape, dtype)?;
    if !use_clips {
        return Ok(ClippedLinear::plain(linear));
    }
    let input_clip = (
        load_scalar(weights, &format!("{base}.input_min"))?,
        load_scalar(weights, &format!("{base}.input_max"))?,
    );
    let output_clip = (
        load_scalar(weights, &format!("{base}.output_min"))?,
        load_scalar(weights, &format!("{base}.output_max"))?,
    );
    Ok(ClippedLinear {
        linear,
        input_clip: Some(input_clip),
        output_clip: Some(output_clip),
    })
}

fn load_rmsnorm(
    weights: &WeightLoader,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let w = weights
        .get(name, act_dtype(dtype))
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("{name}: expected [{}], got {:?}", dim, d);
    }
    Ok(RmsNorm::new(w, eps))
}

impl Gemma4VisionTower {
    pub fn config(&self) -> &Gemma4VisionConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn text_hidden_size(&self) -> usize {
        self.text_hidden_size
    }

    pub fn layer_clips(&self, idx: usize) -> Vec<Option<(f64, f64)>> {
        let l = &self.layers[idx];
        vec![
            l.q_proj.input_clip(),
            l.q_proj.output_clip(),
            l.down_proj.input_clip(),
            l.down_proj.output_clip(),
        ]
    }

    pub fn new_synthetic(
        cfg: Gemma4VisionConfig,
        text_hidden_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if cfg.num_key_value_heads != cfg.num_attention_heads {
            anyhow::bail!(
                "gemma4 vision tower expects MHA (kv heads {} != heads {})",
                cfg.num_key_value_heads,
                cfg.num_attention_heads
            );
        }
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let pp = cfg.patch_pixels();
        let pes = cfg.position_embedding_size;
        let eps = cfg.rms_norm_eps;

        let table = synth_tensor(&[2, pes, h], 7, 0.02, dtype, device)?;
        let patch_embedder = PatchEmbedder {
            input_proj: synth_linear(h, pp, 11, dtype, device)?,
            table_x: table.i(0)?.contiguous()?,
            table_y: table.i(1)?.contiguous()?,
            position_embedding_size: pes,
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let s = (i as u64 + 1) * 1000;
            layers.push(VisionLayer {
                input_layernorm: ones_norm(h, eps, dtype, device)?,
                post_attention_layernorm: ones_norm(h, eps, dtype, device)?,
                pre_feedforward_layernorm: ones_norm(h, eps, dtype, device)?,
                post_feedforward_layernorm: ones_norm(h, eps, dtype, device)?,
                q_proj: ClippedLinear::plain(synth_linear(h, h, s + 1, dtype, device)?),
                k_proj: ClippedLinear::plain(synth_linear(h, h, s + 2, dtype, device)?),
                v_proj: ClippedLinear::plain(synth_linear(h, h, s + 3, dtype, device)?),
                o_proj: ClippedLinear::plain(synth_linear(h, h, s + 4, dtype, device)?),
                q_norm: ones_norm(cfg.head_dim, eps, dtype, device)?,
                k_norm: ones_norm(cfg.head_dim, eps, dtype, device)?,
                gate_proj: ClippedLinear::plain(synth_linear(inter, h, s + 5, dtype, device)?),
                up_proj: ClippedLinear::plain(synth_linear(inter, h, s + 6, dtype, device)?),
                down_proj: ClippedLinear::plain(synth_linear(h, inter, s + 7, dtype, device)?),
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim,
            });
        }

        Ok(Self {
            embed_pre_projection_norm: ones_norm(h, eps, dtype, device)?,
            embedding_projection: synth_linear(text_hidden_size, h, 999, dtype, device)?,
            patch_embedder,
            layers,
            text_hidden_size,
            cfg,
            device: device.clone(),
            dtype: act_dtype(dtype),
        })
    }

    pub fn load(
        cfg: Gemma4VisionConfig,
        weights: &WeightLoader,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if cfg.num_key_value_heads != cfg.num_attention_heads {
            anyhow::bail!(
                "gemma4 vision tower expects MHA (kv heads {} != heads {})",
                cfg.num_key_value_heads,
                cfg.num_attention_heads
            );
        }
        let tower_prefix = if weights.has("model.vision_tower.patch_embedder.input_proj.weight") {
            "model.vision_tower"
        } else if weights.has("vision_tower.patch_embedder.input_proj.weight") {
            "vision_tower"
        } else {
            anyhow::bail!("no gemma4 vision tower tensors found in checkpoint")
        };
        let embed_name = if weights.has("model.embed_vision.embedding_projection.weight") {
            "model.embed_vision.embedding_projection.weight"
        } else if weights.has("embed_vision.embedding_projection.weight") {
            "embed_vision.embedding_projection.weight"
        } else {
            anyhow::bail!("no gemma4 embed_vision projection found in checkpoint")
        };

        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let pp = cfg.patch_pixels();
        let pes = cfg.position_embedding_size;
        let eps = cfg.rms_norm_eps;
        let clips = cfg.use_clipped_linears;

        let proj_shape = weights
            .shape_of(embed_name)
            .context("shape of embed_vision projection")?;
        if proj_shape.len() != 2 || proj_shape[1] != h {
            anyhow::bail!("{embed_name}: expected [text_hidden, {h}], got {proj_shape:?}");
        }
        let text_hidden_size = proj_shape[0];
        let embedding_projection = load_linear(weights, embed_name, (text_hidden_size, h), dtype)?;

        let table = weights
            .get(
                &format!("{tower_prefix}.patch_embedder.position_embedding_table"),
                act_dtype(dtype),
            )
            .context("load position_embedding_table")?;
        let td = table.dims();
        if td != [2, pes, h] {
            anyhow::bail!("position_embedding_table: expected [2, {pes}, {h}], got {td:?}");
        }
        let patch_embedder = PatchEmbedder {
            input_proj: load_linear(
                weights,
                &format!("{tower_prefix}.patch_embedder.input_proj.weight"),
                (h, pp),
                dtype,
            )?,
            table_x: table.i(0)?.contiguous()?,
            table_y: table.i(1)?.contiguous()?,
            position_embedding_size: pes,
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{tower_prefix}.encoder.layers.{i}");
            layers.push(VisionLayer {
                input_layernorm: load_rmsnorm(
                    weights,
                    &format!("{p}.input_layernorm.weight"),
                    h,
                    eps,
                    dtype,
                )?,
                post_attention_layernorm: load_rmsnorm(
                    weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    h,
                    eps,
                    dtype,
                )?,
                pre_feedforward_layernorm: load_rmsnorm(
                    weights,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    h,
                    eps,
                    dtype,
                )?,
                post_feedforward_layernorm: load_rmsnorm(
                    weights,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    h,
                    eps,
                    dtype,
                )?,
                q_proj: load_clipped(
                    weights,
                    &format!("{p}.self_attn.q_proj"),
                    (h, h),
                    dtype,
                    clips,
                )?,
                k_proj: load_clipped(
                    weights,
                    &format!("{p}.self_attn.k_proj"),
                    (h, h),
                    dtype,
                    clips,
                )?,
                v_proj: load_clipped(
                    weights,
                    &format!("{p}.self_attn.v_proj"),
                    (h, h),
                    dtype,
                    clips,
                )?,
                o_proj: load_clipped(
                    weights,
                    &format!("{p}.self_attn.o_proj"),
                    (h, h),
                    dtype,
                    clips,
                )?,
                q_norm: load_rmsnorm(
                    weights,
                    &format!("{p}.self_attn.q_norm.weight"),
                    cfg.head_dim,
                    eps,
                    dtype,
                )?,
                k_norm: load_rmsnorm(
                    weights,
                    &format!("{p}.self_attn.k_norm.weight"),
                    cfg.head_dim,
                    eps,
                    dtype,
                )?,
                gate_proj: load_clipped(
                    weights,
                    &format!("{p}.mlp.gate_proj"),
                    (inter, h),
                    dtype,
                    clips,
                )?,
                up_proj: load_clipped(
                    weights,
                    &format!("{p}.mlp.up_proj"),
                    (inter, h),
                    dtype,
                    clips,
                )?,
                down_proj: load_clipped(
                    weights,
                    &format!("{p}.mlp.down_proj"),
                    (h, inter),
                    dtype,
                    clips,
                )?,
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim,
            });
        }

        Ok(Self {
            embed_pre_projection_norm: ones_norm(h, eps, dtype, device)?,
            embedding_projection,
            patch_embedder,
            layers,
            text_hidden_size,
            cfg,
            device: device.clone(),
            dtype: act_dtype(dtype),
        })
    }

    pub fn plan_full_grid(&self, grid_w: usize, grid_h: usize) -> Result<FullGridPlan> {
        let pk = self.cfg.pooling_kernel_size;
        anyhow::ensure!(
            grid_w > 0
                && grid_h > 0
                && grid_w.is_multiple_of(pk)
                && grid_h.is_multiple_of(pk),
            "plan_full_grid: grid {grid_w}x{grid_h} not a positive multiple of pooling kernel {pk}"
        );
        let positions = full_grid_positions(grid_w, grid_h);
        let (x_ids, y_ids) = position_id_tensors(
            &positions,
            self.patch_embedder.position_embedding_size,
            &self.device,
        )?;
        let rope = Rope2d::build(
            &positions,
            self.cfg.head_dim,
            self.cfg.rope_theta(),
            &self.device,
        )?;
        Ok(FullGridPlan {
            grid_w,
            grid_h,
            x_ids,
            y_ids,
            rope,
        })
    }

    pub fn forward_full_grid(&self, pixel_values: &Tensor, plan: &FullGridPlan) -> Result<Tensor> {
        let (n, pp) = pixel_values.dims2()?;
        anyhow::ensure!(
            pp == self.cfg.patch_pixels() && n == plan.num_patches(),
            "forward_full_grid: pixel_values [{n}, {pp}] does not match plan grid {}x{} with \
             patch_pixels {}",
            plan.grid_w,
            plan.grid_h,
            self.cfg.patch_pixels()
        );
        let mut wall = VisionWall::new(&self.device);
        let pv = pixel_values.to_dtype(self.dtype)?;
        let mut x = self
            .patch_embedder
            .forward_with_ids(&pv, &plan.x_ids, &plan.y_ids)?;
        lap_into(&mut wall, |w| &mut w.patch_embed_ms);
        for layer in &self.layers {
            x = layer.forward(&x, &plan.rope, None, &mut wall)?;
        }
        let pooled = self.pool_full_grid(&x, plan.grid_w, plan.grid_h)?;
        lap_into(&mut wall, |w| &mut w.pool_ms);
        let normed = self.embed_pre_projection_norm.forward(&pooled)?;
        let out = self.embedding_projection.forward(&normed)?;
        lap_into(&mut wall, |w| &mut w.proj_ms);
        if let Some(w) = &wall {
            w.report(n, "full_grid");
        }
        Ok(out)
    }

    fn pool_full_grid(&self, hidden: &Tensor, grid_w: usize, grid_h: usize) -> Result<Tensor> {
        let pk = self.cfg.pooling_kernel_size;
        let h = hidden.dim(1)?;
        let cells = (grid_h / pk) * (grid_w / pk);
        let x = hidden
            .reshape((grid_h / pk, pk, grid_w / pk, pk, h))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((cells, pk * pk, h))?;
        Ok(x.mean(1)?)
    }

    fn pool(&self, hidden: &Tensor, positions: &[(i64, i64)], valid: &[bool]) -> Result<Tensor> {
        let pk = self.cfg.pooling_kernel_size as i64;
        let mut cells: BTreeMap<(i64, i64), Vec<u32>> = BTreeMap::new();
        for (i, &(x, y)) in positions.iter().enumerate() {
            if !valid[i] {
                continue;
            }
            cells.entry((y / pk, x / pk)).or_default().push(i as u32);
        }
        if cells.is_empty() {
            anyhow::bail!("gemma4 vision pool: no valid patches");
        }
        let mut pooled = Vec::with_capacity(cells.len());
        for rows in cells.values() {
            let ids = Tensor::from_vec(rows.clone(), rows.len(), hidden.device())?;
            let sel = hidden.index_select(&ids, 0)?;
            pooled.push(sel.mean(0)?);
        }
        let refs: Vec<&Tensor> = pooled.iter().collect();
        Ok(Tensor::stack(&refs, 0)?)
    }

    pub fn forward(&self, pixel_values: &Tensor, pixel_position_ids: &Tensor) -> Result<Tensor> {
        let (n, pp) = pixel_values.dims2()?;
        if pp != self.cfg.patch_pixels() {
            anyhow::bail!(
                "pixel_values patch dim {} != patch_pixels {}",
                pp,
                self.cfg.patch_pixels()
            );
        }
        let pd = pixel_position_ids.dims();
        if pd != [n, 2] {
            anyhow::bail!("pixel_position_ids: expected [{n}, 2], got {pd:?}");
        }
        let pos_rows = pixel_position_ids.to_dtype(DType::I64)?.to_vec2::<i64>()?;
        let positions: Vec<(i64, i64)> = pos_rows.iter().map(|r| (r[0], r[1])).collect();
        let valid: Vec<bool> = positions
            .iter()
            .map(|&(x, y)| !(x == -1 && y == -1))
            .collect();

        if let Some((gw, gh)) = detect_row_major_full_grid(&positions, &valid) {
            let pk = self.cfg.pooling_kernel_size;
            if gw.is_multiple_of(pk) && gh.is_multiple_of(pk) {
                let plan = self.plan_full_grid(gw, gh)?;
                return self.forward_full_grid(pixel_values, &plan);
            }
        }

        let mut wall = VisionWall::new(&self.device);
        let pv = pixel_values.to_dtype(self.dtype)?;
        let mut x = self.patch_embedder.forward(&pv, &positions, &valid)?;
        lap_into(&mut wall, |w| &mut w.patch_embed_ms);

        let rope = Rope2d::build(
            &positions,
            self.cfg.head_dim,
            self.cfg.rope_theta(),
            &self.device,
        )?;
        lap_into(&mut wall, |w| &mut w.rope_ms);
        let mask = if valid.iter().all(|&b| b) {
            None
        } else {
            let m: Vec<f32> = valid.iter().map(|&b| if b { 0.0 } else { -1e9 }).collect();
            Some(Tensor::from_vec(m, (1, 1, n), &self.device)?)
        };

        for layer in &self.layers {
            x = layer.forward(&x, &rope, mask.as_ref(), &mut wall)?;
        }

        let pooled = self.pool(&x, &positions, &valid)?;
        lap_into(&mut wall, |w| &mut w.pool_ms);
        let normed = self.embed_pre_projection_norm.forward(&pooled)?;
        let out = self.embedding_projection.forward(&normed)?;
        lap_into(&mut wall, |w| &mut w.proj_ms);
        if let Some(w) = &wall {
            w.report(n, "irregular");
        }
        Ok(out)
    }
}
