use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::norm::RmsNorm;
use nv_weights::WeightLoader;
use serde::Deserialize;
use std::path::Path;

pub const GEMMA4_AUDIO_MEL_BINS: usize = 128;
pub const GEMMA4_AUDIO_SAMPLE_RATE: usize = 16_000;
pub const GEMMA4_AUDIO_FRAME_LENGTH: usize = 320;
pub const GEMMA4_AUDIO_HOP_LENGTH: usize = 160;
pub const GEMMA4_AUDIO_SEQ_LENGTH: usize = 750;
pub const GEMMA4_AUDIO_MS_PER_TOKEN: usize = 40;

const CUMULATIVE_NORM_EPS: f64 = 1e-3;
const SSCP_KERNEL: usize = 3;
const SSCP_STRIDE: usize = 2;
const REL_POS_MAX_TIMESCALE: f64 = 1.0e4;

fn default_invalid_logits() -> f64 {
    -1.0e9
}

fn default_logit_cap() -> f64 {
    50.0
}

fn default_residual_weight() -> f64 {
    0.5
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

#[derive(Clone, Debug, Deserialize)]
pub struct Gemma4AudioConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "default_invalid_logits")]
    pub attention_invalid_logits_value: f64,
    #[serde(default = "default_logit_cap")]
    pub attention_logit_cap: f64,
    pub conv_kernel_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub output_proj_dims: usize,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f64,
    pub rms_norm_eps: f64,
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub use_clipped_linears: bool,
}

impl Gemma4AudioConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn past_horizon(&self) -> usize {
        self.attention_context_left.saturating_sub(1)
    }

    pub fn context_size(&self) -> usize {
        self.attention_chunk_size + self.past_horizon() + self.attention_context_right
    }

    pub fn time_stride(&self) -> usize {
        SSCP_STRIDE * SSCP_STRIDE
    }

    pub fn subsampled_freq(&self) -> usize {
        let mut f = GEMMA4_AUDIO_MEL_BINS;
        for _ in 0..2 {
            f = (f + 1 - SSCP_KERNEL) / SSCP_STRIDE + 1;
        }
        f
    }

    pub fn subsample_input_dim(&self) -> usize {
        self.subsampling_conv_channels[self.subsampling_conv_channels.len() - 1]
            * self.subsampled_freq()
    }

    pub fn subsampled_seq_len(&self, mel_frames: usize) -> usize {
        let mut t = mel_frames;
        for _ in 0..2 {
            if t == 0 {
                return 0;
            }
            t = (t + 2 - SSCP_KERNEL) / SSCP_STRIDE + 1;
        }
        t
    }
}

#[derive(Clone, Debug)]
pub struct Gemma4MmAudioSection {
    pub tower: Option<Gemma4AudioConfig>,
    pub audio_token_id: Option<u32>,
    pub boa_token_id: Option<u32>,
    pub eoa_token_id: Option<u32>,
}

impl Gemma4MmAudioSection {
    pub fn from_full_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).context("parse gemma4 mm config json")?;
        let root = v.as_object().context("gemma4 mm config not an object")?;
        let tower = match root.get("audio_config") {
            None | Some(serde_json::Value::Null) => None,
            Some(obj) => Some(
                serde_json::from_value(obj.clone()).context("deserialize gemma4 audio_config")?,
            ),
        };
        let id_of = |k: &str| root.get(k).and_then(|x| x.as_u64()).map(|x| x as u32);
        Ok(Self {
            tower,
            audio_token_id: id_of("audio_token_id"),
            boa_token_id: id_of("boa_token_id"),
            eoa_token_id: id_of("eoa_token_id"),
        })
    }

    nv_weights::hf_json_from_file!(from_full_hf_json_file, from_full_hf_json_str);
}

pub fn chunked_local_valid_mask(
    chunk_size: usize,
    context_left: usize,
    context_right: usize,
) -> Vec<Vec<bool>> {
    let past = context_left.saturating_sub(1);
    let context = chunk_size + past + context_right;
    (0..chunk_size)
        .map(|w| {
            (0..context)
                .map(|c| c >= w && c <= w + past + context_right)
                .collect()
        })
        .collect()
}

pub fn attended_key_range(q: usize, context_left: usize, context_right: usize) -> (usize, usize) {
    (
        q.saturating_sub(context_left.saturating_sub(1)),
        q + context_right,
    )
}

struct ClippedLinear {
    weight_t: Tensor,
    bias: Option<Tensor>,
    input_clip: Option<(f64, f64)>,
    output_clip: Option<(f64, f64)>,
    in_features: usize,
    out_features: usize,
}

impl ClippedLinear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims2()?;
        Ok(Self {
            weight_t: weight.t()?.contiguous()?,
            bias,
            input_clip: None,
            output_clip: None,
            in_features: dims.1,
            out_features: dims.0,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let last = *dims.last().context("clipped linear on scalar")?;
        if last != self.in_features {
            bail!(
                "clipped linear expects {} in-features, got {}",
                self.in_features,
                last
            );
        }
        let mut flat = x.reshape(((), self.in_features))?;
        if let Some((lo, hi)) = self.input_clip {
            flat = flat.clamp(lo, hi)?;
        }
        let mut y = flat.matmul(&self.weight_t)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        if let Some((lo, hi)) = self.output_clip {
            y = y.clamp(lo, hi)?;
        }
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(out_dims)?)
    }
}

fn scalar_of(loader: &WeightLoader, name: &str) -> Result<f64> {
    let t = loader.get(name, DType::F32)?;
    Ok(t.flatten_all()?.to_vec1::<f32>()?[0] as f64)
}

fn load_clipped(
    loader: &WeightLoader,
    prefix: &str,
    use_clip: bool,
    with_bias: bool,
) -> Result<ClippedLinear> {
    let wrapped = format!("{prefix}.linear.weight");
    let plain = format!("{prefix}.weight");
    let wname = if loader.has(&wrapped) { wrapped } else { plain };
    let weight = loader.get(&wname, DType::F32)?;
    let bias = if with_bias {
        Some(loader.get(&format!("{prefix}.bias"), DType::F32)?)
    } else {
        None
    };
    let mut lin = ClippedLinear::new(weight, bias)?;
    if use_clip {
        let imin = format!("{prefix}.input_min");
        let imax = format!("{prefix}.input_max");
        if loader.has(&imin) && loader.has(&imax) {
            lin.input_clip = Some((scalar_of(loader, &imin)?, scalar_of(loader, &imax)?));
        }
        let omin = format!("{prefix}.output_min");
        let omax = format!("{prefix}.output_max");
        if loader.has(&omin) && loader.has(&omax) {
            lin.output_clip = Some((scalar_of(loader, &omin)?, scalar_of(loader, &omax)?));
        }
    }
    Ok(lin)
}

fn load_norm(loader: &WeightLoader, name: &str, eps: f64) -> Result<RmsNorm> {
    Ok(RmsNorm::new(loader.get(name, DType::F32)?, eps))
}

struct CumulativeGroupNorm {
    weight: Tensor,
    eps: f64,
}

impl CumulativeGroupNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, t, f, c) = x.dims4()?;
        let group = (f * c) as f32;
        let counts: Vec<f32> = (1..=t).map(|i| i as f32 * group).collect();
        let counts = Tensor::from_vec(counts, (1, t, 1, 1), x.device())?;
        let sum_t = x.sum_keepdim(3)?.sum_keepdim(2)?;
        let mean = sum_t.cumsum(1)?.broadcast_div(&counts)?;
        let sq = x
            .broadcast_sub(&mean)?
            .sqr()?
            .sum_keepdim(3)?
            .sum_keepdim(2)?;
        let var = sq.cumsum(1)?.broadcast_div(&counts)?;
        let denom = var.affine(1.0, self.eps)?.sqrt()?;
        let y = x.broadcast_sub(&mean)?.broadcast_div(&denom)?;
        Ok(y.broadcast_mul(&self.weight.reshape((1, 1, 1, c))?)?)
    }
}

struct SscpConvBlock {
    conv_weight: Tensor,
    norm: CumulativeGroupNorm,
}

impl SscpConvBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 0, 1)?;
        let y = padded.conv2d(&self.conv_weight, 0, SSCP_STRIDE, 1, 1)?;
        let yp = y.permute((0, 2, 3, 1))?.contiguous()?;
        let yn = self.norm.forward(&yp)?;
        Ok(yn.relu()?.permute((0, 3, 1, 2))?.contiguous()?)
    }
}

struct SubsampleConvProjection {
    conv0: SscpConvBlock,
    conv1: SscpConvBlock,
    input_proj: ClippedLinear,
}

impl SubsampleConvProjection {
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.unsqueeze(1)?;
        let x = self.conv0.forward(&x)?;
        let x = self.conv1.forward(&x)?;
        let (b, c, t, f) = x.dims4()?;
        let flat = x
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b, t, f * c))?;
        self.input_proj.forward(&flat)
    }
}

pub struct Gemma4AudioAttention {
    q_proj: ClippedLinear,
    k_proj: ClippedLinear,
    v_proj: ClippedLinear,
    post: ClippedLinear,
    per_dim_scale: Tensor,
    rel_pos_emb: Tensor,
    causal_mask: Tensor,
    chunk_size: usize,
    past_horizon: usize,
    future_horizon: usize,
    context_size: usize,
    num_heads: usize,
    head_dim: usize,
    logit_cap: f64,
    invalid_logit: f64,
}

fn relative_timing_signal(past: usize, future: usize, channels: usize) -> Vec<f32> {
    let f_len = past + future + 1;
    let num_ts = channels / 2;
    let log_inc = REL_POS_MAX_TIMESCALE.ln() / (num_ts.saturating_sub(1).max(1) as f64);
    let mut out = Vec::with_capacity(f_len * channels);
    for f in 0..f_len {
        let pos = past as f64 - f as f64;
        for i in 0..num_ts {
            out.push((pos * (-(i as f64) * log_inc).exp()).sin() as f32);
        }
        for i in 0..num_ts {
            out.push((pos * (-(i as f64) * log_inc).exp()).cos() as f32);
        }
    }
    out
}

impl Gemma4AudioAttention {
    #[allow(clippy::too_many_arguments)]
    fn build(
        cfg: &Gemma4AudioConfig,
        q_proj: ClippedLinear,
        k_proj: ClippedLinear,
        v_proj: ClippedLinear,
        post: ClippedLinear,
        per_dim_scale: Tensor,
        rel_k_proj_weight: Tensor,
        device: &Device,
    ) -> Result<Self> {
        let past = cfg.past_horizon();
        let future = cfg.attention_context_right;
        let f_len = past + future + 1;
        let n = cfg.num_attention_heads;
        let h = cfg.head_dim();
        let timing = Tensor::from_vec(
            relative_timing_signal(past, future, cfg.hidden_size),
            (f_len, cfg.hidden_size),
            device,
        )?;
        let projected = timing.matmul(&rel_k_proj_weight.t()?.contiguous()?)?;
        let rel_pos_emb = projected
            .reshape((f_len, n, h))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let mask_rows = chunked_local_valid_mask(
            cfg.attention_chunk_size,
            cfg.attention_context_left,
            cfg.attention_context_right,
        );
        let context = cfg.context_size();
        let mask_data: Vec<f32> = mask_rows
            .iter()
            .flat_map(|r| r.iter().map(|&v| if v { 1.0 } else { 0.0 }))
            .collect();
        let causal_mask = Tensor::from_vec(mask_data, (cfg.attention_chunk_size, context), device)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            post,
            per_dim_scale,
            rel_pos_emb,
            causal_mask,
            chunk_size: cfg.attention_chunk_size,
            past_horizon: past,
            future_horizon: future,
            context_size: context,
            num_heads: n,
            head_dim: h,
            logit_cap: cfg.attention_logit_cap,
            invalid_logit: cfg.attention_invalid_logits_value,
        })
    }

    pub fn synthetic(cfg: &Gemma4AudioConfig, seed: &mut u64, device: &Device) -> Result<Self> {
        let d = cfg.hidden_size;
        let scale = 1.0 / (d as f32).sqrt();
        let lin = |seed: &mut u64| -> Result<ClippedLinear> {
            ClippedLinear::new(synth_tensor(&[d, d], scale, seed, device)?, None)
        };
        Self::build(
            cfg,
            lin(seed)?,
            lin(seed)?,
            lin(seed)?,
            lin(seed)?,
            synth_tensor(&[cfg.head_dim()], 0.2, seed, device)?,
            synth_tensor(&[d, d], scale, seed, device)?,
            device,
        )
    }

    fn window_slices(&self, padded: &Tensor, num_blocks: usize) -> Result<Tensor> {
        let mut wins = Vec::with_capacity(num_blocks);
        for u in 0..num_blocks {
            wins.push(
                padded
                    .narrow(1, u * self.chunk_size, self.context_size)?
                    .unsqueeze(1)?,
            );
        }
        Ok(Tensor::cat(&wins, 1)?)
    }

    pub fn forward(&self, x: &Tensor, valid: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let n = self.num_heads;
        let h = self.head_dim;
        let w = self.chunk_size;
        let c = self.context_size;
        let u = t.div_ceil(w);

        let q = self.q_proj.forward(x)?.reshape((b, t, n, h))?;
        let k = self.k_proj.forward(x)?.reshape((b, t, n, h))?;
        let v = self.v_proj.forward(x)?.reshape((b, t, n, h))?;

        let q_scale = (h as f64).powf(-0.5) / std::f64::consts::LN_2;
        let pds = self
            .per_dim_scale
            .exp()?
            .affine(1.0, 1.0)?
            .log()?
            .reshape((1, 1, 1, h))?;
        let q = q.affine(q_scale, 0.0)?.broadcast_mul(&pds)?;

        let qb = q
            .pad_with_zeros(1, 0, u * w - t)?
            .reshape((b, u, w, n, h))?
            .permute((0, 1, 3, 2, 4))?
            .contiguous()?;

        let pad_right = self.future_horizon + w - 1;
        let kp = k.pad_with_zeros(1, self.past_horizon, pad_right)?;
        let vp = v.pad_with_zeros(1, self.past_horizon, pad_right)?;
        let kb = self
            .window_slices(&kp, u)?
            .permute((0, 1, 3, 2, 4))?
            .contiguous()?;
        let vb = self
            .window_slices(&vp, u)?
            .permute((0, 1, 3, 2, 4))?
            .contiguous()?;

        let q4 = qb.reshape((b * u * n, w, h))?;
        let k4 = kb.reshape((b * u * n, c, h))?;
        let term_ac = q4.matmul(&k4.transpose(1, 2)?.contiguous()?)?;

        let f_len = self.past_horizon + self.future_horizon + 1;
        let qh = qb
            .permute((2, 0, 1, 3, 4))?
            .contiguous()?
            .reshape((n, b * u * w, h))?;
        let term_bd_unshifted = qh
            .matmul(&self.rel_pos_emb)?
            .reshape((n, b, u, w, f_len))?
            .permute((1, 2, 0, 3, 4))?
            .contiguous()?;
        let pad_amt = c + 1 - f_len;
        let term_bd = term_bd_unshifted
            .pad_with_zeros(4, 0, pad_amt)?
            .reshape((b, u, n, w * (c + 1)))?
            .narrow(3, 0, w * c)?
            .reshape((b, u, n, w, c))?;

        let logits = (term_ac.reshape((b, u, n, w, c))? + term_bd)?;
        let capped = logits
            .affine(1.0 / self.logit_cap, 0.0)?
            .tanh()?
            .affine(self.logit_cap, 0.0)?;

        let validp = valid.pad_with_zeros(1, self.past_horizon, pad_right)?;
        let vmask = self.window_slices(&validp, u)?.reshape((b, u, 1, 1, c))?;
        let combined = vmask.broadcast_mul(&self.causal_mask.reshape((1, 1, 1, w, c))?)?;
        let masked = capped
            .broadcast_mul(&combined)?
            .broadcast_add(&combined.affine(-self.invalid_logit, self.invalid_logit)?)?;

        let probs = candle_nn::ops::softmax(&masked, D::Minus1)?;
        let ctx = probs
            .reshape((b * u * n, w, c))?
            .matmul(&vb.reshape((b * u * n, c, h))?)?
            .reshape((b, u, n, w, h))?
            .permute((0, 1, 3, 2, 4))?
            .contiguous()?
            .reshape((b, u * w, n * h))?
            .narrow(1, 0, t)?;

        let _ = d;
        self.post.forward(&ctx)
    }
}

struct ConformerFeedForward {
    pre_layer_norm: RmsNorm,
    ffw_layer_1: ClippedLinear,
    ffw_layer_2: ClippedLinear,
    post_layer_norm: RmsNorm,
    residual_weight: f64,
}

impl ConformerFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.pre_layer_norm.forward(x)?;
        let y = self.ffw_layer_1.forward(&y)?.silu()?;
        let y = self.ffw_layer_2.forward(&y)?;
        let y = self.post_layer_norm.forward(&y)?;
        Ok((x + y.affine(self.residual_weight, 0.0)?)?)
    }
}

struct ConformerLightConv1d {
    pre_layer_norm: RmsNorm,
    linear_start: ClippedLinear,
    depthwise_weight: Tensor,
    conv_norm: RmsNorm,
    linear_end: ClippedLinear,
    kernel_size: usize,
}

impl ConformerLightConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _t, d) = x.dims3()?;
        let y = self.pre_layer_norm.forward(x)?;
        let y = self.linear_start.forward(&y)?;
        let a = y.narrow(2, 0, d)?;
        let g = y.narrow(2, d, d)?;
        let y = (a * candle_nn::ops::sigmoid(&g)?)?;
        let y = y
            .pad_with_zeros(1, self.kernel_size - 1, 0)?
            .transpose(1, 2)?
            .contiguous()?;
        let y = y.conv1d(&self.depthwise_weight, 0, 1, 1, d)?;
        let y = y.transpose(1, 2)?.contiguous()?;
        let y = self.conv_norm.forward(&y)?.silu()?;
        let y = self.linear_end.forward(&y)?;
        Ok((y + x)?)
    }
}

struct ConformerBlock {
    feed_forward1: ConformerFeedForward,
    norm_pre_attn: RmsNorm,
    self_attn: Gemma4AudioAttention,
    norm_post_attn: RmsNorm,
    lconv1d: ConformerLightConv1d,
    feed_forward2: ConformerFeedForward,
    norm_out: RmsNorm,
}

impl ConformerBlock {
    fn forward(&self, x: &Tensor, valid: &Tensor, valid_col: &Tensor) -> Result<Tensor> {
        let x = self.feed_forward1.forward(x)?;
        let y = self.norm_pre_attn.forward(&x)?;
        let y = self.self_attn.forward(&y, valid)?;
        let x = (x + self.norm_post_attn.forward(&y)?)?;
        let x = x.broadcast_mul(valid_col)?;
        let x = self.lconv1d.forward(&x)?;
        let x = self.feed_forward2.forward(&x)?;
        self.norm_out.forward(&x)
    }
}

pub struct Gemma4AudioEncoder {
    pub cfg: Gemma4AudioConfig,
    subsample: SubsampleConvProjection,
    layers: Vec<ConformerBlock>,
    output_proj: ClippedLinear,
    device: Device,
}

fn synth_tensor(shape: &[usize], scale: f32, seed: &mut u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((*seed >> 40) & 0xFFFF) as f32 / 65535.0;
        v.push((unit * 2.0 - 1.0) * scale);
    }
    Ok(Tensor::from_vec(v, shape, device)?)
}

fn ones(shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::ones(shape, DType::F32, device)?)
}

impl Gemma4AudioEncoder {
    pub fn from_loader(
        loader: &WeightLoader,
        cfg: &Gemma4AudioConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = if loader.has("model.audio_tower.output_proj.weight") {
            "model.audio_tower"
        } else if loader.has("audio_tower.output_proj.weight") {
            "audio_tower"
        } else {
            bail!("no gemma4 audio tower weights found in checkpoint")
        };
        Self::from_loader_with_prefix(loader, cfg, prefix, device)
    }

    pub fn from_loader_with_prefix(
        loader: &WeightLoader,
        cfg: &Gemma4AudioConfig,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        if cfg.subsampling_conv_channels.len() != 2 {
            bail!(
                "expected 2 subsampling conv channels, got {:?}",
                cfg.subsampling_conv_channels
            );
        }
        let eps = cfg.rms_norm_eps;
        let clip = cfg.use_clipped_linears;
        let sscp = |idx: usize| -> Result<SscpConvBlock> {
            let p = format!("{prefix}.subsample_conv_projection.layer{idx}");
            Ok(SscpConvBlock {
                conv_weight: loader.get(&format!("{p}.conv.weight"), DType::F32)?,
                norm: CumulativeGroupNorm {
                    weight: loader.get(&format!("{p}.norm.weight"), DType::F32)?,
                    eps: CUMULATIVE_NORM_EPS,
                },
            })
        };
        let input_proj = load_clipped(
            loader,
            &format!("{prefix}.subsample_conv_projection.input_proj_linear"),
            clip,
            false,
        )?;
        if input_proj.in_features != cfg.subsample_input_dim() {
            bail!(
                "input_proj_linear expects in-features {}, config implies {}",
                input_proj.in_features,
                cfg.subsample_input_dim()
            );
        }
        let subsample = SubsampleConvProjection {
            conv0: sscp(0)?,
            conv1: sscp(1)?,
            input_proj,
        };
        let ffw = |p: String| -> Result<ConformerFeedForward> {
            Ok(ConformerFeedForward {
                pre_layer_norm: load_norm(loader, &format!("{p}.pre_layer_norm.weight"), eps)?,
                ffw_layer_1: load_clipped(loader, &format!("{p}.ffw_layer_1"), clip, false)?,
                ffw_layer_2: load_clipped(loader, &format!("{p}.ffw_layer_2"), clip, false)?,
                post_layer_norm: load_norm(loader, &format!("{p}.post_layer_norm.weight"), eps)?,
                residual_weight: cfg.residual_weight,
            })
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}.layers.{i}");
            let ap = format!("{lp}.self_attn");
            let self_attn = Gemma4AudioAttention::build(
                cfg,
                load_clipped(loader, &format!("{ap}.q_proj"), clip, false)?,
                load_clipped(loader, &format!("{ap}.k_proj"), clip, false)?,
                load_clipped(loader, &format!("{ap}.v_proj"), clip, false)?,
                load_clipped(loader, &format!("{ap}.post"), clip, false)?,
                loader.get(&format!("{ap}.per_dim_scale"), DType::F32)?,
                loader.get(&format!("{ap}.relative_k_proj.weight"), DType::F32)?,
                device,
            )?;
            let cp = format!("{lp}.lconv1d");
            let lconv1d = ConformerLightConv1d {
                pre_layer_norm: load_norm(loader, &format!("{cp}.pre_layer_norm.weight"), eps)?,
                linear_start: load_clipped(loader, &format!("{cp}.linear_start"), clip, false)?,
                depthwise_weight: loader
                    .get(&format!("{cp}.depthwise_conv1d.weight"), DType::F32)?,
                conv_norm: load_norm(loader, &format!("{cp}.conv_norm.weight"), eps)?,
                linear_end: load_clipped(loader, &format!("{cp}.linear_end"), clip, false)?,
                kernel_size: cfg.conv_kernel_size,
            };
            layers.push(ConformerBlock {
                feed_forward1: ffw(format!("{lp}.feed_forward1"))?,
                norm_pre_attn: load_norm(loader, &format!("{lp}.norm_pre_attn.weight"), eps)?,
                self_attn,
                norm_post_attn: load_norm(loader, &format!("{lp}.norm_post_attn.weight"), eps)?,
                lconv1d,
                feed_forward2: ffw(format!("{lp}.feed_forward2"))?,
                norm_out: load_norm(loader, &format!("{lp}.norm_out.weight"), eps)?,
            });
        }
        let output_proj = load_clipped(loader, &format!("{prefix}.output_proj"), clip, true)?;
        if output_proj.out_features != cfg.output_proj_dims {
            bail!(
                "output_proj out-features {} != config output_proj_dims {}",
                output_proj.out_features,
                cfg.output_proj_dims
            );
        }
        Ok(Self {
            cfg: cfg.clone(),
            subsample,
            layers,
            output_proj,
            device: device.clone(),
        })
    }

    pub fn synthetic(cfg: &Gemma4AudioConfig, device: &Device) -> Result<Self> {
        if cfg.subsampling_conv_channels.len() != 2 {
            bail!(
                "expected 2 subsampling conv channels, got {:?}",
                cfg.subsampling_conv_channels
            );
        }
        let mut seed = 0x9e3779b97f4a7c15u64;
        let s = &mut seed;
        let d = cfg.hidden_size;
        let c0 = cfg.subsampling_conv_channels[0];
        let c1 = cfg.subsampling_conv_channels[1];
        let lin_scale = 1.0 / (d as f32).sqrt();
        let subsample = SubsampleConvProjection {
            conv0: SscpConvBlock {
                conv_weight: synth_tensor(&[c0, 1, SSCP_KERNEL, SSCP_KERNEL], 0.3, s, device)?,
                norm: CumulativeGroupNorm {
                    weight: ones(&[c0], device)?,
                    eps: CUMULATIVE_NORM_EPS,
                },
            },
            conv1: SscpConvBlock {
                conv_weight: synth_tensor(&[c1, c0, SSCP_KERNEL, SSCP_KERNEL], 0.1, s, device)?,
                norm: CumulativeGroupNorm {
                    weight: ones(&[c1], device)?,
                    eps: CUMULATIVE_NORM_EPS,
                },
            },
            input_proj: ClippedLinear::new(
                synth_tensor(&[d, cfg.subsample_input_dim()], 0.05, s, device)?,
                None,
            )?,
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let ffw = |s: &mut u64| -> Result<ConformerFeedForward> {
                Ok(ConformerFeedForward {
                    pre_layer_norm: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
                    ffw_layer_1: ClippedLinear::new(
                        synth_tensor(&[4 * d, d], lin_scale, s, device)?,
                        None,
                    )?,
                    ffw_layer_2: ClippedLinear::new(
                        synth_tensor(&[d, 4 * d], lin_scale * 0.5, s, device)?,
                        None,
                    )?,
                    post_layer_norm: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
                    residual_weight: cfg.residual_weight,
                })
            };
            layers.push(ConformerBlock {
                feed_forward1: ffw(s)?,
                norm_pre_attn: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
                self_attn: Gemma4AudioAttention::synthetic(cfg, s, device)?,
                norm_post_attn: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
                lconv1d: ConformerLightConv1d {
                    pre_layer_norm: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
                    linear_start: ClippedLinear::new(
                        synth_tensor(&[2 * d, d], lin_scale, s, device)?,
                        None,
                    )?,
                    depthwise_weight: synth_tensor(&[d, 1, cfg.conv_kernel_size], 0.2, s, device)?,
                    conv_norm: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
                    linear_end: ClippedLinear::new(
                        synth_tensor(&[d, d], lin_scale, s, device)?,
                        None,
                    )?,
                    kernel_size: cfg.conv_kernel_size,
                },
                feed_forward2: ffw(s)?,
                norm_out: RmsNorm::new(ones(&[d], device)?, cfg.rms_norm_eps),
            });
        }
        let output_proj = ClippedLinear::new(
            synth_tensor(&[cfg.output_proj_dims, d], lin_scale, s, device)?,
            Some(synth_tensor(&[cfg.output_proj_dims], 0.01, s, device)?),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            subsample,
            layers,
            output_proj,
            device: device.clone(),
        })
    }

    pub fn subsampled_valid_lens(&self, mel_frames: usize, valid_lens: &[usize]) -> Vec<usize> {
        let t_sub = self.cfg.subsampled_seq_len(mel_frames);
        let stride = self.cfg.time_stride();
        valid_lens
            .iter()
            .map(|&vl| {
                (0..t_sub)
                    .take_while(|&t| (t * stride).min(mel_frames.saturating_sub(1)) < vl)
                    .count()
            })
            .collect()
    }

    pub fn forward(&self, mel: &Tensor, valid_lens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let (b, t_mel, f) = mel.dims3()?;
        if f != GEMMA4_AUDIO_MEL_BINS {
            bail!("expected {GEMMA4_AUDIO_MEL_BINS} mel bins, got {f}");
        }
        if valid_lens.len() != b {
            bail!("valid_lens length {} != batch {}", valid_lens.len(), b);
        }
        let x = self.subsample.forward(mel)?;
        let t_sub = x.dims3()?.1;
        let sub_lens = self.subsampled_valid_lens(t_mel, valid_lens);
        let mut mask_data = Vec::with_capacity(b * t_sub);
        for &sl in &sub_lens {
            for t in 0..t_sub {
                mask_data.push(if t < sl { 1.0f32 } else { 0.0 });
            }
        }
        let valid = Tensor::from_vec(mask_data, (b, t_sub), &self.device)?;
        let valid_col = valid.reshape((b, t_sub, 1))?;
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &valid, &valid_col)?;
        }
        let x = self.output_proj.forward(&x)?;
        let x = x.broadcast_mul(&valid_col)?;
        Ok((x, sub_lens))
    }
}

pub struct Gemma4AudioEmbedder {
    projection_t: Tensor,
    eps: f64,
    in_features: usize,
    pub text_hidden: usize,
}

impl Gemma4AudioEmbedder {
    pub fn from_loader(loader: &WeightLoader, eps: f64) -> Result<Self> {
        let name = if loader.has("model.embed_audio.embedding_projection.weight") {
            "model.embed_audio.embedding_projection.weight"
        } else if loader.has("embed_audio.embedding_projection.weight") {
            "embed_audio.embedding_projection.weight"
        } else {
            bail!("no gemma4 embed_audio projection found in checkpoint")
        };
        let w = loader.get(name, DType::F32)?;
        let (out, inf) = w.dims2()?;
        Ok(Self {
            projection_t: w.t()?.contiguous()?,
            eps,
            in_features: inf,
            text_hidden: out,
        })
    }

    pub fn synthetic(
        in_features: usize,
        text_hidden: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let mut seed = 0xc0ffee123456789u64;
        let w = synth_tensor(
            &[text_hidden, in_features],
            1.0 / (in_features as f32).sqrt(),
            &mut seed,
            device,
        )?;
        Ok(Self {
            projection_t: w.t()?.contiguous()?,
            eps,
            in_features,
            text_hidden,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let last = *dims.last().context("embedder on scalar")?;
        if last != self.in_features {
            bail!(
                "embedder expects {} features, got {}",
                self.in_features,
                last
            );
        }
        let flat = x.reshape(((), self.in_features))?;
        let mean_sq = flat.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = mean_sq.affine(1.0, self.eps)?.sqrt()?;
        let normed = flat.broadcast_div(&denom)?;
        let y = normed.matmul(&self.projection_t)?;
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.text_hidden;
        Ok(y.reshape(out_dims)?)
    }
}

pub struct Gemma4AudioTower {
    pub encoder: Gemma4AudioEncoder,
    pub embedder: Gemma4AudioEmbedder,
    pub audio_token_id: Option<u32>,
}

impl Gemma4AudioTower {
    pub fn maybe_from_model_dir(dir: &Path, device: &Device) -> Result<Option<Self>> {
        let section = Gemma4MmAudioSection::from_full_hf_json_file(&dir.join("config.json"))?;
        let Some(cfg) = section.tower else {
            return Ok(None);
        };
        let loader = WeightLoader::open_dir(dir, device)?;
        let encoder = Gemma4AudioEncoder::from_loader(&loader, &cfg, device)?;
        let embedder = Gemma4AudioEmbedder::from_loader(&loader, cfg.rms_norm_eps)?;
        if embedder.in_features != cfg.output_proj_dims {
            bail!(
                "embed_audio in-features {} != output_proj_dims {}",
                embedder.in_features,
                cfg.output_proj_dims
            );
        }
        Ok(Some(Self {
            encoder,
            embedder,
            audio_token_id: section.audio_token_id,
        }))
    }
}
