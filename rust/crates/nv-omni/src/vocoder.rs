use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};

use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;

pub const NUM_CODEBOOKS: usize = 16;

pub const SAMPLE_RATE_HZ: usize = 24_000;

pub const SAMPLES_PER_FRAME: usize = 1_920;

pub const FRAME_RATE_HZ: usize = 12;

pub const UPSAMPLE_STRIDES: [usize; 4] = [8, 5, 4, 3];

pub const PRE_UPSAMPLE_STRIDES: [usize; 2] = [2, 2];

pub const PRE_TRANSFORMER_LAYERS: usize = 8;

const VQ_EPS: f64 = 1.0e-5;

const SNAKE_EPS: f64 = 1.0e-9;

pub const CHUNK_FRAMES: usize = 300;

pub const CHUNK_LEFT_CONTEXT: usize = 25;

pub const STREAM_CHUNK_FRAMES: usize = 25;

pub const STREAM_LEFT_CONTEXT_FRAMES: usize = 72;

#[derive(Clone, Debug)]
pub struct VocoderConfig {
    pub codebook_size: usize,
    pub num_codebooks: usize,

    pub codebook_dim: usize,

    pub quant_proj_dim: usize,

    pub latent_dim: usize,

    pub pre_transformer_hidden: usize,

    pub pre_transformer_layers: usize,

    pub pre_transformer_heads: usize,
    pub pre_transformer_kv_heads: usize,
    pub pre_transformer_head_dim: usize,
    pub pre_transformer_intermediate: usize,
    pub pre_transformer_rope_theta: f32,
    pub pre_transformer_rms_eps: f64,
    pub pre_transformer_sliding_window: usize,
    pub pre_transformer_max_seq_len: usize,

    pub pre_upsample_dim: usize,
    pub pre_upsample_strides: [usize; 2],

    pub decoder_dim: usize,
    pub upsample_strides: [usize; 4],

    pub res_kernel: usize,
    pub final_kernel: usize,

    pub dtype: DType,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            codebook_size: 2_048,
            num_codebooks: NUM_CODEBOOKS,
            codebook_dim: 256,
            quant_proj_dim: 512,
            latent_dim: 1024,
            pre_transformer_hidden: 512,
            pre_transformer_layers: PRE_TRANSFORMER_LAYERS,
            pre_transformer_heads: 16,
            pre_transformer_kv_heads: 16,
            pre_transformer_head_dim: 64,
            pre_transformer_intermediate: 1024,
            pre_transformer_rope_theta: 10_000.0,
            pre_transformer_rms_eps: 1.0e-5,
            pre_transformer_sliding_window: 72,
            pre_transformer_max_seq_len: 8000,
            pre_upsample_dim: 1024,
            pre_upsample_strides: PRE_UPSAMPLE_STRIDES,
            decoder_dim: 1536,
            upsample_strides: UPSAMPLE_STRIDES,
            res_kernel: 7,
            final_kernel: 7,
            dtype: DType::F32,
        }
    }
}

impl VocoderConfig {
    pub fn upsample_factor(&self) -> usize {
        let pre: usize = self.pre_upsample_strides.iter().product();
        let main: usize = self.upsample_strides.iter().product();
        pre * main
    }

    pub fn channels_after_stage(&self, stage: usize) -> usize {
        let mut c = self.decoder_dim;
        for s in 0..stage {
            let _ = s;
            c = (c / 2).max(1);
        }
        c
    }
}

struct CausalConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    dilation: usize,
    groups: usize,
    left_pad: usize,
}

impl CausalConv1d {
    fn new(weight: Tensor, bias: Option<Tensor>, dilation: usize, groups: usize) -> Result<Self> {
        let k = weight.dims()[2];
        let left_pad = (k - 1) * dilation;
        Ok(Self {
            weight,
            bias,
            dilation,
            groups,
            left_pad,
        })
    }

    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        dilation: usize,
        groups: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let w = loader
            .get(&format!("{prefix}.weight"), dtype)
            .map_err(|e| anyhow!("load {prefix}.weight: {e}"))?
            .to_device(device)?;
        let b_name = format!("{prefix}.bias");
        let b = if loader.has(&b_name) {
            Some(loader.get(&b_name, dtype)?.to_device(device)?)
        } else {
            None
        };
        Self::new(w, b, dilation, groups)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.left_pad > 0 {
            x.pad_with_zeros(D::Minus1, self.left_pad, 0)?
        } else {
            x.clone()
        };
        let mut y = x.conv1d(&self.weight, 0, 1, self.dilation, self.groups)?;
        if let Some(b) = &self.bias {
            let c = b.dims()[0];
            y = y.broadcast_add(&b.reshape((1usize, c, 1usize))?)?;
        }
        Ok(y)
    }
}

struct CausalConvT1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    right_trim: usize,
}

impl CausalConvT1d {
    fn new(weight: Tensor, bias: Option<Tensor>, stride: usize) -> Result<Self> {
        let k = weight.dims()[2];
        if k < stride {
            anyhow::bail!("CausalConvT1d: kernel {k} < stride {stride}");
        }
        Ok(Self {
            weight,
            bias,
            stride,
            right_trim: k - stride,
        })
    }

    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        stride: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let w = loader
            .get(&format!("{prefix}.weight"), dtype)
            .map_err(|e| anyhow!("load {prefix}.weight: {e}"))?
            .to_device(device)?;
        let b = if loader.has(&format!("{prefix}.bias")) {
            Some(
                loader
                    .get(&format!("{prefix}.bias"), dtype)?
                    .to_device(device)?,
            )
        } else {
            None
        };
        Self::new(w, b, stride)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        if let Some(b) = &self.bias {
            let c = b.dims()[0];
            y = y.broadcast_add(&b.reshape((1usize, c, 1usize))?)?;
        }
        if self.right_trim > 0 {
            let t = y.dims()[2];
            y = y.narrow(2, 0, t - self.right_trim)?;
        }
        Ok(y)
    }
}

struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    fn from_raw(alpha_raw: &Tensor, beta_raw: &Tensor, dtype: DType) -> Result<Self> {
        let c = alpha_raw.dims()[0];
        let af = alpha_raw.to_dtype(DType::F32)?;
        let bf = beta_raw.to_dtype(DType::F32)?;
        let alpha = af.exp()?.reshape((1usize, c, 1usize))?.to_dtype(dtype)?;
        let inv_beta = (bf.exp()? + SNAKE_EPS)?
            .recip()?
            .reshape((1usize, c, 1usize))?
            .to_dtype(dtype)?;
        Ok(Self { alpha, inv_beta })
    }

    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let alpha_raw = loader
            .get(&format!("{prefix}.alpha"), DType::F32)
            .map_err(|e| anyhow!("load {prefix}.alpha: {e}"))?
            .to_device(device)?;
        let beta_raw = loader
            .get(&format!("{prefix}.beta"), DType::F32)
            .map_err(|e| anyhow!("load {prefix}.beta: {e}"))?
            .to_device(device)?;
        Self::from_raw(&alpha_raw, &beta_raw, dtype)
    }

    fn zero(channels: usize, device: &Device, dtype: DType) -> Result<Self> {
        let alpha_raw = Tensor::zeros(channels, DType::F32, device)?;
        let beta_raw = Tensor::zeros(channels, DType::F32, device)?;
        Self::from_raw(&alpha_raw, &beta_raw, dtype)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sin = x.broadcast_mul(&self.alpha)?.sin()?;
        let bump = sin.sqr()?.broadcast_mul(&self.inv_beta)?;
        Ok(x.add(&bump)?)
    }
}

struct VqCodebook {
    weight: Tensor,
}

impl VqCodebook {
    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let emb_sum_name = format!("{prefix}._codebook.embedding_sum");
        let usage_name = format!("{prefix}._codebook.cluster_usage");
        let emb_sum = loader
            .get(&emb_sum_name, DType::F32)
            .map_err(|e| anyhow!("load {emb_sum_name}: {e}"))?;
        let usage = loader
            .get(&usage_name, DType::F32)
            .map_err(|e| anyhow!("load {usage_name}: {e}"))?;
        let usage = usage.clamp(VQ_EPS as f32, f32::INFINITY)?;
        let weight = emb_sum
            .broadcast_div(&usage.unsqueeze(1)?)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok(Self { weight })
    }

    fn zero(
        codebook_size: usize,
        codebook_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            weight: Tensor::zeros((codebook_size, codebook_dim), dtype, device)?,
        })
    }

    fn lookup(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.weight.index_select(ids, 0)?)
    }

    fn dims(&self) -> (usize, usize) {
        let d = self.weight.dims();
        (d[0], d[1])
    }
}

struct Quantizer {
    rvq_first: VqCodebook,
    rvq_rest: Vec<VqCodebook>,

    first_out_proj: CausalConv1d,
    rest_out_proj: CausalConv1d,
}

impl Quantizer {
    fn from_loader(
        loader: &WeightLoader,
        cfg: &VocoderConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let first_out_w = loader
            .get("decoder.quantizer.rvq_first.output_proj.weight", dtype)?
            .to_device(device)?;
        let rest_out_w = loader
            .get("decoder.quantizer.rvq_rest.output_proj.weight", dtype)?
            .to_device(device)?;

        let rvq_first = VqCodebook::from_loader(
            loader,
            "decoder.quantizer.rvq_first.vq.layers.0",
            device,
            dtype,
        )?;
        let rest_count = cfg.num_codebooks.saturating_sub(1);
        let mut rvq_rest = Vec::with_capacity(rest_count);
        for i in 0..rest_count {
            let prefix = format!("decoder.quantizer.rvq_rest.vq.layers.{i}");
            rvq_rest.push(VqCodebook::from_loader(loader, &prefix, device, dtype)?);
        }
        Ok(Self {
            rvq_first,
            rvq_rest,
            first_out_proj: CausalConv1d::new(first_out_w, None, 1, 1)?,
            rest_out_proj: CausalConv1d::new(rest_out_w, None, 1, 1)?,
        })
    }

    fn zero(cfg: &VocoderConfig, device: &Device, dtype: DType) -> Result<Self> {
        let mk_proj_w = |oc, ic| Tensor::zeros((oc, ic, 1usize), dtype, device);
        let rvq_first = VqCodebook::zero(cfg.codebook_size, cfg.codebook_dim, device, dtype)?;
        let rest_count = cfg.num_codebooks.saturating_sub(1);
        let mut rvq_rest = Vec::with_capacity(rest_count);
        for _ in 0..rest_count {
            rvq_rest.push(VqCodebook::zero(
                cfg.codebook_size,
                cfg.codebook_dim,
                device,
                dtype,
            )?);
        }
        let first_out = mk_proj_w(cfg.quant_proj_dim, cfg.codebook_dim)?;
        let rest_out = mk_proj_w(cfg.quant_proj_dim, cfg.codebook_dim)?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            first_out_proj: CausalConv1d::new(first_out, None, 1, 1)?,
            rest_out_proj: CausalConv1d::new(rest_out, None, 1, 1)?,
        })
    }

    fn decode(
        &self,
        tokens: &[[u32; NUM_CODEBOOKS]],
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let t = tokens.len();
        let mut first_ids: Vec<u32> = Vec::with_capacity(t);
        for f in tokens.iter() {
            first_ids.push(f[0]);
        }
        let ids_t = Tensor::from_vec(first_ids, (t,), device)?;
        let first_emb = self.rvq_first.lookup(&ids_t)?.to_dtype(dtype)?;

        let first_emb = first_emb.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let first_proj = self.first_out_proj.forward(&first_emb)?;

        let mut rest_acc = Tensor::zeros((1usize, self.rvq_rest[0].dims().1, t), dtype, device)?;
        for (k, cb) in self.rvq_rest.iter().enumerate() {
            let mut ids: Vec<u32> = Vec::with_capacity(t);
            for f in tokens.iter() {
                ids.push(f[k + 1]);
            }
            let ids_t = Tensor::from_vec(ids, (t,), device)?;
            let e = cb.lookup(&ids_t)?.to_dtype(dtype)?;
            let e = e.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            rest_acc = (rest_acc + e)?;
        }
        let rest_proj = self.rest_out_proj.forward(&rest_acc)?;

        let summed = (first_proj + rest_proj)?;
        Ok(summed)
    }
}

struct ResBlock {
    act1: Snake,
    conv1: CausalConv1d,
    act2: Snake,
    conv2: CausalConv1d,
}

impl ResBlock {
    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        dilation: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let act1 = Snake::from_loader(loader, &format!("{prefix}.act1"), device, dtype)?;
        let act2 = Snake::from_loader(loader, &format!("{prefix}.act2"), device, dtype)?;
        let conv1 = CausalConv1d::from_loader(
            loader,
            &format!("{prefix}.conv1.conv"),
            dilation,
            1,
            device,
            dtype,
        )?;
        let conv2 = CausalConv1d::from_loader(
            loader,
            &format!("{prefix}.conv2.conv"),
            1,
            1,
            device,
            dtype,
        )?;
        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        Ok((x + h)?)
    }
}

struct DecoderStage {
    pre_act: Snake,
    upsample: CausalConvT1d,
    res_blocks: Vec<ResBlock>,
}

impl DecoderStage {
    fn from_loader(
        loader: &WeightLoader,
        stage_index: usize,
        stride: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("decoder.decoder.{stage_index}.block");

        let pre_act = Snake::from_loader(loader, &format!("{prefix}.0"), device, dtype)?;
        let upsample =
            CausalConvT1d::from_loader(loader, &format!("{prefix}.1.conv"), stride, device, dtype)?;

        let dilations = [1usize, 3, 9];
        let mut res_blocks = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            let rb_prefix = format!("{prefix}.{}", i + 2);
            res_blocks.push(ResBlock::from_loader(loader, &rb_prefix, d, device, dtype)?);
        }
        Ok(Self {
            pre_act,
            upsample,
            res_blocks,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.pre_act.forward(x)?;
        h = self.upsample.forward(&h)?;
        for rb in &self.res_blocks {
            h = rb.forward(&h)?;
        }
        Ok(h)
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let weight = loader
            .get(&format!("{prefix}.weight"), dtype)?
            .to_device(device)?;
        let bias = loader
            .get(&format!("{prefix}.bias"), dtype)?
            .to_device(device)?;
        Ok(Self {
            weight,
            bias,
            eps: 1.0e-6,
        })
    }

    fn forward_channels_last(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let mean = xf.mean_keepdim(D::Minus1)?;
        let centered = xf.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_t = Tensor::new(self.eps as f32, x.device())?;
        let denom = var.broadcast_add(&eps_t)?.sqrt()?;
        let normed = centered.broadcast_div(&denom)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let b = self.bias.to_dtype(DType::F32)?;
        let out = normed.broadcast_mul(&w)?.broadcast_add(&b)?;
        Ok(out.to_dtype(x.dtype())?)
    }
}

struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1_w: Tensor,
    pwconv1_b: Tensor,
    pwconv2_w: Tensor,
    pwconv2_b: Tensor,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn from_loader(
        loader: &WeightLoader,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let dw_w = loader
            .get(&format!("{prefix}.dwconv.conv.weight"), dtype)?
            .to_device(device)?;
        let dw_b = if loader.has(&format!("{prefix}.dwconv.conv.bias")) {
            Some(
                loader
                    .get(&format!("{prefix}.dwconv.conv.bias"), dtype)?
                    .to_device(device)?,
            )
        } else {
            None
        };
        let channels = dw_w.dims()[0];
        let dwconv = CausalConv1d::new(dw_w, dw_b, 1, channels)?;
        let norm = LayerNorm::from_loader(loader, &format!("{prefix}.norm"), device, dtype)?;
        let pwconv1_w = loader
            .get(&format!("{prefix}.pwconv1.weight"), dtype)?
            .to_device(device)?;
        let pwconv1_b = loader
            .get(&format!("{prefix}.pwconv1.bias"), dtype)?
            .to_device(device)?;
        let pwconv2_w = loader
            .get(&format!("{prefix}.pwconv2.weight"), dtype)?
            .to_device(device)?;
        let pwconv2_b = loader
            .get(&format!("{prefix}.pwconv2.bias"), dtype)?
            .to_device(device)?;
        let gamma = loader
            .get(&format!("{prefix}.gamma"), dtype)?
            .to_device(device)?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1_w,
            pwconv1_b,
            pwconv2_w,
            pwconv2_b,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dwconv.forward(x)?;
        let h_cl = h.transpose(1, 2)?.contiguous()?;
        let h_cl = self.norm.forward_channels_last(&h_cl)?;

        let dims = h_cl.dims();
        let (b, t, c) = (dims[0], dims[1], dims[2]);
        let flat = h_cl.reshape((b * t, c))?;
        let w1 = self.pwconv1_w.t()?.contiguous()?;
        let mid = flat.matmul(&w1)?.broadcast_add(&self.pwconv1_b)?;
        let mid = mid.gelu_erf()?;
        let w2 = self.pwconv2_w.t()?.contiguous()?;
        let out_flat = mid.matmul(&w2)?.broadcast_add(&self.pwconv2_b)?;
        let out_cl = out_flat.reshape((b, t, c))?;
        let scaled = out_cl.broadcast_mul(&self.gamma)?;
        let scaled_cf = scaled.transpose(1, 2)?.contiguous()?;
        Ok((x + scaled_cf)?)
    }
}

struct PreTransformerLayer {
    input_layernorm: RmsNorm,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    attn_layer_scale: Tensor,
    post_attention_layernorm: RmsNorm,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    mlp_layer_scale: Tensor,
}

impl PreTransformerLayer {
    fn from_loader(
        loader: &WeightLoader,
        layer_idx: usize,
        eps: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let p = format!("decoder.pre_transformer.layers.{layer_idx}");
        let ln_w = loader
            .get(&format!("{p}.input_layernorm.weight"), dtype)?
            .to_device(device)?;
        let q_w = loader
            .get(&format!("{p}.self_attn.q_proj.weight"), dtype)?
            .to_device(device)?;
        let k_w = loader
            .get(&format!("{p}.self_attn.k_proj.weight"), dtype)?
            .to_device(device)?;
        let v_w = loader
            .get(&format!("{p}.self_attn.v_proj.weight"), dtype)?
            .to_device(device)?;
        let o_w = loader
            .get(&format!("{p}.self_attn.o_proj.weight"), dtype)?
            .to_device(device)?;
        let attn_scale = loader
            .get(&format!("{p}.self_attn_layer_scale.scale"), dtype)?
            .to_device(device)?;
        let post_ln_w = loader
            .get(&format!("{p}.post_attention_layernorm.weight"), dtype)?
            .to_device(device)?;
        let gate_w = loader
            .get(&format!("{p}.mlp.gate_proj.weight"), dtype)?
            .to_device(device)?;
        let up_w = loader
            .get(&format!("{p}.mlp.up_proj.weight"), dtype)?
            .to_device(device)?;
        let down_w = loader
            .get(&format!("{p}.mlp.down_proj.weight"), dtype)?
            .to_device(device)?;
        let mlp_scale = loader
            .get(&format!("{p}.mlp_layer_scale.scale"), dtype)?
            .to_device(device)?;
        Ok(Self {
            input_layernorm: RmsNorm::new(ln_w, eps),
            q_proj: q_w,
            k_proj: k_w,
            v_proj: v_w,
            o_proj: o_w,
            attn_layer_scale: attn_scale,
            post_attention_layernorm: RmsNorm::new(post_ln_w, eps),
            gate_proj: gate_w,
            up_proj: up_w,
            down_proj: down_w,
            mlp_layer_scale: mlp_scale,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &Rope,
        positions: &Tensor,
        attn_mask: Option<&Tensor>,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;

        let dims = normed.dims();
        let (b, t, _hidden) = (dims[0], dims[1], dims[2]);
        let flat = normed.reshape((b * t, dims[2]))?;
        let q = flat
            .matmul(&self.q_proj.t()?.contiguous()?)?
            .reshape((b, t, num_heads, head_dim))?;
        let k = flat
            .matmul(&self.k_proj.t()?.contiguous()?)?
            .reshape((b, t, num_heads, head_dim))?;
        let v = flat
            .matmul(&self.v_proj.t()?.contiguous()?)?
            .reshape((b, t, num_heads, head_dim))?;

        let (q_rot, k_rot) = rope.apply(
            &q.reshape((b * t, num_heads, head_dim))?,
            &k.reshape((b * t, num_heads, head_dim))?,
            positions,
        )?;
        let q_rot = q_rot.reshape((b, t, num_heads, head_dim))?;
        let k_rot = k_rot.reshape((b, t, num_heads, head_dim))?;

        let attn_out = sdpa_masked(&q_rot, &k_rot, &v, attn_mask)?;

        let attn_flat = attn_out.reshape((b * t, num_heads * head_dim))?;
        let proj = attn_flat
            .matmul(&self.o_proj.t()?.contiguous()?)?
            .reshape((b, t, dims[2]))?;
        let scaled_attn = proj.broadcast_mul(&self.attn_layer_scale)?;
        let h = (x + scaled_attn)?;

        let normed2 = self.post_attention_layernorm.forward(&h)?;
        let flat2 = normed2.reshape((b * t, dims[2]))?;
        let gate = flat2.matmul(&self.gate_proj.t()?.contiguous()?)?;
        let up = flat2.matmul(&self.up_proj.t()?.contiguous()?)?;
        let activated = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let down = activated
            .matmul(&self.down_proj.t()?.contiguous()?)?
            .reshape((b, t, dims[2]))?;
        let scaled_mlp = down.broadcast_mul(&self.mlp_layer_scale)?;
        Ok((&h + scaled_mlp)?)
    }
}

struct PreTransformer {
    layers: Vec<PreTransformerLayer>,
    rope: Rope,
    num_heads: usize,
    head_dim: usize,
    sliding_window: usize,
}

impl PreTransformer {
    fn from_loader(
        loader: &WeightLoader,
        cfg: &VocoderConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.pre_transformer_layers);
        for i in 0..cfg.pre_transformer_layers {
            layers.push(PreTransformerLayer::from_loader(
                loader,
                i,
                cfg.pre_transformer_rms_eps,
                device,
                dtype,
            )?);
        }
        let rope_cfg = RopeConfig {
            head_dim: cfg.pre_transformer_head_dim,
            max_seq_len: cfg.pre_transformer_max_seq_len,
            base: cfg.pre_transformer_rope_theta,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(rope_cfg, device)?;
        Ok(Self {
            layers,
            rope,
            num_heads: cfg.pre_transformer_heads,
            head_dim: cfg.pre_transformer_head_dim,
            sliding_window: cfg.pre_transformer_sliding_window,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dims()[1];
        let positions_vec: Vec<u32> = (0..t as u32).collect();
        let positions = Tensor::from_vec(positions_vec, (t,), x.device())?;

        let mask = if t > 1 {
            Some(build_sliding_causal_mask(
                t,
                self.sliding_window,
                x.device(),
            )?)
        } else {
            None
        };

        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(
                &h,
                &self.rope,
                &positions,
                mask.as_ref(),
                self.num_heads,
                self.head_dim,
            )?;
        }
        Ok(h)
    }
}

fn sdpa_masked(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let qd = q.dims();
    let kd = k.dims();
    let (b, sq, h, d) = (qd[0], qd[1], qd[2], qd[3]);
    let sk = kd[1];
    let orig_dtype = q.dtype();

    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    let q_t = q.permute((0, 2, 1, 3))?.contiguous()?;
    let k_t = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v_t = v.permute((0, 2, 1, 3))?.contiguous()?;

    let q_flat = q_t.reshape((b * h, sq, d))?;
    let k_flat = k_t.reshape((b * h, sk, d))?;
    let v_flat = v_t.reshape((b * h, sk, d))?;

    let k_perm = k_flat.permute((0, 2, 1))?.contiguous()?;
    let scale = (d as f32).sqrt().recip();
    let scale_t = Tensor::new(scale, q_flat.device())?;
    let mut scores = q_flat.matmul(&k_perm)?.broadcast_mul(&scale_t)?;

    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v_flat)?;
    let out = out
        .reshape((b, h, sq, d))?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    Ok(out.to_dtype(orig_dtype)?)
}

fn build_sliding_causal_mask(seq_len: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let visible = j <= i && i - j < window;
            if !visible {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let t = Tensor::from_vec(mask, (1, seq_len, seq_len), device)?;
    Ok(t)
}

struct UpsampleStage {
    upsample: CausalConvT1d,
    block: ConvNeXtBlock,
}

impl UpsampleStage {
    fn from_loader(
        loader: &WeightLoader,
        stage_index: usize,
        stride: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let upsample = CausalConvT1d::from_loader(
            loader,
            &format!("decoder.upsample.{stage_index}.0.conv"),
            stride,
            device,
            dtype,
        )?;
        let block = ConvNeXtBlock::from_loader(
            loader,
            &format!("decoder.upsample.{stage_index}.1"),
            device,
            dtype,
        )?;
        Ok(Self { upsample, block })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.upsample.forward(x)?;
        self.block.forward(&h)
    }
}

pub struct Vocoder {
    cfg: VocoderConfig,
    device: Device,
    dtype: DType,

    quantizer: Quantizer,
    pre_conv: CausalConv1d,

    pre_transformer_in: Option<(Tensor, Tensor)>,
    pre_transformer_out: Option<(Tensor, Tensor)>,
    pre_transformer_norm: Option<Tensor>,
    pre_transformer: Option<PreTransformer>,

    upsample_stages: Vec<UpsampleStage>,
    init_conv: CausalConv1d,
    decoder_stages: Vec<DecoderStage>,
    final_act: Snake,
    final_conv: CausalConv1d,

    is_zero_init: bool,
}

impl Vocoder {
    pub fn new(cfg: VocoderConfig, device: &Device) -> Result<Self> {
        if cfg.codebook_size == 0 {
            anyhow::bail!("VocoderConfig: codebook_size must be > 0");
        }
        if cfg.num_codebooks == 0 {
            anyhow::bail!("VocoderConfig: num_codebooks must be > 0");
        }
        let factor = cfg.upsample_factor();
        if factor != SAMPLES_PER_FRAME {
            anyhow::bail!(
                "VocoderConfig: upsample_factor {factor} != SAMPLES_PER_FRAME = {}",
                SAMPLES_PER_FRAME
            );
        }
        if cfg.final_kernel == 0 || cfg.final_kernel.is_multiple_of(2) {
            anyhow::bail!(
                "VocoderConfig: final_kernel must be odd > 0 (got {})",
                cfg.final_kernel
            );
        }

        let dtype = cfg.dtype;
        let quantizer = Quantizer::zero(&cfg, device, dtype)?;

        let pre_conv_w =
            Tensor::zeros((cfg.latent_dim, cfg.quant_proj_dim, 3usize), dtype, device)?;
        let pre_conv_b = Tensor::zeros(cfg.latent_dim, dtype, device)?;
        let pre_conv = CausalConv1d::new(pre_conv_w, Some(pre_conv_b), 1, 1)?;

        let mut upsample_stages = Vec::with_capacity(cfg.pre_upsample_strides.len());
        let in_c = cfg.pre_upsample_dim;
        for &stride in cfg.pre_upsample_strides.iter() {
            let up_w = Tensor::zeros((in_c, in_c, stride), dtype, device)?;
            let up_b = Tensor::zeros(in_c, dtype, device)?;
            let upsample = CausalConvT1d::new(up_w, Some(up_b), stride)?;
            let dw_w = Tensor::zeros((in_c, 1usize, 7usize), dtype, device)?;
            let dw_b = Tensor::zeros(in_c, dtype, device)?;
            let dwconv = CausalConv1d::new(dw_w, Some(dw_b), 1, in_c)?;
            let norm = LayerNorm {
                weight: Tensor::zeros(in_c, dtype, device)?,
                bias: Tensor::zeros(in_c, dtype, device)?,
                eps: 1.0e-6,
            };
            let pwconv1_w = Tensor::zeros((4 * in_c, in_c), dtype, device)?;
            let pwconv1_b = Tensor::zeros(4 * in_c, dtype, device)?;
            let pwconv2_w = Tensor::zeros((in_c, 4 * in_c), dtype, device)?;
            let pwconv2_b = Tensor::zeros(in_c, dtype, device)?;
            let gamma = Tensor::zeros(in_c, dtype, device)?;
            let block = ConvNeXtBlock {
                dwconv,
                norm,
                pwconv1_w,
                pwconv1_b,
                pwconv2_w,
                pwconv2_b,
                gamma,
            };
            upsample_stages.push(UpsampleStage { upsample, block });
        }

        let init_w = Tensor::zeros(
            (cfg.decoder_dim, cfg.pre_upsample_dim, cfg.final_kernel),
            dtype,
            device,
        )?;
        let init_b = Tensor::zeros(cfg.decoder_dim, dtype, device)?;
        let init_conv = CausalConv1d::new(init_w, Some(init_b), 1, 1)?;

        let mut decoder_stages = Vec::with_capacity(cfg.upsample_strides.len());
        let mut cur_c = cfg.decoder_dim;
        for &stride in cfg.upsample_strides.iter() {
            let next_c = (cur_c / 2).max(1);
            let pre_act = Snake::zero(cur_c, device, dtype)?;
            let up_kernel = 2 * stride;
            let up_w = Tensor::zeros((cur_c, next_c, up_kernel), dtype, device)?;
            let up_b = Tensor::zeros(next_c, dtype, device)?;
            let upsample = CausalConvT1d::new(up_w, Some(up_b), stride)?;
            let dilations = [1usize, 3, 9];
            let mut res_blocks = Vec::with_capacity(dilations.len());
            for &d in dilations.iter() {
                let act1 = Snake::zero(next_c, device, dtype)?;
                let act2 = Snake::zero(next_c, device, dtype)?;
                let c1_w = Tensor::zeros((next_c, next_c, cfg.res_kernel), dtype, device)?;
                let c1_b = Tensor::zeros(next_c, dtype, device)?;
                let conv1 = CausalConv1d::new(c1_w, Some(c1_b), d, 1)?;
                let c2_w = Tensor::zeros((next_c, next_c, 1usize), dtype, device)?;
                let c2_b = Tensor::zeros(next_c, dtype, device)?;
                let conv2 = CausalConv1d::new(c2_w, Some(c2_b), 1, 1)?;
                res_blocks.push(ResBlock {
                    act1,
                    conv1,
                    act2,
                    conv2,
                });
            }
            decoder_stages.push(DecoderStage {
                pre_act,
                upsample,
                res_blocks,
            });
            cur_c = next_c;
        }
        let last_c = cur_c;
        let final_act = Snake::zero(last_c, device, dtype)?;
        let final_w = Tensor::zeros((1usize, last_c, cfg.final_kernel), dtype, device)?;
        let final_b = Tensor::zeros(1usize, dtype, device)?;
        let final_conv = CausalConv1d::new(final_w, Some(final_b), 1, 1)?;

        Ok(Self {
            cfg,
            device: device.clone(),
            dtype,
            quantizer,
            pre_conv,
            pre_transformer_in: None,
            pre_transformer_out: None,
            pre_transformer_norm: None,
            pre_transformer: None,
            upsample_stages,
            init_conv,
            decoder_stages,
            final_act,
            final_conv,
            is_zero_init: true,
        })
    }

    pub fn from_qwen3_weights(
        loader: &WeightLoader,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let cfg = VocoderConfig {
            dtype,
            ..VocoderConfig::default()
        };
        Self::from_qwen3_weights_with_cfg(loader, &cfg, device)
    }

    pub fn from_qwen3_weights_with_cfg(
        loader: &WeightLoader,
        cfg: &VocoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let dtype = cfg.dtype;
        let quantizer = Quantizer::from_loader(loader, cfg, device, dtype)?;
        let pre_conv =
            CausalConv1d::from_loader(loader, "decoder.pre_conv.conv", 1, 1, device, dtype)?;

        let (pti_w, pti_b) = (
            loader
                .get("decoder.pre_transformer.input_proj.weight", dtype)?
                .to_device(device)?,
            loader
                .get("decoder.pre_transformer.input_proj.bias", dtype)?
                .to_device(device)?,
        );
        let (pto_w, pto_b) = (
            loader
                .get("decoder.pre_transformer.output_proj.weight", dtype)?
                .to_device(device)?,
            loader
                .get("decoder.pre_transformer.output_proj.bias", dtype)?
                .to_device(device)?,
        );
        let ptn = loader
            .get("decoder.pre_transformer.norm.weight", dtype)?
            .to_device(device)?;

        let pre_transformer = PreTransformer::from_loader(loader, cfg, device, dtype)?;

        let mut upsample_stages = Vec::with_capacity(cfg.pre_upsample_strides.len());
        for (i, &stride) in cfg.pre_upsample_strides.iter().enumerate() {
            upsample_stages.push(UpsampleStage::from_loader(
                loader, i, stride, device, dtype,
            )?);
        }

        let init_conv =
            CausalConv1d::from_loader(loader, "decoder.decoder.0.conv", 1, 1, device, dtype)?;

        let mut decoder_stages = Vec::with_capacity(cfg.upsample_strides.len());
        for (i, &stride) in cfg.upsample_strides.iter().enumerate() {
            decoder_stages.push(DecoderStage::from_loader(
                loader,
                i + 1,
                stride,
                device,
                dtype,
            )?);
        }

        let n_stages = cfg.upsample_strides.len();
        let final_act = Snake::from_loader(
            loader,
            &format!("decoder.decoder.{}", n_stages + 1),
            device,
            dtype,
        )?;
        let final_conv = CausalConv1d::from_loader(
            loader,
            &format!("decoder.decoder.{}.conv", n_stages + 2),
            1,
            1,
            device,
            dtype,
        )?;

        Ok(Self {
            cfg: cfg.clone(),
            device: device.clone(),
            dtype,
            quantizer,
            pre_conv,
            pre_transformer_in: Some((pti_w, pti_b)),
            pre_transformer_out: Some((pto_w, pto_b)),
            pre_transformer_norm: Some(ptn),
            pre_transformer: Some(pre_transformer),
            upsample_stages,
            init_conv,
            decoder_stages,
            final_act,
            final_conv,
            is_zero_init: false,
        })
    }

    pub fn config(&self) -> &VocoderConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_zero_init(&self) -> bool {
        self.is_zero_init
    }

    pub fn decode(&self, speech_tokens: &[[u32; NUM_CODEBOOKS]]) -> Result<Vec<f32>> {
        if speech_tokens.is_empty() {
            return Ok(Vec::new());
        }
        if self.cfg.num_codebooks != NUM_CODEBOOKS {
            anyhow::bail!(
                "Vocoder::decode: cfg.num_codebooks {} != const NUM_CODEBOOKS {}",
                self.cfg.num_codebooks,
                NUM_CODEBOOKS
            );
        }
        if speech_tokens.len() < 2 {
            anyhow::bail!(
                "Vocoder::decode: need at least 2 frames (got {})",
                speech_tokens.len()
            );
        }
        for (i, frame) in speech_tokens.iter().enumerate() {
            for (k, &tok) in frame.iter().enumerate() {
                if (tok as usize) >= self.cfg.codebook_size {
                    anyhow::bail!(
                        "Vocoder::decode: speech_tokens[{i}][{k}] = {tok} >= codebook_size {}",
                        self.cfg.codebook_size
                    );
                }
            }
        }

        let spf = self.cfg.upsample_factor();
        let mut out = Vec::with_capacity(speech_tokens.len() * spf);
        let mut streamer = self.streamer(CHUNK_FRAMES, CHUNK_LEFT_CONTEXT)?;
        for frame in speech_tokens {
            if let Some(pcm) = streamer.push(*frame)? {
                out.extend_from_slice(&pcm);
            }
        }
        out.extend_from_slice(&streamer.finish()?);
        Ok(out)
    }

    pub fn streamer(
        &self,
        chunk_frames: usize,
        left_context: usize,
    ) -> Result<VocoderStreamer<'_>> {
        if self.cfg.num_codebooks != NUM_CODEBOOKS {
            anyhow::bail!(
                "Vocoder::streamer: cfg.num_codebooks {} != const NUM_CODEBOOKS {}",
                self.cfg.num_codebooks,
                NUM_CODEBOOKS
            );
        }
        if chunk_frames < 2 {
            anyhow::bail!("Vocoder::streamer: chunk_frames must be >= 2 (got {chunk_frames})");
        }
        Ok(VocoderStreamer {
            voc: self,
            frames: Vec::new(),
            emitted: 0,
            chunk_frames,
            left_context,
        })
    }

    fn decode_window(&self, frames: &[[u32; NUM_CODEBOOKS]]) -> Result<Vec<f32>> {
        let device = &self.device;
        let dtype = self.dtype;

        let mut h = self.quantizer.decode(frames, device, dtype)?;

        h = self.pre_conv.forward(&h)?;

        if let (Some((wi, bi)), Some((wo, bo)), Some(nw)) = (
            &self.pre_transformer_in,
            &self.pre_transformer_out,
            &self.pre_transformer_norm,
        ) {
            let h_cl = h.transpose(1, 2)?.contiguous()?;
            let dims = h_cl.dims();
            let (b, t, c) = (dims[0], dims[1], dims[2]);
            let hidden = wi.dims()[0];
            let flat = h_cl.reshape((b * t, c))?;
            let mid = flat.matmul(&wi.t()?.contiguous()?)?.broadcast_add(bi)?;
            let mid = mid.reshape((b, t, hidden))?;

            let transformed = if let Some(ref pt) = self.pre_transformer {
                pt.forward(&mid)?
            } else {
                mid
            };

            let xf = transformed.to_dtype(DType::F32)?;
            let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
            let eps_t = Tensor::new(self.cfg.pre_transformer_rms_eps as f32, xf.device())?;
            let denom = ms.broadcast_add(&eps_t)?.sqrt()?;
            let normed = xf.broadcast_div(&denom)?;
            let w_f = nw.to_dtype(DType::F32)?;
            let normed = normed.broadcast_mul(&w_f)?.to_dtype(transformed.dtype())?;

            let tf_flat = normed.reshape((b * t, hidden))?;
            let out_flat = tf_flat.matmul(&wo.t()?.contiguous()?)?.broadcast_add(bo)?;
            let out_cl = out_flat.reshape((b, t, wo.dims()[0]))?;
            h = out_cl.transpose(1, 2)?.contiguous()?;
        }

        for stage in &self.upsample_stages {
            h = stage.forward(&h)?;
        }

        h = self.init_conv.forward(&h)?;

        for stage in &self.decoder_stages {
            h = stage.forward(&h)?;
        }
        h = self.final_act.forward(&h)?;
        let y = self.final_conv.forward(&h)?;
        let y = y.clamp(-1.0f32, 1.0f32)?;

        let flat = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(flat)
    }

    pub fn decode_base_only(&self, base_tokens: &[u32]) -> Result<Vec<f32>> {
        let frames: Vec<[u32; NUM_CODEBOOKS]> = base_tokens
            .iter()
            .map(|&t| {
                let mut row = [0u32; NUM_CODEBOOKS];
                row[0] = t;
                row
            })
            .collect();
        self.decode(&frames)
    }
}

pub struct VocoderStreamer<'a> {
    voc: &'a Vocoder,
    frames: Vec<[u32; NUM_CODEBOOKS]>,
    emitted: usize,
    chunk_frames: usize,
    left_context: usize,
}

impl VocoderStreamer<'_> {
    pub fn push(&mut self, frame: [u32; NUM_CODEBOOKS]) -> Result<Option<Vec<f32>>> {
        for (k, &tok) in frame.iter().enumerate() {
            if (tok as usize) >= self.voc.cfg.codebook_size {
                anyhow::bail!(
                    "VocoderStreamer::push: frame {} codebook {k} id {tok} >= codebook_size {}",
                    self.frames.len(),
                    self.voc.cfg.codebook_size
                );
            }
        }
        self.frames.push(frame);
        if self.frames.len() - self.emitted >= self.chunk_frames {
            return Ok(Some(self.emit()?));
        }
        Ok(None)
    }

    pub fn finish(mut self) -> Result<Vec<f32>> {
        if self.frames.len() == self.emitted {
            return Ok(Vec::new());
        }
        if self.frames.len() < 2 {
            anyhow::bail!(
                "VocoderStreamer::finish: need at least 2 frames total (got {})",
                self.frames.len()
            );
        }
        self.emit()
    }

    pub fn pending_frames(&self) -> usize {
        self.frames.len() - self.emitted
    }

    fn emit(&mut self) -> Result<Vec<f32>> {
        let spf = self.voc.cfg.upsample_factor();
        let start = self.emitted;
        let end = self.frames.len();
        let ctx = start.min(self.left_context);
        let pcm = self.voc.decode_window(&self.frames[start - ctx..end])?;
        self.emitted = end;
        Ok(pcm[ctx * spf..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_cfg() -> VocoderConfig {
        VocoderConfig {
            codebook_size: 8,
            num_codebooks: NUM_CODEBOOKS,
            codebook_dim: 4,
            quant_proj_dim: 8,
            latent_dim: 16,
            pre_transformer_hidden: 8,
            pre_transformer_layers: 1,
            pre_transformer_heads: 2,
            pre_transformer_kv_heads: 2,
            pre_transformer_head_dim: 4,
            pre_transformer_intermediate: 16,
            pre_transformer_rope_theta: 10_000.0,
            pre_transformer_rms_eps: 1.0e-5,
            pre_transformer_sliding_window: 72,
            pre_transformer_max_seq_len: 8000,
            pre_upsample_dim: 16,
            pre_upsample_strides: PRE_UPSAMPLE_STRIDES,
            decoder_dim: 32,
            upsample_strides: UPSAMPLE_STRIDES,
            res_kernel: 7,
            final_kernel: 7,
            dtype: DType::F32,
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((self.0 >> 33) as u32) as f32 / u32::MAX as f32;
            (v - 0.25) * 0.4
        }

        fn tensor(&mut self, shape: &[usize], dev: &Device) -> Tensor {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
            Tensor::from_vec(v, shape, dev).unwrap()
        }

        fn positive_tensor(&mut self, shape: &[usize], dev: &Device) -> Tensor {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| self.next_f32().abs() + 0.5).collect();
            Tensor::from_vec(v, shape, dev).unwrap()
        }
    }

    fn synthetic_weight_map(cfg: &VocoderConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut r = Lcg(0x5eed_cafe);
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let cs = cfg.codebook_size;
        let cd = cfg.codebook_dim;
        let qp = cfg.quant_proj_dim;
        let lat = cfg.latent_dim;
        let hid = cfg.pre_transformer_hidden;
        let inter = cfg.pre_transformer_intermediate;
        let pu = cfg.pre_upsample_dim;
        let dec = cfg.decoder_dim;

        m.insert(
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum".into(),
            r.tensor(&[cs, cd], dev),
        );
        m.insert(
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage".into(),
            r.positive_tensor(&[cs], dev),
        );
        m.insert(
            "decoder.quantizer.rvq_first.output_proj.weight".into(),
            r.tensor(&[qp, cd, 1], dev),
        );
        for i in 0..cfg.num_codebooks - 1 {
            m.insert(
                format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"),
                r.tensor(&[cs, cd], dev),
            );
            m.insert(
                format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"),
                r.positive_tensor(&[cs], dev),
            );
        }
        m.insert(
            "decoder.quantizer.rvq_rest.output_proj.weight".into(),
            r.tensor(&[qp, cd, 1], dev),
        );

        m.insert(
            "decoder.pre_conv.conv.weight".into(),
            r.tensor(&[lat, qp, 3], dev),
        );
        m.insert("decoder.pre_conv.conv.bias".into(), r.tensor(&[lat], dev));

        m.insert(
            "decoder.pre_transformer.input_proj.weight".into(),
            r.tensor(&[hid, lat], dev),
        );
        m.insert(
            "decoder.pre_transformer.input_proj.bias".into(),
            r.tensor(&[hid], dev),
        );
        m.insert(
            "decoder.pre_transformer.output_proj.weight".into(),
            r.tensor(&[lat, hid], dev),
        );
        m.insert(
            "decoder.pre_transformer.output_proj.bias".into(),
            r.tensor(&[lat], dev),
        );
        m.insert(
            "decoder.pre_transformer.norm.weight".into(),
            r.positive_tensor(&[hid], dev),
        );
        for l in 0..cfg.pre_transformer_layers {
            let p = format!("decoder.pre_transformer.layers.{l}");
            m.insert(
                format!("{p}.input_layernorm.weight"),
                r.positive_tensor(&[hid], dev),
            );
            m.insert(
                format!("{p}.self_attn.q_proj.weight"),
                r.tensor(&[hid, hid], dev),
            );
            m.insert(
                format!("{p}.self_attn.k_proj.weight"),
                r.tensor(&[hid, hid], dev),
            );
            m.insert(
                format!("{p}.self_attn.v_proj.weight"),
                r.tensor(&[hid, hid], dev),
            );
            m.insert(
                format!("{p}.self_attn.o_proj.weight"),
                r.tensor(&[hid, hid], dev),
            );
            m.insert(
                format!("{p}.self_attn_layer_scale.scale"),
                r.tensor(&[hid], dev),
            );
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                r.positive_tensor(&[hid], dev),
            );
            m.insert(
                format!("{p}.mlp.gate_proj.weight"),
                r.tensor(&[inter, hid], dev),
            );
            m.insert(
                format!("{p}.mlp.up_proj.weight"),
                r.tensor(&[inter, hid], dev),
            );
            m.insert(
                format!("{p}.mlp.down_proj.weight"),
                r.tensor(&[hid, inter], dev),
            );
            m.insert(format!("{p}.mlp_layer_scale.scale"), r.tensor(&[hid], dev));
        }

        for (i, &stride) in cfg.pre_upsample_strides.iter().enumerate() {
            m.insert(
                format!("decoder.upsample.{i}.0.conv.weight"),
                r.tensor(&[pu, pu, stride], dev),
            );
            m.insert(
                format!("decoder.upsample.{i}.0.conv.bias"),
                r.tensor(&[pu], dev),
            );
            let bp = format!("decoder.upsample.{i}.1");
            m.insert(
                format!("{bp}.dwconv.conv.weight"),
                r.tensor(&[pu, 1, 7], dev),
            );
            m.insert(format!("{bp}.dwconv.conv.bias"), r.tensor(&[pu], dev));
            m.insert(format!("{bp}.norm.weight"), r.positive_tensor(&[pu], dev));
            m.insert(format!("{bp}.norm.bias"), r.tensor(&[pu], dev));
            m.insert(format!("{bp}.pwconv1.weight"), r.tensor(&[4 * pu, pu], dev));
            m.insert(format!("{bp}.pwconv1.bias"), r.tensor(&[4 * pu], dev));
            m.insert(format!("{bp}.pwconv2.weight"), r.tensor(&[pu, 4 * pu], dev));
            m.insert(format!("{bp}.pwconv2.bias"), r.tensor(&[pu], dev));
            m.insert(format!("{bp}.gamma"), r.tensor(&[pu], dev));
        }

        m.insert(
            "decoder.decoder.0.conv.weight".into(),
            r.tensor(&[dec, pu, 7], dev),
        );
        m.insert("decoder.decoder.0.conv.bias".into(), r.tensor(&[dec], dev));

        let mut cur = dec;
        for (i, &stride) in cfg.upsample_strides.iter().enumerate() {
            let next = (cur / 2).max(1);
            let bp = format!("decoder.decoder.{}.block", i + 1);
            m.insert(format!("{bp}.0.alpha"), r.tensor(&[cur], dev));
            m.insert(format!("{bp}.0.beta"), r.tensor(&[cur], dev));
            m.insert(
                format!("{bp}.1.conv.weight"),
                r.tensor(&[cur, next, 2 * stride], dev),
            );
            m.insert(format!("{bp}.1.conv.bias"), r.tensor(&[next], dev));
            for (u, _d) in [1usize, 3, 9].iter().enumerate() {
                let up = format!("{bp}.{}", u + 2);
                m.insert(format!("{up}.act1.alpha"), r.tensor(&[next], dev));
                m.insert(format!("{up}.act1.beta"), r.tensor(&[next], dev));
                m.insert(
                    format!("{up}.conv1.conv.weight"),
                    r.tensor(&[next, next, cfg.res_kernel], dev),
                );
                m.insert(format!("{up}.conv1.conv.bias"), r.tensor(&[next], dev));
                m.insert(format!("{up}.act2.alpha"), r.tensor(&[next], dev));
                m.insert(format!("{up}.act2.beta"), r.tensor(&[next], dev));
                m.insert(
                    format!("{up}.conv2.conv.weight"),
                    r.tensor(&[next, next, 1], dev),
                );
                m.insert(format!("{up}.conv2.conv.bias"), r.tensor(&[next], dev));
            }
            cur = next;
        }
        let n = cfg.upsample_strides.len();
        m.insert(
            format!("decoder.decoder.{}.alpha", n + 1),
            r.tensor(&[cur], dev),
        );
        m.insert(
            format!("decoder.decoder.{}.beta", n + 1),
            r.tensor(&[cur], dev),
        );
        m.insert(
            format!("decoder.decoder.{}.conv.weight", n + 2),
            r.tensor(&[1, cur, 7], dev),
        );
        m.insert(
            format!("decoder.decoder.{}.conv.bias", n + 2),
            r.tensor(&[1], dev),
        );
        m
    }

    #[test]
    fn upsample_factor_matches_real_model() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.upsample_factor(), SAMPLES_PER_FRAME);
        assert_eq!(cfg.upsample_factor(), 1_920);
    }

    #[test]
    fn snake_applies_exp_reparameterization() {
        let dev = Device::Cpu;
        let alpha_raw = Tensor::from_vec(vec![0.3f32, -0.7], (2,), &dev).unwrap();
        let beta_raw = Tensor::from_vec(vec![-0.5f32, 0.2], (2,), &dev).unwrap();
        let s = Snake::from_raw(&alpha_raw, &beta_raw, DType::F32).unwrap();
        let x_vals = vec![0.5f32, -1.25, 2.0, 0.1];
        let x = Tensor::from_vec(x_vals.clone(), (1, 2, 2), &dev).unwrap();
        let y = s
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let a = [0.3f32.exp(), (-0.7f32).exp()];
        let b = [(-0.5f32).exp(), 0.2f32.exp()];
        for (i, &xv) in x_vals.iter().enumerate() {
            let c = i / 2;
            let expected = xv + (xv * a[c]).sin().powi(2) / (b[c] + 1.0e-9f32);
            assert!(
                (y[i] - expected).abs() < 1.0e-5,
                "i={i} got {} expected {expected}",
                y[i]
            );
        }
    }

    #[test]
    fn causal_conv_output_ignores_future_inputs() {
        let dev = Device::Cpu;
        let mut r = Lcg(7);
        let w = r.tensor(&[4, 4, 7], &dev);
        let b = r.tensor(&[4], &dev);
        let conv = CausalConv1d::new(w, Some(b), 2, 1).unwrap();

        let t = 12usize;
        let x1 = r.tensor(&[1, 4, t], &dev);
        let mut vals = x1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in 0..4 {
            vals[c * t + (t - 1)] += 5.0;
        }
        let x2 = Tensor::from_vec(vals, (1, 4, t), &dev).unwrap();

        let y1 = conv
            .forward(&x1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let y2 = conv
            .forward(&x2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y1.len(), 4 * t);
        for c in 0..4 {
            for tt in 0..t - 1 {
                assert_eq!(
                    y1[c * t + tt],
                    y2[c * t + tt],
                    "future leak at c={c} t={tt}"
                );
            }
        }
        assert_ne!(y1[t - 1], y2[t - 1]);
    }

    #[test]
    fn causal_transconv_length_and_causality() {
        let dev = Device::Cpu;
        let mut r = Lcg(11);
        let stride = 3usize;
        let k = 6usize;
        let w = r.tensor(&[2, 2, k], &dev);
        let b = r.tensor(&[2], &dev);
        let tc = CausalConvT1d::new(w, Some(b), stride).unwrap();

        let t = 5usize;
        let x1 = r.tensor(&[1, 2, t], &dev);
        let mut vals = x1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in 0..2 {
            vals[c * t + (t - 1)] += 3.0;
        }
        let x2 = Tensor::from_vec(vals, (1, 2, t), &dev).unwrap();

        let y1 = tc.forward(&x1).unwrap();
        assert_eq!(y1.dims(), &[1, 2, t * stride]);
        let y1 = y1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let y2 = tc
            .forward(&x2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let ot = t * stride;
        for c in 0..2 {
            for tt in 0..(t - 1) * stride {
                assert_eq!(
                    y1[c * ot + tt],
                    y2[c * ot + tt],
                    "future leak at c={c} t={tt}"
                );
            }
        }
    }

    #[test]
    fn sliding_window_mask_is_causal() {
        let dev = Device::Cpu;
        let t = 6usize;
        let window = 3usize;
        let m = build_sliding_causal_mask(t, window, &dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for i in 0..t {
            for j in 0..t {
                let v = m[i * t + j];
                if j > i || i - j >= window {
                    assert_eq!(v, f32::NEG_INFINITY, "i={i} j={j}");
                } else {
                    assert_eq!(v, 0.0, "i={i} j={j}");
                }
            }
        }
    }

    #[test]
    fn synthetic_checkpoint_decode_is_causal_and_bounded() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dir =
            std::env::temp_dir().join(format!("nv_omni_vocoder_synth_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let map = synthetic_weight_map(&cfg, &dev);
        candle_core::safetensors::save(&map, &path).unwrap();

        let loader = WeightLoader::open_file(&path, &dev).unwrap();
        let voc = Vocoder::from_qwen3_weights_with_cfg(&loader, &cfg, &dev).unwrap();
        assert!(!voc.is_zero_init());

        let t = 6usize;
        let mk_frames = || -> Vec<[u32; NUM_CODEBOOKS]> {
            (0..t)
                .map(|i| {
                    let mut f = [0u32; NUM_CODEBOOKS];
                    for (k, slot) in f.iter_mut().enumerate() {
                        *slot = ((i * 31 + k * 7) % cfg.codebook_size) as u32;
                    }
                    f
                })
                .collect()
        };
        let frames_a = mk_frames();
        let mut frames_b = mk_frames();
        for f in frames_b.iter_mut().skip(t - 2) {
            for v in f.iter_mut() {
                *v = (*v + 1) % cfg.codebook_size as u32;
            }
        }

        let pa = voc.decode(&frames_a).unwrap();
        let pb = voc.decode(&frames_b).unwrap();
        let spf = cfg.upsample_factor();
        assert_eq!(pa.len(), t * spf);
        assert_eq!(pb.len(), t * spf);

        let keep = (t - 2) * spf;
        assert_eq!(
            &pa[..keep],
            &pb[..keep],
            "future frames leaked into past samples"
        );
        assert_ne!(&pa[keep..], &pb[keep..]);

        let peak = pa.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(peak <= 1.0, "output must be clamped to [-1, 1], got {peak}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_zero_weights_produces_zero_waveform_of_expected_length() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg.clone(), &dev).expect("build vocoder");
        assert!(voc.is_zero_init());

        let t_frames = 4usize;
        let zero_frame = [0u32; NUM_CODEBOOKS];
        let frames = vec![zero_frame; t_frames];

        let pcm = voc.decode(&frames).expect("decode");
        assert_eq!(pcm.len(), t_frames * cfg.upsample_factor());
        assert_eq!(pcm.len(), 4 * 1_920);
        let max_abs = pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert_eq!(
            max_abs, 0.0,
            "zero-init vocoder must produce exact-zero waveform"
        );
    }

    #[test]
    fn decode_handles_arbitrary_frame_count() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg.clone(), &dev).expect("build vocoder");
        for &t in &[2usize, 3, 5] {
            let frames = vec![[0u32; NUM_CODEBOOKS]; t];
            let pcm = voc.decode(&frames).unwrap();
            assert_eq!(pcm.len(), t * cfg.upsample_factor(), "T={t}");
        }
    }

    #[test]
    fn decode_rejects_out_of_range_token() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg.clone(), &dev).unwrap();
        let mut bad = [0u32; NUM_CODEBOOKS];
        bad[0] = cfg.codebook_size as u32;
        let frames = vec![bad, [0u32; NUM_CODEBOOKS]];
        let err = voc.decode(&frames).expect_err("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("speech_tokens"), "msg = {msg}");
    }

    #[test]
    fn decode_base_only_wraps_to_full_frames() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg.clone(), &dev).unwrap();
        let pcm = voc.decode_base_only(&[1, 2, 3]).unwrap();
        assert_eq!(pcm.len(), 3 * cfg.upsample_factor());
    }

    #[test]
    fn decode_rejects_single_frame_with_clean_error() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg, &dev).unwrap();
        let err = voc.decode(&[[0u32; NUM_CODEBOOKS]]).expect_err("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("at least 2 frames"), "msg = {msg}");
    }

    fn synth_vocoder(cfg: &VocoderConfig, dev: &Device, tag: &str) -> Vocoder {
        let dir = std::env::temp_dir().join(format!(
            "nv_omni_vocoder_streamer_{tag}_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let map = synthetic_weight_map(cfg, dev);
        candle_core::safetensors::save(&map, &path).unwrap();
        let loader = WeightLoader::open_file(&path, dev).unwrap();
        let voc = Vocoder::from_qwen3_weights_with_cfg(&loader, cfg, dev).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        voc
    }

    fn varied_frames(cfg: &VocoderConfig, t: usize) -> Vec<[u32; NUM_CODEBOOKS]> {
        (0..t)
            .map(|i| {
                let mut f = [0u32; NUM_CODEBOOKS];
                for (k, r) in f.iter_mut().enumerate() {
                    *r = ((i * 31 + k * 7) % cfg.codebook_size) as u32;
                }
                f
            })
            .collect()
    }

    fn stream_all(
        voc: &Vocoder,
        frames: &[[u32; NUM_CODEBOOKS]],
        chunk: usize,
        ctx: usize,
    ) -> Vec<f32> {
        let mut streamer = voc.streamer(chunk, ctx).unwrap();
        let mut out = Vec::new();
        for f in frames {
            if let Some(pcm) = streamer.push(*f).unwrap() {
                out.extend_from_slice(&pcm);
            }
        }
        out.extend_from_slice(&streamer.finish().unwrap());
        out
    }

    #[test]
    fn streamer_with_full_left_context_matches_batch_decode() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = synth_vocoder(&cfg, &dev, "fullctx");
        let frames = varied_frames(&cfg, 10);
        let batch = voc.decode(&frames).unwrap();
        for &chunk in &[2usize, 3, 4, 7] {
            let streamed = stream_all(&voc, &frames, chunk, 1_000);
            assert_eq!(streamed.len(), batch.len(), "chunk={chunk}");
            let max_delta = streamed
                .iter()
                .zip(&batch)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_delta <= 1.0e-5,
                "chunk={chunk}: causal windowed decode diverged from batch, max delta {max_delta}"
            );
        }
    }

    fn narrow_cfg() -> VocoderConfig {
        let mut cfg = tiny_cfg();
        cfg.latent_dim = 8;
        cfg.pre_upsample_dim = 8;
        cfg.decoder_dim = 8;
        cfg
    }

    fn max_abs_dev(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn rms_dev(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n).sqrt()
    }

    const SEAM_MAX_ABS_DEV: f32 = 1.0e-7;

    #[test]
    fn decode_geometry_chunks_and_its_seam_matches_the_unchunked_decode() {
        let dev = Device::Cpu;
        let cfg = narrow_cfg();
        let voc = synth_vocoder(&cfg, &dev, "chunkseam");
        let spf = cfg.upsample_factor();
        let frames = varied_frames(&cfg, CHUNK_FRAMES + 2);

        let mut st = voc.streamer(CHUNK_FRAMES, CHUNK_LEFT_CONTEXT).unwrap();
        let mut chunk_lens: Vec<usize> = Vec::new();
        let mut streamed: Vec<f32> = Vec::new();
        for f in &frames {
            if let Some(pcm) = st.push(*f).unwrap() {
                chunk_lens.push(pcm.len());
                streamed.extend_from_slice(&pcm);
            }
        }
        let tail = st.finish().unwrap();
        assert_eq!(
            chunk_lens,
            vec![CHUNK_FRAMES * spf],
            "decode geometry must emit exactly one chunk at frame CHUNK_FRAMES={CHUNK_FRAMES}"
        );
        assert_eq!(tail.len(), 2 * spf);
        streamed.extend_from_slice(&tail);

        let unchunked = voc.decode_window(&frames).unwrap();
        assert_eq!(streamed.len(), unchunked.len());
        let seam = max_abs_dev(&streamed, &unchunked);
        assert!(
            seam <= SEAM_MAX_ABS_DEV,
            "chunk seam under CHUNK_LEFT_CONTEXT={CHUNK_LEFT_CONTEXT} diverged from the unchunked \
             decode: max abs dev {seam} > {SEAM_MAX_ABS_DEV}, a bound calibrated on a measured \
             7.45e-9 (one f32 ulp at unit magnitude, 13x headroom)"
        );

        let starved = stream_all(&voc, &frames, CHUNK_FRAMES, 1);
        let starved_seam = max_abs_dev(&starved, &unchunked);
        assert!(
            starved_seam > SEAM_MAX_ABS_DEV,
            "bound gates nothing: one frame of left context must break it (measured 1.10e-6, 11x \
             the bound; zero frames measure 1.60e-2) but got max abs dev {starved_seam}"
        );
    }

    const TRUNCATED_CTX_MAX_RMS_DEV: f32 = 4.0e-6;

    #[test]
    fn streamer_truncated_context_stays_bounded_vs_batch() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = synth_vocoder(&cfg, &dev, "trunc");
        let frames = varied_frames(&cfg, 14);
        let batch = voc.decode(&frames).unwrap();
        let streamed = stream_all(&voc, &frames, 3, 2);
        assert_eq!(streamed.len(), batch.len());
        assert!(streamed.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let dev_2 = rms_dev(&streamed, &batch);
        assert!(
            dev_2 <= TRUNCATED_CTX_MAX_RMS_DEV,
            "truncated-context decode drifted: rms dev {dev_2} > {TRUNCATED_CTX_MAX_RMS_DEV}, a \
             bound calibrated on a measured 1.33e-6 with 3x headroom"
        );

        let dev_1 = rms_dev(&stream_all(&voc, &frames, 3, 1), &batch);
        assert!(
            dev_1 > TRUNCATED_CTX_MAX_RMS_DEV,
            "bound gates nothing: one frame of left context must break it (measured 1.03e-5, 2.6x \
             the bound; zero frames measure 1.82e-4) but got rms dev {dev_1}"
        );
    }

    #[test]
    fn streamer_rejects_bad_inputs() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg.clone(), &dev).unwrap();

        let Err(err) = voc.streamer(1, 25) else {
            panic!("chunk_frames=1 must fail")
        };
        assert!(format!("{err}").contains("chunk_frames"), "{err}");

        let mut st = voc.streamer(4, 4).unwrap();
        let mut bad = [0u32; NUM_CODEBOOKS];
        bad[2] = cfg.codebook_size as u32;
        let err = st.push(bad).expect_err("out-of-range id must fail");
        assert!(format!("{err}").contains("codebook_size"), "{err}");

        let mut st = voc.streamer(4, 4).unwrap();
        st.push([0u32; NUM_CODEBOOKS]).unwrap();
        assert_eq!(st.pending_frames(), 1);
        let err = st.finish().expect_err("single-frame stream must fail");
        assert!(format!("{err}").contains("at least 2 frames"), "{err}");

        let st = voc.streamer(4, 4).unwrap();
        assert_eq!(st.finish().unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn streamer_emits_expected_chunk_sample_counts() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let voc = Vocoder::new(cfg.clone(), &dev).unwrap();
        let spf = cfg.upsample_factor();
        let mut st = voc.streamer(3, 2).unwrap();
        let mut emitted: Vec<usize> = Vec::new();
        for i in 0..7usize {
            let mut f = [0u32; NUM_CODEBOOKS];
            f[0] = (i % cfg.codebook_size) as u32;
            if let Some(pcm) = st.push(f).unwrap() {
                emitted.push(pcm.len());
            }
        }
        emitted.push(st.finish().unwrap().len());
        assert_eq!(emitted, vec![3 * spf, 3 * spf, spf]);
    }

    #[test]
    fn config_rejects_wrong_upsample_product() {
        let dev = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.upsample_strides = [2, 2, 2, 2];
        let err = Vocoder::new(cfg, &dev).err().expect("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("upsample_factor"), "msg = {msg}");
    }
}
