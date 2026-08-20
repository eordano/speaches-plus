use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::attn::{flash_attn, sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
#[cfg(feature = "cuda")]
use nv_weights::{QuantScheme, QuantizationConfig};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

use crate::CausalLm;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl Qwen3Config {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let mut v: serde_json::Value =
            serde_json::from_str(s).context("parse qwen3 config json")?;
        Self::normalize_eos(&mut v);
        let cfg: Qwen3Config = serde_json::from_value(v).context("deserialize qwen3 config")?;
        Ok(cfg)
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    fn normalize_eos(v: &mut serde_json::Value) {
        let Some(map) = v.as_object_mut() else { return };
        if let Some(eos) = map.get_mut("eos_token_id") {
            if let Some(arr) = eos.as_array() {
                if let Some(first) = arr.first().cloned() {
                    *eos = first;
                }
            }
        }
    }
}

pub struct KvCache {
    layers: Vec<(Tensor, Tensor)>,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
    dtype: DType,
}

impl KvCache {
    pub fn new(
        config: &Qwen3Config,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let shape = (
            1usize,
            max_seq_len,
            config.num_key_value_heads,
            config.head_dim,
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let k = Tensor::zeros(shape, dtype, device)?;
            let v = Tensor::zeros(shape, dtype, device)?;
            layers.push((k, v));
        }
        Ok(Self {
            layers,
            current_len: 0,
            max_seq_len,
            n_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            device: device.clone(),
            dtype,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn write_at(
        &mut self,
        layer: usize,
        start: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != self.n_kv_heads || dims[3] != self.head_dim
        {
            anyhow::bail!(
                "KvCache.write_at: expected [1, t, {}, {}], got {:?}",
                self.n_kv_heads,
                self.head_dim,
                dims
            );
        }
        if v_new.dims() != dims {
            anyhow::bail!(
                "KvCache.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        let end = start + t;
        if end > self.max_seq_len {
            anyhow::bail!(
                "KvCache.write_at: end {} exceeds max_seq_len {}",
                end,
                self.max_seq_len
            );
        }
        if layer >= self.layers.len() {
            anyhow::bail!("KvCache.write_at: layer {} out of range", layer);
        }
        let (k_buf, v_buf) = &self.layers[layer];
        let k_updated = k_buf.slice_assign(
            &[0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim],
            k_new,
        )?;
        let v_updated = v_buf.slice_assign(
            &[0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim],
            v_new,
        )?;
        self.layers[layer] = (k_updated, v_updated);
        Ok(())
    }

    pub fn append(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let t = k_new.dims().get(1).copied().unwrap_or(0);
        self.write_at(layer, self.current_len, k_new, v_new)?;
        if layer + 1 == self.layers.len() {
            self.current_len += t;
        }
        Ok(())
    }

    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }

    pub fn get(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        self.view(layer, self.current_len)
    }

    pub fn view(&self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        if layer >= self.layers.len() {
            anyhow::bail!("KvCache.view: layer {} out of range", layer);
        }
        if len > self.max_seq_len {
            anyhow::bail!("KvCache.view: len {} > max {}", len, self.max_seq_len);
        }
        let (k, v) = &self.layers[layer];
        let k = k.narrow(1, 0, len)?;
        let v = v.narrow(1, 0, len)?;
        Ok((k, v))
    }
}

pub struct Qwen3Layer {
    pre_attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    pre_mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl Qwen3Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pre_attn_norm: RmsNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        pre_mlp_norm: RmsNorm,
        mlp: Mlp,
    ) -> Self {
        Self {
            pre_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            pre_mlp_norm,
            mlp,
        }
    }

    pub fn from_loader(
        config: &Qwen3Config,
        idx: usize,
        weights: &WeightLoader,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{}", idx);
        let hidden = config.hidden_size;
        let qd = config.num_attention_heads * config.head_dim;
        let kvd = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let pre_attn_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let pre_mlp_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let q_proj = load_linear(
            weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            qd,
            hidden,
            dtype,
        )?;
        let k_proj = load_linear(
            weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            kvd,
            hidden,
            dtype,
        )?;
        let v_proj = load_linear(
            weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            kvd,
            hidden,
            dtype,
        )?;
        let o_proj = load_linear(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden,
            qd,
            dtype,
        )?;
        let q_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            config.head_dim,
            eps,
            dtype,
        )?;
        let k_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            config.head_dim,
            eps,
            dtype,
        )?;
        let gate_proj = load_linear(
            weights,
            &format!("{prefix}.mlp.gate_proj.weight"),
            inter,
            hidden,
            dtype,
        )?;
        let up_proj = load_linear(
            weights,
            &format!("{prefix}.mlp.up_proj.weight"),
            inter,
            hidden,
            dtype,
        )?;
        let down_proj = load_linear(
            weights,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden,
            inter,
            dtype,
        )?;
        let mlp = Mlp::new(gate_proj, up_proj, down_proj)?;
        Ok(Self {
            pre_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            pre_mlp_norm,
            mlp,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_quantized(
        config: &Qwen3Config,
        idx: usize,
        weights: &WeightLoader,
        dtype: DType,
        qconfig: &QuantizationConfig,
        runners: &QuantRunners,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{}", idx);
        let hidden = config.hidden_size;
        let qd = config.num_attention_heads * config.head_dim;
        let kvd = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let pre_attn_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let pre_mlp_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let q_proj = load_linear_quant(
            weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            qd,
            hidden,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let k_proj = load_linear_quant(
            weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            kvd,
            hidden,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let v_proj = load_linear_quant(
            weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            kvd,
            hidden,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let o_proj = load_linear_quant(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden,
            qd,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let q_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            config.head_dim,
            eps,
            dtype,
        )?;
        let k_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            config.head_dim,
            eps,
            dtype,
        )?;
        let gate_proj = load_linear_quant(
            weights,
            &format!("{prefix}.mlp.gate_proj.weight"),
            inter,
            hidden,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let up_proj = load_linear_quant(
            weights,
            &format!("{prefix}.mlp.up_proj.weight"),
            inter,
            hidden,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let down_proj = load_linear_quant(
            weights,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden,
            inter,
            dtype,
            qconfig,
            runners,
            device,
        )?;
        let mlp = Mlp::new(gate_proj, up_proj, down_proj)?;
        Ok(Self {
            pre_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            pre_mlp_norm,
            mlp,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct QuantRunners {
    pub fp8: Option<Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>>,
    pub nvfp4: Option<Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>>,
}

#[cfg(feature = "cuda")]
impl QuantRunners {
    pub fn new(device: &Device, scheme: QuantScheme) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("QuantRunners requires a CUDA device"),
        };
        let stream = dev.cuda_stream();
        let fp8 = matches!(scheme, QuantScheme::Fp8E4m3)
            .then(|| {
                nv_quant::fp8::Fp8GemmRunner::new(stream.clone()).map(|r| Arc::new(Mutex::new(r)))
            })
            .transpose()?;
        let nvfp4 = matches!(scheme, QuantScheme::Nvfp4)
            .then(|| {
                nv_quant::nvfp4::Nvfp4GemmRunner::new(stream.clone())
                    .map(|r| Arc::new(Mutex::new(r)))
            })
            .transpose()?;
        Ok(Self { fp8, nvfp4 })
    }
}

pub struct Qwen3 {
    config: Qwen3Config,
    embed_weight: Tensor,
    layers: Vec<Qwen3Layer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    dtype: DType,
    device: Device,
}

impl Qwen3 {
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<KvCache> {
        KvCache::new(&self.config, max_seq_len, &self.device, self.dtype)
    }

    pub fn from_loader(
        config: Qwen3Config,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::BF16;
        let embed_name = resolve_name(weights, "model.embed_tokens.weight")
            .context("locate model.embed_tokens.weight")?;
        let embed_weight = weights
            .get(&embed_name, dtype)
            .with_context(|| format!("load {embed_name}"))?;
        let embed_dims = embed_weight.dims();
        if embed_dims.len() != 2
            || embed_dims[0] != config.vocab_size
            || embed_dims[1] != config.hidden_size
        {
            anyhow::bail!(
                "embedding shape mismatch: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                embed_dims
            );
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3Layer::from_loader(&config, i, weights, dtype)?);
        }
        let final_norm = load_rmsnorm(
            weights,
            "model.norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            dtype,
        )?;
        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            let lm_name =
                resolve_name(weights, "lm_head.weight").context("locate lm_head.weight")?;
            weights
                .get(&lm_name, dtype)
                .with_context(|| format!("load {lm_name}"))?
        };
        let lm_head = Linear::new(lm_head_weight, None)?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                base: config.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            rope,
            dtype,
            device: device.clone(),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_quantized(
        config: Qwen3Config,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::BF16;
        if matches!(qconfig.scheme, QuantScheme::None) {
            return Self::from_loader(config, weights, device);
        }
        let runners = QuantRunners::new(device, qconfig.scheme)?;
        let embed_name = resolve_name(weights, "model.embed_tokens.weight")
            .context("locate model.embed_tokens.weight")?;
        let embed_weight = weights
            .get(&embed_name, dtype)
            .with_context(|| format!("load {embed_name}"))?;
        let embed_dims = embed_weight.dims();
        if embed_dims.len() != 2
            || embed_dims[0] != config.vocab_size
            || embed_dims[1] != config.hidden_size
        {
            anyhow::bail!(
                "embedding shape mismatch: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                embed_dims
            );
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3Layer::from_loader_quantized(
                &config, i, weights, dtype, qconfig, &runners, device,
            )?);
        }
        let final_norm = load_rmsnorm(
            weights,
            "model.norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            dtype,
        )?;
        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            let lm_name =
                resolve_name(weights, "lm_head.weight").context("locate lm_head.weight")?;
            weights
                .get(&lm_name, dtype)
                .with_context(|| format!("load {lm_name}"))?
        };
        let lm_head = Linear::new(lm_head_weight, None)?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                base: config.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            rope,
            dtype,
            device: device.clone(),
        })
    }

    pub fn from_parts(
        config: Qwen3Config,
        embed_weight: Tensor,
        layers: Vec<Qwen3Layer>,
        final_norm: RmsNorm,
        lm_head: Linear,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let rope = Rope::new(
            RopeConfig {
                head_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                base: config.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            rope,
            dtype,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let x = self.forward_hidden(tokens, positions, cache)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn forward_hidden(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let tok_dims = tokens.dims();
        if tok_dims.len() != 2 || tok_dims[0] != 1 {
            anyhow::bail!(
                "Qwen3.forward_hidden: tokens must be [1, seq], got {:?}",
                tok_dims
            );
        }
        let seq = tok_dims[1];
        let pos_dims = positions.dims();
        if pos_dims.len() != 1 || pos_dims[0] != seq {
            anyhow::bail!(
                "Qwen3.forward_hidden: positions must be [{}], got {:?}",
                seq,
                pos_dims
            );
        }

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let mut x = self.embed_weight.index_select(&tokens_flat, 0)?.reshape((
            1usize,
            seq,
            self.config.hidden_size,
        ))?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }

        let group_size = self.config.num_attention_heads / self.config.num_key_value_heads;
        let write_start = cache.current_len();
        let new_total = write_start + seq;

        for i in 0..self.layers.len() {
            x = self.layer_forward(
                i,
                &x,
                positions,
                cache,
                seq,
                group_size,
                write_start,
                new_total,
            )?;
        }

        cache.advance(seq);

        let x = self.final_norm.forward(&x)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut KvCache,
        seq: usize,
        _group_size: usize,
        write_start: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];
        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        let normed = layer.pre_attn_norm.forward(x)?;
        let q = layer.q_proj.forward(&normed)?;
        let k = layer.k_proj.forward(&normed)?;
        let v = layer.v_proj.forward(&normed)?;

        let q = q.reshape((1usize, seq, n_heads, head_dim))?;
        let k = k.reshape((1usize, seq, n_kv_heads, head_dim))?;
        let v = v.reshape((1usize, seq, n_kv_heads, head_dim))?;

        let q = layer.q_norm.forward(&q)?;
        let k = layer.k_norm.forward(&k)?;

        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k.to_dtype(DType::F32)?;
        let positions_2d = host_tile_positions(positions, 1, seq, x.device())?;
        let (q_rot, k_rot) = self.rope.apply(&q_f32, &k_f32, &positions_2d)?;
        let q = q_rot.to_dtype(self.dtype)?;
        let k_new = k_rot.to_dtype(self.dtype)?;

        cache.write_at(idx, write_start, &k_new.contiguous()?, &v.contiguous()?)?;
        let (k_full, v_full) = cache.view(idx, new_total)?;

        let attn_cfg = AttnConfig {
            num_heads: n_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = if q.device().is_cuda() {
            flash_attn(
                &q.contiguous()?,
                &k_full.contiguous()?,
                &v_full.contiguous()?,
                &attn_cfg,
            )?
        } else {
            sdpa(
                &q.contiguous()?,
                &k_full.contiguous()?,
                &v_full.contiguous()?,
                &attn_cfg,
            )?
        };
        let attn_out = attn_out.reshape((1usize, seq, n_heads * head_dim))?;
        let attn_out = layer.o_proj.forward(&attn_out)?;

        let x_after = x.add(&attn_out)?;
        let normed2 = layer.pre_mlp_norm.forward(&x_after)?;
        let mlp_out = layer.mlp.forward(&normed2)?;
        let out = x_after.add(&mlp_out)?;
        if out.dims() != [1, seq, hidden] {
            anyhow::bail!(
                "Qwen3 layer {} produced unexpected shape {:?}",
                idx,
                out.dims()
            );
        }
        Ok(out)
    }
}

impl CausalLm for Qwen3 {
    fn forward(&mut self, _tokens: &[u32], _positions: &[u32]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Qwen3::forward(&[u32], &[u32]) shim not implemented; call Qwen3::forward(&Tensor, &Tensor, &mut KvCache)")
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

fn resolve_name(weights: &WeightLoader, name: &str) -> Result<String> {
    if weights.has(name) {
        return Ok(name.to_string());
    }
    if let Some(stripped) = name.strip_prefix("model.") {
        if weights.has(stripped) {
            return Ok(stripped.to_string());
        }
    }
    anyhow::bail!("tensor not found (tried {name} and stripped variant)")
}

fn load_rmsnorm(
    weights: &WeightLoader,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let resolved = resolve_name(weights, name)?;
    let w = weights
        .get(&resolved, dtype)
        .with_context(|| format!("load {resolved}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("rmsnorm {resolved}: expected [{}], got {:?}", dim, d);
    }
    Ok(RmsNorm::new(w, eps))
}

fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let resolved = resolve_name(weights, name)?;
    let w = weights
        .get(&resolved, dtype)
        .with_context(|| format!("load {resolved}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!(
            "linear {resolved}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Linear::new(w, None)
}

#[cfg(feature = "cuda")]
fn load_linear_quant(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    qconfig: &QuantizationConfig,
    runners: &QuantRunners,
    device: &Device,
) -> Result<Linear> {
    let resolved = resolve_name(weights, name)?;
    let module_name = strip_dot_weight(&resolved);
    let is_ignored = qconfig.is_module_ignored(&module_name)
        || qconfig.is_module_ignored(&format!("model.{module_name}"))
        || qconfig.is_module_ignored(&resolved);
    if is_ignored || matches!(qconfig.scheme, QuantScheme::None) {
        return load_linear(weights, name, out_features, in_features, dtype);
    }

    let w = weights
        .get(&resolved, dtype)
        .with_context(|| format!("load {resolved}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!(
            "linear {resolved}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    match qconfig.scheme {
        QuantScheme::Fp8E4m3 => {
            let runner = runners
                .fp8
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("FP8 runner missing"))?
                .clone();
            Linear::from_bf16_quantized_fp8(&w, None, device, runner)
        }
        QuantScheme::Nvfp4 => {
            if in_features < nv_quant::nvfp4::MIN_TILE || out_features < nv_quant::nvfp4::MIN_TILE {
                return load_linear(weights, name, out_features, in_features, dtype);
            }
            let runner = runners
                .nvfp4
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("NVFP4 runner missing"))?
                .clone();
            Linear::from_bf16_quantized_nvfp4(&w, None, device, runner)
        }
        QuantScheme::None => unreachable!(),
    }
}

#[cfg(feature = "cuda")]
fn strip_dot_weight(name: &str) -> String {
    name.strip_suffix(".weight").unwrap_or(name).to_string()
}

pub(crate) fn host_tile_positions(positions: &Tensor, b: usize, t: usize, device: &Device) -> Result<Tensor> {
    let n = positions.elem_count();
    if n != t {
        anyhow::bail!("positions length {} != seq {}", n, t);
    }
    let cpu = positions.to_device(&Device::Cpu)?;
    let row: Vec<i32> = match cpu.dtype() {
        DType::I32 => cpu.to_vec1::<i32>()?,
        DType::I64 => cpu
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        DType::U32 => cpu
            .to_vec1::<u32>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        other => anyhow::bail!("unsupported positions dtype {other:?}"),
    };
    let mut tiled = Vec::with_capacity(b * t);
    for _ in 0..b {
        tiled.extend_from_slice(&row);
    }
    Ok(Tensor::from_vec(tiled, (b, t), device)?)
}
