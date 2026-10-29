use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::load_rmsnorm;
use nv_weights::WeightLoader;
#[cfg(feature = "cuda")]
use nv_weights::{QuantScheme, QuantizationConfig};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

use crate::CausalLm;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeFullParams {
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f32,
    rope_theta: f32,

    #[serde(default)]
    #[allow(dead_code)]
    rope_type: Option<String>,
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn de_softcap_opt<'de, D>(deserializer: D) -> std::result::Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<f32>::deserialize(deserializer)?;
    Ok(opt.unwrap_or(0.0))
}

#[derive(Clone, Debug, Deserialize)]
struct RopeSlidingParams {
    rope_theta: f32,
    #[serde(default)]
    #[allow(dead_code)]
    rope_type: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    full_attention: RopeFullParams,
    sliding_attention: RopeSlidingParams,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Gemma4Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    #[serde(default, deserialize_with = "de_softcap_opt")]
    pub final_logit_softcapping: f32,
    pub layer_types: Vec<LayerType>,
    pub attention_k_eq_v: bool,
    pub tie_word_embeddings: bool,
    pub hidden_activation: String,
    rope_parameters: RopeParameters,

    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub enable_moe_block: bool,
}

impl Gemma4Config {
    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let mut v: Value = serde_json::from_str(s).context("parse gemma4 config json")?;

        let root = v.as_object_mut().context("gemma4 config not an object")?;
        let tie = root
            .get("tie_word_embeddings")
            .cloned()
            .unwrap_or(Value::Bool(true));
        let mut text_obj = match root.remove("text_config") {
            Some(Value::Object(o)) => o,
            Some(_) => bail!("text_config must be an object"),
            None => root.clone(),
        };
        text_obj.insert("tie_word_embeddings".into(), tie);
        let merged = Value::Object(text_obj);
        let cfg: Gemma4Config =
            serde_json::from_value(merged).context("deserialize gemma4 text_config")?;
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "layer_types len {} != num_hidden_layers {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }
        Ok(cfg)
    }

    pub fn layer_kind(&self, idx: usize) -> LayerType {
        self.layer_types[idx]
    }

    pub fn head_dim_for(&self, kind: LayerType) -> usize {
        match kind {
            LayerType::SlidingAttention => self.head_dim,
            LayerType::FullAttention => self.global_head_dim,
        }
    }

    pub fn num_kv_heads_for(&self, kind: LayerType) -> usize {
        match kind {
            LayerType::SlidingAttention => self.num_key_value_heads,
            LayerType::FullAttention => self
                .num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads),
        }
    }

    pub fn has_per_layer_embeddings(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    pub fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    pub fn is_kv_shared_layer(&self, idx: usize) -> bool {
        self.num_kv_shared_layers > 0 && idx >= self.first_kv_shared_layer_idx()
    }

    pub fn kv_source_layer(&self, idx: usize) -> Option<usize> {
        if !self.is_kv_shared_layer(idx) {
            return None;
        }
        let first = self.first_kv_shared_layer_idx();
        let want = self.layer_types[idx];
        self.layer_types[..first].iter().rposition(|&t| t == want)
    }

    pub fn vocab_size_per_layer(&self) -> usize {
        self.vocab_size_per_layer_input.unwrap_or(self.vocab_size)
    }

    pub fn rope_theta_for(&self, kind: LayerType) -> f32 {
        match kind {
            LayerType::SlidingAttention => self.rope_parameters.sliding_attention.rope_theta,
            LayerType::FullAttention => self.rope_parameters.full_attention.rope_theta,
        }
    }

    pub fn rope_partial_factor_for(&self, kind: LayerType) -> f32 {
        match kind {
            LayerType::SlidingAttention => 1.0,
            LayerType::FullAttention => self.rope_parameters.full_attention.partial_rotary_factor,
        }
    }
}

pub struct Gemma4Attention {
    pub kind: LayerType,

    pub qkv_proj: Linear,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub has_v: bool,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,

    pub v_norm: RmsNorm,

    #[cfg(feature = "cuda")]
    pub qkv_prefill_fp4: Option<Linear>,
    #[cfg(feature = "cuda")]
    pub o_prefill_fp4: Option<Linear>,
}

fn megakernel_fuse1_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        env::var("NV_MEGAKERNEL_FUSE1")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

const NORM_GEMV_FUSED_ARM_DEFAULT_OFF_BECAUSE_EVERY_PUBLISHED_GEMMA4_DECODE_NUMBER_WAS_MEASURED_ON_THE_INPUT_LAYERNORM_AS_ITS_OWN_GRAPH_NODE:
    bool = false;

fn input_layernorm_rides_the_qkv_gemv_env_nv_norm_gemv_fused() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        NORM_GEMV_FUSED_ARM_DEFAULT_OFF_BECAUSE_EVERY_PUBLISHED_GEMMA4_DECODE_NUMBER_WAS_MEASURED_ON_THE_INPUT_LAYERNORM_AS_ITS_OWN_GRAPH_NODE
            || env::var("NV_NORM_GEMV_FUSED")
                .ok()
                .and_then(|v| v.trim().parse::<i32>().ok())
                .is_some_and(|rows_per_warp_rung| rows_per_warp_rung > 0)
    })
}

impl Gemma4Attention {
    #[cfg(feature = "cuda")]
    fn prefill_qkv(&self, m: usize) -> Option<&Linear> {
        if !prefill_w4a4_selects(m) || self.qkv_proj.has_lora() {
            return None;
        }
        self.qkv_prefill_fp4.as_ref()
    }

    pub fn o_forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let dims = x.dims();
            let m: usize = dims[..dims.len() - 1].iter().product();
            if !self.o_proj.has_lora() && prefill_w4a4_selects(m) {
                if let Some(fp4) = self.o_prefill_fp4.as_ref() {
                    return fp4.forward(x);
                }
            }
        }
        self.o_proj.forward(x)
    }

    pub fn qkv_forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        use candle_core::D;
        let dims = x.dims();
        let m: usize = dims[..dims.len() - 1].iter().product();
        #[cfg(feature = "cuda")]
        let proj: &Linear = self.prefill_qkv(m).unwrap_or(&self.qkv_proj);
        #[cfg(not(feature = "cuda"))]
        let proj: &Linear = &self.qkv_proj;
        let qkv_bf16 = matches!(proj.kind(), nv_quant::LinearKind::Bf16);
        if !qkv_bf16 || nv_quant::matmul::fused_qkv_bitwise_safe(m, self.has_v) {
            let fused = if qkv_bf16 {
                proj.forward_dense_det(x)?
            } else {
                proj.forward(x)?
            };

            #[cfg(feature = "cuda")]
            if fused.device().is_cuda() {
                let mut parts = vec![(0usize, self.q_dim), (self.q_dim, self.kv_dim)];
                if self.has_v {
                    parts.push((self.q_dim + self.kv_dim, self.kv_dim));
                }
                let mut outs = split_cols_bf16_raw(&fused, &parts)?;
                let v = if self.has_v {
                    outs.pop().unwrap()
                } else {
                    outs[1].clone()
                };
                let k = outs.pop().unwrap();
                let q = outs.pop().unwrap();
                return Ok((q, k, v));
            }
            let q = fused.narrow(D::Minus1, 0, self.q_dim)?.contiguous()?;
            let k = fused
                .narrow(D::Minus1, self.q_dim, self.kv_dim)?
                .contiguous()?;
            let v = if self.has_v {
                fused
                    .narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?
                    .contiguous()?
            } else {
                k.clone()
            };
            Ok((q, k, v))
        } else {
            let q = proj.forward_rows(x, 0, self.q_dim)?;
            let k = proj.forward_rows(x, self.q_dim, self.kv_dim)?;
            let v = if self.has_v {
                proj.forward_rows(x, self.q_dim + self.kv_dim, self.kv_dim)?
            } else {
                k.clone()
            };
            Ok((q, k, v))
        }
    }

    pub fn qkv_forward_prenorm(
        &self,
        x_pre: &Tensor,
        norm: &RmsNorm,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            let dims = x_pre.dims();
            let m: usize = dims[..dims.len() - 1].iter().product();
            let proj: &Linear = self.prefill_qkv(m).unwrap_or(&self.qkv_proj);
            let qkv_bf16 = matches!(proj.kind(), nv_quant::LinearKind::Bf16);

            if !qkv_bf16
                && proj.prenorm_nvfp4_eligible()
                && x_pre.device().is_cuda()
            {
                let fused =
                    proj.forward_prenorm_nvfp4(x_pre, norm.weight_bf16(), norm.eps() as f32)?;
                let mut parts = vec![(0usize, self.q_dim), (self.q_dim, self.kv_dim)];
                if self.has_v {
                    parts.push((self.q_dim + self.kv_dim, self.kv_dim));
                }
                let mut outs = split_cols_bf16_raw(&fused, &parts)?;
                let v = if self.has_v {
                    outs.pop().unwrap()
                } else {
                    outs[1].clone()
                };
                let k = outs.pop().unwrap();
                let q = outs.pop().unwrap();
                return Ok((q, k, v));
            }
        }

        let normed = norm.forward(x_pre)?;
        self.qkv_forward(&normed)
    }
}

pub struct Gemma4Mlp {
    pub gate_up_proj: Linear,
    pub down_proj: Linear,
}

pub struct Gemma4Layer {
    pub kind: LayerType,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub layer_scalar: Tensor,

    pub layer_scalar_host: f32,
    pub self_attn: Gemma4Attention,
    pub mlp: Gemma4Mlp,
}

pub struct Gemma4 {
    config: Gemma4Config,
    embed_weight: Tensor,
    layers: Vec<Gemma4Layer>,
    final_norm: RmsNorm,
    lm_head: Linear,

    #[cfg(feature = "cuda")]
    lm_head_i8: Option<(CudaSlice<i8>, CudaSlice<f32>)>,

    sliding_rope: Rope,

    full_rope: Rope,

    embed_scale: f32,
    dtype: DType,
    device: Device,
}

impl Gemma4 {
    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn from_loader(
        config: Gemma4Config,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        Self::from_loader_inner(config, weights, None, device)
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_quantized(
        config: Gemma4Config,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        device: &Device,
    ) -> Result<Self> {
        if matches!(qconfig.scheme, QuantScheme::None) {
            return Self::from_loader(config, weights, device);
        }
        Self::from_loader_inner(config, weights, Some(qconfig), device)
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn from_loader_inner(
        config: Gemma4Config,
        weights: &WeightLoader,
        #[cfg(feature = "cuda")] qconfig: Option<&QuantizationConfig>,
        #[cfg(not(feature = "cuda"))] _qconfig: Option<&()>,
        device: &Device,
    ) -> Result<Self> {
        ensure!(
            !config.has_per_layer_embeddings(),
            "gemma4::Gemma4 is the dense decoder and has no per-layer-embedding stack, but this \
             config sets hidden_size_per_layer_input={}: it is an E4B/MatFormer checkpoint. \
             Load it with nv_models::gemma4_e4b::Gemma4E4b instead.",
            config.hidden_size_per_layer_input
        );
        ensure!(
            config.num_kv_shared_layers == 0,
            "gemma4::Gemma4 does not implement per-layer KV sharing, but this config sets \
             num_kv_shared_layers={} (layers {}..{} would reuse a source layer's KV). \
             Load it with nv_models::gemma4_e4b::Gemma4E4b instead.",
            config.num_kv_shared_layers,
            config.first_kv_shared_layer_idx(),
            config.num_hidden_layers
        );
        let dtype = DType::BF16;

        #[cfg(feature = "cuda")]
        let nvfp4_runner: Option<Arc<Mutex<Nvfp4GemmRunner>>> = match qconfig {
            Some(q) if matches!(q.scheme, QuantScheme::Nvfp4) => match device {
                Device::Cuda(d) => {
                    let stream = d.cuda_stream();
                    Some(Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone())?)))
                }
                _ => bail!("NVFP4 requires a CUDA device"),
            },
            _ => None,
        };

        #[cfg(feature = "cuda")]
        let fp8_attn_runner: Option<Arc<Mutex<Fp8GemmRunner>>> =
            match (&nvfp4_runner, attn_proj_scheme(), device) {
                (Some(_), AttnProjScheme::Fp8E4m3, Device::Cuda(d)) => Some(Arc::new(Mutex::new(
                    Fp8GemmRunner::new(d.cuda_stream().clone())?,
                ))),
                _ => None,
            };

        let embed_name = "model.language_model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_name, dtype)
            .with_context(|| format!("load {embed_name}"))?;
        let embed_dims = embed_weight.dims();
        if embed_dims.len() != 2
            || embed_dims[0] != config.vocab_size
            || embed_dims[1] != config.hidden_size
        {
            bail!(
                "gemma4 embed: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                embed_dims
            );
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let kind = config.layer_kind(i);
            #[cfg(feature = "cuda")]
            let layer = Gemma4Layer::from_loader(
                &config,
                i,
                kind,
                weights,
                qconfig,
                nvfp4_runner.as_ref().cloned(),
                fp8_attn_runner.as_ref().cloned(),
                device,
                dtype,
            )?;
            #[cfg(not(feature = "cuda"))]
            let layer = Gemma4Layer::from_loader(&config, i, kind, weights, device, dtype)?;
            layers.push(layer);
        }

        let final_norm = load_rmsnorm(
            weights,
            "model.language_model.norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            dtype,
        )?;

        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", dtype)
                .with_context(|| "load lm_head.weight")?
        };
        let lm_head = Linear::new(lm_head_weight.clone(), None)?;

        #[cfg(feature = "cuda")]
        let lm_head_i8 = if env::var("NV_VERIFY_LMHEAD_INT8").as_deref() != Ok("0")
            && device.is_cuda()
        {
            Some(nv_layers::lm_head_resident_quant::rowquant_i8_dev(
                &lm_head_weight,
            )?)
        } else {
            None
        };

        let sliding_rope = build_sliding_rope(&config, device)?;
        let full_rope = build_full_rope(&config, device)?;
        let embed_scale = (config.hidden_size as f32).sqrt();

        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            #[cfg(feature = "cuda")]
            lm_head_i8,
            sliding_rope,
            full_rope,
            embed_scale,
            dtype,
            device: device.clone(),
        })
    }

    pub fn embed_weight(&self) -> &Tensor {
        &self.embed_weight
    }
    pub fn embed_scale(&self) -> f32 {
        self.embed_scale
    }
    pub fn layers(&self) -> &[Gemma4Layer] {
        &self.layers
    }
    pub fn final_norm(&self) -> &RmsNorm {
        &self.final_norm
    }
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }
    #[cfg(feature = "cuda")]
    pub fn lm_head_i8_quantized_copy_absent_when_nv_verify_lmhead_int8_is_0(
        &self,
    ) -> Option<&(CudaSlice<i8>, CudaSlice<f32>)> {
        self.lm_head_i8.as_ref()
    }
    pub fn sliding_rope(&self) -> &Rope {
        &self.sliding_rope
    }
    pub fn full_rope(&self) -> &Rope {
        &self.full_rope
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<Gemma4KvCache> {
        Gemma4KvCache::new(&self.config, max_seq_len, &self.device, self.dtype)
    }

    #[cfg(feature = "cuda")]
    pub fn new_kv_cache_fp8(&self, max_seq_len: usize) -> Result<Gemma4KvCacheFp8> {
        Gemma4KvCacheFp8::new(&self.config, max_seq_len, &self.device)
    }

    #[cfg(feature = "cuda")]
    pub fn new_kv_cache_fp8_windowed(&self, max_seq_len: usize) -> Result<Gemma4KvCacheFp8> {
        Gemma4KvCacheFp8::new_windowed(&self.config, max_seq_len, &self.device)
    }

    pub fn forward(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("Gemma4.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        #[cfg(feature = "cuda")]
        if nv_config::env_arm_present_at_all_reread_every_call("NV_GEMMA4_FP8_KV") {
            let mut cache = self.new_kv_cache_fp8(seq.max(1))?;
            return self.forward_with_cache(tokens, positions, &mut cache);
        }
        let mut cache = self.new_kv_cache(seq.max(1))?;
        self.forward_with_cache(tokens, positions, &mut cache)
    }

    pub fn forward_with_cache<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        let logits = self.forward_with_cache_body(tokens, positions, cache)?;
        let out = tanh_softcap_bf16_to_f32_op(
            &logits,
            self.config.final_logit_softcapping,
            &self.device,
        )?;
        Ok(out)
    }

    pub fn forward_with_cache_last<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        let logits =
            self.forward_with_cache_body_rows(tokens, positions, cache, None, None, Some(1), None)?;
        let out = tanh_softcap_bf16_to_f32_op(
            &logits,
            self.config.final_logit_softcapping,
            &self.device,
        )?;
        Ok(out)
    }

    pub fn forward_with_cache_last_embeds<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        embeds: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        let logits = self.forward_with_cache_body_rows(
            tokens,
            positions,
            cache,
            None,
            None,
            Some(1),
            Some(embeds),
        )?;
        tanh_softcap_bf16_to_f32_op(&logits, self.config.final_logit_softcapping, &self.device)
    }

    pub fn forward_with_cache_hooked<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        hook: &mut dyn Gemma4LayerHook,
    ) -> Result<Tensor> {
        let logits = self.forward_with_cache_body_hooked(tokens, positions, cache, Some(hook))?;
        let out = tanh_softcap_bf16_to_f32_op(
            &logits,
            self.config.final_logit_softcapping,
            &self.device,
        )?;
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_into<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        out_logits: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let logits = self.forward_with_cache_body(tokens, positions, cache)?;
        tanh_softcap_bf16_to_f32_into_op(
            &logits,
            self.config.final_logit_softcapping,
            out_logits,
            &self.device,
        )
    }

    fn forward_with_cache_body<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        self.forward_with_cache_body_hooked(tokens, positions, cache, None)
    }

    fn forward_with_cache_body_hooked<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        hook: Option<&mut dyn Gemma4LayerHook>,
    ) -> Result<Tensor> {
        self.forward_with_cache_body_hooked_masked(tokens, positions, cache, hook, None)
    }

    fn forward_with_cache_body_hooked_masked<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        hook: Option<&mut dyn Gemma4LayerHook>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_cache_body_rows(tokens, positions, cache, hook, attn_mask, None, None)
    }

    fn forward_with_cache_body_rows<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        mut hook: Option<&mut dyn Gemma4LayerHook>,
        attn_mask: Option<&Tensor>,
        logit_rows: Option<usize>,
        embeds_override: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("Gemma4.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            bail!(
                "Gemma4.forward: positions must be [{}], got {:?}",
                seq,
                positions.dims()
            );
        }
        if let Some(m) = attn_mask {
            if m.dims() != [seq, seq] {
                bail!(
                    "Gemma4.forward: attn_mask must be [{seq},{seq}], got {:?}",
                    m.dims()
                );
            }
        }

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let mut hidden = match embeds_override {
            None => {
                let x_flat = embed_lookup_bf16_op(&self.embed_weight, &tokens_flat, &self.device)?;
                let x = x_flat
                    .reshape((1usize, seq, self.config.hidden_size))?
                    .to_dtype(self.dtype)?;
                scale_bf16_op(&x, self.embed_scale, &self.device)?
            }
            Some(e) => {
                ensure!(
                    e.dims() == [seq, self.config.hidden_size],
                    "Gemma4.forward: embeds override must be [{seq}, {}] pre-scaled rows, got {:?}",
                    self.config.hidden_size,
                    e.dims()
                );
                e.to_dtype(self.dtype)?
                    .reshape((1usize, seq, self.config.hidden_size))?
            }
        };

        let write_start = cache.current_len();
        let new_total = write_start + seq;
        cache.prepare_for_decode(write_start, new_total)?;

        for li in 0..self.layers.len() {
            hidden = self
                .layer_forward_masked(li, &hidden, positions, cache, seq, new_total, attn_mask)?;
            if let Some(h) = hook.as_deref_mut() {
                h.after_layer(li, &hidden)?;
            }
        }
        cache.advance(seq);

        let hidden_lr = match logit_rows {
            Some(n) if n < seq => hidden.narrow(1, seq - n.max(1), n.max(1))?.contiguous()?,
            _ => hidden,
        };
        self.final_norm_then_lm_head_rows(&hidden_lr)
    }

    pub fn residual_rows_after_layer<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        layer: usize,
    ) -> Result<Tensor> {
        ensure!(
            layer < self.layers.len(),
            "capture layer {layer} out of range for {} blocks",
            self.layers.len()
        );
        let dims = tokens.dims();
        ensure!(
            dims.len() == 2 && dims[0] == 1,
            "Gemma4 capture: tokens must be [1, seq], got {dims:?}"
        );
        let seq = dims[1];
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = embed_lookup_bf16_op(&self.embed_weight, &tokens_flat, &self.device)?
            .reshape((1usize, seq, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        let mut hidden = scale_bf16_op(&x, self.embed_scale, &self.device)?;
        let write_start = cache.current_len();
        let new_total = write_start + seq;
        cache.prepare_for_decode(write_start, new_total)?;
        for li in 0..=layer {
            hidden =
                self.layer_forward_masked(li, &hidden, positions, cache, seq, new_total, None)?;
        }
        cache.advance(seq);
        Ok(hidden
            .reshape((seq, self.config.hidden_size))?
            .to_dtype(DType::F32)?)
    }

    fn final_norm_then_lm_head_rows(&self, hidden_lr: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if decode_lmhead_i8_arm_enabled_env_nv_g4_lmhead_i8_shared_with_the_batch_graph_step_arm() {
            if let Some(logits) =
                self.lm_head_i8_normed_rows_when_the_mk_kernel_serves_this_row_count(hidden_lr)?
            {
                return Ok(logits);
            }
        }
        let normed = self.final_norm.forward(hidden_lr)?;
        let logits = self.lm_head.forward(&normed)?;
        Ok(logits)
    }

    #[cfg(feature = "cuda")]
    fn lm_head_i8_normed_rows_when_the_mk_kernel_serves_this_row_count(
        &self,
        hidden_lr: &Tensor,
    ) -> Result<Option<Tensor>> {
        let dims = hidden_lr.dims();
        ensure!(
            dims.len() == 3 && dims[0] == 1 && dims[2] == self.config.hidden_size,
            "lm_head epilogue expects [1, rows, hidden], got {dims:?}"
        );
        let rows = dims[1];
        let Some((wq, rs)) = self.lm_head_i8.as_ref() else {
            bail!(
                "NV_G4_LMHEAD_I8=1 needs the i8 lm_head copy; it is absent because \
                 NV_VERIFY_LMHEAD_INT8=0 disabled building it"
            );
        };
        if rows > lm_head_i8_rows_per_call(self.config.hidden_size) {
            return Ok(None);
        }
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("lm_head_i8 epilogue requires a CUDA device")
        };
        let hflat = hidden_lr.reshape((rows, self.config.hidden_size))?;
        let rstd = crate::gemma4_e4b::rstd_op(&hflat, self.config.rms_norm_eps as f32)?;
        let wn = self.final_norm.weight_bf16();
        let raw = nv_layers::lm_head_resident_quant::lm_head_i8_normed_mk_op(
            wq, rs, &hflat, wn, &rstd, rows, &dev,
        )?;
        Ok(Some(raw.reshape((1usize, rows, self.config.vocab_size))?))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut PagedGemma4Cache],
    ) -> Result<Tensor> {
        let b = tokens.len();
        if b == 0 {
            bail!("forward_decode_batched: empty batch");
        }
        if positions.len() != b || caches.len() != b {
            bail!(
                "forward_decode_batched: ragged batch tokens={} positions={} caches={}",
                b,
                positions.len(),
                caches.len()
            );
        }

        let tokens_u32: Vec<u32> = tokens.to_vec();
        let tokens_t = Tensor::from_vec(tokens_u32, b, &self.device)?.to_dtype(DType::U32)?;
        let x_flat = embed_lookup_bf16_op(&self.embed_weight, &tokens_t, &self.device)?;
        let x = x_flat
            .reshape((1usize, b, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        let mut hidden = scale_bf16_op(&x, self.embed_scale, &self.device)?;

        for (i, cache) in caches.iter_mut().enumerate() {
            let len = cache.current_len();
            if len != positions[i] {
                bail!(
                    "forward_decode_batched: seq {i} cache len {len} != position {}",
                    positions[i]
                );
            }
            cache.prepare_for_decode(len, len + 1)?;
        }

        let positions_t = {
            let p: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
            Tensor::from_vec(p, b, &self.device)?
        };

        for li in 0..self.layers.len() {
            hidden = self.layer_forward_decode_batched(li, &hidden, &positions_t, caches, b)?;
        }

        for cache in caches.iter_mut() {
            cache.advance(1);
        }

        let normed = self.final_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&normed)?;
        let dims = logits.dims();
        let vocab = dims[dims.len() - 1];
        let out = logits.reshape((b, vocab))?;
        rowdiff_rows("logits", usize::MAX, &out, 0, b);
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn layer_forward_decode_batched(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut PagedGemma4Cache],
        b: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];

        rowdiff_rows("layer_in", idx, x, 1, b);
        let residual_attn = x.clone();
        let normed_pre_attn = layer.input_layernorm.forward(x)?;
        rowdiff_rows("normed_pre_attn", idx, &normed_pre_attn, 1, b);
        let attn_out =
            self.attention_forward_decode_batched(idx, &normed_pre_attn, positions, caches, b)?;
        let attn_post = layer.post_attention_layernorm.forward(&attn_out)?;

        let (normed_pre_mlp, after_attn) = layer
            .pre_feedforward_layernorm
            .forward_residual(&attn_post, &residual_attn)?;
        rowdiff_rows("normed_pre_mlp", idx, &normed_pre_mlp, 1, b);

        let residual_mlp = after_attn.clone();
        let mlp_out = mlp_forward(&layer.mlp, &normed_pre_mlp)?;
        rowdiff_rows("mlp_out", idx, &mlp_out, 1, b);
        let mlp_post = layer.post_feedforward_layernorm.forward(&mlp_out)?;

        let out = residual_add_scale_bf16_op(
            &residual_mlp,
            &mlp_post,
            layer.layer_scalar_host,
            &self.device,
        )?;
        rowdiff_rows("layer_out", idx, &out, 1, b);
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn attention_forward_decode_batched(
        &self,
        layer_idx: usize,
        x: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut PagedGemma4Cache],
        b: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = self.config.head_dim_for(kind);
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_kv_heads_for(kind);
        let rope = match kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        let window = match kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };

        rowdiff_rows("attn_in", layer_idx, x, 1, b);
        let (q_raw, k_raw, v_raw) = attn.qkv_forward(x)?;
        rowdiff_rows("q_raw", layer_idx, &q_raw, 1, b);
        rowdiff_rows("k_raw", layer_idx, &k_raw, 1, b);
        rowdiff_rows("v_raw", layer_idx, &v_raw, 1, b);
        let q = q_raw.reshape((1usize, b, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k = k_raw.reshape((1usize, b, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
        let v = v_raw.reshape((1usize, b, n_kv, head_dim))?;
        let v_normed = attn.v_norm.forward(&v)?;
        rowdiff_rows("q_normed", layer_idx, &q_normed, 1, b);
        rowdiff_rows("k_normed", layer_idx, &k_normed, 1, b);
        rowdiff_rows("v_normed", layer_idx, &v_normed, 1, b);

        let (q_rot, k_rot) = rope.apply(&q_normed, &k_normed, positions)?;
        let (q_rot, k_rot) = crate::hadamard_kv::maybe_rotate_qk(q_rot, k_rot, head_dim)?;
        let q_rot = q_rot.contiguous()?;
        let k_rot = k_rot.contiguous()?;
        let v_for_cache = v_normed.contiguous()?;
        rowdiff_rows("q_rot", layer_idx, &q_rot, 1, b);
        rowdiff_rows("k_rot", layer_idx, &k_rot, 1, b);

        let mut rows: Vec<Tensor> = Vec::with_capacity(b);
        for i in 0..b {
            let q_i = q_rot.narrow(1, i, 1)?.contiguous()?;
            let k_i = k_rot.narrow(1, i, 1)?.contiguous()?;
            let v_i = v_for_cache.narrow(1, i, 1)?.contiguous()?;
            rowdiff_slot("q_i", layer_idx, i, &q_i);
            rowdiff_slot("k_i", layer_idx, i, &k_i);
            rowdiff_slot("v_i", layer_idx, i, &v_i);

            let cache = &mut *caches[i];
            cache.write_at(layer_idx, &k_i, &v_i)?;
            let total = cache.current_len() + 1;

            let out_i = match cache.try_decode_attention_fp8(layer_idx, &q_i, n_q, window, 1.0)? {
                Some(out) => out,
                None => {
                    let (k_full, v_full) = cache.view(layer_idx, total)?;
                    rowdiff_slot("k_full", layer_idx, i, &k_full);
                    rowdiff_slot("v_full", layer_idx, i, &v_full);

                    {
                        flash_attention(
                            &q_i,
                            &k_full,
                            &v_full,
                            n_q,
                            n_kv,
                            head_dim,
                            1,
                            window.map(|w| w.saturating_sub(1)),
                        )?
                    }
                }
            };
            rowdiff_slot("attn_out_i", layer_idx, i, &out_i);
            rows.push(out_i.reshape((1usize, 1usize, n_q * head_dim))?);
        }

        let attn_out = Tensor::cat(&rows, 1)?;
        rowdiff_rows("attn_cat", layer_idx, &attn_out, 1, b);
        let o = attn.o_proj.forward(&attn_out)?;
        rowdiff_rows("attn_o", layer_idx, &o, 1, b);
        Ok(o)
    }

    pub fn forward_with_aux_hidden_masked(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        layers: &[usize],
        attn_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("Gemma4.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        for &li in layers {
            if li >= self.layers.len() {
                bail!(
                    "forward_with_aux_hidden_masked: layer index {} out of range (have {} layers)",
                    li,
                    self.layers.len()
                );
            }
        }
        let mut collector = Gemma4HiddenCollector::new(layers);
        let mut cache = self.new_kv_cache(seq.max(1))?;
        let raw = self.forward_with_cache_body_hooked_masked(
            tokens,
            positions,
            &mut cache,
            Some(&mut collector),
            attn_mask,
        )?;
        let logits =
            tanh_softcap_bf16_to_f32_op(&raw, self.config.final_logit_softcapping, &self.device)?;
        let hidden_states = collector.into_hidden(layers, seq, self.config.hidden_size)?;
        Ok((logits, hidden_states))
    }

    pub fn forward_with_aux_hidden(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("Gemma4.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        for &li in layers {
            if li >= self.layers.len() {
                bail!(
                    "forward_with_aux_hidden: layer index {} out of range (have {} layers)",
                    li,
                    self.layers.len()
                );
            }
        }
        let mut collector = Gemma4HiddenCollector::new(layers);
        #[cfg(feature = "cuda")]
        let logits = if nv_config::env_arm_present_at_all_reread_every_call("NV_GEMMA4_FP8_KV") {
            let mut cache = self.new_kv_cache_fp8(seq.max(1))?;
            let raw = self.forward_with_cache_body_hooked(
                tokens,
                positions,
                &mut cache,
                Some(&mut collector),
            )?;
            tanh_softcap_bf16_to_f32_op(&raw, self.config.final_logit_softcapping, &self.device)?
        } else {
            let mut cache = self.new_kv_cache(seq.max(1))?;
            let raw = self.forward_with_cache_body_hooked(
                tokens,
                positions,
                &mut cache,
                Some(&mut collector),
            )?;
            tanh_softcap_bf16_to_f32_op(&raw, self.config.final_logit_softcapping, &self.device)?
        };
        #[cfg(not(feature = "cuda"))]
        let logits = {
            let mut cache = self.new_kv_cache(seq.max(1))?;
            let raw = self.forward_with_cache_body_hooked(
                tokens,
                positions,
                &mut cache,
                Some(&mut collector),
            )?;
            tanh_softcap_bf16_to_f32_op(&raw, self.config.final_logit_softcapping, &self.device)?
        };
        let hidden_states = collector.into_hidden(layers, seq, self.config.hidden_size)?;
        Ok((logits, hidden_states))
    }

    fn layer_forward_masked<C: Gemma4Cache>(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        seq: usize,
        new_total: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];

        let residual_attn = x.clone();

        let attn_out = if megakernel_fuse1_enabled()
            || input_layernorm_rides_the_qkv_gemv_env_nv_norm_gemv_fused()
        {
            self.attention_forward_masked(
                idx,
                x,
                positions,
                cache,
                seq,
                new_total,
                attn_mask,
                Some(&layer.input_layernorm),
            )?
        } else {
            let normed_pre_attn = layer.input_layernorm.forward(x)?;
            self.attention_forward_masked(
                idx,
                &normed_pre_attn,
                positions,
                cache,
                seq,
                new_total,
                attn_mask,
                None,
            )?
        };
        let det_sub = det_debug_enabled() && seq > 1 && attn_mask.is_none();
        if det_sub {
            det_hash_tensor("o_proj", idx, &attn_out);
        }
        let (normed_pre_mlp, after_attn) = decode_post_attn_pre_ff(
            &layer.post_attention_layernorm,
            &layer.pre_feedforward_layernorm,
            &attn_out,
            &residual_attn,
            &self.device,
        )?;

        let residual_mlp = after_attn.clone();
        let mlp_out = mlp_forward(&layer.mlp, &normed_pre_mlp)?;
        if det_sub {
            det_hash_tensor("mlp_out", idx, &mlp_out);
        }

        decode_post_ff_residual(
            &layer.post_feedforward_layernorm,
            &mlp_out,
            &residual_mlp,
            layer.layer_scalar_host,
            &self.device,
        )
    }

    fn attention_forward_masked<C: Gemma4Cache>(
        &self,
        layer_idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        seq: usize,
        new_total: usize,
        attn_mask: Option<&Tensor>,
        prenorm: Option<&RmsNorm>,
    ) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = self.config.head_dim_for(kind);
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_kv_heads_for(kind);
        let rope = match kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };

        let (q_raw, k_raw, v_raw) = match prenorm {
            Some(norm) => attn.qkv_forward_prenorm(x, norm)?,
            None => attn.qkv_forward(x)?,
        };
        let q = q_raw.reshape((1usize, seq, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k = k_raw.reshape((1usize, seq, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
        let v = v_raw.reshape((1usize, seq, n_kv, head_dim))?;
        let v_normed = attn.v_norm.forward(&v)?;

        let (q_rot, k_rot) = rope.apply(&q_normed, &k_normed, positions)?;
        let (q_rot, k_rot) = crate::hadamard_kv::maybe_rotate_qk(q_rot, k_rot, head_dim)?;
        let q_rot = q_rot.contiguous()?;
        let k_rot = k_rot.contiguous()?;
        let v_for_cache = v_normed.contiguous()?;

        let det_sub = det_debug_enabled() && seq > 1 && attn_mask.is_none();
        if det_sub {
            det_hash_tensor("q_raw", layer_idx, &q_raw);
            det_hash_tensor("q_rot", layer_idx, &q_rot);
            det_hash_tensor("k_rot", layer_idx, &k_rot);
            det_hash_tensor("v_cache_in", layer_idx, &v_for_cache);
        }

        cache.write_at(layer_idx, &k_rot, &v_for_cache)?;

        let window = match kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };

        let attn_out = if let Some(mask) = attn_mask {
            let (k_full, v_full) = cache.view(layer_idx, new_total)?;
            sdpa_with_mask(
                &q_rot, &k_full, &v_full, n_q, n_kv, head_dim, seq, mask, window,
            )?
        } else {
            let fp8_fast_path_out = if seq == 1 {
                cache.try_decode_attention_fp8(layer_idx, &q_rot, n_q, window, 1.0)?
            } else {
                cache.try_prefill_attention_fp8(layer_idx, &q_rot, n_q, seq, window, 1.0)?
            };
            if let Some(out) = fp8_fast_path_out {
                out
            } else {
                let (k_full, v_full) = cache.view(layer_idx, new_total)?;
                if det_sub {
                    det_hash_tensor("k_full", layer_idx, &k_full);
                    det_hash_tensor("v_full", layer_idx, &v_full);
                }
                if matches!(kind, LayerType::FullAttention) && seq > 1 {
                    causal_attention_chunked(
                        &q_rot,
                        &k_full,
                        &v_full,
                        n_q,
                        n_kv,
                        head_dim,
                        seq,
                        new_total - seq,
                    )?
                } else {
                    flash_attention(
                        &q_rot,
                        &k_full,
                        &v_full,
                        n_q,
                        n_kv,
                        head_dim,
                        seq,
                        window.map(|w| w.saturating_sub(1)),
                    )?
                }
            }
        };
        if det_sub {
            det_hash_tensor("attn_out", layer_idx, &attn_out);
        }

        let attn_out_flat = attn_out.reshape((1usize, seq, n_q * head_dim))?;
        attn.o_forward(&attn_out_flat)
    }
}

pub trait Gemma4LayerHook {
    fn after_layer(&mut self, layer_idx: usize, hidden: &Tensor) -> Result<()>;
}

fn det_debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| nv_config::env_arm_present_at_all_reread_every_call("NV_DEBUG_DETERMINISM"))
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn batch_rowdiff_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| nv_config::env_arm_present_at_all_reread_every_call("NV_BATCH_ROWDIFF"))
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
static ROWDIFF_TRIPPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn rowdiff_hash(t: &Tensor) -> Result<(usize, u64)> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in &v {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok((v.len(), h))
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn rowdiff_rows(tag: &str, layer: usize, t: &Tensor, dim: usize, b: usize) {
    if !batch_rowdiff_enabled()
        || b < 2
        || ROWDIFF_TRIPPED.load(std::sync::atomic::Ordering::Relaxed)
    {
        return;
    }
    let mut hs = Vec::with_capacity(b);
    for i in 0..b {
        match t
            .narrow(dim, i, 1)
            .map_err(anyhow::Error::from)
            .and_then(|r| rowdiff_hash(&r))
        {
            Ok(h) => hs.push(h),
            Err(e) => {
                eprintln!("[ROWDIFF] tag={tag} layer={layer} row={i} ERR {e}");
                return;
            }
        }
    }
    if hs.iter().any(|h| *h != hs[0]) {
        eprintln!("[ROWDIFF] FIRST-DIVERGENCE tag={tag} layer={layer} rows={hs:?}");
        ROWDIFF_TRIPPED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn rowdiff_slot(tag: &str, layer: usize, slot: usize, t: &Tensor) {
    if !batch_rowdiff_enabled() || ROWDIFF_TRIPPED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashMap<String, (usize, u64)>>> =
        std::sync::OnceLock::new();
    let m = SEEN.get_or_init(Default::default);
    let h = match rowdiff_hash(t) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[ROWDIFF] tag={tag} layer={layer} slot={slot} ERR {e}");
            return;
        }
    };
    let key = format!("{tag}@{layer}");
    let mut g = m.lock().unwrap();
    if slot == 0 {
        g.insert(key, h);
        return;
    }
    if let Some(&h0) = g.get(&key) {
        if h0 != h {
            eprintln!(
                "[ROWDIFF] FIRST-DIVERGENCE tag={tag} layer={layer} slot0={h0:?} slot{slot}={h:?}"
            );
            ROWDIFF_TRIPPED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn det_hash_tensor(tag: &str, layer: usize, t: &Tensor) {
    let go = || -> Result<(usize, u64)> {
        let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in &v {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Ok((v.len(), h))
    };
    match go() {
        Ok((n, h)) => eprintln!("[NV_DEBUG_DET_SUB] layer={layer} tag={tag} n={n} hash={h:016x}"),
        Err(e) => eprintln!("[NV_DEBUG_DET_SUB] layer={layer} tag={tag} ERR {e}"),
    }
}

struct Gemma4HiddenCollector {
    wanted: Vec<usize>,
    captured: Vec<Option<Tensor>>,
}

impl Gemma4HiddenCollector {
    fn new(layers: &[usize]) -> Self {
        Self {
            wanted: layers.to_vec(),
            captured: vec![None; layers.len()],
        }
    }

    fn into_hidden(self, layers: &[usize], seq: usize, hidden: usize) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(layers.len());
        for (slot, &li) in self.captured.into_iter().zip(layers.iter()) {
            let t = slot.ok_or_else(|| {
                anyhow::anyhow!("Gemma4HiddenCollector: layer {} was never captured", li)
            })?;
            let dims = t.dims();
            if dims != [1, seq, hidden] {
                bail!(
                    "Gemma4HiddenCollector: layer {} hidden has dims {:?}, expected [1,{seq},{hidden}]",
                    li,
                    dims
                );
            }
            out.push(t);
        }
        Ok(out)
    }
}

impl Gemma4LayerHook for Gemma4HiddenCollector {
    fn after_layer(&mut self, layer_idx: usize, hidden: &Tensor) -> Result<()> {
        for (slot, &li) in self.captured.iter_mut().zip(self.wanted.iter()) {
            if li == layer_idx && slot.is_none() {
                *slot = Some(hidden.clone());
            }
        }
        Ok(())
    }
}

pub trait Gemma4Cache {
    fn current_len(&self) -> usize;
    fn advance(&mut self, n: usize);

    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()>;

    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()>;
    fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)>;

    fn try_decode_attention_fp8(
        &mut self,
        _layer: usize,
        _q_rot: &Tensor,
        _n_q: usize,
        _sliding_window: Option<usize>,
        _scaling: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn try_prefill_attention_fp8(
        &mut self,
        _layer: usize,
        _q_rot: &Tensor,
        _n_q: usize,
        _seq: usize,
        _sliding_window: Option<usize>,
        _scaling: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn try_decode_attention_ring(
        &mut self,
        _layer: usize,
        _q_rot: &Tensor,
        _n_q: usize,
        _sliding_window: Option<usize>,
        _scaling: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn try_decode_attention_gqa(
        &mut self,
        _layer: usize,
        _q_rot: &Tensor,
        _n_q: usize,
        _sliding_window: Option<usize>,
        _scaling: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

impl<T: Gemma4Cache + ?Sized> Gemma4Cache for Box<T> {
    fn current_len(&self) -> usize {
        (**self).current_len()
    }
    fn advance(&mut self, n: usize) {
        (**self).advance(n)
    }
    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        (**self).prepare_for_decode(write_pos, n_total)
    }
    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        (**self).write_at(layer, k_new, v_new)
    }
    fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        (**self).view(layer, len)
    }
    fn try_decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        (**self).try_decode_attention_fp8(layer, q_rot, n_q, sliding_window, scaling)
    }
    fn try_prefill_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        seq: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        (**self).try_prefill_attention_fp8(layer, q_rot, n_q, seq, sliding_window, scaling)
    }
    fn try_decode_attention_ring(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        (**self).try_decode_attention_ring(layer, q_rot, n_q, sliding_window, scaling)
    }
    fn try_decode_attention_gqa(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        (**self).try_decode_attention_gqa(layer, q_rot, n_q, sliding_window, scaling)
    }
}

const SLIDING_COMPACT_SLACK: usize = 256;

pub fn tree_layer_window(kind: LayerType, sliding_window: usize) -> i32 {
    match kind {
        LayerType::SlidingAttention => sliding_window as i32,
        LayerType::FullAttention => 0,
    }
}

pub fn tree_window_attends(qpos: i64, kpos: i64, window: i64) -> bool {
    if kpos > qpos {
        return false;
    }
    if window <= 0 {
        return true;
    }
    qpos - kpos < window
}

pub fn check_kv_window(what: &str, write_pos: usize, t: usize, max_seq_len: usize) -> Result<()> {
    let end = write_pos
        .checked_add(t)
        .ok_or_else(|| anyhow::anyhow!("{what}: write_pos {write_pos} + {t} overflows usize"))?;
    if end > max_seq_len {
        bail!(
            "{what}: write of {t} token(s) at position {write_pos} ends at {end}, past max_seq_len {max_seq_len}"
        );
    }
    Ok(())
}

pub struct Gemma4KvCache {
    layers: Vec<(Tensor, Tensor)>,
    layer_shapes: Vec<(usize, usize)>,

    layer_caps: Vec<usize>,

    layer_windows: Vec<Option<usize>>,

    layer_stored: Vec<usize>,
    current_len: usize,

    pending_write_pos: usize,
    max_seq_len: usize,
    device: Device,
    dtype: DType,
}

impl Gemma4KvCache {
    pub fn new(
        config: &Gemma4Config,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_shapes = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_caps = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_windows = Vec::with_capacity(config.num_hidden_layers);
        for kind in &config.layer_types {
            let head_dim = config.head_dim_for(*kind);
            let n_kv = config.num_kv_heads_for(*kind);

            let (cap, window) = match kind {
                LayerType::FullAttention => (max_seq_len, None),
                LayerType::SlidingAttention => {
                    let w = config.sliding_window.max(1);
                    (max_seq_len.min(w + SLIDING_COMPACT_SLACK), Some(w))
                }
            };
            let shape = (1usize, cap, n_kv, head_dim);
            let k = Tensor::zeros(shape, dtype, device)?;
            let v = Tensor::zeros(shape, dtype, device)?;
            layers.push((k, v));
            layer_shapes.push((n_kv, head_dim));
            layer_caps.push(cap);
            layer_windows.push(window);
        }
        let layer_stored = vec![0usize; layers.len()];
        Ok(Self {
            layers,
            layer_shapes,
            layer_caps,
            layer_windows,
            layer_stored,
            current_len: 0,
            pending_write_pos: 0,
            max_seq_len,
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
        for s in self.layer_stored.iter_mut() {
            *s = 0;
        }
    }
    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let start = self.pending_write_pos;
        let (n_kv, head_dim) = self.layer_shapes[layer];
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != n_kv || dims[3] != head_dim {
            bail!(
                "Gemma4KvCache.write_at layer {layer}: expected [1, t, {n_kv}, {head_dim}], got {:?}",
                dims
            );
        }
        if v_new.dims() != dims {
            bail!(
                "Gemma4KvCache.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        let cap = self.layer_caps[layer];

        let start = if let Some(window) = self.layer_windows[layer] {
            let mut stored = self.layer_stored[layer];
            if stored + t > cap {
                let keep = stored.min(window);
                let src_start = stored - keep;
                if src_start > 0 && keep > 0 {
                    let (k_buf, v_buf) = &self.layers[layer];
                    let k_keep = k_buf.narrow(1, src_start, keep)?.copy()?;
                    let v_keep = v_buf.narrow(1, src_start, keep)?.copy()?;
                    k_buf.slice_set(&k_keep, 1, 0)?;
                    v_buf.slice_set(&v_keep, 1, 0)?;
                }
                stored = keep;
                self.layer_stored[layer] = stored;
            }
            if stored + t > cap {
                bail!(
                    "Gemma4KvCache.write_at: sliding layer {layer} write of {t} tokens exceeds \
                     capacity {cap} (window+slack); prompts longer than the window need chunked \
                     prefill"
                );
            }
            stored
        } else {
            let end = start + t;
            if end > self.max_seq_len {
                bail!(
                    "Gemma4KvCache.write_at: end {} exceeds max_seq_len {}",
                    end,
                    self.max_seq_len
                );
            }
            start
        };
        let end = start + t;
        let (k_buf, v_buf) = &self.layers[layer];
        let k_src = k_new.contiguous()?;
        let v_src = v_new.contiguous()?;
        k_buf.slice_set(&k_src, 1, start)?;
        v_buf.slice_set(&v_src, 1, start)?;
        self.layer_stored[layer] = end;
        Ok(())
    }

    pub fn view(&self, layer: usize, _len: usize) -> Result<(Tensor, Tensor)> {
        if layer >= self.layers.len() {
            bail!("Gemma4KvCache.view: layer {layer} out of range");
        }

        let stored = self.layer_stored[layer];
        let (k, v) = &self.layers[layer];
        let k = k.narrow(1, 0, stored)?;
        let v = v.narrow(1, 0, stored)?;
        Ok((k, v))
    }
}

impl Gemma4Cache for Gemma4KvCache {
    fn current_len(&self) -> usize {
        Gemma4KvCache::current_len(self)
    }
    fn advance(&mut self, n: usize) {
        Gemma4KvCache::advance(self, n)
    }
    fn prepare_for_decode(&mut self, write_pos: usize, _n_total: usize) -> Result<()> {
        self.pending_write_pos = write_pos;
        Ok(())
    }
    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        Gemma4KvCache::write_at(self, layer, k_new, v_new)
    }
    fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        Gemma4KvCache::view(self, layer, len)
    }
}

#[cfg(feature = "cuda")]
impl Gemma4Cache for Gemma4KvCacheFp8 {
    fn current_len(&self) -> usize {
        Gemma4KvCacheFp8::current_len(self)
    }
    fn advance(&mut self, n: usize) {
        Gemma4KvCacheFp8::advance(self, n)
    }
    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        Gemma4KvCacheFp8::prepare_for_decode_dev(self, write_pos, n_total)
    }
    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        Gemma4KvCacheFp8::write_at(self, layer, k_new, v_new)
    }
    fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        Gemma4KvCacheFp8::view_bf16(self, layer, len)
    }
    fn try_decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        Gemma4KvCacheFp8::decode_attention_fp8(self, layer, q_rot, n_q, sliding_window, scaling)
            .map(Some)
    }
}

#[cfg(feature = "cuda")]
pub struct Gemma4KvCacheFp8 {
    layers: Vec<Gemma4KvCacheFp8Layer>,
    layer_shapes: Vec<(usize, usize)>,

    layer_rings: Vec<usize>,
    layer_windows: Vec<usize>,
    current_len: usize,
    max_seq_len: usize,
    device: Device,

    #[cfg(feature = "cuda")]
    write_pos_dev: CudaSlice<i32>,
    #[cfg(feature = "cuda")]
    n_total_dev: CudaSlice<i32>,
    #[cfg(feature = "cuda")]
    host_write_pos: Box<[i32; 1]>,
    #[cfg(feature = "cuda")]
    host_n_total: Box<[i32; 1]>,

    #[cfg(feature = "cuda")]
    flash_scratch: CudaSlice<f32>,
    #[cfg(feature = "cuda")]
    flash_fan_in: CudaSlice<u32>,
}

#[cfg(feature = "cuda")]
struct Gemma4KvCacheFp8Layer {
    k_fp8: CudaSlice<u8>,
    v_fp8: CudaSlice<u8>,
    k_scales: CudaSlice<f32>,
    v_scales: CudaSlice<f32>,
}

pub fn kv_fp8_ring_slots(sliding_window: usize) -> usize {
    sliding_window + VERIFY_PREFILL_CHUNK + VERIFY_RING_HEADROOM
}

#[derive(Clone, Copy, Debug)]
pub struct KvBudget {
    pub verify_full_bytes: usize,
    pub verify_sliding_bytes: usize,
    pub verify_scratch_bytes: usize,
    pub decode_full_bytes: usize,
    pub decode_sliding_bytes: usize,
    pub drafter_kv_bytes: usize,
    pub ring_slots: usize,
}

impl KvBudget {
    pub fn verify_total(&self) -> usize {
        self.verify_full_bytes + self.verify_sliding_bytes + self.verify_scratch_bytes
    }
    pub fn decode_total(&self) -> usize {
        self.decode_full_bytes + self.decode_sliding_bytes
    }

    pub fn worst_total(&self) -> usize {
        self.verify_total() + self.decode_total() + self.drafter_kv_bytes
    }
}

pub fn kv_budget(
    config: &Gemma4Config,
    kv_max: usize,
    verify_fp8: bool,
    rings_on: bool,
    drafter_kv_elems_per_row: usize,
) -> KvBudget {
    kv_budget_capped(
        config,
        kv_max,
        verify_fp8,
        rings_on,
        drafter_kv_elems_per_row,
        kv_max,
    )
}

pub fn kv_budget_capped(
    config: &Gemma4Config,
    kv_max: usize,
    verify_fp8: bool,
    rings_on: bool,
    drafter_kv_elems_per_row: usize,
    drafter_kv_rows: usize,
) -> KvBudget {
    let ring_slots = config.sliding_window + VERIFY_PREFILL_CHUNK + 128;
    let mut verify_full = 0usize;
    let mut verify_sliding = 0usize;
    let mut decode_full = 0usize;
    let mut decode_sliding = 0usize;
    let mut max_stride = 0usize;
    let mut max_nkv = 0usize;
    for kind in &config.layer_types {
        let nkv = config.num_kv_heads_for(*kind);
        let hd = config.head_dim_for(*kind);
        let stride = nkv * hd;
        max_stride = max_stride.max(stride);
        max_nkv = max_nkv.max(nkv);
        let sliding = matches!(kind, LayerType::SlidingAttention);

        let vslots = if sliding && verify_fp8 && rings_on {
            ring_slots.min(kv_max)
        } else {
            kv_max
        };
        let v_bytes_per_row = if verify_fp8 {
            2 * stride + 2 * nkv * 4
        } else {
            2 * stride * 2
        };
        if sliding {
            verify_sliding += vslots * v_bytes_per_row;
        } else {
            verify_full += vslots * v_bytes_per_row;
        }

        let dslots = if sliding && rings_on {
            ring_slots.min(kv_max)
        } else {
            kv_max
        };
        let d_bytes_per_row = 2 * stride + 2 * nkv * 4;
        if sliding {
            decode_sliding += dslots * d_bytes_per_row;
        } else {
            decode_full += dslots * d_bytes_per_row;
        }
    }
    let scratch_rows = kv_max.min(4096);
    let verify_scratch = if verify_fp8 {
        2 * scratch_rows * max_stride + 2 * scratch_rows * max_nkv * 4
    } else {
        2 * scratch_rows * max_stride * 2
    } + scratch_rows * 4;

    let drafter = drafter_kv_rows * drafter_kv_elems_per_row * 2 * 2;
    KvBudget {
        verify_full_bytes: verify_full,
        verify_sliding_bytes: verify_sliding,
        verify_scratch_bytes: verify_scratch,
        decode_full_bytes: decode_full,
        decode_sliding_bytes: decode_sliding,
        drafter_kv_bytes: drafter,
        ring_slots,
    }
}

#[cfg(feature = "cuda")]
impl Gemma4KvCacheFp8 {
    pub fn new(config: &Gemma4Config, max_seq_len: usize, device: &Device) -> Result<Self> {
        Self::build(config, max_seq_len, device, false)
    }

    pub fn new_windowed(
        config: &Gemma4Config,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::build(config, max_seq_len, device, true)
    }

    fn build(
        config: &Gemma4Config,
        max_seq_len: usize,
        device: &Device,
        windowed: bool,
    ) -> Result<Self> {
        let Device::Cuda(dev) = device.clone() else {
            bail!("Gemma4KvCacheFp8 requires a CUDA device")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_shapes = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_rings = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_windows = Vec::with_capacity(config.num_hidden_layers);
        for kind in &config.layer_types {
            let head_dim = config.head_dim_for(*kind);
            let n_kv = config.num_kv_heads_for(*kind);
            let (slots, ring, window) = if windowed
                && kv_ring_enabled()
                && matches!(kind, LayerType::SlidingAttention)
                && max_seq_len > kv_fp8_ring_slots(config.sliding_window)
            {
                let r = kv_fp8_ring_slots(config.sliding_window);
                (r, r, config.sliding_window)
            } else {
                (max_seq_len, 0, 0)
            };
            layer_rings.push(ring);
            layer_windows.push(window);
            let elem_count = slots * n_kv * head_dim;
            let scale_count = slots * n_kv;
            let k_fp8 = stream.alloc_zeros::<u8>(elem_count)?;
            let v_fp8 = stream.alloc_zeros::<u8>(elem_count)?;
            let k_scales = stream.alloc_zeros::<f32>(scale_count)?;
            let v_scales = stream.alloc_zeros::<f32>(scale_count)?;
            layers.push(Gemma4KvCacheFp8Layer {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            });
            layer_shapes.push((n_kv, head_dim));
        }
        let write_pos_dev = stream.alloc_zeros::<i32>(1)?;
        let n_total_dev = stream.alloc_zeros::<i32>(1)?;

        let n_q = config.num_attention_heads;
        let hd_max = layer_shapes.iter().map(|&(_, hd)| hd).max().unwrap_or(0);
        let scratch_elems = nv_kernels::cuda::flash_splitk_scratch_elems(n_q as i32, hd_max as i32);
        ensure!(
            scratch_elems > 0,
            "Gemma4KvCacheFp8: bad flash scratch size for n_q={n_q} hd_max={hd_max}"
        );
        let flash_scratch = stream.alloc_zeros::<f32>(scratch_elems as usize)?;
        let flash_fan_in = stream.alloc_zeros::<u32>(n_q.max(1))?;

        Ok(Self {
            layers,
            layer_shapes,
            layer_rings,
            layer_windows,
            current_len: 0,
            max_seq_len,
            device: device.clone(),
            write_pos_dev,
            n_total_dev,
            host_write_pos: Box::new([0i32; 1]),
            host_n_total: Box::new([0i32; 1]),
            flash_scratch,
            flash_fan_in,
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
    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn layer_shape(&self, layer: usize) -> (usize, usize) {
        self.layer_shapes[layer]
    }

    pub fn set_pending_pos_host_only(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        check_kv_window(
            "Gemma4KvCacheFp8.set_pending_pos",
            write_pos,
            1,
            self.max_seq_len,
        )?;
        if n_total > self.max_seq_len {
            bail!(
                "Gemma4KvCacheFp8.set_pending_pos: n_total {} exceeds max_seq_len {}",
                n_total,
                self.max_seq_len
            );
        }
        self.host_write_pos[0] = write_pos as i32;
        self.host_n_total[0] = n_total as i32;
        Ok(())
    }

    pub fn prepare_for_decode_dev(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        check_kv_window(
            "Gemma4KvCacheFp8.prepare_for_decode",
            write_pos,
            1,
            self.max_seq_len,
        )?;
        if n_total > self.max_seq_len {
            bail!(
                "Gemma4KvCacheFp8.prepare_for_decode: n_total {} exceeds max_seq_len {}",
                n_total,
                self.max_seq_len
            );
        }
        self.host_write_pos[0] = write_pos as i32;
        self.host_n_total[0] = n_total as i32;
        let Device::Cuda(dev) = self.device.clone() else {
            unreachable!()
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        stream
            .memcpy_htod(&self.host_write_pos[..], &mut self.write_pos_dev)
            .map_err(|e| anyhow::anyhow!("htod write_pos: {e:?}"))?;
        stream
            .memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev)
            .map_err(|e| anyhow::anyhow!("htod n_total: {e:?}"))?;
        Ok(())
    }

    pub fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;

        if layer >= self.layer_shapes.len() {
            bail!(
                "Gemma4KvCacheFp8.write_at: layer {layer} out of range ({} layers)",
                self.layer_shapes.len()
            );
        }
        let (n_kv, head_dim) = self.layer_shapes[layer];
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != n_kv || dims[3] != head_dim {
            bail!(
                "Gemma4KvCacheFp8.write_at layer {layer}: expected [1, t, {n_kv}, {head_dim}], got {:?}",
                dims
            );
        }
        if v_new.dims() != dims {
            bail!(
                "Gemma4KvCacheFp8.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        let start = usize::try_from(self.host_write_pos[0]).map_err(|_| {
            anyhow::anyhow!(
                "Gemma4KvCacheFp8.write_at: negative pending write_pos {}",
                self.host_write_pos[0]
            )
        })?;
        check_kv_window("Gemma4KvCacheFp8.write_at", start, t, self.max_seq_len)?;
        let ring = self.layer_rings[layer];
        if ring > 0 {
            let window = self.layer_windows[layer];
            ensure!(
                t <= ring - window + 1,
                "Gemma4KvCacheFp8.write_at: single append of {t} rows exceeds ring \
                 capacity {ring} for window {window}; chunk the prefill"
            );
        }
        let Device::Cuda(dev) = self.device.clone() else {
            unreachable!()
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let (start_dev_ptr, _gsp) = self.write_pos_dev.device_ptr(&stream);
        let layer_mut = &mut self.layers[layer];

        let (k_storage, kl) = k_new.storage_and_layout();
        let (v_storage, vl) = v_new.storage_and_layout();
        let k_cuda = match &*k_storage {
            Storage::Cuda(s) => s,
            _ => bail!("k_new must be on the CUDA device"),
        };
        let v_cuda = match &*v_storage {
            Storage::Cuda(s) => s,
            _ => bail!("v_new must be on the CUDA device"),
        };
        let k_slice = k_cuda.as_cuda_slice::<bf16>()?;
        let v_slice = v_cuda.as_cuda_slice::<bf16>()?;
        let k_view = k_slice.slice(kl.start_offset()..);
        let v_view = v_slice.slice(vl.start_offset()..);

        let (k_in_ptr, _gki) = k_view.device_ptr(&stream);
        let (v_in_ptr, _gvi) = v_view.device_ptr(&stream);
        let (k_fp8_base, _gkf) = layer_mut.k_fp8.device_ptr_mut(&stream);
        let (v_fp8_base, _gvf) = layer_mut.v_fp8.device_ptr_mut(&stream);
        let (k_sc_base, _gks) = layer_mut.k_scales.device_ptr_mut(&stream);
        let (v_sc_base, _gvs) = layer_mut.v_scales.device_ptr_mut(&stream);

        let s_raw = stream.cu_stream() as *mut c_void;
        let rc_k = unsafe {
            nv_kernels::cuda::quantize_kv_fp8(
                s_raw,
                k_in_ptr as *const u16,
                k_fp8_base as *mut u8,
                k_sc_base as *mut f32,
                start_dev_ptr as *const i32,
                t as i32,
                n_kv as i32,
                head_dim as i32,
                ring as i32,
            )
        };
        if rc_k != 0 {
            bail!("quantize_kv_fp8(k) rc={rc_k}");
        }
        let rc_v = unsafe {
            nv_kernels::cuda::quantize_kv_fp8(
                s_raw,
                v_in_ptr as *const u16,
                v_fp8_base as *mut u8,
                v_sc_base as *mut f32,
                start_dev_ptr as *const i32,
                t as i32,
                n_kv as i32,
                head_dim as i32,
                ring as i32,
            )
        };
        if rc_v != 0 {
            bail!("quantize_kv_fp8(v) rc={rc_v}");
        }
        Ok(())
    }

    pub fn view_bf16(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;

        let (n_kv, head_dim) = self.layer_shapes[layer];
        if len > self.max_seq_len {
            bail!(
                "Gemma4KvCacheFp8.view_bf16: len {len} > max_seq_len {}",
                self.max_seq_len
            );
        }

        let ring = self.layer_rings[layer];
        let (start, eff) = if ring > 0 && len > ring {
            (len - ring, ring)
        } else {
            (0usize, len)
        };
        let Device::Cuda(dev) = self.device.clone() else {
            unreachable!()
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let need = eff * n_kv * head_dim;

        let mut k_out = unsafe { stream.alloc::<bf16>(need)? };
        let mut v_out = unsafe { stream.alloc::<bf16>(need)? };

        let layer_ref = &self.layers[layer];
        {
            let (k_fp8_ptr, _gk) = layer_ref.k_fp8.device_ptr(&stream);
            let (v_fp8_ptr, _gv) = layer_ref.v_fp8.device_ptr(&stream);
            let (k_sc_ptr, _gks) = layer_ref.k_scales.device_ptr(&stream);
            let (v_sc_ptr, _gvs) = layer_ref.v_scales.device_ptr(&stream);
            let (k_out_ptr, _gko) = k_out.device_ptr_mut(&stream);
            let (v_out_ptr, _gvo) = v_out.device_ptr_mut(&stream);
            let s_raw = stream.cu_stream() as *mut c_void;
            let rc_k = unsafe {
                nv_kernels::cuda::dequantize_kv_fp8(
                    s_raw,
                    k_fp8_ptr as *const u8,
                    k_sc_ptr as *const f32,
                    k_out_ptr as *mut u16,
                    start as i32,
                    eff as i32,
                    n_kv as i32,
                    head_dim as i32,
                    ring as i32,
                )
            };
            if rc_k != 0 {
                bail!("dequantize_kv_fp8(k) rc={rc_k}");
            }
            let rc_v = unsafe {
                nv_kernels::cuda::dequantize_kv_fp8(
                    s_raw,
                    v_fp8_ptr as *const u8,
                    v_sc_ptr as *const f32,
                    v_out_ptr as *mut u16,
                    start as i32,
                    eff as i32,
                    n_kv as i32,
                    head_dim as i32,
                    ring as i32,
                )
            };
            if rc_v != 0 {
                bail!("dequantize_kv_fp8(v) rc={rc_v}");
            }
        }

        let k_storage = CudaStorage::wrap_cuda_slice(k_out, dev.clone());
        let v_storage = CudaStorage::wrap_cuda_slice(v_out, dev);
        let k = candle_core::Tensor::from_storage(
            Storage::Cuda(k_storage),
            (1usize, eff, n_kv, head_dim),
            BackpropOp::none(),
            false,
        );
        let v = candle_core::Tensor::from_storage(
            Storage::Cuda(v_storage),
            (1usize, eff, n_kv, head_dim),
            BackpropOp::none(),
            false,
        );
        Ok((k, v))
    }

    pub fn decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;

        let (n_kv, head_dim) = self.layer_shapes[layer];
        let dims = q_rot.dims();

        let expected = n_q * head_dim;
        let total: usize = dims.iter().product();
        if total != expected {
            bail!(
                "decode_attention_fp8 layer {layer}: expected total {expected}, got dims {:?}",
                dims
            );
        }
        let Device::Cuda(dev) = self.device.clone() else {
            unreachable!()
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let mut out = unsafe { stream.alloc::<bf16>(expected)? };

        let q_c = q_rot.contiguous()?;
        let (q_storage, _ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_storage {
            Storage::Cuda(s) => s,
            _ => bail!("q_rot must be on CUDA"),
        };
        let q_slice = q_cuda.as_cuda_slice::<bf16>()?;

        let (n_total_ptr, _gnt) = self.n_total_dev.device_ptr(&stream);
        let layer_ref = &self.layers[layer];

        let (q_ptr, _gq) = q_slice.device_ptr(&stream);
        let (k_ptr, _gk) = layer_ref.k_fp8.device_ptr(&stream);
        let (v_ptr, _gv) = layer_ref.v_fp8.device_ptr(&stream);
        let (ks_ptr, _gks) = layer_ref.k_scales.device_ptr(&stream);
        let (vs_ptr, _gvs) = layer_ref.v_scales.device_ptr(&stream);
        let (out_ptr, _go) = out.device_ptr_mut(&stream);
        let (scr_ptr, _gsc) = self.flash_scratch.device_ptr_mut(&stream);
        let (fan_ptr, _gfi) = self.flash_fan_in.device_ptr_mut(&stream);

        let sw_i32 = sliding_window.map(|w| w as i32).unwrap_or(0);
        let ring_i32 = self.layer_rings[layer] as i32;
        ensure!(
            ring_i32 == 0 || sw_i32 > 0,
            "decode_attention_fp8: ring layer {layer} requires a sliding window"
        );
        let s_raw = stream.cu_stream() as *mut c_void;

        let rc = unsafe {
            nv_kernels::cuda::flash_decode_fused_fp8kv(
                s_raw,
                q_ptr as *const u16,
                k_ptr as *const u8,
                v_ptr as *const u8,
                ks_ptr as *const f32,
                vs_ptr as *const f32,
                out_ptr as *mut u16,
                n_total_ptr as *const i32,
                scr_ptr as *mut f32,
                fan_ptr as *mut u32,
                n_q as i32,
                n_kv as i32,
                head_dim as i32,
                sw_i32,
                ring_i32,
                scaling,
            )
        };
        if rc != 0 {
            bail!("flash_decode_fused_fp8kv rc={rc}");
        }

        drop(_go);
        drop(_gsc);
        drop(_gfi);
        drop(_gq);
        drop(_gk);
        drop(_gv);
        drop(_gks);
        drop(_gvs);
        drop(_gnt);
        drop(q_storage);

        let storage = CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            Storage::Cuda(storage),
            (1usize, 1usize, n_q, head_dim),
            BackpropOp::none(),
            false,
        );
        Ok(tensor)
    }
}

impl Gemma4Layer {
    #[cfg(feature = "cuda")]
    fn from_loader(
        config: &Gemma4Config,
        idx: usize,
        kind: LayerType,
        weights: &WeightLoader,
        qconfig: Option<&QuantizationConfig>,
        nvfp4_runner: Option<Arc<Mutex<Nvfp4GemmRunner>>>,
        fp8_attn_runner: Option<Arc<Mutex<Fp8GemmRunner>>>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{idx}");
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let head_dim = config.head_dim_for(kind);
        let n_q = config.num_attention_heads;
        let n_kv = config.num_kv_heads_for(kind);
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;

        let input_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let post_attention_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let pre_feedforward_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let post_feedforward_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let layer_scalar = weights
            .get(&format!("{prefix}.layer_scalar"), dtype)
            .with_context(|| format!("load {prefix}.layer_scalar"))?;
        if layer_scalar.dims() != [1] {
            bail!(
                "{prefix}.layer_scalar expected [1], got {:?}",
                layer_scalar.dims()
            );
        }

        let layer_scalar_host = {
            let host: Vec<half::bf16> = layer_scalar
                .to_dtype(DType::BF16)?
                .flatten_all()?
                .to_vec1()?;
            host[0].to_f32()
        };

        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_proj = load_qkv_fused(weights, &prefix, q_dim, kv_dim, hidden, has_v, dtype)?;
        let o_proj = load_attn_proj_lean(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden,
            q_dim,
            dtype,
        )?;
        let (quant_qkv, quant_o) = attn_proj_quant_mode(qconfig);
        let attn_scheme = attn_proj_scheme();
        let qkv_proj = if quant_qkv {
            match attn_scheme {
                AttnProjScheme::Fp8E4m3 => {
                    quantize_attn_proj_fp8(qkv_proj, fp8_attn_runner.as_ref(), device)
                        .with_context(|| format!("fp8 attn quant {prefix}.self_attn.qkv"))?
                }
                AttnProjScheme::Nvfp4 => {
                    quantize_attn_proj_nvfp4(qkv_proj, nvfp4_runner.as_ref(), device)
                        .with_context(|| format!("nvfp4 attn quant {prefix}.self_attn.qkv"))?
                }
            }
        } else {
            qkv_proj
        };
        let o_proj = if quant_o {
            match attn_scheme {
                AttnProjScheme::Fp8E4m3 => {
                    quantize_attn_proj_fp8(o_proj, fp8_attn_runner.as_ref(), device)
                        .with_context(|| format!("fp8 attn quant {prefix}.self_attn.o_proj"))?
                }
                AttnProjScheme::Nvfp4 => {
                    quantize_attn_proj_nvfp4(o_proj, nvfp4_runner.as_ref(), device)
                        .with_context(|| format!("nvfp4 attn quant {prefix}.self_attn.o_proj"))?
                }
            }
        } else {
            o_proj
        };
        let qkv_prefill_fp4 = prefill_fp4_copy(&qkv_proj, nvfp4_runner.as_ref(), device)
            .with_context(|| format!("nvfp4 prefill quant {prefix}.self_attn.qkv"))?;
        let o_prefill_fp4 = prefill_fp4_copy(&o_proj, nvfp4_runner.as_ref(), device)
            .with_context(|| format!("nvfp4 prefill quant {prefix}.self_attn.o_proj"))?;
        let q_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            head_dim,
            eps,
            dtype,
        )?;
        let k_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            head_dim,
            eps,
            dtype,
        )?;
        let v_norm = build_v_norm_no_scale(head_dim, eps, dtype, device)?;
        let self_attn = Gemma4Attention {
            kind,
            qkv_proj,
            q_dim,
            kv_dim,
            has_v,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            qkv_prefill_fp4,
            o_prefill_fp4,
        };

        let gate_up_proj = load_mlp_proj_fused_pair(
            weights,
            &format!("{prefix}.mlp.gate_proj"),
            &format!("{prefix}.mlp.up_proj"),
            inter,
            hidden,
            dtype,
            qconfig,
            nvfp4_runner.clone(),
            device,
        )?;
        let down_proj = load_mlp_proj(
            weights,
            &format!("{prefix}.mlp.down_proj"),
            hidden,
            inter,
            dtype,
            qconfig,
            nvfp4_runner,
            device,
        )?;
        let mlp = Gemma4Mlp {
            gate_up_proj,
            down_proj,
        };

        Ok(Self {
            kind,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
            layer_scalar_host,
            self_attn,
            mlp,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn from_loader(
        config: &Gemma4Config,
        idx: usize,
        kind: LayerType,
        weights: &WeightLoader,
        _device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{idx}");
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let head_dim = config.head_dim_for(kind);
        let n_q = config.num_attention_heads;
        let n_kv = config.num_kv_heads_for(kind);
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let input_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let post_attention_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let pre_feedforward_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let post_feedforward_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            hidden,
            eps,
            dtype,
        )?;
        let layer_scalar = weights.get(&format!("{prefix}.layer_scalar"), dtype)?;
        let layer_scalar_host = {
            let host: Vec<half::bf16> = layer_scalar
                .to_dtype(DType::BF16)?
                .flatten_all()?
                .to_vec1()?;
            host[0].to_f32()
        };
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_proj = load_qkv_fused(weights, &prefix, q_dim, kv_dim, hidden, has_v, dtype)?;
        let o_proj = load_attn_proj_lean(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden,
            q_dim,
            dtype,
        )?;
        let q_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            head_dim,
            eps,
            dtype,
        )?;
        let k_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            head_dim,
            eps,
            dtype,
        )?;
        let v_norm = build_v_norm_no_scale(head_dim, eps, dtype, _device)?;
        let self_attn = Gemma4Attention {
            kind,
            qkv_proj,
            q_dim,
            kv_dim,
            has_v,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
        };

        let gate_w = weights.get(&format!("{prefix}.mlp.gate_proj.weight"), dtype)?;
        let up_w = weights.get(&format!("{prefix}.mlp.up_proj.weight"), dtype)?;
        let fused = candle_core::Tensor::cat(&[&gate_w, &up_w], 0)?.contiguous()?;
        let gate_up_proj = Linear::new(fused, None)?;
        let down_proj = load_attn_proj(
            weights,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden,
            inter,
            dtype,
        )?;
        let mlp = Gemma4Mlp {
            gate_up_proj,
            down_proj,
        };
        Ok(Self {
            kind,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
            layer_scalar_host,
            self_attn,
            mlp,
        })
    }
}

impl CausalLm for Gemma4 {
    fn forward(&mut self, _tokens: &[u32], _positions: &[u32]) -> Result<Vec<f32>> {
        bail!("Gemma4 CausalLm shim not wired; call Gemma4::forward(&Tensor, &Tensor) directly")
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

fn build_v_norm_no_scale(dim: usize, eps: f64, dtype: DType, device: &Device) -> Result<RmsNorm> {
    let ones = Tensor::ones(dim, dtype, device)?;
    Ok(RmsNorm::new(ones, eps))
}

fn build_sliding_rope(config: &Gemma4Config, device: &Device) -> Result<Rope> {
    let head_dim = config.head_dim;
    let base = config.rope_theta_for(LayerType::SlidingAttention);
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((i as f32 * 2.0) / (head_dim as f32)))
        .collect();
    Rope::from_inv_freq(
        RopeConfig {
            head_dim,
            max_seq_len: config.max_position_embeddings,
            base,
            kind: RopeKind::Standard,
        },
        &inv_freq,
        device,
    )
}

fn build_full_rope(config: &Gemma4Config, device: &Device) -> Result<Rope> {
    let head_dim = config.global_head_dim;
    let base = config.rope_theta_for(LayerType::FullAttention);
    let half = head_dim / 2;
    let partial = config.rope_partial_factor_for(LayerType::FullAttention);
    let rope_angles = ((partial * head_dim as f32 / 2.0) as usize).min(half);
    let mut inv_freq = vec![0f32; half];
    for (i, f) in inv_freq[..rope_angles].iter_mut().enumerate() {
        *f = 1.0 / base.powf((i as f32 * 2.0) / (head_dim as f32));
    }
    Rope::from_inv_freq(
        RopeConfig {
            head_dim,
            max_seq_len: config.max_position_embeddings,
            base,
            kind: RopeKind::Standard,
        },
        &inv_freq,
        device,
    )
}

#[cfg(feature = "cuda")]
pub enum VerifyKvStore {
    Fp8 {
        k: Vec<CudaSlice<u8>>,
        v: Vec<CudaSlice<u8>>,
        k_scale: Vec<CudaSlice<f32>>,
        v_scale: Vec<CudaSlice<f32>>,
        scratch_k: CudaSlice<u8>,
        scratch_v: CudaSlice<u8>,
        scratch_ks: CudaSlice<f32>,
        scratch_vs: CudaSlice<f32>,
    },
}

#[cfg(feature = "cuda")]
pub struct Gemma4VerifyCache {
    store: VerifyKvStore,
    n_committed: CudaSlice<i32>,

    row_stride: Vec<usize>,
    head_dims: Vec<usize>,
    ring: Vec<usize>,

    scratch_rows: usize,
    path_dev: CudaSlice<i32>,
    max_seq: usize,
    device: candle_core::CudaDevice,
    mk_scratch: Option<CudaSlice<f32>>,
    mk_fan_in: Option<CudaSlice<u32>>,
    gqa512_scratch: Option<CudaSlice<f32>>,
    prefill_shadow: HashMap<usize, (Tensor, Tensor)>,
    prefill_shadow_disabled: bool,
}

#[cfg(feature = "cuda")]
pub fn mk_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mk_verify_gate(
            env::var("NV_EAGLE3_MK_VERIFY").ok().as_deref(),
            env::var("NV_EAGLE3_TREE").ok().as_deref(),
        )
    })
}

pub fn mk_verify_gate(mk_raw: Option<&str>, tree_raw: Option<&str>) -> bool {
    mk_raw == Some("1") && tree_raw.is_none()
}

pub fn mk_verify_hd512_gate(tree_raw: Option<&str>) -> bool {
    tree_raw.is_none()
}

#[cfg(feature = "cuda")]
fn mk_verify_hd512_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mk_verify_hd512_gate(env::var("NV_EAGLE3_TREE").ok().as_deref()))
}

pub fn gqa512_verify_geometry(n_q: usize, n_kv_full: usize, hd_full: usize) -> bool {
    hd_full == 512 && n_kv_full > 0 && n_q == 8 * n_kv_full
}

#[cfg(feature = "cuda")]
fn gqa512_verify_scratch_elems_for(config: &Gemma4Config) -> Option<usize> {
    if !mk_verify_hd512_enabled() {
        return None;
    }
    let n_q = config.num_attention_heads;
    let n_kv = config.num_kv_heads_for(LayerType::FullAttention);
    let hd = config.head_dim_for(LayerType::FullAttention);
    if !gqa512_verify_geometry(n_q, n_kv, hd) {
        return None;
    }
    Some(nv_kernels::cuda::gqa512_scratch_elems(
        n_q as i32,
        8,
        GQA512_VERIFY_SPLITS,
    ))
}

#[cfg(feature = "cuda")]
pub fn gqa512_verify_scratch_bytes(config: &Gemma4Config) -> usize {
    gqa512_verify_scratch_elems_for(config)
        .map(|e| e * std::mem::size_of::<f32>())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
const GQA512_VERIFY_SPLITS: i32 = 64;

#[cfg(feature = "cuda")]
fn gqa512_dispatch_note(site: &str, idx: usize, k: usize, n_q: usize, n_kv: usize, hd: usize) {
    static ONCE: Once = Once::new();
    if !nv_config::env_arm_present_at_all_reread_every_call("NV_PROF_CHAT") {
        return;
    }
    ONCE.call_once(|| {
        eprintln!(
            "[NV_PROF_CHAT][hd512] gqa512_verify_fp8 dispatched site={site} layer={idx} k={k} \
             n_q={n_q} n_kv={n_kv} hd={hd} splits={GQA512_VERIFY_SPLITS}"
        );
    });
}

pub fn verify_mask_is_chain(mask: &[u8], k: usize) -> bool {
    k > 0
        && mask.len() == k * k
        && (0..k).all(|i| (0..k).all(|j| (mask[i * k + j] != 0) == (j <= i)))
}

pub fn lm_head_i8_rows_per_call_gate(kernel_ceiling_m: usize) -> usize {
    kernel_ceiling_m.max(1)
}

#[cfg(feature = "cuda")]
fn lm_head_i8_rows_per_call(hidden: usize) -> usize {
    lm_head_i8_rows_per_call_gate(
        nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize,
    )
}

pub const LMHEAD_I8_DECODE_ARM_DEFAULT_ON: bool = false;

pub const LMHEAD_I8_DECODE_ARM_IS_DEFAULT_OFF_BECAUSE_I8_LOGITS_ARE_NOT_BIT_IDENTICAL_TO_THE_BF16_HEAD_AND_EVERY_PUBLISHED_GEMMA4_DECODE_NUMBER_WAS_MEASURED_ON_THE_BF16_HEAD: &str =
    "NV_G4_LMHEAD_I8 arms the resident int8 lm_head copy on the m=1 decode epilogue as well as \
     on the Gemma4BatchGraphFamily step it already armed. The copy itself is built by default \
     (unless NV_VERIFY_LMHEAD_INT8=0) and forward_verify_dev already reads it, so this arm moves \
     no weights -- it moves which copy the decode epilogue reads: 2.819 GB of bf16 rows become \
     1.409 GB of i8 rows plus f32 row scales, and the separate final RMSNorm launch is folded \
     into the mk kernel through rstd_op. It stays default-off because the logits are not \
     bit-identical to the bf16 head.";

#[cfg(feature = "cuda")]
pub fn decode_lmhead_i8_arm_enabled_env_nv_g4_lmhead_i8_shared_with_the_batch_graph_step_arm(
) -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_G4_LMHEAD_I8")
}

#[cfg(feature = "cuda")]
pub(crate) fn current_stream_is_mid_graph_capture(dev: &candle_core::CudaDevice) -> bool {
    use cudarc::driver::sys as drv;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let mut st = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let rc = unsafe { drv::cuStreamIsCapturing(stream.cu_stream(), &mut st) };
    rc == drv::CUresult::CUDA_SUCCESS
        && st != drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
}

pub fn verify_qkv_fused_from(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub fn spec_prefill_flash_from(deterministic: bool) -> bool {
    !deterministic
}

pub const SHADOW_JIT_WARM_AT_CACHE_CREATION_BECAUSE_FIRST_USE_PTX_JIT_OF_THE_CANDLE_CAST_CMP_AFFINE_REDUCE_MODULES_MEASURED_5_9S_ON_THE_FIRST_PREFILL_SHADOW_CALL_WHEN_THE_DRIVER_JIT_CACHE_CANNOT_PERSIST_AND_MUST_NOT_LAND_ON_FIRST_DEEP_PROMPT_TTFT:
    usize = 2;

#[cfg(feature = "cuda")]
fn shadow_route_jit_warm_once_at_cache_creation(dev: &Device) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let s = SHADOW_JIT_WARM_AT_CACHE_CREATION_BECAUSE_FIRST_USE_PTX_JIT_OF_THE_CANDLE_CAST_CMP_AFFINE_REDUCE_MODULES_MEASURED_5_9S_ON_THE_FIRST_PREFILL_SHADOW_CALL_WHEN_THE_DRIVER_JIT_CACHE_CANNOT_PERSIST_AND_MUST_NOT_LAND_ON_FIRST_DEEP_PROMPT_TTFT;
        let warm = (|| -> Result<()> {
            let q = Tensor::zeros((1usize, s, s, 8), DType::BF16, dev)?;
            let kv = Tensor::zeros((1usize, s, 1, 8), DType::BF16, dev)?;
            causal_attention_chunked(&q, &kv, &kv, s, 1, 8, s, 0)?;
            Ok(())
        })();
        let _ = warm;
    });
}

#[cfg(feature = "cuda")]
fn verify_qkv_fused_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| verify_qkv_fused_from(env::var("NV_VERIFY_QKV_FUSED").ok().as_deref()))
}

#[cfg(feature = "cuda")]
fn verify_qkv_fused_layer_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        env::var("NV_VERIFY_QKV_FUSED_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

#[cfg(feature = "cuda")]
fn prefill_shadow_fallback_warn(reason: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[gemma4] spec prefill GEMM shadow unavailable ({reason}); \
             falling back to tree-verify kernel"
        )
    });
}

pub fn prefill_shadow_extend(
    prev: Option<(Tensor, Tensor)>,
    k_new: &Tensor,
    v_new: &Tensor,
    committed: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    if committed == 0 {
        return Ok(Some((k_new.clone(), v_new.clone())));
    }
    match prev {
        Some((pk, pv)) if pk.dims().len() == 4 && pk.dims()[1] == committed => {
            let ks = Tensor::cat(&[&pk, k_new], 1)?;
            let vs = Tensor::cat(&[&pv, v_new], 1)?;
            Ok(Some((ks, vs)))
        }
        _ => Ok(None),
    }
}

pub fn sliding_shadow_extend(
    prev: Option<(Tensor, Tensor)>,
    k_new: &Tensor,
    v_new: &Tensor,
    committed: usize,
    keep: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    let expect = committed.min(keep);
    if expect == 0 {
        return Ok(Some((k_new.clone(), v_new.clone())));
    }
    match prev {
        Some((pk, pv)) if pk.dims().len() == 4 && pk.dims()[1] == expect => {
            let ks = Tensor::cat(&[&pk, k_new], 1)?;
            let vs = Tensor::cat(&[&pv, v_new], 1)?;
            Ok(Some((ks, vs)))
        }
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
fn spec_prefill_flash_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        spec_prefill_flash_from(nv_quant::matmul::deterministic_mode())
    })
}

#[cfg(feature = "cuda")]
const VERIFY_COMPACT_MAX_ROWS: usize = 4096;

pub const VERIFY_PREFILL_CHUNK: usize = 1024;

const VERIFY_RING_HEADROOM: usize = 128;

#[cfg(feature = "cuda")]
pub fn kv_ring_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env::var("NV_KV_RING").as_deref() != Ok("0"))
}

#[cfg(feature = "cuda")]
fn verify_ring_slots(sliding_window: usize) -> Option<usize> {
    if !kv_ring_enabled() {
        return None;
    }
    Some(sliding_window + VERIFY_PREFILL_CHUNK + VERIFY_RING_HEADROOM)
}

#[cfg(feature = "cuda")]
impl Gemma4VerifyCache {
    pub fn n_committed_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.n_committed
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn end_prefill(&mut self) {
        self.prefill_shadow.clear();
        self.prefill_shadow_disabled = false;
    }

    pub fn compact_path(&mut self, path: &[usize], base: usize) -> Result<()> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let a = path.len();
        if a == 0 {
            return Ok(());
        }
        ensure!(
            a <= self.scratch_rows,
            "compact_path: {a} rows exceeds compact scratch capacity {}",
            self.scratch_rows
        );
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let path_i32: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        stream.memcpy_htod(&path_i32, &mut self.path_dev.slice_mut(0..a))?;
        let n_layers = self.row_stride.len();
        for idx in 0..n_layers {
            let stride = self.row_stride[idx];
            let ring = self.ring[idx];
            let rc = match &mut self.store {
                VerifyKvStore::Fp8 {
                    k,
                    v,
                    k_scale,
                    v_scale,
                    scratch_k,
                    scratch_v,
                    scratch_ks,
                    scratch_vs,
                } => {
                    let nkv = stride / self.head_dims[idx];
                    let hd = self.head_dims[idx];
                    let (pp, _a) = self.path_dev.device_ptr(&stream);
                    let (kp, _b) = k[idx].device_ptr_mut(&stream);
                    let (vp, _c) = v[idx].device_ptr_mut(&stream);
                    let (ksp, _f) = k_scale[idx].device_ptr_mut(&stream);
                    let (vsp, _g) = v_scale[idx].device_ptr_mut(&stream);
                    let (skp, _d) = scratch_k.device_ptr_mut(&stream);
                    let (svp, _e) = scratch_v.device_ptr_mut(&stream);
                    let (sksp, _h) = scratch_ks.device_ptr_mut(&stream);
                    let (svsp, _i) = scratch_vs.device_ptr_mut(&stream);
                    unsafe {
                        nv_kernels::cuda::kv_compact_fp8(
                            stream.cu_stream() as *mut _,
                            kp as *mut u8,
                            vp as *mut u8,
                            ksp as *mut f32,
                            vsp as *mut f32,
                            skp as *mut u8,
                            svp as *mut u8,
                            sksp as *mut f32,
                            svsp as *mut f32,
                            pp as *const i32,
                            base as i32,
                            a as i32,
                            nkv as i32,
                            hd as i32,
                            ring as i32,
                        )
                    }
                }
            };
            ensure!(rc == 0, "kv_compact returned {rc}");
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Gemma4 {
    pub fn new_verify_cache(&self, max_seq: usize) -> Result<Gemma4VerifyCache> {
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("verify cache requires cuda")
        };
        shadow_route_jit_warm_once_at_cache_creation(&self.device);
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut row_stride = Vec::with_capacity(self.layers.len());
        let mut head_dims = Vec::with_capacity(self.layers.len());
        let mut ring = Vec::with_capacity(self.layers.len());
        let mut slots = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let nkv = self.config.num_kv_heads_for(layer.kind);
            let hd = self.config.head_dim_for(layer.kind);
            row_stride.push(nkv * hd);
            head_dims.push(hd);

            let layer_slots = if matches!(layer.kind, LayerType::SlidingAttention) {
                match verify_ring_slots(self.config.sliding_window) {
                    Some(r) if max_seq > r => {
                        ring.push(r);
                        r
                    }
                    _ => {
                        ring.push(0);
                        max_seq
                    }
                }
            } else {
                ring.push(0);
                max_seq
            };
            slots.push(layer_slots);
        }
        let max_row_stride = row_stride.iter().copied().max().unwrap_or(0);
        let scratch_rows = max_seq.min(VERIFY_COMPACT_MAX_ROWS);
        let scratch_elems = scratch_rows * max_row_stride;
        let store = {
            let mut k = Vec::with_capacity(self.layers.len());
            let mut v = Vec::with_capacity(self.layers.len());
            let mut k_scale = Vec::with_capacity(self.layers.len());
            let mut v_scale = Vec::with_capacity(self.layers.len());
            for (i, stride) in row_stride.iter().enumerate() {
                let elems = slots[i] * stride;
                let scale_elems = slots[i] * (stride / head_dims[i]);
                k.push(stream.alloc_zeros::<u8>(elems)?);
                v.push(stream.alloc_zeros::<u8>(elems)?);
                k_scale.push(stream.alloc_zeros::<f32>(scale_elems)?);
                v_scale.push(stream.alloc_zeros::<f32>(scale_elems)?);
            }
            let max_nkv = row_stride
                .iter()
                .zip(head_dims.iter())
                .map(|(s, h)| s / h)
                .max()
                .unwrap_or(0);
            VerifyKvStore::Fp8 {
                k,
                v,
                k_scale,
                v_scale,
                scratch_k: stream.alloc_zeros::<u8>(scratch_elems)?,
                scratch_v: stream.alloc_zeros::<u8>(scratch_elems)?,
                scratch_ks: stream.alloc_zeros::<f32>(scratch_rows * max_nkv)?,
                scratch_vs: stream.alloc_zeros::<f32>(scratch_rows * max_nkv)?,
            }
        };
        let n_committed = stream.alloc_zeros::<i32>(1)?;
        let path_dev = stream.alloc_zeros::<i32>(scratch_rows)?;
        let (mk_scratch, mk_fan_in) = if mk_verify_enabled() {
            let n_q = self.config.num_attention_heads;
            let max_hd = head_dims.iter().copied().max().unwrap_or(0);
            let elems = n_q * 8 * 32 * (max_hd + 2);
            (
                Some(stream.alloc_zeros::<f32>(elems)?),
                Some(stream.alloc_zeros::<u32>(n_q)?),
            )
        } else {
            (None, None)
        };
        let gqa512_scratch = match gqa512_verify_scratch_elems_for(&self.config) {
            Some(elems) => Some(stream.alloc_zeros::<f32>(elems)?),
            None => None,
        };
        Ok(Gemma4VerifyCache {
            store,
            n_committed,
            row_stride,
            head_dims,
            ring,
            scratch_rows,
            path_dev,
            max_seq,
            device: dev,
            mk_scratch,
            mk_fan_in,
            gqa512_scratch,
            prefill_shadow: HashMap::new(),
            prefill_shadow_disabled: false,
        })
    }

    pub fn attention_tree(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        mask_dev: &CudaSlice<u8>,
        k: usize,
        cache: &mut Gemma4VerifyCache,
    ) -> Result<Tensor> {
        self.attention_tree_impl(idx, x, positions, mask_dev, k, cache, None)
    }

    fn attention_tree_impl(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        mask_dev: &CudaSlice<u8>,
        k: usize,
        cache: &mut Gemma4VerifyCache,
        prefill: Option<usize>,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let layer = &self.layers[idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let hd = self.config.head_dim_for(kind);
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_kv_heads_for(kind);
        let rope = match kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };

        let window: i32 = tree_layer_window(kind, self.config.sliding_window);

        let pos_flat = positions.flatten_all()?;
        ensure!(
            pos_flat.dtype() == candle_core::DType::I32,
            "attention_tree: positions must be i32 (got {:?})",
            pos_flat.dtype()
        );
        ensure!(
            pos_flat.dims() == [k],
            "attention_tree: positions must have {k} entries (got {:?})",
            pos_flat.dims()
        );
        let pos_c = pos_flat.contiguous()?;

        if prefill.is_none() && (!cache.prefill_shadow.is_empty() || cache.prefill_shadow_disabled)
        {
            cache.end_prefill();
        }

        if prefill.is_none() && verify_qkv_fused_enabled() && idx < verify_qkv_fused_layer_cap() {
            if let Some(out) = self.attention_tree_fused_impl(idx, x, &pos_c, mask_dev, k, cache)? {
                return Ok(out);
            }
        }

        let (q_raw, k_raw, v_raw) = attn.qkv_forward(x)?;
        let q = q_raw.reshape((1usize, k, n_q, hd))?;
        let q = attn.q_norm.forward(&q)?;
        let kk = k_raw.reshape((1usize, k, n_kv, hd))?;
        let kk = attn.k_norm.forward(&kk)?;
        let vv = v_raw.reshape((1usize, k, n_kv, hd))?;
        let vv = attn.v_norm.forward(&vv)?;

        let (q_rot, k_rot) = rope.apply(&q, &kk, positions)?;
        let (q_rot, k_rot) = crate::hadamard_kv::maybe_rotate_qk(q_rot, k_rot, hd)?;
        let q_rot = q_rot.reshape((k, n_q * hd))?.contiguous()?;
        let k_rot = k_rot.reshape((k, n_kv * hd))?.contiguous()?;
        let vv = vv.reshape((k, n_kv * hd))?.contiguous()?;
        let gemm_out: Option<Tensor> = match prefill {
            Some(committed)
                if matches!(kind, LayerType::FullAttention)
                    && !cache.prefill_shadow_disabled =>
            {
                let attempt = (|| -> Result<Option<Tensor>> {
                    let k4 = k_rot.reshape((1usize, k, n_kv, hd))?;
                    let v4 = vv.reshape((1usize, k, n_kv, hd))?;
                    let prev = cache.prefill_shadow.remove(&idx);
                    let Some((ks, vs)) = prefill_shadow_extend(prev, &k4, &v4, committed)? else {
                        return Ok(None);
                    };
                    let q4 = q_rot.reshape((1usize, k, n_q, hd))?;
                    let out = if spec_prefill_flash_enabled() {
                        flash_attention(&q4, &ks, &vs, n_q, n_kv, hd, k, None)?
                    } else {
                        causal_attention_chunked(&q4, &ks, &vs, n_q, n_kv, hd, k, committed)?
                    };
                    cache.prefill_shadow.insert(idx, (ks, vs));
                    Ok(Some(out))
                })();
                match attempt {
                    Ok(Some(out)) => Some(out),
                    Ok(None) => {
                        prefill_shadow_fallback_warn("chunk offset mismatch");
                        None
                    }
                    Err(e) => {
                        cache.prefill_shadow_disabled = true;
                        cache.prefill_shadow.clear();
                        prefill_shadow_fallback_warn(&format!("{e}"));
                        None
                    }
                }
            }
            Some(committed)
                if matches!(kind, LayerType::SlidingAttention)
                    && spec_prefill_flash_enabled()
                    && !cache.prefill_shadow_disabled =>
            {
                let attempt = (|| -> Result<Option<Tensor>> {
                    let keep = self.config.sliding_window.saturating_sub(1);
                    let k4 = k_rot.reshape((1usize, k, n_kv, hd))?;
                    let v4 = vv.reshape((1usize, k, n_kv, hd))?;
                    let prev = cache.prefill_shadow.remove(&idx);
                    let Some((ks, vs)) = sliding_shadow_extend(prev, &k4, &v4, committed, keep)?
                    else {
                        return Ok(None);
                    };
                    let q4 = q_rot.reshape((1usize, k, n_q, hd))?;
                    let out = flash_attention(&q4, &ks, &vs, n_q, n_kv, hd, k, Some(keep))?;
                    let rows = ks.dims()[1];
                    let tail = rows.min(keep);
                    if tail > 0 {
                        let ks_t = ks.narrow(1, rows - tail, tail)?.contiguous()?;
                        let vs_t = vs.narrow(1, rows - tail, tail)?.contiguous()?;
                        cache.prefill_shadow.insert(idx, (ks_t, vs_t));
                    }
                    Ok(Some(out))
                })();
                match attempt {
                    Ok(Some(out)) => Some(out),
                    Ok(None) => {
                        prefill_shadow_fallback_warn("sliding chunk offset mismatch");
                        None
                    }
                    Err(e) => {
                        cache.prefill_shadow_disabled = true;
                        cache.prefill_shadow.clear();
                        prefill_shadow_fallback_warn(&format!("{e}"));
                        None
                    }
                }
            }
            _ => None,
        };
        let use_gemm = gemm_out.is_some();

        let Device::Cuda(dev) = self.device.clone() else {
            bail!("cuda")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let out_elems = if use_gemm { 1 } else { k * n_q * hd };
        let mut out_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(out_elems)? };

        let rc = {
            let (qs, _ql) = q_rot.storage_and_layout();
            let (ks, _kl) = k_rot.storage_and_layout();
            let (vs, _vl) = vv.storage_and_layout();
            let (ps, _pl) = pos_c.storage_and_layout();
            let qc = match &*qs {
                Storage::Cuda(s) => s,
                _ => bail!("q not cuda"),
            };
            let kc = match &*ks {
                Storage::Cuda(s) => s,
                _ => bail!("k not cuda"),
            };
            let vc = match &*vs {
                Storage::Cuda(s) => s,
                _ => bail!("v not cuda"),
            };
            let pc = match &*ps {
                Storage::Cuda(s) => s,
                _ => bail!("positions not cuda"),
            };
            let qsl = qc.as_cuda_slice::<bf16>()?;
            let ksl = kc.as_cuda_slice::<bf16>()?;
            let vsl = vc.as_cuda_slice::<bf16>()?;
            let psl = pc.as_cuda_slice::<i32>()?;
            let (qp, _a) = qsl.device_ptr(&stream);
            let (knp, _b) = ksl.device_ptr(&stream);
            let (vnp, _c) = vsl.device_ptr(&stream);
            let (pp, _i) = psl.device_ptr(&stream);
            let (ncp, _f) = cache.n_committed.device_ptr(&stream);
            let (mp, _g) = mask_dev.device_ptr(&stream);
            let (op, _h) = out_dev.device_ptr_mut(&stream);
            let ring = cache.ring[idx] as i32;
            if ring > 0 {
                ensure!(
                    (k as i32) <= ring - window + 1,
                    "attention_tree: single append of {k} rows exceeds ring capacity \
                     {ring} for window {window}; prefill must be chunked to at most \
                     {VERIFY_PREFILL_CHUNK} rows"
                );
            }
            match &mut cache.store {
                VerifyKvStore::Fp8 {
                    k: kf,
                    v: vf,
                    k_scale,
                    v_scale,
                    ..
                } => {
                    let (kcp, _d) = kf[idx].device_ptr_mut(&stream);
                    let (vcp, _e) = vf[idx].device_ptr_mut(&stream);
                    let (ksp, _j) = k_scale[idx].device_ptr_mut(&stream);
                    let (vsp, _l) = v_scale[idx].device_ptr_mut(&stream);
                    let rc1 = unsafe {
                        nv_kernels::cuda::kv_append_fp8(
                            stream.cu_stream() as *mut _,
                            knp as *const u16,
                            vnp as *const u16,
                            kcp as *mut u8,
                            vcp as *mut u8,
                            ksp as *mut f32,
                            vsp as *mut f32,
                            ncp as *const i32,
                            k as i32,
                            n_kv as i32,
                            hd as i32,
                            ring,
                        )
                    };
                    let use_mk = cache.mk_scratch.is_some()
                        && k <= 8
                        && hd <= 512
                        && window == 0
                        && ring == 0;
                    let use_gqa512 = cache.gqa512_scratch.is_some()
                        && k <= 8
                        && hd == 512
                        && window == 0
                        && ring == 0
                        && n_q == 8 * n_kv;
                    if rc1 != 0 {
                        rc1
                    } else if use_gemm {
                        0
                    } else if use_gqa512 {
                        gqa512_dispatch_note("tree", idx, k, n_q, n_kv, hd);
                        let gs = cache.gqa512_scratch.as_mut().unwrap();
                        let (gsp, _g0) = gs.device_ptr_mut(&stream);
                        unsafe {
                            nv_kernels::cuda::gqa512_verify_fp8(
                                stream.cu_stream() as *mut _,
                                qp as *const u16,
                                kcp as *const u8,
                                vcp as *const u8,
                                ksp as *const f32,
                                vsp as *const f32,
                                op as *mut u16,
                                ncp as *const i32,
                                -(k as i32),
                                k as i32,
                                gsp as *mut f32,
                                n_q as i32,
                                n_kv as i32,
                                hd as i32,
                                GQA512_VERIFY_SPLITS,
                                1.0f32,
                            )
                        }
                    } else if use_mk {
                        let ms = cache.mk_scratch.as_mut().unwrap();
                        let mf = cache.mk_fan_in.as_mut().unwrap();
                        let (msp, _m0) = ms.device_ptr_mut(&stream);
                        let (mfp, _m1) = mf.device_ptr_mut(&stream);
                        unsafe {
                            nv_kernels::cuda::flash_decode_fused_fp8kv_mk(
                                stream.cu_stream() as *mut _,
                                qp as *const u16,
                                kcp as *const u8,
                                vcp as *const u8,
                                ksp as *const f32,
                                vsp as *const f32,
                                op as *mut u16,
                                ncp as *const i32,
                                -(k as i32),
                                k as i32,
                                msp as *mut f32,
                                mfp as *mut u32,
                                n_q as i32,
                                n_kv as i32,
                                hd as i32,
                                window,
                                ring,
                                1.0f32,
                            )
                        }
                    } else {
                        unsafe {
                            nv_kernels::cuda::tree_verify_attn_fp8(
                                stream.cu_stream() as *mut _,
                                qp as *const u16,
                                kcp as *const u8,
                                vcp as *const u8,
                                ksp as *const f32,
                                vsp as *const f32,
                                ncp as *const i32,
                                mp as *const u8,
                                pp as *const i32,
                                op as *mut u16,
                                n_q as i32,
                                n_kv as i32,
                                hd as i32,
                                k as i32,
                                window,
                                ring,
                            )
                        }
                    }
                }
            }
        };
        ensure!(rc == 0, "tree attn kernels returned {rc}");

        if let Some(out) = gemm_out {
            let attn_out = out.reshape((k, n_q * hd))?.contiguous()?;
            return attn.o_forward(&attn_out);
        }

        let storage = CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
        let storage = Storage::Cuda(storage);
        let attn_out = Tensor::from_storage(storage, (k, n_q * hd), BackpropOp::none(), false);
        attn.o_forward(&attn_out)
    }

    fn attention_tree_fused_impl(
        &self,
        idx: usize,
        x: &Tensor,
        pos_c: &Tensor,
        mask_dev: &CudaSlice<u8>,
        k: usize,
        cache: &mut Gemma4VerifyCache,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let layer = &self.layers[idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let hd = self.config.head_dim_for(kind);
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_kv_heads_for(kind);
        if hd == 0 || !hd.is_multiple_of(2) || hd > 512 {
            return Ok(None);
        }
        if attn.q_dim != n_q * hd || attn.kv_dim != n_kv * hd {
            return Ok(None);
        }
        let eps = attn.q_norm.eps();
        if attn.k_norm.eps() != eps || attn.v_norm.eps() != eps {
            return Ok(None);
        }
        let qkv_bf16 = matches!(attn.qkv_proj.kind(), nv_quant::LinearKind::Bf16);
        if qkv_bf16 && !nv_quant::matmul::fused_qkv_bitwise_safe(k, attn.has_v) {
            return Ok(None);
        }
        let rope = match kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        if rope.cos().dtype() != DType::F32 || rope.sin().dtype() != DType::F32 {
            return Ok(None);
        }
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let window: i32 = tree_layer_window(kind, self.config.sliding_window);
        let ring = cache.ring[idx] as i32;
        if ring > 0 {
            ensure!(
                (k as i32) <= ring - window + 1,
                "attention_tree: single append of {k} rows exceeds ring capacity \
                 {ring} for window {window}"
            );
        }

        let fused = if qkv_bf16 {
            attn.qkv_proj.forward_dense_det(x)?
        } else {
            attn.qkv_proj.forward(x)?
        };
        if fused.dtype() != DType::BF16 {
            return Ok(None);
        }
        let fdims = fused.dims().to_vec();
        let width = *fdims.last().unwrap();
        let rows: usize = fdims[..fdims.len() - 1].iter().product();
        let expect_w = attn.q_dim + attn.kv_dim * if attn.has_v { 2 } else { 1 };
        if rows != k || width != expect_w {
            return Ok(None);
        }
        let fused_c = fused.contiguous()?;
        let q_off = 0i64;
        let k_off = attn.q_dim as i64;
        let v_off = if attn.has_v {
            (attn.q_dim + attn.kv_dim) as i64
        } else {
            attn.q_dim as i64
        };
        let cos_c = rope.cos().contiguous()?;
        let sin_c = rope.sin().contiguous()?;

        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut q_out: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(k * n_q * hd)? };
        let mut out_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(k * n_q * hd)? };

        let use_mk = cache.mk_scratch.is_some() && k <= 8 && hd <= 512 && window == 0 && ring == 0;
        let rc = {
            let (fs, fl) = fused_c.storage_and_layout();
            let (cs, _cl) = cos_c.storage_and_layout();
            let (ss, _sl) = sin_c.storage_and_layout();
            let (ps, _pl) = pos_c.storage_and_layout();
            let qw_t = attn.q_norm.weight_bf16();
            let kw_t = attn.k_norm.weight_bf16();
            let vw_t = attn.v_norm.weight_bf16();
            let (qws, _qwl) = qw_t.storage_and_layout();
            let (kws, _kwl) = kw_t.storage_and_layout();
            let (vws, _vwl) = vw_t.storage_and_layout();
            let fc = match &*fs {
                Storage::Cuda(s) => s,
                _ => bail!("qkv not cuda"),
            };
            let cc = match &*cs {
                Storage::Cuda(s) => s,
                _ => bail!("cos not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("sin not cuda"),
            };
            let pc = match &*ps {
                Storage::Cuda(s) => s,
                _ => bail!("positions not cuda"),
            };
            let qwc = match &*qws {
                Storage::Cuda(s) => s,
                _ => bail!("q_norm weight not cuda"),
            };
            let kwc = match &*kws {
                Storage::Cuda(s) => s,
                _ => bail!("k_norm weight not cuda"),
            };
            let vwc = match &*vws {
                Storage::Cuda(s) => s,
                _ => bail!("v_norm weight not cuda"),
            };
            let fsl = fc.as_cuda_slice::<bf16>()?;
            let f_view = fsl.slice(fl.start_offset()..);
            let csl = cc.as_cuda_slice::<f32>()?;
            let ssl = sc.as_cuda_slice::<f32>()?;
            let psl = pc.as_cuda_slice::<i32>()?;
            let qwsl = qwc.as_cuda_slice::<bf16>()?;
            let kwsl = kwc.as_cuda_slice::<bf16>()?;
            let vwsl = vwc.as_cuda_slice::<bf16>()?;
            let (fp, _gf) = f_view.device_ptr(&stream);
            let (cp, _gc) = csl.device_ptr(&stream);
            let (sp, _gs) = ssl.device_ptr(&stream);
            let (pp, _gp) = psl.device_ptr(&stream);
            let (qwp, _gqw) = qwsl.device_ptr(&stream);
            let (kwp, _gkw) = kwsl.device_ptr(&stream);
            let (vwp, _gvw) = vwsl.device_ptr(&stream);
            let (ncp, _gnc) = cache.n_committed.device_ptr(&stream);
            let (mp, _gm) = mask_dev.device_ptr(&stream);
            let (qop, _gqo) = q_out.device_ptr_mut(&stream);
            let (op, _go) = out_dev.device_ptr_mut(&stream);
            match &mut cache.store {
                VerifyKvStore::Fp8 {
                    k: kf,
                    v: vf,
                    k_scale,
                    v_scale,
                    ..
                } => {
                    let (kcp, _d) = kf[idx].device_ptr_mut(&stream);
                    let (vcp, _e) = vf[idx].device_ptr_mut(&stream);
                    let (ksp, _j) = k_scale[idx].device_ptr_mut(&stream);
                    let (vsp, _l) = v_scale[idx].device_ptr_mut(&stream);
                    let rc1 = unsafe {
                        nv_kernels::cuda::verify_qkv_prep(
                            stream.cu_stream() as *mut _,
                            fp as *const u16,
                            width as i64,
                            q_off,
                            k_off,
                            v_off,
                            qwp as *const u16,
                            kwp as *const u16,
                            vwp as *const u16,
                            eps as f32,
                            cp as *const f32,
                            sp as *const f32,
                            pp as *const i32,
                            qop as *mut u16,
                            kcp as *mut u8,
                            vcp as *mut u8,
                            ksp as *mut f32,
                            vsp as *mut f32,
                            ncp as *const i32,
                            k as i32,
                            n_q as i32,
                            n_kv as i32,
                            hd as i32,
                            ring,
                        )
                    };
                    let use_gqa512 = cache.gqa512_scratch.is_some()
                        && k <= 8
                        && hd == 512
                        && window == 0
                        && ring == 0
                        && n_q == 8 * n_kv;
                    if rc1 != 0 {
                        rc1
                    } else if use_gqa512 {
                        gqa512_dispatch_note("fused", idx, k, n_q, n_kv, hd);
                        let gs = cache.gqa512_scratch.as_mut().unwrap();
                        let (gsp, _g0) = gs.device_ptr_mut(&stream);
                        unsafe {
                            nv_kernels::cuda::gqa512_verify_fp8(
                                stream.cu_stream() as *mut _,
                                qop as *const u16,
                                kcp as *const u8,
                                vcp as *const u8,
                                ksp as *const f32,
                                vsp as *const f32,
                                op as *mut u16,
                                ncp as *const i32,
                                -(k as i32),
                                k as i32,
                                gsp as *mut f32,
                                n_q as i32,
                                n_kv as i32,
                                hd as i32,
                                GQA512_VERIFY_SPLITS,
                                1.0f32,
                            )
                        }
                    } else if use_mk {
                        let ms = cache.mk_scratch.as_mut().unwrap();
                        let mf = cache.mk_fan_in.as_mut().unwrap();
                        let (msp, _m0) = ms.device_ptr_mut(&stream);
                        let (mfp, _m1) = mf.device_ptr_mut(&stream);
                        unsafe {
                            nv_kernels::cuda::flash_decode_fused_fp8kv_mk(
                                stream.cu_stream() as *mut _,
                                qop as *const u16,
                                kcp as *const u8,
                                vcp as *const u8,
                                ksp as *const f32,
                                vsp as *const f32,
                                op as *mut u16,
                                ncp as *const i32,
                                -(k as i32),
                                k as i32,
                                msp as *mut f32,
                                mfp as *mut u32,
                                n_q as i32,
                                n_kv as i32,
                                hd as i32,
                                window,
                                ring,
                                1.0f32,
                            )
                        }
                    } else {
                        unsafe {
                            nv_kernels::cuda::tree_verify_attn_fp8(
                                stream.cu_stream() as *mut _,
                                qop as *const u16,
                                kcp as *const u8,
                                vcp as *const u8,
                                ksp as *const f32,
                                vsp as *const f32,
                                ncp as *const i32,
                                mp as *const u8,
                                pp as *const i32,
                                op as *mut u16,
                                n_q as i32,
                                n_kv as i32,
                                hd as i32,
                                k as i32,
                                window,
                                ring,
                            )
                        }
                    }
                }
            }
        };
        ensure!(rc == 0, "verify_qkv_prep/attention kernels returned {rc}");

        let storage = CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
        let storage = Storage::Cuda(storage);
        let attn_out = Tensor::from_storage(storage, (k, n_q * hd), BackpropOp::none(), false);
        Ok(Some(attn.o_proj.forward(&attn_out)?))
    }

    pub fn forward_verify(
        &self,
        ids: &[u32],
        positions: &[i32],
        mask_dev: &CudaSlice<u8>,
        committed: usize,
        aux_layers: &[usize],
        cache: &mut Gemma4VerifyCache,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let k = ids.len();
        let (logits, aux) =
            self.forward_verify_tail(ids, positions, mask_dev, committed, aux_layers, cache, k)?;
        Ok((logits.expect("logit_rows == k always yields logits"), aux))
    }

    pub fn forward_verify_tail(
        &self,
        ids: &[u32],
        positions: &[i32],
        mask_dev: &CudaSlice<u8>,
        committed: usize,
        aux_layers: &[usize],
        cache: &mut Gemma4VerifyCache,
        logit_rows: usize,
    ) -> Result<(Option<Tensor>, Vec<Tensor>)> {
        let k = ids.len();
        ensure!(positions.len() == k, "positions len mismatch");
        ensure!(committed + k <= cache.max_seq, "verify cache overflow");
        let prefill = if logit_rows < k
            && positions.first().copied() == Some(committed as i32)
            && positions.windows(2).all(|w| w[1] == w[0] + 1)
        {
            Some(committed)
        } else {
            None
        };
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("cuda")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        stream.memcpy_htod(&[committed as i32], &mut cache.n_committed)?;
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, k), &self.device)?;
        let pos_t = Tensor::from_vec(positions.to_vec(), (1usize, k), &self.device)?;
        self.forward_verify_dev_rows(
            &tokens, &pos_t, mask_dev, k, aux_layers, cache, logit_rows, prefill,
        )
    }

    pub fn forward_verify_dev(
        &self,
        ids_t: &Tensor,
        pos_t: &Tensor,
        mask_dev: &CudaSlice<u8>,
        k: usize,
        aux_layers: &[usize],
        cache: &mut Gemma4VerifyCache,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (logits, aux) =
            self.forward_verify_dev_rows(ids_t, pos_t, mask_dev, k, aux_layers, cache, k, None)?;
        Ok((logits.expect("logit_rows == k always yields logits"), aux))
    }

    fn forward_verify_dev_rows(
        &self,
        ids_t: &Tensor,
        pos_t: &Tensor,
        mask_dev: &CudaSlice<u8>,
        k: usize,
        aux_layers: &[usize],
        cache: &mut Gemma4VerifyCache,
        logit_rows: usize,
        prefill: Option<usize>,
    ) -> Result<(Option<Tensor>, Vec<Tensor>)> {
        ensure!(logit_rows <= k, "logit_rows {logit_rows} > rows {k}");

        nv_layers::linear::with_dense_bf16(|| {
            let hidden_size = self.config.hidden_size;
            let tokens_flat = ids_t.flatten_all()?.to_dtype(DType::U32)?;
            let x_flat = embed_lookup_bf16_op(&self.embed_weight, &tokens_flat, &self.device)?;
            let x = x_flat
                .reshape((1usize, k, hidden_size))?
                .to_dtype(self.dtype)?;
            let mut hidden = scale_bf16_op(&x, self.embed_scale, &self.device)?;

            let mut aux: Vec<Tensor> = Vec::new();
            for idx in 0..self.layers.len() {
                let layer = &self.layers[idx];
                let residual_attn = hidden.clone();
                let normed_pre_attn = layer.input_layernorm.forward(&hidden)?;
                let attn_out = self.attention_tree_impl(
                    idx,
                    &normed_pre_attn.reshape((k, hidden_size))?,
                    pos_t,
                    mask_dev,
                    k,
                    cache,
                    prefill,
                )?;
                let attn_out = attn_out.reshape((1usize, k, hidden_size))?;
                let (normed_pre_mlp, after_attn) = verify_post_attn_pre_ff(
                    &layer.post_attention_layernorm,
                    &layer.pre_feedforward_layernorm,
                    &attn_out,
                    &residual_attn,
                    &self.device,
                )?;
                let mlp_out = mlp_forward(&layer.mlp, &normed_pre_mlp)?;
                hidden = verify_post_ff_residual(
                    &layer.post_feedforward_layernorm,
                    &mlp_out,
                    &after_attn,
                    layer.layer_scalar_host,
                    &self.device,
                )?;
                if aux_layers.contains(&idx) {
                    aux.push(hidden.reshape((k, hidden_size))?);
                }
            }

            if logit_rows == 0 {
                return Ok((None, aux));
            }

            let rows = logit_rows;
            let hidden = if rows == k {
                hidden
            } else {
                hidden.narrow(1, k - rows, rows)?.contiguous()?
            };
            let raw = {
                #[cfg(feature = "cuda")]
                {
                    match &self.lm_head_i8 {
                        Some((wq, rs)) => {
                            let Device::Cuda(dev) = self.device.clone() else {
                                bail!("cuda")
                            };
                            let hflat = hidden.reshape((rows, hidden_size))?;
                            let rstd = crate::gemma4_e4b::rstd_op(
                                &hflat,
                                self.config.rms_norm_eps as f32,
                            )?;
                            let wn = self.final_norm.weight_bf16();

                            let max_m = lm_head_i8_rows_per_call(hidden_size);
                            if rows <= max_m {
                                nv_layers::lm_head_resident_quant::lm_head_i8_normed_mk_op(
                                    wq, rs, &hflat, wn, &rstd, rows, &dev,
                                )?
                            } else {
                                ensure!(
                                    !current_stream_is_mid_graph_capture(&dev),
                                    "lm_head_i8 rows={rows} exceeds the mk single-call ceiling \
                                     {max_m} and the chunked fallback concatenates via candle \
                                     Tensor::cat, which launches on the device stream: a forked \
                                     verify capture cannot record it and every replay would read \
                                     the freed cat output as CUDA_ERROR_ILLEGAL_ADDRESS. Lower \
                                     spec k to <= {max_m} or set NV_VERIFY_LMHEAD_INT8=0"
                                );
                                let mut chunks: Vec<Tensor> = Vec::new();
                                let mut off = 0usize;
                                while off < rows {
                                    let m = (rows - off).min(max_m);
                                    let xc = hflat.narrow(0, off, m)?.contiguous()?;
                                    let rc = rstd.narrow(0, off, m)?.contiguous()?;
                                    chunks.push(
                                        nv_layers::lm_head_resident_quant::lm_head_i8_normed_mk_op(
                                            wq, rs, &xc, wn, &rc, m, &dev,
                                        )?,
                                    );
                                    off += m;
                                }
                                Tensor::cat(&chunks, 0)?
                            }
                        }
                        None => self
                            .lm_head
                            .forward_dense(&self.final_norm.forward(&hidden)?)?,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.lm_head
                        .forward_dense(&self.final_norm.forward(&hidden)?)?
                }
            };
            let logits = tanh_softcap_bf16_to_f32_op(
                &raw,
                self.config.final_logit_softcapping,
                &self.device,
            )?;
            let logits = logits.reshape((rows, self.config.vocab_size))?;
            Ok((Some(logits), aux))
        })
    }
}

#[cfg(feature = "cuda")]
const DECODE_NORM_FUSED_ARM_DEFAULT_OFF_BECAUSE_EVERY_PUBLISHED_GEMMA4_DECODE_NUMBER_WAS_MEASURED_ON_THE_THREE_LAUNCH_NORM_CHAIN:
    bool = false;

#[cfg(feature = "cuda")]
fn decode_norm_fused_enabled_env_nv_g4_decode_norm_fused_rides_the_post_norm_on_the_residual_norm_dispatch(
) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        DECODE_NORM_FUSED_ARM_DEFAULT_OFF_BECAUSE_EVERY_PUBLISHED_GEMMA4_DECODE_NUMBER_WAS_MEASURED_ON_THE_THREE_LAUNCH_NORM_CHAIN
            || env::var("NV_G4_DECODE_NORM_FUSED").as_deref() == Ok("1")
    })
}

#[cfg(feature = "cuda")]
fn decode_post_attn_pre_ff(
    post_ln: &RmsNorm,
    pre_ln: &RmsNorm,
    attn_out: &Tensor,
    residual: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if decode_norm_fused_enabled_env_nv_g4_decode_norm_fused_rides_the_post_norm_on_the_residual_norm_dispatch()
        && attn_out.dtype() == DType::BF16
        && residual.dtype() == DType::BF16
        && attn_out.dims() == residual.dims()
        && post_ln.eps() == pre_ln.eps()
        && device.is_cuda()
    {
        if let Some(out) = rmsnorm2_residual_op(post_ln, pre_ln, attn_out, residual, device)? {
            return Ok(out);
        }
    }
    let attn_post = post_ln.forward(attn_out)?;
    pre_ln.forward_residual(&attn_post, residual)
}

#[cfg(not(feature = "cuda"))]
fn decode_post_attn_pre_ff(
    post_ln: &RmsNorm,
    pre_ln: &RmsNorm,
    attn_out: &Tensor,
    residual: &Tensor,
    _device: &Device,
) -> Result<(Tensor, Tensor)> {
    let attn_post = post_ln.forward(attn_out)?;
    pre_ln.forward_residual(&attn_post, residual)
}

#[cfg(feature = "cuda")]
fn decode_post_ff_residual(
    post_ln: &RmsNorm,
    mlp_out: &Tensor,
    after_attn: &Tensor,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    if decode_norm_fused_enabled_env_nv_g4_decode_norm_fused_rides_the_post_norm_on_the_residual_norm_dispatch()
        && mlp_out.dtype() == DType::BF16
        && after_attn.dtype() == DType::BF16
        && mlp_out.dims() == after_attn.dims()
        && device.is_cuda()
    {
        if let Some(t) = rmsnorm_residual_scale_op(post_ln, mlp_out, after_attn, scale, device)? {
            return Ok(t);
        }
    }
    let mlp_post = post_ln.forward(mlp_out)?;
    residual_add_scale_bf16_op(after_attn, &mlp_post, scale, device)
}

#[cfg(not(feature = "cuda"))]
fn decode_post_ff_residual(
    post_ln: &RmsNorm,
    mlp_out: &Tensor,
    after_attn: &Tensor,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let mlp_post = post_ln.forward(mlp_out)?;
    residual_add_scale_bf16_op(after_attn, &mlp_post, scale, device)
}

#[cfg(feature = "cuda")]
pub(crate) fn verify_post_attn_pre_ff(
    post_ln: &RmsNorm,
    pre_ln: &RmsNorm,
    attn_out: &Tensor,
    residual: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if attn_out.dtype() == DType::BF16
        && residual.dtype() == DType::BF16
        && attn_out.dims() == residual.dims()
        && post_ln.eps() == pre_ln.eps()
        && device.is_cuda()
    {
        if let Some(out) = rmsnorm2_residual_op(post_ln, pre_ln, attn_out, residual, device)? {
            return Ok(out);
        }
    }
    let attn_post = post_ln.forward(attn_out)?;
    pre_ln.forward_residual(&attn_post, residual)
}

#[cfg(feature = "cuda")]
fn rmsnorm2_residual_op(
    post_ln: &RmsNorm,
    pre_ln: &RmsNorm,
    x: &Tensor,
    residual: &Tensor,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let Device::Cuda(dev) = device.clone() else {
        return Ok(None);
    };
    let x_c = x.contiguous()?;
    let res_c = residual.contiguous()?;
    let dims = x_c.dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let w1 = post_ln.weight_bf16();
    let w2 = pre_ln.weight_bf16();
    if w1.elem_count() != hidden || w2.elem_count() != hidden {
        return Ok(None);
    }
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut sum_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(batch * hidden)? };
    let mut normed_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(batch * hidden)? };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (rs, rl) = res_c.storage_and_layout();
        let (w1s, _w1l) = w1.storage_and_layout();
        let (w2s, _w2l) = w2.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("x not cuda"),
        };
        let rc_ = match &*rs {
            Storage::Cuda(s) => s,
            _ => bail!("residual not cuda"),
        };
        let w1c = match &*w1s {
            Storage::Cuda(s) => s,
            _ => bail!("w1 not cuda"),
        };
        let w2c = match &*w2s {
            Storage::Cuda(s) => s,
            _ => bail!("w2 not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let x_view = xsl.slice(xl.start_offset()..);
        let rsl = rc_.as_cuda_slice::<bf16>()?;
        let r_view = rsl.slice(rl.start_offset()..);
        let w1sl = w1c.as_cuda_slice::<bf16>()?;
        let w2sl = w2c.as_cuda_slice::<bf16>()?;
        let (px, _gx) = x_view.device_ptr(&stream);
        let (pr, _gr) = r_view.device_ptr(&stream);
        let (pw1, _gw1) = w1sl.device_ptr(&stream);
        let (pw2, _gw2) = w2sl.device_ptr(&stream);
        let (psum, _gsum) = sum_dev.device_ptr_mut(&stream);
        let (pnorm, _gnorm) = normed_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::rmsnorm2_residual_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pr as *const u16,
                pw1 as *const u16,
                pw2 as *const u16,
                psum as *mut u16,
                pnorm as *mut u16,
                batch,
                hidden,
                post_ln.eps() as f32,
            )
        }
    };
    ensure!(rc == 0, "rmsnorm2_residual_bf16 returned {rc}");
    let shape: Shape = dims.into();
    let sum_storage = CudaStorage::wrap_cuda_slice(sum_dev, dev.clone());
    let sum_t = Tensor::from_storage(
        Storage::Cuda(sum_storage),
        shape.clone(),
        BackpropOp::none(),
        false,
    );
    let norm_storage = CudaStorage::wrap_cuda_slice(normed_dev, dev);
    let norm_t = Tensor::from_storage(
        Storage::Cuda(norm_storage),
        shape,
        BackpropOp::none(),
        false,
    );
    Ok(Some((norm_t, sum_t)))
}

#[cfg(feature = "cuda")]
pub(crate) fn verify_post_ff_residual(
    post_ln: &RmsNorm,
    mlp_out: &Tensor,
    after_attn: &Tensor,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    if mlp_out.dtype() == DType::BF16
        && after_attn.dtype() == DType::BF16
        && mlp_out.dims() == after_attn.dims()
        && device.is_cuda()
    {
        if let Some(t) = rmsnorm_residual_scale_op(post_ln, mlp_out, after_attn, scale, device)? {
            return Ok(t);
        }
    }
    let mlp_post = post_ln.forward(mlp_out)?;
    residual_add_scale_bf16_op(after_attn, &mlp_post, scale, device)
}

#[cfg(feature = "cuda")]
fn rmsnorm_residual_scale_op(
    post_ln: &RmsNorm,
    x: &Tensor,
    residual: &Tensor,
    scale: f32,
    device: &Device,
) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let Device::Cuda(dev) = device.clone() else {
        return Ok(None);
    };
    let x_c = x.contiguous()?;
    let res_c = residual.contiguous()?;
    let dims = x_c.dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let w = post_ln.weight_bf16();
    if w.elem_count() != hidden {
        return Ok(None);
    }
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(batch * hidden)? };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (rs, rl) = res_c.storage_and_layout();
        let (ws, _wl) = w.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("x not cuda"),
        };
        let rc_ = match &*rs {
            Storage::Cuda(s) => s,
            _ => bail!("residual not cuda"),
        };
        let wc = match &*ws {
            Storage::Cuda(s) => s,
            _ => bail!("w not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let x_view = xsl.slice(xl.start_offset()..);
        let rsl = rc_.as_cuda_slice::<bf16>()?;
        let r_view = rsl.slice(rl.start_offset()..);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let (px, _gx) = x_view.device_ptr(&stream);
        let (pr, _gr) = r_view.device_ptr(&stream);
        let (pw, _gw) = wsl.device_ptr(&stream);
        let (po, _go) = out_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::rmsnorm_residual_scale_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                po as *mut u16,
                batch,
                hidden,
                post_ln.eps() as f32,
                scale,
            )
        }
    };
    ensure!(rc == 0, "rmsnorm_residual_scale_bf16 returned {rc}");
    let shape: Shape = dims.into();
    let storage = CudaStorage::wrap_cuda_slice(out_dev, dev);
    Ok(Some(Tensor::from_storage(
        Storage::Cuda(storage),
        shape,
        BackpropOp::none(),
        false,
    )))
}

pub fn mlp_forward(mlp: &Gemma4Mlp, x: &Tensor) -> Result<Tensor> {
    let fused = mlp.gate_up_proj.forward(x)?;

    #[cfg(feature = "cuda")]
    {
        if fused.dtype() == DType::BF16 && fused.device().is_cuda() {
            let act = gelu_tanh_mul_fused_cuda_bf16(&fused)?;
            return mlp.down_proj.forward(&act);
        }
    }
    let last = fused.dims().len() - 1;
    let total = fused.dims()[last];
    let inter = total / 2;
    let gate = fused.narrow(last, 0, inter)?;
    let up = fused.narrow(last, inter, inter)?;
    let act = gelu_tanh_mul(&gate, &up)?;
    mlp.down_proj.forward(&act)
}

fn gelu_tanh_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.dims() != up.dims() {
        bail!(
            "gelu_tanh_mul: gate dims {:?} != up dims {:?}",
            gate.dims(),
            up.dims()
        );
    }
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda()
        && gate.dtype() == DType::BF16
        && up.dtype() == DType::BF16
    {
        return gelu_tanh_mul_cuda_bf16(gate, up);
    }
    gelu_tanh_mul_candle(gate, up)
}

#[cfg(feature = "cuda")]
fn gelu_tanh_mul_cuda_bf16(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    use nv_kernels::cuda as nvk;

    let gate_c = gate.contiguous()?;
    let up_c = up.contiguous()?;
    let dims = gate_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let Device::Cuda(dev) = gate_c.device().clone() else {
        unreachable!()
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(n)?;

    let rc = {
        let (gs, _gl) = gate_c.storage_and_layout();
        let (us, _ul) = up_c.storage_and_layout();
        let g_cuda = match &*gs {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage"),
        };
        let u_cuda = match &*us {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage"),
        };
        let g_slice = g_cuda.as_cuda_slice::<bf16>()?;
        let u_slice = u_cuda.as_cuda_slice::<bf16>()?;
        let (pg, _) = g_slice.device_ptr(&stream);
        let (pu, _) = u_slice.device_ptr(&stream);
        let (py, _) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nvk::gelu_tanh_mul_bf16(
                stream.cu_stream() as *mut _,
                pg as *const u16,
                pu as *const u16,
                py as *mut u16,
                n,
            )
        }
    };
    if rc != 0 {
        bail!("gelu_tanh_mul_bf16 kernel returned {rc}");
    }
    let storage = CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = Storage::Cuda(storage);
    let shape: Shape = dims.into();
    Ok(candle_core::Tensor::from_storage(
        storage,
        shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn gelu_tanh_mul_fused_cuda_bf16(fused: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    use nv_kernels::cuda as nvk;

    let dims = fused.dims().to_vec();
    let last = dims.len() - 1;
    let two_inter = dims[last];
    if !two_inter.is_multiple_of(2) {
        bail!(
            "gelu_tanh_mul_fused_cuda_bf16: last dim must be even, got {:?}",
            dims
        );
    }
    let inter = two_inter / 2;
    let leading: usize = dims[..last].iter().product();
    let tot_pairs = leading * inter;

    let Device::Cuda(dev) = fused.device().clone() else {
        unreachable!()
    };
    let fused_c = fused.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(tot_pairs)? };
    let rc = {
        let (fs, _fl) = fused_c.storage_and_layout();
        let f_cuda = match &*fs {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for fused"),
        };
        let f_slice = f_cuda.as_cuda_slice::<bf16>()?;
        let (pf, _) = f_slice.device_ptr(&stream);
        let (py, _) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nvk::gelu_tanh_mul_fused_bf16(
                stream.cu_stream() as *mut _,
                pf as *const u16,
                py as *mut u16,
                inter as i32,
                tot_pairs,
            )
        }
    };
    if rc != 0 {
        bail!("gelu_tanh_mul_fused_bf16 kernel returned {rc}");
    }
    let mut out_dims = dims[..last].to_vec();
    out_dims.push(inter);
    let storage = CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = Storage::Cuda(storage);
    let shape: Shape = out_dims.into();
    Ok(candle_core::Tensor::from_storage(
        storage,
        shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn residual_add_scale_bf16_op(
    a: &Tensor,
    b: &Tensor,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    let Device::Cuda(dev) = device.clone() else {
        bail!("residual_add_scale_bf16_op requires CUDA")
    };
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let dims = a_c.dims().to_vec();
    if dims != b_c.dims() {
        bail!(
            "residual_add_scale_bf16_op: shape mismatch a={:?} b={:?}",
            dims,
            b_c.dims()
        );
    }
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(n)? };
    let rc = {
        let (a_st, _al) = a_c.storage_and_layout();
        let (b_st, _bl) = b_c.storage_and_layout();
        let a_cuda = match &*a_st {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for a"),
        };
        let b_cuda = match &*b_st {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for b"),
        };
        let a_slice = a_cuda.as_cuda_slice::<bf16>()?;
        let b_slice = b_cuda.as_cuda_slice::<bf16>()?;
        let (pa, _) = a_slice.device_ptr(&stream);
        let (pb, _) = b_slice.device_ptr(&stream);
        let (py, _) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::residual_add_scale_bf16(
                stream.cu_stream() as *mut _,
                pa as *const u16,
                pb as *const u16,
                py as *mut u16,
                scale,
                n,
            )
        }
    };
    if rc != 0 {
        bail!("residual_add_scale_bf16 kernel returned {rc}");
    }
    let storage = CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = Storage::Cuda(storage);
    let shape: Shape = dims.into();
    Ok(candle_core::Tensor::from_storage(
        storage,
        shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn residual_add_scale_bf16_op(
    a: &Tensor,
    b: &Tensor,
    scale: f32,
    _device: &Device,
) -> Result<Tensor> {
    let scale_t = Tensor::new(scale, a.device())?.to_dtype(a.dtype())?;
    Ok(a.add(b)?.broadcast_mul(&scale_t)?)
}

#[cfg(feature = "cuda")]
pub(crate) fn embed_lookup_bf16_op(
    embed_weight: &Tensor,
    tokens: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    let Device::Cuda(dev) = device.clone() else {
        bail!("embed_lookup_bf16_op requires CUDA")
    };
    let weight_dims = embed_weight.dims();
    if weight_dims.len() != 2 {
        bail!(
            "embed_lookup_bf16_op: embed_weight must be 2-D, got {:?}",
            weight_dims
        );
    }
    let vocab = weight_dims[0];
    let hidden = weight_dims[1];
    let tokens_c = tokens.flatten_all()?.contiguous()?;
    let n_tokens = tokens_c.dims()[0];
    if tokens_c.dtype() != DType::U32 {
        bail!(
            "embed_lookup_bf16_op: tokens dtype must be U32, got {:?}",
            tokens_c.dtype()
        );
    }
    let embed_c = embed_weight.contiguous()?;
    if embed_c.dtype() != DType::BF16 {
        bail!(
            "embed_lookup_bf16_op: embed_weight dtype must be BF16, got {:?}",
            embed_c.dtype()
        );
    }
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(n_tokens * hidden)? };
    let rc = {
        let (ts, _tl) = tokens_c.storage_and_layout();
        let (es, _el) = embed_c.storage_and_layout();
        let t_cuda = match &*ts {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for tokens"),
        };
        let e_cuda = match &*es {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for embed_weight"),
        };
        let t_slice = t_cuda.as_cuda_slice::<u32>()?;
        let e_slice = e_cuda.as_cuda_slice::<bf16>()?;
        let (pt, _) = t_slice.device_ptr(&stream);
        let (pe, _) = e_slice.device_ptr(&stream);
        let (py, _) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gather_rows_bf16(
                stream.cu_stream() as *mut _,
                pe as *const u16,
                pt as *const i32,
                py as *mut u16,
                n_tokens as i32,
                hidden as i32,
                vocab as i32,
            )
        }
    };
    if rc != 0 {
        bail!("gather_rows_bf16 kernel returned {rc}");
    }
    let storage = CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        (n_tokens, hidden),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn embed_lookup_bf16_op(
    embed_weight: &Tensor,
    tokens: &Tensor,
    _device: &Device,
) -> Result<Tensor> {
    let flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
    Ok(embed_weight.index_select(&flat, 0)?)
}

#[cfg(feature = "cuda")]
pub(crate) fn scale_bf16_op(x: &Tensor, scale: f32, device: &Device) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    let Device::Cuda(dev) = device.clone() else {
        bail!("scale_bf16_op requires CUDA")
    };
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(n)? };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let x_cuda = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for x"),
        };
        let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
        let (px, _) = x_slice.device_ptr(&stream);
        let (py, _) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::scale_out_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                py as *mut u16,
                scale,
                n,
            )
        }
    };
    if rc != 0 {
        bail!("scale_out_bf16 kernel returned {rc}");
    }
    let storage = CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = Storage::Cuda(storage);
    let shape: Shape = dims.into();
    Ok(candle_core::Tensor::from_storage(
        storage,
        shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn scale_bf16_op(x: &Tensor, scale: f32, _device: &Device) -> Result<Tensor> {
    let scale_t = Tensor::new(scale, x.device())?.to_dtype(x.dtype())?;
    Ok(x.broadcast_mul(&scale_t)?)
}

#[cfg(feature = "cuda")]
fn tanh_softcap_bf16_to_f32_into_op(
    logits: &Tensor,
    cap: f32,
    out_buf: &mut CudaSlice<f32>,
    device: &Device,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let Device::Cuda(dev) = device.clone() else {
        bail!("tanh_softcap_bf16_to_f32_into_op requires CUDA")
    };
    let logits_c = logits.contiguous()?;
    if logits_c.dtype() != DType::BF16 {
        bail!(
            "tanh_softcap_bf16_to_f32_into_op: logits dtype must be BF16, got {:?}",
            logits_c.dtype()
        );
    }
    let dims = logits_c.dims().to_vec();
    let n: usize = dims.iter().product();
    ensure!(
        out_buf.len() >= n,
        "tanh_softcap_bf16_to_f32_into_op: out_buf len {} < required {}",
        out_buf.len(),
        n
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let (xs, _xl) = logits_c.storage_and_layout();
    let x_cuda = match &*xs {
        Storage::Cuda(s) => s,
        _ => bail!("expected cuda storage for logits"),
    };
    let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
    let (px, _) = x_slice.device_ptr(&stream);
    let (py, _) = out_buf.device_ptr_mut(&stream);
    let rc = unsafe {
        nv_kernels::cuda::tanh_softcap_bf16_to_f32(
            stream.cu_stream() as *mut _,
            px as *const u16,
            py as *mut f32,
            cap,
            n,
        )
    };
    if rc != 0 {
        bail!("tanh_softcap_bf16_to_f32 kernel returned {rc}");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn tanh_softcap_bf16_to_f32_op(
    logits: &Tensor,
    cap: f32,
    device: &Device,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let Device::Cuda(dev) = device.clone() else {
        bail!("tanh_softcap_bf16_to_f32_op requires CUDA")
    };
    let logits_c = logits.contiguous()?;
    if logits_c.dtype() != DType::BF16 {
        bail!(
            "tanh_softcap_bf16_to_f32_op: logits dtype must be BF16, got {:?}",
            logits_c.dtype()
        );
    }
    let dims = logits_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n)? };
    let rc = {
        let (xs, _xl) = logits_c.storage_and_layout();
        let x_cuda = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage for logits"),
        };
        let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
        let (px, _) = x_slice.device_ptr(&stream);
        let (py, _) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::tanh_softcap_bf16_to_f32(
                stream.cu_stream() as *mut _,
                px as *const u16,
                py as *mut f32,
                cap,
                n,
            )
        }
    };
    if rc != 0 {
        bail!("tanh_softcap_bf16_to_f32 kernel returned {rc}");
    }
    let storage = CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = Storage::Cuda(storage);
    let shape: Shape = dims.into();
    Ok(candle_core::Tensor::from_storage(
        storage,
        shape,
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn tanh_softcap_bf16_to_f32_op(
    logits: &Tensor,
    cap: f32,
    device: &Device,
) -> Result<Tensor> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    if cap > 0.0 && cap.is_finite() {
        let c = Tensor::new(cap, device)?;
        let scaled = logits_f32.broadcast_div(&c)?;
        Ok(scaled.tanh()?.broadcast_mul(&c)?)
    } else {
        Ok(logits_f32)
    }
}

fn gelu_tanh_mul_candle(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let in_dtype = gate.dtype();
    let g_f32 = gate.to_dtype(DType::F32)?;
    let dev = g_f32.device();
    let coeff = Tensor::new((2.0f32 / std::f32::consts::PI).sqrt(), dev)?;
    let half = Tensor::new(0.5f32, dev)?;
    let one = Tensor::new(1.0f32, dev)?;
    let cubic_c = Tensor::new(0.044715f32, dev)?;
    let g3 = g_f32.powf(3.0)?;
    let inner = g_f32.add(&g3.broadcast_mul(&cubic_c)?)?;
    let arg = inner.broadcast_mul(&coeff)?;
    let t = arg.tanh()?;
    let one_plus_t = t.broadcast_add(&one)?;
    let gelu = g_f32.broadcast_mul(&half)?.mul(&one_plus_t)?;
    let gelu_b = gelu.to_dtype(in_dtype)?;
    Ok(gelu_b.mul(up)?)
}

pub fn causal_attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    offset: usize,
) -> Result<Tensor> {
    let dev = q.device().clone();
    let out_dtype = q.dtype();
    let group = n_q / n_kv;
    ensure!(group * n_kv == n_q, "GQA group must divide n_q");
    let total = k.dims()[1];
    ensure!(
        total == offset + seq,
        "causal_attention_chunked: keys {total} != offset {offset} + seq {seq}"
    );

    let q_t = q
        .to_dtype(DType::F32)?
        .permute((0, 2, 1, 3))?
        .reshape((n_q, seq, head_dim))?
        .contiguous()?;
    let k_t = k
        .to_dtype(DType::F32)?
        .permute((0, 2, 1, 3))?
        .reshape((n_kv, total, head_dim))?
        .contiguous()?;
    let v_t = v
        .to_dtype(DType::F32)?
        .permute((0, 2, 1, 3))?
        .reshape((n_kv, total, head_dim))?
        .contiguous()?;
    let k_perm = k_t.permute((0, 2, 1))?.contiguous()?;

    let budget_rows = ((1usize << 28) / (4 * n_q * total.max(1))).max(16);
    let qchunk = budget_rows.min(512).min(seq).max(1);

    let mut chunks: Vec<Tensor> = Vec::with_capacity(seq.div_ceil(qchunk));
    let mut start = 0usize;
    while start < seq {
        let len = qchunk.min(seq - start);
        let kv_len = offset + start + len;

        let q_chunk = q_t
            .reshape((n_kv, group, seq, head_dim))?
            .narrow(2, start, len)?
            .reshape((n_kv, group * len, head_dim))?
            .contiguous()?;
        let k_slice = k_perm.narrow(2, 0, kv_len)?.contiguous()?;
        let v_slice = v_t.narrow(1, 0, kv_len)?.contiguous()?;

        let scores = q_chunk.matmul(&k_slice)?;

        let mut bias_host = vec![0f32; len * kv_len];
        for i in 0..len {
            let visible = offset + start + i + 1;
            for j in visible..kv_len {
                bias_host[i * kv_len + j] = -1.0e30;
            }
        }
        let bias = Tensor::from_vec(bias_host, (1usize, 1usize, len, kv_len), &dev)?;

        let scores = scores
            .reshape((n_kv, group, len, kv_len))?
            .broadcast_add(&bias)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs
            .reshape((n_kv, group * len, kv_len))?
            .matmul(&v_slice)?;

        let out = out
            .reshape((n_q, len, head_dim))?
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((1usize, len, n_q, head_dim))?;
        chunks.push(out);
        start += len;
    }

    let full = if chunks.len() == 1 {
        chunks.pop().unwrap()
    } else {
        Tensor::cat(&chunks.iter().collect::<Vec<_>>()[..], 1)?
    };
    Ok(full.to_dtype(out_dtype)?)
}

fn sdpa_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq_q: usize,
    attn_mask: &Tensor,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let qd = q.dims();
    let kd = k.dims();
    if qd.len() != 4 || qd[2] != n_q || qd[3] != head_dim {
        bail!("sdpa_with_mask: bad q dims {:?}", qd);
    }
    let seq_k = kd[1];
    if attn_mask.dims() != [seq_q, seq_k] {
        bail!(
            "sdpa_with_mask: attn_mask must be [{seq_q},{seq_k}], got {:?}",
            attn_mask.dims()
        );
    }
    let b = qd[0];

    let q_f32 = q.to_dtype(DType::F32)?;
    let k_f32 = k.to_dtype(DType::F32)?;
    let v_f32 = v.to_dtype(DType::F32)?;

    let (k_exp, v_exp) = if n_kv == n_q {
        (k_f32, v_f32)
    } else {
        let factor = n_q / n_kv;
        let k_exp = k_f32
            .unsqueeze(3)?
            .expand((b, seq_k, n_kv, factor, head_dim))?
            .reshape((b, seq_k, n_q, head_dim))?;
        let v_exp = v_f32
            .unsqueeze(3)?
            .expand((b, seq_k, n_kv, factor, head_dim))?
            .reshape((b, seq_k, n_q, head_dim))?;
        (k_exp, v_exp)
    };

    let q_t = q_f32.permute((0, 2, 1, 3))?.contiguous()?;
    let k_t = k_exp.permute((0, 2, 1, 3))?.contiguous()?;
    let v_t = v_exp.permute((0, 2, 1, 3))?.contiguous()?;

    let q_flat = q_t.reshape((b * n_q, seq_q, head_dim))?;
    let k_flat = k_t.reshape((b * n_q, seq_k, head_dim))?;
    let v_flat = v_t.reshape((b * n_q, seq_k, head_dim))?;

    let k_perm = k_flat.permute((0, 2, 1))?.contiguous()?;

    let mut scores = q_flat.matmul(&k_perm)?;

    let mask_f32 = attn_mask.to_dtype(DType::F32)?;
    let one = Tensor::new(1.0f32, q_flat.device())?;
    let big = Tensor::new(1.0e30f32, q_flat.device())?;
    let bias_user = mask_f32.broadcast_sub(&one)?.broadcast_mul(&big)?;
    let mut bias = bias_user.reshape((1usize, seq_q, seq_k))?;

    if let Some(window) = sliding_window {
        let w = window.max(1);
        let mut sw = vec![0f32; seq_q * seq_k];
        let offset = seq_k.saturating_sub(seq_q);
        for i in 0..seq_q {
            for j in 0..seq_k {
                let qi = i + offset;
                if j > qi || (qi >= j && qi - j >= w) {
                    sw[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
        let sw_t = Tensor::from_vec(sw, (1usize, seq_q, seq_k), q_flat.device())?;
        bias = bias.add(&sw_t)?;
    }

    scores = scores.broadcast_add(&bias)?;

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v_flat)?;
    let out = out
        .reshape((b, n_q, seq_q, head_dim))?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    let out = out.to_dtype(q.dtype())?;
    Ok(out)
}

fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq_q: usize,
    window_left: Option<usize>,
) -> Result<Tensor> {
    use nv_layers::attn::{flash_attn_windowed, AttnConfig};
    let cfg = AttnConfig {
        num_heads: n_q,
        num_kv_heads: n_kv,
        head_dim,

        softmax_scale: 1.0,
        causal: true,
    };

    let out = flash_attn_windowed(q, k, v, &cfg, window_left, Some(0))?;
    assert_eq!(out.dims(), &[1, seq_q, n_q, head_dim]);
    Ok(out)
}

#[cfg(feature = "cuda")]
use crate::paged_fp8::PagedGemma4Cache;
#[cfg(feature = "cuda")]
use candle_core::op::BackpropOp;
#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
#[cfg(feature = "cuda")]
use candle_core::Shape;
#[cfg(feature = "cuda")]
use candle_core::Storage;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use nv_quant::fp8::Fp8GemmRunner;
#[cfg(feature = "cuda")]
use nv_quant::nvfp4::Nvfp4GemmRunner;
#[cfg(feature = "cuda")]
use nv_quant::nvfp4::MIN_TILE;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Once;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XattnScore {
    Raw,
    Mass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XattnHeadComb {
    Max,
    Mean,
}

#[derive(Clone, Debug)]
pub struct XattnCfg {
    pub block: usize,
    pub stride: usize,
    pub threshold: f32,
    pub score: XattnScore,
    pub headcomb: XattnHeadComb,
}

static XATTN_OVERRIDE: AtomicU8 = AtomicU8::new(0);

static XATTN_SCORE_OVR: AtomicU8 = AtomicU8::new(0);

static XATTN_COMB_OVR: AtomicU8 = AtomicU8::new(0);

static XATTN_THRESH_OVR: AtomicU64 = AtomicU64::new(0);
static XATTN_KEPT: AtomicU64 = AtomicU64::new(0);
static XATTN_CAND: AtomicU64 = AtomicU64::new(0);

pub fn xattn_stats_enabled() -> bool {
    nv_config::env_arm_present_at_all_reread_every_call("NV_XATTN_STATS")
}

pub fn xattn_cfg() -> Option<XattnCfg> {
    let on = match XATTN_OVERRIDE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => nv_config::env_arm_present_at_all_reread_every_call("NV_XATTN_PREFILL"),
    };
    if !on {
        return None;
    }
    let getu = |k: &str, d: usize| env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let getf = |k: &str, d: f32| env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let score = match XATTN_SCORE_OVR.load(Ordering::Relaxed) {
        1 => XattnScore::Raw,
        2 => XattnScore::Mass,
        _ => match env::var("NV_XATTN_SCORE").ok().as_deref() {
            Some("mass") => XattnScore::Mass,
            _ => XattnScore::Raw,
        },
    };
    let headcomb = match XATTN_COMB_OVR.load(Ordering::Relaxed) {
        1 => XattnHeadComb::Max,
        2 => XattnHeadComb::Mean,
        _ => match env::var("NV_XATTN_HEADCOMB").ok().as_deref() {
            Some("mean") => XattnHeadComb::Mean,
            _ => XattnHeadComb::Max,
        },
    };
    let threshold = {
        let ovr = XATTN_THRESH_OVR.load(Ordering::Relaxed);
        if ovr & 1 == 1 {
            f32::from_bits((ovr >> 1) as u32)
        } else {
            getf("NV_XATTN_THRESH", 0.9)
        }
    }
    .clamp(0.0, 1.0);
    Some(XattnCfg {
        block: getu("NV_XATTN_BLOCK", 128).max(1),
        stride: getu("NV_XATTN_STRIDE", 16).max(1),
        threshold,
        score,
        headcomb,
    })
}

pub fn xattn_prefill_bias(
    scores: &Tensor,
    seq: usize,
    stored: usize,
    q_offset: usize,
    k_offset: usize,
    cfg: &XattnCfg,
) -> Result<(Tensor, u64, u64)> {
    if cfg.score == XattnScore::Mass {
        return xattn_prefill_bias_mass(scores, seq, stored, q_offset, k_offset, cfg);
    }
    let dev = scores.device().clone();
    let hm = scores.mean(0)?.to_dtype(DType::F32)?;
    let hm_host = hm.reshape((seq * stored,))?.to_vec1::<f32>()?;

    let block = cfg.block;
    let stride = cfg.stride;
    let nqb = seq.div_ceil(block);

    let mut bias = vec![0f32; seq * stored];
    let mut kept_total: u64 = 0;
    let mut cand_total: u64 = 0;

    for bq in 0..nqb {
        let r0 = bq * block;
        let r1 = ((bq + 1) * block).min(seq);
        let qpos_max = q_offset + r1 - 1;

        let last_vis = qpos_max.saturating_sub(k_offset);
        let last_col = last_vis.min(stored.saturating_sub(1));
        let nkb_cand = last_col / block + 1;

        let mut sblk = vec![f32::NEG_INFINITY; nkb_cand];
        for bk in 0..nkb_cand {
            let c0 = bk * block;
            let c1 = ((bk + 1) * block).min(stored);
            let mut s = 0f32;
            let mut cnt = 0u32;
            for r in r0..r1 {
                let qpos = q_offset + r;
                let row = r * stored;
                for c in c0..c1 {
                    let kpos = k_offset + c;
                    if kpos > qpos {
                        continue;
                    }
                    if !((r - r0) + (c - c0)).is_multiple_of(stride) {
                        continue;
                    }
                    s += hm_host[row + c];
                    cnt += 1;
                }
            }
            sblk[bk] = if cnt > 0 { s } else { f32::NEG_INFINITY };
        }

        let maxs = sblk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = sblk
            .iter()
            .map(|&x| if x.is_finite() { (x - maxs).exp() } else { 0.0 })
            .collect();
        let total: f32 = exps.iter().sum();
        let mut order: Vec<usize> = (0..nkb_cand).collect();
        order.sort_by(|&a, &b| {
            exps[b]
                .partial_cmp(&exps[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut keep = vec![false; nkb_cand];
        let mut acc = 0f32;
        for &f in &[0usize, nkb_cand - 1] {
            if !keep[f] && sblk[f].is_finite() {
                keep[f] = true;
                acc += exps[f];
            }
        }
        let target = cfg.threshold * total;
        for &i in &order {
            if acc >= target {
                break;
            }
            if !keep[i] && sblk[i].is_finite() {
                keep[i] = true;
                acc += exps[i];
            }
        }

        cand_total += nkb_cand as u64;
        kept_total += keep.iter().filter(|&&k| k).count() as u64;

        for bk in 0..nkb_cand {
            if keep[bk] {
                continue;
            }
            let c0 = bk * block;
            let c1 = ((bk + 1) * block).min(stored);
            for r in r0..r1 {
                let row = r * stored;
                for c in c0..c1 {
                    bias[row + c] = f32::NEG_INFINITY;
                }
            }
        }
    }

    XATTN_KEPT.fetch_add(kept_total, Ordering::Relaxed);
    XATTN_CAND.fetch_add(cand_total, Ordering::Relaxed);

    let bias_t = Tensor::from_vec(bias, (1usize, seq, stored), &dev)?;
    Ok((bias_t, kept_total, cand_total))
}

fn xattn_prefill_bias_mass(
    scores: &Tensor,
    seq: usize,
    stored: usize,
    q_offset: usize,
    k_offset: usize,
    cfg: &XattnCfg,
) -> Result<(Tensor, u64, u64)> {
    let dev = scores.device().clone();
    let n_q = scores.dims()[0];
    let block = cfg.block;
    let nqb = seq.div_ceil(block);

    let mut bias = vec![0f32; seq * stored];
    let mut kept_total: u64 = 0;
    let mut cand_total: u64 = 0;

    let mut qblk: Vec<(usize, usize, usize)> = Vec::with_capacity(nqb);
    for bq in 0..nqb {
        let r0 = bq * block;
        let r1 = ((bq + 1) * block).min(seq);
        let qpos_max = q_offset + r1 - 1;
        let last_vis = qpos_max.saturating_sub(k_offset);
        let last_col = last_vis.min(stored.saturating_sub(1));
        let nkb_cand = last_col / block + 1;
        qblk.push((r0, r1, nkb_cand));
    }

    let mut combined: Vec<Vec<f32>> = qblk.iter().map(|&(_, _, nkb)| vec![0f32; nkb]).collect();

    let use_max = cfg.headcomb == XattnHeadComb::Max;

    for h in 0..n_q {
        let head = scores.narrow(0, h, 1)?.reshape((seq, stored))?;
        let hv = head.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (bq, &(r0, r1, nkb_cand)) in qblk.iter().enumerate() {
            let nrows = (r1 - r0).max(1);
            let mut head_mass = vec![0f32; nkb_cand];
            for r in r0..r1 {
                let qpos = q_offset + r;
                let last_vis_col = qpos.saturating_sub(k_offset).min(stored - 1);
                let row = &hv[r];

                let mut m = f32::NEG_INFINITY;
                for c in 0..=last_vis_col {
                    if row[c] > m {
                        m = row[c];
                    }
                }
                let mut denom = 0f32;
                for c in 0..=last_vis_col {
                    denom += (row[c] - m).exp();
                }
                if denom <= 0.0 {
                    continue;
                }
                let inv = 1.0 / denom;
                for c in 0..=last_vis_col {
                    let p = (row[c] - m).exp() * inv;
                    head_mass[c / block] += p;
                }
            }

            let scale = 1.0 / nrows as f32;
            let comb = &mut combined[bq];
            for bk in 0..nkb_cand {
                let v = head_mass[bk] * scale;
                if use_max {
                    if v > comb[bk] {
                        comb[bk] = v;
                    }
                } else {
                    comb[bk] += v;
                }
            }
        }
    }

    for (bq, &(r0, r1, nkb_cand)) in qblk.iter().enumerate() {
        let comb = &combined[bq];
        let mut score = comb.clone();
        if !use_max {
            let inv = 1.0 / n_q as f32;
            for s in score.iter_mut() {
                *s *= inv;
            }
        }
        let total: f32 = score.iter().sum();
        let mut order: Vec<usize> = (0..nkb_cand).collect();
        order.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut keep = vec![false; nkb_cand];
        let mut acc = 0f32;
        for &f in &[0usize, nkb_cand - 1] {
            if !keep[f] {
                keep[f] = true;
                acc += score[f];
            }
        }
        let target = cfg.threshold * total;
        for &i in &order {
            if acc >= target {
                break;
            }
            if !keep[i] {
                keep[i] = true;
                acc += score[i];
            }
        }

        cand_total += nkb_cand as u64;
        kept_total += keep.iter().filter(|&&k| k).count() as u64;

        for bk in 0..nkb_cand {
            if keep[bk] {
                continue;
            }
            let c0 = bk * block;
            let c1 = ((bk + 1) * block).min(stored);
            for r in r0..r1 {
                let rowb = r * stored;
                for c in c0..c1 {
                    bias[rowb + c] = f32::NEG_INFINITY;
                }
            }
        }
    }

    XATTN_KEPT.fetch_add(kept_total, Ordering::Relaxed);
    XATTN_CAND.fetch_add(cand_total, Ordering::Relaxed);

    let bias_t = Tensor::from_vec(bias, (1usize, seq, stored), &dev)?;
    Ok((bias_t, kept_total, cand_total))
}

#[cfg(feature = "cuda")]
fn split_cols_bf16_raw(fused: &Tensor, parts: &[(usize, usize)]) -> Result<Vec<Tensor>> {
    use cudarc::driver::sys as cu_sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    ensure!(
        fused.dtype() == DType::BF16,
        "split_cols_bf16_raw: bf16 only"
    );
    let dims = fused.dims().to_vec();
    let n_total = *dims.last().unwrap();
    let m: usize = dims[..dims.len() - 1].iter().product();
    let Device::Cuda(dev) = fused.device().clone() else {
        bail!("split_cols_bf16_raw: cuda only")
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let f_c = fused.contiguous()?;
    let (fs, fl) = f_c.storage_and_layout();
    let f_cuda = match &*fs {
        Storage::Cuda(s) => s,
        _ => bail!("expected cuda storage"),
    };
    let f_slice = f_cuda.as_cuda_slice::<bf16>()?;
    let f_view = f_slice.slice(fl.start_offset()..);
    let (fp, _gf) = f_view.device_ptr(&stream);

    let elt = std::mem::size_of::<bf16>();
    let mut outs = Vec::with_capacity(parts.len());
    for &(off, w) in parts {
        ensure!(off + w <= n_total, "split_cols_bf16_raw: part out of range");

        let mut out: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m * w)? };
        {
            let (op, _go) = out.device_ptr_mut(&stream);
            let cfg = cu_sys::CUDA_MEMCPY2D_st {
                srcXInBytes: off * elt,
                srcY: 0,
                srcMemoryType: cu_sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(),
                srcDevice: fp,
                srcArray: std::ptr::null_mut(),
                srcPitch: n_total * elt,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: cu_sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(),
                dstDevice: op,
                dstArray: std::ptr::null_mut(),
                dstPitch: w * elt,
                WidthInBytes: w * elt,
                Height: m,
            };
            let rc = unsafe { cu_sys::cuMemcpy2DAsync_v2(&cfg, stream.cu_stream()) };
            ensure!(
                rc == cu_sys::CUresult::CUDA_SUCCESS,
                "cuMemcpy2DAsync_v2 failed: {rc:?}"
            );
        }
        let storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(w);
        outs.push(candle_core::Tensor::from_storage(
            Storage::Cuda(storage),
            Shape::from(out_dims),
            BackpropOp::none(),
            false,
        ));
    }
    Ok(outs)
}

fn load_qkv_fused(
    weights: &WeightLoader,
    prefix: &str,
    q_dim: usize,
    kv_dim: usize,
    hidden: usize,
    has_v: bool,
    dtype: DType,
) -> Result<Linear> {
    let get = |name: String, out: usize| -> Result<Tensor> {
        let w = weights
            .get(&name, dtype)
            .with_context(|| format!("load {name}"))?;
        let d = w.dims();
        if d.len() != 2 || d[0] != out || d[1] != hidden {
            bail!("linear {name}: expected [{}, {}], got {:?}", out, hidden, d);
        }
        Ok(w)
    };
    let q_w = get(format!("{prefix}.self_attn.q_proj.weight"), q_dim)?;
    let k_w = get(format!("{prefix}.self_attn.k_proj.weight"), kv_dim)?;
    let fused = if has_v {
        let v_w = get(format!("{prefix}.self_attn.v_proj.weight"), kv_dim)?;
        Tensor::cat(&[&q_w, &k_w, &v_w], 0)?.contiguous()?
    } else {
        Tensor::cat(&[&q_w, &k_w], 0)?.contiguous()?
    };
    if attn_pretranspose_enabled() {
        Linear::new(fused, None)
    } else {
        Linear::new_no_pretranspose(fused, None)
    }
}

fn load_attn_proj_tensor(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<candle_core::Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        bail!(
            "linear {name}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Ok(w)
}

fn load_attn_proj(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    Linear::new(
        load_attn_proj_tensor(weights, name, out_features, in_features, dtype)?,
        None,
    )
}

pub fn attn_pretranspose_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            env::var("NV_ATTN_PRETRANSPOSE").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}

fn load_attn_proj_lean(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = load_attn_proj_tensor(weights, name, out_features, in_features, dtype)?;
    if attn_pretranspose_enabled() {
        Linear::new(w, None)
    } else {
        Linear::new_no_pretranspose(w, None)
    }
}

pub const PREFILL_W4A4_MIN_M: usize = 256;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn prefill_w4a4_selects(m: usize) -> bool {
    m >= PREFILL_W4A4_MIN_M
}

#[cfg(feature = "cuda")]
fn prefill_fp4_copy(
    lin: &Linear,
    runner: Option<&Arc<Mutex<Nvfp4GemmRunner>>>,
    device: &Device,
) -> Result<Option<Linear>> {
    let Some(runner) = runner else {
        return Ok(None);
    };
    let Some(weight) = lin.weight() else {
        return Ok(None);
    };
    let dims = weight.dims();
    if dims.len() != 2
        || dims[1] % nv_quant::nvfp4::BLOCK_SIZE != 0
        || dims[0] < MIN_TILE
        || dims[1] < MIN_TILE
    {
        return Ok(None);
    }
    let weight = weight.clone();
    Linear::from_bf16_quantized_nvfp4_dev(&weight, None, device, runner.clone()).map(Some)
}

pub const ATTN_PROJ_QUANT_DEFAULT_ON: bool = false;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnProjScheme {
    Nvfp4,
    Fp8E4m3,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn attn_proj_is_fp8_token(raw: &str) -> bool {
    matches!(
        raw.to_ascii_lowercase().as_str(),
        "fp8" | "e4m3" | "fp8_e4m3" | "fp8-e4m3" | "fp8e4m3"
    )
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn attn_proj_scheme_from(raw: Option<&str>) -> AttnProjScheme {
    match raw {
        Some(r) if attn_proj_is_fp8_token(r) => AttnProjScheme::Fp8E4m3,
        _ => AttnProjScheme::Nvfp4,
    }
}

pub const ATTN_PROJ_NVFP4_ALONE_IS_A_MEASURED_QUALITY_REJECT_RESCUED_ONLY_BY_NV_NVFP4_M1_GEMV:
    &str = "NV_ATTN_PROJ_QUANT=nvfp4 on its own scores 17.056 against a 16.308 zero-env basis on \
     ppl_gemma4::gemma4_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode \
     (Gemma-4-31B-IT-NVFP4, 285 prompt tokens, 512 scored, wiki corpus): +0.748 ppl, a REJECT \
     on magnitude at 1.82x the family's own 0.41 reassociation-null spread (the +0.0973 figure \
     once cited here is a qwen3.8 wikitext arm delta, not a floor, and binds no other \
     family). NV_NVFP4_M1_GEMV=1 is what makes it shippable, and not by hiding it: routing the \
     m=1 NVFP4 matmul off the padded cutlass block-scaled tile onto nvfp4_gemv_bf16act is worth \
     -0.742 ppl on its own because the gemv skips the fp4 activation round-trip, so the pair \
     lands at +0.038 and the three-key record at -0.018. The two keys are therefore inseparable \
     and a serving process that quantizes the attention projections without the gemv route is \
     serving a configuration this family's own ship gate rejects.";

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn attn_proj_env_selects_nvfp4_projections(raw: Option<&str>) -> bool {
    attn_proj_quant_from(true, raw) != (false, false)
        && attn_proj_scheme_from(raw) == AttnProjScheme::Nvfp4
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn attn_proj_quant_from(scheme_ok: bool, raw: Option<&str>) -> (bool, bool) {
    if !scheme_ok {
        return (false, false);
    }
    if let Some(r) = raw {
        if attn_proj_is_fp8_token(r) {
            return (true, true);
        }
    }
    match raw {
        Some("off") | Some("0") | Some("bf16") => (false, false),
        Some("qkv") => (true, false),
        Some("o") | Some("oproj") => (false, true),
        Some("1") | Some("on") | Some("true") | Some("yes") | Some("all") | Some("nvfp4")
        | Some("fp4") | Some("qkv+o") => (true, true),
        Some(other) => {
            attn_proj_quant_warn_unknown(other);
            (false, false)
        }
        None => (ATTN_PROJ_QUANT_DEFAULT_ON, ATTN_PROJ_QUANT_DEFAULT_ON),
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn attn_proj_quant_warn_unknown(raw: &str) {
    static WARN: Once = Once::new();
    let raw = raw.to_string();
    WARN.call_once(move || {
        let lower = raw.to_ascii_lowercase();
        if lower.contains("e5m2") {
            eprintln!(
                "NV_ATTN_PROJ_QUANT={raw}: only fp8-e4m3 is implemented for attention \
                 projections (use NV_ATTN_PROJ_QUANT=fp8); e5m2 is not supported. \
                 Attention projections stay bf16."
            );
        } else {
            eprintln!(
                "NV_ATTN_PROJ_QUANT={raw}: unrecognized value, attention projections stay bf16. \
                 Accepted: off|0|bf16, qkv, o|oproj, 1|on|true|yes|all|nvfp4|fp4|qkv+o, fp8|e4m3."
            );
        }
    });
}

#[cfg(feature = "cuda")]
fn attn_proj_quant_mode(qconfig: Option<&QuantizationConfig>) -> (bool, bool) {
    let scheme_ok = qconfig
        .map(|q| matches!(q.scheme, QuantScheme::Nvfp4))
        .unwrap_or(false);
    attn_proj_quant_from(scheme_ok, env::var("NV_ATTN_PROJ_QUANT").ok().as_deref())
}

#[cfg(feature = "cuda")]
fn attn_proj_scheme() -> AttnProjScheme {
    attn_proj_scheme_from(env::var("NV_ATTN_PROJ_QUANT").ok().as_deref())
}

#[cfg(feature = "cuda")]
fn quantize_attn_proj_nvfp4(
    lin: Linear,
    runner: Option<&Arc<Mutex<Nvfp4GemmRunner>>>,
    device: &Device,
) -> Result<Linear> {
    let Some(runner) = runner else {
        return Ok(lin);
    };
    let Some(weight) = lin.weight() else {
        return Ok(lin);
    };
    let dims = weight.dims();
    if dims.len() != 2
        || dims[1] % nv_quant::nvfp4::BLOCK_SIZE != 0
        || dims[0] < MIN_TILE
        || dims[1] < MIN_TILE
    {
        return Ok(lin);
    }
    let weight = weight.clone();
    Linear::from_bf16_quantized_nvfp4_dev(&weight, None, device, runner.clone())
}

#[cfg(feature = "cuda")]
fn quantize_attn_proj_fp8(
    lin: Linear,
    runner: Option<&Arc<Mutex<Fp8GemmRunner>>>,
    device: &Device,
) -> Result<Linear> {
    let Some(runner) = runner else {
        return Ok(lin);
    };
    let Some(weight) = lin.weight() else {
        return Ok(lin);
    };
    let dims = weight.dims();
    if dims.len() != 2 {
        return Ok(lin);
    }
    let weight = weight.clone();
    Linear::from_bf16_quantized_fp8(&weight, None, device, runner.clone())
}

#[cfg(feature = "cuda")]
fn load_mlp_proj(
    weights: &WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    qconfig: Option<&QuantizationConfig>,
    nvfp4_runner: Option<Arc<Mutex<Nvfp4GemmRunner>>>,
    device: &Device,
) -> Result<Linear> {
    let ignored = qconfig.map(|q| q.is_module_ignored(module)).unwrap_or(true);
    let bf16_name = format!("{module}.weight");
    if ignored || qconfig.is_none() {
        return load_attn_proj(weights, &bf16_name, out_features, in_features, dtype);
    }
    let scheme = qconfig.unwrap().scheme;
    if !matches!(scheme, QuantScheme::Nvfp4) {
        return load_attn_proj(weights, &bf16_name, out_features, in_features, dtype);
    }

    let packed_name = format!("{module}.weight");
    let scale2_name = format!("{module}.weight_scale_2");
    if !weights.has(&packed_name) || !weights.has(&scale2_name) {
        return load_attn_proj(weights, &bf16_name, out_features, in_features, dtype);
    }
    if in_features < MIN_TILE || out_features < MIN_TILE {
        return load_attn_proj(weights, &bf16_name, out_features, in_features, dtype);
    }
    let runner =
        nvfp4_runner.ok_or_else(|| anyhow::anyhow!("NVFP4 runner missing for {module}"))?;
    nv_layers::moe::nvfp4_linear_from_disk_with_suffixes(
        weights,
        module,
        out_features,
        in_features,
        runner,
        device,
        nv_layers::moe::Nvfp4Suffixes::GEMMA_MODELOPT,
    )
}

#[cfg(feature = "cuda")]
fn load_mlp_proj_fused_pair(
    weights: &WeightLoader,
    module_a: &str,
    module_b: &str,
    out_features_each: usize,
    in_features: usize,
    dtype: DType,
    qconfig: Option<&QuantizationConfig>,
    nvfp4_runner: Option<Arc<Mutex<Nvfp4GemmRunner>>>,
    device: &Device,
) -> Result<Linear> {
    let ignored = qconfig
        .map(|q| q.is_module_ignored(module_a) || q.is_module_ignored(module_b))
        .unwrap_or(true);
    let bf16_a = format!("{module_a}.weight");
    let bf16_b = format!("{module_b}.weight");
    let scheme = qconfig
        .map(|q| q.scheme)
        .unwrap_or(nv_weights::QuantScheme::None);
    let packed_a = format!("{module_a}.weight");
    let packed_b = format!("{module_b}.weight");
    let scale2_a = format!("{module_a}.weight_scale_2");
    let scale2_b = format!("{module_b}.weight_scale_2");
    let nvfp4_present = weights.has(&packed_a)
        && weights.has(&scale2_a)
        && weights.has(&packed_b)
        && weights.has(&scale2_b);
    let small_tile = in_features < MIN_TILE || out_features_each < MIN_TILE;
    if ignored
        || qconfig.is_none()
        || !matches!(scheme, QuantScheme::Nvfp4)
        || !nvfp4_present
        || small_tile
    {
        let a = weights.get(&bf16_a, dtype)?;
        let b = weights.get(&bf16_b, dtype)?;
        let fused = candle_core::Tensor::cat(&[&a, &b], 0)?.contiguous()?;
        return Linear::new(fused, None);
    }
    let runner = nvfp4_runner
        .ok_or_else(|| anyhow::anyhow!("NVFP4 runner missing for fused {module_a}+{module_b}"))?;
    nv_layers::moe::nvfp4_linear_from_disk_fused_pair(
        weights,
        module_a,
        module_b,
        out_features_each,
        in_features,
        runner,
        device,
        nv_layers::moe::Nvfp4Suffixes::GEMMA_MODELOPT,
    )
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn load_mlp_proj(
    _weights: &WeightLoader,
    _module: &str,
    _out_features: usize,
    _in_features: usize,
    _dtype: DType,
    _qconfig: Option<&()>,
    _nvfp4_runner: Option<()>,
    _device: &Device,
) -> Result<Linear> {
    bail!("gemma4 NVFP4 path requires the `cuda` feature")
}

#[cfg(test)]
mod boxed_prefill_dispatch_tests {
    use super::Gemma4Cache;
    use anyhow::Result;
    use candle_core::{Device, Tensor};
    use std::sync::{Arc, Mutex};

    type Call = (usize, usize, usize, usize, Option<usize>, u32);

    struct DefaultCache;

    impl Gemma4Cache for DefaultCache {
        fn current_len(&self) -> usize {
            0
        }
        fn advance(&mut self, _: usize) {
            panic!("unexpected advance")
        }
        fn prepare_for_decode(&mut self, _: usize, _: usize) -> Result<()> {
            panic!("unexpected prepare")
        }
        fn write_at(&mut self, _: usize, _: &Tensor, _: &Tensor) -> Result<()> {
            panic!("unexpected write")
        }
        fn view(&mut self, _: usize, _: usize) -> Result<(Tensor, Tensor)> {
            panic!("prefill dispatch must not enter a fallback view")
        }
    }

    struct HookCache(Arc<Mutex<Vec<Call>>>);

    #[derive(Debug)]
    struct PrefillFault;

    impl std::fmt::Display for PrefillFault {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("prefill fixture fault")
        }
    }

    impl std::error::Error for PrefillFault {}

    impl Gemma4Cache for HookCache {
        fn current_len(&self) -> usize {
            0
        }
        fn advance(&mut self, _: usize) {
            panic!("unexpected advance")
        }
        fn prepare_for_decode(&mut self, _: usize, _: usize) -> Result<()> {
            panic!("unexpected prepare")
        }
        fn write_at(&mut self, _: usize, _: &Tensor, _: &Tensor) -> Result<()> {
            panic!("unexpected write")
        }
        fn view(&mut self, _: usize, _: usize) -> Result<(Tensor, Tensor)> {
            panic!("prefill dispatch must not enter a fallback view")
        }
        fn try_prefill_attention_fp8(
            &mut self,
            layer: usize,
            q: &Tensor,
            n_q: usize,
            seq: usize,
            window: Option<usize>,
            scaling: f32,
        ) -> Result<Option<Tensor>> {
            self.0.lock().unwrap().push((
                layer,
                q as *const Tensor as usize,
                n_q,
                seq,
                window,
                scaling.to_bits(),
            ));
            if layer == 13 {
                return Err(PrefillFault.into());
            }
            Ok(Some(q.clone()))
        }
    }

    fn invoke<C: Gemma4Cache>(
        cache: &mut C,
        q: &Tensor,
        layer: usize,
        window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        cache.try_prefill_attention_fp8(layer, q, 48, 3, window, 0.125)
    }

    #[test]
    fn boxed_prefill_forwards_override_arguments_errors_and_default_none() {
        let q = Tensor::from_vec(
            vec![1.25f32, -2.5, 3.75, 4.0, 5.5, -6.25],
            (3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut erased: Box<dyn Gemma4Cache + Send> = Box::new(HookCache(calls.clone()));
        let out = invoke(&mut erased, &q, 7, Some(513))
            .unwrap()
            .expect("boxed generic dispatch lost the concrete prefill override");
        assert_eq!(out.dims(), q.dims());
        assert_eq!(out.to_vec2::<f32>().unwrap(), q.to_vec2::<f32>().unwrap());
        let error = invoke(&mut erased, &q, 13, None)
            .expect_err("boxed prefill swallowed an error as None");
        assert!(
            error.downcast_ref::<PrefillFault>().is_some(),
            "typed prefill error must survive delegation: {error:#}"
        );
        let mut concrete = Box::new(HookCache(calls.clone()));
        assert!(invoke(&mut concrete, &q, 11, None).unwrap().is_some());
        let ptr = &q as *const Tensor as usize;
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                (7, ptr, 48, 3, Some(513), 0.125f32.to_bits()),
                (13, ptr, 48, 3, None, 0.125f32.to_bits()),
                (11, ptr, 48, 3, None, 0.125f32.to_bits()),
            ]
        );
        let mut default: Box<dyn Gemma4Cache + Send> = Box::new(DefaultCache);
        assert!(
            invoke(&mut default, &q, 7, None).unwrap().is_none(),
            "a cache without the optional hook must retain default None"
        );
    }
}

#[cfg(test)]
mod falsified_knob_default_pin_tests {
    use super::{spec_prefill_flash_from, verify_qkv_fused_from};

    #[test]
    fn lm_head_i8_rows_default_to_the_kernel_ceiling_verify_qkv_fused_ships_off_and_deterministic_mode_disables_spec_prefill_flash(
    ) {
        use super::lm_head_i8_rows_per_call_gate;
        assert_eq!(lm_head_i8_rows_per_call_gate(8), 8);
        assert_eq!(lm_head_i8_rows_per_call_gate(16), 16);
        assert_eq!(lm_head_i8_rows_per_call_gate(0), 1);
        let gemma4_31b_dflash_default_k = 8usize;
        assert!(
            lm_head_i8_rows_per_call_gate(8) >= gemma4_31b_dflash_default_k,
            "the default gate must keep the k=8 dflash verify body single-call, or the chunked \
             candle cat escapes the forked verify capture and replays freed memory"
        );

        assert!(!verify_qkv_fused_from(None));
        assert!(!verify_qkv_fused_from(Some("0")));
        assert!(!verify_qkv_fused_from(Some("true")));
        assert!(!verify_qkv_fused_from(Some("")));
        assert!(verify_qkv_fused_from(Some("1")));

        assert!(!spec_prefill_flash_from(true));
        assert!(spec_prefill_flash_from(false));
    }
}

#[cfg(test)]
mod mk_verify_gate_tests {
    use super::{
        gqa512_verify_geometry, mk_verify_gate, mk_verify_hd512_gate, verify_mask_is_chain,
    };

    #[test]
    fn mk_requires_exact_opt_in_outside_tree_mode_hd512_defaults_on_without_sharing_that_default_and_its_geometry_needs_8x_gqa(
    ) {
        assert!(mk_verify_gate(Some("1"), None));
        assert!(!mk_verify_gate(None, None));
        assert!(!mk_verify_gate(Some("0"), None));
        assert!(!mk_verify_gate(Some("true"), None));
        assert!(!mk_verify_gate(Some("1"), Some("1")));
        assert!(!mk_verify_gate(Some("1"), Some("0")));
        assert!(!mk_verify_gate(Some("1"), Some("")));
        assert!(!mk_verify_gate(None, Some("1")));

        assert!(mk_verify_hd512_gate(None));
        assert!(!mk_verify_hd512_gate(Some("1")));
        assert!(!mk_verify_hd512_gate(Some("0")));
        assert!(!mk_verify_hd512_gate(Some("")));

        assert!(!mk_verify_gate(None, None));
        assert!(mk_verify_hd512_gate(None));

        assert!(gqa512_verify_geometry(32, 4, 512));
        assert!(gqa512_verify_geometry(8, 1, 512));
        assert!(!gqa512_verify_geometry(32, 4, 256));
        assert!(!gqa512_verify_geometry(32, 4, 128));
        assert!(!gqa512_verify_geometry(32, 8, 512));
        assert!(!gqa512_verify_geometry(16, 4, 512));
        assert!(!gqa512_verify_geometry(0, 0, 512));
    }

    fn lower_tri(k: usize) -> Vec<u8> {
        let mut m = vec![0u8; k * k];
        for i in 0..k {
            for j in 0..=i {
                m[i * k + j] = 1;
            }
        }
        m
    }

    #[test]
    fn chain_predicate_accepts_lower_triangular_masks_and_rejects_every_other_shape() {
        for k in 1..=8 {
            assert!(verify_mask_is_chain(&lower_tri(k), k), "k={k}");
        }

        assert!(!verify_mask_is_chain(&[], 0));
        assert!(!verify_mask_is_chain(&lower_tri(3), 4));
        assert!(!verify_mask_is_chain(&[1u8; 9], 3));

        let mut branching = lower_tri(4);
        branching[3 * 4 + 1] = 0;
        branching[3 * 4 + 2] = 0;
        assert!(!verify_mask_is_chain(&branching, 4));

        let mut upper = lower_tri(3);
        upper[2] = 1;
        assert!(!verify_mask_is_chain(&upper, 3));
    }
}

#[cfg(test)]
mod attn_proj_quant_tests {
    use super::{
        attn_proj_quant_from, attn_proj_scheme_from, AttnProjScheme, ATTN_PROJ_QUANT_DEFAULT_ON,
    };

    #[test]
    fn attn_proj_quant_env_grammar_selects_qkv_o_both_fp8_or_neither_and_never_quantizes_a_non_nvfp4_checkpoint(
    ) {
        for raw in [None, Some("1"), Some("qkv"), Some("o"), Some("off")] {
            assert_eq!(
                attn_proj_quant_from(false, raw),
                (false, false),
                "raw={raw:?}"
            );
        }

        assert_eq!(attn_proj_quant_from(true, Some("off")), (false, false));
        assert_eq!(attn_proj_quant_from(true, Some("0")), (false, false));
        assert_eq!(attn_proj_quant_from(true, Some("bf16")), (false, false));
        assert_eq!(attn_proj_quant_from(true, Some("qkv")), (true, false));
        assert_eq!(attn_proj_quant_from(true, Some("o")), (false, true));
        assert_eq!(attn_proj_quant_from(true, Some("oproj")), (false, true));
        assert_eq!(attn_proj_quant_from(true, Some("1")), (true, true));
        assert_eq!(attn_proj_quant_from(true, Some("nvfp4")), (true, true));

        let d = ATTN_PROJ_QUANT_DEFAULT_ON;
        assert_eq!(attn_proj_quant_from(true, None), (d, d));

        for raw in ["fp8", "e4m3", "fp8_e4m3", "fp8-e4m3", "FP8", "E4M3"] {
            assert_eq!(
                attn_proj_quant_from(true, Some(raw)),
                (true, true),
                "NV_ATTN_PROJ_QUANT={raw} should select both projections"
            );
            assert_eq!(
                attn_proj_scheme_from(Some(raw)),
                AttnProjScheme::Fp8E4m3,
                "NV_ATTN_PROJ_QUANT={raw} should pick the fp8-e4m3 scheme"
            );
        }

        assert_eq!(attn_proj_quant_from(false, Some("fp8")), (false, false));

        assert_eq!(attn_proj_quant_from(true, Some("e5m2")), (false, false));

        assert_eq!(attn_proj_scheme_from(Some("nvfp4")), AttnProjScheme::Nvfp4);
        assert_eq!(attn_proj_scheme_from(None), AttnProjScheme::Nvfp4);

        for raw in ["yolo", "int8", "w4a4", "2", "nvfp4-ish", ""] {
            assert_eq!(
                attn_proj_quant_from(true, Some(raw)),
                (false, false),
                "raw={raw:?}"
            );
        }

        for raw in ["1", "on", "true", "yes", "all", "nvfp4", "fp4", "qkv+o"] {
            assert_eq!(
                attn_proj_quant_from(true, Some(raw)),
                (true, true),
                "raw={raw:?}"
            );
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn shipped_default_is_bf16_until_a_quality_eval_says_otherwise() {
        assert!(
            !ATTN_PROJ_QUANT_DEFAULT_ON,
            "attn-proj NVFP4 stays opt-in. Flipping this constant needs a powered \
             quality eval, not a rebase. Standing evidence (sp-perf/laneE/eval3, \
             MMLU-Pro 440 items, both arms server-verified): the direct probe is \
             -1.6 pts quant-on against a pre-registered -1.5 pt gate, 35/440 answers \
             change, and the paired split is 21 bf16-only vs 14 nvfp4-only \
             (McNemar p=0.311). The B-vs-C negative control measured a zero \
             cross-process noise floor on that metric, so those 35 flips are real \
             quant effects and not run-to-run jitter. Re-flip only by editing this \
             assertion together with a new eval that clears the gate."
        );
        assert_eq!(attn_proj_quant_from(true, None), (false, false));
        assert_eq!(attn_proj_quant_from(true, Some("nvfp4")), (true, true));
    }
}

#[cfg(test)]
mod prefill_w4a4_tests {
    use super::{prefill_w4a4_selects, PREFILL_W4A4_MIN_M};

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn selection_covers_prefill_chunks_not_decode_or_verify() {
        for m in [1usize, 2, 4, 8, 16, 24, 32, 64, 128, 255] {
            assert!(
                !prefill_w4a4_selects(m),
                "m={m} must stay on the bf16 decode path"
            );
        }
        for m in [256usize, 512, 568, 1024, 4096] {
            assert!(
                prefill_w4a4_selects(m),
                "m={m} must take the W4A4 prefill path"
            );
        }
        assert!(PREFILL_W4A4_MIN_M > 128);
    }
}

#[cfg(test)]
mod kv_window_tests {
    use super::check_kv_window;
    use super::{tree_layer_window, tree_window_attends, LayerType};

    #[test]
    fn tree_window_maps_layer_kinds_matches_the_sdpa_mask_and_bounds_the_prefill_shape() {
        assert_eq!(tree_layer_window(LayerType::SlidingAttention, 1024), 1024);
        assert_eq!(tree_layer_window(LayerType::FullAttention, 1024), 0);

        let w = 1024i64;
        assert!(tree_window_attends(2000, 2000, w));
        assert!(tree_window_attends(2000, 977, w));
        assert!(!tree_window_attends(2000, 976, w));
        assert!(!tree_window_attends(2000, 2001, w));

        assert!(tree_window_attends(500_000, 0, 0));
        assert!(tree_window_attends(500_000, 0, -1));

        for k in 0..=10 {
            assert!(tree_window_attends(10, k, w));
        }

        assert!(tree_window_attends(7, 7, 1));
        assert!(!tree_window_attends(7, 6, 1));

        let w = 4i64;
        let prompt = 13i64;
        for qi in 0..prompt {
            let visible = (0..prompt)
                .filter(|&j| tree_window_attends(qi, j, w))
                .count() as i64;
            assert_eq!(visible, (qi + 1).min(w), "query {qi}");
        }
    }

    #[test]
    fn kv_window_check_accepts_writes_that_fit_and_rejects_writes_past_the_end_or_overflowing_positions(
    ) {
        assert!(check_kv_window("t", 0, 128, 128).is_ok());
        assert!(check_kv_window("t", 127, 1, 128).is_ok());

        assert!(check_kv_window("t", 128, 1, 128).is_err());
        assert!(check_kv_window("t", 120, 9, 128).is_err());

        assert!(check_kv_window("t", usize::MAX, 2, usize::MAX).is_err());
    }
}

#[cfg(test)]
mod config_tests {
    use super::Gemma4Config;

    fn base_text() -> String {
        let layer_types: Vec<&str> = vec![
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ];
        serde_json::json!({
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": serde_json::Value::Null,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "sliding_window": 512,
            "final_logit_softcapping": serde_json::Value::Null,
            "num_kv_shared_layers": 4,
            "layer_types": layer_types,
            "attention_k_eq_v": false,
            "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {
                "full_attention": { "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        })
        .to_string()
    }

    #[test]
    fn kv_budget_hybrid_formula_matches_geometry_the_cap_shrinks_only_the_drafter_term_and_the_paged_pool_agrees(
    ) {
        let cfg = Gemma4Config::from_hf_json_str(&base_text()).unwrap();

        let b = super::kv_budget(&cfg, 4096, true, true, 128);
        let ring = 512 + super::VERIFY_PREFILL_CHUNK + 128;
        assert_eq!(b.ring_slots, ring);

        assert_eq!(b.verify_sliding_bytes, 3 * ring * 1040);

        assert_eq!(b.verify_full_bytes, 4096 * 2064);
        assert_eq!(b.decode_sliding_bytes, 3 * ring * 1040);
        assert_eq!(b.decode_full_bytes, 4096 * 2064);
        assert_eq!(b.drafter_kv_bytes, 4096 * 128 * 4);

        let b16 = super::kv_budget(&cfg, 4096, false, true, 0);
        assert_eq!(b16.verify_sliding_bytes, 3 * 4096 * (2 * 512 * 2));
        assert_eq!(b16.verify_full_bytes, 4096 * (2 * 1024 * 2));

        let flat = super::kv_budget(&cfg, 4096, true, false, 0);
        assert_eq!(flat.verify_sliding_bytes, 3 * 4096 * 1040);

        let cfg = Gemma4Config::from_hf_json_str(&base_text()).unwrap();
        let kv_max = 4096usize;
        let elems = 128usize;

        let uncapped = super::kv_budget(&cfg, kv_max, true, true, elems);
        let same = super::kv_budget_capped(&cfg, kv_max, true, true, elems, kv_max);
        assert_eq!(same.verify_full_bytes, uncapped.verify_full_bytes);
        assert_eq!(same.verify_sliding_bytes, uncapped.verify_sliding_bytes);
        assert_eq!(same.verify_scratch_bytes, uncapped.verify_scratch_bytes);
        assert_eq!(same.decode_full_bytes, uncapped.decode_full_bytes);
        assert_eq!(same.decode_sliding_bytes, uncapped.decode_sliding_bytes);
        assert_eq!(same.drafter_kv_bytes, uncapped.drafter_kv_bytes);
        assert_eq!(same.ring_slots, uncapped.ring_slots);

        let rows = 16 + 2048 + 256;
        let capped = super::kv_budget_capped(&cfg, kv_max, true, true, elems, rows);
        assert_eq!(capped.drafter_kv_bytes, rows * elems * 4);
        assert!(capped.drafter_kv_bytes < uncapped.drafter_kv_bytes);
        assert_eq!(capped.verify_full_bytes, uncapped.verify_full_bytes);
        assert_eq!(capped.verify_sliding_bytes, uncapped.verify_sliding_bytes);
        assert_eq!(capped.verify_scratch_bytes, uncapped.verify_scratch_bytes);
        assert_eq!(capped.decode_full_bytes, uncapped.decode_full_bytes);
        assert_eq!(capped.decode_sliding_bytes, uncapped.decode_sliding_bytes);
        assert_eq!(
            capped.worst_total() + (kv_max - rows) * elems * 4,
            uncapped.worst_total()
        );

        let cfg = Gemma4Config::from_hf_json_str(&base_text()).unwrap();
        let p = crate::paged_fp8::PagedPoolConfig::from_gemma4_hybrid(&cfg, 100, 16, 4);
        let rb: usize = (512 + super::VERIFY_PREFILL_CHUNK + 128).div_ceil(16);
        assert_eq!(p.sliding_ring_blocks, rb);
        assert_eq!(p.lanes, 4);
        assert_eq!(p.layer_blocks, vec![4 * rb, 4 * rb, 4 * rb, 100]);
        assert_eq!(p.layer_sliding, vec![true, true, true, false]);
        assert_eq!(p.num_blocks, 100);
        let sliding_block = 2 * 16 * 512 + 2 * 16 * 2 * 4;
        let full_block = 2 * 16 * 1024 + 2 * 16 * 2 * 4;
        assert_eq!(
            p.pool_bytes(),
            3 * (4 * rb) * sliding_block + 100 * full_block
        );

        let p2 = crate::paged_fp8::PagedPoolConfig::from_gemma4_hybrid(&cfg, 101, 16, 4);
        assert_eq!(p2.pool_bytes() - p.pool_bytes(), full_block);
    }

    #[test]
    fn nested_and_flat_configs_parse_null_softcap_and_null_global_kv_and_an_explicit_softcap_is_preserved(
    ) {
        let nested = serde_json::json!({
            "tie_word_embeddings": true,
            "text_config": serde_json::from_str::<serde_json::Value>(&base_text()).unwrap()
        })
        .to_string();
        let cfg = Gemma4Config::from_hf_json_str(&nested).expect("nested E4B-style config parses");
        assert_eq!(cfg.final_logit_softcapping, 0.0);
        assert_eq!(cfg.num_global_key_value_heads, None);
        assert_eq!(cfg.num_kv_shared_layers, 4);
        assert!(cfg.is_kv_shared_layer(3));

        let flat = serde_json::json!({
            "tie_word_embeddings": true,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": serde_json::Value::Null,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "sliding_window": 512,
            "final_logit_softcapping": serde_json::Value::Null,
            "num_kv_shared_layers": 4,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention"],
            "attention_k_eq_v": false,
            "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {
                "full_attention": { "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        })
        .to_string();
        let cfg = Gemma4Config::from_hf_json_str(&flat).expect("flat config parses");
        assert_eq!(cfg.final_logit_softcapping, 0.0);
        assert_eq!(cfg.num_global_key_value_heads, None);

        let with_cap = serde_json::json!({
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 256, "global_head_dim": 512, "vocab_size": 1000,
                "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                "sliding_window": 512, "final_logit_softcapping": 30.0,
                "layer_types": ["sliding_attention","full_attention"],
                "attention_k_eq_v": false, "hidden_activation": "gelu_pytorch_tanh",
                "rope_parameters": {
                    "full_attention": { "rope_theta": 1000000.0 },
                    "sliding_attention": { "rope_theta": 10000.0 }
                }
            }
        })
        .to_string();
        let cfg = Gemma4Config::from_hf_json_str(&with_cap).expect("config parses");
        assert_eq!(cfg.final_logit_softcapping, 30.0);
    }

    #[test]
    fn sliding_window_kv_cache_bounds_storage_and_keeps_window() {
        use super::{Gemma4Cache, Gemma4KvCache, SLIDING_COMPACT_SLACK};
        use candle_core::{DType, Device, Tensor};

        let json = serde_json::json!({
            "tie_word_embeddings": true,
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "num_global_key_value_heads": 1, "head_dim": 2, "global_head_dim": 2,
            "vocab_size": 32, "max_position_embeddings": 100000, "rms_norm_eps": 1e-6,
            "sliding_window": 4, "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention","full_attention","sliding_attention","full_attention"],
            "attention_k_eq_v": false, "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {
                "full_attention": { "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        })
        .to_string();
        let cfg = Gemma4Config::from_hf_json_str(&json).unwrap();
        let window = cfg.sliding_window;
        let dev = Device::Cpu;
        let mut cache = Gemma4KvCache::new(&cfg, 1 << 20, &dev, DType::F32).unwrap();

        let n = 600usize;
        for i in 0..n {
            cache.prepare_for_decode(i, i + 1).unwrap();
            let kv = Tensor::full(i as f32, (1usize, 1usize, 1usize, 2usize), &dev).unwrap();
            for layer in 0..cfg.num_hidden_layers {
                cache.write_at(layer, &kv, &kv).unwrap();
            }
            cache.advance(1);
        }

        let val_at = |t: &Tensor, row: usize| -> f32 {
            t.narrow(1, row, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0]
        };

        for layer in 0..cfg.num_hidden_layers {
            let (k, _v) = cache.view(layer, n).unwrap();
            let stored = k.dims()[1];
            let is_sliding = cfg.layer_types[layer] == super::LayerType::SlidingAttention;

            assert_eq!(
                val_at(&k, stored - 1),
                (n - 1) as f32,
                "layer {layer} newest"
            );
            if is_sliding {
                assert!(
                    stored <= window + SLIDING_COMPACT_SLACK,
                    "sliding layer {layer} storage {stored} exceeded window+slack"
                );

                assert_eq!(
                    val_at(&k, stored - window),
                    (n - window) as f32,
                    "sliding layer {layer} oldest-in-window"
                );
            } else {
                assert_eq!(stored, n, "full layer {layer} must keep the whole context");
            }
        }
    }
}

#[cfg(test)]
mod paper_validation_attention_masks {
    use super::sdpa_with_mask;
    use candle_core::{Device, Tensor};

    fn probs_for(seq_q: usize, seq_k: usize, window: Option<usize>, user_mask: &[f32]) -> Vec<f32> {
        let device = Device::Cpu;
        let d = seq_k;
        let q = Tensor::zeros((1, seq_q, 1, d), candle_core::DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, seq_k, 1, d), candle_core::DType::F32, &device).unwrap();
        let mut v_host = vec![0f32; seq_k * d];
        for j in 0..seq_k {
            v_host[j * d + j] = 1.0;
        }
        let v = Tensor::from_vec(v_host, (1, seq_k, 1, d), &device).unwrap();
        let mask = Tensor::from_vec(user_mask.to_vec(), (seq_q, seq_k), &device).unwrap();
        let out = sdpa_with_mask(&q, &k, &v, 1, 1, d, seq_q, &mask, window).unwrap();
        out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn sliding_window_admits_exactly_w_keys_masks_distance_w_and_composes_conjunctively_with_the_user_mask(
    ) {
        let (seq_q, seq_k, w) = (4usize, 6usize, 2usize);
        let offset = seq_k - seq_q;
        let probs = probs_for(seq_q, seq_k, Some(w), &vec![1.0; seq_q * seq_k]);
        for i in 0..seq_q {
            let qi = i + offset;
            let allowed: Vec<usize> = (0..seq_k).filter(|&j| j <= qi && qi - j < w).collect();
            assert_eq!(allowed.len(), w.min(qi + 1), "window must admit W keys");
            for j in 0..seq_k {
                let p = probs[i * seq_k + j];
                if allowed.contains(&j) {
                    let want = 1.0 / allowed.len() as f32;
                    assert!(
                        (p - want).abs() < 1e-5,
                        "row {i} (qi={qi}) key {j}: prob {p}, want {want}"
                    );
                } else {
                    assert!(
                        p.abs() < 1e-6,
                        "row {i} (qi={qi}) attends key {j} outside the window (p={p}); \
                         the token at distance exactly W must be masked"
                    );
                }
            }
        }

        let (seq_q, seq_k, w) = (1usize, 6usize, 3usize);
        let probs = probs_for(seq_q, seq_k, Some(w), &vec![1.0; seq_q * seq_k]);
        assert!(probs[2].abs() < 1e-6, "distance W must be masked");
        assert!(probs[3] > 0.3, "distance W-1 must attend");
        assert!(probs[5] > 0.3, "current token must attend");

        let (seq_q, seq_k, w) = (4usize, 6usize, 3usize);
        let offset = seq_k - seq_q;
        let mut user = vec![1.0f32; seq_q * seq_k];

        user[3 * seq_k + 5] = 0.0;
        let probs = probs_for(seq_q, seq_k, Some(w), &user);
        let i = 3usize;
        let qi = i + offset;
        let allowed: Vec<usize> = (0..seq_k)
            .filter(|&j| j <= qi && qi - j < w && !(i == 3 && j == 5))
            .collect();
        for j in 0..seq_k {
            let p = probs[i * seq_k + j];
            if allowed.contains(&j) {
                assert!((p - 1.0 / allowed.len() as f32).abs() < 1e-4);
            } else {
                assert!(
                    p.abs() < 1e-4,
                    "row {i} key {j} escaped mask/window conjunction"
                );
            }
        }
    }
}

#[cfg(test)]
mod prefill_shadow_tests {
    use super::{causal_attention_chunked, prefill_shadow_extend};
    use candle_core::{Device, Tensor};

    fn mk(seed: f32, s: usize, h: usize, hd: usize, dev: &Device) -> Tensor {
        let data: Vec<f32> = (0..s * h * hd)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();
        Tensor::from_vec(data, (1usize, s, h, hd), dev).unwrap()
    }

    #[test]
    fn shadow_and_sliding_shadow_bookkeeping_extend_keep_the_tail_and_fall_back_on_a_position_mismatch(
    ) {
        let dev = Device::Cpu;
        let c1 = mk(0.0, 4, 2, 8, &dev);
        let s = prefill_shadow_extend(None, &c1, &c1, 0).unwrap().unwrap();
        assert_eq!(s.0.dims(), [1, 4, 2, 8]);
        assert_eq!(s.1.dims(), [1, 4, 2, 8]);
        let c2 = mk(1.0, 3, 2, 8, &dev);
        let s2 = prefill_shadow_extend(Some(s), &c2, &c2, 4)
            .unwrap()
            .unwrap();
        assert_eq!(s2.0.dims(), [1, 7, 2, 8]);
        assert_eq!(s2.1.dims(), [1, 7, 2, 8]);
        let c3 = mk(2.0, 5, 2, 8, &dev);
        let stale = prefill_shadow_extend(Some(s2), &c3, &c3, 0)
            .unwrap()
            .unwrap();
        assert_eq!(stale.0.dims(), [1, 5, 2, 8]);

        let dev = Device::Cpu;
        let c = mk(0.0, 4, 2, 8, &dev);
        let s = prefill_shadow_extend(None, &c, &c, 0).unwrap().unwrap();
        assert!(prefill_shadow_extend(Some(s), &c, &c, 5).unwrap().is_none());
        assert!(prefill_shadow_extend(None, &c, &c, 4).unwrap().is_none());

        use super::sliding_shadow_extend;
        let dev = Device::Cpu;
        let keep = 5usize;
        let c1 = mk(0.0, 6, 2, 8, &dev);
        let (ks, vs) = sliding_shadow_extend(None, &c1, &c1, 0, keep)
            .unwrap()
            .unwrap();
        assert_eq!(ks.dims(), [1, 6, 2, 8]);
        assert_eq!(vs.dims(), [1, 6, 2, 8]);
        let rows = ks.dims()[1];
        let tail = rows.min(keep);
        let kt = ks
            .narrow(1, rows - tail, tail)
            .unwrap()
            .contiguous()
            .unwrap();
        let vt = vs
            .narrow(1, rows - tail, tail)
            .unwrap()
            .contiguous()
            .unwrap();
        assert_eq!(kt.dims(), [1, 5, 2, 8]);

        let c2 = mk(1.0, 4, 2, 8, &dev);
        let (ks2, _vs2) = sliding_shadow_extend(Some((kt, vt)), &c2, &c2, 6, keep)
            .unwrap()
            .unwrap();
        assert_eq!(ks2.dims(), [1, 9, 2, 8]);

        let expected_tail = c1.narrow(1, 1, 5).unwrap().to_vec3::<f32>();
        let got_tail = ks2.narrow(1, 0, 5).unwrap().to_vec3::<f32>();
        assert_eq!(
            format!("{expected_tail:?}"),
            format!("{got_tail:?}"),
            "tail must be the last keep rows of prior context"
        );

        let dev = Device::Cpu;
        let keep = 5usize;
        let c1 = mk(0.0, 3, 2, 8, &dev);
        let (ks, vs) = sliding_shadow_extend(None, &c1, &c1, 0, keep)
            .unwrap()
            .unwrap();
        assert_eq!(ks.dims(), [1, 3, 2, 8]);
        let c2 = mk(1.0, 2, 2, 8, &dev);
        let (ks2, _) = sliding_shadow_extend(Some((ks, vs)), &c2, &c2, 3, keep)
            .unwrap()
            .unwrap();
        assert_eq!(ks2.dims(), [1, 5, 2, 8]);

        assert!(sliding_shadow_extend(None, &c2, &c2, 3, keep)
            .unwrap()
            .is_none());
        let stale = mk(2.0, 4, 2, 8, &dev);
        assert!(
            sliding_shadow_extend(Some((stale.clone(), stale)), &c2, &c2, 32, keep)
                .unwrap()
                .is_none()
        );
        let (kz, _) = sliding_shadow_extend(None, &c2, &c2, 7, 0)
            .unwrap()
            .unwrap();
        assert_eq!(kz.dims(), [1, 2, 2, 8]);
    }

    #[test]
    fn chunked_prefill_matches_single_shot() {
        let dev = Device::Cpu;
        let (n_q, n_kv, hd, seq, split) = (4usize, 2usize, 8usize, 7usize, 4usize);
        let q = mk(0.1, seq, n_q, hd, &dev);
        let kx = mk(0.2, seq, n_kv, hd, &dev);
        let vx = mk(0.3, seq, n_kv, hd, &dev);
        let full = causal_attention_chunked(&q, &kx, &vx, n_q, n_kv, hd, seq, 0).unwrap();

        let k1 = kx.narrow(1, 0, split).unwrap().contiguous().unwrap();
        let v1 = vx.narrow(1, 0, split).unwrap().contiguous().unwrap();
        let shadow = prefill_shadow_extend(None, &k1, &v1, 0).unwrap().unwrap();
        let k2 = kx
            .narrow(1, split, seq - split)
            .unwrap()
            .contiguous()
            .unwrap();
        let v2 = vx
            .narrow(1, split, seq - split)
            .unwrap()
            .contiguous()
            .unwrap();
        let (ks, vs) = prefill_shadow_extend(Some(shadow), &k2, &v2, split)
            .unwrap()
            .unwrap();
        assert_eq!(ks.dims(), [1, seq, n_kv, hd]);

        let q2 = q
            .narrow(1, split, seq - split)
            .unwrap()
            .contiguous()
            .unwrap();
        let out2 =
            causal_attention_chunked(&q2, &ks, &vs, n_q, n_kv, hd, seq - split, split).unwrap();
        let ref2 = full.narrow(1, split, seq - split).unwrap();
        let d = (out2 - ref2)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-5, "max diff {d}");
    }
}
