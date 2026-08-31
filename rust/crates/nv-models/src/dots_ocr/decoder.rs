use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::{load_linear_with_bias_names, load_rmsnorm};
use serde::Deserialize;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

use crate::deepseek_ocr::decoder::{banned_tokens_windowed_ngram, detect_loop, LoopDetection};

pub const IMGPAD_TOKEN_ID: u32 = 151665;
pub const IMG_TOKEN_ID: u32 = 151666;
pub const ENDOFIMG_TOKEN_ID: u32 = 151667;
pub const USER_TOKEN_ID: u32 = 151670;
pub const ENDOFUSER_TOKEN_ID: u32 = 151671;
pub const ASSISTANT_TOKEN_ID: u32 = 151672;
pub const ENDOFASSISTANT_TOKEN_ID: u32 = 151673;
pub const ENDOFTEXT_TOKEN_ID: u32 = 151643;

pub const EOS_TOKEN_IDS: [u32; 2] = [ENDOFTEXT_TOKEN_ID, ENDOFASSISTANT_TOKEN_ID];

#[derive(Clone, Debug, Deserialize)]
pub struct DotsDecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default)]
    pub head_dim: Option<usize>,
}

fn default_true() -> bool {
    true
}

fn default_image_token_id() -> u32 {
    IMGPAD_TOKEN_ID
}

impl DotsDecoderConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("deserialize dots.ocr config")
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

pub struct DotsKvCache {
    layers: Vec<(Tensor, Tensor)>,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    grow_by_concat_avoids_full_buffer_slice_assign: bool,
    bhsd: bool,
}

impl DotsKvCache {
    pub fn new(
        cfg: &DotsDecoderConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_mode(cfg, max_seq_len, device, dtype, false)
    }

    pub fn new_with_mode(
        cfg: &DotsDecoderConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        grow_by_concat: bool,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let bhsd = !matches!(device, Device::Cuda(_));
        let alloc_len = if grow_by_concat { 0 } else { max_seq_len };
        let shape = if bhsd {
            (1usize, cfg.num_key_value_heads, alloc_len, head_dim)
        } else {
            (1usize, alloc_len, cfg.num_key_value_heads, head_dim)
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push((
                Tensor::zeros(shape, dtype, device)?,
                Tensor::zeros(shape, dtype, device)?,
            ));
        }
        Ok(Self {
            layers,
            current_len: 0,
            max_seq_len,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            grow_by_concat_avoids_full_buffer_slice_assign: grow_by_concat,
            bhsd,
        })
    }

    pub fn bhsd(&self) -> bool {
        self.bhsd
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

    pub fn write_at(&mut self, layer: usize, start: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let t = k.dim(1)?;
        let end = start + t;
        anyhow::ensure!(
            end <= self.max_seq_len,
            "kv cache overflow: {end} > {}",
            self.max_seq_len
        );
        let seq_dim = if self.bhsd { 2 } else { 1 };
        let (k, v) = if self.bhsd {
            if t == 1 {
                let shape = (1, self.n_kv_heads, 1, self.head_dim);
                (k.reshape(shape)?, v.reshape(shape)?)
            } else {
                (
                    k.permute((0, 2, 1, 3))?.contiguous()?,
                    v.permute((0, 2, 1, 3))?.contiguous()?,
                )
            }
        } else {
            (k.clone(), v.clone())
        };
        if self.grow_by_concat_avoids_full_buffer_slice_assign {
            let (kb, vb) = &self.layers[layer];
            let (ku, vu) = if kb.dim(seq_dim)? == 0 {
                (k, v)
            } else {
                (
                    Tensor::cat(&[kb, &k], seq_dim)?,
                    Tensor::cat(&[vb, &v], seq_dim)?,
                )
            };
            self.layers[layer] = (ku, vu);
            return Ok(());
        }
        let (kb, vb) = &self.layers[layer];
        kb.slice_set(&k.contiguous()?, seq_dim, start)?;
        vb.slice_set(&v.contiguous()?, seq_dim, start)?;
        Ok(())
    }

    /// (1, len, h_kv, d) on CUDA; (1, h_kv, len, d) when `bhsd()`.
    pub fn view(&self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let (k, v) = &self.layers[layer];
        let seq_dim = if self.bhsd { 2 } else { 1 };
        Ok((k.narrow(seq_dim, 0, len)?, v.narrow(seq_dim, 0, len)?))
    }

    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
}

struct DotsLayer {
    pre_attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    pre_mlp_norm: RmsNorm,
    mlp: Mlp,
}

pub struct DotsDecoder {
    cfg: DotsDecoderConfig,
    embed: Tensor,
    layers: Vec<DotsLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    rope_span: usize,
    device: Device,
    dtype: DType,
}

trait LinearFactory {
    fn load(
        &self,
        weights: &dyn nv_weights::TensorSource,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        bias: Option<&str>,
    ) -> Result<Linear>;
}

struct Bf16Factory {
    dtype: DType,
}

impl LinearFactory for Bf16Factory {
    fn load(
        &self,
        weights: &dyn nv_weights::TensorSource,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        bias: Option<&str>,
    ) -> Result<Linear> {
        load_linear_with_bias_names(weights, name, out_dim, in_dim, bias, self.dtype)
    }
}

struct GgmlFactory {
    dtype: DType,
    gdt: candle_core::quantized::GgmlDType,
}

impl LinearFactory for GgmlFactory {
    fn load(
        &self,
        weights: &dyn nv_weights::TensorSource,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        bias: Option<&str>,
    ) -> Result<Linear> {
        if !in_dim.is_multiple_of(self.gdt.block_size()) {
            return load_linear_with_bias_names(weights, name, out_dim, in_dim, bias, self.dtype);
        }
        let w = weights
            .get(name, DType::BF16)
            .with_context(|| format!("load {name}"))?;
        anyhow::ensure!(
            w.dims2()? == (out_dim, in_dim),
            "{name}: expected [{out_dim}, {in_dim}], got {:?}",
            w.dims()
        );
        let b = match bias {
            Some(bn) => Some(
                weights
                    .get(bn, self.dtype)
                    .with_context(|| format!("load {bn}"))?,
            ),
            None => None,
        };
        Linear::from_bf16_quantized_ggml(&w.contiguous()?, b, self.gdt)
            .with_context(|| format!("quantize {name} to {:?}", self.gdt))
    }
}

#[cfg(feature = "cuda")]
struct Nvfp4Factory {
    dtype: DType,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: Device,
}

#[cfg(feature = "cuda")]
impl LinearFactory for Nvfp4Factory {
    fn load(
        &self,
        weights: &dyn nv_weights::TensorSource,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        bias: Option<&str>,
    ) -> Result<Linear> {
        use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
        if out_dim < MIN_TILE || in_dim < MIN_TILE || !in_dim.is_multiple_of(BLOCK_SIZE) {
            return load_linear_with_bias_names(weights, name, out_dim, in_dim, bias, self.dtype);
        }
        let w = weights
            .get(name, DType::BF16)
            .with_context(|| format!("load {name}"))?;
        anyhow::ensure!(
            w.dims2()? == (out_dim, in_dim),
            "{name}: expected [{out_dim}, {in_dim}], got {:?}",
            w.dims()
        );
        let b = match bias {
            Some(bn) => Some(
                weights
                    .get(bn, DType::BF16)
                    .with_context(|| format!("load {bn}"))?,
            ),
            None => None,
        };
        Linear::from_bf16_quantized_nvfp4(&w.contiguous()?, b, &self.device, self.runner.clone())
            .with_context(|| format!("quantize {name} to nvfp4"))
    }
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if matches!(q.device(), Device::Cuda(_)) {
        return nv_layers::attn::flash_attn(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            cfg,
        );
    }
    sdpa(q, k, v, cfg)
}

fn decode_attention_grouped(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    let (_b, _one, h, d) = q.dims4()?;
    let (_bk, h_kv, s_k, dk) = k.dims4()?;
    anyhow::ensure!(
        h == cfg.num_heads && h_kv == cfg.num_kv_heads && d == dk && h % h_kv == 0,
        "decode_attention_grouped: bad shapes q={:?} k={:?}",
        q.dims(),
        k.dims()
    );
    let groups = h / h_kv;
    let qg = q
        .reshape((h_kv, groups, d))?
        .affine(cfg.softmax_scale as f64, 0.0)?;
    let kg = k.reshape((h_kv, s_k, d))?;
    let vg = v.reshape((h_kv, s_k, d))?;
    let scores = qg.matmul(&kg.transpose(1, 2)?)?.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(q.dtype())?;
    let out = probs.matmul(&vg)?;
    out.reshape((1, 1, h, d)).map_err(|e| anyhow::anyhow!(e))
}

impl DotsDecoder {
    pub fn from_loader(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        device: &Device,
        dtype: DType,
        rope_max_seq_len: usize,
    ) -> Result<Self> {
        let factory = Bf16Factory { dtype };
        Self::build(cfg, weights, device, dtype, rope_max_seq_len, &factory)
    }

    pub fn from_loader_ggml(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        device: &Device,
        dtype: DType,
        gdt: candle_core::quantized::GgmlDType,
        rope_max_seq_len: usize,
    ) -> Result<Self> {
        let factory = GgmlFactory { dtype, gdt };
        Self::build(cfg, weights, device, dtype, rope_max_seq_len, &factory)
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_nvfp4(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        device: &Device,
        rope_max_seq_len: usize,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("from_loader_nvfp4 requires a CUDA device"),
        };
        let runner = Arc::new(Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
            dev.cuda_stream(),
        )?));
        let factory = Nvfp4Factory {
            dtype: DType::BF16,
            runner,
            device: device.clone(),
        };
        Self::build(
            cfg,
            weights,
            device,
            DType::BF16,
            rope_max_seq_len,
            &factory,
        )
    }

    fn build(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        device: &Device,
        dtype: DType,
        rope_max_seq_len: usize,
        factory: &dyn LinearFactory,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let qd = cfg.num_attention_heads * head_dim;
        let kvd = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let embed = weights
            .get("model.embed_tokens.weight", dtype)
            .context("load model.embed_tokens.weight")?;
        anyhow::ensure!(
            embed.dims2()? == (cfg.vocab_size, hidden),
            "embed_tokens: got {:?}",
            embed.dims()
        );

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let qb = cfg
                .attention_bias
                .then(|| format!("{p}.self_attn.q_proj.bias"));
            let kb = cfg
                .attention_bias
                .then(|| format!("{p}.self_attn.k_proj.bias"));
            let vb = cfg
                .attention_bias
                .then(|| format!("{p}.self_attn.v_proj.bias"));
            layers.push(DotsLayer {
                pre_attn_norm: load_rmsnorm(
                    weights,
                    &format!("{p}.input_layernorm.weight"),
                    hidden,
                    eps,
                    DType::F32,
                )?,
                q_proj: factory.load(
                    weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    qd,
                    hidden,
                    qb.as_deref(),
                )?,
                k_proj: factory.load(
                    weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kvd,
                    hidden,
                    kb.as_deref(),
                )?,
                v_proj: factory.load(
                    weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kvd,
                    hidden,
                    vb.as_deref(),
                )?,
                o_proj: factory.load(
                    weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                    qd,
                    None,
                )?,
                pre_mlp_norm: load_rmsnorm(
                    weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    hidden,
                    eps,
                    DType::F32,
                )?,
                mlp: Mlp::new(
                    factory.load(
                        weights,
                        &format!("{p}.mlp.gate_proj.weight"),
                        inter,
                        hidden,
                        None,
                    )?,
                    factory.load(
                        weights,
                        &format!("{p}.mlp.up_proj.weight"),
                        inter,
                        hidden,
                        None,
                    )?,
                    factory.load(
                        weights,
                        &format!("{p}.mlp.down_proj.weight"),
                        hidden,
                        inter,
                        None,
                    )?,
                )?,
            });
        }

        let final_norm = load_rmsnorm(weights, "model.norm.weight", hidden, eps, DType::F32)?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed.contiguous()?, None)?
        } else {
            load_linear_with_bias_names(
                weights,
                "lm_head.weight",
                cfg.vocab_size,
                hidden,
                None,
                dtype,
            )?
        };
        let rope_span = rope_max_seq_len.clamp(1, cfg.max_position_embeddings);
        let rope = Rope::new(
            RopeConfig {
                head_dim,
                max_seq_len: rope_span,
                base: cfg.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;

        Ok(Self {
            cfg,
            embed,
            layers,
            final_norm,
            lm_head,
            rope,
            rope_span,
            device: device.clone(),
            dtype,
        })
    }

    pub fn rope_span(&self) -> usize {
        self.rope_span
    }

    pub fn config(&self) -> &DotsDecoderConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<DotsKvCache> {
        DotsKvCache::new(&self.cfg, max_seq_len, &self.device, self.dtype)
    }

    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        let idx =
            Tensor::from_slice(tokens, tokens.len(), &Device::Cpu)?.to_device(&self.device)?;
        let e = self.embed.index_select(&idx, 0)?;
        Ok(e.reshape((1, tokens.len(), self.cfg.hidden_size))?
            .to_dtype(self.dtype)?)
    }

    pub fn forward_hidden(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut DotsKvCache,
    ) -> Result<Tensor> {
        let (b, seq, hidden) = embeds.dims3()?;
        anyhow::ensure!(b == 1 && hidden == self.cfg.hidden_size, "bad embeds shape");
        let head_dim = self.cfg.head_dim();
        let n_heads = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let positions: Vec<u32> = (0..seq).map(|i| (start_pos + i) as u32).collect();
        let positions = Tensor::from_vec(positions, (1, seq), &self.device)?;
        let new_total = start_pos + seq;
        let attn_cfg = AttnConfig {
            num_heads: n_heads,
            num_kv_heads: n_kv,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            causal: true,
        };

        let mut x = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = layer.pre_attn_norm.forward(&x)?.to_dtype(self.dtype)?;
            let q = layer
                .q_proj
                .forward(&normed)?
                .reshape((1, seq, n_heads, head_dim))?;
            let k = layer
                .k_proj
                .forward(&normed)?
                .reshape((1, seq, n_kv, head_dim))?;
            let v = layer
                .v_proj
                .forward(&normed)?
                .reshape((1, seq, n_kv, head_dim))?;
            let (q_rot, k_rot) = self.rope.apply(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &positions,
            )?;
            let q = q_rot.to_dtype(self.dtype)?;
            let k = k_rot.to_dtype(self.dtype)?;
            cache.write_at(i, start_pos, &k.contiguous()?, &v.contiguous()?)?;
            let (k_full, v_full) = cache.view(i, new_total)?;
            let attn = if cache.bhsd() {
                if matches!(self.device, Device::Metal(_)) {
                    let q4 = if seq == 1 {
                        q.reshape((1, n_heads, 1, head_dim))?
                    } else {
                        q.permute((0, 2, 1, 3))?.contiguous()?
                    };
                    let (kc, vc) = (k_full.contiguous()?, v_full.contiguous()?);
                    let out = candle_nn::ops::sdpa(
                        &q4,
                        &kc,
                        &vc,
                        None,
                        seq > 1,
                        attn_cfg.softmax_scale,
                        1.0,
                    )?;
                    if seq == 1 {
                        out.reshape((1, 1, n_heads, head_dim))?
                    } else {
                        out.permute((0, 2, 1, 3))?.contiguous()?
                    }
                } else if seq == 1 {
                    decode_attention_grouped(&q, &k_full, &v_full, &attn_cfg)?
                } else {
                    let k_sdpa = k_full.permute((0, 2, 1, 3))?.contiguous()?;
                    let v_sdpa = v_full.permute((0, 2, 1, 3))?.contiguous()?;
                    attention(&q, &k_sdpa, &v_sdpa, &attn_cfg)?
                }
            } else {
                attention(&q, &k_full, &v_full, &attn_cfg)?
            };
            let attn = attn.reshape((1, seq, n_heads * head_dim))?;
            let attn = layer.o_proj.forward(&attn)?;
            x = x.add(&attn)?;
            let normed2 = layer.pre_mlp_norm.forward(&x)?.to_dtype(self.dtype)?;
            x = x.add(&layer.mlp.forward(&normed2)?)?;
        }
        cache.advance(seq);
        Ok(self.final_norm.forward(&x)?.to_dtype(self.dtype)?)
    }

    pub fn last_logits(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut DotsKvCache,
    ) -> Result<Vec<f32>> {
        let h = self.forward_hidden(embeds, start_pos, cache)?;
        let seq = h.dim(1)?;
        let last = h.narrow(1, seq - 1, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&last)?;
        Ok(logits
            .reshape(self.cfg.vocab_size)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
    }
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub ngram_size: Option<usize>,
    pub ngram_window: Option<usize>,
    pub ngram_whitelist: Vec<u32>,
    pub stop_on_loop: bool,
    pub eos_token_ids: Vec<u32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 16384,
            ngram_size: Some(30),
            ngram_window: Some(90),
            ngram_whitelist: Vec::new(),
            stop_on_loop: true,
            eos_token_ids: EOS_TOKEN_IDS.to_vec(),
        }
    }
}

#[derive(Debug)]
pub struct GenerateOutcome {
    pub tokens: Vec<u32>,
    pub loop_detection: Option<LoopDetection>,
    pub hit_eos: bool,
}

pub fn argmax_banned(logits: &[f32], banned: &[u32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if banned.contains(&(i as u32)) {
            continue;
        }
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PromptStyle {
    #[default]
    ChatTemplate,
    UserTurn,
}

impl PromptStyle {
    pub fn from_env() -> Self {
        match std::env::var("NV_DOTS_PROMPT_STYLE").as_deref() {
            Ok("user") | Ok("user-turn") => PromptStyle::UserTurn,
            _ => PromptStyle::ChatTemplate,
        }
    }
}

pub fn build_prompt_tokens(
    encode: impl Fn(&str) -> Result<Vec<u32>>,
    prompt: &str,
    n_vision_tokens: usize,
    style: PromptStyle,
) -> Result<(Vec<u32>, usize)> {
    let mut ids = Vec::with_capacity(n_vision_tokens + 64);
    match style {
        PromptStyle::ChatTemplate => ids.extend(encode(" ")?),
        PromptStyle::UserTurn => ids.push(USER_TOKEN_ID),
    }
    ids.push(IMG_TOKEN_ID);
    let vision_start = ids.len();
    ids.extend(std::iter::repeat_n(IMGPAD_TOKEN_ID, n_vision_tokens));
    ids.push(ENDOFIMG_TOKEN_ID);
    ids.extend(encode(prompt)?);
    if style == PromptStyle::UserTurn {
        ids.push(ENDOFUSER_TOKEN_ID);
    }
    ids.push(ASSISTANT_TOKEN_ID);
    Ok((ids, vision_start))
}

impl DotsDecoder {
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        vision_start: usize,
        vision_features: Option<&Tensor>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let prompt_len = prompt_tokens.len();
        let budget = prompt_len + opts.max_new_tokens + 1;
        anyhow::ensure!(
            prompt_len < self.rope_span,
            "prompt of {prompt_len} tokens exceeds the rope span {} (raise NV_DOTS_MAX_SEQ or lower NV_DOTS_MAX_PIXELS)",
            self.rope_span
        );
        let grow_kv = nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_DOTS_FAST_KV");
        let mut cache = DotsKvCache::new_with_mode(
            &self.cfg,
            budget.min(self.rope_span),
            &self.device,
            self.dtype,
            grow_kv,
        )?;
        let mut embeds = self.embed_tokens(prompt_tokens)?;
        if let Some(feats) = vision_features {
            let n = feats.dim(0)?;
            anyhow::ensure!(
                vision_start + n <= prompt_len,
                "vision span {vision_start}+{n} exceeds prompt {prompt_len}"
            );
            let feats = feats
                .reshape((1, n, self.cfg.hidden_size))?
                .to_dtype(self.dtype)?;
            embeds = embeds.slice_assign(
                &[
                    0..1,
                    vision_start..vision_start + n,
                    0..self.cfg.hidden_size,
                ],
                &feats,
            )?;
        }

        let mut logits = self.last_logits(&embeds, 0, &mut cache)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        for _ in 0..opts.max_new_tokens {
            let banned = match opts.ngram_size {
                Some(n) if n > 0 => banned_tokens_windowed_ngram(
                    &generated,
                    n,
                    opts.ngram_window,
                    &opts.ngram_whitelist,
                ),
                _ => Vec::new(),
            };
            let next = argmax_banned(&logits, &banned);
            if opts.eos_token_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            if opts.stop_on_loop
                && generated.len().is_multiple_of(64)
                && detect_loop(&generated).is_some()
            {
                break;
            }
            let pos = prompt_len + generated.len() - 1;
            if pos + 1 >= cache.max_seq_len() {
                break;
            }
            let step = self.embed_tokens(&[next])?;
            logits = self.last_logits(&step, pos, &mut cache)?;
        }
        let loop_detection = detect_loop(&generated);
        Ok(GenerateOutcome {
            tokens: generated,
            loop_detection,
            hit_eos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_style_matches_the_shipped_jinja_rendering() {
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(match s {
                " " => vec![220],
                _ => vec![7, 8, 9],
            })
        };
        let (ids, start) = build_prompt_tokens(encode, "x", 4, PromptStyle::ChatTemplate).unwrap();
        assert_eq!(start, 2);
        assert_eq!(
            ids,
            vec![
                220,
                IMG_TOKEN_ID,
                IMGPAD_TOKEN_ID,
                IMGPAD_TOKEN_ID,
                IMGPAD_TOKEN_ID,
                IMGPAD_TOKEN_ID,
                ENDOFIMG_TOKEN_ID,
                7,
                8,
                9,
                ASSISTANT_TOKEN_ID,
            ]
        );
        assert_eq!(ids[start..start + 4], [IMGPAD_TOKEN_ID; 4]);
        assert!(!ids.contains(&USER_TOKEN_ID));
        assert!(!ids.contains(&ENDOFUSER_TOKEN_ID));
    }

    #[test]
    fn user_turn_style_wraps_the_turn_in_user_markers() {
        let (ids, start) =
            build_prompt_tokens(|_| Ok(vec![7]), "x", 2, PromptStyle::UserTurn).unwrap();
        assert_eq!(start, 2);
        assert_eq!(
            ids,
            vec![
                USER_TOKEN_ID,
                IMG_TOKEN_ID,
                IMGPAD_TOKEN_ID,
                IMGPAD_TOKEN_ID,
                ENDOFIMG_TOKEN_ID,
                7,
                ENDOFUSER_TOKEN_ID,
                ASSISTANT_TOKEN_ID,
            ]
        );
    }

    #[test]
    fn config_parses_the_shipped_json() {
        let cfg = DotsDecoderConfig::from_hf_json_str(
            r#"{"hidden_size":1536,"num_hidden_layers":28,"num_attention_heads":12,
                "num_key_value_heads":2,"intermediate_size":8960,"vocab_size":151936,
                "max_position_embeddings":131072,"rope_theta":1000000,"rms_norm_eps":1e-06,
                "tie_word_embeddings":false,"attention_bias":true,"image_token_id":151665,
                "vision_config":{"embed_dim":1536}}"#,
        )
        .unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert!(cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.image_token_id, IMGPAD_TOKEN_ID);
    }

    #[test]
    fn argmax_skips_banned_ids() {
        let logits = vec![0.1, 5.0, 2.0, 4.0];
        assert_eq!(argmax_banned(&logits, &[]), 1);
        assert_eq!(argmax_banned(&logits, &[1]), 3);
        assert_eq!(argmax_banned(&logits, &[1, 3]), 2);
    }
}
