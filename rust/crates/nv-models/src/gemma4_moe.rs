use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::{GgufLoader, TensorSource, WeightLoader};
use std::path::Path;

use crate::gemma4::{
    mlp_forward, Gemma4Attention, Gemma4Cache, Gemma4Config, Gemma4KvCache, Gemma4Mlp, LayerType,
};
use crate::CausalLm;

#[derive(Clone, Debug)]
pub struct Gemma4MoeConfig {
    pub base: Gemma4Config,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub moe_intermediate_size: usize,
}

impl Gemma4MoeConfig {
    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let base = Gemma4Config::from_hf_json_str(s)?;
        if !base.enable_moe_block {
            anyhow::bail!("gemma4-moe: config has enable_moe_block=false");
        }
        let v: serde_json::Value = serde_json::from_str(s).context("parse gemma4 moe config")?;
        let text = v.get("text_config").unwrap_or(&v);
        let get_u = |key: &str| -> Result<usize> {
            text.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("gemma4-moe config missing {key}"))
        };
        let cfg = Self {
            base,
            num_experts: get_u("num_experts")?,
            top_k_experts: get_u("top_k_experts")?,
            moe_intermediate_size: get_u("moe_intermediate_size")?,
        };
        if cfg.top_k_experts == 0 || cfg.top_k_experts > cfg.num_experts {
            anyhow::bail!(
                "gemma4-moe: top_k_experts {} invalid for num_experts {}",
                cfg.top_k_experts,
                cfg.num_experts
            );
        }
        Ok(cfg)
    }
}

pub struct Gemma4MoeBlock {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub router_proj: Linear,
    pub router_scale: Tensor,
    pub per_expert_scale: Vec<f32>,

    pub gate_up: Tensor,
    pub down: Tensor,
}

impl Gemma4MoeBlock {
    pub fn forward(&self, x_router: &Tensor, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims().to_vec();
        if *in_dims.last().unwrap() != self.hidden_size {
            anyhow::bail!(
                "Gemma4MoeBlock: last dim {} != hidden_size {}",
                in_dims.last().unwrap(),
                self.hidden_size
            );
        }
        let n_tokens: usize = in_dims[..in_dims.len() - 1].iter().product();
        let in_dtype = x.dtype();
        let device = x.device().clone();
        let x_flat = x.reshape((n_tokens, self.hidden_size))?.contiguous()?;
        let xr_flat = x_router
            .reshape((n_tokens, self.hidden_size))?
            .contiguous()?;

        let xr_f32 = xr_flat.to_dtype(DType::F32)?;
        let mean_sq = xr_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps_t = Tensor::new(self.rms_norm_eps as f32, &device)?;
        let denom = mean_sq.broadcast_add(&eps_t)?.sqrt()?;
        let rn = xr_f32.broadcast_div(&denom)?.to_dtype(in_dtype)?;
        let scalar_root = 1.0 / (self.hidden_size as f64).sqrt();
        let router_in = rn
            .broadcast_mul(&self.router_scale.to_dtype(in_dtype)?)?
            .affine(scalar_root, 0.0)?;
        let logits = self
            .router_proj
            .forward(&router_in)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (sorted_logits, sorted_idx) = logits.sort_last_dim(false)?;
        let k = self.top_k;
        let top_logits = sorted_logits.narrow(1, 0, k)?.contiguous()?;
        let top_idx = sorted_idx.narrow(1, 0, k)?.contiguous()?;
        let top_weights = candle_nn::ops::softmax_last_dim(&top_logits)?.contiguous()?;

        let top_idx_host: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_weights_host: Vec<f32> = top_weights.flatten_all()?.to_vec1::<f32>()?;

        let mut expert_rows: Vec<Vec<u32>> = vec![Vec::new(); self.num_experts];
        let mut expert_w: Vec<Vec<f32>> = vec![Vec::new(); self.num_experts];
        for n in 0..n_tokens {
            for j in 0..k {
                let e = top_idx_host[n * k + j] as usize;
                expert_rows[e].push(n as u32);
                expert_w[e].push(top_weights_host[n * k + j] * self.per_expert_scale[e]);
            }
        }

        let mut acc = Tensor::zeros((n_tokens, self.hidden_size), DType::F32, &device)?;
        for e in 0..self.num_experts {
            let rows = &expert_rows[e];
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let idx_t = Tensor::from_vec(rows.clone(), m, &device)?;
            let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;

            let mlp_e = Gemma4Mlp {
                gate_up_proj: Linear::new(
                    self.gate_up.narrow(0, e, 1)?.squeeze(0)?.contiguous()?,
                    None,
                )?,
                down_proj: Linear::new(self.down.narrow(0, e, 1)?.squeeze(0)?.contiguous()?, None)?,
            };
            let y_e = mlp_forward(&mlp_e, &gathered)?.to_dtype(DType::F32)?;
            let w_t = Tensor::from_vec(expert_w[e].clone(), (m, 1), &device)?;
            let weighted = y_e.broadcast_mul(&w_t)?;
            acc = acc.index_add(&idx_t, &weighted, 0)?;
        }

        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(self.hidden_size);
        Ok(acc.reshape(out_dims)?.to_dtype(in_dtype)?)
    }
}

pub struct Gemma4MoeLayer {
    pub kind: LayerType,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm_1: RmsNorm,
    pub pre_feedforward_layernorm_2: RmsNorm,
    pub post_feedforward_layernorm_2: RmsNorm,
    pub layer_scalar_host: f32,
    pub self_attn: Gemma4Attention,
    pub mlp: Gemma4Mlp,
    pub moe: Gemma4MoeBlock,
}

pub struct Gemma4Moe {
    config: Gemma4MoeConfig,
    embed_weight: Tensor,
    layers: Vec<Gemma4MoeLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    sliding_rope: Rope,
    full_rope: Rope,
    embed_scale: f32,
    dtype: DType,
    device: Device,
}

impl Gemma4Moe {
    pub fn config(&self) -> &Gemma4MoeConfig {
        &self.config
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn layers(&self) -> &[Gemma4MoeLayer] {
        &self.layers
    }
    pub fn embed_weight(&self) -> &Tensor {
        &self.embed_weight
    }
    pub fn embed_scale(&self) -> f32 {
        self.embed_scale
    }

    pub fn from_loader(
        config: Gemma4MoeConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        Self::from_loader_dtype(config, weights, device, DType::BF16)
    }

    pub fn from_gguf(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let loader = GgufLoader::open(path, device)?;
        let config = crate::gemma4_gguf::gemma4_moe_config_from_gguf(&loader)?;
        Self::from_loader_dtype(config, &loader, device, dtype)
    }

    pub fn from_loader_dtype<S: TensorSource>(
        config: Gemma4MoeConfig,
        weights: &S,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let base = &config.base;
        let embed_name = "model.language_model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_name, dtype)
            .with_context(|| format!("load {embed_name}"))?;
        let embed_dims = embed_weight.dims();
        if embed_dims != [base.vocab_size, base.hidden_size] {
            anyhow::bail!(
                "gemma4-moe embed: expected [{}, {}], got {:?}",
                base.vocab_size,
                base.hidden_size,
                embed_dims
            );
        }

        let mut layers = Vec::with_capacity(base.num_hidden_layers);
        for i in 0..base.num_hidden_layers {
            layers.push(load_layer(&config, i, weights, device, dtype)?);
        }

        let final_norm = load_rmsnorm(
            weights,
            "model.language_model.norm.weight",
            base.hidden_size,
            base.rms_norm_eps,
            dtype,
        )?;
        let lm_head_weight = if base.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", dtype)
                .context("load lm_head.weight")?
        };
        let lm_head = Linear::new(lm_head_weight, None)?;

        let sliding_rope = build_rope(
            base.head_dim,
            base.rope_theta_for(LayerType::SlidingAttention),
            1.0,
            base.max_position_embeddings,
            device,
        )?;
        let full_rope = build_rope(
            base.global_head_dim,
            base.rope_theta_for(LayerType::FullAttention),
            base.rope_partial_factor_for(LayerType::FullAttention),
            base.max_position_embeddings,
            device,
        )?;
        let embed_scale = (base.hidden_size as f32).sqrt();

        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            sliding_rope,
            full_rope,
            embed_scale,
            dtype,
            device: device.clone(),
        })
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<Gemma4KvCache> {
        Gemma4KvCache::new(&self.config.base, max_seq_len, &self.device, self.dtype)
    }

    pub fn forward_with_cache<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        self.forward_with_cache_body(tokens, positions, cache, None)
    }

    pub fn forward_with_cache_embeds<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        embeds: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        self.forward_with_cache_body(tokens, positions, cache, Some(embeds))
    }

    fn forward_with_cache_body<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        embeds_override: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!("Gemma4Moe.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Gemma4Moe.forward: positions must be [{}], got {:?}",
                seq,
                positions.dims()
            );
        }
        let base = &self.config.base;

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let mut hidden = match embeds_override {
            None => {
                let x = self
                    .embed_weight
                    .index_select(&tokens_flat, 0)?
                    .reshape((1usize, seq, base.hidden_size))?
                    .to_dtype(self.dtype)?;
                x.affine(self.embed_scale as f64, 0.0)?
            }
            Some(e) => {
                anyhow::ensure!(
                    e.dims() == [seq, base.hidden_size],
                    "Gemma4Moe.forward: embeds override must be [{seq}, {}] pre-scaled rows, got {:?}",
                    base.hidden_size,
                    e.dims()
                );
                e.to_dtype(self.dtype)?
                    .reshape((1usize, seq, base.hidden_size))?
            }
        };

        let write_start = cache.current_len();
        let new_total = write_start + seq;
        cache.prepare_for_decode(write_start, new_total)?;

        for li in 0..self.layers.len() {
            hidden =
                self.layer_forward(li, &hidden, positions, cache, seq, write_start, new_total)?;
        }
        cache.advance(seq);

        let normed = self.final_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        let cap = base.final_logit_softcapping;
        if cap > 0.0 {
            Ok(logits
                .affine(1.0 / cap as f64, 0.0)?
                .tanh()?
                .affine(cap as f64, 0.0)?)
        } else {
            Ok(logits)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward<C: Gemma4Cache>(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        seq: usize,
        write_start: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];

        let normed_pre_attn = layer.input_layernorm.forward(x)?;
        let attn_out = self.attention_forward(
            idx,
            &normed_pre_attn,
            positions,
            cache,
            seq,
            write_start,
            new_total,
        )?;
        let attn_post = layer.post_attention_layernorm.forward(&attn_out)?;

        let after_attn = attn_post.add(x)?;
        let normed_pre_mlp = layer.pre_feedforward_layernorm.forward(&after_attn)?;

        let dense_out = mlp_forward(&layer.mlp, &normed_pre_mlp)?;
        let h1 = layer.post_feedforward_layernorm_1.forward(&dense_out)?;

        let moe_in = layer.pre_feedforward_layernorm_2.forward(&after_attn)?;
        let moe_out = layer.moe.forward(&after_attn, &moe_in)?;
        let h2 = layer.post_feedforward_layernorm_2.forward(&moe_out)?;

        let combined = layer.post_feedforward_layernorm.forward(&h1.add(&h2)?)?;
        Ok(after_attn
            .add(&combined)?
            .affine(layer.layer_scalar_host as f64, 0.0)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward<C: Gemma4Cache>(
        &self,
        layer_idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        seq: usize,
        write_start: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let base = &self.config.base;
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = base.head_dim_for(kind);
        let n_q = base.num_attention_heads;
        let n_kv = base.num_kv_heads_for(kind);
        let rope = match kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        let window = match kind {
            LayerType::SlidingAttention => Some(base.sliding_window),
            LayerType::FullAttention => None,
        };

        let (q_raw, k_raw, v_raw) = attn.qkv_forward(x)?;
        let q = q_raw.reshape((1usize, seq, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k = k_raw.reshape((1usize, seq, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
        let v = v_raw.reshape((1usize, seq, n_kv, head_dim))?;
        let v_normed = attn.v_norm.forward(&v)?;

        let (q_rot, k_rot) = rope.apply(&q_normed, &k_normed, positions)?;
        let q_rot = q_rot.contiguous()?;
        let k_rot = k_rot.contiguous()?;
        let v_for_cache = v_normed.contiguous()?;

        cache.write_at(layer_idx, &k_rot, &v_for_cache)?;
        let (k_full, v_full) = cache.view(layer_idx, new_total)?;

        let attn_out = naive_sdpa(
            &q_rot,
            &k_full,
            &v_full,
            n_q,
            n_kv,
            head_dim,
            seq,
            write_start,
            new_total,
            window,
        )?;

        let attn_out_flat = attn_out
            .to_dtype(x.dtype())?
            .reshape((1usize, seq, n_q * head_dim))?;
        attn.o_proj.forward(&attn_out_flat)
    }
}

#[allow(clippy::too_many_arguments)]
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    write_start: usize,
    new_total: usize,
    window: Option<usize>,
) -> Result<Tensor> {
    let device = q.device().clone();
    let stored = k.dims()[1];
    let q3 = q
        .to_dtype(DType::F32)?
        .reshape((seq, n_q, head_dim))?
        .transpose(0, 1)?
        .contiguous()?;
    let k3 = k
        .to_dtype(DType::F32)?
        .reshape((stored, n_kv, head_dim))?
        .transpose(0, 1)?
        .contiguous()?;
    let v3 = v
        .to_dtype(DType::F32)?
        .reshape((stored, n_kv, head_dim))?
        .transpose(0, 1)?
        .contiguous()?;

    let group = n_q / n_kv;
    let map: Vec<u32> = (0..n_q).map(|h| (h / group) as u32).collect();
    let map_t = Tensor::from_vec(map, n_q, &device)?;
    let k3e = k3.index_select(&map_t, 0)?.contiguous()?;
    let v3e = v3.index_select(&map_t, 0)?.contiguous()?;

    let mut mask = vec![0f32; seq * stored];
    let key_base = new_total - stored;
    for i in 0..seq {
        let qpos = (write_start + i) as i64;
        for j in 0..stored {
            let kpos = (key_base + j) as i64;
            let visible = kpos <= qpos
                && match window {
                    Some(w) => qpos - kpos < w as i64,
                    None => true,
                };
            if !visible {
                mask[i * stored + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from_vec(mask, (1usize, seq, stored), &device)?;

    let raw_scores = q3.matmul(&k3e.transpose(1, 2)?.contiguous()?)?;

    let scores = if window.is_none() && seq > 1 {
        if let Some(cfg) = crate::gemma4::xattn_cfg() {
            let key_base = new_total - stored;
            let (xbias, kept, cand) = crate::gemma4::xattn_prefill_bias(
                &raw_scores,
                seq,
                stored,
                write_start,
                key_base,
                &cfg,
            )?;
            if crate::gemma4::xattn_stats_enabled() {
                eprintln!(
                    "[xattn] full-attn prefill seq={seq} stored={stored} block={} stride={} thr={} kept={kept}/{cand} keep_frac={:.4}",
                    cfg.block,
                    cfg.stride,
                    cfg.threshold,
                    kept as f64 / (cand.max(1) as f64)
                );
            }
            raw_scores.broadcast_add(&mask_t)?.broadcast_add(&xbias)?
        } else {
            raw_scores.broadcast_add(&mask_t)?
        }
    } else {
        raw_scores.broadcast_add(&mask_t)?
    };
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v3e)?;
    Ok(out.transpose(0, 1)?.contiguous()?)
}

fn load_layer<S: TensorSource>(
    config: &Gemma4MoeConfig,
    idx: usize,
    weights: &S,
    device: &Device,
    dtype: DType,
) -> Result<Gemma4MoeLayer> {
    let base = &config.base;
    let prefix = format!("model.language_model.layers.{idx}");
    let hidden = base.hidden_size;
    let eps = base.rms_norm_eps;
    let kind = base.layer_kind(idx);
    let head_dim = base.head_dim_for(kind);
    let n_q = base.num_attention_heads;
    let n_kv = base.num_kv_heads_for(kind);
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;

    let norm = |name: &str, dim: usize| -> Result<RmsNorm> {
        load_rmsnorm(weights, &format!("{prefix}.{name}.weight"), dim, eps, dtype)
    };
    let input_layernorm = norm("input_layernorm", hidden)?;
    let post_attention_layernorm = norm("post_attention_layernorm", hidden)?;
    let pre_feedforward_layernorm = norm("pre_feedforward_layernorm", hidden)?;
    let post_feedforward_layernorm = norm("post_feedforward_layernorm", hidden)?;
    let post_feedforward_layernorm_1 = norm("post_feedforward_layernorm_1", hidden)?;
    let pre_feedforward_layernorm_2 = norm("pre_feedforward_layernorm_2", hidden)?;
    let post_feedforward_layernorm_2 = norm("post_feedforward_layernorm_2", hidden)?;

    let layer_scalar = weights
        .get(&format!("{prefix}.layer_scalar"), dtype)
        .with_context(|| format!("load {prefix}.layer_scalar"))?;
    if layer_scalar.dims() != [1] {
        anyhow::bail!(
            "{prefix}.layer_scalar expected [1], got {:?}",
            layer_scalar.dims()
        );
    }
    let layer_scalar_host = layer_scalar
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let has_v = !matches!(
        (kind, base.attention_k_eq_v),
        (LayerType::FullAttention, true)
    );
    let get_proj = |name: &str, out_f: usize, in_f: usize| -> Result<Tensor> {
        let t = weights
            .get(&format!("{prefix}.self_attn.{name}.weight"), dtype)
            .with_context(|| format!("load {prefix}.self_attn.{name}.weight"))?;
        if t.dims() != [out_f, in_f] {
            anyhow::bail!(
                "{prefix}.self_attn.{name}: expected [{out_f}, {in_f}], got {:?}",
                t.dims()
            );
        }
        Ok(t)
    };
    let q_w = get_proj("q_proj", q_dim, hidden)?;
    let k_w = get_proj("k_proj", kv_dim, hidden)?;
    let fused = if has_v {
        let v_w = get_proj("v_proj", kv_dim, hidden)?;
        Tensor::cat(&[&q_w, &k_w, &v_w], 0)?.contiguous()?
    } else {
        Tensor::cat(&[&q_w, &k_w], 0)?.contiguous()?
    };
    let qkv_proj = Linear::new(fused, None)?;
    let o_proj = Linear::new(get_proj("o_proj", hidden, q_dim)?, None)?;
    let q_norm = norm("self_attn.q_norm", head_dim)?;
    let k_norm = norm("self_attn.k_norm", head_dim)?;
    let v_norm = RmsNorm::new(Tensor::ones(head_dim, dtype, device)?, eps);
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
        #[cfg(feature = "cuda")]
        qkv_prefill_fp4: None,
        #[cfg(feature = "cuda")]
        o_prefill_fp4: None,
    };

    let inter = base.intermediate_size;
    let gate_w = weights.get(&format!("{prefix}.mlp.gate_proj.weight"), dtype)?;
    let up_w = weights.get(&format!("{prefix}.mlp.up_proj.weight"), dtype)?;
    if gate_w.dims() != [inter, hidden] || up_w.dims() != [inter, hidden] {
        anyhow::bail!(
            "{prefix}.mlp gate/up: expected [{inter}, {hidden}], got {:?} / {:?}",
            gate_w.dims(),
            up_w.dims()
        );
    }
    let down_w = weights.get(&format!("{prefix}.mlp.down_proj.weight"), dtype)?;
    if down_w.dims() != [hidden, inter] {
        anyhow::bail!(
            "{prefix}.mlp.down_proj: expected [{hidden}, {inter}], got {:?}",
            down_w.dims()
        );
    }
    let mlp = Gemma4Mlp {
        gate_up_proj: Linear::new(Tensor::cat(&[&gate_w, &up_w], 0)?.contiguous()?, None)?,
        down_proj: Linear::new(down_w, None)?,
    };

    let moe = load_moe_block(config, &prefix, weights, dtype)?;

    Ok(Gemma4MoeLayer {
        kind,
        input_layernorm,
        post_attention_layernorm,
        pre_feedforward_layernorm,
        post_feedforward_layernorm,
        post_feedforward_layernorm_1,
        pre_feedforward_layernorm_2,
        post_feedforward_layernorm_2,
        layer_scalar_host,
        self_attn,
        mlp,
        moe,
    })
}

fn load_moe_block<S: TensorSource>(
    config: &Gemma4MoeConfig,
    prefix: &str,
    weights: &S,
    dtype: DType,
) -> Result<Gemma4MoeBlock> {
    let hidden = config.base.hidden_size;
    let n_e = config.num_experts;
    let inter = config.moe_intermediate_size;

    let router_proj_w = weights
        .get(&format!("{prefix}.router.proj.weight"), dtype)
        .with_context(|| format!("load {prefix}.router.proj.weight"))?;
    if router_proj_w.dims() != [n_e, hidden] {
        anyhow::bail!(
            "{prefix}.router.proj: expected [{n_e}, {hidden}], got {:?}",
            router_proj_w.dims()
        );
    }
    let router_scale = weights.get(&format!("{prefix}.router.scale"), dtype)?;
    if router_scale.dims() != [hidden] {
        anyhow::bail!(
            "{prefix}.router.scale: expected [{hidden}], got {:?}",
            router_scale.dims()
        );
    }
    let per_expert_scale = weights
        .get(&format!("{prefix}.router.per_expert_scale"), DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if per_expert_scale.len() != n_e {
        anyhow::bail!(
            "{prefix}.router.per_expert_scale: expected [{n_e}], got [{}]",
            per_expert_scale.len()
        );
    }

    let gate_up = weights
        .get(&format!("{prefix}.experts.gate_up_proj"), dtype)
        .with_context(|| format!("load {prefix}.experts.gate_up_proj"))?;
    if gate_up.dims() != [n_e, 2 * inter, hidden] {
        anyhow::bail!(
            "{prefix}.experts.gate_up_proj: expected [{n_e}, {}, {hidden}], got {:?}",
            2 * inter,
            gate_up.dims()
        );
    }
    let down = weights
        .get(&format!("{prefix}.experts.down_proj"), dtype)
        .with_context(|| format!("load {prefix}.experts.down_proj"))?;
    if down.dims() != [n_e, hidden, inter] {
        anyhow::bail!(
            "{prefix}.experts.down_proj: expected [{n_e}, {hidden}, {inter}], got {:?}",
            down.dims()
        );
    }

    Ok(Gemma4MoeBlock {
        num_experts: n_e,
        top_k: config.top_k_experts,
        hidden_size: hidden,
        moe_intermediate_size: inter,
        rms_norm_eps: config.base.rms_norm_eps,
        router_proj: Linear::new(router_proj_w, None)?,
        router_scale,
        per_expert_scale,
        gate_up: gate_up.contiguous()?,
        down: down.contiguous()?,
    })
}

fn load_rmsnorm<S: TensorSource>(
    weights: &S,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    if w.dims() != [dim] {
        anyhow::bail!("{name}: expected [{dim}], got {:?}", w.dims());
    }
    Ok(RmsNorm::new(w, eps))
}

pub(crate) fn build_rope(
    head_dim: usize,
    base: f32,
    partial: f32,
    max_seq_len: usize,
    device: &Device,
) -> Result<Rope> {
    let half = head_dim / 2;
    let rope_angles = ((partial * head_dim as f32 / 2.0) as usize).min(half);
    let mut inv_freq = vec![0f32; half];
    for (i, f) in inv_freq.iter_mut().enumerate().take(rope_angles) {
        *f = 1.0 / base.powf((i as f32 * 2.0) / (head_dim as f32));
    }
    Rope::from_inv_freq(
        RopeConfig {
            head_dim,
            max_seq_len,
            base,
            kind: RopeKind::Standard,
        },
        &inv_freq,
        device,
    )
}

impl CausalLm for Gemma4Moe {
    fn forward(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() || tokens.len() != positions.len() {
            anyhow::bail!(
                "gemma4-moe CausalLm.forward: tokens len {} positions len {}",
                tokens.len(),
                positions.len()
            );
        }
        let seq = tokens.len();
        let max_pos = *positions.iter().max().unwrap() as usize;
        let tokens_t = Tensor::from_vec(tokens.to_vec(), (1usize, seq), &self.device)?;
        let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let pos_t = Tensor::from_vec(pos_i32, seq, &self.device)?;
        let mut cache = self.new_kv_cache((max_pos + 1).max(seq))?;
        let logits = self.forward_with_cache(&tokens_t, &pos_t, &mut cache)?;
        let vocab = self.config.base.vocab_size;
        let last = logits
            .narrow(1, seq - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if last.len() != vocab {
            anyhow::bail!(
                "gemma4-moe CausalLm.forward: logits row len {} != vocab {}",
                last.len(),
                vocab
            );
        }
        Ok(last)
    }

    fn vocab_size(&self) -> usize {
        self.config.base.vocab_size
    }
}
