use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::load_rmsnorm;
use nv_weights::{GgufLoader, TensorSource, WeightLoader};
use std::path::Path;

use crate::gemma4::{
    mlp_forward, Gemma4Attention, Gemma4Cache, Gemma4Config, Gemma4KvCache, Gemma4Mlp, LayerType,
};
use crate::CausalLm;

#[cfg(feature = "cuda")]
type ExpertNvfp4Runner = Option<std::sync::Arc<std::sync::Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>>;
#[cfg(not(feature = "cuda"))]
type ExpertNvfp4Runner = Option<()>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaExpertResidency {
    Bf16Stacked,
    Nvfp4WhenEligible,
}

pub fn nv_g4moe_cuda_nvfp4_opt_in_because_post_training_quant_failed_the_ppl_band() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_G4MOE_CUDA_NVFP4")
}

pub fn nv_g4moe_expert_bt_gemm_opt_in_because_the_route_change_is_banded_not_bitwise() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_G4MOE_EXPERT_BT_GEMM")
}

pub fn serving_default_expert_residency_bf16_with_nvfp4_opt_in() -> CudaExpertResidency {
    if nv_g4moe_cuda_nvfp4_opt_in_because_post_training_quant_failed_the_ppl_band() {
        CudaExpertResidency::Nvfp4WhenEligible
    } else {
        CudaExpertResidency::Bf16Stacked
    }
}

#[cfg(feature = "cuda")]
fn expert_dims_meet_nvfp4_min_tile_so_tiny_fixtures_stay_bf16(hidden: usize, inter: usize) -> bool {
    use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
    hidden % BLOCK_SIZE == 0
        && inter % BLOCK_SIZE == 0
        && hidden >= MIN_TILE
        && inter >= MIN_TILE
        && 2 * inter >= MIN_TILE
}

#[cfg(feature = "cuda")]
fn expert_nvfp4_runner_for(
    residency: CudaExpertResidency,
    device: &Device,
) -> Result<ExpertNvfp4Runner> {
    match (residency, device) {
        (CudaExpertResidency::Nvfp4WhenEligible, Device::Cuda(d)) => {
            Ok(Some(std::sync::Arc::new(std::sync::Mutex::new(
                nv_quant::nvfp4::Nvfp4GemmRunner::new(d.cuda_stream().clone())?,
            ))))
        }
        _ => Ok(None),
    }
}

#[cfg(not(feature = "cuda"))]
fn expert_nvfp4_runner_for(
    _residency: CudaExpertResidency,
    _device: &Device,
) -> Result<ExpertNvfp4Runner> {
    Ok(None)
}

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

pub enum Gemma4MoeExperts {
    Bf16Stacked {
        gate_up: Tensor,
        down: Tensor,
    },
    #[cfg(feature = "cuda")]
    Nvfp4Resident {
        mlps: Vec<Gemma4Mlp>,
    },
}

impl Gemma4MoeExperts {
    pub fn is_nvfp4_resident(&self) -> bool {
        match self {
            Self::Bf16Stacked { .. } => false,
            #[cfg(feature = "cuda")]
            Self::Nvfp4Resident { .. } => true,
        }
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

    pub experts: Gemma4MoeExperts,
}

pub mod moe_stall_prof {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static ROUTE_ENQUEUE_NS: AtomicU64 = AtomicU64::new(0);
    pub static DRAIN_NS: AtomicU64 = AtomicU64::new(0);
    pub static DISPATCH_HOST_NS: AtomicU64 = AtomicU64::new(0);
    pub static FORWARDS: AtomicU64 = AtomicU64::new(0);
    pub static EXPERT_PRETRANSPOSE_BYTES: AtomicU64 = AtomicU64::new(0);

    pub fn enabled_via_nv_g4moe_moe_prof_reread_every_call() -> bool {
        nv_config::env_arm_present_at_all_reread_every_call("NV_G4MOE_MOE_PROF")
    }

    pub fn add(cell: &AtomicU64, ns: u128) {
        cell.fetch_add(ns as u64, Ordering::Relaxed);
    }

    pub fn snapshot_and_reset_route_enqueue_drain_dispatch_forwards() -> [u64; 4] {
        [
            ROUTE_ENQUEUE_NS.swap(0, Ordering::Relaxed),
            DRAIN_NS.swap(0, Ordering::Relaxed),
            DISPATCH_HOST_NS.swap(0, Ordering::Relaxed),
            FORWARDS.swap(0, Ordering::Relaxed),
        ]
    }

    pub fn snapshot_and_reset_expert_pretranspose_bytes() -> u64 {
        EXPERT_PRETRANSPOSE_BYTES.swap(0, Ordering::Relaxed)
    }
}

pub struct Gemma4MoeRouting {
    pub top_idx_host: Vec<u32>,
    pub top_weights_host: Vec<f32>,
    pub n_tokens: usize,
}

impl Gemma4MoeBlock {
    pub fn route(&self, x_router: &Tensor) -> Result<Gemma4MoeRouting> {
        let prof = moe_stall_prof::enabled_via_nv_g4moe_moe_prof_reread_every_call();
        let t0 = std::time::Instant::now();
        let in_dims = x_router.dims().to_vec();
        if *in_dims.last().unwrap() != self.hidden_size {
            anyhow::bail!(
                "Gemma4MoeBlock: router last dim {} != hidden_size {}",
                in_dims.last().unwrap(),
                self.hidden_size
            );
        }
        let n_tokens: usize = in_dims[..in_dims.len() - 1].iter().product();
        let in_dtype = x_router.dtype();
        let device = x_router.device().clone();
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
        let t1 = std::time::Instant::now();

        let top_idx_host: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_weights_host: Vec<f32> = top_weights.flatten_all()?.to_vec1::<f32>()?;
        if prof {
            moe_stall_prof::add(
                &moe_stall_prof::ROUTE_ENQUEUE_NS,
                (t1 - t0).as_nanos(),
            );
            moe_stall_prof::add(&moe_stall_prof::DRAIN_NS, t1.elapsed().as_nanos());
        }
        Ok(Gemma4MoeRouting {
            top_idx_host,
            top_weights_host,
            n_tokens,
        })
    }

    pub fn dispatch(&self, routing: &Gemma4MoeRouting, x: &Tensor) -> Result<Tensor> {
        let prof = moe_stall_prof::enabled_via_nv_g4moe_moe_prof_reread_every_call();
        let t0 = std::time::Instant::now();
        let in_dims = x.dims().to_vec();
        if *in_dims.last().unwrap() != self.hidden_size {
            anyhow::bail!(
                "Gemma4MoeBlock: last dim {} != hidden_size {}",
                in_dims.last().unwrap(),
                self.hidden_size
            );
        }
        let n_tokens: usize = in_dims[..in_dims.len() - 1].iter().product();
        anyhow::ensure!(
            n_tokens == routing.n_tokens,
            "Gemma4MoeBlock: dispatch over {} tokens but routing carries {}",
            n_tokens,
            routing.n_tokens
        );
        let in_dtype = x.dtype();
        let device = x.device().clone();
        let x_flat = x.reshape((n_tokens, self.hidden_size))?.contiguous()?;
        let k = self.top_k;

        let mut expert_rows: Vec<Vec<u32>> = vec![Vec::new(); self.num_experts];
        let mut expert_w: Vec<Vec<f32>> = vec![Vec::new(); self.num_experts];
        for n in 0..n_tokens {
            for j in 0..k {
                let e = routing.top_idx_host[n * k + j] as usize;
                expert_rows[e].push(n as u32);
                expert_w[e].push(routing.top_weights_host[n * k + j] * self.per_expert_scale[e]);
            }
        }

        let mut rows_concat: Vec<u32> = Vec::with_capacity(n_tokens * k);
        let mut w_concat: Vec<f32> = Vec::with_capacity(n_tokens * k);
        let mut offsets: Vec<usize> = Vec::with_capacity(self.num_experts);
        for e in 0..self.num_experts {
            offsets.push(rows_concat.len());
            rows_concat.extend_from_slice(&expert_rows[e]);
            w_concat.extend_from_slice(&expert_w[e]);
        }
        let total = rows_concat.len();
        let (rows_all_t, w_all_t) = if total > 0 {
            (
                Some(Tensor::from_vec(rows_concat, total, &device)?),
                Some(Tensor::from_vec(w_concat, total, &device)?),
            )
        } else {
            (None, None)
        };

        let mut acc = Tensor::zeros((n_tokens, self.hidden_size), DType::F32, &device)?;
        for e in 0..self.num_experts {
            let m = expert_rows[e].len();
            if m == 0 {
                continue;
            }
            let idx_t = rows_all_t.as_ref().unwrap().narrow(0, offsets[e], m)?;
            let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;

            let y_raw = match &self.experts {
                Gemma4MoeExperts::Bf16Stacked { gate_up, down } => {
                    let gu_e = gate_up.narrow(0, e, 1)?.squeeze(0)?;
                    let dn_e = down.narrow(0, e, 1)?.squeeze(0)?;
                    let mlp_e =
                        if nv_g4moe_expert_bt_gemm_opt_in_because_the_route_change_is_banded_not_bitwise() {
                            Gemma4Mlp {
                                gate_up_proj: Linear::new_no_pretranspose(gu_e, None)?,
                                down_proj: Linear::new_no_pretranspose(dn_e, None)?,
                            }
                        } else {
                            if prof {
                                moe_stall_prof::add(
                                    &moe_stall_prof::EXPERT_PRETRANSPOSE_BYTES,
                                    ((gu_e.elem_count() + dn_e.elem_count())
                                        * gu_e.dtype().size_in_bytes())
                                        as u128,
                                );
                            }
                            Gemma4Mlp {
                                gate_up_proj: Linear::new(gu_e.contiguous()?, None)?,
                                down_proj: Linear::new(dn_e.contiguous()?, None)?,
                            }
                        };
                    mlp_forward(&mlp_e, &gathered)?
                }
                #[cfg(feature = "cuda")]
                Gemma4MoeExperts::Nvfp4Resident { mlps } => mlp_forward(&mlps[e], &gathered)?,
            };
            let y_e = y_raw.to_dtype(DType::F32)?;
            let w_t = w_all_t
                .as_ref()
                .unwrap()
                .narrow(0, offsets[e], m)?
                .reshape((m, 1))?;
            let weighted = y_e.broadcast_mul(&w_t)?;
            acc = acc.index_add(&idx_t, &weighted, 0)?;
        }

        if prof {
            moe_stall_prof::add(&moe_stall_prof::DISPATCH_HOST_NS, t0.elapsed().as_nanos());
            moe_stall_prof::FORWARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(self.hidden_size);
        Ok(acc.reshape(out_dims)?.to_dtype(in_dtype)?)
    }

    pub fn forward(&self, x_router: &Tensor, x: &Tensor) -> Result<Tensor> {
        let routing = self.route(x_router)?;
        self.dispatch(&routing, x)
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
        Self::from_loader_dtype_residency(
            config,
            weights,
            device,
            DType::BF16,
            serving_default_expert_residency_bf16_with_nvfp4_opt_in(),
        )
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
        Self::from_loader_dtype_residency(
            config,
            weights,
            device,
            dtype,
            CudaExpertResidency::Bf16Stacked,
        )
    }

    pub fn experts_nvfp4_resident_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.moe.experts.is_nvfp4_resident())
            .count()
    }

    pub fn from_loader_dtype_residency<S: TensorSource>(
        config: Gemma4MoeConfig,
        weights: &S,
        device: &Device,
        dtype: DType,
        residency: CudaExpertResidency,
    ) -> Result<Self> {
        let nvfp4_runner: ExpertNvfp4Runner = expert_nvfp4_runner_for(residency, device)?;
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
            layers.push(load_layer(
                &config,
                i,
                weights,
                device,
                dtype,
                &nvfp4_runner,
            )?);
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
        Ok(self
            .forward_with_cache_body(tokens, positions, cache, None)?
            .0)
    }

    pub fn forward_with_cache_embeds<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        embeds: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        Ok(self
            .forward_with_cache_body(tokens, positions, cache, Some(embeds))?
            .0)
    }

    pub fn forward_with_cache_prenorm_hidden_the_row_the_gemma4_assistant_drafter_consumes<
        C: Gemma4Cache,
    >(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with_cache_body(tokens, positions, cache, None)
    }

    fn forward_with_cache_body<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        embeds_override: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
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
        let logits = if cap > 0.0 {
            logits
                .affine(1.0 / cap as f64, 0.0)?
                .tanh()?
                .affine(cap as f64, 0.0)?
        } else {
            logits
        };
        Ok((logits, hidden))
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
        let (k_view, v_view) = cache.view(layer_idx, new_total)?;
        let (k_full, v_full) = kv_that_keeps_this_steps_own_keys_and_values_on_the_graph(
            &k_view,
            &v_view,
            &k_rot,
            &v_for_cache,
            seq,
        )?;

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

pub const GEMMA4_KV_CACHE_WRITE_AT_USES_SLICE_SET_WHICH_COPIES_INTO_A_PREALLOCATED_BUFFER_IN_PLACE_SO_THE_VIEW_IT_HANDS_BACK_IS_A_GRAPH_LEAF_AND_K_AND_V_WOULD_NOT_TRAIN: &str =
    "Gemma4KvCache::write_at stores this step's keys and values with Tensor::slice_set, which \
     writes bytes into a buffer allocated once by new_kv_cache and returns nothing. The buffer \
     is not a Var and carries no BackpropOp, so cache.view returns Narrow(buffer) -- a graph \
     leaf. k_rot and v_for_cache are reachable from the loss only through that write, so a \
     trainer that attends over the cache view gets no gradient entry for k or v at all while q \
     still trains through its own path into the scores. self_attn.k_proj and self_attn.v_proj \
     are DEFAULT LoRA targets of train_runner::run and /v1/fine_tuning, their B factor is \
     zero-initialised, and candle's AdamW::step silently skips a Var with no GradStore entry, so \
     they stay exactly zero and merge as an exact no-op while the loss falls. The cache held a \
     differentiable slice_assign until 3272bc550 (2026-08-15) traded it for the in-place write. \
     Splicing this step's own live k and v back over the tail of the view restores the entry \
     without moving a byte: the view already holds exactly those values, so the spliced tensor \
     is bit-identical to it and only its provenance changes. Everything older than this step is \
     left detached, which is correct -- a previous step's keys are a constant with respect to \
     this step's gradient. Serving never takes it: on an untracked k the view is returned \
     unchanged, so the decoder reads the same tensor it always did.";

fn a_gradient_is_being_built_through_this_kv_write(k_new: &Tensor) -> bool {
    if !k_new.track_op() {
        return false;
    }
    std::env::var("NV_ALLOW_LEAF_GRADIENT_LOSS").ok().as_deref() != Some("1")
}

pub fn kv_that_keeps_this_steps_own_keys_and_values_on_the_graph(
    k_view: &Tensor,
    v_view: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    seq: usize,
) -> Result<(Tensor, Tensor)> {
    if !a_gradient_is_being_built_through_this_kv_write(k_new) {
        let _ = GEMMA4_KV_CACHE_WRITE_AT_USES_SLICE_SET_WHICH_COPIES_INTO_A_PREALLOCATED_BUFFER_IN_PLACE_SO_THE_VIEW_IT_HANDS_BACK_IS_A_GRAPH_LEAF_AND_K_AND_V_WOULD_NOT_TRAIN;
        return Ok((k_view.clone(), v_view.clone()));
    }
    let stored = k_view.dims()[1];
    if stored < seq {
        anyhow::bail!(
            "gemma4_moe: the cache view holds {stored} entries but this step wrote {seq}, so the \
             write this splice is meant to keep on the graph is not in the view"
        );
    }
    let entries_older_than_this_step = stored - seq;
    if entries_older_than_this_step == 0 {
        return Ok((k_new.clone(), v_new.clone()));
    }
    let k_prefix = k_view.narrow(1, 0, entries_older_than_this_step)?;
    let v_prefix = v_view.narrow(1, 0, entries_older_than_this_step)?;
    Ok((
        Tensor::cat(&[&k_prefix, k_new], 1)?.contiguous()?,
        Tensor::cat(&[&v_prefix, v_new], 1)?.contiguous()?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn naive_sdpa(
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
    let probs = nv_layers::attn::softmax_last_dim_that_keeps_q_and_k_on_the_graph(&scores)?;
    let out = probs.matmul(&v3e)?;
    Ok(out.transpose(0, 1)?.contiguous()?)
}

fn load_layer<S: TensorSource>(
    config: &Gemma4MoeConfig,
    idx: usize,
    weights: &S,
    device: &Device,
    dtype: DType,
    nvfp4_runner: &ExpertNvfp4Runner,
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

    let moe = load_moe_block(config, &prefix, weights, dtype, device, nvfp4_runner)?;

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
    device: &Device,
    nvfp4_runner: &ExpertNvfp4Runner,
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

    let experts = build_expert_storage(n_e, inter, hidden, gate_up, down, device, nvfp4_runner)?;

    Ok(Gemma4MoeBlock {
        num_experts: n_e,
        top_k: config.top_k_experts,
        hidden_size: hidden,
        moe_intermediate_size: inter,
        rms_norm_eps: config.base.rms_norm_eps,
        router_proj: Linear::new(router_proj_w, None)?,
        router_scale,
        per_expert_scale,
        experts,
    })
}

#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn build_expert_storage(
    n_e: usize,
    inter: usize,
    hidden: usize,
    gate_up: Tensor,
    down: Tensor,
    device: &Device,
    nvfp4_runner: &ExpertNvfp4Runner,
) -> Result<Gemma4MoeExperts> {
    #[cfg(feature = "cuda")]
    if let (Some(runner), Device::Cuda(_)) = (nvfp4_runner, device) {
        if expert_dims_meet_nvfp4_min_tile_so_tiny_fixtures_stay_bf16(hidden, inter) {
            let mut mlps = Vec::with_capacity(n_e);
            for e in 0..n_e {
                let gu = gate_up.narrow(0, e, 1)?.squeeze(0)?.contiguous()?;
                let dn = down.narrow(0, e, 1)?.squeeze(0)?.contiguous()?;
                mlps.push(Gemma4Mlp {
                    gate_up_proj: Linear::from_bf16_quantized_nvfp4_dev(
                        &gu,
                        None,
                        device,
                        runner.clone(),
                    )
                    .with_context(|| format!("nvfp4-quantize expert {e} gate_up"))?,
                    down_proj: Linear::from_bf16_quantized_nvfp4_dev(
                        &dn,
                        None,
                        device,
                        runner.clone(),
                    )
                    .with_context(|| format!("nvfp4-quantize expert {e} down"))?,
                });
            }
            return Ok(Gemma4MoeExperts::Nvfp4Resident { mlps });
        }
    }
    Ok(Gemma4MoeExperts::Bf16Stacked {
        gate_up: gate_up.contiguous()?,
        down: down.contiguous()?,
    })
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
