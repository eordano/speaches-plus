use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::attn::{flash_attn, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::linear_attn::{LinAttnState, LinearAttention, LinearAttentionConfig};
use nv_layers::mlp::Mlp;
use nv_layers::moe::{MoeBlock, MoeConfig};
use nv_layers::norm::RmsNorm;
use nv_layers::rope::Rope;
#[cfg(feature = "cuda")]
use nv_layers::rope::{RopeConfig, RopeKind};
use nv_weights::{QuantizationConfig, WeightLoader};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use std::ffi::c_void;

pub trait MoeDispatch: Send + Sync {
    fn forward(&self, layer_idx: usize, moe: &MoeBlock, x: &Tensor) -> Result<Tensor>;
    fn dump_phase_ladder_after_replay_when_nv_q36_moe_events_1(&self) {}
}

#[cfg(feature = "cuda")]
#[path = "qwen38_batch.rs"]
pub mod qwen38_batch;

#[cfg(feature = "cuda")]
#[path = "qwen36_batch.rs"]
pub mod qwen36_batch;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug)]
pub struct Qwen3MoeConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub partial_rotary_factor: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub layer_types: Vec<LayerType>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub attn_output_gate: bool,
    pub tie_word_embeddings: bool,
}

fn scoped_token_id(scopes: [&serde_json::Value; 2], k: &str) -> Option<u32> {
    for scope in scopes {
        match scope.get(k) {
            Some(serde_json::Value::Number(n)) => {
                if let Some(x) = n.as_u64() {
                    return Some(x as u32);
                }
            }
            Some(serde_json::Value::Array(a)) => {
                if let Some(x) = a.first().and_then(|x| x.as_u64()) {
                    return Some(x as u32);
                }
            }
            _ => {}
        }
    }
    None
}

fn scoped_bool(scopes: [&serde_json::Value; 2], k: &str) -> Option<bool> {
    scopes
        .into_iter()
        .find_map(|s| s.get(k).and_then(|x| x.as_bool()))
}

impl Qwen3MoeConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).context("parse qwen3_5_moe config json")?;
        let text = v
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("missing text_config"))?;
        let get_u = |k: &str| -> Result<usize> {
            text.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let get_f = |k: &str| -> Result<f64> {
            text.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let layer_types_raw = text
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing layer_types"))?;
        let layer_types: Vec<LayerType> = layer_types_raw
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(LayerType::LinearAttention),
                Some("full_attention") => Ok(LayerType::FullAttention),
                other => Err(anyhow::anyhow!("unknown layer type {:?}", other)),
            })
            .collect::<Result<Vec<_>>>()?;
        let rope_params = text
            .get("rope_parameters")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let rope_theta = rope_params
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("rope_theta").and_then(|x| x.as_f64()))
            .unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rope_params
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("partial_rotary_factor").and_then(|x| x.as_f64()))
            .unwrap_or(1.0) as f32;
        let bos = scoped_token_id([text, &v], "bos_token_id").unwrap_or(0);
        let eos = scoped_token_id([text, &v], "eos_token_id")
            .ok_or_else(|| anyhow::anyhow!("missing eos_token_id"))?;
        let cfg = Self {
            hidden_size: get_u("hidden_size")?,
            num_hidden_layers: get_u("num_hidden_layers")?,
            num_attention_heads: get_u("num_attention_heads")?,
            num_key_value_heads: get_u("num_key_value_heads")?,
            head_dim: get_u("head_dim")?,
            moe_intermediate_size: get_u("moe_intermediate_size")?,
            shared_expert_intermediate_size: get_u("shared_expert_intermediate_size")?,
            num_experts: get_u("num_experts")?,
            num_experts_per_tok: get_u("num_experts_per_tok")?,
            vocab_size: get_u("vocab_size")?,
            max_position_embeddings: get_u("max_position_embeddings")?,
            rope_theta,
            rms_norm_eps: get_f("rms_norm_eps")?,
            partial_rotary_factor,
            bos_token_id: bos,
            eos_token_id: eos,
            layer_types,
            linear_num_key_heads: get_u("linear_num_key_heads")?,
            linear_num_value_heads: get_u("linear_num_value_heads")?,
            linear_key_head_dim: get_u("linear_key_head_dim")?,
            linear_value_head_dim: get_u("linear_value_head_dim")?,
            linear_conv_kernel_dim: get_u("linear_conv_kernel_dim")?,
            attn_output_gate: text
                .get("attn_output_gate")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            tie_word_embeddings: scoped_bool([text, &v], "tie_word_embeddings").unwrap_or(false),
        };
        let _ = cfg.moe_config()?;
        anyhow::ensure!(
            cfg.layer_types.len() == cfg.num_hidden_layers,
            "layer_types has {} entries but num_hidden_layers is {}",
            cfg.layer_types.len(),
            cfg.num_hidden_layers
        );
        Ok(cfg)
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn moe_config(&self) -> Result<MoeConfig> {
        anyhow::ensure!(
            self.num_experts > 0 && self.num_experts_per_tok > 0 && self.moe_intermediate_size > 0,
            "not a MoE config: num_experts={} num_experts_per_tok={} moe_intermediate_size={}",
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size
        );
        Ok(MoeConfig {
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
        })
    }

    pub fn linear_attn_config(&self) -> LinearAttentionConfig {
        LinearAttentionConfig {
            hidden_size: self.hidden_size,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            mamba_ssm_dtype: DType::F32,
            rms_eps: self.rms_norm_eps,
        }
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as usize
    }
}

pub struct AttentionLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    rotary_dim: usize,
}

impl AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attn_output_gate: bool,
        rotary_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_output_gate,
            rotary_dim,
        }
    }

    fn project_qkv(
        &self,
        x: &Tensor,
        rope: &Rope,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            anyhow::bail!("attention forward expects [B, T, H]");
        }
        let b = dims[0];
        let t = dims[1];
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let in_dtype = x.dtype();

        let q_raw = self.q_proj.forward(x)?;
        let k_raw = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;
        #[cfg(feature = "cuda")]
        decode_prof::lap("attn_qkv_proj");

        let gate_fused_arm_reads_the_gate_strided_from_q_raw_skipping_the_qg_slice =
            std::env::var("NV_Q38_ATTN_GATE_FUSED").as_deref() == Ok("1")
                && cfg!(feature = "cuda")
                && x.dtype() == DType::BF16
                && matches!(x.device(), Device::Cuda(_));
        let (q_value, q_gate) = if self.attn_output_gate
            && gate_fused_arm_reads_the_gate_strided_from_q_raw_skipping_the_qg_slice
        {
            let q_view = q_raw.reshape((b, t, n_heads, head_dim * 2))?;
            let rows = b * t * n_heads;
            let qv = slice_cols_bf16(&q_view, rows, head_dim * 2, 0, head_dim)?
                .ok_or_else(|| anyhow::anyhow!("gate-fused arm: qv slice unavailable"))?
                .reshape((b, t, n_heads, head_dim))?;
            (qv, Some(q_view))
        } else if self.attn_output_gate {
            let q_view = q_raw.reshape((b, t, n_heads, head_dim * 2))?;
            let rows = b * t * n_heads;
            #[cfg(feature = "cuda")]
            let fast = {
                let qv = slice_cols_bf16(&q_view, rows, head_dim * 2, 0, head_dim)?;
                let qg = slice_cols_bf16(&q_view, rows, head_dim * 2, head_dim, head_dim)?;
                match (qv, qg) {
                    (Some(qv), Some(qg)) => Some((
                        qv.reshape((b, t, n_heads, head_dim))?,
                        qg.reshape((b, t, n_heads, head_dim))?,
                    )),
                    _ => None,
                }
            };
            #[cfg(not(feature = "cuda"))]
            let fast: Option<(Tensor, Tensor)> = None;
            let _ = rows;
            match fast {
                Some((qv, qg)) => (qv, Some(qg)),
                None => {
                    let qv = q_view.narrow(3, 0, head_dim)?.contiguous()?;
                    let qg = q_view.narrow(3, head_dim, head_dim)?.contiguous()?;
                    (qv, Some(qg))
                }
            }
        } else {
            (q_raw.reshape((b, t, n_heads, head_dim))?, None)
        };
        let k = k_raw.reshape((b, t, n_kv_heads, head_dim))?;
        let v = v_raw.reshape((b, t, n_kv_heads, head_dim))?;

        let q_normed = self.q_norm.forward(&q_value)?;
        let k_normed = self.k_norm.forward(&k)?;

        let (q_rot, k_rot) = apply_partial_rope(
            &q_normed,
            &k_normed,
            rope,
            positions,
            self.rotary_dim,
            head_dim,
        )?;
        let q_final = q_rot.to_dtype(in_dtype)?;
        let k_final = k_rot.to_dtype(in_dtype)?;
        #[cfg(feature = "cuda")]
        decode_prof::lap("attn_qknorm_rope_glue");
        Ok((q_final, k_final, v, q_gate))
    }

    fn finalize_attn(
        &self,
        attn_out: Tensor,
        q_gate: Option<Tensor>,
        b: usize,
        t: usize,
        in_dtype: DType,
    ) -> Result<Tensor> {
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let attn_out = attn_out.reshape((b, t, n_heads * head_dim))?;
        let gated = if let Some(qg) = q_gate {
            #[cfg(feature = "cuda")]
            if *qg.dims().last().unwrap() == head_dim * 2 {
                let fused = fused_attn_gate_sigmoid_mul(&attn_out, &qg, b * t, n_heads, head_dim)?;
                let out = self.o_proj.forward(&fused)?;
                return Ok(out);
            }
            let gate_flat = qg.reshape((b, t, n_heads * head_dim))?;
            let sig =
                candle_nn::ops::sigmoid(&gate_flat.to_dtype(DType::F32)?)?.to_dtype(in_dtype)?;
            attn_out.mul(&sig)?
        } else {
            attn_out
        };
        let out = self.o_proj.forward(&gated)?;
        Ok(out)
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope, positions: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let b = dims[0];
        let t = dims[1];
        let in_dtype = x.dtype();
        let (q_final, k_final, v, q_gate) = self.project_qkv(x, rope, positions)?;
        let attn_cfg = AttnConfig {
            num_heads: self.n_heads,
            num_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = flash_attn(
            &q_final.contiguous()?,
            &k_final.contiguous()?,
            &v.contiguous()?,
            &attn_cfg,
        )?;
        self.finalize_attn(attn_out, q_gate, b, t, in_dtype)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        rope: &Rope,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        cache_slot: usize,
        _write_start: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            anyhow::bail!("attention forward_with_cache expects [B, T, H]");
        }
        let b = dims[0];
        let t = dims[1];
        let in_dtype = x.dtype();
        let scaling_fused = 1.0 / (self.head_dim as f32).sqrt();
        if b == 1
            && t == 1
            && in_dtype == DType::BF16
            && nv_q38_fused_qkv_prep_env_kill_switch_nv_q38_fused_qkv_0()
            && self.head_dim % 32 == 0
            && self.head_dim <= 1024
            && self.rotary_dim >= 2
            && self.rotary_dim % 2 == 0
            && self.rotary_dim <= self.head_dim
            && rope.cos().dtype() == DType::F32
            && matches!(x.device(), Device::Cuda(_))
        {
            let (q_raw, k_raw, v_raw) = match attn_qkv_single_launch_gemv_decode_m1(
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                x,
            )? {
                Some(qkv) => qkv,
                None => (
                    self.q_proj.forward(x)?,
                    self.k_proj.forward(x)?,
                    self.v_proj.forward(x)?,
                ),
            };
            decode_prof::lap("attn_qkv_proj");
            if q_raw.dtype() == DType::BF16
                && k_raw.dtype() == DType::BF16
                && v_raw.dtype() == DType::BF16
            {
                let (q_final, q_sig) = cache
                    .fused_qkv_norm_rope_store_decode_rope_pos_reads_write_pos_dev_because_decode_positions_equal_write_start(
                        cache_slot,
                        &q_raw,
                        &k_raw,
                        &v_raw,
                        self.q_norm.weight_bf16(),
                        self.k_norm.weight_bf16(),
                        rope,
                        self.n_heads,
                        self.rotary_dim,
                        self.q_norm.eps() as f32,
                        self.attn_output_gate,
                    )?;
                decode_prof::lap("attn_fused_prep_store");
                let attn_out =
                    cache.decode_attention_fp8(cache_slot, &q_final, self.n_heads, scaling_fused)?;
                decode_prof::lap("attn_core_splitk");
                let flat = attn_out.reshape((b, t, self.n_heads * self.head_dim))?;
                let gated = match q_sig {
                    Some(sig) => flat.mul(&sig)?,
                    None => flat,
                };
                let out = self.o_proj.forward(&gated)?;
                decode_prof::lap("attn_gate_oproj");
                return Ok(out);
            }
            anyhow::bail!(
                "fused qkv decode: projections not bf16 on a bf16 input; the projection arm \
                 changed dtype and the fused gate must learn its new contract"
            );
        }
        if b == 1
            && (2..=MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP).contains(&t)
            && in_dtype == DType::BF16
            && nv_q36_attn_verify_prep_env_opt_in_nv_q36_attn_verify_prep_1_one_kernel_qknorm_rope_sigmoid_kv_fp8_store_for_2_to_8_verify_rows()
            && self.head_dim % 32 == 0
            && self.head_dim <= 1024
            && self.rotary_dim >= 2
            && self.rotary_dim % 2 == 0
            && self.rotary_dim <= self.head_dim
            && rope.cos().dtype() == DType::F32
            && positions.dtype() == DType::I32
            && matches!(x.device(), Device::Cuda(_))
        {
            let q_raw = self.q_proj.forward(x)?;
            let k_raw = self.k_proj.forward(x)?;
            let v_raw = self.v_proj.forward(x)?;
            decode_prof::lap("attn_qkv_proj");
            anyhow::ensure!(
                q_raw.dtype() == DType::BF16
                    && k_raw.dtype() == DType::BF16
                    && v_raw.dtype() == DType::BF16,
                "fused qkv verify prep: projections not bf16 on a bf16 input; the projection \
                 arm changed dtype and the fused gate must learn its new contract"
            );
            let (q_final, q_sig) = cache
                .fused_qkv_norm_rope_store_verify_rows_positions_read_from_the_lane_pos_buf_the_graph_updates(
                    cache_slot,
                    t,
                    &q_raw,
                    &k_raw,
                    &v_raw,
                    positions,
                    self.q_norm.weight_bf16(),
                    self.k_norm.weight_bf16(),
                    rope,
                    self.n_heads,
                    self.rotary_dim,
                    self.q_norm.eps() as f32,
                    self.attn_output_gate,
                )?;
            decode_prof::lap("attn_fused_prep_store");
            {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "[qwen3_5_moe] NV_Q36_ATTN_VERIFY_PREP verify path active (one \
                         qknorm+rope+sigmoid+kv-fp8-store kernel over t={t} rows, n_q={} \
                         n_kv={} hd={} rot={})",
                        self.n_heads, self.n_kv_heads, self.head_dim, self.rotary_dim
                    );
                });
            }
            let attn_out = match cache.verify_attention_fp8_mk(
                cache_slot,
                &q_final,
                self.n_heads,
                t,
                scaling_fused,
            )? {
                Some(out) => {
                    decode_prof::lap("attn_core_mk_verify");
                    out
                }
                None => {
                    let (k_full, v_full) = cache.view(cache_slot, new_total)?;
                    let attn_cfg = AttnConfig {
                        num_heads: self.n_heads,
                        num_kv_heads: self.n_kv_heads,
                        head_dim: self.head_dim,
                        softmax_scale: scaling_fused,
                        causal: true,
                    };
                    flash_attn(
                        &q_final.contiguous()?,
                        &k_full.contiguous()?,
                        &v_full.contiguous()?,
                        &attn_cfg,
                    )?
                }
            };
            let flat = attn_out.reshape((b, t, self.n_heads * self.head_dim))?;
            let gated = match q_sig {
                Some(sig) => flat.mul(&sig)?,
                None => flat,
            };
            let out = self.o_proj.forward(&gated)?;
            decode_prof::lap("attn_gate_oproj");
            return Ok(out);
        }
        let (q_final, k_final, v, q_gate) = self.project_qkv(x, rope, positions)?;
        decode_prof::lap("attn_qkv_proj");
        let k_contig = k_final.contiguous()?;
        let v_contig = v.contiguous()?;
        cache.write_at(cache_slot, 0, &k_contig, &v_contig)?;
        decode_prof::lap("attn_kv_quant_store");

        let scaling = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out = if t == 1 {
            let out = cache.decode_attention_fp8(
                cache_slot,
                &q_final.contiguous()?,
                self.n_heads,
                scaling,
            )?;
            decode_prof::lap("attn_core_splitk");
            out
        } else if let Some(out) = cache.verify_attention_fp8_mk(
            cache_slot,
            &q_final.contiguous()?,
            self.n_heads,
            t,
            scaling,
        )? {
            decode_prof::lap("attn_core_mk_verify");
            out
        } else {
            let (k_full, v_full) = cache.view(cache_slot, new_total)?;
            decode_prof::lap("attn_kv_dequant_view");
            let attn_cfg = AttnConfig {
                num_heads: self.n_heads,
                num_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                softmax_scale: scaling,
                causal: true,
            };
            let out = flash_attn(
                &q_final.contiguous()?,
                &k_full.contiguous()?,
                &v_full.contiguous()?,
                &attn_cfg,
            )?;
            decode_prof::lap("attn_core_flash_prefill");
            out
        };
        let out = self.finalize_attn(attn_out, q_gate, b, t, in_dtype)?;
        decode_prof::lap("attn_gate_oproj");
        Ok(out)
    }

    pub fn project_qkv_roped_for_a_drafter_owned_kv(
        &self,
        x: &Tensor,
        rope: &Rope,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        self.project_qkv(x, rope, positions)
    }

    pub fn finalize_attn_from_a_drafter_owned_kv(
        &self,
        attn_out: Tensor,
        q_gate: Option<Tensor>,
        b: usize,
        t: usize,
        in_dtype: DType,
    ) -> Result<Tensor> {
        self.finalize_attn(attn_out, q_gate, b, t, in_dtype)
    }
}

pub enum LayerMixer {
    Full(AttentionLayer),
    Linear(LinearAttention),
}

pub enum LayerFfn {
    Moe(MoeBlock),
    Dense(Mlp),
}

impl LayerFfn {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            LayerFfn::Moe(m) => m.forward(x),
            #[cfg(feature = "cuda")]
            LayerFfn::Dense(m) => m.forward_fused_cuda(x),
            #[cfg(not(feature = "cuda"))]
            LayerFfn::Dense(m) => m.forward(x),
        }
    }

    pub fn as_moe(&self) -> Option<&MoeBlock> {
        match self {
            LayerFfn::Moe(m) => Some(m),
            LayerFfn::Dense(_) => None,
        }
    }

    pub fn as_dense(&self) -> Option<&Mlp> {
        match self {
            LayerFfn::Dense(m) => Some(m),
            LayerFfn::Moe(_) => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            LayerFfn::Moe(_) => "moe",
            LayerFfn::Dense(_) => "dense_mlp",
        }
    }
}

#[cfg(feature = "cuda")]
struct Qwen3MoeKvSlotFp8 {
    k_fp8: cudarc::driver::CudaSlice<u8>,
    v_fp8: cudarc::driver::CudaSlice<u8>,
    k_scales: cudarc::driver::CudaSlice<f32>,
    v_scales: cudarc::driver::CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
pub struct Qwen3MoeKvCache {
    layers: Vec<Qwen3MoeKvSlotFp8>,
    full_slot_for_layer: Vec<Option<usize>>,
    lin_attn_for_layer: Vec<Option<usize>>,
    lin_attn_states: Vec<Option<LinAttnState>>,
    lin_ckpts: Vec<Vec<LinAttnState>>,
    fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory: Vec<Vec<LinAttnState>>,
    fused_lin_verify_rows_pending_rollback: usize,
    capture_lin_ckpts: bool,
    fused_lin_attn: bool,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
    write_pos_dev: cudarc::driver::CudaSlice<i32>,
    n_total_dev: cudarc::driver::CudaSlice<i32>,
    host_write_pos: Box<[i32; 1]>,
    host_n_total: Box<[i32; 1]>,
    scores_scratch_because_smem_decode_caps_at_12k_positions_at_48kb:
        Option<cudarc::driver::CudaSlice<f32>>,
    scratch_heads: usize,
    splitk_scratch_and_fan_in_gated_by_nv_q36_graphed_decode_fix_because_the_smem_kernel_serializes_a_block_reduce_per_kv_position_costing_82ms_at_8k:
        Option<(cudarc::driver::CudaSlice<f32>, cudarc::driver::CudaSlice<u32>)>,
    mk_verify_scratch_and_fan_in_routing_2_to_8_row_chains_off_the_full_kv_dequant_view_kill_switch_nv_q38_mk_verify_0:
        Option<(cudarc::driver::CudaSlice<f32>, cudarc::driver::CudaSlice<u32>)>,
    verify_tree_parents_dev_written_once_because_the_two_branch_shape_is_fixed:
        Option<cudarc::driver::CudaSlice<i32>>,
    verify_tree_rows_active: Option<usize>,
}

#[cfg(feature = "cuda")]
pub const VERIFY_TREE_TWO_BRANCH_M6_ROWS: usize = 6;

#[cfg(feature = "cuda")]
pub const VERIFY_TREE_TWO_BRANCH_M6_PARENTS_ANCHOR_THEN_TOP1_CHAIN_THEN_TOP2_SIBLING_WITH_CONTINUATION:
    [i32; VERIFY_TREE_TWO_BRANCH_M6_ROWS] = [-1, 0, 1, 2, 0, 4];

#[cfg(feature = "cuda")]
pub const VERIFY_TREE_TWO_BRANCH_M6_DEPTHS_POSITION_IS_COMMITTED_PLUS_DEPTH_NOT_ROW_INDEX:
    [usize; VERIFY_TREE_TWO_BRANCH_M6_ROWS] = [0, 1, 2, 3, 1, 2];

#[cfg(feature = "cuda")]
pub const VERIFY_TREE_TWO_BRANCH_M6_HOLE_DELTAS_FROM_N_TOTAL_SIBLING_ROWS_HIDE_THE_BRANCH_A_BLOCK:
    [(i32, i32); VERIFY_TREE_TWO_BRANCH_M6_ROWS] =
    [(0, 0), (0, 0), (0, 0), (0, 0), (5, 2), (5, 2)];

#[cfg(feature = "cuda")]
pub const VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW: usize = 4;

#[cfg(feature = "cuda")]
pub const MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP: usize = 8;

#[cfg(feature = "cuda")]
pub const MK_VERIFY_MAX_HEAD_DIM_512_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP: usize = 512;

#[cfg(feature = "cuda")]
pub const MK_VERIFY_SPLITS_CEILING_32_MATCHES_FLASH_SPLITS_PICK: usize = 32;

#[cfg(feature = "cuda")]
pub const SOLO_AND_EVERY_FAMILY_OUTSIDE_THE_QWEN38_LANE_GROUP_ENTERS_AT_ONE_LANE_IN_FLIGHT_SO_THE_SPLIT_PICK_IS_THE_ONE_THE_RECORD_WAS_MEASURED_ON:
    usize = 1;

pub const PRIME_CKPT_MAGIC_NVPRIMEK: &[u8; 8] = b"NVPRIMEK";

pub const PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES:
    u32 = 1;

pub const PRIME_CKPT_FINGERPRINT_MISMATCH: &str = "PrimeCkptFingerprintMismatch";

pub const PRIME_CKPT_GEOMETRY_MISMATCH: &str = "PrimeCkptGeometryMismatch";

pub const PRIME_CKPT_BAD_HEADER: &str = "PrimeCkptBadHeader";

#[cfg(feature = "cuda")]
pub fn nv_q36_moe_fused_chain_env_opt_in_nv_q36_moe_fused_chain_1_single_kernel_route_gather_quant_and_scatter_shared_tail_for_m1_decode(
) -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_MOE_FUSED_CHAIN")
}

pub fn nv_q36_moe_verify_chain_env_opt_in_nv_q36_moe_verify_chain_1_the_m1_fused_chain_arms_extended_to_m_le_8_verify_rows(
) -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_MOE_VERIFY_CHAIN")
}

pub fn nv_q36_moe_router_gemv_env_opt_in_nv_q36_moe_router_gemv_1_router_plus_shared_gate_plus_x_stage_fused_to_one_kernel_at_m_le_8(
) -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_MOE_ROUTER_GEMV")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_moe_shared_fold_env_opt_in_nv_q36_moe_shared_fold_1_shared_expert_becomes_grouped_tile_k_with_sigmoid_gate_applied_at_scatter(
) -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_MOE_SHARED_FOLD")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_moe_events_env_opt_in_nv_q36_moe_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes(
) -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_MOE_EVENTS")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_graphed_decode_fix_env_routes_decode_to_the_gemma4_proven_splitk_flash() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q36_GRAPHED_DECODE_FIX")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_gdn_fp8_env_opt_in_quantizes_bf16_checkpoint_gdn_projections_to_fp8_resident() -> bool
{
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q36_GDN_FP8")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_lm_head_fp8_env_opt_in_quantizes_a_bf16_checkpoint_lm_head_to_fp8_resident() -> bool
{
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q36_LM_HEAD_FP8")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_attn_verify_prep_env_opt_in_nv_q36_attn_verify_prep_1_one_kernel_qknorm_rope_sigmoid_kv_fp8_store_for_2_to_8_verify_rows(
) -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q36_ATTN_VERIFY_PREP")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_fused_qkv_prep_env_kill_switch_nv_q38_fused_qkv_0() -> bool {
    crate::host_serving_util::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
        "NV_Q38_FUSED_QKV",
    )
}

#[cfg(feature = "cuda")]
pub fn nv_q38_fp4_mlp_gemv_env_opt_in_nv_q38_fp4_mlp_1_because_w4a16_matches_but_does_not_beat_the_padded_tc_route_in_graph() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_FP4_MLP")
}

#[cfg(feature = "cuda")]
fn cuda_slice_u8_ptr(
    slice: &cudarc::driver::CudaSlice<u8>,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> u64 {
    use cudarc::driver::DevicePtr;
    let (p, _g) = slice.device_ptr(stream);
    p
}

#[cfg(feature = "cuda")]
fn fused_dense_mlp_nvfp4_w4a16_two_kernels_decode_m1_more_precise_than_the_padded_a4_tensor_core_route(
    ffn: &LayerFfn,
    x: &Tensor,
) -> Result<Option<Tensor>> {
    use cudarc::driver::DevicePtrMut;
    use half::bf16;

    let Some(mlp) = ffn.as_dense() else {
        return Ok(None);
    };
    if !nv_q38_fp4_mlp_gemv_env_opt_in_nv_q38_fp4_mlp_1_because_w4a16_matches_but_does_not_beat_the_padded_tc_route_in_graph() {
        return Ok(None);
    }
    let hidden = mlp.gate_proj().in_features();
    let inter = mlp.gate_proj().out_features();
    if x.dims() != [1, 1, hidden] || x.dtype() != DType::BF16 {
        return Ok(None);
    }
    let (Some(gp), Some(up), Some(dp)) = (
        mlp.gate_proj().nvfp4_parts(),
        mlp.up_proj().nvfp4_parts(),
        mlp.down_proj().nvfp4_parts(),
    ) else {
        return Ok(None);
    };
    if hidden % 16 != 0
        || inter % 16 != 0
        || hidden * 4 > 96 * 1024
        || inter * 4 > 96 * 1024
        || mlp.down_proj().in_features() != inter
        || mlp.down_proj().out_features() != hidden
    {
        return Ok(None);
    }
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let x_c = x.contiguous()?;

    let mut gate_y = unsafe {
        stream
            .alloc::<bf16>(inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut up_y = unsafe {
        stream
            .alloc::<bf16>(inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut down_y = unsafe {
        stream
            .alloc::<bf16>(hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };

    let (gw, gs, g_alpha_recip_wg_times_recip_ig, g_ig) = gp;
    let (uw, us, u_alpha_recip_wg_times_recip_ig, u_ig) = up;
    let (dw, ds, d_alpha_recip_wg_times_recip_ig, d_ig) = dp;
    let ga_w4a16_recip_weight_global_only_because_x_stays_bf16 =
        g_alpha_recip_wg_times_recip_ig * g_ig;
    let ua_w4a16_recip_weight_global_only_because_x_stays_bf16 =
        u_alpha_recip_wg_times_recip_ig * u_ig;
    let da_w4a16_recip_weight_global_only_because_x_stays_bf16 =
        d_alpha_recip_wg_times_recip_ig * d_ig;
    {
        let (x_storage, xl) = x_c.storage_and_layout();
        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
        let x_view = x_slice.slice(xl.start_offset()..);
        let x_ptr = {
            use cudarc::driver::DevicePtr;
            let (p, _g) = x_view.device_ptr(&stream);
            p
        };
        let rc = {
            let (gy_ptr, _g1) = gate_y.device_ptr_mut(&stream);
            let (uy_ptr, _g2) = up_y.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::gemv_nvfp4_w4a16_dual_m1(
                    stream.cu_stream() as *mut c_void,
                    cuda_slice_u8_ptr(gw, &stream) as *const u8,
                    cuda_slice_u8_ptr(gs, &stream) as *const u8,
                    cuda_slice_u8_ptr(uw, &stream) as *const u8,
                    cuda_slice_u8_ptr(us, &stream) as *const u8,
                    x_ptr as *const u16,
                    gy_ptr as *mut u16,
                    uy_ptr as *mut u16,
                    ga_w4a16_recip_weight_global_only_because_x_stays_bf16,
                    ua_w4a16_recip_weight_global_only_because_x_stays_bf16,
                    inter as i32,
                    hidden as i32,
                )
            }
        };
        if rc == -1 {
            return Ok(None);
        }
        anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a16_dual_m1 rc={rc}");
        let (gy_ptr2, _g3) = gate_y.device_ptr_mut(&stream);
        let (uy_ptr2, _g4) = up_y.device_ptr_mut(&stream);
        let (dy_ptr, _g5) = down_y.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gemv_nvfp4_w4a16_silu_gate_up_in_m1(
                stream.cu_stream() as *mut c_void,
                cuda_slice_u8_ptr(dw, &stream) as *const u8,
                cuda_slice_u8_ptr(ds, &stream) as *const u8,
                gy_ptr2 as *const u16,
                uy_ptr2 as *const u16,
                dy_ptr as *mut u16,
                da_w4a16_recip_weight_global_only_because_x_stays_bf16,
                hidden as i32,
                inter as i32,
            )
        };
        if rc == -1 {
            return Ok(None);
        }
        anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a16_silu_gate_up_in_m1 rc={rc}");
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(down_y, dev);
    Ok(Some(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (1usize, 1usize, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    )))
}

#[cfg(feature = "cuda")]
pub fn nv_q38_mlp_w4a8_env_opt_in_nv_q38_mlp_w4a8_1_dp4a_int8_dot_route_for_the_nvfp4_dense_mlp_decode() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_MLP_W4A8")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_siluq_split_env_opt_in_nv_q38_siluq_split_1_multiblock_silu_quant_producer() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_SILUQ_SPLIT")
}

pub const MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM:
    &str = "NV_Q38_MK_VERIFY_KVSHARE";

pub const MK_VERIFY_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER: &str =
    "NV_Q38_MK_VERIFY_KVSHARE_MIN_TOTAL";

pub const KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL: usize = 8192;

pub const KVSHARE_GATE_SELECTS_AT_VERIFY_GRAPH_CAPTURE_SO_A_GRAPHED_SESSION_KEEPS_THE_ARM_ITS_PROMPT_DEPTH_PICKED:
    &str =
    "NV_Q38_MK_VERIFY_KVSHARE=gate compares the total the mk verify kernel reads (host_n_total: \
     committed depth plus the m verify rows) against the knee inside verify_attention_fp8_mk. \
     The graphed serving engine executes that host routing only while recording a rows-keyed \
     verify graph -- capture_round0, immediately after prefill -- and every later round replays \
     the captured kernels, so a graphed session picks one arm from its session-start prompt \
     depth and keeps it even if decode grows the context across the knee; the uncaptured verify \
     path re-evaluates per call. =1 keeps the unconditional kvshare arm, =0/unset keeps the mk \
     arm (the default, unchanged), and NV_Q38_MK_VERIFY_KVSHARE_MIN_TOTAL overrides the knee. \
     The knee comes from the kvgate knee-ladder rows in perf/runs.jsonl (mk vs kvshare \
     round_ms_median on the drafter-splitk basis: mk ahead at 4096, arms dead even at 8192, \
     kvshare ahead from 12288 on and by 1.29x at 32768), hence \
     KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL sits at the dead-even total, \
     matching how TP_ENGAGES_PAST_THE_MEASURED_CROSSOVER_DEPTH_REANCHORED_BY_THE_Q38_TPKNEE_ROWS was placed.";

pub fn mk_verify_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
    raw: Option<&str>,
    knee_override_raw: Option<&str>,
) -> Option<usize> {
    match raw {
        None | Some("") | Some("0") => None,
        Some("1") => Some(0),
        Some("gate") => Some(match knee_override_raw {
            None | Some("") => KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL,
            Some(v) => v.parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "{MK_VERIFY_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER}={v}: \
                     set a plain token total"
                )
            }),
        }),
        Some(other) => panic!(
            "{MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM}\
             ={other}: set 1 (kvshare always), gate (kvshare at or past the knee), or 0"
        ),
    }
}

pub fn nv_q38_mk_verify_kvshare_env_opt_in_row_sequential_group_fold_reads_kv_once_per_row_not_once_per_q_head(
) -> Option<usize> {
    mk_verify_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
        std::env::var(MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM)
            .ok()
            .as_deref(),
        std::env::var(MK_VERIFY_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER)
            .ok()
            .as_deref(),
    )
}

pub const MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING:
    &str = "NV_Q38_MK_VERIFY_ROWSHARED";

pub const MK_VERIFY_ROWSHARED_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER: &str =
    "NV_Q38_MK_VERIFY_ROWSHARED_MIN_TOTAL";

pub const ROWSHARED_GATE_KNEE_TOTAL_SHARES_THE_KVSHARE_CROSSOVER_ITS_OWN_LADDER_FOUND_NO_DEPTH_WORTH_A_DIFFERENT_KNEE:
    usize = KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL;

pub const ROWSHARED_OUTRANKS_KVSHARE_WHEN_BOTH_ENGAGE_BECAUSE_ONE_LAUNCH_READS_KV_ONCE_FOR_ALL_M_ROWS_NOT_ONCE_PER_ROW:
    &str =
    "NV_Q38_MK_VERIFY_ROWSHARED routes verify_attention_fp8_mk to the one-launch row-shared \
     kernel (grid over kv-heads x splits, an smem KV tile scored against all m x group query \
     rows), so each KV byte is read once per round instead of once per row (kvshare) or once \
     per q-head (mk). When both this env and NV_Q38_MK_VERIFY_KVSHARE engage at a depth, the \
     rowshared arm wins the route; graph capture bakes the arm exactly as \
     KVSHARE_GATE_SELECTS_AT_VERIFY_GRAPH_CAPTURE_SO_A_GRAPHED_SESSION_KEEPS_THE_ARM_ITS_PROMPT_DEPTH_PICKED \
     describes. The arm ships default-off and no routing prefers it: on scalar cores the \
     m-fold multiplies per-byte compute by m, so the one-read kernel runs compute-bound where \
     the kvshare arm runs bandwidth-bound (kernel-only at 172k, m=4: rowshared 1.9 ms at 187 \
     effective GB/s vs the m-sequential kvshare 1.7 ms at 833 GB/s; the rowshared knee-ladder \
     it2 rows in perf/runs.jsonl, build_name 'q38 rowshared', measure round_ms_median +2.8% \
     at 8192, +0.5% at 32768, and dead even at 172032 against the shipped gate, and -7% at \
     256 where the gate runs the mk arm anyway). =gate at the shared knee is therefore safe \
     but buys only the mid-depth sliver; reading KV once for all m x group rows only pays \
     when the QK tile leaves the scalar pipe -- the tensor-core mma tile of llama.cpp's fattn \
     is the design that cashes this read structure.";

pub fn mk_verify_rowshared_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
    raw: Option<&str>,
    knee_override_raw: Option<&str>,
) -> Option<usize> {
    match raw {
        None | Some("") | Some("0") => None,
        Some("1") => Some(0),
        Some("gate") => Some(match knee_override_raw {
            None | Some("") => {
                ROWSHARED_GATE_KNEE_TOTAL_SHARES_THE_KVSHARE_CROSSOVER_ITS_OWN_LADDER_FOUND_NO_DEPTH_WORTH_A_DIFFERENT_KNEE
            }
            Some(v) => v.parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "{MK_VERIFY_ROWSHARED_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER}={v}: \
                     set a plain token total"
                )
            }),
        }),
        Some(other) => panic!(
            "{MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING}\
             ={other}: set 1 (rowshared always), gate (rowshared at or past the knee), or 0"
        ),
    }
}

pub fn nv_q38_mk_verify_rowshared_env_opt_in_one_launch_reads_kv_once_for_all_m_by_group_rows(
) -> Option<usize> {
    mk_verify_rowshared_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
        std::env::var(MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING)
            .ok()
            .as_deref(),
        std::env::var(MK_VERIFY_ROWSHARED_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER)
            .ok()
            .as_deref(),
    )
}

pub const MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING:
    &str = "NV_Q38_MK_VERIFY_TCTILE";

pub const MK_VERIFY_TCTILE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER: &str =
    "NV_Q38_MK_VERIFY_TCTILE_MIN_TOTAL";

pub const TCTILE_GATE_KNEE_TOTAL_SHARES_THE_KVSHARE_CROSSOVER_ITS_LADDER_WINS_FROM_8192_AND_LOSES_AT_256: usize =
    KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL;

pub const TCTILE_OUTRANKS_ROWSHARED_AND_KVSHARE_WHEN_ENGAGED_BECAUSE_MMA_TILES_CASH_THE_ONE_READ_STRUCTURE_SCALAR_ROWS_COULD_NOT:
    &str =
    "NV_Q38_MK_VERIFY_TCTILE routes verify_attention_fp8_mk to the one-launch row-shared \
     kernel whose QK scores ride m16n8k16 tensor-core tiles: each warp owns one chain row's \
     6-row q group as an mma A tile and scores 8-position chunks of a block-shared 32-\
     position K stage, so K leaves HBM once for all m x group rows while V is fetched per \
     chain row through L2 by register prefetch issued before the mma (the sibling warps \
     touch the same V lines within the same tile, so HBM still sees each V byte about once); \
     the per-byte score arithmetic leaves the scalar pipe that made the scalar rowshared arm \
     compute-bound at 187 GB/s. Per-row causal tails, window starts and the tree-row hole \
     window mask score fragments per position (masking is row-uniform inside a chain row's \
     group, so a fragment element's liveness depends only on its KV position), and a fully-\
     masked 8-position chunk is skipped before the mma so the online-softmax maxima never \
     see an all-minus-infinity update. The arm is qwen38-shaped only: group 8 (qwen36 \
     16q/2kv) refuses at the launcher with rc -27 because the 50-row half-precision q tile \
     only fits the 48 KiB static smem budget at group 6, and the routing falls through to the \
     rowshared/kvshare/mk ladder there. When several verify-arm envs engage at a depth the \
     route prefers tctile, then rowshared, then kvshare; graph capture bakes the arm exactly \
     as KVSHARE_GATE_SELECTS_AT_VERIFY_GRAPH_CAPTURE_SO_A_GRAPHED_SESSION_KEEPS_THE_ARM_ITS_PROMPT_DEPTH_PICKED \
     describes. The arm ships default-off and no knee routes it: mma tiles reassociate the \
     score sums, so the bf16 attention output sits 1-2 ulp from the mk arm's, and the MoE \
     router amplifies that into confidently different greedy streams -- the emitted-stream \
     tie-class discipline cannot certify a score-reordering arm (the shipped rowshared arm \
     fails the same forced m-invariance sweep at m=8 by the same mechanism), so the \
     losslessness gate for this arm is per-layer cross-arm parity on live serving tensors \
     (`qwen38_tctile_gate`), and any default routing waits on a quality-level gate. \
     Kernel-only at 172032, m=4, 24q/4kv hd256: tctile 0.81 ms at 442 effective GB/s vs the \
     m-sequential kvshare 1.71 ms at 838 GB/s and the scalar rowshared 1.91 ms at 187 GB/s \
     (the rowshared-added kernel-only bench in fp8_mk_ratio_oracle.rs carries the arms side \
     by side). Round-level on the champion basis (perf/runs.jsonl build_name 'q38 tctile', \
     fresh process per depth), round_ms_median control -> tctile: 28.1 -> 30.1 at 256, \
     32.7 -> 30.7 at 8192, 36.6 -> 32.8 at 32768, and 63.8 -> 45.5 at 172032 (effective \
     decode 46.3 -> 61.6 tok/s, +33%), so for an operator who opts in, =gate at the shared \
     8192 knee is the measured setting.";

pub fn mk_verify_tctile_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
    raw: Option<&str>,
    knee_override_raw: Option<&str>,
) -> Option<usize> {
    match raw {
        None | Some("") | Some("0") => None,
        Some("1") => Some(0),
        Some("gate") => Some(match knee_override_raw {
            None | Some("") => TCTILE_GATE_KNEE_TOTAL_SHARES_THE_KVSHARE_CROSSOVER_ITS_LADDER_WINS_FROM_8192_AND_LOSES_AT_256,
            Some(v) => v.parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "{MK_VERIFY_TCTILE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER}={v}: \
                     set a plain token total"
                )
            }),
        }),
        Some(other) => panic!(
            "{MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING}\
             ={other}: set 1 (tctile always), gate (tctile at or past the knee), or 0"
        ),
    }
}

pub const MK_VERIFY_DUMP_DIR_ENV_WRITES_EACH_SLOTS_QKV_ONCE_SO_A_PARITY_GATE_CAN_REPLAY_LIVE_SERVING_TENSORS_THROUGH_BOTH_ARMS:
    &str = "NV_Q38_MK_VERIFY_DUMP";

pub fn nv_q38_mk_verify_tctile_env_opt_in_one_launch_mma_qk_tiles_read_kv_once_for_all_m_by_group_rows(
) -> Option<usize> {
    mk_verify_tctile_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
        std::env::var(MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING)
            .ok()
            .as_deref(),
        std::env::var(MK_VERIFY_TCTILE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER)
            .ok()
            .as_deref(),
    )
}

pub const M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM:
    &str = "NV_Q38_DECODE_KVSHARE";

pub const M1_DECODE_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER: &str =
    "NV_Q38_DECODE_KVSHARE_MIN_TOTAL";

pub const KVSHARE_ENGAGES_PAST_THE_MEASURED_M1_DECODE_CROSSOVER_TOTAL: usize = 16384;

pub const M1_DECODE_KVSHARE_SCRATCH_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS: usize = 128;

pub const M1_DECODE_KVSHARE_GATE_SELECTS_AT_DECODE_GRAPH_CAPTURE_SO_A_GRAPHED_SESSION_KEEPS_THE_ARM_ITS_CAPTURE_DEPTH_PICKED:
    &str =
    "NV_Q38_DECODE_KVSHARE routes the m=1 plain-decode attention (the solo-lane branch of \
     decode_attention_fp8_whose_split_pick_sees_the_lane_group_width) to \
     flash_decode_kvshare_fp8kv_paged, the group-fold kernel whose grid covers kv-heads x \
     splits so each KV byte is read once per kv head instead of once per q head. =gate \
     compares host_n_total against the knee per call; the graphed serving engine executes \
     that host routing only while recording the decode graph, so a graphed session picks one \
     arm from its capture depth and keeps it even if decode grows the context across the \
     knee, exactly as \
     KVSHARE_GATE_SELECTS_AT_VERIFY_GRAPH_CAPTURE_SO_A_GRAPHED_SESSION_KEEPS_THE_ARM_ITS_PROMPT_DEPTH_PICKED \
     describes for the verify side; the eager path re-evaluates per call. =1 is the \
     unconditional kvshare arm, =0/unset keeps the fused arm (the default, unchanged), and \
     NV_Q38_DECODE_KVSHARE_MIN_TOTAL overrides the knee. The arm rides the splitk scratch \
     NV_Q36_GRAPHED_DECODE_FIX allocates and is inert without that fix, and the constructor \
     sizes that scratch for the kernel's 128-split cap only when this env is present at \
     cache construction. The knee comes from the q38_flash_depth_floor kernel-only rows \
     (24q/4kv hd256 fp8, warm L2): fused 331 GB/s vs kvshare 224 at 8192, kvshare 556 vs \
     fused 347 at 32768, and 806 vs 301 at 196608 (2.68x); linear interpolation of the \
     measured microseconds between the 8192 and 32768 rows crosses near 14.3k tokens, hence \
     KVSHARE_ENGAGES_PAST_THE_MEASURED_M1_DECODE_CROSSOVER_TOTAL sits at 16384, the first \
     power-of-two rung past the interpolated crossover, not at the verify side's 8192 where \
     the fused m=1 arm still wins.";

pub fn m1_decode_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
    raw: Option<&str>,
    knee_override_raw: Option<&str>,
) -> Option<usize> {
    match raw {
        None | Some("") | Some("0") => None,
        Some("1") => Some(0),
        Some("gate") => Some(match knee_override_raw {
            None | Some("") => KVSHARE_ENGAGES_PAST_THE_MEASURED_M1_DECODE_CROSSOVER_TOTAL,
            Some(v) => v.parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "{M1_DECODE_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER}={v}: \
                     set a plain token total"
                )
            }),
        }),
        Some(other) => panic!(
            "{M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM}\
             ={other}: set 1 (kvshare always), gate (kvshare at or past the knee), or 0"
        ),
    }
}

pub fn nv_q38_decode_kvshare_env_opt_in_group_fold_reads_kv_once_per_kv_head_not_once_per_q_head(
) -> Option<usize> {
    m1_decode_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
        std::env::var(M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM)
            .ok()
            .as_deref(),
        std::env::var(M1_DECODE_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER)
            .ok()
            .as_deref(),
    )
}

#[cfg(feature = "cuda")]
pub fn nv_q38_norm_quant_fold_env_opt_in_nv_q38_norm_quant_fold_1_post_norm_residual_rowquant_one_kernel() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_NORM_QUANT_FOLD")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_down_qfold_env_opt_in_nv_q38_down_qfold_1_act_quant_in_down_gemv_prologue() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_DOWN_QFOLD")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_attn_qkv_one_env_opt_in_single_launch_qkv_gemv_bitwise_same_rows() -> bool {
    crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q38_ATTN_QKV_ONE")
}

pub const NV_Q38_PREFILL_CHUNK_ENV_DEFAULT_4096_SET_0_FOR_THE_ONE_SHOT_FORWARD_WHOSE_TRANSIENTS_MEASURED_0_21_MIB_PER_TOKEN:
    &str = "NV_Q38_PREFILL_CHUNK";

pub const Q38_PREFILL_CHUNK_DEFAULT_TOKENS_MEASURED_LADDER_140K_AT_6_GIB_OVER_BOOT: usize = 4096;

pub fn q38_prefill_chunk_tokens(raw: Option<&str>) -> Option<usize> {
    match raw {
        None => Some(Q38_PREFILL_CHUNK_DEFAULT_TOKENS_MEASURED_LADDER_140K_AT_6_GIB_OVER_BOOT),
        Some(v) => v.trim().parse::<usize>().ok().filter(|&n| n >= 1),
    }
}

pub fn q38_prefill_chunk_tokens_from_env() -> Option<usize> {
    q38_prefill_chunk_tokens(
        std::env::var(NV_Q38_PREFILL_CHUNK_ENV_DEFAULT_4096_SET_0_FOR_THE_ONE_SHOT_FORWARD_WHOSE_TRANSIENTS_MEASURED_0_21_MIB_PER_TOKEN)
            .ok()
            .as_deref(),
    )
}

pub const NV_Q38_PREFILL_GRAPH_ENV_OPT_IN_1_REPLAYS_GDN_MLP_LAYER_RUNS_AS_CUDA_GRAPH_SEGMENTS_FULL_ATTENTION_STAYS_EAGER_BECAUSE_CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_LEGACY_NULL_STREAM:
    &str = "NV_Q38_PREFILL_GRAPH";

pub fn nv_q38_prefill_graph_env_opt_in() -> bool {
    std::env::var(NV_Q38_PREFILL_GRAPH_ENV_OPT_IN_1_REPLAYS_GDN_MLP_LAYER_RUNS_AS_CUDA_GRAPH_SEGMENTS_FULL_ATTENTION_STAYS_EAGER_BECAUSE_CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_LEGACY_NULL_STREAM)
        .ok()
        .as_deref()
        == Some("1")
}

#[cfg(feature = "cuda")]
pub const PREFILL_SEGMENT_GRAPH_SHAPE_TOKEN_BASE_KEEPS_DECODE_1_AND_VERIFY_0X51XXX_KEYS_UNTOUCHED:
    u64 = 0x53000;

#[cfg(feature = "cuda")]
fn prefill_graph_capture_hostile_env_because_the_inner_gdn_graph_arm_nests_a_capture_and_the_linattn_profiler_syncs_every_lap(
) -> Option<&'static str> {
    ["NV_Q38_GDN_CHUNK_GEMM_GRAPH", "NV_PROF_LINATTN"]
        .into_iter()
        .find(|env| std::env::var(env).is_ok())
}

#[cfg(feature = "cuda")]
pub struct Qwen38PrefillGraphSession {
    runner: nv_kernels::graph::CudaGraphRunner,
    tok_stage: Tensor,
    normed_stage: Tensor,
    residual_stage: Tensor,
    chunk: usize,
    warmed_one_eager_full_chunk_so_plans_and_workspaces_exist: bool,
    pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers:
        Vec<Option<LinAttnState>>,
    capture_failed: bool,
    eager_fallback_reported_once: std::sync::Once,
}

#[cfg(feature = "cuda")]
impl Qwen38PrefillGraphSession {
    fn new(model: &Qwen3Moe, chunk: usize) -> Result<Self> {
        anyhow::ensure!(chunk >= 1, "Qwen38PrefillGraphSession: chunk must be >= 1");
        let dev = match &model.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Qwen38PrefillGraphSession requires a CUDA device"),
        };
        let stream = dev.cuda_stream();
        anyhow::ensure!(
            !stream.cu_stream().is_null(),
            "Qwen38PrefillGraphSession: the legacy NULL stream cannot begin capture; build the \
             device with Device::new_cuda_with_stream"
        );
        let hidden = model.config.hidden_size;
        let tok_stage = Tensor::zeros((1usize, chunk), DType::U32, &model.device)?;
        let normed_stage = Tensor::zeros((1usize, chunk, hidden), DType::BF16, &model.device)?;
        let residual_stage = Tensor::zeros((1usize, chunk, hidden), DType::BF16, &model.device)?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after session alloc: {e:?}"))?;
        let n_lin = model
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, LayerMixer::Linear(_)))
            .count();
        Ok(Self {
            runner: nv_kernels::graph::CudaGraphRunner::new(stream),
            tok_stage,
            normed_stage,
            residual_stage,
            chunk,
            warmed_one_eager_full_chunk_so_plans_and_workspaces_exist: false,
            pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers:
                vec![None; n_lin],
            capture_failed: false,
            eager_fallback_reported_once: std::sync::Once::new(),
        })
    }

    pub fn invalidate(&mut self) {
        self.runner.invalidate();
    }

    pub fn captured_segment_count(&self) -> usize {
        (0..256u64)
            .filter(|i| {
                self.runner.has_cached_token(
                    PREFILL_SEGMENT_GRAPH_SHAPE_TOKEN_BASE_KEEPS_DECODE_1_AND_VERIFY_0X51XXX_KEYS_UNTOUCHED
                        + i,
                )
            })
            .count()
    }

    fn pin_lin_states_from_cache_after_the_eager_warm_chunk(
        &mut self,
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<()> {
        for slot in 0..self
            .pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers
            .len()
        {
            if self
                .pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers
                [slot]
                .is_some()
            {
                continue;
            }
            let cur = cache.lin_attn_states[slot].as_ref().ok_or_else(|| {
                anyhow::anyhow!("pin_lin_states: slot {slot} has no state after the warm chunk")
            })?;
            anyhow::ensure!(
                !cur.is_fused(),
                "pin_lin_states: slot {slot} is a fused decode state; the chunk prefill path \
                 must have produced a non-fused state"
            );
            let pinned = cur.deep_clone()?;
            cache.lin_attn_states[slot] = Some(pinned.clone());
            self.pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers
                [slot] = Some(pinned);
        }
        Ok(())
    }

    fn install_pinned_lin_states(&mut self, cache: &mut Qwen3MoeKvCache) -> Result<()> {
        for slot in 0..self
            .pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers
            .len()
        {
            let Some(pinned) = &self
                .pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers
                [slot]
            else {
                continue;
            };
            match &cache.lin_attn_states[slot] {
                Some(cur) if cur.recurrent_state().id() == pinned.recurrent_state().id() => {}
                Some(cur) => {
                    anyhow::ensure!(
                        !cur.is_fused(),
                        "install_pinned_lin_states: slot {slot} holds a fused decode state whose \
                         layout the pinned prefill state cannot absorb; run this prefill eager"
                    );
                    pinned.copy_data_from(cur)?;
                    cache.lin_attn_states[slot] = Some(pinned.clone());
                }
                None => {
                    pinned.zero_data()?;
                    cache.lin_attn_states[slot] = Some(pinned.clone());
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn cuda_tensor_device_ptr<T: candle_core::cuda_backend::CudaDType>(
    t: &Tensor,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<u64> {
    use cudarc::driver::DevicePtr;
    anyhow::ensure!(t.is_contiguous(), "stage tensor must be contiguous");
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("stage tensor must be on CUDA"),
    };
    let slice = cuda.as_cuda_slice::<T>()?;
    let (ptr, _g) = slice.device_ptr(stream);
    Ok(ptr + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
}

#[cfg(feature = "cuda")]
fn dtod_bf16_tensor_into_stage(
    t: &Tensor,
    stage: &Tensor,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    let contig = t.contiguous()?;
    anyhow::ensure!(
        contig.dtype() == DType::BF16 && stage.dtype() == DType::BF16,
        "stage copy expects bf16, got {:?} -> {:?}",
        contig.dtype(),
        stage.dtype()
    );
    anyhow::ensure!(
        contig.elem_count() == stage.elem_count(),
        "stage copy size mismatch: tensor {} elems vs stage {}",
        contig.elem_count(),
        stage.elem_count()
    );
    let src_ptr = cuda_tensor_device_ptr::<half::bf16>(&contig, stream)?;
    let dst_ptr = cuda_tensor_device_ptr::<half::bf16>(stage, stream)?;
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(
            dst_ptr,
            src_ptr,
            stage.elem_count() * std::mem::size_of::<half::bf16>(),
            stream.cu_stream(),
        )
        .map_err(|e| anyhow::anyhow!("stage dtod: {e:?}"))?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn attn_qkv_single_launch_gemv_decode_m1(
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    x: &Tensor,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    if !nv_q38_attn_qkv_one_env_opt_in_single_launch_qkv_gemv_bitwise_same_rows() {
        return Ok(None);
    }
    if x.dtype() != DType::BF16 {
        return Ok(None);
    }
    let dims = x.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 {
        return Ok(None);
    }
    let k_feat = dims[2];
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let (Some((wq_q, rs_q)), Some((wq_k, rs_k)), Some((wq_v, rs_v))) = (
        q_proj.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(),
        k_proj.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(),
        v_proj.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(),
    ) else {
        return Ok(None);
    };
    if q_proj.in_features() != k_feat
        || k_proj.in_features() != k_feat
        || v_proj.in_features() != k_feat
    {
        return Ok(None);
    }
    let n_q = q_proj.out_features();
    let n_k = k_proj.out_features();
    let n_v = v_proj.out_features();
    if n_q % 16 != 0 || n_k % 16 != 0 || n_v % 16 != 0 || k_feat % 16 != 0 {
        return Ok(None);
    }
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let x_c = x.contiguous()?;
    let mut y_q = unsafe { stream.alloc::<bf16>(n_q).map_err(|e| anyhow::anyhow!(e))? };
    let mut y_k = unsafe { stream.alloc::<bf16>(n_k).map_err(|e| anyhow::anyhow!(e))? };
    let mut y_v = unsafe { stream.alloc::<bf16>(n_v).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let x_cuda = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let x_view = x_cuda.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let (px, _gx) = x_view.device_ptr(&stream);
        let (pwq, _g1) = wq_q.device_ptr(&stream);
        let (psq, _g2) = rs_q.device_ptr(&stream);
        let (pwk, _g3) = wq_k.device_ptr(&stream);
        let (psk, _g4) = rs_k.device_ptr(&stream);
        let (pwv, _g5) = wq_v.device_ptr(&stream);
        let (psv, _g6) = rs_v.device_ptr(&stream);
        let (pyq, _g7) = y_q.device_ptr_mut(&stream);
        let (pyk, _g8) = y_k.device_ptr_mut(&stream);
        let (pyv, _g9) = y_v.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gemv_e4m3_qkv_one_m1(
                stream.cu_stream() as *mut c_void,
                pwq as *const u8,
                psq as *const f32,
                pwk as *const u8,
                psk as *const f32,
                pwv as *const u8,
                psv as *const f32,
                px as *const u16,
                pyq as *mut u16,
                pyk as *mut u16,
                pyv as *mut u16,
                n_q as i32,
                n_k as i32,
                n_v as i32,
                k_feat as i32,
            )
        }
    };
    if rc == -1 {
        return Ok(None);
    }
    anyhow::ensure!(rc == 0, "gemv_e4m3_qkv_one_m1 rc={rc}");
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qwen3_5_moe] NV_Q38_ATTN_QKV_ONE decode path active \
                 (q|gate,k,v fp8 gemvs in one launch, rows {n_q}+{n_k}+{n_v}, k {k_feat})"
            );
        });
    }
    let wrap = |slice: cudarc::driver::CudaSlice<bf16>, n: usize| -> Tensor {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
        Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, 1usize, n),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    Ok(Some((wrap(y_q, n_q), wrap(y_k, n_k), wrap(y_v, n_v))))
}

#[cfg(feature = "cuda")]
thread_local! {
    static GDN_PRENORM_RSTD_PACK_PERSISTENT_SO_CAPTURED_GRAPHS_BAKE_ONE_POINTER:
        std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
fn gdn_prenorm_rstd_pack_tensor_rstd_ssq_count_zeroed_once_then_kernel_maintained(
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    GDN_PRENORM_RSTD_PACK_PERSISTENT_SO_CAPTURED_GRAPHS_BAKE_ONE_POINTER.with(|c| {
        let mut slot = c.borrow_mut();
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let pack: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros::<f32>(4)
            .map_err(|e| anyhow::anyhow!("alloc gdn prenorm rstd pack: {e:?}"))?;
        let storage = candle_core::CudaStorage::wrap_cuda_slice(pack, dev.clone());
        let t = Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (4usize,),
            candle_core::op::BackpropOp::none(),
            false,
        );
        *slot = Some(t.clone());
        Ok(t)
    })
}

#[cfg(feature = "cuda")]
fn dense_mlp_w4a8_eligible_shapes(ffn: &LayerFfn) -> Option<(&Mlp, usize, usize)> {
    if !nv_q38_mlp_w4a8_env_opt_in_nv_q38_mlp_w4a8_1_dp4a_int8_dot_route_for_the_nvfp4_dense_mlp_decode() {
        return None;
    }
    dense_mlp_w4a8_shapes_env_free_because_the_verify_route_self_selects(ffn)
}

#[cfg(feature = "cuda")]
fn dense_mlp_w4a8_shapes_env_free_because_the_verify_route_self_selects(
    ffn: &LayerFfn,
) -> Option<(&Mlp, usize, usize)> {
    let mlp = ffn.as_dense()?;
    let hidden = mlp.gate_proj().in_features();
    let inter = mlp.gate_proj().out_features();
    if mlp.gate_proj().nvfp4_parts().is_none()
        || mlp.up_proj().nvfp4_parts().is_none()
        || mlp.down_proj().nvfp4_parts().is_none()
    {
        return None;
    }
    if hidden % 32 != 0
        || inter % 32 != 0
        || hidden > 96 * 1024
        || inter > 48 * 1024
        || mlp.down_proj().in_features() != inter
        || mlp.down_proj().out_features() != hidden
    {
        return None;
    }
    Some((mlp, hidden, inter))
}

#[cfg(feature = "cuda")]
struct W4a8FoldedPostNormQuant {
    res_out: Tensor,
    x_q8: cudarc::driver::CudaSlice<i8>,
    x_scale: cudarc::driver::CudaSlice<f32>,
    hidden: usize,
    inter: usize,
}

#[cfg(feature = "cuda")]
struct W4a8DecodeScratch {
    inter: usize,
    plen: i32,
    gate_y: cudarc::driver::CudaSlice<half::bf16>,
    up_y: cudarc::driver::CudaSlice<half::bf16>,
    staged: cudarc::driver::CudaSlice<half::bf16>,
    partials: cudarc::driver::CudaSlice<f32>,
    act_q8: cudarc::driver::CudaSlice<i8>,
    act_scale: cudarc::driver::CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
thread_local! {
    static W4A8_DECODE_SCRATCH_POOL_KEYED_BY_INTER_AND_NEVER_FREED_BECAUSE_CAPTURED_GRAPHS_BAKE_ITS_POINTERS:
        std::cell::RefCell<Vec<W4a8DecodeScratch>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "cuda")]
struct W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(
    Option<W4a8DecodeScratch>,
);

#[cfg(feature = "cuda")]
impl std::ops::Deref for W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    type Target = W4a8DecodeScratch;
    fn deref(&self) -> &W4a8DecodeScratch {
        self.0
            .as_ref()
            .expect("lease holds its scratch until drop")
    }
}

#[cfg(feature = "cuda")]
impl std::ops::DerefMut for W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    fn deref_mut(&mut self) -> &mut W4a8DecodeScratch {
        self.0
            .as_mut()
            .expect("lease holds its scratch until drop")
    }
}

#[cfg(feature = "cuda")]
impl Drop for W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            let _ = W4A8_DECODE_SCRATCH_POOL_KEYED_BY_INTER_AND_NEVER_FREED_BECAUSE_CAPTURED_GRAPHS_BAKE_ITS_POINTERS
                .try_with(|c| c.borrow_mut().push(s));
        }
    }
}

#[cfg(feature = "cuda")]
fn w4a8_decode_scratch_take_or_build(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    inter: usize,
) -> Result<W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers> {
    use half::bf16;
    let pooled = W4A8_DECODE_SCRATCH_POOL_KEYED_BY_INTER_AND_NEVER_FREED_BECAUSE_CAPTURED_GRAPHS_BAKE_ITS_POINTERS
        .with(|c| {
            let mut v = c.borrow_mut();
            v.iter()
                .position(|s| s.inter == inter)
                .map(|i| v.swap_remove(i))
        });
    if let Some(s) = pooled {
        return Ok(W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(Some(s)));
    }
    let plen = nv_kernels::cuda::silu_mul_rowquant_i8_mk_partials_len(1, inter as i32);
    anyhow::ensure!(plen > 0, "silu_mul_rowquant_i8_mk_partials_len refused inter={inter}");
    let gate_y = unsafe { stream.alloc::<bf16>(inter).map_err(|e| anyhow::anyhow!(e))? };
    let up_y = unsafe { stream.alloc::<bf16>(inter).map_err(|e| anyhow::anyhow!(e))? };
    let staged = unsafe { stream.alloc::<bf16>(inter).map_err(|e| anyhow::anyhow!(e))? };
    let partials = unsafe {
        stream
            .alloc::<f32>(plen as usize)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let act_q8 = unsafe { stream.alloc::<i8>(inter).map_err(|e| anyhow::anyhow!(e))? };
    let act_scale = unsafe { stream.alloc::<f32>(1).map_err(|e| anyhow::anyhow!(e))? };
    Ok(W4a8ScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(Some(
        W4a8DecodeScratch {
            inter,
            plen,
            gate_y,
            up_y,
            staged,
            partials,
            act_q8,
            act_scale,
        },
    )))
}

#[cfg(feature = "cuda")]
fn w4a8_dual_silu_down_chain_after_x_quant_decode_m1(
    mlp: &Mlp,
    x_q8: &cudarc::driver::CudaSlice<i8>,
    x_scale: &cudarc::driver::CudaSlice<f32>,
    residual: &Tensor,
    hidden: usize,
    inter: usize,
    next_prenorm_rstd_emit_pack_and_eps: Option<(&Tensor, f32)>,
) -> Result<(Tensor, bool)> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    let dev = match residual.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("w4a8 chain: residual not on cuda"),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut prof = nv_layers::linear_attn::gdn_step_prof::SyncLaps::begin_if_armed(&dev);
    let res_c = residual.contiguous()?;
    let (gp, up, dp) = (
        mlp.gate_proj().nvfp4_parts(),
        mlp.up_proj().nvfp4_parts(),
        mlp.down_proj().nvfp4_parts(),
    );
    let (Some(gp), Some(up), Some(dp)) = (gp, up, dp) else {
        anyhow::bail!("w4a8 chain: nvfp4 parts vanished after eligibility");
    };
    let (gw, gs, g_alpha_recip_wg_times_recip_ig, g_ig) = gp;
    let (uw, us, u_alpha_recip_wg_times_recip_ig, u_ig) = up;
    let (dw, ds, d_alpha_recip_wg_times_recip_ig, d_ig) = dp;
    let ga_weight_global_only_because_the_q8_row_scale_rides_separately =
        g_alpha_recip_wg_times_recip_ig * g_ig;
    let ua_weight_global_only_because_the_q8_row_scale_rides_separately =
        u_alpha_recip_wg_times_recip_ig * u_ig;
    let da_weight_global_only_because_the_q8_row_scale_rides_separately =
        d_alpha_recip_wg_times_recip_ig * d_ig;

    let mut down_y = unsafe {
        stream
            .alloc::<bf16>(hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let down_qfold =
        nv_q38_down_qfold_env_opt_in_nv_q38_down_qfold_1_act_quant_in_down_gemv_prologue();
    let siluq_split =
        nv_q38_siluq_split_env_opt_in_nv_q38_siluq_split_1_multiblock_silu_quant_producer();
    let mut rstd_emitted = false;
    let mut scratch_lease = w4a8_decode_scratch_take_or_build(&stream, inter)?;
    let scratch = &mut *scratch_lease;
    let plen = scratch.plen;
    if let Some(p) = prof.as_mut() {
        p.lap("mlp_allocs");
    }
    {
        let (r_storage, rl) = res_c.storage_and_layout();
        let r_cuda = match &*r_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w4a8 chain: residual storage not cuda"),
        };
        let r_slice = r_cuda.as_cuda_slice::<bf16>()?;
        let r_view = r_slice.slice(rl.start_offset()..);
        let (r_ptr, _gr) = r_view.device_ptr(&stream);
        let (xq_ptr, _g1) = x_q8.device_ptr(&stream);
        let (xs_ptr, _g2) = x_scale.device_ptr(&stream);
        let (gy_ptr, _g3) = scratch.gate_y.device_ptr_mut(&stream);
        let (uy_ptr, _g4) = scratch.up_y.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gemv_nvfp4_w4a8_dual_m1(
                stream.cu_stream() as *mut c_void,
                cuda_slice_u8_ptr(gw, &stream) as *const u8,
                cuda_slice_u8_ptr(gs, &stream) as *const u8,
                cuda_slice_u8_ptr(uw, &stream) as *const u8,
                cuda_slice_u8_ptr(us, &stream) as *const u8,
                xq_ptr as *const i8,
                xs_ptr as *const f32,
                gy_ptr as *mut u16,
                uy_ptr as *mut u16,
                ga_weight_global_only_because_the_q8_row_scale_rides_separately,
                ua_weight_global_only_because_the_q8_row_scale_rides_separately,
                inter as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a8_dual_m1 rc={rc}");
        if let Some(p) = prof.as_mut() {
            p.lap("mlp_dual");
        }
        let (dy_ptr, _g9) = down_y.device_ptr_mut(&stream);
        if down_qfold {
            let (st_ptr, _g7) = scratch.staged.device_ptr_mut(&stream);
            let (pp_ptr, _g8) = scratch.partials.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::silu_mul_stage_partial_absmax_m1(
                    stream.cu_stream() as *mut c_void,
                    gy_ptr as *const u16,
                    uy_ptr as *const u16,
                    st_ptr as *mut u16,
                    pp_ptr as *mut f32,
                    inter as i32,
                )
            };
            anyhow::ensure!(rc == 0, "silu_mul_stage_partial_absmax_m1 rc={rc}");
            let rc = unsafe {
                nv_kernels::cuda::gemv_nvfp4_w4a8_down_residual_quant_prologue_m1(
                    stream.cu_stream() as *mut c_void,
                    cuda_slice_u8_ptr(dw, &stream) as *const u8,
                    cuda_slice_u8_ptr(ds, &stream) as *const u8,
                    st_ptr as *const u16,
                    pp_ptr as *const f32,
                    plen,
                    r_ptr as *const u16,
                    dy_ptr as *mut u16,
                    da_weight_global_only_because_the_q8_row_scale_rides_separately,
                    hidden as i32,
                    inter as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a8_down_residual_quant_prologue_m1 rc={rc}");
        } else {
            let (actq_ptr, _g5) = scratch.act_q8.device_ptr_mut(&stream);
            let (acts_ptr, _g6) = scratch.act_scale.device_ptr_mut(&stream);
            let rc = if siluq_split {
                {
                    let (st_ptr, _g7) = scratch.staged.device_ptr_mut(&stream);
                    let (pp_ptr, _g8) = scratch.partials.device_ptr_mut(&stream);
                    unsafe {
                        nv_kernels::cuda::silu_mul_rowquant_i8_mk(
                            stream.cu_stream() as *mut c_void,
                            gy_ptr as *const u16,
                            uy_ptr as *const u16,
                            st_ptr as *mut u16,
                            pp_ptr as *mut f32,
                            actq_ptr as *mut i8,
                            acts_ptr as *mut f32,
                            1,
                            inter as i32,
                        )
                    }
                }
            } else {
                unsafe {
                    nv_kernels::cuda::silu_mul_rowquant_i8_m1(
                        stream.cu_stream() as *mut c_void,
                        gy_ptr as *const u16,
                        uy_ptr as *const u16,
                        actq_ptr as *mut i8,
                        acts_ptr as *mut f32,
                        inter as i32,
                    )
                }
            };
            anyhow::ensure!(rc == 0, "silu quant producer rc={rc} split={siluq_split}");
            if let Some(p) = prof.as_mut() {
                p.lap("mlp_siluq");
            }
            let rc = match next_prenorm_rstd_emit_pack_and_eps {
                Some((pack, eps)) => {
                    let (ps, pl) = pack.storage_and_layout();
                    let p_cuda = match &*ps {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("rstd pack not cuda"),
                    };
                    let p_slice = p_cuda.as_cuda_slice::<f32>()?;
                    let p_view = p_slice.slice(pl.start_offset()..);
                    let (pack_ptr, _gp) = p_view.device_ptr(&stream);
                    rstd_emitted = true;
                    unsafe {
                        nv_kernels::cuda::gemv_nvfp4_w4a8_down_residual_m1_rstd_emit(
                            stream.cu_stream() as *mut c_void,
                            cuda_slice_u8_ptr(dw, &stream) as *const u8,
                            cuda_slice_u8_ptr(ds, &stream) as *const u8,
                            actq_ptr as *const i8,
                            acts_ptr as *const f32,
                            r_ptr as *const u16,
                            dy_ptr as *mut u16,
                            da_weight_global_only_because_the_q8_row_scale_rides_separately,
                            pack_ptr as *mut f32,
                            eps,
                            hidden as i32,
                            inter as i32,
                        )
                    }
                }
                None => unsafe {
                    nv_kernels::cuda::gemv_nvfp4_w4a8_down_residual_m1(
                        stream.cu_stream() as *mut c_void,
                        cuda_slice_u8_ptr(dw, &stream) as *const u8,
                        cuda_slice_u8_ptr(ds, &stream) as *const u8,
                        actq_ptr as *const i8,
                        acts_ptr as *const f32,
                        r_ptr as *const u16,
                        dy_ptr as *mut u16,
                        da_weight_global_only_because_the_q8_row_scale_rides_separately,
                        hidden as i32,
                        inter as i32,
                    )
                },
            };
            anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a8_down_residual_m1 rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("mlp_down");
        }
    }
    drop(scratch_lease);
    let storage = candle_core::CudaStorage::wrap_cuda_slice(down_y, dev);
    Ok((
        Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, 1usize, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ),
        rstd_emitted,
    ))
}

#[cfg(feature = "cuda")]
fn fused_post_norm_residual_rowquant_decode_m1(
    post_norm: &RmsNorm,
    ffn: &LayerFfn,
    mixed: &Tensor,
    residual: &Tensor,
) -> Result<Option<W4a8FoldedPostNormQuant>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    if !nv_q38_norm_quant_fold_env_opt_in_nv_q38_norm_quant_fold_1_post_norm_residual_rowquant_one_kernel()
    {
        return Ok(None);
    }
    let Some((_, hidden, inter)) = dense_mlp_w4a8_eligible_shapes(ffn) else {
        return Ok(None);
    };
    if mixed.dims() != [1, 1, hidden]
        || mixed.dtype() != DType::BF16
        || residual.dims() != [1, 1, hidden]
        || residual.dtype() != DType::BF16
    {
        return Ok(None);
    }
    let dev = match mixed.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let w = post_norm.weight_bf16().clone();
    if w.dtype() != DType::BF16 {
        return Ok(None);
    }
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let x_c = mixed.contiguous()?;
    let res_c = residual.contiguous()?;
    let mut res_out = unsafe {
        stream
            .alloc::<bf16>(hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut x_q8 = unsafe { stream.alloc::<i8>(hidden).map_err(|e| anyhow::anyhow!(e))? };
    let mut x_scale = unsafe { stream.alloc::<f32>(1).map_err(|e| anyhow::anyhow!(e))? };
    {
        let (xs, xl) = x_c.storage_and_layout();
        let (rs, rl) = res_c.storage_and_layout();
        let (ws, wl) = w.storage_and_layout();
        let x_cuda = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let r_cuda = match &*rs {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let w_cuda = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let x_view = x_cuda.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let r_view = r_cuda.as_cuda_slice::<bf16>()?.slice(rl.start_offset()..);
        let w_view = w_cuda.as_cuda_slice::<bf16>()?.slice(wl.start_offset()..);
        let (px, _gx) = x_view.device_ptr(&stream);
        let (pr, _gr) = r_view.device_ptr(&stream);
        let (pw, _gw) = w_view.device_ptr(&stream);
        let (pro, _g1) = res_out.device_ptr_mut(&stream);
        let (pq, _g2) = x_q8.device_ptr_mut(&stream);
        let (ps, _g3) = x_scale.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::rmsnorm_residual_writeout_rowquant_i8_m1(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                pro as *mut u16,
                pq as *mut i8,
                ps as *mut f32,
                hidden as i32,
                post_norm.eps() as f32,
            )
        };
        if rc != 0 {
            return Ok(None);
        }
    }
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qwen3_5_moe] NV_Q38_NORM_QUANT_FOLD decode path active \
                 (post-norm + residual writeout + rowquant in one kernel, hidden={hidden})"
            );
        });
    }
    let res_storage = candle_core::CudaStorage::wrap_cuda_slice(res_out, dev);
    let res_out_t = Tensor::from_storage(
        candle_core::Storage::Cuda(res_storage),
        (1usize, 1usize, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(Some(W4a8FoldedPostNormQuant {
        res_out: res_out_t,
        x_q8,
        x_scale,
        hidden,
        inter,
    }))
}

#[cfg(feature = "cuda")]
fn fused_dense_mlp_nvfp4_w4a8_one_rowquant_shared_by_gate_up_silu_quant_producer_then_down_with_residual_writeback_decode_m1(
    ffn: &LayerFfn,
    x: &Tensor,
    residual: &Tensor,
    next_prenorm_rstd_emit_pack_and_eps: Option<(&Tensor, f32)>,
) -> Result<Option<(Tensor, bool)>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    let Some((mlp, hidden, inter)) = dense_mlp_w4a8_eligible_shapes(ffn) else {
        return Ok(None);
    };
    if x.dims() != [1, 1, hidden]
        || x.dtype() != DType::BF16
        || residual.dims() != [1, 1, hidden]
        || residual.dtype() != DType::BF16
    {
        return Ok(None);
    }
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let x_c = x.contiguous()?;
    let mut x_q8 = unsafe { stream.alloc::<i8>(hidden).map_err(|e| anyhow::anyhow!(e))? };
    let mut x_scale = unsafe { stream.alloc::<f32>(1).map_err(|e| anyhow::anyhow!(e))? };
    {
        let (x_storage, xl) = x_c.storage_and_layout();
        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
        let x_view = x_slice.slice(xl.start_offset()..);
        let (x_ptr, _gx) = x_view.device_ptr(&stream);
        let (xq_ptr, _g1) = x_q8.device_ptr_mut(&stream);
        let (xs_ptr, _g2) = x_scale.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::rowquant_i8(
                stream.cu_stream() as *mut c_void,
                x_ptr as *const u16,
                xq_ptr as *mut i8,
                xs_ptr as *mut f32,
                1,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "rowquant_i8 rc={rc}");
    }
    Ok(Some(w4a8_dual_silu_down_chain_after_x_quant_decode_m1(
        mlp,
        &x_q8,
        &x_scale,
        residual,
        hidden,
        inter,
        next_prenorm_rstd_emit_pack_and_eps,
    )?))
}

#[cfg(feature = "cuda")]
pub fn nv_q38_norm_writeout_env_opt_in_nv_q38_norm_writeout_1() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_NORM_WRITEOUT")
}

#[cfg(feature = "cuda")]
fn rmsnorm_residual_decode_m1_writeout_skipping_the_dtod_residual_copy(
    norm: &RmsNorm,
    x: &Tensor,
    residual: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    if !nv_q38_norm_writeout_env_opt_in_nv_q38_norm_writeout_1() {
        return Ok(None);
    }
    if x.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || x.dims() != residual.dims()
    {
        return Ok(None);
    }
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let x_c = x.contiguous()?;
    let res_c = residual.contiguous()?;
    let w = norm.weight_bf16().clone();
    if w.dtype() != DType::BF16 {
        return Ok(None);
    }
    let dims = x_c.dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out_dev = unsafe {
        stream
            .alloc::<bf16>(batch * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut res_out_dev = unsafe {
        stream
            .alloc::<bf16>(batch * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (rs, rl) = res_c.storage_and_layout();
        let (ws, wl) = w.storage_and_layout();
        let x_cuda = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let r_cuda = match &*rs {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let w_cuda = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let x_view = x_cuda.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let r_view = r_cuda.as_cuda_slice::<bf16>()?.slice(rl.start_offset()..);
        let w_view = w_cuda.as_cuda_slice::<bf16>()?.slice(wl.start_offset()..);
        let (px, _gx) = x_view.device_ptr(&stream);
        let (pr, _gr) = r_view.device_ptr(&stream);
        let (pw, _gw) = w_view.device_ptr(&stream);
        let (pro, _gro) = res_out_dev.device_ptr_mut(&stream);
        let (po, _go) = out_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::rmsnorm_residual_writeout_bf16(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                pro as *mut u16,
                po as *mut u16,
                batch,
                hidden,
                norm.eps() as f32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "rmsnorm_residual_writeout_bf16 rc={rc}");
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qwen3_5_moe] NV_Q38_NORM_WRITEOUT decode norm path active \
                 (residual written out, no dtod copy, hidden={hidden})"
            );
        });
    }
    let shape: candle_core::Shape = dims.into();
    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
    let normed = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let res_storage = candle_core::CudaStorage::wrap_cuda_slice(res_out_dev, dev);
    let new_residual = Tensor::from_storage(
        candle_core::Storage::Cuda(res_storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(Some((normed, new_residual)))
}

#[cfg(feature = "cuda")]
fn dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
    ffn: &LayerFfn,
    normed_post: &Tensor,
    seq: usize,
) -> Result<Tensor> {
    if seq == 1 {
        if let Some(out) =
            fused_dense_mlp_nvfp4_w4a16_two_kernels_decode_m1_more_precise_than_the_padded_a4_tensor_core_route(
                ffn,
                normed_post,
            )?
        {
            return Ok(out);
        }
    }
    if (2..=SMALL_M_W4A8_VERIFY_MAX_TOKENS).contains(&seq) {
        if let Some(out) = fused_dense_mlp_nvfp4_w4a8_small_m_dp4a_for_spec_verify(
            ffn,
            normed_post,
            seq,
        )? {
            return Ok(out);
        }
    }
    ffn.forward(normed_post)
}

#[cfg(feature = "cuda")]
const SMALL_M_W4A8_VERIFY_MAX_TOKENS: usize = 8;

#[cfg(feature = "cuda")]
const MK_DP4A_VERIFY_MLP_WINS_ONLY_AT_M3_INGRAPH_VERIFY_25_54_VS_25_95_MS_LOSES_M4_27_44_VS_27_0_AND_M6_34_26_VS_29_0: usize = 3;

#[cfg(feature = "cuda")]
pub fn nv_q38_w4a8_mk_env_opt_in_nv_q38_w4a8_mk_1_small_m_dp4a_verify_route() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_W4A8_MK")
}

#[cfg(feature = "cuda")]
pub fn verify_mlp_mk_dp4a_selected_for_m_force_1_kill_0_else_auto_under_verify_tc(
    m: usize,
) -> bool {
    match std::env::var("NV_Q38_W4A8_MK").ok().as_deref() {
        Some("1") => true,
        Some("mma") => false,
        Some("0") => false,
        _ => {
            nv_layers::linear_attn::verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms()
                && m <= MK_DP4A_VERIFY_MLP_WINS_ONLY_AT_M3_INGRAPH_VERIFY_25_54_VS_25_95_MS_LOSES_M4_27_44_VS_27_0_AND_M6_34_26_VS_29_0
        }
    }
}

#[cfg(feature = "cuda")]
fn fused_dense_mlp_nvfp4_w4a16_mma_for_spec_verify(
    ffn: &LayerFfn,
    x: &Tensor,
    seq: usize,
) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    use std::ffi::c_void;

    if std::env::var("NV_Q38_W4A8_MK").ok().as_deref() != Some("mma") {
        return Ok(None);
    }
    if !(2..=8).contains(&seq) {
        return Ok(None);
    }
    let Some((mlp, hidden, inter)) = dense_mlp_w4a8_shapes_env_free_because_the_verify_route_self_selects(ffn) else {
        return Ok(None);
    };
    if x.dims() != [1, seq, hidden] || x.dtype() != DType::BF16 {
        return Ok(None);
    }
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let x_c = x.contiguous()?;
    let m = seq;
    let (gp, up, dp) = (
        mlp.gate_proj().nvfp4_parts(),
        mlp.up_proj().nvfp4_parts(),
        mlp.down_proj().nvfp4_parts(),
    );
    let (Some(gp), Some(up), Some(dp)) = (gp, up, dp) else {
        return Ok(None);
    };
    let (gw, gs, g_a, g_ig) = gp;
    let (uw, us, u_a, u_ig) = up;
    let (dw, ds, d_a, d_ig) = dp;
    let (ga, ua, da) = (g_a * g_ig, u_a * u_ig, d_a * d_ig);

    let mut gate_y: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(m * inter).map_err(|e| anyhow::anyhow!(e))? };
    let mut up_y: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(m * inter).map_err(|e| anyhow::anyhow!(e))? };
    let mut act: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(m * inter).map_err(|e| anyhow::anyhow!(e))? };
    let mut down_y: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(m * hidden).map_err(|e| anyhow::anyhow!(e))? };
    {
        let (xs_store, xl) = x_c.storage_and_layout();
        let x_cuda = match &*xs_store {
            candle_core::Storage::Cuda(st) => st,
            _ => return Ok(None),
        };
        let x_view = x_cuda.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let (xp, _gx) = x_view.device_ptr(&stream);
        let (gwp, _g1) = gw.device_ptr(&stream);
        let (gsp, _g2) = gs.device_ptr(&stream);
        let (uwp, _g3) = uw.device_ptr(&stream);
        let (usp, _g4) = us.device_ptr(&stream);
        let (gyp, _g5) = gate_y.device_ptr_mut(&stream);
        let (uyp, _g6) = up_y.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gemm_nvfp4_w4a16_mk_dual(
                stream.cu_stream() as *mut c_void,
                gwp as *const u8,
                gsp as *const u8,
                uwp as *const u8,
                usp as *const u8,
                xp as *const u16,
                gyp as *mut u16,
                uyp as *mut u16,
                ga,
                ua,
                m as i32,
                inter as i32,
                inter as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "w4a16 mma gate/up rc={rc}");
        let (ap, _g7) = act.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::silu_mul_bf16_candle(
                stream.cu_stream() as *mut c_void,
                gyp as *const u16,
                uyp as *const u16,
                ap as *mut u16,
                m * inter,
            )
        };
        anyhow::ensure!(rc == 0, "w4a16 mma silu_mul rc={rc}");
        let (dwp, _g8) = dw.device_ptr(&stream);
        let (dsp, _g9) = ds.device_ptr(&stream);
        let (dyp, _ga2) = down_y.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gemm_nvfp4_w4a16_mk_dual(
                stream.cu_stream() as *mut c_void,
                dwp as *const u8,
                dsp as *const u8,
                std::ptr::null(),
                std::ptr::null(),
                ap as *const u16,
                dyp as *mut u16,
                std::ptr::null_mut(),
                da,
                0.0,
                m as i32,
                hidden as i32,
                hidden as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "w4a16 mma down rc={rc}");
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(down_y, dev);
    Ok(Some(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (1usize, m, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    )))
}

#[cfg(feature = "cuda")]
fn fused_dense_mlp_nvfp4_w4a8_small_m_dp4a_for_spec_verify(
    ffn: &LayerFfn,
    x: &Tensor,
    seq: usize,
) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    if let Some(y) = fused_dense_mlp_nvfp4_w4a16_mma_for_spec_verify(ffn, x, seq)? {
        return Ok(Some(y));
    }
    if !verify_mlp_mk_dp4a_selected_for_m_force_1_kill_0_else_auto_under_verify_tc(seq) {
        return Ok(None);
    }
    let Some((mlp, hidden, inter)) = dense_mlp_w4a8_shapes_env_free_because_the_verify_route_self_selects(ffn) else {
        return Ok(None);
    };
    if x.dims() != [1, seq, hidden] || x.dtype() != DType::BF16 {
        return Ok(None);
    }
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let x_c = x.contiguous()?;
    let m = seq;
    let (gp, up, dp) = (
        mlp.gate_proj().nvfp4_parts(),
        mlp.up_proj().nvfp4_parts(),
        mlp.down_proj().nvfp4_parts(),
    );
    let (Some(gp), Some(up), Some(dp)) = (gp, up, dp) else {
        return Ok(None);
    };
    let (gw, gs, g_alpha_recip_wg_times_recip_ig, g_ig) = gp;
    let (uw, us, u_alpha_recip_wg_times_recip_ig, u_ig) = up;
    let (dw, ds, d_alpha_recip_wg_times_recip_ig, d_ig) = dp;
    let ga = g_alpha_recip_wg_times_recip_ig * g_ig;
    let ua = u_alpha_recip_wg_times_recip_ig * u_ig;
    let da = d_alpha_recip_wg_times_recip_ig * d_ig;

    let plen = nv_kernels::cuda::silu_mul_rowquant_i8_mk_partials_len(m as i32, inter as i32);
    if plen <= 0 {
        return Ok(None);
    }
    let mut prof = nv_layers::linear_attn::gdn_step_prof::SyncLaps::begin_if_armed(&dev);
    let mut x_q8 = unsafe {
        stream
            .alloc::<i8>(m * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut x_scales = unsafe { stream.alloc::<f32>(m).map_err(|e| anyhow::anyhow!(e))? };
    let mut gate_y = unsafe {
        stream
            .alloc::<bf16>(m * inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut up_y = unsafe {
        stream
            .alloc::<bf16>(m * inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut act_staged = unsafe {
        stream
            .alloc::<bf16>(m * inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut partials = unsafe {
        stream
            .alloc::<f32>(plen as usize)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut act_q8 = unsafe {
        stream
            .alloc::<i8>(m * inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut act_scales = unsafe { stream.alloc::<f32>(m).map_err(|e| anyhow::anyhow!(e))? };
    let mut down_y = unsafe {
        stream
            .alloc::<bf16>(m * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    {
        let (x_storage, xl) = x_c.storage_and_layout();
        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
        let x_view = x_slice.slice(xl.start_offset()..);
        let (x_ptr, _gx) = x_view.device_ptr(&stream);
        let (xq_ptr, _g1) = x_q8.device_ptr_mut(&stream);
        let (xs_ptr, _g2) = x_scales.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::rowquant_i8(
                stream.cu_stream() as *mut c_void,
                x_ptr as *const u16,
                xq_ptr as *mut i8,
                xs_ptr as *mut f32,
                m as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "rowquant_i8 m={m} rc={rc}");
        if let Some(p) = prof.as_mut() {
            p.lap("mk_rowquant");
        }
        let (gy_ptr, _g3) = gate_y.device_ptr_mut(&stream);
        let (uy_ptr, _g4) = up_y.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gemv_nvfp4_w4a8_dual_mk(
                stream.cu_stream() as *mut c_void,
                cuda_slice_u8_ptr(gw, &stream) as *const u8,
                cuda_slice_u8_ptr(gs, &stream) as *const u8,
                cuda_slice_u8_ptr(uw, &stream) as *const u8,
                cuda_slice_u8_ptr(us, &stream) as *const u8,
                xq_ptr as *const i8,
                xs_ptr as *const f32,
                gy_ptr as *mut u16,
                uy_ptr as *mut u16,
                ga,
                ua,
                m as i32,
                inter as i32,
                inter as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a8_dual_mk m={m} rc={rc}");
        if let Some(p) = prof.as_mut() {
            p.lap("mk_mlp_dual");
        }
        let (st_ptr, _g5) = act_staged.device_ptr_mut(&stream);
        let (pp_ptr, _g6) = partials.device_ptr_mut(&stream);
        let (aq_ptr, _g7) = act_q8.device_ptr_mut(&stream);
        let (as_ptr, _g8) = act_scales.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::silu_mul_rowquant_i8_mk(
                stream.cu_stream() as *mut c_void,
                gy_ptr as *const u16,
                uy_ptr as *const u16,
                st_ptr as *mut u16,
                pp_ptr as *mut f32,
                aq_ptr as *mut i8,
                as_ptr as *mut f32,
                m as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "silu_mul_rowquant_i8_mk m={m} rc={rc}");
        if let Some(p) = prof.as_mut() {
            p.lap("mk_mlp_siluq");
        }
        let (dy_ptr, _g9) = down_y.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gemv_nvfp4_w4a8_down_residual_mk(
                stream.cu_stream() as *mut c_void,
                cuda_slice_u8_ptr(dw, &stream) as *const u8,
                cuda_slice_u8_ptr(ds, &stream) as *const u8,
                aq_ptr as *const i8,
                as_ptr as *const f32,
                std::ptr::null(),
                dy_ptr as *mut u16,
                da,
                m as i32,
                hidden as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gemv_nvfp4_w4a8_down_residual_mk m={m} rc={rc}");
        if let Some(p) = prof.as_mut() {
            p.lap("mk_mlp_down");
        }
    }
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qwen3_5_moe] NV_Q38_W4A8_MK small-m dp4a verify route active \
                 (m={m}, hidden={hidden}, inter={inter})"
            );
        });
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(down_y, dev);
    Ok(Some(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (1usize, m, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    )))
}

#[cfg(not(feature = "cuda"))]
pub struct Qwen3MoeKvCache {
    full_slot_for_layer: Vec<Option<usize>>,
    lin_attn_for_layer: Vec<Option<usize>>,
    lin_attn_states: Vec<Option<LinAttnState>>,
    current_len: usize,
    max_seq_len: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    device: Device,
}

#[cfg(feature = "cuda")]
impl Qwen3MoeKvCache {
    pub fn new(
        config: &Qwen3MoeConfig,
        max_seq_len: usize,
        device: &Device,
        _dtype: DType,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Qwen3MoeKvCache (FP8) requires CUDA device"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut full_slot_for_layer = Vec::with_capacity(config.num_hidden_layers);
        let mut lin_attn_for_layer = Vec::with_capacity(config.num_hidden_layers);
        let mut n_full = 0usize;
        let mut n_lin = 0usize;
        for ty in &config.layer_types {
            match ty {
                LayerType::FullAttention => {
                    full_slot_for_layer.push(Some(n_full));
                    lin_attn_for_layer.push(None);
                    n_full += 1;
                }
                LayerType::LinearAttention => {
                    full_slot_for_layer.push(None);
                    lin_attn_for_layer.push(Some(n_lin));
                    n_lin += 1;
                }
            }
        }
        let lin_attn_states = (0..n_lin).map(|_| None).collect();
        let lin_ckpts: Vec<Vec<LinAttnState>> = (0..n_lin).map(|_| Vec::new()).collect();
        let fused_lin_verify_ckpts: Vec<Vec<LinAttnState>> = (0..n_lin).map(|_| Vec::new()).collect();
        let n_kv = config.num_key_value_heads;
        let hd = config.head_dim;
        let elem_count = max_seq_len * n_kv * hd;
        let scale_count = max_seq_len * n_kv;
        let mut layers = Vec::with_capacity(n_full);
        for _ in 0..n_full {
            let k_fp8 = stream
                .alloc_zeros::<u8>(elem_count)
                .map_err(|e| anyhow::anyhow!(e))?;
            let v_fp8 = stream
                .alloc_zeros::<u8>(elem_count)
                .map_err(|e| anyhow::anyhow!(e))?;
            let k_scales = stream
                .alloc_zeros::<f32>(scale_count)
                .map_err(|e| anyhow::anyhow!(e))?;
            let v_scales = stream
                .alloc_zeros::<f32>(scale_count)
                .map_err(|e| anyhow::anyhow!(e))?;
            layers.push(Qwen3MoeKvSlotFp8 {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            });
        }
        let write_pos_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let n_total_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let fp8_smem_cap_positions =
            crate::laguna_fp8::LagunaKvCacheFp8::max_seq_len_for_fp8_decode(hd);
        let scratch_heads = if max_seq_len > fp8_smem_cap_positions {
            config.num_attention_heads
        } else {
            0
        };
        let scores_scratch_because_smem_decode_caps_at_12k_positions_at_48kb =
            if scratch_heads > 0 {
                Some(
                    stream
                        .alloc_zeros::<f32>(scratch_heads * max_seq_len)
                        .map_err(|e| anyhow::anyhow!(e))?,
                )
            } else {
                None
            };
        let splitk_scratch_and_fan_in =
            if nv_q36_graphed_decode_fix_env_routes_decode_to_the_gemma4_proven_splitk_flash() {
                let elems = nv_kernels::cuda::flash_splitk_scratch_elems(
                    config.num_attention_heads as i32,
                    hd as i32,
                );
                anyhow::ensure!(
                    elems > 0,
                    "flash_splitk_scratch_elems returned {elems} for nh {} hd {hd}",
                    config.num_attention_heads
                );
                let elems = if nv_q38_decode_kvshare_env_opt_in_group_fold_reads_kv_once_per_kv_head_not_once_per_q_head()
                    .is_some()
                {
                    elems.max(
                        config.num_attention_heads
                            * M1_DECODE_KVSHARE_SCRATCH_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS
                            * (hd + 2),
                    )
                } else {
                    elems
                };
                let scratch = stream
                    .alloc_zeros::<f32>(elems)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let fan_in = stream
                    .alloc_zeros::<u32>(config.num_attention_heads)
                    .map_err(|e| anyhow::anyhow!(e))?;
                Some((scratch, fan_in))
            } else {
                None
            };
        let mk_verify_scratch_and_fan_in = if crate::host_serving_util::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
            "NV_Q38_MK_VERIFY",
        ) && hd <= MK_VERIFY_MAX_HEAD_DIM_512_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
        {
            let elems = config.num_attention_heads
                * MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
                * MK_VERIFY_SPLITS_CEILING_32_MATCHES_FLASH_SPLITS_PICK
                * (hd + 2);
            Some((
                stream
                    .alloc_zeros::<f32>(elems)
                    .map_err(|e| anyhow::anyhow!(e))?,
                stream
                    .alloc_zeros::<u32>(config.num_attention_heads)
                    .map_err(|e| anyhow::anyhow!(e))?,
            ))
        } else {
            None
        };
        Ok(Self {
            layers,
            full_slot_for_layer,
            lin_attn_for_layer,
            lin_attn_states,
            lin_ckpts,
            fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory: fused_lin_verify_ckpts,
            fused_lin_verify_rows_pending_rollback: 0,
            capture_lin_ckpts: false,
            fused_lin_attn: nv_layers::linear_attn::fused_decode_env() == Some(true),
            current_len: 0,
            max_seq_len,
            n_kv_heads: n_kv,
            head_dim: hd,
            device: device.clone(),
            write_pos_dev,
            n_total_dev,
            host_write_pos: Box::new([0i32; 1]),
            host_n_total: Box::new([0i32; 1]),
            scores_scratch_because_smem_decode_caps_at_12k_positions_at_48kb,
            scratch_heads,
            splitk_scratch_and_fan_in_gated_by_nv_q36_graphed_decode_fix_because_the_smem_kernel_serializes_a_block_reduce_per_kv_position_costing_82ms_at_8k: splitk_scratch_and_fan_in,
            mk_verify_scratch_and_fan_in_routing_2_to_8_row_chains_off_the_full_kv_dequant_view_kill_switch_nv_q38_mk_verify_0: mk_verify_scratch_and_fan_in,
            verify_tree_parents_dev_written_once_because_the_two_branch_shape_is_fixed: None,
            verify_tree_rows_active: None,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
        for st in &mut self.lin_attn_states {
            *st = None;
        }
    }

    pub fn full_slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.full_slot_for_layer.get(layer).copied().flatten()
    }

    pub fn lin_attn_slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.lin_attn_for_layer.get(layer).copied().flatten()
    }

    pub fn set_capture_lin_ckpts(&mut self, on: bool) {
        self.capture_lin_ckpts = on;
        for v in &mut self.lin_ckpts {
            v.clear();
        }
    }

    pub fn set_fused_lin_verify_rows_pending(&mut self, rows: usize) {
        self.fused_lin_verify_rows_pending_rollback = rows;
    }

    pub fn rollback_lin_to(&mut self, consumed: usize) -> Result<()> {
        anyhow::ensure!(consumed >= 1, "rollback_lin_to: consumed must be >= 1");
        if self.fused_lin_verify_rows_pending_rollback > 0 {
            anyhow::ensure!(
                consumed <= self.fused_lin_verify_rows_pending_rollback,
                "rollback_lin_to: consumed {} > fused verify rows {}",
                consumed,
                self.fused_lin_verify_rows_pending_rollback
            );
            for slot in 0..self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory.len() {
                let ckpts = &self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory[slot];
                if ckpts.is_empty() {
                    continue;
                }
                let live = self.lin_attn_states[slot].as_ref().ok_or_else(|| {
                    anyhow::anyhow!("rollback_lin_to: fused verify slot {slot} has no live state")
                })?;
                anyhow::ensure!(
                    live.is_fused(),
                    "rollback_lin_to: fused verify ckpts present but live state of slot {slot} \
                     is not fused; the captured graph would be updating buffers this state no \
                     longer aliases"
                );
                live.copy_data_from(&ckpts[consumed - 1])?;
            }
            self.fused_lin_verify_rows_pending_rollback = 0;
            return Ok(());
        }
        for slot in 0..self.lin_ckpts.len() {
            if self.lin_ckpts[slot].is_empty() {
                continue;
            }
            anyhow::ensure!(
                consumed <= self.lin_ckpts[slot].len(),
                "rollback_lin_to: consumed {} > captured {} for slot {}",
                consumed,
                self.lin_ckpts[slot].len(),
                slot
            );
            let restored = self.lin_ckpts[slot][consumed - 1].deep_clone()?;
            self.lin_attn_states[slot] = Some(restored);
        }
        for v in &mut self.lin_ckpts {
            v.clear();
        }
        self.capture_lin_ckpts = false;
        Ok(())
    }

    fn lin_attn_step(
        &mut self,
        slot: usize,
        la: &LinearAttention,
        input: &Tensor,
    ) -> Result<Tensor> {
        let t_len = input.dims().get(1).copied().unwrap_or(0);
        let fused_verify_ready = t_len > 1
            && self.fused_lin_attn
            && self
                .fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory
                .get(slot)
                .map(|v| v.len() >= t_len)
                .unwrap_or(false)
            && self.lin_attn_states[slot]
                .as_ref()
                .map(|s| s.is_fused())
                .unwrap_or(false);
        let tree_rows_this_pass = self.verify_tree_rows_active.filter(|&r| r == t_len);
        if fused_verify_ready
            && (nv_layers::linear_attn::mrow_verify_env_read_per_call_so_one_process_can_ab_both_verify_paths()
                || nv_layers::linear_attn::verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms())
        {
            let st = self.lin_attn_states[slot].as_ref().unwrap();
            let ckpts = &self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory[slot];
            let tree_parents = match tree_rows_this_pass {
                Some(_) => Some(
                    self.verify_tree_parents_dev_written_once_because_the_two_branch_shape_is_fixed
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "lin_attn_step: tree rows active but the parents buffer is gone"
                            )
                        })?,
                ),
                None => None,
            };
            if let Some(out) = la
                .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                    input,
                    st,
                    &ckpts[..t_len],
                    tree_parents,
                )?
            {
                self.fused_lin_verify_rows_pending_rollback = t_len;
                return Ok(out);
            }
        }
        anyhow::ensure!(
            tree_rows_this_pass.is_none(),
            "lin_attn_step: a tree verify reached the row-sequential GDN fallback (slot {slot}); \
             every non-mrow arm advances state row-by-row and a sibling row would read the other \
             branch's state, so the tree round refuses instead of committing lossy logits"
        );
        if fused_verify_ready {
            let out = {
                let st = self.lin_attn_states[slot].as_ref().unwrap();
                let ckpts = &self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory[slot];
                let mut outs: Vec<Tensor> = Vec::with_capacity(t_len);
                for j in 0..t_len {
                    let xj = input.narrow(1, j, 1)?.copy()?;
                    let out_j = la.forward_decode_fused(&xj, st)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "lin_attn_step fused verify: slot {slot} row {j} refused the fused \
                             decode it was preallocated for"
                        )
                    })?;
                    ckpts[j].copy_data_from(st)?;
                    outs.push(out_j);
                }
                let refs: Vec<&Tensor> = outs.iter().collect();
                Tensor::cat(&refs, 1)?
            };
            self.fused_lin_verify_rows_pending_rollback = t_len;
            return Ok(out);
        }
        if self.capture_lin_ckpts && t_len > 1 {
            let mut state = self.lin_attn_states[slot].take();
            let (out, ckpts) = la.forward_with_state_capture(input, &mut state, true)?;
            self.lin_attn_states[slot] = state;
            self.lin_ckpts[slot] = ckpts;
            return Ok(out);
        }
        if self.fused_lin_attn
            && t_len == 1
            && matches!(self.device, Device::Cuda(_))
            && la.fused_decode_supported()
        {
            let already_fused = self.lin_attn_states[slot]
                .as_ref()
                .map(|s| s.is_fused())
                .unwrap_or(false);
            if !already_fused {
                match la.new_fused_state(&self.device) {
                    Ok(fresh) => {
                        let seeded = match &self.lin_attn_states[slot] {
                            Some(prev) => fresh.copy_data_from(prev).is_ok(),
                            None => true,
                        };
                        if seeded {
                            self.lin_attn_states[slot] = Some(fresh);
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[qwen3_5_moe] fused lin-attn state alloc failed (slot {slot}), \
                             falling back: {e:#}"
                        );
                    }
                }
            }
            if let Some(st) = &self.lin_attn_states[slot] {
                if st.is_fused() {
                    if let Some(out) = la.forward_decode_fused(input, st)? {
                        return Ok(out);
                    }
                }
            }
        }
        let mut state = self.lin_attn_states[slot].take();
        let out = la.forward_with_state(input, &mut state)?;
        self.lin_attn_states[slot] = state;
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn lin_attn_step_prenorm_folded(
        &mut self,
        slot: usize,
        la: &LinearAttention,
        x_raw: &Tensor,
        pre_norm_weight_bf16: &Tensor,
        rstd_pack: &Tensor,
    ) -> Result<Option<Tensor>> {
        if !(self.fused_lin_attn
            && matches!(self.device, Device::Cuda(_))
            && la.fused_decode_supported())
        {
            return Ok(None);
        }
        let already_fused = self.lin_attn_states[slot]
            .as_ref()
            .map(|s| s.is_fused())
            .unwrap_or(false);
        if !already_fused {
            match la.new_fused_state(&self.device) {
                Ok(fresh) => {
                    let seeded = match &self.lin_attn_states[slot] {
                        Some(prev) => fresh.copy_data_from(prev).is_ok(),
                        None => true,
                    };
                    if seeded {
                        self.lin_attn_states[slot] = Some(fresh);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[qwen3_5_moe] fused lin-attn state alloc failed (slot {slot}), \
                         prenorm fold falling back: {e:#}"
                    );
                }
            }
        }
        if let Some(st) = &self.lin_attn_states[slot] {
            if st.is_fused() {
                return la
                    .forward_decode_fused_prenorm_folded_reading_raw_x_because_the_layer_rmsnorm_kernel_is_gone(
                        x_raw,
                        pre_norm_weight_bf16,
                        rstd_pack,
                        st,
                    );
            }
        }
        Ok(None)
    }

    fn lin_attn_step_state_pinned_for_prefill_graph(
        &self,
        slot: usize,
        la: &LinearAttention,
        input: &Tensor,
    ) -> Result<Tensor> {
        let pinned = self
            .lin_attn_states
            .get(slot)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!("lin_attn_step pinned: slot {slot} has no installed pinned state")
            })?;
        anyhow::ensure!(
            !pinned.is_fused(),
            "lin_attn_step pinned: slot {slot} holds a fused state; segment graphs bake the \
             non-fused chunk-prefill layout"
        );
        let mut tmp = Some(pinned.clone());
        let out = la.forward_with_state(input, &mut tmp)?;
        let fresh = tmp.ok_or_else(|| {
            anyhow::anyhow!("lin_attn_step pinned: forward_with_state left no state for slot {slot}")
        })?;
        pinned.copy_data_from(&fresh)?;
        Ok(out)
    }

    pub fn set_fused_lin_attn(&mut self, on: bool) {
        self.fused_lin_attn = match nv_layers::linear_attn::fused_decode_env() {
            Some(v) => v,
            None => on,
        };
    }

    pub fn fused_lin_attn(&self) -> bool {
        self.fused_lin_attn
    }

    pub fn has_lin_attn_layers(&self) -> bool {
        self.lin_attn_for_layer.iter().any(|s| s.is_some())
    }

    pub fn snapshot_lin_states(&self) -> Result<Vec<Option<LinAttnState>>> {
        let mut out = Vec::with_capacity(self.lin_attn_states.len());
        for st in &self.lin_attn_states {
            out.push(match st {
                Some(s) => Some(s.deep_clone()?),
                None => None,
            });
        }
        Ok(out)
    }

    pub fn restore_lin_states(&mut self, snaps: &[Option<LinAttnState>]) -> Result<()> {
        anyhow::ensure!(
            snaps.len() == self.lin_attn_states.len(),
            "lin state snapshot length mismatch"
        );
        for (slot, snap) in snaps.iter().enumerate() {
            match (&self.lin_attn_states[slot], snap) {
                (Some(cur), Some(s)) if cur.is_fused() => {
                    cur.copy_data_from(s)?;
                }
                (Some(cur), None) if cur.is_fused() => {
                    cur.zero_data()?;
                }
                _ => {
                    self.lin_attn_states[slot] = snap.clone();
                }
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_fused_verify_ckpt_slot(
        &mut self,
        slot: usize,
        la: &LinearAttention,
        rows: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            slot < self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory.len(),
            "ensure_fused_verify_ckpt_slot: slot {slot} out of range"
        );
        anyhow::ensure!(
            la.fused_decode_supported(),
            "ensure_fused_verify_ckpt_slot: fused decode unsupported for slot {slot}; a captured \
             verify would fall into the host-checkpoint path mid-capture"
        );
        if self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory
            [slot]
            .len()
            < rows
        {
            let slab = la
                .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
                    &self.device,
                    rows,
                )?;
            self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory
                [slot] = slab;
        }
        Ok(())
    }

    pub fn mk_verify_route_available(&self, rows: usize) -> bool {
        rows >= 2
            && rows <= MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
            && self.head_dim <= MK_VERIFY_MAX_HEAD_DIM_512_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
            && self
                .mk_verify_scratch_and_fan_in_routing_2_to_8_row_chains_off_the_full_kv_dequant_view_kill_switch_nv_q38_mk_verify_0
                .is_some()
    }

    pub fn ensure_verify_tree_two_branch_m6(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.head_dim == 256,
            "verify tree: the kvshare hole-mask attention route serves head_dim 256, this cache \
             holds {}",
            self.head_dim
        );
        anyhow::ensure!(
            self.mk_verify_route_available(VERIFY_TREE_TWO_BRANCH_M6_ROWS),
            "verify tree: the mk verify scratch is unavailable (NV_Q38_MK_VERIFY=0 or geometry); \
             tree rows have no lossless attention route without it"
        );
        if self
            .verify_tree_parents_dev_written_once_because_the_two_branch_shape_is_fixed
            .is_some()
        {
            return Ok(());
        }
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("verify tree: CUDA only"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut parents = stream
            .alloc_zeros::<i32>(VERIFY_TREE_TWO_BRANCH_M6_ROWS)
            .map_err(|e| anyhow::anyhow!("alloc verify tree parents: {e:?}"))?;
        stream
            .memcpy_htod(
                &VERIFY_TREE_TWO_BRANCH_M6_PARENTS_ANCHOR_THEN_TOP1_CHAIN_THEN_TOP2_SIBLING_WITH_CONTINUATION[..],
                &mut parents,
            )
            .map_err(|e| anyhow::anyhow!("htod verify tree parents: {e:?}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after tree parents write: {e:?}"))?;
        self.verify_tree_parents_dev_written_once_because_the_two_branch_shape_is_fixed =
            Some(parents);
        Ok(())
    }

    pub fn set_verify_tree_rows_active(&mut self, rows: Option<usize>) -> Result<()> {
        if let Some(r) = rows {
            anyhow::ensure!(
                r == VERIFY_TREE_TWO_BRANCH_M6_ROWS,
                "verify tree: only the two-branch m=6 shape exists, got rows={r}"
            );
            anyhow::ensure!(
                self.verify_tree_parents_dev_written_once_because_the_two_branch_shape_is_fixed
                    .is_some(),
                "verify tree: call ensure_verify_tree_two_branch_m6 before activating tree rows"
            );
        }
        self.verify_tree_rows_active = rows;
        Ok(())
    }

    pub fn verify_tree_rows_active(&self) -> Option<usize> {
        self.verify_tree_rows_active
    }

    pub fn compact_verify_tree_branch_b_rows_down(
        &mut self,
        pos: usize,
        accepted_rows: usize,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        anyhow::ensure!(
            accepted_rows >= 1
                && accepted_rows
                    <= VERIFY_TREE_TWO_BRANCH_M6_ROWS - VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW,
            "verify tree compaction: accepted_rows {accepted_rows} outside the branch-B span"
        );
        let src_slot = pos + VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW;
        let dst_slot = pos + 1;
        anyhow::ensure!(
            src_slot + accepted_rows <= self.max_seq_len,
            "verify tree compaction: rows [{src_slot}, {}) beyond max_seq {}",
            src_slot + accepted_rows,
            self.max_seq_len
        );
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("verify tree compaction: CUDA only"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let row_kv_bytes = self.n_kv_heads * self.head_dim;
        let row_scale_bytes = self.n_kv_heads * std::mem::size_of::<f32>();
        for slot in &self.layers {
            let (k_ptr, _gk) = slot.k_fp8.device_ptr(&stream);
            let (v_ptr, _gv) = slot.v_fp8.device_ptr(&stream);
            let (ks_ptr, _gks) = slot.k_scales.device_ptr(&stream);
            let (vs_ptr, _gvs) = slot.v_scales.device_ptr(&stream);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    k_ptr + (dst_slot * row_kv_bytes) as u64,
                    k_ptr + (src_slot * row_kv_bytes) as u64,
                    accepted_rows * row_kv_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("tree compact k dtod: {e:?}"))?;
                cudarc::driver::result::memcpy_dtod_async(
                    v_ptr + (dst_slot * row_kv_bytes) as u64,
                    v_ptr + (src_slot * row_kv_bytes) as u64,
                    accepted_rows * row_kv_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("tree compact v dtod: {e:?}"))?;
                cudarc::driver::result::memcpy_dtod_async(
                    ks_ptr + (dst_slot * row_scale_bytes) as u64,
                    ks_ptr + (src_slot * row_scale_bytes) as u64,
                    accepted_rows * row_scale_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("tree compact k-scale dtod: {e:?}"))?;
                cudarc::driver::result::memcpy_dtod_async(
                    vs_ptr + (dst_slot * row_scale_bytes) as u64,
                    vs_ptr + (src_slot * row_scale_bytes) as u64,
                    accepted_rows * row_scale_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("tree compact v-scale dtod: {e:?}"))?;
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_fused_slot(&mut self, slot: usize, la: &LinearAttention) -> Result<()> {
        anyhow::ensure!(
            slot < self.lin_attn_states.len(),
            "ensure_fused_slot: slot {slot} out of range"
        );
        if self.lin_attn_states[slot]
            .as_ref()
            .map(|s| s.is_fused())
            .unwrap_or(false)
        {
            return Ok(());
        }
        anyhow::ensure!(
            la.fused_decode_supported(),
            "ensure_fused_slot: fused decode unsupported for slot {slot}"
        );
        let fresh = la.new_fused_state(&self.device)?;
        if let Some(prev) = &self.lin_attn_states[slot] {
            fresh.copy_data_from(prev)?;
        }
        self.lin_attn_states[slot] = Some(fresh);
        Ok(())
    }

    pub fn set_pending_pos_host_only(&mut self, write_start: usize, new_total: usize) {
        self.host_write_pos[0] = write_start as i32;
        self.host_n_total[0] = new_total as i32;
    }

    pub fn set_current_len(&mut self, len: usize) {
        self.current_len = len;
    }

    fn prepare_for_step(&mut self, write_start: usize, new_total: usize) -> Result<()> {
        if new_total > self.max_seq_len {
            anyhow::bail!(
                "Qwen3MoeKvCache: new_total {} > max_seq_len {}",
                new_total,
                self.max_seq_len
            );
        }
        self.set_pending_pos_host_only(write_start, new_total);
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
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

    pub fn write_at(
        &mut self,
        slot: usize,
        _start: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        if slot >= self.layers.len() {
            anyhow::bail!("Qwen3MoeKvCache.write_at: slot {} out of range", slot);
        }
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != self.n_kv_heads || dims[3] != self.head_dim
        {
            anyhow::bail!(
                "Qwen3MoeKvCache.write_at: expected [1, t, {}, {}], got {:?}",
                self.n_kv_heads,
                self.head_dim,
                dims
            );
        }
        if v_new.dims() != dims {
            anyhow::bail!(
                "Qwen3MoeKvCache.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let (start_dev_ptr, _gsp) = self.write_pos_dev.device_ptr(&stream);
        let slot_mut = &mut self.layers[slot];

        let (k_storage, _kl) = k_new.storage_and_layout();
        let (v_storage, _vl) = v_new.storage_and_layout();
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("k_new must be on CUDA"),
        };
        let v_cuda = match &*v_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("v_new must be on CUDA"),
        };
        let k_slice = k_cuda.as_cuda_slice::<bf16>()?;
        let v_slice = v_cuda.as_cuda_slice::<bf16>()?;

        let (k_in_ptr, _gki) = k_slice.device_ptr(&stream);
        let (v_in_ptr, _gvi) = v_slice.device_ptr(&stream);
        let (k_fp8_base, _gkf) = slot_mut.k_fp8.device_ptr_mut(&stream);
        let (v_fp8_base, _gvf) = slot_mut.v_fp8.device_ptr_mut(&stream);
        let (k_sc_base, _gks) = slot_mut.k_scales.device_ptr_mut(&stream);
        let (v_sc_base, _gvs) = slot_mut.v_scales.device_ptr_mut(&stream);

        let s_raw = stream.cu_stream() as *mut c_void;
        let rc_k = unsafe {
            nv_kernels::cuda::quantize_kv_fp8(
                s_raw,
                k_in_ptr as *const u16,
                k_fp8_base as *mut u8,
                k_sc_base as *mut f32,
                start_dev_ptr as *const i32,
                t as i32,
                self.n_kv_heads as i32,
                self.head_dim as i32,
                0,
            )
        };
        if rc_k != 0 {
            anyhow::bail!("quantize_kv_fp8(k) rc={rc_k}");
        }
        let rc_v = unsafe {
            nv_kernels::cuda::quantize_kv_fp8(
                s_raw,
                v_in_ptr as *const u16,
                v_fp8_base as *mut u8,
                v_sc_base as *mut f32,
                start_dev_ptr as *const i32,
                t as i32,
                self.n_kv_heads as i32,
                self.head_dim as i32,
                0,
            )
        };
        if rc_v != 0 {
            anyhow::bail!("quantize_kv_fp8(v) rc={rc_v}");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_norm_rope_store_decode_rope_pos_reads_write_pos_dev_because_decode_positions_equal_write_start(
        &mut self,
        slot: usize,
        q_raw: &Tensor,
        k_raw: &Tensor,
        v_raw: &Tensor,
        q_norm_w_bf16: &Tensor,
        k_norm_w_bf16: &Tensor,
        rope: &Rope,
        n_q: usize,
        rotary_dim: usize,
        eps: f32,
        gated: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        anyhow::ensure!(
            slot < self.layers.len(),
            "fused_qkv_norm_rope_store_decode: slot {slot} out of range"
        );
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let q_stride = if gated { 2 * hd } else { hd };
        anyhow::ensure!(
            q_raw.elem_count() == n_q * q_stride
                && k_raw.elem_count() == n_kv * hd
                && v_raw.elem_count() == n_kv * hd,
            "fused_qkv_norm_rope_store_decode: q/k/v elem counts {} {} {} do not match \
             n_q={n_q} n_kv={n_kv} hd={hd} gated={gated}",
            q_raw.elem_count(),
            k_raw.elem_count(),
            v_raw.elem_count()
        );
        anyhow::ensure!(
            q_raw.dtype() == DType::BF16
                && k_raw.dtype() == DType::BF16
                && v_raw.dtype() == DType::BF16
                && q_norm_w_bf16.dtype() == DType::BF16
                && k_norm_w_bf16.dtype() == DType::BF16
                && rope.cos().dtype() == DType::F32
                && rope.sin().dtype() == DType::F32,
            "fused_qkv_norm_rope_store_decode: dtype contract broken; the caller gate must \
             refuse before projecting"
        );
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let mut q_out = unsafe {
            stream
                .alloc::<bf16>(n_q * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut q_sig = if gated {
            Some(unsafe {
                stream
                    .alloc::<bf16>(n_q * hd)
                    .map_err(|e| anyhow::anyhow!(e))?
            })
        } else {
            None
        };

        let q_c = q_raw.contiguous()?;
        let k_c = k_raw.contiguous()?;
        let v_c = v_raw.contiguous()?;
        let qw_c = q_norm_w_bf16.contiguous()?;
        let kw_c = k_norm_w_bf16.contiguous()?;
        let cos_c = rope.cos().contiguous()?;
        let sin_c = rope.sin().contiguous()?;

        let rc = {
            let get = |t: &Tensor| -> Result<u64> {
                let (storage, layout) = t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("fused_qkv_norm_rope_store_decode: tensor not on CUDA"),
                };
                let ptr = match t.dtype() {
                    DType::BF16 => {
                        let sl = cuda.as_cuda_slice::<bf16>()?;
                        let view = sl.slice(layout.start_offset()..);
                        let (p, _g) = view.device_ptr(&stream);
                        p
                    }
                    DType::F32 => {
                        let sl = cuda.as_cuda_slice::<f32>()?;
                        let view = sl.slice(layout.start_offset()..);
                        let (p, _g) = view.device_ptr(&stream);
                        p
                    }
                    other => anyhow::bail!(
                        "fused_qkv_norm_rope_store_decode: unsupported dtype {other:?}"
                    ),
                };
                Ok(ptr)
            };
            let q_ptr = get(&q_c)?;
            let k_ptr = get(&k_c)?;
            let v_ptr = get(&v_c)?;
            let qw_ptr = get(&qw_c)?;
            let kw_ptr = get(&kw_c)?;
            let cos_ptr = get(&cos_c)?;
            let sin_ptr = get(&sin_c)?;
            let (pos_ptr, _gp) = self.write_pos_dev.device_ptr(&stream);
            let slot_mut = &mut self.layers[slot];
            let (k_fp8_ptr, _gkf) = slot_mut.k_fp8.device_ptr_mut(&stream);
            let (v_fp8_ptr, _gvf) = slot_mut.v_fp8.device_ptr_mut(&stream);
            let (k_sc_ptr, _gks) = slot_mut.k_scales.device_ptr_mut(&stream);
            let (v_sc_ptr, _gvs) = slot_mut.v_scales.device_ptr_mut(&stream);
            let (q_out_ptr, _gqo) = q_out.device_ptr_mut(&stream);
            let q_sig_ptr = match &mut q_sig {
                Some(s) => {
                    let (p, _g) = s.device_ptr_mut(&stream);
                    p
                }
                None => 0u64,
            };
            unsafe {
                nv_kernels::cuda::qkv_norm_rope_kvstore_fp8_decode(
                    stream.cu_stream() as *mut c_void,
                    q_ptr as *const u16,
                    k_ptr as *const u16,
                    v_ptr as *const u16,
                    qw_ptr as *const u16,
                    kw_ptr as *const u16,
                    cos_ptr as *const f32,
                    sin_ptr as *const f32,
                    pos_ptr as *const i32,
                    k_fp8_ptr as *mut u8,
                    v_fp8_ptr as *mut u8,
                    k_sc_ptr as *mut f32,
                    v_sc_ptr as *mut f32,
                    q_out_ptr as *mut u16,
                    q_sig_ptr as *mut u16,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    q_stride as i32,
                    rotary_dim as i32,
                    eps,
                )
            }
        };
        anyhow::ensure!(
            rc == 0,
            "qkv_norm_rope_kvstore_fp8_decode rc={rc}; the caller gate checked geometry so a \
             refusal here is a contract bug, not a fallback"
        );
        let q_tensor = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, n_q, hd),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let sig_tensor = q_sig.map(|s| {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(s, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, n_q * hd),
                candle_core::op::BackpropOp::none(),
                false,
            )
        });
        Ok((q_tensor, sig_tensor))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_norm_rope_store_verify_rows_positions_read_from_the_lane_pos_buf_the_graph_updates(
        &mut self,
        slot: usize,
        rows: usize,
        q_raw: &Tensor,
        k_raw: &Tensor,
        v_raw: &Tensor,
        positions: &Tensor,
        q_norm_w_bf16: &Tensor,
        k_norm_w_bf16: &Tensor,
        rope: &Rope,
        n_q: usize,
        rotary_dim: usize,
        eps: f32,
        gated: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        anyhow::ensure!(
            slot < self.layers.len(),
            "fused_qkv_norm_rope_store_verify_rows: slot {slot} out of range"
        );
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let q_stride = if gated { 2 * hd } else { hd };
        anyhow::ensure!(
            q_raw.elem_count() == rows * n_q * q_stride
                && k_raw.elem_count() == rows * n_kv * hd
                && v_raw.elem_count() == rows * n_kv * hd
                && positions.elem_count() == rows,
            "fused_qkv_norm_rope_store_verify_rows: q/k/v/pos elem counts {} {} {} {} do not \
             match rows={rows} n_q={n_q} n_kv={n_kv} hd={hd} gated={gated}",
            q_raw.elem_count(),
            k_raw.elem_count(),
            v_raw.elem_count(),
            positions.elem_count()
        );
        anyhow::ensure!(
            q_raw.dtype() == DType::BF16
                && k_raw.dtype() == DType::BF16
                && v_raw.dtype() == DType::BF16
                && positions.dtype() == DType::I32
                && q_norm_w_bf16.dtype() == DType::BF16
                && k_norm_w_bf16.dtype() == DType::BF16
                && rope.cos().dtype() == DType::F32
                && rope.sin().dtype() == DType::F32,
            "fused_qkv_norm_rope_store_verify_rows: dtype contract broken; the caller gate must \
             refuse before projecting"
        );
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let mut q_out = unsafe {
            stream
                .alloc::<bf16>(rows * n_q * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut q_sig = if gated {
            Some(unsafe {
                stream
                    .alloc::<bf16>(rows * n_q * hd)
                    .map_err(|e| anyhow::anyhow!(e))?
            })
        } else {
            None
        };

        let q_c = q_raw.contiguous()?;
        let k_c = k_raw.contiguous()?;
        let v_c = v_raw.contiguous()?;
        let p_c = positions.contiguous()?;
        let qw_c = q_norm_w_bf16.contiguous()?;
        let kw_c = k_norm_w_bf16.contiguous()?;
        let cos_c = rope.cos().contiguous()?;
        let sin_c = rope.sin().contiguous()?;

        let rc = {
            let get_bf16 = |t: &Tensor| -> Result<u64> {
                let (storage, layout) = t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("fused_qkv_norm_rope_store_verify_rows: tensor not on CUDA"),
                };
                let ptr = match t.dtype() {
                    DType::BF16 => {
                        let sl = cuda.as_cuda_slice::<bf16>()?;
                        let view = sl.slice(layout.start_offset()..);
                        let (p, _g) = view.device_ptr(&stream);
                        p
                    }
                    DType::F32 => {
                        let sl = cuda.as_cuda_slice::<f32>()?;
                        let view = sl.slice(layout.start_offset()..);
                        let (p, _g) = view.device_ptr(&stream);
                        p
                    }
                    DType::I32 => {
                        let sl = cuda.as_cuda_slice::<i32>()?;
                        let view = sl.slice(layout.start_offset()..);
                        let (p, _g) = view.device_ptr(&stream);
                        p
                    }
                    other => anyhow::bail!(
                        "fused_qkv_norm_rope_store_verify_rows: unsupported dtype {other:?}"
                    ),
                };
                Ok(ptr)
            };
            let q_ptr = get_bf16(&q_c)?;
            let k_ptr = get_bf16(&k_c)?;
            let v_ptr = get_bf16(&v_c)?;
            let p_ptr = get_bf16(&p_c)?;
            let qw_ptr = get_bf16(&qw_c)?;
            let kw_ptr = get_bf16(&kw_c)?;
            let cos_ptr = get_bf16(&cos_c)?;
            let sin_ptr = get_bf16(&sin_c)?;
            let slot_mut = &mut self.layers[slot];
            let (k_fp8_ptr, _gkf) = slot_mut.k_fp8.device_ptr_mut(&stream);
            let (v_fp8_ptr, _gvf) = slot_mut.v_fp8.device_ptr_mut(&stream);
            let (k_sc_ptr, _gks) = slot_mut.k_scales.device_ptr_mut(&stream);
            let (v_sc_ptr, _gvs) = slot_mut.v_scales.device_ptr_mut(&stream);
            let (q_out_ptr, _gqo) = q_out.device_ptr_mut(&stream);
            let q_sig_ptr = match &mut q_sig {
                Some(s) => {
                    let (p, _g) = s.device_ptr_mut(&stream);
                    p
                }
                None => 0u64,
            };
            unsafe {
                nv_kernels::cuda::qkv_norm_rope_kvstore_fp8_rows(
                    stream.cu_stream() as *mut c_void,
                    q_ptr as *const u16,
                    k_ptr as *const u16,
                    v_ptr as *const u16,
                    qw_ptr as *const u16,
                    kw_ptr as *const u16,
                    cos_ptr as *const f32,
                    sin_ptr as *const f32,
                    p_ptr as *const i32,
                    k_fp8_ptr as *mut u8,
                    v_fp8_ptr as *mut u8,
                    k_sc_ptr as *mut f32,
                    v_sc_ptr as *mut f32,
                    q_out_ptr as *mut u16,
                    q_sig_ptr as *mut u16,
                    rows as i32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    q_stride as i32,
                    rotary_dim as i32,
                    eps,
                )
            }
        };
        anyhow::ensure!(
            rc == 0,
            "qkv_norm_rope_kvstore_fp8_rows rc={rc}; the caller gate checked geometry so a \
             refusal here is a contract bug, not a fallback"
        );
        let q_tensor = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, rows, n_q, hd),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let sig_tensor = q_sig.map(|s| {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(s, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, rows, n_q * hd),
                candle_core::op::BackpropOp::none(),
                false,
            )
        });
        Ok((q_tensor, sig_tensor))
    }

    pub fn view(&mut self, slot: usize, len: usize) -> Result<(Tensor, Tensor)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        if slot >= self.layers.len() {
            anyhow::bail!("Qwen3MoeKvCache.view: slot {} out of range", slot);
        }
        if len > self.max_seq_len {
            anyhow::bail!(
                "Qwen3MoeKvCache.view: len {} > max {}",
                len,
                self.max_seq_len
            );
        }
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let need = len * n_kv * hd;
        let mut k_out = unsafe { stream.alloc::<bf16>(need).map_err(|e| anyhow::anyhow!(e))? };
        let mut v_out = unsafe { stream.alloc::<bf16>(need).map_err(|e| anyhow::anyhow!(e))? };
        let slot_ref = &self.layers[slot];
        {
            let (k_fp8_ptr, _gk) = slot_ref.k_fp8.device_ptr(&stream);
            let (v_fp8_ptr, _gv) = slot_ref.v_fp8.device_ptr(&stream);
            let (k_sc_ptr, _gks) = slot_ref.k_scales.device_ptr(&stream);
            let (v_sc_ptr, _gvs) = slot_ref.v_scales.device_ptr(&stream);
            let (k_out_ptr, _gko) = k_out.device_ptr_mut(&stream);
            let (v_out_ptr, _gvo) = v_out.device_ptr_mut(&stream);
            let s_raw = stream.cu_stream() as *mut c_void;
            let rc_k = unsafe {
                nv_kernels::cuda::dequantize_kv_fp8(
                    s_raw,
                    k_fp8_ptr as *const u8,
                    k_sc_ptr as *const f32,
                    k_out_ptr as *mut u16,
                    0,
                    len as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                )
            };
            if rc_k != 0 {
                anyhow::bail!("dequantize_kv_fp8(k) rc={rc_k}");
            }
            let rc_v = unsafe {
                nv_kernels::cuda::dequantize_kv_fp8(
                    s_raw,
                    v_fp8_ptr as *const u8,
                    v_sc_ptr as *const f32,
                    v_out_ptr as *mut u16,
                    0,
                    len as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                )
            };
            if rc_v != 0 {
                anyhow::bail!("dequantize_kv_fp8(v) rc={rc_v}");
            }
        }
        let k_storage = candle_core::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
        let v_storage = candle_core::CudaStorage::wrap_cuda_slice(v_out, dev);
        let k = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(k_storage),
            (1usize, len, n_kv, hd),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let v = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(v_storage),
            (1usize, len, n_kv, hd),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok((k, v))
    }

    pub fn decode_attention_fp8(
        &mut self,
        slot: usize,
        q_rot: &Tensor,
        n_q: usize,
        scaling: f32,
    ) -> Result<Tensor> {
        self.decode_attention_fp8_whose_split_pick_sees_the_lane_group_width(
            slot,
            q_rot,
            n_q,
            scaling,
            SOLO_AND_EVERY_FAMILY_OUTSIDE_THE_QWEN38_LANE_GROUP_ENTERS_AT_ONE_LANE_IN_FLIGHT_SO_THE_SPLIT_PICK_IS_THE_ONE_THE_RECORD_WAS_MEASURED_ON,
        )
    }

    pub fn decode_attention_fp8_whose_split_pick_sees_the_lane_group_width(
        &mut self,
        slot: usize,
        q_rot: &Tensor,
        n_q: usize,
        scaling: f32,
        lanes_in_flight: usize,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let expected = n_q * hd;
        let total: usize = q_rot.dims().iter().product();
        if total != expected {
            anyhow::bail!(
                "decode_attention_fp8: expected total {expected}, got dims {:?}",
                q_rot.dims()
            );
        }
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let mut out = unsafe {
            stream
                .alloc::<bf16>(expected)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let q_c = q_rot.contiguous()?;
        let (q_storage, _ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("q_rot must be on CUDA"),
        };
        let q_slice = q_cuda.as_cuda_slice::<bf16>()?;

        let (n_total_ptr, _gnt) = self.n_total_dev.device_ptr(&stream);
        let slot_ref = &self.layers[slot];

        let (q_ptr, _gq) = q_slice.device_ptr(&stream);
        let (k_ptr, _gk) = slot_ref.k_fp8.device_ptr(&stream);
        let (v_ptr, _gv) = slot_ref.v_fp8.device_ptr(&stream);
        let (ks_ptr, _gks) = slot_ref.k_scales.device_ptr(&stream);
        let (vs_ptr, _gvs) = slot_ref.v_scales.device_ptr(&stream);
        let (out_ptr, _go) = out.device_ptr_mut(&stream);

        let max_total = self.max_seq_len as i32;
        let s_raw = stream.cu_stream() as *mut c_void;
        let kvshare_m1_engaged = lanes_in_flight
            == SOLO_AND_EVERY_FAMILY_OUTSIDE_THE_QWEN38_LANE_GROUP_ENTERS_AT_ONE_LANE_IN_FLIGHT_SO_THE_SPLIT_PICK_IS_THE_ONE_THE_RECORD_WAS_MEASURED_ON
            && hd == 256
            && n_q % n_kv == 0
            && matches!(n_q / n_kv, 6 | 8)
            && nv_q38_decode_kvshare_env_opt_in_group_fold_reads_kv_once_per_kv_head_not_once_per_q_head()
                .is_some_and(|min_total| self.host_n_total[0] as usize >= min_total);
        if let Some((splitk_scratch, splitk_fan_in)) = &mut self.splitk_scratch_and_fan_in_gated_by_nv_q36_graphed_decode_fix_because_the_smem_kernel_serializes_a_block_reduce_per_kv_position_costing_82ms_at_8k {
            let kvshare_scratch_floor_elems =
                n_q * M1_DECODE_KVSHARE_SCRATCH_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS * (hd + 2);
            anyhow::ensure!(
                !kvshare_m1_engaged || splitk_scratch.len() >= kvshare_scratch_floor_elems,
                "{M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM} \
                 engaged over a splitk scratch of {} f32 elems, below the {kvshare_scratch_floor_elems} \
                 the kvshare 128-split grid can address: the env was set after cache construction, \
                 and the constructor sizes the scratch for the kvshare cap only when the env is \
                 present there",
                splitk_scratch.len()
            );
            let (sc_ptr, _gscp) = splitk_scratch.device_ptr_mut(&stream);
            let (fi_ptr, _gfip) = splitk_fan_in.device_ptr_mut(&stream);
            let rc = if kvshare_m1_engaged {
                unsafe {
                    nv_kernels::cuda::flash_decode_kvshare_fp8kv_paged(
                        s_raw,
                        q_ptr as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        out_ptr as *mut u16,
                        n_total_ptr as *const i32,
                        sc_ptr as *mut f32,
                        fi_ptr as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                        0,
                        0,
                        scaling,
                        std::ptr::null(),
                        0,
                        0,
                        0,
                        0,
                    )
                }
            } else if lanes_in_flight
                > SOLO_AND_EVERY_FAMILY_OUTSIDE_THE_QWEN38_LANE_GROUP_ENTERS_AT_ONE_LANE_IN_FLIGHT_SO_THE_SPLIT_PICK_IS_THE_ONE_THE_RECORD_WAS_MEASURED_ON
            {
                unsafe {
                    nv_kernels::cuda::flash_decode_fused_fp8kv_lane_group(
                        s_raw,
                        q_ptr as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        out_ptr as *mut u16,
                        n_total_ptr as *const i32,
                        sc_ptr as *mut f32,
                        fi_ptr as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                        0,
                        scaling,
                        lanes_in_flight as i32,
                    )
                }
            } else {
                unsafe {
                    nv_kernels::cuda::flash_decode_fused_fp8kv(
                        s_raw,
                        q_ptr as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        out_ptr as *mut u16,
                        n_total_ptr as *const i32,
                        sc_ptr as *mut f32,
                        fi_ptr as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                        0,
                        scaling,
                    )
                }
            };
            anyhow::ensure!(
                rc == 0,
                "m1 decode attention rc={rc} kvshare_m1_engaged={kvshare_m1_engaged}"
            );
            drop(_gscp);
            drop(_gfip);
            drop(_go);
            drop(_gq);
            drop(_gk);
            drop(_gv);
            drop(_gks);
            drop(_gvs);
            drop(_gnt);
            drop(q_storage);
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
            let tensor = candle_core::Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, n_q, hd),
                candle_core::op::BackpropOp::none(),
                false,
            );
            return Ok(tensor);
        }
        let rc = match &self.scores_scratch_because_smem_decode_caps_at_12k_positions_at_48kb {
            None => unsafe {
                nv_kernels::cuda::attention_fp8_decode(
                    s_raw,
                    q_ptr as *const u16,
                    k_ptr as *const u8,
                    v_ptr as *const u8,
                    ks_ptr as *const f32,
                    vs_ptr as *const f32,
                    out_ptr as *mut u16,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    n_total_ptr as *const i32,
                    max_total,
                    0,
                    scaling,
                )
            },
            Some(scratch) => {
                anyhow::ensure!(
                    n_q <= self.scratch_heads,
                    "Qwen3MoeKvCache.decode: n_q {n_q} exceeds scores scratch heads {}",
                    self.scratch_heads
                );
                let (sc_ptr, _gsc) = scratch.device_ptr(&stream);
                unsafe {
                    nv_kernels::cuda::attention_fp8_decode_gscores(
                        s_raw,
                        q_ptr as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        out_ptr as *mut u16,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        n_total_ptr as *const i32,
                        max_total,
                        0,
                        scaling,
                        sc_ptr as *mut f32,
                    )
                }
            }
        };
        if rc != 0 {
            anyhow::bail!("attention_fp8_decode rc={rc}");
        }

        drop(_go);
        drop(_gq);
        drop(_gk);
        drop(_gv);
        drop(_gks);
        drop(_gvs);
        drop(_gnt);
        drop(q_storage);

        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, 1usize, n_q, hd),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(tensor)
    }

    pub fn verify_attention_fp8_mk(
        &mut self,
        slot: usize,
        q_rot: &Tensor,
        n_q: usize,
        m: usize,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        if m < 2
            || m > MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
            || self.head_dim > MK_VERIFY_MAX_HEAD_DIM_512_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
            || self
                .mk_verify_scratch_and_fan_in_routing_2_to_8_row_chains_off_the_full_kv_dequant_view_kill_switch_nv_q38_mk_verify_0
                .is_none()
        {
            anyhow::ensure!(
                self.verify_tree_rows_active != Some(m),
                "verify tree: the mk verify route refused rows={m} and every fallback attention \
                 masks by row order, which is lossy for sibling rows"
            );
            return Ok(None);
        }
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let expected = m * n_q * hd;
        let total: usize = q_rot.dims().iter().product();
        anyhow::ensure!(
            total == expected,
            "verify_attention_fp8_mk: expected total {expected} for m={m}, got dims {:?}",
            q_rot.dims()
        );
        let dev = match self.device.clone() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut out = unsafe {
            stream
                .alloc::<bf16>(expected)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let q_c = q_rot.contiguous()?;
        let (q_storage, _ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("q_rot must be on CUDA"),
        };
        let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
        let (n_total_ptr, _gnt) = self.n_total_dev.device_ptr(&stream);
        let slot_ref = &self.layers[slot];
        let (q_ptr, _gq) = q_slice.device_ptr(&stream);
        let (k_ptr, _gk) = slot_ref.k_fp8.device_ptr(&stream);
        let (v_ptr, _gv) = slot_ref.v_fp8.device_ptr(&stream);
        let (ks_ptr, _gks) = slot_ref.k_scales.device_ptr(&stream);
        let (vs_ptr, _gvs) = slot_ref.v_scales.device_ptr(&stream);
        let (out_ptr, _go) = out.device_ptr_mut(&stream);
        let (scratch, fan_in) = self
            .mk_verify_scratch_and_fan_in_routing_2_to_8_row_chains_off_the_full_kv_dequant_view_kill_switch_nv_q38_mk_verify_0
            .as_mut()
            .unwrap();
        let (sc_ptr, _gsc) = scratch.device_ptr_mut(&stream);
        let (fi_ptr, _gfi) = fan_in.device_ptr_mut(&stream);
        let delta_0_because_this_caches_n_total_dev_already_counts_the_m_appended_rows_and_the_kernel_reads_total_as_n_total_minus_delta =
            0i32;
        let tree_active = self.verify_tree_rows_active == Some(m);
        anyhow::ensure!(
            !tree_active || m == VERIFY_TREE_TWO_BRANCH_M6_ROWS,
            "verify tree: active rows {m} != the two-branch shape"
        );
        if let Ok(dump_dir) = std::env::var(MK_VERIFY_DUMP_DIR_ENV_WRITES_EACH_SLOTS_QKV_ONCE_SO_A_PARITY_GATE_CAN_REPLAY_LIVE_SERVING_TENSORS_THROUGH_BOTH_ARMS) {
            let marker = std::path::Path::new(&dump_dir).join(format!("slot{slot}.done"));
            if !marker.exists() {
                stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
                let hq = stream.clone_dtoh(q_slice).map_err(|e| anyhow::anyhow!(e))?;
                let hk = stream
                    .clone_dtoh(&self.layers[slot].k_fp8)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let hv = stream
                    .clone_dtoh(&self.layers[slot].v_fp8)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let hks = stream
                    .clone_dtoh(&self.layers[slot].k_scales)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let hvs = stream
                    .clone_dtoh(&self.layers[slot].v_scales)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let d = std::path::Path::new(&dump_dir);
                let total = self.host_n_total[0] as usize;
                let qb: Vec<u8> = hq.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
                std::fs::write(d.join(format!("slot{slot}.q.bin")), qb)?;
                std::fs::write(
                    d.join(format!("slot{slot}.k.bin")),
                    &hk[..total * n_kv * hd],
                )?;
                std::fs::write(
                    d.join(format!("slot{slot}.v.bin")),
                    &hv[..total * n_kv * hd],
                )?;
                let ksb: Vec<u8> = hks[..total * n_kv]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                std::fs::write(d.join(format!("slot{slot}.ks.bin")), ksb)?;
                let vsb: Vec<u8> = hvs[..total * n_kv]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                std::fs::write(d.join(format!("slot{slot}.vs.bin")), vsb)?;
                std::fs::write(
                    d.join(format!("slot{slot}.meta")),
                    format!("total={total} m={m} n_q={n_q} n_kv={n_kv} hd={hd} scaling={scaling}"),
                )?;
                std::fs::write(&marker, b"1")?;
            }
        }
        let kvshare_geometry = hd == 256 && n_q % n_kv == 0 && matches!(n_q / n_kv, 6 | 8);
        let tctile_geometry = hd == 256 && n_q % n_kv == 0 && n_q / n_kv == 6;
        let tctile_one_launch =
            nv_q38_mk_verify_tctile_env_opt_in_one_launch_mma_qk_tiles_read_kv_once_for_all_m_by_group_rows()
                .is_some_and(|min_total| self.host_n_total[0] as usize >= min_total)
                && tctile_geometry;
        let rowshared_one_launch = !tctile_one_launch
            && nv_q38_mk_verify_rowshared_env_opt_in_one_launch_reads_kv_once_for_all_m_by_group_rows()
                .is_some_and(|min_total| self.host_n_total[0] as usize >= min_total)
            && kvshare_geometry;
        let kvshare_row_sequential = !tctile_one_launch
            && !rowshared_one_launch
            && nv_q38_mk_verify_kvshare_env_opt_in_row_sequential_group_fold_reads_kv_once_per_row_not_once_per_q_head()
                .is_some_and(|min_total| self.host_n_total[0] as usize >= min_total)
            && kvshare_geometry;
        let rc = if tctile_one_launch || rowshared_one_launch {
            let (hole_from_row, hole_lo_delta, hole_hi_delta) = if tree_active {
                let (lo, hi) =
                    VERIFY_TREE_TWO_BRANCH_M6_HOLE_DELTAS_FROM_N_TOTAL_SIBLING_ROWS_HIDE_THE_BRANCH_A_BLOCK
                        [VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW];
                (VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW as i32, lo, hi)
            } else {
                (m as i32, 0, 0)
            };
            let splits_cap_scratch_rows_over_m = (MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
                * MK_VERIFY_SPLITS_CEILING_32_MATCHES_FLASH_SPLITS_PICK
                / m) as i32;
            let one_launch_mrows = if tctile_one_launch {
                nv_kernels::cuda::flash_decode_rowshared_tctile_fp8kv_mk_paged
            } else {
                nv_kernels::cuda::flash_decode_rowshared_fp8kv_mk_paged
            };
            unsafe {
                one_launch_mrows(
                    stream.cu_stream() as *mut c_void,
                    q_ptr as *const u16,
                    k_ptr as *const u8,
                    v_ptr as *const u8,
                    ks_ptr as *const f32,
                    vs_ptr as *const f32,
                    out_ptr as *mut u16,
                    n_total_ptr as *const i32,
                    sc_ptr as *mut f32,
                    fi_ptr as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                    0,
                    0,
                    splits_cap_scratch_rows_over_m,
                    scaling,
                    std::ptr::null(),
                    0,
                    delta_0_because_this_caches_n_total_dev_already_counts_the_m_appended_rows_and_the_kernel_reads_total_as_n_total_minus_delta,
                    m as i32,
                    hole_from_row,
                    hole_lo_delta,
                    hole_hi_delta,
                )
            }
        } else if kvshare_row_sequential {
            let row_stride_bytes = (n_q * hd * std::mem::size_of::<bf16>()) as u64;
            let mut rc = 0i32;
            for qi in 0..m {
                let (hole_lo_delta, hole_hi_delta) = if tree_active {
                    VERIFY_TREE_TWO_BRANCH_M6_HOLE_DELTAS_FROM_N_TOTAL_SIBLING_ROWS_HIDE_THE_BRANCH_A_BLOCK
                        [qi]
                } else {
                    (0, 0)
                };
                rc = unsafe {
                    nv_kernels::cuda::flash_decode_kvshare_fp8kv_paged(
                        stream.cu_stream() as *mut c_void,
                        (q_ptr + qi as u64 * row_stride_bytes) as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        (out_ptr + qi as u64 * row_stride_bytes) as *mut u16,
                        n_total_ptr as *const i32,
                        sc_ptr as *mut f32,
                        fi_ptr as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                        0,
                        0,
                        scaling,
                        std::ptr::null(),
                        0,
                        (m - 1 - qi) as i32,
                        hole_lo_delta,
                        hole_hi_delta,
                    )
                };
                if rc != 0 {
                    break;
                }
            }
            rc
        } else if tree_active {
            unsafe {
                nv_kernels::cuda::flash_decode_fused_fp8kv_mk_hole(
                    stream.cu_stream() as *mut c_void,
                    q_ptr as *const u16,
                    k_ptr as *const u8,
                    v_ptr as *const u8,
                    ks_ptr as *const f32,
                    vs_ptr as *const f32,
                    out_ptr as *mut u16,
                    n_total_ptr as *const i32,
                    delta_0_because_this_caches_n_total_dev_already_counts_the_m_appended_rows_and_the_kernel_reads_total_as_n_total_minus_delta,
                    m as i32,
                    sc_ptr as *mut f32,
                    fi_ptr as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                    0,
                    scaling,
                    VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW as i32,
                    VERIFY_TREE_TWO_BRANCH_M6_HOLE_DELTAS_FROM_N_TOTAL_SIBLING_ROWS_HIDE_THE_BRANCH_A_BLOCK
                        [VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW]
                        .0,
                    VERIFY_TREE_TWO_BRANCH_M6_HOLE_DELTAS_FROM_N_TOTAL_SIBLING_ROWS_HIDE_THE_BRANCH_A_BLOCK
                        [VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW]
                        .1,
                )
            }
        } else {
            unsafe {
                nv_kernels::cuda::flash_decode_fused_fp8kv_mk(
                    stream.cu_stream() as *mut c_void,
                    q_ptr as *const u16,
                    k_ptr as *const u8,
                    v_ptr as *const u8,
                    ks_ptr as *const f32,
                    vs_ptr as *const f32,
                    out_ptr as *mut u16,
                    n_total_ptr as *const i32,
                    delta_0_because_this_caches_n_total_dev_already_counts_the_m_appended_rows_and_the_kernel_reads_total_as_n_total_minus_delta,
                    m as i32,
                    sc_ptr as *mut f32,
                    fi_ptr as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                    0,
                    scaling,
                )
            }
        };
        anyhow::ensure!(
            rc == 0,
            "mk verify attention rc={rc} tctile_one_launch={tctile_one_launch} \
             rowshared_one_launch={rowshared_one_launch} \
             kvshare_row_sequential={kvshare_row_sequential}"
        );
        drop(_gsc);
        drop(_gfi);
        drop(_go);
        drop(_gq);
        drop(_gk);
        drop(_gv);
        drop(_gks);
        drop(_gvs);
        drop(_gnt);
        drop(q_storage);
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, m, n_q, hd),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(tensor))
    }

    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }

    pub fn write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
        &mut self,
        write_start: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        let dims = k_new.dims();
        anyhow::ensure!(
            dims.len() == 4 && dims[1] > 0,
            "synthetic kv fill expects [1, t, n_kv, head_dim] with t > 0, got {dims:?}"
        );
        let t = dims[1];
        self.prepare_for_step(write_start, write_start + t)?;
        for slot in 0..self.layers.len() {
            self.write_at(slot, write_start, k_new, v_new)?;
        }
        self.advance(t);
        Ok(())
    }

    pub fn dump_primed_state_for_reuse(
        &self,
        fingerprint: &str,
        out: &mut dyn std::io::Write,
    ) -> Result<()> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync before prime dump: {e:?}"))?;
        out.write_all(PRIME_CKPT_MAGIC_NVPRIMEK)?;
        out.write_all(
            &PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES
                .to_le_bytes(),
        )?;
        let fp = fingerprint.as_bytes();
        out.write_all(&(fp.len() as u64).to_le_bytes())?;
        out.write_all(fp)?;
        for v in [
            self.current_len as u64,
            self.layers.len() as u64,
            self.n_kv_heads as u64,
            self.head_dim as u64,
            self.lin_attn_states.len() as u64,
        ] {
            out.write_all(&v.to_le_bytes())?;
        }
        for st in &self.lin_attn_states {
            match st {
                Some(s) => {
                    out.write_all(&[1u8])?;
                    s.dump_primed_state_for_reuse(out)?;
                }
                None => out.write_all(&[0u8])?,
            }
        }
        let kv_elems = self.current_len * self.n_kv_heads * self.head_dim;
        let sc_elems = self.current_len * self.n_kv_heads;
        for slot in &self.layers {
            for fp8 in [&slot.k_fp8, &slot.v_fp8] {
                let rows: Vec<u8> = stream
                    .clone_dtoh(&fp8.slice(0..kv_elems))
                    .map_err(|e| anyhow::anyhow!("prime dump fp8 dtoh: {e:?}"))?;
                out.write_all(&rows)?;
            }
            for scales in [&slot.k_scales, &slot.v_scales] {
                let s: Vec<f32> = stream
                    .clone_dtoh(&scales.slice(0..sc_elems))
                    .map_err(|e| anyhow::anyhow!("prime dump scales dtoh: {e:?}"))?;
                let mut b = Vec::with_capacity(s.len() * 4);
                for x in s {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                out.write_all(&b)?;
            }
        }
        Ok(())
    }

    pub fn restore_primed_state_checked(
        &mut self,
        fingerprint: &str,
        input: &mut dyn std::io::Read,
    ) -> Result<()> {
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        anyhow::ensure!(
            &magic == PRIME_CKPT_MAGIC_NVPRIMEK,
            "{PRIME_CKPT_BAD_HEADER}: magic {magic:?} is not a prime checkpoint"
        );
        let mut b4 = [0u8; 4];
        input.read_exact(&mut b4)?;
        let ver = u32::from_le_bytes(b4);
        anyhow::ensure!(
            ver == PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES,
            "{PRIME_CKPT_BAD_HEADER}: cache layout version {ver} != {}, the checkpoint predates or postdates this cache code",
            PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES
        );
        let mut b8 = [0u8; 8];
        input.read_exact(&mut b8)?;
        let fp_len = u64::from_le_bytes(b8) as usize;
        anyhow::ensure!(
            fp_len <= 4096,
            "{PRIME_CKPT_BAD_HEADER}: fingerprint length {fp_len} is not a fingerprint"
        );
        let mut fp = vec![0u8; fp_len];
        input.read_exact(&mut fp)?;
        let found = String::from_utf8_lossy(&fp).into_owned();
        anyhow::ensure!(
            found == fingerprint,
            "{PRIME_CKPT_FINGERPRINT_MISMATCH}: checkpoint carries {found:?}, this run expects {fingerprint:?}"
        );
        let mut hdr = [0u64; 5];
        for h in hdr.iter_mut() {
            input.read_exact(&mut b8)?;
            *h = u64::from_le_bytes(b8);
        }
        let [rows, n_layers, n_kv, hd, n_lin] = hdr.map(|v| v as usize);
        anyhow::ensure!(
            n_layers == self.layers.len()
                && n_kv == self.n_kv_heads
                && hd == self.head_dim
                && n_lin == self.lin_attn_states.len(),
            "{PRIME_CKPT_GEOMETRY_MISMATCH}: checkpoint slots/kv/hd/lin {n_layers}/{n_kv}/{hd}/{n_lin} vs cache {}/{}/{}/{}",
            self.layers.len(),
            self.n_kv_heads,
            self.head_dim,
            self.lin_attn_states.len()
        );
        anyhow::ensure!(
            rows <= self.max_seq_len,
            "{PRIME_CKPT_GEOMETRY_MISMATCH}: primed depth {rows} > cache max_seq_len {}",
            self.max_seq_len
        );
        for i in 0..n_lin {
            let mut present = [0u8; 1];
            input.read_exact(&mut present)?;
            self.lin_attn_states[i] = match present[0] {
                0 => None,
                1 => Some(LinAttnState::restore_primed_state_checked(
                    input,
                    &self.device,
                )?),
                tag => anyhow::bail!(
                    "{PRIME_CKPT_BAD_HEADER}: lin state presence tag {tag} at slot {i}"
                ),
            };
        }
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let kv_elems = rows * n_kv * hd;
        let sc_elems = rows * n_kv;
        let mut kv_bytes = vec![0u8; kv_elems];
        let mut sc_bytes = vec![0u8; sc_elems * 4];
        for slot in &mut self.layers {
            for fp8 in [&mut slot.k_fp8, &mut slot.v_fp8] {
                input.read_exact(&mut kv_bytes)?;
                let mut dst = fp8.slice_mut(0..kv_elems);
                stream
                    .memcpy_htod(&kv_bytes[..], &mut dst)
                    .map_err(|e| anyhow::anyhow!("prime restore fp8 htod: {e:?}"))?;
            }
            for scales in [&mut slot.k_scales, &mut slot.v_scales] {
                input.read_exact(&mut sc_bytes)?;
                let s: Vec<f32> = sc_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut dst = scales.slice_mut(0..sc_elems);
                stream
                    .memcpy_htod(&s[..], &mut dst)
                    .map_err(|e| anyhow::anyhow!("prime restore scales htod: {e:?}"))?;
            }
        }
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after prime restore: {e:?}"))?;
        self.current_len = rows;
        self.fused_lin_verify_rows_pending_rollback = 0;
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl Qwen3MoeKvCache {
    pub fn new(
        config: &Qwen3MoeConfig,
        max_seq_len: usize,
        device: &Device,
        _dtype: DType,
    ) -> Result<Self> {
        let mut full_slot_for_layer: Vec<Option<usize>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut lin_attn_for_layer: Vec<Option<usize>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut n_lin = 0usize;
        for ty in &config.layer_types {
            match ty {
                LayerType::FullAttention => {
                    let idx = full_slot_for_layer.iter().filter(|x| x.is_some()).count();
                    full_slot_for_layer.push(Some(idx));
                    lin_attn_for_layer.push(None);
                }
                LayerType::LinearAttention => {
                    full_slot_for_layer.push(None);
                    lin_attn_for_layer.push(Some(n_lin));
                    n_lin += 1;
                }
            }
        }
        let lin_attn_states: Vec<Option<LinAttnState>> = (0..n_lin).map(|_| None).collect();
        Ok(Self {
            full_slot_for_layer,
            lin_attn_for_layer,
            lin_attn_states,
            current_len: 0,
            max_seq_len,
            _n_kv_heads: config.num_key_value_heads,
            _head_dim: config.head_dim,
            device: device.clone(),
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
        for st in &mut self.lin_attn_states {
            *st = None;
        }
    }

    pub fn full_slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.full_slot_for_layer.get(layer).copied().flatten()
    }

    pub fn lin_attn_slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.lin_attn_for_layer.get(layer).copied().flatten()
    }

    #[allow(dead_code)]
    fn lin_attn_step(
        &mut self,
        slot: usize,
        la: &LinearAttention,
        input: &Tensor,
    ) -> Result<Tensor> {
        let mut state = self.lin_attn_states[slot].take();
        let out = la.forward_with_state(input, &mut state)?;
        self.lin_attn_states[slot] = state;
        Ok(out)
    }

    pub fn set_current_len(&mut self, len: usize) {
        self.current_len = len;
    }

    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
}

pub struct Qwen3MoeLayer {
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
    mixer: LayerMixer,
    ffn: LayerFfn,
}

pub struct Qwen3Moe {
    config: Qwen3MoeConfig,
    dense_intermediate: Option<usize>,
    embed_weight: Tensor,
    layers: Vec<Qwen3MoeLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    dtype: DType,
    device: Device,
}

impl Qwen3Moe {
    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.config
    }

    pub fn moe_block_for_dispatch_parity_tests(&self, layer_idx: usize) -> Option<&LayerFfn> {
        self.layers.get(layer_idx).map(|l| &l.ffn)
    }

    #[cfg(feature = "cuda")]
    pub fn dump_trunk_phase_ladder_after_replay_when_nv_q36_trunk_events_1(&self) {
        if !trunk_events::armed_env_opt_in_nv_q36_trunk_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes()
        {
            return;
        }
        let kinds: Vec<bool> = self
            .layers
            .iter()
            .map(|l| matches!(l.mixer, LayerMixer::Full(_)))
            .collect();
        trunk_events::dump_after_replay_summing_by_layer_kind(&kinds);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn dense_intermediate(&self) -> Option<usize> {
        self.dense_intermediate
    }

    pub fn is_dense(&self) -> bool {
        self.dense_intermediate.is_some()
    }

    pub fn embed_weight(&self) -> &Tensor {
        &self.embed_weight
    }

    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    pub fn rope(&self) -> &Rope {
        &self.rope
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<Qwen3MoeKvCache> {
        Qwen3MoeKvCache::new(&self.config, max_seq_len, &self.device, self.dtype)
    }

    #[cfg(feature = "cuda")]
    pub fn ensure_fused_lin_states(&self, cache: &mut Qwen3MoeKvCache) -> Result<()> {
        for (idx, layer) in self.layers.iter().enumerate() {
            let LayerMixer::Linear(la) = &layer.mixer else {
                continue;
            };
            let Some(slot) = cache.lin_attn_slot_for_layer(idx) else {
                continue;
            };
            cache.ensure_fused_slot(slot, la)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn ensure_fused_lin_verify_ckpts(
        &self,
        cache: &mut Qwen3MoeKvCache,
        rows: usize,
    ) -> Result<()> {
        anyhow::ensure!(rows >= 1, "ensure_fused_lin_verify_ckpts: rows must be >= 1");
        for (idx, layer) in self.layers.iter().enumerate() {
            let LayerMixer::Linear(la) = &layer.mixer else {
                continue;
            };
            let Some(slot) = cache.lin_attn_slot_for_layer(idx) else {
                continue;
            };
            cache.ensure_fused_verify_ckpt_slot(slot, la, rows)?;
        }
        Ok(())
    }

    pub fn from_loader(
        config: Qwen3MoeConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let qconfig = QuantizationConfig::none();
            Self::build(config, None, weights, &qconfig, device)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let qconfig = QuantizationConfig::none();
            Self::build_cpu(config, None, weights, &qconfig, device)
        }
    }

    pub fn from_loader_dense(
        config: Qwen3_5DenseConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        let qconfig = QuantizationConfig::none();
        Self::from_loader_dense_quantized(config, weights, &qconfig, device)
    }

    pub const Q38_27B_MIXED_WEIGHTS_LOAD_AS_ABOUT_22_GB_VRAM_10_6B_FP8_PARAMS_STAY_RESIDENT_E4M3_11_GB_15_0B_NVFP4_MLP_PARAMS_STAY_PACKED_8_GB_EMBED_BF16_3_GB_WAS_54_GB_ON_THE_NV_Q38_FP8_DEQUANT_BF16_ARM: f64 =
        22.1;

    pub fn from_loader_dense_quantized(
        config: Qwen3_5DenseConfig,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        device: &Device,
    ) -> Result<Self> {
        let intermediate_size = config.intermediate_size;
        let base = config.trunk();
        #[cfg(feature = "cuda")]
        {
            Self::build(base, Some(intermediate_size), weights, qconfig, device)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::build_cpu(base, Some(intermediate_size), weights, qconfig, device)
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_quantized(
        config: Qwen3MoeConfig,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::build(config, None, weights, qconfig, device)
    }

    #[cfg(feature = "cuda")]
    fn build(
        config: Qwen3MoeConfig,
        dense_intermediate: Option<usize>,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::BF16;
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Qwen3Moe requires CUDA device"),
        };
        let stream = dev.cuda_stream();
        let runner = Arc::new(Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
            stream.clone(),
        )?));
        let fp8_runner = Arc::new(Mutex::new(nv_quant::fp8::Fp8GemmRunner::new(
            stream.clone(),
        )?));

        let embed_weight = load_named(
            weights,
            &[
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ],
            dtype,
        )?;
        let ed = embed_weight.dims();
        if ed.len() != 2 || ed[0] != config.vocab_size || ed[1] != config.hidden_size {
            anyhow::bail!(
                "embed shape mismatch: got {:?} expected [{}, {}]",
                ed,
                config.vocab_size,
                config.hidden_size
            );
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = build_layer(
                &config,
                dense_intermediate,
                i,
                weights,
                qconfig,
                runner.clone(),
                &fp8_runner,
                device,
                dtype,
            )?;
            layers.push(layer);
        }

        let final_norm = load_rmsnorm_plus_one(
            weights,
            &["model.language_model.norm.weight", "model.norm.weight"],
            config.hidden_size,
            config.rms_norm_eps,
            dtype,
        )?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_weight.clone(), None)?
        } else if nv_layers::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(
            weights, "lm_head",
        ) {
            if q38_fp8_dequant_bf16_env_restores_the_4_byte_prequant_arm() {
                nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
                    weights,
                    "lm_head",
                    config.vocab_size,
                    config.hidden_size,
                    dtype,
                )?
            } else {
                nv_layers::linear::fp8_e4m3_rowscale_checkpoint_resident_linear(
                    weights,
                    "lm_head",
                    config.vocab_size,
                    config.hidden_size,
                    device,
                    fp8_runner.clone(),
                )?
            }
        } else if nv_q36_lm_head_fp8_env_opt_in_quantizes_a_bf16_checkpoint_lm_head_to_fp8_resident(
        ) {
            let w = load_named(weights, &["lm_head.weight"], DType::BF16)?;
            anyhow::ensure!(
                w.dims() == [config.vocab_size, config.hidden_size],
                "lm_head.weight: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                w.dims()
            );
            let weight_host: Vec<half::bf16> = w.flatten_all()?.to_vec1()?;
            let (weight_bytes, row_scales) = nv_layers::linear::fp8_weight_payload(
                &weight_host,
                config.vocab_size,
                config.hidden_size,
                None,
                nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
            )?;
            #[allow(deprecated)]
            let weight_u8 = stream
                .clone_htod(&weight_bytes)
                .map_err(|e| anyhow::anyhow!(e))?;
            Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
                weight_u8,
                row_scales,
                config.hidden_size,
                config.vocab_size,
                None,
                device,
                fp8_runner.clone(),
                nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
            )?
        } else {
            Linear::new(load_named(weights, &["lm_head.weight"], dtype)?, None)?
        };

        let rope = Rope::new(
            RopeConfig {
                head_dim: config.rotary_dim().max(2),
                max_seq_len: config.max_position_embeddings,
                base: config.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;

        Ok(Self {
            config,
            dense_intermediate,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            rope,
            dtype,
            device: device.clone(),
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn build_cpu(
        _config: Qwen3MoeConfig,
        _dense_intermediate: Option<usize>,
        _weights: &WeightLoader,
        _qconfig: &QuantizationConfig,
        _device: &Device,
    ) -> Result<Self> {
        anyhow::bail!("Qwen3Moe::from_loader requires --features cuda")
    }

    #[cfg(feature = "cuda")]
    pub fn prebuild_gdn_nvfp4_verify_twins_if_armed(&self, device: &Device) {
        let Device::Cuda(dev) = device else { return };
        for layer in &self.layers {
            if let LayerMixer::Linear(la) = &layer.mixer {
                la.prebuild_nvfp4_verify_twins_from_the_fp8_residents_the_disk_itself_stores_qkv_z_as_f8e4m3(dev);
            }
        }
    }

    pub fn forward(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!("Qwen3Moe.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Qwen3Moe.forward: positions shape {:?} != [{}]",
                positions.dims(),
                seq
            );
        }

        let prof_enabled = nv_config::env_arm_present_at_all_reread_every_call("NV_PROF_DECODE");
        let prof_dev: Option<candle_core::CudaDevice> = if prof_enabled {
            match tokens.device() {
                Device::Cuda(d) => Some(d.clone()),
                _ => None,
            }
        } else {
            None
        };
        let mut prof: std::collections::BTreeMap<&'static str, f64> =
            std::collections::BTreeMap::new();
        let prof_sync = |d: &Option<candle_core::CudaDevice>| {
            #[cfg(feature = "cuda")]
            if let Some(dev) = d {
                let _ = dev.cuda_stream().synchronize();
            }

            #[cfg(not(feature = "cuda"))]
            let _ = d;
        };

        prof_sync(&prof_dev);
        let t_embed = std::time::Instant::now();
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = self
            .embed_weight
            .index_select(&tokens_flat, 0)?
            .reshape((1usize, seq, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        prof_sync(&prof_dev);
        if prof_enabled {
            *prof.entry("embed").or_default() += t_embed.elapsed().as_secs_f64() * 1000.0;
        }

        let trace = nv_config::env_arm_present_at_all_reread_every_call("NV_TRACE_LAYERS");
        if trace {
            dump_stats("embed", &x)?;
        }

        prof_sync(&prof_dev);
        let t_pre_norm0 = std::time::Instant::now();
        let mut residual = x.clone();
        let mut normed = self.layers[0].pre_norm.forward(&x)?;
        prof_sync(&prof_dev);
        if prof_enabled {
            *prof.entry("pre_norm_first").or_default() +=
                t_pre_norm0.elapsed().as_secs_f64() * 1000.0;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            prof_sync(&prof_dev);
            let t_mix = std::time::Instant::now();
            let (mixed, mix_label): (Tensor, &'static str) = match &layer.mixer {
                LayerMixer::Full(attn) => (
                    attn.forward(&normed, &self.rope, positions)?,
                    "mixer_full_attn",
                ),
                LayerMixer::Linear(la) => (la.forward(&normed)?, "mixer_linear_attn"),
            };
            prof_sync(&prof_dev);
            if prof_enabled {
                *prof.entry(mix_label).or_default() += t_mix.elapsed().as_secs_f64() * 1000.0;
            }
            if trace {
                dump_stats(&format!("L{li}.attn_out"), &mixed)?;
            }
            prof_sync(&prof_dev);
            let t_post = std::time::Instant::now();
            let (normed_post, residual_after_attn) =
                layer.post_norm.forward_residual(&mixed, &residual)?;
            prof_sync(&prof_dev);
            if prof_enabled {
                *prof.entry("post_norm").or_default() += t_post.elapsed().as_secs_f64() * 1000.0;
            }
            prof_sync(&prof_dev);
            let t_moe = std::time::Instant::now();
            let moe_out = layer.ffn.forward(&normed_post)?;
            prof_sync(&prof_dev);
            if prof_enabled {
                *prof.entry(layer.ffn.label()).or_default() +=
                    t_moe.elapsed().as_secs_f64() * 1000.0;
            }
            if trace {
                dump_stats(&format!("L{li}.{}_out", layer.ffn.label()), &moe_out)?;
            }
            if li + 1 < self.layers.len() {
                prof_sync(&prof_dev);
                let t_next = std::time::Instant::now();
                let (normed_next, residual_after_moe) = self.layers[li + 1]
                    .pre_norm
                    .forward_residual(&moe_out, &residual_after_attn)?;
                normed = normed_next;
                residual = residual_after_moe;
                prof_sync(&prof_dev);
                if prof_enabled {
                    *prof.entry("pre_norm_next_fused").or_default() +=
                        t_next.elapsed().as_secs_f64() * 1000.0;
                }
            } else {
                prof_sync(&prof_dev);
                let t_add = std::time::Instant::now();
                residual = residual_after_attn.add(&moe_out)?;
                prof_sync(&prof_dev);
                if prof_enabled {
                    *prof.entry("residual_add_final").or_default() +=
                        t_add.elapsed().as_secs_f64() * 1000.0;
                }
            }
            if trace && matches!(li, 0 | 1 | 3 | 7 | 19 | 39) {
                dump_stats(&format!("L{li}.out"), &residual)?;
            }
        }
        let x = residual;

        prof_sync(&prof_dev);
        let t_final = std::time::Instant::now();
        let x = self.final_norm.forward(&x)?;
        prof_sync(&prof_dev);
        if prof_enabled {
            *prof.entry("final_norm").or_default() += t_final.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            dump_stats("final_norm", &x)?;
        }
        prof_sync(&prof_dev);
        let t_lm = std::time::Instant::now();
        let logits = self.lm_head.forward(&x)?;
        prof_sync(&prof_dev);
        if prof_enabled {
            *prof.entry("lm_head").or_default() += t_lm.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            dump_stats("logits", &logits)?;
            let last = logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..last.len()).collect();
            idx.sort_by(|&a, &b| {
                last[b]
                    .partial_cmp(&last[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top: Vec<(usize, f32)> = idx.iter().take(15).map(|&i| (i, last[i])).collect();
            eprintln!("[trace] top15_logits: {:?}", top);
        }

        if prof_enabled {
            let total: f64 = prof.values().sum();
            eprintln!(
                "[prof] --- decode forward breakdown (sum across {} layers) ---",
                self.layers.len()
            );
            let mut entries: Vec<(&&'static str, &f64)> = prof.iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (label, ms) in &entries {
                eprintln!(
                    "[prof]   {:>22}  {:8.2} ms  ({:.1}%)",
                    label,
                    ms,
                    100.0 * *ms / total.max(1e-12)
                );
            }
            eprintln!("[prof]   {:>22}  {:8.2} ms", "TOTAL", total);
        }
        Ok(logits)
    }

    pub fn forward_hidden(&self, tokens: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!(
                "Qwen3Moe.forward_hidden: tokens must be [1, seq], got {:?}",
                dims
            );
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Qwen3Moe.forward_hidden: positions shape {:?} != [{}]",
                positions.dims(),
                seq
            );
        }
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = self
            .embed_weight
            .index_select(&tokens_flat, 0)?
            .reshape((1usize, seq, self.config.hidden_size))?
            .to_dtype(self.dtype)?;

        let mut residual = x.clone();
        let mut normed = self.layers[0].pre_norm.forward(&x)?;
        for (li, layer) in self.layers.iter().enumerate() {
            let mixed = match &layer.mixer {
                LayerMixer::Full(attn) => attn.forward(&normed, &self.rope, positions)?,
                LayerMixer::Linear(la) => la.forward(&normed)?,
            };
            let (normed_post, residual_after_attn) =
                layer.post_norm.forward_residual(&mixed, &residual)?;
            let moe_out = layer.ffn.forward(&normed_post)?;
            if li + 1 < self.layers.len() {
                let (normed_next, residual_after_moe) = self.layers[li + 1]
                    .pre_norm
                    .forward_residual(&moe_out, &residual_after_attn)?;
                normed = normed_next;
                residual = residual_after_moe;
            } else {
                residual = residual_after_attn.add(&moe_out)?;
            }
        }
        let hidden = residual;
        let x = self.final_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&x)?;
        Ok((logits, hidden))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<Tensor> {
        self.forward_with_cache_dispatched(tokens, positions, cache, None)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_serving_prefill_last_row_logits_because_chat_prefill_samples_only_position_seq_minus_1(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<Tensor> {
        self.forward_with_cache_dispatched_rows(tokens, positions, cache, None, Some(1))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_dispatched(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
    ) -> Result<Tensor> {
        self.forward_with_cache_dispatched_rows(tokens, positions, cache, moe_dispatch, None)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_dispatched_rows(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
        logit_rows: Option<usize>,
    ) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache: tokens must be [1, seq], got {:?}",
                dims
            );
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache: positions shape {:?} != [{}]",
                positions.dims(),
                seq
            );
        }
        let write_start = cache.current_len();
        let new_total = write_start + seq;
        if new_total > cache.max_seq_len() {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache: new_total {} > max_seq_len {}",
                new_total,
                cache.max_seq_len()
            );
        }
        cache.prepare_for_step(write_start, new_total)?;

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = match embed_lookup_bf16(&self.embed_weight, &tokens_flat)? {
            Some(rows) => rows,
            None => self.embed_weight.index_select(&tokens_flat, 0)?,
        }
        .reshape((1usize, seq, self.config.hidden_size))?
        .to_dtype(self.dtype)?;

        self.forward_embedded_with_cache_dispatched_rows(
            x,
            &self.rope,
            positions,
            cache,
            moe_dispatch,
            logit_rows,
            write_start,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
        &self,
        prompt: &[u32],
        chunk: usize,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
    ) -> Result<Tensor> {
        anyhow::ensure!(chunk >= 1, "prefill_chunks: chunk must be >= 1");
        anyhow::ensure!(!prompt.is_empty(), "prefill_chunks: empty prompt");
        let mut fed = 0usize;
        let mut last: Option<Tensor> = None;
        while fed < prompt.len() {
            let n = chunk.min(prompt.len() - fed);
            let start = cache.current_len();
            let toks =
                Tensor::from_vec(prompt[fed..fed + n].to_vec(), (1usize, n), &self.device)?;
            let pos: Vec<i32> = (start as i32..(start + n) as i32).collect();
            let pos_t = Tensor::from_vec(pos, n, &self.device)?;
            last = Some(self.forward_with_cache_dispatched_rows(
                &toks,
                &pos_t,
                cache,
                moe_dispatch,
                Some(1),
            )?);
            fed += n;
        }
        Ok(last.expect("prefill_chunks: the non-empty prompt fed at least one chunk"))
    }

    #[cfg(feature = "cuda")]
    pub fn new_prefill_graph_session(&self, chunk: usize) -> Result<Qwen38PrefillGraphSession> {
        Qwen38PrefillGraphSession::new(self, chunk)
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_chunks_graphed_serving_last_row_logits_replaying_gdn_mlp_segments_with_full_attention_eager_breaks(
        &self,
        prompt: &[u32],
        chunk: usize,
        cache: &mut Qwen3MoeKvCache,
        sess: &mut Qwen38PrefillGraphSession,
    ) -> Result<Tensor> {
        anyhow::ensure!(chunk >= 1, "prefill_chunks_graphed: chunk must be >= 1");
        anyhow::ensure!(!prompt.is_empty(), "prefill_chunks_graphed: empty prompt");
        anyhow::ensure!(
            sess.chunk == chunk,
            "prefill_chunks_graphed: session built for chunk {} but called with {}",
            sess.chunk,
            chunk
        );
        let hostile = prefill_graph_capture_hostile_env_because_the_inner_gdn_graph_arm_nests_a_capture_and_the_linattn_profiler_syncs_every_lap();
        if sess.capture_failed || hostile.is_some() || sess.install_pinned_lin_states(cache).is_err()
        {
            if let Some(env) = hostile {
                sess.eager_fallback_reported_once.call_once(|| {
                    eprintln!("[q38-prefill-graph] eager fallback: {env} is armed");
                });
            }
            return self
                .prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
                    prompt, chunk, cache, None,
                );
        }
        let mut fed = 0usize;
        let mut last: Option<Tensor> = None;
        while fed < prompt.len() {
            let n = chunk.min(prompt.len() - fed);
            let piece = &prompt[fed..fed + n];
            if n < chunk || !sess.warmed_one_eager_full_chunk_so_plans_and_workspaces_exist {
                let start = cache.current_len();
                let toks = Tensor::from_vec(piece.to_vec(), (1usize, n), &self.device)?;
                let pos: Vec<i32> = (start as i32..(start + n) as i32).collect();
                let pos_t = Tensor::from_vec(pos, n, &self.device)?;
                last = Some(self.forward_with_cache_dispatched_rows(
                    &toks,
                    &pos_t,
                    cache,
                    None,
                    Some(1),
                )?);
                if n == chunk {
                    sess.pin_lin_states_from_cache_after_the_eager_warm_chunk(cache)?;
                    sess.warmed_one_eager_full_chunk_so_plans_and_workspaces_exist = true;
                }
            } else {
                last = Some(self.prefill_full_chunk_graphed(piece, cache, sess)?);
            }
            fed += n;
        }
        Ok(last.expect("prefill_chunks_graphed: the non-empty prompt fed at least one chunk"))
    }

    #[cfg(feature = "cuda")]
    fn prefill_full_chunk_graphed(
        &self,
        chunk_tokens: &[u32],
        cache: &mut Qwen3MoeKvCache,
        sess: &mut Qwen38PrefillGraphSession,
    ) -> Result<Tensor> {
        let seq = chunk_tokens.len();
        let hidden = self.config.hidden_size;
        let write_start = cache.current_len();
        let new_total = write_start + seq;
        anyhow::ensure!(
            new_total <= cache.max_seq_len(),
            "prefill_full_chunk_graphed: new_total {} > max_seq_len {}",
            new_total,
            cache.max_seq_len()
        );
        cache.prepare_for_step(write_start, new_total)?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("prefill_full_chunk_graphed requires CUDA"),
        };
        crate::graph_engine::disable_event_tracking_no_earlier_than_the_first_capture(&dev);
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        {
            let tok_ptr = cuda_tensor_device_ptr::<u32>(&sess.tok_stage, &stream)?;
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    tok_ptr,
                    chunk_tokens,
                    stream.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("stage chunk tokens htod: {e:?}"))?;
            }
        }
        let positions_v: Vec<i32> = (write_start as i32..new_total as i32).collect();
        let positions = Tensor::from_vec(positions_v, seq, &self.device)?;

        let n_layers = self.layers.len();
        let mut li = 0usize;
        let mut seg = 0u64;
        let mut eager_final_residual: Option<Tensor> = None;
        while li < n_layers {
            match &self.layers[li].mixer {
                LayerMixer::Linear(_) => {
                    let seg_end = (li..n_layers)
                        .find(|&j| matches!(self.layers[j].mixer, LayerMixer::Full(_)))
                        .unwrap_or(n_layers);
                    let key = PREFILL_SEGMENT_GRAPH_SHAPE_TOKEN_BASE_KEEPS_DECODE_1_AND_VERIFY_0X51XXX_KEYS_UNTOUCHED
                        + seg;
                    let start_li = li;
                    let model = &*self;
                    let cache_ref = &*cache;
                    let tok_stage = &sess.tok_stage;
                    let normed_stage = &sess.normed_stage;
                    let residual_stage = &sess.residual_stage;
                    let capture_result = sess.runner.run(key, |s| {
                        nv_layers::cuda_stream::with_stream(s.clone(), || {
                            let (normed_c, mut residual_c) = if start_li == 0 {
                                let tokens_flat = tok_stage.flatten_all()?.to_dtype(DType::U32)?;
                                let x = match embed_lookup_bf16(&model.embed_weight, &tokens_flat)?
                                {
                                    Some(rows) => rows,
                                    None => model.embed_weight.index_select(&tokens_flat, 0)?,
                                }
                                .reshape((1usize, seq, hidden))?
                                .to_dtype(model.dtype)?;
                                let normed = model.layers[0].pre_norm.forward(&x)?;
                                (normed, x)
                            } else {
                                (normed_stage.clone(), residual_stage.clone())
                            };
                            let mut normed_next: Option<Tensor> = Some(normed_c);
                            for lj in start_li..seg_end {
                                let normed_in = normed_next.take().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "prefill graph segment reached layer {lj} with no normed input"
                                    )
                                })?;
                                let la = match &model.layers[lj].mixer {
                                    LayerMixer::Linear(la) => la,
                                    _ => unreachable!(),
                                };
                                let slot = cache_ref.lin_attn_slot_for_layer(lj).ok_or_else(|| {
                                    anyhow::anyhow!("layer {lj} is LinearAttention without a slot")
                                })?;
                                let mixed = cache_ref
                                    .lin_attn_step_state_pinned_for_prefill_graph(slot, la, &normed_in)?;
                                let (nx, rx) =
                                    model.prefill_graph_unit_tail(lj, &mixed, &residual_c, seq)?;
                                normed_next = nx;
                                residual_c = rx;
                            }
                            if let Some(nx) = &normed_next {
                                dtod_bf16_tensor_into_stage(nx, normed_stage, s)?;
                            }
                            dtod_bf16_tensor_into_stage(&residual_c, residual_stage, s)?;
                            Ok(())
                        })
                    });
                    if let Err(e) = capture_result {
                        sess.capture_failed = true;
                        sess.runner.invalidate();
                        let _ = stream.synchronize();
                        return Err(e).context(
                            "prefill segment graph capture/replay failed; session falls back to \
                             eager on the next prefill, but this chunk's cache state is torn -- \
                             reset and re-prefill",
                        );
                    }
                    li = seg_end;
                    seg += 1;
                }
                LayerMixer::Full(attn) => {
                    let normed_e = sess.normed_stage.clone();
                    let residual_e = sess.residual_stage.clone();
                    let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!("layer {li} is FullAttention without a cache slot")
                    })?;
                    let mixed = attn.forward_with_cache(
                        &normed_e,
                        &self.rope,
                        &positions,
                        cache,
                        slot,
                        write_start,
                        new_total,
                    )?;
                    let (nx, rx) = self.prefill_graph_unit_tail(li, &mixed, &residual_e, seq)?;
                    match nx {
                        Some(nx) => {
                            dtod_bf16_tensor_into_stage(&nx, &sess.normed_stage, &stream)?;
                            dtod_bf16_tensor_into_stage(&rx, &sess.residual_stage, &stream)?;
                        }
                        None => {
                            eager_final_residual = Some(rx);
                        }
                    }
                    li += 1;
                }
            }
        }
        let residual_final = match eager_final_residual {
            Some(r) => r,
            None => sess.residual_stage.clone(),
        };
        let x = residual_final.narrow(1, seq - 1, 1)?;
        let x = self.final_norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        cache.advance(seq);
        Ok(logits)
    }

    #[cfg(feature = "cuda")]
    fn prefill_graph_unit_tail(
        &self,
        li: usize,
        mixed: &Tensor,
        residual: &Tensor,
        seq: usize,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let layer = &self.layers[li];
        let (normed_post, residual_after_attn) = layer.post_norm.forward_residual(mixed, residual)?;
        let moe_out =
            dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(&layer.ffn, &normed_post, seq)?;
        if li + 1 < self.layers.len() {
            let (normed_next, residual_after_moe) = self.layers[li + 1]
                .pre_norm
                .forward_residual(&moe_out, &residual_after_attn)?;
            Ok((Some(normed_next), residual_after_moe))
        } else {
            Ok((None, residual_after_attn.add(&moe_out)?))
        }
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_prefill_image_rows_last_row_logits(
        &self,
        tokens: &Tensor,
        splices: &[crate::qwen3_mm_splice::Qwen3ImageRowSplice],
        mrope: &crate::qwen3_mm_splice::Qwen3MropePositions,
        mrope_section: [usize; 3],
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<Tensor> {
        self.forward_with_cache_prefill_image_rows(
            tokens,
            splices,
            mrope,
            mrope_section,
            cache,
            Some(1),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_prefill_image_rows(
        &self,
        tokens: &Tensor,
        splices: &[crate::qwen3_mm_splice::Qwen3ImageRowSplice],
        mrope: &crate::qwen3_mm_splice::Qwen3MropePositions,
        mrope_section: [usize; 3],
        cache: &mut Qwen3MoeKvCache,
        logit_rows: Option<usize>,
    ) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!(
                "Qwen3Moe.prefill_image_rows: tokens must be [1, seq], got {:?}",
                dims
            );
        }
        let seq = dims[1];
        anyhow::ensure!(
            mrope.len() == seq,
            "Qwen3Moe.prefill_image_rows: {} mrope positions for {seq} tokens",
            mrope.len()
        );
        anyhow::ensure!(
            cache.current_len() == 0,
            "Qwen3Moe.prefill_image_rows: the per-call mrope rope table indexes rows 0..seq, \
             so image prefill must start from an empty cache (current_len {})",
            cache.current_len()
        );
        let write_start = 0usize;
        let new_total = seq;
        if new_total > cache.max_seq_len() {
            anyhow::bail!(
                "Qwen3Moe.prefill_image_rows: new_total {} > max_seq_len {}",
                new_total,
                cache.max_seq_len()
            );
        }
        cache.prepare_for_step(write_start, new_total)?;

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = match embed_lookup_bf16(&self.embed_weight, &tokens_flat)? {
            Some(rows) => rows,
            None => self.embed_weight.index_select(&tokens_flat, 0)?,
        }
        .reshape((1usize, seq, self.config.hidden_size))?
        .to_dtype(self.dtype)?;
        let x = crate::qwen3_mm_splice::splice_image_rows_into_embedded(&x, splices)?;

        let row_indexed_rope = crate::qwen3_mm_splice::mrope_rope_one_row_per_token(
            &self.rope,
            mrope,
            mrope_section,
            &self.device,
        )?;
        let iota: Vec<u32> = (0..seq as u32).collect();
        let positions = Tensor::from_vec(iota, seq, &self.device)?;

        self.forward_embedded_with_cache_dispatched_rows(
            x,
            &row_indexed_rope,
            &positions,
            cache,
            None,
            logit_rows,
            write_start,
        )
    }

    #[cfg(feature = "cuda")]
    fn gdn_prenorm_fold_request_for_next_layer(
        &self,
        li: usize,
        seq: usize,
        device_of: &Tensor,
    ) -> Result<Option<(Tensor, f32)>> {
        if seq != 1
            || li + 1 >= self.layers.len()
            || !nv_layers::linear_attn::gdn_prenorm_fold_env_read_per_call_so_the_kill_switch_works_mid_process()
            || nv_q38_down_qfold_env_opt_in_nv_q38_down_qfold_1_act_quant_in_down_gemv_prologue()
            || !matches!(self.layers[li + 1].mixer, LayerMixer::Linear(_))
        {
            return Ok(None);
        }
        let Device::Cuda(dev) = device_of.device() else {
            return Ok(None);
        };
        let pack =
            gdn_prenorm_rstd_pack_tensor_rstd_ssq_count_zeroed_once_then_kernel_maintained(dev)?;
        Ok(Some((pack, self.layers[li + 1].pre_norm.eps() as f32)))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_embedded_with_cache_dispatched_rows(
        &self,
        x: Tensor,
        rope: &Rope,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
        logit_rows: Option<usize>,
        write_start: usize,
    ) -> Result<Tensor> {
        let seq = x.dims()[1];
        let new_total = write_start + seq;

        let mut splits = PrefillWallSplits::begin_env_nv_prof_prefill(x.device());
        if seq == 1 {
            decode_prof::begin_env_nv_prof_decode_refusing_mid_capture_because_every_lap_syncs(
                x.device(),
            );
        }
        let mut t_lap = std::time::Instant::now();
        let mut residual = x.clone();
        let mut normed = self.layers[0].pre_norm.forward(&x)?;
        let mut gdn_fold_rstd: Option<Tensor> = None;
        t_lap = splits.lap("embed_and_first_norm", t_lap);
        decode_prof::lap("embed_and_first_norm");
        for (li, layer) in self.layers.iter().enumerate() {
            let mixed = match &layer.mixer {
                LayerMixer::Full(attn) => {
                    let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen3Moe.forward_with_cache: layer {} is FullAttention but cache has no slot",
                            li
                        )
                    })?;
                    attn.forward_with_cache(
                        &normed,
                        rope,
                        positions,
                        cache,
                        slot,
                        write_start,
                        new_total,
                    )?
                }
                LayerMixer::Linear(la) => {
                    let lin_slot = cache.lin_attn_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen3Moe.forward_with_cache: layer {} is LinearAttention but cache has no slot",
                            li
                        )
                    })?;
                    let mut folded_out = None;
                    if let Some(rstd_pack) = gdn_fold_rstd.take() {
                        folded_out = cache.lin_attn_step_prenorm_folded(
                            lin_slot,
                            la,
                            &normed,
                            layer.pre_norm.weight_bf16(),
                            &rstd_pack,
                        )?;
                        if folded_out.is_none() {
                            normed = layer.pre_norm.forward(&normed)?;
                        }
                    }
                    let out = match folded_out {
                        Some(o) => o,
                        None => cache.lin_attn_step(lin_slot, la, &normed)?,
                    };
                    decode_prof::lap("gdn_chain");
                    out
                }
            };
            t_lap = splits.lap(
                match &layer.mixer {
                    LayerMixer::Full(_) => "mixer_full_attn",
                    LayerMixer::Linear(_) => "mixer_linear_attn",
                },
                t_lap,
            );
            let mut post_pair = None;
            let mut folded_norm_quant = None;
            if seq == 1 {
                if !matches!((moe_dispatch, layer.ffn.as_moe()), (Some(_), Some(_))) {
                    folded_norm_quant = fused_post_norm_residual_rowquant_decode_m1(
                        &layer.post_norm,
                        &layer.ffn,
                        &mixed,
                        &residual,
                    )?;
                }
                if folded_norm_quant.is_none() {
                    post_pair =
                        rmsnorm_residual_decode_m1_writeout_skipping_the_dtod_residual_copy(
                            &layer.post_norm,
                            &mixed,
                            &residual,
                        )?;
                }
            }
            if let Some(folded) = folded_norm_quant {
                decode_prof::lap("norms_and_residual");
                let mlp = layer.ffn.as_dense().ok_or_else(|| {
                    anyhow::anyhow!("Qwen3Moe.forward_with_cache: folded norm quant on non-dense ffn")
                })?;
                let fold_req = self.gdn_prenorm_fold_request_for_next_layer(li, seq, &folded.res_out)?;
                let (summed, rstd_emitted) = w4a8_dual_silu_down_chain_after_x_quant_decode_m1(
                    mlp,
                    &folded.x_q8,
                    &folded.x_scale,
                    &folded.res_out,
                    folded.hidden,
                    folded.inter,
                    fold_req.as_ref().map(|(p, e)| (p, *e)),
                )?;
                t_lap = splits.lap(layer.ffn.label(), t_lap);
                decode_prof::lap(layer.ffn.label());
                if li + 1 < self.layers.len() {
                    if rstd_emitted {
                        gdn_fold_rstd = fold_req.map(|(p, _)| p);
                        normed = summed.clone();
                    } else {
                        normed = self.layers[li + 1].pre_norm.forward(&summed)?;
                    }
                }
                residual = summed;
                t_lap = splits.lap("norms_and_residual", t_lap);
                decode_prof::lap("norms_and_residual");
                continue;
            }
            let (normed_post, residual_after_attn) = match post_pair {
                Some(p) => p,
                None => layer.post_norm.forward_residual(&mixed, &residual)?,
            };
            decode_prof::lap("norms_and_residual");
            let mut w4a8_summed_with_residual = None;
            let mut w4a8_fold_req = None;
            if seq == 1 && !matches!((moe_dispatch, layer.ffn.as_moe()), (Some(_), Some(_))) {
                w4a8_fold_req =
                    self.gdn_prenorm_fold_request_for_next_layer(li, seq, &residual_after_attn)?;
                w4a8_summed_with_residual =
                    fused_dense_mlp_nvfp4_w4a8_one_rowquant_shared_by_gate_up_silu_quant_producer_then_down_with_residual_writeback_decode_m1(
                        &layer.ffn,
                        &normed_post,
                        &residual_after_attn,
                        w4a8_fold_req.as_ref().map(|(p, e)| (p, *e)),
                    )?;
            }
            if let Some((summed, rstd_emitted)) = w4a8_summed_with_residual {
                t_lap = splits.lap(layer.ffn.label(), t_lap);
                decode_prof::lap(layer.ffn.label());
                if li + 1 < self.layers.len() {
                    if rstd_emitted {
                        gdn_fold_rstd = w4a8_fold_req.map(|(p, _)| p);
                        normed = summed.clone();
                    } else {
                        normed = self.layers[li + 1].pre_norm.forward(&summed)?;
                    }
                }
                residual = summed;
            } else {
                let moe_out = match (moe_dispatch, layer.ffn.as_moe()) {
                    (Some(d), Some(moe)) => d.forward(li, moe, &normed_post)?,
                    _ => dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
                        &layer.ffn,
                        &normed_post,
                        seq,
                    )?,
                };
                t_lap = splits.lap(layer.ffn.label(), t_lap);
                decode_prof::lap(layer.ffn.label());
                if li + 1 < self.layers.len() {
                    let mut pre_pair = None;
                    if seq == 1 {
                        pre_pair =
                            rmsnorm_residual_decode_m1_writeout_skipping_the_dtod_residual_copy(
                                &self.layers[li + 1].pre_norm,
                                &moe_out,
                                &residual_after_attn,
                            )?;
                    }
                    let (normed_next, residual_after_moe) = match pre_pair {
                        Some(p) => p,
                        None => self.layers[li + 1]
                            .pre_norm
                            .forward_residual(&moe_out, &residual_after_attn)?,
                    };
                    normed = normed_next;
                    residual = residual_after_moe;
                } else {
                    residual = residual_after_attn.add(&moe_out)?;
                }
            }
            t_lap = splits.lap("norms_and_residual", t_lap);
            decode_prof::lap("norms_and_residual");
            if std::env::var("NV_DEBUG_NAN_SCAN").as_deref() == Ok("1") {
                let v: Vec<f32> = residual
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                let bad = v.iter().filter(|x| !x.is_finite()).count();
                let amax = v
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold(0f32, |m, x| m.max(x.abs()));
                eprintln!(
                    "[nan-scan] layer={li} mixer={} bad={bad} amax={amax:.1}",
                    match &layer.mixer {
                        LayerMixer::Full(_) => "full",
                        _ => "gdn",
                    }
                );
                anyhow::ensure!(
                    bad == 0,
                    "NV_DEBUG_NAN_SCAN first non-finite residual at layer {li}"
                );
            }
        }
        let x = residual;
        let x = match logit_rows {
            Some(rows) => {
                anyhow::ensure!(
                    rows >= 1 && rows <= seq,
                    "Qwen3Moe.forward_with_cache: logit_rows {rows} out of 1..={seq}"
                );
                if rows < seq {
                    x.narrow(1, seq - rows, rows)?
                } else {
                    x
                }
            }
            None => x,
        };
        let x = self.final_norm.forward(&x)?;
        decode_prof::lap("final_norm");
        let logits = self.lm_head.forward(&x)?;
        splits.lap("final_norm_and_lm_head", t_lap);
        decode_prof::lap("lm_head");
        splits.report(seq, logit_rows);
        decode_prof::report_and_end(write_start);
        cache.advance(seq);
        Ok(logits)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_dispatched_hidden(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with_cache_dispatched_hidden_rows(tokens, positions, cache, moe_dispatch, None)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_dispatched_hidden_rows(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
        logit_rows: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let (logits, hidden, aux) = self.forward_with_cache_dispatched_hidden_rows_aux(
            tokens,
            positions,
            cache,
            moe_dispatch,
            logit_rows,
            &[],
        )?;
        debug_assert!(aux.is_empty());
        Ok((logits, hidden))
    }

    #[cfg(feature = "cuda")]
    pub fn aux_layer_ids_are_post_layer_residuals_because_hf_hidden_states_index_l_plus_1_is_the_output_of_layer_l(
        &self,
        target_layer_ids: &[usize],
    ) -> Result<Vec<usize>> {
        for (i, &li) in target_layer_ids.iter().enumerate() {
            anyhow::ensure!(
                li < self.layers.len(),
                "aux layer id {li} is out of range for a {}-layer target; a DFlash/DSpark drafter \
                 trained against a different depth cannot be fed by this trunk",
                self.layers.len()
            );
            anyhow::ensure!(
                i == 0 || li > target_layer_ids[i - 1],
                "aux layer ids must be strictly increasing, got {target_layer_ids:?}; the trunk \
                 collects in trunk order and the drafter's fc weight is laid out in \
                 target_layer_ids order, so an unsorted list silently permutes the fc input"
            );
        }
        Ok(target_layer_ids.to_vec())
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_dispatched_hidden_rows_aux(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        moe_dispatch: Option<&dyn MoeDispatch>,
        logit_rows: Option<usize>,
        aux_layers: &[usize],
    ) -> Result<(Tensor, Tensor, Vec<Tensor>)> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache_dispatched_hidden: tokens must be [1, seq], got {:?}",
                dims
            );
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache_dispatched_hidden: positions shape {:?} != [{}]",
                positions.dims(),
                seq
            );
        }
        let write_start = cache.current_len();
        let new_total = write_start + seq;
        if new_total > cache.max_seq_len() {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache_dispatched_hidden: new_total {} > max_seq_len {}",
                new_total,
                cache.max_seq_len()
            );
        }
        cache.prepare_for_step(write_start, new_total)?;

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        if (2..=SMALL_M_W4A8_VERIFY_MAX_TOKENS).contains(&seq) {
            decode_prof::begin_env_nv_prof_decode_refusing_mid_capture_because_every_lap_syncs(
                tokens.device(),
            );
        }
        let tm =
            trunk_events::begin_if_armed_verify_rows(seq, tokens.device(), self.layers.len());
        tm.head(0);
        let x = match embed_lookup_bf16(&self.embed_weight, &tokens_flat)? {
            Some(rows) => rows,
            None => self.embed_weight.index_select(&tokens_flat, 0)?,
        }
        .reshape((1usize, seq, self.config.hidden_size))?
        .to_dtype(self.dtype)?;

        let tap = verify_trunk_tap::armed_env_opt_in_nv_q38_verify_trunk_tap_1_eager_verify_rows_only_because_every_tap_is_a_dtoh(
            seq,
            tokens.device(),
        );
        let mut residual = x.clone();
        let mut normed = self.layers[0].pre_norm.forward(&x)?;
        decode_prof::lap("embed_and_first_norm");
        tm.head(1);
        verify_trunk_tap::row0(tap, 0, "embed_x", &x);
        verify_trunk_tap::row0(tap, 0, "norm0", &normed);
        let mut aux: Vec<Tensor> = Vec::with_capacity(aux_layers.len());
        for (li, layer) in self.layers.iter().enumerate() {
            let mixed = match &layer.mixer {
                LayerMixer::Full(attn) => {
                    let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!(
                            "forward_with_cache_dispatched_hidden: layer {} FullAttention but no cache slot",
                            li
                        )
                    })?;
                    attn.forward_with_cache(
                        &normed,
                        &self.rope,
                        positions,
                        cache,
                        slot,
                        write_start,
                        new_total,
                    )?
                }
                LayerMixer::Linear(la) => {
                    let lin_slot = cache.lin_attn_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!(
                            "forward_with_cache_dispatched_hidden: layer {} LinearAttention but no cache slot",
                            li
                        )
                    })?;
                    let out = cache.lin_attn_step(lin_slot, la, &normed)?;
                    decode_prof::lap("gdn_chain");
                    out
                }
            };
            tm.layer(li, 0);
            verify_trunk_tap::row0(tap, li, "mixer", &mixed);
            let mut post_pair = None;
            if seq <= SMALL_M_W4A8_VERIFY_MAX_TOKENS {
                post_pair = rmsnorm_residual_decode_m1_writeout_skipping_the_dtod_residual_copy(
                    &layer.post_norm,
                    &mixed,
                    &residual,
                )?;
            }
            let (normed_post, residual_after_attn) = match post_pair {
                Some(p) => p,
                None => layer.post_norm.forward_residual(&mixed, &residual)?,
            };
            decode_prof::lap("norms_and_residual");
            tm.layer(li, 1);
            verify_trunk_tap::row0(tap, li, "postnorm", &normed_post);
            verify_trunk_tap::row0(tap, li, "resid_attn", &residual_after_attn);
            let moe_out = match (moe_dispatch, layer.ffn.as_moe()) {
                (Some(d), Some(moe)) => d.forward(li, moe, &normed_post)?,
                _ => dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
                    &layer.ffn,
                    &normed_post,
                    seq,
                )?,
            };
            decode_prof::lap(layer.ffn.label());
            tm.layer(li, 2);
            verify_trunk_tap::row0(tap, li, "ffn", &moe_out);
            if li + 1 < self.layers.len() {
                let mut pre_pair = None;
                if seq <= SMALL_M_W4A8_VERIFY_MAX_TOKENS {
                    pre_pair = rmsnorm_residual_decode_m1_writeout_skipping_the_dtod_residual_copy(
                        &self.layers[li + 1].pre_norm,
                        &moe_out,
                        &residual_after_attn,
                    )?;
                }
                let (normed_next, residual_after_moe) = match pre_pair {
                    Some(p) => p,
                    None => self.layers[li + 1]
                        .pre_norm
                        .forward_residual(&moe_out, &residual_after_attn)?,
                };
                normed = normed_next;
                residual = residual_after_moe;
            } else {
                residual = residual_after_attn.add(&moe_out)?;
            }
            decode_prof::lap("norms_and_residual");
            tm.layer(li, 3);
            verify_trunk_tap::row0(tap, li, "resid", &residual);
            if aux_layers.contains(&li) {
                aux.push(residual.reshape((seq, self.config.hidden_size))?);
            }
        }
        let hidden = residual;
        let normed_rows = match logit_rows {
            Some(rows) => {
                anyhow::ensure!(
                    rows >= 1 && rows <= seq,
                    "Qwen3Moe.forward_with_cache_dispatched_hidden: logit_rows {rows} out of 1..={seq}"
                );
                if rows < seq {
                    hidden.narrow(1, seq - rows, rows)?
                } else {
                    hidden.clone()
                }
            }
            None => hidden.clone(),
        };
        let x = self.final_norm.forward(&normed_rows)?;
        decode_prof::lap("final_norm");
        tm.tail(0);
        verify_trunk_tap::row0(tap, self.layers.len(), "final_norm", &x);
        let logits = self.lm_head.forward(&x)?;
        decode_prof::lap("lm_head");
        tm.tail(1);
        decode_prof::report_and_end(write_start);
        cache.advance(seq);
        anyhow::ensure!(
            aux.len() == aux_layers.len(),
            "aux tap collected {} of {} requested layers; a duplicate or out-of-range aux layer id \
             would silently shrink the drafter's fc input and mis-slice every projection",
            aux.len(),
            aux_layers.len()
        );
        Ok((logits, hidden, aux))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_hidden(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<(Tensor, Tensor)> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache_hidden: tokens must be [1, seq], got {:?}",
                dims
            );
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache_hidden: positions shape {:?} != [{}]",
                positions.dims(),
                seq
            );
        }
        let write_start = cache.current_len();
        let new_total = write_start + seq;
        if new_total > cache.max_seq_len() {
            anyhow::bail!(
                "Qwen3Moe.forward_with_cache_hidden: new_total {} > max_seq_len {}",
                new_total,
                cache.max_seq_len()
            );
        }
        cache.prepare_for_step(write_start, new_total)?;

        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = match embed_lookup_bf16(&self.embed_weight, &tokens_flat)? {
            Some(rows) => rows,
            None => self.embed_weight.index_select(&tokens_flat, 0)?,
        }
        .reshape((1usize, seq, self.config.hidden_size))?
        .to_dtype(self.dtype)?;

        let mut residual = x.clone();
        let mut normed = self.layers[0].pre_norm.forward(&x)?;
        for (li, layer) in self.layers.iter().enumerate() {
            let mixed = match &layer.mixer {
                LayerMixer::Full(attn) => {
                    let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!(
                            "forward_with_cache_hidden: layer {} FullAttention but no cache slot",
                            li
                        )
                    })?;
                    attn.forward_with_cache(
                        &normed,
                        &self.rope,
                        positions,
                        cache,
                        slot,
                        write_start,
                        new_total,
                    )?
                }
                LayerMixer::Linear(la) => {
                    let lin_slot = cache.lin_attn_slot_for_layer(li).ok_or_else(|| {
                        anyhow::anyhow!(
                            "forward_with_cache_hidden: layer {} LinearAttention but no cache slot",
                            li
                        )
                    })?;
                    cache.lin_attn_step(lin_slot, la, &normed)?
                }
            };
            let (normed_post, residual_after_attn) =
                layer.post_norm.forward_residual(&mixed, &residual)?;
            let moe_out = dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
                &layer.ffn,
                &normed_post,
                seq,
            )?;
            if li + 1 < self.layers.len() {
                let (normed_next, residual_after_moe) = self.layers[li + 1]
                    .pre_norm
                    .forward_residual(&moe_out, &residual_after_attn)?;
                normed = normed_next;
                residual = residual_after_moe;
            } else {
                residual = residual_after_attn.add(&moe_out)?;
            }
        }
        let hidden = residual;
        let x = self.final_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&x)?;
        cache.advance(seq);
        Ok((logits, hidden))
    }
}

#[cfg(feature = "cuda")]
pub mod verify_trunk_tap {
    use candle_core::{DType, Tensor};
    use std::sync::Mutex;

    #[derive(Clone, Debug, PartialEq)]
    pub struct TapRecord {
        pub m: usize,
        pub li: usize,
        pub label: &'static str,
        pub fnv: u64,
        pub maxabs: f32,
    }

    static RECORDS: Mutex<Vec<TapRecord>> = Mutex::new(Vec::new());

    pub fn armed_env_opt_in_nv_q38_verify_trunk_tap_1_eager_verify_rows_only_because_every_tap_is_a_dtoh(
        seq: usize,
        device: &candle_core::Device,
    ) -> Option<usize> {
        if !(2..=super::SMALL_M_W4A8_VERIFY_MAX_TOKENS).contains(&seq) {
            return None;
        }
        if !nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_VERIFY_TRUNK_TAP") {
            return None;
        }
        match device {
            candle_core::Device::Cuda(d)
                if !crate::gemma4::current_stream_is_mid_graph_capture(d) =>
            {
                Some(seq)
            }
            _ => None,
        }
    }

    pub fn row0(armed_m: Option<usize>, li: usize, label: &'static str, t: &Tensor) {
        let Some(m) = armed_m else {
            return;
        };
        match row0_fnv_and_maxabs(t) {
            Ok((fnv, maxabs)) => {
                eprintln!("[trunk-tap] m={m} li={li:02} {label} fnv={fnv:016x} maxabs={maxabs}");
                RECORDS.lock().unwrap_or_else(|p| p.into_inner()).push(
                    TapRecord {
                        m,
                        li,
                        label,
                        fnv,
                        maxabs,
                    },
                );
            }
            Err(e) => eprintln!("[trunk-tap] m={m} li={li:02} {label} tap failed: {e:#}"),
        }
    }

    pub fn take() -> Vec<TapRecord> {
        std::mem::take(&mut RECORDS.lock().unwrap_or_else(|p| p.into_inner()))
    }

    fn row0_fnv_and_maxabs(t: &Tensor) -> anyhow::Result<(u64, f32)> {
        let row: Vec<f32> = t
            .narrow(1, 0, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let mut h = 0xcbf29ce484222325u64;
        let mut maxabs = 0f32;
        for v in &row {
            maxabs = maxabs.max(v.abs());
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        Ok((h, maxabs))
    }
}

#[cfg(feature = "cuda")]
pub mod decode_prof {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    struct ProfState {
        dev: candle_core::CudaDevice,
        acc: BTreeMap<&'static str, f64>,
        laps: BTreeMap<&'static str, u32>,
        last: std::time::Instant,
    }

    thread_local! {
        static STATE: RefCell<Option<ProfState>> = const { RefCell::new(None) };
    }

    use crate::gemma4::current_stream_is_mid_graph_capture;

    pub fn begin_env_nv_prof_decode_refusing_mid_capture_because_every_lap_syncs(
        device: &candle_core::Device,
    ) {
        if !nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_PROF_DECODE") {
            return;
        }
        let candle_core::Device::Cuda(d) = device else {
            return;
        };
        if current_stream_is_mid_graph_capture(d) {
            return;
        }
        let _ = nv_layers::cuda_stream::current_stream(d).synchronize();
        nv_layers::linear_attn::gdn_step_prof::arm_only_while_an_eager_decode_profiler_is_active_because_every_lap_syncs(true);
        STATE.with(|s| {
            *s.borrow_mut() = Some(ProfState {
                dev: d.clone(),
                acc: BTreeMap::new(),
                laps: BTreeMap::new(),
                last: std::time::Instant::now(),
            });
        });
    }

    pub fn lap(label: &'static str) {
        STATE.with(|s| {
            if let Some(st) = s.borrow_mut().as_mut() {
                let _ = nv_layers::cuda_stream::current_stream(&st.dev).synchronize();
                let now = std::time::Instant::now();
                *st.acc.entry(label).or_default() += (now - st.last).as_secs_f64() * 1000.0;
                *st.laps.entry(label).or_default() += 1;
                st.last = now;
            }
        });
    }

    pub fn report_and_end(pos: usize) {
        STATE.with(|s| {
            if let Some(st) = s.borrow_mut().take() {
                let total: f64 = st.acc.values().sum();
                let mut entries: Vec<(&&'static str, &f64)> = st.acc.iter().collect();
                entries
                    .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (label, ms) in &entries {
                    eprintln!(
                        "[prof-decode] pos={pos} {:>22} {:8.3} ms ({:4.1}%) laps={}",
                        label,
                        ms,
                        100.0 * *ms / total.max(1e-12),
                        st.laps.get(*label).copied().unwrap_or(0)
                    );
                }
                eprintln!("[prof-decode] pos={pos} {:>22} {total:8.3} ms", "TOTAL");
                nv_layers::linear_attn::gdn_step_prof::report_and_reset(pos);
                nv_layers::linear_attn::gdn_step_prof::arm_only_while_an_eager_decode_profiler_is_active_because_every_lap_syncs(false);
            }
        });
    }
}

#[cfg(feature = "cuda")]
pub mod trunk_events {
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex, OnceLock};

    pub const HEAD_SLOTS_ENTRY_THEN_EMBED_FIRST_NORM: usize = 2;
    pub const LAYER_SLOTS_MIXER_NORM_FFN_NORM: usize = 4;
    pub const TAIL_SLOTS_FINAL_NORM_THEN_LM_HEAD: usize = 2;

    pub fn armed_env_opt_in_nv_q36_trunk_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes(
    ) -> bool {
        crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_TRUNK_EVENTS")
    }

    fn slot_count(n_layers: usize) -> usize {
        HEAD_SLOTS_ENTRY_THEN_EMBED_FIRST_NORM
            + n_layers * LAYER_SLOTS_MIXER_NORM_FFN_NORM
            + TAIL_SLOTS_FINAL_NORM_THEN_LM_HEAD
    }

    struct Ladder {
        stamps: cudarc::driver::CudaSlice<u64>,
        stream: Arc<cudarc::driver::CudaStream>,
        n_layers: usize,
    }

    static LADDER: OnceLock<Mutex<Option<Ladder>>> = OnceLock::new();

    pub struct RowsMarker(Option<ActiveRowsMarker>);

    struct ActiveRowsMarker {
        stream: Arc<cudarc::driver::CudaStream>,
        n_layers: usize,
    }

    pub fn m1_opt_in_nv_q36_trunk_events_m1_1_ladders_the_hot_decode_graph_whose_perturbation_the_seq_gate_otherwise_spares(
    ) -> bool {
        crate::env_arm_opt_in_eq_some_1_cached_first_read_wins!("NV_Q36_TRUNK_EVENTS_M1")
    }

    pub fn prebuild_if_armed_because_the_ladder_cannot_allocate_mid_capture(
        device: &candle_core::Device,
        n_layers: usize,
    ) {
        if !armed_env_opt_in_nv_q36_trunk_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes()
        {
            return;
        }
        let candle_core::Device::Cuda(d) = device else {
            return;
        };
        if crate::gemma4::current_stream_is_mid_graph_capture(d) {
            return;
        }
        let stream = nv_layers::cuda_stream::current_stream(d);
        let cell = LADDER.get_or_init(|| Mutex::new(None));
        let Ok(mut guard) = cell.lock() else { return };
        let needs_build = match guard.as_ref() {
            Some(l) => l.n_layers != n_layers,
            None => true,
        };
        if needs_build {
            if let Ok(mut stamps) = stream.alloc_zeros::<u64>(slot_count(n_layers)) {
                {
                    use cudarc::driver::DevicePtrMut;
                    let (base, _g) = stamps.device_ptr_mut(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::prof_timestamp(
                            stream.cu_stream() as *mut c_void,
                            base as *mut u64,
                        )
                    };
                    if rc != 0 {
                        eprintln!(
                            "[q36-trunk-events] prebuild launcher warm-up rc={rc}; the runtime                              lazy-init must happen OUTSIDE capture or every captured mark returns 901"
                        );
                    }
                }
                let _ = stream.synchronize();
                *guard = Some(Ladder {
                    stamps,
                    stream: stream.clone(),
                    n_layers,
                });
            }
        }
    }

    pub fn begin_if_armed_verify_rows(
        seq: usize,
        device: &candle_core::Device,
        n_layers: usize,
    ) -> RowsMarker {
        if (seq < 2
            && !m1_opt_in_nv_q36_trunk_events_m1_1_ladders_the_hot_decode_graph_whose_perturbation_the_seq_gate_otherwise_spares())
            || !armed_env_opt_in_nv_q36_trunk_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes()
        {
            static REFUSAL_PRINTS_ONCE_SO_A_SILENTLY_DISARMED_LADDER_NAMES_ITS_GATE: std::sync::Once =
                std::sync::Once::new();
            REFUSAL_PRINTS_ONCE_SO_A_SILENTLY_DISARMED_LADDER_NAMES_ITS_GATE.call_once(|| {
                eprintln!(
                    "[q36-trunk-events] begin refused: seq={seq} m1_opt_in={} armed={}",
                    m1_opt_in_nv_q36_trunk_events_m1_1_ladders_the_hot_decode_graph_whose_perturbation_the_seq_gate_otherwise_spares(),
                    armed_env_opt_in_nv_q36_trunk_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes()
                );
            });
            return RowsMarker(None);
        }
        let candle_core::Device::Cuda(d) = device else {
            return RowsMarker(None);
        };
        let stream = nv_layers::cuda_stream::current_stream(d);
        let cell = LADDER.get_or_init(|| Mutex::new(None));
        {
            let Ok(mut guard) = cell.lock() else {
                return RowsMarker(None);
            };
            let needs_build = match guard.as_ref() {
                Some(l) => l.n_layers != n_layers,
                None => true,
            };
            if needs_build {
                if crate::gemma4::current_stream_is_mid_graph_capture(d) {
                    return RowsMarker(None);
                }
                let Ok(stamps) = stream.alloc_zeros::<u64>(slot_count(n_layers)) else {
                    return RowsMarker(None);
                };
                *guard = Some(Ladder {
                    stamps,
                    stream: stream.clone(),
                    n_layers,
                });
            }
        }
        RowsMarker(Some(ActiveRowsMarker { stream, n_layers }))
    }

    impl RowsMarker {
        fn mark_idx(&self, idx: usize) {
            let Some(active) = &self.0 else {
                return;
            };
            let Some(cell) = LADDER.get() else {
                return;
            };
            let Ok(mut guard) = cell.lock() else {
                return;
            };
            let Some(l) = guard.as_mut() else {
                return;
            };
            if idx >= slot_count(l.n_layers) {
                return;
            }
            use cudarc::driver::DevicePtrMut;
            let (base, _g) = l.stamps.device_ptr_mut(&active.stream);
            let dst = base + (idx * 8) as u64;
            let rc = unsafe {
                nv_kernels::cuda::prof_timestamp(
                    active.stream.cu_stream() as *mut c_void,
                    dst as *mut u64,
                )
            };
            static FIRST_MARK_PRINTS_ONCE_NAMING_ITS_STREAM: std::sync::Once = std::sync::Once::new();
            FIRST_MARK_PRINTS_ONCE_NAMING_ITS_STREAM.call_once(|| {
                eprintln!(
                    "[q36-trunk-events] first mark: rc={rc} stream={:?}",
                    active.stream.cu_stream()
                );
            });
            if rc != 0 {
                eprintln!("[q36-trunk-events] prof_timestamp rc={rc}");
            }
        }

        pub fn head(&self, i: usize) {
            self.mark_idx(i);
        }

        pub fn layer(&self, li: usize, slot: usize) {
            self.mark_idx(HEAD_SLOTS_ENTRY_THEN_EMBED_FIRST_NORM + li * LAYER_SLOTS_MIXER_NORM_FFN_NORM + slot);
        }

        pub fn tail(&self, i: usize) {
            let n_layers = self.0.as_ref().map(|a| a.n_layers).unwrap_or(0);
            self.mark_idx(
                HEAD_SLOTS_ENTRY_THEN_EMBED_FIRST_NORM
                    + n_layers * LAYER_SLOTS_MIXER_NORM_FFN_NORM
                    + i,
            );
        }
    }

    pub fn read_stamps_for_the_isolation_gate(
        device: &candle_core::Device,
    ) -> Option<Vec<u64>> {
        let candle_core::Device::Cuda(d) = device else {
            return None;
        };
        let cell = LADDER.get()?;
        let guard = cell.lock().ok()?;
        let l = guard.as_ref()?;
        let stream = nv_layers::cuda_stream::current_stream(d);
        stream.synchronize().ok()?;
        let mut host = vec![0u64; slot_count(l.n_layers)];
        l.stream.memcpy_dtoh(&l.stamps, &mut host).ok()?;
        Some(host)
    }

    pub fn dump_after_replay_summing_by_layer_kind(layer_is_full_attention: &[bool]) {
        let Some(cell) = LADDER.get() else {
            eprintln!("[q36-trunk-events] no ladder was ever armed; nothing to dump");
            return;
        };
        let stamps_host: Vec<u64> = {
            let Ok(guard) = cell.lock() else {
                eprintln!("[q36-trunk-events] mutex poisoned");
                return;
            };
            let Some(l) = guard.as_ref() else {
                eprintln!("[q36-trunk-events] ladder never allocated; nothing to dump");
                return;
            };
            if l.n_layers != layer_is_full_attention.len() {
                eprintln!(
                    "[q36-trunk-events] ladder n_layers {} != model layers {}",
                    l.n_layers,
                    layer_is_full_attention.len()
                );
                return;
            }
            if let Err(e) = l.stream.synchronize() {
                eprintln!("[q36-trunk-events] sync failed: {e:?}");
                return;
            }
            #[allow(deprecated)]
            match l.stream.memcpy_dtov(&l.stamps) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[q36-trunk-events] readback failed: {e:?}");
                    return;
                }
            }
        };
        if stamps_host.iter().any(|&t| t == 0) {
            eprintln!(
                "[q36-trunk-events] {} of {} stamps are zero; the last forward did not stamp \
                 every slot -- dump refused",
                stamps_host.iter().filter(|&&t| t == 0).count(),
                stamps_host.len()
            );
            return;
        }
        let n_layers = layer_is_full_attention.len();
        let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 / 1e6;
        let embed_ms = ms(stamps_host[0], stamps_host[1]);
        let mut attn_ms = 0f64;
        let mut gdn_ms = 0f64;
        let mut norms_ms = 0f64;
        let mut ffn_ms = 0f64;
        let mut prev = stamps_host[1];
        for (li, &is_full) in layer_is_full_attention.iter().enumerate() {
            let base = HEAD_SLOTS_ENTRY_THEN_EMBED_FIRST_NORM + li * LAYER_SLOTS_MIXER_NORM_FFN_NORM;
            let mixer = ms(prev, stamps_host[base]);
            if is_full {
                attn_ms += mixer;
            } else {
                gdn_ms += mixer;
            }
            norms_ms += ms(stamps_host[base], stamps_host[base + 1]);
            ffn_ms += ms(stamps_host[base + 1], stamps_host[base + 2]);
            norms_ms += ms(stamps_host[base + 2], stamps_host[base + 3]);
            prev = stamps_host[base + 3];
        }
        let tail_base = HEAD_SLOTS_ENTRY_THEN_EMBED_FIRST_NORM + n_layers * LAYER_SLOTS_MIXER_NORM_FFN_NORM;
        let final_norm_ms = ms(prev, stamps_host[tail_base]);
        let lm_head_ms = ms(stamps_host[tail_base], stamps_host[tail_base + 1]);
        let total_ms = ms(stamps_host[0], stamps_host[tail_base + 1]);
        eprintln!(
            "[q36-trunk-events] layers={n_layers} embed_ms={embed_ms:.3} attn_ms={attn_ms:.3} \
             gdn_ms={gdn_ms:.3} norms_ms={norms_ms:.3} ffn_ms={ffn_ms:.3} \
             final_norm_ms={final_norm_ms:.3} lm_head_ms={lm_head_ms:.3} trunk_total_ms={total_ms:.3}"
        );
    }
}

#[cfg(feature = "cuda")]
struct PrefillWallSplits {
    dev: Option<candle_core::CudaDevice>,
    acc: std::collections::BTreeMap<&'static str, f64>,
}

#[cfg(feature = "cuda")]
impl PrefillWallSplits {
    fn begin_env_nv_prof_prefill(device: &Device) -> Self {
        let on = crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_PROF_PREFILL");
        let dev = match (on, device) {
            (true, Device::Cuda(d)) => Some(d.clone()),
            _ => None,
        };
        Self {
            dev,
            acc: std::collections::BTreeMap::new(),
        }
    }

    fn lap(&mut self, label: &'static str, t0: std::time::Instant) -> std::time::Instant {
        if let Some(d) = &self.dev {
            let _ = d.cuda_stream().synchronize();
            *self.acc.entry(label).or_default() += t0.elapsed().as_secs_f64() * 1000.0;
        }
        std::time::Instant::now()
    }

    fn report(&self, seq: usize, logit_rows: Option<usize>) {
        if self.dev.is_none() {
            return;
        }
        let total: f64 = self.acc.values().sum();
        let mut entries: Vec<(&&'static str, &f64)> = self.acc.iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (label, ms) in &entries {
            eprintln!(
                "[prof-prefill] seq={seq} logit_rows={logit_rows:?} {:>22} {:8.2} ms ({:.1}%)",
                label,
                ms,
                100.0 * *ms / total.max(1e-12)
            );
        }
        eprintln!("[prof-prefill] seq={seq} logit_rows={logit_rows:?} {:>22} {total:8.2} ms", "TOTAL");
    }
}

#[cfg(feature = "cuda")]
pub const MAX_VERIFY_MOE_TOKENS: usize = 32;

#[cfg(feature = "cuda")]
pub struct GroupedMoeDispatch {
    weights: Vec<Arc<nv_layers::moe_grouped::MoeGroupedWeights>>,
    ctx: Mutex<nv_layers::moe_grouped::GroupedDecodeContext>,

    verify_ctx:
        Mutex<std::collections::HashMap<usize, nv_layers::moe_grouped::GroupedDecodeContext>>,

    verify_chain_ctx:
        Mutex<std::collections::HashMap<usize, nv_layers::moe_grouped::GroupedDecodeContext>>,
    hidden_size: usize,
    moe_intermediate_size: usize,
    num_experts_per_tok: usize,
    num_experts: usize,
    max_verify_tokens: usize,
    fused_chain_m1: bool,
    verify_chain_small_m: bool,
    shared_folded: bool,
    router_and_sg_weights_kept_contiguous_bf16_for_the_router_gemv_arm: Option<Vec<(Tensor, Tensor)>>,
    phase_ladder: Option<MoeChainPhaseLadder>,
    ladder_last_stamp_path_0_m1_1_generic_verify_2_fused_verify: std::sync::atomic::AtomicU8,
}

#[cfg(feature = "cuda")]
const MOE_CHAIN_PHASE_SLOTS_ENTRY_GATE_SHARED_GROUPED_TAIL: usize = 5;

#[cfg(feature = "cuda")]
struct MoeChainPhaseLadder {
    stamps: Mutex<cudarc::driver::CudaSlice<u64>>,
    stream: Arc<cudarc::driver::CudaStream>,
    n_layers: usize,
}

#[cfg(feature = "cuda")]
impl MoeChainPhaseLadder {
    fn mark(
        &self,
        layer_idx: usize,
        slot: usize,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        use cudarc::driver::DevicePtrMut;
        anyhow::ensure!(
            layer_idx < self.n_layers && slot < MOE_CHAIN_PHASE_SLOTS_ENTRY_GATE_SHARED_GROUPED_TAIL,
            "phase ladder mark out of range: layer {layer_idx} slot {slot}"
        );
        let mut stamps = self
            .stamps
            .lock()
            .map_err(|e| anyhow::anyhow!("phase ladder mutex poisoned: {e}"))?;
        let (base, _g) = stamps.device_ptr_mut(stream);
        let dst = base
            + ((layer_idx * MOE_CHAIN_PHASE_SLOTS_ENTRY_GATE_SHARED_GROUPED_TAIL + slot) * 8)
                as u64;
        let rc = unsafe {
            nv_kernels::cuda::prof_timestamp(stream.cu_stream() as *mut c_void, dst as *mut u64)
        };
        anyhow::ensure!(rc == 0, "phase ladder prof_timestamp rc={rc}");
        Ok(())
    }

    fn dump(&self, arm_label: &str) {
        let stamps_host: Vec<u64> = {
            let stamps = match self.stamps.lock() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("[q36-moe-events] mutex poisoned: {e}");
                    return;
                }
            };
            if let Err(e) = self.stream.synchronize() {
                eprintln!("[q36-moe-events] sync failed: {e:?}");
                return;
            }
            #[allow(deprecated)]
            match self.stream.memcpy_dtov(&*stamps) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[q36-moe-events] readback failed: {e:?}");
                    return;
                }
            }
        };
        let slots = MOE_CHAIN_PHASE_SLOTS_ENTRY_GATE_SHARED_GROUPED_TAIL;
        let mut phase_ns = [0u64; MOE_CHAIN_PHASE_SLOTS_ENTRY_GATE_SHARED_GROUPED_TAIL - 1];
        let mut counted_layers = 0usize;
        for li in 0..self.n_layers {
            let row = &stamps_host[li * slots..(li + 1) * slots];
            if row.iter().any(|&t| t == 0) {
                continue;
            }
            counted_layers += 1;
            for p in 0..slots - 1 {
                phase_ns[p] += row[p + 1].saturating_sub(row[p]);
            }
        }
        let total_ms: f64 = phase_ns.iter().map(|&ns| ns as f64 / 1e6).sum();
        eprintln!(
            "[q36-moe-events] arm={arm_label} layers_counted={counted_layers} p0_ms={:.3} p1_ms={:.3} p2_ms={:.3} p3_ms={:.3} moe_total_ms={total_ms:.3}",
            phase_ns[0] as f64 / 1e6,
            phase_ns[1] as f64 / 1e6,
            phase_ns[2] as f64 / 1e6,
            phase_ns[3] as f64 / 1e6,
        );
    }
}

#[cfg(feature = "cuda")]
impl GroupedMoeDispatch {
    pub fn from_model(model: &Qwen3Moe) -> Result<Self> {
        let device = model.device().clone();
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("GroupedMoeDispatch requires a CUDA device"),
        };
        let cfg = model.config();
        let fused_chain_m1 =
            nv_q36_moe_fused_chain_env_opt_in_nv_q36_moe_fused_chain_1_single_kernel_route_gather_quant_and_scatter_shared_tail_for_m1_decode();
        let verify_chain_small_m =
            nv_q36_moe_verify_chain_env_opt_in_nv_q36_moe_verify_chain_1_the_m1_fused_chain_arms_extended_to_m_le_8_verify_rows();
        crate::host_serving_util::env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_Q36_MOE_VERIFY_CHAIN",
            verify_chain_small_m,
            "NV_Q36_MOE_FUSED_CHAIN",
            fused_chain_m1,
            "the verify chain is the m1 fused-chain arm stack applied at m<=8 rows and \
             measures nothing without it",
        )?;
        let shared_folded =
            nv_q36_moe_shared_fold_env_opt_in_nv_q36_moe_shared_fold_1_shared_expert_becomes_grouped_tile_k_with_sigmoid_gate_applied_at_scatter();
        crate::host_serving_util::env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_Q36_MOE_SHARED_FOLD",
            shared_folded,
            "NV_Q36_MOE_FUSED_CHAIN",
            fused_chain_m1,
            "the unfused m1 path would scatter the folded shared tile with weight 1.0 instead \
             of its sigmoid gate",
        )?;
        let stream = dev.cuda_stream();
        let mut weights = Vec::with_capacity(model.layers.len());
        for (i, layer) in model.layers.iter().enumerate() {
            let moe = layer.ffn.as_moe().ok_or_else(|| {
                anyhow::anyhow!(
                    "GroupedMoeDispatch: layer {i} has a dense MLP, not a routed MoE block"
                )
            })?;
            let w = moe
                .grouped_weights_built(&device)
                .with_context(|| format!("GroupedMoeDispatch: layer {i} grouped MoE weights"))?;
            if shared_folded {
                let folded = nv_layers::moe_grouped::MoeGroupedWeights::append_shared_expert_as_tile_e_costs_one_extra_expert_copy_on_top_of_the_cached_base(
                    &w,
                    moe.shared_expert(),
                    moe.config().shared_expert_intermediate_size,
                    &stream,
                )
                .with_context(|| format!("GroupedMoeDispatch: layer {i} shared fold"))?;
                weights.push(Arc::new(folded));
            } else {
                weights.push(w);
            }
        }
        let router_gemv_small_m =
            nv_q36_moe_router_gemv_env_opt_in_nv_q36_moe_router_gemv_1_router_plus_shared_gate_plus_x_stage_fused_to_one_kernel_at_m_le_8();
        crate::host_serving_util::env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_Q36_MOE_ROUTER_GEMV",
            router_gemv_small_m,
            "NV_Q36_MOE_FUSED_CHAIN",
            fused_chain_m1,
            "the router gemv arm replaces the fused chain's gate GEMM, logits cast, x dtod \
             and shared-gate gemv; outside the chain nothing reads its outputs",
        )?;
        let router_and_sg_weights = if router_gemv_small_m {
            let mut v: Vec<(Tensor, Tensor)> = Vec::with_capacity(model.layers.len());
            for (i, layer) in model.layers.iter().enumerate() {
                let moe = layer.ffn.as_moe().ok_or_else(|| {
                    anyhow::anyhow!("router gemv arm: layer {i} has no MoE block")
                })?;
                anyhow::ensure!(
                    moe.gate().bias().is_none() && moe.shared_expert_gate().bias().is_none(),
                    "router gemv arm: layer {i} router/shared-gate carries a bias the fused \
                     kernel does not apply"
                );
                let gw = moe
                    .gate()
                    .weight()
                    .ok_or_else(|| {
                        anyhow::anyhow!("router gemv arm: layer {i} router weight is not bf16-resident")
                    })?
                    .contiguous()?;
                anyhow::ensure!(
                    gw.dtype() == DType::BF16
                        && gw.dims() == [cfg.num_experts, cfg.hidden_size],
                    "router gemv arm: layer {i} router weight {:?} {:?}",
                    gw.dims(),
                    gw.dtype()
                );
                let sgw = moe
                    .shared_expert_gate()
                    .weight()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "router gemv arm: layer {i} shared gate weight is not bf16-resident"
                        )
                    })?
                    .contiguous()?;
                anyhow::ensure!(
                    sgw.dtype() == DType::BF16 && sgw.elem_count() == cfg.hidden_size,
                    "router gemv arm: layer {i} shared gate weight {:?} {:?}",
                    sgw.dims(),
                    sgw.dtype()
                );
                v.push((gw, sgw));
            }
            Some(v)
        } else {
            None
        };
        let ctx = if shared_folded {
            nv_layers::moe_grouped::GroupedDecodeContext::new_folded_shared(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                1,
                &stream,
            )?
        } else {
            nv_layers::moe_grouped::GroupedDecodeContext::new(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                &stream,
            )?
        };
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch: post-build sync: {e:?}"))?;
        let max_verify_tokens = MAX_VERIFY_MOE_TOKENS.min(256 / cfg.num_experts_per_tok.max(1));
        crate::host_serving_util::env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_Q36_MOE_FUSED_CHAIN",
            fused_chain_m1,
            "NV_MOE_ROUTE_FUSE",
            nv_layers::moe_grouped::route_fuse_enabled(),
            "without it the shared GroupedDecodeContext chain runs the unfused route and the \
             arm measures nothing",
        )?;
        let phase_ladder =
            if nv_q36_moe_events_env_opt_in_nv_q36_moe_events_1_in_capture_globaltimer_ladder_perturbs_by_its_own_nodes() {
                let stamps = stream.alloc_zeros::<u64>(
                    model.layers.len() * MOE_CHAIN_PHASE_SLOTS_ENTRY_GATE_SHARED_GROUPED_TAIL,
                )?;
                Some(MoeChainPhaseLadder {
                    stamps: Mutex::new(stamps),
                    stream: stream.clone(),
                    n_layers: model.layers.len(),
                })
            } else {
                None
            };
        Ok(Self {
            weights,
            ctx: Mutex::new(ctx),
            verify_ctx: Mutex::new(std::collections::HashMap::new()),
            verify_chain_ctx: Mutex::new(std::collections::HashMap::new()),
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            num_experts_per_tok: cfg.num_experts_per_tok,
            num_experts: cfg.num_experts,
            max_verify_tokens,
            fused_chain_m1,
            verify_chain_small_m,
            shared_folded,
            router_and_sg_weights_kept_contiguous_bf16_for_the_router_gemv_arm:
                router_and_sg_weights,
            phase_ladder,
            ladder_last_stamp_path_0_m1_1_generic_verify_2_fused_verify:
                std::sync::atomic::AtomicU8::new(0),
        })
    }

    pub fn ab_toggle_verify_chain_small_m_for_parity_tests(&mut self, on: bool) {
        self.verify_chain_small_m = on;
    }

    pub fn ab_swap_router_gemv_weights_for_parity_tests(
        &mut self,
        w: Option<Vec<(Tensor, Tensor)>>,
    ) -> Option<Vec<(Tensor, Tensor)>> {
        std::mem::replace(
            &mut self.router_and_sg_weights_kept_contiguous_bf16_for_the_router_gemv_arm,
            w,
        )
    }

    fn forward_verify(
        &self,
        layer_idx: usize,
        moe: &MoeBlock,
        x_flat: &Tensor,
        n_tokens: usize,
    ) -> Result<Option<Tensor>> {
        if n_tokens > self.max_verify_tokens {
            return Ok(None);
        }
        let device = x_flat.device().clone();
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let w = self.weights.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "GroupedMoeDispatch: layer {} out of range ({} layers)",
                layer_idx,
                self.weights.len()
            )
        })?;
        let nan_scan = std::env::var("NV_DEBUG_NAN_SCAN").as_deref() == Ok("1");
        let scan = |label: &str, t: &Tensor| -> Result<()> {
            if !nan_scan {
                return Ok(());
            }
            let v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            if bad > 0 {
                let amax = v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs()));
                anyhow::bail!(
                    "NV_DEBUG_NAN_SCAN first non-finite at layer {layer_idx} stage {label}: \
                     {bad}/{} non-finite, finite_amax={amax}",
                    v.len()
                );
            }
            Ok(())
        };
        let mark_stream = nv_layers::cuda_stream::current_stream(&dev);
        self.ladder_last_stamp_path_0_m1_1_generic_verify_2_fused_verify
            .store(1, std::sync::atomic::Ordering::Relaxed);
        self.mark(layer_idx, 0, &mark_stream);
        scan("input", x_flat)?;
        let logits = moe.gate().forward(x_flat)?;
        scan("router_logits", &logits)?;
        self.mark(layer_idx, 1, &mark_stream);
        let routed = {
            let mut map = self
                .verify_ctx
                .lock()
                .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch verify_ctx poisoned: {e}"))?;
            let ctx = match map.entry(n_tokens) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(v) => {
                    let stream = nv_layers::cuda_stream::current_stream(&dev);
                    v.insert(nv_layers::moe_grouped::GroupedDecodeContext::new_multi(
                        self.hidden_size,
                        self.moe_intermediate_size,
                        self.num_experts_per_tok,
                        self.num_experts,
                        n_tokens,
                        &stream,
                    )?)
                }
            };
            nv_layers::moe_grouped::forward_grouped_decode(
                w, ctx, x_flat, &logits, None, 0, 0.0, false, 1.0, &device,
            )?
        };
        self.mark(layer_idx, 2, &mark_stream);
        scan("routed", &routed)?;
        let shared = moe.shared_contribution_device(x_flat)?;
        self.mark(layer_idx, 3, &mark_stream);
        scan("shared", &shared)?;
        let y = routed.add(&shared)?;
        self.mark(layer_idx, 4, &mark_stream);
        Ok(Some(y))
    }

    fn mark(&self, layer_idx: usize, slot: usize, stream: &Arc<cudarc::driver::CudaStream>) {
        if let Some(ladder) = &self.phase_ladder {
            if let Err(e) = ladder.mark(layer_idx, slot, stream) {
                eprintln!("[q36-moe-events] mark failed: {e:#}");
            }
        }
    }

    fn forward_fused_chain_m1(
        &self,
        layer_idx: usize,
        moe: &MoeBlock,
        x: &Tensor,
    ) -> Result<Tensor> {
        let mut ctx = self
            .ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch ctx mutex poisoned: {e}"))?;
        anyhow::ensure!(
            ctx.n_tokens() == 1,
            "fused chain m1 needs the single-token GroupedDecodeContext"
        );
        self.forward_fused_chain_rows(layer_idx, moe, x, &mut ctx)
    }

    fn forward_fused_chain_verify_small_m(
        &self,
        layer_idx: usize,
        moe: &MoeBlock,
        x: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        let dev = match x.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("GroupedMoeDispatch fused chain requires a CUDA device"),
        };
        let mut map = self
            .verify_chain_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch verify_chain_ctx poisoned: {e}"))?;
        let ctx = match map.entry(n_tokens) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let stream = nv_layers::cuda_stream::current_stream(&dev);
                let built = if self.shared_folded {
                    nv_layers::moe_grouped::GroupedDecodeContext::new_folded_shared(
                        self.hidden_size,
                        self.moe_intermediate_size,
                        self.num_experts_per_tok,
                        self.num_experts,
                        n_tokens,
                        &stream,
                    )?
                } else {
                    nv_layers::moe_grouped::GroupedDecodeContext::new_multi(
                        self.hidden_size,
                        self.moe_intermediate_size,
                        self.num_experts_per_tok,
                        self.num_experts,
                        n_tokens,
                        &stream,
                    )?
                };
                v.insert(built)
            }
        };
        self.forward_fused_chain_rows(layer_idx, moe, x, ctx)
    }

    fn forward_fused_chain_rows(
        &self,
        layer_idx: usize,
        moe: &MoeBlock,
        x: &Tensor,
        ctx: &mut nv_layers::moe_grouped::GroupedDecodeContext,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let in_dims = x.dims().to_vec();
        let hidden = *in_dims.last().unwrap();
        let n_tokens = ctx.n_tokens();
        anyhow::ensure!(
            hidden == self.hidden_size,
            "fused chain: hidden {hidden} != {}",
            self.hidden_size
        );
        anyhow::ensure!(
            x.dtype() == DType::BF16,
            "fused chain expects BF16 activations, got {:?}",
            x.dtype()
        );
        anyhow::ensure!(
            in_dims[..in_dims.len() - 1].iter().product::<usize>() == n_tokens,
            "fused chain: x dims {in_dims:?} do not carry the context's {n_tokens} rows"
        );
        let device = x.device().clone();
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("GroupedMoeDispatch fused chain requires a CUDA device"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let w = self.weights.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "GroupedMoeDispatch: layer {} out of range ({} layers)",
                layer_idx,
                self.weights.len()
            )
        })?;
        self.ladder_last_stamp_path_0_m1_1_generic_verify_2_fused_verify
            .store(
                if n_tokens == 1 { 0 } else { 2 },
                std::sync::atomic::Ordering::Relaxed,
            );
        self.mark(layer_idx, 0, &stream);
        let x_flat = x.reshape((n_tokens, hidden))?.contiguous()?;
        anyhow::ensure!(
            ctx.folds_shared() == self.shared_folded,
            "fused chain needs a GroupedDecodeContext whose fold state matches \
             NV_Q36_MOE_SHARED_FOLD"
        );
        let shared_or_gate: Option<Tensor> = if let Some(wv) =
            &self.router_and_sg_weights_kept_contiguous_bf16_for_the_router_gemv_arm
        {
            let (gw, sgw) = wv.get(layer_idx).ok_or_else(|| {
                anyhow::anyhow!("router gemv arm: layer {layer_idx} weight slot missing")
            })?;
            {
                let (xs, xl) = x_flat.storage_and_layout();
                let x_cuda = match &*xs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("fused chain: x must be CUDA"),
                };
                let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
                let (xp, _g1) = x_slice.device_ptr(&stream);
                let xp = xp + (xl.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
                let (gs, gl) = gw.storage_and_layout();
                let g_cuda = match &*gs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("router gemv arm: router weight must be CUDA"),
                };
                let g_slice = g_cuda.as_cuda_slice::<half::bf16>()?;
                let (gp, _g2) = g_slice.device_ptr(&stream);
                let gp = gp + (gl.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
                let (ss, sl) = sgw.storage_and_layout();
                let s_cuda = match &*ss {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("router gemv arm: shared gate weight must be CUDA"),
                };
                let s_slice = s_cuda.as_cuda_slice::<half::bf16>()?;
                let (sp, _g3) = s_slice.device_ptr(&stream);
                let sp = sp + (sl.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
                ctx.fused_router_gemv_sg_and_x_stage_replacing_gate_gemm_cast_and_dtod(
                    &stream,
                    xp,
                    gp,
                    if self.shared_folded { Some(sp) } else { None },
                )?;
            }
            self.mark(layer_idx, 1, &stream);
            let out = if self.shared_folded {
                None
            } else {
                let shared = moe.shared_contribution_device(&x_flat)?;
                anyhow::ensure!(
                    shared.dtype() == DType::F32 && shared.elem_count() == n_tokens * hidden,
                    "fused chain: unexpected shared contribution {:?} {:?}",
                    shared.dims(),
                    shared.dtype()
                );
                Some(shared)
            };
            self.mark(layer_idx, 2, &stream);
            out
        } else {
            let logits = moe.gate().forward(&x_flat)?;
            anyhow::ensure!(
                logits.dtype() == DType::BF16
                    && logits.elem_count() == n_tokens * self.num_experts,
                "fused chain: unexpected router logits {:?} {:?}",
                logits.dims(),
                logits.dtype()
            );
            {
                let (ls, ll) = logits.storage_and_layout();
                let l_cuda = match &*ls {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("fused chain: router logits must be CUDA"),
                };
                let l_slice = l_cuda.as_cuda_slice::<half::bf16>()?;
                let (lp, _g1) = l_slice.device_ptr(&stream);
                let lp = lp + (ll.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
                ctx.cast_router_logits_bf16_into_logits_f32_softcap_applied_at_route_not_here(
                    &stream, lp,
                )?;
            }
            {
                let (xs, xl) = x_flat.storage_and_layout();
                let x_cuda = match &*xs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("fused chain: x must be CUDA"),
                };
                let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
                let (xp, _g1) = x_slice.device_ptr(&stream);
                let xp = xp + (xl.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
                ctx.stage_x_bf16_rows_dtod(&stream, xp)?;
            }
            self.mark(layer_idx, 1, &stream);
            let out = if self.shared_folded {
                let sg_logit = moe.shared_expert_gate().forward(&x_flat)?;
                anyhow::ensure!(
                    sg_logit.dtype() == DType::BF16 && sg_logit.elem_count() == n_tokens,
                    "fused chain: unexpected shared gate logit {:?} {:?}",
                    sg_logit.dims(),
                    sg_logit.dtype()
                );
                Some(sg_logit)
            } else {
                let shared = moe.shared_contribution_device(&x_flat)?;
                anyhow::ensure!(
                    shared.dtype() == DType::F32 && shared.elem_count() == n_tokens * hidden,
                    "fused chain: unexpected shared contribution {:?} {:?}",
                    shared.dims(),
                    shared.dtype()
                );
                Some(shared)
            };
            self.mark(layer_idx, 2, &stream);
            out
        };
        nv_layers::moe_grouped::forward_grouped_decode_into(
            w, ctx, 0, 0, 0.0, false, 1.0, false, &stream, None, None,
        )?;
        self.mark(layer_idx, 3, &stream);
        let mut out_slice: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc(n_tokens * hidden) }
                .map_err(|e| anyhow::anyhow!("fused chain: out alloc: {e:?}"))?;
        if self.shared_folded {
            match &shared_or_gate {
                None => {
                    let (op, _g) = out_slice.device_ptr_mut(&stream);
                    ctx.unpermute_scatter_tail_reading_the_sg_slot_the_router_gemv_arm_wrote(
                        &stream, op,
                    )?;
                }
                Some(sg_logit) => {
                    let (gs, gl) = sg_logit.storage_and_layout();
                    let g_cuda = match &*gs {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("fused chain: shared gate logit must be CUDA"),
                    };
                    let g_slice = g_cuda.as_cuda_slice::<half::bf16>()?;
                    let (gp, _g1) = g_slice.device_ptr(&stream);
                    let gp = gp + (gl.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
                    let (op, _g2) = out_slice.device_ptr_mut(&stream);
                    ctx.unpermute_scatter_tail_weighting_the_folded_shared_tile_by_sigmoid_of_its_gate_logit(
                        &stream, gp, op,
                    )?;
                }
            }
        } else {
            let shared_or_gate = shared_or_gate
                .as_ref()
                .expect("the non-fold chain always computes the dense shared contribution");
            let (ss, sl) = shared_or_gate.storage_and_layout();
            let s_cuda = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("fused chain: shared must be CUDA"),
            };
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let (sp, _g1) = s_slice.device_ptr(&stream);
            let sp = sp + (sl.start_offset() * std::mem::size_of::<f32>()) as u64;
            let (rp, _g2) = ctx.resid_in.device_ptr(&stream);
            let (op, _g3) = out_slice.device_ptr_mut(&stream);
            ctx.unpermute_scatter_tail_adding_shared_f32_then_resid_bf16(&stream, sp, rp, op)?;
        }
        self.mark(layer_idx, 4, &stream);
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out_slice, dev);
        let out = Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (n_tokens, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(hidden);
        Ok(out.reshape(out_dims)?)
    }
}

#[cfg(feature = "cuda")]
impl MoeDispatch for GroupedMoeDispatch {
    fn forward(&self, layer_idx: usize, moe: &MoeBlock, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims().to_vec();
        let hidden = *in_dims.last().unwrap();
        let n_tokens: usize = in_dims[..in_dims.len() - 1].iter().product();
        if n_tokens != 1 {
            let in_dtype = x.dtype();
            if self.verify_chain_small_m
                && n_tokens <= self.max_verify_tokens
                && n_tokens
                    <= nv_layers::moe_grouped::ROUTE_FUSE_ONE_CTA_PER_TOKEN_HOLDS_TO_8_VERIFY_ROWS
                && in_dtype == DType::BF16
                && matches!(x.device(), Device::Cuda(_))
            {
                return self.forward_fused_chain_verify_small_m(layer_idx, moe, x, n_tokens);
            }
            let x_flat = x.reshape((n_tokens, hidden))?.contiguous()?;
            match self.forward_verify(layer_idx, moe, &x_flat, n_tokens)? {
                Some(y) => {
                    let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
                    out_dims.push(hidden);
                    return Ok(y.reshape(out_dims)?.to_dtype(in_dtype)?);
                }
                None => return moe.forward(x),
            }
        }
        if self.fused_chain_m1 {
            return self.forward_fused_chain_m1(layer_idx, moe, x);
        }
        let w = self.weights.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "GroupedMoeDispatch: layer {} out of range ({} layers)",
                layer_idx,
                self.weights.len()
            )
        })?;
        let in_dtype = x.dtype();
        let device = x.device().clone();
        let mark_stream = match &device {
            Device::Cuda(d) => Some(nv_layers::cuda_stream::current_stream(d)),
            _ => None,
        };
        let mark = |slot: usize| {
            if let Some(s) = &mark_stream {
                self.mark(layer_idx, slot, s);
            }
        };
        self.ladder_last_stamp_path_0_m1_1_generic_verify_2_fused_verify
            .store(0, std::sync::atomic::Ordering::Relaxed);
        mark(0);
        let x_flat = x.reshape((1usize, hidden))?.contiguous()?;
        let logits = moe.gate().forward(&x_flat)?;
        mark(1);
        let routed = {
            let mut ctx = self
                .ctx
                .lock()
                .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch ctx mutex poisoned: {e}"))?;
            nv_layers::moe_grouped::forward_grouped_decode(
                w, &mut ctx, &x_flat, &logits, None, 0, 0.0, false, 1.0, &device,
            )?
        };
        mark(2);
        let shared = moe.shared_contribution_device(&x_flat)?;
        mark(3);
        let y = routed.add(&shared)?;
        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(hidden);
        let out = y.reshape(out_dims)?.to_dtype(in_dtype)?;
        mark(4);
        Ok(out)
    }

    fn dump_phase_ladder_after_replay_when_nv_q36_moe_events_1(&self) {
        if let Some(ladder) = &self.phase_ladder {
            match self
                .ladder_last_stamp_path_0_m1_1_generic_verify_2_fused_verify
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                1 => {
                    ladder.dump("verify(p0=gate_logits,p1=grouped_decode,p2=shared,p3=add_cast)");
                    return;
                }
                2 => {
                    ladder.dump(
                        "verify_chain(p0=gate_logits_xcopy,p1=shared,p2=grouped_route_gemms,p3=scatter_tail)",
                    );
                    return;
                }
                _ => {}
            }
            let arm = if self.fused_chain_m1 && self.shared_folded {
                "fused_fold(p0=gate_logits_xcopy,p1=shared_gate_gemv,p2=grouped_route_gemms_9tile,p3=sig_scatter_tail)"
            } else if self.fused_chain_m1 {
                "fused_chain(p0=gate_logits_xcopy,p1=shared,p2=grouped_route_gemms,p3=scatter_tail)"
            } else {
                "baseline(p0=gate_logits,p1=grouped_decode,p2=shared,p3=add_cast)"
            };
            ladder.dump(arm);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3_5DenseConfig {
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
    pub partial_rotary_factor: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub layer_types: Vec<LayerType>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub attn_output_gate: bool,
    pub tie_word_embeddings: bool,
}

pub const MOE_ONLY_KEYS: [&str; 4] = [
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
];

impl Qwen3_5DenseConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).context("parse qwen3_5 dense config json")?;
        let text = v
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("missing text_config"))?;
        for key in MOE_ONLY_KEYS {
            anyhow::ensure!(
                text.get(key).is_none(),
                "config declares {key}: this is a MoE checkpoint, use Qwen3MoeConfig"
            );
        }
        let get_u = |k: &str| -> Result<usize> {
            text.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let get_f = |k: &str| -> Result<f64> {
            text.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let layer_types_raw = text
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing layer_types"))?;
        let layer_types: Vec<LayerType> = layer_types_raw
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(LayerType::LinearAttention),
                Some("full_attention") => Ok(LayerType::FullAttention),
                other => Err(anyhow::anyhow!("unknown layer type {:?}", other)),
            })
            .collect::<Result<Vec<_>>>()?;
        let rope_params = text
            .get("rope_parameters")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let rope_theta = rope_params
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("rope_theta").and_then(|x| x.as_f64()))
            .unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rope_params
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("partial_rotary_factor").and_then(|x| x.as_f64()))
            .unwrap_or(1.0) as f32;
        let eos = scoped_token_id([text, &v], "eos_token_id")
            .ok_or_else(|| anyhow::anyhow!("missing eos_token_id"))?;
        let tie = scoped_bool([text, &v], "tie_word_embeddings").unwrap_or(false);
        let intermediate_size = get_u("intermediate_size")?;
        anyhow::ensure!(intermediate_size > 0, "intermediate_size must be > 0");
        let num_hidden_layers = get_u("num_hidden_layers")?;
        anyhow::ensure!(
            layer_types.len() == num_hidden_layers,
            "layer_types has {} entries but num_hidden_layers is {}",
            layer_types.len(),
            num_hidden_layers
        );
        Ok(Self {
            hidden_size: get_u("hidden_size")?,
            num_hidden_layers,
            num_attention_heads: get_u("num_attention_heads")?,
            num_key_value_heads: get_u("num_key_value_heads")?,
            head_dim: get_u("head_dim")?,
            intermediate_size,
            vocab_size: get_u("vocab_size")?,
            max_position_embeddings: get_u("max_position_embeddings")?,
            rope_theta,
            rms_norm_eps: get_f("rms_norm_eps")?,
            partial_rotary_factor,
            bos_token_id: scoped_token_id([text, &v], "bos_token_id"),
            eos_token_id: eos,
            layer_types,
            linear_num_key_heads: get_u("linear_num_key_heads")?,
            linear_num_value_heads: get_u("linear_num_value_heads")?,
            linear_key_head_dim: get_u("linear_key_head_dim")?,
            linear_value_head_dim: get_u("linear_value_head_dim")?,
            linear_conv_kernel_dim: get_u("linear_conv_kernel_dim")?,
            attn_output_gate: text
                .get("attn_output_gate")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            tie_word_embeddings: tie,
        })
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as usize
    }

    pub fn trunk(&self) -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            partial_rotary_factor: self.partial_rotary_factor,
            bos_token_id: self.bos_token_id.unwrap_or(0),
            eos_token_id: self.eos_token_id,
            layer_types: self.layer_types.clone(),
            linear_num_key_heads: self.linear_num_key_heads,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            attn_output_gate: self.attn_output_gate,
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }
}

fn dump_stats(label: &str, t: &Tensor) -> Result<()> {
    let v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let absmax = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let first8: Vec<f32> = v.iter().take(8).copied().collect();
    eprintln!(
        "[trace] {:>14}: shape={:?} mean={:+.5e} std={:+.5e} absmax={:+.5e} first8={:?}",
        label,
        t.dims(),
        mean,
        std,
        absmax,
        first8
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn build_layer(
    config: &Qwen3MoeConfig,
    dense_intermediate: Option<usize>,
    idx: usize,
    weights: &WeightLoader,
    qconfig: &QuantizationConfig,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    fp8_runner: &Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    device: &Device,
    dtype: DType,
) -> Result<Qwen3MoeLayer> {
    let prefix = format!("model.language_model.layers.{idx}");
    let alt_prefix = format!("model.layers.{idx}");
    let pre_norm = load_rmsnorm_plus_one(
        weights,
        &[
            &format!("{prefix}.input_layernorm.weight"),
            &format!("{alt_prefix}.input_layernorm.weight"),
        ],
        config.hidden_size,
        config.rms_norm_eps,
        dtype,
    )?;
    let post_norm = load_rmsnorm_plus_one(
        weights,
        &[
            &format!("{prefix}.post_attention_layernorm.weight"),
            &format!("{alt_prefix}.post_attention_layernorm.weight"),
        ],
        config.hidden_size,
        config.rms_norm_eps,
        dtype,
    )?;

    let mixer = match config.layer_types[idx] {
        LayerType::FullAttention => {
            let attn = build_attention(
                config,
                &prefix,
                weights,
                qconfig,
                runner.clone(),
                fp8_runner,
                device,
                dtype,
            )?;
            LayerMixer::Full(attn)
        }
        LayerType::LinearAttention => {
            let la_cfg = config.linear_attn_config();
            let la = if q38_fp8_dequant_bf16_env_restores_the_4_byte_prequant_arm() {
                LinearAttention::from_loader(
                    la_cfg,
                    &format!("{prefix}.linear_attn"),
                    weights,
                    dtype,
                )?
            } else if nv_q36_gdn_fp8_env_opt_in_quantizes_bf16_checkpoint_gdn_projections_to_fp8_resident() {
                LinearAttention::from_loader_bf16_checkpoint_projections_quantized_to_fp8_resident_halving_gdn_decode_weight_traffic(
                    la_cfg,
                    &format!("{prefix}.linear_attn"),
                    weights,
                    dtype,
                    device,
                    fp8_runner,
                )?
            } else {
                LinearAttention::from_loader_fp8_projections_resident_1_byte_per_param(
                    la_cfg,
                    &format!("{prefix}.linear_attn"),
                    weights,
                    dtype,
                    device,
                    fp8_runner,
                )?
            };
            LayerMixer::Linear(la)
        }
    };

    let ffn = match dense_intermediate {
        Some(inter) => LayerFfn::Dense(build_dense_mlp(
            config,
            inter,
            &format!("{prefix}.mlp"),
            weights,
            qconfig,
            runner,
            fp8_runner,
            device,
            dtype,
        )?),
        None => LayerFfn::Moe(MoeBlock::from_loader_quantized(
            config.moe_config()?,
            &format!("{prefix}.mlp"),
            weights,
            dtype,
            qconfig,
            runner,
            device,
        )?),
    };

    Ok(Qwen3MoeLayer {
        pre_norm,
        post_norm,
        mixer,
        ffn,
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn build_dense_mlp(
    config: &Qwen3MoeConfig,
    intermediate: usize,
    prefix: &str,
    weights: &WeightLoader,
    qconfig: &QuantizationConfig,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    fp8_runner: &Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    device: &Device,
    dtype: DType,
) -> Result<Mlp> {
    let hidden = config.hidden_size;
    let gate = load_attn_proj(
        weights,
        &format!("{prefix}.gate_proj"),
        intermediate,
        hidden,
        qconfig,
        runner.clone(),
        fp8_runner,
        device,
        dtype,
    )?;
    let up = load_attn_proj(
        weights,
        &format!("{prefix}.up_proj"),
        intermediate,
        hidden,
        qconfig,
        runner.clone(),
        fp8_runner,
        device,
        dtype,
    )?;
    let down = load_attn_proj(
        weights,
        &format!("{prefix}.down_proj"),
        hidden,
        intermediate,
        qconfig,
        runner,
        fp8_runner,
        device,
        dtype,
    )?;
    Mlp::new(gate, up, down)
}

#[cfg(feature = "cuda")]
fn build_attention(
    config: &Qwen3MoeConfig,
    prefix: &str,
    weights: &WeightLoader,
    qconfig: &QuantizationConfig,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    fp8_runner: &Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    device: &Device,
    dtype: DType,
) -> Result<AttentionLayer> {
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    let q_out = if config.attn_output_gate {
        n_heads * head_dim * 2
    } else {
        n_heads * head_dim
    };
    let kv_out = n_kv_heads * head_dim;
    let q_proj = load_attn_proj(
        weights,
        &format!("{prefix}.self_attn.q_proj"),
        q_out,
        hidden,
        qconfig,
        runner.clone(),
        fp8_runner,
        device,
        dtype,
    )?;
    let k_proj = load_attn_proj(
        weights,
        &format!("{prefix}.self_attn.k_proj"),
        kv_out,
        hidden,
        qconfig,
        runner.clone(),
        fp8_runner,
        device,
        dtype,
    )?;
    let v_proj = load_attn_proj(
        weights,
        &format!("{prefix}.self_attn.v_proj"),
        kv_out,
        hidden,
        qconfig,
        runner.clone(),
        fp8_runner,
        device,
        dtype,
    )?;
    let o_proj = load_attn_proj(
        weights,
        &format!("{prefix}.self_attn.o_proj"),
        hidden,
        n_heads * head_dim,
        qconfig,
        runner,
        fp8_runner,
        device,
        dtype,
    )?;
    let q_norm = load_rmsnorm_plus_one(
        weights,
        &[&format!("{prefix}.self_attn.q_norm.weight")],
        head_dim,
        config.rms_norm_eps,
        dtype,
    )?;
    let k_norm = load_rmsnorm_plus_one(
        weights,
        &[&format!("{prefix}.self_attn.k_norm.weight")],
        head_dim,
        config.rms_norm_eps,
        dtype,
    )?;
    Ok(AttentionLayer {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        n_heads,
        n_kv_heads,
        head_dim,
        attn_output_gate: config.attn_output_gate,
        rotary_dim: config.rotary_dim(),
    })
}

#[cfg(feature = "cuda")]
fn q38_fp8_dequant_bf16_env_restores_the_4_byte_prequant_arm() -> bool {
    crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_FP8_DEQUANT_BF16")
}

#[cfg(feature = "cuda")]
fn load_attn_proj(
    weights: &WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    qconfig: &QuantizationConfig,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    fp8_runner: &Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    device: &Device,
    dtype: DType,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let is_ignored = qconfig.is_module_ignored(module);
    if is_ignored {
        let w = weights.get(&weight_name, dtype)?;
        return Linear::new(w, None);
    }
    if weights.has(&weight_name) {
        if nv_layers::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(weights, module) {
            if q38_fp8_dequant_bf16_env_restores_the_4_byte_prequant_arm() {
                return nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
                    weights,
                    module,
                    out_features,
                    in_features,
                    dtype,
                );
            }
            return nv_layers::linear::fp8_e4m3_rowscale_checkpoint_resident_linear(
                weights,
                module,
                out_features,
                in_features,
                device,
                fp8_runner.clone(),
            );
        }
        let w = weights.get(&weight_name, dtype)?;
        return Linear::new(w, None);
    }
    let packed_name = format!("{module}.weight_packed");
    if !weights.has(&packed_name) {
        anyhow::bail!("neither {weight_name} nor {packed_name} found");
    }
    if crate::host_serving_util::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_MLP_DEQUANT_BF16") {
        return nv_layers::moe::nvfp4_dequant_bf16_linear_from_disk_because_ablating_the_native_gemm_isolates_decode_defects(
            weights,
            module,
            out_features,
            in_features,
            device,
        );
    }
    nv_layers::moe::nvfp4_linear_from_disk_pub(
        weights,
        module,
        out_features,
        in_features,
        runner,
        device,
    )
}

#[cfg(feature = "cuda")]
fn load_named(weights: &WeightLoader, candidates: &[&str], dtype: DType) -> Result<Tensor> {
    for name in candidates {
        if weights.has(name) {
            return weights
                .get(name, dtype)
                .with_context(|| format!("load {name}"));
        }
    }
    anyhow::bail!("none of {candidates:?} found in weights")
}

#[cfg(feature = "cuda")]
fn load_rmsnorm_plus_one(
    weights: &WeightLoader,
    candidates: &[&str],
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let raw = load_named(weights, candidates, dtype)?;
    let d = raw.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("rmsnorm expected [{}], got {:?}", dim, d);
    }
    let raw_f32 = raw.to_dtype(DType::F32)?;
    let plus = raw_f32.affine(1.0, 1.0)?;
    let weight = plus.to_dtype(dtype)?;
    Ok(RmsNorm::new(weight, eps))
}

#[cfg(not(feature = "cuda"))]
fn slice_cols_bf16(
    _src: &Tensor,
    _rows: usize,
    _src_width: usize,
    _start: usize,
    _width: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

#[cfg(feature = "cuda")]
fn slice_cols_bf16(
    src: &Tensor,
    rows: usize,
    src_width: usize,
    start: usize,
    width: usize,
) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    if src.dtype() != DType::BF16 || !src.is_contiguous() {
        return Ok(None);
    }
    let dev = match src.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    anyhow::ensure!(
        src.elem_count() == rows * src_width && start + width <= src_width,
        "slice_cols_bf16: shape mismatch"
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out: cudarc::driver::CudaSlice<half::bf16> = unsafe {
        stream
            .alloc(rows * width)
            .map_err(|e| anyhow::anyhow!("slice_cols alloc: {e:?}"))?
    };
    {
        let (s, l) = src.storage_and_layout();
        let cu = match &*s {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let sl = cu.as_cuda_slice::<half::bf16>()?;
        let (sp, _g1) = sl.device_ptr(&stream);
        let sp = sp + (l.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
        let (dp, _g2) = out.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::copy_cols_bf16(
                stream.cu_stream() as *mut c_void,
                sp as *const u16,
                dp as *mut u16,
                rows as i32,
                width as i32,
                src_width as i64,
                width as i64,
                start as i64,
                0,
            )
        };
        anyhow::ensure!(rc == 0, "copy_cols_bf16 rc={rc}");
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Some(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (rows, width),
        candle_core::op::BackpropOp::none(),
        false,
    )))
}

#[cfg(feature = "cuda")]
fn concat2_cols_bf16(
    a: &Tensor,
    b: &Tensor,
    rows: usize,
    wa: usize,
    wb: usize,
) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    if a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || !a.is_contiguous()
        || !b.is_contiguous()
    {
        return Ok(None);
    }
    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    anyhow::ensure!(
        a.elem_count() == rows * wa && b.elem_count() == rows * wb,
        "concat2_cols_bf16: shape mismatch"
    );
    let w = wa + wb;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out: cudarc::driver::CudaSlice<half::bf16> = unsafe {
        stream
            .alloc(rows * w)
            .map_err(|e| anyhow::anyhow!("concat2_cols alloc: {e:?}"))?
    };
    for (t, width, dst_off) in [(a, wa, 0usize), (b, wb, wa)] {
        let (s, l) = t.storage_and_layout();
        let cu = match &*s {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let sl = cu.as_cuda_slice::<half::bf16>()?;
        let (sp, _g1) = sl.device_ptr(&stream);
        let sp = sp + (l.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
        let (dp, _g2) = out.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::copy_cols_bf16(
                stream.cu_stream() as *mut c_void,
                sp as *const u16,
                dp as *mut u16,
                rows as i32,
                width as i32,
                width as i64,
                w as i64,
                0,
                dst_off as i64,
            )
        };
        anyhow::ensure!(rc == 0, "copy_cols_bf16 (concat) rc={rc}");
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Some(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (rows, w),
        candle_core::op::BackpropOp::none(),
        false,
    )))
}

#[cfg(feature = "cuda")]
fn embed_lookup_bf16(embed: &Tensor, tokens: &Tensor) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    if embed.dtype() != DType::BF16 || !embed.is_contiguous() || tokens.dtype() != DType::U32 {
        return Ok(None);
    }
    let dev = match embed.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    if !tokens.device().same_device(embed.device()) || !tokens.is_contiguous() {
        return Ok(None);
    }
    let (vocab, hidden) = embed.dims2()?;
    let seq = tokens.elem_count();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out: cudarc::driver::CudaSlice<half::bf16> = unsafe {
        stream
            .alloc(seq * hidden)
            .map_err(|e| anyhow::anyhow!("embed_lookup alloc: {e:?}"))?
    };
    {
        let (es, el) = embed.storage_and_layout();
        let e_cu = match &*es {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let e_sl = e_cu.as_cuda_slice::<half::bf16>()?;
        let (ep, _g1) = e_sl.device_ptr(&stream);
        let ep = ep + (el.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
        let (ts, tl) = tokens.storage_and_layout();
        let t_cu = match &*ts {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let t_sl = t_cu.as_cuda_slice::<u32>()?;
        let (tp, _g2) = t_sl.device_ptr(&stream);
        let tp = tp + (tl.start_offset() * std::mem::size_of::<u32>()) as u64;
        let (op, _g3) = out.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_rows_bf16(
                stream.cu_stream() as *mut c_void,
                ep as *const u16,
                tp as *const i32,
                op as *mut u16,
                seq as i32,
                hidden as i32,
                vocab as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gather_rows_bf16 (embed) rc={rc}");
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Some(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (seq, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    )))
}

#[cfg(feature = "cuda")]
fn fused_attn_gate_sigmoid_mul(
    attn_out: &Tensor,
    q_view_value_gate_interleaved: &Tensor,
    rows: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    anyhow::ensure!(
        attn_out.dtype() == DType::BF16 && q_view_value_gate_interleaved.dtype() == DType::BF16,
        "fused attn gate is bf16-only"
    );
    let dev = match attn_out.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("fused attn gate needs CUDA"),
    };
    let attn_c = attn_out.contiguous()?;
    let q_c = q_view_value_gate_interleaved.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(rows * n_heads * head_dim)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (astor, al) = attn_c.storage_and_layout();
        let (qstor, ql) = q_c.storage_and_layout();
        let (a_cuda, q_cuda) = match (&*astor, &*qstor) {
            (candle_core::Storage::Cuda(a), candle_core::Storage::Cuda(q)) => (a, q),
            _ => anyhow::bail!("expected cuda storage"),
        };
        let a_view = a_cuda.as_cuda_slice::<bf16>()?.slice(al.start_offset()..);
        let q_view = q_cuda.as_cuda_slice::<bf16>()?.slice(ql.start_offset()..);
        let (pa, _ga) = a_view.device_ptr(&stream);
        let (pq, _gq) = q_view.device_ptr(&stream);
        let (po, _go) = out.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::attn_gate_sigmoid_mul_bf16(
                stream.cu_stream() as *mut _,
                pa as *const u16,
                pq as *const u16,
                po as *mut u16,
                rows,
                n_heads,
                head_dim,
            )
        }
    };
    anyhow::ensure!(rc == 0, "attn_gate_sigmoid_mul_bf16 rc={rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        attn_out.dims().to_vec(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn apply_partial_rope(
    q: &Tensor,
    k: &Tensor,
    rope: &Rope,
    positions: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    if rotary_dim >= head_dim {
        let q_f = q.to_dtype(DType::F32)?;
        let k_f = k.to_dtype(DType::F32)?;
        let positions_2d = positions_for_rope(positions, q.dims()[0], q.dims()[1], q.device())?;
        return rope.apply(&q_f, &k_f, &positions_2d);
    }
    let dims_q = q.dims().to_vec();
    let dims_k = k.dims().to_vec();
    #[cfg(feature = "cuda")]
    if std::env::var("NV_Q38_ROPE_FUSED").as_deref() == Ok("1")
        && q.dtype() == DType::BF16
        && k.dtype() == DType::BF16
        && matches!(q.device(), Device::Cuda(_))
    {
        let positions_2d = positions_for_rope(positions, dims_q[0], dims_q[1], q.device())?;
        return rope
            .apply_partial_oop_bf16_cuda_one_launch_replacing_slice_cast_rope_cast_concat(
                q,
                k,
                &positions_2d,
                head_dim,
            );
    }
    #[cfg(feature = "cuda")]
    {
        let rows_q: usize = dims_q[..dims_q.len() - 1].iter().product();
        let rows_k: usize = dims_k[..dims_k.len() - 1].iter().product();
        let pass = head_dim - rotary_dim;
        let parts = (
            slice_cols_bf16(q, rows_q, head_dim, 0, rotary_dim)?,
            slice_cols_bf16(q, rows_q, head_dim, rotary_dim, pass)?,
            slice_cols_bf16(k, rows_k, head_dim, 0, rotary_dim)?,
            slice_cols_bf16(k, rows_k, head_dim, rotary_dim, pass)?,
        );
        if let (Some(q_rot), Some(q_pass), Some(k_rot), Some(k_pass)) = parts {
            let mut rot_dims_q = dims_q.clone();
            *rot_dims_q.last_mut().unwrap() = rotary_dim;
            let mut rot_dims_k = dims_k.clone();
            *rot_dims_k.last_mut().unwrap() = rotary_dim;
            let q_rot_f = q_rot.reshape(rot_dims_q)?.to_dtype(DType::F32)?;
            let k_rot_f = k_rot.reshape(rot_dims_k)?.to_dtype(DType::F32)?;
            let positions_2d = positions_for_rope(positions, dims_q[0], dims_q[1], q.device())?;
            let (q_r, k_r) = rope.apply(&q_rot_f, &k_rot_f, &positions_2d)?;
            let q_r_back = q_r.to_dtype(q.dtype())?.contiguous()?;
            let k_r_back = k_r.to_dtype(k.dtype())?.contiguous()?;
            let joined = (
                concat2_cols_bf16(&q_r_back, &q_pass, rows_q, rotary_dim, pass)?,
                concat2_cols_bf16(&k_r_back, &k_pass, rows_k, rotary_dim, pass)?,
            );
            if let (Some(q_out), Some(k_out)) = joined {
                return Ok((q_out.reshape(dims_q)?, k_out.reshape(dims_k)?));
            }
        }
    }
    let q_rot = q.narrow(dims_q.len() - 1, 0, rotary_dim)?.contiguous()?;
    let q_pass = q
        .narrow(dims_q.len() - 1, rotary_dim, head_dim - rotary_dim)?
        .contiguous()?;
    let k_rot = k.narrow(dims_k.len() - 1, 0, rotary_dim)?.contiguous()?;
    let k_pass = k
        .narrow(dims_k.len() - 1, rotary_dim, head_dim - rotary_dim)?
        .contiguous()?;
    let q_rot_f = q_rot.to_dtype(DType::F32)?;
    let k_rot_f = k_rot.to_dtype(DType::F32)?;
    let positions_2d = positions_for_rope(positions, dims_q[0], dims_q[1], q.device())?;
    let (q_r, k_r) = rope.apply(&q_rot_f, &k_rot_f, &positions_2d)?;
    let q_r_back = q_r.to_dtype(q.dtype())?;
    let k_r_back = k_r.to_dtype(k.dtype())?;
    let q_out = Tensor::cat(&[&q_r_back, &q_pass], dims_q.len() - 1)?;
    let k_out = Tensor::cat(&[&k_r_back, &k_pass], dims_k.len() - 1)?;
    Ok((q_out, k_out))
}

fn positions_for_rope(positions: &Tensor, b: usize, t: usize, device: &Device) -> Result<Tensor> {
    if b == 1 && positions.elem_count() == t && positions.device().same_device(device) {
        return Ok(positions.clone());
    }
    crate::qwen3::host_tile_positions(positions, b, t, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    static KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE:
        std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(feature = "cuda")]
    #[test]
    fn attn_gate_fused_kernel_is_bitwise_vs_the_slice_sigmoid_cast_mul_chain() {
        use half::bf16;

        let _env = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap();
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => panic!(
                "no CUDA device 0: this is the attn-gate bitwise gate and must not report \
                 success having compared nothing"
            ),
        };
        for (rows, n_heads, head_dim) in [(4usize, 16usize, 128usize), (169, 8, 96), (5, 3, 64)] {
            let mut seed = 0x243f6a8885a308d3u64;
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                bf16::from_f32(((seed >> 40) as i32 - (1 << 23)) as f32 / (1 << 21) as f32)
            };
            let attn_v: Vec<bf16> = (0..rows * n_heads * head_dim).map(|_| next()).collect();
            let qraw_v: Vec<bf16> =
                (0..rows * n_heads * 2 * head_dim).map(|_| next()).collect();
            let attn =
                Tensor::from_vec(attn_v, (1, rows, n_heads * head_dim), &device).unwrap();
            let q_view =
                Tensor::from_vec(qraw_v, (1, rows, n_heads, 2 * head_dim), &device).unwrap();

            let qg = q_view
                .narrow(3, head_dim, head_dim)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((1, rows, n_heads * head_dim))
                .unwrap();
            let sig = candle_nn::ops::sigmoid(&qg.to_dtype(DType::F32).unwrap())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let chain = attn.mul(&sig).unwrap();

            let fused =
                fused_attn_gate_sigmoid_mul(&attn, &q_view, rows, n_heads, head_dim).unwrap();

            let cv: Vec<u16> = chain
                .flatten_all()
                .unwrap()
                .to_vec1::<bf16>()
                .unwrap()
                .into_iter()
                .map(bf16::to_bits)
                .collect();
            let fv: Vec<u16> = fused
                .flatten_all()
                .unwrap()
                .to_vec1::<bf16>()
                .unwrap()
                .into_iter()
                .map(bf16::to_bits)
                .collect();
            let diff = cv.iter().zip(&fv).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff, 0,
                "not bitwise at rows={rows} heads={n_heads} head_dim={head_dim}: {diff}/{} \
                 elements differ",
                cv.len()
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rope_fused_partial_kernel_is_bitwise_vs_the_slice_cast_rope_cast_concat_chain() {
        use half::bf16;
        use nv_layers::rope::{Rope, RopeConfig, RopeKind};

        let _env = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap();
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => panic!(
                "no CUDA device 0: this is the rope-fused bitwise gate and must not report \
                 success having compared nothing"
            ),
        };
        for (tokens, n_heads, n_kv_heads, head_dim, rotary_dim) in [
            (4usize, 16usize, 4usize, 128usize, 64usize),
            (4, 16, 4, 256, 64),
            (169, 8, 2, 128, 32),
            (5, 3, 1, 96, 32),
        ] {
            let rope = Rope::new(
                RopeConfig {
                    head_dim: rotary_dim,
                    max_seq_len: 4096,
                    base: 1_000_000.0,
                    kind: RopeKind::Standard,
                },
                &device,
            )
            .expect("rotary-sized rope tables");
            let mut seed = 0x9e3779b97f4a7c15u64;
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                bf16::from_f32(((seed >> 40) as i32 - (1 << 23)) as f32 / (1 << 21) as f32)
            };
            let qv: Vec<bf16> = (0..tokens * n_heads * head_dim).map(|_| next()).collect();
            let kv: Vec<bf16> = (0..tokens * n_kv_heads * head_dim).map(|_| next()).collect();
            let q = Tensor::from_vec(qv, (1, tokens, n_heads, head_dim), &device).unwrap();
            let k = Tensor::from_vec(kv, (1, tokens, n_kv_heads, head_dim), &device).unwrap();
            let positions_v: Vec<i64> = (100..100 + tokens as i64).collect();
            let positions = Tensor::from_vec(positions_v, tokens, &device).unwrap();

            unsafe { std::env::remove_var("NV_Q38_ROPE_FUSED") };
            let (q_chain, k_chain) =
                apply_partial_rope(&q, &k, &rope, &positions, rotary_dim, head_dim)
                    .expect("chain path");
            unsafe { std::env::set_var("NV_Q38_ROPE_FUSED", "1") };
            let (q_fused, k_fused) =
                apply_partial_rope(&q, &k, &rope, &positions, rotary_dim, head_dim)
                    .expect("fused path");
            unsafe { std::env::remove_var("NV_Q38_ROPE_FUSED") };

            for (label, a, b) in [("q", &q_chain, &q_fused), ("k", &k_chain, &k_fused)] {
                let av: Vec<u16> = a
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<bf16>()
                    .unwrap()
                    .into_iter()
                    .map(bf16::to_bits)
                    .collect();
                let bv: Vec<u16> = b
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<bf16>()
                    .unwrap()
                    .into_iter()
                    .map(bf16::to_bits)
                    .collect();
                let diff = av.iter().zip(&bv).filter(|(x, y)| x != y).count();
                assert_eq!(
                    diff, 0,
                    "{label} not bitwise at tokens={tokens} heads={n_heads}/{n_kv_heads} \
                     head_dim={head_dim} rotary={rotary_dim}: {diff}/{} elements differ",
                    av.len()
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn w4a8_scratch_pool_keeps_the_inter_a_allocation_alive_across_an_inter_b_take_on_the_same_thread(
    ) {
        use cudarc::driver::DevicePtr;
        const INTER_A_192_MATCHES_THE_TINY_FIXTURE_GEOMETRY: usize = 192;
        const INTER_B_256_A_SECOND_ENGINE_WITH_A_DIFFERENT_MLP_WIDTH: usize = 256;
        let dev = match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => d,
            _ => panic!(
                "no CUDA device 0: this is the w4a8 scratch-lifetime gate and must not report \
                 success having executed nothing"
            ),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let a = w4a8_decode_scratch_take_or_build(
            &stream,
            INTER_A_192_MATCHES_THE_TINY_FIXTURE_GEOMETRY,
        )
        .unwrap();
        let a_gate_ptr = {
            let (p, _g) = a.gate_y.device_ptr(&stream);
            p
        };
        drop(a);
        let b = w4a8_decode_scratch_take_or_build(
            &stream,
            INTER_B_256_A_SECOND_ENGINE_WITH_A_DIFFERENT_MLP_WIDTH,
        )
        .unwrap();
        let b_gate_ptr = {
            let (p, _g) = b.gate_y.device_ptr(&stream);
            p
        };
        drop(b);
        let a2 = w4a8_decode_scratch_take_or_build(
            &stream,
            INTER_A_192_MATCHES_THE_TINY_FIXTURE_GEOMETRY,
        )
        .unwrap();
        let a2_gate_ptr = {
            let (p, _g) = a2.gate_y.device_ptr(&stream);
            p
        };
        drop(a2);
        let b2 = w4a8_decode_scratch_take_or_build(
            &stream,
            INTER_B_256_A_SECOND_ENGINE_WITH_A_DIFFERENT_MLP_WIDTH,
        )
        .unwrap();
        let b2_gate_ptr = {
            let (p, _g) = b2.gate_y.device_ptr(&stream);
            p
        };
        drop(b2);
        assert_eq!(
            a2_gate_ptr, a_gate_ptr,
            "an inter-A retake after an inter-B take must hand back the allocation whose \
             pointer any captured decode graph baked; a fresh pointer means the old scratch \
             was freed and every replay of that graph is a use-after-free"
        );
        assert_eq!(
            b2_gate_ptr, b_gate_ptr,
            "the inter-B scratch must be pooled too: both engines' graphs replay on this thread"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn kvshare_gate_below_min_total_routes_bitwise_to_the_mk_arm_and_at_or_above_it_to_the_kvshare_arm(
    ) {
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        const NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_KVSHARE_ROUTE_ACCEPTS: (usize, usize, usize) =
            (12, 2, 256);
        const BASE_2044_PLUS_M_4_MAKES_TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_SPLIT_DIFFERENTLY_AND_STOP_AGREEING_BITWISE:
            (usize, usize) = (2044, 4);
        let (n_q, n_kv, hd) = NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_KVSHARE_ROUTE_ACCEPTS;
        let (base, m) =
            BASE_2044_PLUS_M_4_MAKES_TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_SPLIT_DIFFERENTLY_AND_STOP_AGREEING_BITWISE;
        let total = base + m;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the kvshare gate routing gate and must not report \
                 success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 4096, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_base = (
            rand_bf16((1, base, n_kv, hd)),
            rand_bf16((1, base, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_base.0, &kv_base.1,
            )
            .unwrap();
        let kv_tail = (rand_bf16((1, m, n_kv, hd)), rand_bf16((1, m, n_kv, hd)));
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                base, &kv_tail.0, &kv_tail.1,
            )
            .unwrap();
        let q = rand_bf16((1, m, n_q, hd));
        let run = |cache: &mut Qwen3MoeKvCache, arm: Option<&str>, min_total: Option<String>| {
            match arm {
                Some(v) => std::env::set_var(
                    MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
                ),
            }
            match min_total {
                Some(v) => std::env::set_var(
                    MK_VERIFY_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                ),
            }
            let out = cache
                .verify_attention_fp8_mk(0, &q, n_q, m, 1.0 / (hd as f32).sqrt())
                .unwrap()
                .expect("mk verify route must be available for this shape");
            std::env::remove_var(
                MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
            );
            std::env::remove_var(
                MK_VERIFY_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
            );
            out.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let mk = run(&mut cache, None, None);
        let kvshare = run(&mut cache, Some("1"), None);
        let gate_below = run(&mut cache, Some("gate"), Some((total + 1).to_string()));
        let gate_at = run(&mut cache, Some("gate"), Some(total.to_string()));
        assert!(
            mk.iter().all(|x| x.is_finite()) && kvshare.iter().all(|x| x.is_finite()),
            "non-finite verify attention output on a pure arm"
        );
        assert!(
            mk != kvshare,
            "the two arms produced bitwise-identical outputs over random fp8 kv; the routing \
             assertions below would then pass with the gate wired to either arm, so this test \
             would cover nothing"
        );
        assert_eq!(
            gate_below, mk,
            "gate with min_total {} above total {total} must route bitwise to the mk arm",
            total + 1
        );
        assert_eq!(
            gate_at, kvshare,
            "gate with min_total {total} at total {total} must route bitwise to the kvshare arm \
             (the gate engages at, not past, the knee)"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn m1_decode_kvshare_gate_below_min_total_routes_bitwise_to_the_fused_arm_and_at_or_above_it_to_the_kvshare_arm(
    ) {
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        const NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_KVSHARE_ROUTE_ACCEPTS: (usize, usize, usize) =
            (12, 2, 256);
        const TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_SPLIT_DIFFERENTLY_AND_STOP_AGREEING_BITWISE:
            usize = 2048;
        let (n_q, n_kv, hd) = NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_KVSHARE_ROUTE_ACCEPTS;
        let total = TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_SPLIT_DIFFERENTLY_AND_STOP_AGREEING_BITWISE;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the m=1 decode kvshare gate routing gate and must not \
                 report success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        std::env::set_var("NV_Q36_GRAPHED_DECODE_FIX", "1");
        std::env::set_var(
            M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
            "1",
        );
        let cache_built = Qwen3MoeKvCache::new(&config, 4096, &device, DType::BF16);
        std::env::remove_var(
            M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
        );
        let mut cache = cache_built.unwrap();
        assert!(
            cache
                .splitk_scratch_and_fan_in_gated_by_nv_q36_graphed_decode_fix_because_the_smem_kernel_serializes_a_block_reduce_per_kv_position_costing_82ms_at_8k
                .as_ref()
                .is_some_and(|(scratch, _)| scratch.len()
                    >= n_q * M1_DECODE_KVSHARE_SCRATCH_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS * (hd + 2)),
            "the constructor saw NV_Q38_DECODE_KVSHARE engaged and must size the splitk scratch \
             for the kvshare 128-split cap, or the routed kernel scribbles past its scratch"
        );
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv = (
            rand_bf16((1, total, n_kv, hd)),
            rand_bf16((1, total, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv.0, &kv.1,
            )
            .unwrap();
        let q = rand_bf16((1, 1, n_q, hd));
        let run = |cache: &mut Qwen3MoeKvCache, arm: Option<&str>, min_total: Option<String>| {
            match arm {
                Some(v) => std::env::set_var(
                    M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
                    v,
                ),
                None => std::env::remove_var(
                    M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
                ),
            }
            match min_total {
                Some(v) => std::env::set_var(
                    M1_DECODE_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                    v,
                ),
                None => std::env::remove_var(
                    M1_DECODE_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                ),
            }
            let out = cache
                .decode_attention_fp8(0, &q, n_q, 1.0 / (hd as f32).sqrt())
                .unwrap();
            std::env::remove_var(
                M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
            );
            std::env::remove_var(
                M1_DECODE_KVSHARE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
            );
            out.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let fused = run(&mut cache, None, None);
        let kvshare = run(&mut cache, Some("1"), None);
        let gate_below = run(&mut cache, Some("gate"), Some((total + 1).to_string()));
        let gate_at = run(&mut cache, Some("gate"), Some(total.to_string()));
        std::env::remove_var("NV_Q36_GRAPHED_DECODE_FIX");
        assert!(
            fused.iter().all(|x| x.is_finite()) && kvshare.iter().all(|x| x.is_finite()),
            "non-finite m=1 decode attention output on a pure arm"
        );
        assert!(
            fused != kvshare,
            "the two arms produced bitwise-identical outputs over random fp8 kv; the routing \
             assertions below would then pass with the gate wired to either arm, so this test \
             would cover nothing"
        );
        let max_mag_fused = fused.iter().map(|a| a.abs()).fold(0f32, f32::max);
        let max_diff = fused
            .iter()
            .zip(&kvshare)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff <= 1e-3 + 0.02 * max_mag_fused,
            "kvshare diverged beyond bf16-rounding from fused at total={total}: \
             max|diff|={max_diff} vs max|fused|={max_mag_fused}; both accumulate f32 \
             online-softmax over the same fp8 rows so disagreement beyond reordering noise \
             means an indexing bug"
        );
        assert_eq!(
            gate_below, fused,
            "gate with min_total {} above total {total} must route bitwise to the fused arm",
            total + 1
        );
        assert_eq!(
            gate_at, kvshare,
            "gate with min_total {total} at total {total} must route bitwise to the kvshare arm \
             (the gate engages at, not past, the knee)"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rowshared_gate_below_min_total_routes_bitwise_to_the_mk_arm_and_at_or_above_it_to_the_rowshared_arm(
    ) {
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        const NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_ROWSHARED_ROUTE_ACCEPTS: (usize, usize, usize) =
            (12, 2, 256);
        const BASE_2044_PLUS_M_4_MAKES_TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_PARTITION_DIFFERENTLY_AND_STOP_AGREEING_BITWISE:
            (usize, usize) = (2044, 4);
        let (n_q, n_kv, hd) = NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_ROWSHARED_ROUTE_ACCEPTS;
        let (base, m) =
            BASE_2044_PLUS_M_4_MAKES_TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_PARTITION_DIFFERENTLY_AND_STOP_AGREEING_BITWISE;
        let total = base + m;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the rowshared gate routing gate and must not report \
                 success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 4096, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_base = (
            rand_bf16((1, base, n_kv, hd)),
            rand_bf16((1, base, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_base.0, &kv_base.1,
            )
            .unwrap();
        let kv_tail = (rand_bf16((1, m, n_kv, hd)), rand_bf16((1, m, n_kv, hd)));
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                base, &kv_tail.0, &kv_tail.1,
            )
            .unwrap();
        let q = rand_bf16((1, m, n_q, hd));
        let run = |cache: &mut Qwen3MoeKvCache, arm: Option<&str>, min_total: Option<String>| {
            match arm {
                Some(v) => std::env::set_var(
                    MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                ),
            }
            match min_total {
                Some(v) => std::env::set_var(
                    MK_VERIFY_ROWSHARED_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_ROWSHARED_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                ),
            }
            let out = cache
                .verify_attention_fp8_mk(0, &q, n_q, m, 1.0 / (hd as f32).sqrt())
                .unwrap()
                .expect("mk verify route must be available for this shape");
            std::env::remove_var(
                MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            );
            std::env::remove_var(
                MK_VERIFY_ROWSHARED_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
            );
            out.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let mk = run(&mut cache, None, None);
        let rowshared = run(&mut cache, Some("1"), None);
        let gate_below = run(&mut cache, Some("gate"), Some((total + 1).to_string()));
        let gate_at = run(&mut cache, Some("gate"), Some(total.to_string()));
        assert!(
            mk.iter().all(|x| x.is_finite()) && rowshared.iter().all(|x| x.is_finite()),
            "non-finite verify attention output on a pure arm"
        );
        assert!(
            mk != rowshared,
            "the two arms produced bitwise-identical outputs over random fp8 kv; the routing \
             assertions below would then pass with the gate wired to either arm, so this test \
             would cover nothing"
        );
        assert_eq!(
            gate_below, mk,
            "gate with min_total {} above total {total} must route bitwise to the mk arm",
            total + 1
        );
        assert_eq!(
            gate_at, rowshared,
            "gate with min_total {total} at total {total} must route bitwise to the rowshared \
             arm (the gate engages at, not past, the knee)"
        );
        let max_mag = mk.iter().fold(0f32, |a, x| a.max(x.abs()));
        let cross = mk
            .iter()
            .zip(rowshared.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            cross <= 1e-3 + 0.02 * max_mag,
            "rowshared vs mk max diff {cross} (max|mk|={max_mag}) exceeds the bf16 \
             reduction-order band the kvshare arm documented; the arms disagree beyond \
             accumulation order"
        );
        std::env::set_var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            "1",
        );
        std::env::set_var(
            MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
            "1",
        );
        let both_engaged = cache
            .verify_attention_fp8_mk(0, &q, n_q, m, 1.0 / (hd as f32).sqrt())
            .unwrap()
            .expect("mk verify route must be available for this shape")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        std::env::remove_var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        std::env::remove_var(
            MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
        );
        assert_eq!(
            both_engaged, rowshared,
            "with both envs engaged the rowshared arm must win the route, per \
             ROWSHARED_OUTRANKS_KVSHARE_WHEN_BOTH_ENGAGE_BECAUSE_ONE_LAUNCH_READS_KV_ONCE_FOR_ALL_M_ROWS_NOT_ONCE_PER_ROW"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rowshared_forced_row0_m_invariance_2_to_8_stays_in_the_cross_arm_band_because_the_warp_to_row_map_depends_on_m(
    ) {
        const NQ_24_NKV_4_HD_256_THE_QWEN38_FULL_ATTENTION_SHAPE: (usize, usize, usize) =
            (24, 4, 256);
        const BASE_1024_MATCHES_THE_MK_ROW0_INVARIANCE_GATE: usize = 1024;
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let (n_q, n_kv, hd) = NQ_24_NKV_4_HD_256_THE_QWEN38_FULL_ATTENTION_SHAPE;
        let base = BASE_1024_MATCHES_THE_MK_ROW0_INVARIANCE_GATE;
        let m_max = MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the rowshared row0 m-invariance gate and must not \
                 report success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 2048, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_base = (
            rand_bf16((1, base, n_kv, hd)),
            rand_bf16((1, base, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_base.0, &kv_base.1,
            )
            .unwrap();
        let kv_tail = (
            rand_bf16((1, m_max, n_kv, hd)),
            rand_bf16((1, m_max, n_kv, hd)),
        );
        let q_all = rand_bf16((1, m_max, n_q, hd));
        let scale = 1.0 / (hd as f32).sqrt();
        std::env::set_var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            "1",
        );
        let mut row0: Vec<(usize, Vec<f32>)> = Vec::new();
        for m in 2..=m_max {
            cache.set_current_len(base);
            let kt = kv_tail.0.narrow(1, 0, m).unwrap().contiguous().unwrap();
            let vt = kv_tail.1.narrow(1, 0, m).unwrap().contiguous().unwrap();
            cache
                .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                    base, &kt, &vt,
                )
                .unwrap();
            let q = q_all.narrow(1, 0, m).unwrap().contiguous().unwrap();
            let out = cache
                .verify_attention_fp8_mk(0, &q, n_q, m, scale)
                .unwrap()
                .expect("mk verify route must be available for this shape");
            let flat: Vec<f32> = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            row0.push((m, flat[..n_q * hd].to_vec()));
        }
        std::env::remove_var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        let (_, base_row) = &row0[0];
        let max_mag = base_row.iter().fold(0f32, |a, x| a.max(x.abs()));
        for (m, r) in &row0 {
            let worst = r
                .iter()
                .zip(base_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                r.iter().all(|x| x.is_finite()),
                "m={m}: non-finite rowshared row0 output"
            );
            assert!(
                worst <= 1e-3 + 0.02 * max_mag,
                "m={m}: rowshared row 0 differs from the m=2 row 0 by {worst} \
                 (max|row0|={max_mag}); row 0 attends the same {base}+1 kv positions at every \
                 m, so beyond the m-dependent warp-partition reduction-order band this is a \
                 row-bound leak"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tctile_forced_row0_m_invariance_2_to_8_stays_in_the_cross_arm_band_because_the_warp_to_row_map_depends_on_m(
    ) {
        const NQ_24_NKV_4_HD_256_THE_QWEN38_FULL_ATTENTION_SHAPE: (usize, usize, usize) =
            (24, 4, 256);
        const BASE_1024_MATCHES_THE_MK_ROW0_INVARIANCE_GATE: usize = 1024;
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let (n_q, n_kv, hd) = NQ_24_NKV_4_HD_256_THE_QWEN38_FULL_ATTENTION_SHAPE;
        let base = BASE_1024_MATCHES_THE_MK_ROW0_INVARIANCE_GATE;
        let m_max = MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the tctile row0 m-invariance gate and must not \
                 report success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 2048, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_base = (
            rand_bf16((1, base, n_kv, hd)),
            rand_bf16((1, base, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_base.0, &kv_base.1,
            )
            .unwrap();
        let kv_tail = (
            rand_bf16((1, m_max, n_kv, hd)),
            rand_bf16((1, m_max, n_kv, hd)),
        );
        let q_all = rand_bf16((1, m_max, n_q, hd));
        let scale = 1.0 / (hd as f32).sqrt();
        std::env::set_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            "1",
        );
        let mut row0: Vec<(usize, Vec<f32>)> = Vec::new();
        for m in 2..=m_max {
            cache.set_current_len(base);
            let kt = kv_tail.0.narrow(1, 0, m).unwrap().contiguous().unwrap();
            let vt = kv_tail.1.narrow(1, 0, m).unwrap().contiguous().unwrap();
            cache
                .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                    base, &kt, &vt,
                )
                .unwrap();
            let q = q_all.narrow(1, 0, m).unwrap().contiguous().unwrap();
            let out = cache
                .verify_attention_fp8_mk(0, &q, n_q, m, scale)
                .unwrap()
                .expect("mk verify route must be available for this shape");
            let flat: Vec<f32> = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            row0.push((m, flat[..n_q * hd].to_vec()));
        }
        std::env::remove_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        let (_, base_row) = &row0[0];
        let max_mag = base_row.iter().fold(0f32, |a, x| a.max(x.abs()));
        for (m, r) in &row0 {
            let worst = r
                .iter()
                .zip(base_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                r.iter().all(|x| x.is_finite()),
                "m={m}: non-finite tctile row0 output"
            );
            assert!(
                worst <= 1e-3 + 0.02 * max_mag,
                "m={m}: tctile row 0 differs from the m=2 row 0 by {worst} \
                 (max|row0|={max_mag}); row 0 attends the same {base}+1 kv positions at every \
                 m, so beyond the m-dependent warp-partition reduction-order band this is a \
                 row-bound leak"
            );
        }
    }

    #[test]
    fn rowshared_env_grammar_unset_empty_0_off_1_always_gate_knee_min_total_overrides_gate_only() {
        let f = mk_verify_rowshared_min_total_from_raw_none_off_zero_always_gate_reads_the_knee;
        assert_eq!(f(None, None), None);
        assert_eq!(f(Some(""), None), None);
        assert_eq!(f(Some("0"), None), None);
        assert_eq!(f(Some("0"), Some("4096")), None);
        assert_eq!(f(Some("1"), None), Some(0));
        assert_eq!(f(Some("1"), Some("4096")), Some(0));
        assert_eq!(
            f(Some("gate"), None),
            Some(ROWSHARED_GATE_KNEE_TOTAL_SHARES_THE_KVSHARE_CROSSOVER_ITS_OWN_LADDER_FOUND_NO_DEPTH_WORTH_A_DIFFERENT_KNEE)
        );
        assert_eq!(f(Some("gate"), Some("12288")), Some(12288));
    }

    #[test]
    #[should_panic(expected = "NV_Q38_MK_VERIFY_ROWSHARED")]
    fn rowshared_env_grammar_rejects_values_outside_1_gate_0_so_a_typo_cannot_silently_pick_an_arm()
    {
        mk_verify_rowshared_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
            Some("yes"),
            None,
        );
    }

    #[test]
    fn tctile_env_grammar_unset_empty_0_off_1_always_gate_knee_min_total_overrides_gate_only() {
        let f = mk_verify_tctile_min_total_from_raw_none_off_zero_always_gate_reads_the_knee;
        assert_eq!(f(None, None), None);
        assert_eq!(f(Some(""), None), None);
        assert_eq!(f(Some("0"), None), None);
        assert_eq!(f(Some("0"), Some("4096")), None);
        assert_eq!(f(Some("1"), None), Some(0));
        assert_eq!(f(Some("1"), Some("4096")), Some(0));
        assert_eq!(
            f(Some("gate"), None),
            Some(TCTILE_GATE_KNEE_TOTAL_SHARES_THE_KVSHARE_CROSSOVER_ITS_LADDER_WINS_FROM_8192_AND_LOSES_AT_256)
        );
        assert_eq!(f(Some("gate"), Some("12288")), Some(12288));
    }

    #[test]
    #[should_panic(expected = "NV_Q38_MK_VERIFY_TCTILE")]
    fn tctile_env_grammar_rejects_values_outside_1_gate_0_so_a_typo_cannot_silently_pick_an_arm() {
        mk_verify_tctile_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
            Some("yes"),
            None,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tctile_gate_below_min_total_routes_bitwise_to_the_mk_arm_at_or_above_it_to_the_tctile_arm_and_outranks_the_other_arms(
    ) {
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        const NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_TCTILE_ROUTE_ACCEPTS: (usize, usize, usize) =
            (12, 2, 256);
        const BASE_2044_PLUS_M_4_MAKES_TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_PARTITION_DIFFERENTLY_AND_STOP_AGREEING_BITWISE:
            (usize, usize) = (2044, 4);
        let (n_q, n_kv, hd) = NQ_12_NKV_2_HD_256_THE_SMALLEST_SHAPE_THE_TCTILE_ROUTE_ACCEPTS;
        let (base, m) =
            BASE_2044_PLUS_M_4_MAKES_TOTAL_2048_DEEP_ENOUGH_THAT_THE_ARMS_PARTITION_DIFFERENTLY_AND_STOP_AGREEING_BITWISE;
        let total = base + m;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the tctile gate routing gate and must not report \
                 success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 4096, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_base = (
            rand_bf16((1, base, n_kv, hd)),
            rand_bf16((1, base, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_base.0, &kv_base.1,
            )
            .unwrap();
        let kv_tail = (rand_bf16((1, m, n_kv, hd)), rand_bf16((1, m, n_kv, hd)));
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                base, &kv_tail.0, &kv_tail.1,
            )
            .unwrap();
        let q = rand_bf16((1, m, n_q, hd));
        let run = |cache: &mut Qwen3MoeKvCache, arm: Option<&str>, min_total: Option<String>| {
            match arm {
                Some(v) => std::env::set_var(
                    MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                ),
            }
            match min_total {
                Some(v) => std::env::set_var(
                    MK_VERIFY_TCTILE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_TCTILE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
                ),
            }
            let out = cache
                .verify_attention_fp8_mk(0, &q, n_q, m, 1.0 / (hd as f32).sqrt())
                .unwrap()
                .expect("mk verify route must be available for this shape");
            std::env::remove_var(
                MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            );
            std::env::remove_var(
                MK_VERIFY_TCTILE_MIN_TOTAL_ENV_OVERRIDES_THE_GATE_KNEE_PLAIN_INTEGER,
            );
            out.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let mk = run(&mut cache, None, None);
        let tctile = run(&mut cache, Some("1"), None);
        let gate_below = run(&mut cache, Some("gate"), Some((total + 1).to_string()));
        let gate_at = run(&mut cache, Some("gate"), Some(total.to_string()));
        assert!(
            mk.iter().all(|x| x.is_finite()) && tctile.iter().all(|x| x.is_finite()),
            "non-finite verify attention output on a pure arm"
        );
        assert!(
            mk != tctile,
            "the two arms produced bitwise-identical outputs over random fp8 kv; the routing \
             assertions below would then pass with the gate wired to either arm, so this test \
             would cover nothing"
        );
        assert_eq!(
            gate_below, mk,
            "gate with min_total {} above total {total} must route bitwise to the mk arm",
            total + 1
        );
        assert_eq!(
            gate_at, tctile,
            "gate with min_total {total} at total {total} must route bitwise to the tctile \
             arm (the gate engages at, not past, the knee)"
        );
        let max_mag = mk.iter().fold(0f32, |a, x| a.max(x.abs()));
        let cross = mk
            .iter()
            .zip(tctile.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            cross <= 1e-3 + 0.02 * max_mag,
            "tctile vs mk max diff {cross} (max|mk|={max_mag}) exceeds the bf16 \
             reduction-order band the kvshare arm documented; the arms disagree beyond \
             accumulation order"
        );
        std::env::set_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            "1",
        );
        std::env::set_var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            "1",
        );
        std::env::set_var(
            MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
            "1",
        );
        let all_engaged = cache
            .verify_attention_fp8_mk(0, &q, n_q, m, 1.0 / (hd as f32).sqrt())
            .unwrap()
            .expect("mk verify route must be available for this shape")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        std::env::remove_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        std::env::remove_var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        std::env::remove_var(
            MK_VERIFY_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_MK_ARM,
        );
        assert_eq!(
            all_engaged, tctile,
            "with every verify-arm env engaged the tctile arm must win the route, per \
             TCTILE_OUTRANKS_ROWSHARED_AND_KVSHARE_WHEN_ENGAGED_BECAUSE_MMA_TILES_CASH_THE_ONE_READ_STRUCTURE_SCALAR_ROWS_COULD_NOT"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tctile_env_on_the_q36_group8_shape_falls_through_to_the_scalar_arms_because_the_launcher_refuses_group8(
    ) {
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        const NQ_16_NKV_2_HD_256_THE_Q36_GROUP8_SHAPE_TCTILE_REFUSES: (usize, usize, usize) =
            (16, 2, 256);
        const BASE_508_PLUS_M_4_KEEPS_THE_ARM_CHEAP_THE_SUBJECT_IS_ROUTING_NOT_DEPTH: (usize, usize) =
            (508, 4);
        let (n_q, n_kv, hd) = NQ_16_NKV_2_HD_256_THE_Q36_GROUP8_SHAPE_TCTILE_REFUSES;
        let (base, m) = BASE_508_PLUS_M_4_KEEPS_THE_ARM_CHEAP_THE_SUBJECT_IS_ROUTING_NOT_DEPTH;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the tctile group-8 fallthrough gate and must not \
                 report success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 1024, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_all = (
            rand_bf16((1, base + m, n_kv, hd)),
            rand_bf16((1, base + m, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_all.0, &kv_all.1,
            )
            .unwrap();
        let q = rand_bf16((1, m, n_q, hd));
        let scale = 1.0 / (hd as f32).sqrt();
        let mk = cache
            .verify_attention_fp8_mk(0, &q, n_q, m, scale)
            .unwrap()
            .expect("mk verify route must be available for this shape")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        std::env::set_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
            "1",
        );
        let with_tctile_env = cache
            .verify_attention_fp8_mk(0, &q, n_q, m, scale)
            .unwrap()
            .expect("mk verify route must be available for this shape")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        std::env::remove_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        assert_eq!(
            with_tctile_env, mk,
            "NV_Q38_MK_VERIFY_TCTILE=1 on the 16q/2kv group-8 shape must route bitwise to the \
             mk arm: the tctile geometry check excludes group 8 before the launcher ever runs"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mk_verify_row0_is_bitwise_m_invariant_2_to_8_at_the_q38_shape_and_serving_depth() {
        const NQ_24_NKV_4_HD_256_THE_QWEN38_FULL_ATTENTION_SHAPE: (usize, usize, usize) =
            (24, 4, 256);
        const BASE_1024_THE_SERVING_DEPTH_THE_SUFSPEC_CAMPAIGN_SAW_THE_M7_ROW0_FLIP_AT: usize =
            1024;
        let _env_guard = KVSHARE_ENV_IS_PROCESS_GLOBAL_SO_TESTS_TOUCHING_OR_READING_IT_SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let (n_q, n_kv, hd) = NQ_24_NKV_4_HD_256_THE_QWEN38_FULL_ATTENTION_SHAPE;
        let base = BASE_1024_THE_SERVING_DEPTH_THE_SUFSPEC_CAMPAIGN_SAW_THE_M7_ROW0_FLIP_AT;
        let m_max = MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP;
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => panic!(
                "no CUDA device 0: this is the mk verify row0 m-invariance gate and must not \
                 report success having executed nothing: {e}"
            ),
        };
        let config = Qwen3MoeConfig {
            hidden_size: n_q * hd,
            num_hidden_layers: 1,
            num_attention_heads: n_q,
            num_key_value_heads: n_kv,
            head_dim: hd,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![LayerType::FullAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 2048, &device, DType::BF16).unwrap();
        let rand_bf16 = |dims: (usize, usize, usize, usize)| {
            Tensor::rand(-1.0f32, 1.0f32, dims, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let kv_base = (
            rand_bf16((1, base, n_kv, hd)),
            rand_bf16((1, base, n_kv, hd)),
        );
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                0, &kv_base.0, &kv_base.1,
            )
            .unwrap();
        let kv_tail = (
            rand_bf16((1, m_max, n_kv, hd)),
            rand_bf16((1, m_max, n_kv, hd)),
        );
        let q_all = rand_bf16((1, m_max, n_q, hd));
        let scale = 1.0 / (hd as f32).sqrt();
        let mut row0: Vec<(usize, Vec<f32>)> = Vec::new();
        for m in 2..=m_max {
            cache.set_current_len(base);
            let kt = kv_tail.0.narrow(1, 0, m).unwrap().contiguous().unwrap();
            let vt = kv_tail.1.narrow(1, 0, m).unwrap().contiguous().unwrap();
            cache
                .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                    base, &kt, &vt,
                )
                .unwrap();
            let q = q_all.narrow(1, 0, m).unwrap().contiguous().unwrap();
            let out = cache
                .verify_attention_fp8_mk(0, &q, n_q, m, scale)
                .unwrap()
                .expect("mk verify route must be available for this shape");
            let flat: Vec<f32> = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            row0.push((m, flat[..n_q * hd].to_vec()));
        }
        let (_, base_row) = &row0[0];
        for (m, r) in &row0 {
            let worst = r
                .iter()
                .zip(base_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                r == base_row,
                "m={m}: row 0 of the mk verify differs from the m=2 row 0 (max abs diff \
                 {worst}) although row 0 attends to the same {base}+1 kv positions in the same \
                 split traversal order at every m; the kernel leaked later rows or mis-derived \
                 row bounds at this template M"
            );
        }
    }

    #[test]
    fn kvshare_env_grammar_unset_empty_0_off_1_always_gate_knee_min_total_overrides_gate_only() {
        let f = mk_verify_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee;
        assert_eq!(f(None, None), None);
        assert_eq!(f(Some(""), None), None);
        assert_eq!(f(Some("0"), None), None);
        assert_eq!(f(Some("0"), Some("4096")), None);
        assert_eq!(f(Some("1"), None), Some(0));
        assert_eq!(f(Some("1"), Some("4096")), Some(0));
        assert_eq!(
            f(Some("gate"), None),
            Some(KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL)
        );
        assert_eq!(
            f(Some("gate"), Some("")),
            Some(KVSHARE_ENGAGES_PAST_THE_MEASURED_VERIFY_CROSSOVER_TOTAL)
        );
        assert_eq!(f(Some("gate"), Some("12288")), Some(12288));
    }

    #[test]
    #[should_panic(expected = "NV_Q38_MK_VERIFY_KVSHARE")]
    fn kvshare_env_grammar_rejects_values_outside_1_gate_0_so_a_typo_cannot_silently_pick_an_arm()
    {
        mk_verify_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
            Some("yes"),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "NV_Q38_MK_VERIFY_KVSHARE_MIN_TOTAL")]
    fn kvshare_gate_knee_override_rejects_non_integers_so_a_typo_cannot_silently_move_the_knee() {
        mk_verify_kvshare_min_total_from_raw_none_off_zero_always_gate_reads_the_knee(
            Some("gate"),
            Some("16k"),
        );
    }

    #[test]
    fn positions_for_rope_passes_through_b1_and_tiles_b2() {
        let dev = Device::Cpu;
        let p = Tensor::from_vec(vec![0i32, 1, 2], 3usize, &dev).unwrap();
        let out = positions_for_rope(&p, 1, 3, &dev).unwrap();
        assert_eq!(out.elem_count(), 3);
        assert_eq!(
            out.flatten_all().unwrap().to_vec1::<i32>().unwrap(),
            vec![0, 1, 2]
        );
        let tiled = positions_for_rope(&p, 2, 3, &dev).unwrap();
        assert_eq!(tiled.dims(), &[2, 3]);
        assert_eq!(
            tiled.flatten_all().unwrap().to_vec1::<i32>().unwrap(),
            vec![0, 1, 2, 0, 1, 2]
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cpu_cache_slot_mapping_tracks_layer_types() {
        let config = Qwen3MoeConfig {
            hidden_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 16,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![
                LayerType::LinearAttention,
                LayerType::FullAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            attn_output_gate: false,
            tie_word_embeddings: true,
        };
        let mut cache = Qwen3MoeKvCache::new(&config, 16, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(cache.full_slot_for_layer(0), None);
        assert_eq!(cache.full_slot_for_layer(1), Some(0));
        assert_eq!(cache.full_slot_for_layer(3), Some(1));
        assert_eq!(cache.lin_attn_slot_for_layer(0), Some(0));
        assert_eq!(cache.lin_attn_slot_for_layer(2), Some(1));
        cache.advance(5);
        assert_eq!(cache.current_len(), 5);
        cache.set_current_len(2);
        assert_eq!(cache.current_len(), 2);
        cache.reset();
        assert_eq!(cache.current_len(), 0);
    }
}
