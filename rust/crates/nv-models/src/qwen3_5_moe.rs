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
}

#[cfg(feature = "cuda")]
#[path = "qwen38_batch.rs"]
pub mod qwen38_batch;

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

        let (q_value, q_gate) = if self.attn_output_gate {
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
            let attn_cfg = AttnConfig {
                num_heads: self.n_heads,
                num_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                softmax_scale: scaling,
                causal: true,
            };
            flash_attn(
                &q_final.contiguous()?,
                &k_full.contiguous()?,
                &v_full.contiguous()?,
                &attn_cfg,
            )?
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
}

#[cfg(feature = "cuda")]
pub const MK_VERIFY_MAX_ROWS_8_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP: usize = 8;

#[cfg(feature = "cuda")]
pub const MK_VERIFY_MAX_HEAD_DIM_512_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP: usize = 512;

#[cfg(feature = "cuda")]
pub const MK_VERIFY_SPLITS_CEILING_32_MATCHES_FLASH_SPLITS_PICK: usize = 32;

pub const PRIME_CKPT_MAGIC_NVPRIMEK: &[u8; 8] = b"NVPRIMEK";

pub const PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES:
    u32 = 1;

pub const PRIME_CKPT_FINGERPRINT_MISMATCH: &str = "PrimeCkptFingerprintMismatch";

pub const PRIME_CKPT_GEOMETRY_MISMATCH: &str = "PrimeCkptGeometryMismatch";

pub const PRIME_CKPT_BAD_HEADER: &str = "PrimeCkptBadHeader";

#[cfg(feature = "cuda")]
pub fn nv_q36_graphed_decode_fix_env_routes_decode_to_the_gemma4_proven_splitk_flash() -> bool {
    std::env::var("NV_Q36_GRAPHED_DECODE_FIX").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_gdn_fp8_env_opt_in_quantizes_bf16_checkpoint_gdn_projections_to_fp8_resident() -> bool
{
    std::env::var("NV_Q36_GDN_FP8").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q36_lm_head_fp8_env_opt_in_quantizes_a_bf16_checkpoint_lm_head_to_fp8_resident() -> bool
{
    std::env::var("NV_Q36_LM_HEAD_FP8").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_fused_qkv_prep_env_kill_switch_nv_q38_fused_qkv_0() -> bool {
    std::env::var("NV_Q38_FUSED_QKV").ok().as_deref() != Some("0")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_fp4_mlp_gemv_env_opt_in_nv_q38_fp4_mlp_1_because_w4a16_matches_but_does_not_beat_the_padded_tc_route_in_graph() -> bool {
    std::env::var("NV_Q38_FP4_MLP").ok().as_deref() == Some("1")
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
    std::env::var("NV_Q38_MLP_W4A8").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_siluq_split_env_opt_in_nv_q38_siluq_split_1_multiblock_silu_quant_producer() -> bool {
    std::env::var("NV_Q38_SILUQ_SPLIT").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_norm_quant_fold_env_opt_in_nv_q38_norm_quant_fold_1_post_norm_residual_rowquant_one_kernel() -> bool {
    std::env::var("NV_Q38_NORM_QUANT_FOLD").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_down_qfold_env_opt_in_nv_q38_down_qfold_1_act_quant_in_down_gemv_prologue() -> bool {
    std::env::var("NV_Q38_DOWN_QFOLD").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn nv_q38_attn_qkv_one_env_opt_in_single_launch_qkv_gemv_bitwise_same_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_Q38_ATTN_QKV_ONE").ok().as_deref() == Some("1"))
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
    std::env::var("NV_Q38_NORM_WRITEOUT").ok().as_deref() == Some("1")
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
    std::env::var("NV_Q38_W4A8_MK").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub fn verify_mlp_mk_dp4a_selected_for_m_force_1_kill_0_else_auto_under_verify_tc(
    m: usize,
) -> bool {
    match std::env::var("NV_Q38_W4A8_MK").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => {
            nv_layers::linear_attn::verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms()
                && m <= MK_DP4A_VERIFY_MLP_WINS_ONLY_AT_M3_INGRAPH_VERIFY_25_54_VS_25_95_MS_LOSES_M4_27_44_VS_27_0_AND_M6_34_26_VS_29_0
        }
    }
}

#[cfg(feature = "cuda")]
fn fused_dense_mlp_nvfp4_w4a8_small_m_dp4a_for_spec_verify(
    ffn: &LayerFfn,
    x: &Tensor,
    seq: usize,
) -> Result<Option<Tensor>> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

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
        let mk_verify_scratch_and_fan_in = if std::env::var("NV_Q38_MK_VERIFY").ok().as_deref()
            != Some("0")
            && hd <= MK_VERIFY_MAX_HEAD_DIM_512_THE_SPLITK_MK_KERNEL_TEMPLATE_CAP
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
        if fused_verify_ready
            && (nv_layers::linear_attn::mrow_verify_env_read_per_call_so_one_process_can_ab_both_verify_paths()
                || nv_layers::linear_attn::verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms())
        {
            let st = self.lin_attn_states[slot].as_ref().unwrap();
            let ckpts = &self.fused_lin_verify_ckpts_preallocated_so_a_captured_verify_graph_owns_no_ckpt_memory[slot];
            if let Some(out) = la
                .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                    input,
                    st,
                    &ckpts[..t_len],
                )?
            {
                self.fused_lin_verify_rows_pending_rollback = t_len;
                return Ok(out);
            }
        }
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
        if let Some((splitk_scratch, splitk_fan_in)) = &mut self.splitk_scratch_and_fan_in_gated_by_nv_q36_graphed_decode_fix_because_the_smem_kernel_serializes_a_block_reduce_per_kv_position_costing_82ms_at_8k {
            let (sc_ptr, _gscp) = splitk_scratch.device_ptr_mut(&stream);
            let (fi_ptr, _gfip) = splitk_fan_in.device_ptr_mut(&stream);
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
                    sc_ptr as *mut f32,
                    fi_ptr as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                    0,
                    scaling,
                )
            };
            anyhow::ensure!(rc == 0, "flash_decode_fused_fp8kv rc={rc}");
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
        {
            return Ok(None);
        }
        if self
            .mk_verify_scratch_and_fan_in_routing_2_to_8_row_chains_off_the_full_kv_dequant_view_kill_switch_nv_q38_mk_verify_0
            .is_none()
        {
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
        let rc = unsafe {
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
        };
        anyhow::ensure!(rc == 0, "flash_decode_fused_fp8kv_mk rc={rc}");
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

        let prof_enabled = std::env::var("NV_PROF_DECODE").is_ok();
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

        let trace = std::env::var("NV_TRACE_LAYERS").is_ok();
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
        let x = match embed_lookup_bf16(&self.embed_weight, &tokens_flat)? {
            Some(rows) => rows,
            None => self.embed_weight.index_select(&tokens_flat, 0)?,
        }
        .reshape((1usize, seq, self.config.hidden_size))?
        .to_dtype(self.dtype)?;

        let mut residual = x.clone();
        let mut normed = self.layers[0].pre_norm.forward(&x)?;
        decode_prof::lap("embed_and_first_norm");
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
            let moe_out = match (moe_dispatch, layer.ffn.as_moe()) {
                (Some(d), Some(moe)) => d.forward(li, moe, &normed_post)?,
                _ => dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
                    &layer.ffn,
                    &normed_post,
                    seq,
                )?,
            };
            decode_prof::lap(layer.ffn.label());
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
        let logits = self.lm_head.forward(&x)?;
        decode_prof::lap("lm_head");
        decode_prof::report_and_end(write_start);
        cache.advance(seq);
        Ok((logits, hidden))
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
        if std::env::var("NV_PROF_DECODE").ok().as_deref() != Some("1") {
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
struct PrefillWallSplits {
    dev: Option<candle_core::CudaDevice>,
    acc: std::collections::BTreeMap<&'static str, f64>,
}

#[cfg(feature = "cuda")]
impl PrefillWallSplits {
    fn begin_env_nv_prof_prefill(device: &Device) -> Self {
        let on = std::env::var("NV_PROF_PREFILL").ok().as_deref() == Some("1");
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
    hidden_size: usize,
    moe_intermediate_size: usize,
    num_experts_per_tok: usize,
    num_experts: usize,
    max_verify_tokens: usize,
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
            weights.push(w);
        }
        let stream = dev.cuda_stream();
        let ctx = nv_layers::moe_grouped::GroupedDecodeContext::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            &stream,
        )?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch: post-build sync: {e:?}"))?;
        let max_verify_tokens = MAX_VERIFY_MOE_TOKENS.min(256 / cfg.num_experts_per_tok.max(1));
        Ok(Self {
            weights,
            ctx: Mutex::new(ctx),
            verify_ctx: Mutex::new(std::collections::HashMap::new()),
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            num_experts_per_tok: cfg.num_experts_per_tok,
            num_experts: cfg.num_experts,
            max_verify_tokens,
        })
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
        let logits = moe.gate().forward(x_flat)?;
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
        let shared = moe.shared_contribution_device(x_flat)?;
        Ok(Some(routed.add(&shared)?))
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
        let w = self.weights.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "GroupedMoeDispatch: layer {} out of range ({} layers)",
                layer_idx,
                self.weights.len()
            )
        })?;
        let in_dtype = x.dtype();
        let device = x.device().clone();
        let x_flat = x.reshape((1usize, hidden))?.contiguous()?;
        let logits = moe.gate().forward(&x_flat)?;
        let routed = {
            let mut ctx = self
                .ctx
                .lock()
                .map_err(|e| anyhow::anyhow!("GroupedMoeDispatch ctx mutex poisoned: {e}"))?;
            nv_layers::moe_grouped::forward_grouped_decode(
                w, &mut ctx, &x_flat, &logits, None, 0, 0.0, false, 1.0, &device,
            )?
        };
        let shared = moe.shared_contribution_device(&x_flat)?;
        let y = routed.add(&shared)?;
        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(hidden);
        Ok(y.reshape(out_dims)?.to_dtype(in_dtype)?)
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
    std::env::var("NV_Q38_FP8_DEQUANT_BF16").ok().as_deref() == Some("1")
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
    if std::env::var("NV_Q38_MLP_DEQUANT_BF16").ok().as_deref() == Some("1") {
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
