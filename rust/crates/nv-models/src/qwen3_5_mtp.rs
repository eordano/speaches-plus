use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;

use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::moe_bf16_grouped::Bf16GroupedExperts;
use nv_layers::norm::RmsNorm;

#[cfg(feature = "cuda")]
use crate::graph_engine::GraphedQwen3Moe;
use crate::qwen3_5_moe::{AttentionLayer, Qwen3Moe, Qwen3MoeConfig};
#[cfg(feature = "cuda")]
use crate::qwen3_5_moe::{GroupedMoeDispatch, MoeDispatch, Qwen3MoeKvCache};

struct MtpMoe {
    gate: Linear,
    experts: Bf16GroupedExperts,
    shared_expert: Mlp,
    shared_expert_gate: Linear,
    top_k: usize,
}

impl MtpMoe {
    fn forward(&self, x_flat: &Tensor) -> Result<Tensor> {
        let (n_tokens, _hidden) = x_flat.dims2()?;

        let logits = self
            .gate
            .forward(x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (sorted_logits, sorted_idx) = logits.sort_last_dim(false)?;
        let k = self.top_k;
        let top_logits = sorted_logits.narrow(1, 0, k)?.contiguous()?;
        let top_idx = sorted_idx.narrow(1, 0, k)?.contiguous()?;
        let top_weights = candle_nn::ops::softmax_last_dim(&top_logits)?.contiguous()?;

        let topk_ids: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let topk_weights: Vec<f32> = top_weights.flatten_all()?.to_vec1::<f32>()?;

        let routed = self.experts.forward(x_flat, &topk_ids, &topk_weights, k)?;

        let shared_out = self.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
        let shared_gate_logits = self
            .shared_expert_gate
            .forward(x_flat)?
            .to_dtype(DType::F32)?;
        let shared_gate = candle_nn::ops::sigmoid(&shared_gate_logits)?;
        let shared = shared_gate.broadcast_mul(&shared_out)?;

        let y = routed.add(&shared)?;
        debug_assert_eq!(y.dims(), &[n_tokens, self.experts.hidden_size()]);
        Ok(y)
    }
}

enum MtpFfn {
    Moe(MtpMoe),
    Dense(Mlp),
}

impl MtpFfn {
    fn forward(&self, x_flat: &Tensor) -> Result<Tensor> {
        match self {
            MtpFfn::Moe(m) => m.forward(x_flat),
            MtpFfn::Dense(mlp) => Ok(mlp.forward(x_flat)?),
        }
    }
}

pub struct MtpDenseDims {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub attn_output_gate: bool,
    pub rotary_dim: usize,
    pub rms_norm_eps: f64,
}

pub const QWEN38_27B_MTP_DIMS: MtpDenseDims = MtpDenseDims {
    hidden_size: 5120,
    intermediate_size: 17408,
    head_dim: 256,
    num_attention_heads: 24,
    num_key_value_heads: 4,
    attn_output_gate: true,
    rotary_dim: 64,
    rms_norm_eps: 1e-6,
};

pub struct MtpHead {
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc: Linear,
    input_layernorm: RmsNorm,
    self_attn: AttentionLayer,
    post_attention_layernorm: RmsNorm,
    ffn: MtpFfn,
    norm: RmsNorm,
    hidden_size: usize,
    dtype: DType,
}

impl MtpHead {
    pub fn from_safetensors(
        path: impl AsRef<std::path::Path>,
        base: &Qwen3Moe,
        device: &Device,
    ) -> Result<Self> {
        let map = candle_core::safetensors::load(path.as_ref(), device)
            .with_context(|| format!("load MTP safetensors {}", path.as_ref().display()))?;
        Self::from_map(&map, base, device)
    }

    pub fn from_map(
        map: &HashMap<String, Tensor>,
        base: &Qwen3Moe,
        _device: &Device,
    ) -> Result<Self> {
        let cfg: &Qwen3MoeConfig = base.config();
        let dtype = base.dtype();
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let eps = cfg.rms_norm_eps;

        let get = |name: &str| -> Result<Tensor> {
            map.get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("MTP tensor missing: {name}"))
        };
        let lin = |name: &str, out_f: usize, in_f: usize| -> Result<Linear> {
            let w = get(name)?.to_dtype(dtype)?.contiguous()?;
            let d = w.dims();
            anyhow::ensure!(
                d.len() == 2 && d[0] == out_f && d[1] == in_f,
                "MTP {name}: expected [{out_f}, {in_f}], got {d:?}"
            );
            Linear::new(w, None)
        };

        let norm = |name: &str, dim: usize| -> Result<RmsNorm> {
            let raw = get(name)?.to_dtype(DType::F32)?;
            let d = raw.dims();
            anyhow::ensure!(
                d.len() == 1 && d[0] == dim,
                "MTP {name}: expected [{dim}], got {d:?}"
            );
            let w = raw.affine(1.0, 1.0)?.to_dtype(dtype)?;
            Ok(RmsNorm::new(w, eps))
        };

        let pre_fc_norm_embedding = norm("mtp.pre_fc_norm_embedding.weight", hidden)?;
        let pre_fc_norm_hidden = norm("mtp.pre_fc_norm_hidden.weight", hidden)?;
        let fc = lin("mtp.fc.weight", hidden, 2 * hidden)?;

        let input_layernorm = norm("mtp.layers.0.input_layernorm.weight", hidden)?;
        let post_attention_layernorm =
            norm("mtp.layers.0.post_attention_layernorm.weight", hidden)?;

        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let q_out = if cfg.attn_output_gate {
            n_heads * head_dim * 2
        } else {
            n_heads * head_dim
        };
        let kv_out = n_kv_heads * head_dim;
        let q_proj = lin("mtp.layers.0.self_attn.q_proj.weight", q_out, hidden)?;
        let k_proj = lin("mtp.layers.0.self_attn.k_proj.weight", kv_out, hidden)?;
        let v_proj = lin("mtp.layers.0.self_attn.v_proj.weight", kv_out, hidden)?;
        let o_proj = lin(
            "mtp.layers.0.self_attn.o_proj.weight",
            hidden,
            n_heads * head_dim,
        )?;
        let q_norm = norm("mtp.layers.0.self_attn.q_norm.weight", head_dim)?;
        let k_norm = norm("mtp.layers.0.self_attn.k_norm.weight", head_dim)?;
        let self_attn = AttentionLayer::from_parts(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.attn_output_gate,
            cfg.rotary_dim(),
        );

        let gate = lin("mtp.layers.0.mlp.gate.weight", cfg.num_experts, hidden)?;
        let gate_up = get("mtp.layers.0.mlp.experts.gate_up_proj")?;
        let gu_d = gate_up.dims();
        anyhow::ensure!(
            gu_d == [cfg.num_experts, 2 * inter, hidden],
            "MTP experts.gate_up_proj: expected [{}, {}, {}], got {:?}",
            cfg.num_experts,
            2 * inter,
            hidden,
            gu_d
        );
        let down = get("mtp.layers.0.mlp.experts.down_proj")?;
        let dn_d = down.dims();
        anyhow::ensure!(
            dn_d == [cfg.num_experts, hidden, inter],
            "MTP experts.down_proj: expected [{}, {}, {}], got {:?}",
            cfg.num_experts,
            hidden,
            inter,
            dn_d
        );
        let experts = Bf16GroupedExperts::new(&gate_up, &down)?;

        let se_inter = cfg.shared_expert_intermediate_size;
        let shared_expert = Mlp::new(
            lin(
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                se_inter,
                hidden,
            )?,
            lin(
                "mtp.layers.0.mlp.shared_expert.up_proj.weight",
                se_inter,
                hidden,
            )?,
            lin(
                "mtp.layers.0.mlp.shared_expert.down_proj.weight",
                hidden,
                se_inter,
            )?,
        )?;
        let shared_expert_gate = lin("mtp.layers.0.mlp.shared_expert_gate.weight", 1, hidden)?;

        let ffn = MtpFfn::Moe(MtpMoe {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: cfg.num_experts_per_tok,
        });

        let norm_final = norm("mtp.norm.weight", hidden)?;

        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            ffn,
            norm: norm_final,
            hidden_size: hidden,
            dtype,
        })
    }

    pub fn from_safetensors_dense(
        path: impl AsRef<std::path::Path>,
        dims: &MtpDenseDims,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let map = candle_core::safetensors::load(path.as_ref(), device)
            .with_context(|| format!("load MTP safetensors {}", path.as_ref().display()))?;
        Self::from_map_dense(&map, dims, dtype)
    }

    pub fn from_map_dense(
        map: &HashMap<String, Tensor>,
        dims: &MtpDenseDims,
        dtype: DType,
    ) -> Result<Self> {
        let hidden = dims.hidden_size;
        let inter = dims.intermediate_size;
        let eps = dims.rms_norm_eps;

        let get = |name: &str| -> Result<Tensor> {
            map.get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("MTP tensor missing: {name}"))
        };
        let lin = |name: &str, out_f: usize, in_f: usize| -> Result<Linear> {
            let w = get(name)?.to_dtype(dtype)?.contiguous()?;
            let d = w.dims();
            anyhow::ensure!(
                d.len() == 2 && d[0] == out_f && d[1] == in_f,
                "MTP {name}: expected [{out_f}, {in_f}], got {d:?}"
            );
            Linear::new(w, None)
        };
        let norm = |name: &str, dim: usize| -> Result<RmsNorm> {
            let raw = get(name)?.to_dtype(DType::F32)?;
            let d = raw.dims();
            anyhow::ensure!(
                d.len() == 1 && d[0] == dim,
                "MTP {name}: expected [{dim}], got {d:?}"
            );
            let w = raw.affine(1.0, 1.0)?.to_dtype(dtype)?;
            Ok(RmsNorm::new(w, eps))
        };

        let pre_fc_norm_embedding = norm("mtp.pre_fc_norm_embedding.weight", hidden)?;
        let pre_fc_norm_hidden = norm("mtp.pre_fc_norm_hidden.weight", hidden)?;
        let fc = lin("mtp.fc.weight", hidden, 2 * hidden)?;
        let input_layernorm = norm("mtp.layers.0.input_layernorm.weight", hidden)?;
        let post_attention_layernorm =
            norm("mtp.layers.0.post_attention_layernorm.weight", hidden)?;

        let head_dim = dims.head_dim;
        let n_heads = dims.num_attention_heads;
        let n_kv_heads = dims.num_key_value_heads;
        let q_out = if dims.attn_output_gate {
            n_heads * head_dim * 2
        } else {
            n_heads * head_dim
        };
        let kv_out = n_kv_heads * head_dim;
        let q_proj = lin("mtp.layers.0.self_attn.q_proj.weight", q_out, hidden)?;
        let k_proj = lin("mtp.layers.0.self_attn.k_proj.weight", kv_out, hidden)?;
        let v_proj = lin("mtp.layers.0.self_attn.v_proj.weight", kv_out, hidden)?;
        let o_proj = lin(
            "mtp.layers.0.self_attn.o_proj.weight",
            hidden,
            n_heads * head_dim,
        )?;
        let q_norm = norm("mtp.layers.0.self_attn.q_norm.weight", head_dim)?;
        let k_norm = norm("mtp.layers.0.self_attn.k_norm.weight", head_dim)?;
        let self_attn = AttentionLayer::from_parts(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            dims.attn_output_gate,
            dims.rotary_dim,
        );

        let ffn = MtpFfn::Dense(Mlp::new(
            lin("mtp.layers.0.mlp.gate_proj.weight", inter, hidden)?,
            lin("mtp.layers.0.mlp.up_proj.weight", inter, hidden)?,
            lin("mtp.layers.0.mlp.down_proj.weight", hidden, inter)?,
        )?);

        let norm_final = norm("mtp.norm.weight", hidden)?;
        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            ffn,
            norm: norm_final,
            hidden_size: hidden,
            dtype,
        })
    }

    pub fn forward(
        &self,
        base: &Qwen3Moe,
        base_hidden: &Tensor,
        next_token_id: u32,
        position: i32,
    ) -> Result<Tensor> {
        Ok(self
            .forward_draft(base, base_hidden, next_token_id, position)?
            .0)
    }

    pub fn forward_draft(
        &self,
        base: &Qwen3Moe,
        base_hidden: &Tensor,
        next_token_id: u32,
        position: i32,
    ) -> Result<(Tensor, Tensor)> {
        let d = base_hidden.dims();
        anyhow::ensure!(
            d == [1, 1, self.hidden_size],
            "MTP base_hidden: expected [1, 1, {}], got {:?}",
            self.hidden_size,
            d
        );
        let device = base_hidden.device().clone();
        let base_hidden = base_hidden.to_dtype(self.dtype)?;

        let tok = Tensor::from_vec(vec![next_token_id], 1usize, &device)?;
        let emb = base
            .embed_weight()
            .index_select(&tok, 0)?
            .reshape((1usize, 1usize, self.hidden_size))?
            .to_dtype(self.dtype)?;

        let norm_emb = self.pre_fc_norm_embedding.forward(&emb)?;
        let norm_hid = self.pre_fc_norm_hidden.forward(&base_hidden)?;
        let fused_in = Tensor::cat(&[&norm_emb, &norm_hid], D::Minus1)?.contiguous()?;
        let mut h = self.fc.forward(&fused_in)?;

        let positions = Tensor::from_vec(vec![position], 1usize, &device)?;
        let residual = h.clone();
        let normed = self.input_layernorm.forward(&h)?;
        let attn = self.self_attn.forward(&normed, base.rope(), &positions)?;
        h = residual.add(&attn)?;

        let residual2 = h.clone();
        let normed2 = self.post_attention_layernorm.forward(&h)?;
        let x_flat = normed2.reshape((1usize, self.hidden_size))?.contiguous()?;
        let moe_out = self
            .ffn
            .forward(&x_flat)?
            .reshape((1usize, 1usize, self.hidden_size))?
            .to_dtype(self.dtype)?;
        h = residual2.add(&moe_out)?;

        let mtp_hidden = h.clone();

        let normed_final = self.norm.forward(&h)?;
        let logits = base.lm_head().forward(&normed_final)?;
        let logits = logits.reshape((1usize, logits.dim(D::Minus1)?))?;
        Ok((logits, mtp_hidden))
    }

    pub fn forward_draft_tok(
        &self,
        base: &Qwen3Moe,
        base_hidden: &Tensor,
        next_token_id: u32,
        position: i32,
    ) -> Result<(u32, Tensor)> {
        let (logits, mtp_hidden) =
            self.forward_draft(base, base_hidden, next_token_id, position)?;
        let tok = logits
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];
        Ok((tok, mtp_hidden))
    }

    pub fn forward_draft_tok_with_top_prob(
        &self,
        base: &Qwen3Moe,
        base_hidden: &Tensor,
        next_token_id: u32,
        position: i32,
    ) -> Result<(u32, f32, Tensor)> {
        let (logits, mtp_hidden) =
            self.forward_draft(base, base_hidden, next_token_id, position)?;
        let tok = logits
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];
        let p = Self::drafter_top_prob_scalar_is_recip_sum_exp_logits_minus_max(&logits)?
            .to_vec1::<f32>()?[0];
        Ok((tok, p, mtp_hidden))
    }

    fn drafter_top_prob_scalar_is_recip_sum_exp_logits_minus_max(
        logits: &Tensor,
    ) -> Result<Tensor> {
        let lf = logits.to_dtype(DType::F32)?;
        let mx = lf.max_keepdim(D::Minus1)?;
        let z = lf.broadcast_sub(&mx)?.exp()?.sum(D::Minus1)?;
        Ok(z.recip()?.flatten_all()?)
    }
}

pub fn qwen_mtp_enabled() -> bool {
    std::env::var("NV_QWEN_MTP")
        .map(|v| v != "0")
        .unwrap_or(false)
}

pub fn qwen_mtp_k() -> usize {
    std::env::var("NV_QWEN_MTP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| k >= 1)
        .unwrap_or(3)
}

pub const NV_Q36_MTP_P_MIN_ENV: &str = "NV_Q36_MTP_P_MIN";

pub fn q36_mtp_draft_p_min(raw: Option<&str>) -> Option<f32> {
    raw.and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|p| *p > 0.0 && *p < 1.0)
}

pub fn q36_mtp_draft_p_min_from_env() -> Option<f32> {
    q36_mtp_draft_p_min(std::env::var(NV_Q36_MTP_P_MIN_ENV).ok().as_deref())
}

pub const ADAPTIVE_MOE_DRAFT_KEEPS_AT_LEAST_ONE_BECAUSE_A_DRAFTLESS_ROUND_IS_A_PLAIN_DECODE_OUTSIDE_EVERY_TWO_PLUS_ROW_VERIFY_LANE:
    usize = 1;

pub fn q36_adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run(
    top_probs: &[f32],
    p_min: f32,
) -> usize {
    top_probs
        .iter()
        .take_while(|&&p| p >= p_min)
        .count()
        .max(ADAPTIVE_MOE_DRAFT_KEEPS_AT_LEAST_ONE_BECAUSE_A_DRAFTLESS_ROUND_IS_A_PLAIN_DECODE_OUTSIDE_EVERY_TWO_PLUS_ROW_VERIFY_LANE)
        .min(top_probs.len().max(1))
}

fn argmax_row(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

#[derive(Default, Clone)]
pub struct MtpSpecStats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
    pub pos0_accepted: usize,
    pub round_ms: Vec<f64>,
    pub accept_len_hist: std::collections::BTreeMap<usize, usize>,
    pub stop_token: Option<u32>,

    pub draft_ms: f64,
    pub verify_ms: f64,
    pub commit_ms: f64,
    pub verify_ms_rounds: Vec<f64>,
}

impl MtpSpecStats {
    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.emitted as f64 / self.rounds as f64
        }
    }
    pub fn pos0_accept_rate(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.pos0_accepted as f64 / self.rounds as f64
        }
    }
}

#[cfg(feature = "cuda")]
fn top2_host_row_for_argmax_margin_triage(row: &[f32]) -> (f32, f32) {
    let mut v1 = f32::NEG_INFINITY;
    let mut v2 = f32::NEG_INFINITY;
    for &v in row {
        if v > v1 {
            v2 = v1;
            v1 = v;
        } else if v > v2 {
            v2 = v;
        }
    }
    (v1, v2)
}

#[cfg(feature = "cuda")]
fn argmax_rows(flat: &[f32], rows: usize, vocab: usize) -> Result<Vec<u32>> {
    anyhow::ensure!(
        flat.len() == rows * vocab,
        "argmax rows: {} logits != {rows} x {vocab}",
        flat.len()
    );
    Ok((0..rows)
        .map(|r| argmax_row(&flat[r * vocab..(r + 1) * vocab]))
        .collect())
}

#[cfg(feature = "cuda")]
fn accept_prefix_len_then_bonus(drafts: &[u32], greedy: &[u32]) -> (usize, u32) {
    let mut accepted = 0usize;
    while accepted < drafts.len() && drafts[accepted] == greedy[accepted] {
        accepted += 1;
    }
    (accepted, greedy[accepted])
}

#[cfg(feature = "cuda")]
pub trait MtpMoeServingRounds {
    fn anchor(&self) -> u32;
    fn round_fits(&self) -> bool;
    fn round(&mut self) -> Result<Vec<u32>>;
    fn stats(&self) -> &MtpSpecStats;
    fn suffix_rounds_drafted_accepted(&self) -> Option<(usize, usize, usize)> {
        None
    }
    fn acceptance_tracking_k2_k3_switches(&self) -> Option<(usize, usize, usize)> {
        None
    }
    fn arm_name(&self) -> &'static str;
    fn capture_active(&self) -> bool;
}

#[cfg(feature = "cuda")]
pub struct MtpMoeEagerDecodeSession<'a> {
    base: &'a Qwen3Moe,
    mtp: &'a MtpHead,
    k: usize,
    p_min_drafter_confidence_cut_none_means_fixed_k: Option<f32>,
    dispatch: Option<&'a dyn MoeDispatch>,
    cache: Qwen3MoeKvCache,
    anchor: u32,
    base_hidden: Tensor,
    max_seq: usize,
    pub stats: MtpSpecStats,
    pub submitted_draft_len_hist: std::collections::BTreeMap<usize, usize>,
    pub emitted_argmax_margins_after_anchor_recorded_only_under_p_min_so_the_default_arm_skips_the_full_logits_readback:
        Vec<f32>,
}

#[cfg(feature = "cuda")]
impl<'a> MtpMoeEagerDecodeSession<'a> {
    pub fn start(
        base: &'a Qwen3Moe,
        mtp: &'a MtpHead,
        k: usize,
        prompt: &[u32],
        max_seq: usize,
        dispatch: Option<&'a dyn MoeDispatch>,
    ) -> Result<Self> {
        anyhow::ensure!(!prompt.is_empty(), "moe mtp session: empty prompt");
        anyhow::ensure!(k >= 1, "moe mtp session: k must be >= 1");
        let p_min = q36_mtp_draft_p_min_from_env();
        anyhow::ensure!(
            p_min.is_none() || k >= 2,
            "{NV_Q36_MTP_P_MIN_ENV} needs k >= 2: a k=1 draft is already the minimum the cut \
             keeps, so there is nothing to shorten"
        );
        anyhow::ensure!(
            prompt.len() + k < max_seq,
            "moe mtp session: prompt {} + k+1 {} > max_seq {max_seq}",
            prompt.len(),
            k + 1
        );
        let device = base.device().clone();
        let mut cache = base.new_kv_cache(max_seq)?;

        let seq = prompt.len();
        let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device)?;
        let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)?;
        let (logits, hidden) = base.forward_with_cache_dispatched_hidden(
            &tokens,
            &pos,
            &mut cache,
            PREFILL_ROUTES_HOST_BECAUSE_THE_GROUPED_MULTI_TOKEN_COST_EXPLODES_WITH_M,
        )?;
        let last: Vec<f32> = logits
            .narrow(1, seq - 1, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let anchor = argmax_row(&last);
        let base_hidden = hidden.narrow(1, seq - 1, 1)?.contiguous()?;
        drop(logits);
        drop(hidden);

        if dispatch.is_some() && seq + (k + 1) <= max_seq {
            let l0 = cache.current_len();
            let snap0 = cache.snapshot_lin_states()?;
            let warm: Vec<u32> = vec![anchor; k + 1];
            let wp: Vec<i32> = (0..(k + 1) as i32).map(|i| l0 as i32 + i).collect();
            let wt = Tensor::from_vec(warm, (1usize, k + 1), &device)?;
            let wpt = Tensor::from_vec(wp, k + 1, &device)?;
            let _ = base.forward_with_cache_dispatched(&wt, &wpt, &mut cache, dispatch)?;
            cache.restore_lin_states(&snap0)?;
            cache.set_current_len(l0);
        }
        let _ = device.synchronize();

        Ok(Self {
            base,
            mtp,
            k,
            p_min_drafter_confidence_cut_none_means_fixed_k: p_min,
            dispatch,
            cache,
            anchor,
            base_hidden,
            max_seq,
            stats: MtpSpecStats::default(),
            submitted_draft_len_hist: std::collections::BTreeMap::new(),
            emitted_argmax_margins_after_anchor_recorded_only_under_p_min_so_the_default_arm_skips_the_full_logits_readback:
                Vec::new(),
        })
    }

    pub fn anchor(&self) -> u32 {
        self.anchor
    }

    pub fn committed_len(&self) -> usize {
        self.cache.current_len()
    }

    pub fn round_fits(&self) -> bool {
        self.cache.current_len() + self.k < self.max_seq
    }

    pub fn round(&mut self) -> Result<Vec<u32>> {
        self.round_inner(None)
    }

    pub fn round_with_drafts_from_a_clairvoyant_test_oracle(
        &mut self,
        drafts: &[u32],
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            drafts.len() == self.k,
            "oracle round needs exactly k={} drafts, got {}",
            self.k,
            drafts.len()
        );
        self.round_inner(Some(drafts))
    }

    fn round_inner(&mut self, forced_drafts: Option<&[u32]>) -> Result<Vec<u32>> {
        let device = self.base.device().clone();
        let round_t0 = std::time::Instant::now();
        let l = self.cache.current_len();
        let k = self.k;
        anyhow::ensure!(
            l + k < self.max_seq,
            "moe mtp verify overflow: committed={l} + k+1={} > max_seq={}",
            k + 1,
            self.max_seq
        );

        let p_min_cut = self
            .p_min_drafter_confidence_cut_none_means_fixed_k
            .filter(|_| forced_drafts.is_none());
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut draft_top_probs: Vec<f32> = Vec::with_capacity(k);
        let mut h = self.base_hidden.clone();
        let mut tok = self.anchor;
        for j in 0..k {
            let position = (l + j) as i32;
            if let Some(pmin) = p_min_cut {
                let (d, p, dhidden) = self
                    .mtp
                    .forward_draft_tok_with_top_prob(self.base, &h, tok, position)?;
                drafts.push(d);
                draft_top_probs.push(p);
                tok = d;
                h = dhidden;
                if p < pmin
                    && drafts.len()
                        >= ADAPTIVE_MOE_DRAFT_KEEPS_AT_LEAST_ONE_BECAUSE_A_DRAFTLESS_ROUND_IS_A_PLAIN_DECODE_OUTSIDE_EVERY_TWO_PLUS_ROW_VERIFY_LANE
                {
                    break;
                }
            } else {
                let (d, dhidden) = self.mtp.forward_draft_tok(self.base, &h, tok, position)?;
                let d = match forced_drafts {
                    Some(f) => f[j],
                    None => d,
                };
                drafts.push(d);
                tok = d;
                h = dhidden;
            }
        }
        if let Some(pmin) = p_min_cut {
            drafts.truncate(
                q36_adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run(
                    &draft_top_probs,
                    pmin,
                ),
            );
        }
        *self
            .submitted_draft_len_hist
            .entry(drafts.len())
            .or_insert(0) += 1;
        let _ = device.synchronize();
        let draft_dt = round_t0.elapsed().as_secs_f64();

        let verify_t0 = std::time::Instant::now();
        let mut block: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
        block.push(self.anchor);
        block.extend_from_slice(&drafts);
        let m = block.len();
        let block_pos: Vec<i32> = (0..m as i32).map(|i| l as i32 + i).collect();
        let bt = Tensor::from_vec(block, (1usize, m), &device)?;
        let bp = Tensor::from_vec(block_pos, m, &device)?;

        self.cache.set_capture_lin_ckpts(true);
        let (vlogits, vhidden) = self.base.forward_with_cache_dispatched_hidden(
            &bt,
            &bp,
            &mut self.cache,
            self.dispatch,
        )?;
        let greedy: Vec<u32> = vlogits
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;
        let _ = device.synchronize();
        let verify_dt = verify_t0.elapsed().as_secs_f64();

        let commit_t0 = std::time::Instant::now();
        let (accepted, bonus) = accept_prefix_len_then_bonus(&drafts, &greedy);
        let consumed = accepted + 1;
        if self
            .p_min_drafter_confidence_cut_none_means_fixed_k
            .is_some()
        {
            let vocab = vlogits.dim(D::Minus1)?;
            let flat: Vec<f32> = vlogits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            for r in 0..consumed {
                let (v1, v2) =
                    top2_host_row_for_argmax_margin_triage(&flat[r * vocab..(r + 1) * vocab]);
                self.emitted_argmax_margins_after_anchor_recorded_only_under_p_min_so_the_default_arm_skips_the_full_logits_readback
                    .push(v1 - v2);
            }
        }

        let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
        emitted.push(bonus);

        self.base_hidden = vhidden.narrow(1, accepted, 1)?.contiguous()?;
        self.cache.set_current_len(l + consumed);
        self.cache.rollback_lin_to(consumed)?;
        self.anchor = bonus;
        let _ = device.synchronize();
        let commit_dt = commit_t0.elapsed().as_secs_f64();

        self.stats
            .round_ms
            .push(1000.0 * round_t0.elapsed().as_secs_f64());
        self.stats.draft_ms += 1000.0 * draft_dt;
        self.stats.verify_ms += 1000.0 * verify_dt;
        self.stats.commit_ms += 1000.0 * commit_dt;
        self.stats.rounds += 1;
        self.stats.drafted += drafts.len();
        self.stats.accepted += accepted;
        self.stats.emitted += emitted.len();
        if accepted > 0 {
            self.stats.pos0_accepted += 1;
        }
        *self.stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
        Ok(emitted)
    }
}

#[cfg(feature = "cuda")]
impl MtpMoeServingRounds for MtpMoeEagerDecodeSession<'_> {
    fn anchor(&self) -> u32 {
        MtpMoeEagerDecodeSession::anchor(self)
    }
    fn round_fits(&self) -> bool {
        MtpMoeEagerDecodeSession::round_fits(self)
    }
    fn round(&mut self) -> Result<Vec<u32>> {
        MtpMoeEagerDecodeSession::round(self)
    }
    fn stats(&self) -> &MtpSpecStats {
        &self.stats
    }
    fn arm_name(&self) -> &'static str {
        "eager"
    }
    fn capture_active(&self) -> bool {
        false
    }
}

#[cfg(feature = "cuda")]
pub const GRAPHED_SESSION_PREFILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN36_CTX_SCALING_PPL_BLOCK: usize =
    512;

#[cfg(feature = "cuda")]
pub const THE_GROUPED_MULTI_TOKEN_VERIFY_CAPTURES_NOW_THAT_ITS_HOST_STALL_IS_THE_OPT_IN_DIAGNOSTIC_NV_MOE_MT_STALL:
    bool = true;

#[cfg(feature = "cuda")]
pub const PREFILL_ROUTES_HOST_BECAUSE_THE_GROUPED_MULTI_TOKEN_COST_EXPLODES_WITH_M: Option<
    &'static dyn MoeDispatch,
> = None;

#[cfg(feature = "cuda")]
pub struct MtpMoeGraphedDecodeSession<'a> {
    engine: &'a mut GraphedQwen3Moe,
    mtp: &'a MtpHead,
    k: usize,
    p_min_drafter_confidence_cut_none_means_fixed_k: Option<f32>,
    anchor: u32,
    base_hidden: Tensor,
    max_seq: usize,
    pub stats: MtpSpecStats,
    pub submitted_draft_len_hist: std::collections::BTreeMap<usize, usize>,
    pub emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption:
        Vec<f32>,
}

#[cfg(feature = "cuda")]
impl<'a> MtpMoeGraphedDecodeSession<'a> {
    pub fn start(
        engine: &'a mut GraphedQwen3Moe,
        mtp: &'a MtpHead,
        k: usize,
        prompt: &[u32],
    ) -> Result<Self> {
        anyhow::ensure!(!prompt.is_empty(), "graphed moe mtp session: empty prompt");
        anyhow::ensure!(k >= 1, "graphed moe mtp session: k must be >= 1");
        let max_seq = engine.cache().max_seq_len();
        anyhow::ensure!(
            prompt.len() + k < max_seq,
            "graphed moe mtp session: prompt {} + k+1 {} > max_seq {max_seq}",
            prompt.len(),
            k + 1
        );
        if let Err(
            the_one_shot_invalid_value_a_captured_graph_teardown_leaves_under_the_decode_fix_arm,
        ) = engine.reset()
        {
            eprintln!(
                "[q36-mtp-graphed-session] reset retry absorbing the one-shot #96 teardown residual: {the_one_shot_invalid_value_a_captured_graph_teardown_leaves_under_the_decode_fix_arm:#}"
            );
            engine.reset().map_err(|e| {
                anyhow::anyhow!(
                    "graphed moe mtp session: second reset after a captured-graph teardown also \
                     failed; the context is genuinely poisoned: {e:#}"
                )
            })?;
        }
        let stashed_hook_matching_prefill_routes_host_because_the_grouped_multi_token_cost_explodes_with_m =
            engine.take_moe_dispatch_for_host_routed_prefill_then_restore_reinstalling_would_drop_and_rebuild_the_folded_weights_a_torn_down_graph_baked();
        let chunk =
            GRAPHED_SESSION_PREFILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN36_CTX_SCALING_PPL_BLOCK;
        let mut fed = 0usize;
        while prompt.len() - fed > chunk {
            let _ = engine.prefill(&prompt[fed..fed + chunk])?;
            fed += chunk;
        }
        let (last_row, hidden) = engine.prefill_hidden_serving_last_row_logits(&prompt[fed..])?;
        let tail = prompt.len() - fed;
        let anchor = argmax_row(&last_row);
        let base_hidden = hidden.narrow(1, tail - 1, 1)?.contiguous()?;
        match stashed_hook_matching_prefill_routes_host_because_the_grouped_multi_token_cost_explodes_with_m {
            Some(hook) => engine.set_moe_dispatch(Some(hook)),
            None => engine.install_grouped_moe()?,
        }
        engine.set_device_routing(THE_GROUPED_MULTI_TOKEN_VERIFY_CAPTURES_NOW_THAT_ITS_HOST_STALL_IS_THE_OPT_IN_DIAGNOSTIC_NV_MOE_MT_STALL);
        engine.ensure_verify_lane(k + 1)?;
        let p_min = q36_mtp_draft_p_min_from_env();
        if p_min.is_some() {
            anyhow::ensure!(
                k >= 2,
                "{NV_Q36_MTP_P_MIN_ENV} needs k >= 2: a k=1 draft is already the minimum the \
                 cut keeps, so there is nothing to shorten"
            );
            let adaptive_rows_shapes: Vec<usize> =
                (1 + ADAPTIVE_MOE_DRAFT_KEEPS_AT_LEAST_ONE_BECAUSE_A_DRAFTLESS_ROUND_IS_A_PLAIN_DECODE_OUTSIDE_EVERY_TWO_PLUS_ROW_VERIFY_LANE
                    ..=k + 1)
                    .collect();
            engine
                .precapture_verify_graphs_then_discard_state_so_no_mid_run_capture_pollutes_verify_ms(
                    &adaptive_rows_shapes,
                )?;
        }
        let device = engine.device().clone();
        let _ = device.synchronize();
        Ok(Self {
            engine,
            mtp,
            k,
            p_min_drafter_confidence_cut_none_means_fixed_k: p_min,
            anchor,
            base_hidden,
            max_seq,
            stats: MtpSpecStats::default(),
            submitted_draft_len_hist: std::collections::BTreeMap::new(),
            emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption:
                Vec::new(),
        })
    }

    pub fn anchor(&self) -> u32 {
        self.anchor
    }

    pub fn committed_len(&self) -> usize {
        self.engine.current_pos()
    }

    pub fn round_fits(&self) -> bool {
        self.engine.current_pos() + self.k < self.max_seq
    }

    pub fn round(&mut self) -> Result<Vec<u32>> {
        self.round_inner(None)
    }

    pub fn round_with_drafts_from_a_clairvoyant_test_oracle(
        &mut self,
        drafts: &[u32],
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            drafts.len() == self.k,
            "oracle round needs exactly k={} drafts, got {}",
            self.k,
            drafts.len()
        );
        self.round_inner(Some(drafts))
    }

    fn round_inner(&mut self, forced_drafts: Option<&[u32]>) -> Result<Vec<u32>> {
        let device = self.engine.device().clone();
        let round_t0 = std::time::Instant::now();
        let l = self.engine.current_pos();
        let k = self.k;
        anyhow::ensure!(
            l + k < self.max_seq,
            "graphed moe mtp verify overflow: committed={l} + k+1={} > max_seq={}",
            k + 1,
            self.max_seq
        );

        let p_min_cut = self
            .p_min_drafter_confidence_cut_none_means_fixed_k
            .filter(|_| forced_drafts.is_none());
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut draft_top_probs: Vec<f32> = Vec::with_capacity(k);
        let mut h = self.base_hidden.clone();
        let mut tok = self.anchor;
        for j in 0..k {
            let position = (l + j) as i32;
            if let Some(pmin) = p_min_cut {
                let (d, p, dhidden) = self.mtp.forward_draft_tok_with_top_prob(
                    self.engine.underlying(),
                    &h,
                    tok,
                    position,
                )?;
                drafts.push(d);
                draft_top_probs.push(p);
                tok = d;
                h = dhidden;
                if p < pmin
                    && drafts.len()
                        >= ADAPTIVE_MOE_DRAFT_KEEPS_AT_LEAST_ONE_BECAUSE_A_DRAFTLESS_ROUND_IS_A_PLAIN_DECODE_OUTSIDE_EVERY_TWO_PLUS_ROW_VERIFY_LANE
                {
                    break;
                }
            } else {
                let (d, dhidden) =
                    self.mtp
                        .forward_draft_tok(self.engine.underlying(), &h, tok, position)?;
                let d = match forced_drafts {
                    Some(f) => f[j],
                    None => d,
                };
                drafts.push(d);
                tok = d;
                h = dhidden;
            }
        }
        if let Some(pmin) = p_min_cut {
            drafts.truncate(
                q36_adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run(
                    &draft_top_probs,
                    pmin,
                ),
            );
        }
        *self
            .submitted_draft_len_hist
            .entry(drafts.len())
            .or_insert(0) += 1;
        let _ = device.synchronize();
        let draft_dt = round_t0.elapsed().as_secs_f64();

        let verify_t0 = std::time::Instant::now();
        let mut block: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
        block.push(self.anchor);
        block.extend_from_slice(&drafts);
        let m = block.len();
        self.engine.ensure_verify_lane(m)?;
        self.engine.forward_verify_chain(&block)?;
        let flat = self.engine.verify_logits_host()?;
        let vocab = self.engine.vocab_size();
        let greedy = argmax_rows(&flat, m, vocab)?;
        let verify_dt = verify_t0.elapsed().as_secs_f64();

        let commit_t0 = std::time::Instant::now();
        let (accepted, bonus) = accept_prefix_len_then_bonus(&drafts, &greedy);
        let consumed = accepted + 1;

        let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
        emitted.push(bonus);
        for r in 0..consumed {
            let (v1, v2) =
                top2_host_row_for_argmax_margin_triage(&flat[r * vocab..(r + 1) * vocab]);
            self.emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption
                .push(v1 - v2);
        }

        let vhidden = self
            .engine
            .verify_hidden_rows_tensor_valid_until_next_forward()?;
        self.base_hidden = vhidden.narrow(1, accepted, 1)?.copy()?;
        self.engine.commit_verify_consumed(consumed)?;
        self.anchor = bonus;
        let _ = device.synchronize();
        let commit_dt = commit_t0.elapsed().as_secs_f64();

        self.stats
            .round_ms
            .push(1000.0 * round_t0.elapsed().as_secs_f64());
        self.stats.draft_ms += 1000.0 * draft_dt;
        self.stats.verify_ms += 1000.0 * verify_dt;
        self.stats.commit_ms += 1000.0 * commit_dt;
        self.stats.rounds += 1;
        self.stats.drafted += drafts.len();
        self.stats.accepted += accepted;
        self.stats.emitted += emitted.len();
        if accepted > 0 {
            self.stats.pos0_accepted += 1;
        }
        *self.stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
        Ok(emitted)
    }
}

#[cfg(feature = "cuda")]
impl MtpMoeServingRounds for MtpMoeGraphedDecodeSession<'_> {
    fn anchor(&self) -> u32 {
        MtpMoeGraphedDecodeSession::anchor(self)
    }
    fn round_fits(&self) -> bool {
        MtpMoeGraphedDecodeSession::round_fits(self)
    }
    fn round(&mut self) -> Result<Vec<u32>> {
        MtpMoeGraphedDecodeSession::round(self)
    }
    fn stats(&self) -> &MtpSpecStats {
        &self.stats
    }
    fn arm_name(&self) -> &'static str {
        "captured-verify"
    }
    fn capture_active(&self) -> bool {
        self.engine.capture_active()
    }
}

#[cfg(feature = "cuda")]
pub struct MtpSpecEngine<'a> {
    base: &'a Qwen3Moe,
    mtp: &'a MtpHead,
    k: usize,
    stop_ids: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl<'a> MtpSpecEngine<'a> {
    pub fn new(base: &'a Qwen3Moe, mtp: &'a MtpHead, k: usize) -> Self {
        Self {
            base,
            mtp,
            k: k.max(1),
            stop_ids: Vec::new(),
        }
    }

    pub fn with_stop_ids(mut self, ids: Vec<u32>) -> Self {
        self.stop_ids = ids;
        self
    }

    pub fn generate_greedy(
        &self,
        prompt: &[u32],
        max_new: usize,
        max_seq: usize,
    ) -> Result<(Vec<u32>, MtpSpecStats)> {
        self.generate_inner(prompt, max_new, max_seq, true)
    }

    pub fn generate_reference(
        &self,
        prompt: &[u32],
        max_new: usize,
        max_seq: usize,
    ) -> Result<(Vec<u32>, MtpSpecStats)> {
        self.generate_inner(prompt, max_new, max_seq, false)
    }

    fn argmax_last(&self, logits: &Tensor) -> Result<u32> {
        let seq = logits.dim(1)?;
        let row: Vec<f32> = logits
            .narrow(1, seq - 1, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok(argmax_row(&row))
    }

    fn generate_inner(
        &self,
        prompt: &[u32],
        max_new: usize,
        max_seq: usize,
        use_draft: bool,
    ) -> Result<(Vec<u32>, MtpSpecStats)> {
        anyhow::ensure!(!prompt.is_empty(), "generate: empty prompt");
        let device = self.base.device().clone();
        let mut cache = self.base.new_kv_cache(max_seq)?;
        let mut stats = MtpSpecStats::default();
        let k = self.k;

        let host_verify = std::env::var("NV_QWEN_MTP_HOST")
            .map(|v| v != "0")
            .unwrap_or(false);
        let dispatch: Option<GroupedMoeDispatch> = if host_verify {
            None
        } else {
            Some(GroupedMoeDispatch::from_model(self.base)?)
        };
        let disp: Option<&dyn MoeDispatch> = dispatch.as_ref().map(|d| d as &dyn MoeDispatch);

        if use_draft {
            let mut session =
                MtpMoeEagerDecodeSession::start(self.base, self.mtp, k, prompt, max_seq, disp)?;
            let anchor = session.anchor();
            let mut generated: Vec<u32> = vec![anchor];
            if self.stop_ids.contains(&anchor) {
                session.stats.stop_token = Some(anchor);
                return Ok((generated, session.stats));
            }
            'rounds: while generated.len() < max_new && session.round_fits() {
                let emitted = session.round()?;
                for &t in &emitted {
                    generated.push(t);
                    if self.stop_ids.contains(&t) {
                        session.stats.stop_token = Some(t);
                        break 'rounds;
                    }
                    if generated.len() >= max_new {
                        break 'rounds;
                    }
                }
            }
            return Ok((generated, session.stats));
        }

        let seq = prompt.len();
        let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device)?;
        let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)?;
        let (logits, hidden) = self.base.forward_with_cache_dispatched_hidden(
            &tokens,
            &pos,
            &mut cache,
            PREFILL_ROUTES_HOST_BECAUSE_THE_GROUPED_MULTI_TOKEN_COST_EXPLODES_WITH_M,
        )?;
        let mut anchor = self.argmax_last(&logits)?;
        drop(logits);
        drop(hidden);

        if disp.is_some() && seq + (k + 1) <= max_seq {
            let l0 = cache.current_len();
            let snap0 = cache.snapshot_lin_states()?;
            let warm: Vec<u32> = vec![anchor; k + 1];
            let wp: Vec<i32> = (0..(k + 1) as i32).map(|i| l0 as i32 + i).collect();
            let wt = Tensor::from_vec(warm, (1usize, k + 1), &device)?;
            let wpt = Tensor::from_vec(wp, k + 1, &device)?;
            let _ = self
                .base
                .forward_with_cache_dispatched(&wt, &wpt, &mut cache, disp)?;
            cache.restore_lin_states(&snap0)?;
            cache.set_current_len(l0);
            let _ = device.synchronize();
        }

        let mut generated: Vec<u32> = vec![anchor];
        if self.stop_ids.contains(&anchor) {
            stats.stop_token = Some(anchor);
            return Ok((generated, stats));
        }

        let _ = device.synchronize();
        while generated.len() < max_new {
            let round_t0 = std::time::Instant::now();
            let l = cache.current_len();
            let bt = Tensor::from_vec(vec![anchor], (1usize, 1usize), &device)?;
            let bp = Tensor::from_vec(vec![l as i32], 1usize, &device)?;
            let (vlogits, _vh) = self
                .base
                .forward_with_cache_dispatched_hidden(&bt, &bp, &mut cache, disp)?;
            anchor = self.argmax_last(&vlogits)?;
            generated.push(anchor);
            stats.rounds += 1;
            stats.emitted += 1;
            stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            if self.stop_ids.contains(&anchor) {
                stats.stop_token = Some(anchor);
                break;
            }
        }
        Ok((generated, stats))
    }
}

#[cfg(test)]
mod adaptive_p_min_cut {
    use super::*;

    #[test]
    fn p_min_parser_accepts_only_the_open_unit_interval() {
        assert_eq!(q36_mtp_draft_p_min(None), None);
        assert_eq!(q36_mtp_draft_p_min(Some("")), None);
        assert_eq!(q36_mtp_draft_p_min(Some("0")), None);
        assert_eq!(q36_mtp_draft_p_min(Some("1")), None);
        assert_eq!(q36_mtp_draft_p_min(Some("1.5")), None);
        assert_eq!(q36_mtp_draft_p_min(Some("-0.3")), None);
        assert_eq!(q36_mtp_draft_p_min(Some("junk")), None);
        assert_eq!(q36_mtp_draft_p_min(Some(" 0.75 ")), Some(0.75));
    }

    #[test]
    fn leading_confident_run_keeps_at_least_one_and_at_most_the_drafted_len() {
        let cut = q36_adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run;
        assert_eq!(cut(&[0.9, 0.9, 0.9], 0.75), 3);
        assert_eq!(cut(&[0.9, 0.5, 0.9], 0.75), 1);
        assert_eq!(cut(&[0.5, 0.9, 0.9], 0.75), 1);
        assert_eq!(cut(&[0.9, 0.9, 0.5], 0.75), 2);
        assert_eq!(cut(&[0.2], 0.75), 1);
        assert_eq!(cut(&[], 0.75), 1);
    }
}
