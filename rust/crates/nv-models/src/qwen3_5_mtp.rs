use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;

use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::moe_bf16_grouped::Bf16GroupedExperts;
use nv_layers::norm::RmsNorm;

use crate::qwen3_5_moe::{AttentionLayer, Qwen3Moe, Qwen3MoeConfig};
#[cfg(feature = "cuda")]
use crate::qwen3_5_moe::{GroupedMoeDispatch, MoeDispatch};

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

pub struct MtpHead {
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc: Linear,
    input_layernorm: RmsNorm,
    self_attn: AttentionLayer,
    post_attention_layernorm: RmsNorm,
    moe: MtpMoe,
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

        let moe = MtpMoe {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: cfg.num_experts_per_tok,
        };

        let norm_final = norm("mtp.norm.weight", hidden)?;

        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
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
            .moe
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

    fn hidden_at(&self, hidden: &Tensor, idx: usize) -> Result<Tensor> {
        Ok(hidden.narrow(1, idx, 1)?.contiguous()?)
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

        let seq = prompt.len();
        let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device)?;
        let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)?;
        let (logits, hidden) = self
            .base
            .forward_with_cache_dispatched_hidden(&tokens, &pos, &mut cache, disp)?;
        let mut anchor = self.argmax_last(&logits)?;
        let mut base_hidden = self.hidden_at(&hidden, seq - 1)?;
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
        if !use_draft {
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
            return Ok((generated, stats));
        }
        while generated.len() < max_new {
            let round_t0 = std::time::Instant::now();
            let l = cache.current_len();

            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            if use_draft {
                let mut h = base_hidden.clone();
                let mut tok = anchor;
                for j in 0..k {
                    let position = (l + j) as i32;
                    let (d, dhidden) = self.mtp.forward_draft_tok(self.base, &h, tok, position)?;
                    drafts.push(d);
                    tok = d;
                    h = dhidden;
                }
            } else {
                drafts = vec![anchor; k];
            }
            let _ = device.synchronize();
            let draft_dt = round_t0.elapsed().as_secs_f64();

            let verify_t0 = std::time::Instant::now();
            let mut block: Vec<u32> = Vec::with_capacity(k + 1);
            block.push(anchor);
            block.extend_from_slice(&drafts);
            let m = block.len();
            let block_pos: Vec<i32> = (0..m as i32).map(|i| l as i32 + i).collect();
            let bt = Tensor::from_vec(block.clone(), (1usize, m), &device)?;
            let bp = Tensor::from_vec(block_pos, m, &device)?;

            cache.set_capture_lin_ckpts(true);
            let (vlogits, vhidden) = self
                .base
                .forward_with_cache_dispatched_hidden(&bt, &bp, &mut cache, disp)?;

            let greedy: Vec<u32> = vlogits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1()?;

            let _ = device.synchronize();
            let verify_dt = verify_t0.elapsed().as_secs_f64();
            let commit_t0 = std::time::Instant::now();

            let mut accepted = 0usize;
            if use_draft {
                while accepted < drafts.len() && drafts[accepted] == greedy[accepted] {
                    accepted += 1;
                }
            }
            let bonus = greedy[accepted];
            let consumed = accepted + 1;

            let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
            emitted.push(bonus);

            base_hidden = self.hidden_at(&vhidden, accepted)?;

            cache.set_current_len(l + consumed);
            cache.rollback_lin_to(consumed)?;
            anchor = bonus;

            let _ = device.synchronize();
            let commit_dt = commit_t0.elapsed().as_secs_f64();
            stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            stats.draft_ms += 1000.0 * draft_dt;
            stats.verify_ms += 1000.0 * verify_dt;
            stats.commit_ms += 1000.0 * commit_dt;
            stats.rounds += 1;
            stats.drafted += drafts.len();
            stats.accepted += accepted;
            stats.emitted += emitted.len();
            if accepted > 0 {
                stats.pos0_accepted += 1;
            }
            *stats.accept_len_hist.entry(accepted).or_insert(0) += 1;

            for (i, &t) in emitted.iter().enumerate() {
                generated.push(t);
                if self.stop_ids.contains(&t) {
                    stats.stop_token = Some(t);
                    return Ok((generated, stats));
                }
                if generated.len() >= max_new {
                    let _ = i;
                    return Ok((generated, stats));
                }
            }
        }
        Ok((generated, stats))
    }
}
