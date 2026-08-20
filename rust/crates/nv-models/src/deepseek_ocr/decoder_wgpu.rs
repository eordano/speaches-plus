use anyhow::{Context, Result};
use candle_core::DType;
use half::bf16;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_weights::WeightLoader;

use super::decoder::DeepseekOcrDecoderConfig;

pub struct WgpuLin {
    w: Vec<u16>,
    n: usize,
    k: usize,
}

pub struct WgpuMlp {
    gate: WgpuLin,
    up: WgpuLin,
    down: WgpuLin,
}

enum WgpuFf {
    Dense(WgpuMlp),
    Moe {
        gate_f32: Vec<f32>,
        experts: Vec<WgpuMlp>,
        shared: WgpuMlp,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    },
}

struct WgpuLayer {
    input_ln: Vec<u16>,
    post_ln: Vec<u16>,
    q: WgpuLin,
    k: WgpuLin,
    v: WgpuLin,
    o: WgpuLin,
    ff: WgpuFf,
}

pub struct DeepseekOcrDecoderWgpu {
    cfg: DeepseekOcrDecoderConfig,
    ctx: &'static WgpuContext,
    embed: Vec<u16>,
    layers: Vec<WgpuLayer>,
    final_norm: Vec<u16>,
    lm_head: WgpuLin,
    cos: Vec<f32>,
    sin: Vec<f32>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    pos: usize,
    max_seq: usize,
}

fn to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
}

fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}

fn host_vec_f32(weights: &WeightLoader, name: &str, expect: &[usize]) -> Result<Vec<f32>> {
    let t = weights
        .get(name, DType::F32)
        .with_context(|| format!("load {name}"))?;
    anyhow::ensure!(
        t.dims() == expect,
        "{name}: expected {:?}, got {:?}",
        expect,
        t.dims()
    );
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

fn host_lin(weights: &WeightLoader, name: &str, n: usize, k: usize) -> Result<WgpuLin> {
    let v = host_vec_f32(weights, name, &[n, k])?;
    Ok(WgpuLin {
        w: to_bf16(&v),
        n,
        k,
    })
}

fn host_norm(weights: &WeightLoader, name: &str, hidden: usize) -> Result<Vec<u16>> {
    let v = host_vec_f32(weights, name, &[hidden])?;
    Ok(to_bf16(&v))
}

fn host_mlp(weights: &WeightLoader, prefix: &str, hidden: usize, inter: usize) -> Result<WgpuMlp> {
    Ok(WgpuMlp {
        gate: host_lin(
            weights,
            &format!("{prefix}.gate_proj.weight"),
            inter,
            hidden,
        )?,
        up: host_lin(weights, &format!("{prefix}.up_proj.weight"), inter, hidden)?,
        down: host_lin(
            weights,
            &format!("{prefix}.down_proj.weight"),
            hidden,
            inter,
        )?,
    })
}

impl DeepseekOcrDecoderWgpu {
    pub fn from_loader(
        cfg: DeepseekOcrDecoderConfig,
        weights: &WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        let ctx =
            WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu adapter unavailable: {e}"))?;
        let h = cfg.hidden_size;
        anyhow::ensure!(
            h.is_multiple_of(2),
            "hidden_size must be even for bf16 kernels, got {h}"
        );
        anyhow::ensure!(
            cfg.head_dim().is_multiple_of(2) && cfg.head_dim() <= wk::attn_decode::MAX_HEAD_DIM,
            "head_dim {} unsupported by wgpu attn_decode (max {})",
            cfg.head_dim(),
            wk::attn_decode::MAX_HEAD_DIM
        );
        let embed_v = host_vec_f32(weights, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
        let embed = to_bf16(&embed_v);
        drop(embed_v);
        let qd = cfg.num_attention_heads * cfg.head_dim();
        let kvd = cfg.num_key_value_heads * cfg.head_dim();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let ff = if cfg.is_moe_layer(i) {
                let gate_f32 = host_vec_f32(
                    weights,
                    &format!("{p}.mlp.gate.weight"),
                    &[cfg.n_routed_experts, h],
                )?;
                let mut experts = Vec::with_capacity(cfg.n_routed_experts);
                for e in 0..cfg.n_routed_experts {
                    experts.push(host_mlp(
                        weights,
                        &format!("{p}.mlp.experts.{e}"),
                        h,
                        cfg.moe_intermediate_size,
                    )?);
                }
                let shared = host_mlp(
                    weights,
                    &format!("{p}.mlp.shared_experts"),
                    h,
                    cfg.shared_expert_intermediate_size(),
                )?;
                WgpuFf::Moe {
                    gate_f32,
                    experts,
                    shared,
                    top_k: cfg.num_experts_per_tok,
                    norm_topk_prob: cfg.norm_topk_prob,
                    routed_scaling_factor: cfg.routed_scaling_factor as f32,
                }
            } else {
                WgpuFf::Dense(host_mlp(
                    weights,
                    &format!("{p}.mlp"),
                    h,
                    cfg.intermediate_size,
                )?)
            };
            layers.push(WgpuLayer {
                input_ln: host_norm(weights, &format!("{p}.input_layernorm.weight"), h)?,
                post_ln: host_norm(weights, &format!("{p}.post_attention_layernorm.weight"), h)?,
                q: host_lin(weights, &format!("{p}.self_attn.q_proj.weight"), qd, h)?,
                k: host_lin(weights, &format!("{p}.self_attn.k_proj.weight"), kvd, h)?,
                v: host_lin(weights, &format!("{p}.self_attn.v_proj.weight"), kvd, h)?,
                o: host_lin(weights, &format!("{p}.self_attn.o_proj.weight"), h, qd)?,
                ff,
            });
        }
        let final_norm = host_norm(weights, "model.norm.weight", h)?;
        let lm_head = host_lin(weights, "lm_head.weight", cfg.vocab_size, h)?;
        let max_seq = max_seq.min(cfg.max_position_embeddings);
        let half = cfg.head_dim() / 2;
        let mut cos = vec![0f32; max_seq * half];
        let mut sin = vec![0f32; max_seq * half];
        for p in 0..max_seq {
            for i in 0..half {
                let inv = 1.0f32 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.head_dim() as f32);
                let theta = p as f32 * inv;
                cos[p * half + i] = theta.cos();
                sin[p * half + i] = theta.sin();
            }
        }
        let kv = (0..cfg.num_hidden_layers)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        Ok(Self {
            cfg,
            ctx,
            embed,
            layers,
            final_norm,
            lm_head,
            cos,
            sin,
            kv,
            pos: 0,
            max_seq,
        })
    }

    pub fn config(&self) -> &DeepseekOcrDecoderConfig {
        &self.cfg
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        for (k, v) in self.kv.iter_mut() {
            k.clear();
            v.clear();
        }
        self.pos = 0;
    }

    fn gemv(&self, lin: &WgpuLin, x: &[u16]) -> Result<Vec<u16>> {
        let mut y = vec![0u16; lin.n];
        let limit = self
            .ctx
            .caps
            .max_storage_buffer_binding_size
            .clamp(1 << 20, 1 << 28) as usize;
        let chunk = (limit / (lin.k * 2)).max(1);
        let mut r0 = 0usize;
        while r0 < lin.n {
            let rows = chunk.min(lin.n - r0);
            wk::gemv_bf16::gemv_bf16(
                self.ctx,
                &lin.w[r0 * lin.k..(r0 + rows) * lin.k],
                x,
                &mut y[r0..r0 + rows],
                rows,
                lin.k,
            )
            .map_err(|e| anyhow::anyhow!("gemv_bf16 [{}x{}]: {e}", lin.n, lin.k))?;
            r0 += rows;
        }
        Ok(y)
    }

    fn rmsnorm(&self, x: &[u16], w: &[u16]) -> Result<Vec<u16>> {
        let h = self.cfg.hidden_size;
        let mut y = vec![0u16; h];
        wk::rmsnorm::rmsnorm_bf16(self.ctx, x, w, &mut y, 1, h, self.cfg.rms_norm_eps as f32)
            .map_err(|e| anyhow::anyhow!("rmsnorm_bf16: {e}"))?;
        Ok(y)
    }

    fn silu_mul(&self, gate: &[u16], up: &[u16]) -> Result<Vec<u16>> {
        let mut y = vec![0u16; gate.len()];
        wk::silu::silu_mul_bf16(self.ctx, gate, up, &mut y, gate.len())
            .map_err(|e| anyhow::anyhow!("silu_mul_bf16: {e}"))?;
        Ok(y)
    }

    fn mlp(&self, m: &WgpuMlp, xn: &[u16]) -> Result<Vec<f32>> {
        let g = self.gemv(&m.gate, xn)?;
        let u = self.gemv(&m.up, xn)?;
        let a = self.silu_mul(&g, &u)?;
        Ok(to_f32(&self.gemv(&m.down, &a)?))
    }

    fn route(
        &self,
        gate_f32: &[f32],
        top_k: usize,
        norm: bool,
        scale: f32,
        xn: &[f32],
    ) -> Vec<(usize, f32)> {
        let h = self.cfg.hidden_size;
        let n_exp = gate_f32.len() / h;
        let mut scores = vec![0f32; n_exp];
        for (e, s) in scores.iter_mut().enumerate() {
            let row = &gate_f32[e * h..(e + 1) * h];
            *s = row.iter().zip(xn.iter()).map(|(a, b)| a * b).sum();
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= z;
        }
        let mut order: Vec<usize> = (0..n_exp).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sel = &order[..top_k];
        let denom = if norm {
            sel.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20)
        } else {
            1.0
        };
        sel.iter().map(|&e| (e, probs[e] / denom * scale)).collect()
    }

    fn ff(&self, ff: &WgpuFf, n2: &[u16]) -> Result<Vec<f32>> {
        match ff {
            WgpuFf::Dense(m) => self.mlp(m, n2),
            WgpuFf::Moe {
                gate_f32,
                experts,
                shared,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
            } => {
                let n2_f32 = to_f32(n2);
                let picks = self.route(
                    gate_f32,
                    *top_k,
                    *norm_topk_prob,
                    *routed_scaling_factor,
                    &n2_f32,
                );
                let mut acc = self.mlp(shared, n2)?;
                for (e, w) in picks {
                    let y = self.mlp(&experts[e], n2)?;
                    for (a, b) in acc.iter_mut().zip(y.iter()) {
                        *a += w * b;
                    }
                }
                Ok(acc)
            }
        }
    }

    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
        anyhow::ensure!(
            (token as usize) < self.cfg.vocab_size,
            "token {token} out of vocab"
        );
        anyhow::ensure!(self.pos < self.max_seq, "kv cache full at {}", self.pos);
        let h = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let half = hd / 2;
        let p = self.pos;

        let mut x_bf16 = vec![0u16; h];
        wk::gather_rows_bf16::gather_rows_bf16(
            self.ctx,
            &self.embed,
            &[token as i32],
            &mut x_bf16,
            1,
            h,
            self.cfg.vocab_size,
        )
        .map_err(|e| anyhow::anyhow!("gather_rows_bf16: {e}"))?;
        let mut x = to_f32(&x_bf16);

        for li in 0..self.layers.len() {
            let xn = self.rmsnorm(&to_bf16(&x), &self.layers[li].input_ln)?;
            let mut q = self.gemv(&self.layers[li].q, &xn)?;
            let mut k = self.gemv(&self.layers[li].k, &xn)?;
            let v = self.gemv(&self.layers[li].v, &xn)?;
            let cos_row = &self.cos[p * half..(p + 1) * half];
            let sin_row = &self.sin[p * half..(p + 1) * half];
            wk::rope_bf16::rope_bf16(
                self.ctx,
                &mut q,
                &mut k,
                cos_row,
                sin_row,
                &[0],
                1,
                heads,
                kv_heads,
                hd,
            )
            .map_err(|e| anyhow::anyhow!("rope_bf16: {e}"))?;
            {
                let (kc, vc) = &mut self.kv[li];
                kc.extend(to_f32(&k));
                vc.extend(to_f32(&v));
            }
            let q_f32 = to_f32(&q);
            let mut attn = vec![0f32; heads * hd];
            let (kc, vc) = &self.kv[li];
            wk::attn_decode::attn_decode_f32(
                self.ctx,
                &q_f32,
                kc,
                vc,
                &mut attn,
                heads,
                kv_heads,
                hd,
                0,
                p + 1,
                1.0 / (hd as f32).sqrt(),
            )
            .map_err(|e| anyhow::anyhow!("attn_decode_f32: {e}"))?;
            let o = to_f32(&self.gemv(&self.layers[li].o, &to_bf16(&attn))?);
            for (a, b) in x.iter_mut().zip(o.iter()) {
                *a += b;
            }
            let n2 = self.rmsnorm(&to_bf16(&x), &self.layers[li].post_ln)?;
            let ffo = self.ff(&self.layers[li].ff, &n2)?;
            for (a, b) in x.iter_mut().zip(ffo.iter()) {
                *a += b;
            }
        }
        self.pos += 1;
        let hn = self.rmsnorm(&to_bf16(&x), &self.final_norm)?;
        Ok(to_f32(&self.gemv(&self.lm_head, &hn)?))
    }

    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "empty token sequence");
        let mut last = Vec::new();
        for &t in tokens {
            last = self.forward_token(t)?;
        }
        Ok(last)
    }

    pub fn greedy_token(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        best as u32
    }
}
