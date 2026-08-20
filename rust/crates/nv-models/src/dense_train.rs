use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::ops::{log_softmax, softmax_last_dim};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::Rope;
use nv_weights::TensorSource;

use crate::gemma4_moe::build_rope;
use crate::gemma4::{
    mlp_forward, Gemma4Attention, Gemma4Config, Gemma4Layer, Gemma4Mlp, LayerType,
};

pub fn max_layers_from_env(num_hidden_layers: usize) -> usize {
    match std::env::var("NV_TRAIN_MAX_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) if n > 0 => n.min(num_hidden_layers),
        _ => num_hidden_layers,
    }
}

pub struct DenseTrainModel {
    config: Gemma4Config,
    embed_weight: Tensor,
    layers: Vec<Gemma4Layer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    sliding_rope: Rope,
    full_rope: Rope,
    embed_scale: f32,
    device: Device,

    full_num_layers: usize,
}

impl DenseTrainModel {
    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn layers(&self) -> &[Gemma4Layer] {
        &self.layers
    }
    pub fn full_num_layers(&self) -> usize {
        self.full_num_layers
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn from_loader_dtype<S: TensorSource>(
        config: Gemma4Config,
        weights: &S,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let full_num_layers = config.num_hidden_layers;
        let n_build = max_layers_from_env(full_num_layers);

        let embed_name = "model.language_model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_name, dtype)
            .with_context(|| format!("load {embed_name}"))?
            .to_device(device)?;
        let embed_dims = embed_weight.dims();
        if embed_dims != [config.vocab_size, config.hidden_size] {
            anyhow::bail!(
                "dense-gemma4 embed: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                embed_dims
            );
        }

        let mut layers = Vec::with_capacity(n_build);
        for i in 0..n_build {
            layers.push(load_dense_layer(&config, i, weights, device, dtype)?);
        }

        let final_norm = load_rmsnorm(weights, "model.language_model.norm.weight", config.hidden_size, config.rms_norm_eps, dtype, device)?;
        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", dtype)
                .context("load lm_head.weight")?
                .to_device(device)?
        };
        let lm_head = Linear::new_no_pretranspose(lm_head_weight, None)?;

        let sliding_rope = build_rope(
            config.head_dim,
            config.rope_theta_for(LayerType::SlidingAttention),
            1.0,
            config.max_position_embeddings,
            device,
        )?;
        let full_rope = build_rope(
            config.global_head_dim,
            config.rope_theta_for(LayerType::FullAttention),
            config.rope_partial_factor_for(LayerType::FullAttention),
            config.max_position_embeddings,
            device,
        )?;
        let embed_scale = (config.hidden_size as f32).sqrt();

        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            sliding_rope,
            full_rope,
            embed_scale,
            device: device.clone(),
            full_num_layers,
        })
    }

    pub fn forward_logits(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] == 0 {
            anyhow::bail!("dense forward: tokens must be [batch, seq], got {:?}", dims);
        }
        let bsz = dims[0];
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "dense forward: positions must be [{}] shared by every same-length row, got {:?}",
                seq,
                positions.dims()
            );
        }
        let row_positions = if bsz == 1 {
            positions.clone()
        } else {
            positions
                .unsqueeze(0)?
                .broadcast_as((bsz, seq))?
                .contiguous()?
                .flatten_all()?
        };

        let mut hidden = self.embed_forward(tokens, seq)?;

        for li in 0..self.layers.len() {
            hidden = self.layer_forward(li, &hidden, &row_positions, seq)?;
        }

        let normed = self.final_norm.forward_candle(&hidden)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        let cap = self.config.final_logit_softcapping;
        if cap > 0.0 {
            Ok(logits
                .affine(1.0 / cap as f64, 0.0)?
                .tanh()?
                .affine(cap as f64, 0.0)?)
        } else {
            Ok(logits)
        }
    }

    fn layer_forward(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];

        let residual_attn = x.clone();
        let normed_pre_attn = layer.input_layernorm.forward_candle(x)?;
        let attn_out = self.attention_forward(idx, &normed_pre_attn, positions, seq)?;
        let attn_post = layer.post_attention_layernorm.forward_candle(&attn_out)?;

        let after_attn = attn_post.add(&residual_attn)?;
        let normed_pre_mlp = layer
            .pre_feedforward_layernorm
            .forward_candle(&after_attn)?;

        let mlp_out = mlp_forward(&layer.mlp, &normed_pre_mlp)?;
        let mlp_post = layer.post_feedforward_layernorm.forward_candle(&mlp_out)?;

        let scale =
            Tensor::new(layer.layer_scalar_host, &self.device)?.to_dtype(after_attn.dtype())?;
        Ok(after_attn.add(&mlp_post)?.broadcast_mul(&scale)?)
    }

    fn attention_forward(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];
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

        let bsz = x.dims()[0];
        let (q_raw, k_raw, v_raw) = self.qkv_split(attn, x)?;
        let q = q_raw.reshape((bsz, seq, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward_candle(&q)?;
        let k = k_raw.reshape((bsz, seq, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward_candle(&k)?;
        let v = v_raw.reshape((bsz, seq, n_kv, head_dim))?;
        let v_normed = attn.v_norm.forward_candle(&v)?;

        let (q_rot, k_rot) = rope.apply_candle(&q_normed, &k_normed, positions)?;
        let q_rot = q_rot.contiguous()?;
        let k_rot = k_rot.contiguous()?;
        let v_for = v_normed.contiguous()?;

        let attn_out = naive_sdpa(&q_rot, &k_rot, &v_for, bsz, n_q, n_kv, head_dim, seq, window)?;
        let attn_out_flat = attn_out
            .to_dtype(x.dtype())?
            .reshape((bsz, seq, n_q * head_dim))?;
        attn.o_proj.forward(&attn_out_flat)
    }

    fn qkv_split(&self, attn: &Gemma4Attention, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if x.device().is_cuda() {
            let fused = attn.qkv_proj.forward(x)?;
            let q = fused.narrow(D::Minus1, 0, attn.q_dim)?.contiguous()?;
            let k = fused
                .narrow(D::Minus1, attn.q_dim, attn.kv_dim)?
                .contiguous()?;
            let v = if attn.has_v {
                fused
                    .narrow(D::Minus1, attn.q_dim + attn.kv_dim, attn.kv_dim)?
                    .contiguous()?
            } else {
                k.clone()
            };
            Ok((q, k, v))
        } else {
            attn.qkv_forward(x)
        }
    }

    fn embed_forward(&self, tokens: &Tensor, seq: usize) -> Result<Tensor> {
        let bsz = tokens.dims()[0];
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = self
            .embed_weight
            .detach()
            .index_select(&tokens_flat, 0)?
            .reshape((bsz, seq, self.config.hidden_size))?
            .to_dtype(DType::F32)?;
        Ok(x.affine(self.embed_scale as f64, 0.0)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    window: Option<usize>,
) -> Result<Tensor> {
    let device = q.device().clone();
    let stored = k.dims()[1];
    let q3 = q
        .to_dtype(DType::F32)?
        .reshape((batch, seq, n_q, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * n_q, seq, head_dim))?;
    let k3 = k
        .to_dtype(DType::F32)?
        .reshape((batch, stored, n_kv, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * n_kv, stored, head_dim))?;
    let v3 = v
        .to_dtype(DType::F32)?
        .reshape((batch, stored, n_kv, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * n_kv, stored, head_dim))?;

    let group = n_q / n_kv;
    let map: Vec<u32> = (0..batch * n_q)
        .map(|i| ((i / n_q) * n_kv + (i % n_q) / group) as u32)
        .collect();
    let map_t = Tensor::from_vec(map, batch * n_q, &device)?;
    let k3e = k3.index_select(&map_t, 0)?.contiguous()?;
    let v3e = v3.index_select(&map_t, 0)?.contiguous()?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut mask = vec![0f32; seq * stored];
    let key_base = stored - seq;
    for i in 0..seq {
        let qpos = (key_base + i) as i64;
        for j in 0..stored {
            let kpos = j as i64;
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

    let raw_scores = q3
        .matmul(&k3e.transpose(1, 2)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let scores = raw_scores.broadcast_add(&mask_t)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v3e)?;
    Ok(out
        .reshape((batch, n_q, seq, head_dim))?
        .transpose(1, 2)?
        .contiguous()?)
}

fn load_dense_layer<S: TensorSource>(
    config: &Gemma4Config,
    idx: usize,
    weights: &S,
    device: &Device,
    dtype: DType,
) -> Result<Gemma4Layer> {
    let prefix = format!("model.language_model.layers.{idx}");
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let kind = config.layer_kind(idx);
    let head_dim = config.head_dim_for(kind);
    let n_q = config.num_attention_heads;
    let n_kv = config.num_kv_heads_for(kind);
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;

    let norm = |name: &str, dim: usize| -> Result<RmsNorm> {
        load_rmsnorm(weights, &format!("{prefix}.{name}.weight"), dim, eps, dtype, device)
    };
    let input_layernorm = norm("input_layernorm", hidden)?;
    let post_attention_layernorm = norm("post_attention_layernorm", hidden)?;
    let pre_feedforward_layernorm = norm("pre_feedforward_layernorm", hidden)?;
    let post_feedforward_layernorm = norm("post_feedforward_layernorm", hidden)?;

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
        (kind, config.attention_k_eq_v),
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
    let qkv_proj = Linear::new_no_pretranspose(fused, None)?;
    let o_proj = Linear::new_no_pretranspose(get_proj("o_proj", hidden, q_dim)?, None)?;
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
        gate_up_proj: Linear::new_no_pretranspose(
            Tensor::cat(&[&gate_w, &up_w], 0)?.contiguous()?,
            None,
        )?,
        down_proj: Linear::new_no_pretranspose(down_w, None)?,
    };

    Ok(Gemma4Layer {
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

fn load_rmsnorm<S: TensorSource>(
    weights: &S,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
    device: &Device,
) -> Result<RmsNorm> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?
        .to_device(device)?;
    if w.dims() != [dim] {
        anyhow::bail!("{name}: expected [{dim}], got {:?}", w.dims());
    }
    Ok(RmsNorm::new(w, eps))
}

pub struct StepGrads {
    pub loss: f32,
    pub grads: Vec<Option<Tensor>>,
}

pub fn lmhead_chunk_from_env() -> usize {
    std::env::var("NV_TRAIN_LMHEAD_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8)
}

impl DenseTrainModel {
    pub fn train_step_checkpointed(
        &self,
        batch: &[Vec<u32>],
        vars: &[Var],
        chunk: usize,
    ) -> Result<StepGrads> {
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut total_loss = 0f32;
        for ids in batch {
            let (loss, seq_grads) = self.seq_backward_checkpointed(ids, vars, chunk)?;
            total_loss += loss;
            for (a, g) in acc.iter_mut().zip(seq_grads) {
                if let Some(g) = g {
                    *a = Some(match a.take() {
                        Some(p) => (p + g)?,
                        None => g,
                    });
                }
            }
        }
        Ok(StepGrads {
            loss: total_loss,
            grads: acc,
        })
    }

    fn seq_backward_checkpointed(
        &self,
        ids: &[u32],
        vars: &[Var],
        chunk: usize,
    ) -> Result<(f32, Vec<Option<Tensor>>)> {
        let dev = &self.device;
        let seq = ids.len();
        if seq < 2 {
            anyhow::bail!("seq_backward_checkpointed: need >=2 tokens, got {seq}");
        }
        let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), dev)?;

        let mut boundaries: Vec<Tensor> = Vec::with_capacity(self.layers.len() + 1);
        boundaries.push(self.embed_forward(&tokens, seq)?.detach());
        for li in 0..self.layers.len() {
            let inp = boundaries[li].clone();
            let out = self.layer_forward(li, &inp, &positions, seq)?.detach();
            boundaries.push(out);
        }
        let h_last = boundaries.last().unwrap().clone();

        let denom = (seq - 1) as f32;
        let targets = &ids[1..];
        let (loss, mut grad_next) = self.head_loss_grad(&h_last, targets, denom, chunk)?;

        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        for li in (0..self.layers.len()).rev() {
            let inp_var = Var::from_tensor(&boundaries[li])?;
            let out = self.layer_forward(li, inp_var.as_tensor(), &positions, seq)?;
            let surrogate = out.broadcast_mul(&grad_next.detach())?.sum_all()?;
            let gs = surrogate.backward()?;
            for (i, v) in vars.iter().enumerate() {
                if let Some(g) = gs.get(v.as_tensor()) {
                    let g = g.clone();
                    acc[i] = Some(match acc[i].take() {
                        Some(p) => (p + g)?,
                        None => g,
                    });
                }
            }
            grad_next = gs
                .get(inp_var.as_tensor())
                .cloned()
                .with_context(|| format!("no grad for layer {li} checkpoint input"))?;
        }
        Ok((loss, acc))
    }

    fn head_loss_grad(
        &self,
        h_last: &Tensor,
        targets: &[u32],
        denom: f32,
        chunk: usize,
    ) -> Result<(f32, Tensor)> {
        let dev = &self.device;
        let hidden = self.config.hidden_size;
        let hv = Var::from_tensor(h_last)?;
        let normed = self.final_norm.forward_candle(hv.as_tensor())?;
        let normed2 = normed.squeeze(0)?;
        let seq = normed2.dim(0)?;
        let pred = normed2.narrow(0, 0, seq - 1)?;
        let (loss, grad_pred) = self.chunked_ce(&pred.detach(), targets, denom, chunk)?;
        let zero_row = Tensor::zeros((1usize, hidden), grad_pred.dtype(), dev)?;
        let grad_normed = Tensor::cat(&[&grad_pred, &zero_row], 0)?.unsqueeze(0)?;
        let surrogate = normed.broadcast_mul(&grad_normed.detach())?.sum_all()?;
        let gs = surrogate.backward()?;
        let grad_hlast = gs
            .get(hv.as_tensor())
            .cloned()
            .context("no grad for final_norm input")?;
        Ok((loss, grad_hlast))
    }

    fn chunked_ce(
        &self,
        pred: &Tensor,
        targets: &[u32],
        denom: f32,
        chunk: usize,
    ) -> Result<(f32, Tensor)> {
        let dev = &self.device;
        let w = self
            .lm_head
            .weight()
            .context("lm_head has no materialised weight")?
            .detach();
        let wdt = w.dtype();
        let wt = w.t()?.contiguous()?;
        let rows = pred.dim(0)?;
        if targets.len() != rows {
            anyhow::bail!("chunked_ce: targets {} != rows {}", targets.len(), rows);
        }
        let cap = self.config.final_logit_softcapping as f64;
        let mut loss = 0f32;
        let mut grad_parts: Vec<Tensor> = Vec::new();
        let mut a = 0usize;
        while a < rows {
            let c = chunk.min(rows - a);
            let hc = pred.narrow(0, a, c)?.to_dtype(wdt)?;
            let raw = hc.matmul(&wt)?.to_dtype(DType::F32)?;
            let z = if cap > 0.0 {
                raw.affine(1.0 / cap, 0.0)?.tanh()?.affine(cap, 0.0)?
            } else {
                raw
            };
            let idx = Tensor::from_vec(targets[a..a + c].to_vec(), (c, 1usize), dev)?;
            let logp = log_softmax(&z, D::Minus1)?;
            let picked = logp.gather(&idx, 1)?;
            loss += picked
                .affine(-1.0 / denom as f64, 0.0)?
                .sum_all()?
                .to_scalar::<f32>()?;

            let probs = softmax_last_dim(&z)?;
            let neg_one = Tensor::from_vec(vec![-1f32; c], (c, 1usize), dev)?;
            let gz = probs.scatter_add(&idx, &neg_one, 1)?;
            let mut gz = gz.affine(1.0 / denom as f64, 0.0)?;
            if cap > 0.0 {
                let zc = z.affine(1.0 / cap, 0.0)?;
                let dcap = (Tensor::ones_like(&zc)? - zc.sqr()?)?;
                gz = gz.mul(&dcap)?;
            }

            let ghc = gz.to_dtype(wdt)?.matmul(&w)?.to_dtype(DType::F32)?;
            grad_parts.push(ghc);
            a += c;
        }
        let grad = Tensor::cat(&grad_parts, 0)?;
        Ok((loss, grad))
    }
}
