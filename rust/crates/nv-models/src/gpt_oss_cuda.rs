use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::attn::{sdpa_with_sinks, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;

use crate::gpt_oss::{
    bf16_val, load_bf16, load_layer, yarn_inv_freq, GptOssConfig, GptOssLayerType, HostBf16Lin,
    HostLayer, HostMxStack, HostWeights, SWIGLU_ALPHA,
};

pub const DEQUANT_TO_BF16_AT_LOAD_IS_THE_FIRST_CUDA_RUNG_AND_COSTS_28_GB_OVER_MXFP4: &str =
    "the cuda gpt-oss decoder dequantizes every mxfp4 expert tensor to bf16 while loading, so it \
     needs no new cuda GEMM: nv_quant::mxfp4::Mxfp4Tensor::dequantize is the same host semantics \
     the wgpu mxfp4 GEMV is already pinned against, and every e2m1 value times its e8m0 block \
     scale is exact in bf16, so the dequant itself loses nothing. It is not free in VRAM. For \
     openai/gpt-oss-20b (24 layers, hidden 2880, 32 experts, expert intermediate 2880) the expert \
     weights are 24*32*(5760*2880 + 2880*2880) = 19.11e9 parameters: 38.2 GB resident as bf16 \
     against roughly 10.2 GB in mxfp4 at 17 bytes per 32 weights. With attention (about 0.64e9 \
     parameters, 1.3 GB, already bf16 on disk) and embed plus lm_head (2*201088*2880 = 1.16e9 \
     parameters, 2.3 GB) the resident total is about 41.8 GB against about 13.7 GB for a native \
     mxfp4 path -- roughly 28 GB more, which fits the 96 GiB sm_120 card alongside a KV cache but \
     leaves much less room for one. A native cuda mxfp4 GEMV is the follow-up rung, and this path \
     is what it must reproduce.";

pub const GPT_OSS_CUDA_ATTENTION_IS_EAGER_BECAUSE_FLASH_CANNOT_CARRY_SINKS: &str =
    "every gpt-oss attention layer folds a learned per-head sink logit into its softmax, and \
     candle_flash_attn is a closed kernel with no argument for one, so the cuda decoder runs \
     nv_layers::attn::sdpa_with_sinks (eager scores, softmax_last_dim over an appended sink \
     column, an appended zero value row) for both prefill and decode. The alternating \
     sliding_window=128 layers are masked in the same builder rather than by \
     flash_attn_windowed. Carrying sinks natively in attn_decode.cu / flash_decode.cu is the \
     follow-up rung; until it exists this path is the reference the kernel must match.";

pub struct GptOssKvCache {
    layers: Vec<(Tensor, Tensor)>,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl GptOssKvCache {
    pub fn new(cfg: &GptOssConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        let shape = (1usize, max_seq_len, cfg.num_key_value_heads, cfg.head_dim);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push((
                Tensor::zeros(shape, DType::BF16, device)?,
                Tensor::zeros(shape, DType::BF16, device)?,
            ));
        }
        Ok(Self {
            layers,
            current_len: 0,
            max_seq_len,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
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

    fn write_at(&mut self, layer: usize, start: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let t = k_new.dims()[1];
        let end = start + t;
        anyhow::ensure!(
            end <= self.max_seq_len,
            "GptOssKvCache: write end {end} exceeds max_seq_len {}",
            self.max_seq_len
        );
        let (k_buf, v_buf) = &self.layers[layer];
        let span = [0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim];
        let k_updated = k_buf.slice_assign(&span, k_new)?;
        let v_updated = v_buf.slice_assign(&span, v_new)?;
        self.layers[layer] = (k_updated, v_updated);
        Ok(())
    }

    fn view(&self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let (k, v) = &self.layers[layer];
        Ok((k.narrow(1, 0, len)?, v.narrow(1, 0, len)?))
    }
}

struct GptOssCudaExperts {
    gate_t: Tensor,
    up_t: Tensor,
    down_t: Tensor,
    gate_bias: Tensor,
    up_bias: Tensor,
    down_bias: Tensor,
    num_experts: usize,
    hidden_size: usize,
}

impl GptOssCudaExperts {
    fn from_host(gate_up: &HostMxStack, down: &HostMxStack, device: &Device) -> Result<Self> {
        let e = gate_up.e;
        let hidden = gate_up.k;
        let inter = down.k;
        anyhow::ensure!(
            gate_up.n == 2 * inter,
            "gate_up rows {} != 2*intermediate {}",
            gate_up.n,
            2 * inter
        );
        anyhow::ensure!(
            down.n == hidden && down.e == e,
            "down stack [{}, {}, {}] does not pair with gate_up hidden {hidden} experts {e}",
            down.e,
            down.n,
            down.k
        );

        let mut gate_t = vec![half::bf16::ZERO; e * hidden * inter];
        let mut up_t = vec![half::bf16::ZERO; e * hidden * inter];
        let mut down_t = vec![half::bf16::ZERO; e * inter * hidden];
        for ex in 0..e {
            let gu = gate_up.expert(ex).dequantize();
            for i in 0..inter {
                let g = &gu[2 * i];
                let u = &gu[2 * i + 1];
                for c in 0..hidden {
                    gate_t[(ex * hidden + c) * inter + i] = half::bf16::from_f32(g[c]);
                    up_t[(ex * hidden + c) * inter + i] = half::bf16::from_f32(u[c]);
                }
            }
            let dn = down.expert(ex).dequantize();
            for (r, row) in dn.iter().enumerate() {
                for (i, w) in row.iter().enumerate().take(inter) {
                    down_t[(ex * inter + i) * hidden + r] = half::bf16::from_f32(*w);
                }
            }
        }

        let mut gate_bias = vec![0f32; e * inter];
        let mut up_bias = vec![0f32; e * inter];
        for ex in 0..e {
            for i in 0..inter {
                gate_bias[ex * inter + i] = bf16_val(gate_up.bias[ex * 2 * inter + 2 * i]);
                up_bias[ex * inter + i] = bf16_val(gate_up.bias[ex * 2 * inter + 2 * i + 1]);
            }
        }
        let down_bias: Vec<f32> = down.bias.iter().map(|b| bf16_val(*b)).collect();

        Ok(Self {
            gate_t: Tensor::from_vec(gate_t, (e, hidden, inter), device)?,
            up_t: Tensor::from_vec(up_t, (e, hidden, inter), device)?,
            down_t: Tensor::from_vec(down_t, (e, inter, hidden), device)?,
            gate_bias: Tensor::from_vec(gate_bias, (e, inter), device)?,
            up_bias: Tensor::from_vec(up_bias, (e, inter), device)?,
            down_bias: Tensor::from_vec(down_bias, (e, hidden), device)?,
            num_experts: e,
            hidden_size: hidden,
        })
    }

    fn forward(
        &self,
        x_flat: &Tensor,
        topk_ids: &[usize],
        topk_weights: &[f32],
        k_top: usize,
        swiglu_limit: f32,
    ) -> Result<Tensor> {
        let (n_tokens, hidden) = x_flat.dims2()?;
        anyhow::ensure!(hidden == self.hidden_size, "moe hidden {hidden} mismatch");
        let device = x_flat.device().clone();

        let mut rows_per_expert: Vec<Vec<u32>> = vec![Vec::new(); self.num_experts];
        let mut w_per_expert: Vec<Vec<f32>> = vec![Vec::new(); self.num_experts];
        for n in 0..n_tokens {
            for j in 0..k_top {
                let e = topk_ids[n * k_top + j];
                rows_per_expert[e].push(n as u32);
                w_per_expert[e].push(topk_weights[n * k_top + j]);
            }
        }

        let mut acc = Tensor::zeros((n_tokens, hidden), DType::F32, &device)?;
        for e in 0..self.num_experts {
            let rows = &rows_per_expert[e];
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let idx = Tensor::from_vec(rows.clone(), m, &device)?;
            let xe = x_flat.index_select(&idx, 0)?.contiguous()?;

            let limit = swiglu_limit as f64;
            let gate = xe
                .matmul(&self.gate_t.narrow(0, e, 1)?.squeeze(0)?.contiguous()?)?
                .to_dtype(DType::F32)?
                .broadcast_add(&self.gate_bias.narrow(0, e, 1)?)?
                .minimum(limit)?;
            let up = xe
                .matmul(&self.up_t.narrow(0, e, 1)?.squeeze(0)?.contiguous()?)?
                .to_dtype(DType::F32)?
                .broadcast_add(&self.up_bias.narrow(0, e, 1)?)?
                .clamp(-limit, limit)?;
            let glu = gate.mul(&candle_nn::ops::sigmoid(
                &gate.affine(SWIGLU_ALPHA as f64, 0.0)?,
            )?)?;
            let act = glu
                .mul(&up.affine(1.0, 1.0)?)?
                .to_dtype(DType::BF16)?
                .contiguous()?;

            let ye = act
                .matmul(&self.down_t.narrow(0, e, 1)?.squeeze(0)?.contiguous()?)?
                .to_dtype(DType::F32)?
                .broadcast_add(&self.down_bias.narrow(0, e, 1)?)?;
            let w_t = Tensor::from_vec(w_per_expert[e].clone(), (m, 1), &device)?;
            acc = acc.index_add(&idx, &ye.broadcast_mul(&w_t)?, 0)?;
        }
        Ok(acc.to_dtype(DType::BF16)?)
    }
}

struct GptOssCudaLayer {
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sinks: Tensor,
    router_t: Tensor,
    router_bias: Tensor,
    experts: GptOssCudaExperts,
    layer_type: GptOssLayerType,
}

pub struct GptOssCuda {
    config: GptOssConfig,
    embed_weight: Tensor,
    layers: Vec<GptOssCudaLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    attention_scaling: f64,
    device: Device,
}

impl GptOssCuda {
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<GptOssKvCache> {
        GptOssKvCache::new(&self.config, max_seq_len, &self.device)
    }

    pub fn from_host(config: GptOssConfig, hw: &HostWeights, device: &Device) -> Result<Self> {
        Self::assemble(
            config,
            &hw.embed,
            &hw.final_norm,
            &hw.lm_head,
            |i| Ok(hw.layers[i].clone()),
            device,
        )
    }

    pub fn from_loader(
        config: GptOssConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        let _ = DEQUANT_TO_BF16_AT_LOAD_IS_THE_FIRST_CUDA_RUNG_AND_COSTS_28_GB_OVER_MXFP4;
        let embed = load_bf16(
            weights,
            &["model.embed_tokens.weight"],
            &[config.vocab_size, config.hidden_size],
        )
        .context("load gpt-oss embedding")?;
        let final_norm = load_bf16(weights, &["model.norm.weight"], &[config.hidden_size])
            .context("load gpt-oss final norm")?;
        let lm_head = if config.tie_word_embeddings {
            embed.clone()
        } else {
            load_bf16(
                weights,
                &["lm_head.weight"],
                &[config.vocab_size, config.hidden_size],
            )
            .context("load gpt-oss lm_head")?
        };
        Self::assemble(
            config.clone(),
            &embed,
            &final_norm,
            &lm_head,
            |i| load_layer(&config, weights, i).with_context(|| format!("load gpt-oss layer {i}")),
            device,
        )
    }

    fn assemble<F>(
        config: GptOssConfig,
        embed: &[u16],
        final_norm: &[u16],
        lm_head: &[u16],
        mut layer_at: F,
        device: &Device,
    ) -> Result<Self>
    where
        F: FnMut(usize) -> Result<HostLayer>,
    {
        let _ = GPT_OSS_CUDA_ATTENTION_IS_EAGER_BECAUSE_FLASH_CANNOT_CARRY_SINKS;
        let hidden = config.hidden_size;
        anyhow::ensure!(
            config.layer_types.len() == config.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            config.layer_types.len(),
            config.num_hidden_layers
        );
        let embed_weight = bf16_tensor(embed, (config.vocab_size, hidden), device)?;
        let lm_head = Linear::new(
            bf16_tensor(lm_head, (config.vocab_size, hidden), device)?,
            None,
        )?;
        let final_norm = RmsNorm::new(bf16_tensor(final_norm, hidden, device)?, config.rms_norm_eps);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let hl = layer_at(i)?;
            anyhow::ensure!(
                hl.attn.sinks.len() == config.num_attention_heads,
                "layer {i}: {} sinks for {} heads",
                hl.attn.sinks.len(),
                config.num_attention_heads
            );
            let n_exp = config.num_local_experts;
            let mut router_host = vec![0f32; hidden * n_exp];
            for r in 0..n_exp {
                for c in 0..hidden {
                    router_host[c * n_exp + r] = bf16_val(hl.moe.router.w[r * hidden + c]);
                }
            }
            let router_t = Tensor::from_vec(router_host, (hidden, n_exp), device)?;
            let router_bias = Tensor::from_vec(
                hl.moe
                    .router
                    .bias
                    .iter()
                    .map(|b| bf16_val(*b))
                    .collect::<Vec<f32>>(),
                (1usize, n_exp),
                device,
            )?;
            layers.push(GptOssCudaLayer {
                input_ln: RmsNorm::new(
                    bf16_tensor(&hl.input_ln, hidden, device)?,
                    config.rms_norm_eps,
                ),
                post_attn_ln: RmsNorm::new(
                    bf16_tensor(&hl.post_attn_ln, hidden, device)?,
                    config.rms_norm_eps,
                ),
                q_proj: host_linear(&hl.attn.q, device)?,
                k_proj: host_linear(&hl.attn.k, device)?,
                v_proj: host_linear(&hl.attn.v, device)?,
                o_proj: host_linear(&hl.attn.o, device)?,
                sinks: Tensor::from_vec(hl.attn.sinks.clone(), config.num_attention_heads, device)?,
                router_t,
                router_bias,
                experts: GptOssCudaExperts::from_host(&hl.moe.gate_up, &hl.moe.down, device)?,
                layer_type: config.layer_types[i],
            });
        }

        let rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                base: config.rope_theta,
                kind: RopeKind::Yarn,
            },
            &yarn_inv_freq(&config),
            device,
        )?;
        let attention_scaling = config.attention_scaling() as f64;

        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            rope,
            attention_scaling,
            device: device.clone(),
        })
    }

    pub fn forward_last_logits(
        &self,
        tokens: &[u32],
        positions: &[u32],
        cache: &mut GptOssKvCache,
    ) -> Result<Vec<f32>> {
        let x = self.forward_hidden(tokens, positions, cache)?;
        let seq = tokens.len();
        let last = x.narrow(1, seq - 1, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&last)?.to_dtype(DType::F32)?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }

    pub fn forward_all_logits(
        &self,
        tokens: &[u32],
        positions: &[u32],
        cache: &mut GptOssKvCache,
    ) -> Result<Tensor> {
        let x = self.forward_hidden(tokens, positions, cache)?;
        Ok(self.lm_head.forward(&x)?.to_dtype(DType::F32)?)
    }

    fn forward_hidden(
        &self,
        tokens: &[u32],
        positions: &[u32],
        cache: &mut GptOssKvCache,
    ) -> Result<Tensor> {
        let seq = tokens.len();
        anyhow::ensure!(seq > 0, "GptOssCuda: empty token chunk");
        anyhow::ensure!(
            positions.len() == seq,
            "GptOssCuda: {} positions for {seq} tokens",
            positions.len()
        );
        let write_start = cache.current_len();
        anyhow::ensure!(
            write_start + seq <= cache.max_seq_len(),
            "GptOssCuda: chunk of {seq} at {write_start} overruns the {}-token KV window",
            cache.max_seq_len()
        );

        let tok_t = Tensor::from_vec(tokens.to_vec(), seq, &self.device)?;
        let mut x = self
            .embed_weight
            .index_select(&tok_t, 0)?
            .reshape((1usize, seq, self.config.hidden_size))?;
        let pos_t = Tensor::from_vec(
            positions.iter().map(|p| *p as i32).collect::<Vec<i32>>(),
            (1usize, seq),
            &self.device,
        )?;

        for i in 0..self.layers.len() {
            x = self.layer_forward(i, &x, &pos_t, cache, seq, write_start)?;
        }
        cache.current_len = write_start + seq;
        Ok(self.final_norm.forward(&x)?)
    }

    fn layer_forward(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut GptOssKvCache,
        seq: usize,
        write_start: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];
        let cfg = &self.config;
        let hd = cfg.head_dim;
        let n_h = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;

        let normed = layer.input_ln.forward(x)?;
        let q = layer
            .q_proj
            .forward(&normed)?
            .reshape((1usize, seq, n_h, hd))?;
        let k = layer
            .k_proj
            .forward(&normed)?
            .reshape((1usize, seq, n_kv, hd))?;
        let v = layer
            .v_proj
            .forward(&normed)?
            .reshape((1usize, seq, n_kv, hd))?;

        let (q_rot, k_rot) = self
            .rope
            .apply(&q.to_dtype(DType::F32)?, &k.to_dtype(DType::F32)?, positions)?;
        let q = q_rot.affine(self.attention_scaling, 0.0)?.to_dtype(DType::BF16)?;
        let k = k_rot.affine(self.attention_scaling, 0.0)?.to_dtype(DType::BF16)?;

        cache.write_at(idx, write_start, &k.contiguous()?, &v.contiguous()?)?;
        let total = write_start + seq;
        let (k_full, v_full) = cache.view(idx, total)?;

        let window = match layer.layer_type {
            GptOssLayerType::Sliding => cfg.sliding_window,
            GptOssLayerType::Full => 0,
        };
        let attn_cfg = AttnConfig {
            num_heads: n_h,
            num_kv_heads: n_kv,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: true,
        };
        let attn_out = sdpa_with_sinks(
            &q.contiguous()?,
            &k_full.contiguous()?,
            &v_full.contiguous()?,
            &attn_cfg,
            &layer.sinks,
            window,
        )?
        .reshape((1usize, seq, n_h * hd))?;
        let x = x.add(&layer.o_proj.forward(&attn_out)?)?;

        let normed_post = layer.post_attn_ln.forward(&x)?;
        let flat = normed_post.reshape((seq, cfg.hidden_size))?;
        let (ids, weights) = self.route(layer, &flat, seq)?;
        let moe_out = layer.experts.forward(
            &flat,
            &ids,
            &weights,
            cfg.num_experts_per_tok,
            cfg.swiglu_limit,
        )?;
        Ok(x.add(&moe_out.reshape((1usize, seq, cfg.hidden_size))?)?)
    }

    fn route(
        &self,
        layer: &GptOssCudaLayer,
        x_flat: &Tensor,
        seq: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let e = self.config.num_local_experts;
        let k_top = self.config.num_experts_per_tok;
        let logits = x_flat
            .to_dtype(DType::F32)?
            .matmul(&layer.router_t)?
            .broadcast_add(&layer.router_bias)?
            .to_vec2::<f32>()?;
        let mut ids = Vec::with_capacity(seq * k_top);
        let mut weights = Vec::with_capacity(seq * k_top);
        for row in logits.iter().take(seq) {
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|a, b| {
                row[*b]
                    .partial_cmp(&row[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            let sel = &order[..k_top];
            let mx = sel.iter().fold(f32::NEG_INFINITY, |m, i| m.max(row[*i]));
            let mut exps: Vec<f32> = sel.iter().map(|i| (row[*i] - mx).exp()).collect();
            let z: f32 = exps.iter().sum();
            for w in exps.iter_mut() {
                *w /= z;
            }
            ids.extend_from_slice(sel);
            weights.extend_from_slice(&exps);
        }
        Ok((ids, weights))
    }
}

fn bf16_tensor<S: Into<candle_core::Shape>>(
    host: &[u16],
    shape: S,
    device: &Device,
) -> Result<Tensor> {
    let vals: Vec<half::bf16> = host.iter().map(|b| half::bf16::from_bits(*b)).collect();
    Ok(Tensor::from_vec(vals, shape, device)?)
}

fn host_linear(lin: &HostBf16Lin, device: &Device) -> Result<Linear> {
    let w = bf16_tensor(&lin.w, (lin.n, lin.k), device)?;
    let bias = if lin.bias.is_empty() {
        None
    } else {
        Some(bf16_tensor(&lin.bias, lin.n, device)?)
    };
    Linear::new(w, bias)
}
