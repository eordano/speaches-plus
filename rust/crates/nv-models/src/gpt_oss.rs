use anyhow::{Context, Result};

use nv_quant::mxfp4::{Mxfp4Tensor, BLOCK_BYTES as MX_BLOCK_BYTES, BLOCK_SIZE as MX_BLOCK};

pub const SWIGLU_ALPHA: f32 = 1.702;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptOssLayerType {
    Sliding,
    Full,
}

#[derive(Clone, Debug)]
pub struct GptOssConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub swiglu_limit: f32,
    pub layer_types: Vec<GptOssLayerType>,
    pub yarn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_original_max: usize,
    pub tie_word_embeddings: bool,
}

impl GptOssConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse gpt_oss config json")?;
        let get_u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let get_f = |k: &str| -> Result<f64> {
            v.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let layer_types_raw = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing layer_types"))?;
        let layer_types: Vec<GptOssLayerType> = layer_types_raw
            .iter()
            .map(|x| match x.as_str() {
                Some("sliding_attention") => Ok(GptOssLayerType::Sliding),
                Some("full_attention") => Ok(GptOssLayerType::Full),
                other => Err(anyhow::anyhow!("unknown layer type {other:?}")),
            })
            .collect::<Result<Vec<_>>>()?;
        let scaling = v.get("rope_scaling");
        let sub_f = |k: &str, d: f64| -> f64 {
            scaling
                .and_then(|s| s.get(k))
                .and_then(|x| x.as_f64())
                .unwrap_or(d)
        };
        let sub_u = |k: &str, d: usize| -> usize {
            scaling
                .and_then(|s| s.get(k))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        Ok(Self {
            hidden_size: get_u("hidden_size")?,
            num_hidden_layers: get_u("num_hidden_layers")?,
            num_attention_heads: get_u("num_attention_heads")?,
            num_key_value_heads: get_u("num_key_value_heads")?,
            head_dim: get_u("head_dim")?,
            intermediate_size: get_u("intermediate_size")?,
            num_local_experts: get_u("num_local_experts")?,
            num_experts_per_tok: get_u("num_experts_per_tok")?,
            vocab_size: get_u("vocab_size")?,
            max_position_embeddings: get_u("max_position_embeddings").unwrap_or(131072),
            sliding_window: get_u("sliding_window")?,
            rope_theta: get_f("rope_theta")? as f32,
            rms_norm_eps: get_f("rms_norm_eps")?,
            swiglu_limit: get_f("swiglu_limit").unwrap_or(7.0) as f32,
            layer_types,
            yarn_factor: sub_f("factor", 1.0) as f32,
            yarn_beta_fast: sub_f("beta_fast", 32.0) as f32,
            yarn_beta_slow: sub_f("beta_slow", 1.0) as f32,
            yarn_original_max: sub_u("original_max_position_embeddings", 0),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str, as_ref);

    pub fn attention_scaling(&self) -> f32 {
        if self.yarn_factor <= 1.0 || self.yarn_original_max == 0 {
            1.0
        } else {
            0.1 * self.yarn_factor.ln() + 1.0
        }
    }
}

pub fn yarn_inv_freq(cfg: &GptOssConfig) -> Vec<f32> {
    let dim = cfg.head_dim;
    let half = dim / 2;
    let base = cfg.rope_theta as f64;
    let default: Vec<f32> = (0..half)
        .map(|i| (1.0 / base.powf((i as f64 * 2.0) / dim as f64)) as f32)
        .collect();
    let factor = cfg.yarn_factor as f64;
    let orig = cfg.yarn_original_max as f64;
    if factor <= 1.0 || orig <= 0.0 {
        return default;
    }
    let beta_fast = cfg.yarn_beta_fast as f64;
    let beta_slow = cfg.yarn_beta_slow as f64;
    let find_correction_dim = |num_rot: f64| -> f64 {
        (dim as f64) * (orig / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
    };
    let mut low = find_correction_dim(beta_fast).floor();
    let mut high = find_correction_dim(beta_slow).ceil();
    low = low.max(0.0);
    high = high.min(dim as f64 - 1.0);
    if (high - low).abs() < f64::EPSILON {
        high += 0.001;
    }
    (0..half)
        .map(|i| {
            let pos_freq = base.powf((i as f64 * 2.0) / dim as f64);
            let extrap = 1.0 / pos_freq;
            let interp = 1.0 / (factor * pos_freq);
            let ramp = (((i as f64) - low) / (high - low)).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            (interp * (1.0 - extrap_factor) + extrap * extrap_factor) as f32
        })
        .collect()
}

pub fn rope_tables(cfg: &GptOssConfig, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim / 2;
    let inv = yarn_inv_freq(cfg);
    let mscale = cfg.attention_scaling();
    let mut cos = vec![0f32; rows * half.max(1)];
    let mut sin = vec![0f32; rows * half.max(1)];
    for p in 0..rows {
        for i in 0..half {
            let t = (p as f32) * inv[i];
            cos[p * half + i] = t.cos() * mscale;
            sin[p * half + i] = t.sin() * mscale;
        }
    }
    (cos, sin)
}

#[derive(Clone)]
pub struct HostBf16Lin {
    pub w: Vec<u16>,
    pub bias: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone)]
pub struct HostMxStack {
    pub blocks: Vec<u8>,
    pub scales: Vec<u8>,
    pub bias: Vec<u16>,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

impl HostMxStack {
    pub fn expert(&self, e: usize) -> Mxfp4Tensor {
        let bpe = self.n * self.k / 2;
        let spe = self.n * self.k / MX_BLOCK;
        Mxfp4Tensor::from_gpt_oss_row_major(
            &self.blocks[e * bpe..(e + 1) * bpe],
            &self.scales[e * spe..(e + 1) * spe],
            self.n,
            self.k,
        )
    }
}

pub fn stack_mx_host(mats: &[Mxfp4Tensor], biases: &[Vec<u16>]) -> HostMxStack {
    let n = mats[0].rows;
    let k = mats[0].cols;
    let mut blocks = Vec::new();
    let mut scales = Vec::new();
    let mut bias = Vec::new();
    for (m, b) in mats.iter().zip(biases) {
        assert_eq!(m.rows, n);
        assert_eq!(m.cols, k);
        assert_eq!(b.len(), n);
        blocks.extend_from_slice(&m.data);
        scales.extend_from_slice(&m.scales);
        bias.extend_from_slice(b);
    }
    HostMxStack {
        blocks,
        scales,
        bias,
        e: mats.len(),
        n,
        k,
    }
}

#[derive(Clone)]
pub struct HostAttn {
    pub q: HostBf16Lin,
    pub k: HostBf16Lin,
    pub v: HostBf16Lin,
    pub o: HostBf16Lin,
    pub sinks: Vec<f32>,
}

#[derive(Clone)]
pub struct HostMoe {
    pub router: HostBf16Lin,
    pub gate_up: HostMxStack,
    pub down: HostMxStack,
}

#[derive(Clone)]
pub struct HostLayer {
    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub attn: HostAttn,
    pub moe: HostMoe,
}

pub struct HostWeights {
    pub embed: Vec<u16>,
    pub final_norm: Vec<u16>,
    pub lm_head: Vec<u16>,
    pub layers: Vec<HostLayer>,
}

pub(crate) fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

pub(crate) fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub(crate) fn load_bf16(w: &nv_weights::WeightLoader, names: &[&str], shape: &[usize]) -> Result<Vec<u16>> {
    for n in names {
        if w.has(n) {
            let t = w
                .get(n, candle_core::DType::BF16)
                .with_context(|| format!("load {n}"))?;
            anyhow::ensure!(t.dims() == shape, "{n}: shape {:?} != {shape:?}", t.dims());
            let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
            return Ok(v.into_iter().map(|x| x.to_bits()).collect());
        }
    }
    anyhow::bail!("none of {names:?} found")
}

pub(crate) fn load_f32_vec(w: &nv_weights::WeightLoader, name: &str, dim: usize) -> Result<Vec<f32>> {
    let raw = load_bf16(w, &[name], &[dim])?;
    Ok(raw.into_iter().map(bf16_val).collect())
}

pub(crate) fn load_lin(
    w: &nv_weights::WeightLoader,
    module: &str,
    n: usize,
    k: usize,
    bias: bool,
) -> Result<HostBf16Lin> {
    Ok(HostBf16Lin {
        w: load_bf16(w, &[&format!("{module}.weight")], &[n, k])?,
        bias: if bias {
            load_bf16(w, &[&format!("{module}.bias")], &[n])?
        } else {
            Vec::new()
        },
        n,
        k,
    })
}

pub(crate) fn load_mx_stack(
    w: &nv_weights::WeightLoader,
    base: &str,
    e: usize,
    n: usize,
    k: usize,
) -> Result<HostMxStack> {
    let kb = k / MX_BLOCK;
    let bl = format!("{base}_blocks");
    let sc = format!("{base}_scales");
    let bi = format!("{base}_bias");
    let bshape = w
        .shape_of(&bl)
        .ok_or_else(|| anyhow::anyhow!("missing {bl}"))?;
    anyhow::ensure!(
        bshape == [e, n, kb, MX_BLOCK_BYTES],
        "{bl}: shape {bshape:?}, want [{e}, {n}, {kb}, {MX_BLOCK_BYTES}]"
    );
    let sshape = w
        .shape_of(&sc)
        .ok_or_else(|| anyhow::anyhow!("missing {sc}"))?;
    anyhow::ensure!(
        sshape == [e, n, kb],
        "{sc}: shape {sshape:?}, want [{e}, {n}, {kb}]"
    );
    let blocks = w.raw_bytes(&bl)?.to_vec();
    let scales = w.raw_bytes(&sc)?.to_vec();
    anyhow::ensure!(
        blocks.len() == e * n * kb * MX_BLOCK_BYTES,
        "{bl}: byte count"
    );
    anyhow::ensure!(scales.len() == e * n * kb, "{sc}: byte count");
    let bias = load_bf16(w, &[&bi], &[e, n])?;
    Ok(HostMxStack {
        blocks,
        scales,
        bias,
        e,
        n,
        k,
    })
}

pub(crate) fn load_layer(cfg: &GptOssConfig, w: &nv_weights::WeightLoader, idx: usize) -> Result<HostLayer> {
    let p = format!("model.layers.{idx}");
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd;
    let kv_out = cfg.num_key_value_heads * hd;
    let hidden = cfg.hidden_size;
    Ok(HostLayer {
        input_ln: load_bf16(w, &[&format!("{p}.input_layernorm.weight")], &[hidden])?,
        post_attn_ln: load_bf16(
            w,
            &[&format!("{p}.post_attention_layernorm.weight")],
            &[hidden],
        )?,
        attn: HostAttn {
            q: load_lin(w, &format!("{p}.self_attn.q_proj"), q_out, hidden, true)?,
            k: load_lin(w, &format!("{p}.self_attn.k_proj"), kv_out, hidden, true)?,
            v: load_lin(w, &format!("{p}.self_attn.v_proj"), kv_out, hidden, true)?,
            o: load_lin(w, &format!("{p}.self_attn.o_proj"), hidden, q_out, true)?,
            sinks: load_f32_vec(w, &format!("{p}.self_attn.sinks"), cfg.num_attention_heads)?,
        },
        moe: HostMoe {
            router: load_lin(
                w,
                &format!("{p}.mlp.router"),
                cfg.num_local_experts,
                hidden,
                true,
            )?,
            gate_up: load_mx_stack(
                w,
                &format!("{p}.mlp.experts.gate_up_proj"),
                cfg.num_local_experts,
                2 * cfg.intermediate_size,
                hidden,
            )?,
            down: load_mx_stack(
                w,
                &format!("{p}.mlp.experts.down_proj"),
                cfg.num_local_experts,
                hidden,
                cfg.intermediate_size,
            )?,
        },
    })
}

pub(crate) fn rbf(x: f32) -> f32 {
    bf16_val(bf16_bits(x))
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn ref_gemv_bf16(w: &HostBf16Lin, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; w.n];
    for (r, out) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (c, xv) in x.iter().enumerate().take(w.k) {
            acc += bf16_val(w.w[r * w.k + c]) * xv;
        }
        if !w.bias.is_empty() {
            acc += bf16_val(w.bias[r]);
        }
        *out = acc;
    }
    y
}

pub(crate) fn ref_rmsnorm(x: &[f32], w: &[u16], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| rbf(x[i] * inv * bf16_val(w[i]))).collect()
}

pub struct RefState {
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
}

impl RefState {
    pub fn new(cfg: &GptOssConfig) -> Self {
        let l = cfg.num_hidden_layers;
        Self {
            kc: vec![Vec::new(); l],
            vc: vec![Vec::new(); l],
            pos: 0,
        }
    }
}

pub fn reference_step(
    cfg: &GptOssConfig,
    hw: &HostWeights,
    st: &mut RefState,
    token: u32,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let pos = st.pos;

    let mut res: Vec<f32> = (0..hidden)
        .map(|i| bf16_val(hw.embed[token as usize * hidden + i]))
        .collect();

    for li in 0..cfg.num_hidden_layers {
        let layer = &hw.layers[li];
        let normed = ref_rmsnorm(&res, &layer.input_ln, eps);
        let mixed = ref_attn(cfg, layer, li, &normed, st, pos)?;
        for i in 0..hidden {
            res[i] = rbf(res[i] + mixed[i]);
        }
        let normed_post = ref_rmsnorm(&res, &layer.post_attn_ln, eps);
        let moe_out = ref_moe(cfg, &layer.moe, &normed_post)?;
        for i in 0..hidden {
            res[i] = rbf(res[i] + moe_out[i]);
        }
    }

    let fx = ref_rmsnorm(&res, &hw.final_norm, eps);
    let lm = HostBf16Lin {
        w: hw.lm_head.clone(),
        bias: Vec::new(),
        n: cfg.vocab_size,
        k: hidden,
    };
    st.pos += 1;
    Ok(ref_gemv_bf16(&lm, &fx))
}

fn ref_attn(
    cfg: &GptOssConfig,
    layer: &HostLayer,
    li: usize,
    x: &[f32],
    st: &mut RefState,
    pos: usize,
) -> Result<Vec<f32>> {
    let a = &layer.attn;
    let hd = cfg.head_dim;
    let half = hd / 2;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let window = match cfg.layer_types[li] {
        GptOssLayerType::Sliding => cfg.sliding_window,
        GptOssLayerType::Full => 0,
    };

    let q_raw: Vec<f32> = ref_gemv_bf16(&a.q, x).into_iter().map(rbf).collect();
    let k_raw: Vec<f32> = ref_gemv_bf16(&a.k, x).into_iter().map(rbf).collect();
    let v_raw: Vec<f32> = ref_gemv_bf16(&a.v, x).into_iter().map(rbf).collect();

    let inv = yarn_inv_freq(cfg);
    let mscale = cfg.attention_scaling();
    let rope = |row: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; hd];
        for i in 0..half {
            let t = (pos as f32) * inv[i];
            let (c, s) = ((t.cos() * mscale), (t.sin() * mscale));
            out[i] = rbf(row[i] * c - row[i + half] * s);
            out[i + half] = rbf(row[i + half] * c + row[i] * s);
        }
        out
    };

    let mut q = vec![0f32; n_h * hd];
    for h in 0..n_h {
        let row: Vec<f32> = q_raw[h * hd..(h + 1) * hd].to_vec();
        q[h * hd..(h + 1) * hd].copy_from_slice(&rope(&row));
    }
    let mut kk = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        let row: Vec<f32> = k_raw[h * hd..(h + 1) * hd].to_vec();
        kk[h * hd..(h + 1) * hd].copy_from_slice(&rope(&row));
    }

    st.kc[li].extend_from_slice(&kk);
    st.vc[li].extend_from_slice(&v_raw);
    let total = pos + 1;
    let start = if window > 0 && total > window {
        total - window
    } else {
        0
    };
    let group = n_h / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut out = vec![0f32; n_h * hd];
    for h in 0..n_h {
        let kv = h / group;
        let mut scores = vec![0f32; total - start];
        let mut m = a.sinks[h];
        for (j, t) in (start..total).enumerate() {
            let base = (t * n_kv + kv) * hd;
            let mut dot = 0f32;
            for i in 0..hd {
                dot += st.kc[li][base + i] * q[h * hd + i];
            }
            scores[j] = dot * scale;
            m = m.max(scores[j]);
        }
        let mut z = (a.sinks[h] - m).exp();
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            z += *s;
        }
        for i in 0..hd {
            let mut acc = 0f32;
            for (j, t) in (start..total).enumerate() {
                acc += scores[j] * st.vc[li][(t * n_kv + kv) * hd + i];
            }
            out[h * hd + i] = acc / z;
        }
    }

    let packed: Vec<f32> = out.iter().map(|v| rbf(*v)).collect();
    Ok(ref_gemv_bf16(&a.o, &packed).into_iter().map(rbf).collect())
}

fn ref_moe(cfg: &GptOssConfig, m: &HostMoe, x: &[f32]) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let k_top = cfg.num_experts_per_tok;

    let logits = ref_gemv_bf16(&m.router, x);
    let mut order: Vec<usize> = (0..cfg.num_local_experts).collect();
    order.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let ids: Vec<usize> = order[..k_top].to_vec();
    let mut wts: Vec<f32> = ids.iter().map(|e| logits[*e]).collect();
    let mx = wts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut zsum = 0f32;
    for w in wts.iter_mut() {
        *w = (*w - mx).exp();
        zsum += *w;
    }
    for w in wts.iter_mut() {
        *w /= zsum;
    }

    let limit = cfg.swiglu_limit;
    let mut acc = vec![0f32; hidden];
    for (j, e) in ids.iter().enumerate() {
        let gu = m.gate_up.expert(*e);
        let gu_deq = gu.dequantize();
        let mut y_gu = vec![0f32; 2 * inter];
        for (r, row) in gu_deq.iter().enumerate() {
            let mut d = 0f32;
            for c in 0..hidden {
                d += row[c] * x[c];
            }
            y_gu[r] = d + bf16_val(m.gate_up.bias[*e * 2 * inter + r]);
        }
        let mut act = vec![0f32; inter];
        for i in 0..inter {
            let g = y_gu[2 * i].min(limit);
            let u = y_gu[2 * i + 1].clamp(-limit, limit);
            let glu = g * sigmoid(SWIGLU_ALPHA * g);
            act[i] = rbf(glu * (u + 1.0));
        }
        let dn = m.down.expert(*e);
        let dn_deq = dn.dequantize();
        for (r, row) in dn_deq.iter().enumerate() {
            let mut d = 0f32;
            for c in 0..inter {
                d += row[c] * act[c];
            }
            d += bf16_val(m.down.bias[*e * hidden + r]);
            acc[r] += d * wts[j];
        }
    }
    Ok(acc.into_iter().map(rbf).collect())
}
