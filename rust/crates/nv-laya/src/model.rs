use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

const HEAD_DIM: usize = 64;
const MASKED_SCORE_BIAS_IS_FINITE_SO_A_PADDED_QUERY_WITH_NO_VALID_KEY_IN_ITS_WINDOW_NEVER_TURNS_NAN: f32 = -30000.0;
const A_PADDED_GROUP_ADMITS_ROWS_AT_LEAST_THIS_FRACTION_OF_ITS_LONGEST: (usize, usize) = (3, 4);
pub const A_GRAPHED_PASS_IS_PADDED_BY_ONE_UNREAD_SEQUENCE_TO_THE_NEXT_MULTIPLE_OF_THIS_MANY_TOKENS: usize = 64;
pub const A_PASS_THAT_PADS_BEYOND_THIS_MANY_TOKENS_IS_COMPUTE_BOUND_AND_RUNS_EAGER: usize = 1600;
pub const REQUESTS_THAT_PROVE_A_SHAPE_REPEATS_BECAUSE_A_CAPTURE_COSTS_ABOUT_THIS_MANY_REPLAY_SAVINGS: usize = 8;

#[cfg(feature = "flash-attn")]
mod graph;
#[cfg(feature = "flash-attn")]
pub use graph::{
    GraphStats, LIVE_GRAPHS_A_REPLICA_KEEPS_BEFORE_IT_DESTROYS_THE_LEAST_RECENTLY_REPLAYED,
    UNPROVEN_SHAPES_A_REPLICA_COUNTS_BEFORE_IT_FORGETS_THE_LEAST_RECENTLY_REQUESTED,
};

pub fn graph_token_bucket(real_tokens: usize) -> Option<usize> {
    let step =
        A_GRAPHED_PASS_IS_PADDED_BY_ONE_UNREAD_SEQUENCE_TO_THE_NEXT_MULTIPLE_OF_THIS_MANY_TOKENS;
    Some((real_tokens / step + 1) * step).filter(|bucket| {
        *bucket <= A_PASS_THAT_PADS_BEYOND_THIS_MANY_TOKENS_IS_COMPUTE_BOUND_AND_RUNS_EAGER
    })
}

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub hidden: usize,
    pub heads: usize,
    pub norm_eps: f32,
    pub half_window: usize,
    pub global: Vec<bool>,
    pub theta_global: f64,
    pub theta_local: f64,
}

impl EncoderConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let num = |k: &str| {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| anyhow!("encoder config lacks {k}"))
        };
        let layers = num("num_hidden_layers")?;
        let every = v
            .get("global_attn_every_n_layers")
            .and_then(Value::as_u64)
            .unwrap_or(3) as usize;
        let global = match v.get("layer_types").and_then(Value::as_array) {
            Some(types) => types
                .iter()
                .map(|t| t.as_str() == Some("full_attention"))
                .collect(),
            None => (0..layers).map(|i| i % every == 0).collect(),
        };
        let theta = |kind: &str, legacy: &str, fallback: f64| {
            v.pointer(&format!("/rope_parameters/{kind}/rope_theta"))
                .or_else(|| v.get(legacy))
                .and_then(Value::as_f64)
                .unwrap_or(fallback)
        };
        Ok(Self {
            hidden: num("hidden_size")?,
            heads: num("num_attention_heads")?,
            norm_eps: v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            half_window: num("local_attention")? / 2,
            global,
            theta_global: theta("full_attention", "global_rope_theta", 160000.0),
            theta_local: theta("sliding_attention", "local_rope_theta", 10000.0),
        })
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Norm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::ops::layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}

struct Linear {
    weight_t: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.matmul(&self.weight_t)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

struct EncoderLayer {
    attn_norm: Option<Norm>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Norm,
    wi: Linear,
    mlp_wo: Linear,
    global: bool,
}

struct HeadLayer {
    norm1: Norm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: Norm,
    linear1: Linear,
    linear2: Linear,
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

pub struct ActHead {
    w0: Vec<f32>,
    b0: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    hidden: usize,
    inputs: usize,
    outputs: usize,
}

impl ActHead {
    pub fn probabilities(&self, pooled: &[f32], feats: [f32; 4]) -> Vec<f32> {
        let mut mid = vec![0f32; self.hidden];
        for (j, m) in mid.iter_mut().enumerate() {
            let row = &self.w0[j * self.inputs..(j + 1) * self.inputs];
            let mut acc = self.b0[j];
            for (w, x) in row.iter().zip(pooled.iter().chain(feats.iter())) {
                acc += w * x;
            }
            *m = 0.5 * acc * (1.0 + erf(acc / std::f32::consts::SQRT_2));
        }
        let mut out: Vec<f32> = (0..self.outputs)
            .map(|o| {
                self.b1[o]
                    + self.w1[o * self.hidden..(o + 1) * self.hidden]
                        .iter()
                        .zip(&mid)
                        .map(|(w, m)| w * m)
                        .sum::<f32>()
            })
            .collect();
        softmax_in_place(&mut out);
        out
    }
}

fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

pub fn softmax_in_place(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

pub struct Batch<'a> {
    pub sequences: &'a [Vec<u32>],
    pub markers: &'a [Vec<usize>],
    pub qtypes: &'a [u32],
}

pub struct Output {
    pub logits: Vec<Vec<f32>>,
    pub pooled: Vec<Vec<f32>>,
}

pub struct DecisionModel {
    cfg: EncoderConfig,
    device: Device,
    dtype: DType,
    tok_embeddings: Tensor,
    emb_norm: Norm,
    layers: Vec<EncoderLayer>,
    final_norm: Norm,
    head: Vec<HeadLayer>,
    type_emb: Tensor,
    scorer_norm: Norm,
    scorer_hidden: Linear,
    scorer_out: Linear,
    rope_global: Rope,
    rope_local: Rope,
    pub act_head: ActHead,
    pad_id: u32,
    flash: bool,
    pub full_final_head: bool,
    pub cuda_graph: bool,
    pub requests_that_prove_a_shape_repeats: usize,
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    max_len: usize,
    #[cfg(feature = "flash-attn")]
    graphs: std::sync::Mutex<graph::Graphs>,
    #[cfg(feature = "flash-attn")]
    lane: std::sync::Arc<graph::Lane>,
    #[cfg(feature = "flash-attn")]
    teardown_holds_this_lanes_turn_and_excludes_captures_until_every_field_above_has_dropped:
        Option<graph::Teardown>,
}

#[cfg(feature = "flash-attn")]
impl Drop for DecisionModel {
    fn drop(&mut self) {
        self.teardown_holds_this_lanes_turn_and_excludes_captures_until_every_field_above_has_dropped =
            Some(graph::Teardown::begin(&self.lane));
    }
}

#[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
struct VarlenRows {
    cu: Vec<u32>,
    positions: Vec<u32>,
}

struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
}

impl Weights {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow!("checkpoint lacks tensor {name}"))?;
        Ok(t.to_dtype(self.dtype)?.to_device(&self.device)?)
    }
    fn cpu_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow!("checkpoint lacks tensor {name}"))?;
        Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }
    fn linear(&mut self, prefix: &str, bias: bool) -> Result<Linear> {
        let weight_t = self.take(&format!("{prefix}.weight"))?.t()?.contiguous()?;
        let bias = if bias {
            Some(self.take(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Linear { weight_t, bias })
    }
    fn prescaled_qkv(&mut self, weight: &str, bias: Option<&str>) -> Result<Linear> {
        let scale_q = |t: Tensor| -> Result<Tensor> {
            let third = t.dim(0)? / 3;
            let q = (t.narrow(0, 0, third)?.to_dtype(DType::F32)?
                * (1.0 / (HEAD_DIM as f64).sqrt()))?
            .to_dtype(t.dtype())?;
            Ok(Tensor::cat(&[q, t.narrow(0, third, 2 * third)?], 0)?)
        };
        let weight_t = scale_q(self.take(weight)?)?.t()?.contiguous()?;
        let bias = match bias {
            Some(name) => Some(scale_q(self.take(name)?)?),
            None => None,
        };
        Ok(Linear { weight_t, bias })
    }
    fn norm(&mut self, prefix: &str, bias: bool, eps: f32) -> Result<Norm> {
        let weight = self.take(&format!("{prefix}.weight"))?;
        let bias = if bias {
            self.take(&format!("{prefix}.bias"))?
        } else {
            weight.zeros_like()?
        };
        Ok(Norm { weight, bias, eps })
    }
}

fn rope_table(theta: f64, max_len: usize, device: &Device, dtype: DType) -> Result<Rope> {
    let half = HEAD_DIM / 2;
    let mut cos = Vec::with_capacity(max_len * half);
    let mut sin = Vec::with_capacity(max_len * half);
    for pos in 0..max_len {
        for i in 0..half {
            let angle = pos as f64 / theta.powf(2.0 * i as f64 / HEAD_DIM as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok(Rope {
        cos: Tensor::from_vec(cos, (max_len, half), device)?.to_dtype(dtype)?,
        sin: Tensor::from_vec(sin, (max_len, half), device)?.to_dtype(dtype)?,
    })
}

enum Layout {
    Padded {
        n: usize,
        len: usize,
        global_bias: Option<Tensor>,
        local_bias: Option<Tensor>,
    },
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    Varlen {
        cu: Tensor,
        max_len: usize,
        positions: Tensor,
    },
}

impl DecisionModel {
    pub fn load(
        dir: &Path,
        device: &Device,
        dtype: DType,
        max_len: usize,
        pad_id: u32,
        head_layers: usize,
        flash: bool,
    ) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(dir.join("encoder/config.json"))
            .context("reading encoder/config.json")?;
        let cfg = EncoderConfig::from_json(&serde_json::from_str(&cfg_text)?)?;
        anyhow::ensure!(
            cfg.hidden == cfg.heads * HEAD_DIM,
            "only 64-wide attention heads are supported"
        );
        let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu)
            .context("reading model.safetensors")?;
        #[cfg(feature = "flash-attn")]
        let lane = graph::Lane::of(device);
        #[cfg(feature = "flash-attn")]
        let _uploading = lane.take_turn();
        let mut w = Weights {
            tensors,
            device: device.clone(),
            dtype,
        };
        let eps = cfg.norm_eps;
        let mut layers = Vec::with_capacity(cfg.global.len());
        for (i, global) in cfg.global.iter().enumerate() {
            let p = format!("encoder.layers.{i}");
            layers.push(EncoderLayer {
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(w.norm(&format!("{p}.attn_norm"), false, eps)?)
                },
                wqkv: w.prescaled_qkv(&format!("{p}.attn.Wqkv.weight"), None)?,
                wo: w.linear(&format!("{p}.attn.Wo"), false)?,
                mlp_norm: w.norm(&format!("{p}.mlp_norm"), false, eps)?,
                wi: w.linear(&format!("{p}.mlp.Wi"), false)?,
                mlp_wo: w.linear(&format!("{p}.mlp.Wo"), false)?,
                global: *global,
            });
        }
        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{i}");
            head.push(HeadLayer {
                norm1: w.norm(&format!("{p}.norm1"), true, 1e-5)?,
                in_proj: w.prescaled_qkv(
                    &format!("{p}.self_attn.in_proj_weight"),
                    Some(&format!("{p}.self_attn.in_proj_bias")),
                )?,
                out_proj: w.linear(&format!("{p}.self_attn.out_proj"), true)?,
                norm2: w.norm(&format!("{p}.norm2"), true, 1e-5)?,
                linear1: w.linear(&format!("{p}.linear1"), true)?,
                linear2: w.linear(&format!("{p}.linear2"), true)?,
            });
        }
        let b0 = w.cpu_f32("act_head.0.bias")?;
        let b1 = w.cpu_f32("act_head.2.bias")?;
        let act_head = ActHead {
            w0: w.cpu_f32("act_head.0.weight")?,
            w1: w.cpu_f32("act_head.2.weight")?,
            hidden: b0.len(),
            inputs: cfg.hidden + 4,
            outputs: b1.len(),
            b0,
            b1,
        };
        Ok(Self {
            tok_embeddings: w.take("encoder.embeddings.tok_embeddings.weight")?,
            emb_norm: w.norm("encoder.embeddings.norm", false, eps)?,
            final_norm: w.norm("encoder.final_norm", false, eps)?,
            type_emb: w.take("type_emb.weight")?,
            scorer_norm: w.norm("scorer.0", true, 1e-5)?,
            scorer_hidden: w.linear("scorer.1", true)?,
            scorer_out: w.linear("scorer.3", true)?,
            rope_global: rope_table(cfg.theta_global, max_len, device, dtype)?,
            rope_local: rope_table(cfg.theta_local, max_len, device, dtype)?,
            layers,
            head,
            act_head,
            cfg,
            device: device.clone(),
            dtype,
            pad_id,
            flash,
            full_final_head: false,
            cuda_graph: true,
            requests_that_prove_a_shape_repeats: REQUESTS_THAT_PROVE_A_SHAPE_REPEATS_BECAUSE_A_CAPTURE_COSTS_ABOUT_THIS_MANY_REPLAY_SAVINGS,
            max_len,
            #[cfg(feature = "flash-attn")]
            graphs: Default::default(),
            #[cfg(feature = "flash-attn")]
            lane: lane.clone(),
            #[cfg(feature = "flash-attn")]
            teardown_holds_this_lanes_turn_and_excludes_captures_until_every_field_above_has_dropped:
                None,
        })
    }

    fn padded_layout(&self, lens: &[usize], len: usize) -> Result<Layout> {
        let n = lens.len();
        let masked = MASKED_SCORE_BIAS_IS_FINITE_SO_A_PADDED_QUERY_WITH_NO_VALID_KEY_IN_ITS_WINDOW_NEVER_TURNS_NAN;
        let key_bias = if lens.iter().all(|l| *l == len) {
            None
        } else {
            let mut bias = vec![0f32; n * len];
            for (r, l) in lens.iter().enumerate() {
                bias[r * len + l..(r + 1) * len].fill(masked);
            }
            Some(Tensor::from_vec(bias, (n, 1, 1, len), &self.device)?.to_dtype(self.dtype)?)
        };
        let half = self.cfg.half_window;
        let window_bias = if len <= half + 1 {
            None
        } else {
            let mut bias = vec![masked; len * len];
            for i in 0..len {
                bias[i * len + i.saturating_sub(half)..i * len + (i + half + 1).min(len)].fill(0.0);
            }
            Some(Tensor::from_vec(bias, (1, 1, len, len), &self.device)?.to_dtype(self.dtype)?)
        };
        let local_bias = match (&key_bias, window_bias) {
            (Some(key), Some(window)) => Some(key.broadcast_add(&window)?),
            (None, window) => window,
            (Some(key), None) => Some(key.clone()),
        };
        Ok(Layout::Padded {
            n,
            len,
            global_bias: key_bias,
            local_bias,
        })
    }

    fn rope_for(&self, table: &Rope, layout: &Layout) -> Result<Rope> {
        Ok(match layout {
            Layout::Padded { len, .. } => Rope {
                cos: table.cos.narrow(0, 0, *len)?,
                sin: table.sin.narrow(0, 0, *len)?,
            },
            Layout::Varlen { positions, .. } => Rope {
                cos: table.cos.index_select(positions, 0)?,
                sin: table.sin.index_select(positions, 0)?,
            },
        })
    }

    fn attention(
        &self,
        qkv: &Tensor,
        layout: &Layout,
        rope: Option<&Rope>,
        local: bool,
    ) -> Result<Tensor> {
        let heads = self.cfg.heads;
        match layout {
            Layout::Padded {
                n,
                len,
                global_bias,
                local_bias,
            } if self.device.is_metal() => {
                let qkv = qkv.reshape((*n, *len, 3 * heads, HEAD_DIM))?;
                let mut qk = qkv.narrow(2, 0, 2 * heads)?;
                if let Some(rope) = rope {
                    qk = candle_nn::rotary_emb::rope_thd(&qk.contiguous()?, &rope.cos, &rope.sin)?;
                }
                let head_rows =
                    |t: &Tensor, first: usize| t.narrow(2, first, heads)?.transpose(1, 2);
                let (q, k, v) = (
                    head_rows(&qk, 0)?,
                    head_rows(&qk, heads)?,
                    head_rows(&qkv, 2 * heads)?,
                );
                let bias = if local { local_bias } else { global_bias };
                let mask = bias
                    .as_ref()
                    .map(|b| b.broadcast_as((*n, heads, *len, *len)))
                    .transpose()?;
                let mixed = candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, 1.0, 1.0)?;
                Ok(mixed
                    .transpose(1, 2)?
                    .reshape((n * len, heads * HEAD_DIM))?)
            }
            Layout::Padded {
                n,
                len,
                global_bias,
                local_bias,
            } => {
                let qkv = qkv
                    .reshape((*n, *len, 3, heads, HEAD_DIM))?
                    .permute((2, 0, 3, 1, 4))?
                    .contiguous()?;
                let mut qk = qkv
                    .narrow(0, 0, 2)?
                    .reshape((2 * n, heads, *len, HEAD_DIM))?;
                if let Some(rope) = rope {
                    qk = candle_nn::rotary_emb::rope(&qk, &rope.cos, &rope.sin)?;
                }
                let (q, k, v) = (qk.narrow(0, 0, *n)?, qk.narrow(0, *n, *n)?, qkv.get(2)?);
                let mut scores = q.matmul(&k.transpose(2, 3)?)?;
                if let Some(bias) = if local { local_bias } else { global_bias } {
                    scores = scores.broadcast_add(bias)?;
                }
                let mixed = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
                Ok(mixed
                    .transpose(1, 2)?
                    .reshape((n * len, heads * HEAD_DIM))?)
            }
            #[cfg(feature = "flash-attn")]
            Layout::Varlen { cu, max_len, .. } => {
                let total = qkv.dim(0)?;
                let qkv = qkv.reshape((total, 3 * heads, HEAD_DIM))?;
                let mut qk = qkv.narrow(1, 0, 2 * heads)?;
                if let Some(rope) = rope {
                    qk = candle_nn::rotary_emb::rope_thd(
                        &qk.contiguous()?.unsqueeze(0)?,
                        &rope.cos,
                        &rope.sin,
                    )?
                    .squeeze(0)?;
                }
                let (q, k, v) = (
                    qk.narrow(1, 0, heads)?,
                    qk.narrow(1, heads, heads)?,
                    qkv.narrow(1, 2 * heads, heads)?,
                );
                let window = if local {
                    Some(self.cfg.half_window)
                } else {
                    None
                };
                let out = candle_flash_attn::flash_attn_varlen_windowed(
                    &q, &k, &v, cu, cu, *max_len, *max_len, 1.0, window, window,
                )?;
                Ok(out.reshape((total, heads * HEAD_DIM))?)
            }
            #[cfg(not(feature = "flash-attn"))]
            Layout::Varlen { .. } => unreachable!("varlen layout needs the flash-attn feature"),
        }
    }

    fn varlen(&self) -> bool {
        self.flash && cfg!(feature = "flash-attn")
    }

    pub fn forward(&self, batch: &Batch) -> Result<Output> {
        #[cfg(feature = "flash-attn")]
        let _turn = self.lane.take_turn();
        #[cfg(feature = "metal")]
        if self.device.is_metal() {
            return candle_metal_kernels::kernels::mlx_gemm::with_split_k_on_this_thread_unless_the_environment_decides(|| self.forward_groups(batch));
        }
        self.forward_groups(batch)
    }

    fn forward_groups(&self, batch: &Batch) -> Result<Output> {
        let n = batch.sequences.len();
        let len_of = |r: usize| batch.sequences[r].len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        if self.varlen() {
            groups.push(order);
        } else {
            let (num, den) = A_PADDED_GROUP_ADMITS_ROWS_AT_LEAST_THIS_FRACTION_OF_ITS_LONGEST;
            order.sort_by_key(|r| std::cmp::Reverse(len_of(*r)));
            for r in order {
                match groups.last_mut() {
                    Some(group) if len_of(r) * den >= len_of(group[0]) * num => group.push(r),
                    _ => groups.push(vec![r]),
                }
            }
        }
        let mut out = Output {
            logits: vec![Vec::new(); n],
            pooled: vec![Vec::new(); n],
        };
        for group in &groups {
            let part = self.forward_rows(batch, group)?;
            for (r, (logits, pooled)) in group.iter().zip(part.logits.into_iter().zip(part.pooled))
            {
                out.logits[*r] = logits;
                out.pooled[*r] = pooled;
            }
        }
        Ok(out)
    }

    fn forward_rows(&self, batch: &Batch, picked_rows: &[usize]) -> Result<Output> {
        let n = picked_rows.len();
        let rows_of = || {
            picked_rows
                .iter()
                .map(|r| (&batch.sequences[*r], batch.qtypes[*r], &batch.markers[*r]))
        };
        let lens: Vec<usize> = rows_of().map(|(seq, ..)| seq.len()).collect();
        let longest = lens.iter().copied().max().unwrap_or(0);

        let mut ids = Vec::new();
        let mut types = Vec::new();
        let mut starts = Vec::with_capacity(n);
        let mut varlen = self.varlen().then(|| VarlenRows {
            cu: vec![0u32],
            positions: Vec::new(),
        });
        for (r, (seq, qt, _)) in rows_of().enumerate() {
            ids.extend_from_slice(seq);
            match &mut varlen {
                Some(rows) => {
                    starts.push(rows.positions.len());
                    types.extend(std::iter::repeat_n(qt, seq.len()));
                    rows.positions.extend(0..seq.len() as u32);
                    rows.cu.push(ids.len() as u32);
                }
                None => {
                    starts.push(r * longest);
                    ids.extend(std::iter::repeat_n(self.pad_id, longest - seq.len()));
                    types.extend(std::iter::repeat_n(qt, longest));
                }
            }
        }

        let mut consumed: Vec<u32> = Vec::new();
        for (start, (.., markers)) in starts.iter().zip(rows_of()) {
            consumed.extend(markers.iter().map(|m| (start + m) as u32));
        }
        let n_markers = consumed.len();
        consumed.extend(starts.iter().map(|s| *s as u32));

        #[cfg(feature = "flash-attn")]
        let graphed = match &varlen {
            Some(rows) => self.graphed(&ids, &types, rows, &consumed, n_markers)?,
            None => None,
        };
        #[cfg(not(feature = "flash-attn"))]
        let graphed = None;
        let flat: Vec<f32> = match graphed {
            Some(flat) => flat,
            None => {
                let layout = match varlen {
                    Some(VarlenRows { cu, positions }) => Layout::Varlen {
                        cu: Tensor::from_vec(cu, n + 1, &self.device)?,
                        max_len: longest,
                        positions: Tensor::from_vec(positions, ids.len(), &self.device)?,
                    },
                    None => self.padded_layout(&lens, longest)?,
                };
                let total = ids.len();
                let ids = Tensor::from_vec(ids, total, &self.device)?;
                let types = Tensor::from_vec(types, total, &self.device)?;
                let consumed = Tensor::from_vec(consumed, n_markers + n, &self.device)?;
                self.encode(&ids, &types, &consumed, &layout, n_markers)?
                    .to_vec1()?
            }
        };
        anyhow::ensure!(
            flat.iter().all(|v| v.is_finite()),
            "the forward pass overflowed in {:?}; restart with --dtype bf16",
            self.dtype
        );

        let mut logits = Vec::with_capacity(n);
        let mut at = 0;
        for (.., markers) in rows_of() {
            logits.push(flat[at..at + markers.len()].to_vec());
            at += markers.len();
        }
        let pooled = flat[n_markers..]
            .chunks(self.cfg.hidden)
            .map(<[f32]>::to_vec)
            .collect();
        Ok(Output { logits, pooled })
    }

    fn encode(
        &self,
        ids: &Tensor,
        types: &Tensor,
        consumed: &Tensor,
        layout: &Layout,
        n_markers: usize,
    ) -> Result<Tensor> {
        let n = consumed.dim(0)? - n_markers;
        let rope_global = self.rope_for(&self.rope_global, layout)?;
        let rope_local = self.rope_for(&self.rope_local, layout)?;
        let mut h = self
            .emb_norm
            .forward(&self.tok_embeddings.index_select(ids, 0)?)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => norm.forward(&h)?,
                None => h.clone(),
            };
            let rope = if layer.global {
                &rope_global
            } else {
                &rope_local
            };
            let attn = self.attention(
                &layer.wqkv.forward(&normed)?,
                layout,
                Some(rope),
                !layer.global,
            )?;
            h = (h + layer.wo.forward(&attn)?)?;
            let wi = layer.wi.forward(&layer.mlp_norm.forward(&h)?)?;
            let inter = wi.dim(D::Minus1)? / 2;
            let half = |first: usize| -> candle_core::Result<Tensor> {
                let half = wi.narrow(D::Minus1, first, inter)?;
                if self.device.is_metal() {
                    half.contiguous()
                } else {
                    Ok(half)
                }
            };
            let gated = (half(0)?.gelu_erf()? * half(inter)?)?;
            h = (h + layer.mlp_wo.forward(&gated)?)?;
        }
        h = self.final_norm.forward(&h)?;

        h = (h + self.type_emb.index_select(types, 0)?)?;
        let prune_at = if self.full_final_head {
            None
        } else {
            self.head.len().checked_sub(1)
        };
        for (i, layer) in self.head.iter().enumerate() {
            let mut attn = self.attention(
                &layer.in_proj.forward(&layer.norm1.forward(&h)?)?,
                layout,
                None,
                false,
            )?;
            if prune_at == Some(i) {
                attn = attn.index_select(consumed, 0)?;
                h = h.index_select(consumed, 0)?;
            }
            h = (h + layer.out_proj.forward(&attn)?)?;
            let ff = layer
                .linear2
                .forward(&layer.linear1.forward(&layer.norm2.forward(&h)?)?.relu()?)?;
            h = (h + ff)?;
        }
        let picked = if prune_at.is_some() {
            h
        } else {
            h.index_select(consumed, 0)?
        };

        let scored = self.scorer_out.forward(
            &self
                .scorer_hidden
                .forward(&self.scorer_norm.forward(&picked.narrow(0, 0, n_markers)?)?)?
                .gelu_erf()?,
        )?;
        let flat = Tensor::cat(
            &[
                scored.flatten_all()?,
                picked.narrow(0, n_markers, n)?.flatten_all()?,
            ],
            0,
        )?;
        Ok(flat.to_dtype(DType::F32)?)
    }
}
