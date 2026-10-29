use crate::{save_peft, DetRng, LoraConfig, LoraTrainable, PeftEntry};
use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use nv_laya::engine::{Checkpoint, Settings};
use nv_laya::json::Json;
use nv_laya::model::EncoderConfig;
use nv_laya::seq::QType;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const LAYA_QLORA_BASE_IS_THE_MLX_AFFINE_ROUND_TRIP_OF_EVERY_ENCODER_LINEAR_AT_THIS_GROUP_SIZE: usize = 64;
pub const BASE_BITS_THE_TRAINER_ACCEPTS_WHERE_SIXTEEN_IS_THE_STORED_WEIGHTS_UNTOUCHED: [u32; 3] =
    [4, 8, 16];
pub const BASE_BITS_DEFAULT_IS_THE_STORED_WEIGHTS_BECAUSE_NOTHING_THIS_SMALL_NEEDS_A_QUANTIZED_BASE: u32 = 16;
pub const MERGED_CHECKPOINT_LEAF_BESIDE_THE_ADAPTER: &str = "merged";
pub const THE_TWIN_MASKS_WITH_THE_SAME_FINITE_BIAS_SERVING_USES: f32 = -30000.0;
pub const UPSTREAM_PROPER_REWARD_WEIGHS_THE_SPHERICAL_SCORE_BY: f64 = 0.5;
pub const UPSTREAM_PROPER_REWARD_WEIGHS_THE_RANKED_PROBABILITY_SCORE_OF_ORDINAL_QUESTIONS_BY: f64 =
    1.0;
pub const UPSTREAM_PROPER_REWARD_FLOORS_THE_LOG_SCORE_AT: f64 = -9.21;
pub const LAYA_TRAINS_THROUGH_A_DIFFERENTIABLE_TWIN_BECAUSE_SERVING_LAYER_NORM_ROPE_SOFTMAX_AND_FLASH_ATTENTION_CARRY_NO_BACKWARD: &str =
    "candle_nn::ops::layer_norm, rotary_emb::rope/rope_thd and softmax_last_dim are apply_op*_no_bwd and flash-attn has no \
     backward at all, so a loss taken through nv_laya::model::DecisionModel reaches no adapter below the last layer while \
     still falling. The twin composes the same functions from tracked primitives; tests/laya_qlora.rs holds it to the \
     serving forward on the merged checkpoint, and reads every lora_B back off zero.";
const HEAD_DIM: usize = 64;
pub const ENCODER_TARGETS: [&str; 4] = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"];

pub struct LayaTrainArgs {
    pub base: PathBuf,
    pub data: PathBuf,
    pub eval_data: Option<PathBuf>,
    pub out: PathBuf,
    pub merged_out: Option<PathBuf>,
    pub rank: usize,
    pub alpha: Option<f64>,
    pub lr: f64,
    pub steps: usize,
    pub batch: usize,
    pub seed: u64,
    pub targets: Vec<String>,
    pub base_bits: u32,
    pub shuffle_options: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Eval {
    pub loss: f32,
    pub accuracy: f32,
}

#[derive(Debug)]
pub struct LayaTrainSummary {
    pub device: String,
    pub train_items: usize,
    pub eval_items: usize,
    pub losses: Vec<f32>,
    pub seconds_per_step: f64,
    pub eval_dense_base: Option<Eval>,
    pub eval_trained_base: Eval,
    pub eval_adapter: Eval,
    pub every_gradient_was_finite: bool,
    pub smallest_lora_b_amax: f32,
    pub adapter_modules: usize,
}

#[derive(Clone)]
struct Record {
    state: Json,
    id: String,
    def: Json,
}

struct Item {
    ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: QType,
    target: Vec<f32>,
    label: usize,
}

fn records(path: &Path) -> Result<Vec<Record>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let row = Json::parse(line)
            .with_context(|| format!("{}:{} is not JSON", path.display(), n + 1))?;
        let state = row
            .get("state")
            .ok_or_else(|| anyhow!("{}:{} has no state", path.display(), n + 1))?
            .clone();
        let questions = row
            .get("questions")
            .and_then(Json::as_obj)
            .ok_or_else(|| anyhow!("{}:{} has no questions map", path.display(), n + 1))?;
        for (id, def) in questions {
            out.push(Record {
                state: state.clone(),
                id: id.clone(),
                def: def.clone(),
            });
        }
    }
    ensure!(
        !out.is_empty(),
        "{} holds no labelled questions",
        path.display()
    );
    Ok(out)
}

fn target_of(def: &Json, keys: &[String], qtype: QType) -> Result<Vec<f32>> {
    let k = keys.len();
    if let Some(soft) = def.get("soft").and_then(Json::as_arr) {
        ensure!(
            soft.len() == k,
            "soft has {} entries for {k} options",
            soft.len()
        );
        let t: Vec<f32> = soft
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let sum: f32 = t.iter().sum();
        ensure!(sum > 0.0, "soft target sums to zero");
        return Ok(t.iter().map(|v| v / sum).collect());
    }
    let answer = def
        .get("answer")
        .ok_or_else(|| anyhow!("a training question needs \"answer\" or \"soft\""))?;
    let index = match (qtype, answer) {
        (QType::Choice, Json::Str(key)) => keys
            .iter()
            .position(|c| c == key)
            .ok_or_else(|| anyhow!("answer {key:?} is not one of {keys:?}"))?,
        (QType::Noul, Json::Bool(yes)) => usize::from(*yes),
        (QType::Score, level) => level
            .as_f64()
            .map(|l| l as usize)
            .filter(|l| *l < k)
            .ok_or_else(|| anyhow!("score answer must be a level index below {k}"))?,
        (kind, other) => bail!("{} answer {other:?} has the wrong type", kind.name()),
    };
    Ok((0..k).map(|i| if i == index { 1.0 } else { 0.0 }).collect())
}

fn item(framing: &Checkpoint, record: &Record, shuffle: Option<&mut DetRng>) -> Result<Item> {
    let plain = framing.prepare(&record.state, &[(record.id.clone(), record.def.clone())])?;
    let qtype = plain.questions[0].qtype;
    let target = target_of(&record.def, &plain.questions[0].keys, qtype)?;
    let k = target.len();
    let order: Vec<usize> = match shuffle {
        Some(rng) if qtype == QType::Choice => {
            let mut order: Vec<usize> = (0..k).collect();
            for i in (1..k).rev() {
                order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
            }
            order
        }
        _ => (0..k).collect(),
    };
    let prepared = if order.iter().enumerate().all(|(i, o)| i == *o) {
        plain
    } else {
        let mut def = record.def.clone();
        let criteria = match record.def.get("criteria") {
            Some(Json::Obj(entries)) => {
                Json::Obj(order.iter().map(|o| entries[*o].clone()).collect())
            }
            Some(Json::Arr(list)) => Json::Arr(order.iter().map(|o| list[*o].clone()).collect()),
            _ => bail!("choice question {:?} has no criteria to reorder", record.id),
        };
        def.set("criteria", criteria);
        framing.prepare(&record.state, &[(record.id.clone(), def)])?
    };
    let target: Vec<f32> = order.iter().map(|o| target[*o]).collect();
    let label = (0..k)
        .max_by(|a, b| target[*a].total_cmp(&target[*b]).then(b.cmp(a)))
        .unwrap_or(0);
    Ok(Item {
        ids: prepared.sequences[0].clone(),
        markers: prepared.markers[0].clone(),
        qtype,
        target,
        label,
    })
}

struct Proj {
    module: String,
    w_t: Tensor,
    bias: Option<Tensor>,
    lora: Option<Arc<LoraTrainable>>,
    quantized: bool,
}

impl Proj {
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut y = x.matmul(&self.w_t)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(bias)?;
        }
        if let Some(lora) = &self.lora {
            y = (y + lora.delta_forward(x, train)?)?;
        }
        Ok(y)
    }

    fn merged(&self) -> Result<Tensor> {
        let w = self.w_t.t()?.contiguous()?;
        Ok(match &self.lora {
            Some(lora) => (w + (lora.b_tensor().matmul(lora.a_tensor())? * lora.scaling())?)?,
            None => w,
        })
    }
}

struct Norm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let y = centered
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)?;
        Ok(match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        })
    }
}

struct EncoderLayer {
    attn_norm: Option<Norm>,
    wqkv: Proj,
    wo: Proj,
    mlp_norm: Norm,
    wi: Proj,
    mlp_wo: Proj,
    global: bool,
}

struct HeadLayer {
    norm1: Norm,
    in_proj: Proj,
    out_proj: Proj,
    norm2: Norm,
    linear1: Proj,
    linear2: Proj,
}

struct Loader {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl Loader {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow!("checkpoint lacks tensor {name}"))?;
        Ok(t.to_dtype(DType::F32)?.to_device(&self.device)?)
    }

    fn norm(&mut self, prefix: &str, bias: bool, eps: f64) -> Result<Norm> {
        Ok(Norm {
            weight: self.take(&format!("{prefix}.weight"))?,
            bias: if bias {
                Some(self.take(&format!("{prefix}.bias"))?)
            } else {
                None
            },
            eps,
        })
    }

    fn frozen(&mut self, weight: &str, bias: Option<&str>) -> Result<Proj> {
        let w_t = self.take(weight)?.t()?.contiguous()?;
        let bias = bias.map(|b| self.take(b)).transpose()?;
        Ok(Proj {
            module: weight.trim_end_matches(".weight").to_string(),
            w_t,
            bias,
            lora: None,
            quantized: false,
        })
    }

    fn encoder_linear(
        &mut self,
        module: String,
        bits: u32,
        lora: Option<(LoraConfig, u64)>,
    ) -> Result<Proj> {
        let dense = self
            .tensors
            .remove(&format!("{module}.weight"))
            .ok_or_else(|| anyhow!("checkpoint lacks tensor {module}.weight"))?
            .to_dtype(DType::F32)?;
        let (rows, cols) = dense.dims2()?;
        let mut values: Vec<f32> = dense.flatten_all()?.to_vec1()?;
        let group =
            LAYA_QLORA_BASE_IS_THE_MLX_AFFINE_ROUND_TRIP_OF_EVERY_ENCODER_LINEAR_AT_THIS_GROUP_SIZE;
        ensure!(bits == 16 || cols % group == 0, "{module} has {cols} input features, not a multiple of the {group}-wide quantization group");
        match bits {
            4 => {
                let (words, scales, biases) =
                    nv_quant::mlx_affine::quantize_mlx_affine_linear(&values, rows, cols, group);
                values = nv_quant::mlx_affine::dequantize_mlx_affine_linear(
                    &words, &scales, &biases, rows, cols, group,
                );
            }
            8 => values
                .chunks_mut(group)
                .for_each(mlx_affine_u8_round_trip_of_one_group),
            _ => {}
        }
        let w_t = Tensor::from_vec(values, (rows, cols), &self.device)?
            .t()?
            .contiguous()?;
        let lora = lora
            .map(|(cfg, seed)| {
                LoraTrainable::from_features(rows, cols, cfg, seed, &self.device).map(Arc::new)
            })
            .transpose()?;
        Ok(Proj {
            module,
            w_t,
            bias: None,
            lora,
            quantized: bits < 16,
        })
    }
}

fn mlx_affine_u8_round_trip_of_one_group(group: &mut [f32]) {
    let levels = 255f32;
    let (w_min, w_max) = group
        .iter()
        .fold((f32::MAX, 0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let anchored_low = w_min.abs() > w_max.abs();
    let edge = if anchored_low { w_min } else { w_max };
    let mut scale = ((w_max - w_min) / levels).max(1e-7) * if anchored_low { 1.0 } else { -1.0 };
    let q0 = (edge / scale).round();
    let bias = if q0 == 0.0 {
        0.0
    } else {
        scale = edge / q0;
        edge
    };
    let (stored_scale, stored_bias) = (
        half::bf16::from_f32(scale).to_f32(),
        half::bf16::from_f32(bias).to_f32(),
    );
    for v in group.iter_mut() {
        *v = ((*v - bias) / scale).round().clamp(0.0, levels) * stored_scale + stored_bias;
    }
}

pub struct Twin {
    cfg: EncoderConfig,
    device: Device,
    tok_embeddings: Tensor,
    emb_norm: Norm,
    layers: Vec<EncoderLayer>,
    final_norm: Norm,
    head: Vec<HeadLayer>,
    type_emb: Tensor,
    scorer_norm: Norm,
    scorer_hidden: Proj,
    scorer_out: Proj,
    rope_global: (Tensor, Tensor),
    rope_local: (Tensor, Tensor),
}

fn rope_table(theta: f64, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let angle = |pos: usize, i: usize| pos as f64 / theta.powf(2.0 * i as f64 / HEAD_DIM as f64);
    let table = |f: fn(f64) -> f64| -> Result<Tensor> {
        let v: Vec<f32> = (0..len)
            .flat_map(|p| (0..half).map(move |i| (p, i)))
            .map(|(p, i)| f(angle(p, i)) as f32)
            .collect();
        Ok(Tensor::from_vec(v, (len, half), device)?)
    };
    Ok((table(f64::cos)?, table(f64::sin)?))
}

impl Twin {
    pub fn load(
        dir: &Path,
        device: &Device,
        max_len: usize,
        head_layers: usize,
        base_bits: u32,
        lora: Option<(LoraConfig, u64, &[String])>,
    ) -> Result<Self> {
        ensure!(
            BASE_BITS_THE_TRAINER_ACCEPTS_WHERE_SIXTEEN_IS_THE_STORED_WEIGHTS_UNTOUCHED
                .contains(&base_bits),
            "--base-bits takes 4, 8 or 16, not {base_bits}"
        );
        let cfg_text = std::fs::read_to_string(dir.join("encoder/config.json"))
            .context("reading encoder/config.json")?;
        let cfg = EncoderConfig::from_json(&serde_json::from_str(&cfg_text)?)?;
        ensure!(
            cfg.hidden == cfg.heads * HEAD_DIM,
            "only 64-wide attention heads are supported"
        );
        let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu)
            .context("reading model.safetensors")?;
        let mut w = Loader {
            tensors,
            device: device.clone(),
        };
        let eps = cfg.norm_eps as f64;
        let mut layers = Vec::with_capacity(cfg.global.len());
        for (i, global) in cfg.global.iter().enumerate() {
            let p = format!("encoder.layers.{i}");
            let mut linear = |leaf: &str, slot: u64| -> Result<Proj> {
                let adapter = lora.and_then(|(c, seed, targets)| {
                    targets
                        .iter()
                        .any(|t| t == leaf)
                        .then_some((c, seed ^ ((i as u64) << 8) ^ slot))
                });
                w.encoder_linear(format!("{p}.{leaf}"), base_bits, adapter)
            };
            let (wqkv, wo, wi, mlp_wo) = (
                linear(ENCODER_TARGETS[0], 0)?,
                linear(ENCODER_TARGETS[1], 1)?,
                linear(ENCODER_TARGETS[2], 2)?,
                linear(ENCODER_TARGETS[3], 3)?,
            );
            layers.push(EncoderLayer {
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(w.norm(&format!("{p}.attn_norm"), false, eps)?)
                },
                mlp_norm: w.norm(&format!("{p}.mlp_norm"), false, eps)?,
                wqkv,
                wo,
                wi,
                mlp_wo,
                global: *global,
            });
        }
        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{i}");
            head.push(HeadLayer {
                norm1: w.norm(&format!("{p}.norm1"), true, 1e-5)?,
                in_proj: w.frozen(
                    &format!("{p}.self_attn.in_proj_weight"),
                    Some(&format!("{p}.self_attn.in_proj_bias")),
                )?,
                out_proj: w.frozen(
                    &format!("{p}.self_attn.out_proj.weight"),
                    Some(&format!("{p}.self_attn.out_proj.bias")),
                )?,
                norm2: w.norm(&format!("{p}.norm2"), true, 1e-5)?,
                linear1: w.frozen(
                    &format!("{p}.linear1.weight"),
                    Some(&format!("{p}.linear1.bias")),
                )?,
                linear2: w.frozen(
                    &format!("{p}.linear2.weight"),
                    Some(&format!("{p}.linear2.bias")),
                )?,
            });
        }
        Ok(Self {
            tok_embeddings: w.take("encoder.embeddings.tok_embeddings.weight")?,
            emb_norm: w.norm("encoder.embeddings.norm", false, eps)?,
            final_norm: w.norm("encoder.final_norm", false, eps)?,
            type_emb: w.take("type_emb.weight")?,
            scorer_norm: w.norm("scorer.0", true, 1e-5)?,
            scorer_hidden: w.frozen("scorer.1.weight", Some("scorer.1.bias"))?,
            scorer_out: w.frozen("scorer.3.weight", Some("scorer.3.bias"))?,
            rope_global: rope_table(cfg.theta_global, max_len, device)?,
            rope_local: rope_table(cfg.theta_local, max_len, device)?,
            layers,
            head,
            cfg,
            device: device.clone(),
        })
    }

    fn projections(&self) -> impl Iterator<Item = &Proj> {
        self.layers
            .iter()
            .flat_map(|l| [&l.wqkv, &l.wo, &l.wi, &l.mlp_wo])
    }

    pub fn adapters(&self) -> Vec<(String, Arc<LoraTrainable>)> {
        self.projections()
            .filter_map(|p| p.lora.clone().map(|l| (p.module.clone(), l)))
            .collect()
    }

    fn attention(
        &self,
        qkv: &Tensor,
        n: usize,
        len: usize,
        rope: Option<&(Tensor, Tensor)>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let heads = self.cfg.heads;
        let qkv = qkv
            .reshape((n, len, 3, heads, HEAD_DIM))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let (mut q, mut k, v) = (
            (qkv.get(0)? * (1.0 / (HEAD_DIM as f64).sqrt()))?,
            qkv.get(1)?,
            qkv.get(2)?,
        );
        if let Some((cos, sin)) = rope {
            let (cos, sin) = (cos.narrow(0, 0, len)?, sin.narrow(0, 0, len)?);
            q = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
            k = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
        }
        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        if let Some(bias) = bias {
            scores = scores.broadcast_add(bias)?;
        }
        let probs = nv_layers::attn::softmax_last_dim_that_keeps_q_and_k_on_the_graph(&scores)?;
        Ok(probs
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n * len, heads * HEAD_DIM))?)
    }

    fn logits(&self, items: &[&Item], train: bool) -> Result<Vec<Tensor>> {
        let n = items.len();
        let len = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
        let masked = THE_TWIN_MASKS_WITH_THE_SAME_FINITE_BIAS_SERVING_USES;
        let mut ids = vec![0u32; n * len];
        let mut types = vec![0u32; n * len];
        let mut key_bias = vec![0f32; n * len];
        let mut rows: Vec<u32> = Vec::new();
        for (r, it) in items.iter().enumerate() {
            ids[r * len..r * len + it.ids.len()].copy_from_slice(&it.ids);
            types[r * len..(r + 1) * len].fill(it.qtype as u32);
            key_bias[r * len + it.ids.len()..(r + 1) * len].fill(masked);
            rows.extend(it.markers.iter().map(|m| (r * len + m) as u32));
        }
        let half = self.cfg.half_window;
        let mut window = vec![masked; len * len];
        for i in 0..len {
            window[i * len + i.saturating_sub(half)..i * len + (i + half + 1).min(len)].fill(0.0);
        }
        let global_bias = Tensor::from_vec(key_bias, (n, 1, 1, len), &self.device)?;
        let local_bias = global_bias.broadcast_add(&Tensor::from_vec(
            window,
            (1, 1, len, len),
            &self.device,
        )?)?;
        let ids = Tensor::from_vec(ids, n * len, &self.device)?;
        let types = Tensor::from_vec(types, n * len, &self.device)?;

        let mut h = self
            .emb_norm
            .forward(&self.tok_embeddings.index_select(&ids, 0)?)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => norm.forward(&h)?,
                None => h.clone(),
            };
            let (rope, bias) = if layer.global {
                (&self.rope_global, &global_bias)
            } else {
                (&self.rope_local, &local_bias)
            };
            let attn = self.attention(
                &layer.wqkv.forward(&normed, train)?,
                n,
                len,
                Some(rope),
                Some(bias),
            )?;
            h = (h + layer.wo.forward(&attn, train)?)?;
            let wi = layer.wi.forward(&layer.mlp_norm.forward(&h)?, train)?;
            let inter = wi.dim(D::Minus1)? / 2;
            let gated = (wi.narrow(D::Minus1, 0, inter)?.contiguous()?.gelu_erf()?
                * wi.narrow(D::Minus1, inter, inter)?.contiguous()?)?;
            h = (h + layer.mlp_wo.forward(&gated, train)?)?;
        }
        h = self.final_norm.forward(&h)?;
        h = (h + self.type_emb.index_select(&types, 0)?)?;
        for layer in &self.head {
            let attn = self.attention(
                &layer.in_proj.forward(&layer.norm1.forward(&h)?, train)?,
                n,
                len,
                None,
                Some(&global_bias),
            )?;
            h = (h + layer.out_proj.forward(&attn, train)?)?;
            let ff = layer.linear2.forward(
                &layer
                    .linear1
                    .forward(&layer.norm2.forward(&h)?, train)?
                    .relu()?,
                train,
            )?;
            h = (h + ff)?;
        }
        let total = rows.len();
        let picked = h.index_select(&Tensor::from_vec(rows, total, &self.device)?, 0)?;
        let scored = self
            .scorer_out
            .forward(
                &self
                    .scorer_hidden
                    .forward(&self.scorer_norm.forward(&picked)?, train)?
                    .gelu_erf()?,
                train,
            )?
            .flatten_all()?;
        let mut out = Vec::with_capacity(n);
        let mut at = 0;
        for it in items {
            out.push(scored.narrow(0, at, it.markers.len())?);
            at += it.markers.len();
        }
        Ok(out)
    }

    pub fn marker_logits(
        &self,
        sequences: &[Vec<u32>],
        markers: &[Vec<usize>],
        qtypes: &[QType],
    ) -> Result<Vec<Vec<f32>>> {
        let items: Vec<Item> = sequences
            .iter()
            .zip(markers)
            .zip(qtypes)
            .map(|((ids, markers), qtype)| Item {
                ids: ids.clone(),
                markers: markers.clone(),
                qtype: *qtype,
                target: Vec::new(),
                label: 0,
            })
            .collect();
        let refs: Vec<&Item> = items.iter().collect();
        self.logits(&refs, false)?
            .iter()
            .map(|t| Ok(t.to_vec1::<f32>()?))
            .collect()
    }

    fn loss_of(&self, logits: &Tensor, it: &Item) -> Result<Tensor> {
        let k = it.target.len();
        let target = Tensor::from_vec(it.target.clone(), k, &self.device)?;
        let shifted = logits.broadcast_sub(&logits.max_keepdim(0)?.detach())?;
        let log_p = shifted.broadcast_sub(&shifted.exp()?.sum_keepdim(0)?.log()?)?;
        let p = log_p.exp()?;
        let floor = Tensor::full(
            UPSTREAM_PROPER_REWARD_FLOORS_THE_LOG_SCORE_AT as f32,
            k,
            &self.device,
        )?;
        let log_score = (&target * log_p.maximum(&floor)?)?.sum_all()?;
        let spherical = ((&target * &p)?.sum_all()? / p.sqr()?.sum_all()?.sqrt()?)?;
        let mut reward =
            (log_score + (spherical * UPSTREAM_PROPER_REWARD_WEIGHS_THE_SPHERICAL_SCORE_BY)?)?;
        if it.qtype == QType::Score && k > 1 {
            let lower: Vec<f32> = (0..k * k)
                .map(|c| if c % k <= c / k { 1.0 } else { 0.0 })
                .collect();
            let lower = Tensor::from_vec(lower, (k, k), &self.device)?;
            let gap = lower.matmul(&(p - target)?.reshape((k, 1))?)?;
            let rps = (gap.sqr()?.sum_all()? / (k - 1) as f64)?;
            reward = (reward - (rps * UPSTREAM_PROPER_REWARD_WEIGHS_THE_RANKED_PROBABILITY_SCORE_OF_ORDINAL_QUESTIONS_BY)?)?;
        }
        Ok(reward.neg()?)
    }

    fn batch_loss(&self, items: &[&Item], train: bool) -> Result<(Tensor, usize)> {
        let logits = self.logits(items, train)?;
        let mut total = Tensor::zeros((), DType::F32, &self.device)?;
        let mut correct = 0;
        for (l, it) in logits.iter().zip(items) {
            total = (total + self.loss_of(l, it)?)?;
            if !train {
                let v: Vec<f32> = l.to_vec1()?;
                let best = (0..v.len())
                    .max_by(|a, b| v[*a].total_cmp(&v[*b]).then(b.cmp(a)))
                    .unwrap_or(0);
                correct += usize::from(best == it.label);
            }
        }
        Ok(((total / items.len() as f64)?, correct))
    }

    fn evaluate(&self, items: &[Item], batch: usize) -> Result<Eval> {
        let (mut loss, mut correct) = (0f64, 0usize);
        for chunk in items.chunks(batch.max(1)) {
            let refs: Vec<&Item> = chunk.iter().collect();
            let (l, c) = self.batch_loss(&refs, false)?;
            loss += l.to_scalar::<f32>()? as f64 * chunk.len() as f64;
            correct += c;
        }
        Ok(Eval {
            loss: (loss / items.len() as f64) as f32,
            accuracy: correct as f32 / items.len() as f32,
        })
    }

    pub fn write_merged_checkpoint(&self, base: &Path, out: &Path) -> Result<()> {
        let mut tensors =
            candle_core::safetensors::load(base.join("model.safetensors"), &Device::Cpu)?;
        for proj in self
            .projections()
            .filter(|p| p.quantized || p.lora.is_some())
        {
            let name = format!("{}.weight", proj.module);
            let stored = tensors
                .get(&name)
                .ok_or_else(|| anyhow!("base checkpoint lacks {name}"))?
                .dtype();
            tensors.insert(
                name,
                proj.merged()?.to_dtype(stored)?.to_device(&Device::Cpu)?,
            );
        }
        for dir in ["encoder", "tokenizer"] {
            std::fs::create_dir_all(out.join(dir))?;
            for entry in std::fs::read_dir(base.join(dir))? {
                let entry = entry?;
                if entry.path().is_file() {
                    std::fs::copy(entry.path(), out.join(dir).join(entry.file_name()))?;
                }
            }
        }
        std::fs::copy(
            base.join("rl_agent_config.json"),
            out.join("rl_agent_config.json"),
        )?;
        candle_core::safetensors::save(&tensors, out.join("model.safetensors"))?;
        Ok(())
    }
}

fn framing(base: &Path) -> Result<(Checkpoint, usize, usize)> {
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(base.join("rl_agent_config.json"))
            .context("reading rl_agent_config.json")?,
    )?;
    let number = |key: &str, default: u64| {
        cfg.get(key)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(default) as usize
    };
    let tokenizer_only = Settings {
        devices: Vec::new(),
        dtype: DType::F32,
        flash: false,
    };
    Ok((
        Checkpoint::load(base, &tokenizer_only)?,
        number("max_len", 512),
        number("head_layers", 2),
    ))
}

pub fn run(args: &LayaTrainArgs, device: &Device) -> Result<LayaTrainSummary> {
    let _ = LAYA_TRAINS_THROUGH_A_DIFFERENTIABLE_TWIN_BECAUSE_SERVING_LAYER_NORM_ROPE_SOFTMAX_AND_FLASH_ATTENTION_CARRY_NO_BACKWARD;
    let (framing, max_len, head_layers) = framing(&args.base)?;
    let train = records(&args.data)?;
    let held_out = match &args.eval_data {
        Some(path) => records(path)?,
        None => train.clone(),
    };
    let eval_items: Vec<Item> = held_out
        .iter()
        .map(|r| item(&framing, r, None))
        .collect::<Result<_>>()?;

    let eval_dense_base = if args.base_bits < 16 {
        Some(
            Twin::load(&args.base, device, max_len, head_layers, 16, None)?
                .evaluate(&eval_items, args.batch)?,
        )
    } else {
        None
    };

    let cfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha.unwrap_or(args.rank as f64),
        dropout: 0.0,
    };
    let twin = Twin::load(
        &args.base,
        device,
        max_len,
        head_layers,
        args.base_bits,
        Some((cfg, args.seed, &args.targets)),
    )?;
    let adapters = twin.adapters();
    ensure!(
        !adapters.is_empty(),
        "no encoder linear matches --target {:?}; choose from {ENCODER_TARGETS:?}",
        args.targets
    );
    let vars: Vec<Var> = adapters
        .iter()
        .flat_map(|(_, l)| l.trainable_vars())
        .collect();
    let eval_trained_base = twin.evaluate(&eval_items, args.batch)?;

    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: args.lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let mut rng = DetRng::new(args.seed);
    let mut losses = Vec::with_capacity(args.steps);
    let mut every_gradient_was_finite = true;
    let started = std::time::Instant::now();
    for step in 0..args.steps {
        let drawn: Vec<Item> = (0..args.batch.max(1))
            .map(|_| {
                let record = &train[(rng.next_u64() % train.len() as u64) as usize];
                item(&framing, record, args.shuffle_options.then_some(&mut rng))
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&Item> = drawn.iter().collect();
        let (loss, _) = twin.batch_loss(&refs, true)?;
        let grads = loss.backward()?;
        let mut norm = Tensor::zeros((), DType::F32, device)?;
        for (v, (module, _)) in vars.iter().zip(adapters.iter().flat_map(|a| [a, a])) {
            let g = grads.get(v).ok_or_else(|| anyhow!("step {step}: a LoRA factor of {module} received no gradient at all, so it would save untrained"))?;
            norm = (norm + g.sqr()?.sum_all()?)?;
        }
        every_gradient_was_finite &= norm.to_scalar::<f32>()?.is_finite();
        opt.step(&grads)?;
        losses.push(loss.to_scalar::<f32>()?);
    }
    let seconds_per_step = started.elapsed().as_secs_f64() / args.steps.max(1) as f64;

    let eval_adapter = twin.evaluate(&eval_items, args.batch)?;
    let mut smallest_lora_b_amax = f32::INFINITY;
    for (_, lora) in &adapters {
        smallest_lora_b_amax =
            smallest_lora_b_amax.min(lora.b_tensor().abs()?.max_all()?.to_scalar::<f32>()?);
    }
    let entries: Vec<PeftEntry<'_>> = adapters
        .iter()
        .map(|(module, lora)| PeftEntry {
            module: module.clone(),
            lora: lora.as_ref(),
        })
        .collect();
    save_peft(&args.out, &entries, &args.base.display().to_string())?;
    if let Some(merged) = &args.merged_out {
        twin.write_merged_checkpoint(&args.base, merged)?;
    }
    Ok(LayaTrainSummary {
        device: format!("{device:?}"),
        train_items: train.len(),
        eval_items: eval_items.len(),
        losses,
        seconds_per_step,
        eval_dense_base,
        eval_trained_base,
        eval_adapter,
        every_gradient_was_finite,
        smallest_lora_b_amax,
        adapter_modules: adapters.len(),
    })
}
