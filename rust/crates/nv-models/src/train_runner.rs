use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use nv_layers::linear::{Linear, LoraDeltaHook};
use nv_train::qwen38::{
    quant_resident_arm_opted_in,
    NV_TRAIN_QUANT_RESIDENT_KEEPS_NVFP4_AND_FP8_MODULES_RESIDENT_SO_TRAINING_RUNS_THE_SERVING_QUANT_GEMM as NV_TRAIN_QUANT_RESIDENT,
};
use nv_train::{
    ce_mean_over_unmasked_positions,
    eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order,
    grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum,
    load_jsonl_token_dataset as load_dataset, lora_delta, max_abs_diff, save_peft_raw, LoraConfig,
    LoraTrainable, RawLora, TokenRow, TrainingLoraHook,
    JSONL_MAX_SEQ_IS_512_SHARED_BY_BOTH_ENDPOINT_ARMS_BECAUSE_THE_DEFAULT_QWEN_WINDOWS_GDN_GRAPH_AT_512_IS_ABOUT_14_GB as JSONL_MAX_SEQ,
};
use nv_layers::norm::RmsNorm;
use nv_layers::rope::Rope;
use nv_weights::{TensorSource, WeightLoader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::dense_train::{naive_sdpa, DenseTrainModel};
use crate::eagle3_train::{Eagle3TrainConfig, Eagle3TrainHead};
use crate::gemma4::{mlp_forward, Gemma4Attention, Gemma4Config, Gemma4Mlp, LayerType};
use crate::gemma4_moe::{build_rope, Gemma4Moe, Gemma4MoeConfig};

pub enum BaseModel {
    Moe(Gemma4Moe),
    Dense(DenseTrainModel),
}

impl BaseModel {
    pub fn vocab(&self) -> usize {
        match self {
            BaseModel::Moe(m) => m.config().base.vocab_size,
            BaseModel::Dense(d) => d.vocab_size(),
        }
    }
    pub fn hidden(&self) -> usize {
        match self {
            BaseModel::Moe(m) => m.config().base.hidden_size,
            BaseModel::Dense(d) => d.config().hidden_size,
        }
    }
    pub fn device(&self) -> Device {
        match self {
            BaseModel::Moe(m) => m.device().clone(),
            BaseModel::Dense(d) => d.device().clone(),
        }
    }

    fn attn_mlp_layers(&self) -> Vec<(&Gemma4Attention, &Gemma4Mlp)> {
        match self {
            BaseModel::Moe(m) => m.layers().iter().map(|l| (&l.self_attn, &l.mlp)).collect(),
            BaseModel::Dense(d) => d.layers().iter().map(|l| (&l.self_attn, &l.mlp)).collect(),
        }
    }

    pub fn forward_logits(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        match self {
            BaseModel::Moe(m) => {
                let seq = tokens.dims().get(1).copied().unwrap_or(0);
                let mut cache = m.new_kv_cache(seq.max(2))?;
                m.forward_with_cache(tokens, positions, &mut cache)
            }
            BaseModel::Dense(d) => d.forward_logits(tokens, positions),
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, BaseModel::Dense(_))
    }

    pub fn layer_counts(&self) -> (usize, usize) {
        match self {
            BaseModel::Moe(m) => {
                let n = m.layers().len();
                (n, n)
            }
            BaseModel::Dense(d) => (d.layers().len(), d.full_num_layers()),
        }
    }
}

struct Nvfp4DequantSource<'a> {
    inner: &'a WeightLoader,
    device: Device,
}

impl<'a> Nvfp4DequantSource<'a> {
    fn dense_nvfp4_module(&self, name: &str) -> Option<String> {
        let module = name.strip_suffix(".weight")?;
        let has_scale = self.inner.has(&format!("{module}.weight_scale"))
            && self.inner.has(&format!("{module}.weight_scale_2"));
        if has_scale {
            Some(module.to_string())
        } else {
            None
        }
    }

    fn dense_w4a16_pack_module(&self, name: &str) -> Option<String> {
        let module = name.strip_suffix(".weight")?;
        if self.inner.has(&format!("{module}.weight_packed"))
            && self.inner.has(&format!("{module}.weight_scale"))
            && !self.inner.has(&format!("{module}.weight_scale_2"))
        {
            Some(module.to_string())
        } else {
            None
        }
    }

    fn dequantize_w4a16_pack(&self, module: &str, dtype: DType) -> Result<Tensor> {
        let packed_name = format!("{module}.weight_packed");
        let shape = self
            .inner
            .shape_of(&packed_name)
            .with_context(|| format!("missing shape for {packed_name}"))?;
        if shape.len() != 2 {
            bail!("{packed_name}: expected rank-2 packed [n, k/8 words], got {shape:?}");
        }
        let raw = self.inner.raw_bytes(&packed_name)?;
        let words: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let scale_raw = self.inner.raw_bytes(&format!("{module}.weight_scale"))?;
        let scales: Vec<f32> = scale_raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        let n = shape[0];
        let words_per_row = words.len() / n;
        let k = words_per_row * 8;
        let groups_per_row = scales.len() / n;
        if groups_per_row == 0 || !k.is_multiple_of(groups_per_row) {
            bail!(
                "{module}: scale length {} does not divide {n}x{k}",
                scales.len()
            );
        }
        let gs = k / groups_per_row;
        let mut vals = vec![0f32; n * k];
        for row in 0..n {
            let wbase = row * words_per_row;
            let sbase = row * groups_per_row;
            for kw in 0..words_per_row {
                let word = words[wbase + kw];
                for e in 0..8usize {
                    let k_idx = kw * 8 + e;
                    let q = ((word >> (4 * e)) & 0xF) as f32 - 8.0;
                    vals[row * k + k_idx] = q * scales[sbase + k_idx / gs];
                }
            }
        }
        let t = Tensor::from_vec(vals, (n, k), &Device::Cpu)?.to_dtype(dtype)?;
        Ok(t.to_device(&self.device)?)
    }
}

fn scalar_f32(inner: &WeightLoader, name: &str) -> Result<f32> {
    let v: Vec<f32> = inner.get(name, DType::F32)?.flatten_all()?.to_vec1()?;
    Ok(*v.first().unwrap_or(&1.0))
}

impl<'a> TensorSource for Nvfp4DequantSource<'a> {
    fn has(&self, name: &str) -> bool {
        self.inner.has(name) || self.dense_w4a16_pack_module(name).is_some()
    }

    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        if let Some(module) = self.dense_w4a16_pack_module(name) {
            return self.dequantize_w4a16_pack(&module, dtype);
        }
        if let Some(module) = self.dense_nvfp4_module(name) {
            let shape = self
                .inner
                .shape_of(name)
                .with_context(|| format!("missing shape for packed nvfp4 {name}"))?;
            if shape.len() != 2 {
                bail!("packed nvfp4 {name}: expected rank-2 [out, in/2], got {shape:?}");
            }
            let out_f = shape[0];
            let in_f = shape[1] * 2;
            let packed = self.inner.raw_bytes(name)?.to_vec();
            let scales = self
                .inner
                .raw_bytes(&format!("{module}.weight_scale"))?
                .to_vec();
            let safe_recip = |x: f32| {
                if x == 0.0 || !x.is_finite() {
                    1.0
                } else {
                    1.0 / x
                }
            };
            let weight_mult = safe_recip(safe_recip(scalar_f32(
                self.inner,
                &format!("{module}.weight_scale_2"),
            )?));
            let vals = nv_quant::nvfp4::dequantize_packed_linear(
                &packed,
                &scales,
                out_f,
                in_f,
                weight_mult,
            );

            let t = Tensor::from_vec(vals, (out_f, in_f), &Device::Cpu)?.to_dtype(dtype)?;
            return Ok(t.to_device(&self.device)?);
        }
        Ok(self.inner.get(name, dtype)?.to_device(&self.device)?)
    }
}

fn is_packed_nvfp4(weights: &WeightLoader) -> bool {
    weights
        .names()
        .iter()
        .any(|n| n.ends_with(".weight_scale_2"))
}

fn is_packed_w4a16(weights: &WeightLoader) -> bool {
    weights
        .names()
        .iter()
        .any(|n| n.ends_with(".weight_packed"))
}

fn open_base_weights(path: &Path, device: &Device) -> Result<WeightLoader> {
    let st = path.join("model.safetensors");
    if st.is_file() {
        WeightLoader::open_file(&st, device)
            .with_context(|| format!("open weights {}", st.display()))
    } else {
        WeightLoader::open_dir(path, device)
            .with_context(|| format!("open sharded weights {}", path.display()))
    }
}

pub fn base_is_packed_nvfp4(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    open_base_weights(path, &Device::Cpu)
        .map(|w| is_packed_nvfp4(&w))
        .unwrap_or(false)
}

#[derive(Clone, Debug)]
pub struct TrainArgs {
    pub base: PathBuf,
    pub data: PathBuf,
    pub eval_data: Option<PathBuf>,
    pub out: PathBuf,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
}

impl TrainArgs {
    pub fn default_targets() -> Vec<String> {
        ["q", "k", "v", "o", "gate", "up", "down"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct TrainSummary {
    pub losses: Vec<f32>,
    pub eval_loss_adapter: Option<f32>,
    pub eval_loss_base: Option<f32>,
    pub eval_ce_adapter: Vec<f32>,
    pub eval_ce_base: Vec<f32>,
    pub serving_equiv_maxabs: f32,
    pub trainable_vars: usize,
    pub num_examples: usize,
    pub num_eval_examples: usize,
    pub base_dtype: String,
    pub resident_base: bool,
    pub nvfp4_base: bool,
    pub dense_base: bool,
    pub layers_built: usize,
    pub config_layers: usize,
    pub deterministic: bool,
    pub device: String,
    pub checkpointed: bool,
    pub all_grads_finite: bool,
    pub lmhead_chunk: usize,
    pub modules: Vec<String>,
    pub adapter_path: PathBuf,
    pub config_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct SpeculatorSpec {
    pub drafter_dir: PathBuf,
    pub rank: usize,
    pub lr_ratio: f64,
}

impl SpeculatorSpec {
    pub fn eagle3_medusa2_defaults_rank_32_lr_ratio_4(drafter_dir: PathBuf) -> Self {
        Self {
            drafter_dir,
            rank: 32,
            lr_ratio: 4.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeculatorTrainSummary {
    pub kind: String,
    pub losses: Vec<f32>,
    pub trainable_vars: usize,
    pub all_grads_finite: bool,
    pub modules: Vec<String>,
    pub adapter_path: PathBuf,
    pub config_path: PathBuf,
}

pub struct DenseAuxTapForward {
    embed_weight: Tensor,
    embed_scale: f32,
    final_norm: RmsNorm,
    lm_head: Linear,
    sliding_rope: Rope,
    full_rope: Rope,
}

impl DenseAuxTapForward {
    pub fn open(base: &Path, model: &DenseTrainModel, base_dtype: DType) -> Result<Self> {
        let device = model.device().clone();
        let cfg = model.config();
        let load_device = if device.is_cuda() {
            Device::Cpu
        } else {
            device.clone()
        };
        let weights = open_base_weights(base, &load_device)?;
        let embed_weight = weights
            .get("model.language_model.embed_tokens.weight", base_dtype)?
            .to_device(&device)?;
        let final_norm = RmsNorm::new(
            weights
                .get("model.language_model.norm.weight", base_dtype)?
                .to_device(&device)?,
            cfg.rms_norm_eps,
        );
        let lm_head_weight = if cfg.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", base_dtype)?
                .to_device(&device)?
        };
        let lm_head = Linear::new_no_pretranspose(lm_head_weight, None)?;
        let sliding_rope = build_rope(
            cfg.head_dim,
            cfg.rope_theta_for(LayerType::SlidingAttention),
            1.0,
            cfg.max_position_embeddings,
            &device,
        )?;
        let full_rope = build_rope(
            cfg.global_head_dim,
            cfg.rope_theta_for(LayerType::FullAttention),
            cfg.rope_partial_factor_for(LayerType::FullAttention),
            cfg.max_position_embeddings,
            &device,
        )?;
        Ok(Self {
            embed_weight,
            embed_scale: (cfg.hidden_size as f32).sqrt(),
            final_norm,
            lm_head,
            sliding_rope,
            full_rope,
        })
    }

    pub fn taps_and_teacher_logits_detached(
        &self,
        model: &DenseTrainModel,
        ids: &[u32],
        tap_layers: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let device = model.device();
        let cfg = model.config();
        let seq = ids.len();
        if seq < 2 {
            bail!("tap forward: need >=2 tokens, got {seq}");
        }
        for &li in tap_layers {
            if li >= model.layers().len() {
                bail!(
                    "tap forward: drafter taps layer {li} but only {} layers are built",
                    model.layers().len()
                );
            }
        }
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), device)?;
        let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, device)?;
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let mut hidden = self
            .embed_weight
            .detach()
            .index_select(&tokens_flat, 0)?
            .reshape((1usize, seq, cfg.hidden_size))?
            .to_dtype(DType::F32)?
            .affine(self.embed_scale as f64, 0.0)?;
        let mut taps: Vec<Option<Tensor>> = vec![None; tap_layers.len()];
        for li in 0..model.layers().len() {
            hidden = self
                .layer_forward_replica(model, li, &hidden, &positions, seq)?
                .detach();
            for (slot, &tl) in taps.iter_mut().zip(tap_layers.iter()) {
                if tl == li && slot.is_none() {
                    *slot = Some(hidden.clone());
                }
            }
        }
        let normed = self.final_norm.forward_candle(&hidden)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        let cap = cfg.final_logit_softcapping;
        let logits = if cap > 0.0 {
            logits
                .affine(1.0 / cap as f64, 0.0)?
                .tanh()?
                .affine(cap as f64, 0.0)?
        } else {
            logits
        };
        let mut parts: Vec<Tensor> = Vec::with_capacity(tap_layers.len());
        for (slot, &tl) in taps.into_iter().zip(tap_layers.iter()) {
            let t = slot.with_context(|| format!("tap layer {tl} was never captured"))?;
            parts.push(t.squeeze(0)?);
        }
        let aux = Tensor::cat(&parts, 1)?.detach();
        Ok((aux, logits.squeeze(0)?.detach()))
    }

    fn layer_forward_replica(
        &self,
        model: &DenseTrainModel,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let layer = &model.layers()[idx];
        let residual_attn = x.clone();
        let normed_pre_attn = layer.input_layernorm.forward_candle(x)?;
        let attn_out = self.attention_forward_replica(model, idx, &normed_pre_attn, positions, seq)?;
        let attn_post = layer.post_attention_layernorm.forward_candle(&attn_out)?;
        let after_attn = attn_post.add(&residual_attn)?;
        let normed_pre_mlp = layer
            .pre_feedforward_layernorm
            .forward_candle(&after_attn)?;
        let mlp_out = mlp_forward(&layer.mlp, &normed_pre_mlp)?;
        let mlp_post = layer.post_feedforward_layernorm.forward_candle(&mlp_out)?;
        let scale =
            Tensor::new(layer.layer_scalar_host, model.device())?.to_dtype(after_attn.dtype())?;
        Ok(after_attn.add(&mlp_post)?.broadcast_mul(&scale)?)
    }

    fn attention_forward_replica(
        &self,
        model: &DenseTrainModel,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let cfg = model.config();
        let layer = &model.layers()[idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = cfg.head_dim_for(kind);
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_kv_heads_for(kind);
        let rope = match kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        let window = match kind {
            LayerType::SlidingAttention => Some(cfg.sliding_window),
            LayerType::FullAttention => None,
        };
        let bsz = x.dims()[0];
        let (q_raw, k_raw, v_raw) = if x.device().is_cuda() {
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
            (q, k, v)
        } else {
            attn.qkv_forward(x)?
        };
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
        let attn_out = naive_sdpa(
            &q_rot, &k_rot, &v_for, bsz, n_q, n_kv, head_dim, seq, window,
        )?;
        let attn_out_flat = attn_out
            .to_dtype(x.dtype())?
            .reshape((bsz, seq, n_q * head_dim))?;
        attn.o_proj.forward(&attn_out_flat)
    }
}

const SPECULATOR_LORA_SEEDS_LIVE_IN_A_DISJOINT_SPACE_FROM_TARGET_SEEDS: u64 = 0xEA61_E300_0000;

struct Eagle3Cotrain {
    head: Eagle3TrainHead,
    taps: DenseAuxTapForward,
    tap_layers: Vec<usize>,
    loras: Vec<(String, Arc<nv_train::LoraTrainable>)>,
    vars: Vec<Var>,
    opt: AdamW,
    losses: Vec<f32>,
    all_grads_finite: bool,
}

impl Eagle3Cotrain {
    fn build(
        args: &TrainArgs,
        spec: &SpeculatorSpec,
        model: &BaseModel,
        base_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let dense = match model {
            BaseModel::Dense(d) => d,
            BaseModel::Moe(_) => bail!(
                "speculator co-training expected a dense gemma-4 trunk (the eagle3 drafter \
                 consumes dense layer-output taps); base {} loaded as MoE, which has no tap path",
                args.base.display()
            ),
        };
        if spec.rank == 0 {
            bail!("speculator rank must be positive");
        }
        if !(spec.lr_ratio > 0.0) {
            bail!("speculator lr_ratio must be positive, got {}", spec.lr_ratio);
        }
        let cfg = Eagle3TrainConfig::from_drafter_dir(&spec.drafter_dir)?;
        let hidden = dense.config().hidden_size;
        if cfg.hidden_size != hidden {
            bail!(
                "eagle3 drafter at {} is dimensioned for hidden_size {} but base {} has \
                 hidden_size {hidden}; a drafter only co-trains against the target it was \
                 built for (the RedHatAI drafter fits gemma-4 31B, not the E4B testbed)",
                spec.drafter_dir.display(),
                cfg.hidden_size,
                args.base.display()
            );
        }
        let vocab = dense.vocab_size();
        if cfg.target_vocab_size != vocab {
            bail!(
                "eagle3 drafter at {} expects target vocab {} but base {} has vocab {vocab}",
                spec.drafter_dir.display(),
                cfg.target_vocab_size,
                args.base.display()
            );
        }
        let built = dense.layers().len();
        if let Some(&max_tap) = cfg.aux_layer_ids.iter().max() {
            if max_tap >= built {
                bail!(
                    "eagle3 drafter taps layer {max_tap} but the base built only {built} of {} \
                     layers (NV_TRAIN_MAX_LAYERS truncation removes the tap)",
                    dense.full_num_layers()
                );
            }
        }
        let head = Eagle3TrainHead::load(&cfg, &spec.drafter_dir, device, base_dtype)?;
        let taps = DenseAuxTapForward::open(&args.base, dense, base_dtype)?;
        let loras = head.attach_head_loras(
            spec.rank,
            args.seed
                .wrapping_add(SPECULATOR_LORA_SEEDS_LIVE_IN_A_DISJOINT_SPACE_FROM_TARGET_SEEDS),
        )?;
        let mut vars: Vec<Var> = Vec::new();
        for (_, l) in &loras {
            vars.extend(l.trainable_vars());
        }
        let opt = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: args.lr * spec.lr_ratio,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        let tap_layers = cfg.aux_layer_ids.clone();
        Ok(Self {
            head,
            taps,
            tap_layers,
            loras,
            vars,
            opt,
            losses: Vec::new(),
            all_grads_finite: true,
        })
    }

    fn step(&mut self, dense: &DenseTrainModel, dataset: &[TokenRow]) -> Result<()> {
        let mut total: Option<Tensor> = None;
        for row in dataset {
            let (aux, teacher) =
                self.taps
                    .taps_and_teacher_logits_detached(dense, &row.ids, &self.tap_layers)?;
            let draft = self
                .head
                .forward_train_shift_pairs_embed_of_token_i_plus_1_with_aux_row_i_like_serving(
                    &row.ids, &aux,
                )?;
            let loss = self
                .head
                .shift_distill_loss_pairs_draft_row_i_with_teacher_row_i_plus_1(
                    &draft,
                    &teacher,
                    row.completion_start,
                )?;
            total = accumulate(total, loss)?;
        }
        let loss = total.context("empty batch for speculator step")?;
        let lv = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        for v in &self.vars {
            if let Some(g) = grads.get(v) {
                if !grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(g)? {
                    self.all_grads_finite = false;
                }
            }
        }
        self.opt.step(&grads)?;
        self.losses.push(lv);
        Ok(())
    }

    fn save(self, out: &Path, spec: &SpeculatorSpec) -> Result<SpeculatorTrainSummary> {
        let entries = Eagle3TrainHead::raw_entries(&self.loras);
        let spec_dir = out.join("speculator");
        save_peft_raw(
            &spec_dir,
            &entries,
            spec.rank,
            spec.rank as f64,
            0.0,
            &spec.drafter_dir.to_string_lossy(),
        )?;
        let cfg_path = out.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read back {}", cfg_path.display()))?;
        let mut v: serde_json::Value = serde_json::from_str(&raw)?;
        v.as_object_mut()
            .context("adapter_config.json is not a json object")?
            .insert(
                "nv_speculator_kind".to_string(),
                serde_json::Value::String("eagle3".to_string()),
            );
        std::fs::write(&cfg_path, serde_json::to_string_pretty(&v)?)?;
        Ok(SpeculatorTrainSummary {
            kind: "eagle3".to_string(),
            losses: self.losses,
            trainable_vars: self.vars.len(),
            all_grads_finite: self.all_grads_finite,
            modules: entries.iter().map(|e| e.module.clone()).collect(),
            adapter_path: spec_dir.join("adapter_model.safetensors"),
            config_path: spec_dir.join("adapter_config.json"),
        })
    }
}

pub fn select_device() -> Result<Device> {
    match std::env::var("NV_TRAIN_DEVICE").ok().as_deref() {
        Some("cuda") | Some("gpu") => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|e| anyhow::anyhow!("open CUDA device 0: {e}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("NV_TRAIN_DEVICE=cuda requires building nvk-train with --features cuda")
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(|e| anyhow::anyhow!("open Metal device 0: {e}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("NV_TRAIN_DEVICE=metal requires building nvk-train with --features metal")
            }
        }
        Some("cpu") => Ok(Device::Cpu),
        _ => {
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            {
                return Device::new_metal(0)
                    .map_err(|e| anyhow::anyhow!("open Metal device 0 (default on this build; NV_TRAIN_DEVICE=cpu opts out): {e}"));
            }
            #[allow(unreachable_code)]
            Ok(Device::Cpu)
        }
    }
}

fn checkpointing_tristate_defaults_on_for_cuda_dense_because_the_full_graph_oomed_on_real_e4b(
    device: &Device,
    dense: bool,
) -> Result<bool> {
    match std::env::var("NV_TRAIN_CKPT").ok().as_deref() {
        Some("1") => {
            if !dense {
                bail!("NV_TRAIN_CKPT=1 is only supported for a dense base");
            }
            Ok(true)
        }
        Some("0") => Ok(false),
        _ => Ok(dense && device.is_cuda()),
    }
}

fn grad_store_from(
    vars: &[Var],
    grads: &[Option<Tensor>],
) -> Result<candle_core::backprop::GradStore> {
    if let Some(i) = grads.iter().position(|g| g.is_none()) {
        bail!(
            "trainable var {i} of {} received no gradient at all; stepping it would use \
             the all-ones GradStore seed instead of a real gradient",
            vars.len()
        );
    }
    let mut gs = vars[0].as_tensor().sum_all()?.backward()?;
    for (v, g) in vars.iter().zip(grads.iter()) {
        if let Some(g) = g {
            gs.insert(v.as_tensor(), g.clone());
        }
    }
    Ok(gs)
}

pub fn load_base(path: &Path, device: &Device) -> Result<(BaseModel, DType)> {
    let (model, dtype, _resident) = load_base_resident_aware(path, device)?;
    Ok((model, dtype))
}

fn load_base_resident_aware(path: &Path, device: &Device) -> Result<(BaseModel, DType, bool)> {
    let resident = quant_resident_arm_opted_in();
    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        ensure!(
            !resident,
            "{NV_TRAIN_QUANT_RESIDENT}=1 on a gguf base: the gguf loader dequantizes to bf16 \
             and has no resident-Linear plumbing"
        );
        let model = Gemma4Moe::from_gguf(path, device, DType::BF16)
            .with_context(|| format!("load gguf base {}", path.display()))?;
        return Ok((BaseModel::Moe(model), DType::BF16, false));
    }
    if path.is_dir() {
        let cfg_path = path.join("config.json");

        let base_cfg = Gemma4Config::from_hf_json_file(&cfg_path)
            .with_context(|| format!("read config {}", cfg_path.display()))?;

        let load_device = if device.is_cuda() {
            Device::Cpu
        } else {
            device.clone()
        };
        let weights = open_base_weights(path, &load_device)?;
        let packed = is_packed_nvfp4(&weights) || is_packed_w4a16(&weights);

        if base_cfg.enable_moe_block {
            ensure!(
                !resident,
                "{NV_TRAIN_QUANT_RESIDENT}=1 on a MoE base: Gemma4Moe loads through a generic \
                 TensorSource with per-expert fused tensors and has no resident-Linear \
                 plumbing; only the dense gemma arm and the qwen38 windowed trainer carry the \
                 resident arm"
            );
            let cfg = Gemma4MoeConfig::from_hf_json_file(&cfg_path)
                .with_context(|| format!("read moe config {}", cfg_path.display()))?;
            let model = if packed {
                let src = Nvfp4DequantSource {
                    inner: &weights,
                    device: device.clone(),
                };
                Gemma4Moe::from_loader_dtype(cfg, &src, device, DType::F32)
                    .with_context(|| format!("load packed-nvfp4 moe base {}", path.display()))?
            } else {
                Gemma4Moe::from_loader_dtype(cfg, &weights, device, DType::F32)?
            };
            return Ok((BaseModel::Moe(model), DType::F32, false));
        }

        let base_dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        if resident {
            ensure!(
                is_packed_nvfp4(&weights),
                "{NV_TRAIN_QUANT_RESIDENT}=1 but {} holds no modelopt-style nvfp4-packed linear \
                 ({}); w4a16-packed and fp8 gemma bases have no dense-trainer resident path",
                path.display(),
                nv_train::qwen38::QUANT_RESIDENT_REFUSES_AN_ALL_BF16_CHECKPOINT_BECAUSE_A_SILENT_FALLTHROUGH_WOULD_FAKE_THE_ARM
            );
            let model = load_dense_resident(base_cfg, &weights, device, base_dtype, path)?;
            return Ok((BaseModel::Dense(model), base_dtype, true));
        }

        let plain_weights;
        let weights = if !packed && device.is_cuda() {
            plain_weights = open_base_weights(path, device)?;
            &plain_weights
        } else {
            &weights
        };
        let model = if packed {
            let src = Nvfp4DequantSource {
                inner: &weights,
                device: device.clone(),
            };
            DenseTrainModel::from_loader_dtype(base_cfg, &src, device, base_dtype)
                .with_context(|| format!("load packed-nvfp4 dense base {}", path.display()))?
        } else {
            DenseTrainModel::from_loader_dtype(base_cfg, weights, device, base_dtype)?
        };
        return Ok((BaseModel::Dense(model), base_dtype, false));
    }
    bail!(
        "--base {} is neither a .gguf file nor a directory with config.json + model.safetensors",
        path.display()
    )
}

pub const DENSE_RESIDENT_NVFP4_REQUANTIZES_THE_FUSED_MATRIX_BECAUSE_PER_MODULE_GLOBAL_SCALES_CANNOT_SHARE_ONE_WEIGHT_ALPHA: &str =
    "The dense gemma trainer fuses q/k/v and gate/up into one Linear per GEMM, but each \
     checkpoint module carries its own weight_scale_2 global scale and a resident nvfp4 \
     Linear holds exactly one weight_alpha, so exact packed-byte fusion is unrepresentable. \
     The resident arm therefore host-dequantizes each module (the default trainer's exact \
     bf16 values), fuses, and requantizes on device (from_bf16_quantized_nvfp4_dev) with a \
     fresh per-matrix global scale; the transient peak is one fused matrix of bf16, and the \
     resident storage is the serving W4A4 GEMM's.";

#[cfg(feature = "cuda")]
fn load_dense_resident(
    base_cfg: Gemma4Config,
    weights: &WeightLoader,
    device: &Device,
    base_dtype: DType,
    path: &Path,
) -> Result<DenseTrainModel> {
    let _ = DENSE_RESIDENT_NVFP4_REQUANTIZES_THE_FUSED_MATRIX_BECAUSE_PER_MODULE_GLOBAL_SCALES_CANNOT_SHARE_ONE_WEIGHT_ALPHA;
    let cuda = match device {
        Device::Cuda(d) => d.clone(),
        _ => bail!(
            "{NV_TRAIN_QUANT_RESIDENT}=1 needs a CUDA device: quantized-resident storages do \
             not exist on CPU"
        ),
    };
    let runner = Arc::new(std::sync::Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
        cuda.cuda_stream(),
    )?));
    let src = Nvfp4DequantSource {
        inner: weights,
        device: device.clone(),
    };
    let resident_count = std::cell::Cell::new(0usize);
    let module_is_nvfp4 = |m: &str| {
        weights.has(&format!("{m}.weight"))
            && weights.has(&format!("{m}.weight_scale"))
            && weights.has(&format!("{m}.weight_scale_2"))
    };
    let ingest = |names: &[String], w: Tensor| -> Result<Linear> {
        let quantized: Vec<bool> = names.iter().map(|n| module_is_nvfp4(n)).collect();
        if quantized.iter().all(|q| !q) {
            for n in names {
                ensure!(
                    !weights.has(&format!("{n}.weight_packed")),
                    "{n}: w4a16-packed storage has no resident-Linear path; unset \
                     {NV_TRAIN_QUANT_RESIDENT} for this base"
                );
            }
            return Linear::new_no_pretranspose(w, None);
        }
        ensure!(
            quantized.iter().all(|q| *q),
            "fused target {names:?} mixes nvfp4-packed and non-packed modules; a half-resident \
             fused GEMM would misreport the arm"
        );
        let lin = Linear::from_bf16_quantized_nvfp4_dev(&w, None, device, runner.clone())
            .with_context(|| format!("resident nvfp4 requant of fused {names:?}"))?;
        resident_count.set(resident_count.get() + names.len());
        Ok(lin)
    };
    let model =
        DenseTrainModel::from_loader_dtype_resident(base_cfg, &src, device, base_dtype, Some(&ingest))
            .with_context(|| format!("load quant-resident dense base {}", path.display()))?;
    ensure!(
        resident_count.get() > 0,
        "{NV_TRAIN_QUANT_RESIDENT}=1 built zero resident modules for {}; {}",
        path.display(),
        nv_train::qwen38::QUANT_RESIDENT_REFUSES_AN_ALL_BF16_CHECKPOINT_BECAUSE_A_SILENT_FALLTHROUGH_WOULD_FAKE_THE_ARM
    );
    eprintln!(
        "[train-runner] quant-resident dense base: {} checkpoint modules resident as nvfp4",
        resident_count.get()
    );
    Ok(model)
}

#[cfg(not(feature = "cuda"))]
fn load_dense_resident(
    _base_cfg: Gemma4Config,
    _weights: &WeightLoader,
    _device: &Device,
    _base_dtype: DType,
    _path: &Path,
) -> Result<DenseTrainModel> {
    bail!(
        "{NV_TRAIN_QUANT_RESIDENT}=1 but this build has no cuda feature; quantized-resident \
         storages only exist under cuda"
    )
}

struct FusedTarget<'a> {
    lin: &'a Linear,
    in_features: usize,

    components: Vec<(String, usize, usize, bool)>,

    seed: u64,
}

fn want(targets: &[String], comp: &str) -> bool {
    targets.iter().any(|t| t == comp)
}

fn build_targets<'a>(
    layers: &[(&'a Gemma4Attention, &'a Gemma4Mlp)],
    targets: &[String],
    seed0: u64,
) -> Result<Vec<FusedTarget<'a>>> {
    let mut out = Vec::new();
    let mut si = seed0;
    for (li, (attn, mlp)) in layers.iter().enumerate() {
        let p = format!("model.language_model.layers.{li}");
        let attn = *attn;
        let q_dim = attn.q_dim;
        let kv_dim = attn.kv_dim;
        let has_v = attn.has_v;

        if want(targets, "q") || want(targets, "k") || want(targets, "v") {
            let mut comps = vec![(
                format!("{p}.self_attn.q_proj"),
                0,
                q_dim,
                want(targets, "q"),
            )];
            if kv_dim > 0 {
                comps.push((
                    format!("{p}.self_attn.k_proj"),
                    q_dim,
                    kv_dim,
                    want(targets, "k"),
                ));
            }
            if has_v {
                comps.push((
                    format!("{p}.self_attn.v_proj"),
                    q_dim + kv_dim,
                    kv_dim,
                    want(targets, "v"),
                ));
            }
            out.push(FusedTarget {
                lin: &attn.qkv_proj,
                in_features: attn.qkv_proj.in_features(),
                components: comps,
                seed: si,
            });
            si += 1;
        }

        if want(targets, "o") {
            out.push(FusedTarget {
                lin: &attn.o_proj,
                in_features: attn.o_proj.in_features(),
                components: vec![(
                    format!("{p}.self_attn.o_proj"),
                    0,
                    attn.o_proj.out_features(),
                    true,
                )],
                seed: si,
            });
            si += 1;
        }

        if want(targets, "gate") || want(targets, "up") {
            let gu = &mlp.gate_up_proj;
            let out_f = gu.out_features();
            let inter = out_f / 2;
            out.push(FusedTarget {
                lin: gu,
                in_features: gu.in_features(),
                components: vec![
                    (
                        format!("{p}.mlp.gate_proj"),
                        0,
                        inter,
                        want(targets, "gate"),
                    ),
                    (
                        format!("{p}.mlp.up_proj"),
                        inter,
                        inter,
                        want(targets, "up"),
                    ),
                ],
                seed: si,
            });
            si += 1;
        }

        if want(targets, "down") {
            let dp = &mlp.down_proj;
            out.push(FusedTarget {
                lin: dp,
                in_features: dp.in_features(),
                components: vec![(format!("{p}.mlp.down_proj"), 0, dp.out_features(), true)],
                seed: si,
            });
            si += 1;
        }
    }
    if out.is_empty() {
        bail!("no trainable targets selected (--target was {:?})", targets);
    }
    Ok(out)
}

fn accumulate(total: Option<Tensor>, loss: Tensor) -> Result<Option<Tensor>> {
    Ok(Some(match total {
        Some(t) => (t + loss)?,
        None => loss,
    }))
}

fn solo_example_loss(model: &BaseModel, row: &TokenRow, dev: &Device) -> Result<Tensor> {
    let ids = &row.ids;
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    let logits = model.forward_logits(&tokens, &positions)?;
    let logits2 = logits.squeeze(0)?;
    let inp = logits2.narrow(0, 0, seq - 1)?.contiguous()?;
    let tgt = Tensor::from_vec(ids[1..].to_vec(), seq - 1, dev)?;
    ce_mean_over_unmasked_positions(
        &inp,
        &tgt,
        &row.target_mask_where_predicting_a_prompt_token_weighs_zero(),
    )
}

fn same_length_bucket_per_row_ce_means_over_unmasked_targets(
    model: &BaseModel,
    rows: &[&TokenRow],
    seq: usize,
    dev: &Device,
) -> Result<Vec<Tensor>> {
    let count = rows.len();
    let mut flat: Vec<u32> = Vec::with_capacity(count * seq);
    for row in rows {
        flat.extend_from_slice(&row.ids);
    }
    let tokens = Tensor::from_vec(flat, (count, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    let logits = model.forward_logits(&tokens, &positions)?;
    let mut out = Vec::with_capacity(count);
    for (bi, row) in rows.iter().enumerate() {
        let inp = logits
            .narrow(0, bi, 1)?
            .squeeze(0)?
            .narrow(0, 0, seq - 1)?
            .contiguous()?;
        let tgt = Tensor::from_vec(row.ids[1..].to_vec(), seq - 1, dev)?;
        out.push(ce_mean_over_unmasked_positions(
            &inp,
            &tgt,
            &row.target_mask_where_predicting_a_prompt_token_weighs_zero(),
        )?);
    }
    Ok(out)
}

fn per_row_ce_means_in_dataset_row_order(
    model: &BaseModel,
    batch: &[TokenRow],
) -> Result<Vec<Tensor>> {
    let dev = model.device();
    let solo = !model.is_dense()
        || nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_TRAIN_SOLO");
    if solo {
        return batch
            .iter()
            .map(|row| solo_example_loss(model, row, &dev))
            .collect();
    }
    let mut buckets: std::collections::BTreeMap<usize, Vec<(usize, &TokenRow)>> =
        std::collections::BTreeMap::new();
    for (i, row) in batch.iter().enumerate() {
        buckets.entry(row.ids.len()).or_default().push((i, row));
    }
    let mut out: Vec<Option<Tensor>> = vec![None; batch.len()];
    for (seq, indexed) in &buckets {
        let rows: Vec<&TokenRow> = indexed.iter().map(|(_, r)| *r).collect();
        let per_row =
            same_length_bucket_per_row_ce_means_over_unmasked_targets(model, &rows, *seq, &dev)?;
        for ((i, _), loss) in indexed.iter().zip(per_row) {
            out[*i] = Some(loss);
        }
    }
    out.into_iter()
        .map(|t| t.context("a bucketed row lost its dataset index"))
        .collect()
}

fn batch_loss(model: &BaseModel, batch: &[TokenRow]) -> Result<Tensor> {
    let mut total: Option<Tensor> = None;
    for loss in per_row_ce_means_in_dataset_row_order(model, batch)? {
        total = accumulate(total, loss)?;
    }
    total.context("empty batch")
}

fn per_row_eval_ce_in_dataset_row_order(
    model: &BaseModel,
    batch: &[TokenRow],
) -> Result<Vec<f32>> {
    per_row_ce_means_in_dataset_row_order(model, batch)?
        .into_iter()
        .map(|t| Ok(t.to_scalar::<f32>()?))
        .collect()
}

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn serving_equivalence(
    targets: &[FusedTarget],
    loras: &[Arc<LoraTrainable>],
    device: &Device,
) -> Result<f32> {
    let mut worst = 0f32;
    for (t, lora) in targets.iter().zip(loras.iter()) {
        let mut rng = Lcg(0xC0FFEE ^ t.seed);
        let n = 4usize;
        let data: Vec<f32> = (0..n * t.in_features).map(|_| rng.f32()).collect();
        let x = Tensor::from_vec(data, (n, t.in_features), device)?;
        let full = lora.delta_forward(&x, false)?;
        let mut pieces = Vec::new();
        for (_name, off, rows, _emit) in &t.components {
            let b_slice = lora.b_tensor().narrow(0, *off, *rows)?;
            let piece = lora_delta(lora.a_tensor(), &b_slice, lora.scaling(), &x, None)?;
            pieces.push(piece);
        }
        let cat = Tensor::cat(&pieces, 1)?;
        worst = worst.max(max_abs_diff(&full, &cat)?);
    }
    Ok(worst)
}

pub fn run(args: &TrainArgs) -> Result<TrainSummary> {
    Ok(run_with_speculator(args, None)?.0)
}

pub fn run_with_speculator(
    args: &TrainArgs,
    spec: Option<&SpeculatorSpec>,
) -> Result<(TrainSummary, Option<SpeculatorTrainSummary>)> {
    let device = select_device()?;
    let device_label = if device.is_cuda() { "cuda" } else { "cpu" };
    let (model, base_dtype, resident_base) = load_base_resident_aware(&args.base, &device)?;
    let mut cotrain = match spec {
        None => None,
        Some(s) => Some(Eagle3Cotrain::build(args, s, &model, base_dtype, &device)?),
    };
    let vocab = model.vocab();
    if model.is_dense() {
        let (built, full) = model.layer_counts();
        eprintln!(
            "DENSE_BASE 1  layers_built {built}  config_layers {full}  hidden {}  vocab {vocab}  \
             device {device_label}  base_dtype {base_dtype:?}",
            model.hidden()
        );
    }

    let attn_mlp = model.attn_mlp_layers();
    let targets = build_targets(&attn_mlp, &args.targets, args.seed)?;

    let cfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
    };
    let mut loras: Vec<Arc<LoraTrainable>> = Vec::new();
    let mut vars: Vec<Var> = Vec::new();
    for t in &targets {
        let lora = Arc::new(LoraTrainable::from_features(
            t.lin.out_features(),
            t.in_features,
            cfg,
            t.seed,
            &device,
        )?);
        vars.extend(lora.trainable_vars());
        let hook: Arc<dyn LoraDeltaHook> = Arc::new(TrainingLoraHook::new(lora.clone(), true));
        t.lin.attach_lora(hook)?;
        loras.push(lora);
    }

    let dataset = load_dataset(&args.data, &args.base, vocab, JSONL_MAX_SEQ)?;

    let checkpointed =
        checkpointing_tristate_defaults_on_for_cuda_dense_because_the_full_graph_oomed_on_real_e4b(
            &device,
            model.is_dense(),
        )?;
    let lmhead_chunk = crate::dense_train::lmhead_chunk_from_env();

    let mut losses = Vec::with_capacity(args.steps);
    let mut all_grads_finite = true;
    if args.steps > 0 {
        let mut opt = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: args.lr,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        for step in 0..args.steps {
            let lv = if checkpointed {
                let dense = match &model {
                    BaseModel::Dense(d) => d,
                    _ => unreachable!("checkpointed guarded to dense"),
                };
                let sg = dense.train_step_checkpointed(&dataset, &vars, lmhead_chunk)?;
                for g in sg.grads.iter().flatten() {
                    if !grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(g)?
                    {
                        all_grads_finite = false;
                    }
                }
                let gs = grad_store_from(&vars, &sg.grads)?;
                opt.step(&gs)?;
                sg.loss
            } else {
                let loss = batch_loss(&model, &dataset)?;
                let lv = loss.to_scalar::<f32>()?;
                let grads = loss.backward()?;
                for v in &vars {
                    if let Some(g) = grads.get(v) {
                        if !grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(
                            g,
                        )? {
                            all_grads_finite = false;
                        }
                    }
                }
                opt.step(&grads)?;
                lv
            };
            losses.push(lv);
            if step < 3 || step % 10 == 0 || step == args.steps - 1 {
                eprintln!("step {step:4}  loss = {lv:.6e}");
            }
            if let Some(c) = cotrain.as_mut() {
                let dense = match &model {
                    BaseModel::Dense(d) => d,
                    _ => unreachable!("cotrain build refused every non-dense base"),
                };
                c.step(dense, &dataset)?;
            }
        }
    }

    let (eval_ce_adapter, eval_ce_base) = match &args.eval_data {
        Some(path) => {
            let eval_set = load_dataset(path, &args.base, vocab, JSONL_MAX_SEQ)?;
            let with_adapter = per_row_eval_ce_in_dataset_row_order(&model, &eval_set)?;
            for t in &targets {
                t.lin.detach_lora();
            }
            let base_only = per_row_eval_ce_in_dataset_row_order(&model, &eval_set)?;
            for (t, lora) in targets.iter().zip(loras.iter()) {
                let hook: Arc<dyn LoraDeltaHook> =
                    Arc::new(TrainingLoraHook::new(lora.clone(), true));
                t.lin.attach_lora(hook)?;
            }
            eprintln!(
                "eval  loss_adapter = {:.6e}  loss_base = {:.6e}  examples {}",
                with_adapter.iter().sum::<f32>(),
                base_only.iter().sum::<f32>(),
                eval_set.len()
            );
            (with_adapter, base_only)
        }
        None => (Vec::new(), Vec::new()),
    };
    let eval_loss_adapter =
        eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(&eval_ce_adapter);
    let eval_loss_base =
        eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(&eval_ce_base);

    let serving_equiv_maxabs = serving_equivalence(&targets, &loras, &device)?;

    let mut entries: Vec<RawLora> = Vec::new();
    for (t, lora) in targets.iter().zip(loras.iter()) {
        for (name, off, rows, emit) in &t.components {
            if !*emit {
                continue;
            }
            let a = lora.a_tensor().clone();
            let b = lora.b_tensor().narrow(0, *off, *rows)?;
            entries.push(RawLora {
                module: name.clone(),
                a,
                b,
            });
        }
    }
    let base_name = args.base.to_string_lossy().to_string();
    save_peft_raw(&args.out, &entries, args.rank, args.alpha, 0.0, &base_name)?;

    let speculator = match (cotrain, spec) {
        (Some(c), Some(s)) => Some(c.save(&args.out, s)?),
        _ => None,
    };

    let (layers_built, config_layers) = model.layer_counts();
    let dense_base = model.is_dense();
    let modules: Vec<String> = entries.iter().map(|e| e.module.clone()).collect();
    let num_eval_examples = eval_ce_base.len();
    let summary = TrainSummary {
        losses,
        eval_loss_adapter,
        eval_loss_base,
        eval_ce_adapter,
        eval_ce_base,
        serving_equiv_maxabs,
        trainable_vars: vars.len(),
        num_examples: dataset.len(),
        num_eval_examples,
        base_dtype: if resident_base {
            format!("{base_dtype:?}+quant_resident")
        } else {
            format!("{base_dtype:?}")
        },
        resident_base,
        nvfp4_base: base_is_packed_nvfp4(&args.base),
        dense_base,
        layers_built,
        config_layers,
        deterministic: nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_DETERMINISTIC"),
        device: device_label.to_string(),
        checkpointed,
        all_grads_finite,
        lmhead_chunk,
        modules,
        adapter_path: args.out.join("adapter_model.safetensors"),
        config_path: args.out.join("adapter_config.json"),
    };
    Ok((summary, speculator))
}
