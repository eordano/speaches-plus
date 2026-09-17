use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::{log_softmax, silu, softmax_last_dim};
use nv_layers::linear::{Linear, LoraDeltaHook};
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::{load_linear, load_rmsnorm, load_tensor};
use nv_train::{LoraConfig, LoraTrainable, RawLora, TrainingLoraHook};
use nv_weights::WeightLoader;
use std::path::Path;
use std::sync::Arc;

use crate::dense_train::naive_sdpa;

pub const EAGLE3_HEAD_LORA_EXCLUDES_EMBED_AND_LM_HEAD_BECAUSE_THE_UPSTREAM_DRAFTER_RECIPE_FREEZES_BOTH:
    [&str; 8] = [
    "fc",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
];

#[derive(Clone, Debug)]
pub struct Eagle3TrainConfig {
    pub hidden_size: usize,
    pub draft_vocab_size: usize,
    pub target_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub norm_before_residual: bool,
    pub norm_before_fc: bool,
    pub aux_layer_ids: Vec<usize>,
}

impl Eagle3TrainConfig {
    pub fn fc_in_dim(&self) -> usize {
        self.aux_layer_ids.len() * self.hidden_size
    }

    pub fn block_in_dim(&self) -> usize {
        2 * self.hidden_size
    }

    pub fn q_out_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_out_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn from_drafter_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        if !cfg_path.is_file() {
            bail!(
                "speculator co-training expected an eagle3 drafter checkpoint at {} \
                 (config.json with speculators_model_type \"eagle3\" plus model.safetensors), \
                 but config.json is missing",
                dir.display()
            );
        }
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", cfg_path.display()))?;
        let kind = v
            .get("speculators_model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("<absent>");
        if kind != "eagle3" {
            bail!(
                "speculator co-training expected speculators_model_type \"eagle3\" in {}, got \
                 \"{kind}\"",
                cfg_path.display()
            );
        }
        let tlc = v.get("transformer_layer_config").with_context(|| {
            format!(
                "eagle3 drafter config {} has no transformer_layer_config",
                cfg_path.display()
            )
        })?;
        let usize_field = |key: &str| -> Result<usize> {
            tlc.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| {
                    format!(
                        "eagle3 drafter config {}: transformer_layer_config.{key} missing",
                        cfg_path.display()
                    )
                })
        };
        let mut aux_layer_ids: Vec<usize> = Vec::new();
        if let Some(arr) = v
            .get("eagle_aux_hidden_state_layer_ids")
            .and_then(|x| x.as_array())
        {
            for (i, x) in arr.iter().enumerate() {
                let id = x.as_u64().with_context(|| {
                    format!(
                        "eagle3 drafter config {}: eagle_aux_hidden_state_layer_ids[{i}] is {x}, \
                         not a non-negative integer; tap ids must be absolute layer indices \
                         (negative python-style indices are not accepted), and silently dropping \
                         the entry would shrink fc_in_dim and resurface as an unexplained \
                         fc.weight shape mismatch",
                        cfg_path.display()
                    )
                })?;
                aux_layer_ids.push(id as usize);
            }
        }
        if aux_layer_ids.is_empty() {
            bail!(
                "eagle3 drafter config {} has no eagle_aux_hidden_state_layer_ids; without the \
                 tap list the fc projector input is undefined",
                cfg_path.display()
            );
        }
        let draft_vocab_size = v
            .get("draft_vocab_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .with_context(|| {
                format!(
                    "eagle3 drafter config {} has no draft_vocab_size",
                    cfg_path.display()
                )
            })?;
        Ok(Self {
            hidden_size: usize_field("hidden_size")?,
            draft_vocab_size,
            target_vocab_size: usize_field("vocab_size")?,
            num_attention_heads: usize_field("num_attention_heads")?,
            num_key_value_heads: usize_field("num_key_value_heads")?,
            head_dim: usize_field("head_dim")?,
            intermediate_size: usize_field("intermediate_size")?,
            max_position_embeddings: usize_field("max_position_embeddings")?,
            rms_norm_eps: tlc
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6),
            rope_theta: tlc
                .get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .or_else(|| tlc.get("rope_theta").and_then(|x| x.as_f64()))
                .map(|t| t as f32)
                .unwrap_or(10000.0),
            norm_before_residual: v
                .get("norm_before_residual")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            norm_before_fc: v
                .get("norm_before_fc")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            aux_layer_ids,
        })
    }
}

pub struct Eagle3TrainHead {
    cfg: Eagle3TrainConfig,
    device: Device,
    embed_tokens: Tensor,
    fc: Linear,
    input_layernorm: RmsNorm,
    hidden_norm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    d2t_target_ids: Tensor,
}

impl Eagle3TrainHead {
    pub fn config(&self) -> &Eagle3TrainConfig {
        &self.cfg
    }

    pub fn load(
        cfg: &Eagle3TrainConfig,
        dir: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if cfg.norm_before_fc {
            bail!(
                "eagle3 drafter at {}: norm_before_fc=true is not supported (the RedHatAI \
                 checkpoint sets it false and ships no pre-fc norm weight)",
                dir.display()
            );
        }
        let st_path = dir.join("model.safetensors");
        if !st_path.is_file() {
            bail!(
                "speculator co-training expected an eagle3 drafter checkpoint at {} with \
                 model.safetensors, but the file is missing",
                dir.display()
            );
        }
        let weights = WeightLoader::open_file(&st_path, device)
            .with_context(|| format!("open {}", st_path.display()))?;

        let embed_tokens = load_tensor(
            &weights,
            "embed_tokens.weight",
            &[cfg.target_vocab_size, cfg.hidden_size],
            dtype,
        )?;
        let fc = load_linear(
            &weights,
            "fc.weight",
            cfg.hidden_size,
            cfg.fc_in_dim(),
            dtype,
        )?;
        let input_layernorm = load_rmsnorm(
            &weights,
            "layers.0.input_layernorm.weight",
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype,
        )?;
        let hidden_norm = load_rmsnorm(
            &weights,
            "layers.0.hidden_norm.weight",
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = load_rmsnorm(
            &weights,
            "layers.0.post_attention_layernorm.weight",
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype,
        )?;
        let q_proj = load_linear(
            &weights,
            "layers.0.self_attn.q_proj.weight",
            cfg.q_out_dim(),
            cfg.block_in_dim(),
            dtype,
        )?;
        let k_proj = load_linear(
            &weights,
            "layers.0.self_attn.k_proj.weight",
            cfg.kv_out_dim(),
            cfg.block_in_dim(),
            dtype,
        )?;
        let v_proj = load_linear(
            &weights,
            "layers.0.self_attn.v_proj.weight",
            cfg.kv_out_dim(),
            cfg.block_in_dim(),
            dtype,
        )?;
        let o_proj = load_linear(
            &weights,
            "layers.0.self_attn.o_proj.weight",
            cfg.hidden_size,
            cfg.q_out_dim(),
            dtype,
        )?;
        let gate_proj = load_linear(
            &weights,
            "layers.0.mlp.gate_proj.weight",
            cfg.intermediate_size,
            cfg.hidden_size,
            dtype,
        )?;
        let up_proj = load_linear(
            &weights,
            "layers.0.mlp.up_proj.weight",
            cfg.intermediate_size,
            cfg.hidden_size,
            dtype,
        )?;
        let down_proj = load_linear(
            &weights,
            "layers.0.mlp.down_proj.weight",
            cfg.hidden_size,
            cfg.intermediate_size,
            dtype,
        )?;
        let norm = load_rmsnorm(
            &weights,
            "norm.weight",
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype,
        )?;
        let lm_head = load_linear(
            &weights,
            "lm_head.weight",
            cfg.draft_vocab_size,
            cfg.hidden_size,
            dtype,
        )?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: cfg.head_dim,
                max_seq_len: cfg.max_position_embeddings,
                base: cfg.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;

        let d2t_raw = load_tensor(&weights, "d2t", &[cfg.draft_vocab_size], DType::I64)?;
        let d2t_offsets: Vec<i64> = d2t_raw.to_vec1()?;
        let mut mapped: Vec<u32> = Vec::with_capacity(cfg.draft_vocab_size);
        for (j, off) in d2t_offsets.iter().enumerate() {
            let target = j as i64 + off;
            if *off < 0 || target < 0 || target >= cfg.target_vocab_size as i64 {
                bail!(
                    "eagle3 drafter d2t[{j}] = {off} maps draft token {j} to target id {target}, \
                     out of range for target_vocab_size {}",
                    cfg.target_vocab_size
                );
            }
            mapped.push(target as u32);
        }
        let d2t_target_ids = Tensor::from_vec(mapped, cfg.draft_vocab_size, device)?;

        Ok(Self {
            cfg: cfg.clone(),
            device: device.clone(),
            embed_tokens,
            fc,
            input_layernorm,
            hidden_norm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            norm,
            lm_head,
            rope,
            d2t_target_ids,
        })
    }

    fn lora_target_linears(&self) -> [(&'static str, &Linear); 8] {
        [
            ("fc", &self.fc),
            ("layers.0.self_attn.q_proj", &self.q_proj),
            ("layers.0.self_attn.k_proj", &self.k_proj),
            ("layers.0.self_attn.v_proj", &self.v_proj),
            ("layers.0.self_attn.o_proj", &self.o_proj),
            ("layers.0.mlp.gate_proj", &self.gate_proj),
            ("layers.0.mlp.up_proj", &self.up_proj),
            ("layers.0.mlp.down_proj", &self.down_proj),
        ]
    }

    pub fn attach_head_loras(
        &self,
        rank: usize,
        seed0: u64,
    ) -> Result<Vec<(String, Arc<LoraTrainable>)>> {
        let mut out = Vec::new();
        for (i, (name, lin)) in self.lora_target_linears().into_iter().enumerate() {
            let base = lin
                .weight()
                .with_context(|| format!("eagle3 drafter {name} has no materialised weight"))?
                .clone();
            let lora = Arc::new(LoraTrainable::new(
                &base,
                LoraConfig::unit_scaling(rank),
                seed0.wrapping_add(i as u64),
                &self.device,
            )?);
            let hook: Arc<dyn LoraDeltaHook> = Arc::new(TrainingLoraHook::new(lora.clone(), true));
            lin.attach_lora(hook)?;
            out.push((
                EAGLE3_HEAD_LORA_EXCLUDES_EMBED_AND_LM_HEAD_BECAUSE_THE_UPSTREAM_DRAFTER_RECIPE_FREEZES_BOTH
                    [i]
                    .to_string(),
                lora,
            ));
        }
        Ok(out)
    }

    pub fn raw_entries(loras: &[(String, Arc<LoraTrainable>)]) -> Vec<RawLora> {
        loras
            .iter()
            .map(|(name, lora)| RawLora {
                module: name.clone(),
                a: lora.a_tensor().clone(),
                b: lora.b_tensor().clone(),
            })
            .collect()
    }

    pub fn forward_train_shift_pairs_embed_of_token_i_plus_1_with_aux_row_i_like_serving(
        &self,
        ids: &[u32],
        aux_cat_detached: &Tensor,
    ) -> Result<Tensor> {
        let full = ids.len();
        if full < 2 {
            bail!(
                "eagle3 shift forward: row has {full} tokens but shift conditioning embeds token \
                 i+1 against aux row i (the serving default in resolve_cond_mode), so a row needs \
                 at least 2 tokens"
            );
        }
        if aux_cat_detached.dims() != [full, self.cfg.fc_in_dim()] {
            bail!(
                "eagle3 shift forward: aux rows must be [{full}, {}] (taps {:?} concatenated in \
                 config order, one row per context token), got {:?}",
                self.cfg.fc_in_dim(),
                self.cfg.aux_layer_ids,
                aux_cat_detached.dims()
            );
        }
        let seq = full - 1;
        let wdt = self.fc.weight().context("fc weight")?.dtype();
        let ids_t = Tensor::from_vec(ids[1..].to_vec(), seq, &self.device)?;
        let token_embed = self.embed_tokens.detach().index_select(&ids_t, 0)?;
        let aux = aux_cat_detached.detach().narrow(0, 0, seq)?.to_dtype(wdt)?;
        let aux_proj = self.fc.forward(&aux)?;

        let token_normed = self.input_layernorm.forward_candle(&token_embed)?;
        let aux_normed = self.hidden_norm.forward_candle(&aux_proj)?;
        let residual = if self.cfg.norm_before_residual {
            aux_normed.clone()
        } else {
            aux_proj.clone()
        };
        let block_in = Tensor::cat(&[&token_normed, &aux_normed], 1)?;

        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let q = self
            .q_proj
            .forward(&block_in)?
            .reshape((1usize, seq, nh, hd))?;
        let k = self
            .k_proj
            .forward(&block_in)?
            .reshape((1usize, seq, nkv, hd))?;
        let v = self
            .v_proj
            .forward(&block_in)?
            .reshape((1usize, seq, nkv, hd))?;
        let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, &self.device)?;
        let (q_rot, k_rot) = self.rope.apply_candle(&q, &k, &positions)?;
        let attn = naive_sdpa(
            &q_rot.contiguous()?,
            &k_rot.contiguous()?,
            &v.contiguous()?,
            1,
            nh,
            nkv,
            hd,
            seq,
            None,
        )?;
        let attn_flat = attn.to_dtype(wdt)?.reshape((seq, nh * hd))?;
        let attn_out = self.o_proj.forward(&attn_flat)?;
        let post_attn = residual.add(&attn_out)?;
        let mlp_in = self.post_attention_layernorm.forward_candle(&post_attn)?;
        let gate = self.gate_proj.forward(&mlp_in)?;
        let up = self.up_proj.forward(&mlp_in)?;
        let act = silu(&gate)?.mul(&up)?;
        let mlp_out = self.down_proj.forward(&act)?;
        let block_out = post_attn.add(&mlp_out)?;

        let normed = self.norm.forward_candle(&block_out)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        debug_assert_eq!(logits.dims(), &[seq, self.cfg.draft_vocab_size]);
        Ok(logits)
    }

    pub fn shift_distill_loss_pairs_draft_row_i_with_teacher_row_i_plus_1(
        &self,
        draft_shift_logits: &Tensor,
        teacher_logits_all_rows: &Tensor,
        completion_start: usize,
    ) -> Result<Tensor> {
        let tdims = teacher_logits_all_rows.dims();
        if tdims.len() != 2 || tdims[1] != self.cfg.target_vocab_size {
            bail!(
                "shift distill loss: teacher logits must be [seq, {}], got {:?}",
                self.cfg.target_vocab_size,
                tdims
            );
        }
        let full = tdims[0];
        if full < 2 {
            bail!("shift distill loss: teacher has {full} rows, shift pairing needs at least 2");
        }
        let rows = full - 1;
        if draft_shift_logits.dims() != [rows, self.cfg.draft_vocab_size] {
            bail!(
                "shift distill loss: draft logits must be [{rows}, {}] (draft row i embeds token \
                 i+1 and matches teacher row i+1, so a {full}-token row has {rows} shift rows), \
                 got {:?}",
                self.cfg.draft_vocab_size,
                draft_shift_logits.dims()
            );
        }
        let t_pred = teacher_logits_all_rows.narrow(0, 1, rows)?;
        let draft_row_i_predicts_absolute_position_i_plus_2_and_the_final_row_is_the_bonus_draft =
            |i: usize| -> f32 {
                if i + 2 >= completion_start {
                    1.0
                } else {
                    0.0
                }
            };
        let mask: Vec<f32> = (0..rows)
            .map(draft_row_i_predicts_absolute_position_i_plus_2_and_the_final_row_is_the_bonus_draft)
            .collect();
        self.soft_ce_to_the_d2t_gathered_teacher_which_is_forward_kl_minus_constant_teacher_entropy(
            draft_shift_logits,
            &t_pred,
            &mask,
        )
    }

    fn soft_ce_to_the_d2t_gathered_teacher_which_is_forward_kl_minus_constant_teacher_entropy(
        &self,
        draft_logits_pred: &Tensor,
        teacher_logits_pred: &Tensor,
        mask: &[f32],
    ) -> Result<Tensor> {
        let rows = draft_logits_pred.dims();
        if rows.len() != 2 || rows[1] != self.cfg.draft_vocab_size {
            bail!(
                "distill loss: draft logits must be [t, {}], got {:?}",
                self.cfg.draft_vocab_size,
                rows
            );
        }
        let t = rows[0];
        let tdims = teacher_logits_pred.dims();
        if tdims != [t, self.cfg.target_vocab_size] {
            bail!(
                "distill loss: teacher logits must be [{t}, {}], got {:?}",
                self.cfg.target_vocab_size,
                tdims
            );
        }
        if mask.len() != t {
            bail!("distill loss: mask has {} weights for {t} rows", mask.len());
        }
        let denom: f32 = mask.iter().sum();
        if denom <= 0.0 {
            bail!("distill loss: every drafted position is masked, the row has no loss");
        }
        let gathered = teacher_logits_pred
            .detach()
            .to_dtype(DType::F32)?
            .index_select(&self.d2t_target_ids, 1)?;
        let p_teacher = softmax_last_dim(&gathered)?;
        let logq = log_softmax(draft_logits_pred, D::Minus1)?;
        let per_row = p_teacher.mul(&logq)?.sum(D::Minus1)?;
        let m = Tensor::from_vec(mask.to_vec(), t, &self.device)?;
        Ok(per_row
            .mul(&m)?
            .sum_all()?
            .affine(-1.0 / denom as f64, 0.0)?)
    }
}
