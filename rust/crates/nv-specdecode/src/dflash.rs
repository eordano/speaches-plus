use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
use std::path::Path;

use crate::eagle3_loader::{load_d2t, load_t2d, validate_d2t, validate_t2d};
use crate::util::{load_linear, load_rmsnorm, load_tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFlashLayerAttn {
    pub sliding: bool,
    pub causal: bool,
}

impl DFlashLayerAttn {
    pub const FULL_NON_CAUSAL: Self = Self {
        sliding: false,
        causal: false,
    };
}

#[derive(Clone, Debug)]
pub struct DFlashSpeculatorConfig {
    pub hidden_size: usize,
    pub target_hidden_size: usize,
    pub draft_vocab_size: usize,
    pub target_vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub aux_hidden_state_layer_ids: Vec<usize>,
    pub layer_attn: Vec<DFlashLayerAttn>,
    pub sliding_window: usize,
    pub logit_softcap: Option<f64>,
    pub tied_embeddings: bool,
}

impl Default for DFlashSpeculatorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 5376,
            target_hidden_size: 5376,
            draft_vocab_size: 32000,
            target_vocab_size: 262144,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 256,
            intermediate_size: 21504,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            block_size: 8,
            mask_token_id: 4,
            aux_hidden_state_layer_ids: vec![1, 17, 29, 47, 58],
            layer_attn: Vec::new(),
            sliding_window: 0,
            logit_softcap: None,
            tied_embeddings: false,
        }
    }
}

impl DFlashSpeculatorConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse dflash config json")?;
        if v.get("transformer_layer_config").is_some() {
            Self::from_speculators_json(&v)
        } else {
            Self::from_flat_json(&v)
        }
    }

    fn from_speculators_json(v: &serde_json::Value) -> Result<Self> {
        let c = crate::eagle3_loader::parse_speculators_tlc(v)?;
        let mut cfg = Self {
            hidden_size: c.hidden_size,
            num_hidden_layers: c
                .num_hidden_layers
                .ok_or_else(|| anyhow!("missing num_hidden_layers"))?,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            intermediate_size: c.intermediate_size,
            max_position_embeddings: c.max_position_embeddings,
            target_vocab_size: c.target_vocab_size,
            ..Self::default()
        };
        if let Some(eps) = c.rms_norm_eps {
            cfg.rms_norm_eps = eps;
        }
        if c.has_rope_parameters {
            if let Some(theta) = c.rope_theta_params {
                cfg.rope_theta = theta;
            }
        } else if let Some(theta) = c.rope_theta_flat {
            cfg.rope_theta = theta;
        }
        if let Some(d) = c.draft_vocab_size {
            cfg.draft_vocab_size = d;
        }
        if let Some(b) = v.get("block_size").and_then(|x| x.as_u64()) {
            cfg.block_size = b as usize;
        }
        cfg.mask_token_id = v
            .get("mask_token_id")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32)
            .ok_or_else(|| anyhow!("missing mask_token_id"))?;
        cfg.target_hidden_size = v
            .get("target_hidden_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(cfg.hidden_size);
        let arr = v
            .get("aux_hidden_state_layer_ids")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("missing aux_hidden_state_layer_ids"))?;
        cfg.aux_hidden_state_layer_ids = arr
            .iter()
            .filter_map(|x| x.as_u64().map(|v| v as usize))
            .collect();
        cfg.validated()
    }

    fn from_flat_json(v: &serde_json::Value) -> Result<Self> {
        let usize_field = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("missing {key}"))
        };
        let mut cfg = Self::default();
        cfg.hidden_size = usize_field("hidden_size")?;
        cfg.num_hidden_layers = usize_field("num_hidden_layers")?;
        cfg.num_attention_heads = usize_field("num_attention_heads")?;
        cfg.num_key_value_heads = usize_field("num_key_value_heads")?;
        cfg.head_dim = usize_field("head_dim")?;
        cfg.intermediate_size = usize_field("intermediate_size")?;
        cfg.max_position_embeddings = usize_field("max_position_embeddings")?;
        cfg.target_vocab_size = usize_field("vocab_size")?;
        cfg.draft_vocab_size = v
            .get("draft_vocab_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(cfg.target_vocab_size);
        cfg.target_hidden_size = v
            .get("target_hidden_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(cfg.hidden_size);
        if let Some(eps) = v.get("rms_norm_eps").and_then(|x| x.as_f64()) {
            cfg.rms_norm_eps = eps;
        }
        if let Some(theta) = v.get("rope_theta").and_then(|x| x.as_f64()) {
            cfg.rope_theta = theta as f32;
        } else if let Some(theta) = v
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|x| x.as_f64())
        {
            cfg.rope_theta = theta as f32;
        }
        if let Some(b) = v.get("block_size").and_then(|x| x.as_u64()) {
            cfg.block_size = b as usize;
        }
        let dfc = v.get("dflash_config");
        let dfc_get = |key: &str| dfc.and_then(|d| d.get(key));
        cfg.mask_token_id = dfc_get("mask_token_id")
            .or_else(|| v.get("mask_token_id"))
            .and_then(|x| x.as_u64())
            .map(|x| x as u32)
            .ok_or_else(|| anyhow!("missing mask_token_id"))?;
        let aux = v
            .get("aux_hidden_state_layer_ids")
            .or_else(|| dfc_get("target_layer_ids"))
            .and_then(|x| x.as_array())
            .ok_or_else(|| {
                anyhow!("missing aux_hidden_state_layer_ids / dflash_config.target_layer_ids")
            })?;
        cfg.aux_hidden_state_layer_ids = aux
            .iter()
            .filter_map(|x| x.as_u64().map(|v| v as usize))
            .collect();
        cfg.logit_softcap = v
            .get("final_logit_softcapping")
            .and_then(|x| x.as_f64())
            .filter(|&c| c > 0.0);
        cfg.tied_embeddings = v
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);

        let causal_override = dfc_get("causal").and_then(|x| x.as_bool());
        let use_swa = dfc_get("use_swa")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let layer_types: Option<Vec<String>> =
            v.get("layer_types").and_then(|x| x.as_array()).map(|a| {
                a.iter()
                    .map(|t| t.as_str().unwrap_or_default().to_string())
                    .collect()
            });
        if let Some(lt) = &layer_types {
            if lt.len() != cfg.num_hidden_layers {
                bail!(
                    "layer_types has {} entries for {} layers",
                    lt.len(),
                    cfg.num_hidden_layers
                );
            }
        }
        let any_sliding = layer_types
            .as_ref()
            .map(|lt| lt.iter().any(|t| t == "sliding_attention"))
            .unwrap_or(false);
        cfg.layer_attn = (0..cfg.num_hidden_layers)
            .map(|i| {
                let is_sliding = match layer_types.as_ref() {
                    Some(lt) if !(use_swa && !any_sliding) => lt[i] == "sliding_attention",
                    _ => use_swa,
                };
                let type_causal = layer_types
                    .as_ref()
                    .map(|lt| lt[i] == "sliding_attention")
                    .unwrap_or(false);
                DFlashLayerAttn {
                    sliding: is_sliding,
                    causal: causal_override.unwrap_or(type_causal),
                }
            })
            .collect();
        if cfg.layer_attn.iter().any(|a| a.sliding) {
            cfg.sliding_window = dfc_get("swa_window_size")
                .or_else(|| v.get("sliding_window"))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("sliding layers need swa_window_size or sliding_window"))?;
            if cfg.sliding_window == 0 {
                bail!("sliding_window must be >= 1");
            }
        }
        cfg.validated()
    }

    const MAX_BLOCK_SIZE: usize = 512;
    const MAX_HIDDEN_DIM: usize = 65536;
    const MAX_INTERMEDIATE: usize = 1 << 20;
    const MAX_HEAD_DIM: usize = 1024;
    const MAX_HEADS: usize = 1024;
    const MAX_LAYERS: usize = 512;
    const MAX_VOCAB: usize = 4 << 20;

    fn validated(self) -> Result<Self> {
        if self.aux_hidden_state_layer_ids.is_empty() {
            bail!("aux_hidden_state_layer_ids is empty");
        }
        let bounded = |name: &str, v: usize, max: usize| -> Result<()> {
            if v == 0 || v > max {
                bail!("dflash config: {name} {v} is outside sanity bounds 1..={max}");
            }
            Ok(())
        };
        bounded("block_size", self.block_size, Self::MAX_BLOCK_SIZE)?;
        bounded("hidden_size", self.hidden_size, Self::MAX_HIDDEN_DIM)?;
        bounded(
            "target_hidden_size",
            self.target_hidden_size,
            Self::MAX_HIDDEN_DIM,
        )?;
        bounded(
            "intermediate_size",
            self.intermediate_size,
            Self::MAX_INTERMEDIATE,
        )?;
        bounded("head_dim", self.head_dim, Self::MAX_HEAD_DIM)?;
        bounded(
            "num_attention_heads",
            self.num_attention_heads,
            Self::MAX_HEADS,
        )?;
        bounded(
            "num_key_value_heads",
            self.num_key_value_heads,
            Self::MAX_HEADS,
        )?;
        bounded(
            "num_hidden_layers",
            self.num_hidden_layers,
            Self::MAX_LAYERS,
        )?;
        bounded("draft_vocab_size", self.draft_vocab_size, Self::MAX_VOCAB)?;
        bounded("target_vocab_size", self.target_vocab_size, Self::MAX_VOCAB)?;
        Ok(self)
    }

    pub fn layer_attn_for(&self, idx: usize) -> DFlashLayerAttn {
        self.layer_attn
            .get(idx)
            .copied()
            .unwrap_or(DFlashLayerAttn::FULL_NON_CAUSAL)
    }

    pub fn any_masked_layer(&self) -> bool {
        self.layer_attn.iter().any(|a| a.causal || a.sliding)
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn fc_in_dim(&self) -> usize {
        self.target_hidden_size * self.aux_hidden_state_layer_ids.len()
    }

    pub fn q_out_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_out_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn query_rows(&self) -> usize {
        1 + self.block_size
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const FP4_STAGING_ROWS: usize = 128;

#[cfg(feature = "cuda")]
const _: () = assert!(FP4_STAGING_ROWS == nv_quant::nvfp4::MIN_TILE);

pub(crate) fn draft_f32_enabled(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| v != "0")
}

pub struct DFlashLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DFlashLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    ) -> Self {
        Self {
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    fn load(
        weights: &WeightLoader,
        idx: usize,
        cfg: &DFlashSpeculatorConfig,
        dtype: DType,
    ) -> Result<Self> {
        let p = format!("layers.{idx}");
        let h = cfg.hidden_size;
        Ok(Self {
            input_layernorm: load_rmsnorm(
                weights,
                &format!("{p}.input_layernorm.weight"),
                h,
                cfg.rms_norm_eps,
                dtype,
            )?,
            post_attention_layernorm: load_rmsnorm(
                weights,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
                cfg.rms_norm_eps,
                dtype,
            )?,
            q_proj: load_linear(
                weights,
                &format!("{p}.self_attn.q_proj.weight"),
                cfg.q_out_dim(),
                h,
                dtype,
            )?,
            k_proj: load_linear(
                weights,
                &format!("{p}.self_attn.k_proj.weight"),
                cfg.kv_out_dim(),
                h,
                dtype,
            )?,
            v_proj: load_linear(
                weights,
                &format!("{p}.self_attn.v_proj.weight"),
                cfg.kv_out_dim(),
                h,
                dtype,
            )?,
            o_proj: load_linear(
                weights,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                cfg.q_out_dim(),
                dtype,
            )?,
            q_norm: load_rmsnorm(
                weights,
                &format!("{p}.self_attn.q_norm.weight"),
                cfg.head_dim,
                cfg.rms_norm_eps,
                dtype,
            )?,
            k_norm: load_rmsnorm(
                weights,
                &format!("{p}.self_attn.k_norm.weight"),
                cfg.head_dim,
                cfg.rms_norm_eps,
                dtype,
            )?,
            gate_proj: load_linear(
                weights,
                &format!("{p}.mlp.gate_proj.weight"),
                cfg.intermediate_size,
                h,
                dtype,
            )?,
            up_proj: load_linear(
                weights,
                &format!("{p}.mlp.up_proj.weight"),
                cfg.intermediate_size,
                h,
                dtype,
            )?,
            down_proj: load_linear(
                weights,
                &format!("{p}.mlp.down_proj.weight"),
                h,
                cfg.intermediate_size,
                dtype,
            )?,
        })
    }
}

#[derive(Debug)]
pub struct DFlashContextKv {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    cap: usize,
    len: usize,
    next_pos: u32,
}

impl DFlashContextKv {
    pub fn empty() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            cap: 0,
            len: 0,
            next_pos: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn next_pos(&self) -> u32 {
        self.next_pos
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.next_pos = 0;
    }

    pub fn fingerprint(&self) -> Result<f32> {
        let mut acc = 0f32;
        if self.len == 0 {
            return Ok(acc);
        }
        for t in self.k.iter().chain(self.v.iter()) {
            acc += t
                .narrow(1, 0, self.len)?
                .to_dtype(DType::F32)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?;
        }
        Ok(acc)
    }
}

pub struct LoadedDFlashDrafter {
    cfg: DFlashSpeculatorConfig,
    device: Device,
    dtype: DType,
    embed_tokens: Tensor,
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DFlashLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    d2t: Vec<u32>,
    t2d: Vec<bool>,
    d2t_map_dev: std::sync::OnceLock<Tensor>,
}

impl LoadedDFlashDrafter {
    pub fn config(&self) -> &DFlashSpeculatorConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn d2t(&self) -> &[u32] {
        &self.d2t
    }

    pub fn t2d(&self) -> &[bool] {
        &self.t2d
    }

    pub fn d2t_map(&self, draft_token: u32) -> u32 {
        crate::eagle3_loader::d2t_apply(&self.d2t, draft_token)
    }

    pub fn t2d_supports(&self, target_token: u32) -> bool {
        let i = target_token as usize;
        i < self.t2d.len() && self.t2d[i]
    }

    pub fn try_load(model_dir: &Path, device: &Device) -> Result<Self> {
        Self::try_load_with_target_embed(model_dir, device, None)
    }

    pub fn try_load_with_target_embed(
        model_dir: &Path,
        device: &Device,
        target_embed: Option<&Tensor>,
    ) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let cfg = if cfg_path.is_file() {
            DFlashSpeculatorConfig::from_hf_json_file(&cfg_path)?
        } else {
            DFlashSpeculatorConfig::default()
        };
        let st_path = model_dir.join("model.safetensors");
        if !st_path.is_file() {
            bail!("missing model.safetensors at {}", st_path.display());
        }
        Self::load_from_safetensors_with_embed(&cfg, &st_path, device, target_embed)
    }

    pub fn load_from_safetensors(
        cfg: &DFlashSpeculatorConfig,
        safetensors_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        Self::load_from_safetensors_with_embed(cfg, safetensors_path, device, None)
    }

    pub fn load_from_safetensors_with_embed(
        cfg: &DFlashSpeculatorConfig,
        safetensors_path: &Path,
        device: &Device,
        target_embed: Option<&Tensor>,
    ) -> Result<Self> {
        let weights = WeightLoader::open_file(safetensors_path, device)
            .with_context(|| format!("open {}", safetensors_path.display()))?;

        let dtype = if draft_f32_enabled(std::env::var("NV_DFLASH_DRAFT_F32").ok().as_deref()) {
            DType::F32
        } else {
            DType::BF16
        };

        let embed_tokens = if weights.has("embed_tokens.weight") {
            load_tensor(
                &weights,
                "embed_tokens.weight",
                &[cfg.target_vocab_size, cfg.hidden_size],
                dtype,
            )?
        } else {
            if !cfg.tied_embeddings {
                bail!(
                    "checkpoint ships no embed_tokens.weight and config does not set \
                     tie_word_embeddings"
                );
            }
            let src = target_embed.ok_or_else(|| {
                anyhow!(
                    "tied-embedding dflash checkpoint needs the target model's embed_tokens; \
                     load via try_load_with_target_embed"
                )
            })?;
            if src.dims() != [cfg.target_vocab_size, cfg.hidden_size] {
                bail!(
                    "target embed shape {:?} != [{}, {}]",
                    src.dims(),
                    cfg.target_vocab_size,
                    cfg.hidden_size
                );
            }
            src.to_device(device)?.to_dtype(dtype)?.contiguous()?
        };
        let fc = load_linear(
            &weights,
            "fc.weight",
            cfg.hidden_size,
            cfg.fc_in_dim(),
            dtype,
        )?;
        let hidden_norm = load_rmsnorm(
            &weights,
            "hidden_norm.weight",
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype,
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DFlashLayer::load(&weights, i, cfg, dtype)?);
        }
        let norm = load_rmsnorm(
            &weights,
            "norm.weight",
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype,
        )?;
        let lm_head = if weights.has("lm_head.weight") {
            load_linear(
                &weights,
                "lm_head.weight",
                cfg.draft_vocab_size,
                cfg.hidden_size,
                dtype,
            )?
        } else {
            if !cfg.tied_embeddings {
                bail!(
                    "checkpoint ships no lm_head.weight and config does not set \
                     tie_word_embeddings"
                );
            }
            if cfg.draft_vocab_size != cfg.target_vocab_size {
                bail!(
                    "tied lm_head needs draft_vocab_size == vocab_size ({} != {})",
                    cfg.draft_vocab_size,
                    cfg.target_vocab_size
                );
            }
            Linear::new(embed_tokens.clone(), None)?
        };
        let (d2t, t2d) = if weights.has("d2t") || weights.has("t2d") {
            let d2t = load_d2t(&weights, "d2t", cfg.draft_vocab_size)?;
            let t2d = load_t2d(&weights, "t2d", cfg.target_vocab_size)?;
            (d2t, t2d)
        } else {
            if cfg.draft_vocab_size != cfg.target_vocab_size {
                bail!(
                    "checkpoint ships no d2t/t2d but draft_vocab_size {} != vocab_size {}",
                    cfg.draft_vocab_size,
                    cfg.target_vocab_size
                );
            }
            (
                vec![0u32; cfg.draft_vocab_size],
                vec![true; cfg.target_vocab_size],
            )
        };
        validate_d2t(&d2t, cfg.draft_vocab_size, cfg.target_vocab_size)
            .with_context(|| format!("speculator {}", safetensors_path.display()))?;
        validate_t2d(&t2d, cfg.target_vocab_size)
            .with_context(|| format!("speculator {}", safetensors_path.display()))?;

        #[cfg(feature = "cuda")]
        let (layers, lm_head) = maybe_quantize_dflash_nvfp4(layers, lm_head, device, dtype)?;

        Self::from_parts(
            cfg.clone(),
            device.clone(),
            dtype,
            embed_tokens,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
            d2t,
            t2d,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        cfg: DFlashSpeculatorConfig,
        device: Device,
        dtype: DType,
        embed_tokens: Tensor,
        fc: Linear,
        hidden_norm: RmsNorm,
        layers: Vec<DFlashLayer>,
        norm: RmsNorm,
        lm_head: Linear,
        d2t: Vec<u32>,
        t2d: Vec<bool>,
    ) -> Result<Self> {
        if layers.len() != cfg.num_hidden_layers {
            bail!(
                "expected {} layers, got {}",
                cfg.num_hidden_layers,
                layers.len()
            );
        }
        validate_d2t(&d2t, cfg.draft_vocab_size, cfg.target_vocab_size)?;
        validate_t2d(&t2d, cfg.target_vocab_size)?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: cfg.head_dim,
                max_seq_len: cfg.max_position_embeddings,
                base: cfg.rope_theta,
                kind: RopeKind::Standard,
            },
            &device,
        )?;
        Ok(Self {
            cfg,
            device,
            dtype,
            embed_tokens,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
            rope,
            d2t,
            t2d,
            d2t_map_dev: std::sync::OnceLock::new(),
        })
    }

    pub fn project_aux(&self, aux_rows: &Tensor) -> Result<Tensor> {
        let dims = aux_rows.dims();
        if dims.len() != 2 || dims[1] != self.cfg.fc_in_dim() {
            bail!(
                "project_aux: expected [n, {}], got {:?}",
                self.cfg.fc_in_dim(),
                dims
            );
        }
        self.fc.forward(&aux_rows.to_dtype(self.dtype)?)
    }

    fn per_head_norm(&self, x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
        let dims = x.dims();
        if dims.last() != Some(&self.cfg.head_dim) {
            bail!(
                "per_head_norm: expected trailing head_dim {}, got {:?}",
                self.cfg.head_dim,
                dims
            );
        }
        norm.forward(&x.contiguous()?)
    }

    fn compute_chunk_kv(
        &self,
        ctx_states: &Tensor,
        ctx_positions: &[u32],
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let dims = ctx_states.dims();
        if dims.len() != 2 || dims[1] != self.cfg.hidden_size {
            bail!(
                "precompute_context_kv: expected [n, {}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let n = dims[0];
        if n == 0 {
            bail!("precompute_context_kv: empty context");
        }
        if ctx_positions.len() != n {
            bail!(
                "precompute_context_kv: {} positions for {} context rows",
                ctx_positions.len(),
                n
            );
        }
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let normed = self
            .hidden_norm
            .forward(&ctx_states.to_dtype(self.dtype)?)?;
        let pos_t = Tensor::from_vec(ctx_positions.to_vec(), (1usize, n), &self.device)?;
        let mut ks = Vec::with_capacity(self.layers.len());
        let mut vs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let k = layer
                .k_proj
                .forward(&normed)?
                .reshape((1usize, n, nkv, hd))?;
            let k = self.per_head_norm(&k, &layer.k_norm)?;
            let (k_rot, _) = self.rope.apply(&k, &k, &pos_t)?;
            let v = layer
                .v_proj
                .forward(&normed)?
                .reshape((1usize, n, nkv, hd))?;
            ks.push(k_rot.contiguous()?);
            vs.push(v.contiguous()?);
        }
        Ok((ks, vs))
    }

    pub fn new_context_kv(&self, cap: usize) -> Result<DFlashContextKv> {
        if cap == 0 {
            bail!("new_context_kv: cap must be >= 1");
        }
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let mut ks = Vec::with_capacity(self.layers.len());
        let mut vs = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            ks.push(Tensor::zeros(
                (1usize, cap, nkv, hd),
                self.dtype,
                &self.device,
            )?);
            vs.push(Tensor::zeros(
                (1usize, cap, nkv, hd),
                self.dtype,
                &self.device,
            )?);
        }
        Ok(DFlashContextKv {
            k: ks,
            v: vs,
            cap,
            len: 0,
            next_pos: 0,
        })
    }

    fn ensure_ctx_capacity(&self, ctx: &mut DFlashContextKv, needed: usize) -> Result<()> {
        if !ctx.k.is_empty() && ctx.cap >= needed {
            return Ok(());
        }
        let new_cap = needed.max(ctx.cap * 2).max(1024);
        let mut fresh = self.new_context_kv(new_cap)?;
        if ctx.len > 0 {
            for (dst, src) in fresh.k.iter().zip(ctx.k.iter()) {
                dst.slice_set(&src.narrow(1, 0, ctx.len)?.contiguous()?, 1, 0)?;
            }
            for (dst, src) in fresh.v.iter().zip(ctx.v.iter()) {
                dst.slice_set(&src.narrow(1, 0, ctx.len)?.contiguous()?, 1, 0)?;
            }
        }
        fresh.len = ctx.len;
        fresh.next_pos = ctx.next_pos;
        *ctx = fresh;
        Ok(())
    }

    pub fn precompute_context_kv(
        &self,
        ctx_states: &Tensor,
        ctx_positions: &[u32],
    ) -> Result<DFlashContextKv> {
        let mut ctx = DFlashContextKv::empty();
        self.append_context_kv(&mut ctx, ctx_states, ctx_positions)?;
        if ctx.is_empty() {
            bail!("precompute_context_kv: empty context");
        }
        Ok(ctx)
    }

    pub fn append_context_kv(
        &self,
        ctx: &mut DFlashContextKv,
        new_states: &Tensor,
        positions: &[u32],
    ) -> Result<()> {
        if positions.is_empty() {
            return Ok(());
        }
        if !ctx.is_empty() && positions[0] != ctx.next_pos {
            bail!(
                "append_context_kv: expected first position {}, got {}",
                ctx.next_pos,
                positions[0]
            );
        }
        let (ks, vs) = self.compute_chunk_kv(new_states, positions)?;
        let rows = positions.len();
        let next_pos = positions[rows - 1]
            .checked_add(1)
            .ok_or_else(|| anyhow!("context position overflow"))?;
        self.ensure_ctx_capacity(ctx, ctx.len + rows + self.cfg.query_rows())?;
        for (dst, src) in ctx.k.iter().zip(ks.iter()) {
            dst.slice_set(src, 1, ctx.len)?;
        }
        for (dst, src) in ctx.v.iter().zip(vs.iter()) {
            dst.slice_set(src, 1, ctx.len)?;
        }
        ctx.len += rows;
        ctx.next_pos = next_pos;
        Ok(())
    }

    pub fn block_input_ids(&self, anchor: u32) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.cfg.query_rows());
        ids.push(anchor);
        ids.extend(std::iter::repeat_n(
            self.cfg.mask_token_id,
            self.cfg.block_size,
        ));
        ids
    }

    fn build_block_mask(
        &self,
        ctx: &DFlashContextKv,
        attn: DFlashLayerAttn,
        m: usize,
    ) -> Result<Tensor> {
        let n_ctx = ctx.len;
        let n_all = n_ctx + m;
        let base = ctx.next_pos as i64 - n_ctx as i64;
        let q0 = ctx.next_pos as i64;
        let w = self.cfg.sliding_window as i64;
        let mut mask = vec![0f32; m * n_all];
        for i in 0..m {
            let qpos = q0 + i as i64;
            for j in 0..n_all {
                let kpos = if j < n_ctx {
                    base + j as i64
                } else {
                    q0 + (j - n_ctx) as i64
                };
                let hidden_by_causal = attn.causal && kpos > qpos;
                let hidden_by_window = attn.sliding && qpos - kpos >= w;
                if hidden_by_causal || hidden_by_window {
                    mask[i * n_all + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(mask, (m, n_all), &self.device)?)
    }

    fn attn_masked(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        let (_, _m, nh, hd) = q.dims4()?;
        let (_, n_all, nkv, _) = k_all.dims4()?;
        if nh % nkv != 0 {
            bail!("attn_masked: nh {nh} not divisible by nkv {nkv}");
        }
        let group = nh / nkv;
        let expand = |t: &Tensor| -> Result<Tensor> {
            let t = t
                .to_dtype(DType::F32)?
                .squeeze(0)?
                .transpose(0, 1)?
                .contiguous()?;
            if group == 1 {
                return Ok(t);
            }
            Ok(t.unsqueeze(1)?
                .repeat((1usize, group, 1usize, 1usize))?
                .reshape((nh, n_all, hd))?)
        };
        let q_h = q
            .to_dtype(DType::F32)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?;
        let k_h = expand(k_all)?;
        let v_h = expand(v_all)?;
        let scores = (q_h.matmul(&k_h.transpose(1, 2)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v_h)?;
        Ok(out.transpose(0, 1)?.contiguous()?.unsqueeze(0)?)
    }

    pub fn forward_block_hidden(&self, ctx: &DFlashContextKv, anchor: u32) -> Result<Tensor> {
        if ctx.k.len() != self.layers.len() {
            bail!(
                "forward_block_hidden: context KV has {} layers, drafter has {}",
                ctx.k.len(),
                self.layers.len()
            );
        }
        let m = self.cfg.query_rows();
        let scratch = ctx.cap >= ctx.len + m;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let ids = Tensor::from_vec(self.block_input_ids(anchor), m, &self.device)?;
        let mut x = self.embed_tokens.index_select(&ids, 0)?;
        let positions: Vec<u32> = (ctx.next_pos..ctx.next_pos + m as u32).collect();
        let pos_t = Tensor::from_vec(positions, (1usize, m), &self.device)?;
        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0f32 / (hd as f32).sqrt(),
            causal: false,
        };
        for (li, (layer, (k_ctx, v_ctx))) in self
            .layers
            .iter()
            .zip(ctx.k.iter().zip(ctx.v.iter()))
            .enumerate()
        {
            let h = layer.input_layernorm.forward(&x)?;
            let q = layer.q_proj.forward(&h)?.reshape((1usize, m, nh, hd))?;
            let q = self.per_head_norm(&q, &layer.q_norm)?;
            let k = layer.k_proj.forward(&h)?.reshape((1usize, m, nkv, hd))?;
            let k = self.per_head_norm(&k, &layer.k_norm)?;
            let v = layer.v_proj.forward(&h)?.reshape((1usize, m, nkv, hd))?;
            let (q_rot, k_rot) = self.rope.apply(&q, &k, &pos_t)?;
            let (k_all, v_all) = if scratch {
                k_ctx.slice_set(&k_rot.contiguous()?, 1, ctx.len)?;
                v_ctx.slice_set(&v.contiguous()?, 1, ctx.len)?;
                (
                    k_ctx.narrow(1, 0, ctx.len + m)?,
                    v_ctx.narrow(1, 0, ctx.len + m)?,
                )
            } else {
                (
                    Tensor::cat(&[&k_ctx.narrow(1, 0, ctx.len)?, &k_rot], 1)?,
                    Tensor::cat(&[&v_ctx.narrow(1, 0, ctx.len)?, &v], 1)?,
                )
            };
            let la = self.cfg.layer_attn_for(li);
            let attn_out = if la.causal || la.sliding {
                let mask = self.build_block_mask(ctx, la, m)?;
                self.attn_masked(&q_rot, &k_all, &v_all, &mask, attn_cfg.softmax_scale as f64)?
            } else {
                sdpa(&q_rot, &k_all, &v_all, &attn_cfg)?
            };
            let attn_out = attn_out
                .squeeze(0)?
                .reshape((m, nh * hd))?
                .to_dtype(self.dtype)?;
            let attn_out = layer.o_proj.forward(&attn_out)?;
            let post_attn = x.add(&attn_out)?;
            let mlp_in = layer.post_attention_layernorm.forward(&post_attn)?;
            let gate = layer.gate_proj.forward(&mlp_in)?;
            let up = layer.up_proj.forward(&mlp_in)?;
            let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
            let mlp_out = layer.down_proj.forward(&act)?;
            x = post_attn.add(&mlp_out)?;
        }
        self.norm.forward(&x)
    }

    fn draft_indices(&self, ctx: &DFlashContextKv, anchor: u32) -> Result<Tensor> {
        let hidden = self.forward_block_hidden(ctx, anchor)?;
        let logits = self.lm_head.forward(&hidden)?.to_dtype(DType::F32)?;
        let idx_all = logits
            .reshape((self.cfg.query_rows(), self.cfg.draft_vocab_size))?
            .argmax(D::Minus1)?;
        Ok(idx_all.narrow(0, 1, self.cfg.block_size)?.contiguous()?)
    }

    fn d2t_map_dev_tensor(&self) -> Result<Tensor> {
        crate::eagle3_loader::d2t_dev_tensor(
            &self.d2t_map_dev,
            &self.d2t,
            self.cfg.draft_vocab_size,
            &self.device,
        )
    }

    pub fn draft_block(&self, ctx: &DFlashContextKv, anchor: u32) -> Result<Vec<u32>> {
        let idx = self.draft_indices(ctx, anchor)?.to_vec1::<u32>()?;
        Ok(idx.into_iter().map(|d| self.d2t_map(d)).collect())
    }

    pub fn draft_block_all_rows(&self, ctx: &DFlashContextKv, anchor: u32) -> Result<Vec<u32>> {
        let hidden = self.forward_block_hidden(ctx, anchor)?;
        let logits = self.lm_head.forward(&hidden)?.to_dtype(DType::F32)?;
        let idx = logits
            .reshape((self.cfg.query_rows(), self.cfg.draft_vocab_size))?
            .argmax(D::Minus1)?
            .to_vec1::<u32>()?;
        Ok(idx.into_iter().map(|d| self.d2t_map(d)).collect())
    }

    pub fn draft_block_all_rows_with_conf(
        &self,
        ctx: &DFlashContextKv,
        anchor: u32,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let hidden = self.forward_block_hidden(ctx, anchor)?;
        let logits = self.lm_head.forward(&hidden)?.to_dtype(DType::F32)?;
        let rows: Vec<Vec<f32>> = logits
            .reshape((self.cfg.query_rows(), self.cfg.draft_vocab_size))?
            .to_vec2()?;
        let mut ids = Vec::with_capacity(rows.len());
        let mut top_prob = Vec::with_capacity(rows.len());
        let mut margin = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut best = (0usize, f32::NEG_INFINITY);
            let mut second = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best.1 {
                    second = best.1;
                    best = (i, v);
                } else if v > second {
                    second = v;
                }
            }
            let denom: f32 = row.iter().map(|&v| (v - best.1).exp()).sum();
            ids.push(self.d2t_map(best.0 as u32));
            top_prob.push(1.0 / denom);
            margin.push(best.1 - second);
        }
        Ok((ids, top_prob, margin))
    }

    pub fn draft_block_dev(&self, ctx: &DFlashContextKv, anchor: u32) -> Result<Tensor> {
        let idx = self.draft_indices(ctx, anchor)?;
        Ok(self.d2t_map_dev_tensor()?.index_select(&idx, 0)?)
    }

    pub fn draft_block_from_aux(
        &self,
        aux_rows: &Tensor,
        ctx_positions: &[u32],
        anchor: u32,
    ) -> Result<Vec<u32>> {
        let proj = self.project_aux(aux_rows)?;
        let ctx = self.precompute_context_kv(&proj, ctx_positions)?;
        self.draft_block(&ctx, anchor)
    }
}

#[cfg(feature = "cuda")]
fn dflash_quant_nvfp4_enabled() -> bool {
    std::env::var("NV_DFLASH_QUANT")
        .map(|v| v == "nvfp4")
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn dflash_quant_lmhead_enabled() -> bool {
    std::env::var("NV_DFLASH_QUANT_LMHEAD")
        .map(|v| v != "0")
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn nvfp4_quantizable(l: &Linear) -> bool {
    use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
    l.weight().is_some()
        && l.bias().is_none()
        && l.in_features() % BLOCK_SIZE == 0
        && l.in_features() >= MIN_TILE
        && l.out_features() >= MIN_TILE
}

#[cfg(feature = "cuda")]
fn maybe_quantize_dflash_nvfp4(
    mut layers: Vec<DFlashLayer>,
    mut lm_head: Linear,
    device: &Device,
    dtype: DType,
) -> Result<(Vec<DFlashLayer>, Linear)> {
    if !dflash_quant_nvfp4_enabled() || dtype != DType::BF16 || !matches!(device, Device::Cuda(_)) {
        return Ok((layers, lm_head));
    }
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let runner = std::sync::Arc::new(std::sync::Mutex::new(
        nv_quant::nvfp4::Nvfp4GemmRunner::new(stream)?,
    ));
    let quant = |l: &mut Linear, name: &str| -> Result<()> {
        if !nvfp4_quantizable(l) {
            eprintln!("[dflash-quant] skipping {name}: not nvfp4-eligible");
            return Ok(());
        }
        let w = l
            .weight()
            .ok_or_else(|| anyhow!("{name}: missing bf16 weight"))?
            .clone();
        *l = Linear::from_bf16_quantized_nvfp4_dev(&w, None, device, runner.clone())
            .with_context(|| format!("nvfp4 quantize dflash {name}"))?;
        Ok(())
    };
    for (i, layer) in layers.iter_mut().enumerate() {
        quant(&mut layer.q_proj, &format!("layers.{i}.q_proj"))?;
        quant(&mut layer.k_proj, &format!("layers.{i}.k_proj"))?;
        quant(&mut layer.v_proj, &format!("layers.{i}.v_proj"))?;
        quant(&mut layer.o_proj, &format!("layers.{i}.o_proj"))?;
        quant(&mut layer.gate_proj, &format!("layers.{i}.gate_proj"))?;
        quant(&mut layer.up_proj, &format!("layers.{i}.up_proj"))?;
        quant(&mut layer.down_proj, &format!("layers.{i}.down_proj"))?;
    }
    let lm_quant = dflash_quant_lmhead_enabled();
    if lm_quant {
        quant(&mut lm_head, "lm_head")?;
    }
    eprintln!(
        "[dflash-quant] drafter transformer weights quantized to nvfp4 (lm_head {})",
        if lm_quant { "nvfp4" } else { "bf16" }
    );
    Ok((layers, lm_head))
}

#[cfg(feature = "cuda")]
pub struct DFlashBlockGraph {
    forked: std::sync::Arc<cudarc::driver::CudaStream>,
    runner: nv_kernels::graph::CudaGraphRunner,
    cap: usize,
    m: usize,
    disabled: bool,
    warmed: bool,
    ids_host: Vec<i32>,
    pos_host: Vec<i32>,
    ids_buf: cudarc::driver::CudaSlice<i32>,
    pos_buf: cudarc::driver::CudaSlice<i32>,
    n_buf: cudarc::driver::CudaSlice<i32>,
    mask_buf: cudarc::driver::CudaSlice<u8>,
    mask_causal_buf: Option<cudarc::driver::CudaSlice<u8>>,
    out_buf: cudarc::driver::CudaSlice<u32>,
    draft_idx: cudarc::driver::CudaSlice<u32>,
    amax_val: cudarc::driver::CudaSlice<f32>,
    amax_idx: cudarc::driver::CudaSlice<i32>,
    x_buf: cudarc::driver::CudaSlice<half::bf16>,
    normed_buf: cudarc::driver::CudaSlice<half::bf16>,
    q_buf: cudarc::driver::CudaSlice<half::bf16>,
    k_buf: cudarc::driver::CudaSlice<half::bf16>,
    v_buf: cudarc::driver::CudaSlice<half::bf16>,
    attn_buf: cudarc::driver::CudaSlice<half::bf16>,
    o_buf: cudarc::driver::CudaSlice<half::bf16>,
    pa_buf: cudarc::driver::CudaSlice<half::bf16>,
    nm_buf: cudarc::driver::CudaSlice<half::bf16>,
    gate_buf: cudarc::driver::CudaSlice<half::bf16>,
    up_buf: cudarc::driver::CudaSlice<half::bf16>,
    act_buf: cudarc::driver::CudaSlice<half::bf16>,
    mlp_buf: cudarc::driver::CudaSlice<half::bf16>,
    logits_buf: cudarc::driver::CudaSlice<half::bf16>,
    logits_f32: cudarc::driver::CudaSlice<f32>,
    fp4_a: Option<cudarc::driver::CudaSlice<u8>>,
    fp4_sc: Option<cudarc::driver::CudaSlice<u8>>,
    fp4_ws: Option<cudarc::driver::CudaSlice<u8>>,
    q_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    k_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    v_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    o_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    gate_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    up_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    mlp_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
    lm_pad: Option<cudarc::driver::CudaSlice<half::bf16>>,
}

#[cfg(feature = "cuda")]
impl Drop for DFlashBlockGraph {
    fn drop(&mut self) {
        let teardown =
            nv_models::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked);
        let runner = &mut self.runner;
        teardown.run(|| runner.invalidate());
    }
}

#[cfg(feature = "cuda")]
impl DFlashBlockGraph {
    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn disabled(&self) -> bool {
        self.disabled
    }

    pub fn disable(&mut self) {
        self.disabled = true;
    }

    pub fn graph_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }
}

#[cfg(feature = "cuda")]
impl LoadedDFlashDrafter {
    fn graph_linear_is_fp4(l: &Linear, name: &str) -> Result<bool> {
        if l.bias().is_some() {
            bail!("block graph: {name} bias unsupported");
        }
        if let Some(w) = l.weight() {
            if !w.is_contiguous() || w.dtype() != DType::BF16 {
                bail!("block graph: {name} weight must be contiguous bf16");
            }
            Ok(false)
        } else if l.nvfp4_parts_full().is_some() {
            Ok(true)
        } else {
            bail!("block graph: {name} is neither bf16-dense nor nvfp4");
        }
    }

    fn block_graph_eligible(&self) -> Result<[bool; 8]> {
        if self.dtype != DType::BF16 || self.embed_tokens.dtype() != DType::BF16 {
            bail!("block graph requires a bf16 drafter");
        }
        if !self.embed_tokens.is_contiguous() {
            bail!("block graph: embed_tokens must be contiguous");
        }
        if self.cfg.head_dim % 2 != 0 {
            bail!("block graph: head_dim must be even");
        }
        let mut flags: Option<[bool; 7]> = None;
        for layer in &self.layers {
            let f = [
                Self::graph_linear_is_fp4(&layer.q_proj, "q_proj")?,
                Self::graph_linear_is_fp4(&layer.k_proj, "k_proj")?,
                Self::graph_linear_is_fp4(&layer.v_proj, "v_proj")?,
                Self::graph_linear_is_fp4(&layer.o_proj, "o_proj")?,
                Self::graph_linear_is_fp4(&layer.gate_proj, "gate_proj")?,
                Self::graph_linear_is_fp4(&layer.up_proj, "up_proj")?,
                Self::graph_linear_is_fp4(&layer.down_proj, "down_proj")?,
            ];
            match &flags {
                None => flags = Some(f),
                Some(prev) if *prev == f => {}
                Some(_) => bail!("block graph: mixed per-layer quantization is unsupported"),
            }
        }
        let f = flags.ok_or_else(|| anyhow!("block graph: no layers"))?;
        let lm_fp4 = Self::graph_linear_is_fp4(&self.lm_head, "lm_head")?;
        if (f.iter().any(|&x| x) || lm_fp4) && self.cfg.query_rows() > FP4_STAGING_ROWS {
            bail!(
                "block graph: nvfp4 activation staging holds at most {FP4_STAGING_ROWS} \
                 query rows but block_size {} gives {} rows; lower block_size or disable \
                 NV_DFLASH_QUANT=nvfp4",
                self.cfg.block_size,
                self.cfg.query_rows()
            );
        }
        Ok([f[0], f[1], f[2], f[3], f[4], f[5], f[6], lm_fp4])
    }

    pub fn new_block_graph(&self, cap: usize) -> Result<DFlashBlockGraph> {
        let fp4_flags = self.block_graph_eligible()?;
        let any_fp4 = fp4_flags.iter().any(|&f| f);
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("new_block_graph requires cuda"),
        };
        let cfg = &self.cfg;
        let m = cfg.query_rows();
        let md = cfg.block_size;
        if cap < m {
            bail!("new_block_graph: cap {cap} < query rows {m}");
        }
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let dv = cfg.draft_vocab_size;
        let parts = nv_kernels::cuda::argmax_parts();

        let raw_ctx: std::sync::Arc<cudarc::driver::CudaContext> =
            dev.cuda_stream().context().clone();
        let forked = raw_ctx.new_stream().map_err(|e| anyhow!(e))?;

        let abf = |n: usize| -> Result<cudarc::driver::CudaSlice<half::bf16>> {
            forked
                .alloc_zeros::<half::bf16>(n)
                .map_err(|e| anyhow!("block graph buf alloc ({n}): {e:?}"))
        };
        let ids_buf = forked.alloc_zeros::<i32>(m).map_err(|e| anyhow!(e))?;
        let pos_buf = forked.alloc_zeros::<i32>(m).map_err(|e| anyhow!(e))?;
        let n_buf = forked.alloc_zeros::<i32>(1).map_err(|e| anyhow!(e))?;
        let mut mask_buf = forked.alloc_zeros::<u8>(m * m).map_err(|e| anyhow!(e))?;
        let mask_host = vec![1u8; m * m];
        forked
            .memcpy_htod(&mask_host[..], &mut mask_buf)
            .map_err(|e| anyhow!("mask htod: {e:?}"))?;
        let any_causal = (0..self.layers.len()).any(|i| cfg.layer_attn_for(i).causal);
        let mask_causal_buf = if any_causal {
            let mut buf = forked.alloc_zeros::<u8>(m * m).map_err(|e| anyhow!(e))?;
            let mut host = vec![0u8; m * m];
            for i in 0..m {
                for j in 0..=i {
                    host[i * m + j] = 1;
                }
            }
            forked
                .memcpy_htod(&host[..], &mut buf)
                .map_err(|e| anyhow!("causal mask htod: {e:?}"))?;
            Some(buf)
        } else {
            None
        };
        let out_buf = forked.alloc_zeros::<u32>(md).map_err(|e| anyhow!(e))?;
        let draft_idx = forked.alloc_zeros::<u32>(md).map_err(|e| anyhow!(e))?;
        let amax_val = forked
            .alloc_zeros::<f32>(md * parts)
            .map_err(|e| anyhow!(e))?;
        let amax_idx = forked
            .alloc_zeros::<i32>(md * parts)
            .map_err(|e| anyhow!(e))?;
        let x_buf = abf(m * h)?;
        let normed_buf = abf(m * h)?;
        let q_buf = abf(m * nh * hd)?;
        let k_buf = abf(m * nkv * hd)?;
        let v_buf = abf(m * nkv * hd)?;
        let attn_buf = abf(m * nh * hd)?;
        let o_buf = abf(m * h)?;
        let pa_buf = abf(m * h)?;
        let nm_buf = abf(m * h)?;
        let gate_buf = abf(m * inter)?;
        let up_buf = abf(m * inter)?;
        let act_buf = abf(m * inter)?;
        let mlp_buf = abf(m * h)?;
        let logits_buf = abf(md * dv)?;
        let logits_f32 = forked.alloc_zeros::<f32>(md * dv).map_err(|e| anyhow!(e))?;

        let mpad = nv_quant::nvfp4::MIN_TILE;
        let (fp4_a, fp4_sc, fp4_ws) = if any_fp4 {
            let kmax = h.max(nh * hd).max(inter);
            let sc_bytes =
                mpad.div_ceil(128) * 128 * (kmax / nv_quant::nvfp4::BLOCK_SIZE + 3).div_ceil(4) * 4;
            (
                Some(
                    forked
                        .alloc_zeros::<u8>(mpad * kmax / 2)
                        .map_err(|e| anyhow!(e))?,
                ),
                Some(forked.alloc_zeros::<u8>(sc_bytes).map_err(|e| anyhow!(e))?),
                Some(
                    forked
                        .alloc_zeros::<u8>(nv_quant::nvfp4::WORKSPACE_BYTES)
                        .map_err(|e| anyhow!(e))?,
                ),
            )
        } else {
            (None, None, None)
        };
        let opt_abf =
            |on: bool, n: usize| -> Result<Option<cudarc::driver::CudaSlice<half::bf16>>> {
                if on {
                    Ok(Some(abf(n)?))
                } else {
                    Ok(None)
                }
            };
        let q_pad = opt_abf(fp4_flags[0], mpad * nh * hd)?;
        let k_pad = opt_abf(fp4_flags[1], mpad * nkv * hd)?;
        let v_pad = opt_abf(fp4_flags[2], mpad * nkv * hd)?;
        let o_pad = opt_abf(fp4_flags[3], mpad * h)?;
        let gate_pad = opt_abf(fp4_flags[4], mpad * inter)?;
        let up_pad = opt_abf(fp4_flags[5], mpad * inter)?;
        let mlp_pad = opt_abf(fp4_flags[6], mpad * h)?;
        let lm_pad = opt_abf(fp4_flags[7], mpad * dv)?;
        forked.synchronize().map_err(|e| anyhow!(e))?;

        let _ = self.d2t_map_dev_tensor()?;
        let runner = nv_kernels::graph::CudaGraphRunner::new(forked.clone());
        Ok(DFlashBlockGraph {
            forked,
            runner,
            cap,
            m,
            disabled: false,
            warmed: false,
            ids_host: vec![cfg.mask_token_id as i32; m],
            pos_host: vec![0i32; m],
            ids_buf,
            pos_buf,
            n_buf,
            mask_buf,
            mask_causal_buf,
            out_buf,
            draft_idx,
            amax_val,
            amax_idx,
            x_buf,
            normed_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_buf,
            o_buf,
            pa_buf,
            nm_buf,
            gate_buf,
            up_buf,
            act_buf,
            mlp_buf,
            logits_buf,
            logits_f32,
            fp4_a,
            fp4_sc,
            fp4_ws,
            q_pad,
            k_pad,
            v_pad,
            o_pad,
            gate_pad,
            up_pad,
            mlp_pad,
            lm_pad,
        })
    }

    pub fn draft_block_graphed(
        &self,
        ctx: &DFlashContextKv,
        g: &mut DFlashBlockGraph,
        anchor: u32,
        eager_body: bool,
    ) -> Result<Vec<u32>> {
        use crate::eagle3_loader::{chain_raw_bf16, chain_raw_f32, chain_raw_u32, chain_slice_ptr};

        if g.disabled {
            return self.draft_block(ctx, anchor);
        }
        let cfg = &self.cfg;
        let any_sliding = (0..self.layers.len()).any(|i| cfg.layer_attn_for(i).sliding);
        if any_sliding && ctx.next_pos as usize != ctx.len {
            return self.draft_block(ctx, anchor);
        }
        let m = g.m;
        if ctx.k.len() != self.layers.len() {
            bail!(
                "draft_block_graphed: context KV has {} layers, drafter has {}",
                ctx.k.len(),
                self.layers.len()
            );
        }
        if ctx.cap < ctx.len + m || g.cap < ctx.len + m {
            bail!(
                "draft_block_graphed: ctx len {} + block {m} exceeds cap (ctx {}, graph {})",
                ctx.len,
                ctx.cap,
                g.cap
            );
        }
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("draft_block_graphed requires cuda"),
        };
        let raw_ctx = dev.cuda_stream().context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            dev.cuda_stream().synchronize().map_err(|e| anyhow!(e))?;
        }
        nv_layers::cuda_stream::current_stream(&dev)
            .synchronize()
            .map_err(|e| anyhow!(e))?;

        g.ids_host[0] = anchor as i32;
        for (i, p) in g.pos_host.iter_mut().enumerate() {
            *p = ctx.next_pos as i32 + i as i32;
        }
        let n_host = [ctx.len as i32];
        g.forked
            .memcpy_htod(&g.ids_host[..], &mut g.ids_buf)
            .map_err(|e| anyhow!("ids htod: {e:?}"))?;
        g.forked
            .memcpy_htod(&g.pos_host[..], &mut g.pos_buf)
            .map_err(|e| anyhow!("pos htod: {e:?}"))?;
        g.forked
            .memcpy_htod(&n_host[..], &mut g.n_buf)
            .map_err(|e| anyhow!("n htod: {e:?}"))?;

        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let dv = cfg.draft_vocab_size;
        let tv = cfg.target_vocab_size;
        let md = cfg.block_size;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let eps = cfg.rms_norm_eps as f32;

        let fk = g.forked.clone();
        let d2t_t = self.d2t_map_dev_tensor()?;

        enum GW {
            Bf16(u64),
            Fp4 {
                w: u64,
                ws: u64,
                alpha_dev: u64,
                sg: f32,
            },
        }
        let need_w = |l: &Linear, name: &str| -> Result<GW> {
            if let Some(w) = l.weight() {
                Ok(GW::Bf16(chain_raw_bf16(w, &fk)?))
            } else if let Some((wu, wsc, adv, _alpha, sg)) = l.nvfp4_parts_full() {
                Ok(GW::Fp4 {
                    w: chain_slice_ptr(wu, &fk),
                    ws: chain_slice_ptr(wsc, &fk),
                    alpha_dev: chain_slice_ptr(adv, &fk),
                    sg,
                })
            } else {
                Err(anyhow!(
                    "block graph: {name} is neither bf16-dense nor nvfp4"
                ))
            }
        };

        let p_emb_w = chain_raw_bf16(&self.embed_tokens, &fk)?;
        let p_wn = chain_raw_bf16(self.norm.weight_bf16(), &fk)?;
        let p_wlm = need_w(&self.lm_head, "lm_head")?;
        let p_cos = chain_raw_f32(self.rope.cos(), &fk)?;
        let p_sin = chain_raw_f32(self.rope.sin(), &fk)?;
        let p_d2t = chain_raw_u32(&d2t_t, &fk)?;

        struct LayerPtrs {
            wil: u64,
            wpl: u64,
            wqn: u64,
            wkn: u64,
            wq: GW,
            wk: GW,
            wv: GW,
            wo: GW,
            wg: GW,
            wu: GW,
            wd: GW,
            kc: u64,
            vc: u64,
            window: i32,
            causal: bool,
        }
        let mut lp: Vec<LayerPtrs> = Vec::with_capacity(self.layers.len());
        for (li, (layer, (kt, vt))) in self
            .layers
            .iter()
            .zip(ctx.k.iter().zip(ctx.v.iter()))
            .enumerate()
        {
            let la = cfg.layer_attn_for(li);
            if la.causal && g.mask_causal_buf.is_none() {
                bail!("draft_block_graphed: causal layer {li} but graph has no causal mask");
            }
            lp.push(LayerPtrs {
                wil: chain_raw_bf16(layer.input_layernorm.weight_bf16(), &fk)?,
                wpl: chain_raw_bf16(layer.post_attention_layernorm.weight_bf16(), &fk)?,
                wqn: chain_raw_bf16(layer.q_norm.weight_bf16(), &fk)?,
                wkn: chain_raw_bf16(layer.k_norm.weight_bf16(), &fk)?,
                wq: need_w(&layer.q_proj, "q_proj")?,
                wk: need_w(&layer.k_proj, "k_proj")?,
                wv: need_w(&layer.v_proj, "v_proj")?,
                wo: need_w(&layer.o_proj, "o_proj")?,
                wg: need_w(&layer.gate_proj, "gate_proj")?,
                wu: need_w(&layer.up_proj, "up_proj")?,
                wd: need_w(&layer.down_proj, "down_proj")?,
                kc: chain_raw_bf16(kt, &fk)?,
                vc: chain_raw_bf16(vt, &fk)?,
                window: if la.sliding {
                    cfg.sliding_window as i32
                } else {
                    0
                },
                causal: la.causal,
            });
        }

        let p_ids = chain_slice_ptr(&g.ids_buf, &fk);
        let p_pos = chain_slice_ptr(&g.pos_buf, &fk);
        let p_n = chain_slice_ptr(&g.n_buf, &fk);
        let p_mask = chain_slice_ptr(&g.mask_buf, &fk);
        let p_mask_causal = g
            .mask_causal_buf
            .as_ref()
            .map(|b| chain_slice_ptr(b, &fk))
            .unwrap_or(p_mask);
        let p_out = chain_slice_ptr(&g.out_buf, &fk);
        let p_didx = chain_slice_ptr(&g.draft_idx, &fk);
        let p_aval = chain_slice_ptr(&g.amax_val, &fk);
        let p_aidx = chain_slice_ptr(&g.amax_idx, &fk);
        let p_x = chain_slice_ptr(&g.x_buf, &fk);
        let p_normed = chain_slice_ptr(&g.normed_buf, &fk);
        let p_q = chain_slice_ptr(&g.q_buf, &fk);
        let p_k = chain_slice_ptr(&g.k_buf, &fk);
        let p_v = chain_slice_ptr(&g.v_buf, &fk);
        let p_attn = chain_slice_ptr(&g.attn_buf, &fk);
        let p_o = chain_slice_ptr(&g.o_buf, &fk);
        let p_pa = chain_slice_ptr(&g.pa_buf, &fk);
        let p_nm = chain_slice_ptr(&g.nm_buf, &fk);
        let p_gate = chain_slice_ptr(&g.gate_buf, &fk);
        let p_up = chain_slice_ptr(&g.up_buf, &fk);
        let p_act = chain_slice_ptr(&g.act_buf, &fk);
        let p_mlp = chain_slice_ptr(&g.mlp_buf, &fk);
        let p_logits = chain_slice_ptr(&g.logits_buf, &fk);
        let p_logits_f32 = chain_slice_ptr(&g.logits_f32, &fk);

        let opt_ptr = |o: &Option<cudarc::driver::CudaSlice<half::bf16>>, dflt: u64| -> u64 {
            o.as_ref().map(|s| chain_slice_ptr(s, &fk)).unwrap_or(dflt)
        };
        let p_qo = opt_ptr(&g.q_pad, p_q);
        let p_ko = opt_ptr(&g.k_pad, p_k);
        let p_vo = opt_ptr(&g.v_pad, p_v);
        let p_oo = opt_ptr(&g.o_pad, p_o);
        let p_gateo = opt_ptr(&g.gate_pad, p_gate);
        let p_upo = opt_ptr(&g.up_pad, p_up);
        let p_mlpo = opt_ptr(&g.mlp_pad, p_mlp);
        let p_logitso = opt_ptr(&g.lm_pad, p_logits);
        let p_fp4a = g
            .fp4_a
            .as_ref()
            .map(|s| chain_slice_ptr(s, &fk))
            .unwrap_or(0);
        let p_fp4sc = g
            .fp4_sc
            .as_ref()
            .map(|s| chain_slice_ptr(s, &fk))
            .unwrap_or(0);
        let p_fp4ws = g
            .fp4_ws
            .as_ref()
            .map(|s| chain_slice_ptr(s, &fk))
            .unwrap_or(0);
        let mpad = nv_quant::nvfp4::MIN_TILE;

        let use_mk_gemm = std::env::var("NV_DFLASH_GEMM")
            .map(|v| v == "mk")
            .unwrap_or(false);
        let splitk = nv_quant::matmul::splitk_enabled();
        let body = |s: &std::sync::Arc<cudarc::driver::CudaStream>| -> Result<()> {
            let cu = s.cu_stream() as *mut std::ffi::c_void;
            let gemm = |gw: &GW,
                        x: u64,
                        y: u64,
                        n: usize,
                        k: usize,
                        mm: usize,
                        what: &str|
             -> Result<()> {
                let w = match gw {
                    GW::Fp4 {
                        w,
                        ws,
                        alpha_dev,
                        sg,
                    } => {
                        let rc = unsafe {
                            nv_kernels::cuda::quantize_nvfp4_bf16_rows(
                                cu,
                                x as *const u16,
                                p_fp4a as *mut u8,
                                p_fp4sc as *mut u8,
                                *sg,
                                mm as i32,
                                k as i32,
                            )
                        };
                        anyhow::ensure!(rc == 0, "{what} act quant rc={rc}");
                        unsafe {
                            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_streamk(
                                cu,
                                p_fp4a as *const std::ffi::c_void,
                                p_fp4sc as *const std::ffi::c_void,
                                *w as *const std::ffi::c_void,
                                *ws as *const std::ffi::c_void,
                                *alpha_dev as *const f32,
                                y as *mut std::ffi::c_void,
                                mpad as i32,
                                n as i32,
                                k as i32,
                                p_fp4ws as *mut std::ffi::c_void,
                                nv_quant::nvfp4::WORKSPACE_BYTES,
                            )
                        }
                        .map_err(|rc| anyhow!("{what} fp4 gemm rc={rc}"))?;
                        return Ok(());
                    }
                    GW::Bf16(w) => *w,
                };
                if use_mk_gemm {
                    let rc = unsafe {
                        nv_kernels::cuda::gemm_bf16_mk(
                            cu,
                            w as *const u16,
                            x as *const u16,
                            y as *mut u16,
                            n as i32,
                            k as i32,
                            mm as i32,
                        )
                    };
                    anyhow::ensure!(rc == 0, "{what} rc={rc}");
                    return Ok(());
                }
                unsafe {
                    nv_quant::matmul::bf16_bt_matmul_det_raw(
                        s,
                        x as *const std::ffi::c_void,
                        w as *const std::ffi::c_void,
                        y as *mut std::ffi::c_void,
                        mm as u64,
                        n as u64,
                        k as u64,
                        1.0,
                        0.0,
                        splitk,
                    )
                }
                .map_err(|e| anyhow!("{what}: {e}"))
            };
            let rms = |x: u64, w: u64, y: u64, rows: usize, dim: usize, what: &str| -> Result<()> {
                let rc = unsafe {
                    nv_kernels::cuda::rmsnorm_bf16(
                        cu,
                        x as *const u16,
                        w as *const u16,
                        y as *mut u16,
                        rows,
                        dim,
                        eps,
                    )
                };
                anyhow::ensure!(rc == 0, "{what} rc={rc}");
                Ok(())
            };

            let rc = unsafe {
                nv_kernels::cuda::gather_rows_bf16(
                    cu,
                    p_emb_w as *const u16,
                    p_ids as *const i32,
                    p_x as *mut u16,
                    m as i32,
                    h as i32,
                    tv as i32,
                )
            };
            anyhow::ensure!(rc == 0, "block embed gather rc={rc}");

            for l in lp.iter() {
                rms(p_x, l.wil, p_normed, m, h, "block input_ln")?;
                gemm(&l.wq, p_normed, p_qo, nh * hd, h, m, "block q_proj")?;
                gemm(&l.wk, p_normed, p_ko, nkv * hd, h, m, "block k_proj")?;
                gemm(&l.wv, p_normed, p_vo, nkv * hd, h, m, "block v_proj")?;
                rms(p_qo, l.wqn, p_qo, m * nh, hd, "block q_norm")?;
                rms(p_ko, l.wkn, p_ko, m * nkv, hd, "block k_norm")?;
                let rc = unsafe {
                    nv_kernels::cuda::rope_bf16(
                        cu,
                        p_qo as *mut u16,
                        p_ko as *mut u16,
                        p_cos as *const f32,
                        p_sin as *const f32,
                        p_pos as *const i32,
                        m,
                        nh,
                        nkv,
                        hd,
                    )
                };
                anyhow::ensure!(rc == 0, "block rope rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::scale_inplace_bf16(cu, p_qo as *mut u16, scale, m * nh * hd)
                };
                anyhow::ensure!(rc == 0, "block q scale rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::kv_append_bf16(
                        cu,
                        p_ko as *const u16,
                        p_vo as *const u16,
                        l.kc as *mut u16,
                        l.vc as *mut u16,
                        p_n as *const i32,
                        m as i32,
                        nkv as i32,
                        hd as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "block kv_append rc={rc}");
                let l_mask = if l.causal { p_mask_causal } else { p_mask };
                let l_positions = if l.window > 0 {
                    p_pos as *const i32
                } else {
                    std::ptr::null()
                };
                let rc = unsafe {
                    nv_kernels::cuda::tree_verify_attn_bf16(
                        cu,
                        p_qo as *const u16,
                        l.kc as *const u16,
                        l.vc as *const u16,
                        p_n as *const i32,
                        l_mask as *const u8,
                        l_positions,
                        p_attn as *mut u16,
                        nh as i32,
                        nkv as i32,
                        hd as i32,
                        m as i32,
                        l.window,
                    )
                };
                anyhow::ensure!(rc == 0, "block attn rc={rc}");
                gemm(&l.wo, p_attn, p_oo, h, nh * hd, m, "block o_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::residual_add_scale_bf16(
                        cu,
                        p_x as *const u16,
                        p_oo as *const u16,
                        p_pa as *mut u16,
                        1.0,
                        m * h,
                    )
                };
                anyhow::ensure!(rc == 0, "block resid add rc={rc}");
                rms(p_pa, l.wpl, p_nm, m, h, "block post_ln")?;
                gemm(&l.wg, p_nm, p_gateo, inter, h, m, "block gate_proj")?;
                gemm(&l.wu, p_nm, p_upo, inter, h, m, "block up_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::silu_mul_bf16(
                        cu,
                        p_gateo as *const u16,
                        p_upo as *const u16,
                        p_act as *mut u16,
                        m * inter,
                    )
                };
                anyhow::ensure!(rc == 0, "block silu_mul rc={rc}");
                gemm(&l.wd, p_act, p_mlpo, h, inter, m, "block down_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::residual_add_scale_bf16(
                        cu,
                        p_pa as *const u16,
                        p_mlpo as *const u16,
                        p_x as *mut u16,
                        1.0,
                        m * h,
                    )
                };
                anyhow::ensure!(rc == 0, "block h add rc={rc}");
            }

            rms(p_x, p_wn, p_normed, m, h, "block final norm")?;
            gemm(
                &p_wlm,
                p_normed + (h as u64) * 2,
                p_logitso,
                dv,
                h,
                md,
                "block lm_head",
            )?;
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    cu,
                    p_logitso as *const u16,
                    p_logits_f32 as *mut f32,
                    (md * dv) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "block logits cast rc={rc}");
            let rc = unsafe {
                nv_kernels::cuda::argmax_f32_rows(
                    cu,
                    p_logits_f32 as *const f32,
                    md as i32,
                    dv as i32,
                    p_aval as *mut f32,
                    p_aidx as *mut i32,
                    p_didx as *mut u32,
                )
            };
            anyhow::ensure!(rc == 0, "block argmax rc={rc}");
            for i in 0..md {
                let rc = unsafe {
                    nv_kernels::cuda::token_map_u32(
                        cu,
                        p_d2t as *const u32,
                        (p_didx + (i as u64) * 4) as *const u32,
                        (p_out + (i as u64) * 4) as *mut u32,
                    )
                };
                anyhow::ensure!(rc == 0, "block d2t map rc={rc}");
            }
            Ok(())
        };

        if eager_body {
            body(&fk)?;
            fk.synchronize().map_err(|e| anyhow!(e))?;
            let out: Vec<u32> = fk
                .clone_dtoh(&g.out_buf)
                .map_err(|e| anyhow!("block out d2h: {e:?}"))?;
            return Ok(out);
        }

        if !g.warmed {
            body(&fk)?;
            fk.synchronize().map_err(|e| anyhow!(e))?;
            g.warmed = true;
        }
        let mut token = 0xdf1a5du64;
        for l in lp.iter() {
            token = token
                .rotate_left(7)
                .wrapping_add(l.kc)
                .rotate_left(9)
                .wrapping_add(l.vc);
        }
        token = token.rotate_left(11).wrapping_add(m as u64);
        let dbg = std::env::var_os("NV_DFLASH_GRAPH_DEBUG").is_some();
        let t_run = std::time::Instant::now();
        g.runner.run(token, body)?;
        let launch_ms = t_run.elapsed().as_secs_f64() * 1000.0;
        fk.synchronize().map_err(|e| anyhow!(e))?;
        let gpu_ms = t_run.elapsed().as_secs_f64() * 1000.0;
        let out: Vec<u32> = fk
            .clone_dtoh(&g.out_buf)
            .map_err(|e| anyhow!("block out d2h: {e:?}"))?;
        if dbg {
            eprintln!(
                "[dflash-graph-dbg] ctx={} launch {launch_ms:.3} ms, launch+exec {gpu_ms:.3} ms, total {:.3} ms",
                ctx.len,
                t_run.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn tiny_cfg(block_size: usize) -> DFlashSpeculatorConfig {
        DFlashSpeculatorConfig {
            hidden_size: 8,
            target_hidden_size: 8,
            draft_vocab_size: 16,
            target_vocab_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            block_size,
            mask_token_id: 3,
            aux_hidden_state_layer_ids: vec![0, 1, 2],
            layer_attn: Vec::new(),
            sliding_window: 0,
            logit_softcap: None,
            tied_embeddings: false,
        }
    }

    fn mk_linear(dev: &Device, out: usize, inp: usize, seed: f32) -> Linear {
        let data: Vec<bf16> = (0..out * inp)
            .map(|i| bf16::from_f32(((i as f32) * 0.0137 + seed).sin() * 0.1))
            .collect();
        let t = Tensor::from_vec(data, (out, inp), dev).unwrap();
        Linear::new(t, None).unwrap()
    }

    fn mk_rms(dev: &Device, dim: usize, eps: f64) -> RmsNorm {
        let data: Vec<bf16> = (0..dim).map(|_| bf16::from_f32(1.0)).collect();
        RmsNorm::new(Tensor::from_vec(data, dim, dev).unwrap(), eps)
    }

    fn tiny_drafter(block_size: usize) -> LoadedDFlashDrafter {
        tiny_drafter_from(tiny_cfg(block_size))
    }

    fn tiny_drafter_from(cfg: DFlashSpeculatorConfig) -> LoadedDFlashDrafter {
        let dev = Device::Cpu;
        let h = cfg.hidden_size;
        let embed_data: Vec<bf16> = (0..cfg.target_vocab_size * h)
            .map(|i| bf16::from_f32(((i as f32) * 0.021).cos() * 0.1))
            .collect();
        let embed = Tensor::from_vec(embed_data, (cfg.target_vocab_size, h), &dev).unwrap();
        let fc = mk_linear(&dev, h, cfg.fc_in_dim(), 0.11);
        let hidden_norm = mk_rms(&dev, h, cfg.rms_norm_eps);
        let layers: Vec<DFlashLayer> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let s = i as f32;
                DFlashLayer::new(
                    mk_rms(&dev, h, cfg.rms_norm_eps),
                    mk_rms(&dev, h, cfg.rms_norm_eps),
                    mk_linear(&dev, cfg.q_out_dim(), h, 0.3 + s),
                    mk_linear(&dev, cfg.kv_out_dim(), h, 0.5 + s),
                    mk_linear(&dev, cfg.kv_out_dim(), h, 0.7 + s),
                    mk_linear(&dev, h, cfg.q_out_dim(), 0.9 + s),
                    mk_rms(&dev, cfg.head_dim, cfg.rms_norm_eps),
                    mk_rms(&dev, cfg.head_dim, cfg.rms_norm_eps),
                    mk_linear(&dev, cfg.intermediate_size, h, 1.1 + s),
                    mk_linear(&dev, cfg.intermediate_size, h, 1.3 + s),
                    mk_linear(&dev, h, cfg.intermediate_size, 1.5 + s),
                )
            })
            .collect();
        let norm = mk_rms(&dev, h, cfg.rms_norm_eps);
        let lm_head = mk_linear(&dev, cfg.draft_vocab_size, h, 1.7);
        let d2t: Vec<u32> = (0..cfg.draft_vocab_size as u32).map(|i| i % 3).collect();
        let t2d: Vec<bool> = (0..cfg.target_vocab_size)
            .map(|i| i < cfg.draft_vocab_size + 2)
            .collect();
        LoadedDFlashDrafter::from_parts(
            cfg,
            dev,
            DType::BF16,
            embed,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
            d2t,
            t2d,
        )
        .unwrap()
    }

    fn tiny_aux(d: &LoadedDFlashDrafter, rows: usize) -> Tensor {
        let n = rows * d.cfg.fc_in_dim();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0093).sin() * 0.4).collect();
        Tensor::from_vec(data, (rows, d.cfg.fc_in_dim()), &d.device).unwrap()
    }

    const REDHAT_CONFIG: &str = r#"{
        "architectures": ["DFlashDraftModel"],
        "aux_hidden_state_layer_ids": [1, 17, 29, 47, 58],
        "block_size": 8,
        "draft_vocab_size": 32000,
        "dtype": "bfloat16",
        "mask_token_id": 4,
        "max_anchors": 3072,
        "speculators_model_type": "dflash",
        "target_hidden_size": null,
        "transformer_layer_config": {
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "max_position_embeddings": 262144,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 16,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "vocab_size": 262144
        }
    }"#;

    #[test]
    fn config_parses_redhat_checkpoint_json() {
        let cfg = DFlashSpeculatorConfig::from_hf_json_str(REDHAT_CONFIG).expect("parse");
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.target_hidden_size, 5376);
        assert_eq!(cfg.num_hidden_layers, 5);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 16);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 21504);
        assert_eq!(cfg.draft_vocab_size, 32000);
        assert_eq!(cfg.target_vocab_size, 262144);
        assert_eq!(cfg.block_size, 8);
        assert_eq!(cfg.mask_token_id, 4);
        assert_eq!(cfg.aux_hidden_state_layer_ids, vec![1, 17, 29, 47, 58]);
        assert_eq!(cfg.fc_in_dim(), 26880);
        assert_eq!(cfg.query_rows(), 9);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn block_input_ids_are_anchor_then_masks() {
        let d = tiny_drafter(4);
        assert_eq!(d.block_input_ids(7), vec![7, 3, 3, 3, 3]);
    }

    #[test]
    fn draft_block_yields_block_size_valid_target_tokens() {
        let d = tiny_drafter(4);
        let aux = tiny_aux(&d, 6);
        let positions: Vec<u32> = (0..6).collect();
        let out = d.draft_block_from_aux(&aux, &positions, 5).expect("draft");
        assert_eq!(out.len(), 4);
        for &t in &out {
            assert!((t as usize) < d.cfg.target_vocab_size);
            assert!(d.t2d_supports(t), "draft image {t} must be t2d-supported");
        }
    }

    #[test]
    fn device_variant_matches_host_variant() {
        let d = tiny_drafter(4);
        let aux = tiny_aux(&d, 5);
        let positions: Vec<u32> = (0..5).collect();
        let proj = d.project_aux(&aux).expect("project");
        let ctx = d.precompute_context_kv(&proj, &positions).expect("ctx kv");
        let host = d.draft_block(&ctx, 2).expect("host draft");
        let dev = d
            .draft_block_dev(&ctx, 2)
            .expect("dev draft")
            .to_vec1::<u32>()
            .expect("to host");
        assert_eq!(host, dev);
    }

    #[test]
    fn draft_is_deterministic_and_anchor_sensitive() {
        let d = tiny_drafter(4);
        let aux = tiny_aux(&d, 5);
        let positions: Vec<u32> = (0..5).collect();
        let proj = d.project_aux(&aux).expect("project");
        let ctx = d.precompute_context_kv(&proj, &positions).expect("ctx kv");
        let a1 = d.draft_block(&ctx, 2).expect("draft");
        let a2 = d.draft_block(&ctx, 2).expect("draft again");
        assert_eq!(a1, a2);
        let h1 = d.forward_block_hidden(&ctx, 2).expect("hidden anchor 2");
        let h2 = d.forward_block_hidden(&ctx, 9).expect("hidden anchor 9");
        let diff = h1
            .sub(&h2)
            .unwrap()
            .abs()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-4,
            "anchor must influence hidden states, diff={diff}"
        );
    }

    #[test]
    fn block_attention_is_non_causal() {
        let wide = tiny_drafter(4);
        let narrow = tiny_drafter(1);
        let aux = tiny_aux(&wide, 5);
        let positions: Vec<u32> = (0..5).collect();
        let proj = wide.project_aux(&aux).expect("project");
        let ctx_w = wide.precompute_context_kv(&proj, &positions).expect("ctx");
        let ctx_n = narrow
            .precompute_context_kv(&proj, &positions)
            .expect("ctx");
        let h_w = wide.forward_block_hidden(&ctx_w, 2).expect("wide");
        let h_n = narrow.forward_block_hidden(&ctx_n, 2).expect("narrow");
        let row_w = h_w.narrow(0, 0, 1).unwrap();
        let row_n = h_n.narrow(0, 0, 1).unwrap();
        let diff = row_w
            .sub(&row_n)
            .unwrap()
            .abs()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-5,
            "anchor row must attend to later mask positions (non-causal), diff={diff}"
        );
    }

    #[test]
    fn context_kv_shapes_and_positions() {
        let d = tiny_drafter(4);
        let aux = tiny_aux(&d, 7);
        let positions: Vec<u32> = (3..10).collect();
        let proj = d.project_aux(&aux).expect("project");
        let ctx = d.precompute_context_kv(&proj, &positions).expect("ctx kv");
        assert_eq!(ctx.len(), 7);
        assert_eq!(ctx.next_pos(), 10);
        assert_eq!(ctx.k.len(), d.cfg.num_hidden_layers);
        assert!(ctx.capacity() >= 7 + d.cfg.query_rows());
        for (k, v) in ctx.k.iter().zip(ctx.v.iter()) {
            assert_eq!(
                k.dims(),
                &[1, ctx.capacity(), d.cfg.num_key_value_heads, d.cfg.head_dim]
            );
            assert_eq!(k.dims(), v.dims());
        }
        let err = d
            .precompute_context_kv(&proj, &positions[..6])
            .expect_err("mismatched positions must fail");
        assert!(err.to_string().contains("positions"), "got: {err}");
    }

    #[test]
    fn incremental_append_matches_one_shot_precompute() {
        let d = tiny_drafter(4);
        let aux = tiny_aux(&d, 9);
        let positions: Vec<u32> = (0..9).collect();
        let proj = d.project_aux(&aux).expect("project");
        let full = d
            .precompute_context_kv(&proj, &positions)
            .expect("full ctx");

        let mut inc = DFlashContextKv::empty();
        for (start, len) in [(0usize, 4usize), (4, 2), (6, 3)] {
            let rows = proj.narrow(0, start, len).unwrap();
            d.append_context_kv(&mut inc, &rows, &positions[start..start + len])
                .expect("append");
        }
        assert_eq!(inc.len(), full.len());
        assert_eq!(inc.next_pos(), full.next_pos());
        let n = inc.len();
        for (a, b) in inc
            .k
            .iter()
            .zip(full.k.iter())
            .chain(inc.v.iter().zip(full.v.iter()))
        {
            let a = a.narrow(1, 0, n).unwrap();
            let b = b.narrow(1, 0, n).unwrap();
            let diff = a
                .sub(&b)
                .unwrap()
                .abs()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(
                diff, 0.0,
                "append must be bit-identical to one-shot precompute"
            );
        }

        let d1 = d.draft_block(&full, 3).expect("draft full");
        let d2 = d.draft_block(&inc, 3).expect("draft inc");
        assert_eq!(d1, d2);

        let bad_rows = proj.narrow(0, 0, 2).unwrap();
        let err = d
            .append_context_kv(&mut inc, &bad_rows, &[11, 12])
            .expect_err("gapped positions must fail");
        assert!(err.to_string().contains("position"), "got: {err}");
    }

    #[test]
    fn capacity_growth_matches_one_shot() {
        let d = tiny_drafter(4);
        let n = 1100usize;
        let aux = tiny_aux(&d, n);
        let positions: Vec<u32> = (0..n as u32).collect();
        let proj = d.project_aux(&aux).expect("project");
        let full = d.precompute_context_kv(&proj, &positions).expect("full");

        let mut inc = DFlashContextKv::empty();
        for start in (0..n).step_by(150) {
            let len = 150.min(n - start);
            let rows = proj.narrow(0, start, len).unwrap();
            d.append_context_kv(&mut inc, &rows, &positions[start..start + len])
                .expect("append");
        }
        assert_eq!(inc.len(), full.len());
        assert!(inc.capacity() > 1024, "growth path must have triggered");
        let d1 = d.draft_block(&full, 3).expect("draft full");
        let d2 = d.draft_block(&inc, 3).expect("draft inc");
        assert_eq!(d1, d2);
        let f1 = full.fingerprint().expect("fp full");
        let f2 = inc.fingerprint().expect("fp inc");
        assert!((f1 - f2).abs() < 1e-3 * f1.abs().max(1.0), "{f1} vs {f2}");
    }

    #[test]
    fn rejects_wrong_aux_width_and_layer_count() {
        let d = tiny_drafter(4);
        let bad = Tensor::zeros((3usize, 5usize), DType::F32, &d.device).unwrap();
        let err = d.project_aux(&bad).expect_err("wrong fc width");
        assert!(err.to_string().contains("expected"), "got: {err}");
        let aux = tiny_aux(&d, 4);
        let proj = d.project_aux(&aux).unwrap();
        let positions: Vec<u32> = (0..4).collect();
        let mut ctx = d.precompute_context_kv(&proj, &positions).unwrap();
        ctx.k.pop();
        ctx.v.pop();
        let err = d
            .forward_block_hidden(&ctx, 1)
            .expect_err("layer count mismatch must fail");
        assert!(err.to_string().contains("layers"), "got: {err}");
    }

    fn bf_tensor(n: usize, shape: Vec<usize>, seed: f32, dev: &Device) -> Tensor {
        let data: Vec<bf16> = (0..n)
            .map(|i| bf16::from_f32(((i as f32) * 0.017 + seed).sin() * 0.1))
            .collect();
        Tensor::from_vec(data, shape, dev).unwrap()
    }

    fn synth_body_tensors(
        cfg: &DFlashSpeculatorConfig,
        dev: &Device,
    ) -> std::collections::HashMap<String, Tensor> {
        let h = cfg.hidden_size;
        let bf = |n: usize, shape: Vec<usize>, seed: f32| bf_tensor(n, shape, seed, dev);
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "fc.weight".to_string(),
            bf(h * cfg.fc_in_dim(), vec![h, cfg.fc_in_dim()], 0.2),
        );
        tensors.insert("hidden_norm.weight".to_string(), bf(h, vec![h], 0.3));
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            let s = i as f32;
            tensors.insert(
                format!("{p}.input_layernorm.weight"),
                bf(h, vec![h], 0.4 + s),
            );
            tensors.insert(
                format!("{p}.post_attention_layernorm.weight"),
                bf(h, vec![h], 0.5 + s),
            );
            tensors.insert(
                format!("{p}.self_attn.q_proj.weight"),
                bf(cfg.q_out_dim() * h, vec![cfg.q_out_dim(), h], 0.6 + s),
            );
            tensors.insert(
                format!("{p}.self_attn.k_proj.weight"),
                bf(cfg.kv_out_dim() * h, vec![cfg.kv_out_dim(), h], 0.7 + s),
            );
            tensors.insert(
                format!("{p}.self_attn.v_proj.weight"),
                bf(cfg.kv_out_dim() * h, vec![cfg.kv_out_dim(), h], 0.8 + s),
            );
            tensors.insert(
                format!("{p}.self_attn.o_proj.weight"),
                bf(h * cfg.q_out_dim(), vec![h, cfg.q_out_dim()], 0.9 + s),
            );
            tensors.insert(
                format!("{p}.self_attn.q_norm.weight"),
                bf(cfg.head_dim, vec![cfg.head_dim], 1.0 + s),
            );
            tensors.insert(
                format!("{p}.self_attn.k_norm.weight"),
                bf(cfg.head_dim, vec![cfg.head_dim], 1.1 + s),
            );
            tensors.insert(
                format!("{p}.mlp.gate_proj.weight"),
                bf(
                    cfg.intermediate_size * h,
                    vec![cfg.intermediate_size, h],
                    1.2 + s,
                ),
            );
            tensors.insert(
                format!("{p}.mlp.up_proj.weight"),
                bf(
                    cfg.intermediate_size * h,
                    vec![cfg.intermediate_size, h],
                    1.3 + s,
                ),
            );
            tensors.insert(
                format!("{p}.mlp.down_proj.weight"),
                bf(
                    h * cfg.intermediate_size,
                    vec![h, cfg.intermediate_size],
                    1.4 + s,
                ),
            );
        }
        tensors.insert("norm.weight".to_string(), bf(h, vec![h], 1.5));
        tensors
    }

    fn save_tmp_safetensors(
        tensors: &std::collections::HashMap<String, Tensor>,
        tag: &str,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let st = dir.join("model.safetensors");
        candle_core::safetensors::save(tensors, &st).unwrap();
        (dir, st)
    }

    #[test]
    fn loader_accepts_redhat_tensor_schema() {
        let cfg = tiny_cfg(4);
        let dev = Device::Cpu;
        let h = cfg.hidden_size;
        let mut tensors = synth_body_tensors(&cfg, &dev);
        tensors.insert(
            "embed_tokens.weight".to_string(),
            bf_tensor(
                cfg.target_vocab_size * h,
                vec![cfg.target_vocab_size, h],
                0.1,
                &dev,
            ),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            bf_tensor(
                cfg.draft_vocab_size * h,
                vec![cfg.draft_vocab_size, h],
                1.6,
                &dev,
            ),
        );
        let d2t: Vec<i64> = (0..cfg.draft_vocab_size as i64).map(|i| i % 3).collect();
        tensors.insert(
            "d2t".to_string(),
            Tensor::from_vec(d2t, cfg.draft_vocab_size, &dev).unwrap(),
        );
        let t2d: Vec<u8> = (0..cfg.target_vocab_size)
            .map(|i| u8::from(i < cfg.draft_vocab_size + 2))
            .collect();
        tensors.insert(
            "t2d".to_string(),
            Tensor::from_vec(t2d, cfg.target_vocab_size, &dev).unwrap(),
        );

        let (dir, st) = save_tmp_safetensors(&tensors, "nv-dflash-loader-test");

        let loaded = LoadedDFlashDrafter::load_from_safetensors(&cfg, &st, &dev).expect("load");
        assert_eq!(loaded.config().num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(loaded.d2t().len(), cfg.draft_vocab_size);
        assert_eq!(loaded.t2d().len(), cfg.target_vocab_size);
        assert_eq!(loaded.d2t_map(4), 5);

        let aux = tiny_aux(&loaded, 3);
        let positions: Vec<u32> = (0..3).collect();
        let out = loaded
            .draft_block_from_aux(&aux, &positions, 1)
            .expect("draft from loaded weights");
        assert_eq!(out.len(), cfg.block_size);

        let _ = std::fs::remove_dir_all(&dir);
    }

    const ZLAB_CONFIG: &str = r#"{
        "architectures": ["DFlashDraftModel"],
        "attention_bias": false,
        "block_size": 16,
        "dflash_config": {
            "mask_token_id": 4,
            "target_layer_ids": [1, 12, 23, 35, 46, 57]
        },
        "dtype": "bfloat16",
        "final_logit_softcapping": 30.0,
        "head_dim": 128,
        "hidden_size": 5376,
        "intermediate_size": 10752,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ],
        "max_position_embeddings": 262144,
        "model_type": "qwen3",
        "num_attention_heads": 64,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 60,
        "rms_norm_eps": 1e-06,
        "sliding_window": 2048,
        "tie_word_embeddings": true,
        "use_sliding_window": true,
        "vocab_size": 262144,
        "rope_theta": 1000000,
        "rope_scaling": null
    }"#;

    #[test]
    fn config_parses_zlab_flat_checkpoint_json() {
        let cfg = DFlashSpeculatorConfig::from_hf_json_str(ZLAB_CONFIG).expect("parse");
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.num_hidden_layers, 5);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 10752);
        assert_eq!(cfg.draft_vocab_size, 262144);
        assert_eq!(cfg.target_vocab_size, 262144);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.query_rows(), 17);
        assert_eq!(cfg.mask_token_id, 4);
        assert_eq!(cfg.aux_hidden_state_layer_ids, vec![1, 12, 23, 35, 46, 57]);
        assert_eq!(cfg.fc_in_dim(), 32256);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
        assert_eq!(cfg.sliding_window, 2048);
        assert_eq!(cfg.logit_softcap, Some(30.0));
        assert!(cfg.tied_embeddings);
        assert_eq!(cfg.layer_attn.len(), 5);
        for i in 0..4 {
            assert_eq!(
                cfg.layer_attn[i],
                DFlashLayerAttn {
                    sliding: true,
                    causal: true
                },
                "layer {i} must be sliding+causal"
            );
        }
        assert_eq!(cfg.layer_attn[4], DFlashLayerAttn::FULL_NON_CAUSAL);
        assert!(cfg.any_masked_layer());
    }

    #[test]
    fn config_redhat_speculators_schema_stays_full_non_causal() {
        let cfg = DFlashSpeculatorConfig::from_hf_json_str(REDHAT_CONFIG).expect("parse");
        assert!(cfg.layer_attn.is_empty());
        assert!(!cfg.any_masked_layer());
        assert_eq!(cfg.layer_attn_for(0), DFlashLayerAttn::FULL_NON_CAUSAL);
        assert!(!cfg.tied_embeddings);
        assert_eq!(cfg.logit_softcap, None);
    }

    #[test]
    fn config_rejects_oversized_block_size() {
        let mut v: serde_json::Value = serde_json::from_str(ZLAB_CONFIG).unwrap();
        v["block_size"] = serde_json::json!(100000);
        let err = DFlashSpeculatorConfig::from_hf_json_str(&v.to_string()).unwrap_err();
        assert!(err.to_string().contains("block_size"), "{err}");

        let mut v: serde_json::Value = serde_json::from_str(REDHAT_CONFIG).unwrap();
        v["block_size"] = serde_json::json!(100000);
        let err = DFlashSpeculatorConfig::from_hf_json_str(&v.to_string()).unwrap_err();
        assert!(err.to_string().contains("block_size"), "{err}");
    }

    #[test]
    fn config_rejects_zero_and_oversized_dims() {
        for (key, bad) in [
            ("block_size", 0u64),
            ("hidden_size", 0),
            ("hidden_size", 1 << 20),
            ("intermediate_size", 1 << 24),
            ("head_dim", 65536),
            ("num_attention_heads", 65536),
            ("num_hidden_layers", 65536),
            ("vocab_size", 1 << 30),
        ] {
            let mut v: serde_json::Value = serde_json::from_str(ZLAB_CONFIG).unwrap();
            v[key] = serde_json::json!(bad);
            if key == "num_hidden_layers" {
                v.as_object_mut().unwrap().remove("layer_types");
            }
            let err = DFlashSpeculatorConfig::from_hf_json_str(&v.to_string())
                .expect_err(&format!("{key}={bad} must be rejected"));
            assert!(
                err.to_string().contains("sanity bounds"),
                "{key}={bad}: {err}"
            );
        }
    }

    #[test]
    fn config_within_bounds_still_parses() {
        let mut v: serde_json::Value = serde_json::from_str(ZLAB_CONFIG).unwrap();
        v["block_size"] = serde_json::json!(200);
        let cfg = DFlashSpeculatorConfig::from_hf_json_str(&v.to_string()).expect("parse");
        assert_eq!(cfg.query_rows(), 201);
        assert!(cfg.query_rows() > FP4_STAGING_ROWS);
    }

    #[test]
    fn draft_f32_env_zero_means_off() {
        assert!(!draft_f32_enabled(None));
        assert!(!draft_f32_enabled(Some("0")));
        assert!(draft_f32_enabled(Some("1")));
        assert!(draft_f32_enabled(Some("")));
        assert!(draft_f32_enabled(Some("true")));
    }

    #[test]
    fn config_dflash_causal_override_applies_to_all_layers() {
        let mut v: serde_json::Value = serde_json::from_str(ZLAB_CONFIG).unwrap();
        v["dflash_config"]["causal"] = serde_json::Value::Bool(false);
        let cfg = DFlashSpeculatorConfig::from_hf_json_str(&v.to_string()).expect("parse");
        assert!(cfg.layer_attn.iter().all(|a| !a.causal));
        assert!(cfg.layer_attn[0].sliding && !cfg.layer_attn[4].sliding);
    }

    fn causal_sliding_cfg(block_size: usize, window: usize) -> DFlashSpeculatorConfig {
        let mut cfg = tiny_cfg(block_size);
        cfg.layer_attn = vec![
            DFlashLayerAttn {
                sliding: true,
                causal: true
            };
            cfg.num_hidden_layers
        ];
        cfg.sliding_window = window;
        cfg
    }

    #[test]
    fn causal_layers_hide_future_mask_rows_from_anchor() {
        let wide = tiny_drafter_from(causal_sliding_cfg(4, 4096));
        let narrow = tiny_drafter_from(causal_sliding_cfg(1, 4096));
        let aux = tiny_aux(&wide, 5);
        let positions: Vec<u32> = (0..5).collect();
        let proj = wide.project_aux(&aux).expect("project");
        let ctx_w = wide.precompute_context_kv(&proj, &positions).expect("ctx");
        let ctx_n = narrow
            .precompute_context_kv(&proj, &positions)
            .expect("ctx");
        let h_w = wide.forward_block_hidden(&ctx_w, 2).expect("wide");
        let h_n = narrow.forward_block_hidden(&ctx_n, 2).expect("narrow");
        let diff = h_w
            .narrow(0, 0, 1)
            .unwrap()
            .sub(&h_n.narrow(0, 0, 1).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "causal anchor row must not see later mask rows, diff={diff}"
        );
    }

    #[test]
    fn sliding_window_hides_distant_context() {
        let d = tiny_drafter_from(causal_sliding_cfg(4, 4));
        let n_ctx = 6usize;
        let aux_a = tiny_aux(&d, n_ctx);
        let host: Vec<f32> = aux_a
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i < 2 * d.cfg.fc_in_dim() {
                    x + 5.0
                } else {
                    x
                }
            })
            .collect();
        let aux_b = Tensor::from_vec(host, (n_ctx, d.cfg.fc_in_dim()), &d.device).unwrap();
        let positions: Vec<u32> = (0..n_ctx as u32).collect();
        let run = |aux: &Tensor| -> Vec<f32> {
            let proj = d.project_aux(aux).expect("project");
            let ctx = d.precompute_context_kv(&proj, &positions).expect("ctx");
            d.forward_block_hidden(&ctx, 2)
                .expect("hidden")
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let h_a = run(&aux_a);
        let h_b = run(&aux_b);
        let max_diff = h_a
            .iter()
            .zip(h_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "context rows 0-1 are outside window 4 for all query rows (positions >= 6); \
             perturbing them must not change the block, diff={max_diff}"
        );

        let full = tiny_drafter_from(tiny_cfg(4));
        let run_full = |aux: &Tensor| -> Vec<f32> {
            let proj = full.project_aux(aux).expect("project");
            let ctx = full.precompute_context_kv(&proj, &positions).expect("ctx");
            full.forward_block_hidden(&ctx, 2)
                .expect("hidden")
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let f_a = run_full(&aux_a);
        let f_b = run_full(&aux_b);
        let full_diff = f_a
            .iter()
            .zip(f_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            full_diff > 1e-4,
            "full-attention control must see the perturbation, diff={full_diff}"
        );
    }

    #[test]
    fn tied_loader_reuses_target_embed_and_identity_token_maps() {
        let mut cfg = tiny_cfg(4);
        cfg.draft_vocab_size = cfg.target_vocab_size;
        cfg.tied_embeddings = true;
        let dev = Device::Cpu;
        let h = cfg.hidden_size;
        let tensors = synth_body_tensors(&cfg, &dev);
        let (dir, st) = save_tmp_safetensors(&tensors, "nv-dflash-tied-test");

        let err = match LoadedDFlashDrafter::load_from_safetensors(&cfg, &st, &dev) {
            Ok(_) => panic!("tied load without target embed must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("target"), "got: {err}");

        let target_embed = bf_tensor(
            cfg.target_vocab_size * h,
            vec![cfg.target_vocab_size, h],
            0.05,
            &dev,
        );
        let loaded = LoadedDFlashDrafter::load_from_safetensors_with_embed(
            &cfg,
            &st,
            &dev,
            Some(&target_embed),
        )
        .expect("tied load");
        for t in 0..cfg.draft_vocab_size as u32 {
            assert_eq!(loaded.d2t_map(t), t, "d2t must be identity");
        }
        assert!(loaded.t2d().iter().all(|&b| b), "t2d must be all-true");
        let aux = tiny_aux(&loaded, 3);
        let positions: Vec<u32> = (0..3).collect();
        let out = loaded
            .draft_block_from_aux(&aux, &positions, 1)
            .expect("draft from tied weights");
        assert_eq!(out.len(), cfg.block_size);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
