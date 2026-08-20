use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
use std::path::Path;

use crate::util::{load_linear, load_rmsnorm, load_tensor};
use crate::DraftScorer;

#[derive(Clone, Debug)]
pub struct Eagle3SpeculatorConfig {
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
    pub eagle_aux_hidden_state_layer_ids: Vec<usize>,
}

impl Default for Eagle3SpeculatorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 5376,
            draft_vocab_size: 32000,
            target_vocab_size: 262144,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 256,
            intermediate_size: 21504,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            norm_before_residual: true,
            norm_before_fc: false,
            eagle_aux_hidden_state_layer_ids: vec![2, 30, 57],
        }
    }
}

pub(crate) struct TlcCommon {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub target_vocab_size: usize,
    pub num_hidden_layers: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub has_rope_parameters: bool,
    pub rope_theta_params: Option<f32>,
    pub rope_theta_flat: Option<f32>,
    pub draft_vocab_size: Option<usize>,
}

pub(crate) fn parse_speculators_tlc(v: &serde_json::Value) -> Result<TlcCommon> {
    let tlc = v
        .get("transformer_layer_config")
        .ok_or_else(|| anyhow!("missing transformer_layer_config"))?;
    let usize_field = |obj: &serde_json::Value, key: &str| -> Result<usize> {
        obj.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| anyhow!("missing {key}"))
    };
    Ok(TlcCommon {
        hidden_size: usize_field(tlc, "hidden_size")?,
        num_attention_heads: usize_field(tlc, "num_attention_heads")?,
        num_key_value_heads: usize_field(tlc, "num_key_value_heads")?,
        head_dim: usize_field(tlc, "head_dim")?,
        intermediate_size: usize_field(tlc, "intermediate_size")?,
        max_position_embeddings: usize_field(tlc, "max_position_embeddings")?,
        target_vocab_size: usize_field(tlc, "vocab_size")?,
        num_hidden_layers: tlc
            .get("num_hidden_layers")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
        rms_norm_eps: tlc.get("rms_norm_eps").and_then(|x| x.as_f64()),
        has_rope_parameters: tlc.get("rope_parameters").is_some(),
        rope_theta_params: tlc
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .map(|t| t as f32),
        rope_theta_flat: tlc
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .map(|t| t as f32),
        draft_vocab_size: v
            .get("draft_vocab_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
    })
}

impl Eagle3SpeculatorConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse eagle3 config json")?;
        let c = parse_speculators_tlc(&v)?;
        let mut cfg = Self {
            hidden_size: c.hidden_size,
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
        if let Some(theta) = c.rope_theta_params {
            cfg.rope_theta = theta;
        }
        if let Some(d) = c.draft_vocab_size {
            cfg.draft_vocab_size = d;
        }
        if let Some(b) = v.get("norm_before_residual").and_then(|x| x.as_bool()) {
            cfg.norm_before_residual = b;
        }
        if let Some(b) = v.get("norm_before_fc").and_then(|x| x.as_bool()) {
            cfg.norm_before_fc = b;
        }
        if let Some(arr) = v
            .get("eagle_aux_hidden_state_layer_ids")
            .and_then(|x| x.as_array())
        {
            cfg.eagle_aux_hidden_state_layer_ids = arr
                .iter()
                .filter_map(|x| x.as_u64().map(|v| v as usize))
                .collect();
        }
        Ok(cfg)
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn fc_in_dim(&self) -> usize {
        3 * self.hidden_size
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
}

pub const DRAFTER_ENCODE_CHUNK: usize = 1024;
pub const EAGLE3_ENCODE_ATTN_SCORE_F32_BYTES_CAP_BOUNDS_PREENCODE_PEAK: usize = 512 * 1024 * 1024;
pub const DRAFTER_DEVICE_KV_MAX_TAIL: usize = 32;

fn encode_attn_score_cap_bytes() -> usize {
    env_usize("NV_EAGLE3_ENCODE_ATTN_SCORE_CAP")
        .filter(|&c| c > 0)
        .unwrap_or(EAGLE3_ENCODE_ATTN_SCORE_F32_BYTES_CAP_BOUNDS_PREENCODE_PEAK)
}

pub fn encode_attn_query_block(nh: usize, m: usize, sk: usize, cap_bytes: usize) -> usize {
    let per_query = nh.saturating_mul(sk).saturating_mul(4);
    if per_query == 0 {
        return m.max(1);
    }
    (cap_bytes / per_query).clamp(1, m.max(1))
}
pub const DRAFTER_DEVICE_KV_ALIGN: usize = 4096;
pub const DRAFTER_KV_CAP_SLACK: usize = 256;
pub const DRAFTER_KV_CAP_DEFAULT_SINK: usize = 16;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

#[derive(Default)]
pub struct DrafterKvCache {
    k: Option<Tensor>,

    v: Option<Tensor>,

    last_h: Option<Tensor>,
    len: usize,
    cap_sink: usize,
    cap_window: usize,
    evicted: usize,
    compactions: u64,
    #[cfg(feature = "cuda")]
    dev_k: Option<Tensor>,
    #[cfg(feature = "cuda")]
    dev_v: Option<Tensor>,
    #[cfg(feature = "cuda")]
    dev_cap: usize,
    #[cfg(feature = "cuda")]
    dev_align: usize,
    #[cfg(feature = "cuda")]
    dev_n: Option<cudarc::driver::CudaSlice<i32>>,
    #[cfg(feature = "cuda")]
    dev_masks: std::collections::HashMap<usize, cudarc::driver::CudaSlice<u8>>,
    #[cfg(feature = "cuda")]
    dev_off: bool,
    #[cfg(feature = "cuda")]
    dev_warned: bool,
}

impl DrafterKvCache {
    pub fn new() -> Self {
        let mut c = Self::default();
        if let Some(w) = env_usize("NV_DRAFTER_KV_WINDOW") {
            let sink = env_usize("NV_DRAFTER_KV_SINK").unwrap_or(DRAFTER_KV_CAP_DEFAULT_SINK);
            c.set_kv_cap(sink, w);
        }
        c
    }

    pub fn with_kv_cap(sink: usize, window: usize) -> Self {
        let mut c = Self::default();
        c.set_kv_cap(sink, window);
        c
    }

    pub fn set_kv_cap(&mut self, sink: usize, window: usize) {
        self.cap_sink = sink;
        self.cap_window = window;
    }

    pub fn kv_cap(&self) -> Option<(usize, usize)> {
        if self.cap_window > 0 {
            Some((self.cap_sink, self.cap_window))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn phys_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[1]).unwrap_or(0)
    }

    pub fn evicted(&self) -> usize {
        self.evicted
    }

    pub fn compactions(&self) -> u64 {
        self.compactions
    }

    fn maybe_compact(&mut self) -> Result<()> {
        if self.cap_window == 0 {
            return Ok(());
        }
        let Some(k) = self.k.as_ref() else {
            return Ok(());
        };
        let phys = k.dims()[1];
        let keep = self.cap_sink + self.cap_window;
        if phys <= keep + DRAFTER_KV_CAP_SLACK {
            return Ok(());
        }
        let v = self
            .v
            .as_ref()
            .ok_or_else(|| anyhow!("DrafterKvCache.maybe_compact: k set but v missing"))?;
        let win_from = phys - self.cap_window;
        let (new_k, new_v) = if self.cap_sink == 0 {
            (
                k.narrow(1, win_from, self.cap_window)?.contiguous()?,
                v.narrow(1, win_from, self.cap_window)?.contiguous()?,
            )
        } else {
            (
                Tensor::cat(
                    &[
                        k.narrow(1, 0, self.cap_sink)?,
                        k.narrow(1, win_from, self.cap_window)?,
                    ],
                    1,
                )?
                .contiguous()?,
                Tensor::cat(
                    &[
                        v.narrow(1, 0, self.cap_sink)?,
                        v.narrow(1, win_from, self.cap_window)?,
                    ],
                    1,
                )?
                .contiguous()?,
            )
        };
        self.k = Some(new_k);
        self.v = Some(new_v);
        self.evicted += phys - keep;
        self.compactions += 1;
        #[cfg(feature = "cuda")]
        self.clear_device_kv();
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn disable_device_kv(&mut self) {
        self.dev_off = true;
    }

    #[cfg(feature = "cuda")]
    pub fn device_kv_armed(&self) -> bool {
        self.dev_k.is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn set_device_kv_align(&mut self, align: usize) {
        self.dev_align = align;
    }

    #[cfg(feature = "cuda")]
    fn device_kv_align(&self) -> usize {
        if self.dev_align == 0 {
            DRAFTER_DEVICE_KV_ALIGN
        } else {
            self.dev_align
        }
    }

    #[cfg(feature = "cuda")]
    fn clear_device_kv(&mut self) {
        self.dev_k = None;
        self.dev_v = None;
        self.dev_cap = 0;
    }
}

pub struct LoadedEagle3Scorer {
    cfg: Eagle3SpeculatorConfig,
    device: Device,
    dtype: DType,

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

    d2t: Vec<u32>,
    t2d: Vec<bool>,

    d2t_map_dev: std::sync::OnceLock<Tensor>,
}

impl LoadedEagle3Scorer {
    pub fn config(&self) -> &Eagle3SpeculatorConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn d2t_map(&self, draft_token: u32) -> u32 {
        d2t_apply(&self.d2t, draft_token)
    }

    pub fn d2t_offset(&self, draft_token: u32) -> u32 {
        let i = draft_token as usize;
        if i < self.d2t.len() {
            self.d2t[i]
        } else {
            0
        }
    }

    pub fn d2t(&self) -> &[u32] {
        &self.d2t
    }

    pub fn t2d_supports(&self, target_token: u32) -> bool {
        let i = target_token as usize;
        i < self.t2d.len() && self.t2d[i]
    }

    pub fn t2d(&self) -> &[bool] {
        &self.t2d
    }

    pub fn share_embed_tokens_with_target(&mut self, target_embed: &Tensor) -> Result<bool> {
        if target_embed.dims() != self.embed_tokens.dims()
            || target_embed.dtype() != self.embed_tokens.dtype()
            || !target_embed
                .device()
                .same_device(self.embed_tokens.device())
        {
            return Ok(false);
        }
        let rows = self.embed_tokens.dims()[0];
        let cols = self.embed_tokens.dims()[1];

        let chunk = (16_000_000 / cols.max(1)).max(1);
        let mut r = 0usize;
        while r < rows {
            let n = chunk.min(rows - r);
            let eq_count = self
                .embed_tokens
                .narrow(0, r, n)?
                .eq(&target_embed.narrow(0, r, n)?)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            if eq_count as usize != n * cols {
                return Ok(false);
            }
            r += n;
        }
        self.embed_tokens = target_embed.clone();
        Ok(true)
    }

    pub fn try_load(model_dir: &Path, device: &Device) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let cfg = if cfg_path.is_file() {
            Eagle3SpeculatorConfig::from_hf_json_file(&cfg_path)?
        } else {
            Eagle3SpeculatorConfig::default()
        };
        let st_path = model_dir.join("model.safetensors");
        if !st_path.is_file() {
            bail!("missing model.safetensors at {}", st_path.display());
        }
        Self::load_from_safetensors(&cfg, &st_path, device)
    }

    pub fn load_from_safetensors(
        cfg: &Eagle3SpeculatorConfig,
        safetensors_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        let weights = WeightLoader::open_file(safetensors_path, device)
            .with_context(|| format!("open {}", safetensors_path.display()))?;

        let dtype = if std::env::var("NV_EAGLE_DRAFT_F32").is_ok() {
            DType::F32
        } else {
            DType::BF16
        };

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

        let d2t = load_d2t(&weights, "d2t", cfg.draft_vocab_size)?;
        let t2d = load_t2d(&weights, "t2d", cfg.target_vocab_size)?;
        validate_d2t(&d2t, cfg.draft_vocab_size, cfg.target_vocab_size)
            .with_context(|| format!("speculator {}", safetensors_path.display()))?;
        validate_t2d(&t2d, cfg.target_vocab_size)
            .with_context(|| format!("speculator {}", safetensors_path.display()))?;

        Ok(Self {
            cfg: cfg.clone(),
            device: device.clone(),
            dtype,
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
            d2t,
            t2d,
            d2t_map_dev: std::sync::OnceLock::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        cfg: Eagle3SpeculatorConfig,
        device: Device,
        dtype: DType,
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
        d2t: Vec<u32>,
        t2d: Vec<bool>,
    ) -> Result<Self> {
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
            d2t,
            t2d,
            d2t_map_dev: std::sync::OnceLock::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn block_step(
        &self,
        block_in: &Tensor,
        h_cond_for_residual: &Tensor,
        pos: usize,
        k_ctx: &Tensor,
        v_ctx: &Tensor,
        new_k: &mut Vec<Tensor>,
        new_v: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let residual = if self.cfg.norm_before_residual {
            self.hidden_norm.forward(h_cond_for_residual)?
        } else {
            h_cond_for_residual.clone()
        };
        let q = self
            .q_proj
            .forward(block_in)?
            .reshape((1usize, 1usize, nh, hd))?;
        let kk = self
            .k_proj
            .forward(block_in)?
            .reshape((1usize, 1usize, nkv, hd))?;
        let vv = self
            .v_proj
            .forward(block_in)?
            .reshape((1usize, 1usize, nkv, hd))?;
        let pos_t = Tensor::from_vec(vec![pos as u32], (1usize, 1usize), &self.device)?;
        let (q_rot, k_rot) = self.rope.apply(&q, &kk, &pos_t)?;
        new_k.push(k_rot);
        new_v.push(vv);
        let mut kparts: Vec<&Tensor> = Vec::with_capacity(1 + new_k.len());
        kparts.push(k_ctx);
        for t in new_k.iter() {
            kparts.push(t);
        }
        let mut vparts: Vec<&Tensor> = Vec::with_capacity(1 + new_v.len());
        vparts.push(v_ctx);
        for t in new_v.iter() {
            vparts.push(t);
        }
        let k_all = Tensor::cat(&kparts[..], 1)?;
        let v_all = Tensor::cat(&vparts[..], 1)?;
        #[cfg(feature = "cuda")]
        let attn_out = match self.single_query_attn_dev(&q_rot, &k_all, &v_all)? {
            Some(out) => out,
            None => self.single_query_attn_sdpa(&q_rot, &k_all, &v_all)?,
        };
        #[cfg(not(feature = "cuda"))]
        let attn_out = self.single_query_attn_sdpa(&q_rot, &k_all, &v_all)?;
        if std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_DEBUG").is_some()
            && std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_EAGER").is_some()
        {
            let v: Vec<f32> = attn_out
                .to_dtype(DType::F32)
                .and_then(|x| x.flatten_all())
                .and_then(|x| x.to_vec1())
                .unwrap_or_default();
            let head: Vec<f32> = v.iter().take(4).copied().collect();
            eprintln!(
                "[chain-e] attn (pos {pos}, kv {}): {head:?}",
                k_all.dims()[1]
            );
        }
        self.block_tail_from_attn(&attn_out, &residual)
    }

    fn single_query_attn_sdpa(
        &self,
        q_rot: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
    ) -> Result<Tensor> {
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0f32 / (hd as f32).sqrt(),
            causal: false,
        };
        let attn_out = sdpa(q_rot, k_all, v_all, &attn_cfg)?;
        attn_out
            .squeeze(0)?
            .reshape((1usize, nh * hd))?
            .to_dtype(self.dtype)
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    fn single_query_attn_dev(
        &self,
        q_rot: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::DevicePtrMut;
        if std::env::var_os("NV_EAGLE3_NO_DEVICE_CHAIN").is_some() {
            return Ok(None);
        }
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        if self.dtype != DType::BF16 {
            return Ok(None);
        }
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        if hd > 512 || nkv == 0 || nh % nkv != 0 {
            return Ok(None);
        }
        let n = k_all.dims()[1];
        let scale = 1.0f64 / (hd as f64).sqrt();
        let q_scaled = (q_rot.clone() * scale)?
            .reshape((1usize, nh * hd))?
            .contiguous()?;
        let k_c = k_all.contiguous()?;
        let v_c = v_all.contiguous()?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let n_dev = stream
            .clone_htod(&[n as i32])
            .map_err(|e| anyhow!("single_query_attn n htod: {e:?}"))?;
        let mask = stream
            .alloc_zeros::<u8>(1)
            .map_err(|e| anyhow!("single_query_attn mask alloc: {e:?}"))?;
        let mut out_dev = stream
            .alloc_zeros::<half::bf16>(nh * hd)
            .map_err(|e| anyhow!("single_query_attn out alloc: {e:?}"))?;
        let rc = {
            let p_q = chain_raw_bf16(&q_scaled, &stream)?;
            let p_k = chain_raw_bf16(&k_c, &stream)?;
            let p_v = chain_raw_bf16(&v_c, &stream)?;
            let p_n = chain_slice_ptr(&n_dev, &stream);
            let p_mask = chain_slice_ptr(&mask, &stream);
            let (p_out, _g) = out_dev.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::tree_verify_attn_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    p_q as *const u16,
                    p_k as *const u16,
                    p_v as *const u16,
                    p_n as *const i32,
                    p_mask as *const u8,
                    std::ptr::null(),
                    p_out as *mut u16,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    1,
                    0,
                )
            }
        };
        if rc != 0 {
            eprintln!(
                "[eagle3] tree_verify_attn_bf16 rc={rc} in single_query_attn_dev; \
                 falling back to sdpa, whose summation order can diverge from the \
                 graphed arm by 1 ulp on tied argmaxes"
            );
            return Ok(None);
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev);
        let attn_out = Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, nh * hd),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(attn_out))
    }

    fn draft_from_block(&self, block_out: &Tensor) -> Result<u32> {
        let normed = self.norm.forward(block_out)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        let idx = logits
            .reshape((1usize, self.cfg.draft_vocab_size))?
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        Ok(idx[0])
    }

    fn d2t_map_dev_tensor(&self) -> Result<Tensor> {
        d2t_dev_tensor(
            &self.d2t_map_dev,
            &self.d2t,
            self.cfg.draft_vocab_size,
            &self.device,
        )
    }

    fn draft_next_dev(&self, block_out: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(block_out)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        let idx = logits
            .reshape((1usize, self.cfg.draft_vocab_size))?
            .argmax(D::Minus1)?;
        Ok(self.d2t_map_dev_tensor()?.index_select(&idx, 0)?)
    }

    fn step_block_in_dev(&self, token_dev: &Tensor, h_cond: &Tensor) -> Result<Tensor> {
        let emb = self.embed_tokens.index_select(token_dev, 0)?;
        let tn = self.input_layernorm.forward(&emb)?;
        let an = self.hidden_norm.forward(h_cond)?;
        Tensor::cat(&[&tn, &an], 1).map_err(Into::into)
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

    pub fn chain_draft(&self, context: &[u32], aux_full: &Tensor, k: usize) -> Result<Vec<u32>> {
        let ctx_len = context.len();
        if aux_full.dims() != [ctx_len, self.cfg.fc_in_dim()] {
            bail!(
                "chain_draft: aux must be [{ctx_len}, {}], got {:?}",
                self.cfg.fc_in_dim(),
                aux_full.dims()
            );
        }
        self.chain_draft_projected(context, &self.project_aux(aux_full)?, k)
    }

    pub fn chain_draft_projected(
        &self,
        context: &[u32],
        aux_proj: &Tensor,
        k: usize,
    ) -> Result<Vec<u32>> {
        let ctx_len = context.len();
        if ctx_len == 0 {
            bail!("chain_draft: empty context");
        }
        if aux_proj.dims() != [ctx_len, self.cfg.hidden_size] {
            bail!(
                "chain_draft: projected aux must be [{ctx_len}, {}], got {:?}",
                self.cfg.hidden_size,
                aux_proj.dims()
            );
        }
        let ids = Tensor::from_vec(context.to_vec(), ctx_len, &self.device)?;
        let token_embed = self.embed_tokens.index_select(&ids, 0)?;
        let token_normed = self.input_layernorm.forward(&token_embed)?;
        let aux_proj = aux_proj.to_dtype(self.dtype)?;
        let aux_normed = self.hidden_norm.forward(&aux_proj)?;
        let residual_ctx = if self.cfg.norm_before_residual {
            aux_normed.clone()
        } else {
            aux_proj.clone()
        };
        let block_in = Tensor::cat(&[&token_normed, &aux_normed], 1)?;
        let (block_out_all, k_rot, v_ctx) = self.block_seq(&block_in, &residual_ctx, 0)?;

        let mut new_k: Vec<Tensor> = Vec::with_capacity(k);
        let mut new_v: Vec<Tensor> = Vec::with_capacity(k);

        let last_idx = Tensor::from_vec(vec![(ctx_len - 1) as u32], 1, &self.device)?;
        let mut h_cond = block_out_all.index_select(&last_idx, 0)?.contiguous()?;
        let draft0 = self.draft_from_block(&h_cond)?;
        let mut out = Vec::with_capacity(k);
        let mut token = self.d2t_map(draft0);
        out.push(token);

        for j in 1..k {
            let ids = Tensor::from_vec(vec![token], 1, &self.device)?;
            let emb = self.embed_tokens.index_select(&ids, 0)?;
            let tn = self.input_layernorm.forward(&emb)?;
            let an = self.hidden_norm.forward(&h_cond)?;
            let bin = Tensor::cat(&[&tn, &an], 1)?;
            let block_out = self.block_step(
                &bin,
                &h_cond,
                ctx_len + j - 1,
                &k_rot,
                &v_ctx,
                &mut new_k,
                &mut new_v,
            )?;
            let d = self.draft_from_block(&block_out)?;
            token = self.d2t_map(d);
            out.push(token);
            h_cond = block_out;
        }
        Ok(out)
    }

    pub fn chain_draft_cached(
        &self,
        cache: &mut DrafterKvCache,
        context: &[u32],
        aux_proj: &Tensor,
        k: usize,
    ) -> Result<Vec<u32>> {
        self.chain_draft_cached_cond(cache, context, aux_proj, k, None, false)
    }

    pub fn chain_draft_cached_cond(
        &self,
        cache: &mut DrafterKvCache,
        context: &[u32],
        aux_proj: &Tensor,
        k: usize,
        bonus: Option<u32>,
        shift: bool,
    ) -> Result<Vec<u32>> {
        self.chain_draft_cached_cond_tail(cache, context, aux_proj, 0, k, bonus, shift)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_context_rows(
        &self,
        cache: &mut DrafterKvCache,
        context: &[u32],
        aux_proj: &Tensor,
        aux_base: usize,
        encode_to: usize,
        shift: bool,
        bonus: Option<u32>,
    ) -> Result<()> {
        let ctx_len = context.len();
        while cache.len < encode_to {
            let target = (cache.len + DRAFTER_ENCODE_CHUNK).min(encode_to);
            let m = target - cache.len;
            let new_tokens: Vec<u32> = if shift {
                (cache.len..target)
                    .map(|i| {
                        if i + 1 < ctx_len {
                            Ok(context[i + 1])
                        } else {
                            bonus.ok_or_else(|| {
                                anyhow!("encode_context_rows: shift row {i} needs a bonus token")
                            })
                        }
                    })
                    .collect::<Result<_>>()?
            } else {
                context[cache.len..target].to_vec()
            };
            let ids = Tensor::from_vec(new_tokens, m, &self.device)?;
            let token_embed = self.embed_tokens.index_select(&ids, 0)?;
            let token_normed = self.input_layernorm.forward(&token_embed)?;

            let row_ids: Vec<u32> =
                ((cache.len - aux_base) as u32..(target - aux_base) as u32).collect();
            let row_ids = Tensor::from_vec(row_ids, m, &self.device)?;
            let aux_new = aux_proj.index_select(&row_ids, 0)?.to_dtype(self.dtype)?;
            let aux_normed = self.hidden_norm.forward(&aux_new)?;
            let residual = if self.cfg.norm_before_residual {
                aux_normed.clone()
            } else {
                aux_new.clone()
            };
            let block_in = Tensor::cat(&[&token_normed, &aux_normed], 1)?;
            let block_out = self.block_seq_append_cached(cache, &block_in, &residual)?;
            let last_idx = Tensor::from_vec(vec![(m - 1) as u32], 1, &self.device)?;
            cache.last_h = Some(block_out.index_select(&last_idx, 0)?.contiguous()?);
            cache.len = target;
        }
        Ok(())
    }

    pub fn preencode_context(
        &self,
        cache: &mut DrafterKvCache,
        context: &[u32],
        aux_proj: &Tensor,
        aux_base: usize,
        encode_to: usize,
        shift: bool,
    ) -> Result<()> {
        let ctx_len = context.len();
        if encode_to > ctx_len || (shift && encode_to >= ctx_len) {
            bail!(
                "preencode_context: encode_to {encode_to} out of range for context {ctx_len} (shift={shift})"
            );
        }
        if encode_to <= cache.len {
            return Ok(());
        }
        if aux_base > cache.len {
            bail!(
                "preencode_context: aux_base {aux_base} is past the encoded prefix {}",
                cache.len
            );
        }
        let dims = aux_proj.dims();
        if dims.len() != 2 || dims[1] != self.cfg.hidden_size || aux_base + dims[0] < encode_to {
            bail!(
                "preencode_context: aux rows [{aux_base}, {aux_base}+{:?}) do not cover encode_to {encode_to}",
                dims
            );
        }
        self.encode_context_rows(cache, context, aux_proj, aux_base, encode_to, shift, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chain_draft_cached_cond_tail(
        &self,
        cache: &mut DrafterKvCache,
        context: &[u32],
        aux_proj: &Tensor,
        aux_base: usize,
        k: usize,
        bonus: Option<u32>,
        shift: bool,
    ) -> Result<Vec<u32>> {
        if shift && bonus.is_none() {
            bail!("chain_draft_cached_cond: shift mode requires a bonus token");
        }
        let ctx_len = context.len();
        if ctx_len == 0 {
            bail!("chain_draft_cached: empty context");
        }
        if aux_proj.dims() != [ctx_len - aux_base, self.cfg.hidden_size] {
            bail!(
                "chain_draft_cached: projected aux tail must be [{} - {aux_base}, {}], got {:?}",
                ctx_len,
                self.cfg.hidden_size,
                aux_proj.dims()
            );
        }
        if cache.len > ctx_len {
            bail!(
                "chain_draft_cached: cache holds {} rows but context has only {ctx_len}; \
                 the cache is append-only and cannot rewind",
                cache.len
            );
        }
        if aux_base > cache.len {
            bail!(
                "chain_draft_cached: aux tail starts at {aux_base} but only {} rows are encoded",
                cache.len
            );
        }

        self.encode_context_rows(cache, context, aux_proj, aux_base, ctx_len, shift, bonus)?;

        let k_ctx = cache.k.as_ref().expect("cache.k set after encode");
        let v_ctx = cache.v.as_ref().expect("cache.v set after encode");
        let mut h_cond = cache
            .last_h
            .as_ref()
            .expect("cache.last_h set after encode")
            .clone();
        let mut new_k: Vec<Tensor> = Vec::with_capacity(k);
        let mut new_v: Vec<Tensor> = Vec::with_capacity(k);

        if std::env::var_os("NV_EAGLE3_NO_DEVICE_CHAIN").is_none() {
            let dbg = std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_DEBUG").is_some()
                && std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_EAGER").is_some();
            let dump = |tag: &str, t: &Tensor| {
                if dbg {
                    let v: Vec<f32> = t
                        .to_dtype(DType::F32)
                        .and_then(|x| x.flatten_all())
                        .and_then(|x| x.to_vec1())
                        .unwrap_or_default();
                    let head: Vec<f32> = v.iter().take(4).copied().collect();
                    eprintln!("[chain-e] {tag}: {head:?}");
                }
            };
            dump("h0", &h_cond);
            let mut tok_dev = match (shift, bonus) {
                (false, Some(b)) => Tensor::from_vec(vec![b], 1, &self.device)?,
                _ => self.draft_next_dev(&h_cond)?,
            };
            let mut toks_dev: Vec<Tensor> = Vec::with_capacity(k);
            toks_dev.push(tok_dev.clone());
            for j in 1..k {
                let bin = self.step_block_in_dev(&tok_dev, &h_cond)?;
                dump(&format!("s{j} bin"), &bin);
                let block_out = self.block_step(
                    &bin,
                    &h_cond,
                    ctx_len + j - 1,
                    k_ctx,
                    v_ctx,
                    &mut new_k,
                    &mut new_v,
                )?;
                if dbg {
                    if let (Some(kr), Some(_vr)) = (new_k.last(), new_v.last()) {
                        dump(&format!("s{j} k_rot"), kr);
                    }
                    dump(&format!("s{j} h"), &block_out);
                }
                tok_dev = self.draft_next_dev(&block_out)?;
                toks_dev.push(tok_dev.clone());
                h_cond = block_out;
            }
            let all = Tensor::cat(&toks_dev[..], 0)?;
            return Ok(all.to_vec1()?);
        }

        let mut token = match (shift, bonus) {
            (false, Some(b)) => b,
            _ => {
                let draft0 = self.draft_from_block(&h_cond)?;
                self.d2t_map(draft0)
            }
        };
        let mut out = Vec::with_capacity(k);
        out.push(token);
        for j in 1..k {
            let bin = self.step_block_in(token, &h_cond)?;
            let block_out = self.block_step(
                &bin,
                &h_cond,
                ctx_len + j - 1,
                k_ctx,
                v_ctx,
                &mut new_k,
                &mut new_v,
            )?;
            let d = self.draft_from_block(&block_out)?;
            token = self.d2t_map(d);
            out.push(token);
            h_cond = block_out;
        }
        Ok(out)
    }

    fn block_seq_appended(
        &self,
        block_in: &Tensor,
        residual: &Tensor,
        pos_base: usize,
        k_prev: Option<&Tensor>,
        v_prev: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let m = block_in.dims()[0];
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;

        let q = self
            .q_proj
            .forward(block_in)?
            .reshape((1usize, m, nh, hd))?;
        let kk = self
            .k_proj
            .forward(block_in)?
            .reshape((1usize, m, nkv, hd))?;
        let vv = self
            .v_proj
            .forward(block_in)?
            .reshape((1usize, m, nkv, hd))?;
        let positions: Vec<u32> = (pos_base as u32..(pos_base + m) as u32).collect();
        let pos_t = Tensor::from_vec(positions, (1usize, m), &self.device)?;
        let (q_rot, k_rot) = self.rope.apply(&q, &kk, &pos_t)?;

        let (k_all, v_all) = match (k_prev, v_prev) {
            (Some(kp), Some(vp)) => (Tensor::cat(&[kp, &k_rot], 1)?, Tensor::cat(&[vp, &vv], 1)?),
            (None, None) => (k_rot, vv),
            _ => bail!("block_seq_appended: k_prev/v_prev must both be set or both absent"),
        };

        let attn_out =
            self.causal_sdpa_qblocked(&q_rot, &k_all, &v_all, pos_base, encode_attn_score_cap_bytes())?;
        let attn_out = attn_out
            .squeeze(0)?
            .reshape((m, nh * hd))?
            .to_dtype(self.dtype)?;
        let block_out = self.block_tail_from_attn(&attn_out, residual)?;
        Ok((block_out, k_all, v_all))
    }

    fn causal_sdpa_qblocked(
        &self,
        q_rot: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        pos_base: usize,
        score_cap_bytes: usize,
    ) -> Result<Tensor> {
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let m = q_rot.dims()[1];
        let sk = k_all.dims()[1];
        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0f32 / (hd as f32).sqrt(),
            causal: true,
        };
        let qs = encode_attn_query_block(nh, m, sk, score_cap_bytes);
        if qs >= m {
            return sdpa(q_rot, k_all, v_all, &attn_cfg);
        }
        let mut outs: Vec<Tensor> = Vec::with_capacity(m.div_ceil(qs));
        let mut o = 0usize;
        while o < m {
            let cur = qs.min(m - o);
            let visible = pos_base + o + cur;
            let q_blk = q_rot.narrow(1, o, cur)?;
            let k_blk = k_all.narrow(1, 0, visible)?;
            let v_blk = v_all.narrow(1, 0, visible)?;
            outs.push(sdpa(&q_blk, &k_blk, &v_blk, &attn_cfg)?);
            o += cur;
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, 1).map_err(Into::into)
    }

    fn block_tail_from_attn(&self, attn_out: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let attn_out = self.o_proj.forward(attn_out)?;
        let post_attn = residual.add(&attn_out)?;
        let normed_mlp_in = self.post_attention_layernorm.forward(&post_attn)?;
        let gate = self.gate_proj.forward(&normed_mlp_in)?;
        let up = self.up_proj.forward(&normed_mlp_in)?;
        let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = self.down_proj.forward(&act)?;
        post_attn.add(&mlp_out).map_err(Into::into)
    }

    fn block_seq_append_cached(
        &self,
        cache: &mut DrafterKvCache,
        block_in: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let m = block_in.dims()[0];
            if m <= DRAFTER_DEVICE_KV_MAX_TAIL
                && !cache.dev_off
                && self.dtype == DType::BF16
                && matches!(self.device, Device::Cuda(_))
            {
                match self.tail_append_device(cache, block_in, residual, m) {
                    Ok(Some(block_out)) => {
                        cache.maybe_compact()?;
                        return Ok(block_out);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        if !cache.dev_warned {
                            eprintln!(
                                "[eagle3] drafter device-KV tail append failed ({e}); using eager sdpa path"
                            );
                            cache.dev_warned = true;
                        }
                        cache.dev_off = true;
                    }
                }
            }
        }
        let (block_out, k_all, v_all) = self.block_seq_appended(
            block_in,
            residual,
            cache.len,
            cache.k.as_ref(),
            cache.v.as_ref(),
        )?;
        cache.k = Some(k_all);
        cache.v = Some(v_all);
        #[cfg(feature = "cuda")]
        cache.clear_device_kv();
        cache.maybe_compact()?;
        Ok(block_out)
    }

    #[cfg(feature = "cuda")]
    fn tail_append_device(
        &self,
        cache: &mut DrafterKvCache,
        block_in: &Tensor,
        residual: &Tensor,
        m: usize,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::DevicePtrMut;
        if std::env::var_os("NV_EAGLE3_NO_DEVICE_KV").is_some() {
            cache.dev_off = true;
            return Ok(None);
        }
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        if hd > 512 || nkv == 0 || nh % nkv != 0 {
            cache.dev_off = true;
            return Ok(None);
        }
        let phys = cache.phys_len();
        anyhow::ensure!(
            phys + cache.evicted == cache.len,
            "tail_append_device: phys {phys} + evicted {} != len {}",
            cache.evicted,
            cache.len
        );
        let need = phys + m;
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        if cache.dev_k.is_none() || cache.dev_cap < need {
            let align = cache.device_kv_align().max(1);
            let cap = need.div_ceil(align) * align + align;
            let k_full = Tensor::zeros((1usize, cap, nkv, hd), DType::BF16, &self.device)?;
            let v_full = Tensor::zeros((1usize, cap, nkv, hd), DType::BF16, &self.device)?;
            if phys > 0 {
                let src_k = cache
                    .k
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow!("tail_append_device: cache.k missing at len {}", cache.len)
                    })?
                    .contiguous()?;
                let src_v = cache
                    .v
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow!("tail_append_device: cache.v missing at len {}", cache.len)
                    })?
                    .contiguous()?;
                anyhow::ensure!(
                    src_k.dims() == [1, phys, nkv, hd] && src_k.dtype() == DType::BF16,
                    "tail_append_device: unexpected cache.k {:?} {:?}",
                    src_k.dims(),
                    src_k.dtype()
                );
                k_full.slice_set(&src_k, 1, 0)?;
                v_full.slice_set(&src_v, 1, 0)?;
            }
            cache.dev_k = Some(k_full);
            cache.dev_v = Some(v_full);
            cache.dev_cap = cap;
        }
        if cache.dev_n.is_none() {
            cache.dev_n = Some(
                stream
                    .alloc_zeros::<i32>(1)
                    .map_err(|e| anyhow!("dev_n alloc: {e:?}"))?,
            );
        }
        if !cache.dev_masks.contains_key(&m) {
            let mask = stream
                .clone_htod(&crate::chain::lower_tri_mask(m))
                .map_err(|e| anyhow!("tail mask htod: {e:?}"))?;
            cache.dev_masks.insert(m, mask);
        }

        let q = self
            .q_proj
            .forward(block_in)?
            .reshape((1usize, m, nh, hd))?;
        let kk = self
            .k_proj
            .forward(block_in)?
            .reshape((1usize, m, nkv, hd))?;
        let vv = self
            .v_proj
            .forward(block_in)?
            .reshape((1usize, m, nkv, hd))?;
        let positions: Vec<u32> = (cache.len as u32..(cache.len + m) as u32).collect();
        let pos_t = Tensor::from_vec(positions, (1usize, m), &self.device)?;
        let (q_rot, k_rot) = self.rope.apply(&q, &kk, &pos_t)?;
        let scale = 1.0f64 / (hd as f64).sqrt();
        let q_scaled = (q_rot * scale)?.reshape((m, nh * hd))?.contiguous()?;
        let k_rot = k_rot.contiguous()?;
        let vv = vv.contiguous()?;

        let k_full = cache.dev_k.as_ref().expect("dev_k armed above");
        let v_full = cache.dev_v.as_ref().expect("dev_v armed above");
        k_full.slice_set(&k_rot, 1, phys)?;
        v_full.slice_set(&vv, 1, phys)?;

        {
            let host = [phys as i32];
            let n_dev = cache.dev_n.as_mut().expect("dev_n armed above");
            stream
                .memcpy_htod(&host[..], n_dev)
                .map_err(|e| anyhow!("dev_n htod: {e:?}"))?;
        }

        let mut out_dev = stream
            .alloc_zeros::<half::bf16>(m * nh * hd)
            .map_err(|e| anyhow!("tail attn out alloc: {e:?}"))?;
        let rc = {
            let p_q = chain_raw_bf16(&q_scaled, &stream)?;
            let p_k = chain_raw_bf16(k_full, &stream)?;
            let p_v = chain_raw_bf16(v_full, &stream)?;
            let p_n = chain_slice_ptr(cache.dev_n.as_ref().expect("dev_n"), &stream);
            let p_mask = chain_slice_ptr(cache.dev_masks.get(&m).expect("mask"), &stream);
            let (p_out, _g) = out_dev.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::tree_verify_attn_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    p_q as *const u16,
                    p_k as *const u16,
                    p_v as *const u16,
                    p_n as *const i32,
                    p_mask as *const u8,
                    std::ptr::null(),
                    p_out as *mut u16,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    m as i32,
                    0,
                )
            }
        };
        if rc != 0 {
            if !cache.dev_warned {
                eprintln!(
                    "[eagle3] tree_verify_attn_bf16 rejected drafter dims (rc={rc}); using eager sdpa path"
                );
                cache.dev_warned = true;
            }
            cache.dev_off = true;
            cache.clear_device_kv();
            return Ok(None);
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev);
        let attn_out = Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (m, nh * hd),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let block_out = self.block_tail_from_attn(&attn_out, residual)?;
        cache.k = Some(cache.dev_k.as_ref().expect("dev_k").narrow(1, 0, need)?);
        cache.v = Some(cache.dev_v.as_ref().expect("dev_v").narrow(1, 0, need)?);
        Ok(Some(block_out))
    }

    fn draft_logits_vec(&self, block_out: &Tensor) -> Result<Vec<f32>> {
        let normed = self.norm.forward(block_out)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        Ok(logits.reshape((self.cfg.draft_vocab_size,))?.to_vec1()?)
    }

    #[allow(clippy::too_many_arguments)]
    fn block_step_tree(
        &self,
        block_in: &Tensor,
        h_cond: &Tensor,
        pos: usize,
        k_ctx: &Tensor,
        v_ctx: &Tensor,
        path_k: &[Tensor],
        path_v: &[Tensor],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let residual = if self.cfg.norm_before_residual {
            self.hidden_norm.forward(h_cond)?
        } else {
            h_cond.clone()
        };
        let q = self
            .q_proj
            .forward(block_in)?
            .reshape((1usize, 1usize, nh, hd))?;
        let kk = self
            .k_proj
            .forward(block_in)?
            .reshape((1usize, 1usize, nkv, hd))?;
        let vv = self
            .v_proj
            .forward(block_in)?
            .reshape((1usize, 1usize, nkv, hd))?;
        let pos_t = Tensor::from_vec(vec![pos as u32], (1usize, 1usize), &self.device)?;
        let (q_rot, k_rot) = self.rope.apply(&q, &kk, &pos_t)?;

        let mut kparts: Vec<&Tensor> = Vec::with_capacity(2 + path_k.len());
        kparts.push(k_ctx);
        for t in path_k.iter() {
            kparts.push(t);
        }
        kparts.push(&k_rot);
        let mut vparts: Vec<&Tensor> = Vec::with_capacity(2 + path_v.len());
        vparts.push(v_ctx);
        for t in path_v.iter() {
            vparts.push(t);
        }
        vparts.push(&vv);
        let k_all = Tensor::cat(&kparts[..], 1)?;
        let v_all = Tensor::cat(&vparts[..], 1)?;

        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0f32 / (hd as f32).sqrt(),
            causal: false,
        };
        let attn_out = sdpa(&q_rot, &k_all, &v_all, &attn_cfg)?;
        let attn_out = attn_out
            .squeeze(0)?
            .reshape((1usize, nh * hd))?
            .to_dtype(self.dtype)?;
        let block_out = self.block_tail_from_attn(&attn_out, &residual)?;
        Ok((block_out, k_rot, vv))
    }

    fn step_block_in(&self, token: u32, h_cond: &Tensor) -> Result<Tensor> {
        let ids = Tensor::from_vec(vec![token], 1, &self.device)?;
        let emb = self.embed_tokens.index_select(&ids, 0)?;
        let tn = self.input_layernorm.forward(&emb)?;
        let an = self.hidden_norm.forward(h_cond)?;
        Tensor::cat(&[&tn, &an], 1).map_err(Into::into)
    }

    pub fn tree_draft(
        &self,
        context: &[u32],
        aux_full: &Tensor,
        branch: usize,
        max_depth: usize,
        budget: usize,
    ) -> Result<crate::eagle3::DraftTree> {
        let ctx_len = context.len();
        if ctx_len == 0 {
            bail!("tree_draft: empty context");
        }
        if aux_full.dims() != [ctx_len, self.cfg.fc_in_dim()] {
            bail!(
                "tree_draft: aux must be [{ctx_len}, {}], got {:?}",
                self.cfg.fc_in_dim(),
                aux_full.dims()
            );
        }
        let ids = Tensor::from_vec(context.to_vec(), ctx_len, &self.device)?;
        let token_embed = self.embed_tokens.index_select(&ids, 0)?;
        let token_normed = self.input_layernorm.forward(&token_embed)?;
        let aux_proj = self.fc.forward(&aux_full.to_dtype(self.dtype)?)?;
        let aux_normed = self.hidden_norm.forward(&aux_proj)?;
        let residual_ctx = if self.cfg.norm_before_residual {
            aux_normed.clone()
        } else {
            aux_proj.clone()
        };
        let block_in = Tensor::cat(&[&token_normed, &aux_normed], 1)?;
        let (block_out_all, k_ctx, v_ctx) = self.block_seq(&block_in, &residual_ctx, 0)?;

        let last_idx = Tensor::from_vec(vec![(ctx_len - 1) as u32], 1, &self.device)?;
        let root_hidden = block_out_all.index_select(&last_idx, 0)?.contiguous()?;
        let root_logits = self.draft_logits_vec(&root_hidden)?;

        let mut tokens: Vec<u32> = Vec::new();
        let mut parents: Vec<Option<usize>> = Vec::new();
        let mut depths: Vec<usize> = Vec::new();
        let mut node_k: Vec<Tensor> = Vec::new();
        let mut node_v: Vec<Tensor> = Vec::new();

        #[allow(clippy::type_complexity)]
        let mut queue: std::collections::VecDeque<(
            Option<usize>,
            Tensor,
            Vec<f32>,
            Vec<usize>,
            usize,
        )> = std::collections::VecDeque::new();
        queue.push_back((None, root_hidden, root_logits, Vec::new(), 1));

        while let Some((parent, p_hidden, p_logits, path, depth)) = queue.pop_front() {
            if depth > max_depth || tokens.len() >= budget {
                continue;
            }
            let top = top_k(&p_logits, branch);
            let path_k: Vec<Tensor> = path.iter().map(|&i| node_k[i].clone()).collect();
            let path_v: Vec<Tensor> = path.iter().map(|&i| node_v[i].clone()).collect();
            for d_idx in top {
                if tokens.len() >= budget {
                    break;
                }
                let tok = self.d2t_map(d_idx);
                let bin = self.step_block_in(tok, &p_hidden)?;
                let (block_out, sk, sv) = self.block_step_tree(
                    &bin,
                    &p_hidden,
                    ctx_len + depth - 1,
                    &k_ctx,
                    &v_ctx,
                    &path_k,
                    &path_v,
                )?;
                let idx = tokens.len();
                tokens.push(tok);
                parents.push(parent);
                depths.push(depth);
                node_k.push(sk);
                node_v.push(sv);
                if depth < max_depth {
                    let child_logits = self.draft_logits_vec(&block_out)?;
                    let mut child_path = path.clone();
                    child_path.push(idx);
                    queue.push_back((Some(idx), block_out, child_logits, child_path, depth + 1));
                }
            }
        }

        Ok(crate::eagle3::DraftTree {
            tokens,
            parents,
            depths,
        })
    }

    pub fn score_with_aux(&mut self, context: &[u32], aux_hidden: &Tensor) -> Result<Vec<f32>> {
        let logits = self.forward(context, Some(aux_hidden))?;
        last_row_to_f32(&logits)
    }

    pub fn forward(&self, token_ids: &[u32], aux_hidden: Option<&Tensor>) -> Result<Tensor> {
        if token_ids.is_empty() {
            bail!("LoadedEagle3Scorer::forward: empty context");
        }
        let seq = token_ids.len();
        let hidden = self.cfg.hidden_size;

        let ids = Tensor::from_vec(token_ids.to_vec(), seq, &self.device)?;
        let token_embed = self.embed_tokens.index_select(&ids, 0)?;
        debug_assert_eq!(token_embed.dims(), &[seq, hidden]);

        let aux_proj = if let Some(aux) = aux_hidden {
            let aux_dims = aux.dims();
            if aux_dims != [seq, self.cfg.fc_in_dim()] {
                bail!(
                    "aux_hidden: expected [{seq}, {}], got {:?}",
                    self.cfg.fc_in_dim(),
                    aux_dims
                );
            }
            if self.cfg.norm_before_fc {
                bail!(
                    "norm_before_fc=true is not implemented in this loader \
                     (the RedHatAI checkpoint sets it to false; enabling it \
                     requires loading an extra RMSNorm weight that is not in \
                     the artifact)"
                );
            }
            let aux_cast = aux.to_dtype(self.dtype)?;
            self.fc.forward(&aux_cast)?
        } else {
            Tensor::zeros((seq, hidden), self.dtype, &self.device)?
        };
        debug_assert_eq!(aux_proj.dims(), &[seq, hidden]);

        let token_normed = self.input_layernorm.forward(&token_embed)?;
        let aux_normed = self.hidden_norm.forward(&aux_proj)?;
        let residual = if self.cfg.norm_before_residual {
            aux_normed.clone()
        } else {
            aux_proj.clone()
        };

        let block_in = Tensor::cat(&[&token_normed, &aux_normed], 1)?;
        debug_assert_eq!(block_in.dims(), &[seq, self.cfg.block_in_dim()]);

        let (block_out, _k_rot, _v) = self.block_seq(&block_in, &residual, 0)?;

        let normed_out = self.norm.forward(&block_out)?;
        let logits = self.lm_head.forward(&normed_out)?.to_dtype(self.dtype)?;
        debug_assert_eq!(logits.dims(), &[seq, self.cfg.draft_vocab_size]);
        Ok(logits)
    }

    fn block_seq(
        &self,
        block_in: &Tensor,
        residual: &Tensor,
        pos_base: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq = block_in.dims()[0];
        let n_heads = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim;

        let q = self
            .q_proj
            .forward(block_in)?
            .reshape((1usize, seq, n_heads, head_dim))?;
        let k = self
            .k_proj
            .forward(block_in)?
            .reshape((1usize, seq, n_kv, head_dim))?;
        let v = self
            .v_proj
            .forward(block_in)?
            .reshape((1usize, seq, n_kv, head_dim))?;

        let positions: Vec<u32> = (pos_base as u32..(pos_base + seq) as u32).collect();
        let pos_t = Tensor::from_vec(positions, (1usize, seq), &self.device)?;
        let (q_rot, k_rot) = self.rope.apply(&q, &k, &pos_t)?;

        let attn_cfg = AttnConfig {
            num_heads: n_heads,
            num_kv_heads: n_kv,
            head_dim,
            softmax_scale: 1.0f32 / (head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = sdpa(&q_rot, &k_rot, &v, &attn_cfg)?;
        let attn_out = attn_out
            .squeeze(0)?
            .reshape((seq, n_heads * head_dim))?
            .to_dtype(self.dtype)?;
        let block_out = self.block_tail_from_attn(&attn_out, residual)?;
        Ok((block_out, k_rot, v))
    }
}

fn top_k(xs: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(xs.len()).max(1);
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.into_iter().map(|i| i as u32).collect()
}

impl DraftScorer for LoadedEagle3Scorer {
    fn score(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        let logits = self.forward(context, None)?;
        last_row_to_f32(&logits)
    }
}

fn last_row_to_f32(logits: &Tensor) -> Result<Vec<f32>> {
    let dims = logits.dims();
    if dims.len() != 2 {
        bail!("last_row_to_f32: expected rank-2 logits, got {:?}", dims);
    }
    let last = dims[0] - 1;
    let row = logits.narrow(0, last, 1)?.squeeze(0)?;
    let row_f32 = row.to_dtype(DType::F32)?.contiguous()?;
    Ok(row_f32.to_vec1::<f32>()?)
}

pub(crate) fn d2t_apply(d2t: &[u32], draft_token: u32) -> u32 {
    let i = draft_token as usize;
    if i < d2t.len() {
        draft_token.wrapping_add(d2t[i])
    } else {
        draft_token
    }
}

pub(crate) fn d2t_dev_tensor(
    cache: &std::sync::OnceLock<Tensor>,
    d2t: &[u32],
    draft_vocab: usize,
    device: &Device,
) -> Result<Tensor> {
    if let Some(t) = cache.get() {
        return Ok(t.clone());
    }
    let mapped: Vec<u32> = (0..draft_vocab as u32).map(|i| d2t_apply(d2t, i)).collect();
    let t = Tensor::from_vec(mapped, draft_vocab, device)?;
    let _ = cache.set(t.clone());
    Ok(t)
}

pub fn validate_d2t(d2t: &[u32], draft_vocab_size: usize, target_vocab_size: usize) -> Result<()> {
    if d2t.len() != draft_vocab_size {
        bail!(
            "d2t: expected {draft_vocab_size} entries, got {}",
            d2t.len()
        );
    }
    for (draft, &off) in d2t.iter().enumerate() {
        let target = (draft as u64) + (off as u64);
        if target >= target_vocab_size as u64 {
            bail!(
                "d2t[{draft}] = {off} maps draft token {draft} to target id {target}, \
                 out of range for target_vocab_size {target_vocab_size}"
            );
        }
    }
    Ok(())
}

pub fn validate_t2d(t2d: &[bool], target_vocab_size: usize) -> Result<()> {
    if t2d.len() != target_vocab_size {
        bail!(
            "t2d: expected {target_vocab_size} entries, got {}",
            t2d.len()
        );
    }
    Ok(())
}

pub(crate) fn load_d2t(weights: &WeightLoader, name: &str, len: usize) -> Result<Vec<u32>> {
    if !weights.has(name) {
        bail!("missing tensor {name}");
    }
    let shape = weights
        .shape_of(name)
        .ok_or_else(|| anyhow!("no shape for {name}"))?;
    if shape != [len] {
        bail!("d2t {name}: expected [{len}], got {:?}", shape);
    }
    let t = weights
        .get(name, DType::I64)
        .with_context(|| format!("load {name}"))?;
    let vals: Vec<i64> = t.to_vec1()?;
    let mut out = Vec::with_capacity(len);
    for v in vals {
        if v < 0 {
            bail!("d2t {name}: negative value {v}");
        }
        if v > u32::MAX as i64 {
            bail!("d2t {name}: value {v} > u32::MAX");
        }
        out.push(v as u32);
    }
    Ok(out)
}

pub(crate) fn load_t2d(weights: &WeightLoader, name: &str, len: usize) -> Result<Vec<bool>> {
    if !weights.has(name) {
        bail!("missing tensor {name}");
    }
    let shape = weights
        .shape_of(name)
        .ok_or_else(|| anyhow!("no shape for {name}"))?;
    if shape != [len] {
        bail!("t2d {name}: expected [{len}], got {:?}", shape);
    }
    let t = weights
        .get(name, DType::U8)
        .with_context(|| format!("load {name}"))?;
    let bytes: Vec<u8> = t.to_vec1()?;
    Ok(bytes.iter().map(|b| *b != 0).collect())
}

#[cfg(feature = "cuda")]
pub struct DraftChainGraph {
    forked: std::sync::Arc<cudarc::driver::CudaStream>,
    runner: nv_kernels::graph::CudaGraphRunner,
    kd: usize,
    cap: usize,

    disabled: bool,
    captured: bool,
    k_cache: cudarc::driver::CudaSlice<half::bf16>,
    v_cache: cudarc::driver::CudaSlice<half::bf16>,
    h0_buf: cudarc::driver::CudaSlice<half::bf16>,

    n_buf: cudarc::driver::CudaSlice<i32>,
    out_buf: cudarc::driver::CudaSlice<u32>,
    attn_out_buf: cudarc::driver::CudaSlice<half::bf16>,

    mask1: cudarc::driver::CudaSlice<u8>,
    host_ns: Vec<i32>,
    host_pos: Vec<i32>,
    mirrored_rows: usize,
    mirrored_compactions: u64,

    pos_steps: Vec<cudarc::driver::CudaSlice<i32>>,

    normed_buf: cudarc::driver::CudaSlice<half::bf16>,
    logits_buf: cudarc::driver::CudaSlice<half::bf16>,
    amax_val: cudarc::driver::CudaSlice<f32>,
    amax_idx: cudarc::driver::CudaSlice<i32>,
    idx_buf: cudarc::driver::CudaSlice<u32>,
    emb_buf: cudarc::driver::CudaSlice<half::bf16>,
    bin_buf: cudarc::driver::CudaSlice<half::bf16>,
    q_buf: cudarc::driver::CudaSlice<half::bf16>,
    k_buf: cudarc::driver::CudaSlice<half::bf16>,
    v_buf: cudarc::driver::CudaSlice<half::bf16>,
    qr_buf: cudarc::driver::CudaSlice<half::bf16>,
    kr_buf: cudarc::driver::CudaSlice<half::bf16>,
    o_buf: cudarc::driver::CudaSlice<half::bf16>,
    pa_buf: cudarc::driver::CudaSlice<half::bf16>,
    nm_buf: cudarc::driver::CudaSlice<half::bf16>,
    gate_buf: cudarc::driver::CudaSlice<half::bf16>,
    up_buf: cudarc::driver::CudaSlice<half::bf16>,
    act_buf: cudarc::driver::CudaSlice<half::bf16>,
    mlp_buf: cudarc::driver::CudaSlice<half::bf16>,
    h_bufs: Vec<cudarc::driver::CudaSlice<half::bf16>>,
}

#[cfg(feature = "cuda")]
impl Drop for DraftChainGraph {
    fn drop(&mut self) {
        let teardown =
            nv_models::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked);
        let runner = &mut self.runner;
        teardown.run(|| runner.invalidate());
    }
}

#[cfg(feature = "cuda")]
impl DraftChainGraph {
    pub fn kd(&self) -> usize {
        self.kd
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn disabled(&self) -> bool {
        self.disabled
    }

    pub fn graph_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    fn mirror_rows(
        &mut self,
        k_all: &Tensor,
        v_all: &Tensor,
        from: usize,
        to: usize,
        nkv: usize,
        hd: usize,
        device: &Device,
    ) -> Result<()> {
        if to <= from {
            return Ok(());
        }
        let stride = nkv * hd;
        anyhow::ensure!(to <= self.cap, "mirror_rows: {to} rows > cap {}", self.cap);
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("mirror_rows requires cuda"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        for (src_t, dst) in [(k_all, &mut self.k_cache), (v_all, &mut self.v_cache)] {
            let c = src_t.contiguous()?;
            let (st, l) = c.storage_and_layout();
            let cuda = match &*st {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("mirror_rows: expected cuda storage"),
            };
            let sl = cuda.as_cuda_slice::<half::bf16>()?;
            let off = l.start_offset();
            anyhow::ensure!(
                sl.len() >= off + to * stride,
                "mirror_rows: cache tensor has {} elems, need {} + off {}",
                sl.len(),
                to * stride,
                off
            );
            let src = sl.slice(off + from * stride..off + to * stride);
            let mut d = dst.slice_mut(from * stride..to * stride);
            stream
                .memcpy_dtod(&src, &mut d)
                .map_err(|e| anyhow!("mirror_rows dtod: {e:?}"))?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl LoadedEagle3Scorer {
    pub fn new_chain_graph(&self, cap: usize, kd: usize) -> Result<DraftChainGraph> {
        if kd == 0 {
            bail!("new_chain_graph: kd must be >= 1");
        }
        self.chain_graph_eligible()?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("new_chain_graph requires cuda"),
        };
        let raw_ctx: std::sync::Arc<cudarc::driver::CudaContext> =
            dev.cuda_stream().context().clone();
        let forked = raw_ctx.new_stream().map_err(|e| anyhow!(e))?;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let stride = nkv * hd;
        let k_cache = forked
            .alloc_zeros::<half::bf16>(cap * stride)
            .map_err(|e| anyhow!("chain k_cache alloc ({cap} x {stride}): {e:?}"))?;
        let v_cache = forked
            .alloc_zeros::<half::bf16>(cap * stride)
            .map_err(|e| anyhow!("chain v_cache alloc: {e:?}"))?;
        let h0_buf = forked
            .alloc_zeros::<half::bf16>(self.cfg.hidden_size)
            .map_err(|e| anyhow!(e))?;
        let n_buf = forked.alloc_zeros::<i32>(kd).map_err(|e| anyhow!(e))?;
        let out_buf = forked.alloc_zeros::<u32>(kd).map_err(|e| anyhow!(e))?;
        let attn_out_buf = forked
            .alloc_zeros::<half::bf16>(nh * hd)
            .map_err(|e| anyhow!(e))?;
        let mask1 = forked.alloc_zeros::<u8>(1).map_err(|e| anyhow!(e))?;
        let mut pos_steps = Vec::with_capacity(kd.saturating_sub(1));
        for _ in 0..kd.saturating_sub(1) {
            pos_steps.push(forked.alloc_zeros::<i32>(1).map_err(|e| anyhow!(e))?);
        }

        let hidden = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let dv = self.cfg.draft_vocab_size;
        let bin_dim = self.cfg.block_in_dim();
        let parts = nv_kernels::cuda::argmax_parts();
        let abf = |n: usize| -> Result<cudarc::driver::CudaSlice<half::bf16>> {
            forked
                .alloc_zeros::<half::bf16>(n)
                .map_err(|e| anyhow!("chain buf alloc ({n}): {e:?}"))
        };
        let normed_buf = abf(hidden)?;
        let logits_buf = abf(dv)?;
        let amax_val = forked.alloc_zeros::<f32>(parts).map_err(|e| anyhow!(e))?;
        let amax_idx = forked.alloc_zeros::<i32>(parts).map_err(|e| anyhow!(e))?;
        let idx_buf = forked.alloc_zeros::<u32>(1).map_err(|e| anyhow!(e))?;
        let emb_buf = abf(hidden)?;
        let bin_buf = abf(bin_dim)?;
        let q_buf = abf(nh * hd)?;
        let k_buf = abf(nkv * hd)?;
        let v_buf = abf(nkv * hd)?;
        let qr_buf = abf(nh * hd)?;
        let kr_buf = abf(nkv * hd)?;
        let o_buf = abf(hidden)?;
        let pa_buf = abf(hidden)?;
        let nm_buf = abf(hidden)?;
        let gate_buf = abf(inter)?;
        let up_buf = abf(inter)?;
        let act_buf = abf(inter)?;
        let mlp_buf = abf(hidden)?;
        let h_bufs = vec![abf(hidden)?, abf(hidden)?];
        forked.synchronize().map_err(|e| anyhow!(e))?;

        let _ = self.d2t_map_dev_tensor()?;
        let runner = nv_kernels::graph::CudaGraphRunner::new(forked.clone());
        Ok(DraftChainGraph {
            forked,
            runner,
            kd,
            cap,
            disabled: false,
            captured: false,
            k_cache,
            v_cache,
            h0_buf,
            n_buf,
            out_buf,
            attn_out_buf,
            mask1,
            host_ns: vec![0i32; kd],
            host_pos: vec![0i32; kd],
            mirrored_rows: 0,
            mirrored_compactions: 0,
            pos_steps,
            normed_buf,
            logits_buf,
            amax_val,
            amax_idx,
            idx_buf,
            emb_buf,
            bin_buf,
            q_buf,
            k_buf,
            v_buf,
            qr_buf,
            kr_buf,
            o_buf,
            pa_buf,
            nm_buf,
            gate_buf,
            up_buf,
            act_buf,
            mlp_buf,
            h_bufs,
        })
    }

    fn chain_graph_eligible(&self) -> Result<()> {
        if self.dtype != DType::BF16 || self.embed_tokens.dtype() != DType::BF16 {
            bail!("chain graph requires a bf16 drafter");
        }
        if !self.embed_tokens.is_contiguous() {
            bail!("chain graph: embed_tokens must be contiguous");
        }
        if self.cfg.head_dim % 2 != 0 {
            bail!("chain graph: head_dim must be even");
        }
        for (name, l) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
            ("gate_proj", &self.gate_proj),
            ("up_proj", &self.up_proj),
            ("down_proj", &self.down_proj),
            ("lm_head", &self.lm_head),
        ] {
            let w = l
                .weight()
                .ok_or_else(|| anyhow!("chain graph: {name} is not bf16-dense"))?;
            if !w.is_contiguous() || w.dtype() != DType::BF16 {
                bail!("chain graph: {name} weight must be contiguous bf16");
            }
            if l.in_features() % 2 != 0 {
                bail!("chain graph: {name} in_features must be even");
            }
            if l.bias().is_some() {
                bail!("chain graph: {name} bias unsupported");
            }
        }
        Ok(())
    }

    pub fn chain_draft_cached_shift_graphed(
        &self,
        cache: &mut DrafterKvCache,
        g: &mut DraftChainGraph,
        context: &[u32],
        aux_proj: &Tensor,
        kd: usize,
        bonus: u32,
    ) -> Result<Vec<u32>> {
        let eager_body = std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_EAGER").is_some();
        self.chain_draft_cached_shift_graphed_full(
            cache, g, context, aux_proj, 0, kd, bonus, eager_body,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chain_draft_cached_shift_graphed_tail(
        &self,
        cache: &mut DrafterKvCache,
        g: &mut DraftChainGraph,
        context: &[u32],
        aux_proj: &Tensor,
        aux_base: usize,
        kd: usize,
        bonus: u32,
    ) -> Result<Vec<u32>> {
        let eager_body = std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_EAGER").is_some();
        self.chain_draft_cached_shift_graphed_full(
            cache, g, context, aux_proj, aux_base, kd, bonus, eager_body,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chain_draft_cached_shift_graphed_mode(
        &self,
        cache: &mut DrafterKvCache,
        g: &mut DraftChainGraph,
        context: &[u32],
        aux_proj: &Tensor,
        kd: usize,
        bonus: u32,
        eager_body: bool,
    ) -> Result<Vec<u32>> {
        self.chain_draft_cached_shift_graphed_full(
            cache, g, context, aux_proj, 0, kd, bonus, eager_body,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chain_draft_cached_shift_graphed_full(
        &self,
        cache: &mut DrafterKvCache,
        g: &mut DraftChainGraph,
        context: &[u32],
        aux_proj: &Tensor,
        aux_base: usize,
        kd: usize,
        bonus: u32,
        eager_body: bool,
    ) -> Result<Vec<u32>> {
        let ctx_len = context.len();

        let projected_phys = {
            let pending = ctx_len.saturating_sub(cache.len);
            let p = cache.phys_len() + pending;
            match cache.kv_cap() {
                Some((s, w)) => p.min(s + w + DRAFTER_KV_CAP_SLACK),
                None => p,
            }
        };
        if g.disabled
            || kd != g.kd
            || kd < 1
            || projected_phys + kd > g.cap
            || ctx_len + kd > self.cfg.max_position_embeddings
            || std::env::var_os("NV_EAGLE3_NO_DEVICE_CHAIN").is_some()
        {
            return self.chain_draft_cached_cond_tail(
                cache,
                context,
                aux_proj,
                aux_base,
                kd,
                Some(bonus),
                true,
            );
        }
        if ctx_len == 0 {
            bail!("chain_draft_cached: empty context");
        }
        if aux_proj.dims() != [ctx_len - aux_base, self.cfg.hidden_size] {
            bail!(
                "chain_draft_cached: projected aux tail must be [{} - {aux_base}, {}], got {:?}",
                ctx_len,
                self.cfg.hidden_size,
                aux_proj.dims()
            );
        }
        if cache.len > ctx_len {
            bail!(
                "chain_draft_cached: cache holds {} rows but context has only {ctx_len}; \
                 the cache is append-only and cannot rewind",
                cache.len
            );
        }
        if aux_base > cache.len {
            bail!(
                "chain_draft_cached: aux tail starts at {aux_base} but only {} rows are encoded",
                cache.len
            );
        }

        self.encode_context_rows(
            cache,
            context,
            aux_proj,
            aux_base,
            ctx_len,
            true,
            Some(bonus),
        )?;

        {
            let phys = cache.phys_len();
            anyhow::ensure!(
                phys + cache.evicted() == ctx_len,
                "graphed tail encode: cache holds {phys} rows + {} evicted, expected {ctx_len}",
                cache.evicted()
            );
            let from = if g.mirrored_compactions != cache.compactions() {
                0
            } else {
                g.mirrored_rows.min(phys)
            };
            if phys > from {
                let k_all = cache.k.clone().expect("cache.k set by encode");
                let v_all = cache.v.clone().expect("cache.v set by encode");
                g.mirror_rows(
                    &k_all,
                    &v_all,
                    from,
                    phys,
                    self.cfg.num_key_value_heads,
                    self.cfg.head_dim,
                    &self.device,
                )?;
            }
            g.mirrored_rows = phys;
            g.mirrored_compactions = cache.compactions();
        }

        match self.run_chain_graph(cache, g, ctx_len, kd, eager_body) {
            Ok(out) => Ok(out),
            Err(e) => {
                eprintln!("[eagle3] graphed draft chain failed ({e}); falling back to eager chain permanently");
                g.disabled = true;
                let _ = g.forked.synchronize();
                self.chain_draft_cached_cond_tail(
                    cache,
                    context,
                    aux_proj,
                    aux_base,
                    kd,
                    Some(bonus),
                    true,
                )
            }
        }
    }

    fn run_chain_graph(
        &self,
        cache: &DrafterKvCache,
        g: &mut DraftChainGraph,
        ctx_len: usize,
        kd: usize,
        eager_body: bool,
    ) -> Result<Vec<u32>> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("run_chain_graph requires cuda"),
        };

        let raw_ctx = dev.cuda_stream().context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            dev.cuda_stream().synchronize().map_err(|e| anyhow!(e))?;
        }

        let phys = cache.phys_len();
        anyhow::ensure!(
            phys + cache.evicted() == ctx_len,
            "run_chain_graph: phys {phys} + evicted {} != ctx {ctx_len}",
            cache.evicted()
        );
        anyhow::ensure!(
            phys + kd <= g.cap,
            "run_chain_graph: phys {phys} + kd {kd} > cap {}",
            g.cap
        );
        for i in 0..kd {
            g.host_ns[i] = (phys + i) as i32;
            g.host_pos[i] = (ctx_len + i) as i32;
        }

        nv_layers::cuda_stream::current_stream(&dev)
            .synchronize()
            .map_err(|e| anyhow!(e))?;

        {
            let last_h = cache
                .last_h
                .as_ref()
                .ok_or_else(|| anyhow!("run_chain_graph: cache.last_h missing"))?
                .contiguous()?;
            let (st, l) = last_h.storage_and_layout();
            let cuda = match &*st {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("last_h: expected cuda storage"),
            };
            let sl = cuda.as_cuda_slice::<half::bf16>()?;
            let off = l.start_offset();
            anyhow::ensure!(
                sl.len() >= off + self.cfg.hidden_size,
                "last_h slice bounds"
            );
            let view = sl.slice(off..off + self.cfg.hidden_size);
            g.forked
                .memcpy_dtod(&view, &mut g.h0_buf)
                .map_err(|e| anyhow!("h0 stage: {e:?}"))?;
        }

        let hidden = self.cfg.hidden_size;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;
        let dv = self.cfg.draft_vocab_size;
        let tv = self.cfg.target_vocab_size;
        let bin_dim = self.cfg.block_in_dim();
        let nbr = self.cfg.norm_before_residual;
        let scale = (1.0f64 / (hd as f64).sqrt()) as f32;
        let was_captured = g.captured;

        g.forked
            .memcpy_htod(&g.host_ns[..], &mut g.n_buf)
            .map_err(|e| anyhow!("ns htod: {e:?}"))?;
        for i in 0..g.pos_steps.len() {
            let v = [g.host_pos[i]];
            g.forked
                .memcpy_htod(&v[..], &mut g.pos_steps[i])
                .map_err(|e| anyhow!("pos htod: {e:?}"))?;
        }

        let d2t_t = self.d2t_map_dev_tensor()?;
        let eps_n = self.norm.eps() as f32;
        let eps_il = self.input_layernorm.eps() as f32;
        let eps_hn = self.hidden_norm.eps() as f32;
        let eps_pl = self.post_attention_layernorm.eps() as f32;
        let fk = g.forked.clone();
        let need_w = |l: &'_ Linear, name: &str| -> Result<Tensor> {
            Ok(l.weight()
                .ok_or_else(|| anyhow!("chain graph: {name} is not bf16-dense"))?
                .clone())
        };
        let p_wn = chain_raw_bf16(self.norm.weight_bf16(), &fk)?;
        let p_wil = chain_raw_bf16(self.input_layernorm.weight_bf16(), &fk)?;
        let p_whn = chain_raw_bf16(self.hidden_norm.weight_bf16(), &fk)?;
        let p_wpl = chain_raw_bf16(self.post_attention_layernorm.weight_bf16(), &fk)?;
        let p_wq = chain_raw_bf16(&need_w(&self.q_proj, "q_proj")?, &fk)?;
        let p_wk = chain_raw_bf16(&need_w(&self.k_proj, "k_proj")?, &fk)?;
        let p_wv = chain_raw_bf16(&need_w(&self.v_proj, "v_proj")?, &fk)?;
        let p_wo = chain_raw_bf16(&need_w(&self.o_proj, "o_proj")?, &fk)?;
        let p_wg = chain_raw_bf16(&need_w(&self.gate_proj, "gate_proj")?, &fk)?;
        let p_wu = chain_raw_bf16(&need_w(&self.up_proj, "up_proj")?, &fk)?;
        let p_wd = chain_raw_bf16(&need_w(&self.down_proj, "down_proj")?, &fk)?;
        let p_wlm = chain_raw_bf16(&need_w(&self.lm_head, "lm_head")?, &fk)?;
        let p_emb_w = chain_raw_bf16(&self.embed_tokens, &fk)?;
        let p_cos = chain_raw_f32(self.rope.cos(), &fk)?;
        let p_sin = chain_raw_f32(self.rope.sin(), &fk)?;
        let p_d2t = chain_raw_u32(&d2t_t, &fk)?;

        let p_kc = chain_slice_ptr(&g.k_cache, &fk);
        let p_vc = chain_slice_ptr(&g.v_cache, &fk);
        let p_h0 = chain_slice_ptr(&g.h0_buf, &fk);
        let p_n = chain_slice_ptr(&g.n_buf, &fk);
        let p_out = chain_slice_ptr(&g.out_buf, &fk);
        let p_attn = chain_slice_ptr(&g.attn_out_buf, &fk);
        let p_mask = chain_slice_ptr(&g.mask1, &fk);
        let p_normed = chain_slice_ptr(&g.normed_buf, &fk);
        let p_logits = chain_slice_ptr(&g.logits_buf, &fk);
        let p_aval = chain_slice_ptr(&g.amax_val, &fk);
        let p_aidx = chain_slice_ptr(&g.amax_idx, &fk);
        let p_idx = chain_slice_ptr(&g.idx_buf, &fk);
        let p_emb = chain_slice_ptr(&g.emb_buf, &fk);
        let p_bin = chain_slice_ptr(&g.bin_buf, &fk);
        let p_q = chain_slice_ptr(&g.q_buf, &fk);
        let p_k = chain_slice_ptr(&g.k_buf, &fk);
        let p_v = chain_slice_ptr(&g.v_buf, &fk);
        let p_qr = chain_slice_ptr(&g.qr_buf, &fk);
        let p_kr = chain_slice_ptr(&g.kr_buf, &fk);
        let p_o = chain_slice_ptr(&g.o_buf, &fk);
        let p_pa = chain_slice_ptr(&g.pa_buf, &fk);
        let p_nm = chain_slice_ptr(&g.nm_buf, &fk);
        let p_gate = chain_slice_ptr(&g.gate_buf, &fk);
        let p_up = chain_slice_ptr(&g.up_buf, &fk);
        let p_act = chain_slice_ptr(&g.act_buf, &fk);
        let p_mlp = chain_slice_ptr(&g.mlp_buf, &fk);
        let p_h = [
            chain_slice_ptr(&g.h_bufs[0], &fk),
            chain_slice_ptr(&g.h_bufs[1], &fk),
        ];
        let p_pos: Vec<u64> = g
            .pos_steps
            .iter()
            .map(|b| chain_slice_ptr(b, &fk))
            .collect();

        let mut body = |s: &std::sync::Arc<cudarc::driver::CudaStream>| -> Result<()> {
            let cu = s.cu_stream() as *mut std::ffi::c_void;
            let gemv = |w: u64, x: u64, y: u64, n: usize, k: usize, what: &str| -> Result<()> {
                let rc = unsafe {
                    nv_kernels::cuda::gemv_bf16(
                        cu,
                        w as *const u16,
                        x as *const u16,
                        y as *mut u16,
                        n as i32,
                        k as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "{what} rc={rc}");
                Ok(())
            };
            let rms = |x: u64, w: u64, y: u64, eps: f32, what: &str| -> Result<()> {
                let rc = unsafe {
                    nv_kernels::cuda::rmsnorm_bf16(
                        cu,
                        x as *const u16,
                        w as *const u16,
                        y as *mut u16,
                        1,
                        hidden,
                        eps,
                    )
                };
                anyhow::ensure!(rc == 0, "{what} rc={rc}");
                Ok(())
            };
            let draft_next = |h: u64, slot: usize| -> Result<()> {
                rms(h, p_wn, p_normed, eps_n, "chain norm")?;
                gemv(p_wlm, p_normed, p_logits, dv, hidden, "chain lm_head")?;
                let rc = unsafe {
                    nv_kernels::cuda::argmax_bf16(
                        cu,
                        p_logits as *const u16,
                        dv as i32,
                        p_aval as *mut f32,
                        p_aidx as *mut i32,
                        std::ptr::null(),
                        p_idx as *mut u32,
                        std::ptr::null_mut(),
                        0,
                    )
                };
                anyhow::ensure!(rc == 0, "chain argmax rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::token_map_u32(
                        cu,
                        p_d2t as *const u32,
                        p_idx as *const u32,
                        (p_out + (slot as u64) * 4) as *mut u32,
                    )
                };
                anyhow::ensure!(rc == 0, "chain d2t rc={rc}");
                Ok(())
            };

            draft_next(p_h0, 0)?;
            for j in 1..kd {
                let h_in = if j == 1 { p_h0 } else { p_h[j % 2] };
                let h_out = p_h[(j - 1) % 2];
                let rc = unsafe {
                    nv_kernels::cuda::gather_rows_bf16(
                        cu,
                        p_emb_w as *const u16,
                        (p_out + ((j - 1) as u64) * 4) as *const i32,
                        p_emb as *mut u16,
                        1,
                        hidden as i32,
                        tv as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "chain embed gather rc={rc}");
                rms(p_emb, p_wil, p_bin, eps_il, "chain input_ln")?;
                rms(
                    h_in,
                    p_whn,
                    p_bin + (hidden as u64) * 2,
                    eps_hn,
                    "chain hidden_norm",
                )?;
                let resid = if nbr {
                    p_bin + (hidden as u64) * 2
                } else {
                    h_in
                };
                gemv(p_wq, p_bin, p_q, nh * hd, bin_dim, "chain q_proj")?;
                gemv(p_wk, p_bin, p_k, nkv * hd, bin_dim, "chain k_proj")?;
                gemv(p_wv, p_bin, p_v, nkv * hd, bin_dim, "chain v_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::rope_bf16_oop(
                        cu,
                        p_q as *const u16,
                        p_k as *const u16,
                        p_qr as *mut u16,
                        p_kr as *mut u16,
                        p_cos as *const f32,
                        p_sin as *const f32,
                        p_pos[j - 1] as *const i32,
                        1,
                        nh,
                        nkv,
                        hd,
                    )
                };
                anyhow::ensure!(rc == 0, "chain rope rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::scale_inplace_bf16(cu, p_qr as *mut u16, scale, nh * hd)
                };
                anyhow::ensure!(rc == 0, "chain q scale rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::kv_append_bf16(
                        cu,
                        p_kr as *const u16,
                        p_v as *const u16,
                        p_kc as *mut u16,
                        p_vc as *mut u16,
                        (p_n + ((j - 1) as u64) * 4) as *const i32,
                        1,
                        nkv as i32,
                        hd as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "kv_append_bf16 rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::tree_verify_attn_bf16(
                        cu,
                        p_qr as *const u16,
                        p_kc as *const u16,
                        p_vc as *const u16,
                        (p_n + (j as u64) * 4) as *const i32,
                        p_mask as *const u8,
                        std::ptr::null(),
                        p_attn as *mut u16,
                        nh as i32,
                        nkv as i32,
                        hd as i32,
                        1,
                        0,
                    )
                };
                anyhow::ensure!(rc == 0, "tree_verify_attn_bf16 rc={rc}");
                gemv(p_wo, p_attn, p_o, hidden, nh * hd, "chain o_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::residual_add_scale_bf16(
                        cu,
                        resid as *const u16,
                        p_o as *const u16,
                        p_pa as *mut u16,
                        1.0,
                        hidden,
                    )
                };
                anyhow::ensure!(rc == 0, "chain resid add rc={rc}");
                rms(p_pa, p_wpl, p_nm, eps_pl, "chain post_ln")?;
                gemv(p_wg, p_nm, p_gate, inter, hidden, "chain gate_proj")?;
                gemv(p_wu, p_nm, p_up, inter, hidden, "chain up_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::silu_mul_bf16(
                        cu,
                        p_gate as *const u16,
                        p_up as *const u16,
                        p_act as *mut u16,
                        inter,
                    )
                };
                anyhow::ensure!(rc == 0, "chain silu_mul rc={rc}");
                gemv(p_wd, p_act, p_mlp, hidden, inter, "chain down_proj")?;
                let rc = unsafe {
                    nv_kernels::cuda::residual_add_scale_bf16(
                        cu,
                        p_pa as *const u16,
                        p_mlp as *const u16,
                        h_out as *mut u16,
                        1.0,
                        hidden,
                    )
                };
                anyhow::ensure!(rc == 0, "chain h add rc={rc}");
                draft_next(h_out, j)?;
            }
            Ok(())
        };

        if eager_body {
            body(&fk)?;
            fk.synchronize().map_err(|e| anyhow!(e))?;
            if std::env::var_os("NV_EAGLE3_GRAPH_CHAIN_DEBUG").is_some() {
                let ns: Vec<i32> = fk.clone_dtoh(&g.n_buf).map_err(|e| anyhow!(e))?;
                eprintln!("[chain-dbg] ctx={ctx_len} n_buf={ns:?}");
                if let Some(kt) = cache.k.as_ref() {
                    let stride = nkv * hd;
                    let kc_host: Vec<half::bf16> =
                        fk.clone_dtoh(&g.k_cache).map_err(|e| anyhow!(e))?;
                    let want: Vec<half::bf16> = kt
                        .contiguous()?
                        .flatten_all()?
                        .to_vec1()
                        .map_err(|e| anyhow!("cache.k d2h: {e}"))?;
                    let upto = (phys * stride).min(want.len()).min(kc_host.len());
                    let mut bad = 0usize;
                    let mut first_bad = None;
                    for i in 0..upto {
                        if kc_host[i] != want[i] {
                            bad += 1;
                            if first_bad.is_none() {
                                first_bad = Some(i);
                            }
                        }
                    }
                    eprintln!(
                        "[chain-dbg] mirror K parity: {bad}/{upto} mismatched, first={first_bad:?} \
                         (row {:?})",
                        first_bad.map(|i| i / stride)
                    );
                    let s0 = phys * stride;
                    eprintln!(
                        "[chain-dbg] chain slot ctx first4: {:?}",
                        &kc_host[s0..(s0 + 4).min(kc_host.len())]
                    );
                }
                let ao: Vec<half::bf16> = fk.clone_dtoh(&g.attn_out_buf).map_err(|e| anyhow!(e))?;
                eprintln!(
                    "[chain-dbg] last attn_out first4: {:?}",
                    &ao[..4.min(ao.len())]
                );
            }
            let out: Vec<u32> = fk
                .clone_dtoh(&g.out_buf)
                .map_err(|e| anyhow!("chain out d2h: {e:?}"))?;
            return Ok(out);
        }

        if !was_captured {
            body(&fk)?;
            fk.synchronize().map_err(|e| anyhow!(e))?;
        }
        g.runner.run(kd as u64, &mut body)?;
        g.captured = true;
        fk.synchronize().map_err(|e| anyhow!(e))?;
        let out: Vec<u32> = fk
            .clone_dtoh(&g.out_buf)
            .map_err(|e| anyhow!("chain out d2h: {e:?}"))?;
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn chain_raw<T>(
    t: &Tensor,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<u64>
where
    T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType,
{
    use cudarc::driver::DevicePtr;
    anyhow::ensure!(t.is_contiguous(), "chain graph: tensor must be contiguous");
    let (st, l) = t.storage_and_layout();
    let cuda = match &*st {
        candle_core::Storage::Cuda(c) => c,
        _ => bail!("chain graph: expected cuda storage"),
    };
    let sl = cuda.as_cuda_slice::<T>()?;
    let view = sl.slice(l.start_offset()..);
    let (p, _g) = view.device_ptr(s);
    Ok(p as u64)
}

#[cfg(feature = "cuda")]
pub(crate) fn chain_raw_bf16(
    t: &Tensor,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<u64> {
    chain_raw::<half::bf16>(t, s)
}

#[cfg(feature = "cuda")]
pub(crate) fn chain_raw_f32(
    t: &Tensor,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<u64> {
    chain_raw::<f32>(t, s)
}

#[cfg(feature = "cuda")]
pub(crate) fn chain_raw_u32(
    t: &Tensor,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<u64> {
    chain_raw::<u32>(t, s)
}

#[cfg(feature = "cuda")]
pub(crate) fn chain_slice_ptr<T: cudarc::driver::DeviceRepr>(
    sl: &cudarc::driver::CudaSlice<T>,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> u64 {
    use cudarc::driver::DevicePtr;
    let (p, _g) = sl.device_ptr(s);
    p as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eagle3::{Eagle3Config as TreeCfg, Eagle3Proposer};
    use half::bf16;

    fn synthetic_scorer() -> Result<LoadedEagle3Scorer> {
        synthetic_scorer_with_d2t((0..32u32).collect())
    }

    fn synthetic_scorer_with_d2t(d2t: Vec<u32>) -> Result<LoadedEagle3Scorer> {
        synthetic_scorer_with_d2t_on(d2t, &Device::Cpu, 16)
    }

    #[allow(dead_code)]
    fn synthetic_scorer_on(dev: &Device, max_pos: usize) -> Result<LoadedEagle3Scorer> {
        synthetic_scorer_with_d2t_on((0..32u32).collect(), dev, max_pos)
    }

    fn synthetic_scorer_with_d2t_on(
        d2t: Vec<u32>,
        dev: &Device,
        max_pos: usize,
    ) -> Result<LoadedEagle3Scorer> {
        let cfg = Eagle3SpeculatorConfig {
            hidden_size: 16,
            draft_vocab_size: 32,
            target_vocab_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            max_position_embeddings: max_pos,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            norm_before_residual: true,
            norm_before_fc: false,
            eagle_aux_hidden_state_layer_ids: vec![0],
        };
        let dev = dev.clone();
        let dtype = DType::BF16;
        let h = cfg.hidden_size;

        let mk_linear = |out: usize, inp: usize| -> Result<Linear> {
            let total = out * inp;
            let data: Vec<bf16> = (0..total)
                .map(|i| bf16::from_f32(((i as f32) * 0.001).sin() * 0.05))
                .collect();
            let t = Tensor::from_vec(data, (out, inp), &dev)?;
            Linear::new(t, None)
        };
        let mk_rms = |dim: usize| -> Result<RmsNorm> {
            let data: Vec<bf16> = (0..dim).map(|_| bf16::from_f32(1.0)).collect();
            let t = Tensor::from_vec(data, dim, &dev)?;
            Ok(RmsNorm::new(t, cfg.rms_norm_eps))
        };

        let embed_total = cfg.target_vocab_size * h;
        let embed_data: Vec<bf16> = (0..embed_total)
            .map(|i| bf16::from_f32(((i as f32) * 0.013).cos() * 0.05))
            .collect();
        let embed = Tensor::from_vec(embed_data, (cfg.target_vocab_size, h), &dev)?;

        let fc = mk_linear(h, cfg.fc_in_dim())?;
        let input_ln = mk_rms(h)?;
        let hidden_ln = mk_rms(h)?;
        let post_ln = mk_rms(h)?;
        let q = mk_linear(cfg.q_out_dim(), cfg.block_in_dim())?;
        let k = mk_linear(cfg.kv_out_dim(), cfg.block_in_dim())?;
        let v = mk_linear(cfg.kv_out_dim(), cfg.block_in_dim())?;
        let o = mk_linear(h, cfg.q_out_dim())?;
        let gate = mk_linear(cfg.intermediate_size, h)?;
        let up = mk_linear(cfg.intermediate_size, h)?;
        let down = mk_linear(h, cfg.intermediate_size)?;
        let norm = mk_rms(h)?;
        let lm_head = mk_linear(cfg.draft_vocab_size, h)?;

        let t2d: Vec<bool> = (0..cfg.target_vocab_size)
            .map(|i| i < cfg.draft_vocab_size)
            .collect();

        LoadedEagle3Scorer::from_parts(
            cfg, dev, dtype, embed, fc, input_ln, hidden_ln, post_ln, q, k, v, o, gate, up, down,
            norm, lm_head, d2t, t2d,
        )
    }

    #[test]
    fn synthetic_zero_weight_forward_shape() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let logits = scorer.forward(&[1u32, 2, 3], None).expect("forward");
        assert_eq!(logits.dims(), &[3, 32]);
        assert_eq!(logits.dtype(), DType::BF16);
    }

    #[test]
    fn synthetic_score_returns_draft_vocab_floats() {
        let mut scorer = synthetic_scorer().expect("build synthetic scorer");
        let row = scorer.score(&[1, 2, 3, 4]).expect("score");
        assert_eq!(row.len(), 32);
        assert!(row.iter().all(|x| x.is_finite()));
    }

    fn synthetic_aux(scorer: &LoadedEagle3Scorer, rows: usize) -> Tensor {
        let n = rows * scorer.cfg.fc_in_dim();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).sin() * 0.3).collect();
        Tensor::from_vec(data, (rows, scorer.cfg.fc_in_dim()), &scorer.device).unwrap()
    }

    #[test]
    fn project_aux_is_row_wise_so_it_can_be_appended() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let rows = 7usize;
        let aux = synthetic_aux(&scorer, rows);
        let whole = scorer.project_aux(&aux).expect("project whole");
        for split in 1..rows {
            let head = scorer
                .project_aux(&aux.narrow(0, 0, split).unwrap())
                .unwrap();
            let tail = scorer
                .project_aux(&aux.narrow(0, split, rows - split).unwrap())
                .unwrap();
            let appended = Tensor::cat(&[&head, &tail], 0).unwrap();
            let a: Vec<f32> = whole
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = appended
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(
                a, b,
                "split={split}: append-only projection must be bit-identical"
            );
        }
    }

    #[test]
    fn chain_draft_projected_matches_chain_draft() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let context: Vec<u32> = vec![1, 5, 9, 13, 2];
        let aux = synthetic_aux(&scorer, context.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let a = scorer.chain_draft(&context, &aux, 4).expect("chain_draft");
        let b = scorer
            .chain_draft_projected(&context, &proj, 4)
            .expect("chain_draft_projected");
        assert_eq!(a, b);
    }

    #[test]
    fn chain_draft_cached_matches_projected_across_rounds() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");

        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7, 3, 11, 4, 6, 8, 10];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let mut cache = DrafterKvCache::new();
        assert!(cache.is_empty());
        let k = 4usize;
        let growth = [0usize, 1, 3, 2, 1];
        let mut n = 5usize;
        for (round, g) in growth.iter().enumerate() {
            n += g;
            let ctx = &full[..n];
            let proj_n = proj.narrow(0, 0, n).unwrap();
            let a = scorer
                .chain_draft_projected(ctx, &proj_n, k)
                .expect("chain_draft_projected");
            let b = scorer
                .chain_draft_cached(&mut cache, ctx, &proj_n, k)
                .expect("chain_draft_cached");
            assert_eq!(a, b, "round {round} (n={n})");
            assert_eq!(cache.len(), n);
        }
    }

    #[test]
    fn chain_draft_cached_cond_shift_incremental_matches_fresh() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7, 3, 11, 4, 6, 8, 10];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 3usize;
        let mut cache = DrafterKvCache::new();
        let mut n = 5usize;
        let growth = [0usize, 1, 3, 2];
        for (round, g) in growth.iter().enumerate() {
            n += g;
            let ctx = &full[..n];

            let bonus = full[n];
            let proj_n = proj.narrow(0, 0, n).unwrap();
            let a = scorer
                .chain_draft_cached_cond(&mut cache, ctx, &proj_n, k, Some(bonus), true)
                .expect("incremental shift");
            let mut fresh = DrafterKvCache::new();
            let b = scorer
                .chain_draft_cached_cond(&mut fresh, ctx, &proj_n, k, Some(bonus), true)
                .expect("fresh shift");
            assert_eq!(a, b, "round {round} (n={n})");
            assert_eq!(cache.len(), n);
            assert_eq!(a.len(), k);
        }
    }

    #[test]
    fn preencode_then_tail_draft_matches_full_cond() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7, 3, 11, 4, 6, 8, 10];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 3usize;
        for shift in [true, false] {
            for n in [6usize, 9, 12] {
                let ctx = &full[..n];
                let bonus = if n < full.len() { full[n] } else { 42 };
                let proj_n = proj.narrow(0, 0, n).unwrap();
                let mut fresh = DrafterKvCache::new();
                let a = scorer
                    .chain_draft_cached_cond(&mut fresh, ctx, &proj_n, k, Some(bonus), shift)
                    .expect("full-aux draft");
                for pre in [1usize, 3, n - 1] {
                    let mut cache = DrafterKvCache::new();
                    scorer
                        .preencode_context(
                            &mut cache,
                            ctx,
                            &proj_n.narrow(0, 0, pre).unwrap(),
                            0,
                            pre,
                            shift,
                        )
                        .expect("preencode");
                    assert_eq!(cache.len(), pre);
                    let tail = proj_n.narrow(0, pre, n - pre).unwrap();
                    let b = scorer
                        .chain_draft_cached_cond_tail(
                            &mut cache,
                            ctx,
                            &tail,
                            pre,
                            k,
                            Some(bonus),
                            shift,
                        )
                        .expect("tail draft");
                    assert_eq!(a, b, "shift={shift} n={n} pre={pre}");
                    assert_eq!(cache.len(), n);
                }
            }
        }
    }

    #[test]
    fn encode_attn_query_block_shrinks_with_prefix_and_never_zero() {
        let cap = 512 * 1024 * 1024;
        assert_eq!(encode_attn_query_block(4, 1024, 12, cap), 1024);
        let big = encode_attn_query_block(32, 1024, 24576, cap);
        assert!(big >= 1 && big < 1024, "24576-prefix must sub-block: {big}");
        assert_eq!(
            encode_attn_query_block(32, 1024, usize::MAX, 4),
            1,
            "an unbounded prefix floors the query block at 1, never 0"
        );
    }

    #[test]
    fn causal_sdpa_qblocked_matches_single_block() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let nh = scorer.cfg.num_attention_heads;
        let nkv = scorer.cfg.num_key_value_heads;
        let hd = scorer.cfg.head_dim;
        let m = 6usize;
        let sk = 10usize;
        let pos_base = sk - m;
        let q: Vec<f32> = (0..m * nh * hd).map(|i| ((i as f32) * 0.03).sin()).collect();
        let kk: Vec<f32> = (0..sk * nkv * hd).map(|i| ((i as f32) * 0.017).cos()).collect();
        let vv: Vec<f32> = (0..sk * nkv * hd).map(|i| ((i as f32) * 0.021).sin()).collect();
        let dev = &scorer.device;
        let q_rot = Tensor::from_vec(q, (1, m, nh, hd), dev).unwrap();
        let k_all = Tensor::from_vec(kk, (1, sk, nkv, hd), dev).unwrap();
        let v_all = Tensor::from_vec(vv, (1, sk, nkv, hd), dev).unwrap();
        let full = scorer
            .causal_sdpa_qblocked(&q_rot, &k_all, &v_all, pos_base, usize::MAX)
            .expect("single block");
        let blocked = scorer
            .causal_sdpa_qblocked(&q_rot, &k_all, &v_all, pos_base, 1)
            .expect("forced sub-block");
        assert_eq!(full.dims(), &[1, m, nh, hd]);
        assert!(
            max_abs_diff(&full, &blocked) < 1e-5,
            "query sub-blocking must be numerically identical to a single causal sdpa"
        );
    }

    #[test]
    fn preencode_at_24k_completes_with_bounded_query_blocks() {
        let seq = 24_576usize;
        let scorer =
            synthetic_scorer_on(&Device::Cpu, seq + 8).expect("build large-context scorer");
        let context: Vec<u32> = (0..seq).map(|i| (i % 60) as u32).collect();
        let aux = synthetic_aux(&scorer, seq);
        let proj = scorer.project_aux(&aux).expect("project");
        let mut cache = DrafterKvCache::new();
        scorer
            .preencode_context(&mut cache, &context, &proj, 0, seq, false)
            .expect("large-seq preencode must complete without OOM");
        assert_eq!(cache.len(), seq);
        assert_eq!(
            cache.k.as_ref().expect("cache.k").dims(),
            &[1, seq, scorer.cfg.num_key_value_heads, scorer.cfg.head_dim]
        );
    }

    #[test]
    fn small_tail_appends_keep_cache_dims() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7, 3, 11, 4, 6, 8, 10, 12, 14];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 3usize;
        let mut cache = DrafterKvCache::new();
        let mut n = 4usize;
        for g in [0usize, 1, 2, 3, 1] {
            n += g;
            let ctx = &full[..n];
            let bonus = full[n];
            let proj_n = proj.narrow(0, 0, n).unwrap();
            let out = scorer
                .chain_draft_cached_cond(&mut cache, ctx, &proj_n, k, Some(bonus), true)
                .expect("tail append round");
            assert_eq!(out.len(), k);
            assert_eq!(cache.len(), n);
            let kc = cache.k.as_ref().expect("cache.k");
            let vc = cache.v.as_ref().expect("cache.v");
            assert_eq!(
                kc.dims(),
                &[1, n, scorer.cfg.num_key_value_heads, scorer.cfg.head_dim]
            );
            assert_eq!(kc.dims(), vc.dims());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn device_kv_tail_matches_sdpa_path_cuda() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no cuda device ({e}); skipping");
                return;
            }
        };
        let scorer = synthetic_scorer_on(&dev, 512).expect("build synthetic cuda scorer");
        let total = 96usize;
        let full: Vec<u32> = (0..total).map(|i| ((i * 13 + 3) % 60) as u32).collect();
        let aux = synthetic_aux(&scorer, total);
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 4usize;

        for start in [64usize, 4] {
            let mut cache_dev = DrafterKvCache::new();
            cache_dev.set_device_kv_align(8);
            let mut cache_ref = DrafterKvCache::new();
            cache_ref.disable_device_kv();
            let mut n = start;
            let mut rounds = 0usize;
            while n + 2 < total.min(start + 24) {
                let ctx = &full[..n];
                let bonus = full[n];
                let proj_n = proj.narrow(0, 0, n).unwrap();
                let a = scorer
                    .chain_draft_cached_cond(&mut cache_dev, ctx, &proj_n, k, Some(bonus), true)
                    .expect("device-kv round");
                let b = scorer
                    .chain_draft_cached_cond(&mut cache_ref, ctx, &proj_n, k, Some(bonus), true)
                    .expect("sdpa round");
                assert_eq!(a, b, "start={start} n={n} device-KV tokens diverged");
                assert_eq!(cache_dev.len(), cache_ref.len());
                if rounds > 0 && start == 4 {
                    assert!(
                        cache_dev.device_kv_armed(),
                        "device KV should be armed after a small tail append"
                    );
                }
                n += 2;
                rounds += 1;
            }
            assert!(rounds >= 8, "expected enough rounds to exercise growth");
        }
    }

    #[test]
    fn preencode_context_rejects_bonus_row_and_uncovered_aux() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let mut cache = DrafterKvCache::new();
        let err = scorer
            .preencode_context(&mut cache, &full, &proj, 0, full.len(), true)
            .expect_err("shift preencode of the bonus row must be rejected");
        assert!(err.to_string().contains("out of range"), "got: {err}");
        let err = scorer
            .preencode_context(
                &mut cache,
                &full,
                &proj.narrow(0, 0, 2).unwrap(),
                0,
                4,
                false,
            )
            .expect_err("aux not covering encode_to must be rejected");
        assert!(err.to_string().contains("cover"), "got: {err}");
        scorer
            .preencode_context(&mut cache, &full, &proj, 0, full.len(), false)
            .expect("non-shift preencode of every row is fine");
        assert_eq!(cache.len(), full.len());
    }

    #[test]
    fn chain_draft_cached_cond_default_matches_plain_cached() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let mut c1 = DrafterKvCache::new();
        let mut c2 = DrafterKvCache::new();
        let a = scorer
            .chain_draft_cached(&mut c1, &full, &proj, 4)
            .expect("plain");
        let b = scorer
            .chain_draft_cached_cond(&mut c2, &full, &proj, 4, None, false)
            .expect("cond default");
        assert_eq!(a, b);
    }

    #[test]
    fn chain_draft_cached_rejects_rewound_context() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let full: Vec<u32> = vec![1, 5, 9, 13, 2, 7];
        let aux = synthetic_aux(&scorer, full.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let mut cache = DrafterKvCache::new();
        scorer
            .chain_draft_cached(&mut cache, &full, &proj, 3)
            .expect("full context");
        let err = scorer
            .chain_draft_cached(&mut cache, &full[..4], &proj.narrow(0, 0, 4).unwrap(), 3)
            .expect_err("shrunk context must be rejected");
        assert!(err.to_string().contains("append-only"), "got: {err}");
    }

    #[test]
    fn tensor_argmax_matches_first_max_scan() {
        let dev = Device::Cpu;
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3.0, 3.0, 1.0, 3.0],
            vec![-1.0, -2.0, -0.5, -0.5],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY],
            vec![5.0, 5.0, 5.0, 5.0],
        ];
        for row in rows {
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for (i, &x) in row.iter().enumerate() {
                if x > bestv {
                    bestv = x;
                    best = i;
                }
            }
            let n = row.len();
            let t = Tensor::from_vec(row.clone(), (1usize, n), &dev).unwrap();
            let got = t
                .argmax(D::Minus1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(got[0] as usize, best, "row={row:?}");
        }
    }

    #[test]
    fn validate_d2t_accepts_in_range_map() {
        let d2t: Vec<u32> = (0..8u32).map(|i| 16 - i).collect();
        assert!(validate_d2t(&d2t, 8, 17).is_ok());
        assert!(validate_t2d(&[false; 17], 17).is_ok());
    }

    #[test]
    fn validate_d2t_rejects_out_of_range_target() {
        let mut d2t: Vec<u32> = vec![0; 8];
        d2t[7] = 56;
        assert!(validate_d2t(&d2t, 8, 64).is_ok());
        d2t[7] = 57;
        let err = validate_d2t(&d2t, 8, 64).unwrap_err().to_string();
        assert!(err.contains("target id 64"), "unexpected error: {err}");
    }

    #[test]
    fn validate_d2t_rejects_wrapping_offset() {
        let d2t: Vec<u32> = vec![u32::MAX; 4];
        assert!(validate_d2t(&d2t, 4, 262144).is_err());
    }

    #[test]
    fn validate_d2t_rejects_length_mismatch() {
        assert!(validate_d2t(&[0u32; 3], 4, 64).is_err());
        assert!(validate_t2d(&[false; 3], 4).is_err());
    }

    #[test]
    fn from_parts_rejects_malformed_d2t() {
        let mut d2t: Vec<u32> = (0..32u32).collect();
        d2t[31] = 1000;
        assert!(synthetic_scorer_with_d2t(d2t).is_err());
    }

    #[test]
    fn d2t_and_t2d_helpers() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");

        assert_eq!(scorer.d2t_offset(5), 5);
        assert_eq!(scorer.d2t_map(5), 10);

        assert!(scorer.t2d_supports(10));
        assert!(!scorer.t2d_supports(40));
    }

    #[test]
    fn config_parses_redhat_json() {
        let json = r#"{
            "draft_vocab_size": 32000,
            "norm_before_fc": false,
            "norm_before_residual": true,
            "eagle_aux_hidden_state_layer_ids": [2, 30, 57],
            "transformer_layer_config": {
                "head_dim": 256,
                "hidden_size": 5376,
                "intermediate_size": 21504,
                "max_position_embeddings": 262144,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
                "vocab_size": 262144
            }
        }"#;
        let cfg = Eagle3SpeculatorConfig::from_hf_json_str(json).expect("parse");
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.draft_vocab_size, 32000);
        assert_eq!(cfg.target_vocab_size, 262144);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 16);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 21504);
        assert!(cfg.norm_before_residual);
        assert!(!cfg.norm_before_fc);
        assert_eq!(cfg.eagle_aux_hidden_state_layer_ids, vec![2, 30, 57]);
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let av: Vec<f32> = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let bv: Vec<f32> = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(av.len(), bv.len());
        av.iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    #[ignore]
    fn gpu_cached_matches_projected_real_checkpoint() {
        let Some(dir) = require(
            "gpu_cached_matches_projected_real_checkpoint",
            "NV_EAGLE3_DRAFT_DIR",
            std::env::var_os("NV_EAGLE3_DRAFT_DIR").map(std::path::PathBuf::from),
        ) else {
            return;
        };
        let dev = Device::new_cuda(0).expect("cuda device");
        let scorer = LoadedEagle3Scorer::try_load(&dir, &dev).expect("load real checkpoint");
        let cfg = scorer.config().clone();
        let total = 64usize;
        let toks: Vec<u32> = (0..total)
            .map(|i| ((i * 7919 + 13) % cfg.target_vocab_size) as u32)
            .collect();
        let n_aux = total * cfg.fc_in_dim();
        let aux_host: Vec<f32> = (0..n_aux).map(|i| ((i as f32) * 0.0137).sin()).collect();
        let aux = Tensor::from_vec(aux_host, (total, cfg.fc_in_dim()), &dev).unwrap();
        let proj = scorer.project_aux(&aux).expect("project");

        let mut cache = DrafterKvCache::new();
        let k = 8usize;
        let growth = [0usize, 1, 2, 3, 1, 5, 2, 1, 4, 2, 3];
        let mut n = 40usize;
        let mut mismatches = 0usize;
        for (round, g) in growth.iter().enumerate() {
            n += g;
            let ctx = &toks[..n];
            let proj_n = proj.narrow(0, 0, n).unwrap();
            let a = scorer
                .chain_draft_projected(ctx, &proj_n, k)
                .expect("chain_draft_projected");
            let b = scorer
                .chain_draft_cached(&mut cache, ctx, &proj_n, k)
                .expect("chain_draft_cached");

            let mut fresh = DrafterKvCache::new();
            let _ = scorer
                .chain_draft_cached(&mut fresh, ctx, &proj_n, k)
                .expect("fresh cached");
            let kdiff = max_abs_diff(cache.k.as_ref().unwrap(), fresh.k.as_ref().unwrap());
            let vdiff = max_abs_diff(cache.v.as_ref().unwrap(), fresh.v.as_ref().unwrap());
            let hdiff = max_abs_diff(
                cache.last_h.as_ref().unwrap(),
                fresh.last_h.as_ref().unwrap(),
            );
            let same = a == b;
            eprintln!(
                "round {round} n={n} tokens_equal={same} kdiff={kdiff:.6} vdiff={vdiff:.6} hdiff={hdiff:.6}\n  projected={a:?}\n  cached   ={b:?}"
            );

            assert!(
                kdiff < 0.5 && vdiff < 0.5 && hdiff < 2.0,
                "round {round}: cached encode diverged beyond bf16 noise \
                 (kdiff={kdiff} vdiff={vdiff} hdiff={hdiff})"
            );
            if a[0] != b[0] {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "cached first draft token diverged from full re-encode"
        );
    }

    const ALLOW_SKIP: &str = "NV_SPECDECODE_ALLOW_SKIP";

    fn require<T>(test: &str, what: &str, found: Option<T>) -> Option<T> {
        if found.is_none() {
            if std::env::var(ALLOW_SKIP).as_deref() != Ok("1") {
                panic!(
                    "{test}: no {what}. This test is #[ignore]d, so it runs only when asked for \
                     by name; reporting a pass without the artifact answers a question that was \
                     never put. Provide it or set {ALLOW_SKIP}=1."
                );
            }
            eprintln!("SKIP ({ALLOW_SKIP}=1): {test}: no {what}; nothing was exercised");
        }
        found
    }

    fn cached_snapshot_dir() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let root = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/\
             models--RedHatAI--gemma-4-31B-it-speculator.eagle3/snapshots",
        );
        if !root.is_dir() {
            return None;
        }

        for entry in std::fs::read_dir(&root).ok()? {
            let p = entry.ok()?.path();
            if p.is_dir() && p.join("model.safetensors").is_file() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    #[ignore]
    fn loads_redhat_checkpoint_when_present() {
        let Some(dir) = require(
            "loads_redhat_checkpoint_when_present",
            "RedHatAI eagle3 snapshot in the HF cache",
            cached_snapshot_dir(),
        ) else {
            return;
        };
        let scorer =
            LoadedEagle3Scorer::try_load(&dir, &Device::Cpu).expect("load real checkpoint");
        let cfg = scorer.config();
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.draft_vocab_size, 32000);
        assert_eq!(cfg.target_vocab_size, 262144);

        assert_eq!(scorer.d2t().len(), 32000);
        assert_eq!(scorer.t2d().len(), 262144);
    }

    #[test]
    #[ignore]
    fn scores_real_checkpoint_verifier_less() {
        let Some(dir) = require(
            "scores_real_checkpoint_verifier_less",
            "RedHatAI eagle3 snapshot in the HF cache",
            cached_snapshot_dir(),
        ) else {
            return;
        };
        let mut scorer =
            LoadedEagle3Scorer::try_load(&dir, &Device::Cpu).expect("load real checkpoint");
        let row = scorer.score(&[1u32, 2, 3]).expect("score");
        assert_eq!(row.len(), 32000);
        assert!(row.iter().all(|x| x.is_finite()), "non-finite logit");
    }

    #[test]
    #[ignore]
    fn drives_existing_eagle3_proposer() {
        let Some(dir) = require(
            "drives_existing_eagle3_proposer",
            "RedHatAI eagle3 snapshot in the HF cache",
            cached_snapshot_dir(),
        ) else {
            return;
        };
        let scorer =
            LoadedEagle3Scorer::try_load(&dir, &Device::Cpu).expect("load real checkpoint");
        let draft_vocab = scorer.config().draft_vocab_size;

        let target_vocab = scorer.config().target_vocab_size;

        let mut proposer = Eagle3Proposer::new(
            scorer,
            TreeCfg {
                max_depth: 2,
                branch_factor: 2,
                total_budget: 6,
                vocab_size: draft_vocab,
            },
        );
        let tree = proposer
            .expand_tree(&[1u32])
            .expect("expand_tree on real scorer");
        assert!(!tree.is_empty(), "tree must contain at least one draft");
        for (i, &tok) in tree.tokens.iter().enumerate() {
            assert!(
                (tok as usize) < draft_vocab,
                "draft token {tok} out of range"
            );

            assert!(tree.depths[i] <= 2);

            let target = proposer.scorer().d2t_map(tok);
            assert!((target as usize) < target_vocab);

            if let Some(p) = tree.parents[i] {
                assert!(p < i);
                assert_eq!(tree.depths[i], tree.depths[p] + 1);
            } else {
                assert_eq!(tree.depths[i], 1);
            }
        }
    }

    fn patterned_ctx(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 7 + 3) % 60) as u32).collect()
    }

    #[test]
    fn kv_cap_no_eviction_is_identical() {
        let scorer = synthetic_scorer().expect("build synthetic scorer");
        let ctx: Vec<u32> = patterned_ctx(12);
        let aux = synthetic_aux(&scorer, ctx.len());
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 4usize;
        let mut plain = DrafterKvCache::new();
        let a = scorer
            .chain_draft_cached(&mut plain, &ctx, &proj, k)
            .expect("uncapped draft");
        let mut capped = DrafterKvCache::with_kv_cap(2, 8);
        let b = scorer
            .chain_draft_cached(&mut capped, &ctx, &proj, k)
            .expect("capped draft");
        assert_eq!(a, b);
        assert_eq!(capped.evicted(), 0);
        assert_eq!(capped.compactions(), 0);
        assert_eq!(capped.phys_len(), capped.len());
    }

    #[test]
    fn kv_cap_evicts_and_keeps_sink_plus_window_kv_rows() {
        let dev = Device::Cpu;
        let scorer = synthetic_scorer_on(&dev, 512).expect("build synthetic scorer");
        let (sink, window) = (2usize, 8usize);
        let n = sink + window + DRAFTER_KV_CAP_SLACK + 34;
        let ctx = patterned_ctx(n);
        let aux = synthetic_aux(&scorer, n);
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 3usize;

        let mut plain = DrafterKvCache::new();
        let _ = scorer
            .chain_draft_cached(&mut plain, &ctx, &proj, k)
            .expect("uncapped draft");
        let mut capped = DrafterKvCache::with_kv_cap(sink, window);
        let out = scorer
            .chain_draft_cached(&mut capped, &ctx, &proj, k)
            .expect("capped draft");
        assert_eq!(out.len(), k);

        assert_eq!(capped.len(), n);
        assert!(capped.evicted() > 0, "expected eviction at n={n}");
        assert!(capped.compactions() >= 1);
        let phys = capped.phys_len();
        assert_eq!(phys + capped.evicted(), n);
        assert!(phys <= sink + window + DRAFTER_KV_CAP_SLACK);

        let rows = |t: &Tensor, from: usize, len: usize| -> Vec<f32> {
            t.narrow(1, from, len)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let plain_k = plain.k.as_ref().expect("plain k");
        let cap_k = capped.k.as_ref().expect("capped k");
        assert_eq!(rows(cap_k, 0, sink), rows(plain_k, 0, sink), "sink K rows");
        let tail = phys - sink;
        assert_eq!(
            rows(cap_k, sink, tail),
            rows(plain_k, n - tail, tail),
            "window K rows"
        );
        let plain_v = plain.v.as_ref().expect("plain v");
        let cap_v = capped.v.as_ref().expect("capped v");
        assert_eq!(rows(cap_v, 0, sink), rows(plain_v, 0, sink), "sink V rows");
        assert_eq!(
            rows(cap_v, sink, tail),
            rows(plain_v, n - tail, tail),
            "window V rows"
        );
    }

    #[test]
    fn kv_cap_incremental_shift_rounds_stay_bounded() {
        let dev = Device::Cpu;
        let scorer = synthetic_scorer_on(&dev, 512).expect("build synthetic scorer");
        let (sink, window) = (2usize, 8usize);
        let bound = sink + window + DRAFTER_KV_CAP_SLACK;
        let total = bound + 40;
        let full = patterned_ctx(total + 1);
        let aux = synthetic_aux(&scorer, total + 1);
        let proj = scorer.project_aux(&aux).expect("project");
        let k = 3usize;
        let mut cache = DrafterKvCache::with_kv_cap(sink, window);
        let mut n = bound - 6;
        while n < total {
            let ctx = &full[..n];
            let bonus = full[n];
            let proj_n = proj.narrow(0, 0, n).unwrap();
            let out = scorer
                .chain_draft_cached_cond(&mut cache, ctx, &proj_n, k, Some(bonus), true)
                .expect("capped shift round");
            assert_eq!(out.len(), k);
            assert_eq!(cache.len(), n);
            assert_eq!(cache.phys_len() + cache.evicted(), n);
            assert!(
                cache.phys_len() <= bound,
                "phys {} exceeds bound {bound} at n={n}",
                cache.phys_len()
            );
            n += 3;
        }
        assert!(cache.compactions() >= 1);
    }
}
