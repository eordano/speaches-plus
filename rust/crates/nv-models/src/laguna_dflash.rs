use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
use serde::Deserialize;

#[cfg(feature = "cuda")]
use crate::laguna::Laguna;
use crate::laguna::{attention, softplus_f32, LagunaEmbedding};

#[cfg(feature = "cuda")]
pub(crate) mod context_commit;
#[cfg(feature = "cuda")]
pub mod prefix;

#[cfg(feature = "cuda")]
use candle_core::op::BackpropOp;
#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
#[cfg(feature = "cuda")]
use candle_core::Storage;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use half::bf16;
#[cfg(feature = "cuda")]
use nv_kernels::graph::CudaGraphRunner;
use nv_quant::nvfp4::MIN_TILE;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::env;
#[cfg(feature = "cuda")]
use std::sync::Arc;

pub fn dflash_nvfp4_enabled() -> Result<bool> {
    match env::var("NV_LAGUNA_DFLASH_NVFP4") {
        Ok(v) if v == "1" => Ok(true),
        Ok(v) if v == "0" => Ok(false),
        Err(std::env::VarError::NotPresent) => Ok(false),
        _ => bail!("NV_LAGUNA_DFLASH_NVFP4 must be 0 or 1"),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DflashQuantizationSummary {
    pub weight_format: &'static str,
    pub packed_linears: usize,
    pub fp8_linears: usize,
    pub fp8_norms: usize,
    pub excluded_linears: usize,
    pub excluded_norms: usize,
    pub bf16_fallback_linears: usize,
    pub bf16act_rows: usize,
    pub bf16act_armed_linears: usize,
    pub packed_weight_bytes: usize,
    pub scale_bytes: usize,
}

pub fn dflash_graph_enabled() -> bool {
    env::var("NV_LAGUNA_DFLASH_GRAPH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub const DFLASH_ADAPT_THRESH_DEFAULT: f32 = 2.0;

pub fn dflash_adapt_enabled() -> bool {
    env::var("NV_DFLASH_ADAPT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn dflash_adapt_thresh() -> f32 {
    env::var("NV_DFLASH_ADAPT_THRESH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DFLASH_ADAPT_THRESH_DEFAULT)
}

pub fn adapt_truncate_len(conf: &[f32], thresh: f32) -> usize {
    let j = conf.iter().position(|&c| c < thresh).unwrap_or(conf.len());
    j.max(1)
}

pub const LOOKUP_MIN_MATCH_DEFAULT: usize = 4;
pub const LOOKUP_EMA_ALPHA: f64 = 0.2;
pub const LOOKUP_EMA_INIT: f64 = 2.0;

pub fn lookup_draft_enabled() -> bool {
    env::var("NV_LAGUNA_LOOKUP_DRAFT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn lookup_min_match() -> usize {
    env::var("NV_LOOKUP_MIN_MATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(LOOKUP_MIN_MATCH_DEFAULT)
        .clamp(1, 64)
}

pub fn lookup_ema_enabled() -> bool {
    env::var("NV_LAGUNA_LOOKUP_EMA")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

pub struct LookupState {
    sam: nv_lookup::SuffixAutomaton,
    min_match: usize,
    ema_guard: bool,
    ema: nv_lookup::AcceptEma,
}

impl LookupState {
    pub fn from_env() -> Option<Self> {
        lookup_draft_enabled().then(|| Self::new(lookup_min_match(), lookup_ema_enabled()))
    }

    pub fn new(min_match: usize, ema_guard: bool) -> Self {
        Self {
            sam: nv_lookup::SuffixAutomaton::new(),
            min_match: min_match.clamp(1, 64),
            ema_guard,
            ema: nv_lookup::AcceptEma::new(LOOKUP_EMA_ALPHA, LOOKUP_EMA_INIT),
        }
    }

    pub fn reset(&mut self) {
        self.sam = nv_lookup::SuffixAutomaton::new();
        self.ema = nv_lookup::AcceptEma::new(LOOKUP_EMA_ALPHA, LOOKUP_EMA_INIT);
    }

    pub fn extend(&mut self, tok: u32) {
        self.sam.extend(tok);
    }

    pub fn extend_slice(&mut self, toks: &[u32]) {
        self.sam.extend_slice(toks);
    }

    pub fn propose(&self, max_len: usize) -> Option<Vec<u32>> {
        let p = self.sam.propose(max_len, self.min_match)?;
        if !self.ema_guard
            || nv_lookup::suffix_arm_wins(
                p.tokens.len(),
                self.min_match,
                p.match_len,
                self.ema.value(),
            )
        {
            Some(p.tokens)
        } else {
            None
        }
    }

    pub fn observe_dflash_round(&mut self, accepted: usize) {
        self.ema.observe(accepted);
    }

    pub fn describe(&self) -> String {
        format!("min_match={} ema_guard={}", self.min_match, self.ema_guard)
    }
}

pub fn margins_from_part_max(part_val: &[f32], rows: usize, parts: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for &v in &part_val[r * parts..(r + 1) * parts] {
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        out.push(top1 - top2);
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapList {
    Target,
    Eagle,
}

pub fn tap_list_mode() -> TapList {
    match env::var("NV_LAGUNA_TAP_LIST") {
        Ok(v) if v.eq_ignore_ascii_case("eagle") => TapList::Eagle,
        _ => TapList::Target,
    }
}

pub fn resolve_tap_layers(cfg: &LagunaDflashConfig, mode: TapList) -> Vec<usize> {
    match mode {
        TapList::Eagle if !cfg.eagle_aux_hidden_state_layer_ids.is_empty() => {
            let n_target = cfg.dflash_config.num_target_layers;
            let mut out = Vec::with_capacity(cfg.eagle_aux_hidden_state_layer_ids.len());
            for &li in &cfg.eagle_aux_hidden_state_layer_ids {
                if li < n_target {
                    out.push(li);
                } else {
                    eprintln!(
                        "[dflash] NV_LAGUNA_TAP_LIST=eagle: dropping out-of-range aux layer {li} (target has {n_target} layers)"
                    );
                }
            }
            if out.len() != cfg.num_aux() {
                eprintln!(
                    "[dflash] NV_LAGUNA_TAP_LIST=eagle: {} taps after filtering but fc expects {}; combine_aux will reject",
                    out.len(),
                    cfg.num_aux()
                );
            }
            out
        }
        TapList::Eagle => {
            eprintln!(
                "[dflash] NV_LAGUNA_TAP_LIST=eagle but config has no eagle_aux_hidden_state_layer_ids; using target_layer_ids"
            );
            cfg.dflash_config.target_layer_ids.clone()
        }
        TapList::Target => cfg.dflash_config.target_layer_ids.clone(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormMode {
    Checkpoint,
    Reference,
}

pub fn norm_mode() -> NormMode {
    match env::var("NV_LAGUNA_NORM_MODE") {
        Ok(v) if v.eq_ignore_ascii_case("reference") => NormMode::Reference,
        _ => NormMode::Checkpoint,
    }
}

pub const VEGAS_K_DEFAULT: usize = 512;

pub fn vegas_enabled() -> bool {
    env::var("NV_SPEC_VEGAS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn vegas_k() -> usize {
    env::var("NV_SPEC_VEGAS_K")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|k| *k >= 1)
        .unwrap_or(VEGAS_K_DEFAULT)
}

#[cfg(feature = "cuda")]
fn vegas_topk_ctx_idx(
    q_rot: &Tensor,
    ctx_k: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
    top_k: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    let ctx_len = ctx_k.dims()[1];
    if ctx_len <= top_k {
        return Ok(None);
    }

    let q = q_rot
        .narrow(1, 0, 1)?
        .to_dtype(DType::F32)?
        .reshape((n_q, hd))?
        .reshape((n_q, 1, hd))?;
    let mut k = ctx_k.to_dtype(DType::F32)?.reshape((ctx_len, n_kv, hd))?;
    if n_kv != n_q {
        let factor = n_q / n_kv;
        k = k
            .unsqueeze(2)?
            .expand((ctx_len, n_kv, factor, hd))?
            .reshape((ctx_len, n_q, hd))?;
    }

    let kt = k.permute((1, 2, 0))?.contiguous()?;

    let scores = q
        .matmul(&kt)?
        .reshape((n_q, ctx_len))?
        .affine(scale as f64, 0.0)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let crit = probs.sum(0)?.to_vec1::<f32>()?;
    let mut order: Vec<usize> = (0..ctx_len).collect();
    order.sort_unstable_by(|&a, &b| {
        crit[b]
            .partial_cmp(&crit[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(top_k);
    order.sort_unstable();
    let idx: Vec<u32> = order.into_iter().map(|i| i as u32).collect();
    Ok(Some(Tensor::from_vec(idx, top_k, device)?))
}

#[derive(Clone, Debug, Deserialize)]
pub struct DflashParams {
    pub block_size: usize,
    pub mask_token_id: u32,
    pub num_target_layers: usize,
    pub target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub causal: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LagunaDflashConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub sliding_window: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    pub dflash_config: DflashParams,
    #[serde(default)]
    pub eagle_aux_hidden_state_layer_ids: Vec<usize>,
}

impl LagunaDflashConfig {
    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let cfg: LagunaDflashConfig =
            serde_json::from_str(s).context("deserialize laguna dflash config")?;
        ensure!(
            (1..=i32::MAX as usize).contains(&cfg.max_position_embeddings),
            "laguna dflash: max_position_embeddings must fit positive i32 positions, got {}",
            cfg.max_position_embeddings
        );
        if !cfg.layer_types.is_empty() {
            if cfg.layer_types.len() != cfg.num_hidden_layers {
                bail!(
                    "laguna dflash: layer_types len {} != num_hidden_layers {}",
                    cfg.layer_types.len(),
                    cfg.num_hidden_layers
                );
            }
            if cfg.layer_types.iter().any(|t| t != "sliding_attention") {
                bail!("laguna dflash: only uniform sliding_attention layers supported");
            }
        }
        if !cfg.dflash_config.causal {
            bail!("laguna dflash: non-causal drafting not supported");
        }
        let d = &cfg.dflash_config;
        if d.block_size < 2 {
            bail!(
                "laguna dflash: block_size must be >= 2, got {}",
                d.block_size
            );
        }
        if d.target_layer_ids.is_empty() {
            bail!("laguna dflash: empty target_layer_ids");
        }
        for w in d.target_layer_ids.windows(2) {
            if w[1] <= w[0] {
                bail!("laguna dflash: target_layer_ids must be ascending");
            }
        }
        Ok(cfg)
    }

    pub fn num_aux(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }
}

struct DflashLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    g_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    mlp: Mlp,
    #[cfg(feature = "cuda")]
    w8: Option<crate::laguna::LagunaAttnW8>,
}

impl DflashLayer {
    #[cfg(feature = "cuda")]
    fn w8_try(
        &self,
        pick: fn(&crate::laguna::LagunaAttnW8) -> &crate::laguna::LagunaProjW8,
        bit: u8,
        x: &Tensor,
        m: usize,
    ) -> Result<Option<Tensor>> {
        if let Some(w8) = &self.w8 {
            if !w8.cfg_slot_on(bit) {
                return Ok(None);
            }
            let p = pick(w8);
            if m >= 1 && m <= p.max_m {
                return w8.forward(p, x, m);
            }
        }
        Ok(None)
    }

    fn proj_q(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.q, crate::laguna::ATTN_W8_Q, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.q_proj.forward(x)
    }

    fn proj_k(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.k, crate::laguna::ATTN_W8_K, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.k_proj.forward(x)
    }

    fn proj_v(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.v, crate::laguna::ATTN_W8_V, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.v_proj.forward(x)
    }

    fn proj_o(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.o, crate::laguna::ATTN_W8_O, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.o_proj.forward(x)
    }
}

pub struct LagunaDflash {
    w4a16_rows: usize,
    w4a16_armed: usize,
    config: LagunaDflashConfig,
    aux_hidden_norms: Vec<RmsNorm>,
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DflashLayer>,
    norm: RmsNorm,
    rope_theta: f32,
    rope: Rope,
    dtype: DType,
    device: Device,
}

pub struct DflashCtxCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
    #[cfg(feature = "cuda")]
    ring: Option<DflashCtxRing>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DflashCtxRing {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    scratch: Tensor,
    stored: usize,
    cap: usize,
    window: usize,
    row_elems: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct CtxAppendPlan {
    input_skip: usize,
    keep: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn plan_ctx_append(stored: usize, window: usize, budget: usize, rows: usize) -> CtxAppendPlan {
    if rows >= window {
        CtxAppendPlan {
            input_skip: rows - window,
            keep: 0,
        }
    } else if stored + rows > budget {
        CtxAppendPlan {
            input_skip: 0,
            keep: stored.min(window - rows),
        }
    } else {
        CtxAppendPlan {
            input_skip: 0,
            keep: stored,
        }
    }
}

impl DflashCtxCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            len: 0,
            #[cfg(feature = "cuda")]
            ring: None,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn has_ring(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.ring.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            *l = None;
        }
        self.len = 0;
        #[cfg(feature = "cuda")]
        if let Some(r) = self.ring.as_mut() {
            r.stored = 0;
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn initialized_ring_rows_for_test(&self) -> Result<Vec<u16>> {
        let ring = self
            .ring
            .as_ref()
            .context("draft ring inspection requires ring storage")?;
        let mut values = Vec::new();
        for tensor in ring.k.iter().chain(&ring.v) {
            values.extend(
                tensor
                    .narrow(1, 0, ring.stored)?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?
                    .into_iter()
                    .map(half::bf16::to_bits),
            );
        }
        Ok(values)
    }

    fn layer_ctx(&self, li: usize) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some(r) = &self.ring {
            ensure!(r.stored > 0, "ctx ring: layer {li} empty");
            return Ok((
                r.k[li].narrow(1, 0, r.stored)?,
                r.v[li].narrow(1, 0, r.stored)?,
            ));
        }
        match &self.layers[li] {
            Some((k, v)) => Ok((k.clone(), v.clone())),
            None => bail!("ctx cache: layer {li} context missing"),
        }
    }
}

#[cfg(feature = "cuda")]
fn copy_rows_bf16(
    stream: &Arc<CudaStream>,
    src: &Tensor,
    src_row: usize,
    dst: &Tensor,
    dst_row: usize,
    n_rows: usize,
    row_elems: usize,
) -> Result<()> {
    if n_rows == 0 {
        return Ok(());
    }
    let (ss, sl) = src.storage_and_layout();
    let (ds, dl) = dst.storage_and_layout();
    ensure!(
        sl.is_contiguous() && dl.is_contiguous(),
        "ctx ring copy: non-contiguous tensor"
    );
    let s_cuda = match &*ss {
        Storage::Cuda(s) => s,
        _ => bail!("ctx ring copy: src not CUDA"),
    };
    let d_cuda = match &*ds {
        Storage::Cuda(s) => s,
        _ => bail!("ctx ring copy: dst not CUDA"),
    };
    let s_slice = s_cuda.as_cuda_slice::<bf16>()?;
    let d_slice = d_cuda.as_cuda_slice::<bf16>()?;
    let s_view = s_slice.slice(
        sl.start_offset() + src_row * row_elems..sl.start_offset() + (src_row + n_rows) * row_elems,
    );
    let d_view = d_slice.slice(
        dl.start_offset() + dst_row * row_elems..dl.start_offset() + (dst_row + n_rows) * row_elems,
    );
    let (sp, _g1) = s_view.device_ptr(stream);
    let (dp, _g2) = d_view.device_ptr(stream);
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(
            dp,
            sp,
            n_rows * row_elems * std::mem::size_of::<bf16>(),
            stream.cu_stream(),
        )
    }
    .map_err(|e| anyhow::anyhow!("ctx ring dtod: {e:?}"))
}

impl LagunaDflash {
    pub fn config(&self) -> &LagunaDflashConfig {
        &self.config
    }

    pub fn quantization_summary(&self) -> DflashQuantizationSummary {
        let mut summary = DflashQuantizationSummary {
            weight_format: "bf16",
            packed_linears: 0,
            excluded_linears: 0,
            fp8_linears: 0,
            fp8_norms: 0,
            excluded_norms: self.aux_hidden_norms.len() + 2 + 4 * self.layers.len(),
            bf16_fallback_linears: 0,
            packed_weight_bytes: 0,
            scale_bytes: 0,
            bf16act_rows: self.w4a16_rows,
            bf16act_armed_linears: self.w4a16_armed,
        };
        let linears = std::iter::once(&self.fc).chain(self.layers.iter().flat_map(|l| {
            [
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.g_proj,
                l.mlp.gate_proj(),
                l.mlp.up_proj(),
                l.mlp.down_proj(),
            ]
        }));
        for linear in linears {
            #[cfg(feature = "cuda")]
            if let Some((packed, scales, _, _)) = linear.nvfp4_parts() {
                summary.packed_linears += 1;
                summary.packed_weight_bytes += packed.len();
                summary.scale_bytes += scales.len();
                continue;
            }
            #[cfg(feature = "cuda")]
            if linear
                .fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h()
                .is_some()
            {
                summary.fp8_linears += 1;
            }
            if linear.out_features() < MIN_TILE || linear.in_features() < MIN_TILE {
                summary.excluded_linears += 1;
            } else {
                summary.bf16_fallback_linears += 1;
            }
        }
        #[cfg(feature = "cuda")]
        for norm in self
            .aux_hidden_norms
            .iter()
            .chain([&self.hidden_norm, &self.norm])
            .chain(self.layers.iter().flat_map(|layer| {
                [
                    &layer.input_layernorm,
                    &layer.post_attention_layernorm,
                    &layer.q_norm,
                    &layer.k_norm,
                ]
            }))
        {
            if norm.fp8_e4m3_active() {
                summary.fp8_norms += 1;
                summary.excluded_norms -= 1;
            }
        }
        if summary.packed_linears > 0 {
            summary.weight_format = "nvfp4-derived";
        }
        summary
    }

    pub fn attn_fp8_active(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.layers.iter().any(|l| l.w8.is_some())
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    pub fn from_loader(
        config: LagunaDflashConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        fn dflash_w4a16_rows_env() -> Result<Option<usize>> {
            match env::var("NV_DFLASH_W4A16_ROWS") {
                Err(std::env::VarError::NotPresent) => Ok(None),
                Ok(v) if v == "0" => Ok(None),
                Ok(v) => {
                    let rows: usize = v.parse().context("NV_DFLASH_W4A16_ROWS must be an integer row limit (8 = proposal M5 and catch-up M1..M5 use bf16 activations; prompt-context rows above it stay W4A4)")?;
                    ensure!(
                        (1..=8).contains(&rows),
                        "NV_DFLASH_W4A16_ROWS must be 1..=8, got {rows}"
                    );
                    Ok(Some(rows))
                }
                Err(e) => Err(e).context("NV_DFLASH_W4A16_ROWS must be valid UTF-8"),
            }
        }
        let nvfp4 = dflash_nvfp4_enabled()?;
        let fp8 = crate::laguna::weight_linears_fp8_enabled();
        ensure!(!nvfp4 || !nv_config::env_arm_present_at_all_reread_every_call("NV_DFLASH_ATTN_FP8"), "NV_LAGUNA_DFLASH_NVFP4=1 conflicts with NV_DFLASH_ATTN_FP8; packed projections must not be replaced by an FP8 copy");
        #[cfg(feature = "cuda")]
        let nvfp4_runner = if nvfp4 {
            let Device::Cuda(dev) = device else {
                bail!("NV_LAGUNA_DFLASH_NVFP4=1 requires CUDA; refusing BF16 fallback");
            };
            Some(Arc::new(std::sync::Mutex::new(
                nv_quant::nvfp4::Nvfp4GemmRunner::new(dev.cuda_stream())?,
            )))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        ensure!(
            !nvfp4,
            "NV_LAGUNA_DFLASH_NVFP4=1 requires the CUDA build; refusing BF16 fallback"
        );
        #[cfg(not(feature = "cuda"))]
        ensure!(!fp8, "NV_LAGUNA_WEIGHT_LINEARS_FP8 requires the CUDA build for DFlash; refusing BF16 fallback");
        #[cfg(feature = "cuda")]
        let fp8_runner: crate::laguna::Fp8RunnerHandle = if fp8 {
            let Device::Cuda(dev) = device else {
                bail!("NV_LAGUNA_WEIGHT_LINEARS_FP8 requires CUDA for the DFlash gate projections");
            };
            Some(Arc::new(std::sync::Mutex::new(
                nv_quant::fp8::Fp8GemmRunner::new(dev.cuda_stream().clone())?,
            )))
        } else {
            None
        };
        ensure!(
            (1..=i32::MAX as usize).contains(&config.max_position_embeddings),
            "laguna dflash: max_position_embeddings must fit positive i32 positions, got {}",
            config.max_position_embeddings
        );
        let dtype = DType::BF16;
        let h = config.hidden_size;
        let hd = config.head_dim;
        let n_q = config.num_attention_heads;
        let n_kv = config.num_key_value_heads;
        let eps = config.rms_norm_eps;
        let num_aux = config.num_aux();

        let norm1 = |name: &str, dim: usize| -> Result<RmsNorm> {
            let w = weights
                .get(name, dtype)
                .with_context(|| format!("load {name}"))?;
            let d = w.dims();
            if d != [dim] {
                bail!("rmsnorm {name}: expected [{dim}], got {d:?}");
            }
            #[cfg(feature = "cuda")]
            if fp8 {
                return RmsNorm::new_fp8_e4m3(w, eps);
            }
            Ok(RmsNorm::new(w, eps))
        };
        let w4a16_rows = dflash_w4a16_rows_env()?;
        ensure!(w4a16_rows.is_none() || nvfp4, "NV_DFLASH_W4A16_ROWS requires NV_LAGUNA_DFLASH_NVFP4=1; large draft weights must remain packed four-bit");
        let w4a16_armed = std::cell::Cell::new(0usize);
        let from_tensor = |name: &str, w: Tensor, small_gate: bool| -> Result<Linear> {
            #[cfg(feature = "cuda")]
            if let Some(runner) = &nvfp4_runner {
                if !small_gate || w.dims()[0] >= MIN_TILE {
                    let mut packed = Linear::from_bf16_quantized_nvfp4_dev(
                        &w,
                        None,
                        device,
                        runner.clone(),
                    )
                    .with_context(|| {
                        format!("pack DFlash projection {name} as NVFP4; BF16 fallback is disabled")
                    })?;
                    if let Some(rows) = w4a16_rows {
                        packed
                            .set_nvfp4_bf16act_rows_before_sharing(rows)
                            .with_context(|| format!("arm draft-only W4A16 (bf16 activations, rows<={rows}) on {name}"))?;
                        w4a16_armed.set(w4a16_armed.get() + 1);
                    }
                    return Ok(packed);
                }
            }
            #[cfg(feature = "cuda")]
            if fp8_runner.is_some() {
                return crate::laguna::load_linear_from_tensor_maybe_fp8(w, &fp8_runner, device)
                    .with_context(|| format!("load DFlash projection {name} as e4m3 row-scaled (NV_LAGUNA_WEIGHT_LINEARS_FP8)"));
            }
            let _ = (name, small_gate);
            Linear::new(w, None)
        };
        let lin = |name: &str, out: usize, inp: usize| -> Result<Linear> {
            let w = weights
                .get(name, dtype)
                .with_context(|| format!("load {name}"))?;
            let d = w.dims();
            if d != [out, inp] {
                bail!("linear {name}: expected [{out}, {inp}], got {d:?}");
            }
            from_tensor(name, w, name.ends_with(".g_proj.weight"))
        };

        let mut aux_hidden_norms = Vec::with_capacity(num_aux);
        for i in 0..num_aux {
            aux_hidden_norms.push(norm1(&format!("aux_hidden_norms.{i}.weight"), h)?);
        }
        let fc = lin("fc.weight", h, num_aux * h)?;
        let hidden_norm = norm1("hidden_norm.weight", h)?;
        let norm = norm1("norm.weight", h)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("layers.{i}");
            let qkv_name = format!("{p}.self_attn.qkv_proj.weight");
            let qkv = weights
                .get(&qkv_name, dtype)
                .with_context(|| format!("load {qkv_name}"))?;
            let qkv_dims = qkv.dims();
            let q_rows = n_q * hd;
            let kv_rows = n_kv * hd;
            if qkv_dims != [q_rows + 2 * kv_rows, h] {
                bail!(
                    "{qkv_name}: expected [{}, {h}], got {qkv_dims:?}",
                    q_rows + 2 * kv_rows
                );
            }
            let q_proj = from_tensor(
                &format!("{p}.self_attn.q_proj"),
                qkv.narrow(0, 0, q_rows)?.contiguous()?,
                false,
            )?;
            let k_proj = from_tensor(
                &format!("{p}.self_attn.k_proj"),
                qkv.narrow(0, q_rows, kv_rows)?.contiguous()?,
                false,
            )?;
            let v_proj = from_tensor(
                &format!("{p}.self_attn.v_proj"),
                qkv.narrow(0, q_rows + kv_rows, kv_rows)?.contiguous()?,
                false,
            )?;
            drop(qkv);
            let o_proj = lin(&format!("{p}.self_attn.o_proj.weight"), h, q_rows)?;
            #[cfg(feature = "cuda")]
            let w8 = if nv_config::env_arm_present_at_all_reread_every_call("NV_DFLASH_ATTN_FP8")
                && device.is_cuda()
            {
                let built = crate::laguna::LagunaAttnW8::build(
                    &q_proj, &k_proj, &v_proj, &o_proj, device, true,
                )?;
                ensure!(
                    built.is_some(),
                    "NV_DFLASH_ATTN_FP8: drafter layer {i} projections not quantizable"
                );
                built
            } else {
                None
            };
            layers.push(DflashLayer {
                input_layernorm: norm1(&format!("{p}.input_layernorm.weight"), h)?,
                post_attention_layernorm: norm1(
                    &format!("{p}.post_attention_layernorm.weight"),
                    h,
                )?,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                g_proj: lin(&format!("{p}.self_attn.g_proj.weight"), n_q, h)?,
                q_norm: norm1(&format!("{p}.self_attn.q_norm.weight"), hd)?,
                k_norm: norm1(&format!("{p}.self_attn.k_norm.weight"), hd)?,
                mlp: Mlp::new(
                    lin(
                        &format!("{p}.mlp.gate_proj.weight"),
                        config.intermediate_size,
                        h,
                    )?,
                    lin(
                        &format!("{p}.mlp.up_proj.weight"),
                        config.intermediate_size,
                        h,
                    )?,
                    lin(
                        &format!("{p}.mlp.down_proj.weight"),
                        h,
                        config.intermediate_size,
                    )?,
                )?,
                #[cfg(feature = "cuda")]
                w8,
            });
        }

        let env_theta: Option<f32> = env::var("NV_DFLASH_ROPE_THETA")
            .ok()
            .and_then(|v| v.parse().ok());
        let rope_theta = env_theta.unwrap_or(config.rope_theta);
        let rope = Rope::new(
            RopeConfig {
                head_dim: hd,
                max_seq_len: config.max_position_embeddings,
                base: rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;

        let draft = Self {
            w4a16_rows: w4a16_rows.unwrap_or(0),
            w4a16_armed: w4a16_armed.get(),
            config,
            aux_hidden_norms,
            fc,
            hidden_norm,
            layers,
            norm,
            rope_theta,
            rope,
            dtype,
            device: device.clone(),
        };
        if nvfp4 {
            let summary = draft.quantization_summary();
            ensure!(summary.packed_linears > 0 && summary.bf16_fallback_linears == 0,
                "NV_LAGUNA_DFLASH_NVFP4=1 did not produce a fully packed large-projection draft: {summary:?}");
            if fp8 {
                ensure!(summary.packed_linears == 1 + 7 * draft.layers.len()
                    && summary.fp8_linears == draft.layers.len()
                    && summary.fp8_norms == draft.aux_hidden_norms.len() + 2 + 4 * draft.layers.len()
                    && summary.excluded_norms == 0,
                    "FP8 DFlash mode must retain packed large projections, FP8 gates and all learned norms: {summary:?}");
                eprintln!("[laguna_dflash_fp8] packed_linears={} fp8_linears={} fp8_norms={} excluded_norms={}",
                    summary.packed_linears, summary.fp8_linears, summary.fp8_norms, summary.excluded_norms);
            }
        }
        Ok(draft)
    }

    pub fn rope_max_seq_len(&self) -> usize {
        self.rope.config().max_seq_len
    }

    pub(crate) fn check_rope_span(&self, start: usize, rows: usize) -> Result<()> {
        let capacity = self.rope_max_seq_len();
        ensure!(
            start.checked_add(rows).is_some_and(|end| end <= capacity),
            "dflash: positions starting at {start} with {rows} rows exceed RoPE capacity {capacity}"
        );
        Ok(())
    }

    pub fn active_rope_theta(&self) -> f32 {
        self.rope_theta
    }

    pub fn new_ctx_cache(&self) -> DflashCtxCache {
        #[allow(unused_mut)]
        let mut cache = DflashCtxCache::new(self.layers.len());
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            match self.new_ctx_ring() {
                Ok(r) => cache.ring = Some(r),
                Err(e) => eprintln!("[dflash] ctx ring alloc failed, legacy cat path: {e:#}"),
            }
        }
        cache
    }

    #[cfg(feature = "cuda")]
    fn new_ctx_ring(&self) -> Result<DflashCtxRing> {
        let n_kv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let window = self.config.sliding_window.max(1);

        let cap = window + self.config.dflash_config.block_size;
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            k.push(Tensor::zeros(
                (1usize, cap, n_kv, hd),
                self.dtype,
                &self.device,
            )?);
            v.push(Tensor::zeros(
                (1usize, cap, n_kv, hd),
                self.dtype,
                &self.device,
            )?);
        }
        let scratch = Tensor::zeros((1usize, window, n_kv, hd), self.dtype, &self.device)?;
        Ok(DflashCtxRing {
            k,
            v,
            scratch,
            stored: 0,
            cap,
            window,
            row_elems: n_kv * hd,
        })
    }

    pub fn combine_aux(&self, aux: &[Tensor]) -> Result<Tensor> {
        self.combine_aux_mode(aux, norm_mode())
    }

    pub fn combine_aux_mode(&self, aux: &[Tensor], mode: NormMode) -> Result<Tensor> {
        let num_aux = self.config.num_aux();
        if aux.len() != num_aux {
            bail!(
                "combine_aux: expected {num_aux} aux tensors, got {}",
                aux.len()
            );
        }
        let mut normed = Vec::with_capacity(num_aux);
        for (i, a) in aux.iter().enumerate() {
            let a = a.to_dtype(self.dtype)?;
            normed.push(match mode {
                NormMode::Checkpoint => self.aux_hidden_norms[i].forward(&a)?,
                NormMode::Reference => a,
            });
        }
        let refs: Vec<&Tensor> = normed.iter().collect();
        let cat = Tensor::cat(&refs, D::Minus1)?;
        let fc_out = self.fc.forward(&cat)?;
        self.hidden_norm.forward(&fc_out)
    }

    pub fn append_context(
        &self,
        cache: &mut DflashCtxCache,
        combined: &Tensor,
        positions: &[i32],
    ) -> Result<()> {
        let dims = combined.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != self.config.hidden_size {
            bail!(
                "append_context: combined must be [1, rows, {}], got {dims:?}",
                self.config.hidden_size
            );
        }
        let rows = dims[1];
        if positions.len() != rows {
            bail!(
                "append_context: positions must be [{rows}], got {:?}",
                positions.len()
            );
        }
        self.check_rope_span(cache.len, rows)?;
        ensure!(
            positions
                .iter()
                .enumerate()
                .all(|(i, &p)| p >= 0 && p as usize == cache.len + i),
            "append_context: positions must be contiguous from context length {}",
            cache.len
        );
        let positions = Tensor::from_slice(positions, rows, &self.device)?;
        #[cfg(feature = "cuda")]
        let plan = cache
            .ring
            .as_ref()
            .map(|r| plan_ctx_append(r.stored, r.window, r.window, rows));
        let mode = norm_mode();
        for (li, layer) in self.layers.iter().enumerate() {
            let (k_rot, v) = self.context_layer_kv(layer, combined, &positions, rows, mode)?;
            #[cfg(feature = "cuda")]
            if let (Some(r), Some(p)) = (cache.ring.as_ref(), plan) {
                self.ring_append_layer(r, li, &k_rot, &v, p, rows)?;
                continue;
            }
            let merged = match &cache.layers[li] {
                Some((k_old, v_old)) => (
                    Tensor::cat(&[k_old, &k_rot], 1)?,
                    Tensor::cat(&[v_old, &v], 1)?,
                ),
                None => (k_rot, v),
            };
            cache.layers[li] = Some(merged);
        }
        #[cfg(feature = "cuda")]
        if let (Some(r), Some(p)) = (cache.ring.as_mut(), plan) {
            r.stored = p.keep + (rows - p.input_skip);
        }
        cache.len += rows;
        Ok(())
    }

    fn context_layer_kv(
        &self,
        layer: &DflashLayer,
        combined: &Tensor,
        positions: &Tensor,
        rows: usize,
        mode: NormMode,
    ) -> Result<(Tensor, Tensor)> {
        let n_kv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let normed = match mode {
            NormMode::Checkpoint => layer.input_layernorm.forward(combined)?,
            NormMode::Reference => combined.clone(),
        };
        let k = layer
            .proj_k(&normed, rows)?
            .reshape((1usize, rows, n_kv, hd))?;
        let k = layer.k_norm.forward(&k)?;
        let kf = k.to_dtype(DType::F32)?;
        let (k_rot, _) = self.rope.apply(&kf, &kf, positions)?;
        let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
        let v = layer
            .proj_v(&normed, rows)?
            .reshape((1usize, rows, n_kv, hd))?
            .to_dtype(self.dtype)?
            .contiguous()?;
        Ok((k_rot, v))
    }

    #[cfg(feature = "cuda")]
    fn ring_append_layer(
        &self,
        ring: &DflashCtxRing,
        li: usize,
        k_rot: &Tensor,
        v: &Tensor,
        plan: CtxAppendPlan,
        rows: usize,
    ) -> Result<()> {
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("ctx ring append requires CUDA")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let re = ring.row_elems;
        if plan.keep > 0 && plan.keep < ring.stored {
            let src_row = ring.stored - plan.keep;
            copy_rows_bf16(
                &stream,
                &ring.k[li],
                src_row,
                &ring.scratch,
                0,
                plan.keep,
                re,
            )?;
            copy_rows_bf16(&stream, &ring.scratch, 0, &ring.k[li], 0, plan.keep, re)?;
            copy_rows_bf16(
                &stream,
                &ring.v[li],
                src_row,
                &ring.scratch,
                0,
                plan.keep,
                re,
            )?;
            copy_rows_bf16(&stream, &ring.scratch, 0, &ring.v[li], 0, plan.keep, re)?;
        }
        let take = rows - plan.input_skip;
        copy_rows_bf16(
            &stream,
            k_rot,
            plan.input_skip,
            &ring.k[li],
            plan.keep,
            take,
            re,
        )?;
        copy_rows_bf16(
            &stream,
            v,
            plan.input_skip,
            &ring.v[li],
            plan.keep,
            take,
            re,
        )?;
        Ok(())
    }

    pub fn propose(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<Vec<u32>> {
        let k = self.config.dflash_config.block_size - 1;
        self.propose_k(cache, anchor, anchor_pos, k, embed_weight, lm_head)
    }

    pub fn propose_k(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        k: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<Vec<u32>> {
        let logits = self.propose_logits(cache, anchor, anchor_pos, k, embed_weight, lm_head)?;
        let drafts: Vec<u32> = logits
            .narrow(1, 1, k)?
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_vec1()?;
        Ok(drafts)
    }

    pub fn propose_k_conf(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        k: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let logits = self.propose_logits(cache, anchor, anchor_pos, k, embed_weight, lm_head)?;
        let rows: Vec<f32> = logits.narrow(1, 1, k)?.flatten_all()?.to_vec1()?;
        let vocab = self.config.vocab_size;
        let mut drafts = Vec::with_capacity(k);
        let mut conf = Vec::with_capacity(k);
        for r in 0..k {
            let row = &rows[r * vocab..(r + 1) * vocab];
            let mut top1 = f32::NEG_INFINITY;
            let mut top2 = f32::NEG_INFINITY;
            let mut idx = 0u32;
            for (j, &v) in row.iter().enumerate() {
                if v > top1 {
                    top2 = top1;
                    top1 = v;
                    idx = j as u32;
                } else if v > top2 {
                    top2 = v;
                }
            }
            drafts.push(idx);
            conf.push(top1 - top2);
        }
        Ok((drafts, conf))
    }

    pub fn propose_logits(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        k: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<Tensor> {
        if cache.is_empty() {
            bail!("propose: empty context cache");
        }
        if anchor_pos != cache.len {
            bail!(
                "propose: anchor_pos {} != context len {} (context must be contiguous)",
                anchor_pos,
                cache.len
            );
        }
        if k == 0 || k >= self.config.dflash_config.block_size {
            bail!(
                "propose: k must be in 1..{}, got {k}",
                self.config.dflash_config.block_size
            );
        }
        let bs = k + 1;
        self.check_rope_span(anchor_pos, bs)?;
        let mask_id = self.config.dflash_config.mask_token_id;
        let h = self.config.hidden_size;
        let hd = self.config.head_dim;
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;

        let mut block_tokens = vec![mask_id; bs];
        block_tokens[0] = anchor;
        let tokens_t = Tensor::from_vec(block_tokens, bs, &self.device)?.to_dtype(DType::U32)?;
        let mut x = embed_weight
            .lookup(&tokens_t)?
            .reshape((1usize, bs, h))?
            .to_dtype(self.dtype)?;

        let block_pos: Vec<i32> = (0..bs).map(|i| (anchor_pos + i) as i32).collect();
        let pos_t = Tensor::from_vec(block_pos, bs, &self.device)?;

        let scale = (hd as f32).powf(-0.5);
        #[cfg(feature = "cuda")]
        let vegas = vegas_enabled();
        #[cfg(feature = "cuda")]
        let vegas_budget = vegas_k();
        #[cfg(feature = "cuda")]
        let mut vegas_idx: Option<Tensor> = None;
        for (li, layer) in self.layers.iter().enumerate() {
            let normed = layer.input_layernorm.forward(&x)?;
            let q = layer.proj_q(&normed, bs)?.reshape((1usize, bs, n_q, hd))?;
            let q = layer.q_norm.forward(&q)?;
            let k = layer.proj_k(&normed, bs)?.reshape((1usize, bs, n_kv, hd))?;
            let k = layer.k_norm.forward(&k)?;
            let v = layer.proj_v(&normed, bs)?.reshape((1usize, bs, n_kv, hd))?;
            let (q_rot, k_rot) =
                self.rope
                    .apply(&q.to_dtype(DType::F32)?, &k.to_dtype(DType::F32)?, &pos_t)?;
            let q_rot = q_rot.to_dtype(self.dtype)?.contiguous()?;
            let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
            let v = v.to_dtype(self.dtype)?.contiguous()?;

            let (ctx_k, ctx_v) = cache.layer_ctx(li)?;
            #[cfg(feature = "cuda")]
            let (ctx_k, ctx_v) = if vegas {
                if vegas_idx.is_none() {
                    vegas_idx = vegas_topk_ctx_idx(
                        &q_rot,
                        &ctx_k,
                        n_q,
                        n_kv,
                        hd,
                        scale,
                        vegas_budget,
                        &self.device,
                    )?;
                }
                match &vegas_idx {
                    Some(idx) => (ctx_k.index_select(idx, 1)?, ctx_v.index_select(idx, 1)?),
                    None => (ctx_k, ctx_v),
                }
            } else {
                let stored = ctx_k.dims()[1];
                let w = self.config.sliding_window.min(stored);
                (
                    ctx_k.narrow(1, stored - w, w)?,
                    ctx_v.narrow(1, stored - w, w)?,
                )
            };
            #[cfg(not(feature = "cuda"))]
            let (ctx_k, ctx_v) = {
                let stored = ctx_k.dims()[1];
                let w = self.config.sliding_window.min(stored);
                (
                    ctx_k.narrow(1, stored - w, w)?,
                    ctx_v.narrow(1, stored - w, w)?,
                )
            };
            let k_full = Tensor::cat(&[&ctx_k, &k_rot], 1)?;
            let v_full = Tensor::cat(&[&ctx_v, &v], 1)?;

            let attn_out = attention(&q_rot, &k_full, &v_full, n_q, n_kv, hd, bs, scale, None)?;

            let g = softplus_f32(&layer.g_proj.forward(&normed)?.to_dtype(DType::F32)?)?;
            let g = g
                .reshape((1usize, bs, n_q, 1usize))?
                .to_dtype(attn_out.dtype())?;
            let gated = attn_out.broadcast_mul(&g)?;
            let attn_flat = gated.reshape((1usize, bs, n_q * hd))?;
            let o = layer.proj_o(&attn_flat, bs)?;
            let after_attn = x.add(&o.to_dtype(x.dtype())?)?;

            let normed_mlp = layer.post_attention_layernorm.forward(&after_attn)?;
            let ffn = layer.mlp.forward(&normed_mlp)?;
            x = after_attn.add(&ffn.to_dtype(after_attn.dtype())?)?;
        }

        let final_hidden = self.norm.forward(&x)?;
        lm_head
            .forward(&final_hidden)?
            .to_dtype(DType::F32)
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    fn forward_block_tree(
        &self,
        ring: &DflashCtxRing,
        tokens_t: &Tensor,
        pos_t: &Tensor,
        mask_dev: &CudaSlice<u8>,
        committed_dev: &CudaSlice<i32>,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<Tensor> {
        let bs = tokens_t.dims()[0];
        let h = self.config.hidden_size;
        let hd = self.config.head_dim;
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let scale = (hd as f32).powf(-0.5);
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("forward_block_tree requires CUDA")
        };
        let device = self.device.clone();

        let mut x = embed_weight
            .lookup_captured(tokens_t, &device)?
            .reshape((1usize, bs, h))?;

        for (li, layer) in self.layers.iter().enumerate() {
            let normed = layer.input_layernorm.forward(&x)?;
            let q = layer.proj_q(&normed, bs)?.reshape((1usize, bs, n_q, hd))?;
            let q = layer.q_norm.forward(&q)?;
            let kb = layer.proj_k(&normed, bs)?.reshape((1usize, bs, n_kv, hd))?;
            let kb = layer.k_norm.forward(&kb)?;
            let vb = layer
                .proj_v(&normed, bs)?
                .reshape((1usize, bs, n_kv, hd))?
                .contiguous()?;
            let (q_rot, k_rot) = self.rope.apply(&q, &kb, pos_t)?;

            let q_scaled = crate::gemma4::scale_bf16_op(&q_rot, scale, &device)?;

            let attn_out = tree_attention_bf16_op(
                &q_scaled,
                &k_rot,
                &vb,
                &ring.k[li],
                &ring.v[li],
                committed_dev,
                mask_dev,
                n_q,
                n_kv,
                hd,
                bs,
                &dev,
            )?;

            let g = layer.g_proj.forward(&normed)?;
            let gated = softplus_gate_bf16_op(&attn_out, &g, n_q, hd, bs, &dev)?;
            let o = layer.proj_o(&gated, bs)?;
            let after_attn = crate::gemma4::residual_add_scale_bf16_op(&x, &o, 1.0, &device)?;

            let normed_mlp = layer.post_attention_layernorm.forward(&after_attn)?;
            let ffn = layer.mlp.forward_fused_cuda(&normed_mlp)?;
            x = crate::gemma4::residual_add_scale_bf16_op(&after_attn, &ffn, 1.0, &device)?;
        }

        let final_hidden = self.norm.forward(&x)?;
        let raw = lm_head.forward(&final_hidden)?;
        crate::gemma4::tanh_softcap_bf16_to_f32_op(&raw, 0.0, &device)?
            .reshape((1usize, bs, self.config.vocab_size))
            .map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
fn tree_attention_bf16_op(
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    ring_k: &Tensor,
    ring_v: &Tensor,
    committed_dev: &CudaSlice<i32>,
    mask_dev: &CudaSlice<u8>,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    bs: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let q_c = q.reshape((bs, n_q * hd))?.contiguous()?;
    let k_c = k_new.reshape((bs, n_kv * hd))?.contiguous()?;
    let v_c = v_new.reshape((bs, n_kv * hd))?.contiguous()?;
    let mut out_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(bs * n_q * hd)? };
    let rc = {
        let (qs, ql) = q_c.storage_and_layout();
        let (ks, kl) = k_c.storage_and_layout();
        let (vs, vl) = v_c.storage_and_layout();
        let (rks, rkl) = ring_k.storage_and_layout();
        let (rvs, rvl) = ring_v.storage_and_layout();
        macro_rules! cuda_slice {
            ($st:expr) => {
                match &*$st {
                    candle_core::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                    _ => anyhow::bail!("tree attention op: expected CUDA storage"),
                }
            };
        }
        let q_sl = cuda_slice!(qs);
        let k_sl = cuda_slice!(ks);
        let v_sl = cuda_slice!(vs);
        let rk_sl = cuda_slice!(rks);
        let rv_sl = cuda_slice!(rvs);
        let elem = std::mem::size_of::<bf16>() as u64;
        let (qp0, _gq) = q_sl.device_ptr(&stream);
        let (kp0, _gk) = k_sl.device_ptr(&stream);
        let (vp0, _gv) = v_sl.device_ptr(&stream);
        let (rkp0, _grk) = rk_sl.device_ptr(&stream);
        let (rvp0, _grv) = rv_sl.device_ptr(&stream);
        let qp = qp0 + ql.start_offset() as u64 * elem;
        let kp = kp0 + kl.start_offset() as u64 * elem;
        let vp = vp0 + vl.start_offset() as u64 * elem;
        let rkp = rkp0 + rkl.start_offset() as u64 * elem;
        let rvp = rvp0 + rvl.start_offset() as u64 * elem;
        let (ncp, _g1) = committed_dev.device_ptr(&stream);
        let (mp, _g2) = mask_dev.device_ptr(&stream);
        let (op, _g3) = out_dev.device_ptr_mut(&stream);
        let rc1 = unsafe {
            nv_kernels::cuda::kv_append_bf16(
                stream.cu_stream() as *mut std::ffi::c_void,
                kp as *const u16,
                vp as *const u16,
                rkp as *mut u16,
                rvp as *mut u16,
                ncp as *const i32,
                bs as i32,
                n_kv as i32,
                hd as i32,
            )
        };
        if rc1 != 0 {
            rc1
        } else {
            unsafe {
                nv_kernels::cuda::tree_verify_attn_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    qp as *const u16,
                    rkp as *const u16,
                    rvp as *const u16,
                    ncp as *const i32,
                    mp as *const u8,
                    std::ptr::null::<i32>(),
                    op as *mut u16,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    bs as i32,
                    0,
                )
            }
        }
    };
    ensure!(rc == 0, "dflash tree attention kernels rc={rc}");
    let storage = CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        (1usize, bs, n_q * hd),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn softplus_gate_bf16_op(
    attn: &Tensor,
    gate: &Tensor,
    n_q: usize,
    hd: usize,
    bs: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let a_c = attn.contiguous()?;
    let g_c = gate.contiguous()?;
    ensure!(
        a_c.elem_count() == bs * n_q * hd && g_c.elem_count() == bs * n_q,
        "softplus gate op: shape mismatch"
    );
    let mut out_dev: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(bs * n_q * hd)? };
    let rc = {
        let (as_, al) = a_c.storage_and_layout();
        let (gs, gl) = g_c.storage_and_layout();
        let a_cuda = match &*as_ {
            Storage::Cuda(s) => s,
            _ => bail!("softplus gate op: attn not CUDA"),
        };
        let g_cuda = match &*gs {
            Storage::Cuda(s) => s,
            _ => bail!("softplus gate op: gate not CUDA"),
        };
        let a_slice = a_cuda.as_cuda_slice::<bf16>()?;
        let g_slice = g_cuda.as_cuda_slice::<bf16>()?;
        let a_view = a_slice.slice(al.start_offset()..al.start_offset() + bs * n_q * hd);
        let g_view = g_slice.slice(gl.start_offset()..gl.start_offset() + bs * n_q);
        let (ap, _g1) = a_view.device_ptr(&stream);
        let (gp, _g2) = g_view.device_ptr(&stream);
        let (op, _g3) = out_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::softplus_gate_bf16(
                stream.cu_stream() as *mut std::ffi::c_void,
                ap as *const u16,
                gp as *const u16,
                op as *mut u16,
                (bs * n_q) as i32,
                hd as i32,
            )
        }
    };
    ensure!(rc == 0, "softplus_gate_bf16 rc={rc}");
    let storage = CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        (1usize, bs, n_q * hd),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
pub struct DflashGraphProposer {
    forked: Arc<CudaStream>,
    runner: CudaGraphRunner,
    bs: usize,
    cap: usize,
    tok_buf: CudaSlice<u32>,
    pos_buf: CudaSlice<i32>,
    mask_buf: CudaSlice<u8>,
    committed_buf: CudaSlice<i32>,
    out_buf: CudaSlice<u32>,
    dummy_drafts: CudaSlice<u32>,
    accept_out: CudaSlice<u32>,
    part_val: CudaSlice<f32>,
    part_idx: CudaSlice<i32>,
    host_toks: Vec<u32>,
    host_pos: Vec<i32>,
    host_committed: Box<[i32; 1]>,
    parts: usize,
    collect_conf: bool,
    last_conf: Option<Vec<f32>>,
    recorded_logits: Option<CudaSlice<f32>>,
    captured: bool,
}

#[cfg(feature = "cuda")]
impl DflashGraphProposer {
    pub fn new(draft: &LagunaDflash, num_speculative: usize) -> Result<Self> {
        ensure!(
            num_speculative >= 1 && num_speculative < draft.config.dflash_config.block_size,
            "dflash graph proposer: speculative count {num_speculative} must be in 1..{}",
            draft.config.dflash_config.block_size
        );
        let Device::Cuda(dev) = draft.device.clone() else {
            bail!("dflash graph proposer requires a CUDA device")
        };
        let bs = num_speculative + 1;
        let window = draft.config.sliding_window.max(1);
        let cap = window + draft.config.dflash_config.block_size;
        let parts = nv_kernels::cuda::dflash_accept_parts();
        ensure!(parts > 0, "dflash accept kernel unavailable");
        let raw_ctx = dev.cuda_stream().context().clone();
        crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(&raw_ctx);
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("dflash graph stream: {e:?}"))?;
        let tok_buf = forked.alloc_zeros::<u32>(bs)?;
        let pos_buf = forked.alloc_zeros::<i32>(bs)?;
        let mut mask_buf = forked.alloc_zeros::<u8>(bs * bs)?;
        let committed_buf = forked.alloc_zeros::<i32>(1)?;
        let out_buf = forked.alloc_zeros::<u32>(bs - 1)?;
        let dummy_drafts = forked.alloc_zeros::<u32>(bs - 1)?;
        let accept_out = forked.alloc_zeros::<u32>(bs)?;
        let part_val = forked.alloc_zeros::<f32>((bs - 1) * parts)?;
        let part_idx = forked.alloc_zeros::<i32>((bs - 1) * parts)?;

        let mut mask_host = vec![0u8; bs * bs];
        for i in 0..bs {
            for j in 0..=i {
                mask_host[i * bs + j] = 1;
            }
        }
        forked
            .memcpy_htod(&mask_host[..], &mut mask_buf)
            .map_err(|e| anyhow::anyhow!("mask htod: {e:?}"))?;
        forked.synchronize()?;
        let runner = CudaGraphRunner::new(forked.clone());
        let mask_id = draft.config.dflash_config.mask_token_id;
        ctor_guard.the_built_engine_owns_teardown_now();
        Ok(Self {
            forked,
            runner,
            bs,
            cap,
            tok_buf,
            pos_buf,
            mask_buf,
            committed_buf,
            out_buf,
            dummy_drafts,
            accept_out,
            part_val,
            part_idx,
            host_toks: vec![mask_id; bs],
            host_pos: vec![0i32; bs],
            host_committed: Box::new([0i32; 1]),
            parts,
            collect_conf: dflash_adapt_enabled(),
            last_conf: None,
            recorded_logits: None,
            captured: false,
        })
    }

    pub fn new_recording_logits(draft: &LagunaDflash, num_speculative: usize) -> Result<Self> {
        let mut proposer = Self::new(draft, num_speculative)?;
        proposer.recorded_logits = Some(
            proposer
                .forked
                .alloc_zeros::<f32>(proposer.bs * draft.config.vocab_size)
                .map_err(|e| anyhow::anyhow!("allocate draft diagnostic logits: {e:?}"))?,
        );
        Ok(proposer)
    }

    pub fn recorded_logits_host(&self) -> Result<Vec<f32>> {
        ensure!(self.captured, "no captured draft logits are available");
        let logits = self
            .recorded_logits
            .as_ref()
            .context("construct the diagnostic proposer with new_recording_logits")?;
        self.forked.synchronize()?;
        let values = self.forked.clone_dtoh(logits);
        values.map_err(|e| anyhow::anyhow!("draft diagnostic logits dtoh: {e:?}"))
    }

    pub fn last_conf(&self) -> Option<&[f32]> {
        self.last_conf.as_deref()
    }

    pub(crate) fn out_buf(&self) -> &CudaSlice<u32> {
        &self.out_buf
    }

    pub fn propose(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<Vec<u32>> {
        self.propose_inner(
            draft,
            cache,
            anchor,
            anchor_pos,
            embed_weight,
            lm_head,
            true,
        )
    }

    pub(crate) fn propose_device(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
    ) -> Result<usize> {
        self.propose_inner(
            draft,
            cache,
            anchor,
            anchor_pos,
            embed_weight,
            lm_head,
            false,
        )?;
        Ok(self.bs - 1)
    }

    fn prepare_inputs<'a>(
        &mut self,
        draft: &LagunaDflash,
        cache: &'a DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
    ) -> Result<&'a DflashCtxRing> {
        let ring = cache
            .ring
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("graph propose requires the ring ctx cache"))?;
        ensure!(
            ring.cap == self.cap,
            "graph propose: ring cap {} != proposer cap {}",
            ring.cap,
            self.cap
        );
        ensure!(
            ring.stored + self.bs <= ring.cap,
            "graph propose: no headroom (stored {}, cap {})",
            ring.stored,
            ring.cap
        );
        ensure!(ring.stored > 0, "graph propose: empty context");
        ensure!(
            anchor_pos == cache.len,
            "graph propose: anchor_pos {} != context len {}",
            anchor_pos,
            cache.len
        );
        draft.check_rope_span(anchor_pos, self.bs)?;
        self.host_toks[0] = anchor;
        for (i, p) in self.host_pos.iter_mut().enumerate() {
            *p = (anchor_pos + i) as i32;
        }
        self.host_committed[0] = ring.stored as i32;
        Ok(ring)
    }

    fn propose_inner(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &dyn LagunaEmbedding,
        lm_head: &Linear,
        want_dtoh: bool,
    ) -> Result<Vec<u32>> {
        let ring = self.prepare_inputs(draft, cache, anchor, anchor_pos)?;
        let Device::Cuda(dev) = draft.device.clone() else {
            bail!("cuda")
        };

        let legacy = dev.cuda_stream();
        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured;
        let forked = self.forked.clone();
        let vocab = draft.config.vocab_size;
        let DflashGraphProposer {
            runner,
            bs,
            tok_buf,
            pos_buf,
            mask_buf,
            committed_buf,
            out_buf,
            dummy_drafts,
            accept_out,
            part_val,
            part_idx,
            host_toks,
            host_pos,
            host_committed,
            recorded_logits,
            ..
        } = self;
        let bs = *bs;
        let dev2 = dev.clone();
        let mut body = |s: &Arc<CudaStream>| -> Result<()> {
            nv_layers::cuda_stream::with_stream(s.clone(), || -> Result<()> {
                s.memcpy_htod(&host_toks[..], tok_buf)
                    .map_err(|e| anyhow::anyhow!("htod toks: {e:?}"))?;
                s.memcpy_htod(&host_pos[..], pos_buf)
                    .map_err(|e| anyhow::anyhow!("htod pos: {e:?}"))?;
                s.memcpy_htod(&host_committed[..], committed_buf)
                    .map_err(|e| anyhow::anyhow!("htod committed: {e:?}"))?;
                let tc = tok_buf.try_clone()?;
                let pc = pos_buf.try_clone()?;
                let toks_t = {
                    let st = CudaStorage::wrap_cuda_slice(tc, dev2.clone());
                    Tensor::from_storage(Storage::Cuda(st), (bs,), BackpropOp::none(), false)
                };
                let pos_t = {
                    let st = CudaStorage::wrap_cuda_slice(pc, dev2.clone());
                    Tensor::from_storage(Storage::Cuda(st), (bs,), BackpropOp::none(), false)
                };
                let logits = draft.forward_block_tree(
                    ring,
                    &toks_t,
                    &pos_t,
                    mask_buf,
                    committed_buf,
                    embed_weight,
                    lm_head,
                )?;
                ensure!(
                    logits.dtype() == DType::F32 && logits.dims() == [1, bs, vocab],
                    "graph propose: unexpected logits {:?} {:?}",
                    logits.dims(),
                    logits.dtype()
                );
                let lc = logits.contiguous()?;
                let (ls, ll) = lc.storage_and_layout();
                let l_cuda = match &*ls {
                    Storage::Cuda(st) => st,
                    _ => bail!("graph logits not CUDA"),
                };
                let l_slice = l_cuda.as_cuda_slice::<f32>()?;
                if let Some(recorded) = recorded_logits.as_mut() {
                    let src = l_slice.slice(ll.start_offset()..ll.start_offset() + bs * vocab);
                    s.memcpy_dtod(&src, recorded)
                        .map_err(|e| anyhow::anyhow!("record captured draft logits: {e:?}"))?;
                }
                let (lp0, _gl) = l_slice.device_ptr(s);

                let lp = lp0 + ((ll.start_offset() + vocab) * std::mem::size_of::<f32>()) as u64;
                let (dp, _g1) = dummy_drafts.device_ptr(s);
                let (rp, _g2) = out_buf.device_ptr_mut(s);
                let (op, _g3) = accept_out.device_ptr_mut(s);
                let (pv, _g4) = part_val.device_ptr_mut(s);
                let (pi, _g5) = part_idx.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::dflash_accept_f32(
                        s.cu_stream() as *mut std::ffi::c_void,
                        lp as *const f32,
                        dp as *const u32,
                        rp as *mut u32,
                        op as *mut u32,
                        pv as *mut f32,
                        pi as *mut i32,
                        (bs - 1) as i32,
                        vocab as i32,
                    )
                };
                ensure!(rc == 0, "graph draft argmax rc={rc}");
                Ok(())
            })
        };

        if !was_captured {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            body(&forked).context("dflash graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }
        runner
            .run_on(bs as u64, Some(&legacy), |s| body(s))
            .context("dflash graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
        }
        self.captured = true;
        if !want_dtoh {
            return Ok(Vec::new());
        }
        let out: Vec<u32> = legacy
            .clone_dtoh(&self.out_buf)
            .map_err(|e| anyhow::anyhow!("drafts dtoh: {e:?}"))?;
        if self.collect_conf {
            let pv: Vec<f32> = legacy
                .clone_dtoh(&self.part_val)
                .map_err(|e| anyhow::anyhow!("part_val dtoh: {e:?}"))?;
            self.last_conf = Some(margins_from_part_max(&pv, self.bs - 1, self.parts));
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
impl Drop for DflashGraphProposer {
    fn drop(&mut self) {
        let td = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked);
        let runner = &mut self.runner;
        td.run(|| runner.invalidate());
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct AcceptSlots {
    drafts: CudaSlice<u32>,
    row_argmax: CudaSlice<u32>,
    out: CudaSlice<u32>,
    part_val: CudaSlice<f32>,
    part_idx: CudaSlice<i32>,
    prof_ts: Option<CudaSlice<u64>>,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct DflashInvalidLogits {
    row: usize,
}

#[cfg(feature = "cuda")]
impl std::fmt::Display for DflashInvalidLogits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "engine busy: DFlash verification logits are invalid at row {}; try later.",
            self.row
        )
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for DflashInvalidLogits {}

#[cfg(feature = "cuda")]
pub(crate) fn checked_target_argmax_row(row: &[f32]) -> Option<u32> {
    if row.len() > u32::MAX as usize {
        return None;
    }
    let mut best = f32::NEG_INFINITY;
    let mut best_id = 0u32;
    for (index, &value) in row.iter().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            return None;
        }
        if value > best {
            best = value;
            best_id = index as u32;
        }
    }
    best.is_finite().then_some(best_id)
}

#[cfg(feature = "cuda")]
pub(crate) fn accept_block_on_host(vlogits: &Tensor, drafts: &[u32]) -> Result<(usize, Vec<u32>)> {
    let k = drafts.len();
    let m = k
        .checked_add(1)
        .context("DFlash acceptance row count overflow")?;
    let dims = vlogits.dims();
    ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == m && dims[2] > 0,
        "host DFlash acceptance requires [1,{m},vocab>0], got {dims:?}"
    );
    let vocab = dims[2];
    ensure!(
        vocab <= i32::MAX as usize && vlogits.dtype() == DType::F32,
        "host DFlash acceptance requires F32 logits and vocab <= INT_MAX"
    );
    let logits: Vec<f32> = vlogits.flatten_all()?.to_vec1()?;
    let mut vtoks = Vec::with_capacity(m);
    for (row_index, row) in logits.chunks_exact(vocab).enumerate() {
        vtoks.push(checked_target_argmax_row(row).ok_or(DflashInvalidLogits { row: row_index })?);
    }
    let accepted = drafts
        .iter()
        .zip(&vtoks)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut emitted = drafts[..accepted].to_vec();
    emitted.push(vtoks[accepted]);
    Ok((accepted, emitted))
}

#[cfg(feature = "cuda")]
pub(crate) fn accept_block_on_device_with_status(
    target: &Laguna,
    slots_map: &mut HashMap<usize, AcceptSlots>,
    vlogits: &Tensor,
    drafts: &[u32],
    device_drafts: Option<u64>,
    xqa_status_device: Option<u64>,
) -> Result<(usize, Vec<u32>)> {
    accept_block_on_device_for_vocab(
        target.device(),
        target.config().vocab_size,
        slots_map,
        vlogits,
        drafts,
        device_drafts,
        xqa_status_device,
    )
}

#[cfg(feature = "cuda")]
fn accept_block_on_device_for_vocab(
    device: &Device,
    vocab: usize,
    slots_map: &mut HashMap<usize, AcceptSlots>,
    vlogits: &Tensor,
    drafts: &[u32],
    device_drafts: Option<u64>,
    xqa_status_device: Option<u64>,
) -> Result<(usize, Vec<u32>)> {
    let Device::Cuda(dev) = device.clone() else {
        bail!("accept_on_device requires a CUDA device")
    };
    let stream = dev.cuda_stream();
    let k = drafts.len();
    let m = k + 1;
    let parts = nv_kernels::cuda::dflash_accept_parts();
    ensure!(parts > 0, "dflash accept kernel unavailable");
    let out_words = m + 1 + usize::from(xqa_status_device.is_some());
    if let std::collections::hash_map::Entry::Vacant(entry) = slots_map.entry(k) {
        let prof_ts = if crate::laguna_step_graph::verify_prof_enabled() {
            Some(stream.alloc_zeros::<u64>(2)?)
        } else {
            None
        };
        entry.insert(AcceptSlots {
            drafts: stream.alloc_zeros::<u32>(k)?,
            row_argmax: stream.alloc_zeros::<u32>(m)?,
            out: stream.alloc_zeros::<u32>(out_words)?,
            part_val: stream.alloc_zeros::<f32>(m * parts)?,
            part_idx: stream.alloc_zeros::<i32>(m * parts)?,
            prof_ts,
        });
    }
    let slots = slots_map.get_mut(&k).unwrap();
    if slots.out.len() < out_words {
        slots.out = stream.alloc_zeros::<u32>(out_words)?;
    }
    ensure!(
        slots.drafts.len() == k && slots.row_argmax.len() == m,
        "accept slots sized for a different k"
    );
    if device_drafts.is_none() {
        stream
            .memcpy_htod(drafts, &mut slots.drafts)
            .map_err(|e| anyhow::anyhow!("accept drafts htod: {e:?}"))?;
    }

    let logits_c = vlogits.contiguous()?;
    let dims = logits_c.dims();
    ensure!(
        dims == [1, m, vocab],
        "accept_on_device: logits must be [1, {m}, {vocab}], got {dims:?}"
    );
    ensure!(logits_c.dtype() == DType::F32, "accept logits must be f32");
    let (ls, ll) = logits_c.storage_and_layout();
    let l_cuda = match &*ls {
        Storage::Cuda(s) => s,
        _ => bail!("accept logits must be CUDA"),
    };
    let l_slice = l_cuda.as_cuda_slice::<f32>()?;
    {
        let (lp, _g0) = l_slice.device_ptr(&stream);
        let lp = lp + (ll.start_offset() * std::mem::size_of::<f32>()) as u64;
        let (dp_own, _g1) = slots.drafts.device_ptr(&stream);
        let dp = device_drafts.unwrap_or(dp_own);
        let (rp, _g2) = slots.row_argmax.device_ptr_mut(&stream);
        let (op, _g3) = slots.out.device_ptr_mut(&stream);
        let (pv, _g4) = slots.part_val.device_ptr_mut(&stream);
        let (pi, _g5) = slots.part_idx.device_ptr_mut(&stream);
        if let Some(ts) = slots.prof_ts.as_ref() {
            crate::laguna_step_graph::prof_timestamp_at(ts, 0, &stream)?;
        }
        let rc = unsafe {
            nv_kernels::cuda::dflash_accept_f32(
                stream.cu_stream() as *mut std::ffi::c_void,
                lp as *const f32,
                dp as *const u32,
                rp as *mut u32,
                op as *mut u32,
                pv as *mut f32,
                pi as *mut i32,
                m as i32,
                vocab as i32,
            )
        };
        ensure!(rc == 0, "dflash_accept_f32 rc={rc}");
        if let Some(ts) = slots.prof_ts.as_ref() {
            crate::laguna_step_graph::prof_timestamp_at(ts, 1, &stream)?;
        }
    }
    if let Some(status_device) = xqa_status_device {
        ensure!(
            status_device != 0,
            "engine busy: XQA status pointer missing; try later."
        );
        let mut dst = slots.out.slice_mut(m + 1..m + 2);
        let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst_ptr,
                status_device,
                std::mem::size_of::<u32>(),
                stream.cu_stream(),
            )
        }
        .map_err(|e| anyhow::anyhow!("engine busy: XQA status copy failed; try later: {e:?}"))?;
    }
    let out_host: Vec<u32> = stream
        .clone_dtoh(&slots.out.slice(0..out_words))
        .map_err(|e| anyhow::anyhow!("accept out dtoh: {e:?}"))?;
    if xqa_status_device.is_some() {
        ensure!(
            out_host[m + 1] == 0,
            "engine busy: XQA verification status {} (invalid committed span); try later.",
            out_host[m + 1]
        );
    }
    if let Some(ts) = slots.prof_ts.as_ref() {
        let t: Vec<u64> = stream
            .clone_dtoh(ts)
            .map_err(|e| anyhow::anyhow!("accept prof dtoh: {e:?}"))?;
        let ms = t[1].saturating_sub(t[0]) as f64 / 1e6;
        eprintln!("[laguna_verify_prof] accept={ms:.3}");
    }
    if out_host[0] == u32::MAX {
        ensure!(
            (out_host[1] as usize) < m,
            "DFlash acceptance invalid-row status exceeds M"
        );
        return Err(DflashInvalidLogits {
            row: out_host[1] as usize,
        }
        .into());
    }
    let accepted = out_host[0] as usize;
    ensure!(accepted <= k, "accept count {accepted} > k {k}");
    let emitted: Vec<u32> = out_host[1..2 + accepted].to_vec();
    Ok((accepted, emitted))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAFT_CONFIG: &str = r#"{
      "attention_bias": false,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "intermediate_size": 8192,
      "max_position_embeddings": 262144,
      "model_type": "laguna",
      "num_attention_heads": 64,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_theta": 500000.0,
      "sliding_window": 512,
      "vocab_size": 100352,
      "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention"
      ],
      "gating": "per-head",
      "architectures": ["DFlashLagunaForCausalLM"],
      "draft_vocab_size": 100352,
      "torch_dtype": "bfloat16",
      "eagle_aux_hidden_state_layer_ids": [2, 14, 26, 34, 40],
      "dflash_config": {
        "block_size": 16,
        "mask_token_id": 12,
        "num_target_layers": 40,
        "target_layer_ids": [1, 13, 25, 33, 39],
        "causal": true
      },
      "num_experts": 0
    }"#;

    #[test]
    fn config_rejects_non_causal_or_mixed_layers_and_resolves_tap_lists() {
        let non_causal = DRAFT_CONFIG.replace("\"causal\": true", "\"causal\": false");
        assert!(LagunaDflashConfig::from_hf_json_str(&non_causal).is_err());
        let mixed = DRAFT_CONFIG.replacen("sliding_attention", "full_attention", 1);
        assert!(LagunaDflashConfig::from_hf_json_str(&mixed).is_err());
        let cfg = LagunaDflashConfig::from_hf_json_str(DRAFT_CONFIG).unwrap();
        assert_eq!(
            resolve_tap_layers(&cfg, TapList::Target),
            vec![1, 13, 25, 33, 39]
        );
        assert_eq!(
            resolve_tap_layers(&cfg, TapList::Eagle),
            vec![2, 14, 26, 34]
        );
        let no_eagle = DRAFT_CONFIG.replace(
            "\"eagle_aux_hidden_state_layer_ids\": [2, 14, 26, 34, 40],",
            "",
        );
        let cfg2 = LagunaDflashConfig::from_hf_json_str(&no_eagle).unwrap();
        assert_eq!(
            resolve_tap_layers(&cfg2, TapList::Eagle),
            vec![1, 13, 25, 33, 39]
        );
        assert_eq!(
            resolve_tap_layers(&cfg2, TapList::Target),
            vec![1, 13, 25, 33, 39]
        );
    }

    #[test]
    fn ctx_append_plan_adapt_truncate_and_part_max_margins_follow_their_host_arithmetic() {
        let (w, cap) = (512usize, 768usize);
        for (stored, rows, input_skip, keep) in [
            (100, 8, 0, 100),
            (760, 16, 0, 496),
            (0, 700, 188, 0),
            (300, 512, 0, 0),
            (600, 300, 0, 212),
            (100, 480, 0, 100),
            (400, 480, 0, 32),
        ] {
            assert_eq!(
                plan_ctx_append(stored, w, cap, rows),
                CtxAppendPlan { input_skip, keep },
                "stored {stored} rows {rows}"
            );
        }
        let mut stored = 0usize;
        let seq = [
            1usize, 5, 8, 3, 256, 240, 512, 600, 7, 7, 7, 250, 250, 250, 4, 1023, 2, 2,
        ];
        for _ in 0..4 {
            for &rows in &seq {
                let p = plan_ctx_append(stored, w, cap, rows);
                assert!(p.keep <= stored, "keep {} > stored {}", p.keep, stored);
                assert!(p.input_skip <= rows);
                let take = rows - p.input_skip;
                let next = p.keep + take;
                assert!(next <= cap, "stored {next} > cap after rows {rows}");

                assert!(next >= w.min(stored + rows));
                stored = next;
            }
        }
        assert_eq!(adapt_truncate_len(&[5.0, 4.0, 3.0, 2.0], 1.0), 4);
        assert_eq!(adapt_truncate_len(&[5.0, 4.0, 0.5, 2.0], 1.0), 2);
        assert_eq!(adapt_truncate_len(&[0.2, 4.0, 3.0, 2.0], 1.0), 1);
        assert_eq!(adapt_truncate_len(&[0.2, 0.1, 0.0, 0.0], 1.0), 1);
        assert_eq!(adapt_truncate_len(&[5.0], 1.0), 1);
        assert_eq!(adapt_truncate_len(&[5.0, 4.0], 10.0), 1);
        assert_eq!(adapt_truncate_len(&[5.0, 4.0], 0.0), 2);
        let pv = [1.0f32, 7.0, 3.0, 2.0, -1.0, -4.0, -2.0, -8.0];
        let m = margins_from_part_max(&pv, 2, 4);
        assert_eq!(m.len(), 2);
        assert!((m[0] - 4.0).abs() < 1e-6);
        assert!((m[1] - 1.0).abs() < 1e-6);
    }

    const TINY_CONFIG: &str = r#"{
      "head_dim": 2,
      "hidden_size": 4,
      "intermediate_size": 8,
      "max_position_embeddings": 64,
      "num_attention_heads": 2,
      "num_hidden_layers": 0,
      "num_key_value_heads": 1,
      "rms_norm_eps": 1e-06,
      "rope_theta": 500000.0,
      "sliding_window": 8,
      "vocab_size": 16,
      "eagle_aux_hidden_state_layer_ids": [1, 2],
      "dflash_config": {
        "block_size": 2,
        "mask_token_id": 3,
        "num_target_layers": 2,
        "target_layer_ids": [0, 1],
        "causal": true
      }
    }"#;

    fn tiny_dflash(dev: &Device) -> LagunaDflash {
        let h = 4usize;
        let config = LagunaDflashConfig::from_hf_json_str(TINY_CONFIG).unwrap();
        let w2 = Tensor::full(2.0f32, (h,), dev).unwrap();
        let w1 = Tensor::ones((h,), DType::F32, dev).unwrap();
        let fc_w = Tensor::arange(0f32, (h * 2 * h) as f32, dev)
            .unwrap()
            .reshape((h, 2 * h))
            .unwrap();
        LagunaDflash {
            w4a16_rows: 0,
            w4a16_armed: 0,
            config,
            aux_hidden_norms: vec![RmsNorm::new(w2.clone(), 1e-6), RmsNorm::new(w2, 1e-6)],
            fc: Linear::new(fc_w, None).unwrap(),
            hidden_norm: RmsNorm::new(w1.clone(), 1e-6),
            layers: vec![],
            norm: RmsNorm::new(w1, 1e-6),
            rope_theta: 1.0e4,
            rope: Rope::new(
                RopeConfig {
                    head_dim: 2,
                    max_seq_len: 1,
                    base: 1.0e4,
                    kind: RopeKind::Standard,
                },
                dev,
            )
            .unwrap(),
            dtype: DType::F32,
            device: dev.clone(),
        }
    }

    fn loaded_tiny_dflash_with_capacity(capacity: usize) -> LagunaDflash {
        let mut config = LagunaDflashConfig::from_hf_json_str(TINY_CONFIG).unwrap();
        config.max_position_embeddings = capacity;
        config.head_dim = 8;
        let dir =
            std::env::temp_dir().join(format!("laguna-rope-{}-{capacity}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut weights = std::collections::HashMap::<String, Tensor>::new();
        for name in [
            "aux_hidden_norms.0.weight",
            "aux_hidden_norms.1.weight",
            "hidden_norm.weight",
            "norm.weight",
        ] {
            weights.insert(
                name.into(),
                Tensor::ones(4, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        weights.insert(
            "fc.weight".into(),
            Tensor::ones((4, 8), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&weights, dir.join("model.safetensors")).unwrap();
        let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
        let draft = LagunaDflash::from_loader(config, &loader, &Device::Cpu).unwrap();
        drop(loader);
        std::fs::remove_dir_all(&dir).unwrap();
        draft
    }

    #[test]
    fn declared_rope_capacity_preserves_prefix_and_covers_positions_above_64k() {
        let old = loaded_tiny_dflash_with_capacity(65536);
        let draft = loaded_tiny_dflash_with_capacity(262144);
        assert_eq!(draft.rope_max_seq_len(), 262144);
        for (name, old_table, full_table) in [
            ("cos", old.rope.cos(), draft.rope.cos()),
            ("sin", old.rope.sin(), draft.rope.sin()),
        ] {
            let before = old_table.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let after = full_table
                .narrow(0, 0, 65536)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                before
                    .iter()
                    .zip(&after)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "extending {name} changed existing frequencies"
            );
        }
        let input = Tensor::from_vec(
            vec![1f32, 1., 1., 1., 0., 0., 0., 0.],
            (1, 1, 1, 8),
            &Device::Cpu,
        )
        .unwrap();
        let mut cache = DflashCtxCache::new(0);
        let combined = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        for position in [65535usize, 65536, 168000, 262143] {
            cache.len = position;
            draft
                .append_context(&mut cache, &combined, &[position as i32])
                .unwrap();
            assert_eq!(cache.len(), position + 1);
            draft.check_rope_span(position, 1).unwrap();
            let positions = Tensor::from_vec(vec![position as u32], 1, &Device::Cpu).unwrap();
            let (rotated, _) = draft.rope.apply(&input, &input, &positions).unwrap();
            let rotated = rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                (rotated[0] - (position as f32).cos()).abs() < 1e-6,
                "cos at {position}"
            );
            assert!(
                (rotated[4] - (position as f32).sin()).abs() < 1e-6,
                "sin at {position}"
            );
        }
        for (start, rows) in [(262144, 1), (262143, 2), (usize::MAX, 1)] {
            assert!(draft
                .check_rope_span(start, rows)
                .unwrap_err()
                .to_string()
                .contains("RoPE capacity"));
        }
        cache.len = 262144;
        assert!(draft
            .append_context(&mut cache, &combined, &[262144])
            .unwrap_err()
            .to_string()
            .contains("RoPE capacity"));
        assert_eq!(cache.len(), 262144);
        cache.len = 0;
        for positions in [&[-1][..], &[1][..], &[262144][..]] {
            assert!(draft
                .append_context(&mut cache, &combined, positions)
                .unwrap_err()
                .to_string()
                .contains("contiguous"));
            assert_eq!(cache.len(), 0);
        }
        draft.append_context(&mut cache, &combined, &[0]).unwrap();
        assert_eq!(cache.len(), 1);
        assert!(LagunaDflashConfig::from_hf_json_str(&TINY_CONFIG.replace(
            "\"max_position_embeddings\": 64",
            "\"max_position_embeddings\": 0"
        ))
        .is_err());
    }

    #[test]
    fn combine_aux_reference_mode_skips_aux_norms() {
        let dev = Device::Cpu;
        let d = tiny_dflash(&dev);
        let a0 = Tensor::new(&[[[1f32, 2., 3., 4.]]], &dev).unwrap();
        let a1 = Tensor::new(&[[[40f32, 30., 20., 10.]]], &dev).unwrap();
        let aux = [a0.clone(), a1.clone()];
        let ck = d.combine_aux_mode(&aux, NormMode::Checkpoint).unwrap();
        let rf = d.combine_aux_mode(&aux, NormMode::Reference).unwrap();
        let cat = Tensor::cat(&[&a0, &a1], candle_core::D::Minus1).unwrap();
        let expect = d.hidden_norm.forward(&d.fc.forward(&cat).unwrap()).unwrap();
        let rf_v: Vec<f32> = rf.flatten_all().unwrap().to_vec1().unwrap();
        let ex_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in rf_v.iter().zip(&ex_v) {
            assert!(
                (a - b).abs() < 1e-5,
                "reference mode != single-hidden_norm path"
            );
        }
        let ck_v: Vec<f32> = ck.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            ck_v.iter().zip(&rf_v).any(|(a, b)| (a - b).abs() > 1e-3),
            "checkpoint mode should differ when aux norms are non-trivial"
        );
    }

    #[test]
    fn lookup_env_knobs_threshold_arm_ema_guard_and_reset() {
        unsafe { std::env::remove_var("NV_LAGUNA_LOOKUP_DRAFT") };
        unsafe { std::env::remove_var("NV_LOOKUP_MIN_MATCH") };
        unsafe { std::env::remove_var("NV_LAGUNA_LOOKUP_EMA") };
        assert!(!lookup_draft_enabled());
        assert!(LookupState::from_env().is_none());
        assert_eq!(lookup_min_match(), LOOKUP_MIN_MATCH_DEFAULT);
        assert!(lookup_ema_enabled());
        unsafe { std::env::set_var("NV_LAGUNA_LOOKUP_DRAFT", "1") };
        unsafe { std::env::set_var("NV_LOOKUP_MIN_MATCH", "0") };
        unsafe { std::env::set_var("NV_LAGUNA_LOOKUP_EMA", "0") };
        assert!(lookup_draft_enabled());
        assert!(LookupState::from_env().is_some());
        assert_eq!(lookup_min_match(), 1);
        assert!(!lookup_ema_enabled());
        unsafe { std::env::set_var("NV_LOOKUP_MIN_MATCH", "999") };
        assert_eq!(lookup_min_match(), 64);
        unsafe { std::env::set_var("NV_LOOKUP_MIN_MATCH", "bogus") };
        assert_eq!(lookup_min_match(), LOOKUP_MIN_MATCH_DEFAULT);
        unsafe { std::env::set_var("NV_LAGUNA_LOOKUP_DRAFT", "0") };
        assert!(!lookup_draft_enabled());
        unsafe { std::env::remove_var("NV_LAGUNA_LOOKUP_DRAFT") };
        unsafe { std::env::remove_var("NV_LOOKUP_MIN_MATCH") };
        unsafe { std::env::remove_var("NV_LAGUNA_LOOKUP_EMA") };

        let mut l = LookupState::new(2, false);
        assert!(l.propose(4).is_none());
        l.extend_slice(&[10, 20, 30, 40, 10, 20]);
        let p = l.propose(4).unwrap();
        assert_eq!(p, vec![30, 40, 10, 20]);
        assert_eq!(l.propose(2).unwrap(), vec![30, 40]);
        let mut strict = LookupState::new(3, false);
        strict.extend_slice(&[10, 20, 30, 40, 10, 20]);
        assert!(strict.propose(4).is_none());

        let mut l = LookupState::new(1, true);
        l.extend_slice(&[7, 1, 2, 3, 4, 5, 6, 8, 9, 7]);
        assert!(l.propose(1).is_none());
        assert!(l.propose(8).is_some());
        for _ in 0..64 {
            l.observe_dflash_round(0);
        }
        assert_eq!(l.propose(1).unwrap(), vec![1]);
        for _ in 0..64 {
            l.observe_dflash_round(6);
        }
        assert!(l.propose(4).is_none());
        assert_eq!(l.propose(8).unwrap(), vec![1, 2, 3, 4, 5, 6, 8, 9]);

        let mut l = LookupState::new(1, true);
        l.extend_slice(&[10, 20, 30, 40, 10, 20]);
        for _ in 0..64 {
            l.observe_dflash_round(0);
        }
        assert!(l.propose(4).is_some());
        l.reset();
        assert!(l.propose(4).is_none());
        assert!((l.ema.value() - LOOKUP_EMA_INIT).abs() < 1e-12);
    }
}
