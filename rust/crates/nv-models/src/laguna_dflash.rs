use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
use serde::Deserialize;

use crate::laguna::{attention, softplus_f32, Laguna};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use half::bf16;
#[cfg(feature = "cuda")]
use nv_kernels::graph::CudaGraphRunner;
#[cfg(feature = "cuda")]
use std::sync::Arc;

pub const ROPE_TABLE_CAP: usize = 65536;

pub const DFLASH_CODE_THETA: f32 = 1.0e4;
pub const DFLASH_PROSE_THETA: f32 = 5.0e5;

pub fn dflash_graph_enabled() -> bool {
    std::env::var("NV_LAGUNA_DFLASH_GRAPH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub const DFLASH_ADAPT_THRESH_DEFAULT: f32 = 2.0;

pub fn dflash_adapt_enabled() -> bool {
    std::env::var("NV_DFLASH_ADAPT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn dflash_adapt_thresh() -> f32 {
    std::env::var("NV_DFLASH_ADAPT_THRESH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DFLASH_ADAPT_THRESH_DEFAULT)
}

pub fn adapt_truncate_len(conf: &[f32], thresh: f32) -> usize {
    let j = conf.iter().position(|&c| c < thresh).unwrap_or(conf.len());
    j.max(1)
}

pub const SPEC_ENTROPY_TAU_DEFAULT: f32 = 0.10;

pub fn spec_entropy_stop_enabled() -> bool {
    std::env::var("NV_SPEC_ENTROPY_STOP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn spec_entropy_tau() -> f32 {
    std::env::var("NV_SPEC_ENTROPY_TAU")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(SPEC_ENTROPY_TAU_DEFAULT)
}

pub fn spec_entropy_cap(fixed_k: usize, block_size: usize) -> usize {
    let ceil = block_size.saturating_sub(1).max(1);
    let dflt = (2 * fixed_k).clamp(1, ceil);
    std::env::var("NV_SPEC_ENTROPY_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|c| c.clamp(1, ceil))
        .unwrap_or(dflt)
}

pub fn entropy_stop_len(top1: &[f32], tau: f32) -> usize {
    let mut cum = 1.0f32;
    let mut len = 0usize;
    for &p in top1 {
        cum *= p.clamp(0.0, 1.0);
        len += 1;
        if cum < tau {
            break;
        }
    }
    len.max(1)
}

pub const LOOKUP_MIN_MATCH_DEFAULT: usize = 4;
pub const LOOKUP_EMA_ALPHA: f64 = 0.2;
pub const LOOKUP_EMA_INIT: f64 = 2.0;

pub fn lookup_draft_enabled() -> bool {
    std::env::var("NV_LAGUNA_LOOKUP_DRAFT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn lookup_min_match() -> usize {
    std::env::var("NV_LOOKUP_MIN_MATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(LOOKUP_MIN_MATCH_DEFAULT)
        .clamp(1, 64)
}

pub fn lookup_ema_enabled() -> bool {
    std::env::var("NV_LAGUNA_LOOKUP_EMA")
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
pub enum PromptClass {
    Code,
    Prose,
}

impl PromptClass {
    pub fn theta(self) -> f32 {
        match self {
            PromptClass::Code => DFLASH_CODE_THETA,
            PromptClass::Prose => DFLASH_PROSE_THETA,
        }
    }

    pub fn default_k(self) -> usize {
        match self {
            PromptClass::Code => DFLASH_DEFAULT_K,
            PromptClass::Prose => DFLASH_DEFAULT_K,
        }
    }
}

pub const DFLASH_DEFAULT_K: usize = 15;

pub fn default_num_speculative(block_size: usize) -> usize {
    DFLASH_DEFAULT_K.min(block_size.saturating_sub(1)).max(1)
}

pub fn classify_prompt(text: &str) -> PromptClass {
    const KEYWORDS: [&str; 14] = [
        "```",
        "fn ",
        "def ",
        "class ",
        "function",
        "struct",
        "impl ",
        "return",
        "import ",
        "#include",
        "programme",
        "program",
        "code",
        "script",
    ];
    let lower = text.to_lowercase();
    let mut score = 0i32;
    for kw in KEYWORDS {
        if lower.contains(kw) {
            score += 1;
        }
    }
    for lang in [
        "python",
        "rust",
        "javascript",
        "typescript",
        "c++",
        "java",
        "golang",
        " sql",
    ] {
        if lower.contains(lang) {
            score += 2;
        }
    }
    for line in text.lines() {
        if line.starts_with("    ") || line.starts_with('\t') {
            score += 1;
        }
    }
    let symbolic = text
        .chars()
        .filter(|c| matches!(c, '{' | '}' | ';' | '='))
        .count();
    if !text.is_empty() && symbolic * 100 > text.chars().count() {
        score += 2;
    }
    if score >= 2 {
        PromptClass::Code
    } else {
        PromptClass::Prose
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeThetaPolicy {
    Fixed,
    Auto,
    Depth,
}

pub fn rope_theta_policy() -> RopeThetaPolicy {
    match std::env::var("NV_DFLASH_ROPE_THETA_POLICY") {
        Ok(v) if v.eq_ignore_ascii_case("auto") => RopeThetaPolicy::Auto,
        Ok(v) if v.eq_ignore_ascii_case("depth") => RopeThetaPolicy::Depth,
        _ => RopeThetaPolicy::Fixed,
    }
}

pub fn rope_theta_policy_auto() -> bool {
    rope_theta_policy() == RopeThetaPolicy::Auto
}

pub const DFLASH_DEPTH_THETA_CTX_DEFAULT: usize = 8192;

pub fn rope_theta_depth_ctx() -> usize {
    std::env::var("NV_DFLASH_ROPE_THETA_DEPTH_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DFLASH_DEPTH_THETA_CTX_DEFAULT)
}

pub fn depth_gated_theta(class: PromptClass, ctx_tokens: usize, depth_ctx: usize) -> f32 {
    if class == PromptClass::Code && ctx_tokens <= depth_ctx {
        DFLASH_CODE_THETA
    } else {
        DFLASH_PROSE_THETA
    }
}

pub fn resolve_rope_thetas(env_override: Option<f32>, config_theta: f32, auto: bool) -> Vec<f32> {
    let mut thetas = vec![env_override.unwrap_or(config_theta)];
    if auto {
        for t in [config_theta, DFLASH_CODE_THETA, DFLASH_PROSE_THETA] {
            if !thetas.iter().any(|&x| theta_eq(x, t)) {
                thetas.push(t);
            }
        }
    }
    thetas
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapList {
    Target,
    Eagle,
}

pub fn tap_list_mode() -> TapList {
    match std::env::var("NV_LAGUNA_TAP_LIST") {
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
    match std::env::var("NV_LAGUNA_NORM_MODE") {
        Ok(v) if v.eq_ignore_ascii_case("reference") => NormMode::Reference,
        _ => NormMode::Checkpoint,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DflashWindowMode {
    Strict,
    Relaxed,
}

pub fn dflash_window_mode() -> DflashWindowMode {
    match std::env::var("NV_DFLASH_WINDOW_MODE") {
        Ok(v) if v.eq_ignore_ascii_case("strict") => DflashWindowMode::Strict,
        _ => DflashWindowMode::Relaxed,
    }
}

pub const VEGAS_K_DEFAULT: usize = 512;

pub fn vegas_enabled() -> bool {
    std::env::var("NV_SPEC_VEGAS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn vegas_k() -> usize {
    std::env::var("NV_SPEC_VEGAS_K")
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
        if !cfg.layer_types.is_empty() {
            if cfg.layer_types.len() != cfg.num_hidden_layers {
                anyhow::bail!(
                    "laguna dflash: layer_types len {} != num_hidden_layers {}",
                    cfg.layer_types.len(),
                    cfg.num_hidden_layers
                );
            }
            if cfg.layer_types.iter().any(|t| t != "sliding_attention") {
                anyhow::bail!("laguna dflash: only uniform sliding_attention layers supported");
            }
        }
        if !cfg.dflash_config.causal {
            anyhow::bail!("laguna dflash: non-causal drafting not supported");
        }
        let d = &cfg.dflash_config;
        if d.block_size < 2 {
            anyhow::bail!(
                "laguna dflash: block_size must be >= 2, got {}",
                d.block_size
            );
        }
        if d.target_layer_ids.is_empty() {
            anyhow::bail!("laguna dflash: empty target_layer_ids");
        }
        for w in d.target_layer_ids.windows(2) {
            if w[1] <= w[0] {
                anyhow::bail!("laguna dflash: target_layer_ids must be ascending");
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
    config: LagunaDflashConfig,
    aux_hidden_norms: Vec<RmsNorm>,
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DflashLayer>,
    norm: RmsNorm,
    ropes: Vec<(f32, Rope)>,
    rope_active: std::sync::atomic::AtomicUsize,
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

fn theta_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 * a.abs().max(b.abs())
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

    fn layer_ctx(&self, li: usize) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some(r) = &self.ring {
            anyhow::ensure!(r.stored > 0, "ctx ring: layer {li} empty");
            return Ok((
                r.k[li].narrow(1, 0, r.stored)?,
                r.v[li].narrow(1, 0, r.stored)?,
            ));
        }
        match &self.layers[li] {
            Some((k, v)) => Ok((k.clone(), v.clone())),
            None => anyhow::bail!("ctx cache: layer {li} context missing"),
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
    anyhow::ensure!(
        sl.is_contiguous() && dl.is_contiguous(),
        "ctx ring copy: non-contiguous tensor"
    );
    let s_cuda = match &*ss {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("ctx ring copy: src not CUDA"),
    };
    let d_cuda = match &*ds {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("ctx ring copy: dst not CUDA"),
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
                anyhow::bail!("rmsnorm {name}: expected [{dim}], got {d:?}");
            }
            Ok(RmsNorm::new(w, eps))
        };
        let lin = |name: &str, out: usize, inp: usize| -> Result<Linear> {
            let w = weights
                .get(name, dtype)
                .with_context(|| format!("load {name}"))?;
            let d = w.dims();
            if d != [out, inp] {
                anyhow::bail!("linear {name}: expected [{out}, {inp}], got {d:?}");
            }
            Linear::new(w, None)
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
                anyhow::bail!(
                    "{qkv_name}: expected [{}, {h}], got {qkv_dims:?}",
                    q_rows + 2 * kv_rows
                );
            }
            let q_proj = Linear::new(qkv.narrow(0, 0, q_rows)?.contiguous()?, None)?;
            let k_proj = Linear::new(qkv.narrow(0, q_rows, kv_rows)?.contiguous()?, None)?;
            let v_proj = Linear::new(
                qkv.narrow(0, q_rows + kv_rows, kv_rows)?.contiguous()?,
                None,
            )?;
            let o_proj = lin(&format!("{p}.self_attn.o_proj.weight"), h, q_rows)?;
            #[cfg(feature = "cuda")]
            let w8 = if std::env::var_os("NV_DFLASH_ATTN_FP8").is_some()
                && matches!(device, Device::Cuda(_))
            {
                let built = crate::laguna::LagunaAttnW8::build(
                    &q_proj, &k_proj, &v_proj, &o_proj, device, true,
                )?;
                anyhow::ensure!(
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

        let env_theta: Option<f32> = std::env::var("NV_DFLASH_ROPE_THETA")
            .ok()
            .and_then(|v| v.parse().ok());
        let thetas = resolve_rope_thetas(
            env_theta,
            config.rope_theta,
            rope_theta_policy() != RopeThetaPolicy::Fixed,
        );
        let max_seq = config.max_position_embeddings.min(ROPE_TABLE_CAP);
        let mut ropes = Vec::with_capacity(thetas.len());
        for t in thetas {
            let r = Rope::new(
                RopeConfig {
                    head_dim: hd,
                    max_seq_len: max_seq,
                    base: t,
                    kind: RopeKind::Standard,
                },
                device,
            )?;
            ropes.push((t, r));
        }

        Ok(Self {
            config,
            aux_hidden_norms,
            fc,
            hidden_norm,
            layers,
            norm,
            ropes,
            rope_active: std::sync::atomic::AtomicUsize::new(0),
            dtype,
            device: device.clone(),
        })
    }

    fn rope(&self) -> &Rope {
        let i = self.rope_active.load(std::sync::atomic::Ordering::Relaxed);
        &self.ropes[i].1
    }

    pub fn active_rope_theta(&self) -> f32 {
        let i = self.rope_active.load(std::sync::atomic::Ordering::Relaxed);
        self.ropes[i].0
    }

    pub fn available_rope_thetas(&self) -> Vec<f32> {
        self.ropes.iter().map(|(t, _)| *t).collect()
    }

    pub fn select_rope_theta(&self, theta: f32) -> Result<f32> {
        for (i, (t, _)) in self.ropes.iter().enumerate() {
            if theta_eq(*t, theta) {
                self.rope_active
                    .store(i, std::sync::atomic::Ordering::Relaxed);
                return Ok(*t);
            }
        }
        anyhow::bail!(
            "dflash: no rope table for theta {theta} (available: {:?}; set NV_DFLASH_ROPE_THETA_POLICY=auto at load)",
            self.available_rope_thetas()
        )
    }

    pub fn select_for_prompt(&self, text: &str) -> PromptClass {
        self.select_for_prompt_ctx(text, text.len() / 4)
    }

    pub fn select_for_prompt_ctx(&self, text: &str, ctx_tokens: usize) -> PromptClass {
        let class = classify_prompt(text);
        let theta = match rope_theta_policy() {
            RopeThetaPolicy::Fixed => return class,
            RopeThetaPolicy::Auto => class.theta(),
            RopeThetaPolicy::Depth => depth_gated_theta(class, ctx_tokens, rope_theta_depth_ctx()),
        };
        if let Err(e) = self.select_rope_theta(theta) {
            eprintln!("[dflash] theta policy select failed, keeping current: {e:#}");
        }
        class
    }

    pub fn new_ctx_cache(&self) -> DflashCtxCache {
        #[allow(unused_mut)]
        let mut cache = DflashCtxCache::new(self.layers.len());
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::Cuda(_))
            && std::env::var_os("NV_DFLASH_HOST_CTX").is_none()
        {
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
            anyhow::bail!(
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
        let cat = Tensor::cat(&refs, candle_core::D::Minus1)?;
        let fc_out = self.fc.forward(&cat)?;
        self.hidden_norm.forward(&fc_out)
    }

    pub fn append_context(
        &self,
        cache: &mut DflashCtxCache,
        combined: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        let dims = combined.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != self.config.hidden_size {
            anyhow::bail!(
                "append_context: combined must be [1, rows, {}], got {dims:?}",
                self.config.hidden_size
            );
        }
        let rows = dims[1];
        if positions.dims() != [rows] {
            anyhow::bail!(
                "append_context: positions must be [{rows}], got {:?}",
                positions.dims()
            );
        }
        let n_kv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        #[cfg(feature = "cuda")]
        let plan = cache
            .ring
            .as_ref()
            .map(|r| plan_ctx_append(r.stored, r.window, r.window, rows));
        let mode = norm_mode();
        for (li, layer) in self.layers.iter().enumerate() {
            let normed = match mode {
                NormMode::Checkpoint => layer.input_layernorm.forward(combined)?,
                NormMode::Reference => combined.clone(),
            };
            let k = layer
                .proj_k(&normed, rows)?
                .reshape((1usize, rows, n_kv, hd))?;
            let k = layer.k_norm.forward(&k)?;
            let kf = k.to_dtype(DType::F32)?;
            let (k_rot, _) = self.rope().apply(&kf, &kf, positions)?;
            let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
            let v = layer
                .proj_v(&normed, rows)?
                .reshape((1usize, rows, n_kv, hd))?
                .to_dtype(self.dtype)?
                .contiguous()?;
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
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("ctx ring append requires CUDA"),
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
        embed_weight: &Tensor,
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
        embed_weight: &Tensor,
        lm_head: &Linear,
    ) -> Result<Vec<u32>> {
        let logits = self.propose_logits(cache, anchor, anchor_pos, k, embed_weight, lm_head)?;
        let drafts: Vec<u32> = logits
            .narrow(1, 1, k)?
            .argmax(candle_core::D::Minus1)?
            .flatten_all()?
            .to_vec1()?;
        Ok(drafts)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn propose_k_conf(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        k: usize,
        embed_weight: &Tensor,
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

    #[allow(clippy::too_many_arguments)]
    pub fn propose_k_logconf(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        k: usize,
        embed_weight: &Tensor,
        lm_head: &Linear,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let logits = self.propose_logits(cache, anchor, anchor_pos, k, embed_weight, lm_head)?;
        let logit_rows = logits.narrow(1, 1, k)?;

        let drafts: Vec<u32> = logit_rows
            .argmax(candle_core::D::Minus1)?
            .flatten_all()?
            .to_vec1()?;
        let rows: Vec<f32> = logit_rows.flatten_all()?.to_vec1()?;
        let vocab = self.config.vocab_size;
        let mut top1_prob = Vec::with_capacity(k);
        for r in 0..k {
            let row = &rows[r * vocab..(r + 1) * vocab];
            let mut max = f32::NEG_INFINITY;
            for &v in row {
                if v > max {
                    max = v;
                }
            }

            let mut denom = 0.0f32;
            for &v in row {
                denom += (v - max).exp();
            }
            let p = if denom > 0.0 { 1.0 / denom } else { 1.0 };
            top1_prob.push(p);
        }
        Ok((drafts, top1_prob))
    }

    fn propose_logits(
        &self,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        k: usize,
        embed_weight: &Tensor,
        lm_head: &Linear,
    ) -> Result<Tensor> {
        if cache.is_empty() {
            anyhow::bail!("propose: empty context cache");
        }
        if anchor_pos != cache.len {
            anyhow::bail!(
                "propose: anchor_pos {} != context len {} (context must be contiguous)",
                anchor_pos,
                cache.len
            );
        }
        if k == 0 || k >= self.config.dflash_config.block_size {
            anyhow::bail!(
                "propose: k must be in 1..{}, got {k}",
                self.config.dflash_config.block_size
            );
        }
        let bs = k + 1;
        let mask_id = self.config.dflash_config.mask_token_id;
        let h = self.config.hidden_size;
        let hd = self.config.head_dim;
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;

        let mut block_tokens = vec![mask_id; bs];
        block_tokens[0] = anchor;
        let tokens_t = Tensor::from_vec(block_tokens, bs, &self.device)?.to_dtype(DType::U32)?;
        let mut x = embed_weight
            .index_select(&tokens_t, 0)?
            .reshape((1usize, bs, h))?
            .to_dtype(self.dtype)?;

        let block_pos: Vec<i32> = (0..bs).map(|i| (anchor_pos + i) as i32).collect();
        let pos_t = Tensor::from_vec(block_pos, bs, &self.device)?;

        let scale = (hd as f32).powf(-0.5);
        let relaxed = dflash_window_mode() == DflashWindowMode::Relaxed;
        let window = if relaxed {
            None
        } else {
            Some(self.config.sliding_window)
        };

        #[cfg(feature = "cuda")]
        let vegas = vegas_enabled();
        #[cfg(feature = "cuda")]
        let vegas_budget = vegas_k();
        #[cfg(feature = "cuda")]
        let mut vegas_idx: Option<Tensor> = None;
        #[cfg(feature = "cuda")]
        let mut vegas_window = window;
        for (li, layer) in self.layers.iter().enumerate() {
            let normed = layer.input_layernorm.forward(&x)?;
            let q = layer.proj_q(&normed, bs)?.reshape((1usize, bs, n_q, hd))?;
            let q = layer.q_norm.forward(&q)?;
            let k = layer.proj_k(&normed, bs)?.reshape((1usize, bs, n_kv, hd))?;
            let k = layer.k_norm.forward(&k)?;
            let v = layer.proj_v(&normed, bs)?.reshape((1usize, bs, n_kv, hd))?;
            let (q_rot, k_rot) =
                self.rope()
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
                    vegas_window = None;
                }
                match &vegas_idx {
                    Some(idx) => (ctx_k.index_select(idx, 1)?, ctx_v.index_select(idx, 1)?),
                    None => (ctx_k, ctx_v),
                }
            } else if relaxed {
                let stored = ctx_k.dims()[1];
                let w = self.config.sliding_window.min(stored);
                (
                    ctx_k.narrow(1, stored - w, w)?,
                    ctx_v.narrow(1, stored - w, w)?,
                )
            } else {
                (ctx_k, ctx_v)
            };
            #[cfg(not(feature = "cuda"))]
            let (ctx_k, ctx_v) = if relaxed {
                let stored = ctx_k.dims()[1];
                let w = self.config.sliding_window.min(stored);
                (
                    ctx_k.narrow(1, stored - w, w)?,
                    ctx_v.narrow(1, stored - w, w)?,
                )
            } else {
                (ctx_k, ctx_v)
            };
            let k_full = Tensor::cat(&[&ctx_k, &k_rot], 1)?;
            let v_full = Tensor::cat(&[&ctx_v, &v], 1)?;

            #[cfg(feature = "cuda")]
            let attn_window = vegas_window;
            #[cfg(not(feature = "cuda"))]
            let attn_window = window;
            let attn_out = attention(
                &q_rot,
                &k_full,
                &v_full,
                n_q,
                n_kv,
                hd,
                bs,
                scale,
                attn_window,
            )?;

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
    #[allow(clippy::too_many_arguments)]
    fn forward_block_tree(
        &self,
        ring: &DflashCtxRing,
        tokens_t: &Tensor,
        pos_t: &Tensor,
        mask_dev: &CudaSlice<u8>,
        committed_dev: &CudaSlice<i32>,
        embed_weight: &Tensor,
        lm_head: &Linear,
    ) -> Result<Tensor> {
        let bs = tokens_t.dims()[0];
        let h = self.config.hidden_size;
        let hd = self.config.head_dim;
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let scale = (hd as f32).powf(-0.5);
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("forward_block_tree requires CUDA"),
        };
        let device = self.device.clone();

        let mut x = crate::gemma4::embed_lookup_bf16_op(embed_weight, tokens_t, &device)?
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
            let (q_rot, k_rot) = self.rope().apply(&q, &kb, pos_t)?;

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
#[allow(clippy::too_many_arguments)]
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
    let mut out_dev: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(bs * n_q * hd)
            .map_err(|e| anyhow::anyhow!(e))?
    };
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
    anyhow::ensure!(rc == 0, "dflash tree attention kernels rc={rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (1usize, bs, n_q * hd),
        candle_core::op::BackpropOp::none(),
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
    anyhow::ensure!(
        a_c.elem_count() == bs * n_q * hd && g_c.elem_count() == bs * n_q,
        "softplus gate op: shape mismatch"
    );
    let mut out_dev: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(bs * n_q * hd)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (as_, al) = a_c.storage_and_layout();
        let (gs, gl) = g_c.storage_and_layout();
        let a_cuda = match &*as_ {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("softplus gate op: attn not CUDA"),
        };
        let g_cuda = match &*gs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("softplus gate op: gate not CUDA"),
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
    anyhow::ensure!(rc == 0, "softplus_gate_bf16 rc={rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (1usize, bs, n_q * hd),
        candle_core::op::BackpropOp::none(),
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
    captured: bool,
}

#[cfg(feature = "cuda")]
impl DflashGraphProposer {
    pub fn new(draft: &LagunaDflash, num_speculative: usize) -> Result<Self> {
        let dev = match &draft.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("dflash graph proposer requires a CUDA device"),
        };
        let bs = num_speculative + 1;
        let window = draft.config.sliding_window.max(1);
        let cap = window + draft.config.dflash_config.block_size;
        let parts = nv_kernels::cuda::dflash_accept_parts();
        anyhow::ensure!(parts > 0, "dflash accept kernel unavailable");
        let raw_ctx = dev.cuda_stream().context().clone();
        crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(&raw_ctx);
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("dflash graph stream: {e:?}"))?;
        let tok_buf = forked
            .alloc_zeros::<u32>(bs)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pos_buf = forked
            .alloc_zeros::<i32>(bs)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut mask_buf = forked
            .alloc_zeros::<u8>(bs * bs)
            .map_err(|e| anyhow::anyhow!(e))?;
        let committed_buf = forked
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let out_buf = forked
            .alloc_zeros::<u32>(bs - 1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let dummy_drafts = forked
            .alloc_zeros::<u32>(bs - 1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let accept_out = forked
            .alloc_zeros::<u32>(bs)
            .map_err(|e| anyhow::anyhow!(e))?;
        let part_val = forked
            .alloc_zeros::<f32>((bs - 1) * parts)
            .map_err(|e| anyhow::anyhow!(e))?;
        let part_idx = forked
            .alloc_zeros::<i32>((bs - 1) * parts)
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut mask_host = vec![0u8; bs * bs];
        for i in 0..bs {
            for j in 0..=i {
                mask_host[i * bs + j] = 1;
            }
        }
        forked
            .memcpy_htod(&mask_host[..], &mut mask_buf)
            .map_err(|e| anyhow::anyhow!("mask htod: {e:?}"))?;
        forked.synchronize().map_err(|e| anyhow::anyhow!(e))?;
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
            captured: false,
        })
    }

    pub fn last_conf(&self) -> Option<&[f32]> {
        self.last_conf.as_deref()
    }

    pub(crate) fn out_buf(&self) -> &CudaSlice<u32> {
        &self.out_buf
    }

    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &Tensor,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propose_device(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &Tensor,
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

    #[allow(clippy::too_many_arguments)]
    fn propose_inner(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        anchor: u32,
        anchor_pos: usize,
        embed_weight: &Tensor,
        lm_head: &Linear,
        want_dtoh: bool,
    ) -> Result<Vec<u32>> {
        let ring = cache
            .ring
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("graph propose requires the ring ctx cache"))?;
        anyhow::ensure!(
            ring.cap == self.cap,
            "graph propose: ring cap {} != proposer cap {}",
            ring.cap,
            self.cap
        );
        anyhow::ensure!(
            ring.stored + self.bs <= ring.cap,
            "graph propose: no headroom (stored {}, cap {})",
            ring.stored,
            ring.cap
        );
        anyhow::ensure!(ring.stored > 0, "graph propose: empty context");
        anyhow::ensure!(
            anchor_pos == cache.len,
            "graph propose: anchor_pos {} != context len {}",
            anchor_pos,
            cache.len
        );
        anyhow::ensure!(
            anchor_pos + self.bs <= ROPE_TABLE_CAP,
            "graph propose: position {} beyond rope table",
            anchor_pos
        );
        let dev = match &draft.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("cuda"),
        };
        self.host_toks[0] = anchor;
        for (i, p) in self.host_pos.iter_mut().enumerate() {
            *p = (anchor_pos + i) as i32;
        }
        self.host_committed[0] = ring.stored as i32;

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
                let tc = tok_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let pc = pos_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let toks_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(tc, dev2.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (bs,),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                let pos_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(pc, dev2.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (bs,),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
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
                anyhow::ensure!(
                    logits.dtype() == DType::F32 && logits.dims() == [1, bs, vocab],
                    "graph propose: unexpected logits {:?} {:?}",
                    logits.dims(),
                    logits.dtype()
                );
                let lc = logits.contiguous()?;
                let (ls, ll) = lc.storage_and_layout();
                let l_cuda = match &*ls {
                    candle_core::Storage::Cuda(st) => st,
                    _ => anyhow::bail!("graph logits not CUDA"),
                };
                let l_slice = l_cuda.as_cuda_slice::<f32>()?;
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
                anyhow::ensure!(rc == 0, "graph draft argmax rc={rc}");
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
        #[allow(deprecated)]
        let out: Vec<u32> = legacy
            .memcpy_dtov(&self.out_buf)
            .map_err(|e| anyhow::anyhow!("drafts dtoh: {e:?}"))?;
        decandle_stats::tick_dtoh();
        if self.collect_conf {
            #[allow(deprecated)]
            let pv: Vec<f32> = legacy
                .memcpy_dtov(&self.part_val)
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

#[derive(Clone, Debug, Default)]
pub struct DflashStats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
    pub pos0_accepted: usize,
    pub lookup_rounds: usize,
    pub lookup_drafted: usize,
    pub lookup_accepted: usize,
    pub round_ms: Vec<f64>,
    pub bucket_hits: std::collections::BTreeMap<usize, usize>,
    pub bucket_capture_ms: Vec<(usize, f64)>,
    pub accept_len_hist: std::collections::BTreeMap<usize, usize>,
    pub stop_reason: LagunaStopReason,
    pub stop_token: Option<u32>,
    pub discarded_after_stop: usize,
}

impl DflashStats {
    pub fn ended_turn(&self) -> bool {
        self.stop_reason == LagunaStopReason::HitStopToken
    }

    pub fn termination(&self, max_new: usize) -> String {
        match self.stop_token {
            Some(t) => format!("ENDED-TURN stop_id={t}"),
            None => format!("MAX-NEW({max_new}) NOT-TERMINATED"),
        }
    }

    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.drafted as f64
    }

    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.emitted as f64 / self.rounds as f64
    }

    pub fn pos0_accept_rate(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.pos0_accepted as f64 / self.rounds as f64
    }

    pub fn pos_accept_curve(&self, k: usize) -> Vec<f64> {
        if self.rounds == 0 {
            return vec![0.0; k];
        }
        let r = self.rounds as f64;
        (0..k)
            .map(|j| {
                let n: usize = self
                    .accept_len_hist
                    .iter()
                    .filter(|(&len, _)| len > j)
                    .map(|(_, &c)| c)
                    .sum();
                n as f64 / r
            })
            .collect()
    }
}

pub const LAGUNA_SHIPPED_EOS_IDS: [u32; 2] = [2, 24];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LagunaStopReason {
    #[default]
    ReachedMaxNew,
    HitStopToken,
}

impl std::fmt::Display for LagunaStopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LagunaStopReason::ReachedMaxNew => write!(f, "max-new"),
            LagunaStopReason::HitStopToken => write!(f, "stop-token"),
        }
    }
}

pub fn non_empty_stop_ids(ids: Vec<u32>) -> Vec<u32> {
    if ids.is_empty() {
        eprintln!(
            "[dflash] empty stop set requested; keeping LAGUNA_SHIPPED_EOS_IDS {:?}",
            LAGUNA_SHIPPED_EOS_IDS
        );
        return LAGUNA_SHIPPED_EOS_IDS.to_vec();
    }
    ids
}

pub fn stop_ids_from_generation_config(dir: &std::path::Path) -> Result<Vec<u32>> {
    let p = dir.join("generation_config.json");
    let raw =
        std::fs::read_to_string(&p).map_err(|e| anyhow::anyhow!("read {}: {e}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let ids: Vec<u32> = match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => vec![n.as_u64().unwrap_or_default() as u32],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        _ => anyhow::bail!("{} has no eos_token_id", p.display()),
    };
    anyhow::ensure!(!ids.is_empty(), "{} has an empty eos_token_id", p.display());
    Ok(ids)
}

pub struct LagunaDflashEngine<'a> {
    target: &'a Laguna,
    draft: &'a LagunaDflash,
    aux_layers: Vec<usize>,
    num_speculative: usize,
    stop_ids: Vec<u32>,
}

pub enum SpecTargetCache {
    Bf16(crate::laguna::LagunaKvCache),
    #[cfg(feature = "cuda")]
    Fp8(crate::laguna_fp8::LagunaKvCacheFp8),
}

impl SpecTargetCache {
    fn rollback(&mut self, n: usize) -> Result<()> {
        match self {
            SpecTargetCache::Bf16(c) => c.rollback(n),
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => c.rollback(n),
        }
    }
}

impl crate::gemma4::Gemma4Cache for SpecTargetCache {
    fn current_len(&self) -> usize {
        match self {
            SpecTargetCache::Bf16(c) => crate::gemma4::Gemma4Cache::current_len(c),
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => crate::gemma4::Gemma4Cache::current_len(c),
        }
    }
    fn advance(&mut self, n: usize) {
        match self {
            SpecTargetCache::Bf16(c) => crate::gemma4::Gemma4Cache::advance(c, n),
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => crate::gemma4::Gemma4Cache::advance(c, n),
        }
    }
    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        match self {
            SpecTargetCache::Bf16(c) => {
                crate::gemma4::Gemma4Cache::prepare_for_decode(c, write_pos, n_total)
            }
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => {
                crate::gemma4::Gemma4Cache::prepare_for_decode(c, write_pos, n_total)
            }
        }
    }
    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        match self {
            SpecTargetCache::Bf16(c) => {
                crate::gemma4::Gemma4Cache::write_at(c, layer, k_new, v_new)
            }
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => crate::gemma4::Gemma4Cache::write_at(c, layer, k_new, v_new),
        }
    }
    fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        match self {
            SpecTargetCache::Bf16(c) => crate::gemma4::Gemma4Cache::view(c, layer, len),
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => crate::gemma4::Gemma4Cache::view(c, layer, len),
        }
    }
    fn try_decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        match self {
            SpecTargetCache::Bf16(c) => crate::gemma4::Gemma4Cache::try_decode_attention_fp8(
                c,
                layer,
                q_rot,
                n_q,
                sliding_window,
                scaling,
            ),
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => crate::gemma4::Gemma4Cache::try_decode_attention_fp8(
                c,
                layer,
                q_rot,
                n_q,
                sliding_window,
                scaling,
            ),
        }
    }
    fn try_decode_attention_ring(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        match self {
            SpecTargetCache::Bf16(c) => crate::gemma4::Gemma4Cache::try_decode_attention_ring(
                c,
                layer,
                q_rot,
                n_q,
                sliding_window,
                scaling,
            ),
            #[cfg(feature = "cuda")]
            SpecTargetCache::Fp8(c) => crate::gemma4::Gemma4Cache::try_decode_attention_ring(
                c,
                layer,
                q_rot,
                n_q,
                sliding_window,
                scaling,
            ),
        }
    }
}

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
pub(crate) mod decandle_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static COUNT_ENABLED: OnceLock<bool> = OnceLock::new();
    static DEVICE_DRAFTS: OnceLock<bool> = OnceLock::new();
    static DEVICE_ROUND: OnceLock<bool> = OnceLock::new();
    pub static HTOD: AtomicU64 = AtomicU64::new(0);
    pub static DTOH: AtomicU64 = AtomicU64::new(0);
    pub static SYNC: AtomicU64 = AtomicU64::new(0);

    pub fn count_enabled() -> bool {
        *COUNT_ENABLED.get_or_init(|| std::env::var_os("NV_DFLASH_SYNC_COUNT").is_some())
    }
    pub fn device_drafts_enabled() -> bool {
        *DEVICE_DRAFTS.get_or_init(|| std::env::var_os("NV_DFLASH_DEVICE_DRAFTS").is_some())
    }

    pub fn device_round_enabled() -> bool {
        *DEVICE_ROUND.get_or_init(|| std::env::var_os("NV_DFLASH_DEVICE_ROUND").is_some())
    }
    pub fn tick_htod() {
        if count_enabled() {
            HTOD.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn tick_dtoh() {
        if count_enabled() {
            DTOH.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn tick_sync() {
        if count_enabled() {
            SYNC.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn snapshot() -> (u64, u64, u64) {
        (
            HTOD.load(Ordering::Relaxed),
            DTOH.load(Ordering::Relaxed),
            SYNC.load(Ordering::Relaxed),
        )
    }
    pub fn reset() {
        HTOD.store(0, Ordering::Relaxed);
        DTOH.store(0, Ordering::Relaxed);
        SYNC.store(0, Ordering::Relaxed);
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

pub(crate) fn accept_block_on_host(vlogits: &Tensor, drafts: &[u32]) -> Result<(usize, Vec<u32>)> {
    let k = drafts.len();
    let vtoks: Vec<u32> = vlogits
        .argmax(candle_core::D::Minus1)?
        .flatten_all()?
        .to_vec1()?;
    let mut accepted = 0usize;
    let mut emitted: Vec<u32> = Vec::with_capacity(k + 1);
    for (i, &d) in drafts.iter().enumerate() {
        let vtok = vtoks[i];
        if vtok == d {
            emitted.push(d);
            accepted += 1;
        } else {
            emitted.push(vtok);
            break;
        }
    }
    if accepted == k {
        emitted.push(vtoks[k]);
    }
    Ok((accepted, emitted))
}

#[cfg(feature = "cuda")]
pub(crate) fn accept_block_on_device(
    target: &Laguna,
    slots_map: &mut std::collections::HashMap<usize, AcceptSlots>,
    vlogits: &Tensor,
    drafts: &[u32],
    device_drafts: Option<u64>,
) -> Result<(usize, Vec<u32>)> {
    let dev = match target.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("accept_on_device requires a CUDA device"),
    };
    let stream = dev.cuda_stream();
    let k = drafts.len();
    let m = k + 1;
    let vocab = target.config().vocab_size;
    let parts = nv_kernels::cuda::dflash_accept_parts();
    anyhow::ensure!(parts > 0, "dflash accept kernel unavailable");
    if !slots_map.contains_key(&k) {
        let prof_ts = if crate::laguna_step_graph::verify_prof_enabled() {
            Some(stream.alloc_zeros::<u64>(2)?)
        } else {
            None
        };
        slots_map.insert(
            k,
            AcceptSlots {
                drafts: stream.alloc_zeros::<u32>(k)?,
                row_argmax: stream.alloc_zeros::<u32>(m)?,
                out: stream.alloc_zeros::<u32>(m + 1)?,
                part_val: stream.alloc_zeros::<f32>(m * parts)?,
                part_idx: stream.alloc_zeros::<i32>(m * parts)?,
                prof_ts,
            },
        );
    }
    let slots = slots_map.get_mut(&k).unwrap();
    anyhow::ensure!(
        slots.drafts.len() == k && slots.row_argmax.len() == m,
        "accept slots sized for a different k"
    );
    if device_drafts.is_none() {
        stream
            .memcpy_htod(drafts, &mut slots.drafts)
            .map_err(|e| anyhow::anyhow!("accept drafts htod: {e:?}"))?;
        decandle_stats::tick_htod();
    }

    let logits_c = vlogits.contiguous()?;
    let dims = logits_c.dims();
    anyhow::ensure!(
        dims == [1, m, vocab],
        "accept_on_device: logits must be [1, {m}, {vocab}], got {dims:?}"
    );
    anyhow::ensure!(logits_c.dtype() == DType::F32, "accept logits must be f32");
    let (ls, ll) = logits_c.storage_and_layout();
    let l_cuda = match &*ls {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("accept logits must be CUDA"),
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
        anyhow::ensure!(rc == 0, "dflash_accept_f32 rc={rc}");
        if let Some(ts) = slots.prof_ts.as_ref() {
            crate::laguna_step_graph::prof_timestamp_at(ts, 1, &stream)?;
        }
    }
    #[allow(deprecated)]
    let out_host: Vec<u32> = stream
        .memcpy_dtov(&slots.out)
        .map_err(|e| anyhow::anyhow!("accept out dtoh: {e:?}"))?;
    decandle_stats::tick_dtoh();
    if let Some(ts) = slots.prof_ts.as_ref() {
        #[allow(deprecated)]
        let t: Vec<u64> = stream
            .memcpy_dtov(ts)
            .map_err(|e| anyhow::anyhow!("accept prof dtoh: {e:?}"))?;
        let ms = t[1].saturating_sub(t[0]) as f64 / 1e6;
        eprintln!("[laguna_verify_prof] accept={ms:.3}");
    }
    let accepted = out_host[0] as usize;
    anyhow::ensure!(accepted <= k, "accept count {accepted} > k {k}");
    let emitted: Vec<u32> = out_host[1..2 + accepted].to_vec();
    Ok((accepted, emitted))
}

impl<'a> LagunaDflashEngine<'a> {
    pub fn new(target: &'a Laguna, draft: &'a LagunaDflash) -> Result<Self> {
        let k = default_num_speculative(draft.config.dflash_config.block_size);
        Self::with_num_speculative(target, draft, k)
    }

    pub fn with_num_speculative(
        target: &'a Laguna,
        draft: &'a LagunaDflash,
        num_speculative: usize,
    ) -> Result<Self> {
        let aux_layers = resolve_tap_layers(&draft.config, tap_list_mode());
        if num_speculative == 0 || num_speculative >= draft.config.dflash_config.block_size {
            anyhow::bail!(
                "dflash: num_speculative must be in 1..{}, got {num_speculative}",
                draft.config.dflash_config.block_size
            );
        }
        let n_target = target.config().num_hidden_layers;
        if draft.config.dflash_config.num_target_layers != n_target {
            anyhow::bail!(
                "dflash: draft expects target with {} layers, got {}",
                draft.config.dflash_config.num_target_layers,
                n_target
            );
        }
        for &li in &aux_layers {
            if li >= n_target {
                anyhow::bail!("dflash: aux layer {li} out of range for target ({n_target})");
            }
        }
        if target.config().hidden_size != draft.config.hidden_size {
            anyhow::bail!("dflash: target/draft hidden size mismatch");
        }
        if target.config().vocab_size != draft.config.vocab_size {
            anyhow::bail!("dflash: target/draft vocab size mismatch");
        }
        target.set_device_verify_routing(true);
        Ok(Self {
            target,
            draft,
            aux_layers,
            num_speculative,
            stop_ids: LAGUNA_SHIPPED_EOS_IDS.to_vec(),
        })
    }

    pub fn with_stop_ids(mut self, ids: Vec<u32>) -> Self {
        self.stop_ids = non_empty_stop_ids(ids);
        self
    }

    pub fn set_stop_ids(&mut self, ids: Vec<u32>) {
        self.stop_ids = non_empty_stop_ids(ids);
    }

    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    pub fn with_stop_ids_from_dir(self, dir: &std::path::Path) -> Result<Self> {
        let ids = stop_ids_from_generation_config(dir)?;
        Ok(self.with_stop_ids(ids))
    }

    pub fn select_for_prompt(&self, text: &str) -> PromptClass {
        self.draft.select_for_prompt(text)
    }

    pub fn select_for_prompt_ctx(&self, text: &str, ctx_tokens: usize) -> PromptClass {
        self.draft.select_for_prompt_ctx(text, ctx_tokens)
    }

    pub fn active_rope_theta(&self) -> f32 {
        self.draft.active_rope_theta()
    }

    fn accept_on_host(&self, vlogits: &Tensor, drafts: &[u32]) -> Result<(usize, Vec<u32>)> {
        accept_block_on_host(vlogits, drafts)
    }

    #[cfg(feature = "cuda")]
    fn accept_on_device(
        &self,
        slots_map: &mut std::collections::HashMap<usize, AcceptSlots>,
        vlogits: &Tensor,
        drafts: &[u32],
        device_drafts: Option<u64>,
    ) -> Result<(usize, Vec<u32>)> {
        accept_block_on_device(self.target, slots_map, vlogits, drafts, device_drafts)
    }

    pub fn generate_greedy(
        &self,
        prompt: &[u32],
        max_new: usize,
        max_seq: usize,
    ) -> Result<(Vec<u32>, DflashStats)> {
        self.generate_greedy_inner(prompt, max_new, max_seq, true)
    }

    pub fn generate_greedy_target_only(
        &self,
        prompt: &[u32],
        max_new: usize,
        max_seq: usize,
    ) -> Result<(Vec<u32>, DflashStats)> {
        self.generate_greedy_inner(prompt, max_new, max_seq, false)
    }

    fn generate_greedy_inner(
        &self,
        prompt: &[u32],
        max_new: usize,
        max_seq: usize,
        use_draft: bool,
    ) -> Result<(Vec<u32>, DflashStats)> {
        if prompt.is_empty() {
            anyhow::bail!("generate_greedy: empty prompt");
        }
        let device = self.target.device().clone();
        let mask_id = self.draft.config.dflash_config.mask_token_id;

        #[cfg(feature = "cuda")]
        let mut target_cache = if std::env::var_os("NV_LAGUNA_FP8_KV").is_some() {
            match self.target.new_kv_cache_fp8(max_seq) {
                Ok(c) => SpecTargetCache::Fp8(c),
                Err(e) => {
                    eprintln!(
                        "[dflash] NV_LAGUNA_FP8_KV set but fp8 KV cache unavailable, bf16: {e:#}"
                    );
                    SpecTargetCache::Bf16(self.target.new_kv_cache(max_seq)?)
                }
            }
        } else {
            SpecTargetCache::Bf16(self.target.new_kv_cache(max_seq)?)
        };
        #[cfg(not(feature = "cuda"))]
        let mut target_cache = SpecTargetCache::Bf16(self.target.new_kv_cache(max_seq)?);
        let mut ctx = self.draft.new_ctx_cache();
        let mut stats = DflashStats::default();
        let mut lookup = if use_draft {
            LookupState::from_env()
        } else {
            None
        };

        let seq = prompt.len();
        let prefill_chunk = std::env::var("NV_DFLASH_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(seq);
        let mut last_logits: Option<(Tensor, usize)> = None;
        let mut offset = 0usize;
        while offset < seq {
            let n = prefill_chunk.min(seq - offset);
            let tokens_t =
                Tensor::from_vec(prompt[offset..offset + n].to_vec(), (1usize, n), &device)?;
            let pos_host: Vec<i32> = (offset as i32..(offset + n) as i32).collect();
            let pos_t = Tensor::from_vec(pos_host.clone(), n, &device)?;
            let (logits, aux) = self.target.forward_with_cache_aux_scoped(
                &tokens_t,
                &pos_t,
                &mut target_cache,
                &self.aux_layers,
                true,
            )?;
            if use_draft {
                let combined = self.draft.combine_aux(&aux)?;
                let ctx_pos = Tensor::from_vec(pos_host, n, &device)?;
                self.draft.append_context(&mut ctx, &combined, &ctx_pos)?;
            }
            last_logits = Some((logits, n));
            offset += n;
        }
        let (logits, last_n) = last_logits.expect("non-empty prompt");

        let mut anchor: u32 = logits
            .narrow(1, last_n - 1, 1)?
            .argmax(candle_core::D::Minus1)?
            .flatten_all()?
            .to_vec1::<u32>()?[0];
        drop(logits);
        let mut generated: Vec<u32> = vec![anchor];
        let mut num_ctx = seq;
        let stop_ids = self.stop_ids.clone();
        if let Some(&t) = generated.first().filter(|t| stop_ids.contains(t)) {
            stats.stop_reason = LagunaStopReason::HitStopToken;
            stats.stop_token = Some(t);
            return Ok((generated, stats));
        }
        if let Some(l) = lookup.as_mut() {
            l.extend_slice(prompt);
            l.extend(anchor);
        }

        let prof = std::env::var_os("NV_DFLASH_PROF").is_some();

        let block_size = self.draft.config.dflash_config.block_size;
        let entropy_stop = use_draft && spec_entropy_stop_enabled();
        let entropy_cap = spec_entropy_cap(self.num_speculative, block_size);
        let entropy_tau = spec_entropy_tau();
        if entropy_stop {
            eprintln!(
                "[dflash] NV_SPEC_ENTROPY_STOP=1: draft-length gated by cumulative confidence \
                 tau={entropy_tau:.3} cap={entropy_cap} (fixed_k={}); accept test unchanged, \
                 accepts byte-identical",
                self.num_speculative
            );
        }

        #[cfg(feature = "cuda")]
        let mut graph_prop: Option<DflashGraphProposer> = None;
        #[cfg(feature = "cuda")]
        if use_draft
            && !entropy_stop
            && dflash_graph_enabled()
            && matches!(device, Device::Cuda(_))
            && ctx.has_ring()
        {
            if dflash_window_mode() == DflashWindowMode::Strict {
                eprintln!(
                    "[dflash] window_mode=strict: graph drafter is relaxed-only, eager drafting"
                );
            } else {
                match DflashGraphProposer::new(self.draft, self.num_speculative) {
                    Ok(p) => graph_prop = Some(p),
                    Err(e) => {
                        eprintln!("[dflash] graph proposer init failed, eager drafting: {e:#}")
                    }
                }
            }
        }

        let adapt = use_draft && dflash_adapt_enabled();
        let adapt_thresh = dflash_adapt_thresh();
        #[cfg(feature = "cuda")]
        let mut verify_graphs: std::collections::HashMap<
            usize,
            crate::laguna_step_graph::LagunaVerifyGraph,
        > = std::collections::HashMap::new();
        #[cfg(feature = "cuda")]
        let mut verify_graph_on =
            crate::laguna_graph::whole_step_graph_enabled() && matches!(device, Device::Cuda(_));
        #[cfg(feature = "cuda")]
        if verify_graph_on && matches!(target_cache, SpecTargetCache::Fp8(_)) {
            if std::env::var("NV_LAGUNA_FP8_VERIFY_GRAPH")
                .map(|v| v == "0")
                .unwrap_or(false)
            {
                eprintln!("[dflash] NV_LAGUNA_FP8_VERIFY_GRAPH=0: fp8 KV run uses eager verify");
                verify_graph_on = false;
            } else {
                eprintln!(
                    "[dflash] fp8 KV + whole-step verify graph (e4m3 full-attn KV read in-graph; \
                     NV_LAGUNA_FP8_VERIFY_GRAPH=0 restores eager verify)"
                );
            }
        }
        #[cfg(feature = "cuda")]
        let mut accept_slots: std::collections::HashMap<usize, AcceptSlots> =
            std::collections::HashMap::new();
        #[cfg(feature = "cuda")]
        let dev_accept = matches!(device, Device::Cuda(_))
            && std::env::var_os("NV_DFLASH_HOST_ACCEPT").is_none();
        let mut t_draft = 0f64;
        let mut t_verify = 0f64;
        let mut t_ctx = 0f64;
        let sync = |on: bool| {
            if on {
                let _ = device.synchronize();
            }
        };

        #[cfg(feature = "cuda")]
        decandle_stats::reset();
        let _ = device.synchronize();
        while generated.len() < max_new {
            let k = self.num_speculative;

            let draft_k = if entropy_stop { entropy_cap } else { k };
            sync(prof);
            let round_t0 = std::time::Instant::now();
            let t0 = round_t0;
            let lookup_prop = lookup.as_ref().and_then(|l| l.propose(k));
            let from_lookup = lookup_prop.is_some();

            #[cfg(feature = "cuda")]
            let dr_eligible = decandle_stats::device_round_enabled()
                && use_draft
                && !from_lookup
                && !adapt
                && !entropy_stop
                && dev_accept
                && verify_graph_on
                && graph_prop.is_some()
                && verify_graphs.contains_key(&(k + 1));
            #[cfg(not(feature = "cuda"))]
            let _dr_eligible = false;

            #[allow(unused_mut, unused_variables, unused_assignments)]
            let mut dr_round = false;
            let (mut drafts, conf): (Vec<u32>, Option<Vec<f32>>) = if let Some(d) = lookup_prop {
                (d, None)
            } else if use_draft {
                #[cfg(feature = "cuda")]
                {
                    let mut got: Option<(Vec<u32>, Option<Vec<f32>>)> = None;
                    let mut kill = false;
                    if let Some(p) = graph_prop.as_mut() {
                        if dr_eligible {
                            match p.propose_device(
                                self.draft,
                                &ctx,
                                anchor,
                                num_ctx,
                                self.target.embed_weight(),
                                self.target.lm_head(),
                            ) {
                                Ok(n) => {
                                    got = Some((vec![0u32; n], None));
                                    dr_round = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[dflash] device-round propose failed, eager fallback: {e:#}"
                                    );
                                    kill = true;
                                }
                            }
                        } else {
                            match p.propose(
                                self.draft,
                                &ctx,
                                anchor,
                                num_ctx,
                                self.target.embed_weight(),
                                self.target.lm_head(),
                            ) {
                                Ok(v) => {
                                    let c = p.last_conf().map(|s| s.to_vec());
                                    got = Some((v, c));
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[dflash] graph propose failed, eager fallback: {e:#}"
                                    );
                                    kill = true;
                                }
                            }
                        }
                    }
                    if kill {
                        graph_prop = None;
                    }
                    match got {
                        Some(v) => v,
                        None if entropy_stop => {
                            let (d, c) = self.draft.propose_k_logconf(
                                &ctx,
                                anchor,
                                num_ctx,
                                draft_k,
                                self.target.embed_weight(),
                                self.target.lm_head(),
                            )?;
                            (d, Some(c))
                        }
                        None if adapt => {
                            let (d, c) = self.draft.propose_k_conf(
                                &ctx,
                                anchor,
                                num_ctx,
                                draft_k,
                                self.target.embed_weight(),
                                self.target.lm_head(),
                            )?;
                            (d, Some(c))
                        }
                        None => (
                            self.draft.propose_k(
                                &ctx,
                                anchor,
                                num_ctx,
                                draft_k,
                                self.target.embed_weight(),
                                self.target.lm_head(),
                            )?,
                            None,
                        ),
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    if entropy_stop {
                        let (d, c) = self.draft.propose_k_logconf(
                            &ctx,
                            anchor,
                            num_ctx,
                            draft_k,
                            self.target.embed_weight(),
                            self.target.lm_head(),
                        )?;
                        (d, Some(c))
                    } else if adapt {
                        let (d, c) = self.draft.propose_k_conf(
                            &ctx,
                            anchor,
                            num_ctx,
                            draft_k,
                            self.target.embed_weight(),
                            self.target.lm_head(),
                        )?;
                        (d, Some(c))
                    } else {
                        (
                            self.draft.propose_k(
                                &ctx,
                                anchor,
                                num_ctx,
                                draft_k,
                                self.target.embed_weight(),
                                self.target.lm_head(),
                            )?,
                            None,
                        )
                    }
                }
            } else {
                (vec![mask_id; k], None)
            };
            if entropy_stop {
                if let Some(c) = conf.as_deref() {
                    drafts.truncate(entropy_stop_len(c, entropy_tau));
                }
            } else if adapt {
                if let Some(c) = conf.as_deref() {
                    drafts.truncate(adapt_truncate_len(c, adapt_thresh));
                }
            }
            let drafts = drafts;
            sync(prof);
            t_draft += t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();

            let mut block: Vec<u32> = Vec::with_capacity(k + 1);
            block.push(anchor);
            block.extend_from_slice(&drafts);
            *stats.bucket_hits.entry(block.len()).or_insert(0) += 1;
            #[cfg(feature = "cuda")]
            let graph_verify_res = {
                let mut got = None;
                if verify_graph_on {
                    let vbs = block.len();
                    let run_verify =
                        |g: &mut crate::laguna_step_graph::LagunaVerifyGraph,
                         cache: &mut SpecTargetCache,
                         block: &[u32]| match cache {
                            SpecTargetCache::Bf16(c) => g.verify(c, block),
                            SpecTargetCache::Fp8(c) => g.verify_fp8(c, block),
                        };
                    if let Some(g) = verify_graphs.get_mut(&vbs) {
                        let vres = if dr_round {
                            let drafts_dev = graph_prop
                                .as_ref()
                                .expect("dr_round implies graph_prop")
                                .out_buf();
                            match &mut target_cache {
                                SpecTargetCache::Bf16(c) => g.verify_device(c, anchor, drafts_dev),
                                SpecTargetCache::Fp8(c) => {
                                    g.verify_fp8_device(c, anchor, drafts_dev)
                                }
                            }
                        } else {
                            run_verify(g, &mut target_cache, &block)
                        };
                        match vres {
                            Ok(r) => got = Some(r),
                            Err(e) => {
                                eprintln!("[dflash] verify graph failed, eager fallback: {e:#}");
                                verify_graphs.clear();
                                verify_graph_on = false;
                            }
                        }
                    } else {
                        let cap_t0 = std::time::Instant::now();
                        let built = match &target_cache {
                            SpecTargetCache::Bf16(c) => {
                                crate::laguna_step_graph::LagunaVerifyGraph::new(
                                    self.target,
                                    c,
                                    vbs,
                                    &self.aux_layers,
                                )
                            }
                            SpecTargetCache::Fp8(c) => {
                                crate::laguna_step_graph::LagunaVerifyGraph::new_fp8(
                                    self.target,
                                    c,
                                    vbs,
                                    &self.aux_layers,
                                )
                            }
                        };
                        match built {
                            Ok(mut g) => match run_verify(&mut g, &mut target_cache, &block) {
                                Ok(r) => {
                                    stats
                                        .bucket_capture_ms
                                        .push((vbs, 1000.0 * cap_t0.elapsed().as_secs_f64()));
                                    verify_graphs.insert(vbs, g);
                                    got = Some(r);
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[dflash] verify graph capture failed (bs={vbs}), eager verify: {e:#}"
                                    );
                                    verify_graphs.clear();
                                    verify_graph_on = false;
                                }
                            },
                            Err(e) => {
                                eprintln!(
                                    "[dflash] verify graph init failed (bs={vbs}), eager verify: {e:#}"
                                );
                                verify_graphs.clear();
                                verify_graph_on = false;
                            }
                        }
                    }
                }
                got
            };
            #[cfg(not(feature = "cuda"))]
            let graph_verify_res: Option<(Tensor, Vec<Tensor>)> = None;
            #[cfg(feature = "cuda")]
            let used_graph_verify = graph_verify_res.is_some();
            let (vlogits, vaux) = match graph_verify_res {
                Some(r) => r,
                None => {
                    let block_pos: Vec<i32> =
                        (0..block.len()).map(|i| (num_ctx + i) as i32).collect();
                    let bt = Tensor::from_vec(block.clone(), (1usize, block.len()), &device)?;
                    let bp = Tensor::from_vec(block_pos, block.len(), &device)?;
                    self.target.forward_with_cache_aux_scoped(
                        &bt,
                        &bp,
                        &mut target_cache,
                        &self.aux_layers,
                        true,
                    )?
                }
            };
            #[allow(unused_mut)]
            let mut accept_res: Option<(usize, Vec<u32>)> = None;
            #[cfg(feature = "cuda")]
            if dev_accept {
                let device_drafts: Option<u64> =
                    if (decandle_stats::device_drafts_enabled() || dr_round) && used_graph_verify {
                        match &device {
                            Device::Cuda(d) => verify_graphs.get(&block.len()).map(|g| {
                                let (p, n) = g.drafts_device_ptr(&d.cuda_stream());
                                debug_assert_eq!(n, drafts.len(), "device drafts len mismatch");
                                p
                            }),
                            _ => None,
                        }
                    } else {
                        None
                    };
                match self.accept_on_device(&mut accept_slots, &vlogits, &drafts, device_drafts) {
                    Ok(r) => accept_res = Some(r),
                    Err(e) => {
                        eprintln!("[dflash] device accept failed, host fallback: {e:#}")
                    }
                }
            }
            let (accepted, emitted) = match accept_res {
                Some(r) => r,
                #[cfg(feature = "cuda")]
                None if dr_round => anyhow::bail!(
                    "device-round: device accept failed but drafts are device-resident \
                     (no host copy for the host-accept fallback)"
                ),
                None => self.accept_on_host(&vlogits, &drafts)?,
            };
            sync(prof);
            t_verify += t1.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();

            let consumed = 1 + accepted;
            target_cache.rollback(block.len() - consumed)?;

            if use_draft {
                let mut vaux_kept = Vec::with_capacity(vaux.len());
                for a in &vaux {
                    vaux_kept.push(a.narrow(1, 0, consumed)?);
                }
                let combined = self.draft.combine_aux(&vaux_kept)?;
                let new_pos: Vec<i32> = (0..consumed).map(|i| (num_ctx + i) as i32).collect();
                let np = Tensor::from_vec(new_pos, consumed, &device)?;
                self.draft.append_context(&mut ctx, &combined, &np)?;
            }

            sync(prof);
            t_ctx += t2.elapsed().as_secs_f64();
            let _ = device.synchronize();
            #[cfg(feature = "cuda")]
            decandle_stats::tick_sync();
            stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());

            stats.rounds += 1;
            stats.drafted += drafts.len();
            stats.accepted += accepted;
            stats.emitted += emitted.len();
            if accepted > 0 {
                stats.pos0_accepted += 1;
            }
            *stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
            if let Some(l) = lookup.as_mut() {
                if from_lookup {
                    stats.lookup_rounds += 1;
                    stats.lookup_drafted += drafts.len();
                    stats.lookup_accepted += accepted;
                } else {
                    l.observe_dflash_round(accepted);
                }
                l.extend_slice(&emitted);
            }

            if let Some(i) = emitted.iter().position(|t| stop_ids.contains(t)) {
                generated.extend_from_slice(&emitted[..=i]);
                stats.stop_reason = LagunaStopReason::HitStopToken;
                stats.stop_token = Some(emitted[i]);
                stats.discarded_after_stop = emitted.len() - i - 1;
                break;
            }
            generated.extend_from_slice(&emitted);
            anchor = *generated.last().unwrap();
            num_ctx += consumed;
        }

        #[cfg(feature = "cuda")]
        if decandle_stats::count_enabled() && stats.rounds > 0 {
            let (htod, dtoh, sync_n) = decandle_stats::snapshot();
            let r = stats.rounds as f64;
            eprintln!(
                "[dflash_synccount] device_drafts={} device_round={} rounds={} htod={htod} dtoh={dtoh} sync={sync_n} \
                 per_round=(htod {:.3}, dtoh {:.3}, sync {:.3}) total_per_round={:.3}",
                decandle_stats::device_drafts_enabled(),
                decandle_stats::device_round_enabled(),
                stats.rounds,
                htod as f64 / r,
                dtoh as f64 / r,
                sync_n as f64 / r,
                (htod + dtoh + sync_n) as f64 / r,
            );
        }

        if prof && stats.rounds > 0 {
            let r = stats.rounds as f64;
            eprintln!(
                "[dflash_prof] use_draft={use_draft} rounds={} draft={:.2}ms verify={:.2}ms ctx={:.2}ms per round",
                stats.rounds,
                1000.0 * t_draft / r,
                1000.0 * t_verify / r,
                1000.0 * t_ctx / r
            );
        }

        if generated.len() > max_new {
            stats.discarded_after_stop += generated.len() - max_new;
            generated.truncate(max_new);
            if !generated.last().is_some_and(|t| stop_ids.contains(t)) {
                stats.stop_reason = LagunaStopReason::ReachedMaxNew;
                stats.stop_token = None;
            }
        }
        Ok((generated, stats))
    }
}

pub fn argmax_row(row: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (j, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = j as u32;
        }
    }
    best_idx
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
    fn config_parses() {
        let cfg = LagunaDflashConfig::from_hf_json_str(DRAFT_CONFIG).unwrap();
        assert_eq!(cfg.num_hidden_layers, 5);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.dflash_config.block_size, 16);
        assert_eq!(cfg.dflash_config.mask_token_id, 12);
        assert_eq!(cfg.dflash_config.target_layer_ids, vec![1, 13, 25, 33, 39]);
        assert!(cfg.dflash_config.causal);
        assert_eq!(cfg.num_aux(), 5);
    }

    #[test]
    fn config_rejects_non_causal_or_mixed_layers() {
        let non_causal = DRAFT_CONFIG.replace("\"causal\": true", "\"causal\": false");
        assert!(LagunaDflashConfig::from_hf_json_str(&non_causal).is_err());
        let mixed = DRAFT_CONFIG.replacen("sliding_attention", "full_attention", 1);
        assert!(LagunaDflashConfig::from_hf_json_str(&mixed).is_err());
    }

    #[test]
    fn ctx_cache_tracks_length() {
        let cache = DflashCtxCache::new(5);
        assert!(cache.is_empty());
        assert_eq!(cache.layers.len(), 5);
    }

    #[test]
    fn ctx_append_plan_cases() {
        let (w, cap) = (512usize, 768usize);
        assert_eq!(
            plan_ctx_append(100, w, cap, 8),
            CtxAppendPlan {
                input_skip: 0,
                keep: 100
            }
        );
        assert_eq!(
            plan_ctx_append(760, w, cap, 16),
            CtxAppendPlan {
                input_skip: 0,
                keep: 496
            }
        );
        assert_eq!(
            plan_ctx_append(0, w, cap, 700),
            CtxAppendPlan {
                input_skip: 188,
                keep: 0
            }
        );
        assert_eq!(
            plan_ctx_append(300, w, cap, 512),
            CtxAppendPlan {
                input_skip: 0,
                keep: 0
            }
        );
        assert_eq!(
            plan_ctx_append(600, w, cap, 300),
            CtxAppendPlan {
                input_skip: 0,
                keep: 212
            }
        );
        assert_eq!(
            plan_ctx_append(100, w, cap, 480),
            CtxAppendPlan {
                input_skip: 0,
                keep: 100
            }
        );
        assert_eq!(
            plan_ctx_append(400, w, cap, 480),
            CtxAppendPlan {
                input_skip: 0,
                keep: 32
            }
        );
    }

    #[test]
    fn ctx_append_plan_invariants() {
        let (w, cap) = (512usize, 768usize);
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
    }

    fn laguna_snapshot_dir() -> Option<std::path::PathBuf> {
        if let Ok(d) = std::env::var("NV_LAGUNA_DIR") {
            let p = std::path::PathBuf::from(d);
            if p.join("generation_config.json").is_file() {
                return Some(p);
            }
        }
        let base = format!(
            "{}/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("generation_config.json").is_file())
    }

    #[test]
    fn shipped_eos_ids_match_the_snapshot_generation_config() {
        let Some(dir) = laguna_snapshot_dir() else {
            eprintln!("skip: no cached Laguna-XS-2.1-NVFP4 snapshot");
            return;
        };
        let ids = stop_ids_from_generation_config(&dir).expect("read eos_token_id");
        eprintln!(
            "{}/generation_config.json eos_token_id = {ids:?}; LAGUNA_SHIPPED_EOS_IDS = {:?}",
            dir.display(),
            LAGUNA_SHIPPED_EOS_IDS
        );
        assert_eq!(
            ids,
            LAGUNA_SHIPPED_EOS_IDS.to_vec(),
            "the built-in default stop set drifted from the shipped generation_config.json. \
             Every free-running Laguna number is measured against this set; update the constant \
             and re-measure rather than letting the two disagree."
        );
    }

    #[test]
    fn a_fresh_engine_has_a_non_empty_stop_set() {
        assert!(
            !LAGUNA_SHIPPED_EOS_IDS.is_empty(),
            "an empty stop set is what free-ran every Laguna acceptance number past </assistant>"
        );
        assert!(LAGUNA_SHIPPED_EOS_IDS.contains(&24), "24 is </assistant>");
        assert!(LAGUNA_SHIPPED_EOS_IDS.contains(&2), "2 is 〈|EOS|〉");
    }

    #[test]
    fn an_empty_stop_set_cannot_be_installed() {
        assert_eq!(non_empty_stop_ids(vec![]), LAGUNA_SHIPPED_EOS_IDS.to_vec());
        assert_eq!(non_empty_stop_ids(vec![7, 9]), vec![7, 9]);
    }

    #[test]
    fn prompt_classifier_on_corpus() {
        assert_eq!(classify_prompt("Hello, my name is"), PromptClass::Prose);
        assert_eq!(
            classify_prompt("〈|EOS|〉<user>Hello, my name is</user>\n<assistant></think>"),
            PromptClass::Prose
        );
        assert_eq!(
            classify_prompt(
                "〈|EOS|〉<system>You are a helpful, conversationally-fluent assistant made by \
                 Poolside. You are here to be helpful to users through natural language \
                 conversations.</system>\n<user>Hello, my name is</user>\n<assistant><think>"
            ),
            PromptClass::Prose
        );
        assert_eq!(
            classify_prompt("Write a Python function that returns the square of a number."),
            PromptClass::Code
        );
        assert_eq!(
            classify_prompt(
                "Write a Rust program that reads a CSV file whose first row is a header, \
                 parses every remaining row into a struct with named fields."
            ),
            PromptClass::Code
        );
        assert_eq!(
            classify_prompt(
                "I'm planning a two-week trip to Japan in the autumn with my family. \
                 We like food, hiking and museums but want to avoid the biggest crowds."
            ),
            PromptClass::Prose
        );
        assert_eq!(
            classify_prompt("```\nfor i in range(10):\n    print(i)\n```"),
            PromptClass::Code
        );
        assert_eq!(PromptClass::Code.theta(), DFLASH_CODE_THETA);
        assert_eq!(PromptClass::Prose.theta(), DFLASH_PROSE_THETA);
    }

    #[test]
    fn adapt_truncate_len_gates() {
        assert_eq!(adapt_truncate_len(&[5.0, 4.0, 3.0, 2.0], 1.0), 4);
        assert_eq!(adapt_truncate_len(&[5.0, 4.0, 0.5, 2.0], 1.0), 2);
        assert_eq!(adapt_truncate_len(&[0.2, 4.0, 3.0, 2.0], 1.0), 1);
        assert_eq!(adapt_truncate_len(&[0.2, 0.1, 0.0, 0.0], 1.0), 1);
        assert_eq!(adapt_truncate_len(&[5.0], 1.0), 1);
        assert_eq!(adapt_truncate_len(&[5.0, 4.0], 10.0), 1);
        assert_eq!(adapt_truncate_len(&[5.0, 4.0], 0.0), 2);
    }

    #[test]
    fn entropy_stop_len_gates() {
        assert_eq!(entropy_stop_len(&[0.99, 0.99, 0.99, 0.99], 0.10), 4);
        assert_eq!(entropy_stop_len(&[0.9, 0.9, 0.9, 0.9], 0.10), 4);
        assert_eq!(entropy_stop_len(&[0.5, 0.5, 0.5, 0.5], 0.10), 4);
        assert_eq!(entropy_stop_len(&[0.5, 0.3, 0.3, 0.9], 0.10), 3);
        assert_eq!(entropy_stop_len(&[0.05, 0.9, 0.9], 0.10), 1);
        assert_eq!(entropy_stop_len(&[0.99], 0.10), 1);
        assert_eq!(entropy_stop_len(&[0.99, 0.99], 2.0), 1);
    }

    #[test]
    fn entropy_cap_defaults_and_clamps() {
        assert_eq!(spec_entropy_cap(4, 16), 8);
        assert_eq!(spec_entropy_cap(4, 6), 5);
        assert_eq!(spec_entropy_cap(1, 2), 1);
    }

    #[test]
    fn margins_from_part_max_rows() {
        let pv = [1.0f32, 7.0, 3.0, 2.0, -1.0, -4.0, -2.0, -8.0];
        let m = margins_from_part_max(&pv, 2, 4);
        assert_eq!(m.len(), 2);
        assert!((m[0] - 4.0).abs() < 1e-6);
        assert!((m[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn theta_eq_tolerance() {
        assert!(theta_eq(1.0e4, 1.0e4 + 1.0));
        assert!(!theta_eq(1.0e4, 5.0e5));
        assert!(theta_eq(5.0e5, 5.0e5));
    }

    #[test]
    fn default_k_is_recipe_and_clamped() {
        assert_eq!(PromptClass::Code.default_k(), DFLASH_DEFAULT_K);
        assert_eq!(PromptClass::Prose.default_k(), DFLASH_DEFAULT_K);
        assert_eq!(DFLASH_DEFAULT_K, 15);
        assert_eq!(default_num_speculative(16), 15);
        assert_eq!(default_num_speculative(64), 15);
        assert_eq!(default_num_speculative(8), 7);
        assert_eq!(default_num_speculative(2), 1);
        assert_eq!(default_num_speculative(0), 1);
    }

    #[test]
    fn rope_theta_resolution_prefers_config() {
        assert_eq!(resolve_rope_thetas(None, 5.0e5, false), vec![5.0e5]);
        assert_eq!(resolve_rope_thetas(Some(1.0e4), 5.0e5, false), vec![1.0e4]);
        let auto = resolve_rope_thetas(None, 1.0e6, true);
        assert_eq!(auto.len(), 3);
        assert!(theta_eq(auto[0], 1.0e6));
        assert!(auto.iter().any(|&t| theta_eq(t, DFLASH_CODE_THETA)));
        assert!(auto.iter().any(|&t| theta_eq(t, DFLASH_PROSE_THETA)));
        let dedup = resolve_rope_thetas(None, DFLASH_PROSE_THETA, true);
        assert_eq!(dedup, vec![DFLASH_PROSE_THETA, DFLASH_CODE_THETA]);
        let over = resolve_rope_thetas(Some(2.0e6), 5.0e5, true);
        assert!(theta_eq(over[0], 2.0e6));
        assert!(over.iter().any(|&t| theta_eq(t, 5.0e5)));
    }

    #[test]
    fn depth_gated_theta_gates_on_class_and_ctx() {
        let d = DFLASH_DEPTH_THETA_CTX_DEFAULT;
        assert!(theta_eq(
            depth_gated_theta(PromptClass::Code, 2048, d),
            DFLASH_CODE_THETA
        ));
        assert!(theta_eq(
            depth_gated_theta(PromptClass::Code, d, d),
            DFLASH_CODE_THETA
        ));
        assert!(theta_eq(
            depth_gated_theta(PromptClass::Code, d + 1, d),
            DFLASH_PROSE_THETA
        ));
        assert!(theta_eq(
            depth_gated_theta(PromptClass::Code, 16384, d),
            DFLASH_PROSE_THETA
        ));
        assert!(theta_eq(
            depth_gated_theta(PromptClass::Prose, 128, d),
            DFLASH_PROSE_THETA
        ));
        assert!(theta_eq(
            depth_gated_theta(PromptClass::Prose, 16384, d),
            DFLASH_PROSE_THETA
        ));
    }

    #[test]
    fn tap_list_resolution_switches_and_filters() {
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
            config,
            aux_hidden_norms: vec![RmsNorm::new(w2.clone(), 1e-6), RmsNorm::new(w2, 1e-6)],
            fc: Linear::new(fc_w, None).unwrap(),
            hidden_norm: RmsNorm::new(w1.clone(), 1e-6),
            layers: vec![],
            norm: RmsNorm::new(w1, 1e-6),
            ropes: vec![],
            rope_active: std::sync::atomic::AtomicUsize::new(0),
            dtype: DType::F32,
            device: dev.clone(),
        }
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
    fn stats_ratios() {
        let s = DflashStats {
            rounds: 4,
            drafted: 60,
            accepted: 15,
            emitted: 19,
            ..Default::default()
        };
        assert!((s.accept_rate() - 0.25).abs() < 1e-9);
        assert!((s.tokens_per_round() - 4.75).abs() < 1e-9);
    }

    #[test]
    fn lookup_env_knobs() {
        std::env::remove_var("NV_LAGUNA_LOOKUP_DRAFT");
        std::env::remove_var("NV_LOOKUP_MIN_MATCH");
        std::env::remove_var("NV_LAGUNA_LOOKUP_EMA");
        assert!(!lookup_draft_enabled());
        assert!(LookupState::from_env().is_none());
        assert_eq!(lookup_min_match(), LOOKUP_MIN_MATCH_DEFAULT);
        assert!(lookup_ema_enabled());
        std::env::set_var("NV_LAGUNA_LOOKUP_DRAFT", "1");
        std::env::set_var("NV_LOOKUP_MIN_MATCH", "0");
        std::env::set_var("NV_LAGUNA_LOOKUP_EMA", "0");
        assert!(lookup_draft_enabled());
        assert!(LookupState::from_env().is_some());
        assert_eq!(lookup_min_match(), 1);
        assert!(!lookup_ema_enabled());
        std::env::set_var("NV_LOOKUP_MIN_MATCH", "999");
        assert_eq!(lookup_min_match(), 64);
        std::env::set_var("NV_LOOKUP_MIN_MATCH", "bogus");
        assert_eq!(lookup_min_match(), LOOKUP_MIN_MATCH_DEFAULT);
        std::env::set_var("NV_LAGUNA_LOOKUP_DRAFT", "0");
        assert!(!lookup_draft_enabled());
        std::env::remove_var("NV_LAGUNA_LOOKUP_DRAFT");
        std::env::remove_var("NV_LOOKUP_MIN_MATCH");
        std::env::remove_var("NV_LAGUNA_LOOKUP_EMA");
    }

    #[test]
    fn lookup_arm_threshold_only() {
        let mut l = LookupState::new(2, false);
        assert!(l.propose(4).is_none());
        l.extend_slice(&[10, 20, 30, 40, 10, 20]);
        let p = l.propose(4).unwrap();
        assert_eq!(p, vec![30, 40, 10, 20]);
        assert_eq!(l.propose(2).unwrap(), vec![30, 40]);
        let mut strict = LookupState::new(3, false);
        strict.extend_slice(&[10, 20, 30, 40, 10, 20]);
        assert!(strict.propose(4).is_none());
    }

    #[test]
    fn lookup_arm_ema_guard_displaces_short_wins() {
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
    }

    #[test]
    fn lookup_reset_clears_stream_and_ema() {
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
