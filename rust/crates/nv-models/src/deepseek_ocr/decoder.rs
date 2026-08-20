use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use nv_layers::attn::flash_attn;
use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
#[cfg(test)]
use std::path::Path;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

pub mod prof {
    use candle_core::Device;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const EMBED: usize = 0;
    pub const NORM1: usize = 1;
    pub const QKV: usize = 2;
    pub const ROPE: usize = 3;
    pub const KV_WRITE: usize = 4;
    pub const ATTN: usize = 5;
    pub const O_PROJ: usize = 6;
    pub const NORM2: usize = 7;
    pub const FFN_DENSE: usize = 8;
    pub const MOE_GATE: usize = 9;
    pub const MOE_D2H: usize = 10;
    pub const MOE_ROUTE: usize = 11;
    pub const MOE_EXPERTS: usize = 12;
    pub const MOE_SHARED: usize = 13;
    pub const MOE_COMBINE: usize = 14;
    pub const FINAL_NORM: usize = 15;
    pub const N: usize = 16;

    pub const NAMES: [&str; N] = [
        "embed + vision splice",
        "input rmsnorm",
        "q/k/v proj",
        "rope (f32 up/down)",
        "kv write + view",
        "attention",
        "o_proj",
        "post rmsnorm",
        "dense ffn (layer 0)",
        "moe gate mm + softmax",
        "moe probs D2H (sync)",
        "moe routing (CPU topk)",
        "moe routed experts",
        "moe shared expert",
        "moe combine + cast",
        "final rmsnorm",
    ];

    static ACC: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("NV_DSOCR_PREFILL_PROF")
                .map(|v| v != "0")
                .unwrap_or(false)
        })
    }

    pub fn span<T>(idx: usize, dev: &Device, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let _ = dev.synchronize();
        let t = Instant::now();
        let r = f();
        let _ = dev.synchronize();
        ACC[idx].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    pub fn note_call() {
        if enabled() {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn reset() {
        for a in ACC.iter() {
            a.store(0, Ordering::Relaxed);
        }
        CALLS.store(0, Ordering::Relaxed);
    }

    pub fn calls() -> u64 {
        CALLS.load(Ordering::Relaxed)
    }

    pub fn report(pages: f64) -> String {
        if !enabled() || pages <= 0.0 {
            return String::new();
        }
        let vals: Vec<f64> = ACC
            .iter()
            .map(|a| a.load(Ordering::Relaxed) as f64 / 1e6 / pages)
            .collect();
        let total: f64 = vals.iter().sum();
        let mut s = String::new();
        s.push_str("| prefill stage | ms/page | share |\n|---|---|---|\n");
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap());
        for i in idx {
            if vals[i] <= 0.0 {
                continue;
            }
            s.push_str(&format!(
                "| {} | {:.2} | {:.1}% |\n",
                NAMES[i],
                vals[i],
                vals[i] / total * 100.0
            ));
        }
        s.push_str(&format!(
            "| **prefill total (instrumented)** | {total:.2} | 100% |\n"
        ));
        s
    }
}

pub const IMAGE_TOKEN_ID: u32 = 128815;
pub const BOS_TOKEN_ID: u32 = 0;
pub const EOS_TOKEN_ID: u32 = 1;
pub const REF_OPEN_TOKEN_ID: u32 = 128816;
pub const REF_CLOSE_TOKEN_ID: u32 = 128817;
pub const DET_OPEN_TOKEN_ID: u32 = 128818;
pub const DET_CLOSE_TOKEN_ID: u32 = 128819;
pub const GROUNDING_TOKEN_ID: u32 = 128820;
pub const TD_OPEN_TOKEN_ID: u32 = 128821;
pub const TD_CLOSE_TOKEN_ID: u32 = 128822;

pub const LOOP_WINDOW: usize = 256;
pub const LOOP_MAX_PERIOD: usize = 64;
pub const LOOP_CHECK_STRIDE: usize = 64;
pub const LOOP_MIN_EVIDENCE: usize = 128;
pub const LOOP_PERIODIC_RATIO: f32 = 0.70;
pub const LOOP_DISTINCT_RATIO: f32 = 0.45;
pub const LOOP_NGRAM: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopDetection {
    pub onset: usize,
    pub period: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowHit {
    Periodic(usize),
    Repeats,
}

fn window_hit(seq: &[u32], end: usize) -> Option<WindowHit> {
    let w = LOOP_WINDOW.min(end);
    if w < LOOP_MIN_EVIDENCE {
        return None;
    }
    let start = end - w;
    for p in 1..=LOOP_MAX_PERIOD.min(w / 2) {
        let mut matches = 0usize;
        for i in (start + p)..end {
            if seq[i] == seq[i - p] {
                matches += 1;
            }
        }
        if matches as f32 / (w - p) as f32 >= LOOP_PERIODIC_RATIO {
            return Some(WindowHit::Periodic(p));
        }
    }
    if w > LOOP_NGRAM {
        let n_grams = w - LOOP_NGRAM + 1;
        let mut set = std::collections::HashSet::with_capacity(n_grams);
        for i in start..=(end - LOOP_NGRAM) {
            set.insert(&seq[i..i + LOOP_NGRAM]);
        }
        if set.len() as f32 / n_grams as f32 <= LOOP_DISTINCT_RATIO {
            return Some(WindowHit::Repeats);
        }
    }
    None
}

fn backtrack_periodic_onset(seq: &[u32], end: usize, period: usize) -> usize {
    let max_gap = (period / 4).max(2);
    let mut onset = end;
    let mut gap = 0usize;
    let mut i = end;
    while i > period {
        i -= 1;
        if seq[i] == seq[i - period] {
            onset = i;
            gap = 0;
        } else {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    onset
}

fn backtrack_repeat_onset(seq: &[u32], end: usize) -> usize {
    if end < 2 * LOOP_NGRAM {
        return end;
    }
    let mut first_seen: std::collections::HashMap<&[u32], usize> = std::collections::HashMap::new();
    let mut repeated = vec![false; end - LOOP_NGRAM + 1];
    for i in 0..=(end - LOOP_NGRAM) {
        let gram = &seq[i..i + LOOP_NGRAM];
        match first_seen.get(gram) {
            Some(_) => repeated[i] = true,
            None => {
                first_seen.insert(gram, i);
            }
        }
    }
    let max_gap = 2 * LOOP_NGRAM;
    let mut onset = end;
    let mut gap = 0usize;
    let mut i = repeated.len();
    while i > 0 {
        i -= 1;
        if repeated[i] {
            onset = i;
            gap = 0;
        } else {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    onset
}

fn checkpoints(n: usize) -> Vec<usize> {
    let mut cs = Vec::new();
    let mut c = LOOP_MIN_EVIDENCE;
    while c < n {
        cs.push(c);
        c += LOOP_CHECK_STRIDE;
    }
    cs.push(n);
    cs
}

fn onset_from_hit(seq: &[u32], c: usize, hit: WindowHit) -> LoopDetection {
    match hit {
        WindowHit::Periodic(p) => LoopDetection {
            onset: backtrack_periodic_onset(seq, c, p),
            period: p,
        },
        WindowHit::Repeats => LoopDetection {
            onset: backtrack_repeat_onset(seq, c),
            period: 0,
        },
    }
}

pub fn detect_loop(seq: &[u32]) -> Option<LoopDetection> {
    let n = seq.len();
    if n < LOOP_MIN_EVIDENCE + LOOP_CHECK_STRIDE {
        return None;
    }
    let cs = checkpoints(n);
    let hits: Vec<Option<WindowHit>> = cs.iter().map(|&c| window_hit(seq, c)).collect();
    let mut streak = 0usize;
    let mut first_sustained = None;
    for (i, h) in hits.iter().enumerate() {
        if let Some(h) = h {
            streak += 1;
            let required = match h {
                WindowHit::Periodic(_) => 2,
                WindowHit::Repeats => 3,
            };
            if streak >= required {
                first_sustained = Some(i);
                break;
            }
        } else {
            streak = 0;
        }
    }
    let first_sustained = first_sustained?;
    let mut run_start = None;
    for i in (0..cs.len()).rev() {
        if hits[i].is_none() {
            break;
        }
        run_start = Some(i);
    }
    let Some(i) = run_start else {
        return Some(LoopDetection {
            onset: n,
            period: 0,
        });
    };
    let final_run = onset_from_hit(seq, cs[i], hits[i].unwrap());
    let first = onset_from_hit(seq, cs[first_sustained], hits[first_sustained].unwrap());
    Some(if final_run.onset >= first.onset {
        final_run
    } else {
        first
    })
}

pub fn strip_grounding_tokens(tokens: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut skip_until: Option<u32> = None;
    for &t in tokens {
        if let Some(close) = skip_until {
            if t == close {
                skip_until = None;
            }
            continue;
        }
        match t {
            REF_OPEN_TOKEN_ID => skip_until = Some(REF_CLOSE_TOKEN_ID),
            DET_OPEN_TOKEN_ID => skip_until = Some(DET_CLOSE_TOKEN_ID),
            GROUNDING_TOKEN_ID => {}
            _ => out.push(t),
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct GenerateOutcome {
    pub tokens: Vec<u32>,
    pub loop_detection: Option<LoopDetection>,
    pub hit_eos: bool,
}

pub const PROMPT_GROUNDING_MARKDOWN: &str =
    "<image>\n<|grounding|>Convert the document to markdown.";
pub const PROMPT_FREE_OCR: &str = "<image>\nFree OCR.";
pub const PROMPT_PARSE_FIGURE: &str = "<image>\nParse the figure.";

#[derive(Clone, Debug)]
pub struct DeepseekOcrDecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl DeepseekOcrDecoderConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).context("parse deepseek-ocr2 config json")?;
        let lang = v.get("language_config");
        let field = |k: &str| -> Option<serde_json::Value> {
            v.get(k)
                .filter(|x| !x.is_null())
                .or_else(|| lang.and_then(|l| l.get(k)).filter(|x| !x.is_null()))
                .cloned()
        };
        let get_u = |k: &str| -> Result<usize> {
            field(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let get_u_or = |k: &str, d: usize| -> usize {
            field(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        let get_f_or = |k: &str, d: f64| -> f64 { field(k).and_then(|x| x.as_f64()).unwrap_or(d) };
        let get_b_or =
            |k: &str, d: bool| -> bool { field(k).and_then(|x| x.as_bool()).unwrap_or(d) };

        if get_b_or("use_mla", false) {
            anyhow::bail!("use_mla=true checkpoints are not supported (this port is MHA-only)");
        }
        if let Some(x) = field("kv_lora_rank") {
            if !x.is_null() {
                anyhow::bail!("kv_lora_rank set -- MLA checkpoint, not supported");
            }
        }
        let hidden_size = get_u("hidden_size")?;
        let num_attention_heads = get_u("num_attention_heads")?;
        if hidden_size % num_attention_heads != 0 {
            anyhow::bail!(
                "hidden_size {hidden_size} not divisible by num_attention_heads {num_attention_heads}"
            );
        }
        Ok(Self {
            hidden_size,
            num_hidden_layers: get_u("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: get_u_or("num_key_value_heads", num_attention_heads),
            intermediate_size: get_u("intermediate_size")?,
            moe_intermediate_size: get_u("moe_intermediate_size")?,
            n_routed_experts: get_u("n_routed_experts")?,
            n_shared_experts: get_u_or("n_shared_experts", 0),
            num_experts_per_tok: get_u("num_experts_per_tok")?,
            first_k_dense_replace: get_u_or("first_k_dense_replace", 0),
            moe_layer_freq: get_u_or("moe_layer_freq", 1),
            vocab_size: get_u("vocab_size")?,
            max_position_embeddings: get_u_or("max_position_embeddings", 8192),
            rms_norm_eps: get_f_or("rms_norm_eps", 1e-6),
            rope_theta: get_f_or("rope_theta", 10_000.0) as f32,
            norm_topk_prob: get_b_or("norm_topk_prob", false),
            routed_scaling_factor: get_f_or("routed_scaling_factor", 1.0),
            bos_token_id: get_u_or("bos_token_id", 0) as u32,
            eos_token_id: get_u_or("eos_token_id", 1) as u32,
        })
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.n_shared_experts * self.moe_intermediate_size
    }

    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.n_routed_experts > 0
            && idx >= self.first_k_dense_replace
            && idx.is_multiple_of(self.moe_layer_freq.max(1))
    }
}

pub struct DeepseekOcrKvCache {
    layers: Vec<(Tensor, Tensor)>,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl DeepseekOcrKvCache {
    pub fn new(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let shape = (1usize, max_seq_len, n_kv_heads, head_dim);
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = Tensor::zeros(shape, dtype, device)?;
            let v = Tensor::zeros(shape, dtype, device)?;
            layers.push((k, v));
        }
        Ok(Self {
            layers,
            current_len: 0,
            max_seq_len,
            n_kv_heads,
            head_dim,
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

    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }

    fn write_at(
        &mut self,
        layer: usize,
        start: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != self.n_kv_heads || dims[3] != self.head_dim
        {
            anyhow::bail!(
                "KvCache.write_at: expected [1, t, {}, {}], got {:?}",
                self.n_kv_heads,
                self.head_dim,
                dims
            );
        }
        let t = dims[1];
        let end = start + t;
        if end > self.max_seq_len {
            anyhow::bail!(
                "KvCache.write_at: end {} exceeds max_seq_len {}",
                end,
                self.max_seq_len
            );
        }
        let (k_buf, v_buf) = &self.layers[layer];
        let ranges = [0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim];
        let k_updated = k_buf.slice_assign(&ranges, k_new)?;
        let v_updated = v_buf.slice_assign(&ranges, v_new)?;
        self.layers[layer] = (k_updated, v_updated);
        Ok(())
    }

    fn view(&self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let (k, v) = &self.layers[layer];
        Ok((k.narrow(1, 0, len)?, v.narrow(1, 0, len)?))
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn layer_bufs(&self, layer: usize) -> &(Tensor, Tensor) {
        &self.layers[layer]
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct StackedExperts {
    pub(crate) gate: Tensor,
    pub(crate) up: Tensor,
    pub(crate) down: Tensor,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct DeepseekMoe {
    gate_t_f32: Tensor,
    gate_bf16: Linear,
    experts: Vec<Mlp>,
    shared: Mlp,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    hidden: usize,
    stacked: Option<StackedExperts>,
}

impl DeepseekMoe {
    pub fn new(
        gate_weight: Tensor,
        experts: Vec<Mlp>,
        shared: Mlp,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f64,
    ) -> Result<Self> {
        let gd = gate_weight.dims();
        if gd.len() != 2 {
            anyhow::bail!("moe gate weight must be 2-D, got {:?}", gd);
        }
        let num_experts = gd[0];
        let hidden = gd[1];
        if experts.len() != num_experts {
            anyhow::bail!(
                "moe: {} experts loaded, gate expects {}",
                experts.len(),
                num_experts
            );
        }
        if top_k == 0 || top_k > num_experts {
            anyhow::bail!("moe: top_k {} invalid for {} experts", top_k, num_experts);
        }
        let gate_bf16 = Linear::new(gate_weight.to_dtype(DType::BF16)?, None)?;
        let gate_t_f32 = gate_weight.to_dtype(DType::F32)?.t()?.contiguous()?;
        Ok(Self {
            gate_t_f32,
            gate_bf16,
            experts,
            shared,
            num_experts,
            top_k,
            norm_topk_prob,
            routed_scaling_factor,
            hidden,
            stacked: None,
        })
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn gate_bf16(&self) -> &Linear {
        &self.gate_bf16
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn shared_expert(&self) -> &Mlp {
        &self.shared
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn stacked(&self) -> Option<&StackedExperts> {
        self.stacked.as_ref()
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn num_experts(&self) -> usize {
        self.num_experts
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn decode_ready(&self) -> bool {
        self.stacked.is_some() && !self.norm_topk_prob && self.routed_scaling_factor == 1.0
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn stack_experts_cuda(&mut self, hidden: usize, inter: usize) -> Result<bool> {
        if self.stacked.is_some() {
            return Ok(true);
        }
        if self.norm_topk_prob || self.routed_scaling_factor != 1.0 {
            return Ok(false);
        }
        let mut gs = Vec::with_capacity(self.num_experts);
        let mut us = Vec::with_capacity(self.num_experts);
        let mut ds = Vec::with_capacity(self.num_experts);
        for m in &self.experts {
            let (Some(g), Some(u), Some(d)) = (
                m.gate_proj().weight(),
                m.up_proj().weight(),
                m.down_proj().weight(),
            ) else {
                return Ok(false);
            };
            if g.dtype() != DType::BF16 || !matches!(g.device(), Device::Cuda(_)) {
                return Ok(false);
            }
            if g.dims() != [inter, hidden]
                || u.dims() != [inter, hidden]
                || d.dims() != [hidden, inter]
            {
                return Ok(false);
            }
            gs.push(g.unsqueeze(0)?);
            us.push(u.unsqueeze(0)?);
            ds.push(d.unsqueeze(0)?);
        }
        let gate = Tensor::cat(&gs, 0)?.contiguous()?;
        let up = Tensor::cat(&us, 0)?.contiguous()?;
        let down = Tensor::cat(&ds, 0)?.contiguous()?;
        let mut views = Vec::with_capacity(self.num_experts);
        for e in 0..self.num_experts {
            let g = Linear::new(gate.narrow(0, e, 1)?.squeeze(0)?, None)?;
            let u = Linear::new(up.narrow(0, e, 1)?.squeeze(0)?, None)?;
            let d = Linear::new(down.narrow(0, e, 1)?.squeeze(0)?, None)?;
            views.push(Mlp::new(g, u, d)?);
        }
        self.experts = views;
        self.stacked = Some(StackedExperts { gate, up, down });
        Ok(true)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims().to_vec();
        let in_dtype = x.dtype();
        let device = x.device().clone();
        let n_tokens: usize = in_dims[..in_dims.len() - 1].iter().product();
        let x_flat = x.reshape((n_tokens, self.hidden))?.contiguous()?;

        let probs = prof::span(prof::MOE_GATE, &device, || -> Result<Tensor> {
            let scores = x_flat.to_dtype(DType::F32)?.matmul(&self.gate_t_f32)?;
            Ok(candle_nn::ops::softmax_last_dim(&scores)?)
        })?;
        let mut shared_early: Option<Tensor> = None;

        #[allow(unused_mut)]
        let mut overlapped: Option<Vec<f32>> = None;
        #[cfg(feature = "cuda")]
        if moe_overlap_shared() && !prof::enabled() {
            overlapped = read_probs_overlapped(&probs, || {
                let t = self.shared.forward(&x_flat)?;
                shared_early = Some(t.to_dtype(DType::F32)?);
                Ok(())
            })?;
        }
        let probs_host: Vec<f32> = match overlapped {
            Some(h) => h,
            None => {
                shared_early = None;
                prof::span(prof::MOE_D2H, &device, || -> Result<Vec<f32>> {
                    Ok(probs.flatten_all()?.to_vec1::<f32>()?)
                })?
            }
        };

        let k = self.top_k;
        let (expert_rows, expert_w) = prof::span(prof::MOE_ROUTE, &device, || {
            route_topk(
                &probs_host,
                n_tokens,
                self.num_experts,
                k,
                self.norm_topk_prob,
                self.routed_scaling_factor as f32,
            )
        });

        let acc = prof::span(prof::MOE_EXPERTS, &device, || -> Result<Tensor> {
            let mut acc = Tensor::zeros((n_tokens, self.hidden), DType::F32, &device)?;
            if !moe_batch_gather() {
                for e in 0..self.num_experts {
                    let rows = &expert_rows[e];
                    if rows.is_empty() {
                        continue;
                    }
                    let m = rows.len();
                    let idx_t = Tensor::from_vec(rows.clone(), m, &device)?;
                    let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;
                    let y_e = self.experts[e].forward(&gathered)?.to_dtype(DType::F32)?;
                    let w_t = Tensor::from_vec(expert_w[e].clone(), (m, 1), &device)?;
                    acc = acc.index_add(&idx_t, &y_e.broadcast_mul(&w_t)?, 0)?;
                }
                return Ok(acc);
            }

            let total: usize = expert_rows.iter().map(|r| r.len()).sum();
            if total == 0 {
                return Ok(acc);
            }
            let mut idx_all: Vec<u32> = Vec::with_capacity(total);
            let mut w_all: Vec<f32> = Vec::with_capacity(total);
            let mut segs: Vec<(usize, usize)> = Vec::with_capacity(self.num_experts);
            for e in 0..self.num_experts {
                segs.push((idx_all.len(), expert_rows[e].len()));
                idx_all.extend_from_slice(&expert_rows[e]);
                w_all.extend_from_slice(&expert_w[e]);
            }
            let idx_t = Tensor::from_vec(idx_all, total, &device)?;
            let w_t = Tensor::from_vec(w_all, (total, 1), &device)?;
            let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;
            for e in 0..self.num_experts {
                let (off, m) = segs[e];
                if m == 0 {
                    continue;
                }
                let g = gathered.narrow(0, off, m)?;
                let y_e = self.experts[e].forward(&g)?.to_dtype(DType::F32)?;
                let w = w_t.narrow(0, off, m)?;
                let i = idx_t.narrow(0, off, m)?;
                acc = acc.index_add(&i, &y_e.broadcast_mul(&w)?, 0)?;
            }
            Ok(acc)
        })?;

        let shared_out = match shared_early {
            Some(t) => t,
            None => prof::span(prof::MOE_SHARED, &device, || -> Result<Tensor> {
                Ok(self.shared.forward(&x_flat)?.to_dtype(DType::F32)?)
            })?,
        };
        prof::span(prof::MOE_COMBINE, &device, || -> Result<Tensor> {
            let y = acc.add(&shared_out)?;
            Ok(y.reshape(in_dims)?.to_dtype(in_dtype)?)
        })
    }
}

pub(crate) fn moe_batch_gather() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_MOE_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn moe_overlap_shared() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_MOE_OVERLAP")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

#[cfg(feature = "cuda")]
mod probe_copy {
    use anyhow::Result;
    use candle_core::{CudaDevice, DType, Device, Tensor};
    use cudarc::driver::{CudaEvent, CudaStream, DevicePtr};
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::sync::Arc;

    struct HostPinned {
        ptr: *mut f32,
        len: usize,
    }

    impl Drop for HostPinned {
        fn drop(&mut self) {
            unsafe {
                let _ = cudarc::driver::result::free_host(self.ptr as *mut c_void);
            }
        }
    }

    struct Lane {
        stream: Arc<CudaStream>,
        gate: CudaEvent,
        copied: CudaEvent,
        host: Option<HostPinned>,
    }

    thread_local! {
        static LANE: RefCell<Option<Lane>> = const { RefCell::new(None) };
    }

    fn lane(dev: &CudaDevice) -> Result<()> {
        LANE.with(|c| {
            let mut slot = c.borrow_mut();
            if slot.is_some() {
                return Ok(());
            }
            let ctx = dev.cuda_stream().context().clone();
            let stream = ctx
                .new_stream()
                .map_err(|e| anyhow::anyhow!("dsocr moe copy stream: {e:?}"))?;
            let nt = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
            let gate = ctx.new_event(nt).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let copied = ctx.new_event(nt).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            *slot = Some(Lane {
                stream,
                gate,
                copied,
                host: None,
            });
            Ok(())
        })
    }

    pub fn read_overlapped<F: FnOnce() -> Result<()>>(
        probs: &Tensor,
        enqueue: F,
    ) -> Result<Option<Vec<f32>>> {
        let Device::Cuda(dev) = probs.device() else {
            return Ok(None);
        };
        if probs.dtype() != DType::F32 {
            return Ok(None);
        }
        let (storage, layout) = probs.storage_and_layout();
        if !layout.is_contiguous() {
            return Ok(None);
        }
        let cuda = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let n = layout.shape().elem_count();
        let slice = cuda.as_cuda_slice::<f32>()?;
        let legacy = dev.cuda_stream();
        let (src, _g) = slice.device_ptr(&legacy);
        let src = src + (layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        lane(dev)?;
        let out = LANE.with(|c| -> Result<Vec<f32>> {
            let mut slot = c.borrow_mut();
            let l = slot.as_mut().expect("lane initialised");
            if l.host.as_ref().map(|h| h.len).unwrap_or(0) < n {
                l.host = None;
                let ptr = unsafe {
                    cudarc::driver::result::malloc_host(n * std::mem::size_of::<f32>(), 0)
                }
                .map_err(|e| anyhow::anyhow!("pinned moe probs buffer: {e:?}"))?;
                l.host = Some(HostPinned {
                    ptr: ptr as *mut f32,
                    len: n,
                });
            }
            let host = l.host.as_ref().unwrap();

            l.gate
                .record(&legacy)
                .map_err(|e| anyhow::anyhow!("record gate event: {e:?}"))?;
            l.stream
                .wait(&l.gate)
                .map_err(|e| anyhow::anyhow!("copy stream wait: {e:?}"))?;
            unsafe {
                let dst = std::slice::from_raw_parts_mut(host.ptr, n);
                cudarc::driver::result::memcpy_dtoh_async(dst, src, l.stream.cu_stream())
            }
            .map_err(|e| anyhow::anyhow!("moe probs dtoh: {e:?}"))?;
            l.copied
                .record(&l.stream)
                .map_err(|e| anyhow::anyhow!("record copy event: {e:?}"))?;

            enqueue()?;

            l.copied
                .synchronize()
                .map_err(|e| anyhow::anyhow!("wait moe probs copy: {e:?}"))?;
            Ok(unsafe { std::slice::from_raw_parts(host.ptr, n) }.to_vec())
        })?;
        Ok(Some(out))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use probe_copy::read_overlapped as read_probs_overlapped;

pub(crate) fn route_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NV_DSOCR_MOE_ROUTE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(1)
    })
}

fn route_legacy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_DSOCR_MOE_ROUTE").as_deref() == Ok("sort"))
}

pub(crate) fn topk_stable_desc(row: &[f32], k: usize, ids: &mut [u32], vals: &mut [f32]) {
    for j in 0..k {
        ids[j] = u32::MAX;
        vals[j] = f32::NEG_INFINITY;
    }
    for (e, &v) in row.iter().enumerate() {
        if !(v > vals[k - 1]) {
            continue;
        }
        let mut j = k - 1;
        while j > 0 && v > vals[j - 1] {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j -= 1;
        }
        vals[j] = v;
        ids[j] = e as u32;
    }
}

fn route_chunk(
    probs: &[f32],
    n0: usize,
    n1: usize,
    e_count: usize,
    k: usize,
    norm: bool,
    scale: f32,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); e_count];
    let mut ws: Vec<Vec<f32>> = vec![Vec::new(); e_count];
    let mut ids = vec![0u32; k];
    let mut vals = vec![0f32; k];
    for n in n0..n1 {
        let row = &probs[n * e_count..(n + 1) * e_count];
        topk_stable_desc(row, k, &mut ids, &mut vals);
        let denom = if norm {
            let mut s = 0f32;
            for j in 0..k {
                s += row[ids[j] as usize];
            }
            s.max(1e-20)
        } else {
            1.0
        };
        for j in 0..k {
            let e = ids[j] as usize;
            rows[e].push(n as u32);
            ws[e].push(row[e] / denom * scale);
        }
    }
    (rows, ws)
}

pub(crate) fn route_topk(
    probs: &[f32],
    n_tokens: usize,
    e_count: usize,
    k: usize,
    norm: bool,
    scale: f32,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    if route_legacy() {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); e_count];
        let mut ws: Vec<Vec<f32>> = vec![Vec::new(); e_count];
        let mut order: Vec<usize> = Vec::with_capacity(e_count);
        for n in 0..n_tokens {
            let row = &probs[n * e_count..(n + 1) * e_count];
            order.clear();
            order.extend(0..e_count);
            order.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sel = &order[..k];
            let denom = if norm {
                sel.iter().map(|&e| row[e]).sum::<f32>().max(1e-20)
            } else {
                1.0
            };
            for &e in sel {
                rows[e].push(n as u32);
                ws[e].push(row[e] / denom * scale);
            }
        }
        return (rows, ws);
    }

    let nt = route_threads().min(n_tokens.div_ceil(64)).max(1);
    if nt == 1 {
        return route_chunk(probs, 0, n_tokens, e_count, k, norm, scale);
    }
    let per = n_tokens.div_ceil(nt);
    let parts: Vec<(Vec<Vec<u32>>, Vec<Vec<f32>>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt)
            .map(|t| {
                let n0 = t * per;
                let n1 = ((t + 1) * per).min(n_tokens);
                s.spawn(move || route_chunk(probs, n0, n1, e_count, k, norm, scale))
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("route chunk"))
            .collect()
    });

    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); e_count];
    let mut ws: Vec<Vec<f32>> = vec![Vec::new(); e_count];
    for (pr, pw) in parts {
        for e in 0..e_count {
            rows[e].extend_from_slice(&pr[e]);
            ws[e].extend_from_slice(&pw[e]);
        }
    }
    (rows, ws)
}

pub(crate) enum FeedForward {
    Dense(Mlp),
    Moe(DeepseekMoe),
}

impl FeedForward {}

pub(crate) struct DecoderLayer {
    pub(crate) input_layernorm: RmsNorm,
    pub(crate) post_attention_layernorm: RmsNorm,
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) ff: FeedForward,
}

pub struct DeepseekOcrDecoder {
    config: DeepseekOcrDecoderConfig,
    embed_weight: Tensor,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    dtype: DType,
    device: Device,
    #[cfg(feature = "cuda")]
    decode_scratch: Mutex<Option<super::decoder_graph::DecodeScratch>>,
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub ngram_window: Option<usize>,
    pub ngram_whitelist: Vec<u32>,
    pub seed: u64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        match std::env::var("NV_DSOCR_NGRAM").as_deref() {
            Ok("hf20") => Self::hf20(),
            Ok("off") => Self {
                no_repeat_ngram_size: None,
                ngram_window: None,
                ngram_whitelist: Vec::new(),
                ..Self::recipe()
            },
            _ => Self::recipe(),
        }
    }
}

impl GenerateOptions {
    pub fn recipe() -> Self {
        Self {
            max_new_tokens: 8192,
            temperature: 0.0,
            no_repeat_ngram_size: Some(30),
            ngram_window: Some(90),
            ngram_whitelist: vec![TD_OPEN_TOKEN_ID, TD_CLOSE_TOKEN_ID],
            seed: 0,
        }
    }

    pub fn hf20() -> Self {
        Self {
            no_repeat_ngram_size: Some(20),
            ngram_window: None,
            ngram_whitelist: Vec::new(),
            ..Self::recipe()
        }
    }
}

trait LinearFactory {
    fn load(
        &self,
        weights: &WeightLoader,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Result<Linear>;
}

struct Bf16Factory {
    dtype: DType,
}

impl LinearFactory for Bf16Factory {
    fn load(
        &self,
        weights: &WeightLoader,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Result<Linear> {
        load_linear(weights, name, out_features, in_features, self.dtype)
    }
}

#[cfg(feature = "cuda")]
struct Nvfp4Factory {
    dtype: DType,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: Device,
}

#[cfg(feature = "cuda")]
impl LinearFactory for Nvfp4Factory {
    fn load(
        &self,
        weights: &WeightLoader,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Result<Linear> {
        const MIN: usize = nv_quant::nvfp4::MIN_TILE;
        if out_features < MIN || in_features < MIN {
            return load_linear(weights, name, out_features, in_features, self.dtype);
        }
        let w = weights
            .get(name, DType::BF16)
            .with_context(|| format!("load {name}"))?;
        let d = w.dims();
        if d.len() != 2 || d[0] != out_features || d[1] != in_features {
            anyhow::bail!(
                "linear {name}: expected [{}, {}], got {:?}",
                out_features,
                in_features,
                d
            );
        }
        Linear::from_bf16_quantized_nvfp4(&w, None, &self.device, self.runner.clone())
    }
}

pub fn env_decoder_dtype() -> DType {
    match std::env::var("NV_DSOCR_DEC_DTYPE").ok().as_deref() {
        Some("f32") | Some("fp32") | Some("float32") => DType::F32,
        _ => DType::BF16,
    }
}

impl DeepseekOcrDecoder {
    pub fn config(&self) -> &DeepseekOcrDecoderConfig {
        &self.config
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn rope(&self) -> &Rope {
        &self.rope
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn embed_weight_t(&self) -> &Tensor {
        &self.embed_weight
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn final_norm(&self) -> &RmsNorm {
        &self.final_norm
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<DeepseekOcrKvCache> {
        DeepseekOcrKvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim(),
            max_seq_len,
            &self.device,
            self.dtype,
        )
    }

    pub fn from_loader(
        config: DeepseekOcrDecoderConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        Self::from_loader_with_dtype(config, weights, device, env_decoder_dtype())
    }

    pub fn from_loader_with_dtype(
        config: DeepseekOcrDecoderConfig,
        weights: &WeightLoader,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let factory = Bf16Factory { dtype };
        Self::build(config, weights, device, dtype, &factory)
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_nvfp4(
        config: DeepseekOcrDecoderConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("from_loader_nvfp4 requires a CUDA device"),
        };
        let runner = Arc::new(Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
            dev.cuda_stream(),
        )?));
        let factory = Nvfp4Factory {
            dtype: DType::BF16,
            runner,
            device: device.clone(),
        };
        Self::build(config, weights, device, DType::BF16, &factory)
    }

    fn build(
        config: DeepseekOcrDecoderConfig,
        weights: &WeightLoader,
        device: &Device,
        dtype: DType,
        factory: &dyn LinearFactory,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let embed_weight = weights
            .get("model.embed_tokens.weight", dtype)
            .context("load model.embed_tokens.weight")?;
        let ed = embed_weight.dims();
        if ed.len() != 2 || ed[0] != config.vocab_size || ed[1] != hidden {
            anyhow::bail!(
                "embedding shape mismatch: expected [{}, {}], got {:?}",
                config.vocab_size,
                hidden,
                ed
            );
        }

        let qd = config.num_attention_heads * config.head_dim();
        let kvd = config.num_key_value_heads * config.head_dim();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let input_layernorm = load_rmsnorm(
                weights,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                config.rms_norm_eps,
                dtype,
            )?;
            let post_attention_layernorm = load_rmsnorm(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                config.rms_norm_eps,
                dtype,
            )?;
            let q_proj = factory.load(
                weights,
                &format!("{prefix}.self_attn.q_proj.weight"),
                qd,
                hidden,
            )?;
            let k_proj = factory.load(
                weights,
                &format!("{prefix}.self_attn.k_proj.weight"),
                kvd,
                hidden,
            )?;
            let v_proj = factory.load(
                weights,
                &format!("{prefix}.self_attn.v_proj.weight"),
                kvd,
                hidden,
            )?;
            let o_proj = factory.load(
                weights,
                &format!("{prefix}.self_attn.o_proj.weight"),
                hidden,
                qd,
            )?;

            let ff = if config.is_moe_layer(i) {
                let gate_weight = weights
                    .get(&format!("{prefix}.mlp.gate.weight"), DType::F32)
                    .with_context(|| format!("load {prefix}.mlp.gate.weight"))?;
                let gd = gate_weight.dims();
                if gd != [config.n_routed_experts, hidden] {
                    anyhow::bail!(
                        "router {prefix}.mlp.gate.weight: expected [{}, {}], got {:?}",
                        config.n_routed_experts,
                        hidden,
                        gd
                    );
                }
                let mut experts = Vec::with_capacity(config.n_routed_experts);
                for e in 0..config.n_routed_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    experts.push(load_mlp(
                        factory,
                        weights,
                        &ep,
                        hidden,
                        config.moe_intermediate_size,
                    )?);
                }
                let shared = load_mlp(
                    factory,
                    weights,
                    &format!("{prefix}.mlp.shared_experts"),
                    hidden,
                    config.shared_expert_intermediate_size(),
                )?;
                FeedForward::Moe(DeepseekMoe::new(
                    gate_weight,
                    experts,
                    shared,
                    config.num_experts_per_tok,
                    config.norm_topk_prob,
                    config.routed_scaling_factor,
                )?)
            } else {
                FeedForward::Dense(load_mlp(
                    factory,
                    weights,
                    &format!("{prefix}.mlp"),
                    hidden,
                    config.intermediate_size,
                )?)
            };
            layers.push(DecoderLayer {
                input_layernorm,
                post_attention_layernorm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                ff,
            });
        }

        let final_norm = load_rmsnorm(
            weights,
            "model.norm.weight",
            hidden,
            config.rms_norm_eps,
            dtype,
        )?;
        let lm_head_w = weights
            .get("lm_head.weight", dtype)
            .context("load lm_head.weight (untied)")?;
        let lm_head = Linear::new(lm_head_w, None)?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: config.head_dim(),
                max_seq_len: config.max_position_embeddings,
                base: config.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) && dtype == DType::BF16 {
            for layer in layers.iter_mut() {
                if let FeedForward::Moe(m) = &mut layer.ff {
                    let _ = m.stack_experts_cuda(hidden, config.moe_intermediate_size)?;
                }
            }
        }
        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            rope,
            dtype,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            decode_scratch: Mutex::new(None),
        })
    }

    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        self.embed_tokens_with_vision(tokens, None)
    }

    pub fn embed_tokens_with_vision(
        &self,
        tokens: &[u32],
        vision_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq = tokens.len();
        if seq == 0 {
            anyhow::bail!("embed_tokens: empty token sequence");
        }
        let n_img = tokens.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
        if n_img > 0 && vision_features.is_none() {
            anyhow::bail!("{n_img} image placeholder tokens but no vision features supplied");
        }
        let lookup: Vec<u32> = tokens
            .iter()
            .map(|&t| {
                if t == IMAGE_TOKEN_ID && (t as usize) >= self.config.vocab_size {
                    0
                } else {
                    t
                }
            })
            .collect();
        let ids = Tensor::from_vec(lookup, seq, &self.device)?;
        let mut x = self.embed_weight.index_select(&ids, 0)?.reshape((
            1usize,
            seq,
            self.config.hidden_size,
        ))?;
        if x.dtype() != self.dtype {
            x = x.to_dtype(self.dtype)?;
        }

        let Some(vision) = vision_features else {
            return Ok(x);
        };
        let mut v = vision.clone();
        if v.dims().len() == 3 && v.dims()[0] == 1 {
            v = v.squeeze(0)?;
        }
        let vd = v.dims().to_vec();
        if vd.len() != 2 || vd[1] != self.config.hidden_size {
            anyhow::bail!(
                "vision features must be [N, {}], got {:?}",
                self.config.hidden_size,
                vd
            );
        }
        if vd[0] != n_img {
            anyhow::bail!(
                "vision feature count {} != image placeholder count {}",
                vd[0],
                n_img
            );
        }
        if v.dtype() != self.dtype {
            v = v.to_dtype(self.dtype)?;
        }
        let mut offset = 0usize;
        let mut i = 0usize;
        while i < seq {
            if tokens[i] == IMAGE_TOKEN_ID {
                let start = i;
                while i < seq && tokens[i] == IMAGE_TOKEN_ID {
                    i += 1;
                }
                let run = i - start;
                let chunk = v.narrow(0, offset, run)?.unsqueeze(0)?;
                x = x.slice_assign(
                    &[0..1, start..start + run, 0..self.config.hidden_size],
                    &chunk,
                )?;
                offset += run;
            } else {
                i += 1;
            }
        }
        Ok(x)
    }

    pub fn forward_embeds(&self, x: &Tensor, cache: &mut DeepseekOcrKvCache) -> Result<Tensor> {
        let hidden = self.forward_embeds_hidden(x, cache)?;
        self.lm_head_forward(&hidden)
    }

    pub fn lm_head_forward(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden)
    }

    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        vision_features: Option<&Tensor>,
        cache: &mut DeepseekOcrKvCache,
    ) -> Result<Tensor> {
        let x = self.embed_tokens_with_vision(tokens, vision_features)?;
        self.forward_embeds(&x, cache)
    }

    pub fn forward_embeds_hidden(
        &self,
        x: &Tensor,
        cache: &mut DeepseekOcrKvCache,
    ) -> Result<Tensor> {
        self.forward_embeds_hidden_taps(x, cache, None)
    }

    pub fn forward_embeds_hidden_taps(
        &self,
        x: &Tensor,
        cache: &mut DeepseekOcrKvCache,
        mut taps: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != self.config.hidden_size {
            anyhow::bail!(
                "forward_embeds_hidden: expected [1, T, {}], got {:?}",
                self.config.hidden_size,
                dims
            );
        }
        let seq = dims[1];
        let write_start = cache.current_len();
        let new_total = write_start + seq;
        if new_total > cache.max_seq_len() {
            anyhow::bail!(
                "sequence overflows kv cache: {} + {} > {}",
                write_start,
                seq,
                cache.max_seq_len()
            );
        }
        let positions: Vec<u32> = (write_start as u32..new_total as u32).collect();
        let positions = Tensor::from_vec(positions, (1usize, seq), &self.device)?;

        let mut h = x.clone();
        if h.dtype() != self.dtype {
            h = h.to_dtype(self.dtype)?;
        }
        for i in 0..self.layers.len() {
            if let Some(t) = taps.as_deref_mut() {
                t.push(h.clone());
            }
            h = self.layer_forward(i, &h, &positions, cache, seq, write_start, new_total)?;
        }
        cache.advance(seq);
        let out = prof::span(prof::FINAL_NORM, &self.device, || {
            self.final_norm.forward(&h)
        })?;
        prof::note_call();
        if let Some(t) = taps {
            t.push(h.clone());
            t.push(out.clone());
        }
        Ok(out)
    }

    fn layer_forward(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut DeepseekOcrKvCache,
        seq: usize,
        write_start: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();

        let dv = &self.device;
        let normed = prof::span(prof::NORM1, dv, || layer.input_layernorm.forward(x))?;
        let (q, k, v) = prof::span(prof::QKV, dv, || -> Result<_> {
            let q = layer
                .q_proj
                .forward(&normed)?
                .reshape((1usize, seq, n_heads, head_dim))?;
            let k = layer
                .k_proj
                .forward(&normed)?
                .reshape((1usize, seq, n_kv_heads, head_dim))?;
            let v = layer
                .v_proj
                .forward(&normed)?
                .reshape((1usize, seq, n_kv_heads, head_dim))?;
            Ok((q, k, v))
        })?;

        let (q, k_new) = prof::span(prof::ROPE, dv, || -> Result<_> {
            let q_f32 = q.to_dtype(DType::F32)?;
            let k_f32 = k.to_dtype(DType::F32)?;
            let (q_rot, k_rot) = self.rope.apply(&q_f32, &k_f32, positions)?;
            Ok((q_rot.to_dtype(self.dtype)?, k_rot.to_dtype(self.dtype)?))
        })?;

        let (k_full, v_full) = prof::span(prof::KV_WRITE, dv, || -> Result<_> {
            cache.write_at(idx, write_start, &k_new.contiguous()?, &v.contiguous()?)?;
            cache.view(idx, new_total)
        })?;

        let attn_cfg = AttnConfig {
            num_heads: n_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = prof::span(prof::ATTN, dv, || {
            attention(
                &q.contiguous()?,
                &k_full.contiguous()?,
                &v_full.contiguous()?,
                &attn_cfg,
            )
        })?;
        let attn_out = attn_out.reshape((1usize, seq, n_heads * head_dim))?;
        let attn_out = prof::span(prof::O_PROJ, dv, || layer.o_proj.forward(&attn_out))?;

        let x_after = x.add(&attn_out)?;
        let normed2 = prof::span(prof::NORM2, dv, || {
            layer.post_attention_layernorm.forward(&x_after)
        })?;
        #[cfg(feature = "cuda")]
        if seq == 1
            && self.dtype == DType::BF16
            && matches!(self.device, Device::Cuda(_))
            && super::decoder_graph::kernel_decode_enabled()
        {
            if let Some(x_next) = self.try_decode_ffn(idx, &normed2, &x_after)? {
                return Ok(x_next);
            }
        }
        let ff_out = match &layer.ff {
            FeedForward::Dense(mlp) => prof::span(prof::FFN_DENSE, dv, || mlp.forward(&normed2))?,
            FeedForward::Moe(moe) => moe.forward(&normed2)?,
        };
        Ok(x_after.add(&ff_out)?)
    }

    #[cfg(feature = "cuda")]
    fn try_decode_ffn(
        &self,
        idx: usize,
        normed2: &Tensor,
        x_after: &Tensor,
    ) -> Result<Option<Tensor>> {
        use super::decoder_graph;
        match &self.layers[idx].ff {
            FeedForward::Dense(mlp) => Ok(Some(decoder_graph::dense_decode_ffn(
                mlp,
                normed2,
                x_after,
                &self.device,
            )?)),
            FeedForward::Moe(m) => {
                if !m.decode_ready() {
                    return Ok(None);
                }
                let mut guard = self
                    .decode_scratch
                    .lock()
                    .map_err(|e| anyhow::anyhow!("decode scratch lock poisoned: {e}"))?;
                if guard.is_none() {
                    *guard = Some(decoder_graph::DecodeScratch::new(
                        self.config.num_experts_per_tok,
                        self.config.n_routed_experts,
                        self.config.moe_intermediate_size,
                        self.config.hidden_size,
                        &self.device,
                    )?);
                }
                let scratch = guard.as_mut().unwrap();
                Ok(Some(decoder_graph::moe_decode_ffn(
                    m,
                    normed2,
                    x_after,
                    scratch,
                    None,
                    0,
                    &self.device,
                )?))
            }
        }
    }

    pub(crate) fn last_logits(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let t = hidden.dim(1)?;
        let last = hidden.narrow(1, t - 1, 1)?;
        let logits = self.lm_head.forward(&last)?;
        Ok(logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        vision_features: Option<&Tensor>,
        opts: &GenerateOptions,
    ) -> Result<Vec<u32>> {
        let outcome = self.generate_detected(prompt_tokens, vision_features, opts)?;
        let mut tokens = outcome.tokens;
        if let Some(d) = outcome.loop_detection {
            tokens.truncate(d.onset);
        }
        Ok(tokens)
    }

    pub fn generate_detected(
        &self,
        prompt_tokens: &[u32],
        vision_features: Option<&Tensor>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let max_len =
            (prompt_tokens.len() + opts.max_new_tokens).min(self.config.max_position_embeddings);
        if prompt_tokens.len() >= max_len {
            anyhow::bail!(
                "prompt length {} leaves no room to generate (max {})",
                prompt_tokens.len(),
                max_len
            );
        }
        let mut cache = self.new_kv_cache(max_len)?;
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut rng = SplitMix64::new(opts.seed);

        let x = self.embed_tokens_with_vision(prompt_tokens, vision_features)?;
        let mut hidden = self.forward_embeds_hidden(&x, &mut cache)?;

        let mut hit_eos = false;
        loop {
            let mut logits = self.last_logits(&hidden)?;
            let next = select_next_token(&mut logits, &all_tokens, opts, &mut rng)?;
            generated.push(next);
            all_tokens.push(next);
            if next == self.config.eos_token_id {
                hit_eos = true;
                break;
            }
            if generated.len() >= opts.max_new_tokens {
                break;
            }
            if generated.len().is_multiple_of(LOOP_CHECK_STRIDE) {
                if let Some(d) = detect_loop(&generated) {
                    return Ok(GenerateOutcome {
                        tokens: generated,
                        loop_detection: Some(d),
                        hit_eos: false,
                    });
                }
            }
            if cache.current_len() + 1 > cache.max_seq_len() {
                break;
            }
            let x = self.embed_tokens(&[next])?;
            hidden = self.forward_embeds_hidden(&x, &mut cache)?;
        }
        let loop_detection = detect_loop(&generated);
        Ok(GenerateOutcome {
            tokens: generated,
            loop_detection,
            hit_eos,
        })
    }
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if matches!(q.device(), Device::Cuda(_)) && q.dtype() == DType::BF16 {
        if q.dims()[1] == 1 && super::decoder_graph::kernel_decode_enabled() {
            return super::decoder_graph::decode_attention_eager(q, k, v, cfg);
        }
        return flash_attn(q, k, v, cfg);
    }
    sdpa(q, k, v, cfg)
}

pub fn banned_tokens_no_repeat_ngram(seq: &[u32], n: usize) -> Vec<u32> {
    banned_tokens_windowed_ngram(seq, n, None, &[])
}

pub fn banned_tokens_windowed_ngram(
    seq: &[u32],
    n: usize,
    window: Option<usize>,
    whitelist: &[u32],
) -> Vec<u32> {
    if n == 0 || seq.len() + 1 < n {
        return Vec::new();
    }
    let prefix_len = n - 1;
    let prefix = &seq[seq.len() - prefix_len..];
    let start = match window {
        Some(w) => seq.len().saturating_sub(w),
        None => 0,
    };
    let mut banned = Vec::new();
    for i in start..seq.len().saturating_sub(prefix_len) {
        if &seq[i..i + prefix_len] == prefix {
            let t = seq[i + prefix_len];
            if !whitelist.contains(&t) {
                banned.push(t);
            }
        }
    }
    banned
}

pub(crate) fn select_next_token(
    logits: &mut [f32],
    all_tokens: &[u32],
    opts: &GenerateOptions,
    rng: &mut SplitMix64,
) -> Result<u32> {
    if let Some(n) = opts.no_repeat_ngram_size {
        for b in
            banned_tokens_windowed_ngram(all_tokens, n, opts.ngram_window, &opts.ngram_whitelist)
        {
            if (b as usize) < logits.len() {
                logits[b as usize] = f32::NEG_INFINITY;
            }
        }
    }
    sample_token(logits, opts.temperature, rng)
}

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub(crate) fn sample_token(logits: &[f32], temperature: f32, rng: &mut SplitMix64) -> Result<u32> {
    if logits.is_empty() {
        anyhow::bail!("sample_token: empty logits");
    }
    if temperature <= 0.0 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        return Ok(best as u32);
    }
    let inv_t = 1.0 / temperature;
    let max = logits
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        anyhow::bail!("sample_token: all logits are -inf");
    }
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - max) * inv_t).exp()
            } else {
                0.0
            }
        })
        .collect();
    let z: f32 = probs.iter().sum();
    if z <= 0.0 {
        anyhow::bail!("sample_token: zero probability mass");
    }
    for p in probs.iter_mut() {
        *p /= z;
    }
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

pub fn build_prompt_tokens<F>(encode: F, prompt: &str, n_image_tokens: usize) -> Result<Vec<u32>>
where
    F: Fn(&str) -> Result<Vec<u32>>,
{
    let mut out = vec![BOS_TOKEN_ID];
    let mut first = true;
    for part in prompt.trim().split("<image>") {
        if !first {
            out.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, n_image_tokens));
        }
        if !part.is_empty() {
            out.extend(encode(part)?);
        }
        first = false;
    }
    Ok(out)
}

pub fn vision_token_count(n_tiles: usize) -> usize {
    n_tiles * 144 + 256 + 1
}

fn load_rmsnorm(
    weights: &WeightLoader,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("rmsnorm {name}: expected [{}], got {:?}", dim, d);
    }
    Ok(RmsNorm::new(w, eps))
}

fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!(
            "linear {name}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Linear::new(w, None)
}

fn load_mlp(
    factory: &dyn LinearFactory,
    weights: &WeightLoader,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<Mlp> {
    let gate = factory.load(
        weights,
        &format!("{prefix}.gate_proj.weight"),
        intermediate,
        hidden,
    )?;
    let up = factory.load(
        weights,
        &format!("{prefix}.up_proj.weight"),
        intermediate,
        hidden,
    )?;
    let down = factory.load(
        weights,
        &format!("{prefix}.down_proj.weight"),
        hidden,
        intermediate,
    )?;
    Mlp::new(gate, up, down)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn decoder_dtype_defaults_to_bf16() {
        if std::env::var("NV_DSOCR_DEC_DTYPE").is_ok() {
            return;
        }
        assert_eq!(env_decoder_dtype(), DType::BF16);
    }

    const CONFIG_JSON: &str = r#"{
        "global_view_pos": "head",
        "language_config": {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "first_k_dense_replace": 1,
            "hidden_size": 1280,
            "intermediate_size": 6848,
            "kv_lora_rank": null,
            "max_position_embeddings": 8192,
            "moe_intermediate_size": 896,
            "n_group": 1,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_attention_heads": 10,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 12,
            "num_key_value_heads": 10,
            "q_lora_rank": null,
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "topk_group": 1,
            "topk_method": "greedy",
            "torch_dtype": "bfloat16",
            "use_mla": false,
            "v_head_dim": 0,
            "vocab_size": 129280
        },
        "model_type": "deepseek_vl_v2",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "first_k_dense_replace": 1,
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "kv_lora_rank": null,
        "max_position_embeddings": 8192,
        "moe_intermediate_size": 896,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_attention_heads": 10,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 12,
        "num_key_value_heads": 10,
        "topk_group": 1,
        "topk_method": "greedy",
        "use_mla": false,
        "vocab_size": 129280
    }"#;

    #[test]
    fn config_parses_checkpoint_shape() {
        let c = DeepseekOcrDecoderConfig::from_hf_json_str(CONFIG_JSON).unwrap();
        assert_eq!(c.hidden_size, 1280);
        assert_eq!(c.num_hidden_layers, 12);
        assert_eq!(c.num_attention_heads, 10);
        assert_eq!(c.num_key_value_heads, 10);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.intermediate_size, 6848);
        assert_eq!(c.moe_intermediate_size, 896);
        assert_eq!(c.n_routed_experts, 64);
        assert_eq!(c.n_shared_experts, 2);
        assert_eq!(c.shared_expert_intermediate_size(), 1792);
        assert_eq!(c.num_experts_per_tok, 6);
        assert_eq!(c.first_k_dense_replace, 1);
        assert_eq!(c.vocab_size, 129280);
        assert_eq!(c.max_position_embeddings, 8192);
        assert_eq!(c.rope_theta, 10_000.0);
        assert_eq!(c.rms_norm_eps, 1e-6);
        assert!(!c.norm_topk_prob);
        assert_eq!(c.routed_scaling_factor, 1.0);
        assert_eq!(c.bos_token_id, 0);
        assert_eq!(c.eos_token_id, 1);
        assert!(!c.is_moe_layer(0));
        for i in 1..12 {
            assert!(c.is_moe_layer(i));
        }
    }

    #[test]
    fn config_rejects_mla() {
        let raw = CONFIG_JSON.replace("\"use_mla\": false", "\"use_mla\": true");
        assert!(DeepseekOcrDecoderConfig::from_hf_json_str(&raw).is_err());
    }

    fn det(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.7311).sin() * 0.2)
            .collect()
    }

    fn tiny_config() -> DeepseekOcrDecoderConfig {
        DeepseekOcrDecoderConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 40,
            moe_intermediate_size: 24,
            n_routed_experts: 4,
            n_shared_experts: 2,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
            vocab_size: 96,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
        }
    }

    fn tiny_weight_map(c: &DeepseekOcrDecoderConfig) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let h = c.hidden_size;
        let mut m = HashMap::new();
        let t =
            |vals: Vec<f32>, shape: (usize, usize)| Tensor::from_vec(vals, shape, &dev).unwrap();
        let t1 = |vals: Vec<f32>, n: usize| Tensor::from_vec(vals, n, &dev).unwrap();
        m.insert(
            "model.embed_tokens.weight".to_string(),
            t(det(c.vocab_size * h, 1.0), (c.vocab_size, h)),
        );
        m.insert(
            "lm_head.weight".to_string(),
            t(det(c.vocab_size * h, 2.0), (c.vocab_size, h)),
        );
        m.insert(
            "model.norm.weight".to_string(),
            t1(det(h, 3.0).iter().map(|v| 1.0 + v).collect(), h),
        );
        for l in 0..c.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let s = l as f32 * 100.0;
            m.insert(
                format!("{p}.input_layernorm.weight"),
                t1(det(h, 4.0 + s).iter().map(|v| 1.0 + v).collect(), h),
            );
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                t1(det(h, 5.0 + s).iter().map(|v| 1.0 + v).collect(), h),
            );
            for (name, seed) in [
                ("q_proj", 6.0),
                ("k_proj", 7.0),
                ("v_proj", 8.0),
                ("o_proj", 9.0),
            ] {
                m.insert(
                    format!("{p}.self_attn.{name}.weight"),
                    t(det(h * h, seed + s), (h, h)),
                );
            }
            if c.is_moe_layer(l) {
                m.insert(
                    format!("{p}.mlp.gate.weight"),
                    t(
                        det(c.n_routed_experts * h, 10.0 + s),
                        (c.n_routed_experts, h),
                    ),
                );
                for e in 0..c.n_routed_experts {
                    let es = 20.0 + s + e as f32 * 7.0;
                    let inter = c.moe_intermediate_size;
                    m.insert(
                        format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                        t(det(inter * h, es), (inter, h)),
                    );
                    m.insert(
                        format!("{p}.mlp.experts.{e}.up_proj.weight"),
                        t(det(inter * h, es + 1.0), (inter, h)),
                    );
                    m.insert(
                        format!("{p}.mlp.experts.{e}.down_proj.weight"),
                        t(det(h * inter, es + 2.0), (h, inter)),
                    );
                }
                let si = c.shared_expert_intermediate_size();
                m.insert(
                    format!("{p}.mlp.shared_experts.gate_proj.weight"),
                    t(det(si * h, 60.0 + s), (si, h)),
                );
                m.insert(
                    format!("{p}.mlp.shared_experts.up_proj.weight"),
                    t(det(si * h, 61.0 + s), (si, h)),
                );
                m.insert(
                    format!("{p}.mlp.shared_experts.down_proj.weight"),
                    t(det(h * si, 62.0 + s), (h, si)),
                );
            } else {
                let inter = c.intermediate_size;
                m.insert(
                    format!("{p}.mlp.gate_proj.weight"),
                    t(det(inter * h, 70.0 + s), (inter, h)),
                );
                m.insert(
                    format!("{p}.mlp.up_proj.weight"),
                    t(det(inter * h, 71.0 + s), (inter, h)),
                );
                m.insert(
                    format!("{p}.mlp.down_proj.weight"),
                    t(det(h * inter, 72.0 + s), (h, inter)),
                );
            }
        }
        m
    }

    fn tiny_model() -> DeepseekOcrDecoder {
        let c = tiny_config();
        let map = tiny_weight_map(&c);
        let dir = std::env::temp_dir().join(format!(
            "dsocr-decoder-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        candle_core::safetensors::save(&map, &path).unwrap();
        let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
        let model =
            DeepseekOcrDecoder::from_loader_with_dtype(c, &loader, &Device::Cpu, DType::F32)
                .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        model
    }

    #[test]
    fn tiny_full_vs_incremental_forward_matches() {
        let model = tiny_model();
        let tokens: Vec<u32> = vec![0, 5, 17, 42, 9, 88];

        let mut cache_full = model.new_kv_cache(16).unwrap();
        let logits_full = model
            .forward_tokens(&tokens, None, &mut cache_full)
            .unwrap();
        let last_full: Vec<f32> = logits_full
            .narrow(1, tokens.len() - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut cache_inc = model.new_kv_cache(16).unwrap();
        let mut last_inc = Vec::new();
        for &t in &tokens {
            let logits = model.forward_tokens(&[t], None, &mut cache_inc).unwrap();
            last_inc = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        }
        assert_eq!(cache_inc.current_len(), tokens.len());
        assert_eq!(last_full.len(), model.vocab_size());
        let mut max_abs = 0f32;
        for (a, b) in last_full.iter().zip(last_inc.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 2e-4,
            "incremental decode diverges from full forward: max abs diff {max_abs}"
        );
    }

    #[test]
    fn moe_matches_reference_without_topk_renormalization() {
        let dev = Device::Cpu;
        let h = 8usize;
        let inter = 6usize;
        let shared_inter = 10usize;
        let n_exp = 4usize;
        let k = 2usize;
        let n_tok = 5usize;

        let gate_w = det(n_exp * h, 1.5);
        let mut eg = Vec::new();
        let mut eu = Vec::new();
        let mut ed = Vec::new();
        let mut experts = Vec::new();
        let lin = |vals: &[f32], o: usize, i: usize| {
            Linear::new(Tensor::from_vec(vals.to_vec(), (o, i), &dev).unwrap(), None).unwrap()
        };
        for e in 0..n_exp {
            let g = det(inter * h, 10.0 + e as f32);
            let u = det(inter * h, 20.0 + e as f32);
            let d = det(h * inter, 30.0 + e as f32);
            experts
                .push(Mlp::new(lin(&g, inter, h), lin(&u, inter, h), lin(&d, h, inter)).unwrap());
            eg.push(g);
            eu.push(u);
            ed.push(d);
        }
        let sg = det(shared_inter * h, 40.0);
        let su = det(shared_inter * h, 41.0);
        let sd = det(h * shared_inter, 42.0);
        let shared = Mlp::new(
            lin(&sg, shared_inter, h),
            lin(&su, shared_inter, h),
            lin(&sd, h, shared_inter),
        )
        .unwrap();
        let moe = DeepseekMoe::new(
            Tensor::from_vec(gate_w.clone(), (n_exp, h), &dev).unwrap(),
            experts,
            shared,
            k,
            false,
            1.0,
        )
        .unwrap();

        let x_host = det(n_tok * h, 99.0);
        let x = Tensor::from_vec(x_host.clone(), (n_tok, h), &dev).unwrap();
        let y = moe.forward(&x).unwrap();
        let y_host: Vec<f32> = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let matvec = |w: &[f32], x: &[f32], o: usize, i: usize| -> Vec<f32> {
            (0..o)
                .map(|r| (0..i).map(|c| w[r * i + c] * x[c]).sum())
                .collect()
        };
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let mlp_ref = |g: &[f32], u: &[f32], d: &[f32], x: &[f32], inter: usize| -> Vec<f32> {
            let gv = matvec(g, x, inter, h);
            let uv = matvec(u, x, inter, h);
            let act: Vec<f32> = gv
                .iter()
                .zip(uv.iter())
                .map(|(a, b)| silu(*a) * b)
                .collect();
            matvec(d, &act, h, inter)
        };

        for n in 0..n_tok {
            let xr = &x_host[n * h..(n + 1) * h];
            let logits = matvec(&gate_w, xr, n_exp, h);
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / z).collect();
            let mut idx: Vec<usize> = (0..n_exp).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let sel = &idx[..k];
            let sel_sum: f32 = sel.iter().map(|&e| probs[e]).sum();
            assert!(sel_sum < 0.9999, "test degenerate: top-k already sums to 1");

            let mut want = mlp_ref(&sg, &su, &sd, xr, shared_inter);
            for &e in sel {
                let ye = mlp_ref(&eg[e], &eu[e], &ed[e], xr, inter);
                for i in 0..h {
                    want[i] += probs[e] * ye[i];
                }
            }
            for i in 0..h {
                let got = y_host[n * h + i];
                assert!(
                    (got - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                    "token {n} dim {i}: got {got}, want {} (raw-prob mixture + unconditional shared)",
                    want[i]
                );
            }
        }
    }

    #[test]
    fn vision_injection_replaces_placeholder_rows() {
        let model = tiny_model();
        let h = model.config().hidden_size;
        let tokens: Vec<u32> = vec![0, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 7, IMAGE_TOKEN_ID, 9];
        let feats_host: Vec<f32> = (0..3 * h).map(|i| 1000.0 + i as f32).collect();
        let feats = Tensor::from_vec(feats_host.clone(), (3, h), &Device::Cpu).unwrap();
        let x = model
            .embed_tokens_with_vision(&tokens, Some(&feats))
            .unwrap();
        let x_host: Vec<f32> = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (fi, ti) in [(0usize, 1usize), (1, 2), (2, 4)] {
            for d in 0..h {
                assert_eq!(x_host[ti * h + d], feats_host[fi * h + d]);
            }
        }
        let embed_row_7: Vec<f32> = model
            .embed_weight
            .narrow(0, 7, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for d in 0..h {
            assert_eq!(x_host[3 * h + d], embed_row_7[d]);
        }

        assert!(model.embed_tokens_with_vision(&tokens, None).is_err());
        let wrong = Tensor::zeros((2, h), DType::F32, &Device::Cpu).unwrap();
        assert!(model
            .embed_tokens_with_vision(&tokens, Some(&wrong))
            .is_err());
    }

    #[test]
    fn tiny_generate_greedy_is_deterministic_and_respects_cache() {
        let model = tiny_model();
        let opts = GenerateOptions {
            max_new_tokens: 8,
            temperature: 0.0,
            no_repeat_ngram_size: Some(20),
            ngram_window: None,
            ngram_whitelist: vec![],
            seed: 0,
        };
        let out1 = model.generate(&[0, 5, 17], None, &opts).unwrap();
        let out2 = model.generate(&[0, 5, 17], None, &opts).unwrap();
        assert_eq!(out1, out2);
        assert!(!out1.is_empty() && out1.len() <= 8);
        for &t in &out1 {
            assert!((t as usize) < model.vocab_size());
        }
    }

    #[test]
    fn prompt_tokens_layout_matches_reference() {
        let encode =
            |s: &str| -> Result<Vec<u32>> { Ok(s.bytes().map(|b| 200 + b as u32).collect()) };
        let out = build_prompt_tokens(encode, "<image>\nFree OCR. ", 5).unwrap();
        assert_eq!(out[0], BOS_TOKEN_ID);
        assert_eq!(&out[1..6], &[IMAGE_TOKEN_ID; 5]);
        let tail: Vec<u32> = "\nFree OCR.".bytes().map(|b| 200 + b as u32).collect();
        assert_eq!(&out[6..], &tail[..]);

        let canonical = build_prompt_tokens(encode, PROMPT_FREE_OCR, 5).unwrap();
        assert_eq!(out, canonical);

        let no_img = build_prompt_tokens(encode, "hello", 5).unwrap();
        assert_eq!(no_img[0], BOS_TOKEN_ID);
        assert!(!no_img.contains(&IMAGE_TOKEN_ID));

        assert_eq!(vision_token_count(0), 257);
        assert_eq!(vision_token_count(6), 6 * 144 + 257);
    }

    #[test]
    fn no_repeat_ngram_bans_completion_tokens() {
        assert!(banned_tokens_no_repeat_ngram(&[1, 2, 3], 4).is_empty());
        let seq = [7, 1, 2, 9, 5, 1, 2];
        assert_eq!(banned_tokens_no_repeat_ngram(&seq, 3), vec![9]);
        let seq2 = [1, 2, 3, 1, 2, 4, 1, 2];
        assert_eq!(banned_tokens_no_repeat_ngram(&seq2, 3), vec![3, 4]);
        assert_eq!(banned_tokens_no_repeat_ngram(&[5, 6, 5], 1), vec![5, 6, 5]);
    }

    #[test]
    fn windowed_ngram_matches_unwindowed_when_window_covers_sequence() {
        let seq = [1u32, 2, 3, 1, 2, 4, 1, 2];
        assert_eq!(
            banned_tokens_windowed_ngram(&seq, 3, None, &[]),
            banned_tokens_no_repeat_ngram(&seq, 3)
        );
        assert_eq!(
            banned_tokens_windowed_ngram(&seq, 3, Some(seq.len()), &[]),
            banned_tokens_no_repeat_ngram(&seq, 3)
        );
        assert_eq!(
            banned_tokens_windowed_ngram(&seq, 3, Some(1000), &[]),
            banned_tokens_no_repeat_ngram(&seq, 3)
        );
    }

    #[test]
    fn windowed_ngram_ignores_matches_outside_window() {
        let seq = [1u32, 2, 3, 9, 9, 9, 9, 1, 2, 4, 1, 2];
        assert_eq!(banned_tokens_windowed_ngram(&seq, 3, None, &[]), vec![3, 4]);
        assert_eq!(banned_tokens_windowed_ngram(&seq, 3, Some(5), &[]), vec![4]);
        assert!(banned_tokens_windowed_ngram(&seq, 3, Some(2), &[]).is_empty());
    }

    #[test]
    fn windowed_ngram_whitelist_exempts_tokens() {
        let seq = [
            1u32,
            2,
            TD_OPEN_TOKEN_ID,
            5,
            1,
            2,
            TD_CLOSE_TOKEN_ID,
            7,
            1,
            2,
        ];
        assert_eq!(
            banned_tokens_windowed_ngram(&seq, 3, None, &[]),
            vec![TD_OPEN_TOKEN_ID, TD_CLOSE_TOKEN_ID]
        );
        assert!(banned_tokens_windowed_ngram(
            &seq,
            3,
            None,
            &[TD_OPEN_TOKEN_ID, TD_CLOSE_TOKEN_ID]
        )
        .is_empty());
        let seq2 = [1u32, 2, TD_OPEN_TOKEN_ID, 5, 1, 2, 8, 7, 1, 2];
        assert_eq!(
            banned_tokens_windowed_ngram(&seq2, 3, None, &[TD_OPEN_TOKEN_ID, TD_CLOSE_TOKEN_ID]),
            vec![8]
        );
    }

    #[test]
    fn windowed_ngram_window_shorter_than_ngram_bans_nothing() {
        let seq = [1u32, 2, 3, 1, 2, 3, 1, 2];
        assert!(banned_tokens_windowed_ngram(&seq, 3, Some(3), &[]).is_empty());
        assert_eq!(banned_tokens_windowed_ngram(&seq, 3, Some(6), &[]), vec![3]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn graph_supported_rejects_cpu_model() {
        let model = tiny_model();
        let err = crate::deepseek_ocr::decoder_graph::graph_supported(&model).unwrap_err();
        assert!(err.contains("CUDA"), "unexpected reason: {err}");
    }

    #[test]
    fn generate_options_presets_match_recipe_and_hf() {
        let r = GenerateOptions::recipe();
        assert_eq!(r.no_repeat_ngram_size, Some(30));
        assert_eq!(r.ngram_window, Some(90));
        assert_eq!(r.ngram_whitelist, vec![TD_OPEN_TOKEN_ID, TD_CLOSE_TOKEN_ID]);
        assert_eq!(r.temperature, 0.0);
        let h = GenerateOptions::hf20();
        assert_eq!(h.no_repeat_ngram_size, Some(20));
        assert_eq!(h.ngram_window, None);
        assert!(h.ngram_whitelist.is_empty());
    }

    #[test]
    fn select_next_token_applies_windowed_ban() {
        let mut rng = SplitMix64::new(0);
        let all = [9u32, 1, 2, 9, 1];
        let base = GenerateOptions {
            max_new_tokens: 4,
            temperature: 0.0,
            no_repeat_ngram_size: Some(2),
            ngram_window: Some(4),
            ngram_whitelist: vec![],
            seed: 0,
        };
        let mut logits = vec![0.0f32, 0.0, 5.0, 4.0, 3.0];
        let next = select_next_token(&mut logits, &all, &base, &mut rng).unwrap();
        assert_eq!(next, 3);
        let mut logits2 = vec![0.0f32, 0.0, 5.0, 4.0, 3.0];
        let whitelisted = GenerateOptions {
            ngram_window: None,
            ngram_whitelist: vec![2],
            ..base.clone()
        };
        let next2 = select_next_token(&mut logits2, &all, &whitelisted, &mut rng).unwrap();
        assert_eq!(next2, 2);
        let mut logits3 = vec![0.0f32, 0.0, 5.0, 4.0, 3.0];
        let tiny_window = GenerateOptions {
            ngram_window: Some(3),
            ..base.clone()
        };
        let next3 = select_next_token(&mut logits3, &all, &tiny_window, &mut rng).unwrap();
        assert_eq!(next3, 2);
    }

    fn pseudo_random_seq(n: usize, vocab: u32, seed: u64) -> Vec<u32> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| (rng.next_u64() % vocab as u64) as u32)
            .collect()
    }

    #[test]
    fn detect_loop_flags_pure_repeat_and_backtracks_onset() {
        let prefix: Vec<u32> = (1000..1100).collect();
        let unit: Vec<u32> = vec![7, 88, 3, 501, 42, 9, 260];
        let mut seq = prefix.clone();
        for _ in 0..60 {
            seq.extend_from_slice(&unit);
        }
        let d = detect_loop(&seq).expect("pure repeat must be detected");
        assert_eq!(d.period % unit.len(), 0);
        assert!(
            d.onset >= prefix.len() && d.onset <= prefix.len() + 2 * unit.len(),
            "onset {} not at loop start (prefix {}, unit {})",
            d.onset,
            prefix.len(),
            unit.len()
        );
        let mut truncated = seq.clone();
        truncated.truncate(d.onset);
        assert_eq!(&truncated[..prefix.len()], &prefix[..]);
    }

    #[test]
    fn detect_loop_flags_mutated_repeat() {
        let prefix: Vec<u32> = (2000..2120).collect();
        let unit: Vec<u32> = vec![11, 12, 13, 14, 15, 16, 17, 18, 19];
        let mut seq = prefix.clone();
        for _ in 0..80 {
            seq.extend_from_slice(&unit);
        }
        let mut i = prefix.len() + 25;
        while i < seq.len() {
            seq[i] = 900 + (i as u32 % 7);
            i += 40;
        }
        let d = detect_loop(&seq).expect("mutated repeat must be detected");
        assert_eq!(d.period % unit.len(), 0);
        assert!(
            d.onset <= prefix.len() + 3 * unit.len(),
            "onset {} too far past loop start {}",
            d.onset,
            prefix.len()
        );
        assert!(
            d.onset >= prefix.len(),
            "onset {} backtracked into clean prefix",
            d.onset
        );
    }

    #[test]
    fn detect_loop_ignores_clean_sequences() {
        assert!(detect_loop(&pseudo_random_seq(3000, 8000, 1)).is_none());
        assert!(detect_loop(&pseudo_random_seq(3000, 60, 2)).is_none());
        assert!(detect_loop(&pseudo_random_seq(100, 8000, 3)).is_none());
    }

    #[test]
    fn detect_loop_truncates_at_final_loop_run_not_first() {
        let unit = [21u32, 22, 23, 24, 25, 26, 27];
        let mut seq = pseudo_random_seq(300, 8000, 6);
        for _ in 0..57 {
            seq.extend_from_slice(&unit);
        }
        seq.extend(pseudo_random_seq(400, 8000, 7));
        let second_loop_start = seq.len();
        for _ in 0..57 {
            seq.extend_from_slice(&unit);
        }
        let d = detect_loop(&seq).expect("interleaved loop must be detected");
        assert!(
            d.onset >= second_loop_start,
            "onset {} cut real content before final loop run at {}",
            d.onset,
            second_loop_start
        );
        assert!(
            d.onset <= second_loop_start + 3 * unit.len(),
            "onset {} too far into final loop run at {}",
            d.onset,
            second_loop_start
        );
    }

    #[test]
    fn detect_loop_ignores_brief_repetition_burst() {
        let mut seq = pseudo_random_seq(1000, 8000, 4);
        let unit = [5u32, 6, 7, 8, 9];
        for _ in 0..30 {
            seq.extend_from_slice(&unit);
        }
        seq.extend(pseudo_random_seq(1000, 8000, 5));
        assert!(detect_loop(&seq).is_none());
    }

    #[test]
    fn strip_grounding_tokens_removes_ref_and_det_spans() {
        let seq = vec![
            GROUNDING_TOKEN_ID,
            REF_OPEN_TOKEN_ID,
            500,
            501,
            REF_CLOSE_TOKEN_ID,
            DET_OPEN_TOKEN_ID,
            600,
            601,
            602,
            DET_CLOSE_TOKEN_ID,
            42,
            43,
            REF_OPEN_TOKEN_ID,
            502,
            REF_CLOSE_TOKEN_ID,
            DET_OPEN_TOKEN_ID,
            603,
            DET_CLOSE_TOKEN_ID,
            44,
        ];
        assert_eq!(strip_grounding_tokens(&seq), vec![42, 43, 44]);
        assert_eq!(strip_grounding_tokens(&[1, 2, 3]), vec![1, 2, 3]);
        let unclosed = vec![10, DET_OPEN_TOKEN_ID, 600, 601];
        assert_eq!(strip_grounding_tokens(&unclosed), vec![10]);
    }

    #[test]
    #[ignore]
    fn loopscan_saved_outputs() {
        let tok_path = std::env::var("NV_LOOPSCAN_TOKENIZER").expect("NV_LOOPSCAN_TOKENIZER");
        let dir = std::env::var("NV_LOOPSCAN_DIR").expect("NV_LOOPSCAN_DIR");
        let tokenizer = nv_tokenizer::load_tokenizer(Path::new(&tok_path)).unwrap();
        let mut files: Vec<std::path::PathBuf> = walk_txt(Path::new(&dir));
        files.sort();
        for f in files {
            let text = std::fs::read_to_string(&f).unwrap();
            let ids = tokenizer.encode(text.as_str(), false).unwrap();
            let seq = ids.get_ids();
            let mut max_periodic = 0f32;
            let mut min_distinct = 1f32;
            let mut c = LOOP_MIN_EVIDENCE;
            while c <= seq.len() {
                let w = LOOP_WINDOW.min(c);
                if w >= LOOP_MIN_EVIDENCE {
                    let start = c - w;
                    for p in 1..=LOOP_MAX_PERIOD.min(w / 2) {
                        let m = ((start + p)..c).filter(|&i| seq[i] == seq[i - p]).count();
                        let r = m as f32 / (w - p) as f32;
                        if r > max_periodic {
                            max_periodic = r;
                        }
                    }
                    let mut set = std::collections::HashSet::new();
                    let n8 = w - 7;
                    for i in start..(c - 7) {
                        set.insert(&seq[i..i + 8]);
                    }
                    let dr = set.len() as f32 / n8 as f32;
                    if dr < min_distinct {
                        min_distinct = dr;
                    }
                }
                if c == seq.len() {
                    break;
                }
                c = (c + LOOP_CHECK_STRIDE).min(seq.len());
            }
            match detect_loop(seq) {
                Some(d) => {
                    if let Ok(outdir) = std::env::var("NV_LOOPSCAN_OUT") {
                        let rel = f.strip_prefix(Path::new(&dir)).unwrap();
                        let out = Path::new(&outdir).join(rel);
                        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
                        let trunc = tokenizer.decode(&seq[..d.onset], true).unwrap();
                        std::fs::write(out, trunc).unwrap();
                    }
                    println!(
                        "LOOP {} tokens={} onset={} period={} kept={:.2} maxper={:.3} mindis={:.3}",
                        f.display(),
                        seq.len(),
                        d.onset,
                        d.period,
                        d.onset as f64 / seq.len() as f64,
                        max_periodic,
                        min_distinct
                    )
                }
                None => println!(
                    "CLEAN {} tokens={} maxper={:.3} mindis={:.3}",
                    f.display(),
                    seq.len(),
                    max_periodic,
                    min_distinct
                ),
            }
        }
    }

    fn walk_txt(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(dir) else {
            return out;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk_txt(&p));
            } else if p.extension().map(|x| x == "txt").unwrap_or(false) {
                out.push(p);
            }
        }
        out
    }

    #[test]
    fn sampling_greedy_and_temperature() {
        let mut rng = SplitMix64::new(42);
        let logits = vec![0.1f32, 3.0, -1.0, 2.9];
        assert_eq!(sample_token(&logits, 0.0, &mut rng).unwrap(), 1);
        let mut counts = [0usize; 4];
        for _ in 0..2000 {
            let t = sample_token(&logits, 1.0, &mut rng).unwrap();
            counts[t as usize] += 1;
        }
        assert!(counts[1] > counts[0]);
        assert!(counts[1] > counts[2]);
        assert!(counts[3] > counts[2]);
        let banned = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        for _ in 0..50 {
            assert_eq!(sample_token(&banned, 0.7, &mut rng).unwrap(), 2);
        }
    }

    fn route_sort_ref(
        probs: &[f32],
        n_tokens: usize,
        e_count: usize,
        k: usize,
        norm: bool,
        scale: f32,
    ) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); e_count];
        let mut ws: Vec<Vec<f32>> = vec![Vec::new(); e_count];
        for n in 0..n_tokens {
            let row = &probs[n * e_count..(n + 1) * e_count];
            let mut order: Vec<usize> = (0..e_count).collect();
            order.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sel = &order[..k];
            let denom = if norm {
                sel.iter().map(|&e| row[e]).sum::<f32>().max(1e-20)
            } else {
                1.0
            };
            for &e in sel {
                rows[e].push(n as u32);
                ws[e].push(row[e] / denom * scale);
            }
        }
        (rows, ws)
    }

    #[test]
    fn fast_topk_routing_matches_stable_sort_bitwise() {
        let e_count = 64usize;
        let k = 6usize;
        let n_tokens = 1200usize;
        let mut rng = SplitMix64::new(0xD50C4);
        for (norm, scale, quant) in [
            (false, 1.0f32, 0u32),
            (true, 2.5f32, 0),
            (false, 1.0, 8),
            (true, 1.0, 3),
        ] {
            let mut probs = vec![0f32; n_tokens * e_count];
            for v in probs.iter_mut() {
                let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
                *v = if quant == 0 {
                    u
                } else {
                    (u * quant as f32).floor() / quant as f32
                };
            }
            let (ra, wa) = route_sort_ref(&probs, n_tokens, e_count, k, norm, scale);
            let (rb, wb) = route_topk(&probs, n_tokens, e_count, k, norm, scale);
            assert_eq!(
                ra, rb,
                "expert row lists differ (norm={norm} quant={quant})"
            );
            for e in 0..e_count {
                for (x, y) in wa[e].iter().zip(&wb[e]) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "weight bits differ for expert {e} (norm={norm} quant={quant})"
                    );
                }
            }
        }
    }
}
