#![allow(dead_code)]
#![allow(unused_imports)]

#[path = "../hub_snapshot/mod.rs"]
mod hub_snapshot;

#[cfg(feature = "wgpu")]
pub mod flash_gqa_fold;

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3MoeConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "cuda")]
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, GenerateOptions, ResolutionMode, RgbImage,
    PROMPT_FREE_OCR,
};
#[cfg(feature = "cuda")]
use nv_models::graph_engine::GraphedQwen3Moe;
#[cfg(feature = "cuda")]
use nv_quant::nvfp4::{swizzle_scales, Nvfp4GemmRunner, Nvfp4Tensor, BLOCK_SIZE};
#[cfg(feature = "cuda")]
use nv_weights::{QuantizationConfig, WeightLoader};
#[cfg(feature = "wgpu")]
use nv_kernels::wgpu_backend::device::WgpuContext;
#[cfg(feature = "wgpu")]
use nv_models::gpt_oss_wgpu as gow;
#[cfg(feature = "wgpu")]
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};
#[cfg(feature = "wgpu")]
use nv_models::qwen3_5_dense_wgpu as q3d;
#[cfg(feature = "wgpu")]
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
#[cfg(feature = "wgpu")]
use nv_models::qwen3_5_moe_wgpu as q3w;
#[cfg(feature = "wgpu")]
use nv_models::qwen3_5_moe_wgpu::{quantize_nvfp4_host, HostBf16Lin};
#[cfg(feature = "wgpu")]
use nv_quant::mxfp4::Mxfp4Tensor;
use nv_models::gemma4_moe::Gemma4MoeConfig;

pub fn argmax_partial_cmp(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[cfg(feature = "cuda")]
pub fn argmax(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &x) in row.iter().enumerate() {
        if x > bestv {
            bestv = x;
            best = i as u32;
        }
    }
    best
}

#[cfg(feature = "cuda")]
pub struct Arm {
    pub label: &'static str,
    pub grouped: bool,
    pub routing: bool,
}

#[cfg(feature = "wgpu")]
pub fn bf16(x: f64) -> f64 {
    half::bf16::from_f32(x as f32).to_f32() as f64
}

pub fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

#[cfg(feature = "wgpu")]
pub fn bf16_bits_from_f64(x: f64) -> u16 {
    half::bf16::from_f32(x as f32).to_bits()
}

#[cfg(feature = "wgpu")]
pub fn bf16_lin_gow_bias(r: &mut LcgSplitMix64TwoSided, n: usize, k: usize, scale: f32, bias: bool) -> gow::HostBf16Lin {
    gow::HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        bias: if bias {
            r.bf16_vec(n, scale)
        } else {
            Vec::new()
        },
        n,
        k,
    }
}

#[cfg(feature = "wgpu")]
pub fn bf16_lin(r: &mut LcgOddSeedShift33SignedUnit, n: usize, k: usize, scale: f32) -> HostBf16Lin {
    HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        n,
        k,
    }
}

#[cfg(feature = "wgpu")]
pub fn bf16_lin_q3w(r: &mut LcgOddSeedShift33SignedUnit, n: usize, k: usize, scale: f32) -> q3w::HostBf16Lin {
    q3w::HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        n,
        k,
    }
}

pub fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(feature = "wgpu")]
pub fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[cfg(feature = "cuda")]
pub fn build(dir: &PathBuf, device: &Device, arm: &Arm) -> GraphedQwen3Moe {
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, device).expect("model");
    let mut eng = GraphedQwen3Moe::new(model, device, 1024).expect("engine");
    if arm.grouped {
        eng.install_grouped_moe().expect("grouped");
    }
    eng.set_device_routing(arm.routing);
    eng.reset().expect("reset");
    eng
}

#[cfg(feature = "cuda")]
pub fn cer(got: &str, want: &str) -> f64 {
    let a: Vec<char> = norm(got).chars().collect();
    let b: Vec<char> = norm(want).chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as f64 / a.len().max(b.len()).max(1) as f64
}

#[cfg(feature = "laguna-wip")]
#[cfg(feature = "wgpu")]
pub const CFG: &str = r#"{
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "vocab_size": 96,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "norm_topk_prob": true,
    "mlp_only_layers": [],
    "decoder_sparse_step": 1,
    "tie_word_embeddings": false,
    "gating": "per-head",
    "sliding_window": 4,
    "moe_routed_scaling_factor": 2.5,
    "moe_router_logit_softcapping": 5.0,
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 64,
            "beta_slow": 1.0,
            "beta_fast": 64.0,
            "attention_factor": 1.3465735902799727,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    },
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "dense"],
    "num_attention_heads_per_layer": [4, 8, 8, 4]
}"#;

#[cfg(feature = "wgpu")]
pub const CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE:
    usize = 256;

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CkParams {
    pub m_live: u32,
    pub base: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[cfg(feature = "wgpu")]
pub fn config_json_wrapped_text_config(layers: usize, hidden: usize, inter: usize, vocab: usize, window: usize) -> String {
    let mut types = Vec::with_capacity(layers);
    for i in 0..layers {
        types.push(if (i + 1) % 3 == 0 {
            "\"full_attention\""
        } else {
            "\"sliding_attention\""
        });
    }
    format!(
        r#"{{
  "text_config": {{
    "hidden_size": {hidden},
    "intermediate_size": {inter},
    "num_hidden_layers": {layers},
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 128,
    "global_head_dim": 256,
    "vocab_size": {vocab},
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "sliding_window": {window},
    "final_logit_softcapping": 0.0,
    "layer_types": [{}],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }}
  }},
  "tie_word_embeddings": true
}}"#,
        types.join(", ")
    )
}

#[cfg(feature = "cuda")]
pub fn config_json() -> String {
    format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "hidden_size": {HIDDEN_128},
  "intermediate_size": {INTER_256},
  "num_hidden_layers": {N_LAYERS_2},
  "num_attention_heads": {N_Q_2},
  "num_key_value_heads": {N_KV_1},
  "num_global_key_value_heads": {N_KV_1},
  "head_dim": {HEAD_DIM_128},
  "global_head_dim": {HEAD_DIM_128},
  "vocab_size": {VOCAB_512},
  "max_position_embeddings": 256,
  "rms_norm_eps": 1e-6,
  "sliding_window": 32,
  "layer_types": ["full_attention", "sliding_attention"],
  "attention_k_eq_v": false,
  "tie_word_embeddings": false,
  "hidden_activation": "gelu_pytorch_tanh",
  "rope_parameters": {{
    "full_attention": {{"rope_theta": 10000.0, "partial_rotary_factor": 1.0}},
    "sliding_attention": {{"rope_theta": 10000.0}}
  }}
}}"#
    )
}

pub const CTX_PANICS_NEVER_SKIPS_BECAUSE_A_RETURN_PRINTS_1_PASSED_IN_0S_FROM_SUITES_THAT_EXIST_TO_PRODUCE_NUMBERS:
    () = ();

#[cfg(feature = "wgpu")]
pub fn ctx() -> &'static WgpuContext {
    let c = WgpuContext::shared().expect("wgpu adapter required for --features wgpu");
    assert!(
        c.qualify().qualified,
        "adapter not qualified: {:?}",
        c.qualify().reason
    );
    c
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip() -> Option<&'static nv_kernels::wgpu_backend::WgpuContext> {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("adapter: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            if require() {
                panic!("NV_KERNELS_WGPU_REQUIRE=1 but no adapter: {e}");
            }
            eprintln!("skipping: no wgpu adapter ({e})");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip_no_require() -> Option<&'static nv_kernels::wgpu_backend::WgpuContext> {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("adapter: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            eprintln!("skipping: no wgpu adapter ({e})");
            None
        }
    }
}

#[cfg(feature = "cuda")]
pub fn ctx_tokens_from_env_default_256_8k_168k() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => vec![256, 8 * 1024, 168 * 1024],
    }
}

#[cfg(feature = "cuda")]
pub fn ctx_tokens_from_env_default_256_8k_196k() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => vec![256, 8 * 1024, 196 * 1024],
    }
}

pub fn distinct(bits: &[u32]) -> usize {
    bits.iter().collect::<std::collections::HashSet<_>>().len()
}

#[cfg(feature = "wgpu")]
pub fn env_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub fn env_usize(var: &str, dflt: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(dflt)
}

pub fn envn(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(d)
}

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FdP {
    pub n_heads: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub total: u32,
    pub start: u32,
    pub splits: u32,
    pub ring: u32,
    pub out_bf16: u32,
    pub scaling: f32,
    pub pad0: u32,
    pub fused: u32,
    pub pad2: u32,
    pub m_rows: u32,
    pub window: u32,
    pub pad3: u32,
    pub pad4: u32,
}

#[cfg(feature = "cuda")]
pub fn fixture_expected_text(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../conformance/fixtures")
        .join(name)
        .join("fixture.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse fixture.json");
    v["expected_text"]
        .as_str()
        .expect("expected_text string")
        .to_string()
}

#[cfg(feature = "cuda")]
pub fn fixture_rgb(name: &str) -> RgbImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../conformance/fixtures")
        .join(name)
        .join("input.png");
    let bytes = std::fs::read(&path).expect("read fixture png");
    RgbImage::decode(&bytes).expect("decode fixture")
}

#[cfg(feature = "cuda")]
pub const FIXTURES: &[&str] = &[
    "070-ocr-clean-line",
    "070-ocr-lowcontrast-gradient",
    "070-ocr-multiword-boxes",
    "070-ocr-noise-gauss",
    "070-ocr-paragraph",
    "070-ocr-photo-noise-surround",
    "070-ocr-realscan-1892",
    "070-ocr-skew-12deg",
    "070-ocr-skew-2deg",
    "070-ocr-small-font",
    "070-ocr-sparse-illustration-1892",
    "071-ocr-layout-invoice",
    "071-ocr-layout-labnotes",
    "071-ocr-layout-letter",
    "071-ocr-layout-newspaper",
    "071-ocr-layout-report",
];

#[cfg(feature = "wgpu")]
pub const GLOBAL_HEAD_DIM: usize = 32;

#[cfg(feature = "wgpu")]
pub fn have_gpu() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[wgpu] adapter: {}", ctx.info.name);
            true
        }
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            false
        }
    }
}

#[cfg(feature = "wgpu")]
pub const HEAD_DIM_16: usize = 16;

#[cfg(feature = "cuda")]
pub const HEAD_DIM_128: usize = 128;

#[cfg(feature = "wgpu")]
pub const HIDDEN_64: usize = 64;

#[cfg(feature = "cuda")]
pub const HIDDEN_128: usize = 128;

#[cfg(feature = "cuda")]
#[allow(deprecated)]
pub fn htod_f32(stream: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<f32> {
    stream.memcpy_stod(v).expect("htod f32")
}

#[cfg(feature = "cuda")]
#[allow(deprecated)]
pub fn htod_u8(stream: &Arc<CudaStream>, v: &[u8]) -> CudaSlice<u8> {
    stream.memcpy_stod(v).expect("htod u8")
}

#[cfg(feature = "cuda")]
pub const INTER_256: usize = 256;

#[cfg(feature = "wgpu")]
pub const INTER_96: usize = 96;

#[cfg(feature = "cuda")]
pub struct LcgInc1HalfCentered(pub u64);

#[cfg(feature = "cuda")]
impl LcgInc1HalfCentered {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / (1u64 << 31) as f32) - 0.5
    }
}

#[cfg(feature = "wgpu")]
pub struct LcgCentered0p1Shift32(pub u64);

#[cfg(feature = "wgpu")]
impl LcgCentered0p1Shift32 {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 32) as u32;
        (bits as f32 / u32::MAX as f32 - 0.5) * 0.2
    }
    pub fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
            .collect()
    }
    pub fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
            .collect()
    }
}

#[cfg(feature = "wgpu")]
pub struct LcgSplitMix64TwoSided(pub u64);

#[cfg(feature = "wgpu")]
impl LcgSplitMix64TwoSided {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let v = ((z >> 40) as u32) as f32 / (1u64 << 23) as f32;
        v - 1.0
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn f32_rows(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_f32() * scale).collect())
            .collect()
    }
}

#[cfg(feature = "wgpu")]
pub struct LcgCentered0p1Shift33(pub u64);

#[cfg(feature = "wgpu")]
impl LcgCentered0p1Shift33 {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32 - 0.5) * 0.2
    }
    pub fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
            .collect()
    }
    pub fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
            .collect()
    }
}

pub struct LcgTop24TwoSided(pub u64);

impl LcgTop24TwoSided {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn token(&mut self, vocab: usize) -> u32 {
        self.next_u32() % vocab as u32
    }
}

#[cfg(feature = "wgpu")]
pub struct LcgOddSeedShift33SignedUnit(pub u64);

#[cfg(feature = "wgpu")]
impl LcgOddSeedShift33SignedUnit {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32;
        v - 1.0
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
}

#[cfg(feature = "wgpu")]
pub fn med(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

pub fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(feature = "wgpu")]
pub const MOE_INTER: usize = 64;

#[cfg(feature = "wgpu")]
pub fn mx_stack(r: &mut LcgSplitMix64TwoSided, e: usize, n: usize, k: usize, scale: f32) -> gow::HostMxStack {
    let mats: Vec<Mxfp4Tensor> = (0..e)
        .map(|_| Mxfp4Tensor::quantize_rows(&r.f32_rows(n, k, scale)))
        .collect();
    let biases: Vec<Vec<u16>> = (0..e).map(|_| r.bf16_vec(n, scale)).collect();
    gow::stack_mx_host(&mats, &biases)
}

#[cfg(feature = "wgpu")]
pub const N_EXPERTS: usize = 8;

#[cfg(feature = "wgpu")]
pub const N_GLOBAL_KV: usize = 1;

#[cfg(feature = "cuda")]
pub const N_KV_1: usize = 1;

#[cfg(feature = "wgpu")]
pub const N_KV_2: usize = 2;

#[cfg(feature = "wgpu")]
pub const N_LAYERS_3: usize = 3;

#[cfg(feature = "cuda")]
pub const N_LAYERS_2: usize = 2;

#[cfg(feature = "cuda")]
pub const N_Q_2: usize = 2;

#[cfg(feature = "wgpu")]
pub const N_Q_4: usize = 4;

#[cfg(feature = "cuda")]
pub fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

#[cfg(feature = "wgpu")]
pub fn norm_tensor(rng: &mut LcgTop24TwoSided, dim: usize) -> Tensor {
    let data: Vec<f32> = (0..dim).map(|_| 1.0 + 0.25 * rng.next_f32()).collect();
    Tensor::from_vec(data, dim, &Device::Cpu).unwrap()
}

#[cfg(feature = "wgpu")]
pub fn norm_vec(r: &mut LcgOddSeedShift33SignedUnit, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn nozi_prof_dump() {
    use nv_kernels::wgpu_backend::dispatch::profile;
    if !profile::enabled() {
        return;
    }
    let mut total = 0.0;
    for (label, count, ns) in profile::report() {
        total += ns;
        eprintln!(
            "[prof] {label} n={count} total={:.3}ms avg={:.2}us",
            ns / 1.0e6,
            ns / 1.0e3 / count.max(1) as f64
        );
    }
    eprintln!("[prof] TOTAL {:.3}ms", total / 1.0e6);
}

#[cfg(feature = "wgpu")]
pub fn nvfp4(r: &mut LcgOddSeedShift33SignedUnit, n: usize, k: usize, scale: f32) -> q3w::HostNvfp4Lin {
    let w = r.bf16_vec(n * k, scale);
    q3w::quantize_nvfp4_host(&w, n, k)
}

#[cfg(feature = "wgpu")]
pub fn nvfp4_dense_lin(l: HostBf16Lin) -> q3d::HostDenseLin {
    q3d::HostDenseLin::Nvfp4(quantize_nvfp4_host(&l.w, l.n, l.k))
}

pub fn ones_tensor(dim: usize) -> Tensor {
    Tensor::ones(dim, DType::BF16, &Device::Cpu).unwrap()
}

#[cfg(feature = "wgpu")]
pub fn ord(bits: u16) -> i64 {
    let mag = (bits & 0x7fff) as i64;
    if bits & 0x8000 != 0 {
        -mag
    } else {
        mag
    }
}

#[cfg(feature = "wgpu")]
pub fn pack(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len() / 2];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

#[cfg(feature = "wgpu")]
pub fn pack_bf16_from_f64(vals: &[f64]) -> Vec<u32> {
    assert!(
        vals.len().is_multiple_of(2),
        "packed bf16 needs an even element count, got {}",
        vals.len()
    );
    vals.chunks(2)
        .map(|c| bf16_bits_from_f64(c[0]) as u32 | ((bf16_bits_from_f64(c[1]) as u32) << 16))
        .collect()
}

pub const PACK_BF16_IS_LOW_HALF_FIRST_THE_LAYOUT_U16_AT_READS: () = ();

#[cfg(feature = "wgpu")]
pub fn pack_bf16(vals: &[f32]) -> Vec<u32> {
    assert!(
        vals.len().is_multiple_of(2),
        "packed bf16 needs even length"
    );
    vals.chunks(2)
        .map(|c| {
            let lo = half::bf16::from_f32(c[0]).to_bits() as u32;
            let hi = half::bf16::from_f32(c[1]).to_bits() as u32;
            lo | (hi << 16)
        })
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn pack_u8(b: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; b.len().div_ceil(4)];
    for (i, x) in b.iter().enumerate() {
        out[i >> 2] |= (*x as u32) << (8 * (i & 3));
    }
    out
}

#[cfg(feature = "cuda")]
pub fn percentile_of_sorted(sorted_ms: &[f64], q: f64) -> f64 {
    let idx = ((sorted_ms.len() as f64 - 1.0) * q).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

pub fn prompt_for(j: usize, len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (((i + 1) * 7919 + j * 613 + 13) % vocab) as u32)
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn prompt_ids(cfg: &Qwen3MoeConfig, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (cfg.vocab_size as u32 - 1)) + 1)
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn rand_tensor_f32_shape(rng: &mut LcgTop24TwoSided, shape: &[usize], scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

#[cfg(feature = "cuda")]
pub fn rand_tensor(rng: &mut LcgTop24TwoSided, shape: (usize, usize), scale: f32) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

#[cfg(feature = "wgpu")]
pub fn real_snapshot() -> std::path::PathBuf {
    match std::env::var("NV_GEMMA4_MOE_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let root = std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
            std::fs::read_dir(&root)
                .unwrap_or_else(|e| panic!("no snapshot dir at {}: {e}", root.display()))
                .filter_map(|d| d.ok())
                .map(|d| d.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn rel_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut maxabs = 0f32;
    let mut scale = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        maxabs = maxabs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    (maxabs, maxabs / scale.max(1e-6))
}

#[cfg(feature = "wgpu")]
pub fn require() -> bool {
    std::env::var("NV_KERNELS_WGPU_REQUIRE").is_ok_and(|v| v != "0")
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[cfg(feature = "cuda")]
pub fn sf_swizzled(t: &Nvfp4Tensor) -> Vec<u8> {
    swizzle_scales(&t.scales, t.rows, t.cols / BLOCK_SIZE)
}

#[cfg(feature = "cuda")]
pub fn snapshot() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    std::fs::read_dir(base)
        .expect("qwen3.6 snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
        .expect("qwen3.6 snapshot with config.json")
}

#[cfg(feature = "wgpu")]
pub fn snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_E4B_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").expect("HOME");
            let base = std::path::PathBuf::from(home).join(
                ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots",
            );
            std::fs::read_dir(&base)
                .expect("hub snapshot dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

#[cfg(feature = "wgpu")]
pub struct Split {
    pub q_rows: usize,
    pub kv_rows: usize,
    pub v_off: usize,
}

#[cfg(feature = "wgpu")]
pub fn tiny_config_qwen35_dense() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 192,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

#[cfg(feature = "wgpu")]
pub fn tiny_config_gpt_oss() -> GptOssConfig {
    GptOssConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 32,
        num_local_experts: 4,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: 64,
        sliding_window: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        swiglu_limit: 7.0,
        layer_types: vec![GptOssLayerType::Sliding, GptOssLayerType::Full],
        yarn_factor: 4.0,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        yarn_original_max: 16,
        tie_word_embeddings: false,
    }
}

#[cfg(feature = "wgpu")]
pub const TINY_CONFIG: &str = r#"{
  "text_config": {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 64,
    "global_head_dim": 128,
    "vocab_size": 512,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"],
    "attention_k_eq_v": true,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;

#[cfg(feature = "wgpu")]
pub fn tiny_config_qwen36_moe() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 64,
        num_experts: 8,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.5,
        bos_token_id: 0,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

#[cfg(feature = "wgpu")]
pub fn tiny_config_json() -> String {
    format!(
        r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": {GLOBAL_HEAD_DIM},
    "head_dim": {HEAD_DIM_16},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN_64},
    "intermediate_size": {INTER_96},
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 64,
    "moe_intermediate_size": {MOE_INTER},
    "num_attention_heads": {N_Q_4},
    "num_experts": {N_EXPERTS},
    "num_global_key_value_heads": {N_GLOBAL_KV},
    "num_hidden_layers": {N_LAYERS_3},
    "num_key_value_heads": {N_KV_2},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }},
    "sliding_window": {WINDOW},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB_160}
  }}
}}"#
    )
}

#[cfg(feature = "wgpu")]
pub const TINY_E4B_CONFIG: &str = r#"{
  "text_config": {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "global_head_dim": 128,
    "vocab_size": 512,
    "vocab_size_per_layer_input": 512,
    "hidden_size_per_layer_input": 32,
    "num_kv_shared_layers": 2,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                    "full_attention", "sliding_attention", "full_attention"],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;

#[cfg(feature = "wgpu")]
pub fn tiny_weights(cfg: &Qwen3MoeConfig, seed: u64) -> q3w::HostWeights {
    let mut r = LcgOddSeedShift33SignedUnit::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
    let hd = cfg.head_dim;
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                in_proj_qkv: bf16_lin_q3w(&mut r, conv_dim, hidden, 0.12),
                in_proj_z: bf16_lin_q3w(&mut r, value_dim, hidden, 0.12),
                in_proj_ab: bf16_lin_q3w(&mut r, 2 * n_v, hidden, 0.12),
                conv1d: r.f32_vec(conv_dim * ks, 0.4),
                a_log: r.f32_vec(n_v, 0.5),
                dt_bias: r.f32_vec(n_v, 0.5),
                norm_w: norm_vec(&mut r, d_v),
                out_proj: bf16_lin_q3w(&mut r, hidden, value_dim, 0.12),
            })),
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                    q: nvfp4(&mut r, q_out, hidden, 0.12),
                    k: nvfp4(&mut r, kv_out, hidden, 0.12),
                    v: nvfp4(&mut r, kv_out, hidden, 0.12),
                    o: nvfp4(&mut r, hidden, cfg.num_attention_heads * hd, 0.12),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        let gates: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let ups: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let downs: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, hidden, inter, 0.15))
            .collect();
        layers.push(q3w::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            moe: q3w::HostMoe {
                router: bf16_lin_q3w(&mut r, cfg.num_experts, hidden, 0.3),
                experts_gate: q3w::stack_nvfp4_host(&gates),
                experts_up: q3w::stack_nvfp4_host(&ups),
                experts_down: q3w::stack_nvfp4_host(&downs),
                shared_gate: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_up: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_down: nvfp4(&mut r, hidden, sinter, 0.15),
                shared_expert_gate: bf16_lin_q3w(&mut r, 1, hidden, 0.3),
            },
        });
    }

    q3w::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[cfg(feature = "wgpu")]
pub const TOP_K: usize = 2;

#[cfg(feature = "wgpu")]
pub fn unpack(words: &[u32], n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| ((words[i / 2] >> (16 * (i % 2))) & 0xffff) as u16)
        .collect()
}

#[cfg(feature = "wgpu")]
pub const VOCAB_160: usize = 160;

#[cfg(feature = "cuda")]
pub const VOCAB_512: usize = 512;

#[cfg(feature = "cuda")]
pub fn wer(got: &str, want: &str) -> f64 {
    let na = norm(got);
    let nb = norm(want);
    let a: Vec<&str> = na.split(' ').filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = nb.split(' ').filter(|s| !s.is_empty()).collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, wa) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, wb) in b.iter().enumerate() {
            let cost = usize::from(wa != wb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as f64 / a.len().max(b.len()).max(1) as f64
}

#[cfg(feature = "wgpu")]
pub const WINDOW: usize = 8;

pub fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs() / scale))
}

#[cfg(feature = "cuda")]
pub fn write_tiny_model(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();

    let mut rng = LcgTop24TwoSided(0x5eed_cafe_f00d_0001);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB_512, HIDDEN_128), 1.0),
    );
    tensors.insert(
        "model.language_model.norm.weight".into(),
        ones_tensor(HIDDEN_128),
    );
    tensors.insert(
        "lm_head.weight".into(),
        rand_tensor(&mut rng, (VOCAB_512, HIDDEN_128), 1.0),
    );
    for i in 0..N_LAYERS_2 {
        let p = format!("model.language_model.layers.{i}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            tensors.insert(format!("{p}.{norm}.weight"), ones_tensor(HIDDEN_128));
        }
        tensors.insert(format!("{p}.layer_scalar"), ones_tensor(1));
        tensors.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(&mut rng, (N_Q_2 * HEAD_DIM_128, HIDDEN_128), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(&mut rng, (N_KV_1 * HEAD_DIM_128, HIDDEN_128), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand_tensor(&mut rng, (N_KV_1 * HEAD_DIM_128, HIDDEN_128), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN_128, N_Q_2 * HEAD_DIM_128), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.q_norm.weight"),
            ones_tensor(HEAD_DIM_128),
        );
        tensors.insert(
            format!("{p}.self_attn.k_norm.weight"),
            ones_tensor(HEAD_DIM_128),
        );
        tensors.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor(&mut rng, (INTER_256, HIDDEN_128), 0.3),
        );
        tensors.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor(&mut rng, (INTER_256, HIDDEN_128), 0.3),
        );
        tensors.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN_128, INTER_256), 0.3),
        );
    }
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
}

pub struct LcgOddSeedShift33SignedUnitRows(pub u64);

impl LcgOddSeedShift33SignedUnitRows {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32;
        v - 1.0
    }
    pub fn bf16_rounded_f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn f32_rows(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_f32() * scale).collect())
            .collect()
    }
    pub fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
    pub fn norm_effective_vec_near_one(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * self.next_f32()).to_f32())
            .collect()
    }
    pub fn norm_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * self.next_f32()).to_bits())
            .collect()
    }
}

pub struct LcgShift33Centered0p1(pub u64);

impl LcgShift33Centered0p1 {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.2
    }
    pub fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
            .collect()
    }
    pub fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
            .collect()
    }
    pub fn token(&mut self, vocab: usize) -> u32 {
        self.next_u32() % vocab as u32
    }
}

pub struct LcgShift32Centered0p1I8(pub u64);

impl LcgShift32Centered0p1I8 {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.2
    }
    pub fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
            .collect()
    }
    pub fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
            .collect()
    }
    pub fn i8_vec(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.next_u32() & 0xff) as u8 as i8).collect()
    }
    pub fn token(&mut self, vocab: usize) -> u32 {
        self.next_u32() % vocab as u32
    }
}

pub struct LcgOddSeedShift33SignedUnitPacks(pub u64);

impl LcgOddSeedShift33SignedUnitPacks {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / (1u64 << 31) as f32) - 1.0
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
}

pub struct LcgOddSeedShift33SignedUnitG4w(pub u64);

impl LcgOddSeedShift33SignedUnitG4w {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn norm_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * self.next_f32()).to_bits())
            .collect()
    }
}

pub struct LcgOddSeedShift32GaussUnit(pub u64);

impl LcgOddSeedShift32GaussUnit {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn gauss(&mut self) -> f32 {
        let s: f32 = (0..12).map(|_| self.unit()).sum();
        s - 6.0
    }
    pub fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 4294967296.0
    }
}

#[cfg(feature = "wgpu")]
pub struct LcgOddSeedNextF64SignedUnit(pub u64);

#[cfg(feature = "wgpu")]
impl LcgOddSeedNextF64SignedUnit {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 32) as u32) as f64 / (1u64 << 31) as f64) - 1.0
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| bf16(self.next() * scale)).collect()
    }
    pub fn f32_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|_| (self.next() * scale) as f32 as f64)
            .collect()
    }
}

pub struct LcgAdd1Shift33SignedUnit(pub u64);

impl LcgAdd1Shift33SignedUnit {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / 2147483648.0) - 1.0
    }
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
    pub fn norm_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * self.next_f32()).to_bits())
            .collect()
    }
}

pub struct LcgCentered0p1Shift32F64(pub u64);

impl LcgCentered0p1Shift32F64 {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 32) as u32;
        ((bits as f64 / 4294967296.0) as f32 - 0.5) * 0.2
    }
    pub fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
            .collect()
    }
    pub fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
            .collect()
    }
}

pub struct LcgOddSeedShift33SignedUnitVec(pub u64);

impl LcgOddSeedShift33SignedUnitVec {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

pub struct LcgShift33TwoSided(pub u64);

impl LcgShift33TwoSided {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

pub fn qwen38_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    hub_snapshot::snapshot_of(
        "unsloth/Qwen3.8-27B-NVFP4",
        &["config.json", "*.safetensors"],
    )
    .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot under the HF hub roots; set NV_QWEN38_DIR")
}

pub fn qwen36_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("qwen3.6 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.join("config.json").is_file()
                && (p.join("model.safetensors").is_file()
                    || p.join("model.safetensors.index.json").is_file())
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no qwen3.6 NVFP4 snapshot with weights under HOME hub; set NV_QWEN36_DIR")
}

pub fn gemma4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("gemma4 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no gemma4 NVFP4 snapshot under HOME hub; set NV_G4_SNAPSHOT")
}

pub fn laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_LAGUNA_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("laguna snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete Laguna-XS-2.1 NVFP4 snapshot under HOME hub; set NV_LAGUNA_DIR")
}

pub fn ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN35_DENSE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--ig1--Qwen3.5-9B-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("ig1 qwen3.5-9b snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete ig1 Qwen3.5-9B NVFP4 snapshot under HOME hub; set NV_QWEN35_DENSE_DIR")
}

pub fn e4b_snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_E4B_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap();
            let base = std::path::PathBuf::from(home)
                .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
            std::fs::read_dir(&base)
                .expect("hub snapshot dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

pub fn got_ocr_snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_GOT_OCR_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stepfun-ai--GOT-OCR-2.0-hf/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

pub fn deepseek_ocr_snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_DSOCR_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

pub struct EnvPins(Vec<(&'static str, Option<String>)>);

impl EnvPins {
    pub fn pin(vars: &[(&'static str, Option<&str>)]) -> Self {
        let saved = vars
            .iter()
            .map(|(k, v)| {
                let prev = std::env::var(k).ok();
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
                (*k, prev)
            })
            .collect();
        EnvPins(saved)
    }
}

impl Drop for EnvPins {
    fn drop(&mut self) {
        for (k, prev) in self.0.drain(..) {
            match prev {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

pub struct TempDir(pub std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

pub fn rand_tensor_two_sided(rng: &mut LcgShift33TwoSided, shape: &[usize], scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

pub fn norm_tensor_two_sided(rng: &mut LcgShift33TwoSided, dim: usize) -> Tensor {
    let data: Vec<f32> = (0..dim).map(|_| 1.0 + 0.25 * rng.next_f32()).collect();
    Tensor::from_vec(data, dim, &Device::Cpu).unwrap()
}

pub fn tensors_for_cfg_two_sided(cfg: &Gemma4MoeConfig, seed: u64) -> HashMap<String, Tensor> {
    use nv_models::gemma4::LayerType;
    let base = &cfg.base;
    let (hidden, inter, vocab) = (base.hidden_size, base.intermediate_size, base.vocab_size);
    let (n_q, n_e, mi) = (
        base.num_attention_heads,
        cfg.num_experts,
        cfg.moe_intermediate_size,
    );
    let mut rng = LcgShift33TwoSided(seed);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor_two_sided(&mut rng, &[vocab, hidden], 1.0),
    );
    t.insert(
        "model.language_model.norm.weight".into(),
        norm_tensor_two_sided(&mut rng, hidden),
    );
    for i in 0..base.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        let kind = base.layer_kind(i);
        let full = kind == LayerType::FullAttention;
        let hd = base.head_dim_for(kind);
        let n_kv = base.num_kv_heads_for(kind);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ] {
            t.insert(format!("{p}.{norm}.weight"), norm_tensor_two_sided(&mut rng, hidden));
        }
        t.insert(
            format!("{p}.layer_scalar"),
            Tensor::from_vec(vec![0.9f32 + 0.1 * rng.next_f32()], 1, &Device::Cpu).unwrap(),
        );
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor_two_sided(&mut rng, &[n_q * hd, hidden], 0.3),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor_two_sided(&mut rng, &[n_kv * hd, hidden], 0.3),
        );
        if !(full && base.attention_k_eq_v) {
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                rand_tensor_two_sided(&mut rng, &[n_kv * hd, hidden], 0.3),
            );
        }
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor_two_sided(&mut rng, &[hidden, n_q * hd], 0.3),
        );
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            norm_tensor_two_sided(&mut rng, hd),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            norm_tensor_two_sided(&mut rng, hd),
        );
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor_two_sided(&mut rng, &[inter, hidden], 0.3),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor_two_sided(&mut rng, &[inter, hidden], 0.3),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor_two_sided(&mut rng, &[hidden, inter], 0.3),
        );
        t.insert(
            format!("{p}.router.proj.weight"),
            rand_tensor_two_sided(&mut rng, &[n_e, hidden], 0.3),
        );
        t.insert(format!("{p}.router.scale"), norm_tensor_two_sided(&mut rng, hidden));
        t.insert(
            format!("{p}.router.per_expert_scale"),
            norm_tensor_two_sided(&mut rng, n_e),
        );
        t.insert(
            format!("{p}.experts.gate_up_proj"),
            rand_tensor_two_sided(&mut rng, &[n_e, 2 * mi, hidden], 0.3),
        );
        t.insert(
            format!("{p}.experts.down_proj"),
            rand_tensor_two_sided(&mut rng, &[n_e, hidden, mi], 0.3),
        );
    }
    t
}

pub fn config_json_gemma4_layers(layers: usize, hidden: usize, inter: usize, vocab: usize) -> String {
    let mut types = Vec::with_capacity(layers);
    for i in 0..layers {
        types.push(if (i + 1) % 3 == 0 {
            "\"full_attention\""
        } else {
            "\"sliding_attention\""
        });
    }
    format!(
        r#"{{
  "text_config": {{
    "hidden_size": {hidden},
    "intermediate_size": {inter},
    "num_hidden_layers": {layers},
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 128,
    "global_head_dim": 256,
    "vocab_size": {vocab},
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4096,
    "final_logit_softcapping": 0.0,
    "layer_types": [{}],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }}
  }},
  "tie_word_embeddings": true
}}"#,
        types.join(", "),
    )
}

pub fn config_json_gemma4_hd64(hidden: usize, inter: usize, vocab: usize) -> String {
    format!(
        r#"{{
  "text_config": {{
    "hidden_size": {hidden},
    "intermediate_size": {inter},
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 64,
    "global_head_dim": 128,
    "vocab_size": {vocab},
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "attention_k_eq_v": true,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }}
  }},
  "tie_word_embeddings": true
}}"#
    )
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip_bool() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            let st = ctx.qualify();
            if !st.qualified {
                eprintln!("SKIP adapter not qualified: {:?}", st.reason);
                return false;
            }
            eprintln!("{}", ctx.summary());
            true
        }
        Err(e) => {
            eprintln!("SKIP no wgpu adapter: {e}");
            false
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip_quiet(what: &str) -> Option<&'static nv_kernels::wgpu_backend::WgpuContext> {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{what}: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            eprintln!("skipping {what}: no wgpu adapter ({e})");
            None
        }
    }
}

pub fn decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state() -> &'static str {
    #[cfg(feature = "cuda")]
    if nv_models::qwen3_5_moe::nv_q36_graphed_decode_fix_env_routes_decode_to_the_gemma4_proven_splitk_flash() {
        return "splitk_flash";
    }
    "serial_smem"
}

pub fn unpack_bf16_bits(words: &[u32], n: usize) -> Vec<u16> {
    (0..n)
        .map(|r| {
            let w = words[r / 2];
            if r.is_multiple_of(2) {
                (w & 0xffff) as u16
            } else {
                (w >> 16) as u16
            }
        })
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn tiny_weights_q3d(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = LcgOddSeedShift33SignedUnit::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(n_v, 0.5),
                    dt_bias: r.f32_vec(n_v, 0.5),
                    norm_w: norm_vec(&mut r, d_v),
                    out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
                }))
            }
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                    q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
                    k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    o: bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12).into(),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
            },
        });
    }

    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[cfg(feature = "wgpu")]
pub fn tiny_config_q3d_mixed_layers() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 96,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

#[cfg(feature = "wgpu")]
pub fn gemma4_host_weights_nvfp4_ffn(config: &nv_models::gemma4::Gemma4Config, seed: u64) -> nv_models::gemma4_wgpu::HostWeights {
    use nv_models::gemma4::LayerType;
    use nv_models::gemma4_wgpu::{quantize_nvfp4_host, HostBf16Lin, HostLayer, HostProj, HostWeights};
    let mut rng = LcgCentered0p1Shift32(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let mk_proj = |rng: &mut LcgCentered0p1Shift32, n: usize, k: usize, quant: bool| {
            let w = rng.bf16_vec(n * k);
            if quant {
                HostProj::Nvfp4(quantize_nvfp4_host(&w, n, k))
            } else {
                HostProj::Bf16(HostBf16Lin { w, n, k })
            }
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk_proj(&mut rng, qkv_rows, hidden, false),
            o: mk_proj(&mut rng, hidden, q_dim, false),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden, true),
            down: mk_proj(&mut rng, hidden, inter, true),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

#[cfg(feature = "wgpu")]
pub fn gemma4_host_weights_bf16_attn_nvfp4_ffn(config: &nv_models::gemma4::Gemma4Config, seed: u64) -> nv_models::gemma4_wgpu::HostWeights {
    use nv_models::gemma4::LayerType;
    use nv_models::gemma4_wgpu::{quantize_nvfp4_host, HostBf16Lin, HostLayer, HostProj, HostWeights};
    let mut rng = LcgCentered0p1Shift33(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let bf16 = |rng: &mut LcgCentered0p1Shift33, n: usize, k: usize| {
            HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(n * k),
                n,
                k,
            })
        };
        let quant = |rng: &mut LcgCentered0p1Shift33, n: usize, k: usize| {
            HostProj::Nvfp4(quantize_nvfp4_host(&rng.bf16_vec(n * k), n, k))
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: bf16(&mut rng, qkv_rows, hidden),
            o: bf16(&mut rng, hidden, q_dim),
            gate_up: quant(&mut rng, 2 * inter, hidden),
            down: quant(&mut rng, hidden, inter),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

#[cfg(feature = "wgpu")]
pub fn gemma4_host_weights_quant_ffn_opt(config: &nv_models::gemma4::Gemma4Config, seed: u64, quant_ffn: bool) -> nv_models::gemma4_wgpu::HostWeights {
    use nv_models::gemma4::LayerType;
    use nv_models::gemma4_wgpu::{quantize_nvfp4_host, HostBf16Lin, HostLayer, HostProj, HostWeights};
    let mut rng = LcgCentered0p1Shift32(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let mk_proj = |rng: &mut LcgCentered0p1Shift32, n: usize, k: usize, quant: bool| {
            let w = rng.bf16_vec(n * k);
            if quant {
                HostProj::Nvfp4(quantize_nvfp4_host(&w, n, k))
            } else {
                HostProj::Bf16(HostBf16Lin { w, n, k })
            }
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk_proj(&mut rng, qkv_rows, hidden, false),
            o: mk_proj(&mut rng, hidden, q_dim, false),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden, quant_ffn),
            down: mk_proj(&mut rng, hidden, inter, quant_ffn),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

#[cfg(feature = "wgpu")]
pub fn tensors_for_cfg(cfg: &Gemma4MoeConfig, seed: u64) -> HashMap<String, Tensor> {
    use nv_models::gemma4::LayerType;
    let base = &cfg.base;
    let (hidden, inter, vocab) = (base.hidden_size, base.intermediate_size, base.vocab_size);
    let (n_q, n_e, mi) = (
        base.num_attention_heads,
        cfg.num_experts,
        cfg.moe_intermediate_size,
    );
    let mut rng = LcgTop24TwoSided(seed);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor_f32_shape(&mut rng, &[vocab, hidden], 1.0),
    );
    t.insert(
        "model.language_model.norm.weight".into(),
        norm_tensor(&mut rng, hidden),
    );
    for i in 0..base.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        let kind = base.layer_kind(i);
        let full = kind == LayerType::FullAttention;
        let hd = base.head_dim_for(kind);
        let n_kv = base.num_kv_heads_for(kind);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ] {
            t.insert(format!("{p}.{norm}.weight"), norm_tensor(&mut rng, hidden));
        }
        t.insert(
            format!("{p}.layer_scalar"),
            Tensor::from_vec(vec![0.9f32 + 0.1 * rng.next_f32()], 1, &Device::Cpu).unwrap(),
        );
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[n_q * hd, hidden], 0.3),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[n_kv * hd, hidden], 0.3),
        );
        if !(full && base.attention_k_eq_v) {
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                rand_tensor_f32_shape(&mut rng, &[n_kv * hd, hidden], 0.3),
            );
        }
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[hidden, n_q * hd], 0.3),
        );
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            norm_tensor(&mut rng, hd),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            norm_tensor(&mut rng, hd),
        );
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[inter, hidden], 0.3),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[inter, hidden], 0.3),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[hidden, inter], 0.3),
        );
        t.insert(
            format!("{p}.router.proj.weight"),
            rand_tensor_f32_shape(&mut rng, &[n_e, hidden], 0.3),
        );
        t.insert(format!("{p}.router.scale"), norm_tensor(&mut rng, hidden));
        t.insert(
            format!("{p}.router.per_expert_scale"),
            norm_tensor(&mut rng, n_e),
        );
        t.insert(
            format!("{p}.experts.gate_up_proj"),
            rand_tensor_f32_shape(&mut rng, &[n_e, 2 * mi, hidden], 0.3),
        );
        t.insert(
            format!("{p}.experts.down_proj"),
            rand_tensor_f32_shape(&mut rng, &[n_e, hidden, mi], 0.3),
        );
    }
    t
}

#[cfg(feature = "wgpu")]
pub fn tiny_e4b_host_weights(config: &nv_models::gemma4::Gemma4Config, seed: u64) -> nv_models::gemma4_e4b_wgpu::E4bHostWeights {
    use nv_models::gemma4::LayerType;
    use nv_models::gemma4_e4b_wgpu::{E4bHostLayer, E4bHostWeights, HostLin};
    let mut rng = LcgShift33Centered0p1(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let hpl = config.hidden_size_per_layer_input;
    let n_q = config.num_attention_heads;
    let n_layers = config.num_hidden_layers;

    let mut layers = Vec::new();
    for i in 0..n_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let kv_source = config.kv_source_layer(i);
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = match kv_source {
            Some(_) => q_dim,
            None => q_dim + kv_dim * if has_v { 2 } else { 1 },
        };
        let k_norm = match kv_source {
            Some(_) => Vec::new(),
            None => rng.bf16_vec_around_one(hd),
        };
        layers.push(E4bHostLayer {
            kind,
            kv_source,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            post_per_layer_input_norm: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm,
            layer_scalar: 0.9,
            has_v,
            qkv: HostLin::new(rng.bf16_vec(qkv_rows * hidden), qkv_rows, hidden),
            o: HostLin::new(rng.bf16_vec(hidden * q_dim), hidden, q_dim),
            gate_up: HostLin::new(rng.bf16_vec(2 * inter * hidden), 2 * inter, hidden),
            down: HostLin::new(rng.bf16_vec(hidden * inter), hidden, inter),
            per_layer_input_gate: HostLin::new(rng.bf16_vec(hpl * hidden), hpl, hidden),
            per_layer_projection: HostLin::new(rng.bf16_vec(hidden * hpl), hidden, hpl),
        });
    }

    let ple_row = n_layers * hpl;
    E4bHostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        embed_per_layer: rng.bf16_vec(config.vocab_size_per_layer() * ple_row),
        per_layer_model_projection: HostLin::new(rng.bf16_vec(ple_row * hidden), ple_row, hidden),
        per_layer_projection_norm: rng.bf16_vec_around_one(hpl),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

pub fn gemma4_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let pinned = PathBuf::from(&home).join(
        ".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
    );
    if pinned.join("config.json").is_file() {
        return pinned;
    }
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("gemma4 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gemma4 NVFP4 snapshot under HOME hub; set NV_G4_SNAPSHOT")
}

#[cfg(feature = "wgpu")]
pub fn greedy_after_prefill(
    gpu: &mut q3d::Qwen3_5DenseWgpu,
    tokens: &[u32],
    chunked: bool,
    continuation: usize,
) -> (Vec<u32>, Vec<f32>) {
    gpu.reset().expect("reset");
    let (last, rest) = tokens.split_last().expect("prompt is non-empty");
    let mut consumed = 0usize;
    if chunked {
        consumed = gpu.prefill_tokens(rest).expect("prefill_tokens");
        assert!(
            consumed > 0 || rest.len() < gpu.prefill_chunk_len(),
            "chunked prefill consumed nothing on a {}-token prompt at m={}",
            rest.len(),
            gpu.prefill_chunk_len()
        );
    }
    for t in &rest[consumed..] {
        gpu.prefill_step(*t).expect("per-token prefill step");
    }
    let (mut next, logits) = gpu.decode_step_logits(*last).expect("last prompt token");
    let mut out = Vec::with_capacity(continuation);
    for _ in 0..continuation {
        out.push(next);
        next = gpu.decode_step(next).expect("decode step");
    }
    (out, logits)
}

#[cfg(feature = "wgpu")]
pub fn gemma4_wgpu_host_weights(
    config: &nv_models::gemma4::Gemma4Config,
    seed: u64,
) -> nv_models::gemma4_wgpu::HostWeights {
    use nv_models::gemma4::LayerType;
    use nv_models::gemma4_wgpu::{
        quantize_nvfp4_host, HostBf16Lin, HostLayer, HostProj, HostWeights,
    };

    let mut rng = LcgShift32Centered0p1I8(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let mk_proj = |rng: &mut LcgShift32Centered0p1I8, n: usize, k: usize, quant: bool| {
            let w = rng.bf16_vec(n * k);
            if quant {
                HostProj::Nvfp4(quantize_nvfp4_host(&w, n, k))
            } else {
                HostProj::Bf16(HostBf16Lin { w, n, k })
            }
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk_proj(&mut rng, qkv_rows, hidden, false),
            o: mk_proj(&mut rng, hidden, q_dim, false),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden, true),
            down: mk_proj(&mut rng, hidden, inter, true),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

pub fn snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
    env_var: &str,
    repos: &[&str],
) -> PathBuf {
    if let Ok(d) = std::env::var(env_var) {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    for repo in repos {
        let base = PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots");
        let Ok(rd) = std::fs::read_dir(&base) else {
            continue;
        };
        let mut candidates: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.join("config.json").is_file()
                    && (p.join("model.safetensors").is_file()
                        || p.join("model.safetensors.index.json").is_file())
            })
            .collect();
        candidates.sort();
        if let Some(dir) = candidates.into_iter().next() {
            return dir;
        }
    }
    panic!("no snapshot with config.json + weights under HOME hub (tried {repos:?}); set {env_var}")
}

#[cfg(feature = "wgpu")]
pub fn gow_norm_vec(r: &mut LcgSplitMix64TwoSided, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn gow_tiny_weights(cfg: &GptOssConfig, seed: u64) -> gow::HostWeights {
    let bf16_lin = bf16_lin_gow_bias;
    type Lcg = LcgSplitMix64TwoSided;
    let norm_vec = gow_norm_vec;

    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd;
    let kv_out = cfg.num_key_value_heads * hd;

    let mut layers = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        layers.push(gow::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            attn: gow::HostAttn {
                q: bf16_lin(&mut r, q_out, hidden, 0.12, true),
                k: bf16_lin(&mut r, kv_out, hidden, 0.12, true),
                v: bf16_lin(&mut r, kv_out, hidden, 0.12, true),
                o: bf16_lin(&mut r, hidden, q_out, 0.12, true),
                sinks: (0..cfg.num_attention_heads)
                    .map(|_| r.next_f32() * 0.5)
                    .collect(),
            },
            moe: gow::HostMoe {
                router: bf16_lin(&mut r, cfg.num_local_experts, hidden, 0.3, true),
                gate_up: mx_stack(&mut r, cfg.num_local_experts, 2 * inter, hidden, 0.15),
                down: mx_stack(&mut r, cfg.num_local_experts, hidden, inter, 0.15),
            },
        });
    }

    gow::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[cfg(feature = "cuda")]
pub fn laguna_chunked_prefill(
    model: &nv_models::laguna::Laguna,
    cache: &mut nv_models::laguna::LagunaKvCache,
    ids: &[u32],
    device: &Device,
) -> Vec<f32> {
    let mut last = Vec::new();
    let mut pos = 0usize;
    for chunk in ids.chunks(256) {
        let t = Tensor::from_vec(chunk.to_vec(), (1usize, chunk.len()), device).unwrap();
        let p = Tensor::from_vec(
            (pos as i32..(pos + chunk.len()) as i32).collect::<Vec<i32>>(),
            chunk.len(),
            device,
        )
        .unwrap();
        let logits = model
            .forward_with_cache(&t, &p, cache)
            .expect("prefill chunk");
        last = logits
            .narrow(1, chunk.len() - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        pos += chunk.len();
    }
    last
}

pub fn e4b_qat_w4a16_ct_snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_E4B_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap();
            let base = std::path::PathBuf::from(home).join(
                ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots",
            );
            std::fs::read_dir(&base)
                .expect("hub snapshot dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}
