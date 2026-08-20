use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::qwen3_5_dense_wgpu::{assert_w8_group_divides_k, rope_tables};
use crate::qwen3_5_moe::{LayerType, Qwen3MoeConfig};

pub use crate::nvfp4_host::NVFP4_BLOCK;
const MAX_HEAD_DIM: usize = 256;
const MAX_LIN_HEAD_DIM: usize = 128;
const MAX_TOPK: usize = 16;
const ARGMAX_GROUPS: usize = 256;
fn flash_splits() -> u32 {
    wk::flash_decode::splits_env() as u32
}

const STAGING_FLUSH_BYTES: u64 = 256 << 20;

const GEMV_BF16_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_gemv_bf16.wgsl");

const GEMV_NVFP4_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_gemv_nvfp4.wgsl");

const GEMV_NVFP4_V2_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_gemv_nvfp4_v2.wgsl");

const QUANT_ROWS_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_quant_rows.wgsl");

const DELTA_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_delta.wgsl");

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_attn.wgsl");

const MOE_WGSL: &str = concat!(
    include_str!("../../nv-kernels/wgsl/q3m_moe.wgsl"),
    include_str!("../../nv-kernels/wgsl/q3w_argmax.wgsl")
);

const PF_DELTA_CHUNK_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_delta.wgsl");

const PF_ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_attn.wgsl");

const PF_KVQ_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_kvq.wgsl");

const PF_MOE_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_moe.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvBf16Params {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemvNvfp4Params {
    pub alpha: f32,
    pub n_rows: u32,
    pub k_blocks: u32,
    pub k_tiles: u32,
    pub groups_x: u32,
    pub w_e_stride_vec2: u32,
    pub sf_e_stride_bytes: u32,
    pub x_slot_stride_vec2: u32,
    pub xsf_slot_stride_bytes: u32,
    pub y_slot_stride_words: u32,
    pub per_expert_alpha: u32,
    pub m_slots_sharing_expert_zero: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantRowsParams {
    pub k_blocks: u32,
    pub n_slots: u32,
    pub use_sel: u32,
    pub x_slot_stride_elems: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SiluPairParams {

    pub u_off_elems: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ResScaleParams {
    n: u32,
    n_words: u32,
    scale: f32,
    cap: f32,
    inv_cap: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    conv_dim: u32,
    kernel: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaQkvParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    ab_off: u32,
    pad1: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatingParams {
    n_v: u32,
    ab_off: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RecurrentParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaOutParams {
    n_v: u32,
    d_v: u32,
    z_off: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormRopeParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    src_off: u32,
    w_off: u32,
    n_q: u32,
    k_src_off: u32,
    k_src_stride: u32,
    k_w_off: u32,
    sin_off: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvFp8Params {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    ring: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    splits: u32,
    ring: u32,
    out_bf16: u32,
    scaling: f32,
    pad0: u32,
    fused: u32,
    pad2: u32,
    m_rows: u32,
    window: u32,
    pad3: u32,
    pad4: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnGateParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RouterParams {
    n_experts: u32,
    k: u32,
    shared_slot: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluMulParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride_words: u32,
    shared_off_words: u32,
    slogit_off: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Clone)]
pub struct HostBf16Lin {
    pub w: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone)]
pub struct HostDeltaNet {
    pub in_proj_qkv: HostBf16Lin,
    pub in_proj_z: HostBf16Lin,
    pub in_proj_ab: HostBf16Lin,
    pub conv1d: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_w: Vec<u16>,
    pub out_proj: HostBf16Lin,
}

#[derive(Clone)]
pub struct HostAttention {
    pub q: HostNvfp4Lin,
    pub k: HostNvfp4Lin,
    pub v: HostNvfp4Lin,
    pub o: HostNvfp4Lin,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
}

#[derive(Clone)]
pub enum HostMixer {
    Delta(Box<HostDeltaNet>),
    Attn(Box<HostAttention>),
}

#[derive(Clone)]
pub struct HostMoe {
    pub router: HostBf16Lin,
    pub experts_gate: HostExpertStack,
    pub experts_up: HostExpertStack,
    pub experts_down: HostExpertStack,
    pub shared_gate: HostNvfp4Lin,
    pub shared_up: HostNvfp4Lin,
    pub shared_down: HostNvfp4Lin,
    pub shared_expert_gate: HostBf16Lin,
}

#[derive(Clone)]
pub struct HostLayer {
    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub mixer: HostMixer,
    pub moe: HostMoe,
}

pub struct HostWeights {
    pub embed: Vec<u16>,
    pub final_norm: Vec<u16>,
    pub lm_head: Vec<u16>,
    pub layers: Vec<HostLayer>,
}

pub enum WeightSource<'a> {
    Host(&'a HostWeights),
    Loader(&'a nv_weights::WeightLoader),
}

fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

use crate::gemma4_wgpu_shared::pack_pairs;
pub use crate::wgpu_ledger::bytes_to_words;

pub use crate::nvfp4_host::{
    dequantize_nvfp4_host, expert_slice, HostNvfp4ExpertStack as HostExpertStack, HostNvfp4Lin,
};

pub use crate::nvfp4_host::{quantize_nvfp4_host, stack_nvfp4_host};

#[derive(Clone)]
struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    label: String,

    bound_bytes: u64,
    widest_bytes: u64,
}

pub const VERIFY_CHAIN_FORWARDS_THE_WHOLE_BAKED_M_ROW_CHUNK_AND_COMMITS_IN_PLACE_ONLY_AT_FULL_WIDTH: &str = "the M-row prefill list is baked for chunks of exactly m tokens -- every chunked delta kernel loops its whole token count from a build-time uniform -- so verify_chain forwards m rows whatever the chain length is, and reads argmaxes for the live prefix only. Rows past the live prefix are causally invisible to it: each M-row kernel is row-independent and the attention is causal, so a garbage row at a later position is never read by an earlier one. DeltaNet recurrent state and the causal short conv still advance across all m rows, which is why advance(n) may commit in place only at n == verify_max_rows(); every shorter accept restores the pre-verify snapshot and replays the accepted prefix through the M=1 stepping path.";

struct Verify {
    rows: usize,
    res2_rows: wgpu::Buffer,
    resadd: Pass,
    tok: wgpu::Buffer,
    row_logits: Option<wgpu::Buffer>,
    rollback: Vec<wgpu::Buffer>,
    validated: bool,
    pending: Option<Vec<u32>>,
}

pub use crate::wgpu_ledger::VramReport;

struct Builder {
    core: crate::wgpu_ledger::VramLedger,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    to_prefill: bool,
    pf_mrow_pass_mix_m_row_then_per_token_copies: (usize, usize),
    nvfp4_v2_routed: usize,
    nvfp4_projs: usize,
    router_par: (usize, usize),
    delta_u4: (usize, usize),
    dn_in_proj: (usize, usize),
    dn_gate: (usize, usize),
    at_qcast: (usize, usize),
    at_kv: (usize, usize),
    at_qknorm: (usize, usize),
    shared_fold: (usize, usize),
    gate_up: (usize, usize),
    gemv_unrolled: (usize, usize),
    quant_lane: (usize, usize),

    w8_proj: (usize, u64, u64),
}

impl std::ops::Deref for Builder {
    type Target = crate::wgpu_ledger::VramLedger;
    fn deref(&self) -> &crate::wgpu_ledger::VramLedger {
        &self.core
    }
}

impl std::ops::DerefMut for Builder {
    fn deref_mut(&mut self) -> &mut crate::wgpu_ledger::VramLedger {
        &mut self.core
    }
}

impl Builder {
    fn make(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<Pass> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        Ok(Pass {
            pipeline,
            bind,
            grid,
            label: format!("{label}:{entry}"),
            bound_bytes: binds.iter().map(|(_, b)| b.size()).sum(),
            widest_bytes: binds.iter().map(|(_, b)| b.size()).max().unwrap_or(0),
        })
    }

    fn push(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pass = self.make(label, source, entry, binds, grid)?;
        if self.to_prefill {
            self.pf_mrow_pass_mix_m_row_then_per_token_copies.0 += 1;
            self.pf_passes.push(pass);
        } else {
            self.passes.push(pass);
        }
        Ok(())
    }

    fn push_pf_off(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer, u64)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        self.push_pf_off_counted(label, source, entry, binds, grid, false)
    }

    fn push_pf_off_m_row(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer, u64)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        self.push_pf_off_counted(label, source, entry, binds, grid, true)
    }

    fn push_pf_off_counted(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer, u64)],
        grid: (u32, u32, u32),
        covers_many_tokens_so_counts_as_m_row: bool,
    ) -> Result<()> {
        for (slot, _, off) in binds {
            anyhow::ensure!(
                off.is_multiple_of(PF_MROW_BIND_OFFSET_ALIGN_BYTES),
                "{label}::{entry} binding {slot}: offset {off} misses the {PF_MROW_BIND_OFFSET_ALIGN_BYTES}B \
                 offset alignment WebGPU guarantees for storage and uniform bindings"
            );
        }
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group_offsets(self.ctx, &pipeline, binds);
        if covers_many_tokens_so_counts_as_m_row {
            self.pf_mrow_pass_mix_m_row_then_per_token_copies.0 += 1;
        } else {
            self.pf_mrow_pass_mix_m_row_then_per_token_copies.1 += 1;
        }
        self.pf_passes.push(Pass {
            pipeline,
            bind,
            grid,
            label: format!("{label}:{entry}"),
            bound_bytes: binds.iter().map(|(_, b, _)| b.size()).sum(),
            widest_bytes: binds.iter().map(|(_, b, _)| b.size()).max().unwrap_or(0),
        });
        Ok(())
    }

}

pub fn nvfp4_gemv_source() -> String {
    format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), GEMV_NVFP4_WGSL)
}

const Q8E_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_q8e.wgsl");

pub fn moe_source() -> String {
    let mut s = MOE_WGSL.to_string();
    for (name, w) in router_rank_entries() {
        s.push('\n');
        s.push_str(&router_rank_wgsl(name, w));
    }
    compose(&s)
}

pub fn router_rank_entries() -> [(&'static str, usize); 3] {
    [
        ("q3w_router_topk_r4", 4),
        ("q3w_router_topk_par", 8),
        ("q3w_router_topk_r16", 16),
    ]
}

fn router_rank_wgsl(name: &str, w: usize) -> String {
    let mut cmp = String::new();
    for i in 0..w {
        cmp.push_str(&format!(
            "            let a{i} = rtp_v[l + {i}u];\n\
             \x20           if (a{i} > vt || (a{i} == vt && l + {i}u < t)) {{ rank = rank + 1u; }}\n"
        ));
    }
    include_str!("../../nv-kernels/wgsl/q3m_router_rank_template.wgsl")
        .replace("RR_ENTRY_POINT", name)
        .replace("RR_UNROLL_WIDTH", &w.to_string())
        .replace("RR_UNROLLED_COMPARES", &cmp)
}

pub fn attn_source() -> String {
    compose(ATTN_WGSL)
}

pub fn delta_source() -> String {
    compose(DELTA_WGSL)
}

pub fn gemv_bf16_source() -> String {
    compose(GEMV_BF16_WGSL)
}

pub fn i8e_source() -> String {
    compose(Q8E_WGSL)
}

pub fn delta_recurrent_variants_source() -> &'static str {
    let u4 = DELTA_WGSL
        .find("fn q3w_delta_recurrent_u4(")
        .expect("q3m_delta.wgsl lost the u4 recurrent variant");
    let start = DELTA_WGSL[..u4]
        .rfind("@compute")
        .expect("u4 variant lost its @compute attribute");
    let end = DELTA_WGSL
        .find("struct Q3doParams")
        .expect("q3m_delta.wgsl lost the delta-out section that bounds the variants");
    &DELTA_WGSL[start..end]
}

pub const U8L32_DEFAULT_BEATS_U16_ON_THE_ISOLATED_DISPATCH_LADDER_AT_48_HEADS_DK128_DV128: &str =
    "on the isolated-dispatch ladder over 48 layer-state buffers, q3w_delta_recurrent_u8l32 \
     (wg32, lane split, grid (48,4,1)) outruns the old default q3w_delta_recurrent_u16 \
     (wg128, grid (48,1,1)); every arm is bit-identical on all five parity corpora; current \
     numbers: perf/runs.jsonl. NV_Q3_WGPU_DELTA_UNROLL=16 restores the old default.";

pub fn delta_recurrent_kernel() -> (&'static str, u32) {
    match std::env::var("NV_Q3_WGPU_DELTA_UNROLL").ok().as_deref() {
        Some("0") | Some("off") | Some("false") => ("q3w_delta_recurrent", 0),
        Some("l32") => ("q3w_delta_recurrent_l32", 32),
        Some("4") => ("q3w_delta_recurrent_u4", 0),
        Some("4l32") => ("q3w_delta_recurrent_u4l32", 32),
        Some("8") => ("q3w_delta_recurrent_u8", 128),
        Some("16") => ("q3w_delta_recurrent_u16", 128),
        Some("16l32") => ("q3w_delta_recurrent_u16l32", 32),
        Some("32") => ("q3w_delta_recurrent_u32", 128),
        Some("32l32") => ("q3w_delta_recurrent_u32l32", 32),
        _ => ("q3w_delta_recurrent_u8l32", 32),
    }
}

pub fn nvfp4_quant_source() -> String {
    let mut s = format!("{}\n{}", wk::gemv_nvfp4::quantize_source(), QUANT_ROWS_WGSL);
    for (name, r#ref, wg) in quant_lane_entries() {
        s.push('\n');
        s.push_str(&quant_lane_wgsl(name, wg, r#ref.contains("silu")));
    }
    s
}

pub fn quant_lane_entries() -> [(&'static str, &'static str, usize); 4] {
    [
        ("q3w_quant_rows_l32", "q3w_quant_rows", 32),
        ("q3w_quant_rows_l256", "q3w_quant_rows", 256),
        ("q3w_silu_mul_quant_l32", "q3w_silu_mul_quant", 32),
        ("q3w_silu_mul_quant_l256", "q3w_silu_mul_quant", 256),
    ]
}

fn quant_lane_wgsl(name: &str, wg: usize, silu: bool) -> String {
    assert!(
        wg.is_multiple_of(32),
        "lane-split quantize needs a whole number of 32-lane subgroups, got wg={wg}"
    );
    let wgb = wg / 8;
    let head = if silu {
        include_str!("../../nv-kernels/wgsl/q3m_quant_lane_head_silu.wgsl")
    } else {
        include_str!("../../nv-kernels/wgsl/q3m_quant_lane_head_plain.wgsl")
    };
    include_str!("../../nv-kernels/wgsl/q3m_quant_lane_template.wgsl")
        .replace("QL_ENTRY_POINT", name)
        .replace("QL_WORKGROUP_THREADS", &wg.to_string())
        .replace("QL_BLOCKS_PER_WORKGROUP", &wgb.to_string())
        .replace("QL_LOAD_HEAD", head)
}

struct Sources {
    gemv_bf16: String,
    i8e: String,
    gemv_nvfp4: String,
    quant: String,
    delta: String,
    attn: String,
    moe: String,
    rms: String,
    rmsres: String,
    resscale: String,
    kvq: String,
    flash: String,
    pf_glue: String,
    pf_attn: String,
    pf_kvq: String,
    pf_moe: String,
}

pub fn pf_attn_source() -> String {
    compose(&format!("{ATTN_WGSL}\n{PF_ATTN_WGSL}"))
}

pub fn pf_kvq_source() -> String {
    compose(&format!("{}\n{}", wk::kv_fp8::WGSL, PF_KVQ_WGSL))
}

pub fn pf_moe_batched_source() -> String {
    compose(&format!("{MOE_WGSL}\n{PF_MOE_WGSL}"))
}

const PF_FLASH_SG_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_flash.wgsl");

pub fn pf_flash_sg_source() -> String {
    compose(&format!("{}\n{}", wk::flash_decode::WGSL, PF_FLASH_SG_WGSL))
}

pub const FLASH_SD_DEFAULT_ON_SHIFT_DECODE_WINS_AT_DEPTH_WITH_QUALITY_UNCHANGED: &str =
    "the exact e4m3_decode spends ~12 ops per KV element (nan guard, subnormal select, sign \
     select) while e4m3_shift_decode_scale_must_carry_2pow120 spends 2 plus a bitcast and \
     folds its 2^120 carry into the per-position k/v scales the kernels already multiply; \
     at deep KV the shift twin wins decode ms/tok with quality unchanged (the \
     e4m3-subnormal flush is below teacher-forced ppl noise); current numbers: \
     perf/runs.jsonl. NV_Q3M_FLASH_SD=0 restores the exact-decode entries";

pub fn flash_sd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !off("NV_Q3M_FLASH_SD"))
}

const PF_SD_K_SCALE_ANCHOR: &str = "ks = fd_k_scales[sp * nkv + kvh];";
const PF_SD_V_SCALE_ANCHOR: &str = "vs = fd_v_scales[sp * nkv + kvh];";
const PF_SD_SCALE_FOLD_2POW120: &str = " * bitcast<f32>(0x7B800000u)";

fn pf_sd_twin(body: &str, entries: &[&str]) -> String {
    for (scales, anchor) in [
        ("fd_k_scales", PF_SD_K_SCALE_ANCHOR),
        ("fd_v_scales", PF_SD_V_SCALE_ANCHOR),
    ] {
        assert_eq!(
            body.matches(anchor).count(),
            1,
            "pf shift-decode twin: scale anchor `{anchor}` must appear exactly once; a missed \
             anchor silently applies exact-decode magnitudes against 2pow120-folded scales"
        );
        assert_eq!(
            body.matches(scales).count(),
            1,
            "pf shift-decode twin: a {scales} read outside the anchored line would escape the \
             2pow120 fold and scale shift-decoded values 2pow120 too small"
        );
    }
    let mut sd = body.to_string();
    for anchor in [PF_SD_K_SCALE_ANCHOR, PF_SD_V_SCALE_ANCHOR] {
        let folded = format!(
            "{}{PF_SD_SCALE_FOLD_2POW120};",
            anchor.strip_suffix(';').expect("scale anchors end the statement")
        );
        sd = sd.replacen(anchor, &folded, 1);
    }
    for e in entries {
        let decl = format!("fn {e}(");
        assert_eq!(
            sd.matches(decl.as_str()).count(),
            1,
            "pf shift-decode twin: entry `{e}` must appear exactly once so the _sd rename \
             cannot leave a stock-named entry reading 2pow120-folded scales"
        );
        sd = sd.replacen(&decl, &format!("fn {e}_sd("), 1);
    }
    assert_eq!(
        sd.matches("0x7B800000").count(),
        2,
        "pf shift-decode twin: exactly one k-scale fold and one v-scale fold must survive the \
         rewrite; any other count silently mismatches decode magnitudes and scales"
    );
    sd
}

pub fn pf_flash_sg_source_sd() -> String {
    assert!(
        PF_FLASH_SG_WGSL.contains("fd_k_fp8(") && PF_FLASH_SG_WGSL.contains("fd_v_fp8("),
        "q3m_pf_flash.wgsl no longer decodes fp8 KV through the exact fd helpers; an _sd twin \
         would change nothing while claiming the shift-decode speedup"
    );
    let sd = pf_sd_twin(PF_FLASH_SG_WGSL, &["q3w_pf_flash1_fp8kv_mk_sg"])
        .replace("fd_k_fp8(", "fd_k_fp8_sd(")
        .replace("fd_v_fp8(", "fd_v_fp8_sd(");
    compose(&format!("{}\n{}", wk::flash_decode::WGSL, sd))
}

const PF_FLASH_TILED_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_flash_tiled.wgsl");

const PF_FLASH_TILED_SG_SECTION_STARTS_AT: &str =
    "fn fdt_warp_sum_butterfly_same_tree_as_pfl_warp_sum";

fn pf_flash_tiled_body(subgroup: bool) -> &'static str {
    if subgroup {
        PF_FLASH_TILED_WGSL
    } else {
        let cut = PF_FLASH_TILED_WGSL
            .find(PF_FLASH_TILED_SG_SECTION_STARTS_AT)
            .expect("q3m_pf_flash_tiled.wgsl keeps the subgroup entry last so the portable arm ships no subgroup intrinsics");
        &PF_FLASH_TILED_WGSL[..cut]
    }
}

pub fn pf_flash_tiled_source(subgroup: bool) -> String {
    compose(&format!(
        "{}\n{}",
        wk::flash_decode::WGSL,
        pf_flash_tiled_body(subgroup)
    ))
}

const PF_TILED_SD_STAGE_DECODES_4_K_AND_4_V_BYTES_PER_LOOP: usize = 8;

pub fn pf_flash_tiled_source_sd(subgroup: bool) -> String {
    let body = pf_flash_tiled_body(subgroup);
    assert_eq!(
        body.matches("e4m3_decode(").count(),
        PF_TILED_SD_STAGE_DECODES_4_K_AND_4_V_BYTES_PER_LOOP,
        "q3m_pf_flash_tiled.wgsl stages KV as 4 k-bytes and 4 v-bytes decoded per loop \
         iteration; a drifted count means exact-decode call sites the _sd twin would miss"
    );
    let mut entries = vec![
        "q3w_pf_flash1_fp8kv_tiled_wg",
        "q3w_pf_flash1_fp8kv_tiled_slotml_wg",
    ];
    if subgroup {
        entries.extend([
            "q3w_pf_flash1_fp8kv_tiled_sg",
            "q3w_pf_flash1_fp8kv_tiled_slotml_sg",
        ]);
    }
    let sd = pf_sd_twin(body, &entries)
        .replace("e4m3_decode(", "e4m3_shift_decode_scale_must_carry_2pow120(");
    compose(&format!("{}\n{}", wk::flash_decode::WGSL, sd))
}

pub fn pf_flash_tiled_default_on_since_slotml_solo_nll_plus_p006_nats_under_p01_bound_and_557_vs_431_tok_s_user_signed_off(
) -> bool {
    !matches!(
        std::env::var("NV_Q3_WGPU_PF_FLASH_TILED").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn pf_flash_tiled_slotml_is_the_default_arm_env_value_1_selects_the_older_tiling_with_larger_nll_penalty(
) -> bool {
    !matches!(
        std::env::var("NV_Q3_WGPU_PF_FLASH_TILED").ok().as_deref(),
        Some("1")
    )
}

pub const PF_TILED_FLASH_ROWS_ONE_KV_STREAM_SERVES_32_ROWS_AT_4_PER_WARP: usize = 32;

const PF_FD_ONE_TRAILING_SLOT_VIEWS_THE_WHOLE_CHUNK_FOR_TILED_FLASH: usize = 1;

const PF_MOE_GROUPED_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_moe_grouped.wgsl");

pub fn pf_moe_grouped_source(cfg: wk::gemv_nvfp4_v2::V2Config) -> String {
    compose(&format!(
        "{}\n{}\n{}",
        wk::gemv_nvfp4_v2::helpers(cfg),
        GEMV_NVFP4_V2_WGSL,
        PF_MOE_GROUPED_WGSL
    ))
}

pub fn pf_moe_grouped_default_on_reuses_expert_weight_loads_across_sorted_slots() -> bool {
    !off("NV_Q3_WGPU_PF_MOE_GROUPED")
}

pub const PF_COOP_NOT_WIRED_HERE_MOE_EXPERTS_STAY_GROUPED_AND_DENSE_ARM_GEMMS_ARE_32PCT: &str =
    "the coop 16x16x16 GEMM wants one dense row-major B swept over all M rows, but the expert \
     stacks select B per sorted slot group, so they keep the grouped slot kernels; the M-row \
     prefill attribution at real qwen3.6-35B shapes (synthetic weights, 8-layer 6DN+2FA \
     replica, m=64, 2048 tokens) puts MoE at 43.2%, DeltaNet at 38.0% (of which the recurrent \
     chunk scan is 14.1%), attention at 17.7%, and every coop-eligible dense-arm GEMM \
     (dn in/out proj, attn q/o/vk proj, router) at 31.9% combined -- a 1.47x ceiling that does \
     not justify the f16-twin staging and per-proj epilogues the wiring costs while the \
     qwen3.8-style 82% GEMM split demonstrably does not transfer to this geometry";

#[doc(hidden)]
pub fn nozi_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("q3m:gemv_bf16", gemv_bf16_source()),
        ("q3m:gemv_nvfp4", nvfp4_gemv_source()),
        ("q3m:quant_rows", nvfp4_quant_source()),
        ("q3m:delta", delta_source()),
        ("q3m:attn", attn_source()),
        ("q3m:moe", moe_source()),
        ("q3m:i8e", i8e_source()),
        (
            "q3m:gemv_nvfp4_v2",
            compose(&format!(
                "{}\n{}",
                wk::gemv_nvfp4_v2::helpers(wk::gemv_nvfp4_v2::V2Config::default()),
                GEMV_NVFP4_V2_WGSL
            )),
        ),
    ]
}

impl Sources {
    fn new() -> Self {
        Self {
            gemv_bf16: compose(GEMV_BF16_WGSL),
            i8e: compose(Q8E_WGSL),
            gemv_nvfp4: format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), GEMV_NVFP4_WGSL),
            quant: nvfp4_quant_source(),
            delta: compose(DELTA_WGSL),
            attn: compose(ATTN_WGSL),
            moe: moe_source(),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            kvq: compose(wk::kv_fp8::WGSL),
            flash: compose(wk::flash_decode::WGSL),
            pf_glue: compose(PF_GLUE_WGSL),
            pf_attn: pf_attn_source(),
            pf_kvq: pf_kvq_source(),
            pf_moe: pf_moe_batched_source(),
        }
    }
}

struct Bf16Gpu {
    w: wgpu::Buffer,
    n: usize,
    k: usize,
    entry: &'static str,
}

use crate::nvfp4_host::{upload_nvfp4, Nvfp4Gpu};

struct ExpertGpu {
    w: wgpu::Buffer,
    scales: wgpu::Buffer,
    alphas: wgpu::Buffer,
    globals: wgpu::Buffer,
    n: usize,
    k: usize,
}

pub struct Qwen3MoeWgpu {
    ctx: &'static WgpuContext,
    config: Qwen3MoeConfig,
    max_seq: usize,
    pos: usize,
    validated: bool,
    prefix_validated: bool,
    passes: Vec<Pass>,
    head_start: usize,
    final_start: usize,
    verify: Option<Verify>,
    _buffers: Vec<wgpu::Buffer>,
    tok_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    fd_buf: wgpu::Buffer,
    fd_base: FdParams,
    res2: wgpu::Buffer,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    recurrent_states: Vec<(wgpu::Buffer, u64)>,
    vocab: usize,
    vram: VramReport,
    nvfp4_v2: (usize, usize),
    router_par: (usize, usize),
    delta_u4: (usize, usize),
    dn_in_proj: (usize, usize),
    dn_gate: (usize, usize),
    at_qcast: (usize, usize),
    at_kv: (usize, usize),
    at_qknorm: (usize, usize),
    shared_fold: (usize, usize),
    gate_up: (usize, usize),
    gemv_unrolled: (usize, usize),
    quant_lane: (usize, usize),
    w8_proj: (usize, u64, u64),
    chain_ring: Option<wgpu::Buffer>,
    pf_list: Option<wgpu::Buffer>,
    pf_m: usize,
    pf_validated: bool,
    pfm: Option<PfMrowExec>,
    pfm_passes: Vec<Pass>,
    pfm_mix: (usize, usize),
    pfm_validated: bool,
    res: wgpu::Buffer,
    embed_gather_end: usize,
    pfm_gather_end: usize,
    splice_validated: bool,

    base_passes: usize,
}

pub use crate::embed_row_splice::EmbedRowSplice as EmbedRowsSplice;

pub const PREFILL_M_MAX: usize = 64;

pub fn prefill_m() -> usize {
    match std::env::var("NV_QWEN35MOE_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(m) => m.clamp(2, PREFILL_M_MAX),
        None => PREFILL_M_MAX,
    }
}

pub fn prefill_list_bytes_per_token_charged_by_mem_fit() -> usize {
    8 + std::mem::size_of::<FdParams>()
}

pub const PF_MROW_GEMM_M_MAX: usize = 16;

pub const PF_MROW_M_MAX_VIA_16_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS: usize = PREFILL_M_MAX;

pub const PF_MROW_BIND_OFFSET_ALIGN_BYTES: u64 = 256;

pub const PF_MROW_POS_STRIDE_WORDS: usize = 64;

pub fn pf_mrow_default_on_since_real_weights_first_token_parity_and_1_84x_at_8k() -> bool {
    !matches!(
        std::env::var("NV_Q3_WGPU_PF_MROW").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn pf_mrow_scratch_bytes_upper_bound_for_mem_fit(cfg: &Qwen3MoeConfig) -> usize {
    let m = PF_MROW_M_MAX_VIA_16_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS;
    let hidden = cfg.hidden_size;
    let moe = m * (cfg.num_experts_per_tok + 1) * (hidden * 2 + hidden * 2 + hidden / 2);
    let n_all = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let delta = m * (n_all * 2 + n_all * 4 + cfg.linear_conv_kernel_dim * n_all * 4);
    let scratch_rows = if pf_flash_tiled_default_on_since_slotml_solo_nll_plus_p006_nats_under_p01_bound_and_557_vs_431_tok_s_user_signed_off() {
        m
    } else {
        PF_MK_FLASH_ROWS_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS
    };
    let attn = 2 * m * cfg.num_attention_heads * cfg.head_dim * 4
        + cfg.num_attention_heads * scratch_rows * flash_splits() as usize * (cfg.head_dim + 2) * 4;
    (moe + delta + attn) * 3 / 2
}

fn pf_pad_bytes_to_bind_offset_align(raw: u64) -> u64 {
    raw.next_multiple_of(PF_MROW_BIND_OFFSET_ALIGN_BYTES)
}

fn pf_stride_elems_bf16(raw_elems: usize) -> usize {
    (pf_pad_bytes_to_bind_offset_align((raw_elems * 2) as u64) / 2) as usize
}

fn pf_stride_bytes_f32(raw_elems: usize) -> u64 {
    pf_pad_bytes_to_bind_offset_align((raw_elems * 4) as u64)
}

const PF_GLUE_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_glue.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfCopyParams {
    rows: u32,
    row_words: u32,
    src_stride_words: u32,
    slots: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGemmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    w_row_words: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    out_f32: u32,
    pad0: u32,
}

fn pf_gemm_bf16_mrow_entry(m: usize) -> String {
    format!("q3w_pf_gemm_bf16_m{m}")
}

fn pf_gemm_bf16_mrow_source(m: usize) -> String {
    use std::fmt::Write as _;
    assert!(
        (2..=PF_MROW_GEMM_M_MAX).contains(&m),
        "the unrolled M-row bf16 GEMM holds one f32 accumulator per token in registers, \
         so m must be 2..={PF_MROW_GEMM_M_MAX}, got {m}"
    );
    let mut b = String::new();
    b.push_str(
        "struct Q3pgParams {\n    n_rows: u32,\n    k_words: u32,\n    groups_x: u32,\n    w_row_words: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n    out_f32: u32,\n    pad0: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> gm_w: array<u32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> gm_x: array<u32>;\n");
    b.push_str("@group(0) @binding(2) var<uniform> gm_p: Q3pgParams;\n");
    b.push_str("@group(0) @binding(3) var<storage, read_write> gm_y: array<u32>;\n\n");
    writeln!(b, "var<workgroup> gm_red: array<f32, {}>;\n", 256 * m).unwrap();
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(b, "fn {}(", pf_gemm_bf16_mrow_entry(m)).unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let half = tid >> 7u;\n");
    b.push_str("    let lane = tid & 127u;\n");
    b.push_str("    let pair = wid.x + wid.y * gm_p.groups_x;\n");
    b.push_str("    let row = pair * 2u + half;\n");
    b.push_str("    let live = row < gm_p.n_rows;\n");
    b.push_str("    let wbase = select(0u, row * gm_p.w_row_words, live);\n");
    b.push_str("    let kw = select(0u, gm_p.k_words, live);\n");
    for mi in 0..m {
        writeln!(b, "    var acc{mi} = 0.0;").unwrap();
    }
    b.push_str("    for (var i = lane; i < kw; i = i + 128u) {\n");
    b.push_str("        let ww = gm_w[wbase + i];\n");
    b.push_str("        let wlo = bf16_lo(ww);\n");
    b.push_str("        let whi = bf16_hi(ww);\n");
    for mi in 0..m {
        writeln!(
            b,
            "        let xw{mi} = gm_x[{mi}u * gm_p.x_stride_words + i];\n\
             \x20       acc{mi} = fma(wlo, bf16_lo(xw{mi}), acc{mi});\n\
             \x20       acc{mi} = fma(whi, bf16_hi(xw{mi}), acc{mi});"
        )
        .unwrap();
    }
    b.push_str("    }\n");
    for mi in 0..m {
        writeln!(b, "    gm_red[{mi}u * 256u + tid] = acc{mi};").unwrap();
    }
    b.push_str("    workgroupBarrier();\n");
    b.push_str("    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {\n");
    b.push_str("        if (lane < stride) {\n");
    for mi in 0..m {
        writeln!(
            b,
            "            gm_red[{mi}u * 256u + tid] = gm_red[{mi}u * 256u + tid] \
             + gm_red[{mi}u * 256u + tid + stride];"
        )
        .unwrap();
    }
    b.push_str("        }\n        workgroupBarrier();\n    }\n");
    writeln!(b, "    if (tid < {m}u) {{").unwrap();
    b.push_str("        let mi = tid;\n");
    b.push_str("        let base = pair * 2u;\n");
    b.push_str("        let lo = gm_red[mi * 256u];\n");
    b.push_str("        var hi = 0.0;\n");
    b.push_str("        if (base + 1u < gm_p.n_rows) {\n");
    b.push_str("            hi = gm_red[mi * 256u + 128u];\n");
    b.push_str("        }\n");
    b.push_str("        if (base < gm_p.n_rows) {\n");
    b.push_str("            if (gm_p.out_f32 == 1u) {\n");
    b.push_str("                gm_y[mi * gm_p.y_stride_words + base] = bitcast<u32>(lo);\n");
    b.push_str("                if (base + 1u < gm_p.n_rows) {\n");
    b.push_str(
        "                    gm_y[mi * gm_p.y_stride_words + base + 1u] = bitcast<u32>(hi);\n",
    );
    b.push_str("                }\n");
    b.push_str("            } else {\n");
    b.push_str("                gm_y[mi * gm_p.y_stride_words + (base >> 1u)] = bf16_pack(lo, hi);\n");
    b.push_str("            }\n");
    b.push_str("        }\n");
    b.push_str("    }\n}\n");
    compose(&b)
}

pub fn pf_token_parallel_default_on_escape_to_per_token_copies() -> bool {
    !off("NV_Q3_WGPU_PF_TOKENPAR")
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfDeltaChunkParams {
    tokens: u32,
    in_stride_elems: u32,
    mixed_stride_elems: u32,
    qk_stride_elems: u32,
    v_stride_elems: u32,
    gb_stride_elems: u32,
    core_stride_elems: u32,
    gated_stride_elems: u32,
}

fn pf_delta_chunk_source(d_k: usize) -> String {
    assert!(
        (1..=128).contains(&d_k),
        "the chunked recurrent scan holds one d_k-long state column per lane in a \
         function-space array sized at pipeline build, so d_k must be 1..=128, got {d_k}"
    );
    compose(&format!(
        "{}\n{}",
        DELTA_WGSL,
        PF_DELTA_CHUNK_WGSL.replace("Q3PD_DK", &d_k.to_string())
    ))
}

pub const PF_MK_FLASH_ROWS_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfAttnRopeParams {
    tokens: u32,
    q_src_stride_elems: u32,
    k_src_stride_elems: u32,
    q_out_stride_elems: u32,
    k_out_stride_elems: u32,
    qf_out_stride_elems: u32,
    pos_stride_words: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfAttnGateParams {
    attn_stride_elems: u32,
    qraw_stride_elems: u32,
    out_stride_words: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfKvqParams {
    tokens: u32,
    x_stride_elems: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfTopkParams {
    tokens: u32,
    rl_stride_words: u32,
    sel_stride_words: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfCombineParams {
    tokens: u32,
    y_stride_words: u32,
    wts_stride_words: u32,
    slogit_stride_words: u32,
    out_stride_words: u32,
    sel_slots_per_token: u32,
    pad1: u32,
    pad2: u32,
}

#[doc(hidden)]
pub fn pf_mrow_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("q3m:pf_glue", compose(PF_GLUE_WGSL)),
        ("q3m:pf_gemm_m2", pf_gemm_bf16_mrow_source(2)),
        (
            "q3m:pf_gemm_m16",
            pf_gemm_bf16_mrow_source(PF_MROW_GEMM_M_MAX),
        ),
        ("q3m:pf_delta_chunk_dk16", pf_delta_chunk_source(16)),
        ("q3m:pf_delta_chunk_dk128", pf_delta_chunk_source(128)),
        ("q3m:pf_attn", pf_attn_source()),
        ("q3m:pf_kvq", pf_kvq_source()),
        ("q3m:pf_moe", pf_moe_batched_source()),
        ("q3m:pf_flash_sg", pf_flash_sg_source()),
        ("q3m:pf_flash_tiled_sg", pf_flash_tiled_source(true)),
        ("q3m:pf_flash_tiled_wg", pf_flash_tiled_source(false)),
        (
            "q3m:pf_moe_grouped_fmlut_cfg",
            pf_moe_grouped_source(wk::gemv_nvfp4_v2::V2Config::new(128, 4)),
        ),
        (
            "q3m:pf_moe_grouped_warp_cfg",
            pf_moe_grouped_source(wk::gemv_nvfp4_v2::V2Config::new(64, 1)),
        ),
    ]
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGroupedParams {
    zn: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfSortParams {
    tokens: u32,
    slots_per_token: u32,
    ids_stride_words: u32,
    bins_cover_n_experts_plus_the_shared_slot: u32,
}

#[allow(clippy::too_many_arguments)]
fn pf_push_gemv_nvfp4_grouped(
    b: &mut Builder,
    label: &str,
    w: &wgpu::Buffer,
    ws: &wgpu::Buffer,
    x: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel_sorted: &wgpu::Buffer,
    perm: &wgpu::Buffer,
    alphas: &wgpu::Buffer,
    n: usize,
    k: usize,
    zslots: usize,
    y_slot_stride_words: usize,
) -> Result<bool> {
    if !nvfp4_v2_enabled(b.ctx) {
        return Ok(false);
    }
    let k_blocks = k / NVFP4_BLOCK;
    if !k_blocks.is_multiple_of(4) || !(n * k_blocks).is_multiple_of(2) || n < 2 {
        return Ok(false);
    }
    let Some((kernel, cfg, pk_entry)) = wk::gemv_nvfp4_v2::select_pk_slots(n, k, zslots) else {
        return Ok(false);
    };
    let (entry, batch, vec4) = match pk_entry {
        wk::gemv_nvfp4_v2::FMLUT_PK_ENTRY => ("q3w_pf_gemv_nvfp4_fmlut_pair_grouped", 2usize, true),
        wk::gemv_nvfp4_v2::WARP_PK_ENTRY => ("q3w_pf_gemv_nvfp4_warp_grouped", 8usize, false),
        _ => return Ok(false),
    };
    let grid = b.grid1(n as u64, cfg.rows_per_group(kernel));
    let source = pf_moe_grouped_source(cfg);
    let p = b.uni(
        "q3w-pf-gemv4g-p",
        GemvNvfp4Params {
            alpha: 1.0,
            n_rows: n as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: (n * k_blocks) as u32,
            sf_e_stride_bytes: wk::gemv_nvfp4::swizzled_scale_len(n, k_blocks) as u32,
            x_slot_stride_vec2: k_blocks as u32,
            xsf_slot_stride_bytes: k_blocks as u32,
            y_slot_stride_words: y_slot_stride_words as u32,
            per_expert_alpha: 1,
            m_slots_sharing_expert_zero: 0,
        },
    );
    let ggp = b.uni(
        "q3w-pf-gemv4g-zn",
        PfGroupedParams {
            zn: zslots as u32,
            ..Default::default()
        },
    );
    let (w_slot, x_slot) = if vec4 { (18, 19) } else { (10, 12) };
    b.nvfp4_projs += 1;
    b.nvfp4_v2_routed += 1;
    b.push(
        label,
        &source,
        entry,
        &[
            (w_slot, w),
            (11, ws),
            (x_slot, x),
            (13, xs),
            (14, &p),
            (15, y),
            (17, alphas),
            (20, sel_sorted),
            (21, perm),
            (22, &ggp),
        ],
        (grid.0, grid.1, zslots.div_ceil(batch) as u32),
    )?;
    Ok(true)
}

struct PfDeltaBufs {
    in_stride_words: usize,
    in_all: wgpu::Buffer,
    mixed_stride_bytes: u64,
    mixed: wgpu::Buffer,
    qk_stride_bytes: u64,
    qg: wgpu::Buffer,
    kg: wgpu::Buffer,
    v_stride_bytes: u64,
    vg: wgpu::Buffer,
    gb_stride_bytes: u64,
    gexp: wgpu::Buffer,
    beta: wgpu::Buffer,
    core_stride_bytes: u64,
    core: wgpu::Buffer,
    gated_stride_words: usize,
    gated: wgpu::Buffer,
}

struct PfAttnBufs {
    fused_kv: bool,
    mk_scratch: Option<wgpu::Buffer>,
    xq: wgpu::Buffer,
    xs: wgpu::Buffer,
    q_raw_stride_words: usize,
    q_raw: wgpu::Buffer,
    vk_raw_stride_words: usize,
    vk_raw: wgpu::Buffer,
    k_raw_stride_words: usize,
    k_raw: wgpu::Buffer,
    q_stride_bytes: u64,
    q: wgpu::Buffer,
    k_stride_bytes: u64,
    k: wgpu::Buffer,
    qf32_stride_bytes: u64,
    q_f32: wgpu::Buffer,
    attn_stride_bytes: u64,
    attn: wgpu::Buffer,
    gated_stride_words: usize,
    gated: wgpu::Buffer,
    oq: wgpu::Buffer,
    os: wgpu::Buffer,
}

struct PfMoeBufs {
    rl_stride_bytes: u64,
    rlogits: wgpu::Buffer,
    ids: wgpu::Buffer,
    wts: wgpu::Buffer,
    sel_sorted: wgpu::Buffer,
    perm: wgpu::Buffer,
    sel_flat: wgpu::Buffer,
    x_rep: wgpu::Buffer,
    xq: wgpu::Buffer,
    xs: wgpu::Buffer,
    y_gate: wgpu::Buffer,
    y_up: wgpu::Buffer,
    aq: wgpu::Buffer,
    as_: wgpu::Buffer,
    y_down: wgpu::Buffer,
}

struct PfMrowBufs {
    m: usize,
    ok: bool,
    off_reason: Option<String>,
    sel_zeros_m: wgpu::Buffer,
    tok: wgpu::Buffer,
    pos_strided_one_i32_per_256b_slot: wgpu::Buffer,
    fd_strided_one_fdparams_per_256b_slot: wgpu::Buffer,
    res: wgpu::Buffer,
    normed: wgpu::Buffer,
    mix: wgpu::Buffer,
    normed_post: wgpu::Buffer,
    moe_out: wgpu::Buffer,
    gemm_tile_rows: usize,
    gemm_src: String,
    gemm_entry: String,
    gemm_tail: Option<(usize, String, String)>,
    delta: Option<PfDeltaBufs>,
    attn: Option<PfAttnBufs>,
    moe: Option<PfMoeBufs>,
}

struct PfMrowExec {
    m: usize,
    tok: wgpu::Buffer,
    pos: wgpu::Buffer,
    fd: wgpu::Buffer,
    res: wgpu::Buffer,
}

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_QWEN36_WGPU_VRAM")
}

pub fn staging_flush_enabled() -> bool {
    !matches!(
        std::env::var("NV_QWEN36_WGPU_STAGING_FLUSH")
            .ok()
            .as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

impl Qwen3MoeWgpu {
    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.config
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn pass_labels(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.label.as_str()).collect()
    }

    pub fn head_pass_start(&self) -> usize {
        self.head_start
    }

    pub fn pass_grid(&self, i: usize) -> (u32, u32, u32) {
        self.passes[i].grid
    }

    pub fn pass_bound_bytes(&self, i: usize) -> (u64, u64) {
        (self.passes[i].bound_bytes, self.passes[i].widest_bytes)
    }

    pub fn probe_at(&mut self, token: u32, pos: usize) {
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytemuck::bytes_of(&(token as i32)));
        self.ctx
            .queue
            .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(pos as i32)));
        let mut fd = self.fd_base;
        fd.total = (pos + 1) as u32;
        self.ctx
            .queue
            .write_buffer(&self.fd_buf, 0, bytemuck::bytes_of(&fd));
    }

    pub fn probe_prefix(&self, n: usize) -> Result<()> {
        anyhow::ensure!(
            n <= self.passes.len(),
            "prefix {n} beyond {} passes",
            self.passes.len()
        );
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.passes[..n] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("probe_prefix drain: {e}"))?;
        Ok(())
    }

    pub fn probe_append(&mut self, needle: &str, copies: usize) -> usize {
        self.passes.truncate(self.base_passes);
        let src: Vec<Pass> = self
            .passes
            .iter()
            .filter(|p| p.label.contains(needle))
            .cloned()
            .collect();
        for _ in 0..copies {
            self.passes.extend(src.iter().cloned());
        }
        src.len() * copies
    }

    pub fn probe_append_clear(&mut self) {
        self.passes.truncate(self.base_passes);
    }

    pub fn probe_prefix_tail(&self, n: usize, tail: usize) -> Result<()> {
        anyhow::ensure!(
            n <= self.passes.len() && tail < self.passes.len(),
            "prefix {n} / tail {tail} beyond {} passes",
            self.passes.len()
        );
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            for p in self.passes[..n].iter().chain(&self.passes[tail..=tail]) {
                cp.set_pipeline(&p.pipeline);
                cp.set_bind_group(0, &p.bind, &[]);
                cp.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("probe_prefix_tail drain: {e}"))?;
        Ok(())
    }

    pub fn probe_encode(&self, n: usize) -> Result<()> {
        anyhow::ensure!(
            n <= self.passes.len(),
            "prefix {n} beyond {} passes",
            self.passes.len()
        );
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.passes[..n] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        drop(enc.finish());
        Ok(())
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        if let Some(v) = self.verify.as_mut() {
            v.pending = None;
        }
        for (buf, bytes) in &self.state_buffers {
            let zeros = vec![0u8; *bytes as usize];
            self.ctx.queue.write_buffer(buf, 0, &zeros);
        }
        Ok(())
    }

    pub fn fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(
        &mut self,
        pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            pos <= self.max_seq,
            "synthetic fill pos {pos} past max_seq {}; window start and gdn state selection are pure functions of pos, so pos alone is the cache-depth state",
            self.max_seq
        );
        let max_bytes = self
            .state_buffers
            .iter()
            .map(|(_, b)| *b as usize)
            .max()
            .unwrap_or(0);
        let pattern = crate::gemma4_moe_wgpu::synthetic_state_bytes_0x30_to_0x3e_finite_small_under_fp8_bf16_and_f32_views_so_no_nan_and_moe_routing_stays_nondegenerate(max_bytes);
        for (buf, bytes) in &self.state_buffers {
            self.ctx
                .queue
                .write_buffer(buf, 0, &pattern[..*bytes as usize]);
        }
        self.pos = pos;
        Ok(())
    }

    pub fn new(config: Qwen3MoeConfig, weights: &HostWeights, max_seq: usize) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq)
    }

    pub fn from_loader(
        config: Qwen3MoeConfig,
        weights: &nv_weights::WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq)
    }

    fn build(config: Qwen3MoeConfig, src: WeightSource<'_>, max_seq: usize) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let s = Sources::new();
        let cfg = &config;

        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        anyhow::ensure!(
            cfg.hidden_size.is_multiple_of(2 * NVFP4_BLOCK),
            "hidden_size {} must be a multiple of {}",
            cfg.hidden_size,
            2 * NVFP4_BLOCK
        );
        anyhow::ensure!(
            cfg.moe_intermediate_size.is_multiple_of(4 * NVFP4_BLOCK),
            "moe_intermediate_size {} must be a multiple of {}",
            cfg.moe_intermediate_size,
            4 * NVFP4_BLOCK
        );
        anyhow::ensure!(
            cfg.shared_expert_intermediate_size
                .is_multiple_of(4 * NVFP4_BLOCK),
            "shared_expert_intermediate_size must be a multiple of {}",
            4 * NVFP4_BLOCK
        );
        anyhow::ensure!(
            cfg.num_experts <= 256,
            "router top-k kernel caps num_experts at 256, got {}",
            cfg.num_experts
        );
        anyhow::ensure!(
            cfg.num_experts_per_tok <= MAX_TOPK,
            "num_experts_per_tok {} exceeds {MAX_TOPK}",
            cfg.num_experts_per_tok
        );
        anyhow::ensure!(
            cfg.head_dim <= MAX_HEAD_DIM && cfg.head_dim.is_multiple_of(4),
            "head_dim {} must be a multiple of 4 (fp8 KV word packing) and <= {MAX_HEAD_DIM}",
            cfg.head_dim
        );
        anyhow::ensure!(
            cfg.linear_key_head_dim <= MAX_LIN_HEAD_DIM
                && cfg.linear_value_head_dim <= MAX_LIN_HEAD_DIM
                && cfg.linear_value_head_dim.is_multiple_of(2),
            "linear head dims must be even and <= {MAX_LIN_HEAD_DIM}"
        );
        anyhow::ensure!(
            cfg.linear_num_value_heads
                .is_multiple_of(cfg.linear_num_key_heads),
            "linear value heads must be a multiple of key heads"
        );
        anyhow::ensure!(
            cfg.num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads),
            "attention heads must be a multiple of kv heads"
        );
        anyhow::ensure!(
            cfg.layer_types.len() == cfg.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            cfg.layer_types.len(),
            cfg.num_hidden_layers
        );

        let hidden = cfg.hidden_size;
        let hidden_words = hidden / 2;
        let eps = cfg.rms_norm_eps as f32;
        let vocab = cfg.vocab_size;

        let mut b = Builder {
            core: crate::wgpu_ledger::VramLedger::new(
                ctx,
                "q3w-",
                staging_flush_enabled,
                STAGING_FLUSH_BYTES,
            ),
            passes: Vec::new(),
            pf_passes: Vec::new(),
            to_prefill: false,
            pf_mrow_pass_mix_m_row_then_per_token_copies: (0, 0),
            nvfp4_v2_routed: 0,
            nvfp4_projs: 0,
            router_par: (0, 0),
            delta_u4: (0, 0),
            dn_in_proj: (0, 0),
            dn_gate: (0, 0),
            at_qcast: (0, 0),
            at_kv: (0, 0),
            at_qknorm: (0, 0),
            shared_fold: (0, 0),
            gate_up: (0, 0),
            gemv_unrolled: (0, 0),
            quant_lane: (0, 0),
            w8_proj: (0, 0, 0),
        };

        let tok_buf = b.upload_u32("q3w-tok", &[0u32]);
        let pos_buf = b.upload_u32("q3w-pos", &[0u32]);

        let fd_base = FdParams {
            n_heads: cfg.num_attention_heads as u32,
            n_kv: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim as u32,
            splits: flash_splits(),
            out_bf16: 0,
            scaling: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        };
        let fd_buf = b.uni("q3w-fd", fd_base);

        let res = b.zeros("q3w-res", (hidden_words * 4) as u64);
        let res2 = b.zeros("q3w-res2", (hidden_words * 4) as u64);
        let normed = b.zeros("q3w-normed", (hidden_words * 4) as u64);
        let mixed_out = b.zeros("q3w-mix", (hidden_words * 4) as u64);
        let moe_out = b.zeros("q3w-moeout", (hidden_words * 4) as u64);

        let pf_mrow_m = if pf_mrow_default_on_since_real_weights_first_token_parity_and_1_84x_at_8k() {
            let mut m = prefill_m().min(PF_MROW_M_MAX_VIA_16_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS);
            if m > PF_MROW_GEMM_M_MAX && m % PF_MROW_GEMM_M_MAX == 1 {
                m -= 1;
            }
            m
        } else {
            0
        };
        let mut pf_mrow: Option<PfMrowBufs> = if pf_mrow_m >= 2 {
            let m = pf_mrow_m;
            anyhow::ensure!(
                hidden.is_multiple_of(128),
                "the M-row list offset-binds hidden-row slices at token strides, so hidden \
                 must keep hidden*2 bytes a multiple of {PF_MROW_BIND_OFFSET_ALIGN_BYTES}"
            );
            let mw = (m * hidden_words * 4) as u64;
            let mk_groups = m.div_ceil(PF_MK_FLASH_ROWS_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS);
            let fd = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q3w-pf-fd"),
                size: (m + mk_groups + PF_FD_ONE_TRAILING_SLOT_VIEWS_THE_WHOLE_CHUNK_FOR_TILED_FLASH)
                    as u64
                    * PF_MROW_BIND_OFFSET_ALIGN_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some(PfMrowBufs {
                m,
                ok: true,
                off_reason: None,
                sel_zeros_m: b.upload_u32("q3w-pf-sel0", &vec![0u32; m]),
                tok: b.upload_u32("q3w-pf-tok", &vec![0u32; m]),
                pos_strided_one_i32_per_256b_slot: b
                    .zeros("q3w-pf-pos", (m * PF_MROW_POS_STRIDE_WORDS * 4) as u64),
                fd_strided_one_fdparams_per_256b_slot: b.store("q3w-pf-fd", fd),
                res: b.zeros("q3w-pf-res", mw),
                normed: b.zeros("q3w-pf-normed", mw),
                mix: b.zeros("q3w-pf-mix", mw),
                normed_post: b.zeros("q3w-pf-normed-post", mw),
                moe_out: b.zeros("q3w-pf-moeout", mw),
                gemm_tile_rows: m.min(PF_MROW_GEMM_M_MAX),
                gemm_src: pf_gemm_bf16_mrow_source(m.min(PF_MROW_GEMM_M_MAX)),
                gemm_entry: pf_gemm_bf16_mrow_entry(m.min(PF_MROW_GEMM_M_MAX)),
                gemm_tail: (m > PF_MROW_GEMM_M_MAX && !m.is_multiple_of(PF_MROW_GEMM_M_MAX)).then(
                    || {
                        let tail = m % PF_MROW_GEMM_M_MAX;
                        (
                            tail,
                            pf_gemm_bf16_mrow_source(tail),
                            pf_gemm_bf16_mrow_entry(tail),
                        )
                    },
                ),
                delta: None,
                attn: None,
                moe: None,
            })
        } else {
            None
        };

        let embed = src.embed(cfg)?;
        anyhow::ensure!(
            embed.len() == vocab * hidden,
            "embed has {} values, want {}",
            embed.len(),
            vocab * hidden
        );
        let chunk_rows = row_chunk(ctx, hidden);
        let mut off = 0usize;
        while off < vocab {
            let rows = chunk_rows.min(vocab - off);
            let buf = b.upload_u32(
                "q3w-embed",
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let p = b.uni(
                "q3w-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "q3w-gather",
                &s.moe,
                "q3w_gather_embed",
                &[(30, &buf), (31, &tok_buf), (32, &res), (33, &p)],
                grid,
            )?;
            if let Some(pfb) = pf_mrow.as_ref().filter(|pfb| pfb.ok) {
                let pp = b.uni(
                    "q3w-pf-embed-p",
                    GatherParams {
                        row_off: off as u32,
                        n_rows: rows as u32,
                        hidden_words: hidden_words as u32,
                        vocab: vocab as u32,
                    },
                );
                b.to_prefill = true;
                b.push(
                    "q3w-pf-gather",
                    &s.pf_glue,
                    "q3w_pf_gather_embed_m",
                    &[(10, &buf), (11, &pfb.tok), (12, &pfb.res), (13, &pp)],
                    ((hidden_words as u32).div_ceil(256), pfb.m as u32, 1),
                )?;
                b.to_prefill = false;
            }
            off += rows;
        }
        drop(embed);
        let embed_gather_end = b.passes.len();
        let pfm_gather_end = b.pf_passes.len();

        let mut state_buffers: Vec<(wgpu::Buffer, u64)> = Vec::new();
        let mut recurrent_states: Vec<(wgpu::Buffer, u64)> = Vec::new();

        for li in 0..cfg.num_hidden_layers {
            let layer = src.layer(cfg, li)?;
            let ln_w = b.upload_u32("q3w-ln", &pack_pairs(&layer.input_ln));
            if li == 0 {
                let p = b.uni(
                    "q3w-rms-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "q3w-rms0",
                    &s.rms,
                    "rmsnorm_bf16",
                    &[(0, &res), (1, &ln_w), (2, &normed), (3, &p)],
                    (1, 1, 1),
                )?;
                if let Some(pfb) = pf_mrow.as_ref().filter(|pfb| pfb.ok) {
                    let pp = b.uni(
                        "q3w-pf-rms-p",
                        RmsParams {
                            hidden: hidden as u32,
                            batch: pfb.m as u32,
                            eps,
                            words_per_row: hidden_words as u32,
                        },
                    );
                    b.to_prefill = true;
                    b.push(
                        "q3w-pf-rms0",
                        &s.rms,
                        "rmsnorm_bf16",
                        &[(0, &pfb.res), (1, &ln_w), (2, &pfb.normed), (3, &pp)],
                        (pfb.m as u32, 1, 1),
                    )?;
                    b.to_prefill = false;
                }
            }

            match &layer.mixer {
                HostMixer::Delta(d) => {
                    let recurrent_from = state_buffers.len();
                    build_delta(
                        &mut b,
                        &s,
                        cfg,
                        d,
                        &normed,
                        &mixed_out,
                        &mut state_buffers,
                        &mut pf_mrow,
                    )?;
                    recurrent_states.extend_from_slice(&state_buffers[recurrent_from..]);
                }
                HostMixer::Attn(a) => {
                    build_attn(
                        &mut b,
                        &s,
                        cfg,
                        a,
                        &normed,
                        &mixed_out,
                        &pos_buf,
                        &fd_buf,
                        max_seq,
                        &mut state_buffers,
                        &mut pf_mrow,
                    )?;
                }
            }

            let post_w = b.upload_u32("q3w-post-ln", &pack_pairs(&layer.post_attn_ln));
            let normed_post = b.zeros("q3w-normed-post", (hidden_words * 4) as u64);
            let rp = b.uni(
                "q3w-rmsres-p",
                RmsParams {
                    hidden: hidden as u32,
                    batch: 1,
                    eps,
                    words_per_row: hidden_words as u32,
                },
            );
            b.push(
                "q3w-rmsres-post",
                &s.rmsres,
                "rmsnorm_residual_bf16",
                &[
                    (0, &mixed_out),
                    (1, &res),
                    (2, &post_w),
                    (3, &normed_post),
                    (4, &rp),
                ],
                (1, 1, 1),
            )?;
            if let Some(pfb) = pf_mrow.as_ref().filter(|pfb| pfb.ok) {
                let pp = b.uni(
                    "q3w-pf-rmsres-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: pfb.m as u32,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.to_prefill = true;
                b.push(
                    "q3w-pf-rmsres-post",
                    &s.rmsres,
                    "rmsnorm_residual_bf16",
                    &[
                        (0, &pfb.mix),
                        (1, &pfb.res),
                        (2, &post_w),
                        (3, &pfb.normed_post),
                        (4, &pp),
                    ],
                    (pfb.m as u32, 1, 1),
                )?;
                b.to_prefill = false;
            }

            build_moe(
                &mut b,
                &s,
                cfg,
                &layer.moe,
                &normed_post,
                &moe_out,
                &mut pf_mrow,
            )?;

            if li + 1 < cfg.num_hidden_layers {
                let next = src.layer_input_ln(cfg, li + 1)?;
                let nw = b.upload_u32("q3w-next-ln", &pack_pairs(&next));
                let rp2 = b.uni(
                    "q3w-rmsres2-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "q3w-rmsres-next",
                    &s.rmsres,
                    "rmsnorm_residual_bf16",
                    &[(0, &moe_out), (1, &res), (2, &nw), (3, &normed), (4, &rp2)],
                    (1, 1, 1),
                )?;
                if let Some(pfb) = pf_mrow.as_ref().filter(|pfb| pfb.ok) {
                    let pp = b.uni(
                        "q3w-pf-rmsres2-p",
                        RmsParams {
                            hidden: hidden as u32,
                            batch: pfb.m as u32,
                            eps,
                            words_per_row: hidden_words as u32,
                        },
                    );
                    b.to_prefill = true;
                    b.push(
                        "q3w-pf-rmsres-next",
                        &s.rmsres,
                        "rmsnorm_residual_bf16",
                        &[
                            (0, &pfb.moe_out),
                            (1, &pfb.res),
                            (2, &nw),
                            (3, &pfb.normed),
                            (4, &pp),
                        ],
                        (pfb.m as u32, 1, 1),
                    )?;
                    b.to_prefill = false;
                }
            } else {
                let sp = b.uni(
                    "q3w-resadd-p",
                    ResScaleParams {
                        n: hidden as u32,
                        n_words: hidden_words as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1(hidden_words as u64, 256);
                b.push(
                    "q3w-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &moe_out), (1, &res), (2, &res2), (3, &sp)],
                    grid,
                )?;
            }
        }

        let verify = match pf_mrow.as_ref().filter(|p| p.ok) {
            Some(p) => {
                let m = p.m;
                let res2_rows = b.zeros("q3w-vf-res2", (m * hidden_words * 4) as u64);
                let vp = b.uni(
                    "q3w-vf-resadd-p",
                    ResScaleParams {
                        n: (m * hidden) as u32,
                        n_words: (m * hidden_words) as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1((m * hidden_words) as u64, 256);
                let (moe_out_rows, res_rows) = (p.moe_out.clone(), p.res.clone());
                let resadd = b.make(
                    "q3w-vf-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &moe_out_rows), (1, &res_rows), (2, &res2_rows), (3, &vp)],
                    grid,
                )?;
                let tok = b.zeros("q3w-vf-tok", (m * 4) as u64);
                Some(Verify {
                    rows: m,
                    res2_rows,
                    resadd,
                    tok,
                    row_logits: None,
                    rollback: Vec::new(),
                    validated: false,
                    pending: None,
                })
            }
            None => None,
        };

        let final_start = b.passes.len();
        let final_w = b.upload_u32("q3w-final-ln", &pack_pairs(&src.final_norm(cfg)?));
        let final_x = b.zeros("q3w-final-x", (hidden_words * 4) as u64);
        let fp = b.uni(
            "q3w-final-p",
            RmsParams {
                hidden: hidden as u32,
                batch: 1,
                eps,
                words_per_row: hidden_words as u32,
            },
        );
        b.push(
            "q3w-final-rms",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &res2), (1, &final_w), (2, &final_x), (3, &fp)],
            (1, 1, 1),
        )?;

        let head_start = b.passes.len();
        let logits = b.zeros("q3w-logits", (vocab * 4) as u64);
        let lm = src.lm_head(cfg)?;
        anyhow::ensure!(
            lm.len() == vocab * hidden,
            "lm_head has {} values, want {}",
            lm.len(),
            vocab * hidden
        );
        let lm_w8 = w8_lmhead_enabled(ctx).then(w8_proj_group);
        let mut off = 0usize;
        while off < vocab {
            let rows = (chunk_rows.min(vocab - off)) & !1usize;
            let rows = if rows == 0 { vocab - off } else { rows };
            if let Some(g) = lm_w8.filter(|g| hidden.is_multiple_of(*g)) {
                let ex = upload_bf16_as_i8(
                    &mut b,
                    "q3w-lmhead",
                    &HostBf16Lin {
                        w: lm[off * hidden..(off + rows) * hidden].to_vec(),
                        n: rows,
                        k: hidden,
                    },
                    g,
                );
                push_gemv_i8_f32(&mut b, &s, "q3w-lmhead", &ex, &final_x, &logits, off)?;
                off += rows;
                continue;
            }
            let wbuf = b.upload_u32(
                "q3w-lmhead",
                &pack_pairs(&lm[off * hidden..(off + rows) * hidden]),
            );
            let pairs = rows.div_ceil(2);
            let grid = b.grid1(pairs as u64, 1);
            let p = b.uni(
                "q3w-lmhead-p",
                GemvBf16Params {
                    n_rows: rows as u32,
                    k_words: hidden_words as u32,
                    groups_x: grid.0,
                    out_f32: 1,
                    w_row_words: hidden_words as u32,
                    x_off_words: 0,
                    y_off_words: off as u32,
                    alpha: 1.0,
                    ..Default::default()
                },
            );
            b.push(
                "q3w-lmhead",
                &s.gemv_bf16,
                "q3w_gemv_bf16",
                &[(0, &wbuf), (1, &final_x), (2, &p), (3, &logits)],
                grid,
            )?;
            off += rows;
        }
        drop(lm);

        let pv = b.zeros("q3w-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("q3w-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("q3w-token", 4);
        let ap = b.uni(
            "q3w-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "q3w-am1",
            &s.moe,
            "q3w_argmax_stage1",
            &[(40, &logits), (41, &pv), (42, &pi), (44, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "q3w-am2",
            &s.moe,
            "q3w_argmax_stage2",
            &[(41, &pv), (42, &pi), (43, &token_out), (44, &ap)],
            (1, 1, 1),
        )?;

        b.flush_staging();
        let vram = b.report();
        if vram_report_enabled() {
            eprint!("[q3w-wgpu] {}", vram.render());
        }

        let Builder {
            core,
            passes,
            pf_passes,
            pf_mrow_pass_mix_m_row_then_per_token_copies,
            nvfp4_v2_routed,
            nvfp4_projs,
            router_par,
            delta_u4,
            dn_in_proj,
            dn_gate,
            at_qcast,
            at_kv,
            at_qknorm,
            shared_fold,
            gate_up,
            gemv_unrolled,
            quant_lane,
            w8_proj,
            ..
        } = b;
        let buffers = core.buffers;
        if let Some(line) = nvfp4_v2_boot_line(nvfp4_v2_enabled(ctx), nvfp4_v2_routed, nvfp4_projs)
        {
            eprintln!("{line}");
        }
        eprintln!(
            "[q3w-wgpu] router top-k: {}/{} layers parallel ({}); \
             delta recurrent: {}/{} layers unrolled ({}); \
             nvfp4 quantize: {}/{} passes lane-split ({})",
            router_par.0,
            router_par.1,
            router_topk_entry().1,
            delta_u4.0,
            delta_u4.1,
            delta_recurrent_kernel().0,
            quant_lane.0,
            quant_lane.1,
            match quant_lane_mode() {
                Some(wg) => format!("wg{wg}"),
                None => "off".to_string(),
            }
        );
        eprintln!(
            "[q3w-wgpu] fusions: delta in_proj {}/{} stacked, gating {}/{} on the \
             split workgroup, q f32-cast {}/{} in norm+rope, attn v+k in one GEMV on \
             {}/{}, attn q+k norm+rope in one pass on {}/{}, shared expert \
             folded into the routed slots on {}/{} layers, MoE gate+up in one GEMV on \
             {}/{}; passes_per_token={}",
            dn_in_proj.0,
            dn_in_proj.1,
            dn_gate.0,
            dn_gate.1,
            at_qcast.0,
            at_qcast.1,
            at_kv.0,
            at_kv.1,
            at_qknorm.0,
            at_qknorm.1,
            shared_fold.0,
            shared_fold.1,
            gate_up.0,
            gate_up.1,
            passes.len()
        );
        eprintln!(
            "[q3w-wgpu] bf16->int8 projections: {} converted, {:.4} -> {:.4} GB \
             of per-token weight traffic (delta={} lmhead={} group={})",
            w8_proj.0,
            w8_proj.1 as f64 / 1e9,
            w8_proj.2 as f64 / 1e9,
            match w8_delta_proj_enabled(ctx) {
                (true, true) => "all",
                (true, false) => "in",
                (false, true) => "out",
                (false, false) => "off",
            },
            u8::from(w8_lmhead_enabled(ctx)),
            w8_proj_group(),
        );

        let base_passes = passes.len();
        let pf_m = prefill_m();
        let pf_list = if pf_m >= 2 {
            Some(dispatch::storage_zeroed(
                ctx,
                "q3w-pf-list",
                (pf_m * prefill_list_bytes_per_token_charged_by_mem_fit()) as u64,
            ))
        } else {
            None
        };
        let (pfm, pfm_passes, pfm_mix) = match pf_mrow {
            Some(p) if p.ok && !pf_passes.is_empty() => {
                eprintln!(
                    "[q3w-wgpu] prefill M-row list: m={}, {} passes per chunk \
                     ({} M-row + {} per-token copies)",
                    p.m,
                    pf_passes.len(),
                    pf_mrow_pass_mix_m_row_then_per_token_copies.0,
                    pf_mrow_pass_mix_m_row_then_per_token_copies.1,
                );
                (
                    Some(PfMrowExec {
                        m: p.m,
                        tok: p.tok.clone(),
                        pos: p.pos_strided_one_i32_per_256b_slot.clone(),
                        fd: p.fd_strided_one_fdparams_per_256b_slot.clone(),
                        res: p.res.clone(),
                    }),
                    pf_passes,
                    pf_mrow_pass_mix_m_row_then_per_token_copies,
                )
            }
            Some(p) => {
                eprintln!(
                    "[q3w-wgpu] prefill M-row list off: {}",
                    p.off_reason
                        .as_deref()
                        .unwrap_or("no layer produced M-row passes")
                );
                (None, Vec::new(), (0, 0))
            }
            None => (None, Vec::new(), (0, 0)),
        };
        Ok(Self {
            ctx,
            config,
            max_seq,
            pos: 0,
            validated: false,
            prefix_validated: false,
            passes,
            head_start,
            final_start,
            verify: verify.filter(|_| pfm.is_some()),
            _buffers: buffers,
            tok_buf,
            pos_buf,
            fd_buf,
            fd_base,
            res2: res2.clone(),
            token_out,
            logits,
            state_buffers,
            recurrent_states,
            vocab,
            vram,
            nvfp4_v2: (nvfp4_v2_routed, nvfp4_projs),
            router_par,
            delta_u4,
            dn_in_proj,
            dn_gate,
            at_qcast,
            at_kv,
            at_qknorm,
            shared_fold,
            gate_up,
            gemv_unrolled,
            quant_lane,
            w8_proj,
            chain_ring: None,
            pf_list,
            pf_m,
            pf_validated: false,
            pfm,
            pfm_passes,
            pfm_mix,
            pfm_validated: false,
            res: res.clone(),
            embed_gather_end,
            pfm_gather_end,
            splice_validated: false,
            base_passes,
        })
    }

    pub fn nvfp4_v2_gemvs(&self) -> (usize, usize) {
        self.nvfp4_v2
    }

    pub fn int8_projection_bytes(&self) -> (usize, u64, u64) {
        self.w8_proj
    }

    pub fn router_parallel_layers(&self) -> (usize, usize) {
        self.router_par
    }

    pub fn delta_unrolled_layers(&self) -> (usize, usize) {
        self.delta_u4
    }

    pub fn delta_in_proj_fused_layers(&self) -> (usize, usize) {
        self.dn_in_proj
    }

    pub fn delta_gate_fused_layers(&self) -> (usize, usize) {
        self.dn_gate
    }

    pub fn attn_qcast_fused_layers(&self) -> (usize, usize) {
        self.at_qcast
    }

    pub fn attn_kv_fused_layers(&self) -> (usize, usize) {
        self.at_kv
    }

    pub fn attn_qknorm_fused_layers(&self) -> (usize, usize) {
        self.at_qknorm
    }

    pub fn shared_expert_folded_layers(&self) -> (usize, usize) {
        self.shared_fold
    }

    pub fn moe_gate_up_fused_layers(&self) -> (usize, usize) {
        self.gate_up
    }

    pub fn gemv_bf16_unrolled_matrices(&self) -> (usize, usize) {
        self.gemv_unrolled
    }

    pub fn quant_lane_passes(&self) -> (usize, usize) {
        self.quant_lane
    }

    pub fn decode_chain(&mut self, token: u32, k: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(k >= 1, "decode_chain needs k >= 1");
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(
            self.pos + k <= self.max_seq,
            "kv cache holds {} more steps, asked for {k}",
            self.max_seq - self.pos
        );
        if k == 1 || (dispatch::profile::enabled() && self.ctx.caps.timestamp_query) {
            let mut out = Vec::with_capacity(k);
            let mut t = token;
            for _ in 0..k {
                t = self.decode_step(t)?;
                out.push(t);
            }
            return Ok(out);
        }

        let ring_bytes = (k * 4) as u64;
        let fresh = !matches!(&self.chain_ring, Some(r) if r.size() >= ring_bytes);
        if fresh {
            self.chain_ring = Some(dispatch::storage_zeroed(
                self.ctx,
                "q3w-chain-ring",
                ring_bytes,
            ));
        }
        let ring = self.chain_ring.clone().expect("ring allocated above");

        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytemuck::bytes_of(&(token as i32)));
        for i in 0..k {
            let pos = self.pos + i;
            self.ctx
                .queue
                .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(pos as i32)));
            let mut fd = self.fd_base;
            fd.total = (pos + 1) as u32;
            self.ctx
                .queue
                .write_buffer(&self.fd_buf, 0, bytemuck::bytes_of(&fd));

            let scope = if self.validated {
                None
            } else {
                Some(
                    self.ctx
                        .device
                        .push_error_scope(wgpu::ErrorFilter::Validation),
                )
            };
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            enc.copy_buffer_to_buffer(&self.token_out, 0, &ring, (i * 4) as u64, 4);
            if i + 1 < k {
                enc.copy_buffer_to_buffer(&self.token_out, 0, &self.tok_buf, 0, 4);
            }
            self.ctx.queue.submit([enc.finish()]);
            if let Some(scope) = scope {
                if let Some(e) = pollster::block_on(scope.pop()) {
                    anyhow::bail!("qwen3_5_moe_wgpu decode chain validation: {e}");
                }
                self.validated = true;
                self.prefix_validated = true;
            }
        }
        self.pos += k;
        dispatch::read_back(self.ctx, &ring, k).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn step_inner(&mut self, token: u32, full: bool) -> Result<()> {
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(
            self.pos < self.max_seq,
            "kv cache full at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytemuck::bytes_of(&(token as i32)));
        self.ctx
            .queue
            .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(self.pos as i32)));
        let mut fd = self.fd_base;
        fd.total = (self.pos + 1) as u32;
        self.ctx
            .queue
            .write_buffer(&self.fd_buf, 0, bytemuck::bytes_of(&fd));

        let need_scope = if full {
            !self.validated
        } else {
            !self.validated && !self.prefix_validated
        };
        let scope = if need_scope {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        } else {
            None
        };
        let passes = if full {
            &self.passes[..]
        } else {
            &self.passes[..self.head_start]
        };
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = passes
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow::anyhow!("profiled submit: {e}"))?;
        } else {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_moe_wgpu decode step validation: {e}");
            }
            if full {
                self.validated = true;
            }
            self.prefix_validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    crate::wgpu_step_readback_api!();

    pub fn last_decode_logits(&self) -> Result<Vec<f32>> {
        dispatch::read_back(self.ctx, &self.logits, self.vocab).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn prefill_chunk_len(&self) -> usize {
        if self.pf_list.is_none() {
            0
        } else {
            self.pf_m
        }
    }

    pub fn prefill_mrow_chunk_len(&self) -> usize {
        self.pfm.as_ref().map_or(0, |e| e.m)
    }

    pub fn prefill_mrow_pass_count(&self) -> usize {
        self.pfm_passes.len()
    }

    pub fn prefill_mrow_pass_mix(&self) -> (usize, usize) {
        self.pfm_mix
    }

    fn pfm_write_host_inputs(&self, e: &PfMrowExec, chunk: &[u32]) {
        let m = e.m;
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        self.ctx
            .queue
            .write_buffer(&e.tok, 0, bytemuck::cast_slice(&ids));
        let mut posw = vec![0u32; m * PF_MROW_POS_STRIDE_WORDS];
        for (t, w) in posw.chunks_exact_mut(PF_MROW_POS_STRIDE_WORDS).enumerate() {
            w[0] = (self.pos + t) as i32 as u32;
        }
        self.ctx
            .queue
            .write_buffer(&e.pos, 0, bytemuck::cast_slice(&posw));
        let stride = PF_MROW_BIND_OFFSET_ALIGN_BYTES as usize;
        let rows = PF_MK_FLASH_ROWS_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
        let groups = m.div_ceil(rows);
        let mut fdb =
            vec![
                0u8;
                (m + groups + PF_FD_ONE_TRAILING_SLOT_VIEWS_THE_WHOLE_CHUNK_FOR_TILED_FLASH)
                    * stride
            ];
        for (t, slot) in fdb.chunks_exact_mut(stride).enumerate() {
            let mut fd = self.fd_base;
            if t < m {
                fd.total = (self.pos + t + 1) as u32;
            } else if t == m + groups {
                fd.total = (self.pos + m) as u32;
                fd.m_rows = m as u32;
            } else {
                let g = t - m;
                let mr_g = rows.min(m - g * rows);
                fd.total = (self.pos + g * rows + mr_g) as u32;
                fd.m_rows = mr_g as u32;
            }
            slot[..std::mem::size_of::<FdParams>()].copy_from_slice(bytemuck::bytes_of(&fd));
        }
        self.ctx.queue.write_buffer(&e.fd, 0, &fdb);
    }

    fn prefill_chunk_m_row_projections(&mut self, chunk: &[u32]) -> Result<()> {
        let e = self.pfm.as_ref().expect("M-row chunk without the pf list");
        let m = e.m;
        anyhow::ensure!(
            chunk.len() == m,
            "the M-row list is baked for chunks of exactly {m} tokens, got {}; the caller \
             routes shorter tails through the per-token replay",
            chunk.len()
        );
        anyhow::ensure!(
            self.pos + m <= self.max_seq,
            "kv cache full at {} + {m} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in chunk {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        self.pfm_write_host_inputs(e, chunk);
        let scope = if self.pfm_validated {
            None
        } else {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        };
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = self
                .pfm_passes
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = self.pfm_passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow::anyhow!("profiled M-row chunk submit: {e}"))?;
        } else {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.pfm_passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(err) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_moe_wgpu M-row prefill chunk validation: {err}");
            }
            self.pfm_validated = true;
        }
        self.pos += m;
        Ok(())
    }

    fn prefill_chunk_one_submission_of_per_token_passes(&mut self, chunk: &[u32]) -> Result<()> {
        let n = chunk.len();
        anyhow::ensure!(
            (2..=self.pf_m).contains(&n),
            "prefill chunk is {n} tokens, want 2..={}; a 1-token chunk is the per-token path \
             wearing the chunked path's name",
            self.pf_m
        );
        anyhow::ensure!(
            self.pos + n <= self.max_seq,
            "kv cache full at {} + {n} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in chunk {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let fd_sz = std::mem::size_of::<FdParams>();
        let mut host: Vec<u8> = Vec::with_capacity(n * (8 + fd_sz));
        for &t in chunk {
            host.extend_from_slice(bytemuck::bytes_of(&(t as i32)));
        }
        for i in 0..n {
            host.extend_from_slice(bytemuck::bytes_of(&((self.pos + i) as i32)));
        }
        for i in 0..n {
            let mut fd = self.fd_base;
            fd.total = (self.pos + i + 1) as u32;
            host.extend_from_slice(bytemuck::bytes_of(&fd));
        }
        let list = self
            .pf_list
            .clone()
            .expect("prefill chunk without pf list");
        self.ctx.queue.write_buffer(&list, 0, &host);
        let scope = if self.pf_validated {
            None
        } else {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        };
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for i in 0..n {
            enc.copy_buffer_to_buffer(&list, (i * 4) as u64, &self.tok_buf, 0, 4);
            enc.copy_buffer_to_buffer(&list, ((n + i) * 4) as u64, &self.pos_buf, 0, 4);
            enc.copy_buffer_to_buffer(
                &list,
                (n * 8 + i * fd_sz) as u64,
                &self.fd_buf,
                0,
                fd_sz as u64,
            );
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.passes[..self.head_start] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
            drop(pass);
        }
        self.ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_moe_wgpu prefill chunk validation: {e}");
            }
            self.pf_validated = true;
            self.prefix_validated = true;
        }
        self.pos += n;
        Ok(())
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let mut done = 0usize;
        if let Some(mm) = self.pfm.as_ref().map(|e| e.m) {
            while tokens.len() - done >= mm && self.pos + mm <= self.max_seq {
                self.prefill_chunk_m_row_projections(&tokens[done..done + mm])?;
                done += mm;
            }
        }
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            return Ok(done);
        }
        let m = self.prefill_chunk_len();
        if m == 0 {
            return Ok(done);
        }
        loop {
            let left = tokens.len() - done;
            if left < 2 {
                return Ok(done);
            }
            let take = m.min(left).min(self.max_seq.saturating_sub(self.pos));
            if take < 2 {
                return Ok(done);
            }
            self.prefill_chunk_one_submission_of_per_token_passes(&tokens[done..done + take])?;
            done += take;
        }
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        anyhow::ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let (last, rest) = tokens.split_last().expect("non-empty");
        let done = self.prefill_tokens(rest)?;
        for t in &rest[done..] {
            self.prefill_step(*t)?;
        }
        self.decode_step(*last)
    }

    pub fn verify_max_rows(&self) -> usize {
        self.verify.as_ref().map_or(0, |v| v.rows)
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        Ok(self.verify_chain_inner(batch, false)?.0)
    }

    pub fn verify_chain_logits(&mut self, batch: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        self.verify_chain_inner(batch, true)
    }

    fn verify_chain_inner(
        &mut self,
        batch: &[u32],
        want_logits: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows >= 2,
            "verify_chain needs the M-row prefill list; it is off (NV_Q3_WGPU_PF_MROW=0 or a \
             build-time bail printed at load)"
        );
        let live = batch.len();
        anyhow::ensure!(
            (1..=rows).contains(&live),
            "verify_chain batch {live} out of 1..={rows}"
        );
        anyhow::ensure!(
            self.verify.as_ref().is_some_and(|v| v.pending.is_none()),
            "verify_chain twice without advance(): {VERIFY_CHAIN_FORWARDS_THE_WHOLE_BAKED_M_ROW_CHUNK_AND_COMMITS_IN_PLACE_ONLY_AT_FULL_WIDTH}"
        );
        anyhow::ensure!(
            self.pos + rows <= self.max_seq,
            "verify_chain forwards the whole {rows}-row chunk, so it needs {rows} free kv rows \
             at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in batch {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let hidden_words = self.config.hidden_size / 2;
        let row_bytes = (hidden_words * 4) as u64;
        let vocab_bytes = (self.vocab * 4) as u64;
        if self.verify.as_ref().is_some_and(|v| v.rollback.is_empty()) {
            let rollback: Vec<wgpu::Buffer> = self
                .recurrent_states
                .iter()
                .map(|(_, bytes)| dispatch::storage_zeroed(self.ctx, "q3w-vf-rollback", *bytes))
                .collect();
            self.verify.as_mut().expect("verify list").rollback = rollback;
        }
        if want_logits && self.verify.as_ref().is_some_and(|v| v.row_logits.is_none()) {
            let buf =
                dispatch::storage_zeroed(self.ctx, "q3w-vf-row-logits", rows as u64 * vocab_bytes);
            self.verify.as_mut().expect("verify list").row_logits = Some(buf);
        }

        let mut padded: Vec<u32> = batch.to_vec();
        padded.resize(rows, *batch.last().expect("non-empty batch"));
        let e = self.pfm.as_ref().expect("verify without the M-row list");
        self.pfm_write_host_inputs(e, &padded);

        let v = self.verify.as_ref().expect("verify list");
        let scope = (!v.validated).then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for ((src, bytes), dst) in self.recurrent_states.iter().zip(&v.rollback) {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, *bytes);
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in self.pfm_passes.iter().chain(std::iter::once(&v.resadd)) {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        for r in 0..live {
            enc.copy_buffer_to_buffer(&v.res2_rows, r as u64 * row_bytes, &self.res2, 0, row_bytes);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.passes[self.final_start..] {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            enc.copy_buffer_to_buffer(&self.token_out, 0, &v.tok, r as u64 * 4, 4);
            if want_logits {
                let rl = v.row_logits.as_ref().expect("row logits buffer");
                enc.copy_buffer_to_buffer(&self.logits, 0, rl, r as u64 * vocab_bytes, vocab_bytes);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(err) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_moe_wgpu verify chain validation: {err}");
            }
        }
        let toks: Vec<u32> =
            dispatch::read_back(self.ctx, &v.tok, live).map_err(|e| anyhow::anyhow!("{e}"))?;
        let logits = match v.row_logits.as_ref().filter(|_| want_logits) {
            Some(rl) => dispatch::read_back(self.ctx, rl, live * self.vocab)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            None => Vec::new(),
        };
        let vm = self.verify.as_mut().expect("verify list");
        vm.validated = true;
        vm.pending = Some(batch.to_vec());
        Ok((toks, logits))
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        let rows = self
            .verify
            .as_ref()
            .and_then(|v| v.pending.as_ref().map(|b| b.len()))
            .ok_or_else(|| anyhow::anyhow!("advance() without a pending verify_chain"))?;
        anyhow::ensure!(
            n <= rows,
            "advance {n} beyond the {rows} rows verify_chain read back"
        );
        let batch = self
            .verify
            .as_mut()
            .and_then(|v| v.pending.take())
            .expect("pending chain");
        if n == self.verify_max_rows() && n == batch.len() {
            self.pos += n;
            return Ok(());
        }
        self.verify_rollback()?;
        for &t in &batch[..n] {
            self.prefill_step(t)?;
        }
        Ok(())
    }

    fn verify_rollback(&mut self) -> Result<()> {
        let v = self.verify.as_ref().expect("verify list");
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for ((dst, bytes), src) in self.recurrent_states.iter().zip(&v.rollback) {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, *bytes);
        }
        self.ctx.queue.submit([enc.finish()]);
        Ok(())
    }

    fn prefill_step_embed_row(&mut self, row_words: &[u32]) -> Result<()> {
        let hidden_words = self.config.hidden_size / 2;
        anyhow::ensure!(
            self.pos < self.max_seq,
            "kv cache full at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        anyhow::ensure!(
            row_words.len() == hidden_words,
            "spliced embed row has {} words, want {hidden_words}",
            row_words.len()
        );
        self.ctx
            .queue
            .write_buffer(&self.res, 0, bytemuck::cast_slice(row_words));
        self.ctx
            .queue
            .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(self.pos as i32)));
        let mut fd = self.fd_base;
        fd.total = (self.pos + 1) as u32;
        self.ctx
            .queue
            .write_buffer(&self.fd_buf, 0, bytemuck::bytes_of(&fd));
        let scope = (!self.splice_validated).then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.passes[self.embed_gather_end..self.head_start] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_moe_wgpu splice prefill step validation: {e}");
            }
            self.splice_validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    fn prefill_chunk_m_rows_spliced(
        &mut self,
        chunk: &[u32],
        rows: &[(usize, &[u32])],
    ) -> Result<()> {
        let e = self.pfm.as_ref().expect("M-row spliced chunk without the pf list");
        let m = e.m;
        let hidden_words = self.config.hidden_size / 2;
        anyhow::ensure!(
            chunk.len() == m,
            "the M-row list is baked for chunks of exactly {m} tokens, got {}",
            chunk.len()
        );
        anyhow::ensure!(
            self.pos + m <= self.max_seq,
            "kv cache full at {} + {m} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in chunk {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        self.pfm_write_host_inputs(e, chunk);
        let scope = (!self.splice_validated).then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.pfm_passes[..self.pfm_gather_end] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        for (row_start, words) in rows {
            anyhow::ensure!(
                words.len() % hidden_words == 0 && row_start + words.len() / hidden_words <= m,
                "spliced segment at chunk row {row_start} ({} words) overflows the {m}-row chunk",
                words.len()
            );
            self.ctx.queue.write_buffer(
                &e.res,
                (row_start * hidden_words * 4) as u64,
                bytemuck::cast_slice(words),
            );
        }
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.pfm_passes[self.pfm_gather_end..] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(err) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_moe_wgpu M-row spliced chunk validation: {err}");
            }
            self.splice_validated = true;
        }
        self.pos += m;
        Ok(())
    }

    pub fn prefill_with_splices(
        &mut self,
        tokens: &[u32],
        splices: &[EmbedRowsSplice],
    ) -> Result<u32> {
        if splices.is_empty() {
            return self.prefill(tokens);
        }
        anyhow::ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let hidden = self.config.hidden_size;
        let hidden_words = hidden / 2;
        let last_plain = tokens.len() - 1;
        let mut cursor = 0usize;
        for (k, sp) in splices.iter().enumerate() {
            anyhow::ensure!(
                !sp.rows_bf16.is_empty() && sp.rows_bf16.len() % hidden == 0,
                "splice {k} rows_bf16 length {} is not a positive multiple of hidden_size {hidden}",
                sp.rows_bf16.len()
            );
            let n_rows = sp.rows_bf16.len() / hidden;
            anyhow::ensure!(
                sp.position >= cursor,
                "splices must be strictly sorted and non-overlapping; splice {k} starts at {} but \
                 the previous splice ends at {cursor}",
                sp.position
            );
            anyhow::ensure!(
                sp.position + n_rows <= last_plain,
                "splice {k} covers tokens [{}, {}) but the final token at {last_plain} must be a \
                 plain token: it is decoded via decode_step, which replays the embed gathers",
                sp.position,
                sp.position + n_rows
            );
            cursor = sp.position + n_rows;
        }
        let packed: Vec<Vec<u32>> = splices.iter().map(|sp| pack_pairs(&sp.rows_bf16)).collect();
        let (last, rest) = tokens.split_last().expect("non-empty");
        let m = self.pfm.as_ref().map(|e| e.m);
        let mut i = 0usize;
        while i < rest.len() {
            if let Some(mm) = m {
                if rest.len() - i >= mm && self.pos + mm <= self.max_seq {
                    let mut chunk_rows: Vec<(usize, &[u32])> = Vec::new();
                    for (si, sp) in splices.iter().enumerate() {
                        let n_rows = sp.rows_bf16.len() / hidden;
                        let a = sp.position.max(i);
                        let b = (sp.position + n_rows).min(i + mm);
                        if a < b {
                            let off = a - sp.position;
                            let cnt = b - a;
                            chunk_rows.push((
                                a - i,
                                &packed[si][off * hidden_words..(off + cnt) * hidden_words],
                            ));
                        }
                    }
                    if chunk_rows.is_empty() {
                        self.prefill_chunk_m_row_projections(&rest[i..i + mm])?;
                    } else {
                        self.prefill_chunk_m_rows_spliced(&rest[i..i + mm], &chunk_rows)?;
                    }
                    i += mm;
                    continue;
                }
            }
            let inside = splices.iter().enumerate().find_map(|(si, sp)| {
                let n_rows = sp.rows_bf16.len() / hidden;
                (sp.position <= i && i < sp.position + n_rows).then(|| (si, i - sp.position))
            });
            if let Some((si, row)) = inside {
                self.prefill_step_embed_row(
                    &packed[si][row * hidden_words..(row + 1) * hidden_words],
                )?;
                i += 1;
            } else {
                let next = splices
                    .iter()
                    .map(|sp| sp.position)
                    .filter(|&p| p > i)
                    .min()
                    .unwrap_or(rest.len());
                let seg = &rest[i..next];
                let done = self.prefill_tokens(seg)?;
                for t in &seg[done..] {
                    self.prefill_step(*t)?;
                }
                i = next;
            }
        }
        self.decode_step(*last)
    }
}

fn row_chunk(ctx: &WgpuContext, hidden: usize) -> usize {
    let limit = ctx
        .caps
        .max_storage_buffer_binding_size
        .clamp(1 << 20, 1u64 << 30);
    let per_row = (hidden * 2) as u64;
    ((limit / per_row) as usize).max(2) & !1usize
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
    y_off_words: usize,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "q3w-gemvb-p",
        GemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_off_words: 0,
            y_off_words: y_off_words as u32,
            alpha: 1.0,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.gemv_bf16,
        w.entry,
        &[(0, &w.w), (1, x), (2, &p), (3, y)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_quant_rows(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    packed: &wgpu::Buffer,
    scales: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    globals: &wgpu::Buffer,
    k: usize,
    slots: usize,
    use_sel: bool,
    x_slot_stride_elems: usize,
) -> Result<()> {
    let k_blocks = k / NVFP4_BLOCK;
    anyhow::ensure!(
        k_blocks.is_multiple_of(4),
        "quant rows needs k/{NVFP4_BLOCK} divisible by 4, got {k_blocks}"
    );
    let p = b.uni(
        "q3w-quant-p",
        QuantRowsParams {
            k_blocks: k_blocks as u32,
            n_slots: slots as u32,
            use_sel: u32::from(use_sel),
            x_slot_stride_elems: x_slot_stride_elems as u32,
        },
    );
    let (entry, wg_blocks) = match quant_lane_entry(b.ctx, "q3w_quant_rows") {
        Some((e, wg)) => (e, wg / 8),
        None => ("q3w_quant_rows", 256),
    };
    b.quant_lane.0 += usize::from(entry != "q3w_quant_rows");
    b.quant_lane.1 += 1;
    let gx = (k_blocks as u32).div_ceil(wg_blocks as u32).max(1);
    b.push(
        label,
        &s.quant,
        entry,
        &[
            (10, x),
            (11, &p),
            (12, packed),
            (13, scales),
            (14, sel),
            (15, globals),
        ],
        (gx, slots as u32, 1),
    )
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Q8eParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    x_slot_stride_elems: u32,
    w_e_stride_words: u32,
    y_slot_stride_words: u32,
    use_sel: u32,
    groups_per_row: u32,
    group_shift: u32,
    y_off_words: u32,
    pad1: u32,
    pad2: u32,
}

fn fuse_silu_quant_enabled() -> bool {
    std::env::var("NV_Q3_WGPU_FUSE_SILU_QUANT").ok().as_deref() != Some("0")
}

pub fn quant_lane_mode() -> Option<usize> {
    match std::env::var("NV_Q3_WGPU_QUANT_LANE").ok().as_deref() {
        Some("0") | Some("off") | Some("false") => None,
        Some("w256") | Some("256") => Some(256),
        _ => Some(32),
    }
}

pub(crate) fn quant_lane_entry(ctx: &WgpuContext, base: &str) -> Option<(&'static str, usize)> {
    let wg = quant_lane_mode()?;
    if !wk::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        return None;
    }
    quant_lane_entries()
        .into_iter()
        .find(|(_, r#ref, w)| *r#ref == base && *w == wg)
        .map(|(e, _, w)| (e, w))
}

fn off(var: &str) -> bool {
    matches!(
        std::env::var(var).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn delta_in_proj_fused() -> bool {
    !off("NV_Q3_WGPU_FUSE_DN_INPROJ")
}

pub fn delta_gate_fused() -> bool {
    !off("NV_Q3_WGPU_FUSE_DN_GATE")
}

pub fn attn_qcast_fused() -> bool {
    !off("NV_Q3_WGPU_FUSE_AT_QCAST")
}

pub fn attn_kv_fused() -> bool {
    !off("NV_Q3_WGPU_FUSE_AT_KV")
}

pub fn attn_qknorm_fused() -> bool {
    !off("NV_Q3_WGPU_FUSE_AT_QKNORM")
}

pub fn router_gate_fused() -> bool {
    !off("NV_Q3_WGPU_FUSE_ROUTER_GATE")
}

pub fn shared_expert_folded() -> bool {
    !off("NV_Q3_WGPU_FUSE_SHARED_EXPERT")
}

#[allow(clippy::too_many_arguments)]
fn push_silu_mul_quant(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    y_gate: &wgpu::Buffer,
    y_up: &wgpu::Buffer,
    packed: &wgpu::Buffer,
    scales: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    globals: &wgpu::Buffer,
    k: usize,
    slots: usize,
    use_sel: bool,
    x_per_slot: bool,
    fused_pair: bool,
) -> Result<()> {
    let k_blocks = k / NVFP4_BLOCK;
    anyhow::ensure!(
        k_blocks.is_multiple_of(4),
        "silu_mul_quant needs k/{NVFP4_BLOCK} divisible by 4, got {k_blocks}"
    );
    anyhow::ensure!(
        !fused_pair || k.is_multiple_of(2),
        "silu_mul_quant fused pair needs an even row count, got {k}"
    );
    let p = b.uni(
        "q3w-siluq-p",
        QuantRowsParams {
            k_blocks: k_blocks as u32,
            n_slots: slots as u32,
            use_sel: u32::from(use_sel),
            x_slot_stride_elems: match (x_per_slot, fused_pair) {
                (true, true) => 2 * k as u32,
                (true, false) => k as u32,
                (false, _) => 0,
            },
        },
    );
    let pp = b.uni(
        "q3w-siluq-pair",
        SiluPairParams {
            u_off_elems: if fused_pair { k as u32 } else { 0 },
            ..Default::default()
        },
    );
    let (entry, wg_blocks) = match quant_lane_entry(b.ctx, "q3w_silu_mul_quant") {
        Some((e, wg)) => (e, wg / 8),
        None => ("q3w_silu_mul_quant", 256),
    };
    b.quant_lane.0 += usize::from(entry != "q3w_silu_mul_quant");
    b.quant_lane.1 += 1;
    let gx = (k_blocks as u32).div_ceil(wg_blocks as u32).max(1);
    b.push(
        label,
        &s.quant,
        entry,
        &[
            (11, &p),
            (12, packed),
            (13, scales),
            (14, sel),
            (15, globals),
            (16, y_gate),
            (17, y_up),
            (18, &pp),
        ],
        (gx, slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_nvfp4(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &wgpu::Buffer,
    ws: &wgpu::Buffer,
    x: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    alphas: &wgpu::Buffer,
    n: usize,
    k: usize,
    slots: usize,
    y_slot_stride_words: usize,
    per_expert: bool,
    alpha: f32,
) -> Result<()> {
    let k_blocks = k / NVFP4_BLOCK;
    let route = nvfp4_v2_route(b.ctx, n, k, k_blocks, slots);
    b.nvfp4_projs += 1;
    b.nvfp4_v2_routed += usize::from(route.is_some());
    let grid = match &route {
        Some(r) => b.grid1(n as u64, r.rows_per_group),
        None => b.grid1(n.div_ceil(2) as u64, 1),
    };
    let p = b.uni(
        "q3w-gemv4-p",
        GemvNvfp4Params {
            alpha,
            n_rows: n as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: if per_expert { (n * k_blocks) as u32 } else { 0 },
            sf_e_stride_bytes: if per_expert {
                wk::gemv_nvfp4::swizzled_scale_len(n, k_blocks) as u32
            } else {
                0
            },
            x_slot_stride_vec2: if slots > 1 { k_blocks as u32 } else { 0 },
            xsf_slot_stride_bytes: if slots > 1 { k_blocks as u32 } else { 0 },
            y_slot_stride_words: y_slot_stride_words as u32,
            per_expert_alpha: u32::from(per_expert),
            m_slots_sharing_expert_zero: 0,
        },
    );
    if let Some(r) = route {
        let (w_slot, x_slot) = if r.vec4 { (18, 19) } else { (10, 12) };
        return b.push(
            label,
            &r.source,
            r.entry,
            &[
                (w_slot, w),
                (11, ws),
                (x_slot, x),
                (13, xs),
                (14, &p),
                (15, y),
                (16, sel),
                (17, alphas),
            ],
            (grid.0, grid.1, slots as u32),
        );
    }
    b.push(
        label,
        &s.gemv_nvfp4,
        "q3w_gemv_nvfp4",
        &[
            (10, w),
            (11, ws),
            (12, x),
            (13, xs),
            (14, &p),
            (15, y),
            (16, sel),
            (17, alphas),
        ],
        (grid.0, grid.1, slots as u32),
    )
}

pub struct Nvfp4V2Route {
    pub source: String,
    pub entry: &'static str,
    pub rows_per_group: u32,
    pub vec4: bool,
}

pub const NVFP4_V2_DEFAULT_ON: bool = true;

pub fn nvfp4_v2_enabled(ctx: &WgpuContext) -> bool {
    if !wk::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        return false;
    }
    match std::env::var("NV_Q3_WGPU_NVFP4_V2") {
        Ok(v) => v != "0",
        Err(_) => NVFP4_V2_DEFAULT_ON,
    }
}

pub fn nvfp4_v2_boot_line(requested: bool, routed: usize, total: usize) -> Option<String> {
    if !requested {
        return None;
    }
    Some(if routed == 0 {
        format!("[q3w-wgpu] nvfp4 v2 requested but 0 of {total} nvfp4 GEMVs routed to it")
    } else {
        format!("[q3w-wgpu] nvfp4 v2 engaged on {routed} of {total} nvfp4 GEMVs")
    })
}

pub fn nvfp4_v2_route(
    ctx: &WgpuContext,
    n: usize,
    k: usize,
    k_blocks: usize,
    slots: usize,
) -> Option<Nvfp4V2Route> {
    if !nvfp4_v2_enabled(ctx) {
        return None;
    }
    if !k_blocks.is_multiple_of(4) || !(n * k_blocks).is_multiple_of(2) || n < 2 {
        return None;
    }
    if slots == 0 {
        return None;
    }
    let (kernel, cfg, pk_entry) = wk::gemv_nvfp4_v2::select_pk_slots(n, k, slots)?;
    let (kernel, pk_entry) = mrow2_upgrade(kernel, cfg, pk_entry, k);
    let entry = match pk_entry {
        wk::gemv_nvfp4_v2::FMLUT_PK_ENTRY => "q3w_gemv_nvfp4_fmlut",
        wk::gemv_nvfp4_v2::FDEC_PK_ENTRY => "q3w_gemv_nvfp4_fdec",
        wk::gemv_nvfp4_v2::WARP_PK_ENTRY => "q3w_gemv_nvfp4_warp",
        wk::gemv_nvfp4_v2::MROW_PK_ENTRY => "q3w_gemv_nvfp4_mrow",
        wk::gemv_nvfp4_v2::MROW2_PK_ENTRY => "q3w_gemv_nvfp4_mrow2",
        _ => return None,
    };
    let rows_per_group = cfg.rows_per_group(kernel);
    let source = compose(&format!(
        "{}\n{}",
        wk::gemv_nvfp4_v2::helpers(cfg),
        GEMV_NVFP4_V2_WGSL
    ));
    Some(Nvfp4V2Route {
        source,
        entry,
        rows_per_group,
        vec4: kernel.vec4_slots(),
    })
}

pub const NVFP4_SLOTSHARED_MAX_SLOTS_ONE_ACC_REGISTER_PER_SLOT: usize = 16;

pub const SLOTSHARED_ENTRY: &str = "q3w_gemv_nvfp4_slotshared";

pub const SLOTSHARED_IS_ONLY_CORRECT_WHEN_EVERY_SLOT_SELECTS_EXPERT_ZERO: &str =
    "q3w_gemv_nvfp4_slotshared reads each weight vec4 once and loops the chunk slots in \
     registers, so an M-slot prefill chunk pays one weight sweep instead of M z-replicated \
     sweeps; it reads q34_sel[0] for every slot, so a caller whose slots select different \
     experts (the MoE stacks) must keep the z-replicated route.";

pub fn nvfp4_slotshared_enabled() -> bool {
    std::env::var("NV_NVFP4_SLOTSHARED").ok().as_deref() != Some("0")
}

pub const SLOTSHARED_MAP_ENTRY: &str = "q3w_i8map_x_rows";

pub fn nvfp4_slotshared_sources() -> String {
    let cfg = wk::gemv_nvfp4_v2::V2Config::new(256, 1);
    compose(&format!(
        "{}\n{}",
        wk::gemv_nvfp4_v2::helpers(cfg),
        GEMV_NVFP4_V2_WGSL
    ))
}

#[doc(hidden)]
pub fn nvfp4_v2_mrow2_source_cfg128x2_matching_the_dense_decode_route() -> String {
    let cfg = wk::gemv_nvfp4_v2::V2Config::new(128, 2);
    compose(&format!(
        "{}\n{}",
        wk::gemv_nvfp4_v2::helpers(cfg),
        GEMV_NVFP4_V2_WGSL
    ))
}

pub fn nvfp4_v2_route_slotshared(
    ctx: &WgpuContext,
    n: usize,
    k: usize,
    k_blocks: usize,
    slots: usize,
) -> Option<Nvfp4V2Route> {
    if !nvfp4_v2_enabled(ctx) || !nvfp4_slotshared_enabled() {
        return None;
    }
    if !(2..=NVFP4_SLOTSHARED_MAX_SLOTS_ONE_ACC_REGISTER_PER_SLOT).contains(&slots) {
        return None;
    }
    if !k.is_multiple_of(NVFP4_BLOCK) || k_blocks == 0 || n < 2 {
        return None;
    }
    let cfg = wk::gemv_nvfp4_v2::V2Config::new(256, 1);
    if !wk::gemv_nvfp4_v2::subgroup32_ok(ctx) || !cfg.subgroups().is_multiple_of(2) {
        return None;
    }
    let source = compose(&format!(
        "{}\n{}",
        wk::gemv_nvfp4_v2::helpers(cfg),
        GEMV_NVFP4_V2_WGSL
    ));
    Some(Nvfp4V2Route {
        source,
        entry: SLOTSHARED_ENTRY,
        rows_per_group: cfg.subgroups(),
        vec4: false,
    })
}

pub const MROW2_UPGRADE_BEATS_MROW_ON_EVERY_DENSE_SINGLE_SLOT_SHAPE_MEASURED: &str =
    "interleaved route A/B (dense_single_slot_route_ab): scalar two-row mrow2 out-streams \
     mrow(128,2) on every dense single-slot projection shape measured, both orientations, \
     bit-identical outputs on every arm; select_slots only returns MRow at slots<=1 so this \
     upgrade is scoped to exactly the measured regime; current numbers: perf/runs.jsonl. \
     NV_Q3_WGPU_NVFP4_MROW2=0 restores mrow.";

fn mrow2_upgrade(
    kernel: wk::gemv_nvfp4_v2::V2Kernel,
    cfg: wk::gemv_nvfp4_v2::V2Config,
    pk_entry: &'static str,
    k: usize,
) -> (wk::gemv_nvfp4_v2::V2Kernel, &'static str) {
    let mrow2 = wk::gemv_nvfp4_v2::V2Kernel::MRow2;
    if pk_entry != wk::gemv_nvfp4_v2::MROW_PK_ENTRY
        || std::env::var("NV_Q3_WGPU_NVFP4_MROW2").ok().as_deref() == Some("0")
        || !mrow2.shape_ok(k)
        || !wk::gemv_nvfp4_v2::pk_capable(mrow2, cfg)
    {
        return (kernel, pk_entry);
    }
    (mrow2, wk::gemv_nvfp4_v2::MROW2_PK_ENTRY)
}

pub fn gemv_bf16_load() -> &'static str {
    match std::env::var("NV_Q3_WGPU_GEMV_BF16_LOAD").ok().as_deref() {
        Some("1") | Some("scalar") | Some("off") => "q3w_gemv_bf16",
        Some("u4") => "q3w_gemv_bf16_u4",
        _ => "q3w_gemv_bf16_u8",
    }
}

fn upload_bf16(b: &mut Builder, label: &str, l: &HostBf16Lin) -> Bf16Gpu {
    let entry = gemv_bf16_load();
    b.gemv_unrolled.0 += usize::from(entry != "q3w_gemv_bf16");
    b.gemv_unrolled.1 += 1;
    Bf16Gpu {
        w: b.upload_u32(label, &pack_pairs(&l.w)),
        n: l.n,
        k: l.k,
        entry,
    }
}

fn nvfp4_rowcat_ok(ctx: &WgpuContext, lo: &HostNvfp4Lin, hi: &HostNvfp4Lin) -> bool {
    if lo.n != hi.n || lo.k != hi.k || !lo.n.is_multiple_of(128) {
        return false;
    }
    if lo.alpha != hi.alpha || lo.input_global != hi.input_global {
        return false;
    }
    if lo.packed.len() != hi.packed.len()
        || lo.scales_swizzled.len() != hi.scales_swizzled.len()
        || !lo.packed.len().is_multiple_of(8)
        || !lo.scales_swizzled.len().is_multiple_of(8)
    {
        return false;
    }
    let kb = lo.k / NVFP4_BLOCK;
    let shape =
        |n: usize| nvfp4_v2_route(ctx, n, lo.k, kb, 1).map(|r| (r.entry, r.rows_per_group, r.vec4));
    shape(lo.n) == shape(2 * lo.n)
}

fn upload_nvfp4_rowcat(
    b: &mut Builder,
    label: &str,
    lo: &HostNvfp4Lin,
    hi: &HostNvfp4Lin,
) -> Nvfp4Gpu {
    let cat = |a: &[u8], c: &[u8]| {
        let mut v: Vec<u32> = Vec::with_capacity((a.len() + c.len()) / 4);
        v.extend_from_slice(&bytes_to_words(a));
        v.extend_from_slice(&bytes_to_words(c));
        v
    };
    Nvfp4Gpu {
        w: b.upload_u32(label, &cat(&lo.packed, &hi.packed)),
        scales: b.upload_u32(
            &format!("{label}-sf"),
            &cat(&lo.scales_swizzled, &hi.scales_swizzled),
        ),
        alpha: lo.alpha,
        input_global: lo.input_global,
        n: lo.n + hi.n,
        k: lo.k,
    }
}

pub fn router_par_enabled() -> bool {
    !matches!(
        std::env::var("NV_Q3_WGPU_ROUTER_PAR").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn router_topk_entry() -> (&'static str, &'static str) {
    match std::env::var("NV_Q3_WGPU_ROUTER_TOPK").ok().as_deref() {
        Some("tree") => ("q3w-moe-topk-tree", "q3w_router_topk_tree"),
        Some("r4") => ("q3w-moe-topk-r4", "q3w_router_topk_r4"),
        Some("r16") => ("q3w-moe-topk-r16", "q3w_router_topk_r16"),
        _ => ("q3w-moe-topk-par", "q3w_router_topk_par"),
    }
}

fn w8_group() -> usize {
    crate::nvfp4_host::w8_group_from_env("NV_Q3_WGPU_W8_GROUP")
}

fn w8_experts_enabled(ctx: &WgpuContext) -> bool {
    std::env::var("NV_Q3_WGPU_W8_EXPERTS").ok().as_deref() == Some("1")
        && wk::gemv_nvfp4_v2::subgroup32_ok(ctx)
}

fn w8_proj_group() -> usize {
    let g = std::env::var("NV_Q3_WGPU_W8_PROJ_GROUP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(128);
    assert!(
        g >= 32 && g.is_power_of_two(),
        "NV_Q3_WGPU_W8_PROJ_GROUP must be a power of two >= 32; got {g}. Per-row \
         (g=0) is deliberately NOT offered here: it measured materially worse weight \
         rms than the control on these exact tensors (numbers: perf/runs.jsonl)."
    );
    g
}

fn w8_delta_proj_enabled(ctx: &WgpuContext) -> (bool, bool) {
    if !wk::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        return (false, false);
    }
    match std::env::var("NV_Q3_WGPU_W8_DELTA").ok().as_deref() {
        Some("1") | Some("all") => (true, true),
        Some("out") => (false, true),
        Some("in") => (true, false),
        _ => (false, false),
    }
}

fn w8_lmhead_enabled(ctx: &WgpuContext) -> bool {
    std::env::var("NV_Q3_WGPU_W8_LMHEAD").ok().as_deref() != Some("0")
        && wk::gemv_nvfp4_v2::subgroup32_ok(ctx)
}

struct ExpertI8Gpu {
    w: wgpu::Buffer,
    s: wgpu::Buffer,
    n: usize,
    k: usize,
    group: usize,
}

fn quantize_bf16_i8(w: &[u16], n: usize, k: usize, group: usize) -> (Vec<u32>, Vec<f32>) {
    assert!(
        group > 0 && k.is_multiple_of(group) && k.is_multiple_of(4),
        "quantize_bf16_i8: k={k} must be a multiple of group={group} and of 4"
    );
    assert_eq!(w.len(), n * k, "quantize_bf16_i8: weight length mismatch");
    let gpr = k / group;
    let mut packed = vec![0u32; n * k / 4];
    let mut scales = vec![0f32; n * gpr];
    for r in 0..n {
        let row = &w[r * k..(r + 1) * k];
        for g in 0..gpr {
            let lo = g * group;
            let mut max_abs = 0f32;
            for &b in &row[lo..lo + group] {
                max_abs = max_abs.max(bf16_val(b).abs());
            }
            let sc = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[r * gpr + g] = sc;
            for (i, &b) in row[lo..lo + group].iter().enumerate() {
                let idx = r * k + lo + i;
                let q = (bf16_val(b) / sc).round().clamp(-127.0, 127.0) as i32 as u32 & 0xff;
                packed[idx / 4] |= q << (8 * (idx % 4));
            }
        }
    }
    (packed, scales)
}

fn upload_bf16_as_i8(b: &mut Builder, label: &str, l: &HostBf16Lin, group: usize) -> ExpertI8Gpu {
    let (packed, scales) = quantize_bf16_i8(&l.w, l.n, l.k, group);
    b.w8_proj.0 += 1;
    b.w8_proj.1 += (l.n * l.k * 2) as u64;
    b.w8_proj.2 += (packed.len() * 4 + scales.len() * 4) as u64;
    ExpertI8Gpu {
        w: b.upload_u32(label, &packed),
        s: b.upload_f32(&format!("{label}-s"), &scales),
        n: l.n,
        k: l.k,
        group,
    }
}

fn upload_experts_i8(b: &mut Builder, label: &str, st: &HostExpertStack) -> ExpertI8Gpu {
    let group = w8_group();
    assert_w8_group_divides_k(label, group, st.k, "NV_Q3_WGPU_W8_GROUP");
    let (packed, scales) = crate::nvfp4_host::quantize_nvfp4_stack_i8(st, group);
    ExpertI8Gpu {
        w: b.upload_u32(label, &packed),
        s: b.upload_f32(&format!("{label}-s"), &scales),
        n: st.n,
        k: st.k,
        group,
    }
}

fn upload_nvfp4_i8(b: &mut Builder, label: &str, l: &HostNvfp4Lin) -> ExpertI8Gpu {
    let st = HostExpertStack {
        packed: l.packed.clone(),
        scales_swizzled: l.scales_swizzled.clone(),
        alphas: vec![l.alpha],
        input_globals: vec![l.input_global],
        e: 1,
        n: l.n,
        k: l.k,
    };
    upload_experts_i8(b, label, &st)
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_i8_experts(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    ex: &ExpertI8Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    use_sel: bool,
    x_per_slot: bool,
) -> Result<()> {
    anyhow::ensure!(
        ex.k.is_multiple_of(4),
        "i8 experts need k % 4 == 0, got {}",
        ex.k
    );
    let groups = (ex.n.div_ceil(8)) as u32;
    let p = b.uni(
        "q3w-i8e-p",
        Q8eParams {
            n_rows: ex.n as u32,
            k_elems: ex.k as u32,
            groups_x: groups,
            x_slot_stride_elems: if x_per_slot { ex.k as u32 } else { 0 },
            w_e_stride_words: (ex.n * ex.k / 4) as u32,
            y_slot_stride_words: (ex.n.div_ceil(2)) as u32,
            use_sel: u32::from(use_sel),
            groups_per_row: ex.k.checked_div(ex.group).unwrap_or(1) as u32,
            group_shift: if ex.group > 0 {
                (ex.group / 4).trailing_zeros()
            } else {
                0
            },
            y_off_words: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    let entry = if ex.group > 0 {
        "q3w_gemv_i8g_experts"
    } else {
        "q3w_gemv_i8_experts"
    };
    b.push(
        label,
        &s.i8e,
        entry,
        &[(0, &ex.w), (1, &ex.s), (2, x), (3, y), (4, &p), (5, sel)],
        (groups, 1, slots as u32),
    )
}

fn push_gemv_i8_f32(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    ex: &ExpertI8Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    y_off_words: usize,
) -> Result<()> {
    anyhow::ensure!(ex.group > 0, "the f32 int8 GEMV is group-scaled only");
    let groups = ex.n.div_ceil(8) as u32;
    anyhow::ensure!(
        groups <= b.ctx.caps.max_compute_workgroups_per_dimension,
        "lm_head int8 chunk needs {groups} workgroups, over the {} per-dimension \
         limit; lower the row chunk",
        b.ctx.caps.max_compute_workgroups_per_dimension
    );
    let p = b.uni(
        "q3w-i8f-p",
        Q8eParams {
            n_rows: ex.n as u32,
            k_elems: ex.k as u32,
            groups_x: groups,
            x_slot_stride_elems: 0,
            w_e_stride_words: (ex.n * ex.k / 4) as u32,
            y_slot_stride_words: 0,
            use_sel: 0,
            groups_per_row: (ex.k / ex.group) as u32,
            group_shift: (ex.group / 4).trailing_zeros(),
            y_off_words: y_off_words as u32,
            pad1: 0,
            pad2: 0,
        },
    );
    b.push(
        label,
        &s.i8e,
        "q3w_gemv_i8g_f32",
        &[(0, &ex.w), (1, &ex.s), (2, x), (3, y), (4, &p)],
        (groups, 1, 1),
    )
}

fn gate_up_fused() -> bool {
    !matches!(
        std::env::var("NV_Q3_WGPU_FUSE_GATEUP").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

fn upload_experts_gate_up(
    b: &mut Builder,
    label: &str,
    g: &HostExpertStack,
    u: &HostExpertStack,
    sh: Option<(&HostNvfp4Lin, &HostNvfp4Lin)>,
) -> Result<ExpertGpu> {
    anyhow::ensure!(
        g.e == u.e && g.n == u.n && g.k == u.k,
        "{label}: gate {}x{}x{} vs up {}x{}x{}",
        g.e,
        g.n,
        g.k,
        u.e,
        u.n,
        u.k
    );
    anyhow::ensure!(
        g.n.is_multiple_of(128),
        "{label}: n={} is not a multiple of the 128-row scale-swizzle tile, so the \
         concatenated scales would not be the swizzle of the concatenated rows",
        g.n
    );
    anyhow::ensure!(
        g.alphas == u.alphas && g.input_globals == u.input_globals,
        "{label}: gate and up disagree on weight_global_scale or input_global_scale, \
         so one GEMV cannot carry both"
    );
    let ws = g.packed.len() / g.e;
    let ss = g.scales_swizzled.len() / g.e;
    anyhow::ensure!(
        g.packed.len().is_multiple_of(g.e)
            && u.packed.len() == g.packed.len()
            && g.scales_swizzled.len().is_multiple_of(g.e)
            && u.scales_swizzled.len() == g.scales_swizzled.len()
            && ws.is_multiple_of(8)
            && ss.is_multiple_of(8),
        "{label}: per-expert strides {ws}/{ss} must be whole u32 pairs and equal across the pair"
    );
    let (mut w, mut sf) = (
        Vec::<u32>::with_capacity((g.packed.len() + u.packed.len()) / 4),
        Vec::<u32>::with_capacity((g.scales_swizzled.len() + u.scales_swizzled.len()) / 4),
    );
    for e in 0..g.e {
        w.extend_from_slice(&bytes_to_words(&g.packed[e * ws..(e + 1) * ws]));
        w.extend_from_slice(&bytes_to_words(&u.packed[e * ws..(e + 1) * ws]));
        sf.extend_from_slice(&bytes_to_words(&g.scales_swizzled[e * ss..(e + 1) * ss]));
        sf.extend_from_slice(&bytes_to_words(&u.scales_swizzled[e * ss..(e + 1) * ss]));
    }
    let mut alphas = g.alphas.clone();
    let mut globals = g.input_globals.clone();
    if let Some((sg, su)) = sh {
        anyhow::ensure!(
            sg.n == g.n && su.n == g.n && sg.k == g.k && su.k == g.k,
            "{label}: shared pair is {}x{}/{}x{}, stack is {}x{}",
            sg.n,
            sg.k,
            su.n,
            su.k,
            g.n,
            g.k
        );
        anyhow::ensure!(
            sg.packed.len() == ws
                && su.packed.len() == ws
                && sg.scales_swizzled.len() == ss
                && su.scales_swizzled.len() == ss,
            "{label}: shared pair does not occupy exactly one expert stride"
        );
        anyhow::ensure!(
            sg.alpha == su.alpha && sg.input_global == su.input_global,
            "{label}: shared gate and up disagree on their global scales"
        );
        w.extend_from_slice(&bytes_to_words(&sg.packed));
        w.extend_from_slice(&bytes_to_words(&su.packed));
        sf.extend_from_slice(&bytes_to_words(&sg.scales_swizzled));
        sf.extend_from_slice(&bytes_to_words(&su.scales_swizzled));
        alphas.push(sg.alpha);
        globals.push(sg.input_global);
    }
    Ok(ExpertGpu {
        w: b.upload_u32(label, &w),
        scales: b.upload_u32(&format!("{label}-sf"), &sf),
        alphas: b.upload_f32(&format!("{label}-a"), &alphas),
        globals: b.upload_f32(&format!("{label}-g"), &globals),
        n: 2 * g.n,
        k: g.k,
    })
}

fn upload_experts(b: &mut Builder, label: &str, st: &HostExpertStack) -> ExpertGpu {
    ExpertGpu {
        w: b.upload_u32(label, &bytes_to_words(&st.packed)),
        scales: b.upload_u32(&format!("{label}-sf"), &bytes_to_words(&st.scales_swizzled)),
        alphas: b.upload_f32(&format!("{label}-a"), &st.alphas),
        globals: b.upload_f32(&format!("{label}-g"), &st.input_globals),
        n: st.n,
        k: st.k,
    }
}

fn upload_experts_with(
    b: &mut Builder,
    label: &str,
    st: &HostExpertStack,
    sh: Option<&HostNvfp4Lin>,
) -> Result<ExpertGpu> {
    let Some(sh) = sh else {
        return Ok(upload_experts(b, label, st));
    };
    let w_stride = st.packed.len() / st.e;
    let sf_stride = st.scales_swizzled.len() / st.e;
    anyhow::ensure!(
        sh.n == st.n && sh.k == st.k,
        "{label}: shared expert is {}x{}, stack is {}x{}",
        sh.n,
        sh.k,
        st.n,
        st.k
    );
    anyhow::ensure!(
        st.packed.len().is_multiple_of(st.e) && sh.packed.len() == w_stride,
        "{label}: shared packed {} bytes, per-expert stride {w_stride}",
        sh.packed.len()
    );
    anyhow::ensure!(
        st.scales_swizzled.len().is_multiple_of(st.e) && sh.scales_swizzled.len() == sf_stride,
        "{label}: shared scales {} bytes, per-expert stride {sf_stride}",
        sh.scales_swizzled.len()
    );
    anyhow::ensure!(
        w_stride.is_multiple_of(8) && sf_stride.is_multiple_of(8),
        "{label}: strides {w_stride}/{sf_stride} must be whole u32 pairs for the \
         word-packed upload to concatenate exactly"
    );
    let cat = |a: &[u8], c: &[u8]| {
        let mut v: Vec<u32> = Vec::with_capacity((a.len() + c.len()) / 4);
        v.extend_from_slice(&bytes_to_words(a));
        v.extend_from_slice(&bytes_to_words(c));
        v
    };
    let w = cat(&st.packed, &sh.packed);
    let sf = cat(&st.scales_swizzled, &sh.scales_swizzled);
    let mut alphas = st.alphas.clone();
    alphas.push(sh.alpha);
    let mut globals = st.input_globals.clone();
    globals.push(sh.input_global);
    Ok(ExpertGpu {
        w: b.upload_u32(label, &w),
        scales: b.upload_u32(&format!("{label}-sf"), &sf),
        alphas: b.upload_f32(&format!("{label}-a"), &alphas),
        globals: b.upload_f32(&format!("{label}-g"), &globals),
        n: st.n,
        k: st.k,
    })
}

enum ProjGpu {
    Bf16(Bf16Gpu),

    I8(ExpertI8Gpu, wgpu::Buffer),
}

fn upload_proj(b: &mut Builder, label: &str, l: &HostBf16Lin, w8: Option<usize>) -> ProjGpu {
    match w8 {
        Some(g) if l.k.is_multiple_of(g) => {
            let sel = b.upload_u32(&format!("{label}-sel"), &[0u32]);
            ProjGpu::I8(upload_bf16_as_i8(b, label, l, g), sel)
        }
        _ => ProjGpu::Bf16(upload_bf16(b, label, l)),
    }
}

fn push_proj(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    p: &ProjGpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Result<()> {
    match p {
        ProjGpu::Bf16(w) => push_gemv_bf16(b, s, label, w, x, y, false, 0),
        ProjGpu::I8(ex, sel) => push_gemv_i8_experts(b, s, label, ex, x, y, sel, 1, false, false),
    }
}

#[allow(clippy::too_many_arguments)]
fn pf_push_gemm_bf16_m(
    b: &mut Builder,
    pf: &PfMrowBufs,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    y: &wgpu::Buffer,
    y_stride_words: usize,
    out_f32: bool,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "q3w-pf-gemm-p",
        PfGemmParams {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            w_row_words: (w.k / 2) as u32,
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
            out_f32: u32::from(out_f32),
            pad0: 0,
        },
    );
    let tile = pf.gemm_tile_rows;
    let mut done = 0usize;
    while done < pf.m {
        let rows = tile.min(pf.m - done);
        let (src, entry): (&str, &str) = if rows == tile {
            (&pf.gemm_src, &pf.gemm_entry)
        } else {
            let (tr, ts, te) = pf.gemm_tail.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "{label}: the {rows}-row GEMM tail has no generated kernel; \
                     pf_mrow_m and gemm_tail disagree"
                )
            })?;
            anyhow::ensure!(*tr == rows, "{label}: tail kernel is {tr} rows, want {rows}");
            (ts, te)
        };
        if done == 0 {
            b.push(label, src, entry, &[(0, &w.w), (1, x), (2, &p), (3, y)], grid)?;
        } else {
            let xb = (done * x_stride_words * 4) as u64;
            let yb = (done * y_stride_words * 4) as u64;
            b.push_pf_off_m_row(
                label,
                src,
                entry,
                &[(0, &w.w, 0), (1, x, xb), (2, &p, 0), (3, y, yb)],
                grid,
            )?;
        }
        done += rows;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pf_push_gemv_i8_slots(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    ex: &ExpertI8Gpu,
    sel_zeros: &wgpu::Buffer,
    x: &wgpu::Buffer,
    x_slot_stride_elems: usize,
    y: &wgpu::Buffer,
    y_slot_stride_words: usize,
    slots: usize,
) -> Result<()> {
    anyhow::ensure!(
        ex.k.is_multiple_of(4),
        "i8 experts need k % 4 == 0, got {}",
        ex.k
    );
    let groups = (ex.n.div_ceil(8)) as u32;
    let p = b.uni(
        "q3w-pf-i8e-p",
        Q8eParams {
            n_rows: ex.n as u32,
            k_elems: ex.k as u32,
            groups_x: groups,
            x_slot_stride_elems: x_slot_stride_elems as u32,
            w_e_stride_words: (ex.n * ex.k / 4) as u32,
            y_slot_stride_words: y_slot_stride_words as u32,
            use_sel: 1,
            groups_per_row: ex.k.checked_div(ex.group).unwrap_or(1) as u32,
            group_shift: if ex.group > 0 {
                (ex.group / 4).trailing_zeros()
            } else {
                0
            },
            y_off_words: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    let entry = if ex.group > 0 {
        "q3w_gemv_i8g_experts"
    } else {
        "q3w_gemv_i8_experts"
    };
    b.push(
        label,
        &s.i8e,
        entry,
        &[(0, &ex.w), (1, &ex.s), (2, x), (3, y), (4, &p), (5, sel_zeros)],
        (groups, 1, slots as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_push_proj_m(
    b: &mut Builder,
    s: &Sources,
    pf: &PfMrowBufs,
    label: &str,
    p: &ProjGpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    match p {
        ProjGpu::Bf16(w) => {
            pf_push_gemm_bf16_m(b, pf, label, w, x, x_stride_words, y, y_stride_words, false)
        }
        ProjGpu::I8(ex, _) => {
            let sel = pf.sel_zeros_m.clone();
            pf_push_gemv_i8_slots(
                b,
                s,
                label,
                ex,
                &sel,
                x,
                x_stride_words * 2,
                y,
                y_stride_words,
                pf.m,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_delta(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3MoeConfig,
    d: &HostDeltaNet,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    pf: &mut Option<PfMrowBufs>,
) -> Result<()> {
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let (w8_in, w8_out) = w8_delta_proj_enabled(b.ctx);
    let g = w8_proj_group();
    let (w8_in, w8_out) = (w8_in.then_some(g), w8_out.then_some(g));

    let w_out = upload_proj(b, "q3w-dn-out", &d.out_proj, w8_out);

    let fuse_in_proj = delta_in_proj_fused()
        && d.in_proj_qkv.k == d.in_proj_z.k
        && d.in_proj_qkv.k == d.in_proj_ab.k
        && conv_dim.is_multiple_of(2)
        && value_dim.is_multiple_of(2);
    let (qkv, z, ab, z_off, ab_off, w_in_all) = if fuse_in_proj {
        let k = d.in_proj_qkv.k;
        let n_all = d.in_proj_qkv.n + d.in_proj_z.n + d.in_proj_ab.n;
        let mut w = Vec::with_capacity(n_all * k);
        w.extend_from_slice(&d.in_proj_qkv.w);
        w.extend_from_slice(&d.in_proj_z.w);
        w.extend_from_slice(&d.in_proj_ab.w);
        let w_all = upload_proj(b, "q3w-dn-inproj", &HostBf16Lin { w, n: n_all, k }, w8_in);
        let buf = b.zeros("q3w-dn-inprojbuf", (n_all * 2) as u64);
        push_proj(b, s, "q3w-dn-inproj", &w_all, x, &buf)?;
        b.dn_in_proj.0 += 1;
        b.dn_in_proj.1 += 1;
        (
            buf.clone(),
            buf.clone(),
            buf,
            conv_dim,
            conv_dim + value_dim,
            Some(w_all),
        )
    } else {
        let w_qkv = upload_proj(b, "q3w-dn-qkv", &d.in_proj_qkv, w8_in);
        let w_z = upload_proj(b, "q3w-dn-z", &d.in_proj_z, w8_in);
        let w_ab = upload_proj(b, "q3w-dn-ab", &d.in_proj_ab, None);
        let qkv = b.zeros("q3w-dn-qkvbuf", (conv_dim * 2) as u64);
        let z = b.zeros("q3w-dn-zbuf", (value_dim * 2) as u64);
        let ab = b.zeros("q3w-dn-abbuf", (2 * n_v * 2).max(4) as u64);
        push_proj(b, s, "q3w-dn-qkv", &w_qkv, x, &qkv)?;
        push_proj(b, s, "q3w-dn-z", &w_z, x, &z)?;
        push_proj(b, s, "q3w-dn-ab", &w_ab, x, &ab)?;
        b.dn_in_proj.1 += 1;
        (qkv, z, ab, 0, 0, None)
    };

    let conv_w = b.upload_f32("q3w-dn-convw", &d.conv1d);
    let conv_state_bytes = (conv_dim * (ks - 1) * 4) as u64;
    let conv_state = b.zeros("q3w-dn-convstate", conv_state_bytes.max(4));
    states.push((conv_state.clone(), conv_state_bytes.max(4)));
    let mixed = b.zeros("q3w-dn-mixed", (conv_dim * 4) as u64);
    let cp = b.uni(
        "q3w-dn-conv-p",
        ConvParams {
            conv_dim: conv_dim as u32,
            kernel: ks as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(conv_dim as u64, 64);
    b.push(
        "q3w-dn-conv",
        &s.delta,
        "q3w_delta_conv",
        &[
            (0, &qkv),
            (1, &conv_w),
            (2, &conv_state),
            (3, &mixed),
            (4, &cp),
        ],
        grid,
    )?;

    let qg = b.zeros("q3w-dn-q", (n_v * d_k * 4) as u64);
    let kg = b.zeros("q3w-dn-k", (n_v * d_k * 4) as u64);
    let vg = b.zeros("q3w-dn-v", (n_v * d_v * 4) as u64);
    let dqp = b.uni(
        "q3w-dn-qkv-p",
        DeltaQkvParams {
            n_v: n_v as u32,
            d_k: d_k as u32,
            d_v: d_v as u32,
            key_dim: key_dim as u32,
            v_per_k: (n_v / n_k) as u32,
            ab_off: ab_off as u32,
            scale: 1.0 / (d_k as f32).sqrt(),
            ..Default::default()
        },
    );

    let mut alogdt = d.a_log.clone();
    alogdt.extend_from_slice(&d.dt_bias);
    let alogdt = b.upload_f32("q3w-dn-alogdt", &alogdt);
    let gexp = b.zeros("q3w-dn-g", (n_v * 4) as u64);
    let beta = b.zeros("q3w-dn-beta", (n_v * 4) as u64);

    if delta_gate_fused() {
        b.push(
            "q3w-dn-split",
            &s.delta,
            "q3w_delta_qkv_gated",
            &[
                (10, &mixed),
                (11, &qg),
                (12, &kg),
                (13, &vg),
                (14, &dqp),
                (20, &ab),
                (21, &alogdt),
                (23, &gexp),
                (24, &beta),
            ],
            (n_v as u32, 1, 1),
        )?;
        b.dn_gate.0 += 1;
        b.dn_gate.1 += 1;
    } else {
        b.push(
            "q3w-dn-split",
            &s.delta,
            "q3w_delta_qkv",
            &[(10, &mixed), (11, &qg), (12, &kg), (13, &vg), (14, &dqp)],
            (n_v as u32, 1, 1),
        )?;
        let gp = b.uni(
            "q3w-dn-gate-p",
            GatingParams {
                n_v: n_v as u32,
                ab_off: ab_off as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1(n_v as u64, 64);
        b.push(
            "q3w-dn-gating",
            &s.delta,
            "q3w_delta_gating",
            &[
                (20, &ab),
                (21, &alogdt),
                (23, &gexp),
                (24, &beta),
                (25, &gp),
            ],
            grid,
        )?;
        b.dn_gate.1 += 1;
    }

    let state_bytes = (n_v * d_k * d_v * 4) as u64;
    let state = b.zeros("q3w-dn-state", state_bytes);
    states.push((state.clone(), state_bytes));
    let core = b.zeros("q3w-dn-core", (n_v * d_v * 4) as u64);
    let rp = b.uni(
        "q3w-dn-rec-p",
        RecurrentParams {
            heads: n_v as u32,
            d_k: d_k as u32,
            d_v: d_v as u32,
            pad0: 0,
        },
    );
    let (rec_entry, rec_lanes) = delta_recurrent_kernel();
    anyhow::ensure!(
        rec_lanes > 0 || d_v <= 128,
        "{rec_entry} covers one lane per invocation of a 128-wide workgroup, got d_v {d_v}"
    );
    let rec_grid_y = if rec_lanes > 0 {
        (d_v as u32).div_ceil(rec_lanes)
    } else {
        1
    };
    b.delta_u4.0 += usize::from(rec_entry != "q3w_delta_recurrent");
    b.delta_u4.1 += 1;
    b.push(
        "q3w-dn-recurrent",
        &s.delta,
        rec_entry,
        &[
            (30, &qg),
            (31, &kg),
            (32, &vg),
            (33, &gexp),
            (34, &beta),
            (35, &core),
            (36, &state),
            (37, &rp),
        ],
        (n_v as u32, rec_grid_y, 1),
    )?;

    let norm_w = b.upload_u32("q3w-dn-normw", &pack_pairs(&d.norm_w));
    let gated = b.zeros("q3w-dn-gated", (value_dim * 2) as u64);
    let dop = b.uni(
        "q3w-dn-out-p",
        DeltaOutParams {
            n_v: n_v as u32,
            d_v: d_v as u32,
            z_off: z_off as u32,
            eps: cfg.rms_norm_eps as f32,
        },
    );
    b.push(
        "q3w-dn-out",
        &s.delta,
        "q3w_delta_out",
        &[
            (40, &core),
            (41, &norm_w),
            (42, &z),
            (43, &gated),
            (44, &dop),
        ],
        (n_v as u32, 1, 1),
    )?;

    push_proj(b, s, "q3w-dn-oproj", &w_out, &gated, out)?;

    let Some(p) = pf.as_mut().filter(|p| p.ok) else {
        return Ok(());
    };
    b.to_prefill = true;
    let r = (|b: &mut Builder| -> Result<()> {
        let w_in_all = w_in_all.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "M-row prefill covers only the fused delta in_proj arm; \
                 NV_Q3_WGPU_FUSE_DN_INPROJ=0 leaves prefill on the per-token replay"
            )
        })?;
        anyhow::ensure!(
            delta_gate_fused(),
            "M-row prefill covers only the fused delta gating arm; \
             NV_Q3_WGPU_FUSE_DN_GATE=0 leaves prefill on the per-token replay"
        );
        let m = p.m;
        let hidden_words = cfg.hidden_size / 2;
        let n_all = conv_dim + value_dim + 2 * n_v;
        if p.delta.is_none() {
            let in_stride_words = pf_stride_elems_bf16(n_all) / 2;
            let mixed_stride_bytes = pf_stride_bytes_f32(conv_dim);
            let qk_stride_bytes = pf_stride_bytes_f32(n_v * d_k);
            let v_stride_bytes = pf_stride_bytes_f32(n_v * d_v);
            let gb_stride_bytes = pf_stride_bytes_f32(n_v);
            let core_stride_bytes = pf_stride_bytes_f32(n_v * d_v);
            let gated_stride_words = pf_stride_elems_bf16(value_dim) / 2;
            p.delta = Some(PfDeltaBufs {
                in_stride_words,
                in_all: b.zeros("q3w-pf-dn-in", (m * in_stride_words * 4) as u64),
                mixed_stride_bytes,
                mixed: b.zeros("q3w-pf-dn-mixed", m as u64 * mixed_stride_bytes),
                qk_stride_bytes,
                qg: b.zeros("q3w-pf-dn-q", m as u64 * qk_stride_bytes),
                kg: b.zeros("q3w-pf-dn-k", m as u64 * qk_stride_bytes),
                v_stride_bytes,
                vg: b.zeros("q3w-pf-dn-v", m as u64 * v_stride_bytes),
                gb_stride_bytes,
                gexp: b.zeros("q3w-pf-dn-g", m as u64 * gb_stride_bytes),
                beta: b.zeros("q3w-pf-dn-beta", m as u64 * gb_stride_bytes),
                core_stride_bytes,
                core: b.zeros("q3w-pf-dn-core", m as u64 * core_stride_bytes),
                gated_stride_words,
                gated: b.zeros("q3w-pf-dn-gated", (m * gated_stride_words * 4) as u64),
            });
        }
        let dl = p.delta.as_ref().expect("created above");
        anyhow::ensure!(
            dl.in_stride_words == pf_stride_elems_bf16(n_all) / 2,
            "pf delta scratch was sized for a different layer shape"
        );
        pf_push_proj_m(
            b,
            s,
            p,
            "q3w-pf-dn-inproj",
            w_in_all,
            &p.normed,
            hidden_words,
            &dl.in_all,
            dl.in_stride_words,
        )?;
        let conv_grid = b.grid1(conv_dim as u64, 64);
        if pf_token_parallel_default_on_escape_to_per_token_copies() {
            anyhow::ensure!(
                ks <= 8,
                "q3w_pf_delta_conv_chunk keeps the rolling window in a fixed 8-slot \
                 register array, got kernel {ks}"
            );
            let src = pf_delta_chunk_source(d_k);
            let pdp = b.uni(
                "q3w-pf-dn-chunk-p",
                PfDeltaChunkParams {
                    tokens: m as u32,
                    in_stride_elems: (dl.in_stride_words * 2) as u32,
                    mixed_stride_elems: (dl.mixed_stride_bytes / 4) as u32,
                    qk_stride_elems: (dl.qk_stride_bytes / 4) as u32,
                    v_stride_elems: (dl.v_stride_bytes / 4) as u32,
                    gb_stride_elems: (dl.gb_stride_bytes / 4) as u32,
                    core_stride_elems: (dl.core_stride_bytes / 4) as u32,
                    gated_stride_elems: (dl.gated_stride_words * 2) as u32,
                },
            );
            b.push(
                "q3w-pf-dn-conv-chunk",
                &src,
                "q3w_pf_delta_conv_chunk",
                &[
                    (0, &dl.in_all),
                    (1, &conv_w),
                    (2, &conv_state),
                    (3, &dl.mixed),
                    (4, &cp),
                    (50, &pdp),
                ],
                conv_grid,
            )?;
            b.push(
                "q3w-pf-dn-split-m",
                &src,
                "q3w_pf_delta_split_gated_m",
                &[
                    (10, &dl.mixed),
                    (11, &dl.qg),
                    (12, &dl.kg),
                    (13, &dl.vg),
                    (14, &dqp),
                    (20, &dl.in_all),
                    (21, &alogdt),
                    (23, &dl.gexp),
                    (24, &dl.beta),
                    (50, &pdp),
                ],
                (n_v as u32, m as u32, 1),
            )?;
            b.push(
                "q3w-pf-dn-recurrent-chunk",
                &src,
                "q3w_pf_delta_recurrent_chunk",
                &[
                    (30, &dl.qg),
                    (31, &dl.kg),
                    (32, &dl.vg),
                    (33, &dl.gexp),
                    (34, &dl.beta),
                    (35, &dl.core),
                    (36, &state),
                    (37, &rp),
                    (50, &pdp),
                ],
                (n_v as u32, (d_v as u32).div_ceil(32), 1),
            )?;
            b.push(
                "q3w-pf-dn-out-m",
                &src,
                "q3w_pf_delta_out_m",
                &[
                    (40, &dl.core),
                    (41, &norm_w),
                    (42, &dl.in_all),
                    (43, &dl.gated),
                    (44, &dop),
                    (50, &pdp),
                ],
                (n_v as u32, m as u32, 1),
            )?;
            return pf_push_proj_m(
                b,
                s,
                p,
                "q3w-pf-dn-oproj",
                &w_out,
                &dl.gated,
                dl.gated_stride_words,
                &p.mix,
                hidden_words,
            );
        }
        let in_stride_bytes = (dl.in_stride_words * 4) as u64;
        let gated_stride_bytes = (dl.gated_stride_words * 4) as u64;
        for t in 0..m as u64 {
            b.push_pf_off(
                "q3w-pf-dn-conv",
                &s.delta,
                "q3w_delta_conv",
                &[
                    (0, &dl.in_all, t * in_stride_bytes),
                    (1, &conv_w, 0),
                    (2, &conv_state, 0),
                    (3, &dl.mixed, t * dl.mixed_stride_bytes),
                    (4, &cp, 0),
                ],
                conv_grid,
            )?;
            b.push_pf_off(
                "q3w-pf-dn-split",
                &s.delta,
                "q3w_delta_qkv_gated",
                &[
                    (10, &dl.mixed, t * dl.mixed_stride_bytes),
                    (11, &dl.qg, t * dl.qk_stride_bytes),
                    (12, &dl.kg, t * dl.qk_stride_bytes),
                    (13, &dl.vg, t * dl.v_stride_bytes),
                    (14, &dqp, 0),
                    (20, &dl.in_all, t * in_stride_bytes),
                    (21, &alogdt, 0),
                    (23, &dl.gexp, t * dl.gb_stride_bytes),
                    (24, &dl.beta, t * dl.gb_stride_bytes),
                ],
                (n_v as u32, 1, 1),
            )?;
            b.push_pf_off(
                "q3w-pf-dn-recurrent",
                &s.delta,
                rec_entry,
                &[
                    (30, &dl.qg, t * dl.qk_stride_bytes),
                    (31, &dl.kg, t * dl.qk_stride_bytes),
                    (32, &dl.vg, t * dl.v_stride_bytes),
                    (33, &dl.gexp, t * dl.gb_stride_bytes),
                    (34, &dl.beta, t * dl.gb_stride_bytes),
                    (35, &dl.core, t * dl.core_stride_bytes),
                    (36, &state, 0),
                    (37, &rp, 0),
                ],
                (n_v as u32, rec_grid_y, 1),
            )?;
            b.push_pf_off(
                "q3w-pf-dn-out",
                &s.delta,
                "q3w_delta_out",
                &[
                    (40, &dl.core, t * dl.core_stride_bytes),
                    (41, &norm_w, 0),
                    (42, &dl.in_all, t * in_stride_bytes),
                    (43, &dl.gated, t * gated_stride_bytes),
                    (44, &dop, 0),
                ],
                (n_v as u32, 1, 1),
            )?;
        }
        pf_push_proj_m(
            b,
            s,
            p,
            "q3w-pf-dn-oproj",
            &w_out,
            &dl.gated,
            dl.gated_stride_words,
            &p.mix,
            hidden_words,
        )
    })(b);
    b.to_prefill = false;
    if let Err(e) = r {
        p.ok = false;
        p.off_reason = Some(format!("delta layer: {e}"));
    }
    Ok(())
}

enum PfKvProj {
    Fused(Nvfp4Gpu),
    Split(Nvfp4Gpu, Nvfp4Gpu),
}

#[allow(clippy::too_many_arguments)]
fn build_attn(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3MoeConfig,
    a: &HostAttention,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pos_buf: &wgpu::Buffer,
    fd_buf: &wgpu::Buffer,
    max_seq: usize,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    pf: &mut Option<PfMrowBufs>,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let rot = cfg.rotary_dim();
    anyhow::ensure!(
        rot.is_multiple_of(2) && rot <= hd,
        "rotary_dim {rot} invalid"
    );
    anyhow::ensure!(
        hd.is_multiple_of(2) && a.q_norm.len() == hd && a.k_norm.len() == hd,
        "q/k norm weights are {}/{} long, want head_dim {hd} (even)",
        a.q_norm.len(),
        a.k_norm.len()
    );

    let wq = upload_nvfp4(b, "q3w-at-q", &a.q);
    let wo = upload_nvfp4(b, "q3w-at-o", &a.o);

    let sel0 = b.upload_u32("q3w-at-sel", &[0u32]);
    let alpha_dummy = b.upload_f32("q3w-at-alpha", &[1.0f32]);

    let k_blocks = hidden / NVFP4_BLOCK;
    let xq = b.zeros("q3w-at-xq", (hidden / 2) as u64);
    let xs = b.zeros("q3w-at-xs", (k_blocks.div_ceil(4) * 4) as u64);
    let glob_in = b.upload_f32("q3w-at-gin", &[wq.input_global]);
    push_quant_rows(
        b,
        s,
        "q3w-at-quant",
        x,
        &xq,
        &xs,
        &sel0,
        &glob_in,
        hidden,
        1,
        false,
        0,
    )?;

    anyhow::ensure!(
        (a.q.input_global - a.k.input_global).abs() <= 1e-6
            && (a.q.input_global - a.v.input_global).abs() <= 1e-6,
        "q/k/v input_global_scale differ: {} {} {}",
        a.q.input_global,
        a.k.input_global,
        a.v.input_global
    );

    let q_raw = b.zeros("q3w-at-qraw", (wq.n * 2) as u64);
    push_gemv_nvfp4(
        b,
        s,
        "q3w-at-qproj",
        &wq.w,
        &wq.scales,
        &xq,
        &xs,
        &q_raw,
        &sel0,
        &alpha_dummy,
        wq.n,
        wq.k,
        1,
        wq.n / 2,
        false,
        wq.alpha,
    )?;

    let fuse_kv = attn_kv_fused() && nvfp4_rowcat_ok(b.ctx, &a.v, &a.k);
    b.at_kv.1 += 1;
    let (v_raw, k_raw, k_src_off, kv_proj) = if fuse_kv {
        let vk = upload_nvfp4_rowcat(b, "q3w-at-vk", &a.v, &a.k);
        let raw = b.zeros("q3w-at-vkraw", (vk.n * 2) as u64);
        push_gemv_nvfp4(
            b,
            s,
            "q3w-at-vkproj",
            &vk.w,
            &vk.scales,
            &xq,
            &xs,
            &raw,
            &sel0,
            &alpha_dummy,
            vk.n,
            vk.k,
            1,
            vk.n / 2,
            false,
            vk.alpha,
        )?;
        b.at_kv.0 += 1;
        (raw.clone(), raw, a.v.n, PfKvProj::Fused(vk))
    } else {
        let wv = upload_nvfp4(b, "q3w-at-v", &a.v);
        let wk_ = upload_nvfp4(b, "q3w-at-k", &a.k);
        let v_raw = b.zeros("q3w-at-vraw", (wv.n * 2) as u64);
        let k_raw = b.zeros("q3w-at-kraw", (wk_.n * 2) as u64);
        push_gemv_nvfp4(
            b,
            s,
            "q3w-at-vproj",
            &wv.w,
            &wv.scales,
            &xq,
            &xs,
            &v_raw,
            &sel0,
            &alpha_dummy,
            wv.n,
            wv.k,
            1,
            wv.n / 2,
            false,
            wv.alpha,
        )?;
        push_gemv_nvfp4(
            b,
            s,
            "q3w-at-kproj",
            &wk_.w,
            &wk_.scales,
            &xq,
            &xs,
            &k_raw,
            &sel0,
            &alpha_dummy,
            wk_.n,
            wk_.k,
            1,
            wk_.n / 2,
            false,
            wk_.alpha,
        )?;
        (v_raw, k_raw, 0, PfKvProj::Split(wv, wk_))
    };

    let (cos, sin) = rope_tables(rot.max(2), cfg.rope_theta, max_seq);
    let sin_off = cos.len();
    let mut cs = cos;
    cs.extend_from_slice(&sin);
    let csb = b.upload_f32("q3w-at-cs", &cs);
    let mut qkn = a.q_norm.clone();
    qkn.extend_from_slice(&a.k_norm);
    let qkn = b.upload_u32("q3w-at-qkn", &pack_pairs(&qkn));
    let q = b.zeros("q3w-at-q", (n_h * hd * 2) as u64);
    let k = b.zeros("q3w-at-k", (n_kv * hd * 2) as u64);
    let src_stride_q = if cfg.attn_output_gate { 2 * hd } else { hd };
    let q_f32 = b.zeros("q3w-at-qf32", (n_h * hd * 4) as u64);
    let fuse_qcast = attn_qcast_fused();
    let fuse_qknorm = fuse_qcast && attn_qknorm_fused();
    if fuse_qknorm {
        let np = b.uni(
            "q3w-at-qknorm-p",
            NormRopeParams {
                n_rows: (n_h + n_kv) as u32,
                head_dim: hd as u32,
                src_stride: src_stride_q as u32,
                rot_half: (rot / 2) as u32,
                n_q: n_h as u32,
                k_src_off: k_src_off as u32,
                k_src_stride: hd as u32,
                k_w_off: hd as u32,
                sin_off: sin_off as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        b.push(
            "q3w-at-qknorm",
            &s.attn,
            "q3w_attn_norm_rope_qk",
            &[
                (0, &q_raw),
                (1, &qkn),
                (2, &csb),
                (3, pos_buf),
                (4, &k_raw),
                (5, &q),
                (6, &np),
                (7, &q_f32),
                (8, &k),
            ],
            ((n_h + n_kv) as u32, 1, 1),
        )?;
        b.at_qknorm.0 += 1;
        b.at_qcast.0 += 1;
    } else {
        let qp = b.uni(
            "q3w-at-qnorm-p",
            NormRopeParams {
                n_rows: n_h as u32,
                head_dim: hd as u32,
                src_stride: src_stride_q as u32,
                rot_half: (rot / 2) as u32,
                sin_off: sin_off as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        if fuse_qcast {
            b.push(
                "q3w-at-qnorm",
                &s.attn,
                "q3w_attn_norm_rope_f32",
                &[
                    (0, &q_raw),
                    (1, &qkn),
                    (2, &csb),
                    (3, pos_buf),
                    (5, &q),
                    (6, &qp),
                    (7, &q_f32),
                ],
                (n_h as u32, 1, 1),
            )?;
            b.at_qcast.0 += 1;
        } else {
            b.push(
                "q3w-at-qnorm",
                &s.attn,
                "q3w_attn_norm_rope",
                &[
                    (0, &q_raw),
                    (1, &qkn),
                    (2, &csb),
                    (3, pos_buf),
                    (5, &q),
                    (6, &qp),
                ],
                (n_h as u32, 1, 1),
            )?;
        }
        let kp = b.uni(
            "q3w-at-knorm-p",
            NormRopeParams {
                n_rows: n_kv as u32,
                head_dim: hd as u32,
                src_stride: hd as u32,
                rot_half: (rot / 2) as u32,
                src_off: k_src_off as u32,
                w_off: hd as u32,
                sin_off: sin_off as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        b.push(
            "q3w-at-knorm",
            &s.attn,
            "q3w_attn_norm_rope",
            &[
                (0, &k_raw),
                (1, &qkn),
                (2, &csb),
                (3, pos_buf),
                (5, &k),
                (6, &kp),
            ],
            (n_kv as u32, 1, 1),
        )?;
    }
    b.at_qcast.1 += 1;
    b.at_qknorm.1 += 1;

    let cache_bytes = (max_seq * n_kv * hd) as u64;
    let scale_bytes = (max_seq * n_kv * 4) as u64;
    let kc = b.zeros("q3w-at-kc", cache_bytes);
    let vc = b.zeros("q3w-at-vc", cache_bytes);
    let ksc = b.zeros("q3w-at-ks", scale_bytes);
    let vsc = b.zeros("q3w-at-vs", scale_bytes);
    states.push((kc.clone(), cache_bytes));
    states.push((vc.clone(), cache_bytes));
    states.push((ksc.clone(), scale_bytes));
    states.push((vsc.clone(), scale_bytes));
    let kvq_p = b.uni(
        "q3w-at-kvq-p",
        KvFp8Params {
            n_tokens: 1,
            n_kv: n_kv as u32,
            head_dim: hd as u32,
            pairs: n_kv as u32,
            slots: max_seq as u32,
            ..Default::default()
        },
    );
    b.push(
        "q3w-at-kvq-k",
        &s.kvq,
        wk::kv_fp8::QUANTIZE_ENTRY,
        &[(0, &k), (1, &kc), (2, &ksc), (3, pos_buf), (4, &kvq_p)],
        (n_kv as u32, 1, 1),
    )?;
    b.push(
        "q3w-at-kvq-v",
        &s.kvq,
        wk::kv_fp8::QUANTIZE_ENTRY,
        &[(0, &v_raw), (1, &vc), (2, &vsc), (3, pos_buf), (4, &kvq_p)],
        (n_kv as u32, 1, 1),
    )?;

    if !fuse_qcast {
        let cast_p = b.uni(
            "q3w-at-cast-p",
            ResScaleParams {
                n: (n_h * hd) as u32,
                n_words: (n_h * hd / 2) as u32,
                scale: 1.0,
                ..Default::default()
            },
        );
        let grid = b.grid1((n_h * hd / 2) as u64, 256);
        b.push(
            "q3w-at-qcast",
            &s.resscale,
            "cast_bf16_to_f32",
            &[(0, &q), (3, &cast_p), (4, &q_f32)],
            grid,
        )?;
    }

    let scratch = b.zeros(
        "q3w-at-scratch",
        (n_h * flash_splits() as usize * (hd + 2) * 4) as u64,
    );
    let attn = b.zeros("q3w-at-out", (n_h * hd * 4) as u64);
    b.push(
        "q3w-at-flash1",
        &s.flash,
        if flash_sd_enabled() {
            wk::flash_decode::ENTRY_STAGE1_FP8_SD
        } else {
            wk::flash_decode::ENTRY_STAGE1_FP8
        },
        &[
            (0, &q_f32),
            (4, fd_buf),
            (5, &kc),
            (6, &vc),
            (7, &scratch),
            (8, &ksc),
            (9, &vsc),
        ],
        (n_h as u32, flash_splits(), 1),
    )?;
    b.push(
        "q3w-at-flash2",
        &s.flash,
        wk::flash_decode::ENTRY_STAGE2,
        &[(3, &attn), (4, fd_buf), (7, &scratch)],
        (n_h as u32, 1, 1),
    )?;

    let gated = b.zeros("q3w-at-gated", (n_h * hd * 2) as u64);
    let agp = b.uni(
        "q3w-at-gate-p",
        AttnGateParams {
            n_words: (n_h * hd / 2) as u32,
            head_dim: hd as u32,
            src_stride: src_stride_q as u32,
            gate_off: hd as u32,
            has_gate: u32::from(cfg.attn_output_gate),
            ..Default::default()
        },
    );
    let grid = b.grid1((n_h * hd / 2) as u64, 64);
    b.push(
        "q3w-at-gate",
        &s.attn,
        "q3w_attn_gate",
        &[(30, &attn), (31, &q_raw), (32, &gated), (33, &agp)],
        grid,
    )?;

    let ok_blocks = wo.k / NVFP4_BLOCK;
    let oq = b.zeros("q3w-at-oq", (wo.k / 2) as u64);
    let os = b.zeros("q3w-at-os", (ok_blocks.div_ceil(4) * 4) as u64);
    let glob_o = b.upload_f32("q3w-at-gout", &[wo.input_global]);
    push_quant_rows(
        b,
        s,
        "q3w-at-oquant",
        &gated,
        &oq,
        &os,
        &sel0,
        &glob_o,
        wo.k,
        1,
        false,
        0,
    )?;
    push_gemv_nvfp4(
        b,
        s,
        "q3w-at-oproj",
        &wo.w,
        &wo.scales,
        &oq,
        &os,
        out,
        &sel0,
        &alpha_dummy,
        wo.n,
        wo.k,
        1,
        wo.n / 2,
        false,
        wo.alpha,
    )?;

    let Some(p) = pf.as_mut().filter(|p| p.ok) else {
        return Ok(());
    };
    b.to_prefill = true;
    let r = (|b: &mut Builder| -> Result<()> {
        anyhow::ensure!(
            fuse_qknorm,
            "M-row prefill covers only the fused q/k norm+rope arm; \
             NV_Q3_WGPU_FUSE_AT_QKNORM=0 or NV_Q3_WGPU_FUSE_AT_QCAST=0 leaves prefill \
             on the per-token replay"
        );
        let m = p.m;
        let hidden_words = hidden / 2;
        if p.attn.is_none() {
            let q_raw_stride_words = pf_stride_elems_bf16(wq.n) / 2;
            let (vk_rows, k_rows) = match &kv_proj {
                PfKvProj::Fused(vk) => (vk.n, 0),
                PfKvProj::Split(wv, wk_) => (wv.n, wk_.n),
            };
            let vk_raw_stride_words = pf_stride_elems_bf16(vk_rows) / 2;
            let vk_raw = b.zeros("q3w-pf-at-vkraw", (m * vk_raw_stride_words * 4) as u64);
            let (k_raw_stride_words, k_raw_buf) = if k_rows == 0 {
                (vk_raw_stride_words, vk_raw.clone())
            } else {
                let st = pf_stride_elems_bf16(k_rows) / 2;
                (st, b.zeros("q3w-pf-at-kraw", (m * st * 4) as u64))
            };
            let mk_scratch = pf_token_parallel_default_on_escape_to_per_token_copies().then(|| {
                let scratch_rows =
                    if pf_flash_tiled_default_on_since_slotml_solo_nll_plus_p006_nats_under_p01_bound_and_557_vs_431_tok_s_user_signed_off() {
                        m
                    } else {
                        PF_MK_FLASH_ROWS_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS
                    };
                b.zeros(
                    "q3w-pf-at-mkscratch",
                    (n_h * scratch_rows * flash_splits() as usize * (hd + 2) * 4) as u64,
                )
            });
            p.attn = Some(PfAttnBufs {
                fused_kv: fuse_kv,
                mk_scratch,
                xq: b.zeros("q3w-pf-at-xq", (m * hidden / 2) as u64),
                xs: b.zeros("q3w-pf-at-xs", (m * k_blocks) as u64),
                q_raw_stride_words,
                q_raw: b.zeros("q3w-pf-at-qraw", (m * q_raw_stride_words * 4) as u64),
                vk_raw_stride_words,
                vk_raw,
                k_raw_stride_words,
                k_raw: k_raw_buf,
                q_stride_bytes: pf_pad_bytes_to_bind_offset_align((n_h * hd * 2) as u64),
                q: b.zeros(
                    "q3w-pf-at-q",
                    m as u64 * pf_pad_bytes_to_bind_offset_align((n_h * hd * 2) as u64),
                ),
                k_stride_bytes: pf_pad_bytes_to_bind_offset_align((n_kv * hd * 2) as u64),
                k: b.zeros(
                    "q3w-pf-at-k",
                    m as u64 * pf_pad_bytes_to_bind_offset_align((n_kv * hd * 2) as u64),
                ),
                qf32_stride_bytes: pf_stride_bytes_f32(n_h * hd),
                q_f32: b.zeros("q3w-pf-at-qf32", m as u64 * pf_stride_bytes_f32(n_h * hd)),
                attn_stride_bytes: pf_stride_bytes_f32(n_h * hd),
                attn: b.zeros("q3w-pf-at-attn", m as u64 * pf_stride_bytes_f32(n_h * hd)),
                gated_stride_words: pf_stride_elems_bf16(n_h * hd) / 2,
                gated: b.zeros(
                    "q3w-pf-at-gated",
                    (m * (pf_stride_elems_bf16(n_h * hd) / 2) * 4) as u64,
                ),
                oq: b.zeros("q3w-pf-at-oq", (m * wo.k / 2) as u64),
                os: b.zeros("q3w-pf-at-os", (m * wo.k / NVFP4_BLOCK) as u64),
            });
        }
        let at = p.attn.as_ref().expect("created above");
        anyhow::ensure!(
            at.fused_kv == fuse_kv,
            "kv fusion arm changed between attention layers; the shared M-row scratch \
             assumes one arm for every layer"
        );
        let glob_in_m = b.upload_f32("q3w-pf-at-gin", &vec![wq.input_global; m]);
        push_quant_rows(
            b,
            s,
            "q3w-pf-at-quant",
            &p.normed,
            &at.xq,
            &at.xs,
            &p.sel_zeros_m,
            &glob_in_m,
            hidden,
            m,
            false,
            hidden,
        )?;
        push_gemv_nvfp4(
            b,
            s,
            "q3w-pf-at-qproj",
            &wq.w,
            &wq.scales,
            &at.xq,
            &at.xs,
            &at.q_raw,
            &p.sel_zeros_m,
            &alpha_dummy,
            wq.n,
            wq.k,
            m,
            at.q_raw_stride_words,
            false,
            wq.alpha,
        )?;
        match &kv_proj {
            PfKvProj::Fused(vk) => {
                push_gemv_nvfp4(
                    b,
                    s,
                    "q3w-pf-at-vkproj",
                    &vk.w,
                    &vk.scales,
                    &at.xq,
                    &at.xs,
                    &at.vk_raw,
                    &p.sel_zeros_m,
                    &alpha_dummy,
                    vk.n,
                    vk.k,
                    m,
                    at.vk_raw_stride_words,
                    false,
                    vk.alpha,
                )?;
            }
            PfKvProj::Split(wv, wk_) => {
                push_gemv_nvfp4(
                    b,
                    s,
                    "q3w-pf-at-vproj",
                    &wv.w,
                    &wv.scales,
                    &at.xq,
                    &at.xs,
                    &at.vk_raw,
                    &p.sel_zeros_m,
                    &alpha_dummy,
                    wv.n,
                    wv.k,
                    m,
                    at.vk_raw_stride_words,
                    false,
                    wv.alpha,
                )?;
                push_gemv_nvfp4(
                    b,
                    s,
                    "q3w-pf-at-kproj",
                    &wk_.w,
                    &wk_.scales,
                    &at.xq,
                    &at.xs,
                    &at.k_raw,
                    &p.sel_zeros_m,
                    &alpha_dummy,
                    wk_.n,
                    wk_.k,
                    m,
                    at.k_raw_stride_words,
                    false,
                    wk_.alpha,
                )?;
            }
        }
        let np = b.uni(
            "q3w-pf-at-qknorm-p",
            NormRopeParams {
                n_rows: (n_h + n_kv) as u32,
                head_dim: hd as u32,
                src_stride: src_stride_q as u32,
                rot_half: (rot / 2) as u32,
                n_q: n_h as u32,
                k_src_off: k_src_off as u32,
                k_src_stride: hd as u32,
                k_w_off: hd as u32,
                sin_off: sin_off as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        let glob_o_m = b.upload_f32("q3w-pf-at-gout", &vec![wo.input_global; m]);
        let qraw_bytes = (at.q_raw_stride_words * 4) as u64;
        let vkraw_bytes = (at.vk_raw_stride_words * 4) as u64;
        let kraw_bytes = (at.k_raw_stride_words * 4) as u64;
        let gated_bytes = (at.gated_stride_words * 4) as u64;
        let gate_grid = b.grid1((n_h * hd / 2) as u64, 64);
        if pf_token_parallel_default_on_escape_to_per_token_copies() {
            anyhow::ensure!(
                at.qf32_stride_bytes == (n_h * hd * 4) as u64
                    && at.attn_stride_bytes == (n_h * hd * 4) as u64,
                "the MK flash kernels index q and out as tightly packed [token][head][dim] \
                 rows, so the pf q_f32/attn strides must equal n_h*hd*4 exactly; \
                 got {} and {} for {}",
                at.qf32_stride_bytes,
                at.attn_stride_bytes,
                n_h * hd * 4
            );
            let mk_scratch = at.mk_scratch.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "token-parallel attn needs the mk scratch allocated by the first \
                     attention layer under the same env; the env changed between layers"
                )
            })?;
            let parp = b.uni(
                "q3w-pf-at-rope-p",
                PfAttnRopeParams {
                    tokens: m as u32,
                    q_src_stride_elems: (at.q_raw_stride_words * 2) as u32,
                    k_src_stride_elems: (at.k_raw_stride_words * 2) as u32,
                    q_out_stride_elems: (at.q_stride_bytes / 2) as u32,
                    k_out_stride_elems: (at.k_stride_bytes / 2) as u32,
                    qf_out_stride_elems: (at.qf32_stride_bytes / 4) as u32,
                    pos_stride_words: PF_MROW_POS_STRIDE_WORDS as u32,
                    pad0: 0,
                },
            );
            b.push(
                "q3w-pf-at-qknorm-m",
                &s.pf_attn,
                "q3w_pf_attn_norm_rope_qk_m",
                &[
                    (0, &at.q_raw),
                    (1, &qkn),
                    (2, &csb),
                    (3, &p.pos_strided_one_i32_per_256b_slot),
                    (4, &at.k_raw),
                    (5, &at.q),
                    (6, &np),
                    (7, &at.q_f32),
                    (8, &at.k),
                    (9, &parp),
                ],
                ((n_h + n_kv) as u32, m as u32, 1),
            )?;
            let pkv_k = b.uni(
                "q3w-pf-at-kvq-k-p",
                PfKvqParams {
                    tokens: m as u32,
                    x_stride_elems: (at.k_stride_bytes / 2) as u32,
                    ..Default::default()
                },
            );
            b.push(
                "q3w-pf-at-kvq-k-m",
                &s.pf_kvq,
                "q3w_pf_quantize_kv_fp8_m",
                &[
                    (0, &at.k),
                    (1, &kc),
                    (2, &ksc),
                    (3, &p.pos_strided_one_i32_per_256b_slot),
                    (4, &kvq_p),
                    (9, &pkv_k),
                ],
                (n_kv as u32, m as u32, 1),
            )?;
            let pkv_v = b.uni(
                "q3w-pf-at-kvq-v-p",
                PfKvqParams {
                    tokens: m as u32,
                    x_stride_elems: (at.vk_raw_stride_words * 2) as u32,
                    ..Default::default()
                },
            );
            b.push(
                "q3w-pf-at-kvq-v-m",
                &s.pf_kvq,
                "q3w_pf_quantize_kv_fp8_m",
                &[
                    (0, &at.vk_raw),
                    (1, &vc),
                    (2, &vsc),
                    (3, &p.pos_strided_one_i32_per_256b_slot),
                    (4, &kvq_p),
                    (9, &pkv_v),
                ],
                (n_kv as u32, m as u32, 1),
            )?;
            let rows = PF_MK_FLASH_ROWS_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
            if pf_flash_tiled_default_on_since_slotml_solo_nll_plus_p006_nats_under_p01_bound_and_557_vs_431_tok_s_user_signed_off() {
                anyhow::ensure!(
                    PF_FLASH_TILED_WGSL.contains("const FDT_ROWS: u32 = 32u"),
                    "the tiled flash tile height is baked as 32 on both sides; \
                     PF_TILED_FLASH_ROWS_ONE_KV_STREAM_SERVES_32_ROWS_AT_4_PER_WARP and the \
                     kernel's FDT_ROWS must move together"
                );
                let subgroup = wk::gemv_nvfp4_v2::subgroup32_ok(b.ctx);
                let sd = flash_sd_enabled();
                let tiled_src = if sd {
                    pf_flash_tiled_source_sd(subgroup)
                } else {
                    pf_flash_tiled_source(subgroup)
                };
                let slotml =
                    pf_flash_tiled_slotml_is_the_default_arm_env_value_1_selects_the_older_tiling_with_larger_nll_penalty();
                let stock_entry = match (slotml, subgroup) {
                    (false, true) => "q3w_pf_flash1_fp8kv_tiled_sg",
                    (false, false) => "q3w_pf_flash1_fp8kv_tiled_wg",
                    (true, true) => "q3w_pf_flash1_fp8kv_tiled_slotml_sg",
                    (true, false) => "q3w_pf_flash1_fp8kv_tiled_slotml_wg",
                };
                let tiled_entry = if sd {
                    format!("{stock_entry}_sd")
                } else {
                    stock_entry.to_string()
                };
                let tiles = m.div_ceil(PF_TILED_FLASH_ROWS_ONE_KV_STREAM_SERVES_32_ROWS_AT_4_PER_WARP);
                let fdb = ((m + m.div_ceil(rows)) as u64) * PF_MROW_BIND_OFFSET_ALIGN_BYTES;
                b.push_pf_off_m_row(
                    "q3w-pf-at-flash1-tiled",
                    &tiled_src,
                    &tiled_entry,
                    &[
                        (0, &at.q_f32, 0),
                        (4, &p.fd_strided_one_fdparams_per_256b_slot, fdb),
                        (5, &kc, 0),
                        (6, &vc, 0),
                        (7, mk_scratch, 0),
                        (8, &ksc, 0),
                        (9, &vsc, 0),
                    ],
                    (n_h as u32, flash_splits(), tiles as u32),
                )?;
                b.push_pf_off_m_row(
                    "q3w-pf-at-flash2-mk",
                    &s.flash,
                    wk::flash_decode::ENTRY_STAGE2_MK,
                    &[
                        (3, &at.attn, 0),
                        (4, &p.fd_strided_one_fdparams_per_256b_slot, fdb),
                        (7, mk_scratch, 0),
                    ],
                    (n_h as u32, m as u32, 1),
                )?;
            } else {
            let flash1_owned = match (wk::gemv_nvfp4_v2::subgroup32_ok(b.ctx), flash_sd_enabled()) {
                (true, true) => Some((pf_flash_sg_source_sd(), "q3w_pf_flash1_fp8kv_mk_sg_sd")),
                (true, false) => Some((pf_flash_sg_source(), "q3w_pf_flash1_fp8kv_mk_sg")),
                (false, true) => Some((
                    compose(&format!(
                        "{}\n{}",
                        wk::flash_decode::WGSL,
                        wk::flash_decode::mk_stage1_source_sd()
                    )),
                    wk::flash_decode::ENTRY_STAGE1_FP8_MK_SD,
                )),
                (false, false) => None,
            };
            let (flash1_src, flash1_entry): (&str, &str) = match &flash1_owned {
                Some((src, entry)) => (src, entry),
                None => (&s.flash, wk::flash_decode::ENTRY_STAGE1_FP8_MK),
            };
            for g in 0..m.div_ceil(rows) {
                let mr_g = rows.min(m - g * rows);
                let qb = (g * rows) as u64 * at.qf32_stride_bytes;
                let ob = (g * rows) as u64 * at.attn_stride_bytes;
                let fdb = ((m + g) as u64) * PF_MROW_BIND_OFFSET_ALIGN_BYTES;
                b.push_pf_off_m_row(
                    "q3w-pf-at-flash1-mk",
                    flash1_src,
                    flash1_entry,
                    &[
                        (0, &at.q_f32, qb),
                        (4, &p.fd_strided_one_fdparams_per_256b_slot, fdb),
                        (5, &kc, 0),
                        (6, &vc, 0),
                        (7, mk_scratch, 0),
                        (8, &ksc, 0),
                        (9, &vsc, 0),
                    ],
                    (n_h as u32, flash_splits(), 1),
                )?;
                b.push_pf_off_m_row(
                    "q3w-pf-at-flash2-mk",
                    &s.flash,
                    wk::flash_decode::ENTRY_STAGE2_MK,
                    &[
                        (3, &at.attn, ob),
                        (4, &p.fd_strided_one_fdparams_per_256b_slot, fdb),
                        (7, mk_scratch, 0),
                    ],
                    (n_h as u32, mr_g as u32, 1),
                )?;
            }
            }
            let pagp = b.uni(
                "q3w-pf-at-gate-p",
                PfAttnGateParams {
                    attn_stride_elems: (at.attn_stride_bytes / 4) as u32,
                    qraw_stride_elems: (at.q_raw_stride_words * 2) as u32,
                    out_stride_words: at.gated_stride_words as u32,
                    pad0: 0,
                },
            );
            b.push(
                "q3w-pf-at-gate-m",
                &s.pf_attn,
                "q3w_pf_attn_gate_m",
                &[
                    (30, &at.attn),
                    (31, &at.q_raw),
                    (32, &at.gated),
                    (33, &agp),
                    (34, &pagp),
                ],
                (((n_h * hd / 2) as u32).div_ceil(64), m as u32, 1),
            )?;
        } else {
        for t in 0..m as u64 {
            let posb = t * (PF_MROW_POS_STRIDE_WORDS as u64 * 4);
            let fdb = t * PF_MROW_BIND_OFFSET_ALIGN_BYTES;
            b.push_pf_off(
                "q3w-pf-at-qknorm",
                &s.attn,
                "q3w_attn_norm_rope_qk",
                &[
                    (0, &at.q_raw, t * qraw_bytes),
                    (1, &qkn, 0),
                    (2, &csb, 0),
                    (3, &p.pos_strided_one_i32_per_256b_slot, posb),
                    (4, &at.k_raw, t * kraw_bytes),
                    (5, &at.q, t * at.q_stride_bytes),
                    (6, &np, 0),
                    (7, &at.q_f32, t * at.qf32_stride_bytes),
                    (8, &at.k, t * at.k_stride_bytes),
                ],
                ((n_h + n_kv) as u32, 1, 1),
            )?;
            b.push_pf_off(
                "q3w-pf-at-kvq-k",
                &s.kvq,
                wk::kv_fp8::QUANTIZE_ENTRY,
                &[
                    (0, &at.k, t * at.k_stride_bytes),
                    (1, &kc, 0),
                    (2, &ksc, 0),
                    (3, &p.pos_strided_one_i32_per_256b_slot, posb),
                    (4, &kvq_p, 0),
                ],
                (n_kv as u32, 1, 1),
            )?;
            b.push_pf_off(
                "q3w-pf-at-kvq-v",
                &s.kvq,
                wk::kv_fp8::QUANTIZE_ENTRY,
                &[
                    (0, &at.vk_raw, t * vkraw_bytes),
                    (1, &vc, 0),
                    (2, &vsc, 0),
                    (3, &p.pos_strided_one_i32_per_256b_slot, posb),
                    (4, &kvq_p, 0),
                ],
                (n_kv as u32, 1, 1),
            )?;
            b.push_pf_off(
                "q3w-pf-at-flash1",
                &s.flash,
                if flash_sd_enabled() {
                    wk::flash_decode::ENTRY_STAGE1_FP8_SD
                } else {
                    wk::flash_decode::ENTRY_STAGE1_FP8
                },
                &[
                    (0, &at.q_f32, t * at.qf32_stride_bytes),
                    (4, &p.fd_strided_one_fdparams_per_256b_slot, fdb),
                    (5, &kc, 0),
                    (6, &vc, 0),
                    (7, &scratch, 0),
                    (8, &ksc, 0),
                    (9, &vsc, 0),
                ],
                (n_h as u32, flash_splits(), 1),
            )?;
            b.push_pf_off(
                "q3w-pf-at-flash2",
                &s.flash,
                wk::flash_decode::ENTRY_STAGE2,
                &[
                    (3, &at.attn, t * at.attn_stride_bytes),
                    (4, &p.fd_strided_one_fdparams_per_256b_slot, fdb),
                    (7, &scratch, 0),
                ],
                (n_h as u32, 1, 1),
            )?;
            b.push_pf_off(
                "q3w-pf-at-gate",
                &s.attn,
                "q3w_attn_gate",
                &[
                    (30, &at.attn, t * at.attn_stride_bytes),
                    (31, &at.q_raw, t * qraw_bytes),
                    (32, &at.gated, t * gated_bytes),
                    (33, &agp, 0),
                ],
                gate_grid,
            )?;
        }
        }
        push_quant_rows(
            b,
            s,
            "q3w-pf-at-oquant",
            &at.gated,
            &at.oq,
            &at.os,
            &p.sel_zeros_m,
            &glob_o_m,
            wo.k,
            m,
            false,
            at.gated_stride_words * 2,
        )?;
        push_gemv_nvfp4(
            b,
            s,
            "q3w-pf-at-oproj",
            &wo.w,
            &wo.scales,
            &at.oq,
            &at.os,
            &p.mix,
            &p.sel_zeros_m,
            &alpha_dummy,
            wo.n,
            wo.k,
            m,
            hidden_words,
            false,
            wo.alpha,
        )
    })(b);
    b.to_prefill = false;
    if let Err(e) = r {
        p.ok = false;
        p.off_reason = Some(format!("attention layer: {e}"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_moe_w8(
    b: &mut Builder,
    s: &Sources,
    m: &HostMoe,
    x: &wgpu::Buffer,
    ids: &wgpu::Buffer,
    wts: &wgpu::Buffer,
    k_top: usize,
    hidden: usize,
    inter: usize,
    sinter: usize,
    hidden_words: usize,
    out: &wgpu::Buffer,
) -> Result<()> {
    let eg = upload_experts_i8(b, "q3w-moe8-eg", &m.experts_gate);
    let eu = upload_experts_i8(b, "q3w-moe8-eu", &m.experts_up);
    let ed = upload_experts_i8(b, "q3w-moe8-ed", &m.experts_down);

    let y_gate = b.zeros("q3w-moe-ygate", (k_top * inter * 2) as u64);
    let y_up = b.zeros("q3w-moe-yup", (k_top * inter * 2) as u64);
    push_gemv_i8_experts(
        b,
        s,
        "q3w-moe-gate8",
        &eg,
        x,
        &y_gate,
        ids,
        k_top,
        true,
        false,
    )?;
    push_gemv_i8_experts(b, s, "q3w-moe-up8", &eu, x, &y_up, ids, k_top, true, false)?;

    let act = b.zeros("q3w-moe-act", (k_top * inter * 2) as u64);
    let smp = b.uni(
        "q3w-moe-silu-p",
        SiluMulParams {
            n_words: (k_top * inter / 2) as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1((k_top * inter / 2) as u64, 64);
    b.push(
        "q3w-moe-silu",
        &s.moe,
        "q3w_silu_mul",
        &[(10, &y_gate), (11, &y_up), (12, &act), (13, &smp)],
        grid,
    )?;

    let y_down = b.zeros("q3w-moe-ydown", (k_top * hidden * 2) as u64);
    push_gemv_i8_experts(
        b,
        s,
        "q3w-moe-down8",
        &ed,
        &act,
        &y_down,
        ids,
        k_top,
        true,
        true,
    )?;

    let sel0 = b.upload_u32("q3w-moe-sel0", &[0u32]);
    let sg = upload_nvfp4_i8(b, "q3w-moe8-sg", &m.shared_gate);
    let su = upload_nvfp4_i8(b, "q3w-moe8-su", &m.shared_up);
    let sd = upload_nvfp4_i8(b, "q3w-moe8-sd", &m.shared_down);

    let sy_g = b.zeros("q3w-moe-syg", (sinter * 2) as u64);
    let sy_u = b.zeros("q3w-moe-syu", (sinter * 2) as u64);
    push_gemv_i8_experts(
        b,
        s,
        "q3w-moe-sgate8",
        &sg,
        x,
        &sy_g,
        &sel0,
        1,
        false,
        false,
    )?;
    push_gemv_i8_experts(b, s, "q3w-moe-sup8", &su, x, &sy_u, &sel0, 1, false, false)?;

    let sact = b.zeros("q3w-moe-sact", (sinter * 2) as u64);
    let ssp = b.uni(
        "q3w-moe-ssilu-p",
        SiluMulParams {
            n_words: (sinter / 2) as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1((sinter / 2) as u64, 64);
    b.push(
        "q3w-moe-ssilu",
        &s.moe,
        "q3w_silu_mul",
        &[(10, &sy_g), (11, &sy_u), (12, &sact), (13, &ssp)],
        grid,
    )?;

    let shared_out = b.zeros("q3w-moe-sout", (hidden * 2) as u64);
    push_gemv_i8_experts(
        b,
        s,
        "q3w-moe-sdown8",
        &sd,
        &sact,
        &shared_out,
        &sel0,
        1,
        false,
        false,
    )?;

    let sgate = upload_bf16(b, "q3w-moe-sgatew", &m.shared_expert_gate);
    let slogit = b.zeros("q3w-moe-slogit", 4);
    push_gemv_bf16(b, s, "q3w-moe-sgatelin", &sgate, x, &slogit, true, 0)?;

    let mcp = b.uni(
        "q3w-moe-comb-p",
        CombineParams {
            hidden_words: hidden_words as u32,
            k: k_top as u32,
            slot_stride_words: hidden_words as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(hidden_words as u64, 64);
    b.push(
        "q3w-moe-combine",
        &s.moe,
        "q3w_moe_combine",
        &[
            (20, &y_down),
            (21, wts),
            (22, &shared_out),
            (23, &slogit),
            (24, out),
            (25, &mcp),
        ],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_moe(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3MoeConfig,
    m: &HostMoe,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pf: &mut Option<PfMrowBufs>,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
    let e = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;

    anyhow::ensure!(
        m.experts_gate.input_globals == m.experts_up.input_globals,
        "expert gate/up input_global_scale differ; the wgpu path quantizes the token once per slot"
    );
    anyhow::ensure!(
        (m.shared_gate.input_global - m.shared_up.input_global).abs() <= 1e-6,
        "shared expert gate/up input_global_scale differ"
    );

    let w8 = w8_experts_enabled(b.ctx);
    let fold = shared_expert_folded()
        && !w8
        && sinter == inter
        && m.shared_gate.n == inter
        && m.shared_up.n == inter
        && m.shared_down.k == inter
        && m.experts_gate.e == e
        && m.experts_up.e == e
        && m.experts_down.e == e;
    b.shared_fold.0 += usize::from(fold);
    b.shared_fold.1 += 1;
    let slots = k_top + usize::from(fold);

    let gate_fused =
        router_gate_fused() && m.shared_expert_gate.n == 1 && m.shared_expert_gate.k == m.router.k;
    let router = if gate_fused {
        let mut w = Vec::with_capacity((m.router.n + 1) * m.router.k);
        w.extend_from_slice(&m.router.w);
        w.extend_from_slice(&m.shared_expert_gate.w);
        upload_bf16(
            b,
            "q3w-moe-router",
            &HostBf16Lin {
                w,
                n: m.router.n + 1,
                k: m.router.k,
            },
        )
    } else {
        upload_bf16(b, "q3w-moe-router", &m.router)
    };
    let rlogits = b.zeros("q3w-moe-rlogits", (router.n * 4) as u64);
    push_gemv_bf16(b, s, "q3w-moe-router", &router, x, &rlogits, true, 0)?;

    let ids = b.zeros("q3w-moe-ids", (slots * 4) as u64);
    let wts = b.zeros("q3w-moe-wts", (k_top * 4) as u64);
    let rp = b.uni(
        "q3w-moe-router-p",
        RouterParams {
            n_experts: e as u32,
            k: k_top as u32,
            shared_slot: u32::from(fold),
            ..Default::default()
        },
    );
    anyhow::ensure!(
        e <= 256,
        "router top-k kernels bound n_experts to 256, got {e}"
    );
    let par = router_par_enabled();
    let (topk_label, topk_entry) = if par {
        router_topk_entry()
    } else {
        ("q3w-moe-topk", "q3w_router_topk")
    };
    b.router_par.0 += usize::from(par);
    b.router_par.1 += 1;
    b.push(
        topk_label,
        &s.moe,
        topk_entry,
        &[(0, &rlogits), (1, &ids), (2, &wts), (3, &rp)],
        (1, 1, 1),
    )?;

    if w8 {
        if let Some(p) = pf.as_mut().filter(|p| p.ok) {
            p.ok = false;
            p.off_reason =
                Some("the NV_Q3_WGPU_W8_EXPERTS arm has no M-row list this round".to_string());
        }
        return emit_moe_w8(
            b,
            s,
            m,
            x,
            &ids,
            &wts,
            k_top,
            hidden,
            inter,
            sinter,
            hidden_words,
            out,
        );
    }
    let hb = hidden / NVFP4_BLOCK;
    let route_of = |n: usize| {
        nvfp4_v2_route(b.ctx, n, hidden, hb, slots)
            .map(|r| (r.entry, r.rows_per_group, r.vec4, r.source))
    };
    let fuse_gu = gate_up_fused()
        && fuse_silu_quant_enabled()
        && m.experts_gate.n.is_multiple_of(128)
        && m.experts_gate.alphas == m.experts_up.alphas
        && m.experts_gate.input_globals == m.experts_up.input_globals
        && (!fold
            || (m.shared_gate.alpha == m.shared_up.alpha
                && m.shared_gate.input_global == m.shared_up.input_global))
        && route_of(m.experts_gate.n) == route_of(2 * m.experts_gate.n);
    b.gate_up.1 += 1;
    let egu = if fuse_gu {
        b.gate_up.0 += 1;
        Some(upload_experts_gate_up(
            b,
            "q3w-moe-egu",
            &m.experts_gate,
            &m.experts_up,
            fold.then_some((&m.shared_gate, &m.shared_up)),
        )?)
    } else {
        None
    };
    let split = match &egu {
        Some(_) => None,
        None => Some((
            upload_experts_with(
                b,
                "q3w-moe-eg",
                &m.experts_gate,
                fold.then_some(&m.shared_gate),
            )?,
            upload_experts_with(b, "q3w-moe-eu", &m.experts_up, fold.then_some(&m.shared_up))?,
        )),
    };
    let ed = upload_experts_with(
        b,
        "q3w-moe-ed",
        &m.experts_down,
        fold.then_some(&m.shared_down),
    )?;

    let ib = inter / NVFP4_BLOCK;
    let xq = b.zeros("q3w-moe-xq", (slots * hidden / 2) as u64);
    let xs = b.zeros("q3w-moe-xs", (slots * hb) as u64);
    let x_globals = match (&egu, &split) {
        (Some(p), _) => &p.globals,
        (None, Some((g, _))) => &g.globals,
        _ => unreachable!("exactly one of the gate stacks is built"),
    };
    push_quant_rows(
        b,
        s,
        "q3w-moe-xquant",
        x,
        &xq,
        &xs,
        &ids,
        x_globals,
        hidden,
        slots,
        true,
        0,
    )?;

    let (y_gate, y_up) = match (&egu, &split) {
        (Some(p), _) => {
            let y = b.zeros("q3w-moe-ygu", (slots * 2 * inter * 2) as u64);
            push_gemv_nvfp4(
                b,
                s,
                "q3w-moe-gateup",
                &p.w,
                &p.scales,
                &xq,
                &xs,
                &y,
                &ids,
                &p.alphas,
                p.n,
                p.k,
                slots,
                p.n / 2,
                true,
                1.0,
            )?;
            (y.clone(), y)
        }
        (None, Some((g, u))) => {
            let y_gate = b.zeros("q3w-moe-ygate", (slots * inter * 2) as u64);
            let y_up = b.zeros("q3w-moe-yup", (slots * inter * 2) as u64);
            push_gemv_nvfp4(
                b,
                s,
                "q3w-moe-gate",
                &g.w,
                &g.scales,
                &xq,
                &xs,
                &y_gate,
                &ids,
                &g.alphas,
                g.n,
                g.k,
                slots,
                g.n / 2,
                true,
                1.0,
            )?;
            push_gemv_nvfp4(
                b,
                s,
                "q3w-moe-up",
                &u.w,
                &u.scales,
                &xq,
                &xs,
                &y_up,
                &ids,
                &u.alphas,
                u.n,
                u.k,
                slots,
                u.n / 2,
                true,
                1.0,
            )?;
            (y_gate, y_up)
        }
        _ => unreachable!("exactly one of the gate stacks is built"),
    };

    let aq = b.zeros("q3w-moe-aq", (slots * inter / 2) as u64);
    let as_ = b.zeros("q3w-moe-as", (slots * ib) as u64);
    if fuse_silu_quant_enabled() {
        push_silu_mul_quant(
            b,
            s,
            "q3w-moe-siluq",
            &y_gate,
            &y_up,
            &aq,
            &as_,
            &ids,
            &ed.globals,
            inter,
            slots,
            true,
            true,
            fuse_gu,
        )?;
    } else {
        let act = b.zeros("q3w-moe-act", (slots * inter * 2) as u64);
        let smp = b.uni(
            "q3w-moe-silu-p",
            SiluMulParams {
                n_words: (slots * inter / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((slots * inter / 2) as u64, 64);
        b.push(
            "q3w-moe-silu",
            &s.moe,
            "q3w_silu_mul",
            &[(10, &y_gate), (11, &y_up), (12, &act), (13, &smp)],
            grid,
        )?;
        push_quant_rows(
            b,
            s,
            "q3w-moe-aquant",
            &act,
            &aq,
            &as_,
            &ids,
            &ed.globals,
            inter,
            slots,
            true,
            inter,
        )?;
    }

    let y_down = b.zeros("q3w-moe-ydown", (slots * hidden * 2) as u64);
    push_gemv_nvfp4(
        b,
        s,
        "q3w-moe-down",
        &ed.w,
        &ed.scales,
        &aq,
        &as_,
        &y_down,
        &ids,
        &ed.alphas,
        ed.n,
        ed.k,
        slots,
        ed.n / 2,
        true,
        1.0,
    )?;

    let shared_out = if fold {
        y_down.clone()
    } else {
        emit_shared_expert(b, s, m, x, hidden, sinter)?
    };

    let slogit = if gate_fused {
        rlogits.clone()
    } else {
        let sgate = upload_bf16(b, "q3w-moe-sgatew", &m.shared_expert_gate);
        let slogit = b.zeros("q3w-moe-slogit", 4);
        push_gemv_bf16(b, s, "q3w-moe-sgatelin", &sgate, x, &slogit, true, 0)?;
        slogit
    };

    let mcp = b.uni(
        "q3w-moe-comb-p",
        CombineParams {
            hidden_words: hidden_words as u32,
            k: k_top as u32,
            slot_stride_words: hidden_words as u32,
            shared_off_words: if fold {
                (k_top * hidden_words) as u32
            } else {
                0
            },
            slogit_off: if gate_fused { e as u32 } else { 0 },
            ..Default::default()
        },
    );
    let grid = b.grid1(hidden_words as u64, 64);
    b.push(
        "q3w-moe-combine",
        &s.moe,
        "q3w_moe_combine",
        &[
            (20, &y_down),
            (21, &wts),
            (22, &shared_out),
            (23, &slogit),
            (24, out),
            (25, &mcp),
        ],
        grid,
    )?;

    let Some(p) = pf.as_mut().filter(|p| p.ok) else {
        return Ok(());
    };
    b.to_prefill = true;
    let r = (|b: &mut Builder| -> Result<()> {
        anyhow::ensure!(
            fold && gate_fused,
            "M-row prefill covers only the folded-shared-expert + fused-router-gate arm; \
             this layer fell back, so prefill stays on the per-token replay"
        );
        anyhow::ensure!(
            fuse_silu_quant_enabled(),
            "M-row prefill covers only the fused silu+quant arm; \
             NV_Q3_WGPU_FUSE_SILU_QUANT=0 leaves prefill on the per-token replay"
        );
        anyhow::ensure!(
            slots <= PF_MROW_POS_STRIDE_WORDS,
            "the per-token top-k copies write ids/wts into one {PF_MROW_BIND_OFFSET_ALIGN_BYTES}B \
             slot each, which holds at most {PF_MROW_POS_STRIDE_WORDS} entries; got {slots} slots"
        );
        let m_tok = p.m;
        let zslots = m_tok * slots;
        let ygu_bytes = if fuse_gu {
            (zslots * 2 * inter * 2) as u64
        } else {
            (zslots * inter * 2) as u64
        };
        if p.moe.is_none() {
            let rl_stride_bytes = pf_stride_bytes_f32(router.n);
            let (y_gate, y_up) = if fuse_gu {
                let y = b.zeros("q3w-pf-moe-ygu", ygu_bytes);
                (y.clone(), y)
            } else {
                (
                    b.zeros("q3w-pf-moe-ygate", ygu_bytes),
                    b.zeros("q3w-pf-moe-yup", ygu_bytes),
                )
            };
            p.moe = Some(PfMoeBufs {
                rl_stride_bytes,
                rlogits: b.zeros("q3w-pf-moe-rlogits", m_tok as u64 * rl_stride_bytes),
                ids: b.zeros(
                    "q3w-pf-moe-ids",
                    (m_tok * PF_MROW_POS_STRIDE_WORDS * 4) as u64,
                ),
                wts: b.zeros(
                    "q3w-pf-moe-wts",
                    (m_tok * PF_MROW_POS_STRIDE_WORDS * 4) as u64,
                ),
                sel_sorted: b.zeros("q3w-pf-moe-sel-sorted", (zslots * 4).max(4) as u64),
                perm: b.zeros("q3w-pf-moe-perm", (zslots * 4).max(4) as u64),
                sel_flat: b.zeros("q3w-pf-moe-sel", (zslots * 4).max(4) as u64),
                x_rep: b.zeros("q3w-pf-moe-xrep", (zslots * hidden_words * 4) as u64),
                xq: b.zeros("q3w-pf-moe-xq", (zslots * hidden / 2) as u64),
                xs: b.zeros("q3w-pf-moe-xs", (zslots * hb) as u64),
                y_gate,
                y_up,
                aq: b.zeros("q3w-pf-moe-aq", (zslots * inter / 2) as u64),
                as_: b.zeros("q3w-pf-moe-as", (zslots * ib) as u64),
                y_down: b.zeros("q3w-pf-moe-ydown", (zslots * hidden * 2) as u64),
            });
        }
        let mo = p.moe.as_ref().expect("created above");
        anyhow::ensure!(
            mo.y_gate.size() == ygu_bytes && mo.rl_stride_bytes == pf_stride_bytes_f32(router.n),
            "the gate/up fusion arm or router width changed between layers; the shared \
             M-row scratch assumes one arm for every layer"
        );
        pf_push_gemm_bf16_m(
            b,
            p,
            "q3w-pf-moe-router",
            &router,
            &p.normed_post,
            hidden_words,
            &mo.rlogits,
            (mo.rl_stride_bytes / 4) as usize,
            true,
        )?;
        if pf_token_parallel_default_on_escape_to_per_token_copies() {
            let ptkp = b.uni(
                "q3w-pf-moe-topk-p",
                PfTopkParams {
                    tokens: m_tok as u32,
                    rl_stride_words: (mo.rl_stride_bytes / 4) as u32,
                    sel_stride_words: PF_MROW_POS_STRIDE_WORDS as u32,
                    pad0: 0,
                },
            );
            b.push(
                "q3w-pf-moe-topk-m",
                &s.pf_moe,
                "q3w_pf_router_topk_m",
                &[
                    (0, &mo.rlogits),
                    (1, &mo.ids),
                    (2, &mo.wts),
                    (3, &rp),
                    (4, &ptkp),
                ],
                (m_tok as u32, 1, 1),
            )?;
        } else {
            for t in 0..m_tok as u64 {
                b.push_pf_off(
                    "q3w-pf-moe-topk",
                    &s.moe,
                    topk_entry,
                    &[
                        (0, &mo.rlogits, t * mo.rl_stride_bytes),
                        (1, &mo.ids, t * PF_MROW_BIND_OFFSET_ALIGN_BYTES),
                        (2, &mo.wts, t * PF_MROW_BIND_OFFSET_ALIGN_BYTES),
                        (3, &rp, 0),
                    ],
                    (1, 1, 1),
                )?;
            }
        }
        let rep_groups = ((zslots * hidden_words) as u32).div_ceil(256);
        anyhow::ensure!(
            rep_groups <= b.ctx.caps.max_compute_workgroups_per_dimension,
            "the x-replicate glue pass needs {rep_groups} workgroups in one dimension, \
             over the device limit {}",
            b.ctx.caps.max_compute_workgroups_per_dimension
        );
        let repp = b.uni(
            "q3w-pf-moe-repp",
            PfCopyParams {
                rows: zslots as u32,
                row_words: hidden_words as u32,
                src_stride_words: hidden_words as u32,
                slots: slots as u32,
            },
        );
        let selp = b.uni(
            "q3w-pf-moe-selp",
            PfCopyParams {
                rows: m_tok as u32,
                row_words: slots as u32,
                src_stride_words: PF_MROW_POS_STRIDE_WORDS as u32,
                slots: slots as u32,
            },
        );
        b.push(
            "q3w-pf-moe-selpack",
            &s.pf_glue,
            "q3w_pf_pack_padded_ids_to_flat_sel",
            &[(0, &mo.ids), (1, &mo.sel_flat), (2, &selp)],
            ((zslots as u32).div_ceil(64), 1, 1),
        )?;
        b.push(
            "q3w-pf-moe-xrep",
            &s.pf_glue,
            "q3w_pf_replicate_token_row_to_slots",
            &[(0, &p.normed_post), (1, &mo.x_rep), (2, &repp)],
            (rep_groups, 1, 1),
        )?;
        let sel_slots: &wgpu::Buffer = &mo.sel_flat;
        let grouped_on = pf_token_parallel_default_on_escape_to_per_token_copies()
            && pf_moe_grouped_default_on_reuses_expert_weight_loads_across_sorted_slots()
            && e + 1 <= 257;
        if grouped_on {
            let psp = b.uni(
                "q3w-pf-moe-sort-p",
                PfSortParams {
                    tokens: m_tok as u32,
                    slots_per_token: slots as u32,
                    ids_stride_words: PF_MROW_POS_STRIDE_WORDS as u32,
                    bins_cover_n_experts_plus_the_shared_slot: (e + 1) as u32,
                },
            );
            b.push(
                "q3w-pf-moe-sort",
                &s.pf_moe,
                "q3w_pf_group_slots_by_expert_for_weight_load_reuse",
                &[
                    (5, &psp),
                    (6, &mo.ids),
                    (7, &mo.sel_sorted),
                    (8, &mo.perm),
                ],
                (1, 1, 1),
            )?;
        }
        push_quant_rows(
            b,
            s,
            "q3w-pf-moe-xquant",
            &mo.x_rep,
            &mo.xq,
            &mo.xs,
            sel_slots,
            x_globals,
            hidden,
            zslots,
            true,
            hidden,
        )?;
        match (&egu, &split) {
            (Some(pgu), _) => {
                let grouped = grouped_on
                    && pf_push_gemv_nvfp4_grouped(
                        b,
                        "q3w-pf-moe-gateup-g",
                        &pgu.w,
                        &pgu.scales,
                        &mo.xq,
                        &mo.xs,
                        &mo.y_gate,
                        &mo.sel_sorted,
                        &mo.perm,
                        &pgu.alphas,
                        pgu.n,
                        pgu.k,
                        zslots,
                        pgu.n / 2,
                    )?;
                if !grouped {
                    push_gemv_nvfp4(
                        b,
                        s,
                        "q3w-pf-moe-gateup",
                        &pgu.w,
                        &pgu.scales,
                        &mo.xq,
                        &mo.xs,
                        &mo.y_gate,
                        sel_slots,
                        &pgu.alphas,
                        pgu.n,
                        pgu.k,
                        zslots,
                        pgu.n / 2,
                        true,
                        1.0,
                    )?;
                }
            }
            (None, Some((g, u))) => {
                push_gemv_nvfp4(
                    b,
                    s,
                    "q3w-pf-moe-gate",
                    &g.w,
                    &g.scales,
                    &mo.xq,
                    &mo.xs,
                    &mo.y_gate,
                    sel_slots,
                    &g.alphas,
                    g.n,
                    g.k,
                    zslots,
                    g.n / 2,
                    true,
                    1.0,
                )?;
                push_gemv_nvfp4(
                    b,
                    s,
                    "q3w-pf-moe-up",
                    &u.w,
                    &u.scales,
                    &mo.xq,
                    &mo.xs,
                    &mo.y_up,
                    sel_slots,
                    &u.alphas,
                    u.n,
                    u.k,
                    zslots,
                    u.n / 2,
                    true,
                    1.0,
                )?;
            }
            _ => unreachable!("exactly one of the gate stacks is built"),
        }
        push_silu_mul_quant(
            b,
            s,
            "q3w-pf-moe-siluq",
            &mo.y_gate,
            &mo.y_up,
            &mo.aq,
            &mo.as_,
            sel_slots,
            &ed.globals,
            inter,
            zslots,
            true,
            true,
            fuse_gu,
        )?;
        let down_grouped = grouped_on
            && pf_push_gemv_nvfp4_grouped(
                b,
                "q3w-pf-moe-down-g",
                &ed.w,
                &ed.scales,
                &mo.aq,
                &mo.as_,
                &mo.y_down,
                &mo.sel_sorted,
                &mo.perm,
                &ed.alphas,
                ed.n,
                ed.k,
                zslots,
                ed.n / 2,
            )?;
        if !down_grouped {
            push_gemv_nvfp4(
                b,
                s,
                "q3w-pf-moe-down",
                &ed.w,
                &ed.scales,
                &mo.aq,
                &mo.as_,
                &mo.y_down,
                sel_slots,
                &ed.alphas,
                ed.n,
                ed.k,
                zslots,
                ed.n / 2,
                true,
                1.0,
            )?;
        }
        let comb_grid = b.grid1(hidden_words as u64, 64);
        let tok_block_bytes = (slots * hidden * 2) as u64;
        if pf_token_parallel_default_on_escape_to_per_token_copies() {
            let pmcp = b.uni(
                "q3w-pf-moe-comb-m-p",
                PfCombineParams {
                    tokens: m_tok as u32,
                    y_stride_words: (tok_block_bytes / 4) as u32,
                    wts_stride_words: PF_MROW_POS_STRIDE_WORDS as u32,
                    slogit_stride_words: (mo.rl_stride_bytes / 4) as u32,
                    out_stride_words: hidden_words as u32,
                    sel_slots_per_token: slots as u32,
                    ..Default::default()
                },
            );
            b.push(
                "q3w-pf-moe-combine-m",
                &s.pf_moe,
                "q3w_pf_moe_combine_m",
                &[
                    (20, &mo.y_down),
                    (21, &mo.wts),
                    (22, &mo.y_down),
                    (23, &mo.rlogits),
                    (24, &p.moe_out),
                    (25, &mcp),
                    (26, &pmcp),
                ],
                ((hidden_words as u32).div_ceil(64), m_tok as u32, 1),
            )?;
            return Ok(());
        }
        for t in 0..m_tok as u64 {
            b.push_pf_off(
                "q3w-pf-moe-combine",
                &s.moe,
                "q3w_moe_combine",
                &[
                    (20, &mo.y_down, t * tok_block_bytes),
                    (21, &mo.wts, t * PF_MROW_BIND_OFFSET_ALIGN_BYTES),
                    (22, &mo.y_down, t * tok_block_bytes),
                    (23, &mo.rlogits, t * mo.rl_stride_bytes),
                    (24, &p.moe_out, t * (hidden * 2) as u64),
                    (25, &mcp, 0),
                ],
                comb_grid,
            )?;
        }
        Ok(())
    })(b);
    b.to_prefill = false;
    if let Err(e) = r {
        p.ok = false;
        p.off_reason = Some(format!("moe layer: {e}"));
    }
    Ok(())
}

fn emit_shared_expert(
    b: &mut Builder,
    s: &Sources,
    m: &HostMoe,
    x: &wgpu::Buffer,
    hidden: usize,
    sinter: usize,
) -> Result<wgpu::Buffer> {
    let hb = hidden / NVFP4_BLOCK;
    let sel0 = b.upload_u32("q3w-moe-sel0", &[0u32]);
    let adummy = b.upload_f32("q3w-moe-adummy", &[1.0f32]);
    let sg = upload_nvfp4(b, "q3w-moe-sg", &m.shared_gate);
    let su = upload_nvfp4(b, "q3w-moe-su", &m.shared_up);
    let sd = upload_nvfp4(b, "q3w-moe-sd", &m.shared_down);

    let sxq = b.zeros("q3w-moe-sxq", (hidden / 2) as u64);
    let sxs = b.zeros("q3w-moe-sxs", (hb.div_ceil(4) * 4) as u64);
    let sglob = b.upload_f32("q3w-moe-sglob", &[sg.input_global]);
    push_quant_rows(
        b,
        s,
        "q3w-moe-sxquant",
        x,
        &sxq,
        &sxs,
        &sel0,
        &sglob,
        hidden,
        1,
        false,
        0,
    )?;
    let sy_g = b.zeros("q3w-moe-syg", (sinter * 2) as u64);
    let sy_u = b.zeros("q3w-moe-syu", (sinter * 2) as u64);
    push_gemv_nvfp4(
        b,
        s,
        "q3w-moe-sgate",
        &sg.w,
        &sg.scales,
        &sxq,
        &sxs,
        &sy_g,
        &sel0,
        &adummy,
        sg.n,
        sg.k,
        1,
        sg.n / 2,
        false,
        sg.alpha,
    )?;
    push_gemv_nvfp4(
        b,
        s,
        "q3w-moe-sup",
        &su.w,
        &su.scales,
        &sxq,
        &sxs,
        &sy_u,
        &sel0,
        &adummy,
        su.n,
        su.k,
        1,
        su.n / 2,
        false,
        su.alpha,
    )?;
    let sib = sinter / NVFP4_BLOCK;
    let saq = b.zeros("q3w-moe-saq", (sinter / 2) as u64);
    let sas = b.zeros("q3w-moe-sas", (sib.div_ceil(4) * 4) as u64);
    let sdglob = b.upload_f32("q3w-moe-sdglob", &[sd.input_global]);
    if fuse_silu_quant_enabled() {
        push_silu_mul_quant(
            b,
            s,
            "q3w-moe-ssiluq",
            &sy_g,
            &sy_u,
            &saq,
            &sas,
            &sel0,
            &sdglob,
            sinter,
            1,
            false,
            false,
            false,
        )?;
    } else {
        let sact = b.zeros("q3w-moe-sact", (sinter * 2) as u64);
        let ssp = b.uni(
            "q3w-moe-ssilu-p",
            SiluMulParams {
                n_words: (sinter / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((sinter / 2) as u64, 64);
        b.push(
            "q3w-moe-ssilu",
            &s.moe,
            "q3w_silu_mul",
            &[(10, &sy_g), (11, &sy_u), (12, &sact), (13, &ssp)],
            grid,
        )?;
        push_quant_rows(
            b,
            s,
            "q3w-moe-saquant",
            &sact,
            &saq,
            &sas,
            &sel0,
            &sdglob,
            sinter,
            1,
            false,
            0,
        )?;
    }
    let shared_out = b.zeros("q3w-moe-sout", (hidden * 2) as u64);
    push_gemv_nvfp4(
        b,
        s,
        "q3w-moe-sdown",
        &sd.w,
        &sd.scales,
        &saq,
        &sas,
        &shared_out,
        &sel0,
        &adummy,
        sd.n,
        sd.k,
        1,
        sd.n / 2,
        false,
        sd.alpha,
    )?;
    Ok(shared_out)
}

impl WeightSource<'_> {
    fn embed(&self, cfg: &Qwen3MoeConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.embed.clone()),
            Self::Loader(w) => load_bf16(
                w,
                &[
                    "model.language_model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                ],
                &[cfg.vocab_size, cfg.hidden_size],
            ),
        }
    }

    fn lm_head(&self, cfg: &Qwen3MoeConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.lm_head.clone()),
            Self::Loader(w) => {
                if cfg.tie_word_embeddings {
                    self.embed(cfg)
                } else {
                    load_bf16(w, &["lm_head.weight"], &[cfg.vocab_size, cfg.hidden_size])
                }
            }
        }
    }

    fn final_norm(&self, cfg: &Qwen3MoeConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.final_norm.clone()),
            Self::Loader(w) => load_norm_plus_one(
                w,
                &["model.language_model.norm.weight", "model.norm.weight"],
                cfg.hidden_size,
            ),
        }
    }

    fn layer_input_ln(&self, cfg: &Qwen3MoeConfig, idx: usize) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].input_ln.clone()),
            Self::Loader(w) => load_norm_plus_one(
                w,
                &[
                    &format!("model.language_model.layers.{idx}.input_layernorm.weight"),
                    &format!("model.layers.{idx}.input_layernorm.weight"),
                ],
                cfg.hidden_size,
            ),
        }
    }

    fn layer(&self, cfg: &Qwen3MoeConfig, idx: usize) -> Result<HostLayer> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].clone()),
            Self::Loader(w) => load_layer(cfg, w, idx),
        }
    }
}

fn load_bf16(w: &nv_weights::WeightLoader, names: &[&str], shape: &[usize]) -> Result<Vec<u16>> {
    for n in names {
        if w.has(n) {
            let t = w
                .get(n, candle_core::DType::BF16)
                .with_context(|| format!("load {n}"))?;
            anyhow::ensure!(t.dims() == shape, "{n}: shape {:?} != {shape:?}", t.dims());
            let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
            return Ok(v.into_iter().map(|x| x.to_bits()).collect());
        }
    }
    anyhow::bail!("none of {names:?} found")
}

pub fn norm_plus_one_enabled() -> bool {
    !matches!(
        std::env::var("NV_QWEN36_NORM_PLUS_ONE").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

fn load_norm_plus_one(
    w: &nv_weights::WeightLoader,
    names: &[&str],
    dim: usize,
) -> Result<Vec<u16>> {
    let raw = load_bf16(w, names, &[dim])?;
    if !norm_plus_one_enabled() {
        return Ok(raw);
    }
    Ok(raw
        .into_iter()
        .map(|b| bf16_bits(bf16_val(b) + 1.0))
        .collect())
}

fn load_f32_vec(w: &nv_weights::WeightLoader, names: &[&str], dim: usize) -> Result<Vec<f32>> {
    let raw = load_bf16(w, names, &[dim])?;
    Ok(raw.into_iter().map(bf16_val).collect())
}

pub fn load_nvfp4(
    w: &nv_weights::WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<HostNvfp4Lin> {
    let packed_name = format!("{module}.weight_packed");
    let shape = w
        .shape_of(&packed_name)
        .ok_or_else(|| anyhow::anyhow!("missing {packed_name}"))?;
    anyhow::ensure!(
        shape.len() == 2 && shape[0] == n && shape[1] == k / 2,
        "{module}: weight_packed shape {shape:?}, want [{n}, {}]",
        k / 2
    );
    let packed = w.raw_bytes(&packed_name)?.to_vec();
    let scale_raw = w.raw_bytes(&format!("{module}.weight_scale"))?.to_vec();
    anyhow::ensure!(
        scale_raw.len() == n * k / NVFP4_BLOCK,
        "{module}: weight_scale {} bytes, want {}",
        scale_raw.len(),
        n * k / NVFP4_BLOCK
    );
    let scales = nv_quant::nvfp4::swizzle_scales(&scale_raw, n, k / NVFP4_BLOCK);
    let gw = scalar_f32(w, &format!("{module}.weight_global_scale"))?;
    let gi = if w.has(&format!("{module}.input_global_scale")) {
        scalar_f32(w, &format!("{module}.input_global_scale"))?
    } else {
        1.0
    };
    let recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };
    Ok(HostNvfp4Lin {
        packed,
        scales_swizzled: scales,
        alpha: recip(gw) * recip(gi),
        input_global: gi,
        n,
        k,
    })
}

fn scalar_f32(w: &nv_weights::WeightLoader, name: &str) -> Result<f32> {
    let t = w.get(name, candle_core::DType::F32)?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(*v.first().unwrap_or(&1.0))
}

fn load_layer(cfg: &Qwen3MoeConfig, w: &nv_weights::WeightLoader, idx: usize) -> Result<HostLayer> {
    let prefix = format!("model.language_model.layers.{idx}");
    let alt = format!("model.layers.{idx}");
    let input_ln = load_norm_plus_one(
        w,
        &[
            &format!("{prefix}.input_layernorm.weight"),
            &format!("{alt}.input_layernorm.weight"),
        ],
        cfg.hidden_size,
    )?;
    let post_attn_ln = load_norm_plus_one(
        w,
        &[
            &format!("{prefix}.post_attention_layernorm.weight"),
            &format!("{alt}.post_attention_layernorm.weight"),
        ],
        cfg.hidden_size,
    )?;

    let mixer = match cfg.layer_types[idx] {
        LayerType::LinearAttention => {
            let p = format!("{prefix}.linear_attn");
            let n_k = cfg.linear_num_key_heads;
            let n_v = cfg.linear_num_value_heads;
            let d_k = cfg.linear_key_head_dim;
            let d_v = cfg.linear_value_head_dim;
            let key_dim = n_k * d_k;
            let value_dim = n_v * d_v;
            let conv_dim = 2 * key_dim + value_dim;
            let ks = cfg.linear_conv_kernel_dim;
            let a_w = load_bf16(
                w,
                &[&format!("{p}.in_proj_a.weight")],
                &[n_v, cfg.hidden_size],
            )?;
            let b_w = load_bf16(
                w,
                &[&format!("{p}.in_proj_b.weight")],
                &[n_v, cfg.hidden_size],
            )?;
            let mut ab = a_w;
            ab.extend_from_slice(&b_w);
            let conv_raw = load_bf16(w, &[&format!("{p}.conv1d.weight")], &[conv_dim, 1, ks])?;
            HostMixer::Delta(Box::new(HostDeltaNet {
                in_proj_qkv: HostBf16Lin {
                    w: load_bf16(
                        w,
                        &[&format!("{p}.in_proj_qkv.weight")],
                        &[conv_dim, cfg.hidden_size],
                    )?,
                    n: conv_dim,
                    k: cfg.hidden_size,
                },
                in_proj_z: HostBf16Lin {
                    w: load_bf16(
                        w,
                        &[&format!("{p}.in_proj_z.weight")],
                        &[value_dim, cfg.hidden_size],
                    )?,
                    n: value_dim,
                    k: cfg.hidden_size,
                },
                in_proj_ab: HostBf16Lin {
                    w: ab,
                    n: 2 * n_v,
                    k: cfg.hidden_size,
                },
                conv1d: conv_raw.into_iter().map(bf16_val).collect(),
                a_log: load_f32_vec(w, &[&format!("{p}.A_log")], n_v)?,
                dt_bias: load_f32_vec(w, &[&format!("{p}.dt_bias")], n_v)?,
                norm_w: load_bf16(w, &[&format!("{p}.norm.weight")], &[d_v])?,
                out_proj: HostBf16Lin {
                    w: load_bf16(
                        w,
                        &[&format!("{p}.out_proj.weight")],
                        &[cfg.hidden_size, value_dim],
                    )?,
                    n: cfg.hidden_size,
                    k: value_dim,
                },
            }))
        }
        LayerType::FullAttention => {
            let p = format!("{prefix}.self_attn");
            let hd = cfg.head_dim;
            let q_out = if cfg.attn_output_gate {
                cfg.num_attention_heads * hd * 2
            } else {
                cfg.num_attention_heads * hd
            };
            let kv_out = cfg.num_key_value_heads * hd;
            HostMixer::Attn(Box::new(HostAttention {
                q: load_nvfp4(w, &format!("{p}.q_proj"), q_out, cfg.hidden_size)?,
                k: load_nvfp4(w, &format!("{p}.k_proj"), kv_out, cfg.hidden_size)?,
                v: load_nvfp4(w, &format!("{p}.v_proj"), kv_out, cfg.hidden_size)?,
                o: load_nvfp4(
                    w,
                    &format!("{p}.o_proj"),
                    cfg.hidden_size,
                    cfg.num_attention_heads * hd,
                )?,
                q_norm: load_norm_plus_one(w, &[&format!("{p}.q_norm.weight")], hd)?,
                k_norm: load_norm_plus_one(w, &[&format!("{p}.k_norm.weight")], hd)?,
            }))
        }
    };

    let mp = format!("{prefix}.mlp");
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
    let mut gates = Vec::with_capacity(cfg.num_experts);
    let mut ups = Vec::with_capacity(cfg.num_experts);
    let mut downs = Vec::with_capacity(cfg.num_experts);
    for e in 0..cfg.num_experts {
        gates.push(load_nvfp4(
            w,
            &format!("{mp}.experts.{e}.gate_proj"),
            inter,
            cfg.hidden_size,
        )?);
        ups.push(load_nvfp4(
            w,
            &format!("{mp}.experts.{e}.up_proj"),
            inter,
            cfg.hidden_size,
        )?);
        downs.push(load_nvfp4(
            w,
            &format!("{mp}.experts.{e}.down_proj"),
            cfg.hidden_size,
            inter,
        )?);
    }

    Ok(HostLayer {
        input_ln,
        post_attn_ln,
        mixer,
        moe: HostMoe {
            router: HostBf16Lin {
                w: load_bf16(
                    w,
                    &[&format!("{mp}.gate.weight")],
                    &[cfg.num_experts, cfg.hidden_size],
                )?,
                n: cfg.num_experts,
                k: cfg.hidden_size,
            },
            experts_gate: stack_nvfp4_host(&gates),
            experts_up: stack_nvfp4_host(&ups),
            experts_down: stack_nvfp4_host(&downs),
            shared_gate: load_nvfp4(
                w,
                &format!("{mp}.shared_expert.gate_proj"),
                sinter,
                cfg.hidden_size,
            )?,
            shared_up: load_nvfp4(
                w,
                &format!("{mp}.shared_expert.up_proj"),
                sinter,
                cfg.hidden_size,
            )?,
            shared_down: load_nvfp4(
                w,
                &format!("{mp}.shared_expert.down_proj"),
                cfg.hidden_size,
                sinter,
            )?,
            shared_expert_gate: HostBf16Lin {
                w: load_bf16(
                    w,
                    &[&format!("{mp}.shared_expert_gate.weight")],
                    &[1, cfg.hidden_size],
                )?,
                n: 1,
                k: cfg.hidden_size,
            },
        },
    })
}

fn rbf(x: f32) -> f32 {
    bf16_val(bf16_bits(x))
}

fn ref_div_rn(a: f32, b: f32) -> f32 {
    let (bb, post) = if b.abs() > 1.0e37 {
        (b * 0.003_906_25, 0.003_906_25)
    } else {
        (b, 1.0)
    };
    let r = 1.0 / bb;
    let q = a * r;
    (-q).mul_add(bb, a).mul_add(r, q) * post
}

fn ref_e4m3_encode(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 31) << 7;
    let mag = bits & 0x7fff_ffff;
    if mag > 0x7f80_0000 {
        return sign | 0x7f;
    }
    let e = (mag >> 23) as i32 - 127;
    if e >= -6 {
        let lsb = (mag >> 20) & 1;
        let r = mag + 0x7_ffff + lsb;
        let e2 = (r >> 23) as i32 - 127;
        let m2 = (r >> 20) & 7;
        if e2 > 8 || (e2 == 8 && m2 == 7) {
            return sign | 0x7e;
        }
        return sign | (((e2 + 7) as u32) << 3) | m2;
    }
    let s = (14 - e) as u32;
    if s >= 32 {
        return sign;
    }
    let full = 0x80_0000u32 | (mag & 0x7f_ffff);
    let q = full >> s;
    let round_bit = (full >> (s - 1)) & 1;
    let rest = full & ((1u32 << (s - 1)) - 1);
    let mut n = q;
    if round_bit == 1 && (rest != 0 || (q & 1) == 1) {
        n += 1;
    }
    sign | n
}

fn ref_e4m3_decode(b: u32) -> f32 {
    let b = b & 255;
    let e = (b >> 3) & 15;
    let m = b & 7;
    let mag = if e == 0 {
        m as f32 * 0.001_953_125
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if (b & 128) != 0 {
        -mag
    } else {
        mag
    }
}

fn ref_kv_fp8(rows: &[f32], n_kv: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        let row = &rows[h * hd..(h + 1) * hd];
        let mut amax = 0f32;
        for v in row {
            amax = amax.max(v.abs());
        }
        let (scale, inv) = if amax > 0.0 {
            (ref_div_rn(amax, 448.0), ref_div_rn(448.0, amax))
        } else {
            (1.0, 1.0)
        };
        for (i, v) in row.iter().enumerate() {
            out[h * hd + i] = ref_e4m3_decode(ref_e4m3_encode(v * inv)) * scale;
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn ref_gemv_bf16(w: &HostBf16Lin, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; w.n];
    for r in 0..w.n {
        let mut acc = 0f32;
        for c in 0..w.k {
            acc += bf16_val(w.w[r * w.k + c]) * x[c];
        }
        y[r] = acc;
    }
    y
}

pub fn ref_quant_x(x: &[f32], global: f32) -> Vec<f32> {
    let row: Vec<f32> = x.to_vec();
    let t = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&[row], global);
    t.dequantize().remove(0)
}

pub fn ref_gemv_nvfp4(lin: &HostNvfp4Lin, x_eff: &[f32]) -> Vec<f32> {
    let w = dequantize_nvfp4_host(lin);
    let mut y = vec![0f32; lin.n];
    for r in 0..lin.n {
        let mut acc = 0f32;
        for c in 0..lin.k {
            acc += w[r * lin.k + c] * x_eff[c];
        }
        y[r] = acc;
    }
    y
}

fn ref_rmsnorm(x: &[f32], w: &[u16], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| rbf(x[i] * inv * bf16_val(w[i]))).collect()
}

pub struct RefState {
    conv: Vec<Vec<f32>>,
    rec: Vec<Vec<f32>>,
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
}

impl RefState {
    pub fn new(cfg: &Qwen3MoeConfig) -> Self {
        let l = cfg.num_hidden_layers;
        Self {
            conv: vec![Vec::new(); l],
            rec: vec![Vec::new(); l],
            kc: vec![Vec::new(); l],
            vc: vec![Vec::new(); l],
            pos: 0,
        }
    }
}

pub fn reference_step(
    cfg: &Qwen3MoeConfig,
    hw: &HostWeights,
    st: &mut RefState,
    token: u32,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let pos = st.pos;

    let mut res: Vec<f32> = (0..hidden)
        .map(|i| bf16_val(hw.embed[token as usize * hidden + i]))
        .collect();

    for li in 0..cfg.num_hidden_layers {
        let layer = &hw.layers[li];
        let normed = ref_rmsnorm(&res, &layer.input_ln, eps);
        let mixed = match &layer.mixer {
            HostMixer::Delta(d) => ref_delta(cfg, d, &normed, st, li),
            HostMixer::Attn(a) => ref_attn(cfg, a, &normed, st, li, pos)?,
        };
        for i in 0..hidden {
            res[i] = rbf(res[i] + mixed[i]);
        }
        let normed_post = ref_rmsnorm(&res, &layer.post_attn_ln, eps);
        let moe_out = ref_moe(cfg, &layer.moe, &normed_post)?;
        for i in 0..hidden {
            res[i] = rbf(res[i] + moe_out[i]);
        }
    }

    let fx = ref_rmsnorm(&res, &hw.final_norm, eps);
    let lm = HostBf16Lin {
        w: hw.lm_head.clone(),
        n: cfg.vocab_size,
        k: hidden,
    };
    st.pos += 1;
    Ok(ref_gemv_bf16(&lm, &fx))
}

fn ref_delta(
    cfg: &Qwen3MoeConfig,
    d: &HostDeltaNet,
    x: &[f32],
    st: &mut RefState,
    li: usize,
) -> Vec<f32> {
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;
    let hist = ks - 1;

    let qkv: Vec<f32> = ref_gemv_bf16(&d.in_proj_qkv, x)
        .into_iter()
        .map(rbf)
        .collect();
    let z: Vec<f32> = ref_gemv_bf16(&d.in_proj_z, x)
        .into_iter()
        .map(rbf)
        .collect();
    let ab: Vec<f32> = ref_gemv_bf16(&d.in_proj_ab, x)
        .into_iter()
        .map(rbf)
        .collect();

    if st.conv[li].is_empty() {
        st.conv[li] = vec![0f32; conv_dim * hist];
    }
    let mut mixed = vec![0f32; conv_dim];
    for c in 0..conv_dim {
        let mut acc = 0f32;
        for j in 0..hist {
            acc += d.conv1d[c * ks + j] * st.conv[li][c * hist + j];
        }
        acc += d.conv1d[c * ks + hist] * qkv[c];
        for j in 0..hist.saturating_sub(1) {
            st.conv[li][c * hist + j] = st.conv[li][c * hist + j + 1];
        }
        if hist > 0 {
            st.conv[li][c * hist + hist - 1] = qkv[c];
        }
        mixed[c] = rbf(silu(acc));
    }

    let v_per_k = n_v / n_k;
    let scale = 1.0 / (d_k as f32).sqrt();
    let mut qg = vec![0f32; n_v * d_k];
    let mut kg = vec![0f32; n_v * d_k];
    let mut vg = vec![0f32; n_v * d_v];
    for h in 0..n_v {
        let kh = h / v_per_k;
        let mut sq = 0f32;
        let mut sk = 0f32;
        for i in 0..d_k {
            let qv = mixed[kh * d_k + i];
            let kv = mixed[key_dim + kh * d_k + i];
            sq += qv * qv;
            sk += kv * kv;
        }
        let nq = (sq + 1e-6).sqrt();
        let nk = (sk + 1e-6).sqrt();
        for i in 0..d_k {
            qg[h * d_k + i] = (mixed[kh * d_k + i] / nq) * scale;
            kg[h * d_k + i] = mixed[key_dim + kh * d_k + i] / nk;
        }
        for i in 0..d_v {
            vg[h * d_v + i] = mixed[2 * key_dim + h * d_v + i];
        }
    }

    let mut gexp = vec![0f32; n_v];
    let mut beta = vec![0f32; n_v];
    for i in 0..n_v {
        let a = ab[i];
        let bb = ab[n_v + i];
        beta[i] = sigmoid(bb);
        let t = a + d.dt_bias[i];
        let sp = t.max(0.0) + (1.0 + (-t.abs()).exp()).ln();
        gexp[i] = (sp * -(d.a_log[i].exp())).exp();
    }

    if st.rec[li].is_empty() {
        st.rec[li] = vec![0f32; n_v * d_k * d_v];
    }
    let mut core = vec![0f32; n_v * d_v];
    for h in 0..n_v {
        let base = h * d_k * d_v;
        for dv in 0..d_v {
            let mut kv_mem = 0f32;
            for dk in 0..d_k {
                let s = st.rec[li][base + dk * d_v + dv] * gexp[h];
                st.rec[li][base + dk * d_v + dv] = s;
                kv_mem += s * kg[h * d_k + dk];
            }
            let delta = (vg[h * d_v + dv] - kv_mem) * beta[h];
            let mut outv = 0f32;
            for dk in 0..d_k {
                let s = st.rec[li][base + dk * d_v + dv] + kg[h * d_k + dk] * delta;
                st.rec[li][base + dk * d_v + dv] = s;
                outv += s * qg[h * d_k + dk];
            }
            core[h * d_v + dv] = outv;
        }
    }

    let mut gated = vec![0f32; value_dim];
    for h in 0..n_v {
        let mut ss = 0f32;
        for i in 0..d_v {
            let c = rbf(core[h * d_v + i]);
            ss += c * c;
        }
        let inv = 1.0 / (ss / d_v as f32 + cfg.rms_norm_eps as f32).sqrt();
        for i in 0..d_v {
            let c = rbf(core[h * d_v + i]);
            let n = rbf(c * inv * bf16_val(d.norm_w[i]));
            let g = rbf(silu(z[h * d_v + i]));
            gated[h * d_v + i] = rbf(n * g);
        }
    }
    ref_gemv_bf16(&d.out_proj, &gated)
        .into_iter()
        .map(rbf)
        .collect()
}

fn ref_attn(
    cfg: &Qwen3MoeConfig,
    a: &HostAttention,
    x: &[f32],
    st: &mut RefState,
    li: usize,
    pos: usize,
) -> Result<Vec<f32>> {
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let rot = cfg.rotary_dim();
    let rh = rot / 2;
    let eps = cfg.rms_norm_eps as f32;

    let xq = ref_quant_x(x, a.q.input_global);
    let q_raw: Vec<f32> = ref_gemv_nvfp4(&a.q, &xq).into_iter().map(rbf).collect();
    let k_raw: Vec<f32> = ref_gemv_nvfp4(&a.k, &xq).into_iter().map(rbf).collect();
    let v_raw: Vec<f32> = ref_gemv_nvfp4(&a.v, &xq).into_iter().map(rbf).collect();

    let stride_q = if cfg.attn_output_gate { 2 * hd } else { hd };
    let rope = |vals: &mut [f32], p: usize| {
        let src: Vec<f32> = vals.to_vec();
        for i in 0..rh {
            let inv = 1.0f32 / cfg.rope_theta.powf((i as f32 * 2.0) / rot as f32);
            let th = (p as f32) * inv;
            let (c, s) = (th.cos(), th.sin());
            vals[i] = src[i] * c - src[i + rh] * s;
            vals[i + rh] = src[i] * s + src[i + rh] * c;
        }
    };

    let mut q = vec![0f32; n_h * hd];
    for h in 0..n_h {
        let row: Vec<f32> = (0..hd).map(|i| q_raw[h * stride_q + i]).collect();
        let mut n = ref_rmsnorm(&row, &a.q_norm, eps);
        rope(&mut n, pos);
        for i in 0..hd {
            q[h * hd + i] = rbf(n[i]);
        }
    }
    let mut kk = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        let row: Vec<f32> = (0..hd).map(|i| k_raw[h * hd + i]).collect();
        let mut n = ref_rmsnorm(&row, &a.k_norm, eps);
        rope(&mut n, pos);
        for i in 0..hd {
            kk[h * hd + i] = rbf(n[i]);
        }
    }

    st.kc[li].extend_from_slice(&ref_kv_fp8(&kk, n_kv, hd));
    st.vc[li].extend_from_slice(&ref_kv_fp8(&v_raw, n_kv, hd));
    let total = pos + 1;
    let group = n_h / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut out = vec![0f32; n_h * hd];
    for h in 0..n_h {
        let kv = h / group;
        let mut scores = vec![0f32; total];
        let mut m = f32::NEG_INFINITY;
        for t in 0..total {
            let base = (t * n_kv + kv) * hd;
            let mut dot = 0f32;
            for i in 0..hd {
                dot += st.kc[li][base + i] * q[h * hd + i];
            }
            scores[t] = dot * scale;
            m = m.max(scores[t]);
        }
        let mut z = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            z += *s;
        }
        for i in 0..hd {
            let mut acc = 0f32;
            for (t, s) in scores.iter().enumerate() {
                acc += s * st.vc[li][(t * n_kv + kv) * hd + i];
            }
            out[h * hd + i] = acc / z;
        }
    }

    let mut gated = vec![0f32; n_h * hd];
    for h in 0..n_h {
        for i in 0..hd {
            let av = rbf(out[h * hd + i]);
            if cfg.attn_output_gate {
                let g = q_raw[h * stride_q + hd + i];
                gated[h * hd + i] = av * rbf(sigmoid(g));
            } else {
                gated[h * hd + i] = av;
            }
        }
    }
    let oq = ref_quant_x(
        &gated.iter().copied().map(rbf).collect::<Vec<f32>>(),
        a.o.input_global,
    );
    Ok(ref_gemv_nvfp4(&a.o, &oq).into_iter().map(rbf).collect())
}

fn ref_moe(cfg: &Qwen3MoeConfig, m: &HostMoe, x: &[f32]) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
    let k_top = cfg.num_experts_per_tok;

    let logits = ref_gemv_bf16(&m.router, x);
    let mut order: Vec<usize> = (0..cfg.num_experts).collect();
    order.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let ids: Vec<usize> = order[..k_top].to_vec();
    let mut wts: Vec<f32> = ids.iter().map(|e| logits[*e]).collect();
    let mx = wts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut zsum = 0f32;
    for w in wts.iter_mut() {
        *w = (*w - mx).exp();
        zsum += *w;
    }
    for w in wts.iter_mut() {
        *w /= zsum;
    }

    let mut acc = vec![0f32; hidden];
    for (j, e) in ids.iter().enumerate() {
        let gate = expert_slice(&m.experts_gate, *e);
        let up = expert_slice(&m.experts_up, *e);
        let down = expert_slice(&m.experts_down, *e);
        let xq = ref_quant_x(x, gate.input_global);
        let yg = ref_gemv_nvfp4(&gate, &xq);
        let yu = ref_gemv_nvfp4(&up, &xq);
        let act: Vec<f32> = (0..inter)
            .map(|i| rbf(rbf(silu(rbf(yg[i]))) * rbf(yu[i])))
            .collect();
        let aq = ref_quant_x(&act, down.input_global);
        let yd = ref_gemv_nvfp4(&down, &aq);
        for i in 0..hidden {
            acc[i] += rbf(yd[i]) * wts[j];
        }
    }

    let sxq = ref_quant_x(x, m.shared_gate.input_global);
    let sg = ref_gemv_nvfp4(&m.shared_gate, &sxq);
    let su = ref_gemv_nvfp4(&m.shared_up, &sxq);
    let sact: Vec<f32> = (0..sinter)
        .map(|i| rbf(rbf(silu(rbf(sg[i]))) * rbf(su[i])))
        .collect();
    let saq = ref_quant_x(&sact, m.shared_down.input_global);
    let sy = ref_gemv_nvfp4(&m.shared_down, &saq);
    let slogit = ref_gemv_bf16(&m.shared_expert_gate, x)[0];
    let sgate = sigmoid(slogit);
    for i in 0..hidden {
        acc[i] = rbf(acc[i] + sgate * rbf(sy[i]));
    }
    Ok(acc)
}

crate::wgpu_state_snapshot::impl_wgpu_state_snapshot!(Qwen3MoeWgpu, max_seq);
