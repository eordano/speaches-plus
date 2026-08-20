use anyhow::Result;

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};
use nv_quant::mxfp4::BLOCK_SIZE as MX_BLOCK;

pub use crate::gpt_oss::{
    reference_step, rope_tables, stack_mx_host, yarn_inv_freq, GptOssConfig, GptOssLayerType,
    HostAttn, HostBf16Lin, HostLayer, HostMoe, HostMxStack, HostWeights, RefState, SWIGLU_ALPHA,
};
pub use crate::qwen3_5_dense_wgpu::ImageRowSplice;

use crate::gpt_oss::{bf16_val, load_bf16, load_layer};
use crate::gemma4_wgpu_shared::pack_pairs;
use crate::wgpu_ledger::VramLedger;
const MAX_HEAD_DIM: usize = 256;
const MAX_TOPK: usize = 16;
const ARGMAX_GROUPS: usize = 256;
const STAGING_FLUSH_BYTES: u64 = 256 << 20;

const GEMV_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_gemv.wgsl");

const MX_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_mx.wgsl");

const MX_SG_SECTION_SPLIT: &str = "const GOW_MX_SECTION_SG";
pub const MX_SCALAR_ENTRY: &str = "gow_gemv_mx";
pub const MX_SG_ENTRY: &str = "gow_gemv_mx_sg";
const MX_SCALAR_ROWS_PER_WG: u32 = 2;
const MX_SG_ROWS_PER_WG_AT_SG32_OVER_WG256: u32 = 8;

pub const MX_SG_GATE_ENV: &str = "NV_GPTOSS_WGPU_MX_SG_REORDERED_SUM";

pub fn mx_sg_reordered_sum_opt_in() -> bool {
    matches!(
        std::env::var(MX_SG_GATE_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

fn mx_scalar_section() -> &'static str {
    MX_WGSL
        .split_once(MX_SG_SECTION_SPLIT)
        .map_or(MX_WGSL, |(head, _)| head)
}

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_attn.wgsl");

const FLASH_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_flash_decode.wgsl");

pub const GPTOSS_WGPU_FLASH_DECODE_DEFAULT_ON: bool = false;

pub const FLASH_DECODE_ENV: &str = "NV_GPTOSS_WGPU_FLASH_DECODE";

pub const FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X: u32 = 64;

const FLASH_WORKGROUP_BYTES_QSH256_RED256_SM8_SL8_SACC2048_F32: u32 =
    (256 + 256 + 8 + 8 + 2048) * 4;

const KVSHARE_STAGE1_IS_SPECIALIZED_TO_HD_64: usize = 64;

const KVSHARE_STAGE1_IS_SPECIALIZED_TO_GROUP_8: usize = 8;

pub const FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH: u32 = 256;

pub fn flash_decode_enabled() -> bool {
    match std::env::var(FLASH_DECODE_ENV) {
        Ok(v) => matches!(v.trim(), "1" | "on" | "true"),
        Err(_) => GPTOSS_WGPU_FLASH_DECODE_DEFAULT_ON,
    }
}

pub fn flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build() -> u32 {
    match std::env::var("NV_GPTOSS_WGPU_FLASH_SPLITS") {
        Ok(v) => {
            let n: u32 = v.trim().parse().unwrap_or_else(|_| {
                panic!("NV_GPTOSS_WGPU_FLASH_SPLITS={v}: expected an integer split count")
            });
            assert!(
                n >= 1 && n <= FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
                "NV_GPTOSS_WGPU_FLASH_SPLITS={n} out of 1..={}: stage2 scans splits serially per \
                 head and the scratch buffer carries splits*(head_dim+2) f32 rows per head",
                FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH
            );
            n
        }
        Err(_) => FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X,
    }
}

pub const KV_FP8_ENV: &str = "NV_GOW_KV_FP8";

pub const GOW_KV_FP8_DEFAULT_ON_THE_DEPTH_STACK_DOMINATES_SERIAL_AT_EVERY_MEASURED_DEPTH: bool = true;

pub fn kv_fp8_enabled() -> bool {
    match std::env::var(KV_FP8_ENV) {
        Ok(v) => !matches!(v.trim(), "0" | "off" | "false"),
        Err(_) => GOW_KV_FP8_DEFAULT_ON_THE_DEPTH_STACK_DOMINATES_SERIAL_AT_EVERY_MEASURED_DEPTH,
    }
}

const PF_KVQ_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_kvq.wgsl");

const MOE_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_moe.wgsl");

const PREFILL_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_prefill.wgsl");

const SPLICE_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_splice.wgsl");

const VERIFY_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_verify.wgsl");

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
    has_bias: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    y_slot_stride: u32,
    use_sel: u32,
    pad0: u32,
    pad1: u32,
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
struct RopeParams {
    n_rows: u32,
    head_dim: u32,
    rot_half: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvWriteParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    pad0: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashDecodeParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    splits: u32,
    total: u32,
    start: u32,
    pad0: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SharedFdParams {
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
struct PfKvqParams {
    tokens: u32,
    x_stride_elems: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PackParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RouterParams {
    n_experts: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SwigluParams {
    n_words: u32,
    inter_words: u32,
    two_inter: u32,
    pad0: u32,
    limit: f32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride: u32,
    pad0: u32,
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

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CkParams {
    m_live: u32,
    base: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGemvBf16Params {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_row_words: u32,
    y_row_words: u32,
    has_bias: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGemvMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    x_tok_stride_words: u32,
    y_slot_stride: u32,
    k_top: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfRouterParams {
    n_experts: u32,
    k: u32,
    m: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfCombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride: u32,
    m: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SpliceParams {
    hidden_words: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VerifyLmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_row_words: u32,
    y_row_words: u32,
    has_bias: u32,
    alpha: f32,
    y_off_words: u32,
    pad0: u32,
    pad1: u32,
}

pub enum WeightSource<'a> {
    Host(&'a HostWeights),
    Loader(&'a nv_weights::WeightLoader),
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    nv_kernels::wgpu_backend::pack::pack_u8_words_padded_to_multiple(bytes, 4)
}

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    label: String,
}

pub use crate::wgpu_ledger::VramReport;

struct Builder {
    core: VramLedger,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    v_passes: Vec<Pass>,
    to_prefill: bool,
    to_verify: bool,
    probes: Vec<(String, wgpu::Buffer, usize, bool)>,
    pf_kvq_pairs: usize,
    pf_kv_write_full_layers: usize,
}

impl std::ops::Deref for Builder {
    type Target = VramLedger;
    fn deref(&self) -> &VramLedger {
        &self.core
    }
}

impl std::ops::DerefMut for Builder {
    fn deref_mut(&mut self) -> &mut VramLedger {
        &mut self.core
    }
}

impl Builder {
    fn push(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        let pass = Pass {
            pipeline,
            bind,
            grid,
            label: if dispatch::profile::enabled() {
                format!("{label}:{entry}")
            } else {
                String::new()
            },
        };
        if self.to_verify {
            self.v_passes.push(pass);
        } else if self.to_prefill {
            self.pf_passes.push(pass);
        } else {
            self.passes.push(pass);
        }
        Ok(())
    }

    fn probe(&mut self, name: &str, buf: &wgpu::Buffer, elems: usize, bf16: bool) {
        if self.probes.iter().any(|(n, _, _, _)| n == name) {
            return;
        }
        self.probes
            .push((name.to_string(), buf.clone(), elems, bf16));
    }
}

struct Sources {
    gemv: String,
    mx: String,
    mx_entry: &'static str,
    mx_rows_per_wg: u32,
    attn: String,
    flash: String,
    flash_stage1_entry: &'static str,
    kvq: String,
    pf_kvq: String,
    moe: String,
    rms: String,
    rmsres: String,
    resscale: String,
    prefill: String,
    verify: String,
}

#[doc(hidden)]
pub fn verify_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("gow:verify", compose(VERIFY_WGSL)),
        (
            "gow:prefill+splice",
            compose(&format!("{PREFILL_WGSL}\n{SPLICE_WGSL}")),
        ),
    ]
}

#[doc(hidden)]
pub fn nozi_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("gow:gemv", compose(GEMV_WGSL)),
        ("gow:mx", compose(MX_WGSL)),
        ("gow:attn", compose(ATTN_WGSL)),
        ("gow:moe", compose(MOE_WGSL)),
    ]
}

impl Sources {
    fn new(ctx: &WgpuContext) -> Self {
        let sg = mx_sg_reordered_sum_opt_in() && wk::gemv_nvfp4::sg32_ok(ctx);
        Self {
            gemv: compose(GEMV_WGSL),
            mx: compose(if sg { MX_WGSL } else { mx_scalar_section() }),
            mx_entry: if sg { MX_SG_ENTRY } else { MX_SCALAR_ENTRY },
            mx_rows_per_wg: if sg {
                MX_SG_ROWS_PER_WG_AT_SG32_OVER_WG256
            } else {
                MX_SCALAR_ROWS_PER_WG
            },
            attn: compose(ATTN_WGSL),
            flash: compose(FLASH_WGSL),
            flash_stage1_entry: if wk::gemv_nvfp4::sg32_ok(ctx) {
                "gow_flash_stage1_sg"
            } else {
                "gow_flash_stage1"
            },
            kvq: compose(wk::kv_fp8::WGSL),
            pf_kvq: compose(&format!("{}\n{}", wk::kv_fp8::WGSL, PF_KVQ_WGSL)),
            moe: compose(MOE_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            prefill: compose(&format!("{PREFILL_WGSL}\n{SPLICE_WGSL}")),
            verify: compose(VERIFY_WGSL),
        }
    }
}

struct Bf16Gpu {
    w: wgpu::Buffer,
    bias: Option<wgpu::Buffer>,
    n: usize,
    k: usize,
}

struct MxGpu {
    w: wgpu::Buffer,
    scales: wgpu::Buffer,
    bias: wgpu::Buffer,
    e: usize,
    n: usize,
    k: usize,
}

struct VerifyState {
    rows: usize,
    head_start: usize,
    passes: Vec<Pass>,
    logits: wgpu::Buffer,
    tokens: wgpu::Buffer,
    validated: bool,
}

struct ChunkRowSplice<'a> {
    rel_pos: usize,
    row_words: &'a [u32],
}

pub struct GptOssWgpu {
    ctx: &'static WgpuContext,
    config: GptOssConfig,
    max_seq: usize,
    pos: usize,
    validated: bool,
    prefix_validated: bool,
    pf_validated: bool,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    pf_m: usize,
    pf_tok: Option<wgpu::Buffer>,
    pf_ck: Option<wgpu::Buffer>,
    pf_splice: Option<wgpu::Buffer>,
    pf_mask: Option<wgpu::Buffer>,
    verify: Option<VerifyState>,
    head_start: usize,
    _buffers: Vec<wgpu::Buffer>,
    tok_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)>,
    kv8_fd: Option<(wgpu::Buffer, SharedFdParams)>,
    pf_kvq: Option<wgpu::Buffer>,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    probes: Vec<(String, wgpu::Buffer, usize, bool)>,
    vocab: usize,
    vram: VramReport,
    mx_entry: &'static str,
}

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_GPTOSS_WGPU_VRAM")
}

pub fn staging_flush_enabled() -> bool {
    !matches!(
        std::env::var("NV_GPTOSS_WGPU_STAGING_FLUSH")
            .ok()
            .as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

impl GptOssWgpu {
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn mx_gemv_entry(&self) -> &'static str {
        self.mx_entry
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
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
            "synthetic fill pos {pos} past max_seq {}; the flat kv caches index slots by pos \
             and the decode window start is a pure function of pos, so pos alone is the \
             cache-depth state",
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

    pub fn new(config: GptOssConfig, weights: &HostWeights, max_seq: usize) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq)
    }

    pub fn from_loader(
        config: GptOssConfig,
        weights: &nv_weights::WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq)
    }

    fn build(config: GptOssConfig, src: WeightSource<'_>, max_seq: usize) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let s = Sources::new(ctx);
        let mx_entry = s.mx_entry;
        let cfg = &config;

        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        anyhow::ensure!(
            cfg.hidden_size.is_multiple_of(2 * MX_BLOCK),
            "hidden_size {} must be a multiple of {}",
            cfg.hidden_size,
            2 * MX_BLOCK
        );
        anyhow::ensure!(
            cfg.intermediate_size.is_multiple_of(MX_BLOCK),
            "intermediate_size {} must be a multiple of {MX_BLOCK}",
            cfg.intermediate_size
        );
        anyhow::ensure!(
            (2 * cfg.intermediate_size * cfg.hidden_size / MX_BLOCK).is_multiple_of(4)
                && (cfg.hidden_size * cfg.intermediate_size / MX_BLOCK).is_multiple_of(4),
            "expert scale bytes must align to whole u32 words"
        );
        anyhow::ensure!(
            cfg.num_local_experts <= 256,
            "router top-k kernel caps num_local_experts at 256, got {}",
            cfg.num_local_experts
        );
        anyhow::ensure!(
            cfg.num_experts_per_tok >= 1 && cfg.num_experts_per_tok <= MAX_TOPK,
            "num_experts_per_tok {} out of range 1..={MAX_TOPK}",
            cfg.num_experts_per_tok
        );
        anyhow::ensure!(
            cfg.head_dim <= MAX_HEAD_DIM && cfg.head_dim.is_multiple_of(4),
            "head_dim {} must be a multiple of 4 and <= {MAX_HEAD_DIM}",
            cfg.head_dim
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
        anyhow::ensure!(cfg.sliding_window > 0, "sliding_window must be positive");

        let hidden = cfg.hidden_size;
        let hidden_words = hidden / 2;
        let eps = cfg.rms_norm_eps as f32;
        let vocab = cfg.vocab_size;

        let mut b = Builder {
            core: VramLedger::new(ctx, "gow-", staging_flush_enabled, STAGING_FLUSH_BYTES),
            passes: Vec::new(),
            pf_passes: Vec::new(),
            v_passes: Vec::new(),
            to_prefill: false,
            to_verify: false,
            probes: Vec::new(),
            pf_kvq_pairs: 0,
            pf_kv_write_full_layers: 0,
        };

        let tok_buf = b.upload_u32("gow-tok", &[0u32]);
        let pos_buf = b.upload_u32("gow-pos", &[0u32]);
        let flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)> =
            flash_decode_enabled().then(|| {
                let p = FlashDecodeParams {
                    n_heads: cfg.num_attention_heads as u32,
                    n_kv: cfg.num_key_value_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    splits: flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
                    total: 0,
                    start: 0,
                    pad0: 0,
                    scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                };
                (b.uni("gow-at-fl-p", p), p)
            });
        let kv8_geometry_ok = cfg.head_dim.is_multiple_of(32)
            && cfg.head_dim <= wk::flash_decode::MAX_HEAD_DIM;
        if kv_fp8_enabled() && !kv8_geometry_ok {
            eprintln!(
                "[gow-wgpu] kv-fp8 depth stack needs head_dim a multiple of 32 up to the fold \
                 generator's cap; head_dim {} keeps the serial bf16 decode arm",
                cfg.head_dim
            );
        }
        let kv8_shared: Option<Kv8Shared> = (kv_fp8_enabled() && kv8_geometry_ok)
            .then(|| -> Result<Kv8Shared> {
                let group = cfg.num_attention_heads / cfg.num_key_value_heads;
                let fold = {
                    let f = group.min(wk::flash_decode::MAX_GQA_FOLD);
                    if f > 1
                        && group.is_multiple_of(f)
                        && cfg.num_attention_heads.is_multiple_of(f)
                    {
                        f
                    } else {
                        1
                    }
                };
                let sg = ctx.caps.subgroup && ctx.subgroup_width() == Some(32);
                let splits =
                    flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
                let fd = SharedFdParams {
                    n_heads: cfg.num_attention_heads as u32,
                    n_kv: cfg.num_key_value_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    splits,
                    scaling: 1.0 / (cfg.head_dim as f32).sqrt(),
                    ..Default::default()
                };
                let gfd = FlashDecodeParams {
                    n_heads: cfg.num_attention_heads as u32,
                    n_kv: cfg.num_key_value_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    splits,
                    scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                    ..Default::default()
                };
                Ok(Kv8Shared {
                    fd_buf: b.uni("gow-at-fd8-p", fd),
                    fd_base: fd,
                    gfd_buf: b.uni("gow-at-fd8-s2-p", gfd),
                    fold_src: compose(&format!(
                        "{}\n{}",
                        wk::flash_decode::WGSL,
                        wk::flash_decode::fold_stage1_source_sd(cfg.head_dim as u32, sg, fold as u32)
                    )),
                    fold_entry: wk::flash_decode::fold_stage1_entry_sd(
                        cfg.head_dim as u32,
                        sg,
                        fold as u32,
                    ),
                    fold,
                    splits,
                })
            })
            .transpose()?;
        let mut pf_m = prefill_m();
        if pf_m > max_seq {
            pf_m = 0;
        }
        let pf = (pf_m > 0).then(|| alloc_pf_scratch(&mut b, cfg, max_seq, pf_m));
        let pf_kvq_buf = (kv8_shared.is_some() && pf.is_some()).then(|| {
            b.uni(
                "gow-pf-kvq-mp",
                PfKvqParams {
                    tokens: 0,
                    x_stride_elems: (cfg.num_key_value_heads * cfg.head_dim) as u32,
                    ..Default::default()
                },
            )
        });

        let res = b.zeros("gow-res", (hidden_words * 4) as u64);
        let res2 = b.zeros("gow-res2", (hidden_words * 4) as u64);
        let normed = b.zeros("gow-normed", (hidden_words * 4) as u64);
        let mixed_out = b.zeros("gow-mix", (hidden_words * 4) as u64);
        let moe_out = b.zeros("gow-moeout", (hidden_words * 4) as u64);

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
                "gow-embed",
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let p = b.uni(
                "gow-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "gow-gather",
                &s.moe,
                "gow_gather_embed",
                &[(30, &buf), (31, &tok_buf), (32, &res), (33, &p)],
                grid,
            )?;
            if let Some(sc) = &pf {
                let pp = b.uni(
                    "gow-pf-embed-p",
                    PfGatherParams {
                        row_off: off as u32,
                        n_rows: rows as u32,
                        hidden_words: hidden_words as u32,
                        vocab: vocab as u32,
                        m: pf_m as u32,
                        ..Default::default()
                    },
                );
                b.to_prefill = true;
                b.push(
                    "gow-pf-gather",
                    &s.prefill,
                    "gow_pf_gather_embed",
                    &[(0, &buf), (1, &sc.tok), (2, &sc.res), (3, &pp)],
                    ((hidden_words as u32).div_ceil(256), pf_m as u32, 1),
                )?;
                b.to_prefill = false;
            }
            off += rows;
        }
        drop(embed);

        if let Some(sc) = &pf {
            let sp = b.uni(
                "gow-pf-splice-p",
                SpliceParams {
                    hidden_words: hidden_words as u32,
                    m: pf_m as u32,
                    ..Default::default()
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.to_prefill = true;
            b.push(
                "gow-pf-splice",
                &s.prefill,
                "gow_pf_splice_embed_rows",
                &[(2, &sc.res), (4, &sc.splice), (5, &sc.mask), (6, &sp)],
                (grid.0, pf_m as u32, 1),
            )?;
            b.to_prefill = false;
        }

        let (cos, sin) = rope_tables(cfg, max_seq);
        let cosb = b.upload_f32("gow-cos", &cos);
        let sinb = b.upload_f32("gow-sin", &sin);

        let mut state_buffers: Vec<(wgpu::Buffer, u64)> = Vec::new();

        for li in 0..cfg.num_hidden_layers {
            let layer = src.layer(cfg, li)?;
            let ln_w = b.upload_u32("gow-ln", &pack_pairs(&layer.input_ln));
            if li == 0 {
                let p = b.uni(
                    "gow-rms-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "gow-rms0",
                    &s.rms,
                    "rmsnorm_bf16",
                    &[(0, &res), (1, &ln_w), (2, &normed), (3, &p)],
                    (1, 1, 1),
                )?;
            }

            let ag = build_attn(
                &mut b,
                &s,
                cfg,
                &layer,
                li,
                &normed,
                &mixed_out,
                &pos_buf,
                &cosb,
                &sinb,
                max_seq,
                &mut state_buffers,
                flash_fd.as_ref().map(|(buf, _)| buf),
                kv8_shared.as_ref(),
            )?;

            let post_w = b.upload_u32("gow-post-ln", &pack_pairs(&layer.post_attn_ln));
            let normed_post = b.zeros("gow-normed-post", (hidden_words * 4) as u64);
            let rp = b.uni(
                "gow-rmsres-p",
                RmsParams {
                    hidden: hidden as u32,
                    batch: 1,
                    eps,
                    words_per_row: hidden_words as u32,
                },
            );
            b.push(
                "gow-rmsres-post",
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

            let mg = build_moe(&mut b, &s, cfg, &layer.moe, &normed_post, &moe_out)?;

            if let Some(sc) = &pf {
                b.to_prefill = true;
                if li == 0 {
                    pf_norm(
                        &mut b,
                        &s,
                        "gow-pf-rms0",
                        &sc.res,
                        &ln_w,
                        &sc.normed,
                        hidden,
                        eps,
                        pf_m,
                    )?;
                }
                build_attn_prefill(
                    &mut b,
                    &s,
                    cfg,
                    &ag,
                    sc,
                    &cosb,
                    &sinb,
                    &sc.normed,
                    &sc.mixed,
                    max_seq,
                    &pos_buf,
                    pf_kvq_buf.as_ref(),
                )?;
                pf_norm_residual(
                    &mut b,
                    &s,
                    "gow-pf-rmsres-post",
                    &sc.mixed,
                    &sc.res,
                    &post_w,
                    &sc.normed_post,
                    hidden,
                    eps,
                    pf_m,
                )?;
                build_moe_prefill(&mut b, &s, cfg, &mg, sc, &sc.normed_post, &sc.moe_out)?;
                b.to_prefill = false;
            }

            if li + 1 < cfg.num_hidden_layers {
                let next = src.layer_input_ln(cfg, li + 1)?;
                let nw = b.upload_u32("gow-next-ln", &pack_pairs(&next));
                let rp2 = b.uni(
                    "gow-rmsres2-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "gow-rmsres-next",
                    &s.rmsres,
                    "rmsnorm_residual_bf16",
                    &[(0, &moe_out), (1, &res), (2, &nw), (3, &normed), (4, &rp2)],
                    (1, 1, 1),
                )?;
                if let Some(sc) = &pf {
                    b.to_prefill = true;
                    pf_norm_residual(
                        &mut b,
                        &s,
                        "gow-pf-rmsres-next",
                        &sc.moe_out,
                        &sc.res,
                        &nw,
                        &sc.normed,
                        hidden,
                        eps,
                        pf_m,
                    )?;
                    b.to_prefill = false;
                }
            } else {
                let sp = b.uni(
                    "gow-resadd-p",
                    ResScaleParams {
                        n: hidden as u32,
                        n_words: hidden_words as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1(hidden_words as u64, 256);
                b.push(
                    "gow-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &moe_out), (1, &res), (2, &res2), (3, &sp)],
                    grid,
                )?;
            }
        }

        let final_w = b.upload_u32("gow-final-ln", &pack_pairs(&src.final_norm(cfg)?));
        let final_x = b.zeros("gow-final-x", (hidden_words * 4) as u64);
        let fp = b.uni(
            "gow-final-p",
            RmsParams {
                hidden: hidden as u32,
                batch: 1,
                eps,
                words_per_row: hidden_words as u32,
            },
        );
        b.push(
            "gow-final-rms",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &res2), (1, &final_w), (2, &final_x), (3, &fp)],
            (1, 1, 1),
        )?;

        let head_start = b.passes.len();
        let logits = b.zeros("gow-logits", (vocab * 4) as u64);
        let lm = src.lm_head(cfg)?;
        anyhow::ensure!(
            lm.len() == vocab * hidden,
            "lm_head has {} values, want {}",
            lm.len(),
            vocab * hidden
        );
        let dummy_bias = b.upload_u32("gow-dummy-bias", &[0u32]);
        let mut lm_chunks: Vec<(wgpu::Buffer, usize, usize)> = Vec::new();
        let mut off = 0usize;
        while off < vocab {
            let rows = (chunk_rows.min(vocab - off)) & !1usize;
            let rows = if rows == 0 { vocab - off } else { rows };
            let wbuf = b.upload_u32(
                "gow-lmhead",
                &pack_pairs(&lm[off * hidden..(off + rows) * hidden]),
            );
            lm_chunks.push((wbuf.clone(), off, rows));
            let pairs = rows.div_ceil(2);
            let grid = b.grid1(pairs as u64, 1);
            let p = b.uni(
                "gow-lmhead-p",
                GemvBf16Params {
                    n_rows: rows as u32,
                    k_words: hidden_words as u32,
                    groups_x: grid.0,
                    out_f32: 1,
                    w_row_words: hidden_words as u32,
                    x_off_words: 0,
                    y_off_words: off as u32,
                    has_bias: 0,
                    alpha: 1.0,
                    ..Default::default()
                },
            );
            b.push(
                "gow-lmhead",
                &s.gemv,
                "gow_gemv_bf16",
                &[
                    (0, &wbuf),
                    (1, &final_x),
                    (2, &p),
                    (3, &logits),
                    (4, &dummy_bias),
                ],
                grid,
            )?;
            off += rows;
        }
        drop(lm);

        let pv = b.zeros("gow-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("gow-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("gow-token", 4);
        let ap = b.uni(
            "gow-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "gow-am1",
            &s.moe,
            "gow_argmax_stage1",
            &[(40, &logits), (41, &pv), (42, &pi), (44, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "gow-am2",
            &s.moe,
            "gow_argmax_stage2",
            &[(41, &pv), (42, &pi), (43, &token_out), (44, &ap)],
            (1, 1, 1),
        )?;

        let verify = match pf.as_ref() {
            Some(sc) if !b.pf_passes.is_empty() && verify_rows(sc.m) > 0 => {
                let rows = verify_rows(sc.m);
                b.to_verify = true;
                let v_res2 = b.zeros("gow-v-res2", (rows * hidden_words * 4) as u64);
                let v_final_x = b.zeros("gow-v-finalx", (rows * hidden_words * 4) as u64);
                let v_logits = b.zeros("gow-v-logits", (rows * vocab * 4) as u64);
                let v_pv = b.zeros("gow-v-am-pv", (rows * ARGMAX_GROUPS * 4) as u64);
                let v_pi = b.zeros("gow-v-am-pi", (rows * ARGMAX_GROUPS * 4) as u64);
                let v_tokens = b.zeros("gow-v-tokens", (rows * 4) as u64);
                let rsp = b.uni(
                    "gow-v-resadd-p",
                    ResScaleParams {
                        n: (rows * hidden) as u32,
                        n_words: (rows * hidden_words) as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1((rows * hidden_words) as u64, 256);
                b.push(
                    "gow-v-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &sc.moe_out), (1, &sc.res), (2, &v_res2), (3, &rsp)],
                    grid,
                )?;
                pf_norm(
                    &mut b,
                    &s,
                    "gow-v-final-rms",
                    &v_res2,
                    &final_w,
                    &v_final_x,
                    hidden,
                    eps,
                    rows,
                )?;
                let head_start = b.v_passes.len();
                for (wbuf, off, n_rows) in &lm_chunks {
                    let pairs = n_rows.div_ceil(2);
                    let grid = b.grid1(pairs as u64, 1);
                    let p = b.uni(
                        "gow-v-lmhead-p",
                        VerifyLmParams {
                            n_rows: *n_rows as u32,
                            k_words: hidden_words as u32,
                            groups_x: grid.0,
                            out_f32: 1,
                            w_row_words: hidden_words as u32,
                            x_row_words: hidden_words as u32,
                            y_row_words: vocab as u32,
                            has_bias: 0,
                            alpha: 1.0,
                            y_off_words: *off as u32,
                            ..Default::default()
                        },
                    );
                    b.push(
                        "gow-v-lmhead",
                        &s.verify,
                        "gow_v_lmhead",
                        &[
                            (0, wbuf),
                            (1, &v_final_x),
                            (2, &p),
                            (3, &v_logits),
                            (4, &dummy_bias),
                        ],
                        (grid.0, grid.1, rows as u32),
                    )?;
                }
                let vap = b.uni(
                    "gow-v-am-p",
                    ArgmaxParams {
                        n: vocab as u32,
                        groups: ARGMAX_GROUPS as u32,
                        ..Default::default()
                    },
                );
                b.push(
                    "gow-v-am1",
                    &s.verify,
                    "gow_v_argmax_stage1",
                    &[(10, &v_logits), (11, &v_pv), (12, &v_pi), (14, &vap)],
                    (ARGMAX_GROUPS as u32, 1, rows as u32),
                )?;
                b.push(
                    "gow-v-am2",
                    &s.verify,
                    "gow_v_argmax_stage2",
                    &[(11, &v_pv), (12, &v_pi), (13, &v_tokens), (14, &vap)],
                    (1, 1, rows as u32),
                )?;
                b.to_verify = false;
                Some(VerifyState {
                    rows,
                    head_start,
                    passes: std::mem::take(&mut b.v_passes),
                    logits: v_logits,
                    tokens: v_tokens,
                    validated: false,
                })
            }
            _ => None,
        };

        if kv8_shared.is_some() && !b.pf_passes.is_empty() {
            anyhow::ensure!(
                b.pf_kvq_pairs == b.pf_kv_write_full_layers,
                "{KV_FP8_ENV} decode reads the fp8 cache for every full-attention row, so every \
                 prefill arm that writes full-layer KV must also record its quantize pair; \
                 counted {} quantize pairs against {} full-layer prefill kv writes -- an arm \
                 that skips the pair would leave chunk rows unread by decode",
                b.pf_kvq_pairs,
                b.pf_kv_write_full_layers
            );
        }
        b.flush_staging();
        let vram = b.report();
        if vram_report_enabled() {
            eprint!("[gow-wgpu] {}", vram.render());
        }

        let Builder {
            core,
            passes,
            pf_passes,
            probes,
            ..
        } = b;
        let buffers = core.buffers;

        let (pf_passes, pf_m, pf_tok, pf_ck, pf_splice, pf_mask) = match pf {
            Some(sc) if !pf_passes.is_empty() => (
                pf_passes,
                sc.m,
                Some(sc.tok.clone()),
                Some(sc.ck.clone()),
                Some(sc.splice.clone()),
                Some(sc.mask.clone()),
            ),
            _ => (Vec::new(), 0, None, None, None, None),
        };
        let verify = if pf_m == 0 { None } else { verify };

        Ok(Self {
            ctx,
            config,
            max_seq,
            pos: 0,
            validated: false,
            prefix_validated: false,
            pf_validated: false,
            passes,
            pf_passes,
            pf_m,
            pf_tok,
            pf_ck,
            pf_splice,
            pf_mask,
            verify,
            head_start,
            _buffers: buffers,
            probes,
            tok_buf,
            pos_buf,
            flash_fd,
            kv8_fd: kv8_shared
                .as_ref()
                .map(|k8| (k8.fd_buf.clone(), k8.fd_base)),
            pf_kvq: pf_kvq_buf,
            token_out,
            logits,
            state_buffers,
            vocab,
            vram,
            mx_entry,
        })
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
        if let Some((buf, base)) = &self.flash_fd {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        if let Some((buf, base)) = &self.kv8_fd {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }

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
                anyhow::bail!("gpt_oss_wgpu decode step validation: {e}");
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

    pub fn prefill_chunk_len(&self) -> usize {
        if self.pf_passes.is_empty() {
            0
        } else {
            self.pf_m
        }
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.pf_passes.len()
    }

    fn write_chunk_inputs(
        &mut self,
        chunk: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        let m = self.pf_m;
        anyhow::ensure!(
            chunk.len() == m && (1..=m).contains(&live),
            "prefill chunk is {} tokens with {live} live, want {m} with 1..={m} live",
            chunk.len()
        );
        anyhow::ensure!(
            self.pos + live <= self.max_seq,
            "kv cache full at {} + {live} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in chunk {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let hidden_words = self.config.hidden_size / 2;
        let mut mask = vec![0u32; m];
        for sp in splices {
            anyhow::ensure!(
                sp.rel_pos < live,
                "splice rel_pos {} is not a live row of this chunk (live {live})",
                sp.rel_pos
            );
            anyhow::ensure!(
                sp.row_words.len() == hidden_words,
                "splice row has {} words, want {hidden_words}",
                sp.row_words.len()
            );
            mask[sp.rel_pos] = 1;
            let splice = self
                .pf_splice
                .as_ref()
                .expect("splice prefill without pf list");
            self.ctx.queue.write_buffer(
                splice,
                (sp.rel_pos * hidden_words * 4) as u64,
                bytemuck::cast_slice(sp.row_words),
            );
        }
        let mask_buf = self
            .pf_mask
            .as_ref()
            .expect("prefill chunk without pf list");
        self.ctx
            .queue
            .write_buffer(mask_buf, 0, bytemuck::cast_slice(&mask));
        let tok = self.pf_tok.as_ref().expect("prefill chunk without pf list");
        let ck = self.pf_ck.as_ref().expect("prefill chunk without pf list");
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        self.ctx
            .queue
            .write_buffer(tok, 0, bytemuck::cast_slice(&ids));
        self.ctx.queue.write_buffer(
            ck,
            0,
            bytemuck::bytes_of(&CkParams {
                m_live: live as u32,
                base: self.pos as u32,
                pad0: 0,
                pad1: 0,
            }),
        );
        if let Some(pf_kvq) = &self.pf_kvq {
            self.ctx
                .queue
                .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(self.pos as i32)));
            self.ctx.queue.write_buffer(
                pf_kvq,
                0,
                bytemuck::bytes_of(&PfKvqParams {
                    tokens: live as u32,
                    x_stride_elems: (self.config.num_key_value_heads * self.config.head_dim)
                        as u32,
                    ..Default::default()
                }),
            );
        }
        Ok(())
    }

    fn prefill_chunk(&mut self, chunk: &[u32], live: usize) -> Result<()> {
        self.prefill_chunk_masked(chunk, live, &[])
    }

    fn prefill_chunk_masked(
        &mut self,
        chunk: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        self.write_chunk_inputs(chunk, live, splices)?;
        let scope = if self.pf_validated {
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
                .pf_passes
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = self.pf_passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow::anyhow!("profiled prefill submit: {e}"))?;
        } else {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.pf_passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gpt_oss_wgpu prefill chunk validation: {e}");
            }
            self.pf_validated = true;
        }
        self.pos += live;
        Ok(())
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let m = self.prefill_chunk_len();
        if m == 0 {
            return Ok(0);
        }
        let mut done = 0usize;
        while tokens.len() - done >= m && self.pos + m <= self.max_seq {
            let chunk: Vec<u32> = tokens[done..done + m].to_vec();
            self.prefill_chunk(&chunk, m)?;
            done += m;
        }
        let left = tokens.len() - done;
        if left >= 2 && self.pos + left <= self.max_seq {
            let mut padded: Vec<u32> = tokens[done..].to_vec();
            let pad = *padded.last().expect("non-empty tail");
            padded.resize(m, pad);
            self.prefill_chunk(&padded, left)?;
            done += left;
        }
        Ok(done)
    }

    pub fn prefill_tokens_with_image_rows(
        &mut self,
        tokens: &[u32],
        splices: &[ImageRowSplice],
    ) -> Result<usize> {
        let m = self.prefill_chunk_len();
        anyhow::ensure!(
            m >= 2,
            "embedding-row splice prefill requires the chunked prefill graph: \
             NV_GPTOSS_WGPU_PREFILL_M>=2 and a live pf pass list"
        );
        let hidden = self.config.hidden_size;
        let hidden_words = hidden / 2;
        let mut prev_end = 0usize;
        let mut packed: Vec<Vec<u32>> = Vec::with_capacity(splices.len());
        for sp in splices {
            anyhow::ensure!(
                sp.rows_bf16.len() % hidden == 0,
                "splice rows_bf16 len {} not a multiple of hidden {hidden}",
                sp.rows_bf16.len()
            );
            let n_slots = sp.rows_bf16.len() / hidden;
            anyhow::ensure!(
                sp.position >= prev_end,
                "embedding-row splices must be sorted and non-overlapping"
            );
            anyhow::ensure!(
                sp.position + n_slots <= tokens.len(),
                "splice at {} with {n_slots} rows exceeds {} tokens",
                sp.position,
                tokens.len()
            );
            prev_end = sp.position + n_slots;
            packed.push(pack_pairs(&sp.rows_bf16));
        }
        let mut done = 0usize;
        while done < tokens.len() {
            let live = m.min(tokens.len() - done);
            let mut chunk: Vec<u32> = tokens[done..done + live].to_vec();
            let pad = *chunk.last().expect("non-empty chunk");
            chunk.resize(m, pad);
            let chunk_end = done + live;
            let mut rows: Vec<ChunkRowSplice> = Vec::new();
            for (si, sp) in splices.iter().enumerate() {
                let n_slots = sp.rows_bf16.len() / hidden;
                let lo = sp.position.max(done);
                let hi = (sp.position + n_slots).min(chunk_end);
                for abs in lo..hi {
                    let w0 = (abs - sp.position) * hidden_words;
                    rows.push(ChunkRowSplice {
                        rel_pos: abs - done,
                        row_words: &packed[si][w0..w0 + hidden_words],
                    });
                }
            }
            self.prefill_chunk_masked(&chunk, live, &rows)?;
            done += live;
        }
        Ok(done)
    }

    pub fn verify_max_rows(&self) -> usize {
        self.verify.as_ref().map(|v| v.rows).unwrap_or(0)
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        anyhow::ensure!(
            self.pos + n <= self.max_seq,
            "advance {n} from {} past max_seq {}",
            self.pos,
            self.max_seq
        );
        self.pos += n;
        Ok(())
    }

    fn verify_forward(&mut self, batch: &[u32]) -> Result<usize> {
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows > 0,
            "verify_chain needs the M-row prefill graph and its verify epilogue: \
             NV_GPTOSS_WGPU_PREFILL_M >= 2 and {VERIFY_M_ENV} != 0"
        );
        let live = batch.len();
        anyhow::ensure!(
            (1..=rows).contains(&live),
            "verify_chain batch of {live} out of 1..={rows}"
        );
        let mut padded = batch.to_vec();
        let pad = *padded.last().expect("non-empty batch");
        padded.resize(self.pf_m, pad);
        self.write_chunk_inputs(&padded, live, &[])?;
        let ctx = self.ctx;
        let verified = self.verify.as_ref().is_some_and(|v| v.validated);
        let scope = if self.pf_validated && verified {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        {
            let vs = self
                .verify
                .as_ref()
                .expect("verify rows are nonzero only with a verify epilogue");
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.pf_passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
                for (i, p) in vs.passes.iter().enumerate() {
                    let z = if i >= vs.head_start {
                        live as u32
                    } else {
                        p.grid.2
                    };
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, z);
                }
            }
            ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gpt_oss_wgpu verify chain validation: {e}");
            }
            self.pf_validated = true;
            if let Some(vs) = self.verify.as_mut() {
                vs.validated = true;
            }
        }
        Ok(live)
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        let live = self.verify_forward(batch)?;
        let vs = self.verify.as_ref().expect("verify epilogue");
        dispatch::read_back::<u32>(self.ctx, &vs.tokens, live)
            .map_err(|e| anyhow::anyhow!("verify token readback: {e}"))
    }

    pub fn verify_chain_logits(&mut self, batch: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let live = self.verify_forward(batch)?;
        let vocab = self.vocab;
        let vs = self.verify.as_ref().expect("verify epilogue");
        let toks = dispatch::read_back::<u32>(self.ctx, &vs.tokens, live)
            .map_err(|e| anyhow::anyhow!("verify token readback: {e}"))?;
        let logits = dispatch::read_back::<f32>(self.ctx, &vs.logits, live * vocab)
            .map_err(|e| anyhow::anyhow!("verify logits readback: {e}"))?;
        Ok((toks, logits))
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

    #[doc(hidden)]
    pub fn debug_probe(&self, name: &str) -> Option<Vec<f32>> {
        let (_, buf, elems, bf16) = self.probes.iter().find(|(n, _, _, _)| n == name)?;
        if *bf16 {
            let words: Vec<u32> = dispatch::read_back(self.ctx, buf, elems.div_ceil(2)).ok()?;
            let mut out = Vec::with_capacity(*elems);
            for i in 0..*elems {
                out.push(bf16_val((words[i / 2] >> (16 * (i % 2))) as u16));
            }
            Some(out)
        } else {
            dispatch::read_back(self.ctx, buf, *elems).ok()
        }
    }

    #[doc(hidden)]
    pub fn debug_probe_names(&self) -> Vec<String> {
        self.probes.iter().map(|(n, _, _, _)| n.clone()).collect()
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
    dummy_bias: &wgpu::Buffer,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "gow-gemvb-p",
        GemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_off_words: 0,
            y_off_words: 0,
            has_bias: u32::from(w.bias.is_some()),
            alpha: 1.0,
            ..Default::default()
        },
    );
    let bias = w.bias.as_ref().unwrap_or(dummy_bias);
    b.push(
        label,
        &s.gemv,
        "gow_gemv_bf16",
        &[(0, &w.w), (1, x), (2, &p), (3, y), (4, bias)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_mx(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &MxGpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    x_slot_stride_words: usize,
) -> Result<()> {
    let k_blocks = w.k / MX_BLOCK;
    let groups = w.n.div_ceil(s.mx_rows_per_wg as usize);
    let grid = b.grid1(groups as u64, 1);
    let p = b.uni(
        "gow-gemvmx-p",
        GemvMxParams {
            n_rows: w.n as u32,
            k_blocks: k_blocks as u32,
            groups_x: grid.0,
            has_bias: 1,
            w_e_stride_v4: (w.n * k_blocks) as u32,
            sf_e_stride_bytes: (w.n * k_blocks) as u32,
            bias_e_stride: w.n as u32,
            x_slot_stride_words: x_slot_stride_words as u32,
            y_slot_stride: w.n as u32,
            use_sel: 1,
            ..Default::default()
        },
    );
    anyhow::ensure!(slots <= w.e, "more slots than experts");
    b.push(
        label,
        &s.mx,
        s.mx_entry,
        &[
            (10, &w.w),
            (11, &w.scales),
            (12, x),
            (13, &p),
            (14, y),
            (15, sel),
            (16, &w.bias),
        ],
        (grid.0, grid.1, slots as u32),
    )
}

pub const PREFILL_M_MAX: usize = 64;

pub fn prefill_m() -> usize {
    match std::env::var("NV_GPTOSS_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(m) => m.clamp(2, PREFILL_M_MAX),
        None => 16,
    }
}

pub const VERIFY_M_ENV: &str = "NV_GPTOSS_WGPU_VERIFY_M";

pub const VERIFY_EPILOGUE_COSTS_ROWS_TIMES_VOCAB_F32_OF_LOGITS: &str =
    "the verify epilogue holds one f32 logit row per verify row (rows * vocab * 4 bytes) plus two \
     bf16 hidden scratch rows; at the published 20b geometry (vocab 201088) that is 804 KiB per \
     row, so the default verify width equals the prefill width and NV_GPTOSS_WGPU_VERIFY_M=0 \
     removes the epilogue entirely";

fn verify_rows(prefill_m: usize) -> usize {
    match std::env::var(VERIFY_M_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(r) => r.min(prefill_m),
        None => prefill_m,
    }
}

struct Kv8Shared {
    fd_buf: wgpu::Buffer,
    fd_base: SharedFdParams,
    gfd_buf: wgpu::Buffer,
    fold_src: String,
    fold_entry: String,
    fold: usize,
    splits: u32,
}

struct Kv8Attn {
    kc8: wgpu::Buffer,
    vc8: wgpu::Buffer,
    ksc: wgpu::Buffer,
    vsc: wgpu::Buffer,
    kvq_p: wgpu::Buffer,
}

struct AttnGpu {
    wq: Bf16Gpu,
    wk: Bf16Gpu,
    wv: Bf16Gpu,
    wo: Bf16Gpu,
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    sinks: wgpu::Buffer,
    dummy_bias: wgpu::Buffer,
    window: usize,
    kv8: Option<Kv8Attn>,
}

struct MoeGpu {
    router: Bf16Gpu,
    gu: MxGpu,
    dn: MxGpu,
    dummy_bias: wgpu::Buffer,
}

struct PfScratch {
    m: usize,
    tok: wgpu::Buffer,
    ck: wgpu::Buffer,
    splice: wgpu::Buffer,
    mask: wgpu::Buffer,
    res: wgpu::Buffer,
    normed: wgpu::Buffer,
    normed_post: wgpu::Buffer,
    mixed: wgpu::Buffer,
    moe_out: wgpu::Buffer,
    q_raw: wgpu::Buffer,
    k_raw: wgpu::Buffer,
    v_raw: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    scores: wgpu::Buffer,
    attn: wgpu::Buffer,
    attn_bf16: wgpu::Buffer,
    rlogits: wgpu::Buffer,
    ids: wgpu::Buffer,
    wts: wgpu::Buffer,
    y_gu: wgpu::Buffer,
    act: wgpu::Buffer,
    y_down: wgpu::Buffer,
}

fn alloc_pf_scratch(b: &mut Builder, cfg: &GptOssConfig, max_seq: usize, m: usize) -> PfScratch {
    let hidden_words = cfg.hidden_size / 2;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let k_top = cfg.num_experts_per_tok;
    let inter = cfg.intermediate_size;
    let mw = (m * hidden_words * 4) as u64;
    PfScratch {
        m,
        tok: b.upload_u32("gow-pf-tok", &vec![0u32; m]),
        ck: b.uni("gow-pf-ck", CkParams::default()),
        splice: b.zeros("gow-pf-splice", mw),
        mask: b.upload_u32("gow-pf-mask", &vec![0u32; m]),
        res: b.zeros("gow-pf-res", mw),
        normed: b.zeros("gow-pf-normed", mw),
        normed_post: b.zeros("gow-pf-normed-post", mw),
        mixed: b.zeros("gow-pf-mix", mw),
        moe_out: b.zeros("gow-pf-moeout", mw),
        q_raw: b.zeros("gow-pf-qraw", (m * n_h * hd * 2) as u64),
        k_raw: b.zeros("gow-pf-kraw", (m * n_kv * hd * 2) as u64),
        v_raw: b.zeros("gow-pf-vraw", (m * n_kv * hd * 2) as u64),
        q: b.zeros("gow-pf-q", (m * n_h * hd * 2) as u64),
        k: b.zeros("gow-pf-k", (m * n_kv * hd * 2) as u64),
        scores: b.zeros("gow-pf-scores", (m * n_h * max_seq * 4) as u64),
        attn: b.zeros("gow-pf-attn", (m * n_h * hd * 4) as u64),
        attn_bf16: b.zeros("gow-pf-packed", (m * n_h * hd * 2) as u64),
        rlogits: b.zeros("gow-pf-rlogits", (m * cfg.num_local_experts * 4) as u64),
        ids: b.zeros("gow-pf-ids", (m * k_top * 4) as u64),
        wts: b.zeros("gow-pf-wts", (m * k_top * 4) as u64),
        y_gu: b.zeros("gow-pf-ygu", (m * k_top * 2 * inter * 4) as u64),
        act: b.zeros("gow-pf-act", (m * k_top * inter * 2) as u64),
        y_down: b.zeros("gow-pf-ydown", (m * k_top * cfg.hidden_size * 4) as u64),
    }
}

fn pf_norm(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    m: usize,
) -> Result<()> {
    let p = b.uni(
        "gow-pf-rms-p",
        RmsParams {
            hidden: hidden as u32,
            batch: m as u32,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    b.push(
        label,
        &s.rms,
        "rmsnorm_bf16",
        &[(0, x), (1, w), (2, y), (3, &p)],
        (m as u32, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_norm_residual(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    res: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    m: usize,
) -> Result<()> {
    let p = b.uni(
        "gow-pf-rmsres-p",
        RmsParams {
            hidden: hidden as u32,
            batch: m as u32,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    b.push(
        label,
        &s.rmsres,
        "rmsnorm_residual_bf16",
        &[(0, x), (1, res), (2, w), (3, y), (4, &p)],
        (m as u32, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    x_row_words: usize,
    y: &wgpu::Buffer,
    y_row_words: usize,
    out_f32: bool,
    dummy_bias: &wgpu::Buffer,
    m: usize,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "gow-pf-gemvb-p",
        PfGemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_row_words: x_row_words as u32,
            y_row_words: y_row_words as u32,
            has_bias: u32::from(w.bias.is_some()),
            alpha: 1.0,
            ..Default::default()
        },
    );
    let bias = w.bias.as_ref().unwrap_or(dummy_bias);
    b.push(
        label,
        &s.prefill,
        "gow_pf_gemv_bf16",
        &[(10, &w.w), (11, x), (12, &p), (13, y), (14, bias)],
        (grid.0, grid.1, m as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_gemv_mx(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &MxGpu,
    x: &wgpu::Buffer,
    x_tok_stride_words: usize,
    x_slot_stride_words: usize,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    m: usize,
    k_top: usize,
) -> Result<()> {
    let k_blocks = w.k / MX_BLOCK;
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "gow-pf-gemvmx-p",
        PfGemvMxParams {
            n_rows: w.n as u32,
            k_blocks: k_blocks as u32,
            groups_x: grid.0,
            has_bias: 1,
            w_e_stride_v4: (w.n * k_blocks) as u32,
            sf_e_stride_bytes: (w.n * k_blocks) as u32,
            bias_e_stride: w.n as u32,
            x_slot_stride_words: x_slot_stride_words as u32,
            x_tok_stride_words: x_tok_stride_words as u32,
            y_slot_stride: w.n as u32,
            k_top: k_top as u32,
            ..Default::default()
        },
    );
    anyhow::ensure!(
        k_top <= w.e,
        "router picks {k_top} of {} experts; the slot axis carries m*k_top rows, but every slot \
         must still name a real expert",
        w.e
    );
    b.push(
        label,
        &s.prefill,
        "gow_pf_gemv_mx",
        &[
            (60, &w.w),
            (61, &w.scales),
            (62, x),
            (63, &p),
            (64, y),
            (65, sel),
            (66, &w.bias),
        ],
        (grid.0, grid.1, (m * k_top) as u32),
    )
}

fn upload_bf16(b: &mut Builder, label: &str, l: &HostBf16Lin) -> Bf16Gpu {
    let bias = if l.bias.is_empty() {
        None
    } else {
        Some(b.upload_u32(&format!("{label}-b"), &pack_pairs(&l.bias)))
    };
    Bf16Gpu {
        w: b.upload_u32(label, &pack_pairs(&l.w)),
        bias,
        n: l.n,
        k: l.k,
    }
}

fn upload_mx(b: &mut Builder, label: &str, st: &HostMxStack) -> MxGpu {
    MxGpu {
        w: b.upload_u32(label, &bytes_to_words(&st.blocks)),
        scales: b.upload_u32(&format!("{label}-sf"), &bytes_to_words(&st.scales)),
        bias: b.upload_u32(&format!("{label}-bias"), &pack_pairs(&st.bias)),
        e: st.e,
        n: st.n,
        k: st.k,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_attn(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    layer: &HostLayer,
    li: usize,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pos_buf: &wgpu::Buffer,
    cosb: &wgpu::Buffer,
    sinb: &wgpu::Buffer,
    max_seq: usize,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    flash_fd: Option<&wgpu::Buffer>,
    kv8: Option<&Kv8Shared>,
) -> Result<AttnGpu> {
    let a = &layer.attn;
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let window = match cfg.layer_types[li] {
        GptOssLayerType::Sliding => cfg.sliding_window,
        GptOssLayerType::Full => 0,
    };
    anyhow::ensure!(
        a.sinks.len() == n_h,
        "sinks length {} != {n_h}",
        a.sinks.len()
    );

    let wq = upload_bf16(b, "gow-at-q", &a.q);
    let wk_ = upload_bf16(b, "gow-at-k", &a.k);
    let wv = upload_bf16(b, "gow-at-v", &a.v);
    let wo = upload_bf16(b, "gow-at-o", &a.o);
    let dummy_bias = b.upload_u32("gow-at-dummy", &[0u32]);

    let q_raw = b.zeros("gow-at-qraw", (wq.n * 2) as u64);
    let k_raw = b.zeros("gow-at-kraw", (wk_.n * 2) as u64);
    let v_raw = b.zeros("gow-at-vraw", (wv.n * 2) as u64);
    push_gemv_bf16(b, s, "gow-at-qproj", &wq, x, &q_raw, false, &dummy_bias)?;
    push_gemv_bf16(b, s, "gow-at-kproj", &wk_, x, &k_raw, false, &dummy_bias)?;
    push_gemv_bf16(b, s, "gow-at-vproj", &wv, x, &v_raw, false, &dummy_bias)?;

    let q = b.zeros("gow-at-q", (n_h * hd * 2) as u64);
    let k = b.zeros("gow-at-k", (n_kv * hd * 2) as u64);
    let qp = b.uni(
        "gow-at-rope-p",
        RopeParams {
            n_rows: n_h as u32,
            head_dim: hd as u32,
            rot_half: (hd / 2) as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-at-ropeq",
        &s.attn,
        "gow_rope",
        &[
            (0, &q_raw),
            (1, cosb),
            (2, sinb),
            (3, pos_buf),
            (4, &q),
            (5, &qp),
        ],
        (n_h as u32, 1, 1),
    )?;
    let kp = b.uni(
        "gow-at-ropek-p",
        RopeParams {
            n_rows: n_kv as u32,
            head_dim: hd as u32,
            rot_half: (hd / 2) as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-at-ropek",
        &s.attn,
        "gow_rope",
        &[
            (0, &k_raw),
            (1, cosb),
            (2, sinb),
            (3, pos_buf),
            (4, &k),
            (5, &kp),
        ],
        (n_kv as u32, 1, 1),
    )?;

    let kv_words = n_kv * hd / 2;
    let cache_bytes = (max_seq * kv_words * 4) as u64;
    let kc = b.zeros("gow-at-kc", cache_bytes);
    let vc = b.zeros("gow-at-vc", cache_bytes);
    states.push((kc.clone(), cache_bytes));
    states.push((vc.clone(), cache_bytes));
    let kvp = b.uni(
        "gow-at-kv-p",
        KvWriteParams {
            words: kv_words as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(kv_words as u64, 64);
    b.push(
        "gow-at-kvwrite",
        &s.attn,
        "gow_kv_write",
        &[
            (10, &k),
            (11, &v_raw),
            (12, &kc),
            (13, &vc),
            (14, pos_buf),
            (15, &kvp),
        ],
        grid,
    )?;

    let sinks = b.upload_f32("gow-at-sinks", &a.sinks);
    b.probe("q0", &q, n_h * hd, true);
    b.probe("kraw0", &k_raw, n_kv * hd, true);
    b.probe("sinks0", &sinks, n_h, false);
    b.probe("mix0", out, cfg.hidden_size, true);
    let attn_bf16 = b.zeros("gow-at-packed", (n_h * hd * 2) as u64);
    b.probe(&format!("attnpk{li}"), &attn_bf16, n_h * hd, true);
    let use_kv8 = kv8.is_some() && window == 0;
    let use_flash = !use_kv8 && flash_fd.is_some() && window == 0;
    let mut kv8_attn: Option<Kv8Attn> = None;
    if let (Some(k8), true) = (kv8, use_kv8) {
        anyhow::ensure!(
            hd.is_multiple_of(32),
            "the shared fp8 fold stage1 packs 4 e4m3 bytes per word and strides head_dim in \
             32-lane strips, got {hd}"
        );
        let cache8_bytes = (max_seq * n_kv * hd) as u64;
        let scale_bytes = (max_seq * n_kv * 4) as u64;
        let kc8 = b.zeros("gow-at-kc8", cache8_bytes);
        let vc8 = b.zeros("gow-at-vc8", cache8_bytes);
        let ksc = b.zeros("gow-at-ks8", scale_bytes);
        let vsc = b.zeros("gow-at-vs8", scale_bytes);
        states.push((kc8.clone(), cache8_bytes));
        states.push((vc8.clone(), cache8_bytes));
        states.push((ksc.clone(), scale_bytes));
        states.push((vsc.clone(), scale_bytes));
        let kvq_p = b.uni(
            "gow-at-kvq-p",
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
            "gow-at-kvq-k",
            &s.kvq,
            wk::kv_fp8::QUANTIZE_ENTRY,
            &[(0, &k), (1, &kc8), (2, &ksc), (3, pos_buf), (4, &kvq_p)],
            (n_kv as u32, 1, 1),
        )?;
        b.push(
            "gow-at-kvq-v",
            &s.kvq,
            wk::kv_fp8::QUANTIZE_ENTRY,
            &[(0, &v_raw), (1, &vc8), (2, &vsc), (3, pos_buf), (4, &kvq_p)],
            (n_kv as u32, 1, 1),
        )?;
        let q_f32 = b.zeros("gow-at-qf32", (n_h * hd * 4) as u64);
        let cast_p = b.uni(
            "gow-at-qcast-p",
            ResScaleParams {
                n: (n_h * hd) as u32,
                n_words: (n_h * hd / 2) as u32,
                ..Default::default()
            },
        );
        let cast_grid = b.grid1((n_h * hd / 2) as u64, 256);
        b.push(
            "gow-at-qcast",
            &s.resscale,
            "cast_bf16_to_f32",
            &[(0, &q), (3, &cast_p), (4, &q_f32)],
            cast_grid,
        )?;
        let splits = k8.splits;
        let scratch = b.zeros("gow-at-fscr8", (n_h * splits as usize * (hd + 2) * 4) as u64);
        b.push(
            "gow-at-flash1",
            &k8.fold_src,
            &k8.fold_entry,
            &[
                (0, &q_f32),
                (4, &k8.fd_buf),
                (5, &kc8),
                (6, &vc8),
                (7, &scratch),
                (8, &ksc),
                (9, &vsc),
            ],
            ((n_h / k8.fold) as u32, splits, 1),
        )?;
        b.push(
            "gow-at-flash2",
            &s.flash,
            "gow_flash_stage2_sink_pk",
            &[(3, &scratch), (4, &attn_bf16), (5, &sinks), (6, &k8.gfd_buf)],
            (n_h as u32, 1, 1),
        )?;
        kv8_attn = Some(Kv8Attn {
            kc8,
            vc8,
            ksc,
            vsc,
            kvq_p,
        });
    } else if let (Some(fp), true) = (flash_fd, use_flash) {
        anyhow::ensure!(
            hd <= MAX_HEAD_DIM && hd.is_multiple_of(2),
            "gow flash stage1 accumulates head_dim in 32-lane strips up to {MAX_HEAD_DIM} and \
             stage2 packs bf16 pairs, got {hd}"
        );
        anyhow::ensure!(
            b.ctx.caps.max_compute_invocations_per_workgroup >= 256
                && b.ctx.caps.max_compute_workgroup_size_x >= 256
                && b.ctx.caps.workgroup_storage_fits(
                    FLASH_WORKGROUP_BYTES_QSH256_RED256_SM8_SL8_SACC2048_F32,
                ),
            "{FLASH_DECODE_ENV}=1 needs a 256-invocation workgroup and {} workgroup bytes; \
             device allows {} invocations / {} bytes -- unset the env to keep the serial arm",
            FLASH_WORKGROUP_BYTES_QSH256_RED256_SM8_SL8_SACC2048_F32,
            b.ctx.caps.max_compute_invocations_per_workgroup,
            b.ctx.caps.max_compute_workgroup_storage_size
        );
        let splits = flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
        let scratch = b.zeros("gow-at-fscr", (n_h * splits as usize * (hd + 2) * 4) as u64);
        let kvshare = s.flash_stage1_entry == "gow_flash_stage1_sg"
            && hd == KVSHARE_STAGE1_IS_SPECIALIZED_TO_HD_64
            && n_h / n_kv == KVSHARE_STAGE1_IS_SPECIALIZED_TO_GROUP_8;
        let (stage1_entry, grid_x) = if kvshare {
            ("gow_flash_stage1_kvshare_sg_group8_hd64", n_kv as u32)
        } else {
            (s.flash_stage1_entry, n_h as u32)
        };
        b.push(
            "gow-at-flash1",
            &s.flash,
            stage1_entry,
            &[(0, &q), (1, &kc), (2, &vc), (3, &scratch), (6, fp)],
            (grid_x, splits, 1),
        )?;
        b.push(
            "gow-at-flash2",
            &s.flash,
            "gow_flash_stage2_sink_pk",
            &[(3, &scratch), (4, &attn_bf16), (5, &sinks), (6, fp)],
            (n_h as u32, 1, 1),
        )?;
    } else {
        let scores = b.zeros("gow-at-scores", (n_h * max_seq * 4) as u64);
        let attn = b.zeros("gow-at-out", (n_h * hd * 4) as u64);
        let adp = b.uni(
            "gow-at-dec-p",
            AttnDecodeParams {
                n_heads: n_h as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                max_seq: max_seq as u32,
                group: (n_h / n_kv) as u32,
                window: window as u32,
                scale: 1.0 / (hd as f32).sqrt(),
                ..Default::default()
            },
        );
        b.push(
            "gow-at-decode",
            &s.attn,
            "gow_attn_decode",
            &[
                (20, &q),
                (21, &kc),
                (22, &vc),
                (23, &scores),
                (24, &attn),
                (25, pos_buf),
                (26, &adp),
                (27, &sinks),
            ],
            (n_h as u32, 1, 1),
        )?;
        b.probe("attn0", &attn, n_h * hd, false);
        let pkp = b.uni(
            "gow-at-pack-p",
            PackParams {
                n_words: (n_h * hd / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((n_h * hd / 2) as u64, 64);
        b.push(
            "gow-at-pack",
            &s.attn,
            "gow_pack_bf16",
            &[(30, &attn), (31, &attn_bf16), (32, &pkp)],
            grid,
        )?;
    }

    push_gemv_bf16(
        b,
        s,
        "gow-at-oproj",
        &wo,
        &attn_bf16,
        out,
        false,
        &dummy_bias,
    )?;

    Ok(AttnGpu {
        wq,
        wk: wk_,
        wv,
        wo,
        kc,
        vc,
        sinks,
        dummy_bias,
        window,
        kv8: kv8_attn,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_attn_prefill(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    ag: &AttnGpu,
    sc: &PfScratch,
    cosb: &wgpu::Buffer,
    sinb: &wgpu::Buffer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    max_seq: usize,
    pos_buf: &wgpu::Buffer,
    pf_kvq: Option<&wgpu::Buffer>,
) -> Result<()> {
    let m = sc.m;
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hidden_words = cfg.hidden_size / 2;

    pf_gemv_bf16(
        b,
        s,
        "gow-pf-qproj",
        &ag.wq,
        x,
        hidden_words,
        &sc.q_raw,
        n_h * hd / 2,
        false,
        &ag.dummy_bias,
        m,
    )?;
    pf_gemv_bf16(
        b,
        s,
        "gow-pf-kproj",
        &ag.wk,
        x,
        hidden_words,
        &sc.k_raw,
        n_kv * hd / 2,
        false,
        &ag.dummy_bias,
        m,
    )?;
    pf_gemv_bf16(
        b,
        s,
        "gow-pf-vproj",
        &ag.wv,
        x,
        hidden_words,
        &sc.v_raw,
        n_kv * hd / 2,
        false,
        &ag.dummy_bias,
        m,
    )?;

    let qp = b.uni(
        "gow-pf-ropeq-p",
        RopeParams {
            n_rows: n_h as u32,
            head_dim: hd as u32,
            rot_half: (hd / 2) as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-pf-ropeq",
        &s.prefill,
        "gow_pf_rope",
        &[
            (20, &sc.q_raw),
            (21, cosb),
            (22, sinb),
            (23, &sc.q),
            (24, &qp),
            (25, &sc.ck),
        ],
        (n_h as u32, m as u32, 1),
    )?;
    let kp = b.uni(
        "gow-pf-ropek-p",
        RopeParams {
            n_rows: n_kv as u32,
            head_dim: hd as u32,
            rot_half: (hd / 2) as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-pf-ropek",
        &s.prefill,
        "gow_pf_rope",
        &[
            (20, &sc.k_raw),
            (21, cosb),
            (22, sinb),
            (23, &sc.k),
            (24, &kp),
            (25, &sc.ck),
        ],
        (n_kv as u32, m as u32, 1),
    )?;

    let kv_words = n_kv * hd / 2;
    let kvp = b.uni(
        "gow-pf-kv-p",
        KvWriteParams {
            words: kv_words as u32,
            ..Default::default()
        },
    );
    b.push(
        "gow-pf-kvwrite",
        &s.prefill,
        "gow_pf_kv_write",
        &[
            (30, &sc.k),
            (31, &sc.v_raw),
            (32, &ag.kc),
            (33, &ag.vc),
            (34, &kvp),
            (35, &sc.ck),
        ],
        ((kv_words as u32).div_ceil(64), m as u32, 1),
    )?;
    if ag.window == 0 {
        b.pf_kv_write_full_layers += 1;
    }
    if let Some(k8) = ag.kv8.as_ref() {
        let pf_kvq = pf_kvq.ok_or_else(|| {
            anyhow::anyhow!(
                "{KV_FP8_ENV} full-layer prefill needs the per-chunk kvq uniform; \
                 build() allocates it whenever kv8 and the prefill graph coexist"
            )
        })?;
        b.push(
            "gow-pf-kvq-k",
            &s.pf_kvq,
            "q3w_pf_quantize_kv_fp8_m",
            &[
                (0, &sc.k),
                (1, &k8.kc8),
                (2, &k8.ksc),
                (3, pos_buf),
                (4, &k8.kvq_p),
                (9, pf_kvq),
            ],
            (n_kv as u32, m as u32, 1),
        )?;
        b.push(
            "gow-pf-kvq-v",
            &s.pf_kvq,
            "q3w_pf_quantize_kv_fp8_m",
            &[
                (0, &sc.v_raw),
                (1, &k8.vc8),
                (2, &k8.vsc),
                (3, pos_buf),
                (4, &k8.kvq_p),
                (9, pf_kvq),
            ],
            (n_kv as u32, m as u32, 1),
        )?;
        b.pf_kvq_pairs += 1;
    }

    let adp = b.uni(
        "gow-pf-attn-p",
        AttnDecodeParams {
            n_heads: n_h as u32,
            n_kv: n_kv as u32,
            head_dim: hd as u32,
            max_seq: max_seq as u32,
            group: (n_h / n_kv) as u32,
            window: ag.window as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            ..Default::default()
        },
    );
    b.push(
        "gow-pf-attn",
        &s.prefill,
        "gow_pf_attn",
        &[
            (40, &sc.q),
            (41, &ag.kc),
            (42, &ag.vc),
            (43, &sc.scores),
            (44, &sc.attn),
            (45, &adp),
            (46, &ag.sinks),
            (47, &sc.ck),
        ],
        (n_h as u32, m as u32, 1),
    )?;

    let pkp = b.uni(
        "gow-pf-pack-p",
        PackParams {
            n_words: (m * n_h * hd / 2) as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1((m * n_h * hd / 2) as u64, 64);
    b.push(
        "gow-pf-pack",
        &s.attn,
        "gow_pack_bf16",
        &[(30, &sc.attn), (31, &sc.attn_bf16), (32, &pkp)],
        grid,
    )?;

    pf_gemv_bf16(
        b,
        s,
        "gow-pf-oproj",
        &ag.wo,
        &sc.attn_bf16,
        n_h * hd / 2,
        out,
        hidden_words,
        false,
        &ag.dummy_bias,
        m,
    )
}

fn build_moe(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    m: &HostMoe,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Result<MoeGpu> {
    let hidden = cfg.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.intermediate_size;
    let e = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;

    anyhow::ensure!(
        m.gate_up.e == e && m.down.e == e,
        "expert stack count mismatch"
    );
    anyhow::ensure!(
        m.gate_up.n == 2 * inter && m.gate_up.k == hidden,
        "gate_up stack is [{}, {}], want [{}, {hidden}]",
        m.gate_up.n,
        m.gate_up.k,
        2 * inter
    );
    anyhow::ensure!(
        m.down.n == hidden && m.down.k == inter,
        "down stack is [{}, {}], want [{hidden}, {inter}]",
        m.down.n,
        m.down.k
    );

    let router = upload_bf16(b, "gow-moe-router", &m.router);
    let dummy_bias = b.upload_u32("gow-moe-dummy", &[0u32]);
    let rlogits = b.zeros("gow-moe-rlogits", (e * 4) as u64);
    push_gemv_bf16(
        b,
        s,
        "gow-moe-router",
        &router,
        x,
        &rlogits,
        true,
        &dummy_bias,
    )?;

    let ids = b.zeros("gow-moe-ids", (k_top * 4) as u64);
    let wts = b.zeros("gow-moe-wts", (k_top * 4) as u64);
    let rp = b.uni(
        "gow-moe-router-p",
        RouterParams {
            n_experts: e as u32,
            k: k_top as u32,
            ..Default::default()
        },
    );
    b.push(
        "gow-moe-topk",
        &s.moe,
        "gow_router_topk",
        &[(0, &rlogits), (1, &ids), (2, &wts), (3, &rp)],
        (1, 1, 1),
    )?;

    let gu = upload_mx(b, "gow-moe-gu", &m.gate_up);
    let dn = upload_mx(b, "gow-moe-dn", &m.down);

    let y_gu = b.zeros("gow-moe-ygu", (k_top * 2 * inter * 4) as u64);
    push_gemv_mx(b, s, "gow-moe-gemv-gu", &gu, x, &y_gu, &ids, k_top, 0)?;

    let act = b.zeros("gow-moe-act", (k_top * inter * 2) as u64);
    let swp = b.uni(
        "gow-moe-swiglu-p",
        SwigluParams {
            n_words: (k_top * inter / 2) as u32,
            inter_words: (inter / 2) as u32,
            two_inter: (2 * inter) as u32,
            limit: cfg.swiglu_limit,
            alpha: SWIGLU_ALPHA,
            ..Default::default()
        },
    );
    let grid = b.grid1((k_top * inter / 2) as u64, 64);
    b.push(
        "gow-moe-swiglu",
        &s.moe,
        "gow_swiglu",
        &[(10, &y_gu), (11, &act), (12, &swp)],
        grid,
    )?;

    let y_down = b.zeros("gow-moe-ydown", (k_top * hidden * 4) as u64);
    push_gemv_mx(
        b,
        s,
        "gow-moe-gemv-dn",
        &dn,
        &act,
        &y_down,
        &ids,
        k_top,
        inter / 2,
    )?;

    b.probe("rlogits0", &rlogits, e, false);
    b.probe("wts0", &wts, k_top, false);
    b.probe("ydown0", &y_down, k_top * hidden, false);
    let cbp = b.uni(
        "gow-moe-comb-p",
        CombineParams {
            hidden_words: hidden_words as u32,
            k: k_top as u32,
            slot_stride: hidden as u32,
            pad0: 0,
        },
    );
    let grid = b.grid1(hidden_words as u64, 64);
    b.push(
        "gow-moe-combine",
        &s.moe,
        "gow_moe_combine",
        &[(20, &y_down), (21, &wts), (22, out), (23, &cbp)],
        grid,
    )?;

    Ok(MoeGpu {
        router,
        gu,
        dn,
        dummy_bias,
    })
}

fn build_moe_prefill(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    mg: &MoeGpu,
    sc: &PfScratch,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Result<()> {
    let m = sc.m;
    let hidden = cfg.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.intermediate_size;
    let e = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;

    pf_gemv_bf16(
        b,
        s,
        "gow-pf-moe-router",
        &mg.router,
        x,
        hidden_words,
        &sc.rlogits,
        e,
        true,
        &mg.dummy_bias,
        m,
    )?;

    let rp = b.uni(
        "gow-pf-router-p",
        PfRouterParams {
            n_experts: e as u32,
            k: k_top as u32,
            m: m as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-pf-moe-topk",
        &s.prefill,
        "gow_pf_router_topk",
        &[(50, &sc.rlogits), (51, &sc.ids), (52, &sc.wts), (53, &rp)],
        (m as u32, 1, 1),
    )?;

    pf_gemv_mx(
        b,
        s,
        "gow-pf-moe-gemv-gu",
        &mg.gu,
        x,
        hidden_words,
        0,
        &sc.y_gu,
        &sc.ids,
        m,
        k_top,
    )?;

    let slots = m * k_top;
    let swp = b.uni(
        "gow-pf-swiglu-p",
        SwigluParams {
            n_words: (slots * inter / 2) as u32,
            inter_words: (inter / 2) as u32,
            two_inter: (2 * inter) as u32,
            limit: cfg.swiglu_limit,
            alpha: SWIGLU_ALPHA,
            ..Default::default()
        },
    );
    let grid = b.grid1((slots * inter / 2) as u64, 64);
    b.push(
        "gow-pf-swiglu",
        &s.moe,
        "gow_swiglu",
        &[(10, &sc.y_gu), (11, &sc.act), (12, &swp)],
        grid,
    )?;

    pf_gemv_mx(
        b,
        s,
        "gow-pf-moe-gemv-dn",
        &mg.dn,
        &sc.act,
        0,
        inter / 2,
        &sc.y_down,
        &sc.ids,
        m,
        k_top,
    )?;

    let cbp = b.uni(
        "gow-pf-comb-p",
        PfCombineParams {
            hidden_words: hidden_words as u32,
            k: k_top as u32,
            slot_stride: hidden as u32,
            m: m as u32,
        },
    );
    b.push(
        "gow-pf-moe-combine",
        &s.prefill,
        "gow_pf_moe_combine",
        &[(70, &sc.y_down), (71, &sc.wts), (72, out), (73, &cbp)],
        ((hidden_words as u32).div_ceil(64), m as u32, 1),
    )
}

impl WeightSource<'_> {
    fn embed(&self, cfg: &GptOssConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.embed.clone()),
            Self::Loader(w) => load_bf16(
                w,
                &["model.embed_tokens.weight"],
                &[cfg.vocab_size, cfg.hidden_size],
            ),
        }
    }

    fn lm_head(&self, cfg: &GptOssConfig) -> Result<Vec<u16>> {
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

    fn final_norm(&self, cfg: &GptOssConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.final_norm.clone()),
            Self::Loader(w) => load_bf16(w, &["model.norm.weight"], &[cfg.hidden_size]),
        }
    }

    fn layer_input_ln(&self, cfg: &GptOssConfig, idx: usize) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].input_ln.clone()),
            Self::Loader(w) => load_bf16(
                w,
                &[&format!("model.layers.{idx}.input_layernorm.weight")],
                &[cfg.hidden_size],
            ),
        }
    }

    fn layer(&self, cfg: &GptOssConfig, idx: usize) -> Result<HostLayer> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].clone()),
            Self::Loader(w) => load_layer(cfg, w, idx),
        }
    }
}

crate::wgpu_state_snapshot::impl_wgpu_state_snapshot!(GptOssWgpu, max_seq);
