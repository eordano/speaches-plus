use anyhow::Result;

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::qwen3_5_dense_wgpu::assert_w8_group_divides_k;

use super::config::{NVFP4_BLOCK, STAGING_FLUSH_BYTES};
use super::weights::{
    bytes_to_words, pack_pairs, HostBf16ExpertStack, HostBf16Lin, HostExperts, HostLin,
    HostNvfp4ExpertStack, HostNvfp4Lin,
};

pub const GEMV_BF16_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_gemv_bf16.wgsl");

pub const GEMV_NVFP4_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_gemv_nvfp4.wgsl");

pub const QUANT_ROWS_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_quant_rows.wgsl");

pub const GEMV_I8_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_gemv_i8.wgsl");

pub const COMMON_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_common.wgsl");

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GemvBf16Params {
    pub n_rows: u32,
    pub k_words: u32,
    pub groups_x: u32,
    pub out_f32: u32,
    pub w_row_words: u32,
    pub x_off_words: u32,
    pub y_off_words: u32,
    pub pad0: u32,
    pub alpha: f32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GemvBf16ExpertParams {
    pub n_rows: u32,
    pub k_words: u32,
    pub groups_x: u32,
    pub out_f32: u32,
    pub w_row_words: u32,
    pub w_e_stride_words: u32,
    pub x_slot_stride_words: u32,
    pub y_slot_stride_words: u32,
    pub alpha: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GemvI8Params {
    pub n_rows: u32,
    pub k_elems: u32,
    pub groups_x: u32,
    pub x_slot_stride_elems: u32,
    pub w_e_stride_words: u32,
    pub y_slot_stride_words: u32,
    pub use_sel: u32,
    pub groups_per_row: u32,
    pub group_shift: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
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
    pub pad0: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct QuantRowsParams {
    pub k_blocks: u32,
    pub n_slots: u32,
    pub use_sel: u32,
    pub x_slot_stride_elems: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RmsParams {
    pub hidden: u32,
    pub batch: u32,
    pub eps: f32,
    pub words_per_row: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ResScaleParams {
    pub n: u32,
    pub n_words: u32,
    pub scale: f32,
    pub cap: f32,
    pub inv_cap: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GatherParams {
    pub row_off: u32,
    pub n_rows: u32,
    pub hidden_words: u32,
    pub vocab: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SiluMulParams {
    pub n_words: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ArgmaxParams {
    pub n: u32,
    pub groups: u32,
    pub pad0: u32,
    pub pad1: u32,
}

pub struct Pass {
    pub pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    pub bind: wgpu::BindGroup,
    pub grid: (u32, u32, u32),
    pub entry: String,
}

pub use crate::wgpu_ledger::VramReport;

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_LAGUNA_WGPU_VRAM")
}

pub fn staging_flush_enabled() -> bool {
    !matches!(
        std::env::var("NV_LAGUNA_WGPU_STAGING_FLUSH")
            .ok()
            .as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub struct Builder {
    pub core: crate::wgpu_ledger::VramLedger,
    pub passes: Vec<Pass>,
    pub state_buffers: Vec<(wgpu::Buffer, u64)>,
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
    pub fn new(ctx: &'static WgpuContext) -> Self {
        Self {
            core: crate::wgpu_ledger::VramLedger::new(
                ctx,
                "lgw-",
                staging_flush_enabled,
                STAGING_FLUSH_BYTES,
            ),
            passes: Vec::new(),
            state_buffers: Vec::new(),
        }
    }

    pub fn state_zeros(&mut self, label: &str, bytes: u64) -> wgpu::Buffer {
        let b = self.zeros(label, bytes);
        self.state_buffers.push((b.clone(), bytes));
        b
    }

    pub fn push(
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
        self.passes.push(Pass {
            pipeline,
            bind,
            grid,
            entry: entry.to_string(),
        });
        Ok(())
    }

    pub fn row_chunk(&self, hidden: usize) -> usize {
        let limit = self
            .ctx
            .caps
            .max_storage_buffer_binding_size
            .clamp(1 << 20, 1u64 << 30);
        let per_row = (hidden * 2) as u64;
        ((limit / per_row) as usize).max(2) & !1usize
    }
}

pub const STEP_TOKEN: usize = 0;
pub const STEP_POS: usize = 1;
pub const STEP_TOTAL: usize = 2;
pub const STEP_SLIDING_START: usize = 3;
pub const STEP_SLOTS: usize = 4;

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct StepUniform {
    pub tok: u32,
    pub pos: u32,
    pub total: u32,
    pub sliding_start: u32,
}

pub struct StepBuffers {
    pub tok: wgpu::Buffer,
    pub pos: wgpu::Buffer,
    pub step: wgpu::Buffer,
    pub uni: wgpu::Buffer,
}

impl StepBuffers {
    pub fn alloc(b: &mut Builder) -> Self {
        Self {
            tok: b.upload_u32("lgw-tok", &[0u32]),
            pos: b.upload_u32("lgw-pos", &[0u32]),
            step: b.upload_u32("lgw-step", &[0u32; STEP_SLOTS]),
            uni: b.uni("lgw-stepu", StepUniform::default()),
        }
    }

    pub fn write(
        &self,
        ctx: &WgpuContext,
        token: u32,
        pos: u32,
        total_tokens: u32,
        sliding_start: u32,
    ) {
        ctx.queue
            .write_buffer(&self.tok, 0, bytemuck::bytes_of(&token));
        ctx.queue
            .write_buffer(&self.pos, 0, bytemuck::bytes_of(&pos));
        let slots: [u32; STEP_SLOTS] = [token, pos, total_tokens, sliding_start];
        ctx.queue
            .write_buffer(&self.step, 0, bytemuck::cast_slice(&slots));
        ctx.queue.write_buffer(
            &self.uni,
            0,
            bytemuck::bytes_of(&StepUniform {
                tok: token,
                pos,
                total: total_tokens,
                sliding_start,
            }),
        );
    }
}

pub struct Sources {
    pub gemv_bf16: String,
    pub gemv_i8: String,
    pub gemv_nvfp4: String,
    pub quant: String,
    pub common: String,
    pub rms: String,
    pub rmsres: String,
    pub resscale: String,
    pub attn: String,
    pub moe: String,
    pub dense: String,
}

impl Sources {
    pub fn new() -> Self {
        Self {
            gemv_bf16: compose(GEMV_BF16_WGSL),
            gemv_i8: compose(GEMV_I8_WGSL),
            gemv_nvfp4: format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), GEMV_NVFP4_WGSL),
            quant: format!("{}\n{}", wk::gemv_nvfp4::quantize_source(), QUANT_ROWS_WGSL),
            common: compose(COMMON_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            attn: compose(super::attn::ATTN_WGSL),
            moe: compose(super::moe::MOE_WGSL),
            dense: compose(super::dense::DENSE_WGSL),
        }
    }
}

impl Default for Sources {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Bf16Gpu {
    pub w: wgpu::Buffer,
    pub n: usize,
    pub k: usize,
}

pub use crate::nvfp4_host::Nvfp4Gpu;

pub struct Bf16ExpertGpu {
    pub w: wgpu::Buffer,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

pub struct Nvfp4ExpertGpu {
    pub w: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub alphas: wgpu::Buffer,
    pub globals: wgpu::Buffer,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

pub struct I8Gpu {
    pub w: wgpu::Buffer,
    pub s: wgpu::Buffer,
    pub e: usize,
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

pub enum LinGpu {
    Bf16(Bf16Gpu),
    Nvfp4(Nvfp4Gpu),
    Int8(I8Gpu),
}

impl LinGpu {
    pub fn n(&self) -> usize {
        match self {
            LinGpu::Bf16(l) => l.n,
            LinGpu::Nvfp4(l) => l.n,
            LinGpu::Int8(l) => l.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            LinGpu::Bf16(l) => l.k,
            LinGpu::Nvfp4(l) => l.k,
            LinGpu::Int8(l) => l.k,
        }
    }
}

pub enum ExpertsGpu {
    Bf16(Bf16ExpertGpu),
    Nvfp4(Nvfp4ExpertGpu),
    Int8(I8Gpu),
}

impl ExpertsGpu {
    pub fn e(&self) -> usize {
        match self {
            ExpertsGpu::Bf16(s) => s.e,
            ExpertsGpu::Nvfp4(s) => s.e,
            ExpertsGpu::Int8(s) => s.e,
        }
    }

    pub fn n(&self) -> usize {
        match self {
            ExpertsGpu::Bf16(s) => s.n,
            ExpertsGpu::Nvfp4(s) => s.n,
            ExpertsGpu::Int8(s) => s.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            ExpertsGpu::Bf16(s) => s.k,
            ExpertsGpu::Nvfp4(s) => s.k,
            ExpertsGpu::Int8(s) => s.k,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum W8Scope {
    Attn,
    Ffn,
}

pub fn w8_mode() -> (bool, bool) {
    let v = std::env::var("NV_LAGUNA_WGPU_W8").unwrap_or_default();
    match v.trim() {
        "ffn" => (false, true),
        "attn" => (true, false),
        "1" | "all" => (true, true),
        _ => (false, false),
    }
}

pub fn w8_group() -> usize {
    crate::nvfp4_host::w8_group_from_env("NV_LAGUNA_WGPU_W8_GROUP")
}

pub fn w8_enabled(ctx: &WgpuContext, scope: W8Scope) -> bool {
    let (attn, ffn) = w8_mode();
    let on = match scope {
        W8Scope::Attn => attn,
        W8Scope::Ffn => ffn,
    };
    on && wk::gemv_nvfp4_v2::subgroup32_ok(ctx)
}

pub use crate::nvfp4_host::quantize_nvfp4_stack_i8;

pub fn upload_nvfp4_experts_i8(b: &mut Builder, label: &str, st: &HostNvfp4ExpertStack) -> I8Gpu {
    let group = w8_group();
    assert_w8_group_divides_k(label, group, st.k, "NV_LAGUNA_WGPU_W8_GROUP");
    let (packed, scales) = quantize_nvfp4_stack_i8(st, group);
    I8Gpu {
        w: b.upload_u32(label, &packed),
        s: b.upload_f32(&format!("{label}-s"), &scales),
        e: st.e,
        n: st.n,
        k: st.k,
        group,
    }
}

pub fn upload_nvfp4_i8(b: &mut Builder, label: &str, l: &HostNvfp4Lin) -> I8Gpu {
    let st = HostNvfp4ExpertStack {
        packed: l.packed.clone(),
        scales_swizzled: l.scales_swizzled.clone(),
        alphas: vec![l.alpha],
        input_globals: vec![l.input_global],
        e: 1,
        n: l.n,
        k: l.k,
    };
    upload_nvfp4_experts_i8(b, label, &st)
}

#[allow(clippy::too_many_arguments)]
pub fn push_gemv_i8_experts(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    ex: &I8Gpu,
    x_packed: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    use_sel: bool,
    x_per_slot: bool,
) -> Result<()> {
    anyhow::ensure!(
        ex.k.is_multiple_of(4),
        "{label}: i8 gemv needs k % 4 == 0, got {}",
        ex.k
    );
    let groups = (ex.n.div_ceil(8)) as u32;
    let p = b.uni(
        "lgw-i8e-p",
        GemvI8Params {
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
            ..Default::default()
        },
    );
    let entry = if ex.group > 0 {
        "lgw_gemv_i8g_experts"
    } else {
        "lgw_gemv_i8_experts"
    };
    b.push(
        label,
        &s.gemv_i8,
        entry,
        &[
            (0, &ex.w),
            (1, &ex.s),
            (2, x_packed),
            (3, y),
            (4, &p),
            (5, sel),
        ],
        (groups, 1, slots as u32),
    )
}

pub fn upload_bf16(b: &mut Builder, label: &str, l: &HostBf16Lin) -> Bf16Gpu {
    Bf16Gpu {
        w: b.upload_u32(label, &pack_pairs(&l.w)),
        n: l.n,
        k: l.k,
    }
}

pub use crate::nvfp4_host::upload_nvfp4;

pub fn upload_bf16_experts(
    b: &mut Builder,
    label: &str,
    st: &HostBf16ExpertStack,
) -> Bf16ExpertGpu {
    Bf16ExpertGpu {
        w: b.upload_u32(label, &pack_pairs(&st.w)),
        e: st.e,
        n: st.n,
        k: st.k,
    }
}

pub fn upload_nvfp4_experts(
    b: &mut Builder,
    label: &str,
    st: &HostNvfp4ExpertStack,
) -> Nvfp4ExpertGpu {
    Nvfp4ExpertGpu {
        w: b.upload_u32(label, &bytes_to_words(&st.packed)),
        scales: b.upload_u32(&format!("{label}-sf"), &bytes_to_words(&st.scales_swizzled)),
        alphas: b.upload_f32(&format!("{label}-a"), &st.alphas),
        globals: b.upload_f32(&format!("{label}-g"), &st.input_globals),
        e: st.e,
        n: st.n,
        k: st.k,
    }
}

pub fn upload_lin(b: &mut Builder, label: &str, l: &HostLin, scope: W8Scope) -> LinGpu {
    match l {
        HostLin::Bf16(x) => LinGpu::Bf16(upload_bf16(b, label, x)),
        HostLin::Nvfp4(x) if w8_enabled(b.ctx, scope) => LinGpu::Int8(upload_nvfp4_i8(b, label, x)),
        HostLin::Nvfp4(x) => LinGpu::Nvfp4(upload_nvfp4(b, label, x)),
    }
}

pub fn upload_experts(b: &mut Builder, label: &str, e: &HostExperts) -> ExpertsGpu {
    match e {
        HostExperts::Bf16(s) => ExpertsGpu::Bf16(upload_bf16_experts(b, label, s)),
        HostExperts::Nvfp4(s) if w8_enabled(b.ctx, W8Scope::Ffn) => {
            ExpertsGpu::Int8(upload_nvfp4_experts_i8(b, label, s))
        }
        HostExperts::Nvfp4(s) => ExpertsGpu::Nvfp4(upload_nvfp4_experts(b, label, s)),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn push_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x_packed: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
    y_off_words: usize,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "lgw-gemvb-p",
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
        "lgw_gemv_bf16",
        &[(0, &w.w), (1, x_packed), (2, &p), (3, y)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn push_gemv_bf16_experts(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16ExpertGpu,
    x_packed: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    x_per_slot: bool,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "lgw-gemvbe-p",
        GemvBf16ExpertParams {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: 0,
            w_row_words: (w.k / 2) as u32,
            w_e_stride_words: (w.n * w.k / 2) as u32,
            x_slot_stride_words: if x_per_slot { (w.k / 2) as u32 } else { 0 },
            y_slot_stride_words: (w.n / 2) as u32,
            alpha: 1.0,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.gemv_bf16,
        "lgw_gemv_bf16_experts",
        &[(0, &w.w), (1, x_packed), (2, &p), (3, y), (4, sel)],
        (grid.0, grid.1, slots as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn push_quant_rows(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x_packed: &wgpu::Buffer,
    packed_out: &wgpu::Buffer,
    scales_out: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    globals: &wgpu::Buffer,
    k_elems: usize,
    slots: usize,
    use_sel: bool,
    x_per_slot: bool,
) -> Result<()> {
    let k_blocks = k_elems / NVFP4_BLOCK;
    anyhow::ensure!(
        k_blocks.is_multiple_of(4),
        "quant rows needs k/{NVFP4_BLOCK} divisible by 4, got {k_blocks}"
    );
    let p = b.uni(
        "lgw-quant-p",
        QuantRowsParams {
            k_blocks: k_blocks as u32,
            n_slots: slots as u32,
            use_sel: u32::from(use_sel),
            x_slot_stride_elems: if x_per_slot { k_elems as u32 } else { 0 },
        },
    );
    let gx = (k_blocks as u32).div_ceil(256).max(1);
    b.push(
        label,
        &s.quant,
        "lgw_quant_rows",
        &[
            (10, x_packed),
            (11, &p),
            (12, packed_out),
            (13, scales_out),
            (14, sel),
            (15, globals),
        ],
        (gx, slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn push_gemv_nvfp4(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &wgpu::Buffer,
    ws: &wgpu::Buffer,
    xq: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    alphas: &wgpu::Buffer,
    n_rows: usize,
    k_elems: usize,
    slots: usize,
    per_expert: bool,
    alpha: f32,
) -> Result<()> {
    let k_blocks = k_elems / NVFP4_BLOCK;
    let pairs = n_rows.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "lgw-gemv4-p",
        GemvNvfp4Params {
            alpha,
            n_rows: n_rows as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: if per_expert {
                (n_rows * k_blocks) as u32
            } else {
                0
            },
            sf_e_stride_bytes: if per_expert {
                wk::gemv_nvfp4::swizzled_scale_len(n_rows, k_blocks) as u32
            } else {
                0
            },
            x_slot_stride_vec2: if slots > 1 { k_blocks as u32 } else { 0 },
            xsf_slot_stride_bytes: if slots > 1 { k_blocks as u32 } else { 0 },
            y_slot_stride_words: (n_rows / 2) as u32,
            per_expert_alpha: u32::from(per_expert),
            pad0: 0,
        },
    );
    b.push(
        label,
        &s.gemv_nvfp4,
        "lgw_gemv_nvfp4",
        &[
            (10, w),
            (11, ws),
            (12, xq),
            (13, xs),
            (14, &p),
            (15, y),
            (16, sel),
            (17, alphas),
        ],
        (grid.0, grid.1, slots as u32),
    )
}

pub struct LinScratch {
    pub xq: wgpu::Buffer,
    pub xs: wgpu::Buffer,
    pub globals: wgpu::Buffer,
    pub sel0: wgpu::Buffer,
    pub alpha_dummy: wgpu::Buffer,
}

pub fn alloc_lin_scratch(b: &mut Builder, label: &str, lin: &LinGpu) -> LinScratch {
    let (input_global, xq_bytes, xs_bytes) = match lin {
        LinGpu::Bf16(_) | LinGpu::Int8(_) => (1.0f32, 4u64, 4u64),
        LinGpu::Nvfp4(l) => {
            let k_blocks = l.k / NVFP4_BLOCK;
            (
                l.input_global,
                (l.k / 2) as u64,
                (k_blocks.div_ceil(4) * 4) as u64,
            )
        }
    };
    LinScratch {
        xq: b.zeros(&format!("{label}-xq"), xq_bytes),
        xs: b.zeros(&format!("{label}-xs"), xs_bytes),
        globals: b.upload_f32(&format!("{label}-gin"), &[input_global]),
        sel0: b.upload_u32(&format!("{label}-sel0"), &[0u32]),
        alpha_dummy: b.upload_f32(&format!("{label}-adummy"), &[1.0f32]),
    }
}

pub fn push_lin_gemv(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    lin: &LinGpu,
    scratch: &LinScratch,
    x_packed: &wgpu::Buffer,
    y_packed: &wgpu::Buffer,
) -> Result<()> {
    match lin {
        LinGpu::Bf16(w) => push_gemv_bf16(b, s, label, w, x_packed, y_packed, false, 0),

        LinGpu::Int8(w) => push_gemv_i8_experts(
            b,
            s,
            label,
            w,
            x_packed,
            y_packed,
            &scratch.sel0,
            1,
            false,
            false,
        ),
        LinGpu::Nvfp4(w) => {
            push_quant_rows(
                b,
                s,
                &format!("{label}-quant"),
                x_packed,
                &scratch.xq,
                &scratch.xs,
                &scratch.sel0,
                &scratch.globals,
                w.k,
                1,
                false,
                false,
            )?;
            push_gemv_nvfp4(
                b,
                s,
                label,
                &w.w,
                &w.scales,
                &scratch.xq,
                &scratch.xs,
                y_packed,
                &scratch.sel0,
                &scratch.alpha_dummy,
                w.n,
                w.k,
                1,
                false,
                w.alpha,
            )
        }
    }
}

pub fn push_silu_mul(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    gate_packed: &wgpu::Buffer,
    up_packed: &wgpu::Buffer,
    out_packed: &wgpu::Buffer,
    n_elems_total: usize,
) -> Result<()> {
    let n_words = n_elems_total / 2;
    let p = b.uni(
        "lgw-silu-p",
        SiluMulParams {
            n_words: n_words as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(n_words as u64, 64);
    b.push(
        label,
        &s.common,
        "lgw_silu_mul",
        &[
            (10, gate_packed),
            (11, up_packed),
            (12, out_packed),
            (13, &p),
        ],
        grid,
    )
}

pub fn push_rmsnorm(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x_packed: &wgpu::Buffer,
    w_packed: &wgpu::Buffer,
    y_packed: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let p = b.uni(
        "lgw-rms-p",
        RmsParams {
            hidden: hidden as u32,
            batch: 1,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    b.push(
        label,
        &s.rms,
        "rmsnorm_bf16",
        &[(0, x_packed), (1, w_packed), (2, y_packed), (3, &p)],
        (1, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn push_rmsnorm_residual(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    delta_packed: &wgpu::Buffer,
    res_packed: &wgpu::Buffer,
    w_packed: &wgpu::Buffer,
    y_packed: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let p = b.uni(
        "lgw-rmsres-p",
        RmsParams {
            hidden: hidden as u32,
            batch: 1,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    b.push(
        label,
        &s.rmsres,
        "rmsnorm_residual_bf16",
        &[
            (0, delta_packed),
            (1, res_packed),
            (2, w_packed),
            (3, y_packed),
            (4, &p),
        ],
        (1, 1, 1),
    )
}

pub fn push_residual_add(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    delta_packed: &wgpu::Buffer,
    res_packed: &wgpu::Buffer,
    out_packed: &wgpu::Buffer,
    hidden: usize,
) -> Result<()> {
    let p = b.uni(
        "lgw-resadd-p",
        ResScaleParams {
            n: hidden as u32,
            n_words: (hidden / 2) as u32,
            scale: 1.0,
            ..Default::default()
        },
    );
    let grid = b.grid1((hidden / 2) as u64, 256);
    b.push(
        label,
        &s.resscale,
        "residual_add_scale_bf16",
        &[(0, delta_packed), (1, res_packed), (2, out_packed), (3, &p)],
        grid,
    )
}
