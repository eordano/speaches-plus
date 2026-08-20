#![allow(clippy::too_many_arguments)]

use std::sync::OnceLock;

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/gemv_bf16.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const LANES_PER_ROW: u32 = 32;
pub const ROWS_PER_GROUP: u32 = 8;
pub const VEC8_ENTRY: &str = "gemv_bf16_vec8";
pub const SCALAR_ENTRY: &str = "gemv_bf16_scalar";
pub const V4_TREE_ENTRY: &str = "gemv_bf16_vec8_v4";
pub const SG_U32_ENTRY: &str = "gemv_bf16_sg_u32";
pub const SG_SCALAR_ENTRY: &str = "gemv_bf16_sg_scalar";
pub const SG_DEFAULT_WG: u32 = 128;
pub const SG_V4_WG: [(u32, &str, u32); 4] = [
    (64, "gemv_bf16_sg_v4_wg64", 2),
    (128, "gemv_bf16_sg_v4_wg128", 4),
    (256, "gemv_bf16_sg_v4_wg256", 8),
    (512, "gemv_bf16_sg_v4_wg512", 16),
];

pub const SG_WGSL: &str = include_str!("../../../wgsl/gemv_bf16_sg.wgsl");

pub fn sg_source() -> String {
    compose(SG_WGSL)
}

pub const SG_PK_ENTRY_WG128: &str = "gemv_bf16_sg_v4_pk_wg128";
pub const SG_PK_ENTRY_WG256: &str = "gemv_bf16_sg_v4_pk_wg256";

pub const SG_PK_WGSL: &str = include_str!("../../../wgsl/gemv_bf16_sg_pk.wgsl");

pub fn sg_pk_source() -> String {
    format!("{}\n{}", sg_source(), SG_PK_WGSL)
}

pub fn sg_pk_entry(wg: u32) -> (&'static str, u32) {
    if wg >= 256 {
        (SG_PK_ENTRY_WG256, 8)
    } else {
        (SG_PK_ENTRY_WG128, 4)
    }
}

pub const ADVERTISED_SUBGROUP_ONLY_ENV: &str = "NV_KERNELS_WGPU_GEMV_BF16_ADVERTISED_SUBGROUP";

pub fn known_subgroup_width(
    subgroup: bool,
    min_size: u32,
    max_size: u32,
    probed: Option<u32>,
) -> Option<u32> {
    if !subgroup {
        return None;
    }
    probed.or(if min_size == max_size {
        Some(min_size)
    } else {
        None
    })
}

pub fn sg32_from(subgroup: bool, min_size: u32, max_size: u32, probed: Option<u32>) -> bool {
    known_subgroup_width(subgroup, min_size, max_size, probed) == Some(LANES_PER_ROW)
}

pub fn probed_subgroup_width(ctx: &WgpuContext) -> Option<u32> {
    static ADVERTISED_ONLY: OnceLock<bool> = OnceLock::new();
    if *ADVERTISED_ONLY
        .get_or_init(|| std::env::var(ADVERTISED_SUBGROUP_ONLY_ENV).is_ok_and(|v| v != "0"))
    {
        return None;
    }
    ctx.subgroup_width()
}

pub fn sg32_ok(ctx: &WgpuContext) -> bool {
    sg32_from(
        ctx.caps.subgroup,
        ctx.caps.subgroup_min_size,
        ctx.caps.subgroup_max_size,
        probed_subgroup_width(ctx),
    )
}

pub const ADAPTIVE_ENTRY: &str = "gemv_bf16_sg_v4_adaptive";
pub const CUDA_WARP_LANES: u32 = 32;
pub const ADAPTIVE_MIN_WIDTH: u32 = 8;

pub fn adaptive_width(
    subgroup: bool,
    min_size: u32,
    max_size: u32,
    probed: Option<u32>,
) -> Option<u32> {
    if !subgroup || max_size < min_size {
        return None;
    }
    let width = probed.unwrap_or(min_size);
    if width < ADAPTIVE_MIN_WIDTH || !width.is_power_of_two() {
        return None;
    }
    Some(width.min(CUDA_WARP_LANES))
}

pub fn reduction_strides(x: u32) -> (Vec<u32>, Vec<u32>) {
    let mut fold = Vec::new();
    let mut s = CUDA_WARP_LANES / 2;
    while s >= x {
        fold.push(s);
        s /= 2;
    }
    let mut shuffle = Vec::new();
    while s >= 1 {
        shuffle.push(s);
        s /= 2;
    }
    (fold, shuffle)
}

pub fn adaptive_source(x: u32, wg: u32) -> String {
    use std::fmt::Write as _;
    let r = CUDA_WARP_LANES / x;
    let rows = wg / x;
    let (fold, shuffle) = reduction_strides(x);
    let mut b = String::new();
    b.push_str(
        "struct GemvAdaptiveParams {\n    n_rows: u32,\n    k_elems: u32,\n    w_row_words: u32,\n    groups_x: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> ga_w4: array<vec4<u32>>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> ga_x4: array<vec4<u32>>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read_write> ga_y: array<u32>;\n");
    b.push_str("@group(0) @binding(3) var<uniform> ga_params: GemvAdaptiveParams;\n\n");
    writeln!(b, "@compute @workgroup_size({wg})").unwrap();
    writeln!(b, "fn {ADAPTIVE_ENTRY}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n");
    b.push_str("    @builtin(subgroup_id) sgid: u32,\n");
    b.push_str("    @builtin(subgroup_size) sgsz: u32,\n");
    b.push_str("    @builtin(subgroup_invocation_id) slane: u32\n) {\n");
    b.push_str("    let vt = sgid * sgsz + slane;\n");
    writeln!(b, "    let slot = vt / {x}u;").unwrap();
    writeln!(b, "    let vlane = slane & {}u;", x - 1).unwrap();
    writeln!(
        b,
        "    let row = (wid.x + wid.y * ga_params.groups_x) * {rows}u + slot;"
    )
    .unwrap();
    b.push_str("    let live = row < ga_params.n_rows;\n");
    b.push_str("    let kv = select(0u, ga_params.k_elems >> 3u, live);\n");
    b.push_str("    let w_base = select(0u, row * (ga_params.w_row_words >> 2u), live);\n");
    writeln!(b, "    var acc: array<f32, {r}>;").unwrap();
    writeln!(
        b,
        "    for (var j = 0u; j < {r}u; j = j + 1u) {{ acc[j] = 0.0; }}"
    )
    .unwrap();
    writeln!(b, "    for (var j = 0u; j < {r}u; j = j + 1u) {{").unwrap();
    writeln!(
        b,
        "        for (var v = vlane + {x}u * j; v < kv; v = v + {CUDA_WARP_LANES}u) {{"
    )
    .unwrap();
    b.push_str("            let ww = ga_w4[w_base + v];\n");
    b.push_str("            let xw = ga_x4[v];\n");
    b.push_str("            for (var c = 0u; c < 4u; c = c + 1u) {\n");
    b.push_str(
        "                acc[j] = acc[j] + (bf16_lo(ww[c]) * bf16_lo(xw[c]) + bf16_hi(ww[c]) * bf16_hi(xw[c]));\n",
    );
    b.push_str("            }\n        }\n    }\n");
    for s in fold {
        let m = s / x;
        b.push_str("    {\n");
        writeln!(b, "        var nxt: array<f32, {r}>;").unwrap();
        writeln!(
            b,
            "        for (var j = 0u; j < {r}u; j = j + 1u) {{ nxt[j] = acc[j] + acc[j ^ {m}u]; }}"
        )
        .unwrap();
        b.push_str("        acc = nxt;\n    }\n");
    }
    b.push_str("    var a0 = acc[0];\n");
    for s in shuffle {
        writeln!(b, "    a0 = a0 + subgroupShuffleXor(a0, {s}u);").unwrap();
    }
    b.push_str("    if (vlane == 0u && live) {\n        ga_y[row] = bf16_encode(a0);\n    }\n}\n");
    compose(&b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvKernel {
    TreeVec8,
    TreeScalar,
    TreeVec8V4,
    SgU32,
    SgV4 { wg: u32 },
    SgV4Adaptive { wg: u32, x: u32 },
    SgScalar,
}

impl GemvKernel {
    pub fn entry(self) -> &'static str {
        match self {
            Self::TreeVec8 => VEC8_ENTRY,
            Self::TreeScalar => SCALAR_ENTRY,
            Self::TreeVec8V4 => V4_TREE_ENTRY,
            Self::SgU32 => SG_U32_ENTRY,
            Self::SgV4 { wg } => SG_V4_WG
                .iter()
                .find(|(size, _, _)| *size == wg)
                .map(|(_, entry, _)| *entry)
                .unwrap_or("gemv_bf16_sg_v4_wg256"),
            Self::SgV4Adaptive { .. } => ADAPTIVE_ENTRY,
            Self::SgScalar => SG_SCALAR_ENTRY,
        }
    }

    pub fn rows_per_group(self) -> u32 {
        match self {
            Self::SgV4 { wg } => (wg / LANES_PER_ROW).max(1),
            Self::SgV4Adaptive { wg, x } => (wg / x.max(1)).max(1),
            _ => ROWS_PER_GROUP,
        }
    }

    pub fn source(self) -> String {
        match self {
            Self::TreeVec8 | Self::TreeScalar | Self::TreeVec8V4 => compose(WGSL),
            Self::SgU32 | Self::SgV4 { .. } | Self::SgScalar => sg_source(),
            Self::SgV4Adaptive { wg, x } => adaptive_source(x, wg),
        }
    }

    pub fn needs_vec8(self) -> bool {
        !matches!(self, Self::TreeScalar | Self::SgScalar)
    }

    fn slots(self) -> (u32, u32) {
        match self {
            Self::TreeVec8 | Self::TreeScalar => (0, 1),
            Self::TreeVec8V4 => (20, 21),
            Self::SgU32 | Self::SgScalar => (4, 5),
            Self::SgV4 { .. } | Self::SgV4Adaptive { .. } => (0, 1),
        }
    }
}

pub fn select_kernel_from(
    subgroup: bool,
    min_size: u32,
    max_size: u32,
    probed: Option<u32>,
    k: usize,
) -> GemvKernel {
    if sg32_from(subgroup, min_size, max_size, probed) {
        if k.is_multiple_of(8) {
            GemvKernel::SgV4 { wg: SG_DEFAULT_WG }
        } else {
            GemvKernel::SgScalar
        }
    } else if k.is_multiple_of(8) {
        match adaptive_width(subgroup, min_size, max_size, probed) {
            Some(x) => GemvKernel::SgV4Adaptive {
                wg: SG_DEFAULT_WG,
                x,
            },
            None => GemvKernel::TreeVec8,
        }
    } else {
        GemvKernel::TreeScalar
    }
}

pub fn select_kernel(ctx: &WgpuContext, k: usize) -> GemvKernel {
    select_kernel_from(
        ctx.caps.subgroup,
        ctx.caps.subgroup_min_size,
        ctx.caps.subgroup_max_size,
        probed_subgroup_width(ctx),
        k,
    )
}

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvBf16Params {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
}

pub fn entry_for(k: usize) -> &'static str {
    if k.is_multiple_of(8) {
        VEC8_ENTRY
    } else {
        SCALAR_ENTRY
    }
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "gemv_bf16", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_binding(ctx: &WgpuContext, what: &str, bytes: u64) -> Result<()> {
    if bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "gemv_bf16 {what} needs {bytes} bytes; device allows {} per storage binding",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

struct Plan {
    w: wgpu::Buffer,
    x: wgpu::Buffer,
    y: wgpu::Buffer,
    params: wgpu::Buffer,
    kernel: GemvKernel,
    groups: (u32, u32, u32),
}

impl Plan {
    fn bindings(&self) -> [(u32, &wgpu::Buffer); 4] {
        let (ws, xs) = self.kernel.slots();
        [
            (ws, &self.w),
            (xs, &self.x),
            (2, &self.y),
            (3, &self.params),
        ]
    }
}

fn plan(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    kernel: GemvKernel,
) -> Result<Plan> {
    if !k.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "gemv_bf16 K must be even so rows start on a u32 word; got {k}"
        )));
    }
    if kernel.needs_vec8() && !k.is_multiple_of(8) {
        return Err(WgpuError::Shape(format!(
            "{} needs K%8==0; got {k}",
            kernel.entry()
        )));
    }
    dispatch::check_len("gemv_bf16 w", w.len(), n * k)?;
    dispatch::check_len("gemv_bf16 x", x.len(), k)?;
    check_device(ctx)?;

    let row_words = k / 2;
    check_binding(ctx, "w", (n as u64) * (row_words as u64) * 4)?;
    check_binding(ctx, "y", (n as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, n as u64, kernel.rows_per_group());
    let params = GemvBf16Params {
        n_rows: n as u32,
        k_elems: k as u32,
        w_row_words: row_words as u32,
        groups_x: groups.0,
    };

    Ok(Plan {
        w: dispatch::storage_from_slice(ctx, "gemv-bf16-w", &pack_u16(w)),
        x: dispatch::storage_from_slice(ctx, "gemv-bf16-x", &pack_u16(x)),
        y: dispatch::storage_zeroed(ctx, "gemv-bf16-y", (n * 4) as u64),
        params: dispatch::uniform_from(ctx, "gemv-bf16-params", &params),
        kernel,
        groups,
    })
}

pub fn gemv_bf16(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("gemv_bf16 y", y.len(), n)?;
    if n == 0 || k == 0 {
        return Ok(());
    }
    let kernel = select_kernel(ctx, k);
    let p = plan(ctx, w, x, n, k, kernel)?;
    dispatch::run(
        ctx,
        "nv_kernels_gemv_bf16",
        &kernel.source(),
        kernel.entry(),
        &p.bindings(),
        p.groups,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &p.y, n)?;
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

pub fn gemv_bf16_probe(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    warmup: usize,
    iters: usize,
    kernel: GemvKernel,
) -> Result<(Vec<u16>, f64)> {
    let p = plan(ctx, w, x, n, k, kernel)?;
    let pipeline = dispatch::cached_compute_pipeline(
        ctx,
        "nv_kernels_gemv_bf16",
        &kernel.source(),
        kernel.entry(),
    )?;
    let group = dispatch::bind_group(ctx, &pipeline, &p.bindings());
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(warmup.max(1));
    ctx.poll_blocking()?;

    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking()?;
    let secs = start.elapsed().as_secs_f64();

    let words: Vec<u32> = dispatch::read_back(ctx, &p.y, n)?;
    let y = words.iter().map(|word| (*word & 0xffff) as u16).collect();
    Ok((y, secs))
}

pub fn gemv_bf16_weight_gbps(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    if n == 0 || k == 0 || iters == 0 {
        return Ok(0.0);
    }
    let kernel = select_kernel(ctx, k);
    let (_, secs) = gemv_bf16_probe(ctx, w, x, n, k, 1, iters, kernel)?;
    if secs <= 0.0 {
        return Ok(0.0);
    }
    let bytes = (n as f64) * (k as f64) * 2.0 * (iters as f64);
    Ok(bytes / secs / 1.0e9)
}

pub const NORMED_ENTRY: &str = "gemv_bf16_normed";
pub const ROWQUANT_ENTRY: &str = "rowquant_i8";
pub const I8_NORMED_ENTRY: &str = "gemv_i8_normed";
pub const I8_NORMED_MK_ENTRY: &str = "gemv_i8_normed_mk";

pub const MAX_SHARED_K: usize = 4096;
pub const MAX_MK_ROWS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvNormedParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
    rstd: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RowQuantParams {
    n_rows: u32,
    k_elems: u32,
    src_row_words: u32,
    dst_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvI8Params {
    n_rows: u32,
    k_elems: u32,
    wq_row_words: u32,
    groups_x: u32,
    m_rows: u32,
    x_row_words: u32,
    pad0: u32,
    pad1: u32,
}

fn pack_u16_rows(src: &[u16], rows: usize, k: usize) -> Vec<u32> {
    let row_words = k.div_ceil(2);
    let mut out = vec![0u32; rows * row_words];
    for r in 0..rows {
        for i in 0..k {
            out[r * row_words + (i >> 1)] |= (src[r * k + i] as u32) << (16 * (i & 1));
        }
    }
    out
}

fn pack_i8_rows(src: &[i8], rows: usize, k: usize) -> Vec<u32> {
    let row_words = k.div_ceil(4);
    let mut out = vec![0u32; rows * row_words];
    for r in 0..rows {
        for i in 0..k {
            out[r * row_words + (i >> 2)] |= ((src[r * k + i] as u8) as u32) << (8 * (i & 3));
        }
    }
    out
}

fn unpack_i8_rows(words: &[u32], dst: &mut [i8], rows: usize, k: usize) {
    let row_words = k.div_ceil(4);
    for r in 0..rows {
        for i in 0..k {
            let w = words[r * row_words + (i >> 2)];
            dst[r * k + i] = ((w >> (8 * (i & 3))) & 0xff) as u8 as i8;
        }
    }
}

fn store_bf16(words: &[u32], y: &mut [u16]) {
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
}

pub fn gemv_bf16_normed(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    norm_weight: &[u16],
    rstd: f32,
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("gemv_bf16_normed y", y.len(), n)?;
    if n == 0 || k == 0 {
        return Ok(());
    }
    if !k.is_multiple_of(8) || k > MAX_SHARED_K {
        return Err(WgpuError::Shape(format!(
            "gemv_bf16_normed needs K%8==0 and K<={MAX_SHARED_K}; got {k}"
        )));
    }
    dispatch::check_len("gemv_bf16_normed w", w.len(), n * k)?;
    dispatch::check_len("gemv_bf16_normed x", x.len(), k)?;
    dispatch::check_len("gemv_bf16_normed wn", norm_weight.len(), k)?;
    check_device(ctx)?;

    let row_words = k / 2;
    check_binding(ctx, "w", (n as u64) * (row_words as u64) * 4)?;
    check_binding(ctx, "y", (n as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, n as u64, ROWS_PER_GROUP);
    let params = GemvNormedParams {
        n_rows: n as u32,
        k_elems: k as u32,
        w_row_words: row_words as u32,
        groups_x: groups.0,
        rstd,
        ..Default::default()
    };

    let bw = dispatch::storage_from_slice(ctx, "gemv-bf16-normed-w", &pack_u16(w));
    let bx = dispatch::storage_from_slice(ctx, "gemv-bf16-normed-x", &pack_u16(x));
    let bn = dispatch::storage_from_slice(ctx, "gemv-bf16-normed-wn", &pack_u16(norm_weight));
    let by = dispatch::storage_zeroed(ctx, "gemv-bf16-normed-y", (n * 4) as u64);
    let bp = dispatch::uniform_from(ctx, "gemv-bf16-normed-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_gemv_bf16_normed",
        &compose(WGSL),
        NORMED_ENTRY,
        &[(4, &bw), (5, &bx), (6, &bn), (7, &by), (8, &bp)],
        groups,
    )?;
    store_bf16(&dispatch::read_back::<u32>(ctx, &by, n)?, y);
    Ok(())
}

pub fn rowquant_i8(
    ctx: &WgpuContext,
    w: &[u16],
    q: &mut [i8],
    scales: &mut [f32],
    rows: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("rowquant_i8 q", q.len(), rows * k)?;
    dispatch::check_len("rowquant_i8 scales", scales.len(), rows)?;
    if rows == 0 || k == 0 {
        return Ok(());
    }
    dispatch::check_len("rowquant_i8 w", w.len(), rows * k)?;
    check_device(ctx)?;

    let src_row_words = k.div_ceil(2);
    let dst_row_words = k.div_ceil(4);
    check_binding(ctx, "w", (rows as u64) * (src_row_words as u64) * 4)?;
    check_binding(ctx, "q", (rows as u64) * (dst_row_words as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, rows as u64, 1);
    let params = RowQuantParams {
        n_rows: rows as u32,
        k_elems: k as u32,
        src_row_words: src_row_words as u32,
        dst_row_words: dst_row_words as u32,
        groups_x: groups.0,
        ..Default::default()
    };

    let bw = dispatch::storage_from_slice(ctx, "rowquant-i8-w", &pack_u16_rows(w, rows, k));
    let bq = dispatch::storage_zeroed(ctx, "rowquant-i8-q", (rows * dst_row_words * 4) as u64);
    let bs = dispatch::storage_zeroed(ctx, "rowquant-i8-scale", (rows * 4) as u64);
    let bp = dispatch::uniform_from(ctx, "rowquant-i8-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_rowquant_i8",
        &compose(WGSL),
        ROWQUANT_ENTRY,
        &[(9, &bw), (10, &bq), (11, &bs), (12, &bp)],
        groups,
    )?;

    let words: Vec<u32> = dispatch::read_back(ctx, &bq, rows * dst_row_words)?;
    unpack_i8_rows(&words, q, rows, k);
    let got: Vec<f32> = dispatch::read_back(ctx, &bs, rows)?;
    scales.copy_from_slice(&got);
    Ok(())
}

struct I8Plan {
    wq: wgpu::Buffer,
    row_scale: wgpu::Buffer,
    x: wgpu::Buffer,
    wn: wgpu::Buffer,
    rstd: wgpu::Buffer,
    y: wgpu::Buffer,
    params: wgpu::Buffer,
    groups: (u32, u32, u32),
}

impl I8Plan {
    fn bindings(&self) -> [(u32, &wgpu::Buffer); 7] {
        [
            (13, &self.wq),
            (14, &self.row_scale),
            (15, &self.x),
            (16, &self.wn),
            (17, &self.rstd),
            (18, &self.y),
            (19, &self.params),
        ]
    }
}

fn i8_plan(
    ctx: &WgpuContext,
    w_i8: &[i8],
    w_scales: &[f32],
    x: &[u16],
    norm_weight: &[u16],
    rstd: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<I8Plan> {
    if !k.is_multiple_of(16) {
        return Err(WgpuError::Shape(format!(
            "gemv_i8_normed needs K%16==0; got {k}"
        )));
    }
    if m == 0 || m > MAX_MK_ROWS {
        return Err(WgpuError::Shape(format!(
            "gemv_i8_normed_mk supports 1..={MAX_MK_ROWS} rows; got {m}"
        )));
    }
    dispatch::check_len("gemv_i8_normed wq", w_i8.len(), n * k)?;
    dispatch::check_len("gemv_i8_normed row_scale", w_scales.len(), n)?;
    dispatch::check_len("gemv_i8_normed x", x.len(), m * k)?;
    dispatch::check_len("gemv_i8_normed wn", norm_weight.len(), k)?;
    dispatch::check_len("gemv_i8_normed rstd", rstd.len(), m)?;
    check_device(ctx)?;

    let wq_row_words = k / 4;
    check_binding(ctx, "wq", (n as u64) * (wq_row_words as u64) * 4)?;
    check_binding(ctx, "y", (m as u64) * (n as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, n as u64, ROWS_PER_GROUP);
    let params = GemvI8Params {
        n_rows: n as u32,
        k_elems: k as u32,
        wq_row_words: wq_row_words as u32,
        groups_x: groups.0,
        m_rows: m as u32,
        x_row_words: (k / 2) as u32,
        ..Default::default()
    };

    Ok(I8Plan {
        wq: dispatch::storage_from_slice(ctx, "gemv-i8-wq", &pack_i8_rows(w_i8, n, k)),
        row_scale: dispatch::storage_from_slice(ctx, "gemv-i8-row-scale", w_scales),
        x: dispatch::storage_from_slice(ctx, "gemv-i8-x", &pack_u16(x)),
        wn: dispatch::storage_from_slice(ctx, "gemv-i8-wn", &pack_u16(norm_weight)),
        rstd: dispatch::storage_from_slice(ctx, "gemv-i8-rstd", rstd),
        y: dispatch::storage_zeroed(ctx, "gemv-i8-y", (m * n * 4) as u64),
        params: dispatch::uniform_from(ctx, "gemv-i8-params", &params),
        groups,
    })
}

pub fn gemv_i8_normed(
    ctx: &WgpuContext,
    w_i8: &[i8],
    w_scales: &[f32],
    x: &[u16],
    norm_weight: &[u16],
    rstd: f32,
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("gemv_i8_normed y", y.len(), n)?;
    if n == 0 || k == 0 {
        return Ok(());
    }
    if k > MAX_SHARED_K {
        return Err(WgpuError::Shape(format!(
            "gemv_i8_normed needs K<={MAX_SHARED_K}; got {k}"
        )));
    }
    let p = i8_plan(ctx, w_i8, w_scales, x, norm_weight, &[rstd], 1, n, k)?;
    dispatch::run(
        ctx,
        "nv_kernels_gemv_i8_normed",
        &compose(WGSL),
        I8_NORMED_ENTRY,
        &p.bindings(),
        p.groups,
    )?;
    store_bf16(&dispatch::read_back::<u32>(ctx, &p.y, n)?, y);
    Ok(())
}

pub fn gemv_i8_normed_mk(
    ctx: &WgpuContext,
    w_i8: &[i8],
    w_scales: &[f32],
    x: &[u16],
    norm_weight: &[u16],
    rstd: &[f32],
    y: &mut [u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("gemv_i8_normed_mk y", y.len(), m * n)?;
    if n == 0 || k == 0 {
        return Ok(());
    }
    let p = i8_plan(ctx, w_i8, w_scales, x, norm_weight, rstd, m, n, k)?;
    dispatch::run(
        ctx,
        "nv_kernels_gemv_i8_normed_mk",
        &compose(WGSL),
        I8_NORMED_MK_ENTRY,
        &p.bindings(),
        p.groups,
    )?;
    store_bf16(&dispatch::read_back::<u32>(ctx, &p.y, m * n)?, y);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_follows_the_cuda_vector_predicate() {
        assert_eq!(entry_for(4096), VEC8_ENTRY);
        assert_eq!(entry_for(8), VEC8_ENTRY);
        assert_eq!(entry_for(6), SCALAR_ENTRY);
        assert_eq!(entry_for(4098), SCALAR_ENTRY);
    }

    #[test]
    fn wgsl_declares_both_entry_points() {
        assert!(WGSL.contains(VEC8_ENTRY));
        assert!(WGSL.contains(SCALAR_ENTRY));
        assert!(WGSL.contains(V4_TREE_ENTRY));
        assert!(compose(WGSL).contains("fn bf16_encode("));
    }

    #[test]
    fn base_wgsl_stays_free_of_subgroup_extensions() {
        assert!(!WGSL.contains("subgroup"));
    }

    #[test]
    fn sg_source_declares_the_subgroup_entries() {
        let src = sg_source();
        assert!(src.contains("subgroupShuffleXor"));
        for (_, entry, _) in SG_V4_WG {
            assert!(src.contains(entry), "missing entry {entry}");
        }
        assert!(src.contains(SG_U32_ENTRY));
        assert!(src.contains(SG_SCALAR_ENTRY));
        assert!(src.contains("fn bf16_encode("));
    }

    #[test]
    fn the_width_this_adapter_runs_at_is_the_probe_when_it_answered() {
        assert_eq!(known_subgroup_width(true, 4, 64, Some(32)), Some(32));
        assert_eq!(known_subgroup_width(true, 4, 64, Some(16)), Some(16));
        assert_eq!(known_subgroup_width(true, 4, 64, None), None);
        assert_eq!(known_subgroup_width(true, 32, 32, None), Some(32));
        assert_eq!(known_subgroup_width(true, 32, 32, Some(64)), Some(64));
        assert_eq!(known_subgroup_width(false, 32, 32, Some(32)), None);
    }

    #[test]
    fn the_advertised_range_alone_never_decides_the_warp32_gate() {
        assert!(sg32_from(true, 4, 64, Some(LANES_PER_ROW)));
        assert!(sg32_from(true, LANES_PER_ROW, LANES_PER_ROW, None));
        assert!(!sg32_from(true, 4, 64, None));
        assert!(!sg32_from(true, 4, 64, Some(16)));
        assert!(!sg32_from(true, LANES_PER_ROW, LANES_PER_ROW, Some(64)));
        assert!(!sg32_from(false, LANES_PER_ROW, LANES_PER_ROW, Some(32)));
    }

    #[test]
    fn a_4_to_64_adapter_probing_32_reaches_the_warp32_kernel() {
        assert_eq!(
            select_kernel_from(true, 4, 64, Some(32), 4096),
            GemvKernel::SgV4 { wg: SG_DEFAULT_WG }
        );
        assert_eq!(
            select_kernel_from(true, 4, 64, Some(32), 4098),
            GemvKernel::SgScalar
        );
        assert_eq!(
            select_kernel_from(true, 4, 64, None, 4096),
            GemvKernel::TreeVec8
        );
        assert_eq!(
            select_kernel_from(true, 4, 64, None, 4098),
            GemvKernel::TreeScalar
        );
    }

    #[test]
    fn a_probed_width_other_than_32_never_reaches_a_warp32_kernel() {
        for probed in [4u32, 8, 16, 64, 128] {
            for k in [4096usize, 4098] {
                let picked = select_kernel_from(true, 4, 64, Some(probed), k);
                assert!(
                    !matches!(picked, GemvKernel::SgV4 { .. } | GemvKernel::SgScalar),
                    "probed subgroup width {probed} routed k={k} to {picked:?}, whose body strides \
                     by {LANES_PER_ROW} lanes and folds a {LANES_PER_ROW}-lane butterfly"
                );
            }
        }
        assert_eq!(
            select_kernel_from(true, 4, 64, Some(16), 4096),
            GemvKernel::SgV4Adaptive {
                wg: SG_DEFAULT_WG,
                x: 16
            }
        );
        assert_eq!(
            select_kernel_from(true, 4, 64, Some(64), 4096),
            GemvKernel::SgV4Adaptive {
                wg: SG_DEFAULT_WG,
                x: 32
            }
        );
        assert_eq!(
            select_kernel_from(true, 4, 64, Some(4), 4096),
            GemvKernel::TreeVec8
        );
    }

    #[test]
    fn the_adaptive_width_never_exceeds_the_width_the_lanes_actually_share() {
        for (min_size, max_size, probed) in [
            (4u32, 64u32, Some(8u32)),
            (4, 64, Some(16)),
            (4, 64, Some(32)),
            (4, 64, Some(64)),
            (8, 32, None),
            (16, 16, None),
            (64, 64, None),
        ] {
            let real = probed.unwrap_or(min_size);
            let x = adaptive_width(true, min_size, max_size, probed).unwrap();
            assert!(
                x <= real && real.is_multiple_of(x),
                "adaptive x={x} for advertised {min_size}..{max_size} probe {probed:?} does not \
                 divide the {real}-lane subgroup its shuffles stay inside"
            );
        }
        assert_eq!(adaptive_width(true, 4, 64, None), None);
        assert_eq!(adaptive_width(true, 4, 4, None), None);
        assert_eq!(adaptive_width(true, 12, 12, None), None);
        assert_eq!(adaptive_width(true, 32, 16, None), None);
        assert_eq!(adaptive_width(false, 32, 32, Some(32)), None);
    }

    #[test]
    fn kernel_rows_track_workgroup_size() {
        assert_eq!(GemvKernel::SgV4 { wg: 64 }.rows_per_group(), 2);
        assert_eq!(GemvKernel::SgV4 { wg: 512 }.rows_per_group(), 16);
        assert_eq!(GemvKernel::TreeVec8.rows_per_group(), ROWS_PER_GROUP);
        assert_eq!(GemvKernel::SgScalar.rows_per_group(), ROWS_PER_GROUP);
        assert_eq!(
            GemvKernel::SgV4 { wg: 128 }.entry(),
            "gemv_bf16_sg_v4_wg128"
        );
        assert_eq!(GemvKernel::TreeVec8V4.slots(), (20, 21));
    }

    #[test]
    fn u16_word_packing_matches_the_shader_layout() {
        let src: Vec<u16> = vec![0x1234, 0xabcd, 0x0001, 0xffff];
        let words = pack_u16(&src);
        assert_eq!(words, vec![0xabcd_1234u32, 0xffff_0001u32]);
    }

    #[test]
    fn wgsl_declares_the_int8_sibling_entry_points() {
        for entry in [
            NORMED_ENTRY,
            ROWQUANT_ENTRY,
            I8_NORMED_ENTRY,
            I8_NORMED_MK_ENTRY,
        ] {
            assert!(WGSL.contains(entry), "missing entry {entry}");
        }
        assert!(compose(WGSL).contains("fn int8_decode("));
    }

    #[test]
    fn odd_k_rows_are_padded_to_whole_words() {
        let src: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        let words = pack_u16_rows(&src, 2, 3);
        assert_eq!(
            words,
            vec![0x0002_0001u32, 0x0000_0003, 0x0005_0004, 0x0000_0006]
        );
    }

    #[test]
    fn int8_row_packing_round_trips() {
        let src: Vec<i8> = vec![1, -2, 3, -4, 5, -128, 127, 0, 9, -9];
        let words = pack_i8_rows(&src, 2, 5);
        assert_eq!(words.len(), 4);
        let mut back = vec![0i8; 10];
        unpack_i8_rows(&words, &mut back, 2, 5);
        assert_eq!(back, src);
    }

    #[test]
    fn one_over_127_constant_matches_the_shader() {
        assert!(WGSL.contains("0x3c010204u"));
        assert_eq!((1.0f32 / 127.0f32).to_bits(), 0x3c01_0204);
    }
}
