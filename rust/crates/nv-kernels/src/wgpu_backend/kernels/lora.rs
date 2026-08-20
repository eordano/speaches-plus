#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_u16};

pub const GROUPED_WGSL: &str = include_str!("../../../wgsl/lora_grouped.wgsl");
pub const FUSED_WGSL: &str = include_str!("../../../wgsl/lora_fused.wgsl");

pub const SHRINK_ENTRY: &str = "lora_shrink";
pub const EXPAND_ENTRY: &str = "lora_expand";
pub const FUSED_ENTRY: &str = "lora_fused";

pub const BLOCK_M: u32 = 16;
pub const BLOCK_N: u32 = 16;
pub const FUSED_N_CHUNK: u32 = 512;
pub const FUSED_WARPS: u32 = 16;
pub const FUSED_LANES: u32 = 32;
pub const FUSED_MAX_RANK: usize = 64;

const GROUPED_INVOCATIONS: u32 = BLOCK_M * BLOCK_N;
const FUSED_INVOCATIONS: u32 = FUSED_LANES * FUSED_WARPS;
const FUSED_SCRATCH_BYTES: u32 = (FUSED_INVOCATIONS + FUSED_MAX_RANK as u32) * 4;

pub use crate::lora_meta::{LoraKernelMeta as LoraMeta, NO_LORA};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GroupedParams {
    m: u32,
    rank: u32,
    k: u32,
    cta_m_num: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    y_row_stride: u32,
    scale: f32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedParams {
    m: u32,
    rank: u32,
    k: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    y_row_stride: u32,
    win_off: u32,
    win_len: u32,
    scale: f32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
}

struct MetaLayout {
    data: Vec<i32>,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
}

fn build_meta(meta: &LoraMeta, m: usize, widths: &[usize], rank: usize) -> Result<MetaLayout> {
    dispatch::check_len(
        "lora token_indices_sorted",
        meta.token_indices_sorted.len(),
        m,
    )?;
    dispatch::check_len(
        "lora num_tokens_per_lora",
        meta.num_tokens_per_lora.len(),
        meta.max_loras + 1,
    )?;
    dispatch::check_len(
        "lora lora_token_start_loc",
        meta.lora_token_start_loc.len(),
        meta.max_loras + 2,
    )?;
    dispatch::check_len(
        "lora active_lora_ids",
        meta.active_lora_ids.len(),
        meta.max_loras + 1,
    )?;

    let mut data = Vec::new();
    data.extend_from_slice(&meta.token_indices_sorted);
    let off_counts = data.len() as u32;
    data.extend_from_slice(&meta.num_tokens_per_lora);
    let off_start = data.len() as u32;
    data.extend_from_slice(&meta.lora_token_start_loc);
    let off_active = data.len() as u32;
    data.extend_from_slice(&meta.active_lora_ids);
    let off_slice_n = data.len() as u32;
    for &w in widths {
        data.push(i32::try_from(w).map_err(|_| overflow("slice_n", w))?);
    }
    let off_slice_start = data.len() as u32;
    let mut acc = 0usize;
    for &w in widths {
        data.push(i32::try_from(acc).map_err(|_| overflow("slice_start", acc))?);
        acc += w;
    }
    let off_b_off = data.len() as u32;
    let mut b_acc = 0usize;
    for &w in widths {
        data.push(i32::try_from(b_acc).map_err(|_| overflow("b_off", b_acc))?);
        b_acc += meta.max_loras * w * rank;
    }
    let off_b_d0 = data.len() as u32;
    for &w in widths {
        let d0 = w * rank;
        data.push(i32::try_from(d0).map_err(|_| overflow("b_d0", d0))?);
    }

    Ok(MetaLayout {
        data,
        off_counts,
        off_start,
        off_active,
        off_slice_n,
        off_slice_start,
        off_b_off,
        off_b_d0,
    })
}

fn overflow(what: &str, v: usize) -> WgpuError {
    WgpuError::Shape(format!("lora {what} {v} exceeds i32 range"))
}

fn concat_check(what: &str, slices: &[&[u16]], want_each: &[usize]) -> Result<Vec<u16>> {
    let mut out = Vec::new();
    for (s, part) in slices.iter().enumerate() {
        dispatch::check_len(&format!("lora {what}[{s}]"), part.len(), want_each[s])?;
        out.extend_from_slice(part);
    }
    Ok(out)
}

fn widen_y(y: &[u16]) -> Vec<u32> {
    y.iter().map(|&v| v as u32).collect()
}

fn store_y(words: &[u32], y: &mut [u16]) {
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
}

fn check_device(ctx: &WgpuContext, invocations: u32, scratch_bytes: u32) -> Result<()> {
    if ctx.caps.max_compute_invocations_per_workgroup < invocations {
        return Err(WgpuError::Unsupported(format!(
            "lora needs a {invocations}-invocation workgroup; device allows {}",
            ctx.caps.max_compute_invocations_per_workgroup
        )));
    }
    if !ctx.caps.workgroup_storage_fits(scratch_bytes) {
        return Err(WgpuError::Unsupported(format!(
            "lora needs {scratch_bytes} bytes of workgroup storage; device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    Ok(())
}

fn check_grid(ctx: &WgpuContext, grid: (u32, u32, u32)) -> Result<()> {
    let limit = ctx.caps.max_compute_workgroups_per_dimension;
    for (axis, g) in [("x", grid.0), ("y", grid.1), ("z", grid.2)] {
        if g == 0 || g > limit {
            return Err(WgpuError::Shape(format!(
                "lora grid {axis}={g} outside 1..={limit}"
            )));
        }
    }
    Ok(())
}

struct Shapes {
    m: usize,
    rank: usize,
    k: usize,
    n_slices: usize,
    max_n: usize,
    a_d0_stride: usize,
    a_slice_stride: usize,
}

fn check_shapes(
    meta: &LoraMeta,
    widths: &[usize],
    m: usize,
    rank: usize,
    k: usize,
) -> Result<Shapes> {
    if m == 0 || rank == 0 || k == 0 || widths.is_empty() {
        return Err(WgpuError::Shape(format!(
            "lora dims must be positive: m={m} rank={rank} k={k} n_slices={}",
            widths.len()
        )));
    }
    if widths.contains(&0) {
        return Err(WgpuError::Shape("lora slice width 0".into()));
    }
    let a_d0_stride = rank * k;
    Ok(Shapes {
        m,
        rank,
        k,
        n_slices: widths.len(),
        max_n: *widths.iter().max().unwrap(),
        a_d0_stride,
        a_slice_stride: meta.max_loras * a_d0_stride,
    })
}

pub fn lora_shrink(
    ctx: &WgpuContext,
    x_bf16: &[u16],
    a_slices: &[&[u16]],
    buffer: &mut [f32],
    meta: &LoraMeta,
    m: usize,
    rank: usize,
    k: usize,
    scale: f32,
) -> Result<()> {
    let widths = vec![1usize; a_slices.len()];
    let s = check_shapes(meta, &widths, m, rank, k)?;
    dispatch::check_len("lora x", x_bf16.len(), s.m * s.k)?;
    dispatch::check_len("lora buffer", buffer.len(), s.n_slices * s.m * s.rank)?;
    check_device(ctx, GROUPED_INVOCATIONS, 0)?;
    let ml = build_meta(meta, s.m, &widths, s.rank)?;
    let a_all = concat_check("a", a_slices, &vec![s.a_slice_stride; s.n_slices])?;

    let cta_m_num = s.m.div_ceil(BLOCK_M as usize) as u32;
    let cta_n_num = s.rank.div_ceil(BLOCK_N as usize) as u32;
    let grid = (
        cta_m_num * cta_n_num,
        s.n_slices as u32,
        meta.grid_loras() as u32,
    );
    check_grid(ctx, grid)?;

    let params = GroupedParams {
        m: s.m as u32,
        rank: s.rank as u32,
        k: s.k as u32,
        cta_m_num,
        a_slice_stride: s.a_slice_stride as u32,
        a_d0_stride: s.a_d0_stride as u32,
        y_row_stride: 0,
        scale,
        off_counts: ml.off_counts,
        off_start: ml.off_start,
        off_active: ml.off_active,
        off_slice_n: ml.off_slice_n,
        off_slice_start: ml.off_slice_start,
        off_b_off: ml.off_b_off,
        pad0: 0,
        pad1: 0,
    };

    let x_buf = dispatch::storage_from_slice(ctx, "lora-shrink-x", &pack_u16(x_bf16));
    let a_buf = dispatch::storage_from_slice(ctx, "lora-shrink-a", &pack_u16(&a_all));
    let buf = dispatch::storage_zeroed(ctx, "lora-shrink-buf", (buffer.len() * 4) as u64);
    let meta_buf = dispatch::storage_from_slice(ctx, "lora-shrink-meta", &ml.data);
    let params_buf = dispatch::uniform_from(ctx, "lora-shrink-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_lora_shrink",
        &compose(GROUPED_WGSL),
        SHRINK_ENTRY,
        &[
            (0, &x_buf),
            (1, &a_buf),
            (2, &buf),
            (3, &meta_buf),
            (4, &params_buf),
        ],
        grid,
    )?;
    let out: Vec<f32> = dispatch::read_back(ctx, &buf, buffer.len())?;
    buffer.copy_from_slice(&out);
    Ok(())
}

pub fn lora_expand(
    ctx: &WgpuContext,
    buffer: &[f32],
    b_slices: &[&[u16]],
    y_bf16: &mut [u16],
    meta: &LoraMeta,
    widths: &[usize],
    m: usize,
    rank: usize,
    y_row_stride: usize,
) -> Result<()> {
    let s = check_shapes(meta, widths, m, rank, 1)?;
    dispatch::check_len("lora b_slices", b_slices.len(), s.n_slices)?;
    dispatch::check_len("lora buffer", buffer.len(), s.n_slices * s.m * s.rank)?;
    dispatch::check_len("lora y", y_bf16.len(), s.m * y_row_stride)?;
    let sum_n: usize = widths.iter().sum();
    if y_row_stride < sum_n {
        return Err(WgpuError::Shape(format!(
            "lora y_row_stride {y_row_stride} < sum of slice widths {sum_n}"
        )));
    }
    check_device(ctx, GROUPED_INVOCATIONS, 0)?;
    let ml = build_meta(meta, s.m, widths, s.rank)?;
    let b_want: Vec<usize> = widths
        .iter()
        .map(|&w| meta.max_loras * w * s.rank)
        .collect();
    let b_all = concat_check("b", b_slices, &b_want)?;

    let cta_m_num = s.m.div_ceil(BLOCK_M as usize) as u32;
    let cta_n_num = s.max_n.div_ceil(BLOCK_N as usize) as u32;
    let grid = (
        cta_m_num * cta_n_num,
        s.n_slices as u32,
        meta.grid_loras() as u32,
    );
    check_grid(ctx, grid)?;

    let params = GroupedParams {
        m: s.m as u32,
        rank: s.rank as u32,
        k: 0,
        cta_m_num,
        a_slice_stride: 0,
        a_d0_stride: 0,
        y_row_stride: y_row_stride as u32,
        scale: 0.0,
        off_counts: ml.off_counts,
        off_start: ml.off_start,
        off_active: ml.off_active,
        off_slice_n: ml.off_slice_n,
        off_slice_start: ml.off_slice_start,
        off_b_off: ml.off_b_off,
        pad0: 0,
        pad1: 0,
    };

    let buf = dispatch::storage_from_slice(ctx, "lora-expand-buf", buffer);
    let meta_buf = dispatch::storage_from_slice(ctx, "lora-expand-meta", &ml.data);
    let params_buf = dispatch::uniform_from(ctx, "lora-expand-params", &params);
    let b_buf = dispatch::storage_from_slice(ctx, "lora-expand-b", &pack_u16(&b_all));
    let y_buf = dispatch::storage_from_slice(ctx, "lora-expand-y", &widen_y(y_bf16));

    dispatch::run(
        ctx,
        "nv_kernels_lora_expand",
        &compose(GROUPED_WGSL),
        EXPAND_ENTRY,
        &[
            (2, &buf),
            (3, &meta_buf),
            (4, &params_buf),
            (5, &b_buf),
            (6, &y_buf),
        ],
        grid,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, y_bf16.len())?;
    store_y(&words, y_bf16);
    Ok(())
}

pub fn lora_grouped(
    ctx: &WgpuContext,
    x_bf16: &[u16],
    a_slices: &[&[u16]],
    b_slices: &[&[u16]],
    y_bf16: &mut [u16],
    meta: &LoraMeta,
    widths: &[usize],
    m: usize,
    rank: usize,
    k: usize,
    y_row_stride: usize,
    scale: f32,
    buffer_out: Option<&mut [f32]>,
) -> Result<()> {
    let s = check_shapes(meta, widths, m, rank, k)?;
    dispatch::check_len("lora a_slices", a_slices.len(), s.n_slices)?;
    dispatch::check_len("lora b_slices", b_slices.len(), s.n_slices)?;
    dispatch::check_len("lora x", x_bf16.len(), s.m * s.k)?;
    dispatch::check_len("lora y", y_bf16.len(), s.m * y_row_stride)?;
    let sum_n: usize = widths.iter().sum();
    if y_row_stride < sum_n {
        return Err(WgpuError::Shape(format!(
            "lora y_row_stride {y_row_stride} < sum of slice widths {sum_n}"
        )));
    }
    check_device(ctx, GROUPED_INVOCATIONS, 0)?;
    let ml = build_meta(meta, s.m, widths, s.rank)?;
    let a_all = concat_check("a", a_slices, &vec![s.a_slice_stride; s.n_slices])?;
    let b_want: Vec<usize> = widths
        .iter()
        .map(|&w| meta.max_loras * w * s.rank)
        .collect();
    let b_all = concat_check("b", b_slices, &b_want)?;

    let cta_m_num = s.m.div_ceil(BLOCK_M as usize) as u32;
    let shrink_grid = (
        cta_m_num * (s.rank.div_ceil(BLOCK_N as usize) as u32),
        s.n_slices as u32,
        meta.grid_loras() as u32,
    );
    let expand_grid = (
        cta_m_num * (s.max_n.div_ceil(BLOCK_N as usize) as u32),
        s.n_slices as u32,
        meta.grid_loras() as u32,
    );
    check_grid(ctx, shrink_grid)?;
    check_grid(ctx, expand_grid)?;

    let params = GroupedParams {
        m: s.m as u32,
        rank: s.rank as u32,
        k: s.k as u32,
        cta_m_num,
        a_slice_stride: s.a_slice_stride as u32,
        a_d0_stride: s.a_d0_stride as u32,
        y_row_stride: y_row_stride as u32,
        scale,
        off_counts: ml.off_counts,
        off_start: ml.off_start,
        off_active: ml.off_active,
        off_slice_n: ml.off_slice_n,
        off_slice_start: ml.off_slice_start,
        off_b_off: ml.off_b_off,
        pad0: 0,
        pad1: 0,
    };

    let buf_len = s.n_slices * s.m * s.rank;
    let x_buf = dispatch::storage_from_slice(ctx, "lora-grouped-x", &pack_u16(x_bf16));
    let a_buf = dispatch::storage_from_slice(ctx, "lora-grouped-a", &pack_u16(&a_all));
    let b_buf = dispatch::storage_from_slice(ctx, "lora-grouped-b", &pack_u16(&b_all));
    let buf = dispatch::storage_zeroed(ctx, "lora-grouped-buf", (buf_len * 4) as u64);
    let meta_buf = dispatch::storage_from_slice(ctx, "lora-grouped-meta", &ml.data);
    let params_buf = dispatch::uniform_from(ctx, "lora-grouped-params", &params);
    let y_buf = dispatch::storage_from_slice(ctx, "lora-grouped-y", &widen_y(y_bf16));

    let source = compose(GROUPED_WGSL);
    dispatch::run(
        ctx,
        "nv_kernels_lora_shrink",
        &source,
        SHRINK_ENTRY,
        &[
            (0, &x_buf),
            (1, &a_buf),
            (2, &buf),
            (3, &meta_buf),
            (4, &params_buf),
        ],
        shrink_grid,
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_lora_expand",
        &source,
        EXPAND_ENTRY,
        &[
            (2, &buf),
            (3, &meta_buf),
            (4, &params_buf),
            (5, &b_buf),
            (6, &y_buf),
        ],
        expand_grid,
    )?;

    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, y_bf16.len())?;
    store_y(&words, y_bf16);
    if let Some(out) = buffer_out {
        dispatch::check_len("lora buffer_out", out.len(), buf_len)?;
        let got: Vec<f32> = dispatch::read_back(ctx, &buf, buf_len)?;
        out.copy_from_slice(&got);
    }
    Ok(())
}

pub fn lora_fused(
    ctx: &WgpuContext,
    x_bf16: &[u16],
    a_slices: &[&[u16]],
    b_slices: &[&[u16]],
    y_bf16: &mut [u16],
    meta: &LoraMeta,
    widths: &[usize],
    m: usize,
    rank: usize,
    k: usize,
    win_off: usize,
    win_len: usize,
    y_row_stride: usize,
    scale: f32,
) -> Result<()> {
    let s = check_shapes(meta, widths, m, rank, k)?;
    if s.rank > FUSED_MAX_RANK {
        return Err(WgpuError::Shape(format!(
            "lora_fused rank {rank} exceeds FUSED_MAX_RANK {FUSED_MAX_RANK}"
        )));
    }
    if win_len == 0 {
        return Err(WgpuError::Shape(
            "lora_fused win_len must be positive".into(),
        ));
    }
    dispatch::check_len("lora a_slices", a_slices.len(), s.n_slices)?;
    dispatch::check_len("lora b_slices", b_slices.len(), s.n_slices)?;
    dispatch::check_len("lora x", x_bf16.len(), s.m * s.k)?;
    dispatch::check_len("lora y", y_bf16.len(), s.m * y_row_stride)?;
    if y_row_stride < win_len {
        return Err(WgpuError::Shape(format!(
            "lora_fused y_row_stride {y_row_stride} < win_len {win_len}"
        )));
    }
    check_device(ctx, FUSED_INVOCATIONS, FUSED_SCRATCH_BYTES)?;
    let ml = build_meta(meta, s.m, widths, s.rank)?;
    let a_all = concat_check("a", a_slices, &vec![s.a_slice_stride; s.n_slices])?;
    let b_want: Vec<usize> = widths
        .iter()
        .map(|&w| meta.max_loras * w * s.rank)
        .collect();
    let b_all = concat_check("b", b_slices, &b_want)?;

    let cta_n_num = s.max_n.div_ceil(FUSED_N_CHUNK as usize) as u32;
    let grid = (
        (s.m as u32) * cta_n_num,
        s.n_slices as u32,
        meta.grid_loras() as u32,
    );
    check_grid(ctx, grid)?;

    let params = FusedParams {
        m: s.m as u32,
        rank: s.rank as u32,
        k: s.k as u32,
        a_slice_stride: s.a_slice_stride as u32,
        a_d0_stride: s.a_d0_stride as u32,
        y_row_stride: y_row_stride as u32,
        win_off: win_off as u32,
        win_len: win_len as u32,
        scale,
        off_counts: ml.off_counts,
        off_start: ml.off_start,
        off_active: ml.off_active,
        off_slice_n: ml.off_slice_n,
        off_slice_start: ml.off_slice_start,
        off_b_off: ml.off_b_off,
        off_b_d0: ml.off_b_d0,
    };

    let x_buf = dispatch::storage_from_slice(ctx, "lora-fused-x", &pack_u16(x_bf16));
    let a_buf = dispatch::storage_from_slice(ctx, "lora-fused-a", &pack_u16(&a_all));
    let b_buf = dispatch::storage_from_slice(ctx, "lora-fused-b", &pack_u16(&b_all));
    let y_buf = dispatch::storage_from_slice(ctx, "lora-fused-y", &widen_y(y_bf16));
    let meta_buf = dispatch::storage_from_slice(ctx, "lora-fused-meta", &ml.data);
    let params_buf = dispatch::uniform_from(ctx, "lora-fused-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_lora_fused",
        &compose(FUSED_WGSL),
        FUSED_ENTRY,
        &[
            (0, &x_buf),
            (1, &a_buf),
            (2, &b_buf),
            (3, &y_buf),
            (4, &meta_buf),
            (5, &params_buf),
        ],
        grid,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, y_bf16.len())?;
    store_y(&words, y_bf16);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_prepare_matches_vllm_semantics() {
        let meta = LoraMeta::prepare(&[2, -1, 0, 2, -1, 2], 4);
        assert_eq!(meta.token_indices_sorted, vec![1, 4, 2, 0, 3, 5]);
        assert_eq!(meta.active_lora_ids, vec![-1, 0, 2, -1, -1]);
        assert_eq!(meta.num_tokens_per_lora, vec![2, 1, 3, 0, 0]);
        assert_eq!(meta.lora_token_start_loc, vec![0, 2, 3, 6, 0, 0]);
        assert_eq!(meta.num_active_loras, 3);
        assert!(!meta.no_lora);

        let meta = LoraMeta::prepare(&[-1, -1], 2);
        assert!(meta.no_lora);

        let meta = LoraMeta::prepare(&[1, 1, 1], 2);
        assert!(!meta.no_lora);
        assert_eq!(meta.active_lora_ids, vec![1, -1, -1]);
        assert_eq!(meta.num_tokens_per_lora, vec![3, 0, 0]);
    }

    #[test]
    fn meta_layout_sections_are_contiguous() {
        let meta = LoraMeta::prepare(&[0, 1, -1, 0], 2);
        let ml = build_meta(&meta, 4, &[6, 2], 8).unwrap();
        assert_eq!(ml.off_counts, 4);
        assert_eq!(ml.off_start, 7);
        assert_eq!(ml.off_active, 11);
        assert_eq!(ml.off_slice_n, 14);
        assert_eq!(ml.off_slice_start, 16);
        assert_eq!(ml.off_b_off, 18);
        assert_eq!(ml.off_b_d0, 20);
        assert_eq!(ml.data.len(), 22);
        assert_eq!(&ml.data[14..16], &[6, 2]);
        assert_eq!(&ml.data[16..18], &[0, 6]);
        assert_eq!(&ml.data[18..20], &[0, 2 * 6 * 8]);
        assert_eq!(&ml.data[20..22], &[48, 16]);
    }

    #[test]
    fn u16_packing_matches_the_shader_word_layout() {
        assert_eq!(
            pack_u16(&[0x1234, 0xabcd, 0x0001]),
            vec![0xabcd_1234u32, 0x0000_0001]
        );
        assert_eq!(pack_u16(&[]), vec![0u32]);
    }

    #[test]
    fn wgsl_declares_the_entry_points() {
        assert!(GROUPED_WGSL.contains(SHRINK_ENTRY));
        assert!(GROUPED_WGSL.contains(EXPAND_ENTRY));
        assert!(FUSED_WGSL.contains(FUSED_ENTRY));
        assert!(compose(GROUPED_WGSL).contains("fn bf16_encode("));
        assert!(compose(FUSED_WGSL).contains("fn u16_at("));
    }
}
