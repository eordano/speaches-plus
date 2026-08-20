#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::kernels::flash_decode as fd;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub use crate::wgpu_backend::kernels::flash_decode::WGSL;

pub const WORKGROUP_SIZE: u32 = fd::WORKGROUP_SIZE;
pub const MAX_M: usize = fd::MAX_MK_ROWS;
pub const MAX_HEAD_DIM: usize = fd::MAX_HEAD_DIM_MK;

pub const ENTRY_STAGE1_F32: &str = "flash_smv2_stage1_f32";
pub const ENTRY_STAGE1_BF16: &str = "flash_smv2_stage1_bf16kv";
pub const ENTRY_STAGE1_FP8: &str = "flash_smv2_stage1_fp8kv";
pub const ENTRY_STAGE2: &str = fd::ENTRY_STAGE2_MK_U;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SmV2Params {
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

pub fn scratch_elems(n_heads: usize, head_dim: usize, m: usize, splits: usize) -> Result<usize> {
    fd::flash_splitk_scratch_elems_mk(n_heads, head_dim, m, splits)
}

struct Plan {
    params: SmV2Params,
    scratch_elems: usize,
    n_heads: usize,
    head_dim: usize,
    m: usize,
}

fn plan(
    ctx: &WgpuContext,
    q_len: usize,
    kv_len_elems: usize,
    out_len: usize,
    scratch_len: usize,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    total: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
    fused: bool,
    out_bf16: bool,
) -> Result<Plan> {
    if head_dim == 0 || head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_v2 head_dim {head_dim} out of range 1..={MAX_HEAD_DIM}"
        )));
    }
    if !(1..=MAX_M).contains(&m) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_v2 m_rows {m} out of range 1..={MAX_M}"
        )));
    }
    if n_heads == 0 || n_kv_heads == 0 {
        return Err(WgpuError::Shape(
            "attn_decode_small_m_v2 n_heads and n_kv_heads must be non-zero".to_string(),
        ));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_v2 n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    if total < m {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_v2 total {total} is smaller than m_rows {m}"
        )));
    }
    if ctx.caps.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || ctx.caps.max_compute_workgroup_size_x < WORKGROUP_SIZE
    {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_v2 needs a {WORKGROUP_SIZE}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    if !ctx.caps.workgroup_storage_fits(fd::SCRATCH_BYTES_MK) {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_v2 scratch needs {} bytes of workgroup storage; device allows {}",
            fd::SCRATCH_BYTES_MK,
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    let splits = fd::normalize_splits(splits);
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    if n_heads > limit || splits.max(m) > limit {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_v2 grid {n_heads}x{} exceeds max_compute_workgroups_per_dimension {limit}",
            splits.max(m)
        )));
    }
    if ring > 0 && window == 0 {
        return Err(WgpuError::Shape(
            "attn_decode_small_m_v2 ring>0 requires window>0".to_string(),
        ));
    }
    dispatch::check_len("attn_decode_small_m_v2 q", q_len, m * n_heads * head_dim)?;
    dispatch::check_len(
        "attn_decode_small_m_v2 out",
        out_len,
        m * n_heads * head_dim,
    )?;
    let per_slot = n_kv_heads * head_dim;
    if !kv_len_elems.is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_v2 k_cache: length {kv_len_elems} is not a multiple of {per_slot}"
        )));
    }
    let slots = kv_len_elems / per_slot;
    if ring > 0 {
        if ring > slots {
            return Err(WgpuError::Shape(format!(
                "attn_decode_small_m_v2 ring {ring} exceeds cache slots {slots}"
            )));
        }
    } else if total > slots {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_v2 k_cache holds {slots} slots but total is {total}"
        )));
    }
    let scratch_elems = scratch_elems(n_heads, head_dim, m, splits)?;
    if scratch_len < scratch_elems {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_v2 scratch: got {scratch_len} want at least {scratch_elems}"
        )));
    }
    Ok(Plan {
        params: SmV2Params {
            n_heads: n_heads as u32,
            n_kv: n_kv_heads as u32,
            head_dim: head_dim as u32,
            total: total as u32,
            start: 0,
            splits: splits as u32,
            ring: ring as u32,
            out_bf16: u32::from(out_bf16),
            scaling,
            pad0: 0,
            fused: u32::from(fused),
            pad2: 0,
            m_rows: m as u32,
            window: window as u32,
            pad3: 0,
            pad4: 0,
        },
        scratch_elems,
        n_heads,
        head_dim,
        m,
    })
}

fn run_stages(
    ctx: &WgpuContext,
    plan: &Plan,
    stage1_entry: &'static str,
    stage1_bindings: &[(u32, &wgpu::Buffer)],
    sb: &wgpu::Buffer,
    ob: &wgpu::Buffer,
    pb: &wgpu::Buffer,
    scratch: &mut [f32],
) -> Result<Vec<u32>> {
    let source = compose(WGSL);
    dispatch::run(
        ctx,
        stage1_entry,
        &source,
        stage1_entry,
        stage1_bindings,
        (plan.n_heads as u32, plan.params.splits, 1),
    )?;
    dispatch::run(
        ctx,
        ENTRY_STAGE2,
        &source,
        ENTRY_STAGE2,
        &[(3, ob), (4, pb), (7, sb)],
        (plan.n_heads as u32, plan.m as u32, 1),
    )?;
    let got_scratch: Vec<f32> = dispatch::read_back(ctx, sb, plan.scratch_elems)?;
    scratch[..plan.scratch_elems].copy_from_slice(&got_scratch);
    dispatch::read_back(ctx, ob, plan.m * plan.n_heads * plan.head_dim)
}

pub fn attn_decode_small_m_v2_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    fused: bool,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len(
        "attn_decode_small_m_v2 v_cache",
        v_cache.len(),
        k_cache.len(),
    )?;
    let plan = plan(
        ctx,
        q.len(),
        k_cache.len(),
        out.len(),
        scratch.len(),
        m_rows,
        n_heads,
        n_kv_heads,
        head_dim,
        total,
        window,
        scaling,
        splits,
        0,
        fused,
        false,
    )?;

    let qb = dispatch::storage_from_slice(ctx, "smv2-q", q);
    let kb = dispatch::storage_from_slice(ctx, "smv2-k-f32", k_cache);
    let vb = dispatch::storage_from_slice(ctx, "smv2-v-f32", v_cache);
    let sb = dispatch::storage_zeroed(ctx, "smv2-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(
        ctx,
        "smv2-out",
        (plan.m * plan.n_heads * plan.head_dim * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "smv2-params", &plan.params);
    let words = run_stages(
        ctx,
        &plan,
        ENTRY_STAGE1_F32,
        &[(0, &qb), (1, &kb), (2, &vb), (4, &pb), (7, &sb)],
        &sb,
        &ob,
        &pb,
        scratch,
    )?;
    fd::words_to_f32(&words, out);
    Ok(())
}

pub fn attn_decode_small_m_v2_bf16kv(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    out: &mut [u16],
    scratch: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
    fused: bool,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len(
        "attn_decode_small_m_v2 v_cache",
        v_cache.len(),
        k_cache.len(),
    )?;
    let plan = plan(
        ctx,
        q.len(),
        k_cache.len(),
        out.len(),
        scratch.len(),
        m_rows,
        n_heads,
        n_kv_heads,
        head_dim,
        total,
        window,
        scaling,
        splits,
        ring,
        fused,
        true,
    )?;

    let qb = dispatch::storage_from_slice(ctx, "smv2-q", q);
    let kb = dispatch::storage_from_slice(ctx, "smv2-k-bf16", &fd::pack_u16(k_cache));
    let vb = dispatch::storage_from_slice(ctx, "smv2-v-bf16", &fd::pack_u16(v_cache));
    let sb = dispatch::storage_zeroed(ctx, "smv2-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(
        ctx,
        "smv2-out",
        (plan.m * plan.n_heads * plan.head_dim * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "smv2-params", &plan.params);
    let words = run_stages(
        ctx,
        &plan,
        ENTRY_STAGE1_BF16,
        &[(0, &qb), (4, &pb), (5, &kb), (6, &vb), (7, &sb)],
        &sb,
        &ob,
        &pb,
        scratch,
    )?;
    fd::words_to_u16(&words, out);
    Ok(())
}

pub fn attn_decode_small_m_v2_fp8kv(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [u16],
    scratch: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("attn_decode_small_m_v2 v_fp8", v_fp8.len(), k_fp8.len())?;
    let plan = plan(
        ctx,
        q.len(),
        k_fp8.len(),
        out.len(),
        scratch.len(),
        m_rows,
        n_heads,
        n_kv_heads,
        head_dim,
        total,
        window,
        scaling,
        splits,
        ring,
        false,
        true,
    )?;
    let slots = k_fp8.len() / (n_kv_heads * head_dim);
    dispatch::check_len(
        "attn_decode_small_m_v2 k_scales",
        k_scales.len(),
        slots * n_kv_heads,
    )?;
    dispatch::check_len(
        "attn_decode_small_m_v2 v_scales",
        v_scales.len(),
        slots * n_kv_heads,
    )?;

    let q_f32: Vec<f32> = q
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();

    let qb = dispatch::storage_from_slice(ctx, "smv2-q", &q_f32);
    let kb = dispatch::storage_from_slice(ctx, "smv2-k-fp8", &fd::pack_u8(k_fp8));
    let vb = dispatch::storage_from_slice(ctx, "smv2-v-fp8", &fd::pack_u8(v_fp8));
    let ksb = dispatch::storage_from_slice(ctx, "smv2-k-scales", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "smv2-v-scales", v_scales);
    let sb = dispatch::storage_zeroed(ctx, "smv2-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(
        ctx,
        "smv2-out",
        (plan.m * plan.n_heads * plan.head_dim * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "smv2-params", &plan.params);
    let words = run_stages(
        ctx,
        &plan,
        ENTRY_STAGE1_FP8,
        &[
            (0, &qb),
            (4, &pb),
            (5, &kb),
            (6, &vb),
            (7, &sb),
            (8, &ksb),
            (9, &vsb),
        ],
        &sb,
        &ob,
        &pb,
        scratch,
    )?;
    fd::words_to_u16(&words, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<SmV2Params>() % 16, 0);
        assert_eq!(
            std::mem::size_of::<SmV2Params>(),
            16 * std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn scratch_accounting_matches_the_flash_mk_formula() {
        assert_eq!(scratch_elems(8, 256, 4, 16).unwrap(), 8 * 4 * 16 * 258);
        assert_eq!(scratch_elems(4, 128, 1, 0).unwrap(), 4 * 16 * 130);
        assert_eq!(
            scratch_elems(32, 256, 8, 32).unwrap(),
            fd::flash_splitk_scratch_elems_mk(32, 256, 8, 32).unwrap()
        );
        assert!(matches!(
            scratch_elems(4, 512, 2, 16).unwrap_err(),
            WgpuError::Unsupported(_)
        ));
        assert!(matches!(
            scratch_elems(4, 128, 0, 16).unwrap_err(),
            WgpuError::Shape(_)
        ));
        assert!(matches!(
            scratch_elems(4, 128, MAX_M + 1, 16).unwrap_err(),
            WgpuError::Shape(_)
        ));
    }

    #[test]
    fn wgsl_declares_every_v2_entry_point() {
        for entry in [
            ENTRY_STAGE1_F32,
            ENTRY_STAGE1_BF16,
            ENTRY_STAGE1_FP8,
            ENTRY_STAGE2,
        ] {
            assert!(WGSL.contains(&format!("fn {entry}(")), "missing {entry}");
        }
        assert!(compose(WGSL).contains("fn e4m3_decode("));
    }

    #[test]
    fn limits_track_the_flash_mk_contract() {
        assert_eq!(MAX_M, 8);
        assert_eq!(MAX_HEAD_DIM, 256);
        assert_eq!(WORKGROUP_SIZE, 256);
    }
}
