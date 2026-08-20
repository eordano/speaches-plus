#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/rope.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    half_dim: u32,
    total_heads: u32,
    total_pairs: u32,
    reserved: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "rope", WORKGROUP_SIZE)
}

fn plan(
    q: usize,
    k: usize,
    cos_tbl: usize,
    sin_tbl: usize,
    positions: usize,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<RopeParams> {
    if !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "rope head_dim must be even; got {head_dim}"
        )));
    }
    let half_dim = head_dim / 2;
    dispatch::check_len("rope q", q, batch * n_heads * head_dim)?;
    dispatch::check_len("rope k", k, batch * n_kv_heads * head_dim)?;
    dispatch::check_len("rope positions", positions, batch)?;
    if cos_tbl != sin_tbl {
        return Err(WgpuError::Shape(format!(
            "rope cos/sin tables must be the same length; got {cos_tbl} and {sin_tbl}"
        )));
    }
    if half_dim > 0 && !cos_tbl.is_multiple_of(half_dim) {
        return Err(WgpuError::Shape(format!(
            "rope cos/sin table length {cos_tbl} is not a multiple of head_dim/2 = {half_dim}"
        )));
    }
    let total_heads = n_heads + n_kv_heads;
    let total_pairs = batch * total_heads * half_dim;
    Ok(RopeParams {
        batch: batch as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        half_dim: half_dim as u32,
        total_heads: total_heads as u32,
        total_pairs: total_pairs as u32,
        reserved: 0,
    })
}

fn check_positions(positions: &[i32], table_rows: usize) -> Result<()> {
    for (i, p) in positions.iter().enumerate() {
        if *p < 0 || (*p as usize) >= table_rows {
            return Err(WgpuError::Shape(format!(
                "rope position[{i}] = {p} is outside the {table_rows}-row cos/sin table"
            )));
        }
    }
    Ok(())
}

fn storage_or_stub<T: bytemuck::Pod>(ctx: &WgpuContext, label: &str, data: &[T]) -> wgpu::Buffer {
    if data.is_empty() {
        dispatch::storage_zeroed(ctx, label, 4)
    } else {
        dispatch::storage_from_slice(ctx, label, data)
    }
}

pub fn rope_f32(
    ctx: &WgpuContext,
    q: &mut [f32],
    k: &mut [f32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let params = plan(
        q.len(),
        k.len(),
        cos_tbl.len(),
        sin_tbl.len(),
        positions.len(),
        batch,
        n_heads,
        n_kv_heads,
        head_dim,
    )?;
    if params.total_pairs == 0 {
        return Ok(());
    }
    check_positions(positions, cos_tbl.len() / (head_dim / 2))?;
    check_device(ctx)?;

    let qb = storage_or_stub(ctx, "rope-f32-q", q);
    let kb = storage_or_stub(ctx, "rope-f32-k", k);
    let cb = dispatch::storage_from_slice(ctx, "rope-f32-cos", cos_tbl);
    let sb = dispatch::storage_from_slice(ctx, "rope-f32-sin", sin_tbl);
    let pb = dispatch::storage_from_slice(ctx, "rope-f32-positions", positions);
    let ub = dispatch::uniform_from(ctx, "rope-f32-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, params.total_pairs as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_rope_f32",
        &compose(WGSL),
        "rope_f32",
        &[(0, &qb), (1, &kb), (2, &cb), (3, &sb), (4, &pb), (5, &ub)],
        groups,
    )?;

    if !q.is_empty() {
        let out: Vec<f32> = dispatch::read_back(ctx, &qb, q.len())?;
        q.copy_from_slice(&out);
    }
    if !k.is_empty() {
        let out: Vec<f32> = dispatch::read_back(ctx, &kb, k.len())?;
        k.copy_from_slice(&out);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_flattens_the_cuda_grid() {
        let p = plan(2 * 4 * 8, 2 * 2 * 8, 16 * 4, 16 * 4, 2, 2, 4, 2, 8).unwrap();
        assert_eq!(p.half_dim, 4);
        assert_eq!(p.total_heads, 6);
        assert_eq!(p.total_pairs, 2 * 6 * 4);
    }

    #[test]
    fn odd_head_dim_is_rejected() {
        let e = plan(0, 0, 0, 0, 0, 0, 0, 0, 7).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let e = plan(10, 0, 8, 8, 1, 1, 2, 0, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn out_of_range_position_is_reported() {
        assert!(check_positions(&[0, 3], 4).is_ok());
        assert!(check_positions(&[0, 4], 4).is_err());
        assert!(check_positions(&[-1], 4).is_err());
    }
}
