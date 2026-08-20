#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/attn_decode.wgsl");

pub const WORKGROUP_SIZE: u32 = 128;
pub const MAX_PER_THREAD: usize = 4;
pub const MAX_HEAD_DIM: usize = WORKGROUP_SIZE as usize * MAX_PER_THREAD;

const SCRATCH_BYTES: u32 = (MAX_HEAD_DIM as u32) * 4 + WORKGROUP_SIZE * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    scaling: f32,
    _pad0: u32,
    _pad1: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "attn_decode", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_shapes(
    q: usize,
    k: usize,
    v: usize,
    out: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    total: usize,
) -> Result<()> {
    dispatch::check_len("attn_decode q", q, n_heads * head_dim)?;
    dispatch::check_len("attn_decode out", out, n_heads * head_dim)?;
    dispatch::check_len("attn_decode k", k, total * n_kv_heads * head_dim)?;
    dispatch::check_len("attn_decode v", v, total * n_kv_heads * head_dim)?;
    Ok(())
}

pub fn attn_decode_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start: usize,
    total: usize,
    scaling: f32,
) -> Result<()> {
    check_shapes(
        q.len(),
        k.len(),
        v.len(),
        out.len(),
        n_heads,
        n_kv_heads,
        head_dim,
        total,
    )?;
    if n_heads == 0 || n_kv_heads == 0 || total == 0 {
        return Ok(());
    }
    if head_dim == 0 {
        return Ok(());
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode head_dim {head_dim} exceeds {MAX_HEAD_DIM}"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WgpuError::Shape(format!(
            "attn_decode n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    check_device(ctx)?;

    let params = AttnDecodeParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        total: total as u32,
        start: start.min(total) as u32,
        scaling,
        _pad0: 0,
        _pad1: 0,
    };

    let qb = dispatch::storage_from_slice(ctx, "attn-decode-q", q);
    let kb = dispatch::storage_from_slice(ctx, "attn-decode-k", k);
    let vb = dispatch::storage_from_slice(ctx, "attn-decode-v", v);
    let ob = dispatch::storage_zeroed(ctx, "attn-decode-out", (n_heads * head_dim * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "attn-decode-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_heads as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_attn_decode_f32",
        &compose(WGSL),
        "attn_decode_f32",
        &[(0, &qb), (1, &kb), (2, &vb), (3, &ob), (4, &pb)],
        groups,
    )?;

    let got: Vec<f32> = dispatch::read_back(ctx, &ob, n_heads * head_dim)?;
    out.copy_from_slice(&got);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_is_reported() {
        let e = check_shapes(8, 16, 16, 8, 2, 1, 4, 8).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn matching_shapes_are_accepted() {
        check_shapes(8, 32, 32, 8, 2, 1, 4, 8).unwrap();
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<AttnDecodeParams>() % 16, 0);
    }
}
