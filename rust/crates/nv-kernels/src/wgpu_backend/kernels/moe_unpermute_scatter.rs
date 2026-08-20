#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_bf16};

pub const WGSL: &str = include_str!("../../../wgsl/moe_unpermute_scatter.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MusParams {
    n_tokens: u32,
    k: u32,
    hidden: u32,
    row_stride: u32,
    hidden_tiles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "moe_unpermute_scatter", WORKGROUP_SIZE)
}

fn check_shapes(
    y_sorted: usize,
    routing_weights: usize,
    inv_perm: usize,
    out: usize,
    n_tokens: usize,
    k: usize,
    hidden: usize,
    y_sorted_row_stride: usize,
) -> Result<()> {
    dispatch::check_len(
        "moe_unpermute_scatter routing_weights",
        routing_weights,
        n_tokens * k,
    )?;
    dispatch::check_len("moe_unpermute_scatter inv_perm", inv_perm, n_tokens * k)?;
    dispatch::check_len("moe_unpermute_scatter out", out, n_tokens * hidden)?;
    if y_sorted_row_stride < hidden {
        return Err(WgpuError::Shape(format!(
            "moe_unpermute_scatter y_sorted_row_stride {y_sorted_row_stride} < hidden {hidden}"
        )));
    }
    let _ = y_sorted;
    Ok(())
}

fn check_rows(inv_perm: &[i32], y_sorted: usize, hidden: usize, stride: usize) -> Result<()> {
    for (slot, p) in inv_perm.iter().enumerate() {
        if *p < 0 {
            return Err(WgpuError::Shape(format!(
                "moe_unpermute_scatter inv_perm[{slot}] = {p} is negative"
            )));
        }
        let end = (*p as usize) * stride + hidden;
        if end > y_sorted {
            return Err(WgpuError::Shape(format!(
                "moe_unpermute_scatter inv_perm[{slot}] = {p} reads element {end} of a {y_sorted}-element y_sorted"
            )));
        }
    }
    Ok(())
}

pub fn moe_unpermute_scatter(
    ctx: &WgpuContext,
    y_sorted: &[u16],
    routing_weights: &[f32],
    inv_perm: &[i32],
    out: &mut [f32],
    n_tokens: usize,
    k: usize,
    hidden: usize,
    y_sorted_row_stride: usize,
) -> Result<()> {
    check_shapes(
        y_sorted.len(),
        routing_weights.len(),
        inv_perm.len(),
        out.len(),
        n_tokens,
        k,
        hidden,
        y_sorted_row_stride,
    )?;
    if n_tokens == 0 || k == 0 || hidden == 0 {
        return Ok(());
    }
    check_rows(inv_perm, y_sorted.len(), hidden, y_sorted_row_stride)?;
    check_device(ctx)?;

    let hidden_tiles = (hidden as u32).div_ceil(WORKGROUP_SIZE);
    let params = MusParams {
        n_tokens: n_tokens as u32,
        k: k as u32,
        hidden: hidden as u32,
        row_stride: y_sorted_row_stride as u32,
        hidden_tiles,
        ..Default::default()
    };

    let yb = dispatch::storage_from_slice(ctx, "mus-y-sorted", &pack_bf16(y_sorted));
    let wb = dispatch::storage_from_slice(ctx, "mus-weights", routing_weights);
    let ib = dispatch::storage_from_slice(ctx, "mus-inv-perm", inv_perm);
    let ob = dispatch::storage_zeroed(ctx, "mus-out", (n_tokens * hidden * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "mus-params", &params);

    let tiles = n_tokens as u64 * hidden_tiles as u64;
    let groups = dispatch::workgroup_count_1d(ctx, tiles, 1);
    dispatch::run(
        ctx,
        "nv_kernels_moe_unpermute_scatter",
        &compose(WGSL),
        "moe_unpermute_scatter",
        &[(0, &yb), (1, &wb), (2, &ib), (3, &ob), (4, &pb)],
        groups,
    )?;

    let got: Vec<f32> = dispatch::read_back(ctx, &ob, n_tokens * hidden)?;
    out.copy_from_slice(&got);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_handles_odd_lengths() {
        let src: Vec<u16> = vec![1, 2, 3];
        let words = pack_bf16(&src);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], 1 | (2 << 16));
        assert_eq!(words[1], 3);
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let e = check_shapes(64, 7, 8, 16, 4, 2, 4, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn stride_below_hidden_is_reported() {
        let e = check_shapes(64, 8, 8, 16, 4, 2, 4, 2).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn out_of_range_row_is_reported() {
        let e = check_rows(&[0, 9], 16, 4, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        let e = check_rows(&[-1], 16, 4, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        assert!(check_rows(&[0, 3], 16, 4, 4).is_ok());
    }
}
