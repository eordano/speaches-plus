#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16, unpack_u16_pairs as unpack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/rmsnorm_residual.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 + 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsResParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "rmsnorm_residual", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_shapes(
    x: usize,
    residual: usize,
    weight: usize,
    out: usize,
    batch: usize,
    hidden: usize,
) -> Result<()> {
    dispatch::check_len("rmsnorm_residual x", x, batch * hidden)?;
    dispatch::check_len("rmsnorm_residual residual", residual, batch * hidden)?;
    dispatch::check_len("rmsnorm_residual weight", weight, hidden)?;
    dispatch::check_len("rmsnorm_residual out", out, batch * hidden)?;
    Ok(())
}

pub fn rmsnorm_residual_f32(
    ctx: &WgpuContext,
    x: &[f32],
    residual: &mut [f32],
    weight: &[f32],
    out: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    check_shapes(
        x.len(),
        residual.len(),
        weight.len(),
        out.len(),
        batch,
        hidden,
    )?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let params = RmsResParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: hidden as u32,
    };
    let xb = dispatch::storage_from_slice(ctx, "rmsnorm-residual-f32-x", x);
    let rb = dispatch::storage_from_slice(ctx, "rmsnorm-residual-f32-res", residual);
    let wb = dispatch::storage_from_slice(ctx, "rmsnorm-residual-f32-weight", weight);
    let ob = dispatch::storage_zeroed(ctx, "rmsnorm-residual-f32-out", (batch * hidden * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "rmsnorm-residual-f32-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm_residual_f32",
        &compose(WGSL),
        "rmsnorm_residual_f32",
        &[(0, &xb), (1, &rb), (2, &wb), (3, &ob), (4, &pb)],
        groups,
    )?;

    let new_res: Vec<f32> = dispatch::read_back(ctx, &rb, batch * hidden)?;
    residual.copy_from_slice(&new_res);
    let y: Vec<f32> = dispatch::read_back(ctx, &ob, batch * hidden)?;
    out.copy_from_slice(&y);
    Ok(())
}

pub fn rmsnorm_residual_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    residual: &mut [u16],
    weight: &[u16],
    out: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    check_shapes(
        x.len(),
        residual.len(),
        weight.len(),
        out.len(),
        batch,
        hidden,
    )?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    if !hidden.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "rmsnorm_residual bf16 hidden must be even so whole u32 words are written; got {hidden}"
        )));
    }
    check_device(ctx)?;

    let words_per_row = hidden / 2;
    let params = RmsResParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: words_per_row as u32,
    };
    let xb = dispatch::storage_from_slice(ctx, "rmsnorm-residual-bf16-x", &pack_u16(x));
    let rb = dispatch::storage_from_slice(ctx, "rmsnorm-residual-bf16-res", &pack_u16(residual));
    let wb = dispatch::storage_from_slice(ctx, "rmsnorm-residual-bf16-weight", &pack_u16(weight));
    let ob = dispatch::storage_zeroed(
        ctx,
        "rmsnorm-residual-bf16-out",
        (batch * words_per_row * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "rmsnorm-residual-bf16-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm_residual_bf16",
        &compose(WGSL),
        "rmsnorm_residual_bf16",
        &[(0, &xb), (1, &rb), (2, &wb), (3, &ob), (4, &pb)],
        groups,
    )?;

    let new_res: Vec<u32> = dispatch::read_back(ctx, &rb, batch * words_per_row)?;
    unpack_u16(&new_res, residual);
    let y: Vec<u32> = dispatch::read_back(ctx, &ob, batch * words_per_row)?;
    unpack_u16(&y, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_word_packing_round_trips() {
        let src: Vec<u16> = (0u16..64).map(|i| i.wrapping_mul(7919)).collect();
        let words = pack_u16(&src);
        assert_eq!(words.len(), src.len() / 2);
        assert_eq!(words[3], src[6] as u32 | ((src[7] as u32) << 16));
        let mut back = vec![0u16; src.len()];
        unpack_u16(&words, &mut back);
        assert_eq!(back, src);
    }

    #[test]
    fn shape_mismatch_is_reported() {
        check_shapes(8, 8, 4, 8, 2, 4).unwrap();
        let e = check_shapes(8, 7, 4, 8, 2, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        let e = check_shapes(8, 8, 3, 8, 2, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        let e = check_shapes(8, 8, 4, 6, 2, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }
}
