#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 * 2;

pub const EPS_JOINS_THE_SUM_OF_SQUARES_NOT_A_MEAN_MATCHING_L2_NORMALIZE_LAST: &str =
    "TrainGdn::head l2-normalizes q/k rows as x / sqrt(sum(x^2) + 1e-6) with eps added to the \
     raw sum of squares, no division by the row length, so the backward uses the same \
     d = sqrt(S + eps) and emits dX_i = dY_i/d - x_i*sum_j(dY_j*x_j)/d^3; the two row sums are \
     256-wide binary tree folds against a sequential f64 host oracle, so gates run at the \
     1e-5 relative class, never bitwise";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct L2NormBwdParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    pad0: u32,
}

const WGSL: &str = include_str!("../../../wgsl/q38_bwd_l2norm.wgsl");

pub fn l2norm_bwd_f32(
    ctx: &WgpuContext,
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    dispatch::drive_blocking(l2norm_bwd_f32_arm(
        ctx,
        x,
        dy,
        dx,
        batch,
        hidden,
        eps,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn l2norm_bwd_f32_arm(
    ctx: &WgpuContext,
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = EPS_JOINS_THE_SUM_OF_SQUARES_NOT_A_MEAN_MATCHING_L2_NORMALIZE_LAST;
    dispatch::check_len("l2norm_bwd x", x.len(), batch * hidden)?;
    dispatch::check_len("l2norm_bwd dy", dy.len(), batch * hidden)?;
    dispatch::check_len("l2norm_bwd dx", dx.len(), batch * hidden)?;
    if batch * hidden == 0 {
        return Ok(());
    }
    dispatch::require_workgroup_and_scratch(ctx, "q38_l2norm_bwd", WORKGROUP_SIZE, SCRATCH_BYTES)?;

    let params = L2NormBwdParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        pad0: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "ql2-x", x);
    let dyb = dispatch::storage_from_slice(ctx, "ql2-dy", dy);
    let dxb = dispatch::storage_zeroed(ctx, "ql2-dx", (batch * hidden * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "ql2-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_q38_l2norm_bwd",
        &compose(WGSL),
        "q38_l2norm_bwd",
        &[(0, &xb), (1, &dyb), (2, &dxb), (3, &pb)],
        dispatch::workgroup_count_1d(ctx, batch as u64, 1),
    )?;

    let out: Vec<f32> = dispatch::read_back_arm(ctx, &dxb, batch * hidden, arm).await?;
    dx.copy_from_slice(&out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_module_stays_inside_the_measured_browser_caps() {
        assert!(
            !WGSL.contains("subgroup"),
            "no subgroup ops, no directive split needed"
        );
        assert!(!WGSL.contains("enable "), "no enable directives");
        assert!(!WGSL.contains("f16"), "f32 only");
        assert_eq!(WGSL.matches("@workgroup_size(256)").count(), 1);
        assert_eq!(WGSL.matches("var<storage").count(), 3);
        assert_eq!(WGSL.matches("var<workgroup>").count(), 2);
        assert_eq!(SCRATCH_BYTES, 2048);
        const { assert!(SCRATCH_BYTES <= 32768) };
    }

    #[test]
    fn params_keep_the_sixteen_byte_uniform_layout() {
        assert_eq!(std::mem::size_of::<L2NormBwdParams>(), 16);
    }

    #[test]
    fn source_declares_the_entry() {
        assert!(compose(WGSL).contains("fn q38_l2norm_bwd("));
    }
}
