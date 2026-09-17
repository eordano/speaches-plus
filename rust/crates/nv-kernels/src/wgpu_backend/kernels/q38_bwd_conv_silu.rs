#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4;

pub const BWD_RECOMPUTES_THE_PREACTIVATION_WITH_THE_FORWARD_ASCENDING_TAP_ORDER: &str =
    "TrainGdn::causal_depthwise_conv_silu pads k-1 zeros before t=0 and sums taps j=0..k-1 in \
     ascending order, so p[t,c] = sum_j w[c,j]*x[t+j-(k-1),c] with negative rows zero; both \
     entries recompute p with that same ascending-j add chain and apply \
     dsilu(p) = sig(p)*(1+p*(1-sig(p))); dx sums the k da taps thread-sequentially and dw \
     tree-folds a 256-wide strided pass over b*t in index order, so gates run at the 1e-5 \
     relative class against a sequential f64 host oracle, never bitwise";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvSiluBwdParams {
    batch: u32,
    seq: u32,
    channels: u32,
    ktaps: u32,
}

const WGSL: &str = include_str!("../../../wgsl/q38_bwd_conv_silu.wgsl");

pub fn conv_silu_bwd_f32(
    ctx: &WgpuContext,
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    batch: usize,
    seq: usize,
    channels: usize,
    ktaps: usize,
) -> Result<()> {
    dispatch::drive_blocking(conv_silu_bwd_f32_arm(
        ctx,
        x,
        w,
        dy,
        dx,
        dw,
        batch,
        seq,
        channels,
        ktaps,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn conv_silu_bwd_f32_arm(
    ctx: &WgpuContext,
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    batch: usize,
    seq: usize,
    channels: usize,
    ktaps: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = BWD_RECOMPUTES_THE_PREACTIVATION_WITH_THE_FORWARD_ASCENDING_TAP_ORDER;
    if ktaps == 0 {
        return Err(WgpuError::Shape("conv tap count must be at least 1".into()));
    }
    let elems = batch * seq * channels;
    dispatch::check_len("conv_silu_bwd x", x.len(), elems)?;
    dispatch::check_len("conv_silu_bwd w", w.len(), channels * ktaps)?;
    dispatch::check_len("conv_silu_bwd dy", dy.len(), elems)?;
    dispatch::check_len("conv_silu_bwd dx", dx.len(), elems)?;
    dispatch::check_len("conv_silu_bwd dw", dw.len(), channels * ktaps)?;
    if elems == 0 {
        return Ok(());
    }
    dispatch::require_workgroup_and_scratch(
        ctx,
        "q38_conv_silu_bwd",
        WORKGROUP_SIZE,
        SCRATCH_BYTES,
    )?;

    let params = ConvSiluBwdParams {
        batch: batch as u32,
        seq: seq as u32,
        channels: channels as u32,
        ktaps: ktaps as u32,
    };
    let xb = dispatch::storage_from_slice(ctx, "qcs-x", x);
    let wb = dispatch::storage_from_slice(ctx, "qcs-w", w);
    let dyb = dispatch::storage_from_slice(ctx, "qcs-dy", dy);
    let dxb = dispatch::storage_zeroed(ctx, "qcs-dx", (elems * 4) as u64);
    let dwb = dispatch::storage_zeroed(ctx, "qcs-dw", (channels * ktaps * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qcs-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_q38_conv_silu_bwd_dx",
        &compose(WGSL),
        "q38_conv_silu_bwd_dx",
        &[(0, &xb), (1, &wb), (2, &dyb), (3, &dxb), (5, &pb)],
        dispatch::workgroup_count_1d(ctx, elems as u64, WORKGROUP_SIZE),
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_q38_conv_silu_bwd_dw",
        &compose(WGSL),
        "q38_conv_silu_bwd_dw",
        &[(0, &xb), (1, &wb), (2, &dyb), (4, &dwb), (5, &pb)],
        dispatch::workgroup_count_1d(ctx, (channels * ktaps) as u64, 1),
    )?;

    let dx_out: Vec<f32> = dispatch::read_back_arm(ctx, &dxb, elems, arm).await?;
    let dw_out: Vec<f32> = dispatch::read_back_arm(ctx, &dwb, channels * ktaps, arm).await?;
    dx.copy_from_slice(&dx_out);
    dw.copy_from_slice(&dw_out);
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
        assert_eq!(WGSL.matches("@workgroup_size(256)").count(), 2);
        assert_eq!(WGSL.matches("var<storage").count(), 5);
        assert_eq!(WGSL.matches("var<workgroup>").count(), 1);
        assert_eq!(SCRATCH_BYTES, 1024);
        const { assert!(SCRATCH_BYTES <= 32768) };
    }

    #[test]
    fn params_keep_the_sixteen_byte_uniform_layout() {
        assert_eq!(std::mem::size_of::<ConvSiluBwdParams>(), 16);
    }

    #[test]
    fn both_entries_are_declared() {
        let s = compose(WGSL);
        assert!(s.contains("fn q38_conv_silu_bwd_dx("));
        assert!(s.contains("fn q38_conv_silu_bwd_dw("));
    }
}
