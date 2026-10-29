use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 * 2;

pub const GAMMA_ARRIVES_RAW_BECAUSE_THE_GDN_INNER_NORM_SHIPS_UNCENTRED_AND_IS_FROZEN: &str =
    "TrainGdn::tail is rmsnorm_last(core) * silu(z): r = inverseSqrt(mean(x^2) + eps), \
     normed_i = g_i*x_i*r with linear_attn.norm.weight loaded raw (it ships centred on its \
     true value, unlike the trunk gammas, per \
     THE_TRUNK_RMSNORM_GAMMAS_ARE_STORED_AS_GAMMA_MINUS_ONE_BUT_THE_GDN_INNER_NORM_IS_NOT), \
     and the norm is frozen so the backward emits dX and dZ only: with e_i = dY_i*silu(z_i), \
     dX_i = g_i*e_i*r - x_i*(r^3/H)*sum_j(e_j*g_j*x_j) and \
     dZ_i = dY_i*g_i*x_i*r*sig(z_i)*(1+z_i*(1-sig(z_i)))";

pub const TREE_REDUCTION_ORDER_IS_DETERMINISTIC_BUT_GATES_STILL_RUN_AT_1E5_REL: &str =
    "each thread accumulates its strided slice in index order and a 256-wide binary tree \
     folds the partials, a fixed order every run; the f64 host oracle sums sequentially and \
     wgpu-hal's Metal path sets no math-mode override, so parity gates run at 1e-5 relative \
     to the oracle max magnitude rather than exact";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatedNormBwdParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    pad0: u32,
}

const WGSL: &str = include_str!("../../../wgsl/q38_bwd_gated_norm.wgsl");

pub fn gated_norm_bwd_f32(
    ctx: &WgpuContext,
    x: &[f32],
    z: &[f32],
    gamma_raw: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dz: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    dispatch::drive_blocking(gated_norm_bwd_f32_arm(
        ctx,
        x,
        z,
        gamma_raw,
        dy,
        dx,
        dz,
        batch,
        hidden,
        eps,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn gated_norm_bwd_f32_arm(
    ctx: &WgpuContext,
    x: &[f32],
    z: &[f32],
    gamma_raw: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dz: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = GAMMA_ARRIVES_RAW_BECAUSE_THE_GDN_INNER_NORM_SHIPS_UNCENTRED_AND_IS_FROZEN;
    let _ = TREE_REDUCTION_ORDER_IS_DETERMINISTIC_BUT_GATES_STILL_RUN_AT_1E5_REL;
    let total = batch * hidden;
    dispatch::check_len("gated_norm_bwd x", x.len(), total)?;
    dispatch::check_len("gated_norm_bwd z", z.len(), total)?;
    dispatch::check_len("gated_norm_bwd gamma", gamma_raw.len(), hidden)?;
    dispatch::check_len("gated_norm_bwd dy", dy.len(), total)?;
    dispatch::check_len("gated_norm_bwd dx", dx.len(), total)?;
    dispatch::check_len("gated_norm_bwd dz", dz.len(), total)?;
    if total == 0 {
        return Ok(());
    }
    dispatch::require_workgroup_and_scratch(
        ctx,
        "q38_gated_norm_bwd",
        WORKGROUP_SIZE,
        SCRATCH_BYTES,
    )?;

    let params = GatedNormBwdParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        pad0: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "qgn-x", x);
    let zb = dispatch::storage_from_slice(ctx, "qgn-z", z);
    let gb = dispatch::storage_from_slice(ctx, "qgn-g", gamma_raw);
    let dyb = dispatch::storage_from_slice(ctx, "qgn-dy", dy);
    let dxb = dispatch::storage_zeroed(ctx, "qgn-dx", (total * 4) as u64);
    let dzb = dispatch::storage_zeroed(ctx, "qgn-dz", (total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qgn-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_q38_gated_norm_bwd",
        &compose(WGSL),
        "q38_gated_norm_bwd",
        &[
            (0, &xb),
            (1, &zb),
            (2, &gb),
            (3, &dyb),
            (4, &dxb),
            (5, &dzb),
            (6, &pb),
        ],
        dispatch::workgroup_count_1d(ctx, batch as u64, 1),
    )?;

    let dx_out: Vec<f32> = dispatch::read_back_arm(ctx, &dxb, total, arm).await?;
    let dz_out: Vec<f32> = dispatch::read_back_arm(ctx, &dzb, total, arm).await?;
    dx.copy_from_slice(&dx_out);
    dz.copy_from_slice(&dz_out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_module_stays_inside_the_measured_browser_caps_with_its_entry_and_uniform_layout() {
        {
            assert!(
                !WGSL.contains("subgroup"),
                "no subgroup ops, no directive split needed"
            );
            assert!(!WGSL.contains("enable "), "no enable directives");
            assert!(!WGSL.contains("f16"), "f32 only");
            assert_eq!(WGSL.matches("@workgroup_size(256)").count(), 1);
            assert_eq!(WGSL.matches("var<storage").count(), 6);
            assert_eq!(WGSL.matches("var<workgroup>").count(), 2);
            assert_eq!(SCRATCH_BYTES, 2048);
            const { assert!(SCRATCH_BYTES <= 32768) };
        }
        {
            assert_eq!(std::mem::size_of::<GatedNormBwdParams>(), 16);
        }
        {
            assert!(compose(WGSL).contains("fn q38_gated_norm_bwd("));
        }
    }
}
