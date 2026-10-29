use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 * 2;

pub const BWD_TAKES_FOLDED_GAMMA_AND_PRODUCES_ONLY_THE_INPUT_GRAD: &str =
    "the mlx export folds qwen's zero-centred +1 into the stored norm gammas \
     (DOTS_NORMS_LOAD_RAW_BECAUSE_THE_MLX_EXPORT_FOLDS_THE_PLUS_ONE), so gamma arrives here \
     as the effective multiplier and no +1 is added in-kernel; norms are frozen in qlora, so \
     the backward emits dX only: dX_i = g_i*dY_i*r - x_i*(r^3/H)*sum_j(dY_j*g_j*x_j) with \
     r = inverseSqrt(sum(x^2)/H + eps), the same r expression the forward kernels use";

pub const TREE_REDUCTION_ORDER_IS_DETERMINISTIC_BUT_GATES_STILL_RUN_AT_1E5_REL: &str =
    "each thread accumulates its strided slice in index order and a 256-wide binary tree \
     folds the partials, a fixed order every run; the CPU reference sums sequentially \
     instead, and wgpu-hal's Metal path sets no math-mode override, so parity gates run at \
     1e-5 relative to the reference max magnitude rather than exact";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormBwdParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    pad0: u32,
}

const WGSL: &str = include_str!("../../../wgsl/qlora_bwd_norm.wgsl");

pub fn rmsnorm_bwd_f32(
    ctx: &WgpuContext,
    x: &[f32],
    gamma_folded: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    dispatch::drive_blocking(rmsnorm_bwd_f32_arm(
        ctx,
        x,
        gamma_folded,
        dy,
        dx,
        batch,
        hidden,
        eps,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn rmsnorm_bwd_f32_arm(
    ctx: &WgpuContext,
    x: &[f32],
    gamma_folded: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = BWD_TAKES_FOLDED_GAMMA_AND_PRODUCES_ONLY_THE_INPUT_GRAD;
    dispatch::check_len("rmsnorm_bwd x", x.len(), batch * hidden)?;
    dispatch::check_len("rmsnorm_bwd gamma", gamma_folded.len(), hidden)?;
    dispatch::check_len("rmsnorm_bwd dy", dy.len(), batch * hidden)?;
    dispatch::check_len("rmsnorm_bwd dx", dx.len(), batch * hidden)?;
    if batch * hidden == 0 {
        return Ok(());
    }
    dispatch::require_workgroup_and_scratch(
        ctx,
        "qlora_bwd_rmsnorm",
        WORKGROUP_SIZE,
        SCRATCH_BYTES,
    )?;

    let params = NormBwdParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        pad0: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "qbn-x", x);
    let gb = dispatch::storage_from_slice(ctx, "qbn-g", gamma_folded);
    let dyb = dispatch::storage_from_slice(ctx, "qbn-dy", dy);
    let dxb = dispatch::storage_zeroed(ctx, "qbn-dx", (batch * hidden * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qbn-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_qlora_bwd_rmsnorm",
        &compose(WGSL),
        "qlora_bwd_rmsnorm",
        &[(0, &xb), (1, &gb), (2, &dyb), (3, &dxb), (4, &pb)],
        groups,
    )?;

    let out: Vec<f32> = dispatch::read_back_arm(ctx, &dxb, batch * hidden, arm).await?;
    dx.copy_from_slice(&out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_scratch_and_entry_follow_the_shader_contract() {
        {
            assert_eq!(std::mem::size_of::<NormBwdParams>(), 16);
        }
        {
            assert_eq!(
                WGSL.matches("var<workgroup>").count(),
                2,
                "a new var<workgroup> must be summed into SCRATCH_BYTES"
            );
            assert_eq!(SCRATCH_BYTES, 2048);
        }
        {
            assert!(compose(WGSL).contains("fn qlora_bwd_rmsnorm("));
        }
    }
}
