use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};

pub const WORKGROUP_SIZE: u32 = 256;

pub const FROZEN_IN_PROJ_AB_STILL_NEED_DA_DB_BECAUSE_THE_DX_CHAIN_THROUGH_THEM_IS_LOAD_BEARING:
    &str = "TrainGdn::head computes beta = sigmoid(b) and g_exp = exp(softplus(a + dt_bias) * \
     -exp(A_log)) with softplus in the stable max(x,0)+log(1+exp(-|x|)) form whose exact \
     derivative is sigmoid(x); in_proj_a/in_proj_b take no weight grads, but dX flows through \
     their frozen matmuls, so this kernel emits da = dg*g_exp*(-exp(A_log))*sigmoid(a+dt_bias) \
     and db = dbeta*beta*(1-beta) for the linear-backward dx chain; every element is one \
     thread with no cross-thread reduction, so only exp/log libcall rounding separates it from \
     the f64 host oracle and gates run at the 1e-5 relative class";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatingBwdParams {
    rows: u32,
    heads: u32,
    pad0: u32,
    pad1: u32,
}

const WGSL: &str = include_str!("../../../wgsl/q38_bwd_gating.wgsl");

pub fn gating_bwd_f32(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    dg: &[f32],
    dbeta: &[f32],
    da: &mut [f32],
    db: &mut [f32],
    rows: usize,
    heads: usize,
) -> Result<()> {
    dispatch::drive_blocking(gating_bwd_f32_arm(
        ctx,
        a,
        b,
        a_log,
        dt_bias,
        dg,
        dbeta,
        da,
        db,
        rows,
        heads,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn gating_bwd_f32_arm(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    dg: &[f32],
    dbeta: &[f32],
    da: &mut [f32],
    db: &mut [f32],
    rows: usize,
    heads: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = FROZEN_IN_PROJ_AB_STILL_NEED_DA_DB_BECAUSE_THE_DX_CHAIN_THROUGH_THEM_IS_LOAD_BEARING;
    let total = rows * heads;
    dispatch::check_len("gating_bwd a", a.len(), total)?;
    dispatch::check_len("gating_bwd b", b.len(), total)?;
    dispatch::check_len("gating_bwd a_log", a_log.len(), heads)?;
    dispatch::check_len("gating_bwd dt_bias", dt_bias.len(), heads)?;
    dispatch::check_len("gating_bwd dg", dg.len(), total)?;
    dispatch::check_len("gating_bwd dbeta", dbeta.len(), total)?;
    dispatch::check_len("gating_bwd da", da.len(), total)?;
    dispatch::check_len("gating_bwd db", db.len(), total)?;
    if total == 0 {
        return Ok(());
    }
    dispatch::require_workgroup(ctx, "q38_gating_bwd", WORKGROUP_SIZE)?;

    let params = GatingBwdParams {
        rows: rows as u32,
        heads: heads as u32,
        pad0: 0,
        pad1: 0,
    };
    let ab = dispatch::storage_from_slice(ctx, "qgt-a", a);
    let bb = dispatch::storage_from_slice(ctx, "qgt-b", b);
    let alb = dispatch::storage_from_slice(ctx, "qgt-alog", a_log);
    let dtb = dispatch::storage_from_slice(ctx, "qgt-dtb", dt_bias);
    let dgb = dispatch::storage_from_slice(ctx, "qgt-dg", dg);
    let dbetab = dispatch::storage_from_slice(ctx, "qgt-dbeta", dbeta);
    let dab = dispatch::storage_zeroed(ctx, "qgt-da", (total * 4) as u64);
    let dbb = dispatch::storage_zeroed(ctx, "qgt-db", (total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qgt-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_q38_gating_bwd",
        &compose(WGSL),
        "q38_gating_bwd",
        &[
            (0, &ab),
            (1, &bb),
            (2, &alb),
            (3, &dtb),
            (4, &dgb),
            (5, &dbetab),
            (6, &dab),
            (7, &dbb),
            (8, &pb),
        ],
        dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE),
    )?;

    let da_out: Vec<f32> = dispatch::read_back_arm(ctx, &dab, total, arm).await?;
    let db_out: Vec<f32> = dispatch::read_back_arm(ctx, &dbb, total, arm).await?;
    da.copy_from_slice(&da_out);
    db.copy_from_slice(&db_out);
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
            assert_eq!(
                WGSL.matches("var<storage").count(),
                8,
                "8 storage declarations, under the measured maxStorageBuffersPerShaderStage=10"
            );
            assert_eq!(WGSL.matches("var<workgroup>").count(), 0);
        }
        {
            assert_eq!(std::mem::size_of::<GatingBwdParams>(), 16);
        }
        {
            assert!(compose(WGSL).contains("fn q38_gating_bwd("));
        }
    }
}
