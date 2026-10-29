use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};

pub const WORKGROUP_SIZE: u32 = 256;

pub const THE_GATE_LOGIT_RIDES_THE_Q_PROJ_SO_DGATE_FEEDS_THE_SAME_LINEAR_BACKWARD_AS_DQ: &str =
    "TrainAttn with attn_output_gate splits q_proj rows into value|gate halves and applies \
     gated = attn_out * sigmoid(gate_logit) elementwise after sdpa; the backward emits \
     dattn = dY*sig and dgate = dY*attn_out*sig*(1-sig), one thread per element with no \
     cross-thread reduction, so only the exp libcall separates it from the f64 host oracle \
     and gates run at the 1e-5 relative class";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnGateBwdParams {
    total: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const WGSL: &str = include_str!("../../../wgsl/q38_bwd_attn_gate.wgsl");

pub fn attn_gate_bwd_f32(
    ctx: &WgpuContext,
    attn: &[f32],
    gate_logit: &[f32],
    dy: &[f32],
    dattn: &mut [f32],
    dgate: &mut [f32],
) -> Result<()> {
    dispatch::drive_blocking(attn_gate_bwd_f32_arm(
        ctx,
        attn,
        gate_logit,
        dy,
        dattn,
        dgate,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn attn_gate_bwd_f32_arm(
    ctx: &WgpuContext,
    attn: &[f32],
    gate_logit: &[f32],
    dy: &[f32],
    dattn: &mut [f32],
    dgate: &mut [f32],
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = THE_GATE_LOGIT_RIDES_THE_Q_PROJ_SO_DGATE_FEEDS_THE_SAME_LINEAR_BACKWARD_AS_DQ;
    let total = attn.len();
    dispatch::check_len("attn_gate_bwd gate_logit", gate_logit.len(), total)?;
    dispatch::check_len("attn_gate_bwd dy", dy.len(), total)?;
    dispatch::check_len("attn_gate_bwd dattn", dattn.len(), total)?;
    dispatch::check_len("attn_gate_bwd dgate", dgate.len(), total)?;
    if total == 0 {
        return Ok(());
    }
    dispatch::require_workgroup(ctx, "q38_attn_gate_bwd", WORKGROUP_SIZE)?;

    let params = AttnGateBwdParams {
        total: total as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let ab = dispatch::storage_from_slice(ctx, "qag-attn", attn);
    let gb = dispatch::storage_from_slice(ctx, "qag-gate", gate_logit);
    let dyb = dispatch::storage_from_slice(ctx, "qag-dy", dy);
    let dab = dispatch::storage_zeroed(ctx, "qag-dattn", (total * 4) as u64);
    let dgb = dispatch::storage_zeroed(ctx, "qag-dgate", (total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qag-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_q38_attn_gate_bwd",
        &compose(WGSL),
        "q38_attn_gate_bwd",
        &[
            (0, &ab),
            (1, &gb),
            (2, &dyb),
            (3, &dab),
            (4, &dgb),
            (5, &pb),
        ],
        dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE),
    )?;

    let da_out: Vec<f32> = dispatch::read_back_arm(ctx, &dab, total, arm).await?;
    let dg_out: Vec<f32> = dispatch::read_back_arm(ctx, &dgb, total, arm).await?;
    dattn.copy_from_slice(&da_out);
    dgate.copy_from_slice(&dg_out);
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
            assert_eq!(WGSL.matches("var<storage").count(), 5);
            assert_eq!(WGSL.matches("var<workgroup>").count(), 0);
        }
        {
            assert_eq!(std::mem::size_of::<AttnGateBwdParams>(), 16);
        }
        {
            assert!(compose(WGSL).contains("fn q38_attn_gate_bwd("));
        }
    }
}
