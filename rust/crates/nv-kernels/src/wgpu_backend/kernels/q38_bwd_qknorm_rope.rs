use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 * 2;

pub const Q38_PARTIAL_ROTARY_IS_ITS_OWN_KERNEL_BECAUSE_THE_DOTS_ROPE_TWIN_PINS_DOTS_SEMANTICS:
    &str = "qlora_bwd_rope mirrors dots_rope.rs and stays untouched; TrainAttn rotates only the \
     first rotary_dim of head_dim (64 of 256, halved convention inside that slice: \
     out_lo = n_lo*c - n_hi*s, out_hi = n_lo*s + n_hi*c) and passes dims rotary_dim..head_dim \
     through, all downstream of a frozen per-head qk rmsnorm, so this kernel fuses the \
     un-rotation with the norm backward and emits dX only";

pub const GAMMA_ARRIVES_FOLDED_BECAUSE_TRUNK_GAMMAS_SHIP_CENTRED_ON_ZERO: &str =
    "q_norm/k_norm weights load through trunk_rmsnorm_gamma, which adds the qwen +1 \
     (THE_TRUNK_RMSNORM_GAMMAS_ARE_STORED_AS_GAMMA_MINUS_ONE_BUT_THE_GDN_INNER_NORM_IS_NOT), \
     so gamma arrives here as the effective multiplier; with dn the un-rotated grad, \
     dX_i = g_i*dn_i*r - x_i*(r^3/H)*sum_j(dn_j*g_j*x_j), r = inverseSqrt(mean(x^2)+eps); \
     the two row sums are 256-wide tree folds against a sequential f64 host oracle, so \
     gates run at the 1e-5 relative class, never bitwise";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QkNormRopeBwdParams {
    tokens: u32,
    heads: u32,
    hidden: u32,
    half: u32,
    max_seq: u32,
    eps: f32,
    pad0: u32,
    pad1: u32,
}

const WGSL: &str = include_str!("../../../wgsl/q38_bwd_qknorm_rope.wgsl");

pub fn qknorm_rope_bwd_f32(
    ctx: &WgpuContext,
    x: &[f32],
    gamma_folded: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    tokens: usize,
    heads: usize,
    hidden: usize,
    rotary_dim: usize,
    max_seq: usize,
    eps: f32,
) -> Result<()> {
    dispatch::drive_blocking(qknorm_rope_bwd_f32_arm(
        ctx,
        x,
        gamma_folded,
        positions,
        cos_tbl,
        sin_tbl,
        dy,
        dx,
        tokens,
        heads,
        hidden,
        rotary_dim,
        max_seq,
        eps,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn qknorm_rope_bwd_f32_arm(
    ctx: &WgpuContext,
    x: &[f32],
    gamma_folded: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    tokens: usize,
    heads: usize,
    hidden: usize,
    rotary_dim: usize,
    max_seq: usize,
    eps: f32,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = Q38_PARTIAL_ROTARY_IS_ITS_OWN_KERNEL_BECAUSE_THE_DOTS_ROPE_TWIN_PINS_DOTS_SEMANTICS;
    let _ = GAMMA_ARRIVES_FOLDED_BECAUSE_TRUNK_GAMMAS_SHIP_CENTRED_ON_ZERO;
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > hidden || max_seq == 0 {
        return Err(WgpuError::Shape(format!(
            "qknorm_rope_bwd needs even rotary_dim in 2..=hidden and max_seq nonzero; got \
             rotary_dim={rotary_dim} hidden={hidden} max_seq={max_seq}"
        )));
    }
    let half = rotary_dim / 2;
    let total = tokens * heads * hidden;
    dispatch::check_len("qknorm_rope_bwd x", x.len(), total)?;
    dispatch::check_len("qknorm_rope_bwd gamma", gamma_folded.len(), hidden)?;
    dispatch::check_len("qknorm_rope_bwd positions", positions.len(), tokens)?;
    dispatch::check_len("qknorm_rope_bwd cos", cos_tbl.len(), max_seq * half)?;
    dispatch::check_len("qknorm_rope_bwd sin", sin_tbl.len(), max_seq * half)?;
    dispatch::check_len("qknorm_rope_bwd dy", dy.len(), total)?;
    dispatch::check_len("qknorm_rope_bwd dx", dx.len(), total)?;
    if total == 0 {
        return Ok(());
    }
    dispatch::require_workgroup_and_scratch(
        ctx,
        "q38_qknorm_rope_bwd",
        WORKGROUP_SIZE,
        SCRATCH_BYTES,
    )?;

    let params = QkNormRopeBwdParams {
        tokens: tokens as u32,
        heads: heads as u32,
        hidden: hidden as u32,
        half: half as u32,
        max_seq: max_seq as u32,
        eps,
        pad0: 0,
        pad1: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "qqr-x", x);
    let gb = dispatch::storage_from_slice(ctx, "qqr-g", gamma_folded);
    let posb = dispatch::storage_from_slice(ctx, "qqr-pos", positions);
    let cb = dispatch::storage_from_slice(ctx, "qqr-cos", cos_tbl);
    let sb = dispatch::storage_from_slice(ctx, "qqr-sin", sin_tbl);
    let dyb = dispatch::storage_from_slice(ctx, "qqr-dy", dy);
    let dxb = dispatch::storage_zeroed(ctx, "qqr-dx", (total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qqr-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_q38_qknorm_rope_bwd",
        &compose(WGSL),
        "q38_qknorm_rope_bwd",
        &[
            (0, &xb),
            (1, &gb),
            (2, &posb),
            (3, &cb),
            (4, &sb),
            (5, &dyb),
            (6, &dxb),
            (7, &pb),
        ],
        dispatch::workgroup_count_1d(ctx, (tokens * heads) as u64, 1),
    )?;

    let out: Vec<f32> = dispatch::read_back_arm(ctx, &dxb, total, arm).await?;
    dx.copy_from_slice(&out);
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
                7,
                "7 storage declarations, under the measured maxStorageBuffersPerShaderStage=10"
            );
            assert_eq!(WGSL.matches("var<workgroup>").count(), 2);
            assert_eq!(SCRATCH_BYTES, 2048);
            const { assert!(SCRATCH_BYTES <= 32768) };
        }
        {
            assert_eq!(std::mem::size_of::<QkNormRopeBwdParams>(), 32);
        }
        {
            assert!(compose(WGSL).contains("fn q38_qknorm_rope_bwd("));
        }
    }

    #[test]
    fn odd_or_oversized_rotary_dim_is_rejected() {
        let ctx = crate::wgpu_backend::device::WgpuContext::shared();
        let Ok(ctx) = ctx else {
            return;
        };
        let mut dx = vec![0f32; 8];
        for (rot, ms) in [(3usize, 4usize), (0, 4), (10, 4), (4, 0)] {
            let e = qknorm_rope_bwd_f32(
                ctx,
                &[0f32; 8],
                &[0f32; 8],
                &[0u32],
                &[],
                &[],
                &[0f32; 8],
                &mut dx,
                1,
                1,
                8,
                rot,
                ms,
                1e-6,
            )
            .unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        }
    }
}
