use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WORKGROUP_SIZE: u32 = 256;

pub const PAIR_MATCHES_DOTS_ROPE_FUSED_SEMANTICS_INCLUDING_THE_POSITION_CLAMP: &str =
    "forward and backward mirror nv-train/src/dots_rope.rs: x is [tokens, heads, 2*half] in \
     the halved convention, positions are u32 per token clamped to max_seq-1, and the \
     nominal rounding sequence is fl(fl(lo*c)-fl(hi*s)) / fl(fl(lo*s)+fl(hi*c)) forward and \
     fl(fl(gh*s)+fl(gl*c)) / fl(fl(gh*c)-fl(gl*s)) backward; the trainer's MSL twin compiles \
     under MTLMathMode::Safe, but wgpu-hal's Metal path sets no math-mode override, so \
     contraction into fma may perturb the last ulp here and parity gates run at 1e-5 \
     relative rather than bitwise";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeBwdParams {
    tokens: u32,
    heads: u32,
    half_dim: u32,
    max_seq: u32,
}

const WGSL: &str = include_str!("../../../wgsl/qlora_bwd_rope.wgsl");

fn plan(
    x: usize,
    out: usize,
    positions: usize,
    cos_tbl: usize,
    sin_tbl: usize,
    tokens: usize,
    heads: usize,
    half_dim: usize,
    max_seq: usize,
) -> Result<RopeBwdParams> {
    if half_dim == 0 || max_seq == 0 {
        return Err(WgpuError::Shape(format!(
            "rope pair needs half_dim and max_seq nonzero; got half_dim={half_dim} max_seq={max_seq}"
        )));
    }
    dispatch::check_len("rope pair x", x, tokens * heads * 2 * half_dim)?;
    dispatch::check_len("rope pair out", out, tokens * heads * 2 * half_dim)?;
    dispatch::check_len("rope pair positions", positions, tokens)?;
    dispatch::check_len("rope pair cos", cos_tbl, max_seq * half_dim)?;
    dispatch::check_len("rope pair sin", sin_tbl, max_seq * half_dim)?;
    Ok(RopeBwdParams {
        tokens: tokens as u32,
        heads: heads as u32,
        half_dim: half_dim as u32,
        max_seq: max_seq as u32,
    })
}

async fn run_pair(
    ctx: &WgpuContext,
    entry: &'static str,
    label: &'static str,
    x: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    out: &mut [f32],
    tokens: usize,
    heads: usize,
    half_dim: usize,
    max_seq: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = PAIR_MATCHES_DOTS_ROPE_FUSED_SEMANTICS_INCLUDING_THE_POSITION_CLAMP;
    let params = plan(
        x.len(),
        out.len(),
        positions.len(),
        cos_tbl.len(),
        sin_tbl.len(),
        tokens,
        heads,
        half_dim,
        max_seq,
    )?;
    if x.is_empty() {
        return Ok(());
    }
    dispatch::require_workgroup(ctx, "qlora_rope", WORKGROUP_SIZE)?;

    let xb = dispatch::storage_from_slice(ctx, "qbr-x", x);
    let posb = dispatch::storage_from_slice(ctx, "qbr-pos", positions);
    let cb = dispatch::storage_from_slice(ctx, "qbr-cos", cos_tbl);
    let sb = dispatch::storage_from_slice(ctx, "qbr-sin", sin_tbl);
    let ob = dispatch::storage_zeroed(ctx, "qbr-out", (out.len() * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qbr-params", &params);

    let total = tokens * heads * half_dim;
    let groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        label,
        &compose(WGSL),
        entry,
        &[(0, &xb), (1, &posb), (2, &cb), (3, &sb), (4, &ob), (5, &pb)],
        groups,
    )?;

    let got: Vec<f32> = dispatch::read_back_arm(ctx, &ob, out.len(), arm).await?;
    out.copy_from_slice(&got);
    Ok(())
}

pub fn rope_fwd_f32(
    ctx: &WgpuContext,
    x: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    out: &mut [f32],
    tokens: usize,
    heads: usize,
    half_dim: usize,
    max_seq: usize,
) -> Result<()> {
    dispatch::drive_blocking(rope_fwd_f32_arm(
        ctx,
        x,
        positions,
        cos_tbl,
        sin_tbl,
        out,
        tokens,
        heads,
        half_dim,
        max_seq,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn rope_fwd_f32_arm(
    ctx: &WgpuContext,
    x: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    out: &mut [f32],
    tokens: usize,
    heads: usize,
    half_dim: usize,
    max_seq: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    run_pair(
        ctx,
        "qlora_rope_fwd",
        "nv_kernels_qlora_rope_fwd",
        x,
        positions,
        cos_tbl,
        sin_tbl,
        out,
        tokens,
        heads,
        half_dim,
        max_seq,
        arm,
    )
    .await
}

pub fn rope_bwd_f32(
    ctx: &WgpuContext,
    grad: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    dx: &mut [f32],
    tokens: usize,
    heads: usize,
    half_dim: usize,
    max_seq: usize,
) -> Result<()> {
    dispatch::drive_blocking(rope_bwd_f32_arm(
        ctx,
        grad,
        positions,
        cos_tbl,
        sin_tbl,
        dx,
        tokens,
        heads,
        half_dim,
        max_seq,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn rope_bwd_f32_arm(
    ctx: &WgpuContext,
    grad: &[f32],
    positions: &[u32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    dx: &mut [f32],
    tokens: usize,
    heads: usize,
    half_dim: usize,
    max_seq: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    run_pair(
        ctx,
        "qlora_rope_bwd",
        "nv_kernels_qlora_rope_bwd",
        grad,
        positions,
        cos_tbl,
        sin_tbl,
        dx,
        tokens,
        heads,
        half_dim,
        max_seq,
        arm,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_and_both_directions_follow_the_shader_contract() {
        {
            assert_eq!(std::mem::size_of::<RopeBwdParams>(), 16);
        }
        {
            let src = compose(WGSL);
            assert!(src.contains("fn qlora_rope_fwd("));
            assert!(src.contains("fn qlora_rope_bwd("));
        }
    }

    #[test]
    fn zero_half_dim_and_shape_mismatch_are_rejected() {
        {
            let e = plan(0, 0, 0, 0, 0, 0, 0, 0, 4).unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
            let e = plan(0, 0, 0, 0, 0, 0, 0, 4, 0).unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        }
        {
            let e = plan(10, 10, 1, 8, 8, 1, 1, 4, 2).unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        }
    }
}
