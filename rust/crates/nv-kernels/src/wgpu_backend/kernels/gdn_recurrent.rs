use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/gdn_recurrent.wgsl");

pub const HEAD_DIM: usize = 128;
pub const WORKGROUP_SIZE: u32 = 128;

pub const W32_ENTRY: &str = "gdn_recurrent_f32_w32";
pub const W32_LANES_ONE_WARP_STAGES_KQ_NO_SUBGROUP_OPS_SO_ANY_WIDTH_IS_CORRECT: u32 = 32;
pub const W32_COL_GROUPS_WG_Z_SPANS_THE_128_COLUMNS: u32 =
    HEAD_DIM as u32 / W32_LANES_ONE_WARP_STAGES_KQ_NO_SUBGROUP_OPS_SO_ANY_WIDTH_IS_CORRECT;
pub const SCAN_W32_ENV_OPT_IN_THE_BARRIER_KERNEL_STAYS_THE_DEFAULT: &str = "NV_Q3D_GDN_SCAN_W32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanArm {
    Barrier128,
    Warp32Smem,
}

pub fn scan_w32_env_opted_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        SCAN_W32_ENV_OPT_IN_THE_BARRIER_KERNEL_STAYS_THE_DEFAULT,
    )
}

pub fn scan_arm() -> ScanArm {
    if scan_w32_env_opted_in() {
        ScanArm::Warp32Smem
    } else {
        ScanArm::Barrier128
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GdnRecParams {
    batch: u32,
    seq: u32,
    heads: u32,
    pairs: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "gdn_recurrent", WORKGROUP_SIZE)
}

fn check_shapes(
    q: usize,
    k: usize,
    v: usize,
    g_exp: usize,
    beta: usize,
    out: usize,
    state: usize,
    b: usize,
    h: usize,
    t: usize,
) -> Result<()> {
    let rows = b * t * h;
    let vecs = rows * HEAD_DIM;
    dispatch::check_len("gdn_recurrent q", q, vecs)?;
    dispatch::check_len("gdn_recurrent k", k, vecs)?;
    dispatch::check_len("gdn_recurrent v", v, vecs)?;
    dispatch::check_len("gdn_recurrent g_exp", g_exp, rows)?;
    dispatch::check_len("gdn_recurrent beta", beta, rows)?;
    dispatch::check_len("gdn_recurrent out", out, vecs)?;
    dispatch::check_len("gdn_recurrent state", state, b * h * HEAD_DIM * HEAD_DIM)?;
    Ok(())
}

pub fn gdn_recurrent_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    out: &mut [f32],
    state: &mut [f32],
    b: usize,
    h: usize,
    t: usize,
) -> Result<()> {
    gdn_recurrent_f32_with_arm(ctx, q, k, v, g_exp, beta, out, state, b, h, t, scan_arm())
}

pub fn gdn_recurrent_f32_with_arm(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    out: &mut [f32],
    state: &mut [f32],
    b: usize,
    h: usize,
    t: usize,
    arm: ScanArm,
) -> Result<()> {
    check_shapes(
        q.len(),
        k.len(),
        v.len(),
        g_exp.len(),
        beta.len(),
        out.len(),
        state.len(),
        b,
        h,
        t,
    )?;
    if b * t * h == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let pairs = b * h;
    let vecs = b * t * h * HEAD_DIM;
    let state_len = pairs * HEAD_DIM * HEAD_DIM;
    let state_bytes = (state_len * 4) as u64;
    if state_bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "gdn_recurrent state needs {state_bytes} bytes; device allows {}",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }

    let params = GdnRecParams {
        batch: b as u32,
        seq: t as u32,
        heads: h as u32,
        pairs: pairs as u32,
    };
    let qb = dispatch::storage_from_slice(ctx, "gdn-recurrent-q", q);
    let kb = dispatch::storage_from_slice(ctx, "gdn-recurrent-k", k);
    let vb = dispatch::storage_from_slice(ctx, "gdn-recurrent-v", v);
    let gb = dispatch::storage_from_slice(ctx, "gdn-recurrent-g", g_exp);
    let bb = dispatch::storage_from_slice(ctx, "gdn-recurrent-beta", beta);
    let ob = dispatch::storage_zeroed(ctx, "gdn-recurrent-out", (vecs * 4) as u64);
    let sb = dispatch::storage_zeroed(ctx, "gdn-recurrent-state", state_bytes);
    let pb = dispatch::uniform_from(ctx, "gdn-recurrent-params", &params);

    let pair_groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    let (label, entry, groups) = match arm {
        ScanArm::Barrier128 => (
            "nv_kernels_gdn_recurrent_f32",
            "gdn_recurrent_f32",
            pair_groups,
        ),
        ScanArm::Warp32Smem => (
            "nv_kernels_gdn_recurrent_f32_w32",
            W32_ENTRY,
            (
                pair_groups.0,
                pair_groups.1,
                W32_COL_GROUPS_WG_Z_SPANS_THE_128_COLUMNS,
            ),
        ),
    };
    dispatch::run(
        ctx,
        label,
        &compose(WGSL),
        entry,
        &[
            (0, &qb),
            (1, &kb),
            (2, &vb),
            (3, &gb),
            (4, &bb),
            (5, &ob),
            (6, &sb),
            (7, &pb),
        ],
        groups,
    )?;

    let y: Vec<f32> = dispatch::read_back(ctx, &ob, vecs)?;
    out.copy_from_slice(&y);
    let s: Vec<f32> = dispatch::read_back(ctx, &sb, state_len)?;
    state.copy_from_slice(&s);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_checks_and_the_four_wg32_column_groups_follow_the_shader_contract() {
        {
            let vecs = 2 * 3 * 4 * HEAD_DIM;
            let rows = 2 * 3 * 4;
            let st = 2 * 4 * HEAD_DIM * HEAD_DIM;
            check_shapes(vecs, vecs, vecs, rows, rows, vecs, st, 2, 4, 3).unwrap();
            let e = check_shapes(vecs - 1, vecs, vecs, rows, rows, vecs, st, 2, 4, 3).unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
            let e = check_shapes(vecs, vecs, vecs, rows - 1, rows, vecs, st, 2, 4, 3).unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
            let e = check_shapes(vecs, vecs, vecs, rows, rows, vecs, st - 4, 2, 4, 3).unwrap_err();
            assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        }
        {
            assert_eq!(
                W32_COL_GROUPS_WG_Z_SPANS_THE_128_COLUMNS
                    * W32_LANES_ONE_WARP_STAGES_KQ_NO_SUBGROUP_OPS_SO_ANY_WIDTH_IS_CORRECT,
                HEAD_DIM as u32
            );
            assert!(WGSL.contains(W32_ENTRY));
            assert!(WGSL.contains("@workgroup_size(32)"));
            assert!(!WGSL.contains("subgroup"));
            assert!(!WGSL.contains("enable subgroups"));
        }
    }
}
