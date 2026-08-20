#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};
#[cfg(test)]
use crate::wgpu_backend::WgpuError;

pub const WGSL: &str = include_str!("../../../wgsl/gdn_gating.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GdnGatingParams {
    total: u32,
    num_heads: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "gdn_gating", WORKGROUP_SIZE)
}

fn check_shapes(
    a: usize,
    b: usize,
    a_log: usize,
    dt_bias: usize,
    g_out: usize,
    beta_out: usize,
    tokens: usize,
    num_heads: usize,
) -> Result<usize> {
    let total = tokens * num_heads;
    dispatch::check_len("gdn_gating a", a, total)?;
    dispatch::check_len("gdn_gating b", b, total)?;
    dispatch::check_len("gdn_gating A_log", a_log, num_heads)?;
    dispatch::check_len("gdn_gating dt_bias", dt_bias, num_heads)?;
    dispatch::check_len("gdn_gating g_out", g_out, total)?;
    dispatch::check_len("gdn_gating beta_out", beta_out, total)?;
    Ok(total)
}

fn widen(src: &[u16]) -> Vec<u32> {
    src.iter().map(|w| *w as u32).collect()
}

pub fn gdn_gating_f32(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    g_out: &mut [f32],
    beta_out: &mut [f32],
    tokens: usize,
    num_heads: usize,
) -> Result<()> {
    let total = check_shapes(
        a.len(),
        b.len(),
        a_log.len(),
        dt_bias.len(),
        g_out.len(),
        beta_out.len(),
        tokens,
        num_heads,
    )?;
    if total == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let params = GdnGatingParams {
        total: total as u32,
        num_heads: num_heads as u32,
    };
    let ab = dispatch::storage_from_slice(ctx, "gdn-gating-f32-a", a);
    let bb = dispatch::storage_from_slice(ctx, "gdn-gating-f32-b", b);
    let lb = dispatch::storage_from_slice(ctx, "gdn-gating-f32-alog", a_log);
    let db = dispatch::storage_from_slice(ctx, "gdn-gating-f32-bias", dt_bias);
    let gb = dispatch::storage_zeroed(ctx, "gdn-gating-f32-g", (total * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "gdn-gating-f32-beta", (total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "gdn-gating-f32-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_gdn_gating_f32",
        &compose(WGSL),
        "gdn_gating_f32",
        &[
            (0, &ab),
            (1, &bb),
            (2, &lb),
            (3, &db),
            (4, &gb),
            (5, &ob),
            (6, &pb),
        ],
        groups,
    )?;

    let g: Vec<f32> = dispatch::read_back(ctx, &gb, total)?;
    g_out.copy_from_slice(&g);
    let beta: Vec<f32> = dispatch::read_back(ctx, &ob, total)?;
    beta_out.copy_from_slice(&beta);
    Ok(())
}

pub fn gdn_gating_bf16(
    ctx: &WgpuContext,
    a: &[u16],
    b: &[u16],
    a_log: &[u16],
    dt_bias: &[u16],
    g_out: &mut [f32],
    beta_out: &mut [u16],
    tokens: usize,
    num_heads: usize,
) -> Result<()> {
    let total = check_shapes(
        a.len(),
        b.len(),
        a_log.len(),
        dt_bias.len(),
        g_out.len(),
        beta_out.len(),
        tokens,
        num_heads,
    )?;
    if total == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let params = GdnGatingParams {
        total: total as u32,
        num_heads: num_heads as u32,
    };
    let ab = dispatch::storage_from_slice(ctx, "gdn-gating-bf16-a", &widen(a));
    let bb = dispatch::storage_from_slice(ctx, "gdn-gating-bf16-b", &widen(b));
    let lb = dispatch::storage_from_slice(ctx, "gdn-gating-bf16-alog", &widen(a_log));
    let db = dispatch::storage_from_slice(ctx, "gdn-gating-bf16-bias", &widen(dt_bias));
    let gb = dispatch::storage_zeroed(ctx, "gdn-gating-bf16-g", (total * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "gdn-gating-bf16-beta", (total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "gdn-gating-bf16-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_gdn_gating_bf16",
        &compose(WGSL),
        "gdn_gating_bf16",
        &[
            (0, &ab),
            (1, &bb),
            (2, &lb),
            (3, &db),
            (4, &gb),
            (5, &ob),
            (6, &pb),
        ],
        groups,
    )?;

    let g: Vec<f32> = dispatch::read_back(ctx, &gb, total)?;
    g_out.copy_from_slice(&g);
    let beta: Vec<u32> = dispatch::read_back(ctx, &ob, total)?;
    for (dst, src) in beta_out.iter_mut().zip(beta.iter()) {
        *dst = (*src & 0xffff) as u16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_is_reported() {
        assert_eq!(check_shapes(16, 16, 4, 4, 16, 16, 4, 4).unwrap(), 16);
        let e = check_shapes(15, 16, 4, 4, 16, 16, 4, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        let e = check_shapes(16, 16, 3, 4, 16, 16, 4, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
        let e = check_shapes(16, 16, 4, 4, 16, 12, 4, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn widen_zero_extends() {
        assert_eq!(
            widen(&[0xbf80u16, 0x0000, 0xffff]),
            vec![0xbf80u32, 0, 0xffff]
        );
    }
}
