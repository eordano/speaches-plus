#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};
#[cfg(test)]
use crate::wgpu_backend::WgpuError;

pub const WGSL: &str = include_str!("../../../wgsl/residual_scale.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

pub const ELEMS_PER_INVOCATION: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    n: u32,
    n_words: u32,
    scale: f32,
    cap: f32,
    inv_cap: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "residual_scale", WORKGROUP_SIZE)
}

fn pack_u16(src: &[u16], words: usize) -> Vec<u32> {
    let mut out = vec![0u32; words];
    for (i, w) in out.iter_mut().enumerate() {
        let lo = src[2 * i] as u32;
        let hi = src.get(2 * i + 1).copied().unwrap_or(0) as u32;
        *w = lo | (hi << 16);
    }
    out
}

fn unpack_u16(words: &[u32], dst: &mut [u16]) {
    for (i, w) in words.iter().enumerate() {
        dst[2 * i] = (*w & 0xffff) as u16;
        if 2 * i + 1 < dst.len() {
            dst[2 * i + 1] = (*w >> 16) as u16;
        }
    }
}

fn word_count(n: usize) -> usize {
    n.div_ceil(ELEMS_PER_INVOCATION)
}

fn params(n: usize, n_words: usize, scale: f32, cap: f32, inv_cap: f32) -> ScaleParams {
    ScaleParams {
        n: n as u32,
        n_words: n_words as u32,
        scale,
        cap,
        inv_cap,
        ..Default::default()
    }
}

pub fn residual_add_scale_bf16(
    ctx: &WgpuContext,
    a: &[u16],
    b: &[u16],
    y: &mut [u16],
    scale: f32,
    n: usize,
) -> Result<()> {
    dispatch::check_len("residual_add_scale_bf16 a", a.len(), n)?;
    dispatch::check_len("residual_add_scale_bf16 b", b.len(), n)?;
    dispatch::check_len("residual_add_scale_bf16 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let words = word_count(n);
    let p = params(n, words, scale, 0.0, 0.0);
    let ab = dispatch::storage_from_slice(ctx, "residual-scale-a", &pack_u16(a, words));
    let bb = dispatch::storage_from_slice(ctx, "residual-scale-b", &pack_u16(b, words));
    let yb = dispatch::storage_zeroed(ctx, "residual-scale-y", (words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "residual-scale-params", &p);

    let groups = dispatch::workgroup_count_1d(ctx, words as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_residual_add_scale_bf16",
        &compose(WGSL),
        "residual_add_scale_bf16",
        &[(0, &ab), (1, &bb), (2, &yb), (3, &pb)],
        groups,
    )?;

    let out: Vec<u32> = dispatch::read_back(ctx, &yb, words)?;
    unpack_u16(&out, y);
    Ok(())
}

fn scale_into(
    ctx: &WgpuContext,
    label: &str,
    x: &[u16],
    y: &mut [u16],
    scale: f32,
    n: usize,
) -> Result<()> {
    let words = word_count(n);
    let p = params(n, words, scale, 0.0, 0.0);
    let xb = dispatch::storage_from_slice(ctx, "scale-bf16-x", &pack_u16(x, words));
    let yb = dispatch::storage_zeroed(ctx, "scale-bf16-y", (words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "scale-bf16-params", &p);

    let groups = dispatch::workgroup_count_1d(ctx, words as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        label,
        &compose(WGSL),
        "scale_bf16",
        &[(0, &xb), (2, &yb), (3, &pb)],
        groups,
    )?;

    let out: Vec<u32> = dispatch::read_back(ctx, &yb, words)?;
    unpack_u16(&out, y);
    Ok(())
}

pub fn scale_inplace_bf16(ctx: &WgpuContext, y: &mut [u16], scale: f32, n: usize) -> Result<()> {
    dispatch::check_len("scale_inplace_bf16 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    check_device(ctx)?;
    let src = y.to_vec();
    scale_into(ctx, "nv_kernels_scale_inplace_bf16", &src, y, scale, n)
}

pub fn scale_out_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    y: &mut [u16],
    scale: f32,
    n: usize,
) -> Result<()> {
    dispatch::check_len("scale_out_bf16 x", x.len(), n)?;
    dispatch::check_len("scale_out_bf16 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    check_device(ctx)?;
    scale_into(ctx, "nv_kernels_scale_out_bf16", x, y, scale, n)
}

pub fn tanh_softcap_bf16_to_f32(
    ctx: &WgpuContext,
    x: &[u16],
    y: &mut [f32],
    cap: f32,
    n: usize,
) -> Result<()> {
    dispatch::check_len("tanh_softcap_bf16_to_f32 x", x.len(), n)?;
    dispatch::check_len("tanh_softcap_bf16_to_f32 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let softcap = cap > 0.0 && cap.is_finite();
    let inv_cap = if softcap { 1.0f32 / cap } else { 0.0 };
    let words = word_count(n);
    let p = params(n, words, 0.0, cap, inv_cap);
    let xb = dispatch::storage_from_slice(ctx, "softcap-x", &pack_u16(x, words));
    let yb = dispatch::storage_zeroed(ctx, "softcap-y", (n * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "softcap-params", &p);

    let entry = if softcap {
        "tanh_softcap_bf16_to_f32"
    } else {
        "cast_bf16_to_f32"
    };
    let groups = dispatch::workgroup_count_1d(ctx, words as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_tanh_softcap_bf16_to_f32",
        &compose(WGSL),
        entry,
        &[(0, &xb), (4, &yb), (3, &pb)],
        groups,
    )?;

    let out: Vec<f32> = dispatch::read_back(ctx, &yb, n)?;
    y.copy_from_slice(&out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_length_packs_and_unpacks() {
        let src: Vec<u16> = vec![1, 2, 3, 4, 5];
        let words = word_count(src.len());
        assert_eq!(words, 3);
        let packed = pack_u16(&src, words);
        assert_eq!(packed[2], 5);
        let mut back = vec![0u16; src.len()];
        unpack_u16(&packed, &mut back);
        assert_eq!(back, src);
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let e = dispatch::check_len("scale_out_bf16 x", 3, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }
}
