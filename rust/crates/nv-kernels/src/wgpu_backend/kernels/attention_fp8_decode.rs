#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/attention_fp8_decode.wgsl");

pub const SUPPORTED_HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];

const SCRATCH_BYTES: u32 = 512 * 4 * 2 + 32 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnFp8Params {
    n_q: u32,
    n_kv: u32,
    head_dim: u32,
    n_total: u32,
    sliding_window: u32,
    score_stride: u32,
    scaling: f32,
    reserved: u32,
}

pub fn entry_for(head_dim: usize) -> Result<&'static str> {
    match head_dim {
        64 => Ok("attention_fp8_decode_hd64"),
        128 => Ok("attention_fp8_decode_hd128"),
        256 => Ok("attention_fp8_decode_hd256"),
        512 => Ok("attention_fp8_decode_hd512"),
        _ => Err(WgpuError::Unsupported(format!(
            "attention_fp8_decode supports head_dim in {SUPPORTED_HEAD_DIMS:?}; got {head_dim}"
        ))),
    }
}

fn check_device(ctx: &WgpuContext, head_dim: usize) -> Result<()> {
    let block = head_dim as u32;
    if ctx.caps.max_compute_invocations_per_workgroup < block
        || ctx.caps.max_compute_workgroup_size_x < block
    {
        return Err(WgpuError::Unsupported(format!(
            "attention_fp8_decode needs a {block}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    if !ctx.caps.workgroup_storage_fits(SCRATCH_BYTES) {
        return Err(WgpuError::Unsupported(format!(
            "attention_fp8_decode scratch needs {SCRATCH_BYTES} bytes of workgroup storage; device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    Ok(())
}

fn widen_u16(src: &[u16]) -> Vec<u32> {
    let mut out: Vec<u32> = src.iter().map(|v| *v as u32).collect();
    if out.is_empty() {
        out.push(0);
    }
    out
}

pub fn attention_fp8_decode(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [u16],
    n_total: &[i32],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    sliding_window: usize,
    scaling: f32,
) -> Result<()> {
    if n_q == 0 || n_kv == 0 || head_dim == 0 {
        return Ok(());
    }
    if !n_q.is_multiple_of(n_kv) {
        return Err(WgpuError::Shape(format!(
            "attention_fp8_decode n_q {n_q} is not a multiple of n_kv {n_kv}"
        )));
    }
    let entry = entry_for(head_dim)?;
    if n_total.is_empty() {
        return Err(WgpuError::Shape(
            "attention_fp8_decode: n_total buffer is empty".to_string(),
        ));
    }
    if n_total[0] < 0 {
        return Err(WgpuError::Shape(format!(
            "attention_fp8_decode: n_total must be non-negative; got {}",
            n_total[0]
        )));
    }
    let total = n_total[0] as usize;
    dispatch::check_len("attention_fp8_decode q", q.len(), n_q * head_dim)?;
    dispatch::check_len("attention_fp8_decode out", out.len(), n_q * head_dim)?;
    let per_slot = n_kv * head_dim;
    if !k_fp8.len().is_multiple_of(per_slot) || !v_fp8.len().is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "attention_fp8_decode: k/v byte counts {} / {} are not multiples of n_kv*head_dim {per_slot}",
            k_fp8.len(),
            v_fp8.len()
        )));
    }
    let slots = k_fp8.len() / per_slot;
    if slots == 0 {
        return Err(WgpuError::Shape(
            "attention_fp8_decode: the kv cache holds no slots".to_string(),
        ));
    }
    dispatch::check_len("attention_fp8_decode v", v_fp8.len(), slots * per_slot)?;
    dispatch::check_len(
        "attention_fp8_decode k_scales",
        k_scales.len(),
        slots * n_kv,
    )?;
    dispatch::check_len(
        "attention_fp8_decode v_scales",
        v_scales.len(),
        slots * n_kv,
    )?;
    if total > slots {
        return Err(WgpuError::Shape(format!(
            "attention_fp8_decode: n_total {total} exceeds {slots} kv slots"
        )));
    }
    check_device(ctx, head_dim)?;

    let params = AttnFp8Params {
        n_q: n_q as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        n_total: total as u32,
        sliding_window: sliding_window as u32,
        score_stride: total.max(1) as u32,
        scaling,
        reserved: 0,
    };

    let qb = dispatch::storage_from_slice(ctx, "attn-fp8-q", &widen_u16(q));
    let kb = dispatch::storage_from_slice(ctx, "attn-fp8-k", &bytes_to_words(k_fp8));
    let vb = dispatch::storage_from_slice(ctx, "attn-fp8-v", &bytes_to_words(v_fp8));
    let ksb = dispatch::storage_from_slice(ctx, "attn-fp8-kscale", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "attn-fp8-vscale", v_scales);
    let ob = dispatch::storage_zeroed(ctx, "attn-fp8-out", (n_q * head_dim * 4) as u64);
    let sb = dispatch::storage_zeroed(ctx, "attn-fp8-scores", (n_q * total.max(1) * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "attn-fp8-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_q as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_attention_fp8_decode",
        &compose(WGSL),
        entry,
        &[
            (0, &qb),
            (1, &kb),
            (2, &vb),
            (3, &ksb),
            (4, &vsb),
            (5, &ob),
            (6, &sb),
            (7, &pb),
        ],
        groups,
    )?;

    let got: Vec<u32> = dispatch::read_back(ctx, &ob, n_q * head_dim)?;
    for (dst, word) in out.iter_mut().zip(got.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_selects_the_matching_entry_point() {
        assert_eq!(entry_for(64).unwrap(), "attention_fp8_decode_hd64");
        assert_eq!(entry_for(512).unwrap(), "attention_fp8_decode_hd512");
        assert!(entry_for(96).is_err());
        assert!(entry_for(1024).is_err());
    }

    #[test]
    fn wgsl_declares_every_entry_point() {
        for hd in SUPPORTED_HEAD_DIMS {
            assert!(WGSL.contains(entry_for(hd).unwrap()));
        }
        assert!(compose(WGSL).contains("fn e4m3_decode("));
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<AttnFp8Params>() % 16, 0);
    }

    #[test]
    fn byte_and_halfword_widening_round_trips() {
        assert_eq!(widen_u16(&[0x3f80, 0xbf00]), vec![0x3f80u32, 0xbf00u32]);
        assert_eq!(bytes_to_words(&[1, 2, 3, 4]), vec![0x04030201u32]);
    }
}
