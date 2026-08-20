#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/attn_decode_small_m_fp8.wgsl");

pub const SUPPORTED_HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];
pub const MAX_M: usize = 10;

const SCRATCH_BYTES: u32 = (MAX_M as u32) * 512 * 4 + 512 * 4 + 32 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sm8Params {
    pub n_q: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub total: u32,
    pub m_rows: u32,
    pub window: u32,
    pub score_stride: u32,
    pub scaling: f32,
}

pub fn entry_for(head_dim: usize) -> Result<&'static str> {
    match head_dim {
        64 => Ok("attn_decode_small_m_fp8_hd64"),
        128 => Ok("attn_decode_small_m_fp8_hd128"),
        256 => Ok("attn_decode_small_m_fp8_hd256"),
        512 => Ok("attn_decode_small_m_fp8_hd512"),
        _ => Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_fp8 supports head_dim in {SUPPORTED_HEAD_DIMS:?}; got {head_dim}"
        ))),
    }
}

pub fn check_m_rows(m_rows: usize) -> Result<()> {
    if !(1..=MAX_M).contains(&m_rows) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_fp8 m_rows {m_rows} out of range 1..={MAX_M}"
        )));
    }
    Ok(())
}

fn check_device(ctx: &WgpuContext, head_dim: usize) -> Result<()> {
    let block = head_dim as u32;
    if ctx.caps.max_compute_invocations_per_workgroup < block
        || ctx.caps.max_compute_workgroup_size_x < block
    {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_fp8 needs a {block}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    if !ctx.caps.workgroup_storage_fits(SCRATCH_BYTES) {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_fp8 scratch needs {SCRATCH_BYTES} bytes of workgroup storage; device allows {}",
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

pub fn attn_decode_small_m_fp8(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [u16],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Result<()> {
    if n_q == 0 || n_kv == 0 {
        return Err(WgpuError::Shape(
            "attn_decode_small_m_fp8 n_q and n_kv must be non-zero".to_string(),
        ));
    }
    if !n_q.is_multiple_of(n_kv) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_fp8 n_q {n_q} is not a multiple of n_kv {n_kv}"
        )));
    }
    check_m_rows(m_rows)?;
    if total < m_rows {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_fp8 total {total} is smaller than m_rows {m_rows}"
        )));
    }
    let entry = entry_for(head_dim)?;
    check_device(ctx, head_dim)?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 q",
        q.len(),
        m_rows * n_q * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 out",
        out.len(),
        m_rows * n_q * head_dim,
    )?;
    let per_slot = n_kv * head_dim;
    if !k_fp8.len().is_multiple_of(per_slot) || v_fp8.len() != k_fp8.len() {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_fp8: k/v byte counts {} / {} are not equal multiples of n_kv*head_dim {per_slot}",
            k_fp8.len(),
            v_fp8.len()
        )));
    }
    let slots = k_fp8.len() / per_slot;
    if total > slots {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_fp8: total {total} exceeds {slots} kv slots"
        )));
    }
    dispatch::check_len(
        "attn_decode_small_m_fp8 k_scales",
        k_scales.len(),
        slots * n_kv,
    )?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 v_scales",
        v_scales.len(),
        slots * n_kv,
    )?;

    let params = Sm8Params {
        n_q: n_q as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        total: total as u32,
        m_rows: m_rows as u32,
        window: window as u32,
        score_stride: total as u32,
        scaling,
    };

    let qb = dispatch::storage_from_slice(ctx, "smk-fp8-q", &widen_u16(q));
    let kb = dispatch::storage_from_slice(ctx, "smk-fp8-k", &bytes_to_words(k_fp8));
    let vb = dispatch::storage_from_slice(ctx, "smk-fp8-v", &bytes_to_words(v_fp8));
    let ksb = dispatch::storage_from_slice(ctx, "smk-fp8-kscale", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "smk-fp8-vscale", v_scales);
    let ob = dispatch::storage_zeroed(ctx, "smk-fp8-out", (m_rows * n_q * head_dim * 4) as u64);
    let sb = dispatch::storage_zeroed(ctx, "smk-fp8-scores", (n_q * m_rows * total * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "smk-fp8-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_q as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_attn_decode_small_m_fp8",
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

    let got: Vec<u32> = dispatch::read_back(ctx, &ob, m_rows * n_q * head_dim)?;
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
        assert_eq!(entry_for(64).unwrap(), "attn_decode_small_m_fp8_hd64");
        assert_eq!(entry_for(512).unwrap(), "attn_decode_small_m_fp8_hd512");
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
        assert_eq!(std::mem::size_of::<Sm8Params>() % 16, 0);
    }

    #[test]
    fn m_rows_out_of_range_is_rejected() {
        assert!(check_m_rows(0).is_err());
        assert!(check_m_rows(MAX_M + 1).is_err());
        for m in 1..=MAX_M {
            assert!(check_m_rows(m).is_ok());
        }
    }

    #[test]
    fn byte_widening_pads_the_tail_word() {
        assert_eq!(bytes_to_words(&[1, 2, 3, 4, 5]), vec![0x04030201u32, 5]);
        assert_eq!(widen_u16(&[]), vec![0]);
        assert_eq!(bytes_to_words(&[]), vec![0]);
    }

    #[test]
    fn wgsl_scratch_matches_the_declared_budget() {
        assert!(WGSL.contains("array<f32, 5120>"));
        assert_eq!(SCRATCH_BYTES, 5120 * 4 + 512 * 4 + 32 * 4);
        assert!(WGSL.contains(&format!("const SM8_MAX_M: u32 = {MAX_M}u;")));
    }
}
