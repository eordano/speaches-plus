#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_bf16};

pub const WGSL: &str = include_str!("../../../wgsl/attn_decode_small_m.wgsl");

pub const WORKGROUP_SIZE: u32 = 128;
pub const MAX_PER_THREAD: usize = 4;
pub const MAX_HEAD_DIM: usize = WORKGROUP_SIZE as usize * MAX_PER_THREAD;
pub const MAX_M: usize = 9;

const SCRATCH_BYTES: u32 = (MAX_M as u32) * (MAX_HEAD_DIM as u32) * 4 + WORKGROUP_SIZE * 4;

pub const FP8_SUPPORTED_HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];

const FP8_SCRATCH_BYTES: u32 = (MAX_M as u32) * (MAX_HEAD_DIM as u32) * 4 + 512 * 4 + 32 * 4;

const ENTRY_F32: &str = "attn_decode_small_m_f32";
const ENTRY_BF16KV: &str = "attn_decode_small_m_bf16kv";

pub fn fp8_entry_for(head_dim: usize) -> Result<&'static str> {
    match head_dim {
        64 => Ok("attn_decode_small_m_fp8_hd64"),
        128 => Ok("attn_decode_small_m_fp8_hd128"),
        256 => Ok("attn_decode_small_m_fp8_hd256"),
        512 => Ok("attn_decode_small_m_fp8_hd512"),
        _ => Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_fp8 supports head_dim in {FP8_SUPPORTED_HEAD_DIMS:?}; got {head_dim}"
        ))),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SmParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    total: u32,
    m_rows: u32,
    window: u32,
    scaling: f32,
    _pad0: u32,
}

pub enum SmallMKv<'a> {
    F32 {
        k: &'a [f32],
        v: &'a [f32],
    },
    Bf16 {
        k: &'a [u16],
        v: &'a [u16],
    },
    Fp8 {
        k: &'a [u8],
        v: &'a [u8],
        k_scales: &'a [f32],
        v_scales: &'a [f32],
    },
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "attn_decode_small_m", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_m_rows(m_rows: usize) -> Result<()> {
    if !(1..=MAX_M).contains(&m_rows) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m m_rows {m_rows} out of range 1..={MAX_M}"
        )));
    }
    Ok(())
}

fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Result<()> {
    if head_dim == 0 || head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m head_dim {head_dim} out of range 1..={MAX_HEAD_DIM}"
        )));
    }
    if n_heads == 0 || n_kv_heads == 0 {
        return Err(WgpuError::Shape(
            "attn_decode_small_m n_heads and n_kv_heads must be non-zero".to_string(),
        ));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    Ok(())
}

fn check_total(m_rows: usize, total: usize) -> Result<()> {
    if total < m_rows {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m total {total} is smaller than m_rows {m_rows}"
        )));
    }
    Ok(())
}

fn make_params(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> SmParams {
    SmParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        total: total as u32,
        m_rows: m_rows as u32,
        window: window as u32,
        scaling,
        _pad0: 0,
    }
}

fn check_device_fp8(ctx: &WgpuContext, head_dim: usize) -> Result<()> {
    let block = head_dim as u32;
    if ctx.caps.max_compute_invocations_per_workgroup < block
        || ctx.caps.max_compute_workgroup_size_x < block
    {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_fp8 needs a {block}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    if !ctx.caps.workgroup_storage_fits(FP8_SCRATCH_BYTES) {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_fp8 scratch needs {FP8_SCRATCH_BYTES} bytes of workgroup storage; device allows {}",
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

fn bf16_bits_from_f32(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = 0x7fffu32 + ((b >> 16) & 1);
    (b.wrapping_add(r) >> 16) as u16
}

pub fn attn_decode_small_m_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Result<()> {
    check_m_rows(m_rows)?;
    check_geometry(n_heads, n_kv_heads, head_dim)?;
    check_total(m_rows, total)?;
    check_device(ctx)?;
    dispatch::check_len(
        "attn_decode_small_m q",
        q.len(),
        m_rows * n_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m out",
        out.len(),
        m_rows * n_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m k",
        k.len(),
        total * n_kv_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m v",
        v.len(),
        total * n_kv_heads * head_dim,
    )?;

    let params = make_params(
        n_heads, n_kv_heads, head_dim, m_rows, total, window, scaling,
    );

    let qb = dispatch::storage_from_slice(ctx, "sm-attn-q", q);
    let kb = dispatch::storage_from_slice(ctx, "sm-attn-k", k);
    let vb = dispatch::storage_from_slice(ctx, "sm-attn-v", v);
    let ob = dispatch::storage_zeroed(ctx, "sm-attn-out", (m_rows * n_heads * head_dim * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "sm-attn-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_heads as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_attn_decode_small_m_f32",
        &compose(WGSL),
        ENTRY_F32,
        &[(0, &qb), (1, &kb), (2, &vb), (3, &ob), (4, &pb)],
        groups,
    )?;

    let got: Vec<f32> = dispatch::read_back(ctx, &ob, m_rows * n_heads * head_dim)?;
    out.copy_from_slice(&got);
    Ok(())
}

pub fn attn_decode_small_m_bf16kv(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Result<()> {
    check_m_rows(m_rows)?;
    check_geometry(n_heads, n_kv_heads, head_dim)?;
    check_total(m_rows, total)?;
    if !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "attn_decode_small_m_bf16kv needs an even head_dim so bf16 pairs land on whole u32 words; got {head_dim}"
        )));
    }
    check_device(ctx)?;
    dispatch::check_len(
        "attn_decode_small_m q",
        q.len(),
        m_rows * n_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m out",
        out.len(),
        m_rows * n_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m k",
        k.len(),
        total * n_kv_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m v",
        v.len(),
        total * n_kv_heads * head_dim,
    )?;

    let params = make_params(
        n_heads, n_kv_heads, head_dim, m_rows, total, window, scaling,
    );

    let qb = dispatch::storage_from_slice(ctx, "sm-attn-q", q);
    let kwords = pack_bf16(k);
    let vwords = pack_bf16(v);
    let kb = dispatch::storage_from_slice(ctx, "sm-attn-k-bf16", &kwords);
    let vb = dispatch::storage_from_slice(ctx, "sm-attn-v-bf16", &vwords);
    let ob = dispatch::storage_zeroed(ctx, "sm-attn-out", (m_rows * n_heads * head_dim * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "sm-attn-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_heads as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_attn_decode_small_m_bf16kv",
        &compose(WGSL),
        ENTRY_BF16KV,
        &[(0, &qb), (3, &ob), (4, &pb), (5, &kb), (6, &vb)],
        groups,
    )?;

    let got: Vec<f32> = dispatch::read_back(ctx, &ob, m_rows * n_heads * head_dim)?;
    out.copy_from_slice(&got);
    Ok(())
}

pub fn attn_decode_small_m_fp8(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Result<()> {
    check_m_rows(m_rows)?;
    check_geometry(n_heads, n_kv_heads, head_dim)?;
    check_total(m_rows, total)?;
    let entry = fp8_entry_for(head_dim)?;
    check_device_fp8(ctx, head_dim)?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 q",
        q.len(),
        m_rows * n_heads * head_dim,
    )?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 out",
        out.len(),
        m_rows * n_heads * head_dim,
    )?;
    let per_slot = n_kv_heads * head_dim;
    if !k_fp8.len().is_multiple_of(per_slot) || !v_fp8.len().is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "attn_decode_small_m_fp8: k/v byte counts {} / {} are not multiples of n_kv*head_dim {per_slot}",
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
    dispatch::check_len("attn_decode_small_m_fp8 v", v_fp8.len(), slots * per_slot)?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 k_scales",
        k_scales.len(),
        slots * n_kv_heads,
    )?;
    dispatch::check_len(
        "attn_decode_small_m_fp8 v_scales",
        v_scales.len(),
        slots * n_kv_heads,
    )?;

    let params = make_params(
        n_heads, n_kv_heads, head_dim, m_rows, total, window, scaling,
    );

    let qb = dispatch::storage_from_slice(ctx, "sm-attn-fp8-q", &widen_u16(q));
    let kb = dispatch::storage_from_slice(ctx, "sm-attn-fp8-k", &bytes_to_words(k_fp8));
    let vb = dispatch::storage_from_slice(ctx, "sm-attn-fp8-v", &bytes_to_words(v_fp8));
    let ksb = dispatch::storage_from_slice(ctx, "sm-attn-fp8-kscale", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "sm-attn-fp8-vscale", v_scales);
    let ob = dispatch::storage_zeroed(
        ctx,
        "sm-attn-fp8-out",
        (m_rows * n_heads * head_dim * 4) as u64,
    );
    let sb = dispatch::storage_zeroed(
        ctx,
        "sm-attn-fp8-scores",
        (m_rows * n_heads * total * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "sm-attn-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_heads as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_attn_decode_small_m_fp8",
        &compose(WGSL),
        entry,
        &[
            (4, &pb),
            (7, &qb),
            (8, &kb),
            (9, &vb),
            (10, &ksb),
            (11, &vsb),
            (12, &ob),
            (13, &sb),
        ],
        groups,
    )?;

    let got: Vec<u32> = dispatch::read_back(ctx, &ob, m_rows * n_heads * head_dim)?;
    for (dst, word) in out.iter_mut().zip(got.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

pub fn attn_decode_small_m_dispatch(
    ctx: &WgpuContext,
    q: &[f32],
    kv: SmallMKv,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Result<()> {
    check_m_rows(m_rows)?;
    match kv {
        SmallMKv::F32 { k, v } => attn_decode_small_m_f32(
            ctx, q, k, v, out, n_heads, n_kv_heads, head_dim, m_rows, total, window, scaling,
        ),
        SmallMKv::Bf16 { k, v } => attn_decode_small_m_bf16kv(
            ctx, q, k, v, out, n_heads, n_kv_heads, head_dim, m_rows, total, window, scaling,
        ),
        SmallMKv::Fp8 {
            k,
            v,
            k_scales,
            v_scales,
        } => {
            let q_bf16: Vec<u16> = q.iter().map(|x| bf16_bits_from_f32(*x)).collect();
            let mut out_bf16 = vec![0u16; out.len()];
            attn_decode_small_m_fp8(
                ctx,
                &q_bf16,
                k,
                v,
                k_scales,
                v_scales,
                &mut out_bf16,
                n_heads,
                n_kv_heads,
                head_dim,
                m_rows,
                total,
                window,
                scaling,
            )?;
            for (dst, bits) in out.iter_mut().zip(out_bf16.iter()) {
                *dst = f32::from_bits((*bits as u32) << 16);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_rows_out_of_range_is_rejected() {
        assert!(check_m_rows(0).is_err());
        assert!(check_m_rows(10).is_err());
        for m in 1..=9 {
            assert!(check_m_rows(m).is_ok());
        }
    }

    #[test]
    fn geometry_rejects_ragged_head_groups() {
        assert!(matches!(
            check_geometry(6, 4, 64).unwrap_err(),
            WgpuError::Shape(_)
        ));
        assert!(matches!(
            check_geometry(8, 4, MAX_HEAD_DIM + 1).unwrap_err(),
            WgpuError::Unsupported(_)
        ));
    }

    #[test]
    fn total_below_m_rows_is_rejected() {
        assert!(check_total(4, 3).is_err());
        assert!(check_total(4, 4).is_ok());
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<SmParams>() % 16, 0);
    }

    #[test]
    fn bf16_pack_handles_odd_lengths() {
        assert_eq!(pack_bf16(&[1, 2, 3]), vec![0x0002_0001, 3]);
        assert_eq!(pack_bf16(&[]), vec![0]);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn scratch_bytes_fit_a_32kb_threadgroup() {
        assert!(
            SCRATCH_BYTES <= 32768,
            "scratch {SCRATCH_BYTES} exceeds 32KB"
        );
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn fp8_scratch_bytes_fit_a_32kb_threadgroup() {
        assert_eq!(FP8_SCRATCH_BYTES, 20608);
        assert!(
            FP8_SCRATCH_BYTES <= 32768,
            "fp8 scratch {FP8_SCRATCH_BYTES} exceeds 32KB"
        );
    }

    #[test]
    fn fp8_head_dim_selects_the_matching_entry_point() {
        assert_eq!(fp8_entry_for(64).unwrap(), "attn_decode_small_m_fp8_hd64");
        assert_eq!(fp8_entry_for(128).unwrap(), "attn_decode_small_m_fp8_hd128");
        assert_eq!(fp8_entry_for(256).unwrap(), "attn_decode_small_m_fp8_hd256");
        assert_eq!(fp8_entry_for(512).unwrap(), "attn_decode_small_m_fp8_hd512");
        assert!(fp8_entry_for(96).is_err());
        assert!(fp8_entry_for(1024).is_err());
    }

    #[test]
    fn wgsl_declares_every_fp8_entry_point() {
        for hd in FP8_SUPPORTED_HEAD_DIMS {
            assert!(WGSL.contains(fp8_entry_for(hd).unwrap()));
        }
        assert!(crate::wgpu_backend::compose(WGSL).contains("fn e4m3_decode("));
    }

    #[test]
    fn fp8_staging_helpers_round_trip() {
        assert_eq!(widen_u16(&[0x3f80, 0xbf00]), vec![0x3f80u32, 0xbf00u32]);
        assert_eq!(widen_u16(&[]), vec![0]);
        assert_eq!(bytes_to_words(&[1, 2, 3, 4]), vec![0x04030201u32]);
        assert_eq!(bytes_to_words(&[]), vec![0]);
    }

    #[test]
    fn bf16_bits_round_to_nearest_even_like_the_wgsl_encoder() {
        assert_eq!(bf16_bits_from_f32(1.0), 0x3f80);
        assert_eq!(bf16_bits_from_f32(-2.0), 0xc000);
        assert_eq!(bf16_bits_from_f32(f32::NAN), 0x7fc0);
        assert_eq!(bf16_bits_from_f32(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16_bits_from_f32(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16_bits_from_f32(f32::from_bits(0x3f80_8001)), 0x3f81);
    }
}
