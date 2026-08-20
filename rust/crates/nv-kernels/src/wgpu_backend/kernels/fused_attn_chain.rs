#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::pack::unpack_u8_by_element as words_to_bytes;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/fused_attn_chain.wgsl");

pub const ENTRY_Q: &str = "e4b_attn_q_rms_rope_f32";
pub const ENTRY_K: &str = "e4b_attn_k_rms_rope_fp8";
pub const ENTRY_V: &str = "e4b_attn_v_rms_fp8";

pub const WORKGROUP_SIZE: u32 = 256;
pub const MAX_HEAD_DIM: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FacParams {
    pub n_heads: u32,
    pub head_dim: u32,
    pub half_dim: u32,
    pub eps: f32,
    pub rows: u32,
    pub ring: u32,
    pub pad0: u32,
    pub pad1: u32,
}

fn check_common(n_heads: usize, head_dim: usize, w_len: usize) -> Result<()> {
    if head_dim == 0 || !head_dim.is_multiple_of(4) || head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Shape(format!(
            "fused_attn_chain head_dim must be a positive multiple of 4 no larger than {MAX_HEAD_DIM}; got {head_dim}"
        )));
    }
    if n_heads == 0 {
        return Err(WgpuError::Shape(
            "fused_attn_chain n_heads must be positive".to_string(),
        ));
    }
    dispatch::check_len("fused_attn_chain w", w_len, head_dim)?;
    Ok(())
}

fn check_tables(cos_tbl: &[f32], sin_tbl: &[f32], positions: &[i32], half: usize) -> Result<()> {
    if cos_tbl.len() != sin_tbl.len() || !cos_tbl.len().is_multiple_of(half) {
        return Err(WgpuError::Shape(format!(
            "fused_attn_chain tables: cos {} sin {} half {half}",
            cos_tbl.len(),
            sin_tbl.len()
        )));
    }
    let rows = cos_tbl.len() / half;
    for (t, p) in positions.iter().enumerate() {
        if *p < 0 || (*p as usize) >= rows {
            return Err(WgpuError::Shape(format!(
                "fused_attn_chain position[{t}]={p} out of table range 0..{rows}"
            )));
        }
    }
    Ok(())
}

fn params_uniform(
    ctx: &WgpuContext,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    tokens: usize,
    ring: usize,
) -> wgpu::Buffer {
    dispatch::uniform_from(
        ctx,
        "fac-params",
        &FacParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: (head_dim / 2) as u32,
            eps,
            rows: (tokens * n_heads) as u32,
            ring: ring as u32,
            pad0: 0,
            pad1: 0,
        },
    )
}

pub fn q_rms_rope_f32(
    ctx: &WgpuContext,
    x: &[u16],
    w: &[u16],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    out: &mut [f32],
    tokens: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    check_common(n_heads, head_dim, w.len())?;
    check_tables(cos_tbl, sin_tbl, positions, head_dim / 2)?;
    dispatch::check_len("fac q x", x.len(), tokens * n_heads * head_dim)?;
    dispatch::check_len("fac q positions", positions.len(), tokens)?;
    dispatch::check_len("fac q out", out.len(), tokens * n_heads * head_dim)?;
    if tokens == 0 {
        return Ok(());
    }
    let rows = tokens * n_heads;
    let xb = dispatch::storage_from_slice(ctx, "fac-q-x", &pack_u16(x));
    let wb = dispatch::storage_from_slice(ctx, "fac-q-w", &pack_u16(w));
    let cb = dispatch::storage_from_slice(ctx, "fac-q-cos", cos_tbl);
    let sb = dispatch::storage_from_slice(ctx, "fac-q-sin", sin_tbl);
    let pb = dispatch::storage_from_slice(ctx, "fac-q-pos", positions);
    let ob = dispatch::storage_zeroed(ctx, "fac-q-out", (out.len() * 4) as u64);
    let up = params_uniform(ctx, n_heads, head_dim, eps, tokens, 0);
    let groups = dispatch::workgroup_count_1d(ctx, rows as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_fac_q",
        &compose(WGSL),
        ENTRY_Q,
        &[
            (0, &xb),
            (1, &wb),
            (2, &cb),
            (3, &sb),
            (4, &pb),
            (6, &ob),
            (9, &up),
        ],
        groups,
    )?;
    let y: Vec<f32> = dispatch::read_back(ctx, &ob, out.len())?;
    out.copy_from_slice(&y);
    Ok(())
}

pub fn k_rms_rope_fp8(
    ctx: &WgpuContext,
    x: &[u16],
    w: &[u16],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    start: &[i32],
    out: &mut [u8],
    scales: &mut [f32],
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
    eps: f32,
) -> Result<()> {
    check_common(n_kv, head_dim, w.len())?;
    check_tables(cos_tbl, sin_tbl, positions, head_dim / 2)?;
    dispatch::check_len("fac k x", x.len(), tokens * n_kv * head_dim)?;
    dispatch::check_len("fac k positions", positions.len(), tokens)?;
    if tokens == 0 {
        return Ok(());
    }
    run_kv_entry(
        ctx,
        ENTRY_K,
        x,
        w,
        Some((cos_tbl, sin_tbl, positions)),
        start,
        out,
        scales,
        tokens,
        n_kv,
        head_dim,
        ring,
        eps,
    )
}

pub fn v_rms_fp8(
    ctx: &WgpuContext,
    x: &[u16],
    w: &[u16],
    start: &[i32],
    out: &mut [u8],
    scales: &mut [f32],
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
    eps: f32,
) -> Result<()> {
    check_common(n_kv, head_dim, w.len())?;
    dispatch::check_len("fac v x", x.len(), tokens * n_kv * head_dim)?;
    if tokens == 0 {
        return Ok(());
    }
    run_kv_entry(
        ctx, ENTRY_V, x, w, None, start, out, scales, tokens, n_kv, head_dim, ring, eps,
    )
}

fn run_kv_entry(
    ctx: &WgpuContext,
    entry: &str,
    x: &[u16],
    w: &[u16],
    tables: Option<(&[f32], &[f32], &[i32])>,
    start: &[i32],
    out: &mut [u8],
    scales: &mut [f32],
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
    eps: f32,
) -> Result<()> {
    if start.is_empty() || start[0] < 0 {
        return Err(WgpuError::Shape(format!(
            "fused_attn_chain start must hold a non-negative slot; got {start:?}"
        )));
    }
    let per_slot = n_kv * head_dim;
    if per_slot == 0 || !out.len().is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "fused_attn_chain out: length {} is not a multiple of {per_slot}",
            out.len()
        )));
    }
    let slots = out.len() / per_slot;
    dispatch::check_len("fac kv scales", scales.len(), slots * n_kv)?;
    let last = if ring > 0 {
        ring - 1
    } else {
        start[0] as usize + tokens - 1
    };
    if last >= slots {
        return Err(WgpuError::Shape(format!(
            "fused_attn_chain: start {} + tokens {tokens} (ring {ring}) exceeds {slots} slots",
            start[0]
        )));
    }
    let rows = tokens * n_kv;
    let xb = dispatch::storage_from_slice(ctx, "fac-kv-x", &pack_u16(x));
    let wb = dispatch::storage_from_slice(ctx, "fac-kv-w", &pack_u16(w));
    let stb = dispatch::storage_from_slice(ctx, "fac-kv-start", &start[..1]);
    let out_words = bytes_to_words(out);
    let ob = dispatch::storage_from_slice(ctx, "fac-kv-out", &out_words);
    let scb = dispatch::storage_from_slice(ctx, "fac-kv-scales", scales);
    let up = params_uniform(ctx, n_kv, head_dim, eps, tokens, ring);
    let groups = dispatch::workgroup_count_1d(ctx, rows as u64, 1);
    let src = compose(WGSL);
    match tables {
        Some((cos_tbl, sin_tbl, positions)) => {
            let cb = dispatch::storage_from_slice(ctx, "fac-kv-cos", cos_tbl);
            let sb = dispatch::storage_from_slice(ctx, "fac-kv-sin", sin_tbl);
            let pb = dispatch::storage_from_slice(ctx, "fac-kv-pos", positions);
            dispatch::run(
                ctx,
                "nv_kernels_fac_k",
                &src,
                entry,
                &[
                    (0, &xb),
                    (1, &wb),
                    (2, &cb),
                    (3, &sb),
                    (4, &pb),
                    (5, &stb),
                    (7, &ob),
                    (8, &scb),
                    (9, &up),
                ],
                groups,
            )?;
        }
        None => {
            dispatch::run(
                ctx,
                "nv_kernels_fac_v",
                &src,
                entry,
                &[(0, &xb), (1, &wb), (5, &stb), (7, &ob), (8, &scb), (9, &up)],
                groups,
            )?;
        }
    }
    let got_out: Vec<u32> = dispatch::read_back(ctx, &ob, out_words.len())?;
    words_to_bytes(&got_out, out);
    let got_scales: Vec<f32> = dispatch::read_back(ctx, &scb, scales.len())?;
    scales.copy_from_slice(&got_scales);
    Ok(())
}
