#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16, unpack_u16_pairs as unpack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/rope_bf16.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    n_heads: u32,
    half_dim: u32,
    total_words: u32,
    table_rows: u32,
}

struct Tables {
    cos: wgpu::Buffer,
    sin: wgpu::Buffer,
    pos: wgpu::Buffer,
    rows: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "rope_bf16", WORKGROUP_SIZE)
}

fn check_tables(
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    batch: usize,
    head_dim: usize,
) -> Result<usize> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "rope_bf16 head_dim must be even and non-zero; got {head_dim}"
        )));
    }
    let half = head_dim / 2;
    dispatch::check_len("rope_bf16 positions", positions.len(), batch)?;
    if cos_tbl.len() != sin_tbl.len() {
        return Err(WgpuError::Shape(format!(
            "rope_bf16 cos/sin tables differ: {} vs {}",
            cos_tbl.len(),
            sin_tbl.len()
        )));
    }
    if !cos_tbl.len().is_multiple_of(half) {
        return Err(WgpuError::Shape(format!(
            "rope_bf16 table length {} is not a multiple of half_dim {half}",
            cos_tbl.len()
        )));
    }
    let rows = cos_tbl.len() / half;
    for (t, p) in positions.iter().enumerate() {
        if *p < 0 || (*p as usize) >= rows {
            return Err(WgpuError::Shape(format!(
                "rope_bf16 position[{t}]={p} out of table range 0..{rows}"
            )));
        }
    }
    Ok(half)
}

fn upload_tables(
    ctx: &WgpuContext,
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    rows: usize,
) -> Tables {
    Tables {
        cos: dispatch::storage_from_slice(ctx, "rope-bf16-cos", cos_tbl),
        sin: dispatch::storage_from_slice(ctx, "rope-bf16-sin", sin_tbl),
        pos: dispatch::storage_from_slice(ctx, "rope-bf16-pos", positions),
        rows: rows as u32,
    }
}

fn rope_one(
    ctx: &WgpuContext,
    tables: &Tables,
    x: &[u16],
    batch: usize,
    n_heads: usize,
    half: usize,
) -> Result<Vec<u32>> {
    let words = pack_u16(x);
    let total_words = batch * n_heads * half;
    let params = RopeParams {
        n_heads: n_heads as u32,
        half_dim: half as u32,
        total_words: total_words as u32,
        table_rows: tables.rows,
    };
    let src = dispatch::storage_from_slice(ctx, "rope-bf16-src", &words);
    let dst = dispatch::storage_zeroed(ctx, "rope-bf16-dst", (total_words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "rope-bf16-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, total_words as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_rope_bf16",
        &compose(WGSL),
        "rope_bf16",
        &[
            (0, &src),
            (1, &dst),
            (2, &tables.cos),
            (3, &tables.sin),
            (4, &tables.pos),
            (5, &pb),
        ],
        groups,
    )?;
    dispatch::read_back(ctx, &dst, total_words)
}

pub fn rope_bf16(
    ctx: &WgpuContext,
    q: &mut [u16],
    k: &mut [u16],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let half = check_tables(cos_tbl, sin_tbl, positions, batch, head_dim)?;
    dispatch::check_len("rope_bf16 q", q.len(), batch * n_heads * head_dim)?;
    dispatch::check_len("rope_bf16 k", k.len(), batch * n_kv_heads * head_dim)?;
    if batch == 0 {
        return Ok(());
    }
    check_device(ctx)?;
    let tables = upload_tables(ctx, cos_tbl, sin_tbl, positions, cos_tbl.len() / half);
    if n_heads > 0 {
        let out = rope_one(ctx, &tables, q, batch, n_heads, half)?;
        unpack_u16(&out, q);
    }
    if n_kv_heads > 0 {
        let out = rope_one(ctx, &tables, k, batch, n_kv_heads, half)?;
        unpack_u16(&out, k);
    }
    Ok(())
}

pub fn rope_bf16_oop(
    ctx: &WgpuContext,
    q_in: &[u16],
    k_in: &[u16],
    q_out: &mut [u16],
    k_out: &mut [u16],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let half = check_tables(cos_tbl, sin_tbl, positions, batch, head_dim)?;
    dispatch::check_len("rope_bf16_oop q_in", q_in.len(), batch * n_heads * head_dim)?;
    dispatch::check_len(
        "rope_bf16_oop k_in",
        k_in.len(),
        batch * n_kv_heads * head_dim,
    )?;
    dispatch::check_len(
        "rope_bf16_oop q_out",
        q_out.len(),
        batch * n_heads * head_dim,
    )?;
    dispatch::check_len(
        "rope_bf16_oop k_out",
        k_out.len(),
        batch * n_kv_heads * head_dim,
    )?;
    if batch == 0 {
        return Ok(());
    }
    check_device(ctx)?;
    let tables = upload_tables(ctx, cos_tbl, sin_tbl, positions, cos_tbl.len() / half);
    if n_heads > 0 {
        let out = rope_one(ctx, &tables, q_in, batch, n_heads, half)?;
        unpack_u16(&out, q_out);
    }
    if n_kv_heads > 0 {
        let out = rope_one(ctx, &tables, k_in, batch, n_kv_heads, half)?;
        unpack_u16(&out, k_out);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_head_dim_is_rejected() {
        let e = check_tables(&[1.0], &[0.0], &[0], 1, 3).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn table_rows_are_derived_from_half_dim() {
        let cos = vec![0f32; 8];
        let sin = vec![0f32; 8];
        let half = check_tables(&cos, &sin, &[0, 3], 2, 4).unwrap();
        assert_eq!(half, 2);
    }

    #[test]
    fn out_of_range_position_is_rejected() {
        let cos = vec![0f32; 8];
        let sin = vec![0f32; 8];
        let e = check_tables(&cos, &sin, &[0, 4], 2, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn u16_word_packing_round_trips() {
        let src: Vec<u16> = (0u16..32).map(|i| i.wrapping_mul(2311)).collect();
        let words = pack_u16(&src);
        let mut back = vec![0u16; src.len()];
        unpack_u16(&words, &mut back);
        assert_eq!(back, src);
    }
}
