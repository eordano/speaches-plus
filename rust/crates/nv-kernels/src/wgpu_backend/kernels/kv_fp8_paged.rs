#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::pack::unpack_u8_by_element as words_to_bytes;
use crate::wgpu_backend::kernels::kv_fp8::{decode_e4m3, encode_e4m3, FP8_E4M3_MAX};
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_even_min_one_word as pack_bf16, unpack_u16_pairs_clamped as unpack_bf16};

pub const WGSL: &str = include_str!("../../../wgsl/kv_fp8_paged.wgsl");

pub const QUANTIZE_ENTRY: &str = "quantize_kv_fp8_paged";
pub const DEQUANTIZE_ENTRY: &str = "dequantize_kv_fp8_paged";
pub const COPY_ENTRY: &str = "copy_kv_block_fp8";
pub const COPY_CROSS_ENTRY: &str = "copy_kv_block_fp8_x";
pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 + 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvPagedParams {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    block_size: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvCopyParams {
    src_block: u32,
    dst_block: u32,
    block_size: u32,
    n_kv: u32,
    head_dim: u32,
    pairs: u32,
    reserved0: u32,
    reserved1: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "kv_fp8_paged", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn slot_count(len: usize, per_slot: usize, what: &str) -> Result<usize> {
    if per_slot == 0 || !len.is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "{what}: length {len} is not a multiple of {per_slot}"
        )));
    }
    Ok(len / per_slot)
}

pub fn paged_slot(block_table: &[i32], block_size: usize, logical: usize) -> usize {
    let blk = logical / block_size;
    let off = logical - blk * block_size;
    block_table[blk] as usize * block_size + off
}

fn check_table(
    what: &str,
    block_table: &[i32],
    block_size: usize,
    first_logical: usize,
    n_logical: usize,
    slots: usize,
) -> Result<()> {
    if block_size == 0 {
        return Err(WgpuError::Shape(format!("{what}: block_size must be > 0")));
    }
    if n_logical == 0 {
        return Ok(());
    }
    let first_blk = first_logical / block_size;
    let last_blk = (first_logical + n_logical - 1) / block_size;
    if block_table.len() <= last_blk {
        return Err(WgpuError::Shape(format!(
            "{what}: block_table has {} entries but logical position {} needs block {last_blk}",
            block_table.len(),
            first_logical + n_logical - 1
        )));
    }
    for (blk, entry) in block_table
        .iter()
        .enumerate()
        .take(last_blk + 1)
        .skip(first_blk)
    {
        if *entry < 0 {
            return Err(WgpuError::Shape(format!(
                "{what}: block_table[{blk}] = {entry} is negative"
            )));
        }
        let page_end = (*entry as usize + 1) * block_size;
        if page_end > slots {
            return Err(WgpuError::Shape(format!(
                "{what}: block_table[{blk}] = {entry} spans slot {} but the cache has {slots} slots",
                page_end - 1
            )));
        }
    }
    Ok(())
}

pub fn quantize_kv_fp8_paged(
    ctx: &WgpuContext,
    x: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    block_table: &[i32],
    start: &[i32],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    if n_tokens == 0 || n_kv == 0 || head_dim == 0 || block_size == 0 {
        return Ok(());
    }
    if !head_dim.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "kv_fp8_paged quantize needs head_dim divisible by 4 so fp8 bytes land on whole u32 words; got {head_dim}"
        )));
    }
    if start.is_empty() {
        return Err(WgpuError::Shape(
            "kv_fp8_paged quantize: start buffer is empty".to_string(),
        ));
    }
    if start[0] < 0 {
        return Err(WgpuError::Shape(format!(
            "kv_fp8_paged quantize: start must be non-negative; got {}",
            start[0]
        )));
    }
    dispatch::check_len("kv_fp8_paged x", x.len(), n_tokens * n_kv * head_dim)?;
    let slots = slot_count(out.len(), n_kv * head_dim, "kv_fp8_paged out")?;
    dispatch::check_len("kv_fp8_paged scales", scales.len(), slots * n_kv)?;
    let start0 = start[0] as usize;
    check_table(
        "kv_fp8_paged quantize",
        block_table,
        block_size,
        start0,
        n_tokens,
        slots,
    )?;
    check_device(ctx)?;

    let pairs = n_tokens * n_kv;
    let params = KvPagedParams {
        n_tokens: n_tokens as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        block_size: block_size as u32,
        pairs: pairs as u32,
        start: start0 as u32,
        slots: slots as u32,
        reserved: 0,
    };

    let x_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.x", &pack_bf16(x));
    let out_words = bytes_to_words(out);
    let out_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.out", &out_words);
    let scales_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.scales", scales);
    let start_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.start", &start[..1]);
    let table_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.table", block_table);
    let params_buf = dispatch::uniform_from(ctx, "kv_fp8_paged.params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_quantize_kv_fp8_paged",
        &compose(WGSL),
        QUANTIZE_ENTRY,
        &[
            (0, &x_buf),
            (1, &out_buf),
            (2, &scales_buf),
            (3, &start_buf),
            (4, &table_buf),
            (5, &params_buf),
        ],
        groups,
    )?;

    let got_out: Vec<u32> = dispatch::read_back(ctx, &out_buf, out_words.len())?;
    words_to_bytes(&got_out, out);
    let got_scales: Vec<f32> = dispatch::read_back(ctx, &scales_buf, scales.len())?;
    scales.copy_from_slice(&got_scales);
    Ok(())
}

pub fn dequantize_kv_fp8_paged(
    ctx: &WgpuContext,
    src: &[u8],
    scales: &[f32],
    block_table: &[i32],
    out: &mut [u16],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    if n_tokens == 0 || n_kv == 0 || head_dim == 0 || block_size == 0 {
        return Ok(());
    }
    if !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "kv_fp8_paged dequantize needs an even head_dim so bf16 pairs land on whole u32 words; got {head_dim}"
        )));
    }
    dispatch::check_len("kv_fp8_paged out", out.len(), n_tokens * n_kv * head_dim)?;
    let slots = slot_count(src.len(), n_kv * head_dim, "kv_fp8_paged src")?;
    dispatch::check_len("kv_fp8_paged scales", scales.len(), slots * n_kv)?;
    check_table(
        "kv_fp8_paged dequantize",
        block_table,
        block_size,
        0,
        n_tokens,
        slots,
    )?;
    check_device(ctx)?;

    let pairs = n_tokens * n_kv;
    let params = KvPagedParams {
        n_tokens: n_tokens as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        block_size: block_size as u32,
        pairs: pairs as u32,
        start: 0,
        slots: slots as u32,
        reserved: 0,
    };

    let src_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.dq_src", &bytes_to_words(src));
    let scales_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.dq_scales", scales);
    let out_words = out.len() / 2;
    let out_buf = dispatch::storage_zeroed(ctx, "kv_fp8_paged.dq_out", (out_words * 4) as u64);
    let table_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.dq_table", block_table);
    let params_buf = dispatch::uniform_from(ctx, "kv_fp8_paged.dq_params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_dequantize_kv_fp8_paged",
        &compose(WGSL),
        DEQUANTIZE_ENTRY,
        &[
            (6, &src_buf),
            (7, &scales_buf),
            (8, &out_buf),
            (9, &table_buf),
            (10, &params_buf),
        ],
        groups,
    )?;

    let got: Vec<u32> = dispatch::read_back(ctx, &out_buf, out_words)?;
    unpack_bf16(&got, out);
    Ok(())
}

fn copy_one(
    ctx: &WgpuContext,
    label: &str,
    fp8: &mut [u8],
    scales: &mut [f32],
    params: &KvCopyParams,
    groups: (u32, u32, u32),
) -> Result<()> {
    let words = bytes_to_words(fp8);
    let fp8_buf = dispatch::storage_from_slice(ctx, label, &words);
    let scales_buf = dispatch::storage_from_slice(ctx, label, scales);
    let params_buf = dispatch::uniform_from(ctx, label, params);
    dispatch::run(
        ctx,
        "nv_kernels_copy_kv_block_fp8",
        &compose(WGSL),
        COPY_ENTRY,
        &[(11, &fp8_buf), (12, &scales_buf), (13, &params_buf)],
        groups,
    )?;
    let got: Vec<u32> = dispatch::read_back(ctx, &fp8_buf, words.len())?;
    words_to_bytes(&got, fp8);
    let got_scales: Vec<f32> = dispatch::read_back(ctx, &scales_buf, scales.len())?;
    scales.copy_from_slice(&got_scales);
    Ok(())
}

pub fn copy_kv_block_fp8(
    ctx: &WgpuContext,
    k_fp8: &mut [u8],
    v_fp8: &mut [u8],
    k_scales: &mut [f32],
    v_scales: &mut [f32],
    src_block: usize,
    dst_block: usize,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<()> {
    if block_size == 0 || n_kv == 0 || head_dim == 0 {
        return Ok(());
    }
    if src_block == dst_block {
        return Ok(());
    }
    if !head_dim.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "kv_fp8_paged copy needs head_dim divisible by 4 so fp8 bytes land on whole u32 words; got {head_dim}"
        )));
    }
    let k_slots = slot_count(k_fp8.len(), n_kv * head_dim, "kv_fp8_paged copy k")?;
    let v_slots = slot_count(v_fp8.len(), n_kv * head_dim, "kv_fp8_paged copy v")?;
    dispatch::check_len("kv_fp8_paged copy k_scales", k_scales.len(), k_slots * n_kv)?;
    dispatch::check_len("kv_fp8_paged copy v_scales", v_scales.len(), v_slots * n_kv)?;
    let need = (src_block.max(dst_block) + 1) * block_size;
    if need > k_slots || need > v_slots {
        return Err(WgpuError::Shape(format!(
            "kv_fp8_paged copy: blocks {src_block}->{dst_block} of size {block_size} need {need} slots; k has {k_slots}, v has {v_slots}"
        )));
    }
    check_device(ctx)?;

    let pairs = block_size * n_kv;
    let params = KvCopyParams {
        src_block: src_block as u32,
        dst_block: dst_block as u32,
        block_size: block_size as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        pairs: pairs as u32,
        reserved0: 0,
        reserved1: 0,
    };
    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    copy_one(ctx, "kv_fp8_paged.copy_k", k_fp8, k_scales, &params, groups)?;
    copy_one(ctx, "kv_fp8_paged.copy_v", v_fp8, v_scales, &params, groups)?;
    Ok(())
}

pub fn copy_kv_block_fp8_into(
    ctx: &WgpuContext,
    src_fp8: &[u8],
    src_scales: &[f32],
    dst_fp8: &mut [u8],
    dst_scales: &mut [f32],
    src_block: usize,
    dst_block: usize,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<()> {
    if block_size == 0 || n_kv == 0 || head_dim == 0 {
        return Ok(());
    }
    if src_block == dst_block {
        return Ok(());
    }
    if !head_dim.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "kv_fp8_paged copy needs head_dim divisible by 4 so fp8 bytes land on whole u32 words; got {head_dim}"
        )));
    }
    let src_slots = slot_count(src_fp8.len(), n_kv * head_dim, "kv_fp8_paged copy src")?;
    let dst_slots = slot_count(dst_fp8.len(), n_kv * head_dim, "kv_fp8_paged copy dst")?;
    dispatch::check_len(
        "kv_fp8_paged copy src_scales",
        src_scales.len(),
        src_slots * n_kv,
    )?;
    dispatch::check_len(
        "kv_fp8_paged copy dst_scales",
        dst_scales.len(),
        dst_slots * n_kv,
    )?;
    if (src_block + 1) * block_size > src_slots {
        return Err(WgpuError::Shape(format!(
            "kv_fp8_paged copy: src block {src_block} of size {block_size} needs {} slots; src has {src_slots}",
            (src_block + 1) * block_size
        )));
    }
    if (dst_block + 1) * block_size > dst_slots {
        return Err(WgpuError::Shape(format!(
            "kv_fp8_paged copy: dst block {dst_block} of size {block_size} needs {} slots; dst has {dst_slots}",
            (dst_block + 1) * block_size
        )));
    }
    check_device(ctx)?;

    let pairs = block_size * n_kv;
    let params = KvCopyParams {
        src_block: src_block as u32,
        dst_block: dst_block as u32,
        block_size: block_size as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        pairs: pairs as u32,
        reserved0: 0,
        reserved1: 0,
    };
    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);

    let src_words = bytes_to_words(src_fp8);
    let dst_words = bytes_to_words(dst_fp8);
    let src_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.x_src", &src_words);
    let src_sc_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.x_src_scales", src_scales);
    let dst_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.x_dst", &dst_words);
    let dst_sc_buf = dispatch::storage_from_slice(ctx, "kv_fp8_paged.x_dst_scales", dst_scales);
    let params_buf = dispatch::uniform_from(ctx, "kv_fp8_paged.x_params", &params);
    dispatch::run(
        ctx,
        "nv_kernels_copy_kv_block_fp8_x",
        &compose(WGSL),
        COPY_CROSS_ENTRY,
        &[
            (14, &src_buf),
            (15, &src_sc_buf),
            (16, &dst_buf),
            (17, &dst_sc_buf),
            (18, &params_buf),
        ],
        groups,
    )?;
    let got: Vec<u32> = dispatch::read_back(ctx, &dst_buf, dst_words.len())?;
    words_to_bytes(&got, dst_fp8);
    let got_scales: Vec<f32> = dispatch::read_back(ctx, &dst_sc_buf, dst_scales.len())?;
    dst_scales.copy_from_slice(&got_scales);
    Ok(())
}

pub fn cpu_quantize_kv_fp8_paged(
    x: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    block_table: &[i32],
    start: usize,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
) {
    for token in 0..n_tokens {
        let slot = paged_slot(block_table, block_size, start + token);
        for kv_head in 0..n_kv {
            let base_src = (token * n_kv + kv_head) * head_dim;
            let base_dst = (slot * n_kv + kv_head) * head_dim;
            let mut amax = 0.0f32;
            for d in 0..head_dim {
                let v = f32::from_bits((x[base_src + d] as u32) << 16);
                amax = amax.max(v.abs());
            }
            let (scale, inv) = if amax > 0.0 {
                (amax / FP8_E4M3_MAX, FP8_E4M3_MAX / amax)
            } else {
                (1.0, 1.0)
            };
            scales[slot * n_kv + kv_head] = scale;
            for d in 0..head_dim {
                let v = f32::from_bits((x[base_src + d] as u32) << 16);
                out[base_dst + d] = encode_e4m3(v * inv);
            }
        }
    }
}

pub fn cpu_dequantize_kv_fp8_paged(
    src: &[u8],
    scales: &[f32],
    block_table: &[i32],
    out: &mut [u16],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
) {
    for token in 0..n_tokens {
        let slot = paged_slot(block_table, block_size, token);
        for kv_head in 0..n_kv {
            let base = (slot * n_kv + kv_head) * head_dim;
            let obase = (token * n_kv + kv_head) * head_dim;
            let scale = scales[slot * n_kv + kv_head];
            for d in 0..head_dim {
                let v = decode_e4m3(src[base + d]) * scale;
                out[obase + d] = half::bf16::from_f32(v).to_bits();
            }
        }
    }
}

pub fn cpu_copy_kv_block_fp8(
    fp8: &mut [u8],
    scales: &mut [f32],
    src_block: usize,
    dst_block: usize,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
) {
    if src_block == dst_block {
        return;
    }
    for slot_in_block in 0..block_size {
        let src_slot = src_block * block_size + slot_in_block;
        let dst_slot = dst_block * block_size + slot_in_block;
        for kv_head in 0..n_kv {
            let src_base = (src_slot * n_kv + kv_head) * head_dim;
            let dst_base = (dst_slot * n_kv + kv_head) * head_dim;
            for d in 0..head_dim {
                fp8[dst_base + d] = fp8[src_base + d];
            }
            scales[dst_slot * n_kv + kv_head] = scales[src_slot * n_kv + kv_head];
        }
    }
}

pub fn cpu_copy_kv_block_fp8_into(
    src_fp8: &[u8],
    src_scales: &[f32],
    dst_fp8: &mut [u8],
    dst_scales: &mut [f32],
    src_block: usize,
    dst_block: usize,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
) {
    if src_block == dst_block {
        return;
    }
    for slot_in_block in 0..block_size {
        let src_slot = src_block * block_size + slot_in_block;
        let dst_slot = dst_block * block_size + slot_in_block;
        for kv_head in 0..n_kv {
            let src_base = (src_slot * n_kv + kv_head) * head_dim;
            let dst_base = (dst_slot * n_kv + kv_head) * head_dim;
            dst_fp8[dst_base..dst_base + head_dim]
                .copy_from_slice(&src_fp8[src_base..src_base + head_dim]);
            dst_scales[dst_slot * n_kv + kv_head] = src_scales[src_slot * n_kv + kv_head];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_slot_maps_through_the_table() {
        let table = [4i32, 0, 2];
        assert_eq!(paged_slot(&table, 8, 0), 32);
        assert_eq!(paged_slot(&table, 8, 7), 39);
        assert_eq!(paged_slot(&table, 8, 8), 0);
        assert_eq!(paged_slot(&table, 8, 15), 7);
        assert_eq!(paged_slot(&table, 8, 16), 16);
    }

    #[test]
    fn table_bounds_are_checked() {
        let table = [1i32, 0];
        assert!(check_table("t", &table, 4, 0, 8, 8).is_ok());
        assert!(check_table("t", &table, 4, 0, 9, 8).is_err());
        assert!(check_table("t", &table, 4, 0, 8, 7).is_err());
        assert!(check_table("t", &[-1i32], 4, 0, 4, 8).is_err());
        assert!(check_table("t", &table, 0, 0, 4, 8).is_err());
        assert!(check_table("t", &[-1i32, 0], 4, 4, 4, 8).is_ok());
        assert!(check_table("t", &[-1i32, 0], 4, 3, 4, 8).is_err());
    }

    #[test]
    fn slot_count_rejects_ragged_lengths() {
        assert!(slot_count(10, 4, "x").is_err());
        assert_eq!(slot_count(12, 4, "x").unwrap(), 3);
    }
}
