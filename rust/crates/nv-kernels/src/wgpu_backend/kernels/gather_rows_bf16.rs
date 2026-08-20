#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_words, unpack_u16_pairs_clamped as unpack_words};

pub const WGSL: &str = include_str!("../../../wgsl/gather_rows_bf16.wgsl");

pub const ENTRY: &str = "gather_rows_bf16";
pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    m_total_padded: u32,
    hidden_words: u32,
    n_tokens: i32,
    pad0: u32,
}

pub fn gather_rows_bf16(
    ctx: &WgpuContext,
    x_bf16: &[u16],
    src_idx: &[i32],
    out_bf16: &mut [u16],
    m_total_padded: usize,
    hidden: usize,
    n_tokens: usize,
) -> Result<()> {
    dispatch::check_len("x_bf16", x_bf16.len(), n_tokens * hidden)?;
    dispatch::check_len("src_idx", src_idx.len(), m_total_padded)?;
    dispatch::check_len("out_bf16", out_bf16.len(), m_total_padded * hidden)?;
    if m_total_padded == 0 || hidden == 0 {
        return Ok(());
    }
    if !hidden.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "gather_rows_bf16: hidden must be even for u32 word copy, got {hidden}"
        )));
    }
    if n_tokens > i32::MAX as usize {
        return Err(WgpuError::Shape(format!(
            "gather_rows_bf16: n_tokens {n_tokens} exceeds i32 range"
        )));
    }

    let hidden_words = hidden / 2;
    let out_words_len = m_total_padded * hidden_words;

    let mut x_words = pack_words(x_bf16);
    if x_words.is_empty() {
        x_words.push(0);
    }

    let x_buf = dispatch::storage_from_slice(ctx, "gather_rows_bf16.x", &x_words);
    let idx_buf = dispatch::storage_from_slice(ctx, "gather_rows_bf16.src_idx", src_idx);
    let out_buf = dispatch::storage_zeroed(
        ctx,
        "gather_rows_bf16.out",
        (out_words_len * std::mem::size_of::<u32>()) as u64,
    );
    let params = Params {
        m_total_padded: m_total_padded as u32,
        hidden_words: hidden_words as u32,
        n_tokens: n_tokens as i32,
        pad0: 0,
    };
    let params_buf = dispatch::uniform_from(ctx, "gather_rows_bf16.params", &params);

    let workgroups = dispatch::workgroup_count_1d(ctx, m_total_padded as u64, 1);
    dispatch::run(
        ctx,
        "gather_rows_bf16",
        WGSL,
        ENTRY,
        &[(0, &x_buf), (1, &idx_buf), (2, &out_buf), (3, &params_buf)],
        workgroups,
    )?;

    let words = dispatch::read_back::<u32>(ctx, &out_buf, out_words_len)?;
    unpack_words(&words, out_bf16);
    Ok(())
}
