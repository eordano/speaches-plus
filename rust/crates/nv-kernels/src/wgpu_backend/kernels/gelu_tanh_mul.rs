#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/gelu_tanh_mul.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

pub const ENTRY_SPLIT: &str = "gelu_tanh_mul_split";
pub const ENTRY_FUSED_EVEN: &str = "gelu_tanh_mul_fused_even";
pub const ENTRY_FUSED_GENERAL: &str = "gelu_tanh_mul_fused_general";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeluParams {
    pub inter: u32,
    pub inter_words: u32,
    pub rows: u32,
    pub tot_pairs: u32,
}

pub fn source() -> String {
    compose(WGSL)
}

pub fn u16s_to_words(v: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; v.len().div_ceil(2).max(1)];
    for (i, x) in v.iter().enumerate() {
        out[i / 2] |= (*x as u32) << (16 * (i % 2));
    }
    out
}

pub fn words_to_u16s(words: &[u32], dst: &mut [u16]) {
    for (i, d) in dst.iter_mut().enumerate() {
        *d = ((words[i / 2] >> (16 * (i % 2))) & 0xffff) as u16;
    }
}

fn groups_x(ctx: &WgpuContext, invocations: usize) -> u32 {
    let limit = ctx.caps.max_compute_workgroups_per_dimension.max(1) as u64;
    let want = (invocations as u64).div_ceil(WORKGROUP_SIZE as u64).max(1);
    want.min(limit) as u32
}

pub fn gelu_tanh_mul_bf16(
    ctx: &WgpuContext,
    gate: &[u16],
    up: &[u16],
    y: &mut [u16],
    n: usize,
) -> Result<()> {
    dispatch::check_len("gelu_tanh_mul gate", gate.len(), n)?;
    dispatch::check_len("gelu_tanh_mul up", up.len(), n)?;
    dispatch::check_len("gelu_tanh_mul y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let n_words = n.div_ceil(2);
    let gate_words = u16s_to_words(gate);
    let up_words = u16s_to_words(up);
    let gate_buf = dispatch::storage_from_slice(ctx, "gelu-gate", &gate_words);
    let up_buf = dispatch::storage_from_slice(ctx, "gelu-up", &up_words);
    let y_buf = dispatch::storage_zeroed(ctx, "gelu-y", (n_words * 4) as u64);
    dispatch::run(
        ctx,
        "gelu_tanh_mul_bf16",
        &source(),
        ENTRY_SPLIT,
        &[(0, &gate_buf), (1, &up_buf), (2, &y_buf)],
        (groups_x(ctx, n_words), 1, 1),
    )?;
    let out = dispatch::read_back::<u32>(ctx, &y_buf, n_words)?;
    words_to_u16s(&out, y);
    Ok(())
}

pub fn gelu_tanh_mul_fused_bf16(
    ctx: &WgpuContext,
    fused: &[u16],
    y: &mut [u16],
    inter: usize,
    tot_pairs: usize,
) -> Result<()> {
    dispatch::check_len("gelu_tanh_mul_fused y", y.len(), tot_pairs)?;
    if tot_pairs == 0 {
        return Ok(());
    }
    if inter == 0 {
        return Err(WgpuError::Shape("gelu_tanh_mul_fused: inter is 0".into()));
    }
    let rows = tot_pairs.div_ceil(inter);
    dispatch::check_len("gelu_tanh_mul_fused fused", fused.len(), rows * 2 * inter)?;
    if u32::try_from(rows * 2 * inter).is_err() {
        return Err(WgpuError::Shape(format!(
            "gelu_tanh_mul_fused: {} elements exceed u32 indexing",
            rows * 2 * inter
        )));
    }

    let out_words = tot_pairs.div_ceil(2);
    let src_words = u16s_to_words(fused);
    let src_buf = dispatch::storage_from_slice(ctx, "gelu-fused-src", &src_words);
    let y_buf = dispatch::storage_zeroed(ctx, "gelu-fused-y", (out_words * 4) as u64);
    let params = GeluParams {
        inter: inter as u32,
        inter_words: (inter / 2) as u32,
        rows: rows as u32,
        tot_pairs: tot_pairs as u32,
    };
    let params_buf = dispatch::uniform_from(ctx, "gelu-fused-params", &params);
    let bindings = [(3, &src_buf), (4, &y_buf), (5, &params_buf)];

    let word_aligned = inter.is_multiple_of(2) && tot_pairs == rows * inter;
    let (entry, groups) = if word_aligned {
        let limit = ctx.caps.max_compute_workgroups_per_dimension.max(1);
        let gy = (rows as u64).min(limit as u64) as u32;
        (ENTRY_FUSED_EVEN, (groups_x(ctx, inter / 2), gy.max(1), 1))
    } else {
        (ENTRY_FUSED_GENERAL, (groups_x(ctx, out_words), 1, 1))
    };

    dispatch::run(
        ctx,
        "gelu_tanh_mul_fused_bf16",
        &source(),
        entry,
        &bindings,
        groups,
    )?;
    let out = dispatch::read_back::<u32>(ctx, &y_buf, out_words)?;
    words_to_u16s(&out, y);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_packing_round_trips() {
        let v: Vec<u16> = (0..7u16).map(|i| i.wrapping_mul(4097)).collect();
        let w = u16s_to_words(&v);
        assert_eq!(w.len(), 4);
        let mut back = vec![0u16; v.len()];
        words_to_u16s(&w, &mut back);
        assert_eq!(back, v);
    }

    #[test]
    fn wgsl_declares_the_entry_points() {
        for e in [ENTRY_SPLIT, ENTRY_FUSED_EVEN, ENTRY_FUSED_GENERAL] {
            assert!(WGSL.contains(&format!("fn {e}(")), "missing entry {e}");
        }
        assert!(WGSL.contains("0.7978845608028654"));
        assert!(WGSL.contains("0.044715"));
    }
}
