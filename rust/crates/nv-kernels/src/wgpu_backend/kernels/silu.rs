#![allow(clippy::too_many_arguments)]

use std::sync::OnceLock;

use crate::wgpu_backend::dequant;
use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{Result, WgpuError};
pub use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_bf16_words, unpack_u16_first_n as unpack_bf16_words};

pub const WGSL: &str = include_str!("../../../wgsl/silu.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

static SOURCE: OnceLock<String> = OnceLock::new();

fn source() -> &'static str {
    SOURCE.get_or_init(|| dequant::compose(WGSL))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn params_for(n: usize) -> Result<SiluParams> {
    if n > u32::MAX as usize {
        return Err(WgpuError::Shape(format!(
            "silu: n {n} exceeds the u32 element index range"
        )));
    }
    Ok(SiluParams {
        n: n as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    })
}

pub fn bf16_dst_words(n: usize) -> usize {
    n.div_ceil(2)
}

pub fn workgroups_f32(ctx: &WgpuContext, n: usize) -> (u32, u32, u32) {
    dispatch::workgroup_count_1d(ctx, n as u64, WORKGROUP_SIZE)
}

pub fn workgroups_bf16(ctx: &WgpuContext, n: usize) -> (u32, u32, u32) {
    dispatch::workgroup_count_1d(ctx, bf16_dst_words(n) as u64, WORKGROUP_SIZE)
}

fn run_f32(
    ctx: &WgpuContext,
    label: &str,
    entry: &str,
    x: &[f32],
    gate: Option<&[f32]>,
    y: &mut [f32],
    n: usize,
) -> Result<()> {
    dispatch::check_len("silu x", x.len(), n)?;
    dispatch::check_len("silu y", y.len(), n)?;
    if let Some(g) = gate {
        dispatch::check_len("silu gate", g.len(), n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let params = params_for(n)?;
    let src = dispatch::storage_from_slice(ctx, "silu-src-f32", x);
    let gate_buf = gate.map(|g| dispatch::storage_from_slice(ctx, "silu-gate-f32", g));
    let dst = dispatch::storage_zeroed(ctx, "silu-dst-f32", (n * 4) as u64);
    let params_buf = dispatch::uniform_from(ctx, "silu-params", &params);

    let mut bindings: Vec<(u32, &wgpu::Buffer)> = vec![(0, &src), (2, &dst), (3, &params_buf)];
    if let Some(g) = gate_buf.as_ref() {
        bindings.push((1, g));
    }
    let groups = workgroups_f32(ctx, n);
    dispatch::run(ctx, label, source(), entry, &bindings, groups)?;
    let out: Vec<f32> = dispatch::read_back(ctx, &dst, n)?;
    y.copy_from_slice(&out);
    Ok(())
}

fn run_bf16(
    ctx: &WgpuContext,
    label: &str,
    entry: &str,
    x: &[u16],
    gate: Option<&[u16]>,
    y: &mut [u16],
    n: usize,
) -> Result<()> {
    dispatch::check_len("silu x", x.len(), n)?;
    dispatch::check_len("silu y", y.len(), n)?;
    if let Some(g) = gate {
        dispatch::check_len("silu gate", g.len(), n)?;
    }
    if n == 0 {
        return Ok(());
    }
    let params = params_for(n)?;
    let words = bf16_dst_words(n);
    let src_words = pack_bf16_words(x);
    let src = dispatch::storage_from_slice(ctx, "silu-src-bf16", &src_words);
    let gate_words = gate.map(pack_bf16_words);
    let gate_buf = gate_words
        .as_ref()
        .map(|g| dispatch::storage_from_slice(ctx, "silu-gate-bf16", g));
    let dst = dispatch::storage_zeroed(ctx, "silu-dst-bf16", (words * 4) as u64);
    let params_buf = dispatch::uniform_from(ctx, "silu-params", &params);

    let mut bindings: Vec<(u32, &wgpu::Buffer)> = vec![(4, &src), (6, &dst), (3, &params_buf)];
    if let Some(g) = gate_buf.as_ref() {
        bindings.push((5, g));
    }
    let groups = workgroups_bf16(ctx, n);
    dispatch::run(ctx, label, source(), entry, &bindings, groups)?;
    let out: Vec<u32> = dispatch::read_back(ctx, &dst, words)?;
    unpack_bf16_words(&out, n, y);
    Ok(())
}

pub fn silu_f32(ctx: &WgpuContext, x: &[f32], y: &mut [f32], n: usize) -> Result<()> {
    run_f32(ctx, "silu-f32", "silu_f32", x, None, y, n)
}

pub fn silu_bf16(ctx: &WgpuContext, x: &[u16], y: &mut [u16], n: usize) -> Result<()> {
    run_bf16(ctx, "silu-bf16", "silu_bf16", x, None, y, n)
}

pub fn silu_mul_f32(
    ctx: &WgpuContext,
    x: &[f32],
    gate: &[f32],
    y: &mut [f32],
    n: usize,
) -> Result<()> {
    run_f32(ctx, "silu-mul-f32", "silu_mul_f32", x, Some(gate), y, n)
}

pub fn silu_mul_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    gate: &[u16],
    y: &mut [u16],
    n: usize,
) -> Result<()> {
    run_bf16(ctx, "silu-mul-bf16", "silu_mul_bf16", x, Some(gate), y, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_word_packing_round_trips() {
        let values: Vec<u16> = vec![0x3f80, 0xbf80, 0x0000, 0x7f80, 0x4049];
        let words = pack_bf16_words(&values);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0], 0xbf80_3f80);
        assert_eq!(words[2] & 0xffff_0000, 0);
        let mut back = vec![0u16; values.len()];
        unpack_bf16_words(&words, values.len(), &mut back);
        assert_eq!(back, values);
    }

    #[test]
    fn shader_declares_every_entry_point() {
        for entry in [
            "fn silu_f32(",
            "fn silu_bf16(",
            "fn silu_mul_f32(",
            "fn silu_mul_bf16(",
        ] {
            assert!(WGSL.contains(entry), "missing {entry}");
        }
        assert!(WGSL.contains("clamp(-x, -SILU_EXP_LIMIT, SILU_EXP_LIMIT)"));
    }

    #[test]
    fn params_reject_indices_past_u32() {
        assert!(params_for(1 << 20).is_ok());
        if (u32::MAX as usize) < usize::MAX {
            assert!(params_for(u32::MAX as usize + 1).is_err());
        }
    }
}
