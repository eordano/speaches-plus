#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16, unpack_u16_pairs as unpack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/fused_norm_chain.wgsl");

pub const ENTRY_RMS_RES_RMS: &str = "e4b_rms_res_rms_bf16";
pub const ENTRY_RES_OF_RMS: &str = "e4b_res_of_rms_bf16";
pub const ENTRY_RMS_RES_RMS_NEXT: &str = "e4b_rms_res_rms_next_bf16";

pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FncParams {
    pub hidden: u32,
    pub batch: u32,
    pub eps: f32,
    pub words_per_row: u32,
    pub scale: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

fn check(batch: usize, hidden: usize, lens: &[(&str, usize, usize)]) -> Result<()> {
    if !hidden.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "fused_norm_chain hidden must be even; got {hidden}"
        )));
    }
    let _ = batch;
    for (name, got, want) in lens {
        dispatch::check_len(name, *got, *want)?;
    }
    Ok(())
}

pub fn rms_res_rms_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    residual: &mut [u16],
    w1: &[u16],
    w2: &[u16],
    out: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    check(
        batch,
        hidden,
        &[
            ("fnc a x", x.len(), batch * hidden),
            ("fnc a residual", residual.len(), batch * hidden),
            ("fnc a w1", w1.len(), hidden),
            ("fnc a w2", w2.len(), hidden),
            ("fnc a out", out.len(), batch * hidden),
        ],
    )?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    let words = hidden / 2;
    let p = FncParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: words as u32,
        scale: 1.0,
        ..Default::default()
    };
    let xb = dispatch::storage_from_slice(ctx, "fnc-a-x", &pack_u16(x));
    let rb = dispatch::storage_from_slice(ctx, "fnc-a-res", &pack_u16(residual));
    let w1b = dispatch::storage_from_slice(ctx, "fnc-a-w1", &pack_u16(w1));
    let w2b = dispatch::storage_from_slice(ctx, "fnc-a-w2", &pack_u16(w2));
    let ob = dispatch::storage_zeroed(ctx, "fnc-a-out", (batch * words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "fnc-a-params", &p);
    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_fnc_rms_res_rms_bf16",
        &compose(WGSL),
        ENTRY_RMS_RES_RMS,
        &[(0, &xb), (1, &rb), (2, &w1b), (3, &w2b), (4, &ob), (5, &pb)],
        groups,
    )?;
    let new_res: Vec<u32> = dispatch::read_back(ctx, &rb, batch * words)?;
    unpack_u16(&new_res, residual);
    let y: Vec<u32> = dispatch::read_back(ctx, &ob, batch * words)?;
    unpack_u16(&y, out);
    Ok(())
}

pub fn res_of_rms_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    residual: &[u16],
    w: &[u16],
    out: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
    scale: f32,
) -> Result<()> {
    check(
        batch,
        hidden,
        &[
            ("fnc b x", x.len(), batch * hidden),
            ("fnc b residual", residual.len(), batch * hidden),
            ("fnc b w", w.len(), hidden),
            ("fnc b out", out.len(), batch * hidden),
        ],
    )?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    let words = hidden / 2;
    let p = FncParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: words as u32,
        scale,
        ..Default::default()
    };
    let xb = dispatch::storage_from_slice(ctx, "fnc-b-x", &pack_u16(x));
    let rb = dispatch::storage_from_slice(ctx, "fnc-b-res", &pack_u16(residual));
    let wb = dispatch::storage_from_slice(ctx, "fnc-b-w", &pack_u16(w));
    let ob = dispatch::storage_zeroed(ctx, "fnc-b-out", (batch * words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "fnc-b-params", &p);
    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_fnc_res_of_rms_bf16",
        &compose(WGSL),
        ENTRY_RES_OF_RMS,
        &[(0, &xb), (1, &rb), (2, &wb), (4, &ob), (5, &pb)],
        groups,
    )?;
    let y: Vec<u32> = dispatch::read_back(ctx, &ob, batch * words)?;
    unpack_u16(&y, out);
    Ok(())
}

pub fn rms_res_rms_next_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    residual: &[u16],
    w1: &[u16],
    w2: &[u16],
    out: &mut [u16],
    out2: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
    scale: f32,
) -> Result<()> {
    check(
        batch,
        hidden,
        &[
            ("fnc c x", x.len(), batch * hidden),
            ("fnc c residual", residual.len(), batch * hidden),
            ("fnc c w1", w1.len(), hidden),
            ("fnc c w2", w2.len(), hidden),
            ("fnc c out", out.len(), batch * hidden),
            ("fnc c out2", out2.len(), batch * hidden),
        ],
    )?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    let words = hidden / 2;
    let p = FncParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: words as u32,
        scale,
        ..Default::default()
    };
    let xb = dispatch::storage_from_slice(ctx, "fnc-c-x", &pack_u16(x));
    let rb = dispatch::storage_from_slice(ctx, "fnc-c-res", &pack_u16(residual));
    let w1b = dispatch::storage_from_slice(ctx, "fnc-c-w1", &pack_u16(w1));
    let w2b = dispatch::storage_from_slice(ctx, "fnc-c-w2", &pack_u16(w2));
    let ob = dispatch::storage_zeroed(ctx, "fnc-c-out", (batch * words * 4) as u64);
    let o2b = dispatch::storage_zeroed(ctx, "fnc-c-out2", (batch * words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "fnc-c-params", &p);
    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_fnc_rms_res_rms_next_bf16",
        &compose(WGSL),
        ENTRY_RMS_RES_RMS_NEXT,
        &[
            (0, &xb),
            (1, &rb),
            (2, &w1b),
            (3, &w2b),
            (4, &ob),
            (5, &pb),
            (6, &o2b),
        ],
        groups,
    )?;
    let y: Vec<u32> = dispatch::read_back(ctx, &ob, batch * words)?;
    unpack_u16(&y, out);
    let y2: Vec<u32> = dispatch::read_back(ctx, &o2b, batch * words)?;
    unpack_u16(&y2, out2);
    Ok(())
}
