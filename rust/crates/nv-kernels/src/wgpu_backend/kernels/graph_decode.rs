#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::kernels::attn_decode;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/graph_decode.wgsl");

pub const WORKGROUP_SIZE: u32 = 128;
pub const ARGMAX_BLOCKS: usize = 256;
pub const MAX_PREP_HEAD_DIM: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct EwParams {
    n: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvParams {
    nkv: u32,
    hd: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    rows: u32,
    dim: u32,
    eps: f32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ApplyParams {
    n: u32,
    dim: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RasParams {
    rows: u32,
    dim: u32,
    eps: f32,
    scale: f32,
    eps_next: f32,
    flags: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QkvParams {
    nh: u32,
    nkv: u32,
    hd: u32,
    delta: i32,
    eps: f32,
    has_kv: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxParams {
    n: u32,
    nparts: u32,
    ring_mask: i32,
    has_ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxRowsParams {
    rows: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ZeroParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn check_device(ctx: &WgpuContext, need: u32) -> Result<()> {
    dispatch::require_workgroup(ctx, "graph_decode", need)
}

fn widen(src: &[u16]) -> Vec<u32> {
    src.iter().map(|v| *v as u32).collect()
}

fn narrow(words: &[u32], dst: &mut [u16]) {
    for (d, w) in dst.iter_mut().zip(words.iter()) {
        *d = (*w & 0xffff) as u16;
    }
}

fn shader() -> String {
    compose(WGSL)
}

fn dummy(ctx: &WgpuContext, label: &str) -> wgpu::Buffer {
    dispatch::storage_zeroed(ctx, label, 4)
}

pub fn incr_pos(ctx: &WgpuContext, pos: &mut [i32]) -> Result<()> {
    if pos.is_empty() {
        return Err(WgpuError::Shape("incr_pos: pos is empty".into()));
    }
    let pb = dispatch::storage_from_slice(ctx, "gd-incr-pos", pos);
    dispatch::run(
        ctx,
        "nv_kernels_incr_pos",
        &shader(),
        "incr_pos",
        &[(11, &pb)],
        (1, 1, 1),
    )?;
    let got: Vec<i32> = dispatch::read_back(ctx, &pb, pos.len())?;
    pos.copy_from_slice(&got);
    Ok(())
}

pub fn incr_pos_rope(ctx: &WgpuContext, pos: &mut [i32], rope_pos: &mut [i32]) -> Result<()> {
    if pos.is_empty() || rope_pos.is_empty() {
        return Err(WgpuError::Shape("incr_pos_rope: empty buffer".into()));
    }
    let pb = dispatch::storage_from_slice(ctx, "gd-incr-pos-rope-pos", pos);
    let rb = dispatch::storage_from_slice(ctx, "gd-incr-pos-rope-out", rope_pos);
    dispatch::run(
        ctx,
        "nv_kernels_incr_pos_rope",
        &shader(),
        "incr_pos_rope",
        &[(11, &pb), (12, &rb)],
        (1, 1, 1),
    )?;
    let gp: Vec<i32> = dispatch::read_back(ctx, &pb, pos.len())?;
    let gr: Vec<i32> = dispatch::read_back(ctx, &rb, rope_pos.len())?;
    pos.copy_from_slice(&gp);
    rope_pos.copy_from_slice(&gr);
    Ok(())
}

pub fn write_kv_f32(
    ctx: &WgpuContext,
    src_k: &[f32],
    src_v: &[f32],
    cache_k: &mut [f32],
    cache_v: &mut [f32],
    pos: &[i32],
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("write_kv_f32 src_k", src_k.len(), n_kv_heads * head_dim)?;
    dispatch::check_len("write_kv_f32 src_v", src_v.len(), n_kv_heads * head_dim)?;
    dispatch::check_len("write_kv_f32 cache", cache_v.len(), cache_k.len())?;
    if pos.is_empty() {
        return Err(WgpuError::Shape("write_kv_f32: pos is empty".into()));
    }
    check_device(ctx, WORKGROUP_SIZE)?;

    let params = KvParams {
        nkv: n_kv_heads as u32,
        hd: head_dim as u32,
        pad0: 0,
        pad1: 0,
    };
    let skb = dispatch::storage_from_slice(ctx, "gd-wkv-sk", src_k);
    let svb = dispatch::storage_from_slice(ctx, "gd-wkv-sv", src_v);
    let ckb = dispatch::storage_from_slice(ctx, "gd-wkv-ck", cache_k);
    let cvb = dispatch::storage_from_slice(ctx, "gd-wkv-cv", cache_v);
    let pb = dispatch::storage_from_slice(ctx, "gd-wkv-pos", pos);
    let ub = dispatch::uniform_from(ctx, "gd-wkv-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, n_kv_heads as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_write_kv_f32",
        &shader(),
        "write_kv_f32",
        &[
            (13, &skb),
            (14, &svb),
            (15, &ckb),
            (16, &cvb),
            (17, &pb),
            (18, &ub),
        ],
        groups,
    )?;
    let gk: Vec<f32> = dispatch::read_back(ctx, &ckb, cache_k.len())?;
    let gv: Vec<f32> = dispatch::read_back(ctx, &cvb, cache_v.len())?;
    cache_k.copy_from_slice(&gk);
    cache_v.copy_from_slice(&gv);
    Ok(())
}

pub fn attn_decode_dev_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    pos: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> Result<()> {
    if pos.is_empty() {
        return Err(WgpuError::Shape("attn_decode_dev_f32: pos is empty".into()));
    }
    if n_heads == 0 || n_kv_heads == 0 {
        return Ok(());
    }
    let total = pos[0].max(0) as usize;
    let start = if window > 0 && total > window {
        total - window
    } else {
        0
    };
    for o in out.iter_mut() {
        *o = 0.0;
    }
    if total == 0 {
        return Ok(());
    }
    let span = total * n_kv_heads * head_dim;
    if k_cache.len() < span || v_cache.len() < span {
        return Err(WgpuError::Shape(format!(
            "attn_decode_dev_f32 cache: got {} want at least {span}",
            k_cache.len().min(v_cache.len())
        )));
    }
    attn_decode::attn_decode_f32(
        ctx,
        q,
        &k_cache[..span],
        &v_cache[..span],
        out,
        n_heads,
        n_kv_heads,
        head_dim,
        start,
        total,
        1.0,
    )
}

fn elementwise_u32(
    ctx: &WgpuContext,
    label: &'static str,
    entry: &str,
    input: &[u32],
    n: usize,
    scale: f32,
) -> Result<Vec<u32>> {
    check_device(ctx, WORKGROUP_SIZE)?;
    let params = EwParams {
        n: n as u32,
        scale,
        pad0: 0,
        pad1: 0,
    };
    let ib = dispatch::storage_from_slice(ctx, "gd-ew-in", input);
    let ob = dispatch::storage_zeroed(ctx, "gd-ew-out", (n * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-ew-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        label,
        &shader(),
        entry,
        &[(0, &ib), (1, &ob), (2, &ub)],
        groups,
    )?;
    dispatch::read_back(ctx, &ob, n)
}

pub fn cast_bf16_f32(ctx: &WgpuContext, x: &[u16], y: &mut [f32], n: usize) -> Result<()> {
    dispatch::check_len("cast_bf16_f32 x", x.len(), n)?;
    dispatch::check_len("cast_bf16_f32 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let got = elementwise_u32(
        ctx,
        "nv_kernels_cast_bf16_f32",
        "cast_bf16_f32",
        &widen(x),
        n,
        0.0,
    )?;
    for (d, w) in y.iter_mut().zip(got.iter()) {
        *d = f32::from_bits(*w);
    }
    Ok(())
}

pub fn cast_f32_bf16(ctx: &WgpuContext, x: &[f32], y: &mut [u16], n: usize) -> Result<()> {
    dispatch::check_len("cast_f32_bf16 x", x.len(), n)?;
    dispatch::check_len("cast_f32_bf16 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let bits: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
    let got = elementwise_u32(
        ctx,
        "nv_kernels_cast_f32_bf16",
        "cast_f32_bf16",
        &bits,
        n,
        0.0,
    )?;
    narrow(&got, y);
    Ok(())
}

pub fn cast_scale_bf16_f32(
    ctx: &WgpuContext,
    x: &[u16],
    y: &mut [f32],
    scale: f32,
    n: usize,
) -> Result<()> {
    dispatch::check_len("cast_scale_bf16_f32 x", x.len(), n)?;
    dispatch::check_len("cast_scale_bf16_f32 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let got = elementwise_u32(
        ctx,
        "nv_kernels_cast_scale_bf16_f32",
        "cast_scale_bf16_f32",
        &widen(x),
        n,
        scale,
    )?;
    for (d, w) in y.iter_mut().zip(got.iter()) {
        *d = f32::from_bits(*w);
    }
    Ok(())
}

pub fn add_scale_f32(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    y: &mut [f32],
    scale: f32,
    n: usize,
) -> Result<()> {
    dispatch::check_len("add_scale_f32 a", a.len(), n)?;
    dispatch::check_len("add_scale_f32 b", b.len(), n)?;
    dispatch::check_len("add_scale_f32 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    check_device(ctx, WORKGROUP_SIZE)?;
    let params = EwParams {
        n: n as u32,
        scale,
        pad0: 0,
        pad1: 0,
    };
    let ab = dispatch::storage_from_slice(ctx, "gd-add-a", a);
    let bb = dispatch::storage_from_slice(ctx, "gd-add-b", b);
    let ob = dispatch::storage_zeroed(ctx, "gd-add-y", (n * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-add-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_add_scale_f32",
        &shader(),
        "add_scale_f32",
        &[(3, &ab), (4, &bb), (5, &ob), (6, &ub)],
        groups,
    )?;
    let got: Vec<f32> = dispatch::read_back(ctx, &ob, n)?;
    y.copy_from_slice(&got);
    Ok(())
}

pub fn gelu_mul_bf16f32(
    ctx: &WgpuContext,
    gate: &[u16],
    pli: &[f32],
    y: &mut [u16],
    n: usize,
) -> Result<()> {
    dispatch::check_len("gelu_mul_bf16f32 gate", gate.len(), n)?;
    dispatch::check_len("gelu_mul_bf16f32 pli", pli.len(), n)?;
    dispatch::check_len("gelu_mul_bf16f32 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    check_device(ctx, WORKGROUP_SIZE)?;
    let params = EwParams {
        n: n as u32,
        scale: 0.0,
        pad0: 0,
        pad1: 0,
    };
    let gb = dispatch::storage_from_slice(ctx, "gd-gelu-gate", &widen(gate));
    let pb = dispatch::storage_from_slice(ctx, "gd-gelu-pli", pli);
    let ob = dispatch::storage_zeroed(ctx, "gd-gelu-y", (n * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-gelu-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_gelu_mul_bf16f32",
        &shader(),
        "gelu_mul_bf16f32",
        &[(7, &gb), (8, &pb), (9, &ob), (10, &ub)],
        groups,
    )?;
    let got: Vec<u32> = dispatch::read_back(ctx, &ob, n)?;
    narrow(&got, y);
    Ok(())
}

fn rms_family(
    ctx: &WgpuContext,
    label: &'static str,
    entry: &str,
    x: &[u16],
    w: Option<&[u16]>,
    out_len: usize,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    check_device(ctx, WORKGROUP_SIZE)?;
    let params = RmsParams {
        rows: rows as u32,
        dim: dim as u32,
        eps,
        pad0: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "gd-rms-x", &widen(x));
    let wb = match w {
        Some(w) => dispatch::storage_from_slice(ctx, "gd-rms-w", &widen(w)),
        None => dummy(ctx, "gd-rms-w-dummy"),
    };
    let ob = dispatch::storage_zeroed(ctx, "gd-rms-y", (out_len * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-rms-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, rows as u64, 1);
    let bindings: Vec<(u32, &wgpu::Buffer)> = if w.is_some() {
        vec![(19, &xb), (20, &wb), (21, &ob), (22, &ub)]
    } else {
        vec![(19, &xb), (21, &ob), (22, &ub)]
    };
    dispatch::run(ctx, label, &shader(), entry, &bindings, groups)?;
    dispatch::read_back(ctx, &ob, out_len)
}

pub fn rms_no_weight_bf16_f32(
    ctx: &WgpuContext,
    x: &[u16],
    y: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    dispatch::check_len("rms_no_weight_bf16_f32 x", x.len(), rows * dim)?;
    dispatch::check_len("rms_no_weight_bf16_f32 y", y.len(), rows * dim)?;
    if rows == 0 || dim == 0 {
        return Ok(());
    }
    let got = rms_family(
        ctx,
        "nv_kernels_rms_no_weight_bf16_f32",
        "rms_no_weight_bf16_f32",
        x,
        None,
        rows * dim,
        rows,
        dim,
        eps,
    )?;
    y.copy_from_slice(&got);
    Ok(())
}

pub fn rmsnorm_bf16w_f32out(
    ctx: &WgpuContext,
    x: &[u16],
    weight: &[u16],
    y: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    dispatch::check_len("rmsnorm_bf16w_f32out x", x.len(), rows * dim)?;
    dispatch::check_len("rmsnorm_bf16w_f32out weight", weight.len(), dim)?;
    dispatch::check_len("rmsnorm_bf16w_f32out y", y.len(), rows * dim)?;
    if rows == 0 || dim == 0 {
        return Ok(());
    }
    let got = rms_family(
        ctx,
        "nv_kernels_rmsnorm_bf16w_f32out",
        "rmsnorm_bf16w_f32out",
        x,
        Some(weight),
        rows * dim,
        rows,
        dim,
        eps,
    )?;
    y.copy_from_slice(&got);
    Ok(())
}

pub fn rstd_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    rstd: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    dispatch::check_len("rstd_bf16 x", x.len(), rows * dim)?;
    dispatch::check_len("rstd_bf16 rstd", rstd.len(), rows)?;
    if rows == 0 || dim == 0 {
        return Ok(());
    }
    let got = rms_family(
        ctx,
        "nv_kernels_rstd_bf16",
        "rstd_bf16",
        x,
        None,
        rows,
        rows,
        dim,
        eps,
    )?;
    rstd.copy_from_slice(&got);
    Ok(())
}

pub fn rms_apply_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    weight: &[u16],
    rstd: &[f32],
    y: &mut [u16],
    n: usize,
    dim: usize,
) -> Result<()> {
    dispatch::check_len("rms_apply_bf16 x", x.len(), n)?;
    dispatch::check_len("rms_apply_bf16 weight", weight.len(), dim)?;
    dispatch::check_len("rms_apply_bf16 y", y.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    if dim == 0 {
        return Err(WgpuError::Shape("rms_apply_bf16: dim is zero".into()));
    }
    dispatch::check_len("rms_apply_bf16 rstd", rstd.len(), n.div_ceil(dim))?;
    check_device(ctx, WORKGROUP_SIZE)?;
    let params = ApplyParams {
        n: n as u32,
        dim: dim as u32,
        pad0: 0,
        pad1: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "gd-ap-x", &widen(x));
    let wb = dispatch::storage_from_slice(ctx, "gd-ap-w", &widen(weight));
    let rb = dispatch::storage_from_slice(ctx, "gd-ap-rstd", rstd);
    let ob = dispatch::storage_zeroed(ctx, "gd-ap-y", (n * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-ap-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_rms_apply_bf16",
        &shader(),
        "rms_apply_bf16",
        &[(23, &xb), (24, &wb), (25, &rb), (26, &ob), (27, &ub)],
        groups,
    )?;
    let got: Vec<u32> = dispatch::read_back(ctx, &ob, n)?;
    narrow(&got, y);
    Ok(())
}

pub fn rmsnorm_add_scale_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    weight: &[u16],
    residual: &[u16],
    y: &mut [u16],
    rstd_out: Option<&mut [f32]>,
    next_weight: Option<&[u16]>,
    normed_out: Option<&mut [u16]>,
    rows: usize,
    dim: usize,
    eps: f32,
    scale: f32,
    eps_next: f32,
) -> Result<()> {
    dispatch::check_len("rmsnorm_add_scale_bf16 x", x.len(), rows * dim)?;
    dispatch::check_len("rmsnorm_add_scale_bf16 weight", weight.len(), dim)?;
    dispatch::check_len(
        "rmsnorm_add_scale_bf16 residual",
        residual.len(),
        rows * dim,
    )?;
    dispatch::check_len("rmsnorm_add_scale_bf16 y", y.len(), rows * dim)?;
    if normed_out.is_some() && next_weight.is_none() {
        return Err(WgpuError::Shape(
            "rmsnorm_add_scale_bf16: normed_out requires next_weight".into(),
        ));
    }
    if rows == 0 || dim == 0 {
        return Ok(());
    }
    check_device(ctx, WORKGROUP_SIZE)?;

    let mut flags = 0u32;
    if rstd_out.is_some() {
        flags |= 1;
    }
    if normed_out.is_some() {
        flags |= 2;
    }
    let params = RasParams {
        rows: rows as u32,
        dim: dim as u32,
        eps,
        scale,
        eps_next,
        flags,
        pad0: 0,
        pad1: 0,
    };

    let xb = dispatch::storage_from_slice(ctx, "gd-ras-x", &widen(x));
    let wb = dispatch::storage_from_slice(ctx, "gd-ras-w", &widen(weight));
    let rb = dispatch::storage_from_slice(ctx, "gd-ras-res", &widen(residual));
    let yb = dispatch::storage_zeroed(ctx, "gd-ras-y", (rows * dim * 4) as u64);
    let sb = dispatch::storage_zeroed(ctx, "gd-ras-rstd", (rows * 4) as u64);
    let nwb = match next_weight {
        Some(w) => dispatch::storage_from_slice(ctx, "gd-ras-nextw", &widen(w)),
        None => dummy(ctx, "gd-ras-nextw-dummy"),
    };
    let nb = dispatch::storage_zeroed(ctx, "gd-ras-normed", (rows * dim * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-ras-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, rows as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm_add_scale_bf16",
        &shader(),
        "rmsnorm_add_scale_bf16",
        &[
            (28, &xb),
            (29, &wb),
            (30, &rb),
            (31, &yb),
            (32, &sb),
            (33, &nwb),
            (34, &nb),
            (35, &ub),
        ],
        groups,
    )?;

    let gy: Vec<u32> = dispatch::read_back(ctx, &yb, rows * dim)?;
    narrow(&gy, y);
    if let Some(rs) = rstd_out {
        dispatch::check_len("rmsnorm_add_scale_bf16 rstd_out", rs.len(), rows)?;
        let g: Vec<f32> = dispatch::read_back(ctx, &sb, rows)?;
        rs.copy_from_slice(&g);
    }
    if let Some(no) = normed_out {
        dispatch::check_len("rmsnorm_add_scale_bf16 normed_out", no.len(), rows * dim)?;
        let g: Vec<u32> = dispatch::read_back(ctx, &nb, rows * dim)?;
        narrow(&g, no);
    }
    Ok(())
}

pub fn qkv_prep(
    ctx: &WgpuContext,
    qkv: &[u16],
    q_norm: &[u16],
    k_norm: Option<&[u16]>,
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    rope_pos: &[i32],
    cache_pos: Option<&[i32]>,
    delta: i32,
    q_out: &mut [f32],
    k_cache: Option<&mut [u16]>,
    v_cache: Option<&mut [u16]>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Err(WgpuError::Shape("qkv_prep: empty shape".into()));
    }
    if head_dim > MAX_PREP_HEAD_DIM || !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "qkv_prep head_dim {head_dim} must be even and <= {MAX_PREP_HEAD_DIM}"
        )));
    }
    let has_kv = k_norm.is_some();
    if has_kv && (n_kv_heads == 0 || cache_pos.is_none() || k_cache.is_none() || v_cache.is_none())
    {
        return Err(WgpuError::Shape(
            "qkv_prep: kv path needs k_norm, cache_pos, k_cache and v_cache".into(),
        ));
    }
    let heads = if has_kv {
        n_heads + 2 * n_kv_heads
    } else {
        n_heads
    };
    dispatch::check_len("qkv_prep qkv", qkv.len(), heads * head_dim)?;
    dispatch::check_len("qkv_prep q_norm", q_norm.len(), head_dim)?;
    dispatch::check_len("qkv_prep q_out", q_out.len(), n_heads * head_dim)?;
    if rope_pos.is_empty() {
        return Err(WgpuError::Shape("qkv_prep: rope_pos is empty".into()));
    }
    check_device(ctx, WORKGROUP_SIZE)?;

    let params = QkvParams {
        nh: n_heads as u32,
        nkv: n_kv_heads as u32,
        hd: head_dim as u32,
        delta,
        eps,
        has_kv: u32::from(has_kv),
        pad0: 0,
        pad1: 0,
    };

    let kc_words: Option<Vec<u32>> = k_cache.as_deref().map(widen);
    let vc_words: Option<Vec<u32>> = v_cache.as_deref().map(widen);
    let kc_len = kc_words.as_ref().map(|c| c.len()).unwrap_or(0);
    let vc_len = vc_words.as_ref().map(|c| c.len()).unwrap_or(0);

    let ib = dispatch::storage_from_slice(ctx, "gd-qkv-in", &widen(qkv));
    let qwb = dispatch::storage_from_slice(ctx, "gd-qkv-qw", &widen(q_norm));
    let kwb = match k_norm {
        Some(w) => dispatch::storage_from_slice(ctx, "gd-qkv-kw", &widen(w)),
        None => dummy(ctx, "gd-qkv-kw-dummy"),
    };
    let cb = dispatch::storage_from_slice(ctx, "gd-qkv-cos", cos_tbl);
    let sb = dispatch::storage_from_slice(ctx, "gd-qkv-sin", sin_tbl);
    let rpb = dispatch::storage_from_slice(ctx, "gd-qkv-ropepos", rope_pos);
    let cpb = match cache_pos {
        Some(p) => dispatch::storage_from_slice(ctx, "gd-qkv-cachepos", p),
        None => dummy(ctx, "gd-qkv-cachepos-dummy"),
    };
    let qob = dispatch::storage_zeroed(ctx, "gd-qkv-qout", (n_heads * head_dim * 4) as u64);
    let kcb = match kc_words.as_ref() {
        Some(c) => dispatch::storage_from_slice(ctx, "gd-qkv-kcache", c),
        None => dummy(ctx, "gd-qkv-kcache-dummy"),
    };
    let vcb = match vc_words.as_ref() {
        Some(c) => dispatch::storage_from_slice(ctx, "gd-qkv-vcache", c),
        None => dummy(ctx, "gd-qkv-vcache-dummy"),
    };
    let ub = dispatch::uniform_from(ctx, "gd-qkv-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, heads as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_qkv_prep",
        &shader(),
        "qkv_prep",
        &[
            (36, &ib),
            (37, &qwb),
            (38, &kwb),
            (39, &cb),
            (40, &sb),
            (41, &rpb),
            (42, &cpb),
            (43, &qob),
            (44, &kcb),
            (45, &vcb),
            (46, &ub),
        ],
        groups,
    )?;

    let gq: Vec<f32> = dispatch::read_back(ctx, &qob, n_heads * head_dim)?;
    q_out.copy_from_slice(&gq);
    if let Some(c) = k_cache {
        let g: Vec<u32> = dispatch::read_back(ctx, &kcb, kc_len)?;
        narrow(&g, c);
    }
    if let Some(c) = v_cache {
        let g: Vec<u32> = dispatch::read_back(ctx, &vcb, vc_len)?;
        narrow(&g, c);
    }
    Ok(())
}

pub fn argmax_bf16_part_count() -> usize {
    ARGMAX_BLOCKS
}

pub fn argmax_bf16_parts(
    ctx: &WgpuContext,
    logits: &[u16],
    part_val: &mut [f32],
    part_idx: &mut [i32],
    vocab: usize,
) -> Result<()> {
    dispatch::check_len("argmax_bf16_parts logits", logits.len(), vocab)?;
    dispatch::check_len("argmax_bf16_parts part_val", part_val.len(), ARGMAX_BLOCKS)?;
    dispatch::check_len("argmax_bf16_parts part_idx", part_idx.len(), ARGMAX_BLOCKS)?;
    if vocab == 0 {
        return Err(WgpuError::Shape("argmax_bf16_parts: vocab is zero".into()));
    }
    check_device(ctx, WORKGROUP_SIZE)?;
    let params = ArgmaxParams {
        n: vocab as u32,
        nparts: ARGMAX_BLOCKS as u32,
        ring_mask: 0,
        has_ring: 0,
    };
    let lb = dispatch::storage_from_slice(ctx, "gd-am-logits", &widen(logits));
    let vb = dispatch::storage_zeroed(ctx, "gd-am-partval", (ARGMAX_BLOCKS * 4) as u64);
    let ib = dispatch::storage_zeroed(ctx, "gd-am-partidx", (ARGMAX_BLOCKS * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-am-params", &params);
    dispatch::run(
        ctx,
        "nv_kernels_argmax_bf16_parts",
        &shader(),
        "argmax_bf16_stage1",
        &[(47, &lb), (48, &vb), (49, &ib), (50, &ub)],
        (ARGMAX_BLOCKS as u32, 1, 1),
    )?;
    let gv: Vec<f32> = dispatch::read_back(ctx, &vb, ARGMAX_BLOCKS)?;
    let gi: Vec<i32> = dispatch::read_back(ctx, &ib, ARGMAX_BLOCKS)?;
    part_val.copy_from_slice(&gv);
    part_idx.copy_from_slice(&gi);
    Ok(())
}

pub fn argmax_bf16(
    ctx: &WgpuContext,
    logits: &[u16],
    pos: &[i32],
    token_out: &mut [u32],
    ring: Option<&mut [u32]>,
    ring_mask: i32,
    vocab: usize,
) -> Result<()> {
    dispatch::check_len("argmax_bf16 logits", logits.len(), vocab)?;
    if token_out.is_empty() {
        return Err(WgpuError::Shape("argmax_bf16: token_out is empty".into()));
    }
    if vocab == 0 {
        return Err(WgpuError::Shape("argmax_bf16: vocab is zero".into()));
    }
    if pos.is_empty() {
        return Err(WgpuError::Shape("argmax_bf16: pos is empty".into()));
    }
    check_device(ctx, ARGMAX_BLOCKS as u32)?;
    let params = ArgmaxParams {
        n: vocab as u32,
        nparts: ARGMAX_BLOCKS as u32,
        ring_mask,
        has_ring: u32::from(ring.is_some()),
    };
    let ring_len = ring.as_deref().map(|r| r.len()).unwrap_or(0);
    let lb = dispatch::storage_from_slice(ctx, "gd-am-logits", &widen(logits));
    let vb = dispatch::storage_zeroed(ctx, "gd-am-partval", (ARGMAX_BLOCKS * 4) as u64);
    let ib = dispatch::storage_zeroed(ctx, "gd-am-partidx", (ARGMAX_BLOCKS * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-am-params", &params);
    let pb = dispatch::storage_from_slice(ctx, "gd-am-pos", pos);
    let tb = dispatch::storage_zeroed(ctx, "gd-am-token", (token_out.len() * 4) as u64);
    let rb = match ring.as_deref() {
        Some(r) => dispatch::storage_from_slice(ctx, "gd-am-ring", r),
        None => dummy(ctx, "gd-am-ring-dummy"),
    };

    dispatch::run(
        ctx,
        "nv_kernels_argmax_bf16_stage1",
        &shader(),
        "argmax_bf16_stage1",
        &[(47, &lb), (48, &vb), (49, &ib), (50, &ub)],
        (ARGMAX_BLOCKS as u32, 1, 1),
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_argmax_bf16_stage2",
        &shader(),
        "argmax_bf16_stage2",
        &[
            (48, &vb),
            (49, &ib),
            (50, &ub),
            (51, &pb),
            (52, &tb),
            (53, &rb),
        ],
        (1, 1, 1),
    )?;

    let gt: Vec<u32> = dispatch::read_back(ctx, &tb, token_out.len())?;
    token_out.copy_from_slice(&gt);
    if let Some(r) = ring {
        let g: Vec<u32> = dispatch::read_back(ctx, &rb, ring_len)?;
        r.copy_from_slice(&g);
    }
    Ok(())
}

pub fn argmax_f32_rows(
    ctx: &WgpuContext,
    logits: &[f32],
    token_out: &mut [u32],
    rows: usize,
    vocab: usize,
) -> Result<()> {
    argmax_f32_rows_with_parts(ctx, logits, token_out, None, None, rows, vocab)
}

pub fn argmax_f32_rows_with_parts(
    ctx: &WgpuContext,
    logits: &[f32],
    token_out: &mut [u32],
    part_val: Option<&mut [f32]>,
    part_idx: Option<&mut [i32]>,
    rows: usize,
    vocab: usize,
) -> Result<()> {
    dispatch::check_len("argmax_f32_rows logits", logits.len(), rows * vocab)?;
    dispatch::check_len("argmax_f32_rows token_out", token_out.len(), rows)?;
    if rows == 0 || vocab == 0 {
        return Err(WgpuError::Shape("argmax_f32_rows: empty shape".into()));
    }
    check_device(ctx, ARGMAX_BLOCKS as u32)?;
    if rows as u32 > ctx.caps.max_compute_workgroups_per_dimension {
        return Err(WgpuError::Unsupported(format!(
            "argmax_f32_rows rows {rows} exceeds workgroup dimension limit {}",
            ctx.caps.max_compute_workgroups_per_dimension
        )));
    }
    let parts = rows * ARGMAX_BLOCKS;
    let params = ArgmaxRowsParams {
        rows: rows as u32,
        n: vocab as u32,
        pad0: 0,
        pad1: 0,
    };
    let lb = dispatch::storage_from_slice(ctx, "gd-amf-logits", logits);
    let vb = dispatch::storage_zeroed(ctx, "gd-amf-partval", (parts * 4) as u64);
    let ib = dispatch::storage_zeroed(ctx, "gd-amf-partidx", (parts * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "gd-amf-out", (rows * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "gd-amf-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_argmax_f32_rows_stage1",
        &shader(),
        "argmax_f32_rows_stage1",
        &[(54, &lb), (55, &vb), (56, &ib), (58, &ub)],
        (ARGMAX_BLOCKS as u32, rows as u32, 1),
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_argmax_f32_rows_stage2",
        &shader(),
        "argmax_f32_rows_stage2",
        &[(55, &vb), (56, &ib), (57, &ob), (58, &ub)],
        dispatch::workgroup_count_1d(ctx, rows as u64, 1),
    )?;

    let go: Vec<u32> = dispatch::read_back(ctx, &ob, rows)?;
    token_out.copy_from_slice(&go);
    if let Some(pv) = part_val {
        dispatch::check_len("argmax_f32_rows part_val", pv.len(), parts)?;
        let g: Vec<f32> = dispatch::read_back(ctx, &vb, parts)?;
        pv.copy_from_slice(&g);
    }
    if let Some(pi) = part_idx {
        dispatch::check_len("argmax_f32_rows part_idx", pi.len(), parts)?;
        let g: Vec<i32> = dispatch::read_back(ctx, &ib, parts)?;
        pi.copy_from_slice(&g);
    }
    Ok(())
}

pub const ARGMAX_SOFTCAP_STAGE1_ENTRY: &str = "argmax_softcap_bf16_stage1";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArgmaxCapParams {
    pub n: u32,
    pub cap: f32,
    pub inv_cap: f32,
    pub softcap: u32,
}

pub fn argmax_softcap_cap_params(vocab: usize, cap: f32) -> ArgmaxCapParams {
    let softcap = cap > 0.0 && cap.is_finite();
    ArgmaxCapParams {
        n: vocab as u32,
        cap,
        inv_cap: if softcap { 1.0 / cap } else { 0.0 },
        softcap: softcap as u32,
    }
}

pub fn argmax_softcap_bf16_fold(
    ctx: &WgpuContext,
    logits_pk: &[u32],
    logits_f32: &mut [f32],
    token_out: &mut [u32],
    part_val: Option<&mut [f32]>,
    part_idx: Option<&mut [i32]>,
    vocab: usize,
    cap: f32,
) -> Result<()> {
    dispatch::check_len(
        "argmax_softcap_bf16_fold logits_pk",
        logits_pk.len(),
        vocab.div_ceil(2),
    )?;
    dispatch::check_len(
        "argmax_softcap_bf16_fold logits_f32",
        logits_f32.len(),
        vocab,
    )?;
    dispatch::check_len("argmax_softcap_bf16_fold token_out", token_out.len(), 1)?;
    if vocab == 0 {
        return Err(WgpuError::Shape(
            "argmax_softcap_bf16_fold: empty shape".into(),
        ));
    }
    check_device(ctx, ARGMAX_BLOCKS as u32)?;
    let p = argmax_softcap_cap_params(vocab, cap);
    let rows_p = ArgmaxRowsParams {
        rows: 1,
        n: vocab as u32,
        pad0: 0,
        pad1: 0,
    };
    let pk = dispatch::storage_from_slice(ctx, "gd-amc-pk", logits_pk);
    let lf = dispatch::storage_zeroed(ctx, "gd-amc-logits", (vocab * 4) as u64);
    let vb = dispatch::storage_zeroed(ctx, "gd-amc-partval", (ARGMAX_BLOCKS * 4) as u64);
    let ib = dispatch::storage_zeroed(ctx, "gd-amc-partidx", (ARGMAX_BLOCKS * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "gd-amc-out", 4);
    let ub = dispatch::uniform_from(ctx, "gd-amc-params", &p);
    let rub = dispatch::uniform_from(ctx, "gd-amc-rows-params", &rows_p);
    dispatch::run(
        ctx,
        "nv_kernels_argmax_softcap_bf16_stage1",
        &shader(),
        ARGMAX_SOFTCAP_STAGE1_ENTRY,
        &[(65, &pk), (66, &lf), (55, &vb), (56, &ib), (67, &ub)],
        (ARGMAX_BLOCKS as u32, 1, 1),
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_argmax_f32_rows_stage2",
        &shader(),
        "argmax_f32_rows_stage2",
        &[(55, &vb), (56, &ib), (57, &ob), (58, &rub)],
        dispatch::workgroup_count_1d(ctx, 1, 1),
    )?;
    let go: Vec<u32> = dispatch::read_back(ctx, &ob, 1)?;
    token_out.copy_from_slice(&go);
    let lo: Vec<f32> = dispatch::read_back(ctx, &lf, vocab)?;
    logits_f32.copy_from_slice(&lo);
    if let Some(pv) = part_val {
        dispatch::check_len("argmax_softcap_bf16_fold part_val", pv.len(), ARGMAX_BLOCKS)?;
        let g: Vec<f32> = dispatch::read_back(ctx, &vb, ARGMAX_BLOCKS)?;
        pv.copy_from_slice(&g);
    }
    if let Some(pi) = part_idx {
        dispatch::check_len("argmax_softcap_bf16_fold part_idx", pi.len(), ARGMAX_BLOCKS)?;
        let g: Vec<i32> = dispatch::read_back(ctx, &ib, ARGMAX_BLOCKS)?;
        pi.copy_from_slice(&g);
    }
    Ok(())
}

pub fn token_map_u32(ctx: &WgpuContext, map: &[u32], idx: &[u32], out: &mut [u32]) -> Result<()> {
    if map.is_empty() || idx.is_empty() || out.is_empty() {
        return Err(WgpuError::Shape("token_map_u32: empty buffer".into()));
    }
    if idx[0] as usize >= map.len() {
        return Err(WgpuError::Shape(format!(
            "token_map_u32: idx {} out of range for map of {}",
            idx[0],
            map.len()
        )));
    }
    let mb = dispatch::storage_from_slice(ctx, "gd-tm-map", map);
    let ib = dispatch::storage_from_slice(ctx, "gd-tm-idx", idx);
    let ob = dispatch::storage_zeroed(ctx, "gd-tm-out", (out.len() * 4) as u64);
    dispatch::run(
        ctx,
        "nv_kernels_token_map_u32",
        &shader(),
        "token_map_u32",
        &[(59, &mb), (60, &ib), (61, &ob)],
        (1, 1, 1),
    )?;
    let got: Vec<u32> = dispatch::read_back(ctx, &ob, out.len())?;
    out.copy_from_slice(&got);
    Ok(())
}

pub fn multi_zero_bf16(ctx: &WgpuContext, buffers: &mut [&mut [u16]]) -> Result<()> {
    if buffers.is_empty() {
        return Ok(());
    }
    check_device(ctx, 256)?;
    let mut desc: Vec<u32> = Vec::with_capacity(buffers.len() * 2);
    let mut data: Vec<u32> = Vec::new();
    for b in buffers.iter() {
        desc.push(data.len() as u32);
        desc.push(b.len() as u32);
        for pair in b.chunks(2) {
            let lo = pair[0] as u32;
            let hi = if pair.len() > 1 { pair[1] as u32 } else { 0 };
            data.push(lo | (hi << 16));
        }
    }
    if data.is_empty() {
        return Ok(());
    }
    let params = ZeroParams {
        n: buffers.len() as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let db = dispatch::storage_from_slice(ctx, "gd-mz-data", &data);
    let sb = dispatch::storage_from_slice(ctx, "gd-mz-desc", &desc);
    let ub = dispatch::uniform_from(ctx, "gd-mz-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, buffers.len() as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_multi_zero_bf16",
        &shader(),
        "multi_zero_bf16",
        &[(62, &db), (63, &sb), (64, &ub)],
        groups,
    )?;
    let got: Vec<u32> = dispatch::read_back(ctx, &db, data.len())?;
    let mut off = 0usize;
    for b in buffers.iter_mut() {
        let words = b.len().div_ceil(2);
        for (i, v) in b.iter_mut().enumerate() {
            let w = got[off + i / 2];
            *v = if i % 2 == 0 {
                (w & 0xffff) as u16
            } else {
                (w >> 16) as u16
            };
        }
        off += words;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<EwParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<KvParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<RmsParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<ApplyParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<RasParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<QkvParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<ArgmaxParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<ArgmaxRowsParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<ZeroParams>() % 16, 0);
    }

    #[test]
    fn part_count_matches_cuda_constant() {
        assert_eq!(argmax_bf16_part_count(), 256);
    }

    #[test]
    fn widen_and_narrow_roundtrip() {
        let src = [0u16, 1, 0x7f80, 0xffff];
        let mut back = [0u16; 4];
        narrow(&widen(&src), &mut back);
        assert_eq!(src, back);
    }
}
