#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
pub(crate) use crate::wgpu_backend::dequant::bytes_to_words;
pub(crate) use crate::wgpu_backend::pack::unpack_u8_by_element as words_to_bytes;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
pub(crate) use crate::wgpu_backend::pack::{pack_u16_even_min_one_word as pack_bf16, unpack_u16_pairs_clamped as unpack_bf16};

pub const WGSL: &str = include_str!("../../../wgsl/tree_verify_attn.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const MAX_HEAD_DIM: usize = 512;

const SCRATCH_BYTES: u32 = 512 * 4 + 256 * 4 + 8 * 4 + 8 * 4 + 8 * 512 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TvParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub k: u32,
    pub window: u32,
    pub scaling: f32,
    pub ring: u32,
    pub n_committed: u32,
    pub base: u32,
    pub n_accept: u32,
    pub stride_words: u32,
    pub total: u32,
}

fn check_device(ctx: &WgpuContext, what: &str, scratch: u32) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, what, WORKGROUP_SIZE, scratch)
}

pub(crate) fn grid_2d(
    ctx: &WgpuContext,
    what: &str,
    x: usize,
    y: usize,
) -> Result<(u32, u32, u32)> {
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    if x > limit || y > limit {
        return Err(WgpuError::Unsupported(format!(
            "{what}: grid {x}x{y} exceeds max workgroups per dimension {limit}"
        )));
    }
    Ok((x as u32, y as u32, 1))
}

pub(crate) fn slot_count(len: usize, per_slot: usize, what: &str) -> Result<usize> {
    if per_slot == 0 || !len.is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "{what}: length {len} is not a multiple of {per_slot}"
        )));
    }
    Ok(len / per_slot)
}

pub fn tree_verify_attn_bf16(
    ctx: &WgpuContext,
    q: &[u16],
    kc: &[u16],
    vc: &[u16],
    mask: &[u8],
    positions: &[i32],
    n_committed: &[i32],
    out: &mut [u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k: usize,
    window: usize,
    scaling: f32,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || k == 0 || head_dim == 0 {
        return Ok(());
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "tree_verify_attn_bf16 head_dim {head_dim} exceeds {MAX_HEAD_DIM}"
        )));
    }
    if !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "tree_verify_attn_bf16 needs an even head_dim so bf16 pairs land on whole u32 words; got {head_dim}"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WgpuError::Shape(format!(
            "tree_verify_attn_bf16 n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    if n_committed.is_empty() || n_committed[0] < 0 {
        return Err(WgpuError::Shape(
            "tree_verify_attn_bf16: n_committed must hold one non-negative entry".to_string(),
        ));
    }
    let nc = n_committed[0] as usize;
    dispatch::check_len("tree_verify_attn_bf16 q", q.len(), k * n_heads * head_dim)?;
    dispatch::check_len(
        "tree_verify_attn_bf16 out",
        out.len(),
        k * n_heads * head_dim,
    )?;
    dispatch::check_len("tree_verify_attn_bf16 mask", mask.len(), k * k)?;
    let slots = slot_count(kc.len(), n_kv_heads * head_dim, "tree_verify_attn_bf16 kc")?;
    dispatch::check_len("tree_verify_attn_bf16 vc", vc.len(), kc.len())?;
    if nc + k > slots {
        return Err(WgpuError::Shape(format!(
            "tree_verify_attn_bf16: n_committed {nc} + k {k} exceeds {slots} cache slots"
        )));
    }
    if window > 0 {
        dispatch::check_len("tree_verify_attn_bf16 positions", positions.len(), k)?;
    }
    check_device(ctx, "tree_verify_attn_bf16", SCRATCH_BYTES)?;

    let params = TvParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        k: k as u32,
        window: window as u32,
        scaling,
        ring: 0,
        n_committed: nc as u32,
        ..Default::default()
    };

    let mut pos = positions.to_vec();
    if pos.is_empty() {
        pos.push(0);
    }

    let qb = dispatch::storage_from_slice(ctx, "tv.q", &pack_bf16(q));
    let kb = dispatch::storage_from_slice(ctx, "tv.kc", &pack_bf16(kc));
    let vb = dispatch::storage_from_slice(ctx, "tv.vc", &pack_bf16(vc));
    let mb = dispatch::storage_from_slice(ctx, "tv.mask", &bytes_to_words(mask));
    let pb = dispatch::storage_from_slice(ctx, "tv.pos", &pos);
    let out_words = out.len() / 2;
    let ob = dispatch::storage_zeroed(ctx, "tv.out", (out_words * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "tv.params", &params);

    let groups = grid_2d(ctx, "tree_verify_attn_bf16", n_heads, k)?;
    dispatch::run(
        ctx,
        "nv_kernels_tree_verify_attn_bf16",
        &compose(WGSL),
        "tree_verify_attn_bf16",
        &[
            (0, &qb),
            (1, &kb),
            (2, &vb),
            (3, &mb),
            (4, &pb),
            (5, &ob),
            (6, &ub),
        ],
        groups,
    )?;

    let got: Vec<u32> = dispatch::read_back(ctx, &ob, out_words)?;
    unpack_bf16(&got, out);
    Ok(())
}

pub fn kv_append_bf16(
    ctx: &WgpuContext,
    k_src: &[u16],
    v_src: &[u16],
    kc: &mut [u16],
    vc: &mut [u16],
    n_committed: &[i32],
    n_kv_heads: usize,
    head_dim: usize,
    k: usize,
    ring: usize,
) -> Result<()> {
    if k == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    let row = n_kv_heads * head_dim;
    if !row.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "kv_append_bf16 needs an even n_kv_heads*head_dim so bf16 pairs land on whole u32 words; got {row}"
        )));
    }
    if n_committed.is_empty() || n_committed[0] < 0 {
        return Err(WgpuError::Shape(
            "kv_append_bf16: n_committed must hold one non-negative entry".to_string(),
        ));
    }
    let nc = n_committed[0] as usize;
    dispatch::check_len("kv_append_bf16 k_src", k_src.len(), k * row)?;
    dispatch::check_len("kv_append_bf16 v_src", v_src.len(), k * row)?;
    let slots = slot_count(kc.len(), row, "kv_append_bf16 kc")?;
    dispatch::check_len("kv_append_bf16 vc", vc.len(), kc.len())?;
    if ring > slots {
        return Err(WgpuError::Shape(format!(
            "kv_append_bf16: ring {ring} exceeds {slots} cache slots"
        )));
    }
    if ring == 0 && nc + k > slots {
        return Err(WgpuError::Shape(format!(
            "kv_append_bf16: n_committed {nc} + k {k} exceeds {slots} cache slots"
        )));
    }
    check_device(ctx, "kv_append_bf16", 0)?;

    let row_words = row / 2;
    let total = k * row_words;
    let params = TvParams {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        k: k as u32,
        ring: ring as u32,
        n_committed: nc as u32,
        stride_words: row_words as u32,
        total: total as u32,
        ..Default::default()
    };

    let ksb = dispatch::storage_from_slice(ctx, "tv.append.ksrc", &pack_bf16(k_src));
    let vsb = dispatch::storage_from_slice(ctx, "tv.append.vsrc", &pack_bf16(v_src));
    let kcb = dispatch::storage_from_slice(ctx, "tv.append.kc", &pack_bf16(kc));
    let vcb = dispatch::storage_from_slice(ctx, "tv.append.vc", &pack_bf16(vc));
    let ub = dispatch::uniform_from(ctx, "tv.append.params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_kv_append_bf16",
        &compose(WGSL),
        "kv_append_bf16",
        &[(7, &ksb), (8, &vsb), (9, &kcb), (10, &vcb), (11, &ub)],
        groups,
    )?;

    let got_k: Vec<u32> = dispatch::read_back(ctx, &kcb, kc.len() / 2)?;
    unpack_bf16(&got_k, kc);
    let got_v: Vec<u32> = dispatch::read_back(ctx, &vcb, vc.len() / 2)?;
    unpack_bf16(&got_v, vc);
    Ok(())
}

pub fn kv_compact_bf16(
    ctx: &WgpuContext,
    kc: &mut [u16],
    vc: &mut [u16],
    path: &[i32],
    base: usize,
    n_accept: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if n_accept == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    let row = n_kv_heads * head_dim;
    if !row.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "kv_compact_bf16 needs an even n_kv_heads*head_dim so bf16 pairs land on whole u32 words; got {row}"
        )));
    }
    dispatch::check_len("kv_compact_bf16 path", path.len(), n_accept)?;
    let slots = slot_count(kc.len(), row, "kv_compact_bf16 kc")?;
    dispatch::check_len("kv_compact_bf16 vc", vc.len(), kc.len())?;
    if base + n_accept > slots {
        return Err(WgpuError::Shape(format!(
            "kv_compact_bf16: base {base} + n_accept {n_accept} exceeds {slots} cache slots"
        )));
    }
    for (i, &p) in path.iter().enumerate() {
        if p < 0 || base + p as usize >= slots {
            return Err(WgpuError::Shape(format!(
                "kv_compact_bf16: path[{i}] = {p} with base {base} is outside {slots} cache slots"
            )));
        }
    }
    check_device(ctx, "kv_compact_bf16", 0)?;

    let row_words = row / 2;
    let total = n_accept * row_words;
    let params = TvParams {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        base: base as u32,
        n_accept: n_accept as u32,
        stride_words: row_words as u32,
        total: total as u32,
        ..Default::default()
    };

    let kcb = dispatch::storage_from_slice(ctx, "tv.compact.kc", &pack_bf16(kc));
    let vcb = dispatch::storage_from_slice(ctx, "tv.compact.vc", &pack_bf16(vc));
    let skb = dispatch::storage_zeroed(ctx, "tv.compact.sk", (total * 4) as u64);
    let svb = dispatch::storage_zeroed(ctx, "tv.compact.sv", (total * 4) as u64);
    let pb = dispatch::storage_from_slice(ctx, "tv.compact.path", path);
    let ub = dispatch::uniform_from(ctx, "tv.compact.params", &params);

    let gather: [(u32, &wgpu::Buffer); 6] = [
        (12, &kcb),
        (13, &vcb),
        (14, &skb),
        (15, &svb),
        (16, &pb),
        (17, &ub),
    ];
    let scatter: [(u32, &wgpu::Buffer); 5] =
        [(12, &kcb), (13, &vcb), (14, &skb), (15, &svb), (17, &ub)];
    let groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    let src = compose(WGSL);
    dispatch::run(
        ctx,
        "nv_kernels_kv_gather_bf16",
        &src,
        "kv_gather_bf16",
        &gather,
        groups,
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_kv_scatter_bf16",
        &src,
        "kv_scatter_bf16",
        &scatter,
        groups,
    )?;

    let got_k: Vec<u32> = dispatch::read_back(ctx, &kcb, kc.len() / 2)?;
    unpack_bf16(&got_k, kc);
    let got_v: Vec<u32> = dispatch::read_back(ctx, &vcb, vc.len() / 2)?;
    unpack_bf16(&got_v, vc);
    Ok(())
}

pub fn cpu_kv_append_bf16(
    k_src: &[u16],
    v_src: &[u16],
    kc: &mut [u16],
    vc: &mut [u16],
    n_committed: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k: usize,
    ring: usize,
) {
    let row = n_kv_heads * head_dim;
    for token in 0..k {
        let mut slot = n_committed + token;
        if ring > 0 {
            slot %= ring;
        }
        let src = token * row;
        let dst = slot * row;
        kc[dst..dst + row].copy_from_slice(&k_src[src..src + row]);
        vc[dst..dst + row].copy_from_slice(&v_src[src..src + row]);
    }
}

pub fn cpu_kv_compact_bf16(
    kc: &mut [u16],
    vc: &mut [u16],
    path: &[i32],
    base: usize,
    n_accept: usize,
    stride: usize,
) {
    let mut sk = vec![0u16; n_accept * stride];
    let mut sv = vec![0u16; n_accept * stride];
    for i in 0..n_accept {
        let src = (base + path[i] as usize) * stride;
        sk[i * stride..(i + 1) * stride].copy_from_slice(&kc[src..src + stride]);
        sv[i * stride..(i + 1) * stride].copy_from_slice(&vc[src..src + stride]);
    }
    for i in 0..n_accept {
        let dst = (base + i) * stride;
        kc[dst..dst + stride].copy_from_slice(&sk[i * stride..(i + 1) * stride]);
        vc[dst..dst + stride].copy_from_slice(&sv[i * stride..(i + 1) * stride]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<TvParams>() % 16, 0);
    }

    #[test]
    fn byte_word_packing_round_trips() {
        let bytes: Vec<u8> = (0u8..13).collect();
        let words = bytes_to_words(&bytes);
        let mut back = vec![0u8; bytes.len()];
        words_to_bytes(&words, &mut back);
        assert_eq!(back, bytes);
    }

    #[test]
    fn cpu_compact_matches_gather_scatter() {
        let stride = 2;
        let mut kc: Vec<u16> = (0u16..12).collect();
        let mut vc: Vec<u16> = (100u16..112).collect();
        cpu_kv_compact_bf16(&mut kc, &mut vc, &[2, 0], 1, 2, stride);
        assert_eq!(&kc[2..6], &[6, 7, 2, 3]);
        assert_eq!(&vc[2..6], &[106, 107, 102, 103]);
    }

    #[test]
    fn slot_bounds_are_checked() {
        assert!(slot_count(10, 4, "x").is_err());
        assert_eq!(slot_count(12, 4, "x").unwrap(), 3);
    }
}
