#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::kernels::kv_fp8;
use crate::wgpu_backend::kernels::tree_verify_attn::{
    bytes_to_words, grid_2d, pack_bf16, slot_count, unpack_bf16, words_to_bytes, TvParams,
};
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/tree_verify_fp8.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const MAX_HEAD_DIM: usize = 512;

const SCRATCH_BYTES: u32 = 512 * 4 + 256 * 4 + 8 * 4 + 8 * 4 + 8 * 512 * 4 + 256 * 4 * 2;

fn source() -> String {
    compose(&format!("{}\n{}", kv_fp8::WGSL, WGSL))
}

fn check_device(ctx: &WgpuContext, what: &str, scratch: u32) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, what, WORKGROUP_SIZE, scratch)
}

pub fn tree_verify_attn_fp8(
    ctx: &WgpuContext,
    q: &[u16],
    kc: &[u8],
    vc: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    mask: &[u8],
    positions: &[i32],
    n_committed: &[i32],
    out: &mut [u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k: usize,
    window: usize,
    ring: usize,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || k == 0 || head_dim == 0 {
        return Ok(());
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "tree_verify_attn_fp8 head_dim {head_dim} exceeds {MAX_HEAD_DIM}"
        )));
    }
    if !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "tree_verify_attn_fp8 needs an even head_dim so the bf16 query lands on whole u32 words; got {head_dim}"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WgpuError::Shape(format!(
            "tree_verify_attn_fp8 n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    if ring > 0 && window == 0 {
        return Err(WgpuError::Shape(
            "tree_verify_attn_fp8: a ring cache requires a positive window".to_string(),
        ));
    }
    if n_committed.is_empty() || n_committed[0] < 0 {
        return Err(WgpuError::Shape(
            "tree_verify_attn_fp8: n_committed must hold one non-negative entry".to_string(),
        ));
    }
    let nc = n_committed[0] as usize;
    dispatch::check_len("tree_verify_attn_fp8 q", q.len(), k * n_heads * head_dim)?;
    dispatch::check_len(
        "tree_verify_attn_fp8 out",
        out.len(),
        k * n_heads * head_dim,
    )?;
    dispatch::check_len("tree_verify_attn_fp8 mask", mask.len(), k * k)?;
    let slots = slot_count(kc.len(), n_kv_heads * head_dim, "tree_verify_attn_fp8 kc")?;
    dispatch::check_len("tree_verify_attn_fp8 vc", vc.len(), kc.len())?;
    dispatch::check_len(
        "tree_verify_attn_fp8 k_scales",
        k_scales.len(),
        slots * n_kv_heads,
    )?;
    dispatch::check_len(
        "tree_verify_attn_fp8 v_scales",
        v_scales.len(),
        slots * n_kv_heads,
    )?;
    if ring > slots {
        return Err(WgpuError::Shape(format!(
            "tree_verify_attn_fp8: ring {ring} exceeds {slots} cache slots"
        )));
    }
    if ring == 0 && nc + k > slots {
        return Err(WgpuError::Shape(format!(
            "tree_verify_attn_fp8: n_committed {nc} + k {k} exceeds {slots} cache slots"
        )));
    }
    if window > 0 {
        dispatch::check_len("tree_verify_attn_fp8 positions", positions.len(), k)?;
    }
    check_device(ctx, "tree_verify_attn_fp8", SCRATCH_BYTES)?;

    let params = TvParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        k: k as u32,
        window: window as u32,
        scaling: 1.0,
        ring: ring as u32,
        n_committed: nc as u32,
        ..Default::default()
    };

    let mut pos = positions.to_vec();
    if pos.is_empty() {
        pos.push(0);
    }

    let qb = dispatch::storage_from_slice(ctx, "tvf.q", &pack_bf16(q));
    let kb = dispatch::storage_from_slice(ctx, "tvf.kc", &bytes_to_words(kc));
    let vb = dispatch::storage_from_slice(ctx, "tvf.vc", &bytes_to_words(vc));
    let ksb = dispatch::storage_from_slice(ctx, "tvf.ks", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "tvf.vs", v_scales);
    let mb = dispatch::storage_from_slice(ctx, "tvf.mask", &bytes_to_words(mask));
    let pb = dispatch::storage_from_slice(ctx, "tvf.pos", &pos);
    let out_words = out.len() / 2;
    let ob = dispatch::storage_zeroed(ctx, "tvf.out", (out_words * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "tvf.params", &params);

    let groups = grid_2d(ctx, "tree_verify_attn_fp8", n_heads, k)?;
    dispatch::run(
        ctx,
        "nv_kernels_tree_verify_attn_fp8",
        &source(),
        "tree_verify_attn_fp8",
        &[
            (20, &qb),
            (21, &kb),
            (22, &vb),
            (23, &ksb),
            (24, &vsb),
            (25, &mb),
            (26, &pb),
            (27, &ob),
            (28, &ub),
        ],
        groups,
    )?;

    let got: Vec<u32> = dispatch::read_back(ctx, &ob, out_words)?;
    unpack_bf16(&got, out);
    Ok(())
}

pub fn kv_append_fp8(
    ctx: &WgpuContext,
    k_src: &[u16],
    v_src: &[u16],
    kc: &mut [u8],
    vc: &mut [u8],
    k_scales: &mut [f32],
    v_scales: &mut [f32],
    n_committed: &[i32],
    n_kv_heads: usize,
    head_dim: usize,
    k: usize,
    ring: usize,
) -> Result<()> {
    if k == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    if !head_dim.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "kv_append_fp8 needs head_dim divisible by 4 so fp8 bytes land on whole u32 words; got {head_dim}"
        )));
    }
    if n_committed.is_empty() || n_committed[0] < 0 {
        return Err(WgpuError::Shape(
            "kv_append_fp8: n_committed must hold one non-negative entry".to_string(),
        ));
    }
    let nc = n_committed[0] as usize;
    let row = n_kv_heads * head_dim;
    dispatch::check_len("kv_append_fp8 k_src", k_src.len(), k * row)?;
    dispatch::check_len("kv_append_fp8 v_src", v_src.len(), k * row)?;
    let slots = slot_count(kc.len(), row, "kv_append_fp8 kc")?;
    dispatch::check_len("kv_append_fp8 vc", vc.len(), kc.len())?;
    dispatch::check_len("kv_append_fp8 k_scales", k_scales.len(), slots * n_kv_heads)?;
    dispatch::check_len("kv_append_fp8 v_scales", v_scales.len(), slots * n_kv_heads)?;
    if ring > slots {
        return Err(WgpuError::Shape(format!(
            "kv_append_fp8: ring {ring} exceeds {slots} cache slots"
        )));
    }
    if ring == 0 && nc + k > slots {
        return Err(WgpuError::Shape(format!(
            "kv_append_fp8: n_committed {nc} + k {k} exceeds {slots} cache slots"
        )));
    }
    check_device(ctx, "kv_append_fp8", 256 * 4 * 2 + 256 * 4 + 4)?;

    let params = TvParams {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        k: k as u32,
        ring: ring as u32,
        n_committed: nc as u32,
        stride_words: (row / 4) as u32,
        total: (k * row / 4) as u32,
        ..Default::default()
    };

    let ksb = dispatch::storage_from_slice(ctx, "tvf.append.ksrc", &pack_bf16(k_src));
    let vsb = dispatch::storage_from_slice(ctx, "tvf.append.vsrc", &pack_bf16(v_src));
    let kcb = dispatch::storage_from_slice(ctx, "tvf.append.kc", &bytes_to_words(kc));
    let vcb = dispatch::storage_from_slice(ctx, "tvf.append.vc", &bytes_to_words(vc));
    let kscb = dispatch::storage_from_slice(ctx, "tvf.append.ksc", k_scales);
    let vscb = dispatch::storage_from_slice(ctx, "tvf.append.vsc", v_scales);
    let ub = dispatch::uniform_from(ctx, "tvf.append.params", &params);

    let groups = grid_2d(ctx, "kv_append_fp8", n_kv_heads, k)?;
    dispatch::run(
        ctx,
        "nv_kernels_kv_append_fp8",
        &source(),
        "kv_append_fp8",
        &[
            (29, &ksb),
            (30, &vsb),
            (31, &kcb),
            (32, &vcb),
            (33, &kscb),
            (34, &vscb),
            (35, &ub),
        ],
        groups,
    )?;

    let got_k: Vec<u32> = dispatch::read_back(ctx, &kcb, kc.len().div_ceil(4))?;
    words_to_bytes(&got_k, kc);
    let got_v: Vec<u32> = dispatch::read_back(ctx, &vcb, vc.len().div_ceil(4))?;
    words_to_bytes(&got_v, vc);
    let got_ks: Vec<f32> = dispatch::read_back(ctx, &kscb, k_scales.len())?;
    k_scales.copy_from_slice(&got_ks);
    let got_vs: Vec<f32> = dispatch::read_back(ctx, &vscb, v_scales.len())?;
    v_scales.copy_from_slice(&got_vs);
    Ok(())
}

pub fn kv_compact_fp8(
    ctx: &WgpuContext,
    kc: &mut [u8],
    vc: &mut [u8],
    k_scales: &mut [f32],
    v_scales: &mut [f32],
    path: &[i32],
    base: usize,
    n_accept: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    kv_compact_fp8_ring(
        ctx, kc, vc, k_scales, v_scales, path, base, n_accept, n_kv_heads, head_dim, 0,
    )
}

pub fn kv_compact_fp8_ring(
    ctx: &WgpuContext,
    kc: &mut [u8],
    vc: &mut [u8],
    k_scales: &mut [f32],
    v_scales: &mut [f32],
    path: &[i32],
    base: usize,
    n_accept: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ring: usize,
) -> Result<()> {
    if n_accept == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    let row = n_kv_heads * head_dim;
    if !row.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "kv_compact_fp8 needs n_kv_heads*head_dim divisible by 4 so fp8 rows land on whole u32 words; got {row}"
        )));
    }
    dispatch::check_len("kv_compact_fp8 path", path.len(), n_accept)?;
    let slots = slot_count(kc.len(), row, "kv_compact_fp8 kc")?;
    dispatch::check_len("kv_compact_fp8 vc", vc.len(), kc.len())?;
    dispatch::check_len(
        "kv_compact_fp8 k_scales",
        k_scales.len(),
        slots * n_kv_heads,
    )?;
    dispatch::check_len(
        "kv_compact_fp8 v_scales",
        v_scales.len(),
        slots * n_kv_heads,
    )?;
    if ring > slots {
        return Err(WgpuError::Shape(format!(
            "kv_compact_fp8: ring {ring} exceeds {slots} cache slots"
        )));
    }
    if ring == 0 && base + n_accept > slots {
        return Err(WgpuError::Shape(format!(
            "kv_compact_fp8: base {base} + n_accept {n_accept} exceeds {slots} cache slots"
        )));
    }
    for (i, &p) in path.iter().enumerate() {
        if p < 0 {
            return Err(WgpuError::Shape(format!(
                "kv_compact_fp8: path[{i}] = {p} is negative"
            )));
        }
        let srow = if ring > 0 {
            (base + p as usize) % ring
        } else {
            base + p as usize
        };
        if srow >= slots {
            return Err(WgpuError::Shape(format!(
                "kv_compact_fp8: path[{i}] = {p} with base {base} is outside {slots} cache slots"
            )));
        }
    }
    check_device(ctx, "kv_compact_fp8", 0)?;

    let row_words = row / 4;
    let total = n_accept * row_words;
    let scale_total = n_accept * n_kv_heads;
    let params = TvParams {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        ring: ring as u32,
        base: base as u32,
        n_accept: n_accept as u32,
        stride_words: row_words as u32,
        total: total as u32,
        ..Default::default()
    };

    let kcb = dispatch::storage_from_slice(ctx, "tvf.compact.kc", &bytes_to_words(kc));
    let vcb = dispatch::storage_from_slice(ctx, "tvf.compact.vc", &bytes_to_words(vc));
    let skb = dispatch::storage_zeroed(ctx, "tvf.compact.sk", (total * 4) as u64);
    let svb = dispatch::storage_zeroed(ctx, "tvf.compact.sv", (total * 4) as u64);
    let pb = dispatch::storage_from_slice(ctx, "tvf.compact.path", path);
    let ub = dispatch::uniform_from(ctx, "tvf.compact.params", &params);
    let kscb = dispatch::storage_from_slice(ctx, "tvf.compact.ksc", k_scales);
    let vscb = dispatch::storage_from_slice(ctx, "tvf.compact.vsc", v_scales);
    let sskb = dispatch::storage_zeroed(ctx, "tvf.compact.ssk", (scale_total * 4) as u64);
    let ssvb = dispatch::storage_zeroed(ctx, "tvf.compact.ssv", (scale_total * 4) as u64);

    let src = source();
    let byte_groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    let scale_groups = dispatch::workgroup_count_1d(ctx, scale_total as u64, WORKGROUP_SIZE);

    let gather_bytes: [(u32, &wgpu::Buffer); 6] = [
        (36, &kcb),
        (37, &vcb),
        (38, &skb),
        (39, &svb),
        (40, &pb),
        (41, &ub),
    ];
    let gather_scales: [(u32, &wgpu::Buffer); 6] = [
        (40, &pb),
        (41, &ub),
        (42, &kscb),
        (43, &vscb),
        (44, &sskb),
        (45, &ssvb),
    ];
    let scatter_bytes: [(u32, &wgpu::Buffer); 5] =
        [(36, &kcb), (37, &vcb), (38, &skb), (39, &svb), (41, &ub)];
    let scatter_scales: [(u32, &wgpu::Buffer); 5] = [
        (41, &ub),
        (42, &kscb),
        (43, &vscb),
        (44, &sskb),
        (45, &ssvb),
    ];

    dispatch::run(
        ctx,
        "nv_kernels_kv_gather_fp8",
        &src,
        "kv_gather_fp8",
        &gather_bytes,
        byte_groups,
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_kv_gather_scales_fp8",
        &src,
        "kv_gather_scales_fp8",
        &gather_scales,
        scale_groups,
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_kv_scatter_fp8",
        &src,
        "kv_scatter_fp8",
        &scatter_bytes,
        byte_groups,
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_kv_scatter_scales_fp8",
        &src,
        "kv_scatter_scales_fp8",
        &scatter_scales,
        scale_groups,
    )?;

    let got_k: Vec<u32> = dispatch::read_back(ctx, &kcb, kc.len().div_ceil(4))?;
    words_to_bytes(&got_k, kc);
    let got_v: Vec<u32> = dispatch::read_back(ctx, &vcb, vc.len().div_ceil(4))?;
    words_to_bytes(&got_v, vc);
    let got_ks: Vec<f32> = dispatch::read_back(ctx, &kscb, k_scales.len())?;
    k_scales.copy_from_slice(&got_ks);
    let got_vs: Vec<f32> = dispatch::read_back(ctx, &vscb, v_scales.len())?;
    v_scales.copy_from_slice(&got_vs);
    Ok(())
}

pub fn cpu_kv_append_fp8(
    k_src: &[u16],
    v_src: &[u16],
    kc: &mut [u8],
    vc: &mut [u8],
    k_scales: &mut [f32],
    v_scales: &mut [f32],
    n_committed: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k: usize,
    ring: usize,
) {
    for token in 0..k {
        for kvh in 0..n_kv_heads {
            let mut slot = n_committed + token;
            if ring > 0 {
                slot %= ring;
            }
            let base_src = (token * n_kv_heads + kvh) * head_dim;
            let base_dst = (slot * n_kv_heads + kvh) * head_dim;
            let mut amax_k = 0.0f32;
            let mut amax_v = 0.0f32;
            for d in 0..head_dim {
                let a = f32::from_bits((k_src[base_src + d] as u32) << 16).abs();
                if a > amax_k {
                    amax_k = a;
                }
                let b = f32::from_bits((v_src[base_src + d] as u32) << 16).abs();
                if b > amax_v {
                    amax_v = b;
                }
            }
            let inv_k = if amax_k > 0.0 {
                kv_fp8::div_rn(kv_fp8::FP8_E4M3_MAX, amax_k)
            } else {
                1.0
            };
            let inv_v = if amax_v > 0.0 {
                kv_fp8::div_rn(kv_fp8::FP8_E4M3_MAX, amax_v)
            } else {
                1.0
            };
            k_scales[slot * n_kv_heads + kvh] = if amax_k > 0.0 {
                kv_fp8::div_rn(amax_k, kv_fp8::FP8_E4M3_MAX)
            } else {
                1.0
            };
            v_scales[slot * n_kv_heads + kvh] = if amax_v > 0.0 {
                kv_fp8::div_rn(amax_v, kv_fp8::FP8_E4M3_MAX)
            } else {
                1.0
            };
            for d in 0..head_dim {
                let kv = f32::from_bits((k_src[base_src + d] as u32) << 16);
                let vv = f32::from_bits((v_src[base_src + d] as u32) << 16);
                kc[base_dst + d] = kv_fp8::encode_e4m3(kv * inv_k);
                vc[base_dst + d] = kv_fp8::encode_e4m3(vv * inv_v);
            }
        }
    }
}

pub fn cpu_kv_compact_fp8(
    kc: &mut [u8],
    vc: &mut [u8],
    k_scales: &mut [f32],
    v_scales: &mut [f32],
    path: &[i32],
    base: usize,
    n_accept: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ring: usize,
) {
    let row = n_kv_heads * head_dim;
    let mut sk = vec![0u8; n_accept * row];
    let mut sv = vec![0u8; n_accept * row];
    let mut ssk = vec![0f32; n_accept * n_kv_heads];
    let mut ssv = vec![0f32; n_accept * n_kv_heads];
    for i in 0..n_accept {
        let mut srow = base + path[i] as usize;
        if ring > 0 {
            srow %= ring;
        }
        sk[i * row..(i + 1) * row].copy_from_slice(&kc[srow * row..srow * row + row]);
        sv[i * row..(i + 1) * row].copy_from_slice(&vc[srow * row..srow * row + row]);
        for e in 0..n_kv_heads {
            ssk[i * n_kv_heads + e] = k_scales[srow * n_kv_heads + e];
            ssv[i * n_kv_heads + e] = v_scales[srow * n_kv_heads + e];
        }
    }
    for i in 0..n_accept {
        let mut drow = base + i;
        if ring > 0 {
            drow %= ring;
        }
        kc[drow * row..drow * row + row].copy_from_slice(&sk[i * row..(i + 1) * row]);
        vc[drow * row..drow * row + row].copy_from_slice(&sv[i * row..(i + 1) * row]);
        for e in 0..n_kv_heads {
            k_scales[drow * n_kv_heads + e] = ssk[i * n_kv_heads + e];
            v_scales[drow * n_kv_heads + e] = ssv[i * n_kv_heads + e];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_source_carries_the_shared_fp8_encoder() {
        let src = source();
        assert!(src.contains("fn kv_encode_e4m3"));
        assert!(src.contains("fn e4m3_decode"));
        assert!(src.contains("fn tree_verify_attn_fp8"));
    }

    #[test]
    fn cpu_append_round_trips_through_the_shared_codec() {
        let (nkv, hd, k) = (2usize, 8usize, 2usize);
        let src: Vec<u16> = (0..k * nkv * hd)
            .map(|i| half::bf16::from_f32(0.25 * (i as f32 + 1.0)).to_bits())
            .collect();
        let mut kc = vec![0u8; 4 * nkv * hd];
        let mut vc = vec![0u8; 4 * nkv * hd];
        let mut ks = vec![0f32; 4 * nkv];
        let mut vs = vec![0f32; 4 * nkv];
        cpu_kv_append_fp8(
            &src, &src, &mut kc, &mut vc, &mut ks, &mut vs, 1, nkv, hd, k, 0,
        );
        assert_eq!(kc, vc);
        assert!(ks[nkv] > 0.0);
        assert_eq!(&kc[..nkv * hd], &vec![0u8; nkv * hd][..]);
    }
}
