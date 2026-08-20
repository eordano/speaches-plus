#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_u16, unpack_u16_by_element as unpack_u16, pack_u8_min_one_word as pack_u8};

pub const WGSL: &str = include_str!("../../../wgsl/verify_fused_norms.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const MAX_HEAD_DIM: usize = 512;
pub const FP8_E4M3_MAX: f32 = 448.0;

pub const QKV_PREP_ENTRY: &str = "verify_qkv_prep";
pub const RMSNORM2_ENTRY: &str = "rmsnorm2_residual_bf16";
pub const RMSNORM_SCALE_ENTRY: &str = "rmsnorm_residual_scale_bf16";

const SCRATCH_BYTES: u32 = 512 * 4 * 2 + WORKGROUP_SIZE * 4 + 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VfParams {
    k: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    ring: u32,
    stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    eps: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VfrParams {
    hidden: u32,
    batch: u32,
    words: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VfsParams {
    hidden: u32,
    batch: u32,
    words: u32,
    eps: f32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "verify_fused_norms", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_grid(ctx: &WgpuContext, x: usize, y: usize) -> Result<()> {
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    if x > limit || y > limit {
        return Err(WgpuError::Unsupported(format!(
            "verify_fused_norms grid {x}x{y} exceeds max_compute_workgroups_per_dimension {limit}"
        )));
    }
    Ok(())
}

fn unpack_u8(words: &[u32], dst: &mut [u8]) {
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = ((words[i >> 2] >> (8 * (i & 3))) & 0xff) as u8;
    }
}

pub fn verify_qkv_prep(
    ctx: &WgpuContext,
    qkv: &[u16],
    qkv_stride: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,
    qw: &[u16],
    kw: &[u16],
    vw: &[u16],
    eps: f32,
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    q_out: &mut [u16],
    kc: &mut [u8],
    vc: &mut [u8],
    k_scale: &mut [f32],
    v_scale: &mut [f32],
    n_committed: &[i32],
    k: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    ring: usize,
) -> Result<()> {
    if k == 0 || nq == 0 || nkv == 0 || hd == 0 {
        return Ok(());
    }
    if hd > MAX_HEAD_DIM || !hd.is_multiple_of(2) || hd / 2 > WORKGROUP_SIZE as usize {
        return Err(WgpuError::Unsupported(format!(
            "verify_qkv_prep head_dim {hd} outside the CUDA contract (even, <= {MAX_HEAD_DIM})"
        )));
    }
    if !hd.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "verify_qkv_prep needs head_dim % 4 == 0 for whole-word fp8 writes; got {hd}"
        )));
    }
    check_device(ctx)?;
    check_grid(ctx, nq + nkv, k)?;

    let half = hd / 2;
    let need_q = q_off + nq * hd;
    let need_k = k_off + nkv * hd;
    let need_v = v_off + nkv * hd;
    let need = (k - 1) * qkv_stride + need_q.max(need_k).max(need_v);
    if qkv.len() < need {
        return Err(WgpuError::Shape(format!(
            "verify_qkv_prep qkv: got {} elements, need at least {need}",
            qkv.len()
        )));
    }
    dispatch::check_len("verify_qkv_prep qw", qw.len(), hd)?;
    dispatch::check_len("verify_qkv_prep kw", kw.len(), hd)?;
    dispatch::check_len("verify_qkv_prep vw", vw.len(), hd)?;
    dispatch::check_len("verify_qkv_prep q_out", q_out.len(), k * nq * hd)?;
    if positions.len() < k {
        return Err(WgpuError::Shape(format!(
            "verify_qkv_prep positions: got {}, need {k}",
            positions.len()
        )));
    }
    let max_pos = positions[..k].iter().copied().max().unwrap_or(0).max(0) as usize;
    if cos_tbl.len() < (max_pos + 1) * half || sin_tbl.len() < (max_pos + 1) * half {
        return Err(WgpuError::Shape(format!(
            "verify_qkv_prep rope tables: need {} floats for max position {max_pos}",
            (max_pos + 1) * half
        )));
    }
    let Some(nc) = n_committed.first() else {
        return Err(WgpuError::Shape(
            "verify_qkv_prep n_committed is empty".to_string(),
        ));
    };
    let nc = (*nc).max(0) as usize;
    if kc.len() != vc.len() || !kc.len().is_multiple_of(nkv * hd) {
        return Err(WgpuError::Shape(format!(
            "verify_qkv_prep kv caches: kc {} vc {} not a slot multiple of {}",
            kc.len(),
            vc.len(),
            nkv * hd
        )));
    }
    let slots = kc.len() / (nkv * hd);
    if ring > 0 {
        if ring > slots {
            return Err(WgpuError::Shape(format!(
                "verify_qkv_prep ring {ring} exceeds cache slots {slots}"
            )));
        }
    } else if nc + k > slots {
        return Err(WgpuError::Shape(format!(
            "verify_qkv_prep cache holds {slots} slots but needs {}",
            nc + k
        )));
    }
    dispatch::check_len("verify_qkv_prep k_scale", k_scale.len(), slots * nkv)?;
    dispatch::check_len("verify_qkv_prep v_scale", v_scale.len(), slots * nkv)?;

    let params = VfParams {
        k: k as u32,
        nq: nq as u32,
        nkv: nkv as u32,
        hd: hd as u32,
        ring: ring as u32,
        stride: qkv_stride as u32,
        q_off: q_off as u32,
        k_off: k_off as u32,
        v_off: v_off as u32,
        eps,
        pad0: 0,
        pad1: 0,
    };

    let qkvb = dispatch::storage_from_slice(ctx, "vf-qkv", &pack_u16(qkv));
    let qwb = dispatch::storage_from_slice(ctx, "vf-qw", &pack_u16(qw));
    let kwb = dispatch::storage_from_slice(ctx, "vf-kw", &pack_u16(kw));
    let vwb = dispatch::storage_from_slice(ctx, "vf-vw", &pack_u16(vw));
    let cosb = dispatch::storage_from_slice(ctx, "vf-cos", cos_tbl);
    let sinb = dispatch::storage_from_slice(ctx, "vf-sin", sin_tbl);
    let posb = dispatch::storage_from_slice(ctx, "vf-pos", &positions[..k]);
    let qob = dispatch::storage_zeroed(ctx, "vf-qout", (k * nq * hd * 2) as u64);
    let kcb = dispatch::storage_from_slice(ctx, "vf-kc", &pack_u8(kc));
    let vcb = dispatch::storage_from_slice(ctx, "vf-vc", &pack_u8(vc));
    let ksb = dispatch::storage_from_slice(ctx, "vf-kscale", &*k_scale);
    let vsb = dispatch::storage_from_slice(ctx, "vf-vscale", &*v_scale);
    let ncb = dispatch::storage_from_slice(ctx, "vf-nc", n_committed);
    let pb = dispatch::uniform_from(ctx, "vf-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_verify_qkv_prep",
        &compose(WGSL),
        QKV_PREP_ENTRY,
        &[
            (0, &qkvb),
            (1, &qwb),
            (2, &kwb),
            (3, &vwb),
            (4, &cosb),
            (5, &sinb),
            (6, &posb),
            (7, &qob),
            (8, &kcb),
            (9, &vcb),
            (10, &ksb),
            (11, &vsb),
            (12, &ncb),
            (13, &pb),
        ],
        ((nq + nkv) as u32, k as u32, 1),
    )?;

    let qw_words: Vec<u32> = dispatch::read_back(ctx, &qob, k * nq * hd / 2)?;
    unpack_u16(&qw_words, q_out);
    let kc_words: Vec<u32> = dispatch::read_back(ctx, &kcb, kc.len().div_ceil(4))?;
    unpack_u8(&kc_words, kc);
    let vc_words: Vec<u32> = dispatch::read_back(ctx, &vcb, vc.len().div_ceil(4))?;
    unpack_u8(&vc_words, vc);
    let ks: Vec<f32> = dispatch::read_back(ctx, &ksb, k_scale.len())?;
    k_scale.copy_from_slice(&ks);
    let vs: Vec<f32> = dispatch::read_back(ctx, &vsb, v_scale.len())?;
    v_scale.copy_from_slice(&vs);
    Ok(())
}

pub fn rmsnorm2_residual_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    residual: &[u16],
    w1: &[u16],
    w2: &[u16],
    sum_out: &mut [u16],
    normed_out: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    if !hidden.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "rmsnorm2_residual_bf16 hidden must be even so whole u32 words are written; got {hidden}"
        )));
    }
    check_device(ctx)?;
    dispatch::check_len("rmsnorm2 x", x.len(), batch * hidden)?;
    dispatch::check_len("rmsnorm2 residual", residual.len(), batch * hidden)?;
    dispatch::check_len("rmsnorm2 w1", w1.len(), hidden)?;
    dispatch::check_len("rmsnorm2 w2", w2.len(), hidden)?;
    dispatch::check_len("rmsnorm2 sum_out", sum_out.len(), batch * hidden)?;
    dispatch::check_len("rmsnorm2 normed_out", normed_out.len(), batch * hidden)?;

    let words = hidden / 2;
    let params = VfrParams {
        hidden: hidden as u32,
        batch: batch as u32,
        words: words as u32,
        eps,
    };
    let xb = dispatch::storage_from_slice(ctx, "vfr-x", &pack_u16(x));
    let rb = dispatch::storage_from_slice(ctx, "vfr-res", &pack_u16(residual));
    let w1b = dispatch::storage_from_slice(ctx, "vfr-w1", &pack_u16(w1));
    let w2b = dispatch::storage_from_slice(ctx, "vfr-w2", &pack_u16(w2));
    let sb = dispatch::storage_zeroed(ctx, "vfr-sum", (batch * words * 4) as u64);
    let nb = dispatch::storage_zeroed(ctx, "vfr-norm", (batch * words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "vfr-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm2_residual_bf16",
        &compose(WGSL),
        RMSNORM2_ENTRY,
        &[
            (14, &xb),
            (15, &rb),
            (16, &w1b),
            (17, &w2b),
            (18, &sb),
            (19, &nb),
            (20, &pb),
        ],
        groups,
    )?;

    let sw: Vec<u32> = dispatch::read_back(ctx, &sb, batch * words)?;
    unpack_u16(&sw, sum_out);
    let nw: Vec<u32> = dispatch::read_back(ctx, &nb, batch * words)?;
    unpack_u16(&nw, normed_out);
    Ok(())
}

pub fn rmsnorm_residual_scale_bf16(
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
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    if !hidden.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "rmsnorm_residual_scale_bf16 hidden must be even so whole u32 words are written; got {hidden}"
        )));
    }
    check_device(ctx)?;
    dispatch::check_len("rmsnorm_residual_scale x", x.len(), batch * hidden)?;
    dispatch::check_len(
        "rmsnorm_residual_scale residual",
        residual.len(),
        batch * hidden,
    )?;
    dispatch::check_len("rmsnorm_residual_scale w", w.len(), hidden)?;
    dispatch::check_len("rmsnorm_residual_scale out", out.len(), batch * hidden)?;

    let words = hidden / 2;
    let params = VfsParams {
        hidden: hidden as u32,
        batch: batch as u32,
        words: words as u32,
        eps,
        scale,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let xb = dispatch::storage_from_slice(ctx, "vfs-x", &pack_u16(x));
    let rb = dispatch::storage_from_slice(ctx, "vfs-res", &pack_u16(residual));
    let wb = dispatch::storage_from_slice(ctx, "vfs-w", &pack_u16(w));
    let ob = dispatch::storage_zeroed(ctx, "vfs-out", (batch * words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "vfs-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm_residual_scale_bf16",
        &compose(WGSL),
        RMSNORM_SCALE_ENTRY,
        &[(21, &xb), (22, &rb), (23, &wb), (24, &ob), (25, &pb)],
        groups,
    )?;

    let ow: Vec<u32> = dispatch::read_back(ctx, &ob, batch * words)?;
    unpack_u16(&ow, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_keep_uniform_layout_multiples() {
        assert_eq!(std::mem::size_of::<VfParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<VfrParams>(), 16);
        assert_eq!(std::mem::size_of::<VfsParams>() % 16, 0);
    }

    #[test]
    fn byte_word_packing_round_trips() {
        let src: Vec<u8> = (0u16..37)
            .map(|i| (i.wrapping_mul(97) & 0xff) as u8)
            .collect();
        let words = pack_u8(&src);
        let mut back = vec![0u8; src.len()];
        unpack_u8(&words, &mut back);
        assert_eq!(back, src);
        let src16: Vec<u16> = (0u16..21).map(|i| i.wrapping_mul(1031)).collect();
        let w16 = pack_u16(&src16);
        let mut back16 = vec![0u16; src16.len()];
        unpack_u16(&w16, &mut back16);
        assert_eq!(back16, src16);
    }
}
