#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::sync::Arc;

fn rng_bf16(seed: u64, n: usize, scale: f32) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            bf16::from_f32((u - 1.0) * scale).to_bits()
        })
        .collect()
}

fn rng_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(7);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / ((1u64 << 32) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn htod(stream: &Arc<CudaStream>, v: &[u16]) -> CudaSlice<u16> {
    #[allow(deprecated)]
    stream.clone_htod(v).unwrap()
}

fn htod_f32(stream: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<f32> {
    #[allow(deprecated)]
    stream.clone_htod(v).unwrap()
}

fn dtoh_u16(stream: &Arc<CudaStream>, d: &CudaSlice<u16>) -> Vec<u16> {
    #[allow(deprecated)]
    stream.memcpy_dtov(d).unwrap()
}

fn dtoh_u8(stream: &Arc<CudaStream>, d: &CudaSlice<u8>) -> Vec<u8> {
    #[allow(deprecated)]
    stream.memcpy_dtov(d).unwrap()
}

fn dtoh_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    #[allow(deprecated)]
    stream.memcpy_dtov(d).unwrap()
}

fn qkv_prep_case(k: usize, nq: usize, nkv: usize, hd: usize, ring: i32, committed: i32) {
    qkv_prep_case_seeded(k, nq, nkv, hd, ring, committed, 0)
}

#[allow(clippy::too_many_arguments)]
fn qkv_prep_case_seeded(
    k: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    ring: i32,
    committed: i32,
    seed: u64,
) {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "parity_verify_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("parity_verify_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let eps = 1e-6f32;
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let width = q_dim + 2 * kv_dim;
    let half = hd / 2;
    let max_pos = 64usize;

    let fused_h = rng_bf16(11 ^ seed.wrapping_mul(2654435761), k * width, 2.0);
    let qw_h = rng_bf16(21 ^ seed.wrapping_mul(97), hd, 1.0);
    let kw_h = rng_bf16(22 ^ seed.wrapping_mul(193), hd, 1.0);
    let vw_h = rng_bf16(23 ^ seed.wrapping_mul(389), hd, 1.0);
    let cos_h = rng_f32(31 ^ seed.wrapping_mul(769), max_pos * half);
    let sin_h = rng_f32(32 ^ seed.wrapping_mul(1543), max_pos * half);
    let pos_h: Vec<i32> = (0..k)
        .map(|i| (committed + i as i32) % max_pos as i32)
        .collect();

    let slots = if ring > 0 {
        ring as usize
    } else {
        (committed as usize) + k + 4
    };

    let d_fused = htod(&stream, &fused_h);
    let d_qw = htod(&stream, &qw_h);
    let d_kw = htod(&stream, &kw_h);
    let d_vw = htod(&stream, &vw_h);
    let d_cos = htod_f32(&stream, &cos_h);
    let d_sin = htod_f32(&stream, &sin_h);
    #[allow(deprecated)]
    let d_pos: CudaSlice<i32> = stream.clone_htod(&pos_h).unwrap();
    #[allow(deprecated)]
    let d_nc: CudaSlice<i32> = stream.clone_htod(&[committed]).unwrap();

    let mut q_raw_h = vec![0u16; k * q_dim];
    let mut k_raw_h = vec![0u16; k * kv_dim];
    let mut v_raw_h = vec![0u16; k * kv_dim];
    for t in 0..k {
        q_raw_h[t * q_dim..(t + 1) * q_dim].copy_from_slice(&fused_h[t * width..t * width + q_dim]);
        k_raw_h[t * kv_dim..(t + 1) * kv_dim]
            .copy_from_slice(&fused_h[t * width + q_dim..t * width + q_dim + kv_dim]);
        v_raw_h[t * kv_dim..(t + 1) * kv_dim]
            .copy_from_slice(&fused_h[t * width + q_dim + kv_dim..t * width + width]);
    }
    let d_qraw = htod(&stream, &q_raw_h);
    let d_kraw = htod(&stream, &k_raw_h);
    let d_vraw = htod(&stream, &v_raw_h);
    let mut d_qn: CudaSlice<u16> = stream.alloc_zeros(k * q_dim).unwrap();
    let mut d_kn: CudaSlice<u16> = stream.alloc_zeros(k * kv_dim).unwrap();
    let mut d_vn: CudaSlice<u16> = stream.alloc_zeros(k * kv_dim).unwrap();
    let mut d_qrot: CudaSlice<u16> = stream.alloc_zeros(k * q_dim).unwrap();
    let mut d_krot: CudaSlice<u16> = stream.alloc_zeros(k * kv_dim).unwrap();
    let mut kc_ref: CudaSlice<u8> = stream.alloc_zeros(slots * kv_dim).unwrap();
    let mut vc_ref: CudaSlice<u8> = stream.alloc_zeros(slots * kv_dim).unwrap();
    let mut ks_ref: CudaSlice<f32> = stream.alloc_zeros(slots * nkv).unwrap();
    let mut vs_ref: CudaSlice<f32> = stream.alloc_zeros(slots * nkv).unwrap();

    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pqr, _a) = d_qraw.device_ptr(&stream);
        let (pqw, _b) = d_qw.device_ptr(&stream);
        let (pqn, _c) = d_qn.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                s,
                pqr as *const u16,
                pqw as *const u16,
                pqn as *mut u16,
                k * nq,
                hd,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pkr, _a) = d_kraw.device_ptr(&stream);
        let (pkw, _b) = d_kw.device_ptr(&stream);
        let (pkn, _c) = d_kn.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                s,
                pkr as *const u16,
                pkw as *const u16,
                pkn as *mut u16,
                k * nkv,
                hd,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pvr, _a) = d_vraw.device_ptr(&stream);
        let (pvw, _b) = d_vw.device_ptr(&stream);
        let (pvn, _c) = d_vn.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                s,
                pvr as *const u16,
                pvw as *const u16,
                pvn as *mut u16,
                k * nkv,
                hd,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pqn, _a) = d_qn.device_ptr(&stream);
        let (pkn, _b) = d_kn.device_ptr(&stream);
        let (pqo, _c) = d_qrot.device_ptr_mut(&stream);
        let (pko, _d) = d_krot.device_ptr_mut(&stream);
        let (pc, _e) = d_cos.device_ptr(&stream);
        let (ps, _f) = d_sin.device_ptr(&stream);
        let (pp, _g) = d_pos.device_ptr(&stream);
        unsafe {
            cuda::rope_bf16_oop(
                s,
                pqn as *const u16,
                pkn as *const u16,
                pqo as *mut u16,
                pko as *mut u16,
                pc as *const f32,
                ps as *const f32,
                pp as *const i32,
                k,
                nq,
                nkv,
                hd,
            )
        }
    };
    assert_eq!(rc, 0);
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pko, _a) = d_krot.device_ptr(&stream);
        let (pvn, _b) = d_vn.device_ptr(&stream);
        let (pkc, _c) = kc_ref.device_ptr_mut(&stream);
        let (pvc, _d) = vc_ref.device_ptr_mut(&stream);
        let (pks, _e) = ks_ref.device_ptr_mut(&stream);
        let (pvs, _f) = vs_ref.device_ptr_mut(&stream);
        let (pnc, _g) = d_nc.device_ptr(&stream);
        unsafe {
            cuda::kv_append_fp8(
                s,
                pko as *const u16,
                pvn as *const u16,
                pkc as *mut u8,
                pvc as *mut u8,
                pks as *mut f32,
                pvs as *mut f32,
                pnc as *const i32,
                k as i32,
                nkv as i32,
                hd as i32,
                ring,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();

    let mut d_qfused: CudaSlice<u16> = stream.alloc_zeros(k * q_dim).unwrap();
    let mut kc_new: CudaSlice<u8> = stream.alloc_zeros(slots * kv_dim).unwrap();
    let mut vc_new: CudaSlice<u8> = stream.alloc_zeros(slots * kv_dim).unwrap();
    let mut ks_new: CudaSlice<f32> = stream.alloc_zeros(slots * nkv).unwrap();
    let mut vs_new: CudaSlice<f32> = stream.alloc_zeros(slots * nkv).unwrap();

    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pf, _a) = d_fused.device_ptr(&stream);
        let (pqw, _b) = d_qw.device_ptr(&stream);
        let (pkw, _c) = d_kw.device_ptr(&stream);
        let (pvw, _d) = d_vw.device_ptr(&stream);
        let (pc, _e) = d_cos.device_ptr(&stream);
        let (ps, _f) = d_sin.device_ptr(&stream);
        let (pp, _g) = d_pos.device_ptr(&stream);
        let (pqo, _h) = d_qfused.device_ptr_mut(&stream);
        let (pkc, _i) = kc_new.device_ptr_mut(&stream);
        let (pvc, _j) = vc_new.device_ptr_mut(&stream);
        let (pks, _l) = ks_new.device_ptr_mut(&stream);
        let (pvs, _m) = vs_new.device_ptr_mut(&stream);
        let (pnc, _n) = d_nc.device_ptr(&stream);
        unsafe {
            cuda::verify_qkv_prep(
                s,
                pf as *const u16,
                width as i64,
                0,
                q_dim as i64,
                (q_dim + kv_dim) as i64,
                pqw as *const u16,
                pkw as *const u16,
                pvw as *const u16,
                eps,
                pc as *const f32,
                ps as *const f32,
                pp as *const i32,
                pqo as *mut u16,
                pkc as *mut u8,
                pvc as *mut u8,
                pks as *mut f32,
                pvs as *mut f32,
                pnc as *const i32,
                k as i32,
                nq as i32,
                nkv as i32,
                hd as i32,
                ring,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();

    assert_eq!(
        dtoh_u16(&stream, &d_qrot),
        dtoh_u16(&stream, &d_qfused),
        "q_rot mismatch (k={k} nq={nq} nkv={nkv} hd={hd} ring={ring})"
    );
    assert_eq!(
        dtoh_u8(&stream, &kc_ref),
        dtoh_u8(&stream, &kc_new),
        "kc mismatch"
    );
    assert_eq!(
        dtoh_u8(&stream, &vc_ref),
        dtoh_u8(&stream, &vc_new),
        "vc mismatch"
    );
    let ksr = dtoh_f32(&stream, &ks_ref);
    let ksn = dtoh_f32(&stream, &ks_new);
    assert_eq!(
        ksr.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        ksn.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "k_scale mismatch"
    );
    let vsr = dtoh_f32(&stream, &vs_ref);
    let vsn = dtoh_f32(&stream, &vs_new);
    assert_eq!(
        vsr.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        vsn.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "v_scale mismatch"
    );
}

#[test]
fn verify_qkv_prep_bitexact_hd256() {
    qkv_prep_case(4, 8, 4, 256, 0, 3);
}

#[test]
fn verify_qkv_prep_bitexact_hd512() {
    qkv_prep_case(4, 4, 2, 512, 0, 7);
}

#[test]
fn verify_qkv_prep_bitexact_ring() {
    qkv_prep_case(4, 8, 4, 256, 24, 21);
}

#[test]
fn verify_qkv_prep_bitexact_m1() {
    qkv_prep_case(1, 8, 4, 256, 0, 0);
}

#[test]
fn verify_qkv_prep_bitexact_ring_alias_last_writer() {
    for seed in 0..2u64 {
        qkv_prep_case_seeded(5, 4, 2, 128, 4, 1000, seed);
        qkv_prep_case_seeded(7, 4, 2, 256, 3, 21, seed.wrapping_add(11));
        qkv_prep_case_seeded(9, 8, 4, 64, 2, 5, seed.wrapping_add(23));
    }
}

#[test]
fn rmsnorm2_residual_bitexact() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "parity_verify_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("parity_verify_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let batch = 4usize;
    let hidden = 5376usize;
    let eps = 1e-6f32;

    let x_h = rng_bf16(41, batch * hidden, 3.0);
    let res_h = rng_bf16(42, batch * hidden, 3.0);
    let w1_h = rng_bf16(43, hidden, 1.0);
    let w2_h = rng_bf16(44, hidden, 1.0);

    let d_x = htod(&stream, &x_h);
    let d_res = htod(&stream, &res_h);
    let d_w1 = htod(&stream, &w1_h);
    let d_w2 = htod(&stream, &w2_h);

    let mut d_t: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let mut d_res_copy = htod(&stream, &res_h);
    let mut d_norm_ref: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();

    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (px, _a) = d_x.device_ptr(&stream);
        let (pw, _b) = d_w1.device_ptr(&stream);
        let (pt, _c) = d_t.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                s,
                px as *const u16,
                pw as *const u16,
                pt as *mut u16,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pt, _a) = d_t.device_ptr(&stream);
        let (pr, _b) = d_res_copy.device_ptr_mut(&stream);
        let (pw, _c) = d_w2.device_ptr(&stream);
        let (po, _d) = d_norm_ref.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_residual_bf16(
                s,
                pt as *const u16,
                pr as *mut u16,
                pw as *const u16,
                po as *mut u16,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();

    let mut d_sum_new: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let mut d_norm_new: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (px, _a) = d_x.device_ptr(&stream);
        let (pr, _b) = d_res.device_ptr(&stream);
        let (pw1, _c) = d_w1.device_ptr(&stream);
        let (pw2, _d) = d_w2.device_ptr(&stream);
        let (psum, _e) = d_sum_new.device_ptr_mut(&stream);
        let (pnorm, _f) = d_norm_new.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm2_residual_bf16(
                s,
                px as *const u16,
                pr as *const u16,
                pw1 as *const u16,
                pw2 as *const u16,
                psum as *mut u16,
                pnorm as *mut u16,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();

    assert_eq!(
        dtoh_u16(&stream, &d_res_copy),
        dtoh_u16(&stream, &d_sum_new),
        "sum mismatch"
    );
    assert_eq!(
        dtoh_u16(&stream, &d_norm_ref),
        dtoh_u16(&stream, &d_norm_new),
        "normed mismatch"
    );
}

#[test]
fn rmsnorm_residual_scale_bitexact() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "parity_verify_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("parity_verify_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let batch = 4usize;
    let hidden = 5376usize;
    let eps = 1e-6f32;
    let scale = 0.7071f32;

    let x_h = rng_bf16(51, batch * hidden, 3.0);
    let res_h = rng_bf16(52, batch * hidden, 3.0);
    let w_h = rng_bf16(53, hidden, 1.0);

    let d_x = htod(&stream, &x_h);
    let d_res = htod(&stream, &res_h);
    let d_w = htod(&stream, &w_h);

    let mut d_norm: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let mut d_ref: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (px, _a) = d_x.device_ptr(&stream);
        let (pw, _b) = d_w.device_ptr(&stream);
        let (pn, _c) = d_norm.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                s,
                px as *const u16,
                pw as *const u16,
                pn as *mut u16,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0);
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (pr, _a) = d_res.device_ptr(&stream);
        let (pn, _b) = d_norm.device_ptr(&stream);
        let (po, _c) = d_ref.device_ptr_mut(&stream);
        unsafe {
            cuda::residual_add_scale_bf16(
                s,
                pr as *const u16,
                pn as *const u16,
                po as *mut u16,
                scale,
                batch * hidden,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();

    let mut d_new: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let rc = {
        let s = stream.cu_stream() as *mut std::ffi::c_void;
        let (px, _a) = d_x.device_ptr(&stream);
        let (pr, _b) = d_res.device_ptr(&stream);
        let (pw, _c) = d_w.device_ptr(&stream);
        let (po, _d) = d_new.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_residual_scale_bf16(
                s,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                po as *mut u16,
                batch,
                hidden,
                eps,
                scale,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();

    assert_eq!(
        dtoh_u16(&stream, &d_ref),
        dtoh_u16(&stream, &d_new),
        "fused post_ff mismatch"
    );
}

#[test]
#[ignore]
fn verify_qkv_prep_stress_ties() {
    let iters: usize = std::env::var("NV_B3_STRESS_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    for i in 0..iters {
        let seed = i as u64 + 1;
        qkv_prep_case_seeded(8, 32, 16, 256, 0, (i % 40) as i32, seed);
        qkv_prep_case_seeded(8, 32, 4, 512, 0, (i % 40) as i32, seed.wrapping_add(7777));
        qkv_prep_case_seeded(
            4,
            32,
            16,
            256,
            2176,
            1000 + (i % 900) as i32,
            seed.wrapping_add(31337),
        );
        qkv_prep_case_seeded(6, 32, 8, 256, 4, (i % 40) as i32, seed.wrapping_add(999));
        if i % 50 == 0 {
            eprintln!("stress iter {i}/{iters}");
        }
    }
}
