#![cfg(feature = "cuda")]

mod common;
use common::e4m3_decode;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn reference(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    mask: &[u8],
    nc: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    k: usize,
    window: usize,
) -> Vec<f32> {
    let group = nh / nkv;
    let mut out = vec![0f32; k * nh * hd];
    for qi in 0..k {
        let qpos = nc + qi;
        let win_start = if window > 0 {
            (qpos + 1).saturating_sub(window)
        } else {
            0
        };
        for h in 0..nh {
            let kvh = h / group;
            let qv = &q[(qi * nh + h) * hd..(qi * nh + h) * hd + hd];
            let mut positions: Vec<usize> = (win_start.min(nc)..nc).collect();
            for j in 0..k {
                if mask[qi * k + j] != 0 && (window == 0 || qpos - (nc + j) < window) {
                    positions.push(nc + j);
                }
            }
            let mut scores = Vec::with_capacity(positions.len());
            for &p in &positions {
                let kp = &kc[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                let mut s = 0f32;
                for d in 0..hd {
                    s += qv[d] * kp[d];
                }
                scores.push(s);
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f64;
            let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
            for &e in &exps {
                denom += e as f64;
            }
            for d in 0..hd {
                let mut acc = 0f64;
                for (idx, &p) in positions.iter().enumerate() {
                    let vp = vc[(p * nkv + kvh) * hd + d];
                    acc += (exps[idx] as f64) * (vp as f64);
                }
                out[(qi * nh + h) * hd + d] = (acc / denom) as f32;
            }
        }
    }
    out
}

fn run_case(nh: usize, nkv: usize, hd: usize, k: usize, nc: usize, window: usize) {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "tree_verify_attn: no CUDA device 0: {e}. This gate refuses to report \
                     success without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on \
                     purpose."
                );
            }
            eprintln!(
                "tree_verify_attn: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0: {e}"
            );
            return;
        }
    };
    let stream = ctx.default_stream();

    let max_seq = nc + k + 3;

    let mut rng = Rng(0x1234_5678 ^ (window as u64) << 32 ^ nc as u64);
    let q: Vec<f32> = (0..k * nh * hd).map(|_| rng.next()).collect();
    let kc: Vec<f32> = (0..max_seq * nkv * hd).map(|_| rng.next()).collect();
    let vc: Vec<f32> = (0..max_seq * nkv * hd).map(|_| rng.next()).collect();

    let mut mask = vec![0u8; k * k];
    for i in 0..k {
        for j in 0..=i {
            mask[i * k + j] = 1;
        }
    }

    let positions: Vec<i32> = (0..k).map(|j| (nc + j) as i32).collect();

    let q_bf: Vec<bf16> = q.iter().map(|&x| bf16::from_f32(x)).collect();
    let kc_bf: Vec<bf16> = kc.iter().map(|&x| bf16::from_f32(x)).collect();
    let vc_bf: Vec<bf16> = vc.iter().map(|&x| bf16::from_f32(x)).collect();

    let q_d = stream.clone_htod(&q_bf).unwrap();
    let kc_d = stream.clone_htod(&kc_bf).unwrap();
    let vc_d = stream.clone_htod(&vc_bf).unwrap();
    let nc_d = stream.clone_htod(&[nc as i32]).unwrap();
    let mask_d = stream.clone_htod(&mask).unwrap();
    let pos_d = stream.clone_htod(&positions).unwrap();
    let mut out_d = stream
        .clone_htod(&vec![bf16::from_f32(0.0); k * nh * hd])
        .unwrap();

    let rc = {
        let (qp, _a) = q_d.device_ptr(&stream);
        let (kp, _b) = kc_d.device_ptr(&stream);
        let (vp, _c) = vc_d.device_ptr(&stream);
        let (np, _e) = nc_d.device_ptr(&stream);
        let (mp, _f) = mask_d.device_ptr(&stream);
        let (pp, _h) = pos_d.device_ptr(&stream);
        let (op, _g) = out_d.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::tree_verify_attn_bf16(
                stream.cu_stream() as *mut _,
                qp as *const u16,
                kp as *const u16,
                vp as *const u16,
                np as *const i32,
                mp as *const u8,
                pp as *const i32,
                op as *mut u16,
                nh as i32,
                nkv as i32,
                hd as i32,
                k as i32,
                window as i32,
            )
        }
    };
    assert_eq!(rc, 0, "kernel returned {rc}");
    stream.synchronize().unwrap();

    let out_host: Vec<bf16> = stream.clone_dtoh(&out_d).unwrap();
    let got: Vec<f32> = out_host.iter().map(|x| x.to_f32()).collect();

    let q_ref: Vec<f32> = q_bf.iter().map(|x| x.to_f32()).collect();
    let kc_ref: Vec<f32> = kc_bf.iter().map(|x| x.to_f32()).collect();
    let vc_ref: Vec<f32> = vc_bf.iter().map(|x| x.to_f32()).collect();
    let want = reference(&q_ref, &kc_ref, &vc_ref, &mask, nc, nh, nkv, hd, k, window);

    let mut max_err = 0f32;
    for i in 0..got.len() {
        let e = (got[i] - want[i]).abs();
        if e > max_err {
            max_err = e;
        }
    }
    eprintln!("tree_verify_attn nc={nc} k={k} window={window} max abs err = {max_err:.5}");
    assert!(max_err < 2.0e-2, "max err {max_err} too large");
}

fn reference_fp8(
    q: &[f32],
    k8: &[u8],
    v8: &[u8],
    ks: &[f32],
    vs: &[f32],
    mask: &[u8],
    nc: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    k: usize,
) -> Vec<f32> {
    let group = nh / nkv;
    let mut out = vec![0f32; k * nh * hd];
    for qi in 0..k {
        for h in 0..nh {
            let kvh = h / group;
            let qv = &q[(qi * nh + h) * hd..(qi * nh + h) * hd + hd];
            let mut positions: Vec<usize> = (0..nc).collect();
            for j in 0..k {
                if mask[qi * k + j] != 0 {
                    positions.push(nc + j);
                }
            }
            let mut scores = Vec::with_capacity(positions.len());
            for &p in &positions {
                let kp = &k8[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                let mut s = 0f64;
                for d in 0..hd {
                    s += qv[d] as f64 * e4m3_decode(kp[d]) as f64;
                }
                scores.push(s * ks[p * nkv + kvh] as f64);
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
            let denom: f64 = exps.iter().sum();
            for d in 0..hd {
                let mut acc = 0f64;
                for (idx, &p) in positions.iter().enumerate() {
                    let vp =
                        e4m3_decode(v8[(p * nkv + kvh) * hd + d]) as f64 * vs[p * nkv + kvh] as f64;
                    acc += exps[idx] * vp;
                }
                out[(qi * nh + h) * hd + d] = (acc / denom) as f32;
            }
        }
    }
    out
}

fn run_case_fp8_global(nh: usize, nkv: usize, hd: usize, k: usize, nc: usize) {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "tree_verify_attn: no CUDA device 0: {e}. This gate refuses to report \
                     success without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on \
                     purpose."
                );
            }
            eprintln!(
                "tree_verify_attn: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0: {e}"
            );
            return;
        }
    };
    let stream = ctx.default_stream();
    let max_seq = nc + k;

    let mut rng = Rng(0x5eed_f00d ^ ((hd as u64) << 32) ^ nc as u64);
    let q: Vec<f32> = (0..k * nh * hd).map(|_| rng.next()).collect();
    let kc: Vec<f32> = (0..max_seq * nkv * hd).map(|_| rng.next()).collect();
    let vc: Vec<f32> = (0..max_seq * nkv * hd).map(|_| rng.next()).collect();

    let mut mask = vec![0u8; k * k];
    for i in 0..k {
        for j in 0..=i {
            mask[i * k + j] = 1;
        }
    }
    let positions: Vec<i32> = (0..k).map(|j| (nc + j) as i32).collect();

    let q_bf: Vec<bf16> = q.iter().map(|&x| bf16::from_f32(x)).collect();
    let kc_bf: Vec<bf16> = kc.iter().map(|&x| bf16::from_f32(x)).collect();
    let vc_bf: Vec<bf16> = vc.iter().map(|&x| bf16::from_f32(x)).collect();

    let q_d = stream.clone_htod(&q_bf).unwrap();
    let kc_d = stream.clone_htod(&kc_bf).unwrap();
    let vc_d = stream.clone_htod(&vc_bf).unwrap();
    let nc_d = stream.clone_htod(&[nc as i32]).unwrap();
    let mask_d = stream.clone_htod(&mask).unwrap();
    let pos_d = stream.clone_htod(&positions).unwrap();
    let mut out_d = stream
        .clone_htod(&vec![bf16::from_f32(0.0); k * nh * hd])
        .unwrap();

    let mut k8_d = stream.alloc_zeros::<u8>(max_seq * nkv * hd).unwrap();
    let mut v8_d = stream.alloc_zeros::<u8>(max_seq * nkv * hd).unwrap();
    let mut ks_d = stream.alloc_zeros::<f32>(max_seq * nkv).unwrap();
    let mut vs_d = stream.alloc_zeros::<f32>(max_seq * nkv).unwrap();
    let zero_d = stream.clone_htod(&[0i32]).unwrap();
    let rc = {
        let (kp, _a) = kc_d.device_ptr(&stream);
        let (vp, _b) = vc_d.device_ptr(&stream);
        let (k8, _c) = k8_d.device_ptr_mut(&stream);
        let (v8, _d) = v8_d.device_ptr_mut(&stream);
        let (ks, _e) = ks_d.device_ptr_mut(&stream);
        let (vs, _f) = vs_d.device_ptr_mut(&stream);
        let (zp, _g) = zero_d.device_ptr(&stream);
        unsafe {
            nv_kernels::cuda::kv_append_fp8(
                stream.cu_stream() as *mut _,
                kp as *const u16,
                vp as *const u16,
                k8 as *mut u8,
                v8 as *mut u8,
                ks as *mut f32,
                vs as *mut f32,
                zp as *const i32,
                max_seq as i32,
                nkv as i32,
                hd as i32,
                0,
            )
        }
    };
    assert_eq!(rc, 0, "kv_append_fp8 rc={rc}");

    let rc = {
        let (qp, _a) = q_d.device_ptr(&stream);
        let (k8, _b) = k8_d.device_ptr(&stream);
        let (v8, _c) = v8_d.device_ptr(&stream);
        let (ks, _d) = ks_d.device_ptr(&stream);
        let (vs, _e) = vs_d.device_ptr(&stream);
        let (np, _f) = nc_d.device_ptr(&stream);
        let (mp, _g) = mask_d.device_ptr(&stream);
        let (pp, _h) = pos_d.device_ptr(&stream);
        let (op, _i) = out_d.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::tree_verify_attn_fp8(
                stream.cu_stream() as *mut _,
                qp as *const u16,
                k8 as *const u8,
                v8 as *const u8,
                ks as *const f32,
                vs as *const f32,
                np as *const i32,
                mp as *const u8,
                pp as *const i32,
                op as *mut u16,
                nh as i32,
                nkv as i32,
                hd as i32,
                k as i32,
                0,
                0,
            )
        }
    };
    assert_eq!(rc, 0, "tree_verify_attn_fp8 rc={rc}");
    stream.synchronize().unwrap();

    let out_host: Vec<bf16> = stream.clone_dtoh(&out_d).unwrap();
    let got: Vec<f32> = out_host.iter().map(|x| x.to_f32()).collect();

    let k8h: Vec<u8> = stream.clone_dtoh(&k8_d).unwrap();
    let v8h: Vec<u8> = stream.clone_dtoh(&v8_d).unwrap();
    let ksh: Vec<f32> = stream.clone_dtoh(&ks_d).unwrap();
    let vsh: Vec<f32> = stream.clone_dtoh(&vs_d).unwrap();
    let q_ref: Vec<f32> = q_bf.iter().map(|x| x.to_f32()).collect();
    let want = reference_fp8(&q_ref, &k8h, &v8h, &ksh, &vsh, &mask, nc, nh, nkv, hd, k);

    let mut max_err = 0f32;
    for i in 0..got.len() {
        let e = (got[i] - want[i]).abs();
        if e > max_err {
            max_err = e;
        }
    }
    eprintln!(
        "tree_verify_attn_fp8 nh={nh} nkv={nkv} hd={hd} nc={nc} k={k} max abs err = {max_err:.5}"
    );
    assert!(max_err < 4.0e-2, "max err {max_err} too large");
}

#[test]
fn tree_verify_attn_matches_reference() {
    run_case(8, 2, 64, 5, 11, 0);
}

#[test]
fn tree_verify_attn_fp8_global_layer_shape_large_nc() {
    run_case_fp8_global(8, 4, 512, 8, 3801);
}

#[test]
fn tree_verify_attn_fp8_hd256_shape() {
    run_case_fp8_global(8, 4, 256, 5, 1027);
}

#[test]
fn tree_verify_attn_window_clamps_prefix() {
    run_case(8, 2, 64, 5, 11, 7);
}

#[test]
fn tree_verify_attn_window_clamps_tree_keys() {
    run_case(8, 2, 64, 13, 0, 4);
}

#[test]
fn tree_verify_attn_window_larger_than_context_is_noop() {
    run_case(8, 2, 64, 5, 11, 1024);
}
