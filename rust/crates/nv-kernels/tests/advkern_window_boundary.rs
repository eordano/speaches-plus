#![cfg(feature = "cuda")]

mod common;
use common::e4m3_decode;
use common::lcg_f32;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    kat: &dyn Fn(usize, usize, usize) -> f32,
    vat: &dyn Fn(usize, usize, usize) -> f32,
    kscale: &dyn Fn(usize, usize) -> f64,
    mask: &[u8],
    positions: &[i32],
    nc: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    k: usize,
    window: i32,
) -> Vec<f32> {
    let group = nh / nkv;
    let mut out = vec![0f32; k * nh * hd];
    for qi in 0..k {
        let qpos = positions[qi];
        let win_start = if window > 0 {
            (qpos - (window - 1)).max(0) as usize
        } else {
            0
        };
        for h in 0..nh {
            let kvh = h / group;
            let qv = &q[(qi * nh + h) * hd..(qi * nh + h) * hd + hd];

            let mut incl: Vec<usize> = (win_start.min(nc)..nc).collect();
            for j in 0..k {
                if mask[qi * k + j] != 0 && (window <= 0 || qpos - positions[j] < window) {
                    incl.push(nc + j);
                }
            }
            let mut scores: Vec<f64> = Vec::with_capacity(incl.len());
            for &p in &incl {
                let mut s = 0f64;
                for d in 0..hd {
                    s += qv[d] as f64 * kat(p, kvh, d) as f64;
                }
                scores.push(s * kscale(p, kvh));
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
            let denom: f64 = exps.iter().sum();
            for d in 0..hd {
                let mut acc = 0f64;
                for (i, &p) in incl.iter().enumerate() {
                    acc += exps[i] * vat(p, kvh, d) as f64;
                }
                out[(qi * nh + h) * hd + d] = if denom > 0.0 {
                    (acc / denom) as f32
                } else {
                    0.0
                };
            }
        }
    }
    out
}

fn run_case(nh: usize, nkv: usize, hd: usize, k: usize, nc: usize, window: i32, plant: bool) {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "advkern_window_boundary: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("advkern_window_boundary: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let max_seq = nc + k;
    let group = nh / nkv;

    let mut st = 0x77aa_11ee ^ ((nc as u64) << 20) ^ ((window as u64) << 4) ^ k as u64;
    let q: Vec<f32> = (0..k * nh * hd).map(|_| lcg_f32(&mut st)).collect();
    let mut kc: Vec<f32> = (0..max_seq * nkv * hd)
        .map(|_| lcg_f32(&mut st) * 0.3)
        .collect();
    let mut vc: Vec<f32> = (0..max_seq * nkv * hd).map(|_| lcg_f32(&mut st)).collect();

    if plant && window > 0 {
        let qpos = nc as i32;
        for (pos, vval) in [(qpos - window, 0.9f32), (qpos - (window - 1), -0.9f32)] {
            if pos < 0 || pos as usize >= nc {
                continue;
            }
            let pos = pos as usize;
            for kvh in 0..nkv {
                for d in 0..hd {
                    let mut qa = 0f32;
                    for g in 0..group {
                        qa += q[(0 * nh + kvh * group + g) * hd + d];
                    }
                    kc[(pos * nkv + kvh) * hd + d] = (qa / group as f32).signum() * 3.0;
                    vc[(pos * nkv + kvh) * hd + d] = vval;
                }
            }
        }
    }

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
    let mut out_d: CudaSlice<bf16> = stream
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
            cuda::tree_verify_attn_bf16(
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
                window,
            )
        }
    };
    assert_eq!(rc, 0, "tree_verify_attn_bf16 rc={rc}");
    stream.synchronize().unwrap();
    let got: Vec<bf16> = stream.clone_dtoh(&out_d).unwrap();

    let q_r: Vec<f32> = q_bf.iter().map(|x| x.to_f32()).collect();
    let kc_r: Vec<f32> = kc_bf.iter().map(|x| x.to_f32()).collect();
    let vc_r: Vec<f32> = vc_bf.iter().map(|x| x.to_f32()).collect();
    let want = reference(
        &q_r,
        &|p, kvh, d| kc_r[(p * nkv + kvh) * hd + d],
        &|p, kvh, d| vc_r[(p * nkv + kvh) * hd + d],
        &|_, _| 1.0,
        &mask,
        &positions,
        nc,
        nh,
        nkv,
        hd,
        k,
        window,
    );
    let mut max_err = 0f32;
    for i in 0..want.len() {
        max_err = max_err.max((got[i].to_f32() - want[i]).abs());
    }
    eprintln!("bf16 tree_verify nc={nc} k={k} window={window} plant={plant}: max_err={max_err:.5}");
    assert!(
        max_err < 2.5e-2,
        "bf16 nc={nc} k={k} window={window}: err {max_err}"
    );

    let mut k8_d: CudaSlice<u8> = stream.alloc_zeros::<u8>(max_seq * nkv * hd).unwrap();
    let mut v8_d: CudaSlice<u8> = stream.alloc_zeros::<u8>(max_seq * nkv * hd).unwrap();
    let mut ks_d: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_seq * nkv).unwrap();
    let mut vs_d: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_seq * nkv).unwrap();
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
            cuda::kv_append_fp8(
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
            cuda::tree_verify_attn_fp8(
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
                window,
                0,
            )
        }
    };
    assert_eq!(rc, 0, "tree_verify_attn_fp8 rc={rc}");
    stream.synchronize().unwrap();
    let got8: Vec<bf16> = stream.clone_dtoh(&out_d).unwrap();
    let k8h: Vec<u8> = stream.clone_dtoh(&k8_d).unwrap();
    let v8h: Vec<u8> = stream.clone_dtoh(&v8_d).unwrap();
    let ksh: Vec<f32> = stream.clone_dtoh(&ks_d).unwrap();
    let vsh: Vec<f32> = stream.clone_dtoh(&vs_d).unwrap();

    let want8 = reference(
        &q_r,
        &|p, kvh, d| e4m3_decode(k8h[(p * nkv + kvh) * hd + d]),
        &|p, kvh, d| e4m3_decode(v8h[(p * nkv + kvh) * hd + d]) * vsh[p * nkv + kvh],
        &|p, kvh| ksh[p * nkv + kvh] as f64,
        &mask,
        &positions,
        nc,
        nh,
        nkv,
        hd,
        k,
        window,
    );
    let mut max_err8 = 0f32;
    for i in 0..want8.len() {
        max_err8 = max_err8.max((got8[i].to_f32() - want8[i]).abs());
    }
    eprintln!(
        "fp8  tree_verify nc={nc} k={k} window={window} plant={plant}: max_err={max_err8:.5}"
    );
    assert!(
        max_err8 < 4.0e-2,
        "fp8 nc={nc} k={k} window={window}: err {max_err8}"
    );
}

#[test]
fn advkern_tree_verify_window_boundary_contexts() {
    for nc in [1023usize, 1024, 1025, 2048] {
        run_case(8, 2, 64, 5, nc, 1024, true);
    }
}

#[test]
fn advkern_tree_verify_window_edge_geometries() {
    run_case(8, 2, 64, 5, 1024, 1, false);
    run_case(8, 2, 64, 5, 1023, 1023, true);
    run_case(8, 2, 64, 5, 1024, 1029, false);
    run_case(8, 2, 64, 4, 1025, 1024, true);
    run_case(8, 2, 64, 2, 1025, 1024, true);
}

#[test]
fn advkern_tree_verify_prefill_shaped_window_inside_tree() {
    run_case(4, 2, 64, 1030, 0, 1024, false);
}
