#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda as nvk;
use std::ffi::c_void;

fn cpu_ref(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    n_q: usize,
    n_kv: usize,
    hd: usize,
    total: usize,
    window: usize,
    scale: f32,
) -> Vec<f32> {
    let group = n_q / n_kv;
    let start = if window > 0 && total > window {
        total - window
    } else {
        0
    };
    let mut out = vec![0f32; n_q * hd];
    for h in 0..n_q {
        let kvh = h / group;
        let mut scores = Vec::with_capacity(total - start);
        for p in start..total {
            let mut dot = 0f64;
            for d in 0..hd {
                dot += (q[h * hd + d].to_f32() * scale) as f64
                    * k[(p * n_kv + kvh) * hd + d].to_f32() as f64;
            }
            scores.push(dot);
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0f64;
        let mut acc = vec![0f64; hd];
        for (i, sc) in scores.iter().enumerate() {
            let w = (sc - m).exp();
            l += w;
            let p = start + i;
            for d in 0..hd {
                acc[d] += w * v[(p * n_kv + kvh) * hd + d].to_f32() as f64;
            }
        }
        for d in 0..hd {
            out[h * hd + d] = (acc[d] / l) as f32;
        }
    }
    out
}

#[test]
fn laguna_flash_decode_gqa_matches_cpu() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "laguna_flash_decode_gqa: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("laguna_flash_decode_gqa: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let hd = 128usize;
    let scale = (hd as f32).powf(-0.5);
    let mut rng_state = 0x12345678u64;
    let mut rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };

    for (n_q, n_kv) in [(48usize, 8usize), (10usize, 10usize)] {
        for (total, window) in [(553usize, 0usize), (553, 512), (517, 512), (5, 512), (1, 0)] {
            let q: Vec<bf16> = (0..n_q * hd).map(|_| bf16::from_f32(rand())).collect();
            let k: Vec<bf16> = (0..total * n_kv * hd)
                .map(|_| bf16::from_f32(rand()))
                .collect();
            let v: Vec<bf16> = (0..total * n_kv * hd)
                .map(|_| bf16::from_f32(rand()))
                .collect();

            let q_d = stream.memcpy_stod(&q).unwrap();
            let k_d = stream.memcpy_stod(&k).unwrap();
            let v_d = stream.memcpy_stod(&v).unwrap();
            let cu = stream.memcpy_stod(&[0i32, total as i32]).unwrap();
            let elems = nvk::laguna_flash_decode_gqa_scratch_elems(n_kv as i32);
            let mut scratch = stream.alloc_zeros::<f32>(elems).unwrap();
            let mut fan_in = stream.alloc_zeros::<u32>(n_kv).unwrap();
            let mut out_d = stream.alloc_zeros::<u16>(n_q * hd).unwrap();

            for _ in 0..2 {
                let rc = unsafe {
                    nvk::laguna_flash_decode_gqa(
                        stream.cu_stream() as *mut c_void,
                        (q_d.device_ptr(&stream).0) as *const u16,
                        (k_d.device_ptr(&stream).0) as *const u16,
                        (v_d.device_ptr(&stream).0) as *const u16,
                        (out_d.device_ptr_mut(&stream).0) as *mut u16,
                        ((cu.device_ptr(&stream).0) + 4) as *const i32,
                        0,
                        (scratch.device_ptr_mut(&stream).0) as *mut f32,
                        (fan_in.device_ptr_mut(&stream).0) as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        window as i32,
                        scale,
                    )
                };
                assert_eq!(rc, 0, "kernel rc (total {total} window {window})");
            }
            stream.synchronize().unwrap();
            let out_raw: Vec<u16> = stream.memcpy_dtov(&out_d).unwrap();
            let expect = cpu_ref(&q, &k, &v, n_q, n_kv, hd, total, window, scale);

            let mut max_err = 0f32;
            for (i, &r) in out_raw.iter().enumerate() {
                let got = bf16::from_bits(r).to_f32();
                let e = expect[i];
                let err = (got - e).abs() / e.abs().max(1.0);
                max_err = max_err.max(err);
            }
            assert!(
                max_err < 2e-2,
                "q{n_q}/kv{n_kv} total {total} window {window}: max rel err {max_err}"
            );
            eprintln!(
                "[gqa] q{n_q}/kv{n_kv} total {total} window {window}: max rel err {max_err:.5}"
            );
        }
    }
}
