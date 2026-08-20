#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

fn cpu_reference(
    y_sorted: &[bf16],
    topk_weights: &[f32],
    inv_perm: &[i32],
    n_tokens: usize,
    k: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut acc = vec![0f32; n_tokens * hidden];
    for n in 0..n_tokens {
        for s in 0..k {
            let slot = n * k + s;
            let sorted_row = inv_perm[slot] as usize;
            let w = topk_weights[slot];
            for h in 0..hidden {
                acc[n * hidden + h] += w * y_sorted[sorted_row * hidden + h].to_f32();
            }
        }
    }
    acc
}

#[test]
fn moe_unpermute_scatter_matches_cpu() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_unpermute_scatter: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("moe_unpermute_scatter: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let n_tokens = 8usize;
    let k = 4usize;
    let hidden = 512usize;
    let m_total = n_tokens * k;

    let mut inv_perm: Vec<i32> = (0..(n_tokens * k) as i32).collect();
    inv_perm.swap(0, 7);
    inv_perm.swap(3, 12);
    inv_perm.swap(5, 22);

    let y_sorted: Vec<bf16> = (0..m_total * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 1.5))
        .collect();
    let topk_weights: Vec<f32> = (0..n_tokens * k)
        .map(|i| 0.1 + (i as f32 * 0.071).cos().abs() * 0.4)
        .collect();

    let cpu = cpu_reference(&y_sorted, &topk_weights, &inv_perm, n_tokens, k, hidden);

    #[allow(deprecated)]
    let y_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(y_sorted.as_ptr() as *const u16, y_sorted.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let w_dev = stream.memcpy_stod(&topk_weights).unwrap();
    #[allow(deprecated)]
    let inv_dev = stream.memcpy_stod(&inv_perm).unwrap();
    let mut acc_dev = stream.alloc_zeros::<f32>(n_tokens * hidden).unwrap();

    let rc = {
        let (y_p, _g1) = y_dev.device_ptr(&stream);
        let (w_p, _g2) = w_dev.device_ptr(&stream);
        let (i_p, _g3) = inv_dev.device_ptr(&stream);
        let (a_p, _g4) = acc_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::moe_unpermute_scatter(
                stream.cu_stream() as *mut c_void,
                y_p as *const u16,
                w_p as *const f32,
                i_p as *const i32,
                a_p as *mut f32,
                n_tokens as i32,
                k as i32,
                hidden as i32,
                hidden as i32,
            )
        }
    };
    assert_eq!(rc, 0, "moe_unpermute_scatter rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&acc_dev).unwrap();

    let mut max_abs_diff = 0f32;
    for (g, c) in got.iter().zip(cpu.iter()) {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    eprintln!("max abs diff GPU vs CPU = {max_abs_diff:.6}");
    assert!(max_abs_diff < 1e-4, "GPU vs CPU diverges by {max_abs_diff}");
}
