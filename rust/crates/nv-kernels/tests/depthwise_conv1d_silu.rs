#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn cpu_reference(x: &[bf16], w: &[bf16], b: usize, c: usize, t: usize, k: usize) -> Vec<bf16> {
    let mut y = vec![bf16::from_f32(0.0); b * c * t];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..t {
                let mut acc = 0f32;
                for kk in 0..k {
                    let src_t = ti as isize - (k - 1) as isize + kk as isize;
                    if src_t >= 0 && (src_t as usize) < t {
                        let xv = x[(bi * c + ci) * t + src_t as usize].to_f32();
                        let wv = w[ci * k + kk].to_f32();
                        acc += xv * wv;
                    }
                }
                y[(bi * c + ci) * t + ti] = bf16::from_f32(silu(acc));
            }
        }
    }
    y
}

fn run(b: usize, c: usize, t: usize, k: usize, label: &str) {
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();

    let x_host: Vec<bf16> = (0..b * c * t)
        .map(|i| bf16::from_f32((i as f32 * 0.013).sin() * 0.5))
        .collect();
    let w_host: Vec<bf16> = (0..c * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.07).cos() - 0.3) * 0.4))
        .collect();

    #[allow(deprecated)]
    let x_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u16, x_host.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let w_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(w_host.as_ptr() as *const u16, w_host.len())
        })
        .unwrap();
    let mut y_dev = stream.alloc_zeros::<bf16>(b * c * t).unwrap();

    let rc = {
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (wp, _g2) = w_dev.device_ptr(&stream);
        let (yp, _g3) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::depthwise_conv1d_silu_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                wp as *const u16,
                yp as *mut u16,
                b as i32,
                c as i32,
                t as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "{label}: rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let y_got = stream.memcpy_dtov(&y_dev).unwrap();

    let y_ref = cpu_reference(&x_host, &w_host, b, c, t, k);

    let mut max_abs = 0f32;
    for (g, r) in y_got.iter().zip(y_ref.iter()) {
        let d = (g.to_f32() - r.to_f32()).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    eprintln!("{label}: B={b} C={c} T={t} K={k} max_abs_diff = {max_abs:.6}");
    assert!(max_abs < 1e-2, "{label}: max_abs_diff {max_abs} too high");
}

#[test]
fn depthwise_conv1d_silu_t1_k4() {
    run(1, 6144, 1, 4, "decode T=1");
}

#[test]
fn depthwise_conv1d_silu_t8_k4() {
    run(1, 6144, 8, 4, "prefill_s8");
}

#[test]
fn depthwise_conv1d_silu_t32_k4() {
    run(1, 6144, 32, 4, "prefill_s32");
}

#[test]
fn depthwise_conv1d_silu_t128_k4() {
    run(1, 6144, 128, 4, "prefill_s128");
}

#[test]
fn depthwise_conv1d_silu_small() {
    run(1, 16, 5, 4, "small smoke");
}
