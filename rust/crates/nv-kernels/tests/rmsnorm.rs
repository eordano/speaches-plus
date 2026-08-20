#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
mod common;
use common::cpu_rmsnorm;

#[test]
fn rmsnorm_f32_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "rmsnorm: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("rmsnorm: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;

    let mut x = Vec::with_capacity(batch * hidden);
    let mut weight = Vec::with_capacity(hidden);
    for i in 0..(batch * hidden) {
        x.push(((i as f32) * 0.0001).sin());
    }
    for i in 0..hidden {
        weight.push(1.0 + ((i as f32) * 0.001).cos());
    }

    #[allow(deprecated)]
    let dx: CudaSlice<f32> = stream.memcpy_stod(&x).unwrap();
    #[allow(deprecated)]
    let dw: CudaSlice<f32> = stream.memcpy_stod(&weight).unwrap();
    let mut dy: CudaSlice<f32> = stream.alloc_zeros::<f32>(batch * hidden).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (pw, _g2) = dw.device_ptr(&stream);
        let (py, _g3) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_f32(
                stream.cu_stream() as *mut _,
                px as *const f32,
                pw as *const f32,
                py as *mut f32,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&dy).unwrap();
    let expect = cpu_rmsnorm(&x, &weight, hidden, eps);
    let mut max_err = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1e-4, "max abs error {max_err} exceeds 1e-4");
}

#[test]
fn rmsnorm_bf16_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "rmsnorm: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("rmsnorm: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;

    let mut x = Vec::with_capacity(batch * hidden);
    let mut weight = Vec::with_capacity(hidden);
    for i in 0..(batch * hidden) {
        x.push(bf16::from_f32(((i as f32) * 0.0001).sin()));
    }
    for i in 0..hidden {
        weight.push(bf16::from_f32(1.0 + ((i as f32) * 0.001).cos()));
    }

    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let w_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
    let expect_f32 = cpu_rmsnorm(&x_f32, &w_f32, hidden, eps);

    let x_u16: Vec<u16> = x.iter().map(|v| v.to_bits()).collect();
    let w_u16: Vec<u16> = weight.iter().map(|v| v.to_bits()).collect();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.memcpy_stod(&x_u16).unwrap();
    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.memcpy_stod(&w_u16).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(batch * hidden).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (pw, _g2) = dw.device_ptr(&stream);
        let (py, _g3) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pw as *const u16,
                py as *mut u16,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got_u16 = stream.memcpy_dtov(&dy).unwrap();
    let got: Vec<f32> = got_u16
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    let mut max_err = 0f32;
    for (g, e) in got.iter().zip(expect_f32.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 0.05, "bf16 rmsnorm drift {max_err}");
}
