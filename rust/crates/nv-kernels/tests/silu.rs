#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
mod common;
use common::cpu_silu_mul;

fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

#[test]
fn silu_f32_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "silu: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("silu: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let n = 4096usize;
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let v = ((i as f32) * 0.0017).sin() * 6.0;
        x.push(v);
    }
    let expect = cpu_silu(&x);

    #[allow(deprecated)]
    let dx: CudaSlice<f32> = stream.memcpy_stod(&x).unwrap();
    let mut dy: CudaSlice<f32> = stream.alloc_zeros::<f32>(n).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (py, _g2) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_f32(
                stream.cu_stream() as *mut _,
                px as *const f32,
                py as *mut f32,
                n,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&dy).unwrap();
    let mut max_err = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1e-5, "silu f32 drift {max_err}");
}

#[test]
fn silu_bf16_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "silu: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("silu: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let n = 4096usize;
    let mut x_f32 = Vec::with_capacity(n);
    for i in 0..n {
        let v = ((i as f32) * 0.0017).sin() * 6.0;
        x_f32.push(v);
    }
    let x: Vec<u16> = x_f32.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    let x_quant: Vec<f32> = x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    let expect = cpu_silu(&x_quant);

    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.memcpy_stod(&x).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (py, _g2) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                py as *mut u16,
                n,
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
    for (g, e) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 0.05, "silu bf16 drift {max_err}");
}

#[test]
fn silu_mul_f32_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "silu: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("silu: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let n = 4096usize;
    let mut x = Vec::with_capacity(n);
    let mut gate = Vec::with_capacity(n);
    for i in 0..n {
        x.push(((i as f32) * 0.0017).sin() * 4.0);
        gate.push(((i as f32) * 0.0023).cos() * 3.0);
    }
    let expect = cpu_silu_mul(&x, &gate);

    #[allow(deprecated)]
    let dx: CudaSlice<f32> = stream.memcpy_stod(&x).unwrap();
    #[allow(deprecated)]
    let dg: CudaSlice<f32> = stream.memcpy_stod(&gate).unwrap();
    let mut dy: CudaSlice<f32> = stream.alloc_zeros::<f32>(n).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (pg, _g2) = dg.device_ptr(&stream);
        let (py, _g3) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_mul_f32(
                stream.cu_stream() as *mut _,
                px as *const f32,
                pg as *const f32,
                py as *mut f32,
                n,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&dy).unwrap();
    let mut max_err = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1e-5, "silu_mul f32 drift {max_err}");
}

#[test]
fn silu_mul_bf16_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "silu: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("silu: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let n = 4096usize;
    let mut x_f32 = Vec::with_capacity(n);
    let mut g_f32 = Vec::with_capacity(n);
    for i in 0..n {
        x_f32.push(((i as f32) * 0.0017).sin() * 4.0);
        g_f32.push(((i as f32) * 0.0023).cos() * 3.0);
    }
    let x: Vec<u16> = x_f32.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    let g: Vec<u16> = g_f32.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    let x_q: Vec<f32> = x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    let g_q: Vec<f32> = g.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    let expect = cpu_silu_mul(&x_q, &g_q);

    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.memcpy_stod(&x).unwrap();
    #[allow(deprecated)]
    let dg: CudaSlice<u16> = stream.memcpy_stod(&g).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (pg, _g2) = dg.device_ptr(&stream);
        let (py, _g3) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_mul_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pg as *const u16,
                py as *mut u16,
                n,
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
    for (g, e) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 0.05, "silu_mul bf16 drift {max_err}");
}
