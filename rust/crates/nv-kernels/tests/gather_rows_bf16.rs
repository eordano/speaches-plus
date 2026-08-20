#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

#[test]
fn gather_rows_bf16_basic() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "gather_rows_bf16: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("gather_rows_bf16: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let n_tokens = 5usize;
    let hidden = 256usize;
    let m_total_padded = 16usize;

    let mut x_host: Vec<bf16> = Vec::with_capacity(n_tokens * hidden);
    for n in 0..n_tokens {
        for h in 0..hidden {
            x_host.push(bf16::from_f32((n * 100 + h) as f32 * 0.001));
        }
    }
    let src_idx: Vec<i32> = vec![0, 1, 2, 3, 4, -1, -1, -1, 2, 0, 4, 1, 3, -1, -1, -1];
    assert_eq!(src_idx.len(), m_total_padded);

    #[allow(deprecated)]
    let x_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u16, x_host.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let src_dev = stream.memcpy_stod(&src_idx).unwrap();
    let mut out_dev = stream.alloc_zeros::<bf16>(m_total_padded * hidden).unwrap();

    let rc = {
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (sp, _g2) = src_dev.device_ptr(&stream);
        let (op, _g3) = out_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gather_rows_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                sp as *const i32,
                op as *mut u16,
                m_total_padded as i32,
                hidden as i32,
                n_tokens as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gather_rows_bf16 rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&out_dev).unwrap();

    for r in 0..m_total_padded {
        let src = src_idx[r];
        for h in 0..hidden {
            let got = out[r * hidden + h].to_f32();
            let expect = if src >= 0 && (src as usize) < n_tokens {
                (src as usize * 100 + h) as f32 * 0.001
            } else {
                0.0
            };
            let expect_bf = bf16::from_f32(expect).to_f32();
            assert!(
                (got - expect_bf).abs() < 1e-4,
                "row {r} h {h}: got {got} expect {expect_bf} (src_idx={src})"
            );
        }
    }
}

#[test]
fn gather_rows_bf16_pad_only() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "gather_rows_bf16: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("gather_rows_bf16: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let n_tokens = 1usize;
    let hidden = 128usize;
    let m_total_padded = 4usize;

    let x_host: Vec<bf16> = vec![bf16::from_f32(123.0); n_tokens * hidden];
    let src_idx: Vec<i32> = vec![-1; m_total_padded];

    #[allow(deprecated)]
    let x_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u16, x_host.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let src_dev = stream.memcpy_stod(&src_idx).unwrap();
    let mut out_dev = stream.alloc_zeros::<bf16>(m_total_padded * hidden).unwrap();

    let rc = {
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (sp, _g2) = src_dev.device_ptr(&stream);
        let (op, _g3) = out_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gather_rows_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                sp as *const i32,
                op as *mut u16,
                m_total_padded as i32,
                hidden as i32,
                n_tokens as i32,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&out_dev).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(v.to_f32(), 0.0, "padding row should be zero, idx {i}");
    }
}
