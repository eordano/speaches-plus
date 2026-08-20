#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{cpu_nvfp4_matmul_weight_row, supports_nvfp4, Nvfp4GemmRunner, Nvfp4Tensor};
use std::ffi::c_void;
mod common;
use common::rel_rms_bf16 as rel_rms;

#[test]
fn cutlass_fp4_gemm_sm120_matches_cublaslt_and_cpu() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "cutlass_fp4_gemm_sm120: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("cutlass_fp4_gemm_sm120: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major}.{minor} lacks NVFP4 support");
        return;
    }
    if (major, minor) != (12, 0) {
        eprintln!("skip: CUTLASS FP4 GEMM TU is SM120a-only; this is SM {major}.{minor}");
        return;
    }
    let stream = ctx.default_stream();

    let (m, n, k) = (128usize, 128usize, 128usize);
    let a_rows: Vec<Vec<f32>> = (0..m)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.07).sin()).collect())
        .collect();
    let b_rows: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.09).cos()).collect())
        .collect();

    let a_q = Nvfp4Tensor::quantize_rows(&a_rows);
    let b_q = Nvfp4Tensor::quantize_rows(&b_rows);

    #[allow(deprecated)]
    let a_data = stream.memcpy_stod(&a_q.data).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_q.scales_swizzled()).unwrap();
    #[allow(deprecated)]
    let b_data = stream.memcpy_stod(&b_q.data).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_q.scales_swizzled()).unwrap();

    let mut d_cublaslt = stream.alloc_zeros::<bf16>(m * n).unwrap();
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    #[allow(deprecated)]
    let alpha_dev = stream.memcpy_stod(&[1.0f32]).unwrap();
    runner
        .matmul_scaled_alpha_dev(
            &a_data,
            &a_scales,
            &b_data,
            &b_scales,
            &mut d_cublaslt,
            m as u64,
            n as u64,
            k as u64,
            &alpha_dev,
            1.0,
        )
        .unwrap();
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let got_cublaslt = stream.memcpy_dtov(&d_cublaslt).unwrap();

    let mut d_cutlass = stream.alloc_zeros::<bf16>(m * n).unwrap();
    let global_sf = stream.memcpy_stod(&[1.0f32]).unwrap();
    let mut workspace = stream.alloc_zeros::<u8>(64 * 1024 * 1024).unwrap();

    let needed = {
        let (a_ptr, _ga) = a_data.device_ptr(&stream);
        let (a_sf_ptr, _gas) = a_scales.device_ptr(&stream);
        let (b_ptr, _gb) = b_data.device_ptr(&stream);
        let (b_sf_ptr, _gbs) = b_scales.device_ptr(&stream);
        let (gsf_ptr, _ggsf) = global_sf.device_ptr(&stream);
        let (d_ptr, _gd) = d_cutlass.device_ptr_mut(&stream);
        let (ws_ptr, _gws) = workspace.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16(
                stream.cu_stream() as *mut c_void,
                a_ptr as *const c_void,
                a_sf_ptr as *const c_void,
                b_ptr as *const c_void,
                b_sf_ptr as *const c_void,
                gsf_ptr as *const f32,
                d_ptr as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ws_ptr as *mut c_void,
                64 * 1024 * 1024,
            )
            .expect("cutlass_fp4_gemm_sm120_bf16 launch")
        }
    };
    stream.synchronize().unwrap();
    eprintln!("cutlass workspace used: {} bytes", needed);

    #[allow(deprecated)]
    let got_cutlass = stream.memcpy_dtov(&d_cutlass).unwrap();

    let expect = cpu_nvfp4_matmul_weight_row(&a_q, &b_q, m, n, k);

    let nz_cublaslt = got_cublaslt.iter().filter(|x| x.to_f32() != 0.0).count();
    let nz_cutlass = got_cutlass.iter().filter(|x| x.to_f32() != 0.0).count();
    let nz_expect = expect.iter().filter(|x| x.to_f32() != 0.0).count();
    eprintln!(
        "non-zero counts: cuBLASLt={nz_cublaslt}/{}  CUTLASS={nz_cutlass}/{}  CPU={nz_expect}/{}",
        m * n,
        m * n,
        m * n
    );
    eprintln!(
        "first 4 outputs: cuBLASLt={:?} CUTLASS={:?} CPU={:?}",
        &got_cublaslt[..4]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>(),
        &got_cutlass[..4]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>(),
        &expect[..4].iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
    );
    assert!(
        nz_cublaslt > (m * n) / 4,
        "cuBLASLt output is mostly zero -- kernel did not run"
    );
    assert!(
        nz_cutlass > (m * n) / 4,
        "CUTLASS output is mostly zero -- kernel did not run"
    );

    let r_cublaslt_vs_cpu = rel_rms(&got_cublaslt, &expect);
    let r_cutlass_vs_cpu = rel_rms(&got_cutlass, &expect);
    let r_cutlass_vs_cublaslt = rel_rms(&got_cutlass, &got_cublaslt);
    eprintln!("cuBLASLt vs CPU rel-rms       = {r_cublaslt_vs_cpu:.5}");
    eprintln!("CUTLASS  vs CPU rel-rms       = {r_cutlass_vs_cpu:.5}");
    eprintln!("CUTLASS  vs cuBLASLt rel-rms  = {r_cutlass_vs_cublaslt:.5}");

    assert!(
        r_cutlass_vs_cpu < 0.20,
        "CUTLASS-vs-CPU rel-rms {r_cutlass_vs_cpu} exceeds 0.20"
    );
    assert!(
        r_cutlass_vs_cublaslt < 0.10,
        "CUTLASS-vs-cuBLASLt rel-rms {r_cutlass_vs_cublaslt} exceeds 0.10 (layout/scale mismatch?)"
    );
}
