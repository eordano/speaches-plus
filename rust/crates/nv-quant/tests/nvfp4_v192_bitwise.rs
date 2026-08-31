#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4Tensor};
use std::ffi::c_void;

const WORKSPACE_BYTES: usize = 4 << 20;

const ALPHA_NON_UNIT_SO_THE_EPILOGUE_SCALE_PATH_IS_UNDER_TEST: f32 = 0.7315;

fn rows(m: usize, k: usize, phase: f32) -> Vec<Vec<f32>> {
    (0..m)
        .map(|i| {
            (0..k)
                .map(|j| ((i * k + j) as f32 * phase).sin() * 2.0)
                .collect()
        })
        .collect()
}

#[test]
fn nvfp4_v192_matches_shipped_cutlass_arm_bitwise_because_k_accumulation_order_is_identical() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4 support");
        return;
    }
    let stream = ctx.default_stream();

    for &(m, n, k) in &[
        (192usize, 128usize, 256usize),
        (129, 384, 128),
        (200, 256, 512),
        (1024, 512, 384),
        (4096, 2048, 512),
    ] {
        let a_q = Nvfp4Tensor::quantize_rows(&rows(m, k, 0.07));
        let b_q = Nvfp4Tensor::quantize_rows(&rows(n, k, 0.013));
        #[allow(deprecated)]
        let a_data: CudaSlice<u8> = stream.memcpy_stod(&a_q.data).unwrap();
        #[allow(deprecated)]
        let a_scales: CudaSlice<u8> = stream.memcpy_stod(&a_q.scales_swizzled()).unwrap();
        #[allow(deprecated)]
        let b_data: CudaSlice<u8> = stream.memcpy_stod(&b_q.data).unwrap();
        #[allow(deprecated)]
        let b_scales: CudaSlice<u8> = stream.memcpy_stod(&b_q.scales_swizzled()).unwrap();
        #[allow(deprecated)]
        let alpha_dev: CudaSlice<f32> = stream
            .memcpy_stod(&[ALPHA_NON_UNIT_SO_THE_EPILOGUE_SCALE_PATH_IS_UNDER_TEST])
            .unwrap();
        let mut d_ctl: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();
        let mut d_new: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();
        let mut ws: CudaSlice<u8> = stream.alloc_zeros::<u8>(WORKSPACE_BYTES).unwrap();

        let ctl_tile = if k % 256 == 0 { 1 } else { 0 };
        {
            let (ap, _ga) = a_data.device_ptr(&stream);
            let (asp, _gas) = a_scales.device_ptr(&stream);
            let (bp, _gb) = b_data.device_ptr(&stream);
            let (bsp, _gbs) = b_scales.device_ptr(&stream);
            let (gp, _gg) = alpha_dev.device_ptr(&stream);
            let (dcp, _gdc) = d_ctl.device_ptr_mut(&stream);
            let (wsp, _gws) = ws.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_tiled(
                    stream.cu_stream() as *mut c_void,
                    ap as *const c_void,
                    asp as *const c_void,
                    bp as *const c_void,
                    bsp as *const c_void,
                    gp as *const f32,
                    dcp as *mut c_void,
                    m as i32,
                    n as i32,
                    k as i32,
                    ctl_tile,
                    0,
                    wsp as *mut c_void,
                    WORKSPACE_BYTES,
                )
            }
            .unwrap_or_else(|rc| panic!("control cutlass arm rc={rc} at m={m} n={n} k={k}"));
            let (dnp, _gdn) = d_new.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::nvfp4_gemm_sm120_v192_bf16(
                    stream.cu_stream() as *mut c_void,
                    ap as *const c_void,
                    asp as *const c_void,
                    bp as *const c_void,
                    bsp as *const c_void,
                    gp as *const f32,
                    dnp as *mut c_void,
                    m as i32,
                    n as i32,
                    k as i32,
                )
            }
            .unwrap_or_else(|rc| panic!("v192 arm rc={rc} at m={m} n={n} k={k}"));
        }
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let ctl: Vec<bf16> = stream.memcpy_dtov(&d_ctl).unwrap();
        #[allow(deprecated)]
        let new: Vec<bf16> = stream.memcpy_dtov(&d_new).unwrap();
        let mut mismatches = 0usize;
        let mut first: Option<(usize, u16, u16)> = None;
        for (i, (c, v)) in ctl.iter().zip(new.iter()).enumerate() {
            if c.to_bits() != v.to_bits() {
                mismatches += 1;
                if first.is_none() {
                    first = Some((i, c.to_bits(), v.to_bits()));
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "v192 diverges from the shipped arm at m={m} n={n} k={k}: {mismatches}/{} elems, \
             first at flat={:?} (row={}, col={}); both arms accumulate k64 mma steps in the \
             same monotonic order so any difference is a fragment-layout or predicate bug, \
             not float noise",
            m * n,
            first,
            first.map_or(0, |(i, _, _)| i / n),
            first.map_or(0, |(i, _, _)| i % n),
        );
        eprintln!("v192 bitwise ok m={m} n={n} k={k}");
    }
}
