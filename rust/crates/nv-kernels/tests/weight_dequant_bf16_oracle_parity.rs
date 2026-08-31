#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;

mod common;
use common::LcgMask23TwoSided;

const RC_REFUSES_NEGATIVE_DIMS: i32 = -3;
const RC_NVFP4_REFUSES_K_NOT_MULTIPLE_OF_16: i32 = -4;

fn stream_or_refuse(test: &str) -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => {
            if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "{test}: no CUDA device 0 ({e}). This gate refuses to report success \
                     without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
            None
        }
    }
}

fn oracle_bits(host_f32: &[f32]) -> Vec<u16> {
    host_f32
        .iter()
        .map(|v| bf16::from_f32(*v).to_bits())
        .collect()
}

fn assert_bitwise(tag: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let mut diff = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            diff += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert_eq!(
        diff,
        0,
        "{tag}: {diff}/{} bf16 words differ from the host oracle bitwise, first (idx, got, want) = {first:?}",
        got.len()
    );
}

fn run_nvfp4_device(
    stream: &Arc<CudaStream>,
    packed: &[u8],
    scales_swizzled: &[u8],
    alpha: f32,
    n: usize,
    k: usize,
) -> (i32, Vec<u16>) {
    #[allow(deprecated)]
    let packed_dev = stream.memcpy_stod(packed).unwrap();
    #[allow(deprecated)]
    let scales_dev = stream.memcpy_stod(scales_swizzled).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n * k).unwrap();
    let rc = {
        let (pp, _g1) = packed_dev.device_ptr(stream);
        let (sp, _g2) = scales_dev.device_ptr(stream);
        let (op, _g3) = out_dev.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::dequant_weight_nvfp4_swizzled_to_bf16_rowmajor(
                stream.cu_stream() as *mut c_void,
                pp as *const u8,
                sp as *const u8,
                op as *mut u16,
                alpha,
                n as i32,
                k as i32,
            )
        }
    };
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&out_dev).unwrap();
    (rc, out)
}

fn run_e4m3_device(
    stream: &Arc<CudaStream>,
    wq: &[u8],
    row_scale: &[f32],
    n: usize,
    k: usize,
) -> (i32, Vec<u16>) {
    #[allow(deprecated)]
    let wq_dev = stream.memcpy_stod(wq).unwrap();
    #[allow(deprecated)]
    let rs_dev = stream.memcpy_stod(row_scale).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n * k).unwrap();
    let rc = {
        let (wp, _g1) = wq_dev.device_ptr(stream);
        let (sp, _g2) = rs_dev.device_ptr(stream);
        let (op, _g3) = out_dev.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::dequant_weight_e4m3_rowscale_to_bf16_rowmajor(
                stream.cu_stream() as *mut c_void,
                wp as *const u8,
                sp as *const f32,
                op as *mut u16,
                n as i32,
                k as i32,
            )
        }
    };
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&out_dev).unwrap();
    (rc, out)
}

const NVFP4_SHAPES_TAIL_HEAVY_ROWS_OFF_128_KBLOCKS_OFF_4: &[(usize, usize)] = &[
    (1, 16),
    (3, 32),
    (31, 48),
    (100, 64),
    (128, 128),
    (130, 80),
    (257, 208),
    (64, 2048),
    (300, 1008),
];

#[test]
fn nvfp4_device_dequant_matches_host_oracle_bitwise_on_quantized_tensors() {
    let Some(stream) = stream_or_refuse("nvfp4_quantized_parity") else {
        return;
    };
    for (case, &(n, k)) in NVFP4_SHAPES_TAIL_HEAVY_ROWS_OFF_128_KBLOCKS_OFF_4
        .iter()
        .enumerate()
    {
        let mut rng = LcgMask23TwoSided(0x9E3779B9u64 ^ ((case as u64) << 32) | 1);
        let stored_global = 1.0 + 0.35 * (case as f32);
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_f32() * 4.0).collect())
            .collect();
        let q = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
        let sw = q.scales_swizzled();
        let alpha = 1.0 / stored_global;
        let host = nv_quant::nvfp4::dequantize_packed_swizzled(&q.data, &sw, n, k, alpha);
        let want = oracle_bits(&host);
        let (rc, got) = run_nvfp4_device(&stream, &q.data, &sw, alpha, n, k);
        assert_eq!(rc, 0, "nvfp4 dequant rc={rc} (n={n} k={k})");
        assert_bitwise(&format!("nvfp4 quantized n={n} k={k}"), &got, &want);
    }
}

#[test]
fn nvfp4_device_dequant_matches_host_oracle_bitwise_on_full_random_bytes() {
    let Some(stream) = stream_or_refuse("nvfp4_random_bytes_parity") else {
        return;
    };
    for (case, &(n, k)) in NVFP4_SHAPES_TAIL_HEAVY_ROWS_OFF_128_KBLOCKS_OFF_4
        .iter()
        .enumerate()
    {
        let mut rng = LcgMask23TwoSided(0xFEEDFACEu64 ^ ((case as u64) << 24) | 1);
        let packed = rng.packed_nibbles(n * k / 2);
        let k_blocks = k / 16;
        let scales_linear: Vec<u8> = (0..n * k_blocks)
            .map(|_| (rng.next_u32() & 0xFF) as u8)
            .collect();
        let sw = nv_quant::nvfp4::swizzle_scales(&scales_linear, n, k_blocks);
        for alpha in [1.0f32, 0.0371, 5.75] {
            let host = nv_quant::nvfp4::dequantize_packed_swizzled(&packed, &sw, n, k, alpha);
            let want = oracle_bits(&host);
            let (rc, got) = run_nvfp4_device(&stream, &packed, &sw, alpha, n, k);
            assert_eq!(rc, 0, "nvfp4 dequant rc={rc} (n={n} k={k} alpha={alpha})");
            assert_bitwise(
                &format!("nvfp4 random bytes n={n} k={k} alpha={alpha}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn nvfp4_device_dequant_refuses_the_shapes_the_host_oracle_refuses() {
    let host_refused = std::panic::catch_unwind(|| {
        let packed = [0u8; 12];
        let sw = nv_quant::nvfp4::swizzle_scales(&[0u8; 2], 1, 2);
        nv_quant::nvfp4::dequantize_packed_swizzled(&packed, &sw, 1, 24, 1.0)
    })
    .is_err();
    assert!(
        host_refused,
        "host oracle accepted cols=24; the packed layout has no tail-block path and the \
         device refusal below would then be stricter than the oracle"
    );
    let Some(stream) = stream_or_refuse("nvfp4_refusal") else {
        return;
    };
    let (rc, _) = run_nvfp4_device(&stream, &[0u8; 16], &[0u8; 512], 1.0, 1, 24);
    assert_eq!(
        rc, RC_NVFP4_REFUSES_K_NOT_MULTIPLE_OF_16,
        "device must refuse k=24 exactly like the host oracle refuses cols=24"
    );
    let rc_neg = unsafe {
        nv_kernels::cuda::dequant_weight_nvfp4_swizzled_to_bf16_rowmajor(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1.0,
            -1,
            16,
        )
    };
    assert_eq!(rc_neg, RC_REFUSES_NEGATIVE_DIMS);
}

const E4M3_SHAPES_INCLUDING_ODD_K: &[(usize, usize)] =
    &[(1, 16), (5, 512), (7, 384), (37, 997), (254, 129)];

#[test]
fn e4m3_device_dequant_matches_host_oracle_bitwise_on_quantized_rows() {
    let Some(stream) = stream_or_refuse("e4m3_quantized_parity") else {
        return;
    };
    for (case, &(n, k)) in E4M3_SHAPES_INCLUDING_ODD_K.iter().enumerate() {
        let mut rng = LcgMask23TwoSided(0xBADC0FFEu64 ^ ((case as u64) << 16) | 1);
        let values: Vec<bf16> = (0..n * k)
            .map(|_| bf16::from_f32(rng.next_f32() * 3.0))
            .collect();
        let (bytes, scales) = nv_quant::fp8::quantize_e4m3_per_row(&values, n, k).unwrap();
        let host = nv_quant::fp8::dequantize_e4m3_per_row(&bytes, n, k, &scales).unwrap();
        let want = oracle_bits(&host);
        let (rc, got) = run_e4m3_device(&stream, &bytes, &scales, n, k);
        assert_eq!(rc, 0, "e4m3 dequant rc={rc} (n={n} k={k})");
        assert_bitwise(&format!("e4m3 quantized n={n} k={k}"), &got, &want);
    }
}

#[test]
fn e4m3_device_dequant_matches_host_oracle_bitwise_over_every_non_nan_code_with_distinct_row_scales(
) {
    let Some(stream) = stream_or_refuse("e4m3_full_code_space_parity") else {
        return;
    };
    let non_nan_codes: Vec<u8> = (0u16..=255)
        .map(|b| b as u8)
        .filter(|b| b & 0x7F != 0x7F)
        .collect();
    assert_eq!(non_nan_codes.len(), 254);
    let (n, k) = (127usize, 254usize);
    let bytes: Vec<u8> = (0..n * k)
        .map(|i| non_nan_codes[(i * 7 + i / k) % non_nan_codes.len()])
        .collect();
    let scales: Vec<f32> = (0..n).map(|r| 0.002 + 0.013 * (r as f32 + 1.0)).collect();
    assert!(
        scales.windows(2).all(|w| w[0] != w[1]),
        "uniform row scales hide row-index bugs; distinct scales are mandatory"
    );
    let host = nv_quant::fp8::dequantize_e4m3_per_row(&bytes, n, k, &scales).unwrap();
    let want = oracle_bits(&host);
    let (rc, got) = run_e4m3_device(&stream, &bytes, &scales, n, k);
    assert_eq!(rc, 0, "e4m3 dequant rc={rc}");
    assert_bitwise("e4m3 full non-NaN code space", &got, &want);
}

#[test]
fn e4m3_device_dequant_refuses_negative_dims_and_the_host_oracle_refuses_len_mismatches() {
    assert!(
        nv_quant::fp8::dequantize_e4m3_per_row(&[0u8; 8], 2, 4, &[1.0]).is_err(),
        "host oracle must refuse a scales vector shorter than rows"
    );
    assert!(
        nv_quant::fp8::dequantize_e4m3_per_row(&[0u8; 7], 2, 4, &[1.0, 1.0]).is_err(),
        "host oracle must refuse a byte count that is not rows*cols"
    );
    let Some(stream) = stream_or_refuse("e4m3_refusal") else {
        return;
    };
    let rc = unsafe {
        nv_kernels::cuda::dequant_weight_e4m3_rowscale_to_bf16_rowmajor(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            2,
            -4,
        )
    };
    assert_eq!(rc, RC_REFUSES_NEGATIVE_DIMS);
}
