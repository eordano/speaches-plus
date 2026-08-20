#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::BLOCK_SIZE;
use std::ffi::c_void;
mod common;
use common::swizzled_dst;

fn quantize(m_padded: usize, m_logical: usize, k: usize, rows_only: bool) -> (Vec<u8>, Vec<u8>) {
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();

    let host_bf: Vec<bf16> = (0..m_logical * k)
        .map(|n| bf16::from_f32(((n as f32) * 0.017 + 0.3).sin() * 3.0))
        .collect();
    #[allow(deprecated)]
    let x_dev = stream
        .clone_htod(unsafe {
            std::slice::from_raw_parts(host_bf.as_ptr() as *const u16, host_bf.len())
        })
        .unwrap();
    let packed_bytes = m_padded * k / 2;
    let scales_bytes = m_padded.div_ceil(128) * 128 * ((k / BLOCK_SIZE).div_ceil(4)) * 4;
    let mut packed_dev = stream.alloc_zeros::<u8>(packed_bytes).unwrap();
    let mut scales_dev = stream.alloc_zeros::<u8>(scales_bytes).unwrap();

    let rc = {
        let (x_ptr, _gx) = x_dev.device_ptr(&stream);
        let (p_ptr, _gp) = packed_dev.device_ptr_mut(&stream);
        let (s_ptr, _gs) = scales_dev.device_ptr_mut(&stream);
        unsafe {
            if rows_only {
                nv_kernels::cuda::quantize_nvfp4_bf16_rows(
                    stream.cu_stream() as *mut c_void,
                    x_ptr as *const u16,
                    p_ptr as *mut u8,
                    s_ptr as *mut u8,
                    1.0,
                    m_logical as i32,
                    k as i32,
                )
            } else {
                nv_kernels::cuda::quantize_nvfp4_bf16(
                    stream.cu_stream() as *mut c_void,
                    x_ptr as *const u16,
                    p_ptr as *mut u8,
                    s_ptr as *mut u8,
                    1.0,
                    m_padded as i32,
                    m_logical as i32,
                    k as i32,
                )
            }
        }
    };
    assert_eq!(rc, 0, "quantize rc");
    stream.synchronize().unwrap();
    let packed = stream.clone_dtoh(&packed_dev).unwrap();
    let scales = stream.clone_dtoh(&scales_dev).unwrap();
    (packed, scales)
}

#[test]
fn rows_launcher_matches_padded_on_logical_rows() {
    for &(m_padded, m_logical, k) in &[
        (128usize, 4usize, 5376usize),
        (128, 1, 2048),
        (128, 8, 336 * 16),
    ] {
        let (p_full, s_full) = quantize(m_padded, m_logical, k, false);
        let (p_rows, s_rows) = quantize(m_padded, m_logical, k, true);
        assert_eq!(
            &p_full[..m_logical * k / 2],
            &p_rows[..m_logical * k / 2],
            "packed rows differ for m={m_logical} k={k}"
        );
        let k_blocks = k / BLOCK_SIZE;
        for row in 0..m_logical {
            for kb in 0..k_blocks {
                let dst = swizzled_dst(row, kb, k_blocks);
                assert_eq!(
                    s_full[dst], s_rows[dst],
                    "scale byte differs at row {row} kb {kb} (m={m_logical} k={k})"
                );
            }
        }
    }
}
