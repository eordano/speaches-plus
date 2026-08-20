#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};
use std::ffi::c_void;

#[test]
fn quantize_per_expert_matches_per_slice() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "quantize_nvfp4_bf16_per_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "quantize_nvfp4_bf16_per_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let stream = ctx.default_stream();

    let m_per_expert = 128usize;
    let a = 4usize;
    let m_total = a * m_per_expert;
    let k = 2048usize;

    let stored_globals: Vec<f32> = vec![1.5, 0.75, 3.0, 0.25];
    assert_eq!(stored_globals.len(), a);

    let mut host_bf: Vec<bf16> = Vec::with_capacity(m_total * k);
    for r in 0..m_total {
        let e = r / m_per_expert;
        for j in 0..k {
            let v = (((e as f32 + 1.0) * 0.3) * ((r + j) as f32 * 0.013).sin()) * 2.0;
            host_bf.push(bf16::from_f32(v));
        }
    }

    #[allow(deprecated)]
    let x_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(host_bf.as_ptr() as *const u16, host_bf.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let globals_dev = stream.memcpy_stod(&stored_globals).unwrap();

    let mut packed_dev = stream.alloc_zeros::<u8>(m_total * k / 2).unwrap();
    let scales_bytes = ((m_total + 127) / 128) * 128 * ((k / BLOCK_SIZE + 3) / 4) * 4;
    let mut scales_dev = stream.alloc_zeros::<u8>(scales_bytes).unwrap();

    let rc = {
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (gp, _g2) = globals_dev.device_ptr(&stream);
        let (pp, _g3) = packed_dev.device_ptr_mut(&stream);
        let (sp, _g4) = scales_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16_per_expert(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                pp as *mut u8,
                sp as *mut u8,
                gp as *const f32,
                m_per_expert as i32,
                m_total as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let packed_got = stream.memcpy_dtov(&packed_dev).unwrap();
    #[allow(deprecated)]
    let scales_got = stream.memcpy_dtov(&scales_dev).unwrap();

    let mut packed_ref = vec![0u8; m_total * k / 2];
    let mut scales_ref = vec![0u8; scales_bytes];
    for e in 0..a {
        let lo = e * m_per_expert;
        let hi = (e + 1) * m_per_expert;
        let rows: Vec<Vec<f32>> = (lo..hi)
            .map(|r| {
                host_bf[r * k..(r + 1) * k]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect()
            })
            .collect();
        let q = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_globals[e]);
        let sw = swizzle_scales(&q.scales, m_per_expert, k / BLOCK_SIZE);

        let row_bytes = k / 2;
        packed_ref[lo * row_bytes..hi * row_bytes].copy_from_slice(&q.data);

        let per_expert_sf_bytes = 128 * ((k / BLOCK_SIZE + 3) / 4) * 4;
        let dst_off = e * per_expert_sf_bytes;
        scales_ref[dst_off..dst_off + per_expert_sf_bytes].copy_from_slice(&sw);
    }

    assert_eq!(packed_got.len(), packed_ref.len());
    let p_mis: usize = packed_got
        .iter()
        .zip(packed_ref.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        p_mis,
        0,
        "packed FP4 differs at {p_mis}/{} bytes between GPU per-expert and CPU per-slice ref",
        packed_got.len()
    );
    let s_mis: usize = scales_got
        .iter()
        .zip(scales_ref.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        s_mis,
        0,
        "swizzled scales differ at {s_mis}/{} bytes",
        scales_got.len()
    );
}
