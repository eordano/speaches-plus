#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};
use std::ffi::c_void;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn fused_matches_silu_mul_then_per_expert_quantize() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "silu_mul_quantize_nvfp4: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("silu_mul_quantize_nvfp4: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let m_per_expert = 128usize;
    let a = 4usize;
    let m_total = a * m_per_expert;
    let k = 512usize;
    let stored_globals: Vec<f32> = vec![1.5, 0.75, 3.0, 0.25];

    let mut g_host: Vec<bf16> = Vec::with_capacity(m_total * k);
    let mut u_host: Vec<bf16> = Vec::with_capacity(m_total * k);
    for r in 0..m_total {
        for j in 0..k {
            let g = ((r * 31 + j) as f32 * 0.013).sin() * 1.5;
            let u = ((r * 17 + j) as f32 * 0.019).cos() * 1.2;
            g_host.push(bf16::from_f32(g));
            u_host.push(bf16::from_f32(u));
        }
    }

    let mut y_act_rows: Vec<Vec<f32>> = Vec::with_capacity(m_total);
    for r in 0..m_total {
        let row: Vec<f32> = (0..k)
            .map(|j| {
                let g = g_host[r * k + j].to_f32();
                let u = u_host[r * k + j].to_f32();
                let v = silu(g) * u;
                bf16::from_f32(v).to_f32()
            })
            .collect();
        y_act_rows.push(row);
    }

    let mut packed_ref = vec![0u8; m_total * k / 2];
    let row_bytes = k / 2;
    let per_expert_sf_bytes = 128 * ((k / BLOCK_SIZE + 3) / 4) * 4;
    let mut scales_ref = vec![0u8; a * per_expert_sf_bytes];
    for e in 0..a {
        let lo = e * m_per_expert;
        let hi = (e + 1) * m_per_expert;
        let q = Nvfp4Tensor::quantize_rows_with_global(&y_act_rows[lo..hi], stored_globals[e]);
        let sw = swizzle_scales(&q.scales, m_per_expert, k / BLOCK_SIZE);
        packed_ref[lo * row_bytes..hi * row_bytes].copy_from_slice(&q.data);
        let off = e * per_expert_sf_bytes;
        scales_ref[off..off + per_expert_sf_bytes].copy_from_slice(&sw);
    }

    #[allow(deprecated)]
    let g_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(g_host.as_ptr() as *const u16, g_host.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let u_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(u_host.as_ptr() as *const u16, u_host.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let globals_dev = stream.memcpy_stod(&stored_globals).unwrap();
    let mut packed_dev = stream.alloc_zeros::<u8>(m_total * k / 2).unwrap();
    let scales_bytes = a * per_expert_sf_bytes;
    let mut scales_dev = stream.alloc_zeros::<u8>(scales_bytes).unwrap();

    let rc = {
        let (gp, _g1) = g_dev.device_ptr(&stream);
        let (up, _g2) = u_dev.device_ptr(&stream);
        let (gl, _g3) = globals_dev.device_ptr(&stream);
        let (pp, _g4) = packed_dev.device_ptr_mut(&stream);
        let (sp, _g5) = scales_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::silu_mul_quantize_nvfp4_bf16_per_expert(
                stream.cu_stream() as *mut c_void,
                gp as *const u16,
                up as *const u16,
                pp as *mut u8,
                sp as *mut u8,
                gl as *const f32,
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

    let total_bytes = packed_got.len();
    let mut off_by_one = 0usize;
    let mut worse = 0usize;
    for (a_byte, b_byte) in packed_got.iter().zip(packed_ref.iter()) {
        for shift in [0, 4] {
            let an = ((*a_byte >> shift) & 0xF) as i32;
            let bn = ((*b_byte >> shift) & 0xF) as i32;
            let lo_a = an & 0x7;
            let lo_b = bn & 0x7;
            let sign_a = an >> 3;
            let sign_b = bn >> 3;
            if sign_a == sign_b && (lo_a - lo_b).abs() <= 1 {
                if (lo_a - lo_b).abs() == 1 {
                    off_by_one += 1;
                }
            } else if an != bn {
                worse += 1;
            }
        }
    }
    eprintln!(
        "fused vs unfused packed: {}/{} nibbles off-by-1, {}/{} larger diff",
        off_by_one,
        total_bytes * 2,
        worse,
        total_bytes * 2,
    );
    assert!(
        worse < total_bytes * 2 / 50,
        "too many large-diff nibbles: {worse}/{}",
        total_bytes * 2
    );
}
