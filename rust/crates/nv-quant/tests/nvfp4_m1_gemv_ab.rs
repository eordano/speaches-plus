#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::Nvfp4GemmRunner;
use std::ffi::c_void;

#[path = "nvfp4_true_m_common.rs"]
mod common;
use common::quantize_dev;

const GEMV_VS_F64_ORACLE_REL_RMS_TOL_IS_F32_ACCUM_NOISE_ONLY: f32 = 1e-4;

fn ctx_or_skip() -> Option<std::sync::Arc<CudaContext>> {
    match CudaContext::new(0) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skip: no CUDA device 0: {e}");
            None
        }
    }
}

fn gen_bf16(n: usize, seed: f32, scale: f32) -> Vec<u16> {
    (0..n)
        .map(|j| bf16::from_f32(((j as f32) * seed).sin() * scale).to_bits())
        .collect()
}

fn run_gemv_bf16act(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    w_q: &CudaSlice<u8>,
    w_sf: &CudaSlice<u8>,
    x_dev: &CudaSlice<u16>,
    y_dev: &mut CudaSlice<u16>,
    alpha: f32,
    n: usize,
    k: usize,
) {
    let rc = {
        let (wp, _g0) = w_q.device_ptr(stream);
        let (sp, _g1) = w_sf.device_ptr(stream);
        let (xp, _g2) = x_dev.device_ptr(stream);
        let (yp, _g3) = y_dev.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::nvfp4_gemv_bf16act(
                stream.cu_stream() as *mut c_void,
                wp as *const u8,
                sp as *const u8,
                xp as *const u16,
                yp as *mut u16,
                alpha,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "nvfp4_gemv_bf16act rc={rc}");
}

#[test]
fn m1_gemv_bf16act_matches_the_swizzled_dequant_oracle() {
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    let stream = ctx.default_stream();
    for &(n, k, alpha) in &[
        (64usize, 256usize, 1.0f32),
        (129, 1024, 1.0),
        (640, 5376, 0.5),
        (5376, 5376, 1.0),
    ] {
        let w_host = gen_bf16(n * k, 0.00071, 1.0);
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        let (w_q, w_sf) = quantize_dev(&stream, &w_dev, n, n, k, 1.0);
        #[allow(deprecated)]
        let w_q_host: Vec<u8> = stream.memcpy_dtov(&w_q).unwrap();
        #[allow(deprecated)]
        let w_sf_host: Vec<u8> = stream.memcpy_dtov(&w_sf).unwrap();
        let w_flat = nv_quant::nvfp4::dequantize_packed_swizzled(&w_q_host, &w_sf_host, n, k, 1.0);

        let x_host = gen_bf16(k, 0.00131, 2.0);
        #[allow(deprecated)]
        let x_dev = stream.clone_htod(&x_host).unwrap();
        let mut y_dev: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        run_gemv_bf16act(&stream, &w_q, &w_sf, &x_dev, &mut y_dev, alpha, n, k);
        #[allow(deprecated)]
        let y_host: Vec<u16> = stream.memcpy_dtov(&y_dev).unwrap();

        let want: Vec<f32> = (0..n)
            .map(|row| {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += w_flat[row * k + kk] as f64 * bf16::from_bits(x_host[kk]).to_f64();
                }
                (acc * alpha as f64) as f32
            })
            .collect();
        let rms =
            (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt() as f32;
        let mut worst = (0f32, 0usize);
        for (row, (g, w)) in y_host.iter().zip(&want).enumerate() {
            let got = bf16::from_bits(*g).to_f32();
            let d = (got - w).abs() / w.abs().max(rms.max(1e-3));
            if d > worst.0 {
                worst = (d, row);
            }
        }
        let bf16_out_rounding_budget = 0.01f32;
        assert!(
            worst.0 < GEMV_VS_F64_ORACLE_REL_RMS_TOL_IS_F32_ACCUM_NOISE_ONLY
                + bf16_out_rounding_budget,
            "n={n} k={k} alpha={alpha}: row {} rel {:.3e}: the bf16act gemv disagrees with the \
             swizzled dequant oracle beyond f32-accum + bf16-output rounding",
            worst.1,
            worst.0
        );
    }
}

#[test]
#[ignore = "timing instrument; set NV_NVFP4_M1_BENCH=1"]
fn m1_gemv_bf16act_vs_padded16_lt_route_bench() {
    assert_eq!(
        std::env::var("NV_NVFP4_M1_BENCH").ok().as_deref(),
        Some("1"),
        "set NV_NVFP4_M1_BENCH=1 -- this loads the GPU with timing loops"
    );
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    let stream = ctx.default_stream();
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    #[allow(deprecated)]
    let alpha_dev = stream.clone_htod(&[1.0f32]).unwrap();

    for &(n, k) in &[(5376usize, 5376usize), (43008, 5376), (5376, 21504)] {
        let w_host = gen_bf16(n * k, 0.00071, 1.0);
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        let (w_q, w_sf) = quantize_dev(&stream, &w_dev, n, n, k, 1.0);
        let x_host = gen_bf16(k, 0.00131, 2.0);
        #[allow(deprecated)]
        let x_dev = stream.clone_htod(&x_host).unwrap();
        let mut y_dev: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

        let m_pad = 16usize;
        let (a_q, a_sf) = quantize_dev(&stream, &x_dev, m_pad, 1, k, 1.0);
        let mut d_pad: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m_pad * n).unwrap();

        let iters = 200usize;
        let time = |f: &mut dyn FnMut()| -> f64 {
            for _ in 0..10 {
                f();
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };

        let t_gemv = time(&mut || {
            run_gemv_bf16act(&stream, &w_q, &w_sf, &x_dev, &mut y_dev, 1.0, n, k);
        });
        let t_lt = time(&mut || {
            runner
                .matmul_scaled_alpha_dev(
                    &a_q,
                    &a_sf,
                    &w_q,
                    &w_sf,
                    &mut d_pad,
                    m_pad as u64,
                    n as u64,
                    k as u64,
                    &alpha_dev,
                    1.0,
                )
                .unwrap();
        });

        let weight_bytes = (n * k) as f64 / 2.0 + (n * k) as f64 / 16.0;
        let gbps = |t: f64| weight_bytes / t / 1e9;
        eprintln!(
            "-- n={n} k={k} fp4 weight+scales={:.1} MB",
            weight_bytes / 1e6
        );
        eprintln!(
            "   padded16 LT route    {:8.4} ms  eff {:7.1} GB/s",
            t_lt * 1e3,
            gbps(t_lt)
        );
        eprintln!(
            "   bf16act gemv m=1     {:8.4} ms  eff {:7.1} GB/s  speedup {:.2}x",
            t_gemv * 1e3,
            gbps(t_gemv),
            t_lt / t_gemv
        );
    }
}
