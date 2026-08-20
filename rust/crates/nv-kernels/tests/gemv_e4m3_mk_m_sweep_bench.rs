#![cfg(feature = "cuda")]

mod common;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

fn bench_us(stream: &Arc<CudaStream>, iters: usize, mut launch: impl FnMut()) -> f64 {
    for _ in 0..3 {
        launch();
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        launch();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- sweeps gemv_e4m3_mk m=1..8 on N=17408 K=5120 (the q38 dense mlp gate/up decode shape) and N=6144 K=5120 (o_proj/out_proj) reporting us + GB/s over fp8 weight bytes; the m>=2 xf32-staged-smem arm pays the bf16->f32 x conversion once per element instead of once per output row per chunk"]
fn gemv_e4m3_mk_m_sweep_gbs_on_q38_decode_shapes() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run the m-sweep bench");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    for (n, k) in [(17408usize, 5120usize), (6144, 5120)] {
        let wq: Vec<u8> = (0..n * k).map(|_| (rng.next_u32() & 0xff) as u8).collect();
        let row_scale: Vec<f32> = (0..n).map(|_| 0.01 + rng.next_f32().abs() * 0.01).collect();
        #[allow(deprecated)]
        let dw: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
        #[allow(deprecated)]
        let ds: CudaSlice<f32> = stream.clone_htod(&row_scale).unwrap();
        for m in [1usize, 2, 4, 8] {
            let x = rng.bf16_words(m * k, 1.0);
            #[allow(deprecated)]
            let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
            let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
            let us = bench_us(&stream, 200, || {
                let (pw, _a) = dw.device_ptr(&stream);
                let (ps, _b) = ds.device_ptr(&stream);
                let (px, _c) = dx.device_ptr(&stream);
                let (py, _d) = dy.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gemv_e4m3_mk(
                        stream.cu_stream() as *mut c_void,
                        pw as *const u8,
                        ps as *const f32,
                        px as *const u16,
                        py as *mut u16,
                        n as i32,
                        k as i32,
                        m as i32,
                    )
                };
                assert_eq!(rc, 0, "gemv_e4m3_mk rc={rc} (n={n} k={k} m={m})");
            });
            let gbs = (n * k) as f64 / us / 1e3;
            eprintln!(
                "GEMV-E4M3-MK-SWEEP n={n} k={k} m={m} us={us:.1} weight_gbs={gbs:.0} rows_per_us={:.2}",
                m as f64 / us
            );
        }
    }
}
