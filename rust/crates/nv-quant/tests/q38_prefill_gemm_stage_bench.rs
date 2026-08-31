#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::fp8::Fp8GemmRunner;
use nv_quant::nvfp4::{Nvfp4GemmRunner, WORKSPACE_BYTES};
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

const Q38_HIDDEN: usize = 5120;
const Q38_INTER: usize = 17408;
const Q38_GDN_QKV: usize = 10240;
const Q38_GDN_Z: usize = 6144;
const Q38_ATTN_QKV: usize = 8192;
const M_DEFAULT_4096_IS_THE_SERVING_PREFILL_CHUNK: usize = 4096;
const L2_COLD_WEIGHT_ROTATION: usize = 4;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bf16_words(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
}

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

fn tflops(m: usize, n: usize, k: usize, us: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / us / 1e6
}

fn quantize_nvfp4_dev(
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    m: usize,
    k: usize,
) -> (CudaSlice<u8>, CudaSlice<u8>) {
    let blocks = k / 16;
    let packed_bytes = m * k / 2;
    let scales_bytes = m.div_ceil(128) * 128 * blocks.div_ceil(4) * 4;
    let mut packed: CudaSlice<u8> = unsafe { stream.alloc::<u8>(packed_bytes).unwrap() };
    let mut scales: CudaSlice<u8> = unsafe { stream.alloc::<u8>(scales_bytes).unwrap() };
    let rc = {
        let (xp, _gx) = x.device_ptr(stream);
        let (pp, _gp) = packed.device_ptr_mut(stream);
        let (sp, _gs) = scales.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                pp as *mut u8,
                sp as *mut u8,
                1.0,
                m as i32,
                m as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "quantize_nvfp4_bf16 rc={rc}");
    (packed, scales)
}

fn rowquant_dev(
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    m: usize,
    k: usize,
) -> (CudaSlice<u8>, CudaSlice<f32>) {
    let mut q: CudaSlice<u8> = unsafe { stream.alloc::<u8>(m * k).unwrap() };
    let mut sc: CudaSlice<f32> = unsafe { stream.alloc::<f32>(m).unwrap() };
    let rc = {
        let (xp, _gx) = x.device_ptr(stream);
        let (qp, _gq) = q.device_ptr_mut(stream);
        let (sp, _gs) = sc.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::rowquant_e4m3(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                qp as *mut u8,
                sp as *mut f32,
                m as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "rowquant_e4m3 rc={rc}");
    (q, sc)
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- prices every stage of \
            the q38 chunked-prefill GEMM stack at the serving chunk shape (m=4096): the fp8 \
            route's rowquant + cublasLt e4m3 + scale_rowcol epilogue per projection shape \
            (gdn qkv/z/fused-qkvz/out, attn qkv, fp8-tail mlp) and the nvfp4 route's quantize + \
            cutlass sm120 arms (dp128 / dp_k256 / dp_n256 / streamk / cublaslt) per mlp shape, \
            so the campaign can name the achieved-vs-ceiling gap per stage instead of per layer; \
            NV_PFB_SQUARE=<s> instead runs only the nvfp4 arms at m=n=k=s, the shape class \
            published external sm_120 rooflines use"]
fn q38_prefill_gemm_stage_ladder_at_serving_chunk_m() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        panic!(
            "NV_KERNELS_BENCH is not 1, so the prefill gemm stage ladder executed nothing. It is \
             already #[ignore]d, which makes --ignored the opt-in; a second env gate behind that \
             can only make an --ignored run print `1 passed` for a ladder that measured nothing. \
             A suite whose precondition is absent must fail or be ignored, never return green."
        );
    }
    let m: usize = std::env::var("NV_PFB_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(M_DEFAULT_4096_IS_THE_SERVING_PREFILL_CHUNK);
    let iters: usize = std::env::var("NV_PFB_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x243f6a8885a308d3);

    let mut fp8_runner = Fp8GemmRunner::new(stream.clone()).unwrap();
    let ones_host = vec![1.0f32; 1];
    #[allow(deprecated)]
    let ones: CudaSlice<f32> = stream.clone_htod(&ones_host).unwrap();

    let square_mnk_for_external_roofline_cross_check: Option<usize> = std::env::var(
        "NV_PFB_SQUARE",
    )
    .ok()
    .and_then(|v| v.parse().ok());

    eprintln!("PFGEMM-STAGE m={m} iters={iters} rotation={L2_COLD_WEIGHT_ROTATION}");

    for (name, n, k) in match square_mnk_for_external_roofline_cross_check {
        Some(_) => vec![],
        None => vec![
            ("gdn_in_qkv", Q38_GDN_QKV, Q38_HIDDEN),
            ("gdn_in_z", Q38_GDN_Z, Q38_HIDDEN),
            ("gdn_in_qkvz_fused", Q38_GDN_QKV + Q38_GDN_Z, Q38_HIDDEN),
            ("gdn_out", Q38_HIDDEN, Q38_GDN_Z),
            ("attn_qkv_fused", Q38_ATTN_QKV, Q38_HIDDEN),
            ("mlp_gate_or_up_fp8tail", Q38_INTER, Q38_HIDDEN),
            ("mlp_gateup_fused_fp8tail", 2 * Q38_INTER, Q38_HIDDEN),
            ("mlp_down_fp8tail", Q38_HIDDEN, Q38_INTER),
        ],
    } {
        let x_host = rng.bf16_words(m * k, 1.0);
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_host).unwrap();

        let rq_us = bench_us(&stream, iters, || {
            let _ = rowquant_dev(&stream, &x_dev, m, k);
        });

        let (xq, _xsc) = rowquant_dev(&stream, &x_dev, m, k);
        let mut wq_rot: Vec<CudaSlice<u8>> = Vec::new();
        for _ in 0..L2_COLD_WEIGHT_ROTATION {
            let w_host = rng.bf16_words(n * k, 0.05);
            #[allow(deprecated)]
            let w_dev: CudaSlice<u16> = stream.clone_htod(&w_host).unwrap();
            let (wq, _wsc) = rowquant_dev(&stream, &w_dev, n, k);
            wq_rot.push(wq);
        }
        let mut d: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();

        let mut it = 0usize;
        let gemm_us = bench_us(&stream, iters, || {
            let wi = it % L2_COLD_WEIGHT_ROTATION;
            it += 1;
            fp8_runner
                .matmul_e4m3_weight_row(
                    &xq,
                    &wq_rot[wi],
                    &mut d,
                    m as u64,
                    n as u64,
                    k as u64,
                    &ones,
                    &ones,
                )
                .unwrap();
        });

        let row_sc_host = vec![1.0f32; m];
        let col_sc_host = vec![1.0f32; n];
        #[allow(deprecated)]
        let row_sc: CudaSlice<f32> = stream.clone_htod(&row_sc_host).unwrap();
        #[allow(deprecated)]
        let col_sc: CudaSlice<f32> = stream.clone_htod(&col_sc_host).unwrap();
        let epi_us = bench_us(&stream, iters, || {
            let (dp, _gd) = d.device_ptr_mut(&stream);
            let (rp, _gr) = row_sc.device_ptr(&stream);
            let (cp, _gc) = col_sc.device_ptr(&stream);
            let rc = unsafe {
                nv_kernels::cuda::scale_rowcol_bf16(
                    stream.cu_stream() as *mut c_void,
                    dp as *mut u16,
                    rp as *const f32,
                    cp as *const f32,
                    m as i32,
                    n as i32,
                )
            };
            assert_eq!(rc, 0, "scale_rowcol_bf16 rc={rc}");
        });

        let chain_us = rq_us + gemm_us + epi_us;
        eprintln!(
            "PFGEMM-STAGE fp8 {name} m={m} n={n} k={k} rowquant_us={rq_us:.1} gemm_us={gemm_us:.1} \
             epilogue_us={epi_us:.1} chain_us={chain_us:.1} gemm_tflops={:.0} chain_tflops={:.0}",
            tflops(m, n, k, gemm_us),
            tflops(m, n, k, chain_us),
        );
    }

    drop(fp8_runner);
    let mut nvfp4_runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    #[allow(deprecated)]
    let alpha_dev: CudaSlice<f32> = stream.clone_htod(&[1.0f32]).unwrap();
    let mut ws: CudaSlice<u8> = stream.alloc_zeros::<u8>(WORKSPACE_BYTES).unwrap();

    let m = square_mnk_for_external_roofline_cross_check.unwrap_or(m);
    for (name, n, k) in match square_mnk_for_external_roofline_cross_check {
        Some(s) => vec![("square_external_roofline", s, s)],
        None => vec![
            ("mlp_gate_or_up", Q38_INTER, Q38_HIDDEN),
            ("mlp_gateup_fused", 2 * Q38_INTER, Q38_HIDDEN),
            ("mlp_down", Q38_HIDDEN, Q38_INTER),
        ],
    } {
        let x_host = rng.bf16_words(m * k, 1.0);
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_host).unwrap();

        let q_us = bench_us(&stream, iters, || {
            let _ = quantize_nvfp4_dev(&stream, &x_dev, m, k);
        });

        let (aq, asf) = quantize_nvfp4_dev(&stream, &x_dev, m, k);
        let mut w_rot: Vec<(CudaSlice<u8>, CudaSlice<u8>)> = Vec::new();
        for _ in 0..L2_COLD_WEIGHT_ROTATION {
            let w_host = rng.bf16_words(n * k, 0.05);
            #[allow(deprecated)]
            let w_dev: CudaSlice<u16> = stream.clone_htod(&w_host).unwrap();
            w_rot.push(quantize_nvfp4_dev(&stream, &w_dev, n, k));
        }
        let mut d: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();

        let arm_filter = std::env::var("NV_PFB_ARM").ok();
        for (arm, tile, sk) in [
            ("ctl_dp128", 0i32, 0i32),
            ("ctl_dp_k256", 1, 0),
            ("ctl_dp_n256", 2, 0),
            ("ctl_sk", 0, 1),
            ("ctl_sk_k256", 1, 1),
            ("v192_12warp", -1, 0),
        ] {
            if arm_filter.as_deref().is_some_and(|f| f != arm) {
                continue;
            }
            let mut it = 0usize;
            let mut failed = false;
            {
                let (ap, _ga) = aq.device_ptr(&stream);
                let (asp, _gas) = asf.device_ptr(&stream);
                let (wp, _gw) = w_rot[0].0.device_ptr(&stream);
                let (wsp, _gws) = w_rot[0].1.device_ptr(&stream);
                let (gp, _gg) = alpha_dev.device_ptr(&stream);
                let (dp, _gd) = d.device_ptr_mut(&stream);
                let (wsptr, _gwsb) = ws.device_ptr_mut(&stream);
                let res = if tile < 0 {
                    unsafe {
                        nv_kernels::cuda::nvfp4_gemm_sm120_v192_bf16(
                            stream.cu_stream() as *mut c_void,
                            ap as *const c_void,
                            asp as *const c_void,
                            wp as *const c_void,
                            wsp as *const c_void,
                            gp as *const f32,
                            dp as *mut c_void,
                            m as i32,
                            n as i32,
                            k as i32,
                        )
                    }
                    .map(|()| 0usize)
                } else {
                    unsafe {
                        nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_tiled(
                            stream.cu_stream() as *mut c_void,
                            ap as *const c_void,
                            asp as *const c_void,
                            wp as *const c_void,
                            wsp as *const c_void,
                            gp as *const f32,
                            dp as *mut c_void,
                            m as i32,
                            n as i32,
                            k as i32,
                            tile,
                            sk,
                            wsptr as *mut c_void,
                            WORKSPACE_BYTES,
                        )
                    }
                };
                if res.is_err() {
                    eprintln!("PFGEMM-STAGE nvfp4 {name} arm={arm} UNSUPPORTED rc={res:?}");
                    failed = true;
                }
            }
            if failed {
                continue;
            }
            let us = bench_us(&stream, iters, || {
                let wi = it % L2_COLD_WEIGHT_ROTATION;
                it += 1;
                let (ap, _ga) = aq.device_ptr(&stream);
                let (asp, _gas) = asf.device_ptr(&stream);
                let (wp, _gw) = w_rot[wi].0.device_ptr(&stream);
                let (wsp, _gws) = w_rot[wi].1.device_ptr(&stream);
                let (gp, _gg) = alpha_dev.device_ptr(&stream);
                let (dp, _gd) = d.device_ptr_mut(&stream);
                let (wsptr, _gwsb) = ws.device_ptr_mut(&stream);
                if tile < 0 {
                    unsafe {
                        nv_kernels::cuda::nvfp4_gemm_sm120_v192_bf16(
                            stream.cu_stream() as *mut c_void,
                            ap as *const c_void,
                            asp as *const c_void,
                            wp as *const c_void,
                            wsp as *const c_void,
                            gp as *const f32,
                            dp as *mut c_void,
                            m as i32,
                            n as i32,
                            k as i32,
                        )
                    }
                    .expect("v192 launch");
                } else {
                    unsafe {
                        nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_tiled(
                            stream.cu_stream() as *mut c_void,
                            ap as *const c_void,
                            asp as *const c_void,
                            wp as *const c_void,
                            wsp as *const c_void,
                            gp as *const f32,
                            dp as *mut c_void,
                            m as i32,
                            n as i32,
                            k as i32,
                            tile,
                            sk,
                            wsptr as *mut c_void,
                            WORKSPACE_BYTES,
                        )
                    }
                    .expect("cutlass launch");
                }
            });
            eprintln!(
                "PFGEMM-STAGE nvfp4 {name} arm={arm} m={m} n={n} k={k} quant_us={q_us:.1} \
                 gemm_us={us:.1} gemm_tflops={:.0} chain_tflops={:.0}",
                tflops(m, n, k, us),
                tflops(m, n, k, us + q_us),
            );
        }

        {
            let mut it = 0usize;
            let us = bench_us(&stream, iters, || {
                let wi = it % L2_COLD_WEIGHT_ROTATION;
                it += 1;
                nvfp4_runner
                    .matmul_scaled_lt_narrow_m(
                        &aq,
                        &asf,
                        &w_rot[wi].0,
                        &w_rot[wi].1,
                        &mut d,
                        m as u64,
                        n as u64,
                        k as u64,
                        1.0,
                    )
                    .unwrap();
            });
            eprintln!(
                "PFGEMM-STAGE nvfp4 {name} arm=cublaslt m={m} n={n} k={k} quant_us={q_us:.1} \
                 gemm_us={us:.1} gemm_tflops={:.0} chain_tflops={:.0}",
                tflops(m, n, k, us),
                tflops(m, n, k, us + q_us),
            );
        }
    }
}
