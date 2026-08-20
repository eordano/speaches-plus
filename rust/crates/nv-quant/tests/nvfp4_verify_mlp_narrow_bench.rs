#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{Nvfp4GemmRunner, WORKSPACE_BYTES};
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

#[path = "nvfp4_true_m_common.rs"]
mod common;
use common::quantize_dev;

const Q38_HIDDEN: usize = 5120;
const Q38_INTER: usize = 17408;
const M_LOGICAL_COVERS_THE_VERIFY_RANGE: usize = 8;
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
    for _ in 0..5 {
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

fn rel_rms(got: &[bf16], expect: &[bf16]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        let gv = g.to_f64();
        let ev = e.to_f64();
        num += (gv - ev) * (gv - ev);
        den += ev * ev;
    }
    (num / den.max(1e-30)).sqrt()
}

struct WeightSet {
    packed: Vec<CudaSlice<u8>>,
    scales: Vec<CudaSlice<u8>>,
}

fn weight_set(stream: &Arc<CudaStream>, rng: &mut Lcg, n: usize, k: usize) -> WeightSet {
    let w_host = rng.bf16_words(n * k, 0.05);
    #[allow(deprecated)]
    let w_dev: CudaSlice<u16> = stream.clone_htod(&w_host).unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    for _ in 0..L2_COLD_WEIGHT_ROTATION {
        let (wq, wsf) = quantize_dev(stream, &w_dev, n, n, k, 1.0);
        packed.push(wq);
        scales.push(wsf);
    }
    WeightSet { packed, scales }
}

fn dtoh_bf16(stream: &Arc<CudaStream>, d: &CudaSlice<bf16>) -> Vec<bf16> {
    stream.memcpy_dtov(d).unwrap()
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- prices the verify-scope dense-mlp GEMM arms (LT pad-128 serving baseline vs LT narrow pads 16/32/64 vs cutlass 128-tile true-m DP/k256/stream-K) at exact q38 shapes so the NV_Q38_VERIFY_MLP_NARROW default can cite a per-layer us table"]
fn q38_verify_mlp_narrow_kernel_ab() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run the verify mlp narrow bench");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    #[allow(deprecated)]
    let alpha_dev: CudaSlice<f32> = stream.clone_htod(&[1.0f32]).unwrap();
    let mut ws: CudaSlice<u8> = stream.alloc_zeros::<u8>(WORKSPACE_BYTES).unwrap();

    let m_l = M_LOGICAL_COVERS_THE_VERIFY_RANGE;
    let lt_arms_only = std::env::var("NV_VMLP_ARMS").ok().as_deref() == Some("lt");
    let mut layer_us: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for (name, n, k, layer_mult) in [
        ("gate_or_up", Q38_INTER, Q38_HIDDEN, 2.0f64),
        ("down", Q38_HIDDEN, Q38_INTER, 1.0f64),
    ] {
        let wset = weight_set(&stream, &mut rng, n, k);
        let x_host = rng.bf16_words(m_l * k, 1.0);
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_host).unwrap();

        let (a128, s128) = quantize_dev(&stream, &x_dev, 128, m_l, k, 1.0);
        let mut d128: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(128 * n).unwrap();
        runner
            .matmul_scaled_alpha_dev(
                &a128,
                &s128,
                &wset.packed[0],
                &wset.scales[0],
                &mut d128,
                128,
                n as u64,
                k as u64,
                &alpha_dev,
                1.0,
            )
            .unwrap();
        let reference = dtoh_bf16(&stream, &d128);
        let reference = &reference[..m_l * n];

        {
            let mut it = 0usize;
            let us = bench_us(&stream, 200, || {
                let wi = it % L2_COLD_WEIGHT_ROTATION;
                it += 1;
                runner
                    .matmul_scaled_alpha_dev(
                        &a128,
                        &s128,
                        &wset.packed[wi],
                        &wset.scales[wi],
                        &mut d128,
                        128,
                        n as u64,
                        k as u64,
                        &alpha_dev,
                        1.0,
                    )
                    .unwrap();
            });
            let weight_bytes = n * k / 2 + wset.scales[0].len();
            let gbs = weight_bytes as f64 / us / 1e3;
            eprintln!(
                "Q38-VMLP-NARROW {name} arm=lt_pad128 n={n} k={k} us={us:.2} weight_gbs={gbs:.0}"
            );
            *layer_us.entry("lt_pad128".into()).or_default() += us * layer_mult;
        }

        for m_pad in [16usize, 32, 64] {
            let (aq, asf) = quantize_dev(&stream, &x_dev, m_pad, m_l, k, 1.0);
            let mut d: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m_pad * n).unwrap();
            match runner.matmul_scaled_lt_narrow_m(
                &aq,
                &asf,
                &wset.packed[0],
                &wset.scales[0],
                &mut d,
                m_pad as u64,
                n as u64,
                k as u64,
                1.0,
            ) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("Q38-VMLP-NARROW {name} arm=lt_pad{m_pad} UNSUPPORTED: {e}");
                    continue;
                }
            }
            let got = dtoh_bf16(&stream, &d);
            let rr = rel_rms(&got[..m_l * n], reference);
            let mut it = 0usize;
            let us = bench_us(&stream, 200, || {
                let wi = it % L2_COLD_WEIGHT_ROTATION;
                it += 1;
                runner
                    .matmul_scaled_lt_narrow_m(
                        &aq,
                        &asf,
                        &wset.packed[wi],
                        &wset.scales[wi],
                        &mut d,
                        m_pad as u64,
                        n as u64,
                        k as u64,
                        1.0,
                    )
                    .unwrap();
            });
            let weight_bytes = n * k / 2 + wset.scales[0].len();
            let gbs = weight_bytes as f64 / us / 1e3;
            eprintln!(
                "Q38-VMLP-NARROW {name} arm=lt_pad{m_pad} n={n} k={k} us={us:.2} weight_gbs={gbs:.0} rel_rms_vs_lt128={rr:.2e}"
            );
            assert!(
                rr < 3e-2,
                "lt_pad{m_pad} diverged from lt_pad128 on {name}: rel_rms={rr}"
            );
            *layer_us.entry(format!("lt_pad{m_pad}")).or_default() += us * layer_mult;
        }

        for m in [3usize, 4, 6, 8] {
            if lt_arms_only {
                break;
            }
            let (aq, asf) = quantize_dev(&stream, &x_dev, m, m.min(m_l), k, 1.0);
            let mut d: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();
            for (arm, tile, sk) in [
                ("ctl_dp128", -1i32, 0i32),
                ("ctl_dp_k256", 1, 0),
                ("ctl_sk", -1, 1),
            ] {
                let launch = |runner_ws: &mut CudaSlice<u8>, d: &mut CudaSlice<bf16>| {
                    let (ap, _ga) = aq.device_ptr(&stream);
                    let (asp, _gas) = asf.device_ptr(&stream);
                    let (wp, _gw) = wset.packed[0].device_ptr(&stream);
                    let (wsp, _gws) = wset.scales[0].device_ptr(&stream);
                    let (gp, _gg) = alpha_dev.device_ptr(&stream);
                    let (dp, _gd) = d.device_ptr_mut(&stream);
                    let (wsptr, _gwsb) = runner_ws.device_ptr_mut(&stream);
                    let res = unsafe {
                        if tile >= 0 {
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
                        } else if sk == 1 {
                            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_streamk(
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
                                wsptr as *mut c_void,
                                WORKSPACE_BYTES,
                            )
                        } else {
                            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16(
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
                                wsptr as *mut c_void,
                                WORKSPACE_BYTES,
                            )
                        }
                    };
                    res.map(|_| ())
                };
                if let Err(rc) = launch(&mut ws, &mut d) {
                    eprintln!("Q38-VMLP-NARROW {name} arm={arm} m={m} UNSUPPORTED rc={rc}");
                    continue;
                }
                let got = dtoh_bf16(&stream, &d);
                let take = m.min(m_l) * n;
                let rr = rel_rms(&got[..take], &reference[..take]);
                let mut it = 0usize;
                let us = bench_us(&stream, 200, || {
                    let wi = it % L2_COLD_WEIGHT_ROTATION;
                    let _ = wi;
                    it += 1;
                    let (ap, _ga) = aq.device_ptr(&stream);
                    let (asp, _gas) = asf.device_ptr(&stream);
                    let (wp, _gw) = wset.packed[it % L2_COLD_WEIGHT_ROTATION].device_ptr(&stream);
                    let (wsp, _gws) =
                        wset.scales[it % L2_COLD_WEIGHT_ROTATION].device_ptr(&stream);
                    let (gp, _gg) = alpha_dev.device_ptr(&stream);
                    let (dp, _gd) = d.device_ptr_mut(&stream);
                    let (wsptr, _gwsb) = ws.device_ptr_mut(&stream);
                    let res = unsafe {
                        if tile >= 0 {
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
                        } else if sk == 1 {
                            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_streamk(
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
                                wsptr as *mut c_void,
                                WORKSPACE_BYTES,
                            )
                        } else {
                            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16(
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
                                wsptr as *mut c_void,
                                WORKSPACE_BYTES,
                            )
                        }
                    };
                    res.map(|_| ()).expect("cutlass launch");
                });
                let weight_bytes = n * k / 2 + wset.scales[0].len();
                let gbs = weight_bytes as f64 / us / 1e3;
                eprintln!(
                    "Q38-VMLP-NARROW {name} arm={arm} m={m} n={n} k={k} us={us:.2} weight_gbs={gbs:.0} rel_rms_vs_lt128={rr:.2e}"
                );
                assert!(
                    rr < 3e-2,
                    "{arm} m={m} diverged from lt_pad128 on {name}: rel_rms={rr}"
                );
                *layer_us.entry(format!("{arm}_m{m}")).or_default() += us * layer_mult;
            }

            {
                let dual = layer_mult > 1.5;
                let mut d_mma: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();
                let mut d_mma_b: Option<CudaSlice<bf16>> = if dual {
                    Some(stream.alloc_zeros::<bf16>(m * n).unwrap())
                } else {
                    None
                };
                let mm = m.min(m_l);
                let launch_mma = |d: &mut CudaSlice<bf16>,
                                  db: Option<&mut CudaSlice<bf16>>,
                                  wi: usize| {
                    let (xp, _gx) = x_dev.device_ptr(&stream);
                    let (wp, _gw) = wset.packed[wi].device_ptr(&stream);
                    let (wsp, _gws) = wset.scales[wi].device_ptr(&stream);
                    let wj = (wi + 1) % L2_COLD_WEIGHT_ROTATION;
                    let (wpb, _gwb) = wset.packed[wj].device_ptr(&stream);
                    let (wspb, _gwsb) = wset.scales[wj].device_ptr(&stream);
                    let (dp, _gd) = d.device_ptr_mut(&stream);
                    let (pb, sb, dpb) = match db {
                        Some(b) => {
                            let (p, _gb) = b.device_ptr_mut(&stream);
                            (wpb as *const u8, wspb as *const u8, p as *mut u16)
                        }
                        None => (
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null_mut(),
                        ),
                    };
                    unsafe {
                        nv_kernels::cuda::gemm_nvfp4_w4a16_mk_dual(
                            stream.cu_stream() as *mut c_void,
                            wp as *const u8,
                            wsp as *const u8,
                            pb,
                            sb,
                            xp as *const u16,
                            dp as *mut u16,
                            dpb,
                            1.0,
                            1.0,
                            mm as i32,
                            n as i32,
                            k as i32,
                        )
                    }
                };
                let rc = launch_mma(&mut d_mma, d_mma_b.as_mut(), 0);
                if rc != 0 {
                    eprintln!("Q38-VMLP-NARROW {name} arm=mk_mma m={m} UNSUPPORTED rc={rc}");
                } else {
                    let got = dtoh_bf16(&stream, &d_mma);
                    let take = mm * n;
                    let rr = rel_rms(&got[..take], &reference[..take]);
                    let mut it = 0usize;
                    let us = bench_us(&stream, 200, || {
                        it += 1;
                        let rc = launch_mma(
                            &mut d_mma,
                            d_mma_b.as_mut(),
                            it % L2_COLD_WEIGHT_ROTATION,
                        );
                        assert_eq!(rc, 0, "gemm_nvfp4_w4a16_mk_dual rc={rc}");
                    });
                    let arms = if dual { 2.0 } else { 1.0 };
                    let weight_bytes = (n * k / 2 + wset.scales[0].len()) as f64 * arms;
                    let gbs = weight_bytes / us / 1e3;
                    let label = if dual { "mk_mma_dual" } else { "mk_mma" };
                    eprintln!(
                        "Q38-VMLP-NARROW {name} arm={label} m={m} n={n} k={k} us={us:.2} weight_gbs={gbs:.0} rel_rms_vs_lt128={rr:.2e}"
                    );
                    let mk_mma_reads_bf16_acts_so_it_differs_from_the_a4_route_by_activation_quant_noise = 1.5e-1;
                    assert!(
                        rr < mk_mma_reads_bf16_acts_so_it_differs_from_the_a4_route_by_activation_quant_noise,
                        "mk_mma m={m} diverged from lt_pad128 on {name}: rel_rms={rr}"
                    );
                    *layer_us.entry(format!("mk_mma_m{m}")).or_default() += us;
                }
            }
        }
    }

    let mut rows: Vec<(String, f64)> = layer_us.into_iter().collect();
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (arm, us) in rows {
        eprintln!("Q38-VMLP-NARROW-LAYER arm={arm} layer_us={us:.1}");
    }
}
