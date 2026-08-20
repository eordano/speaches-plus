#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

const Q38_HIDDEN: usize = 5120;
const Q38_INTER: usize = 17408;
const Q38_CONV_DIM: usize = 10240;
const Q38_CONV_K: usize = 4;
const Q38_VALUE_DIM: usize = 6144;
const Q38_GDN_NK: usize = 16;
const Q38_GDN_NV: usize = 48;
const Q38_GDN_DK: usize = 128;
const Q38_GDN_DV: usize = 128;

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
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u32() & 0xff) as u8).collect()
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

fn nvfp4_scale_bytes(n: usize, k: usize) -> usize {
    let k_tiles = (k / 16).div_ceil(4);
    let m_tiles = n.div_ceil(128);
    m_tiles * k_tiles * 512
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- prices one graph-captured stream-ordered alloc+free pair at replay by diffing a 256-iter alloc-kernel-free graph against a prealloc-kernel graph, because the q38 decode graph carries hundreds of per-step scratch alloc nodes"]
fn q38_graph_alloc_free_node_pair_replay_cost() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run the graph alloc node bench");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.new_stream().expect("capture stream");
    let mut rng = Lcg(0x2545f4914f6cdd1d);
    let iters = 256usize;
    let hidden = 1024usize;
    let x = rng.bf16_words(hidden, 1.0);
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let kernel_on = |s: &Arc<CudaStream>, px: u64, py: u64| {
        let rc = unsafe {
            cuda::rmsnorm_bf16(
                s.cu_stream() as *mut c_void,
                px as *const u16,
                px as *const u16,
                py as *mut u16,
                1,
                hidden,
                1e-6,
            )
        };
        assert_eq!(rc, 0, "rmsnorm_bf16 rc={rc}");
    };

    let mut prealloc_bufs: Vec<CudaSlice<u16>> = (0..iters)
        .map(|_| stream.alloc_zeros::<u16>(hidden).unwrap())
        .collect();
    let mut runner_pre = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
    let replay = |runner: &mut nv_kernels::graph::CudaGraphRunner,
                  token: u64,
                  f: &dyn Fn(&Arc<CudaStream>)| {
        runner
            .run(token, |s| {
                f(s);
                Ok(())
            })
            .unwrap();
        runner.stream().synchronize().unwrap();
        let reps = 100usize;
        let t0 = Instant::now();
        for _ in 0..reps {
            runner
                .run(token, |_| {
                    panic!("recapture on a cached token");
                })
                .unwrap();
        }
        runner.stream().synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };

    let (pdx, _gx) = dx.device_ptr(&stream);
    let pre_ptrs: Vec<u64> = prealloc_bufs
        .iter_mut()
        .map(|b| {
            let (p, _g) = b.device_ptr_mut(&stream);
            p
        })
        .collect();
    let pre_us = replay(&mut runner_pre, 1, &|s| {
        for py in pre_ptrs.iter() {
            kernel_on(s, pdx, *py);
        }
    });

    let mut runner_alloc = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
    let alloc_us = replay(&mut runner_alloc, 2, &|s| {
        for _ in 0..iters {
            let mut tmp: CudaSlice<u16> = unsafe { s.alloc::<u16>(hidden).unwrap() };
            let (py, _g) = tmp.device_ptr_mut(s);
            kernel_on(s, pdx, py);
        }
    });

    let per_pair_ns = (alloc_us - pre_us) * 1e3 / iters as f64;
    eprintln!(
        "Q38-GRAPH-ALLOC-NODE iters={iters} prealloc_graph_us={pre_us:.1} allocfree_graph_us={alloc_us:.1} per_alloc_free_pair_ns={per_pair_ns:.0}"
    );
}

const Q38_VOCAB: usize = 248320;

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- prices the draft lm_head m=1 arms at the exact [248320 x 5120] serving shape: the resident fp8 gemv_e4m3_mk baseline against the nvfp4 candidates (w4a16 full-n, w4a16 dual over row halves, w4a8 dual over row halves), so the NV_Q38_DRAFT_LMHEAD_NVFP4 serving route ships the measured winner"]
fn q38_draft_lmhead_m1_gemv_arm_floor() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run the draft lm_head arm floor bench");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0xd1b54a32d192ed03);
    let n = Q38_VOCAB;
    let k = Q38_HIDDEN;
    let n2 = n / 2;
    assert_eq!(n2 % 128, 0, "half-split rows must stay swizzle-tile aligned");
    let iters = 100usize;

    let x = rng.bf16_words(k, 1.0);
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let xq: Vec<u8> = rng.bytes(k);
    #[allow(deprecated)]
    let dxq: CudaSlice<u8> = stream.clone_htod(&xq).unwrap();
    #[allow(deprecated)]
    let dxs: CudaSlice<f32> = stream.clone_htod(&[0.05f32]).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

    {
        let wq = rng.bytes(n * k);
        let row_scale: Vec<f32> = (0..n).map(|_| 0.01 + rng.next_f32().abs() * 0.01).collect();
        let dws: Vec<CudaSlice<u8>> = (0..2)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
                d
            })
            .collect();
        #[allow(deprecated)]
        let ds: CudaSlice<f32> = stream.clone_htod(&row_scale).unwrap();
        let mut it = 0usize;
        let us = bench_us(&stream, iters, || {
            let (pw, _a) = dws[it % 2].device_ptr(&stream);
            it += 1;
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
                    1,
                )
            };
            assert_eq!(rc, 0, "gemv_e4m3_mk rc={rc}");
        });
        let gbs = (n * k) as f64 / us / 1e3;
        eprintln!("Q38-LMHEAD-FLOOR fp8_baseline gemv_e4m3_mk n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }

    {
        let wq = rng.bytes(n * k / 2);
        let sc = rng.bytes(nvfp4_scale_bytes(n, k));
        let dws: Vec<CudaSlice<u8>> = (0..2)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
                d
            })
            .collect();
        #[allow(deprecated)]
        let dsc: CudaSlice<u8> = stream.clone_htod(&sc).unwrap();
        let mut it = 0usize;
        let us = bench_us(&stream, iters, || {
            let (pw, _a) = dws[it % 2].device_ptr(&stream);
            it += 1;
            let (psc, _b) = dsc.device_ptr(&stream);
            let (px, _c) = dx.device_ptr(&stream);
            let (py, _d) = dy.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::nvfp4_gemv_bf16act(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u8,
                    psc as *const u8,
                    px as *const u16,
                    py as *mut u16,
                    1.0,
                    n as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "nvfp4_gemv_bf16act rc={rc}");
        });
        let bytes = (n * k) as f64 / 2.0 + nvfp4_scale_bytes(n, k) as f64;
        let gbs = bytes / us / 1e3;
        eprintln!("Q38-LMHEAD-FLOOR w4a16_full nvfp4_gemv_bf16act n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }

    {
        let wq_h = rng.bytes(n2 * k / 2);
        let sc_h = rng.bytes(nvfp4_scale_bytes(n2, k));
        let dwa: Vec<CudaSlice<u8>> = (0..2)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq_h).unwrap();
                d
            })
            .collect();
        let dwb: Vec<CudaSlice<u8>> = (0..2)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq_h).unwrap();
                d
            })
            .collect();
        #[allow(deprecated)]
        let dsa: CudaSlice<u8> = stream.clone_htod(&sc_h).unwrap();
        #[allow(deprecated)]
        let dsb: CudaSlice<u8> = stream.clone_htod(&sc_h).unwrap();
        let mut dya: CudaSlice<u16> = stream.alloc_zeros::<u16>(n2).unwrap();
        let mut dyb: CudaSlice<u16> = stream.alloc_zeros::<u16>(n2).unwrap();

        let mut it = 0usize;
        let us = bench_us(&stream, iters, || {
            let (pwa, _a) = dwa[it % 2].device_ptr(&stream);
            let (pwb, _b) = dwb[it % 2].device_ptr(&stream);
            it += 1;
            let (psa, _c) = dsa.device_ptr(&stream);
            let (psb, _d) = dsb.device_ptr(&stream);
            let (px, _e) = dx.device_ptr(&stream);
            let (pya, _f) = dya.device_ptr_mut(&stream);
            let (pyb, _g) = dyb.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a16_dual_m1(
                    stream.cu_stream() as *mut c_void,
                    pwa as *const u8,
                    psa as *const u8,
                    pwb as *const u8,
                    psb as *const u8,
                    px as *const u16,
                    pya as *mut u16,
                    pyb as *mut u16,
                    1.0,
                    1.0,
                    n2 as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_nvfp4_w4a16_dual_m1 rc={rc}");
        });
        let bytes = (n * k) as f64 / 2.0 + 2.0 * nvfp4_scale_bytes(n2, k) as f64;
        let gbs = bytes / us / 1e3;
        eprintln!("Q38-LMHEAD-FLOOR w4a16_dual_halves gemv_nvfp4_w4a16_dual_m1 n_half={n2} k={k} us={us:.2} weight_gbs={gbs:.0}");

        let mut it = 0usize;
        let us = bench_us(&stream, iters, || {
            let (pwa, _a) = dwa[it % 2].device_ptr(&stream);
            let (pwb, _b) = dwb[it % 2].device_ptr(&stream);
            it += 1;
            let (psa, _c) = dsa.device_ptr(&stream);
            let (psb, _d) = dsb.device_ptr(&stream);
            let (pxq, _e) = dxq.device_ptr(&stream);
            let (pxs, _f) = dxs.device_ptr(&stream);
            let (pya, _g) = dya.device_ptr_mut(&stream);
            let (pyb, _h) = dyb.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_dual_m1(
                    stream.cu_stream() as *mut c_void,
                    pwa as *const u8,
                    psa as *const u8,
                    pwb as *const u8,
                    psb as *const u8,
                    pxq as *const i8,
                    pxs as *const f32,
                    pya as *mut u16,
                    pyb as *mut u16,
                    1.0,
                    1.0,
                    n2 as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_nvfp4_w4a8_dual_m1 rc={rc}");
        });
        let bytes = (n * k) as f64 / 2.0 + 2.0 * nvfp4_scale_bytes(n2, k) as f64;
        let gbs = bytes / us / 1e3;
        eprintln!("Q38-LMHEAD-FLOOR w4a8_dual_halves gemv_nvfp4_w4a8_dual_m1 n_half={n2} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- solo-runs every kernel in the q38 decode gdn_chain and dense_mlp buckets at exact serving shapes, reporting us and effective GB/s so the bucket totals can be attributed kernel by kernel"]
fn q38_decode_bucket_floor_solo_kernel_rates() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run the bucket floor bench");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x9e3779b97f4a7c15);

    const L2_COLD_WEIGHT_ROTATION: usize = 8;

    for (name, n, k) in [
        ("gdn_in_qkv", Q38_CONV_DIM, Q38_HIDDEN),
        ("gdn_in_z", Q38_VALUE_DIM, Q38_HIDDEN),
        ("gdn_out", Q38_HIDDEN, Q38_VALUE_DIM),
    ] {
        let wq = rng.bytes(n * k);
        let row_scale: Vec<f32> = (0..n).map(|_| 0.01 + rng.next_f32().abs() * 0.01).collect();
        let dws: Vec<CudaSlice<u8>> = (0..L2_COLD_WEIGHT_ROTATION)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
                d
            })
            .collect();
        #[allow(deprecated)]
        let ds: CudaSlice<f32> = stream.clone_htod(&row_scale).unwrap();
        let x = rng.bf16_words(k, 1.0);
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        let mut it = 0usize;
        let us = bench_us(&stream, 400, || {
            let (pw, _a) = dws[it % L2_COLD_WEIGHT_ROTATION].device_ptr(&stream);
            it += 1;
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
                    1,
                )
            };
            assert_eq!(rc, 0, "gemv_e4m3_mk rc={rc} ({name})");
        });
        let gbs = (n * k) as f64 / us / 1e3;
        eprintln!("Q38-FLOOR-COLD {name} gemv_e4m3_mk n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }

    for (name, n, k) in [("gdn_in_a", Q38_GDN_NV, Q38_HIDDEN), ("gdn_in_b", Q38_GDN_NV, Q38_HIDDEN)] {
        let w = rng.bf16_words(n * k, 0.05);
        #[allow(deprecated)]
        let dw: CudaSlice<u16> = stream.clone_htod(&w).unwrap();
        let x = rng.bf16_words(k, 1.0);
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        let us = bench_us(&stream, 400, || {
            let (pw, _a) = dw.device_ptr(&stream);
            let (px, _c) = dx.device_ptr(&stream);
            let (py, _d) = dy.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_bf16(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u16,
                    px as *const u16,
                    py as *mut u16,
                    n as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_bf16 rc={rc} ({name})");
        });
        let gbs = (2 * n * k) as f64 / us / 1e3;
        eprintln!("Q38-FLOOR {name} gemv_bf16 n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }

    {
        let qkv = rng.bf16_words(Q38_CONV_DIM, 1.0);
        let conv_state = rng.bf16_words(Q38_CONV_DIM * (Q38_CONV_K - 1), 1.0);
        let w = rng.bf16_words(Q38_CONV_DIM * Q38_CONV_K, 0.2);
        #[allow(deprecated)]
        let dq: CudaSlice<u16> = stream.clone_htod(&qkv).unwrap();
        #[allow(deprecated)]
        let dcs: CudaSlice<u16> = stream.clone_htod(&conv_state).unwrap();
        #[allow(deprecated)]
        let dwv: CudaSlice<u16> = stream.clone_htod(&w).unwrap();
        let mut dmix: CudaSlice<u16> = stream.alloc_zeros::<u16>(Q38_CONV_DIM).unwrap();
        let us = bench_us(&stream, 400, || {
            let (pq, _a) = dq.device_ptr(&stream);
            let (pcs, _b) = dcs.device_ptr(&stream);
            let (pw, _c) = dwv.device_ptr(&stream);
            let (pm, _d) = dmix.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_conv_decode_silu_bf16(
                    stream.cu_stream() as *mut c_void,
                    pq as *const u16,
                    pcs as *mut u16,
                    pw as *const u16,
                    pm as *mut u16,
                    Q38_CONV_DIM as i32,
                    Q38_CONV_K as i32,
                )
            };
            assert_eq!(rc, 0, "gdn_conv_decode_silu_bf16 rc={rc}");
        });
        eprintln!("Q38-FLOOR gdn_conv gdn_conv_decode_silu_bf16 conv_dim={Q38_CONV_DIM} k={Q38_CONV_K} us={us:.2}");
    }

    {
        let mixed = rng.bf16_words(Q38_CONV_DIM, 0.5);
        let z = rng.bf16_words(Q38_VALUE_DIM, 0.5);
        let a = rng.bf16_words(Q38_GDN_NV, 0.5);
        let b = rng.bf16_words(Q38_GDN_NV, 0.5);
        let a_log = rng.bf16_words(Q38_GDN_NV, 0.5);
        let dt = rng.bf16_words(Q38_GDN_NV, 0.5);
        let nw = rng.bf16_words(Q38_GDN_DV, 0.5);
        #[allow(deprecated)]
        let dm: CudaSlice<u16> = stream.clone_htod(&mixed).unwrap();
        #[allow(deprecated)]
        let dz: CudaSlice<u16> = stream.clone_htod(&z).unwrap();
        #[allow(deprecated)]
        let da: CudaSlice<u16> = stream.clone_htod(&a).unwrap();
        #[allow(deprecated)]
        let db: CudaSlice<u16> = stream.clone_htod(&b).unwrap();
        #[allow(deprecated)]
        let dal: CudaSlice<u16> = stream.clone_htod(&a_log).unwrap();
        #[allow(deprecated)]
        let ddt: CudaSlice<u16> = stream.clone_htod(&dt).unwrap();
        #[allow(deprecated)]
        let dnw: CudaSlice<u16> = stream.clone_htod(&nw).unwrap();
        let mut dstates: Vec<CudaSlice<f32>> = (0..4)
            .map(|_| {
                stream
                    .alloc_zeros::<f32>(Q38_GDN_NV * Q38_GDN_DK * Q38_GDN_DV)
                    .unwrap()
            })
            .collect();
        let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(Q38_VALUE_DIM).unwrap();
        let mut dqn: CudaSlice<f32> = stream.alloc_zeros::<f32>(Q38_GDN_NK * Q38_GDN_DK).unwrap();
        let mut dkn: CudaSlice<f32> = stream.alloc_zeros::<f32>(Q38_GDN_NK * Q38_GDN_DK).unwrap();
        let mut dge: CudaSlice<f32> = stream.alloc_zeros::<f32>(Q38_GDN_NV).unwrap();
        let mut dbe: CudaSlice<f32> = stream.alloc_zeros::<f32>(Q38_GDN_NV).unwrap();
        let mut dco: CudaSlice<u16> = stream
            .alloc_zeros::<u16>(Q38_GDN_NV * Q38_GDN_DV)
            .unwrap();
        let mut it = 0usize;
        let us = bench_us(&stream, 400, || {
            let (pm, _a) = dm.device_ptr(&stream);
            let (pz, _b) = dz.device_ptr(&stream);
            let (pa, _c) = da.device_ptr(&stream);
            let (pb, _d) = db.device_ptr(&stream);
            let (pal, _e) = dal.device_ptr(&stream);
            let (pdt, _f) = ddt.device_ptr(&stream);
            let (pnw, _g) = dnw.device_ptr(&stream);
            let (pst, _h) = dstates[it % 4].device_ptr_mut(&stream);
            it += 1;
            let (po, _i) = dout.device_ptr_mut(&stream);
            let (pqn, _j) = dqn.device_ptr_mut(&stream);
            let (pkn, _k) = dkn.device_ptr_mut(&stream);
            let (pge, _l) = dge.device_ptr_mut(&stream);
            let (pbe, _m) = dbe.device_ptr_mut(&stream);
            let (pco, _n) = dco.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_decode_step_split_bf16(
                    stream.cu_stream() as *mut c_void,
                    pm as *const u16,
                    pz as *const u16,
                    pa as *const u16,
                    pb as *const u16,
                    pal as *const u16,
                    pdt as *const u16,
                    pnw as *const u16,
                    pst as *mut f32,
                    po as *mut u16,
                    pqn as *mut f32,
                    pkn as *mut f32,
                    pge as *mut f32,
                    pbe as *mut f32,
                    pco as *mut u16,
                    Q38_GDN_NK as i32,
                    Q38_GDN_NV as i32,
                    Q38_GDN_DK as i32,
                    Q38_GDN_DV as i32,
                    1e-6,
                )
            };
            assert_eq!(rc, 0, "gdn_decode_step_split_bf16 rc={rc}");
        });
        let state_bytes = 2 * Q38_GDN_NV * Q38_GDN_DK * Q38_GDN_DV * 4;
        let gbs = state_bytes as f64 / us / 1e3;
        eprintln!("Q38-FLOOR-COLD gdn_step gdn_decode_step_split_bf16 nv={Q38_GDN_NV} dk={Q38_GDN_DK} dv={Q38_GDN_DV} us={us:.2} state_rw_gbs={gbs:.0}");
    }

    {
        let n = Q38_INTER;
        let k = Q38_HIDDEN;
        let wq_a = rng.bytes(n * k / 2);
        let wq_b = rng.bytes(n * k / 2);
        let sc_a = rng.bytes(nvfp4_scale_bytes(n, k));
        let sc_b = rng.bytes(nvfp4_scale_bytes(n, k));
        let xq: Vec<u8> = rng.bytes(k);
        let dwas: Vec<CudaSlice<u8>> = (0..4)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq_a).unwrap();
                d
            })
            .collect();
        let dwbs: Vec<CudaSlice<u8>> = (0..4)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq_b).unwrap();
                d
            })
            .collect();
        #[allow(deprecated)]
        let dsa: CudaSlice<u8> = stream.clone_htod(&sc_a).unwrap();
        #[allow(deprecated)]
        let dsb: CudaSlice<u8> = stream.clone_htod(&sc_b).unwrap();
        #[allow(deprecated)]
        let dxq: CudaSlice<u8> = stream.clone_htod(&xq).unwrap();
        #[allow(deprecated)]
        let dxs: CudaSlice<f32> = stream.clone_htod(&[0.05f32]).unwrap();
        let mut dya: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        let mut dyb: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        let mut it = 0usize;
        let us = bench_us(&stream, 400, || {
            let (pwa, _a) = dwas[it % 4].device_ptr(&stream);
            let (pwb, _c) = dwbs[it % 4].device_ptr(&stream);
            it += 1;
            let (psa, _b) = dsa.device_ptr(&stream);
            let (psb, _d) = dsb.device_ptr(&stream);
            let (pxq, _e) = dxq.device_ptr(&stream);
            let (pxs, _f) = dxs.device_ptr(&stream);
            let (pya, _g) = dya.device_ptr_mut(&stream);
            let (pyb, _h) = dyb.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_dual_m1(
                    stream.cu_stream() as *mut c_void,
                    pwa as *const u8,
                    psa as *const u8,
                    pwb as *const u8,
                    psb as *const u8,
                    pxq as *const i8,
                    pxs as *const f32,
                    pya as *mut u16,
                    pyb as *mut u16,
                    1.0,
                    1.0,
                    n as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_nvfp4_w4a8_dual_m1 rc={rc}");
        });
        let bytes = (n * k) as f64 + 2.0 * nvfp4_scale_bytes(n, k) as f64;
        let gbs = bytes / us / 1e3;
        eprintln!("Q38-FLOOR-COLD mlp_dual gemv_nvfp4_w4a8_dual_m1 n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }

    {
        let n = Q38_HIDDEN;
        let k = Q38_INTER;
        let wq = rng.bytes(n * k / 2);
        let sc = rng.bytes(nvfp4_scale_bytes(n, k));
        let actq: Vec<u8> = rng.bytes(k);
        let res = rng.bf16_words(n, 0.5);
        let dwd: Vec<CudaSlice<u8>> = (0..8)
            .map(|_| {
                #[allow(deprecated)]
                let d: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
                d
            })
            .collect();
        #[allow(deprecated)]
        let dsc: CudaSlice<u8> = stream.clone_htod(&sc).unwrap();
        #[allow(deprecated)]
        let daq: CudaSlice<u8> = stream.clone_htod(&actq).unwrap();
        #[allow(deprecated)]
        let das: CudaSlice<f32> = stream.clone_htod(&[0.05f32]).unwrap();
        #[allow(deprecated)]
        let dres: CudaSlice<u16> = stream.clone_htod(&res).unwrap();
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        let mut it = 0usize;
        let us = bench_us(&stream, 400, || {
            let (pw, _a) = dwd[it % 8].device_ptr(&stream);
            it += 1;
            let (psc, _b) = dsc.device_ptr(&stream);
            let (paq, _c) = daq.device_ptr(&stream);
            let (pas, _d) = das.device_ptr(&stream);
            let (pr, _e) = dres.device_ptr(&stream);
            let (py, _f) = dy.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_down_residual_m1(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u8,
                    psc as *const u8,
                    paq as *const i8,
                    pas as *const f32,
                    pr as *const u16,
                    py as *mut u16,
                    1.0,
                    n as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_m1 rc={rc}");
        });
        let bytes = (n * k) as f64 / 2.0 + nvfp4_scale_bytes(n, k) as f64;
        let gbs = bytes / us / 1e3;
        eprintln!("Q38-FLOOR-COLD mlp_down gemv_nvfp4_w4a8_down_residual_m1 n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
    }

    {
        let gate = rng.bf16_words(Q38_INTER, 0.5);
        let up = rng.bf16_words(Q38_INTER, 0.5);
        #[allow(deprecated)]
        let dg: CudaSlice<u16> = stream.clone_htod(&gate).unwrap();
        #[allow(deprecated)]
        let du: CudaSlice<u16> = stream.clone_htod(&up).unwrap();
        let plen = cuda::silu_mul_rowquant_i8_mk_partials_len(1, Q38_INTER as i32);
        assert!(plen > 0);
        let mut dst: CudaSlice<u16> = stream.alloc_zeros::<u16>(Q38_INTER).unwrap();
        let mut dpp: CudaSlice<f32> = stream.alloc_zeros::<f32>(plen as usize).unwrap();
        let us = bench_us(&stream, 400, || {
            let (pg, _a) = dg.device_ptr(&stream);
            let (pu, _b) = du.device_ptr(&stream);
            let (pst, _c) = dst.device_ptr_mut(&stream);
            let (ppp, _d) = dpp.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::silu_mul_stage_partial_absmax_m1(
                    stream.cu_stream() as *mut c_void,
                    pg as *const u16,
                    pu as *const u16,
                    pst as *mut u16,
                    ppp as *mut f32,
                    Q38_INTER as i32,
                )
            };
            assert_eq!(rc, 0, "silu_mul_stage_partial_absmax_m1 rc={rc}");
        });
        eprintln!("Q38-FLOOR mlp_siluq silu_mul_stage_partial_absmax_m1 k={Q38_INTER} us={us:.2}");
    }

    {
        let n = 256usize * 1024 * 1024;
        let src: CudaSlice<u8> = stream.alloc_zeros::<u8>(n).unwrap();
        let mut dst: CudaSlice<u8> = stream.alloc_zeros::<u8>(n).unwrap();
        let us = bench_us(&stream, 20, || {
            stream.memcpy_dtod(&src, &mut dst).unwrap();
        });
        let gbs = (2 * n) as f64 / us / 1e3;
        eprintln!("Q38-FLOOR-CAL dtod_copy bytes_each_way={n} us={us:.1} rw_gbs={gbs:.0}");
    }

    {
        let x = rng.bf16_words(Q38_HIDDEN, 1.0);
        let w = rng.bf16_words(Q38_HIDDEN, 1.0);
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
        #[allow(deprecated)]
        let dw: CudaSlice<u16> = stream.clone_htod(&w).unwrap();
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(Q38_HIDDEN).unwrap();
        let us = bench_us(&stream, 400, || {
            let (px, _a) = dx.device_ptr(&stream);
            let (pw, _b) = dw.device_ptr(&stream);
            let (py, _c) = dy.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::rmsnorm_bf16(
                    stream.cu_stream() as *mut c_void,
                    px as *const u16,
                    pw as *const u16,
                    py as *mut u16,
                    1,
                    Q38_HIDDEN,
                    1e-6,
                )
            };
            assert_eq!(rc, 0, "rmsnorm_bf16 rc={rc}");
        });
        eprintln!("Q38-FLOOR pre_norm rmsnorm_bf16 hidden={Q38_HIDDEN} us={us:.2}");
    }
}
