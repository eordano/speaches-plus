#![cfg(feature = "cuda")]

mod common;
use common::LcgOddSeedShift32F64TwoSided as Lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::sync::Arc;

const VARIANTS: [(i32, &str); 8] = [
    (0, "warps8  split1 ldcs"),
    (1, "warps8  split1 ldg "),
    (2, "warps4  split1 ldcs"),
    (3, "warps16 split1 ldcs"),
    (4, "warps8  split2 ldcs"),
    (5, "warps16 split2 ldcs"),
    (6, "warps8  split4 ldcs"),
    (7, "warps16 split4 ldcs"),
];

fn gen_inputs(n: usize, k: usize, seed: u64) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.next_u32()).collect();
    let scales: Vec<u16> = (0..n * k / 32)
        .map(|_| bf16::from_f32(0.005 + 0.01 * rng.next_f32().abs()).to_bits())
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    (packed, scales, x)
}

fn cpu_ref(packed: &[u32], scales: &[u16], x: &[u16], n: usize, k: usize) -> Vec<f32> {
    let xf: Vec<f32> = x.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    let sf: Vec<f32> = scales
        .iter()
        .map(|&b| bf16::from_bits(b).to_f32())
        .collect();
    let kw = k / 8;
    let kg = k / 32;
    let mut y = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f64;
        for kk in 0..k {
            let word = packed[row * kw + kk / 8];
            let q = ((word >> (4 * (kk % 8))) & 0xF) as i32 - 8;
            acc += (q as f32 * sf[row * kg + kk / 32] * xf[kk]) as f64;
        }
        y[row] = acc as f32;
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn launch_proto(
    stream: &Arc<CudaStream>,
    packed: &CudaSlice<u32>,
    scales: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    y: &mut CudaSlice<u16>,
    n: usize,
    k: usize,
    variant: i32,
) -> i32 {
    let (pp, _g1) = packed.device_ptr(stream);
    let (sp, _g2) = scales.device_ptr(stream);
    let (xp, _g3) = x.device_ptr(stream);
    let (yp, _g4) = y.device_ptr_mut(stream);
    unsafe {
        cuda::gemv_w4a16_m1_proto(
            stream.cu_stream() as *mut _,
            pp as *const u32,
            sp as *const u16,
            xp as *const u16,
            yp as *mut u16,
            n as i32,
            k as i32,
            32,
            variant,
        )
    }
}

#[test]
fn proto_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "gemv_w4a16_m1_proto: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("gemv_w4a16_m1_proto: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    for &(n, k) in &[(67usize, 320usize), (96, 2560), (33, 10240)] {
        let (packed, scales, x) = gen_inputs(n, k, 0x1234 + (n * k) as u64);
        let expect = cpu_ref(&packed, &scales, &x, n, k);

        #[allow(deprecated)]
        let dp: CudaSlice<u32> = stream.memcpy_stod(&packed).unwrap();
        #[allow(deprecated)]
        let ds: CudaSlice<u16> = stream.memcpy_stod(&scales).unwrap();
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.memcpy_stod(&x).unwrap();

        for &(variant, name) in &VARIANTS {
            let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
            let rc = launch_proto(&stream, &dp, &ds, &dx, &mut dy, n, k, variant);
            assert_eq!(rc, 0, "variant {variant} ({name}) n={n} k={k} rc={rc}");
            stream.synchronize().unwrap();
            #[allow(deprecated)]
            let got_u16 = stream.memcpy_dtov(&dy).unwrap();
            let mut max_rel = 0f32;
            for (i, (&g, &e)) in got_u16.iter().zip(expect.iter()).enumerate() {
                let gf = bf16::from_bits(g).to_f32();
                let rel = (gf - e).abs() / e.abs().max(0.5);
                if rel > max_rel {
                    max_rel = rel;
                }
                assert!(
                    rel <= 1e-2,
                    "variant {variant} ({name}) n={n} k={k} row {i}: got {gf} want {e} rel {rel}"
                );
            }
            eprintln!("variant {variant} ({name}) n={n} k={k}: max rel err {max_rel:.2e}");
        }
    }
}

fn time_launches<F: FnMut(usize)>(stream: &Arc<CudaStream>, iters: usize, mut f: F) -> f64 {
    for i in 0..20 {
        f(i);
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        f(i);
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64()
}

fn bench_shape(stream: &Arc<CudaStream>, name: &str, k: usize, n: usize) {
    const COPIES: usize = 16;
    const ITERS: usize = 200;
    let weight_bytes = (n * k / 2 + n * (k / 32) * 2) as f64;
    println!(
        "=== {name}: K={k} N={n}  weights+scales = {:.1} MB ===",
        weight_bytes / 1e6
    );

    let (packed, scales, x) = gen_inputs(n, k, 0xBEEF);

    let dpacked: Vec<CudaSlice<u32>> = (0..COPIES)
        .map(|_| {
            #[allow(deprecated)]
            stream.memcpy_stod(&packed).unwrap()
        })
        .collect();
    let dscales: Vec<CudaSlice<u16>> = (0..COPIES)
        .map(|_| {
            #[allow(deprecated)]
            stream.memcpy_stod(&scales).unwrap()
        })
        .collect();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.memcpy_stod(&x).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

    for &(variant, vname) in &VARIANTS {
        let rc = launch_proto(
            stream,
            &dpacked[0],
            &dscales[0],
            &dx,
            &mut dy,
            n,
            k,
            variant,
        );
        if rc != 0 {
            println!("  proto {vname}: unsupported (rc={rc})");
            continue;
        }
        let secs = time_launches(stream, ITERS, |i| {
            let c = i % COPIES;
            let rc = launch_proto(
                stream,
                &dpacked[c],
                &dscales[c],
                &dx,
                &mut dy,
                n,
                k,
                variant,
            );
            assert_eq!(rc, 0);
        });
        let tbps = weight_bytes * ITERS as f64 / secs / 1e12;
        println!(
            "  proto {vname}: {:8.2} us/launch  {tbps:5.2} TB/s",
            secs / ITERS as f64 * 1e6
        );
    }
    drop(dpacked);
    drop(dscales);

    let mut rng = Lcg::new(0xCAFE);
    let packed_std: Vec<i32> = (0..k / 8 * n).map(|_| rng.next_u32() as i32).collect();
    let scales_t: Vec<u16> = (0..k / 32 * n)
        .map(|_| bf16::from_f32(0.005 + 0.01 * rng.next_f32().abs()).to_bits())
        .collect();
    #[allow(deprecated)]
    let dstd: CudaSlice<i32> = stream.memcpy_stod(&packed_std).unwrap();
    let dmarlin: Vec<CudaSlice<i32>> = (0..COPIES)
        .map(|_| {
            let mut m: CudaSlice<i32> = stream.alloc_zeros::<i32>(k * n / 8).unwrap();
            let rc = {
                let (sp, _g1) = dstd.device_ptr(stream);
                let (mp, _g2) = m.device_ptr_mut(stream);
                unsafe {
                    cuda::marlin_repack_w4a16(
                        stream.cu_stream() as *mut _,
                        sp as *const std::ffi::c_void,
                        mp as *mut std::ffi::c_void,
                        k as i32,
                        n as i32,
                        4,
                    )
                }
            };
            assert_eq!(rc, 0, "marlin_repack rc={rc}");
            m
        })
        .collect();
    stream.synchronize().unwrap();
    let dscales_t: Vec<CudaSlice<u16>> = (0..COPIES)
        .map(|_| {
            #[allow(deprecated)]
            stream.memcpy_stod(&scales_t).unwrap()
        })
        .collect();

    let mut ws_elems: i32 = 0;
    unsafe { cuda::marlin_workspace_elems(&mut ws_elems as *mut i32) };
    assert!(ws_elems > 0);
    let mut workspace: CudaSlice<i32> = stream.alloc_zeros::<i32>(ws_elems as usize).unwrap();

    for m in [1usize, 4] {
        let xa: Vec<u16> = (0..m * k)
            .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
            .collect();
        #[allow(deprecated)]
        let dxa: CudaSlice<u16> = stream.memcpy_stod(&xa).unwrap();
        let mut dc: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();

        let secs = time_launches(stream, ITERS, |i| {
            let c = i % COPIES;
            let rc = {
                let (ap, _g1) = dxa.device_ptr(stream);
                let (bp, _g2) = dmarlin[c].device_ptr(stream);
                let (sp, _g3) = dscales_t[c].device_ptr(stream);
                let (cp, _g4) = dc.device_ptr_mut(stream);
                let (wp, _g5) = workspace.device_ptr_mut(stream);
                unsafe {
                    cuda::marlin_gemm_w4a16(
                        stream.cu_stream() as *mut _,
                        ap as *const std::ffi::c_void,
                        bp as *const std::ffi::c_void,
                        sp as *const std::ffi::c_void,
                        cp as *mut std::ffi::c_void,
                        wp as *mut std::ffi::c_void,
                        m as i32,
                        n as i32,
                        k as i32,
                        32,
                        1,
                    )
                }
            };
            assert_eq!(rc, 0, "marlin_gemm rc={rc}");
        });
        let tbps = weight_bytes * ITERS as f64 / secs / 1e12;
        println!(
            "  marlin M={m}: {:8.2} us/launch  {tbps:5.2} TB/s",
            secs / ITERS as f64 * 1e6
        );
    }
}

#[test]
#[ignore]
fn bench_proto_vs_marlin() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "gemv_w4a16_m1_proto: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("gemv_w4a16_m1_proto: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    bench_shape(&stream, "gate_up", 2560, 20480);
    bench_shape(&stream, "down", 10240, 2560);
}
