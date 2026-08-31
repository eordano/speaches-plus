#![cfg(feature = "cuda")]

use cudarc::cublaslt::result as cublaslt;
use cudarc::cublaslt::sys;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::mem;

const DECODE_HOT_SHAPES_FROM_THE_LIVE_NV_DEBUG_FP8_LT_SHAPES_LOG: &[(u64, u64, u64)] = &[
    (4, 10240, 5120),
    (4, 6144, 5120),
    (4, 5120, 6144),
    (4, 12288, 5120),
    (4, 1024, 5120),
    (4, 17408, 5120),
    (4, 5120, 17408),
];

const HEURISTIC_CANDIDATES_REQUESTED: usize = 24;
const WEIGHT_RING_COPIES_DEFEAT_L2_RESIDENCY: usize = 4;
const REPS: usize = 64;
const WARMUP: usize = 8;

#[test]
#[ignore]
fn fp8_lt_algo_sweep_default_pick_vs_enumerated_candidates() {
    if std::env::var("NV_FP8_LT_ALGO_SWEEP_TEST").as_deref() != Ok("1") {
        eprintln!("[fp8-lt-sweep] NV_FP8_LT_ALGO_SWEEP_TEST!=1: skip");
        return;
    }
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let handle = cublaslt::create_handle().expect("cublasLt handle");

    let ws_bytes: usize = 32 * 1024 * 1024;
    let mut ws: CudaSlice<u8> = stream.alloc_zeros::<u8>(ws_bytes).unwrap();
    let scale_host = [1.0f32];
    #[allow(deprecated)]
    let a_scale: CudaSlice<f32> = stream.memcpy_stod(&scale_host).unwrap();
    #[allow(deprecated)]
    let b_scale: CudaSlice<f32> = stream.memcpy_stod(&scale_host).unwrap();

    for &(m, n, k) in DECODE_HOT_SHAPES_FROM_THE_LIVE_NV_DEBUG_FP8_LT_SHAPES_LOG {
        let mut host = vec![0u8; (n * k) as usize];
        for (i, v) in host.iter_mut().enumerate() {
            *v = ((i * 2654435761) >> 7) as u8 & 0x77;
        }
        let weights: Vec<CudaSlice<u8>> = (0..WEIGHT_RING_COPIES_DEFEAT_L2_RESIDENCY)
            .map(|_| {
                #[allow(deprecated)]
                stream.memcpy_stod(&host).unwrap()
            })
            .collect();
        let mut xh = vec![0u8; (m * k) as usize];
        for (i, v) in xh.iter_mut().enumerate() {
            *v = ((i * 40503) >> 3) as u8 & 0x77;
        }
        #[allow(deprecated)]
        let x: CudaSlice<u8> = stream.memcpy_stod(&xh).unwrap();
        let mut d: CudaSlice<bf16> = stream.alloc_zeros::<bf16>((m * n) as usize).unwrap();

        let dtype_fp8 = sys::cudaDataType_t::CUDA_R_8F_E4M3;
        let dtype_bf16 = sys::cudaDataType_t::CUDA_R_16BF;
        let compute = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
        let scale_type = sys::cudaDataType_t::CUDA_R_32F;
        let a_layout = cublaslt::create_matrix_layout(dtype_fp8, k, n, k as i64).unwrap();
        let b_layout = cublaslt::create_matrix_layout(dtype_fp8, k, m, k as i64).unwrap();
        let d_layout = cublaslt::create_matrix_layout(dtype_bf16, n, m, n as i64).unwrap();
        let desc = cublaslt::create_matmul_desc(compute, scale_type).unwrap();
        let transa = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
        let transb = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
        unsafe {
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &transa as *const _ as *const c_void,
                mem::size_of_val(&transa),
            )
            .unwrap();
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &transb as *const _ as *const c_void,
                mem::size_of_val(&transb),
            )
            .unwrap();
            let (wsc, _g1) = b_scale.device_ptr(&stream);
            let (xsc, _g2) = a_scale.device_ptr(&stream);
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &wsc as *const _ as *const c_void,
                mem::size_of::<u64>(),
            )
            .unwrap();
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &xsc as *const _ as *const c_void,
                mem::size_of::<u64>(),
            )
            .unwrap();
        }
        let pref = cublaslt::create_matmul_pref().unwrap();
        unsafe {
            cublaslt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const _ as *const c_void,
                mem::size_of::<usize>(),
            )
            .unwrap();
        }

        let mut heur: [sys::cublasLtMatmulHeuristicResult_t; HEURISTIC_CANDIDATES_REQUESTED] =
            unsafe { mem::zeroed() };
        let mut count: i32 = 0;
        let rc = unsafe {
            sys::cublasLtMatmulAlgoGetHeuristic(
                handle,
                desc,
                a_layout,
                b_layout,
                d_layout,
                d_layout,
                pref,
                HEURISTIC_CANDIDATES_REQUESTED as i32,
                heur.as_mut_ptr(),
                &mut count,
            )
        };
        assert_eq!(
            rc,
            sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "heuristic query failed for {m}x{n}x{k}"
        );
        assert!(count > 0, "no fp8 algos for {m}x{n}x{k}");

        let weight_bytes = (n * k) as f64;
        let mut best = (f64::INFINITY, -1i32);
        let mut default_ms = f64::NAN;
        for i in 0..count as usize {
            if heur[i].state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                continue;
            }
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let mut run = |ring: usize| {
                let (wp, _gw) = weights[ring].device_ptr(&stream);
                let (xp, _gx) = x.device_ptr(&stream);
                let (dp, _gd) = d.device_ptr_mut(&stream);
                let (wsp, _gws) = ws.device_ptr_mut(&stream);
                let rc = unsafe {
                    cublaslt::matmul(
                        handle,
                        desc,
                        &alpha as *const _ as *const c_void,
                        &beta as *const _ as *const c_void,
                        wp as *const c_void,
                        a_layout,
                        xp as *const c_void,
                        b_layout,
                        dp as *const c_void,
                        d_layout,
                        dp as *mut c_void,
                        d_layout,
                        &heur[i].algo as *const _,
                        wsp as *mut c_void,
                        ws_bytes,
                        stream.cu_stream() as sys::cudaStream_t,
                    )
                };
                rc.is_ok()
            };
            let mut ok = true;
            for w in 0..WARMUP {
                ok &= run(w % WEIGHT_RING_COPIES_DEFEAT_L2_RESIDENCY);
            }
            if !ok {
                eprintln!("[fp8-lt-sweep] shape={m}x{n}x{k} algo_idx={i} launch refused");
                continue;
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for r in 0..REPS {
                run(r % WEIGHT_RING_COPIES_DEFEAT_L2_RESIDENCY);
            }
            stream.synchronize().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
            let gbs = weight_bytes / (ms * 1e-3) / 1e9;
            let ws_need = heur[i].workspaceSize;
            eprintln!(
                "[fp8-lt-sweep] shape={m}x{n}x{k} algo_idx={i} ms={ms:.4} gbs={gbs:.0} \
                 ws={ws_need} waves={:.2}{}",
                heur[i].wavesCount,
                if i == 0 { " DEFAULT" } else { "" }
            );
            if i == 0 {
                default_ms = ms;
            }
            if ms < best.0 {
                best = (ms, i as i32);
            }
        }
        eprintln!(
            "[fp8-lt-sweep] shape={m}x{n}x{k} VERDICT default_ms={default_ms:.4} \
             best_ms={:.4} best_idx={} speedup_if_switched={:.3}x",
            best.0,
            best.1,
            default_ms / best.0
        );

        unsafe {
            cublaslt::destroy_matmul_pref(pref).unwrap();
            cublaslt::destroy_matmul_desc(desc).unwrap();
            cublaslt::destroy_matrix_layout(a_layout).unwrap();
            cublaslt::destroy_matrix_layout(b_layout).unwrap();
            cublaslt::destroy_matrix_layout(d_layout).unwrap();
        }
    }
}
