#![cfg(feature = "cuda")]

mod common;
use common::xorshift;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

const HEAD_DIM: usize = 256;
const N_KV: usize = 4;
const N_Q: usize = 24;
const BLOCK_SIZE: usize = 16;

struct Fixture {
    stream: Arc<CudaStream>,
    raw: *mut c_void,
    d_q: CudaSlice<u16>,
    d_k: CudaSlice<u8>,
    d_v: CudaSlice<u8>,
    d_ks: CudaSlice<f32>,
    d_vs: CudaSlice<f32>,
    d_ntot: CudaSlice<i32>,
    d_table: CudaSlice<i32>,
    scaling: f32,
}

fn fixture(total: usize) -> Fixture {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;

    let mut state = 0x2545_f491u32;
    let elems = total * N_KV * HEAD_DIM;
    let src: Vec<u16> = (0..elems)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let q: Vec<u16> = (0..N_Q * HEAD_DIM)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();

    let blocks = total.div_ceil(BLOCK_SIZE);
    let table: Vec<i32> = (0..blocks as i32).collect();
    let slots = blocks * BLOCK_SIZE;

    #[allow(deprecated)]
    let d_table: CudaSlice<i32> = stream.clone_htod(&table).unwrap();
    #[allow(deprecated)]
    let d_start: CudaSlice<i32> = stream.clone_htod(&vec![0i32]).unwrap();
    #[allow(deprecated)]
    let d_ntot: CudaSlice<i32> = stream.clone_htod(&vec![total as i32]).unwrap();
    #[allow(deprecated)]
    let d_q: CudaSlice<u16> = stream.clone_htod(&q).unwrap();
    #[allow(deprecated)]
    let d_src: CudaSlice<u16> = stream.clone_htod(&src).unwrap();

    let mut quantize = || -> (CudaSlice<u8>, CudaSlice<f32>) {
        let mut d_fp8: CudaSlice<u8> = stream.alloc_zeros::<u8>(slots * N_KV * HEAD_DIM).unwrap();
        let mut d_sc: CudaSlice<f32> = stream.alloc_zeros::<f32>(slots * N_KV).unwrap();
        let rc = {
            let (p_src, _a) = d_src.device_ptr(&stream);
            let (p_st, _b) = d_start.device_ptr(&stream);
            let (p_tb, _c) = d_table.device_ptr(&stream);
            let (p_fp8, _d) = d_fp8.device_ptr_mut(&stream);
            let (p_sc, _e) = d_sc.device_ptr_mut(&stream);
            unsafe {
                cuda::quantize_kv_fp8_paged(
                    raw,
                    p_src as *const u16,
                    p_fp8 as *mut u8,
                    p_sc as *mut f32,
                    p_st as *const i32,
                    p_tb as *const i32,
                    BLOCK_SIZE as i32,
                    total as i32,
                    N_KV as i32,
                    HEAD_DIM as i32,
                )
            }
        };
        assert_eq!(rc, 0, "quantize rc={rc}");
        stream.synchronize().unwrap();
        (d_fp8, d_sc)
    };
    let (d_k, d_ks) = quantize();
    let (d_v, d_vs) = quantize();

    Fixture {
        stream,
        raw,
        d_q,
        d_k,
        d_v,
        d_ks,
        d_vs,
        d_ntot,
        d_table,
        scaling: 1.0f32 / (HEAD_DIM as f32).sqrt(),
    }
}

fn run_fused(f: &Fixture) -> Vec<u16> {
    let stream = &f.stream;
    let scratch_elems = cuda::flash_splitk_scratch_elems(N_Q as i32, HEAD_DIM as i32);
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(N_Q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(N_Q * HEAD_DIM).unwrap();
    let rc = {
        let (p_q, _a) = f.d_q.device_ptr(stream);
        let (p_k, _b) = f.d_k.device_ptr(stream);
        let (p_v, _c) = f.d_v.device_ptr(stream);
        let (p_ks, _d) = f.d_ks.device_ptr(stream);
        let (p_vs, _e) = f.d_vs.device_ptr(stream);
        let (p_nt, _f) = f.d_ntot.device_ptr(stream);
        let (p_tb, _g) = f.d_table.device_ptr(stream);
        let (p_out, _h) = d_out.device_ptr_mut(stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(stream);
        unsafe {
            cuda::flash_decode_fused_fp8kv_paged(
                f.raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                N_Q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                f.scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    assert_eq!(rc, 0, "fused launch rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.clone_dtoh(&d_out).unwrap()
}

fn run_kvshare(f: &Fixture, splits: usize, time_reps: usize) -> (Vec<u16>, f64) {
    let stream = &f.stream;
    let mut d_scratch: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(N_Q * splits.max(128) * (HEAD_DIM + 2))
        .unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(N_Q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(N_Q * HEAD_DIM).unwrap();
    let go = |d_out: &mut CudaSlice<u16>,
              d_scratch: &mut CudaSlice<f32>,
              d_fan: &mut CudaSlice<u32>|
     -> i32 {
        let (p_q, _a) = f.d_q.device_ptr(stream);
        let (p_k, _b) = f.d_k.device_ptr(stream);
        let (p_v, _c) = f.d_v.device_ptr(stream);
        let (p_ks, _d) = f.d_ks.device_ptr(stream);
        let (p_vs, _e) = f.d_vs.device_ptr(stream);
        let (p_nt, _f) = f.d_ntot.device_ptr(stream);
        let (p_tb, _g) = f.d_table.device_ptr(stream);
        let (p_out, _h) = d_out.device_ptr_mut(stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(stream);
        unsafe {
            cuda::flash_decode_kvshare_fp8kv_paged(
                f.raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                N_Q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                splits as i32,
                f.scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    let rc = go(&mut d_out, &mut d_scratch, &mut d_fan);
    assert_eq!(rc, 0, "kvshare launch rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out: Vec<u16> = stream.clone_dtoh(&d_out).unwrap();
    let mut us = 0.0f64;
    if time_reps > 0 {
        for _ in 0..5 {
            go(&mut d_out, &mut d_scratch, &mut d_fan);
        }
        stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..time_reps {
            go(&mut d_out, &mut d_scratch, &mut d_fan);
        }
        stream.synchronize().unwrap();
        us = t0.elapsed().as_secs_f64() * 1e6 / time_reps as f64;
    }
    (out, us)
}

fn max_abs_diff(a: &[u16], b: &[u16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (bf16::from_bits(*x).to_f32() - bf16::from_bits(*y).to_f32()).abs())
        .fold(0f32, f32::max)
}

fn run_bw_probe(f: &Fixture, total: usize, splits: usize, mode: i32, time_reps: usize) -> f64 {
    let stream = &f.stream;
    let mut d_sink: CudaSlice<f32> = stream.alloc_zeros::<f32>(N_KV).unwrap();
    let go = |d_sink: &mut CudaSlice<f32>| -> i32 {
        let (p_q, _q) = f.d_q.device_ptr(stream);
        let (p_k, _a) = f.d_k.device_ptr(stream);
        let (p_v, _b) = f.d_v.device_ptr(stream);
        let (p_ks, _d) = f.d_ks.device_ptr(stream);
        let (p_vs, _e) = f.d_vs.device_ptr(stream);
        let (p_tb, _g) = f.d_table.device_ptr(stream);
        let (p_s, _c) = d_sink.device_ptr_mut(stream);
        unsafe {
            cuda::kvshare_bw_probe(
                f.raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
                p_s as *mut f32,
                total as i32,
                N_KV as i32,
                splits as i32,
                mode,
            )
        }
    };
    let rc = go(&mut d_sink);
    assert_eq!(rc, 0, "bw probe mode {mode} rc={rc}");
    stream.synchronize().unwrap();
    for _ in 0..5 {
        go(&mut d_sink);
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..time_reps {
        go(&mut d_sink);
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / time_reps as f64
}

#[test]
fn kvshare_bw_probe_separates_dram_pattern_ceiling_from_kernel_internals() {
    let total = 196608usize;
    let f = fixture(total);
    for splits in [64usize, 94, 128] {
        for (mode, name) in [
            (0i32, "staged-pattern"),
            (2, "plain-pattern"),
            (1, "linear-stream"),
            (3, "plus-slots-scales"),
            (4, "plus-dot-shuffle"),
            (5, "full-arithmetic"),
        ] {
            let us = run_bw_probe(&f, total, splits, mode, 30);
            let unique = (N_KV * total * HEAD_DIM * 2) as f64;
            eprintln!(
                "[bwprobe] ctx {total} splits {splits:3} mode {mode} {name:15}: \
                 {us:9.2} us | unique {:7.1} GB/s",
                unique / us / 1e3
            );
        }
    }
}

const DIRECT_ARMS: [usize; 2] = [0, 1];

fn set_arm(direct: usize) {
    std::env::set_var("NV_KVSHARE_DIRECT", direct.to_string());
}

#[test]
fn kvshare_group6_hd256_matches_the_per_q_head_fused_kernel_bit_exactly_at_matched_splits() {
    for total in [1000usize, 8192, 32768] {
        let f = fixture(total);
        let want = run_fused(&f);
        for direct in DIRECT_ARMS {
            set_arm(direct);
            let (got, _) = run_kvshare(&f, 16, 0);
            let n_diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            assert_eq!(
                n_diff,
                0,
                "ctx {total} direct {direct}: kvshare splits=16 \
                 vs fused splits=16 (flash_splits_pick(24)=16) differ in \
                 {n_diff}/{} bf16 lanes, max|diff|={:.3e}; identical per-warp \
                 position order + identical FP expression trees must reproduce \
                 the fused kernel bit-for-bit under every load scheme, tile depth \
                 and occupancy bound",
                want.len(),
                max_abs_diff(&want, &got)
            );
        }
    }
    set_arm(1);
}

#[test]
fn kvshare_group6_hd256_split128_stays_within_bf16_rounding_of_the_fused_kernel() {
    for total in [8192usize, 32768, 196608] {
        let f = fixture(total);
        let want = run_fused(&f);
        for direct in DIRECT_ARMS {
            set_arm(direct);
            for splits in [64usize, 74, 94, 111, 128] {
                let (got, us) = run_kvshare(&f, splits, 30);
                let diff = max_abs_diff(&want, &got);
                let unique = (N_KV * total * HEAD_DIM * 2) as f64;
                eprintln!(
                    "[kvshare] ctx {total:6} direct {direct} \
                     splits {splits:3}: {us:9.2} us | unique {:7.1} GB/s | \
                     max|diff| vs fused {diff:.3e}",
                    unique / us / 1e3
                );
                assert!(
                    diff < 1.5e-2,
                    "ctx {total} direct {direct} splits \
                     {splits}: max|diff| {diff:.3e} vs fused exceeds bf16 \
                     reorder tolerance"
                );
            }
        }
    }
    set_arm(1);
}
