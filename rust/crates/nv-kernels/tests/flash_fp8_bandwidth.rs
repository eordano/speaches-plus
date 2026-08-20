#![cfg(feature = "cuda")]

mod common;
use common::xorshift;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::time::Instant;

const HEAD_DIM: usize = 512;
const N_KV: usize = 4;
const BLOCK_SIZE: usize = 16;

const ACHIEVABLE_GBPS: f64 = 1502.0;

fn bench_gqa(total: usize, n_q: usize, splits: usize) -> (f64, f64, f64, f32) {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;

    let mut state = 0x2545_f491u32;
    let elems = total * N_KV * HEAD_DIM;
    let src: Vec<u16> = (0..elems)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let q: Vec<u16> = (0..n_q * HEAD_DIM)
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

    let mut d_scratch: CudaSlice<f32> =
        stream.alloc_zeros::<f32>(n_q * splits * (HEAD_DIM + 2)).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(n_q.max(N_KV)).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(n_q * HEAD_DIM).unwrap();
    let scaling = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let go = |d_out: &mut CudaSlice<u16>,
              d_scratch: &mut CudaSlice<f32>,
              d_fan: &mut CudaSlice<u32>|
     -> i32 {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = d_k.device_ptr(&stream);
        let (p_v, _c) = d_v.device_ptr(&stream);
        let (p_ks, _d) = d_ks.device_ptr(&stream);
        let (p_vs, _e) = d_vs.device_ptr(&stream);
        let (p_nt, _f) = d_ntot.device_ptr(&stream);
        let (p_tb, _g) = d_table.device_ptr(&stream);
        let (p_out, _h) = d_out.device_ptr_mut(&stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(&stream);
        unsafe {
            cuda::flash_decode_gqa_fp8kv_paged(
                raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                n_q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                splits as i32,
                scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    let rc = go(&mut d_out, &mut d_scratch, &mut d_fan);
    assert_eq!(rc, 0, "gqa launch rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let got: Vec<f32> = stream
        .clone_dtoh(&d_out)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    for _ in 0..5 {
        go(&mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let reps = 30;
    let t0 = Instant::now();
    for _ in 0..reps {
        go(&mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let unique = (N_KV * total * HEAD_DIM * 2) as f64;
    let requested = unique;
    let want = reference_out(total, n_q);
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    (us, unique / us / 1e3, requested / us / 1e3, diff)
}

fn reference_out(total: usize, n_q: usize) -> Vec<f32> {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let mut state = 0x2545_f491u32;
    let elems = total * N_KV * HEAD_DIM;
    let src: Vec<u16> = (0..elems)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let q: Vec<u16> = (0..n_q * HEAD_DIM)
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
        assert_eq!(rc, 0);
        stream.synchronize().unwrap();
        (d_fp8, d_sc)
    };
    let (d_k, d_ks) = quantize();
    let (d_v, d_vs) = quantize();
    let scratch_elems = cuda::flash_splitk_scratch_elems(n_q as i32, HEAD_DIM as i32) as usize;
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(n_q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(n_q * HEAD_DIM).unwrap();
    let scaling = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let rc = {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = d_k.device_ptr(&stream);
        let (p_v, _c) = d_v.device_ptr(&stream);
        let (p_ks, _d) = d_ks.device_ptr(&stream);
        let (p_vs, _e) = d_vs.device_ptr(&stream);
        let (p_nt, _f) = d_ntot.device_ptr(&stream);
        let (p_tb, _g) = d_table.device_ptr(&stream);
        let (p_out, _h) = d_out.device_ptr_mut(&stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(&stream);
        unsafe {
            cuda::flash_decode_fused_fp8kv_paged(
                raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                n_q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    assert_eq!(rc, 0);
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream
        .clone_dtoh(&d_out)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect()
}

fn bench(total: usize, n_q: usize) -> (f64, f64, f64) {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;

    let mut state = 0x2545_f491u32;
    let elems = total * N_KV * HEAD_DIM;
    let src: Vec<u16> = (0..elems)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let q: Vec<u16> = (0..n_q * HEAD_DIM)
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

    let scratch_elems = cuda::flash_splitk_scratch_elems(n_q as i32, HEAD_DIM as i32) as usize;
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(n_q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(n_q * HEAD_DIM).unwrap();
    let scaling = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let go = |d_out: &mut CudaSlice<u16>,
              d_scratch: &mut CudaSlice<f32>,
              d_fan: &mut CudaSlice<u32>|
     -> i32 {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = d_k.device_ptr(&stream);
        let (p_v, _c) = d_v.device_ptr(&stream);
        let (p_ks, _d) = d_ks.device_ptr(&stream);
        let (p_vs, _e) = d_vs.device_ptr(&stream);
        let (p_nt, _f) = d_ntot.device_ptr(&stream);
        let (p_tb, _g) = d_table.device_ptr(&stream);
        let (p_out, _h) = d_out.device_ptr_mut(&stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(&stream);
        unsafe {
            cuda::flash_decode_fused_fp8kv_paged(
                raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                n_q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    assert_eq!(go(&mut d_out, &mut d_scratch, &mut d_fan), 0, "launch");
    for _ in 0..5 {
        go(&mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();

    let reps = 30;
    let t0 = Instant::now();
    for _ in 0..reps {
        go(&mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let unique = (N_KV * total * HEAD_DIM * 2) as f64;
    let requested = (n_q * total * HEAD_DIM * 2) as f64;
    (us, unique / us / 1e3, requested / us / 1e3)
}

#[test]
#[ignore]
fn how_much_of_the_memory_floor_does_fp8_decode_attention_reach() {
    if std::env::var("NV_FLASH_BW_BENCH").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_FLASH_BW_BENCH=1");
    }
    let splits = std::env::var("NV_E4B_FLASH_SPLITS").unwrap_or_else(|_| "auto".into());
    eprintln!("[flash-bw] splits={splits}");
    for total in [8192usize, 32768, 131072] {
        for n_q in [4usize, 8, 16, 32, 64, 128] {
            let (us, unique, requested) = bench(total, n_q);
            eprintln!(
                "[flash-bw] ctx {total:6} nq {n_q:3} group {:3}: {us:9.2} us | unique \
                 {unique:7.1} GB/s ({:5.1}% of DRAM floor) | requested {requested:7.1} GB/s",
                n_q / N_KV,
                unique / ACHIEVABLE_GBPS * 100.0
            );
        }
    }

    eprintln!("[flash-bw] one block per KV head, group 8 (n_q 32, n_kv 4):");
    for total in [8192usize, 32768, 131072] {
        for splits in [16usize, 32, 64, 128] {
            let (us, unique, requested, diff) = bench_gqa(total, 32, splits);
            eprintln!(
                "[flash-bw] ctx {total:6} splits {splits:3}: {us:9.2} us | unique \
                 {unique:7.1} GB/s ({:5.1}% of DRAM floor) | requested {requested:7.1} GB/s \
                 | max|diff| vs per-head kernel {diff:.3e}",
                unique / ACHIEVABLE_GBPS * 100.0
            );
        }
    }
}
