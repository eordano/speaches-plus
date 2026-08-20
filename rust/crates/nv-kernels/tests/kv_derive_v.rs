#![cfg(feature = "cuda")]

mod common;
use common::xorshift;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use common::rel_rms_f32 as rel_rms;

fn inv_freq() -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut inv = vec![0f32; half];
    for (j, f) in inv.iter_mut().enumerate().take(ROPE_ANGLES) {
        *f = (1.0f64 / (BASE as f64).powf((j as f64 * 2.0) / (HEAD_DIM as f64))) as f32;
    }
    inv
}

const HEAD_DIM: usize = 512;
const N_KV: usize = 4;
const LEN: usize = 256;
const BLOCK_SIZE: usize = 16;
const ROPE_ANGLES: usize = 64;
const BASE: f32 = 1_000_000.0;
const W_K: f32 = 0.0615;

fn tables(pos_base: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 2;
    let mut cos = vec![1.0f32; LEN * half];
    let mut sin = vec![0.0f32; LEN * half];
    for p in 0..LEN {
        for j in 0..ROPE_ANGLES {
            let inv = 1.0f64 / (BASE as f64).powf((j as f64 * 2.0) / (HEAD_DIM as f64));
            let th = (pos_base + p) as f64 * inv;
            cos[p * half + j] = th.cos() as f32;
            sin[p * half + j] = th.sin() as f32;
        }
    }
    (cos, sin)
}

fn build_kv(cos: &[f32], sin: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 2;
    let mut state = 0x1234_5678u32;
    let mut k_rot = vec![0f32; LEN * N_KV * HEAD_DIM];
    let mut v_exact = vec![0f32; LEN * N_KV * HEAD_DIM];
    for t in 0..LEN {
        for h in 0..N_KV {
            let raw: Vec<f32> = (0..HEAD_DIM).map(|_| xorshift(&mut state) * 3.0).collect();
            let ms: f32 = raw.iter().map(|x| x * x).sum::<f32>() / HEAD_DIM as f32;
            let inv_rms = 1.0 / (ms + 1e-6).sqrt();
            let base = (t * N_KV + h) * HEAD_DIM;
            let normed: Vec<f32> = raw.iter().map(|x| x * inv_rms).collect();
            for d in 0..HEAD_DIM {
                v_exact[base + d] = normed[d];
            }
            for j in 0..half {
                let (c, s) = (cos[t * half + j], sin[t * half + j]);
                let (lo, hi) = (normed[j] * W_K, normed[j + half] * W_K);
                k_rot[base + j] = lo * c - hi * s;
                k_rot[base + j + half] = lo * s + hi * c;
            }
        }
    }
    (k_rot, v_exact)
}

const ANGLE_TABLE: i32 = 0;
const ANGLE_F32: i32 = 1;
const ANGLE_F64: i32 = 2;

struct Run {
    stored: f32,
    table: f32,
    f32_mode: f32,
    f64_mode: f32,
}

fn run_at(pos_base: usize) -> Run {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw_stream = stream.cu_stream() as *mut c_void;

    let (cos, sin) = tables(pos_base);
    let inv = inv_freq();
    let (k_rot, v_exact) = build_kv(&cos, &sin);

    let blocks = LEN.div_ceil(BLOCK_SIZE);
    let table: Vec<i32> = (0..blocks as i32).collect();
    let slots = blocks * BLOCK_SIZE;
    let to_bf16 =
        |v: &[f32]| -> Vec<u16> { v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect() };

    #[allow(deprecated)]
    let d_table: CudaSlice<i32> = stream.clone_htod(&table).unwrap();
    #[allow(deprecated)]
    let d_start: CudaSlice<i32> = stream.clone_htod(&vec![0i32]).unwrap();
    #[allow(deprecated)]
    let d_cos: CudaSlice<f32> = stream.clone_htod(&cos).unwrap();
    #[allow(deprecated)]
    let d_sin: CudaSlice<f32> = stream.clone_htod(&sin).unwrap();
    #[allow(deprecated)]
    let d_inv: CudaSlice<f32> = stream.clone_htod(&inv).unwrap();

    let mut quantized = |src: &[f32]| -> (CudaSlice<u8>, CudaSlice<f32>) {
        #[allow(deprecated)]
        let d_src: CudaSlice<u16> = stream.clone_htod(&to_bf16(src)).unwrap();
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
                    raw_stream,
                    p_src as *const u16,
                    p_fp8 as *mut u8,
                    p_sc as *mut f32,
                    p_st as *const i32,
                    p_tb as *const i32,
                    BLOCK_SIZE as i32,
                    LEN as i32,
                    N_KV as i32,
                    HEAD_DIM as i32,
                )
            }
        };
        assert_eq!(rc, 0, "quantize_kv_fp8_paged rc={rc}");
        stream.synchronize().unwrap();
        (d_fp8, d_sc)
    };

    let (d_k_fp8, d_k_sc) = quantized(&k_rot);
    let (d_v_fp8, d_v_sc) = quantized(&v_exact);

    let mut d_v_stored: CudaSlice<u16> = stream.alloc_zeros::<u16>(LEN * N_KV * HEAD_DIM).unwrap();
    let rc = {
        let (p_fp8, _a) = d_v_fp8.device_ptr(&stream);
        let (p_sc, _b) = d_v_sc.device_ptr(&stream);
        let (p_tb, _c) = d_table.device_ptr(&stream);
        let (p_out, _d) = d_v_stored.device_ptr_mut(&stream);
        unsafe {
            cuda::dequantize_kv_fp8_paged(
                raw_stream,
                p_fp8 as *const u8,
                p_sc as *const f32,
                p_out as *mut u16,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
                LEN as i32,
                N_KV as i32,
                HEAD_DIM as i32,
            )
        }
    };
    assert_eq!(rc, 0, "dequantize_kv_fp8_paged rc={rc}");

    let mut derive = |mode: i32| -> Vec<f32> {
        let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(LEN * N_KV * HEAD_DIM).unwrap();
        let rc = {
            let (p_k, _a) = d_k_fp8.device_ptr(&stream);
            let (p_ks, _b) = d_k_sc.device_ptr(&stream);
            let (p_cos, _c) = d_cos.device_ptr(&stream);
            let (p_sin, _d) = d_sin.device_ptr(&stream);
            let (p_inv, _e) = d_inv.device_ptr(&stream);
            let (p_tb, _f) = d_table.device_ptr(&stream);
            let (p_out, _g) = d_out.device_ptr_mut(&stream);
            unsafe {
                cuda::derive_v_from_k_fp8_paged(
                    raw_stream,
                    p_k as *const u8,
                    p_ks as *const f32,
                    p_cos as *const f32,
                    p_sin as *const f32,
                    p_inv as *const f32,
                    p_out as *mut u16,
                    p_tb as *const i32,
                    BLOCK_SIZE as i32,
                    LEN as i32,
                    N_KV as i32,
                    HEAD_DIM as i32,
                    ROPE_ANGLES as i32,
                    mode,
                    pos_base as i32,
                    1.0 / W_K,
                )
            }
        };
        assert_eq!(rc, 0, "derive_v_from_k_fp8_paged mode={mode} rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        stream
            .clone_dtoh(&d_out)
            .unwrap()
            .iter()
            .map(|b| bf16::from_bits(*b).to_f32())
            .collect()
    };

    let t = derive(ANGLE_TABLE);
    let a32 = derive(ANGLE_F32);
    let a64 = derive(ANGLE_F64);

    #[allow(deprecated)]
    let stored: Vec<f32> = stream
        .clone_dtoh(&d_v_stored)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    Run {
        stored: rel_rms(&stored, &v_exact),
        table: rel_rms(&t, &v_exact),
        f32_mode: rel_rms(&a32, &v_exact),
        f64_mode: rel_rms(&a64, &v_exact),
    }
}

fn report(what: &str, r: &Run) {
    eprintln!(
        "[derive-v] {what}: stored {:e} | table {:e} ({:.3}x) | invfreq-f32 {:e} ({:.3}x) | \
         invfreq-f64 {:e} ({:.3}x)",
        r.stored,
        r.table,
        r.table / r.stored,
        r.f32_mode,
        r.f32_mode / r.stored,
        r.f64_mode,
        r.f64_mode / r.stored
    );
}

#[test]
fn v_reconstructed_from_the_cached_k_is_no_worse_than_the_v_we_store() {
    let r = run_at(0);
    report("pos 0..256", &r);
    assert!(
        r.stored > 1e-3 && r.stored < 1e-1,
        "the fp8 baseline is implausible ({:e}); the quantizer or the reference \
         is not doing what this test assumes",
        r.stored
    );
    assert!(
        r.table < r.stored * 1.25,
        "reconstructed V is worse than the stored V: derived {:e} vs stored {:e}. \
         Dropping the V slab would then be an accuracy trade, not a free 5.04 GiB.",
        r.table,
        r.stored
    );
}

#[test]
fn the_affordable_angle_source_still_holds_up_at_full_context() {
    let shallow = run_at(0);
    let deep = run_at(262144 - LEN);
    report("pos 0..256", &shallow);
    report("pos 261888..262144", &deep);

    assert!(
        deep.table < deep.stored * 1.25,
        "the table mode should cancel at any depth: {:e} vs stored {:e}",
        deep.table,
        deep.stored
    );
    assert!(
        deep.f64_mode < deep.stored * 1.25,
        "recomputing the angle in f64 does not survive full context: {:e} vs \
         stored {:e}. Then the fused kernel has no affordable angle source and \
         the 5.04 GiB is not reachable this way.",
        deep.f64_mode,
        deep.stored
    );
}
