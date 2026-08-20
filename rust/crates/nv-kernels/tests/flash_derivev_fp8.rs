#![cfg(feature = "cuda")]

mod common;
use common::xorshift;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::time::Instant;
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
const N_Q: usize = 32;
const N_KV: usize = 4;
const BLOCK_SIZE: usize = 16;
const ROPE_ANGLES: usize = 64;
const BASE: f32 = 1_000_000.0;
const W_K: f32 = 0.0615;

fn build_kv(total: usize, inv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 2;
    let mut state = 0x9e37_79b9u32;
    let mut k = vec![0f32; total * N_KV * HEAD_DIM];
    let mut v = vec![0f32; total * N_KV * HEAD_DIM];
    for t in 0..total {
        for h in 0..N_KV {
            let raw: Vec<f32> = (0..HEAD_DIM).map(|_| xorshift(&mut state) * 3.0).collect();
            let ms: f32 = raw.iter().map(|x| x * x).sum::<f32>() / HEAD_DIM as f32;
            let inv_rms = 1.0 / (ms + 1e-6).sqrt();
            let normed: Vec<f32> = raw.iter().map(|x| x * inv_rms).collect();
            let base = (t * N_KV + h) * HEAD_DIM;
            for d in 0..HEAD_DIM {
                v[base + d] = normed[d];
            }
            for j in 0..half {
                let (c, s) = if j < ROPE_ANGLES {
                    let th = t as f64 * inv[j] as f64;
                    (th.cos() as f32, th.sin() as f32)
                } else {
                    (1.0, 0.0)
                };
                let (lo, hi) = (normed[j] * W_K, normed[j + half] * W_K);
                k[base + j] = lo * c - hi * s;
                k[base + j + half] = lo * s + hi * c;
            }
        }
    }
    (k, v)
}

fn reference(q: &[f32], k: &[f32], v: &[f32], total: usize, scaling: f32) -> Vec<f32> {
    let group = N_Q / N_KV;
    let mut out = vec![0f32; N_Q * HEAD_DIM];
    for h in 0..N_Q {
        let kvh = h / group;
        let mut scores = vec![0f64; total];
        for (t, sc) in scores.iter_mut().enumerate() {
            let base = (t * N_KV + kvh) * HEAD_DIM;
            let mut acc = 0f64;
            for d in 0..HEAD_DIM {
                acc += q[h * HEAD_DIM + d] as f64 * k[base + d] as f64;
            }
            *sc = acc * scaling as f64;
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0f64;
        for sc in scores.iter_mut() {
            *sc = (*sc - m).exp();
            denom += *sc;
        }
        for (t, sc) in scores.iter().enumerate() {
            let w = sc / denom;
            let base = (t * N_KV + kvh) * HEAD_DIM;
            for d in 0..HEAD_DIM {
                out[h * HEAD_DIM + d] += (w * v[base + d] as f64) as f32;
            }
        }
    }
    out
}

fn packed_tables(total: usize, inv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut c = vec![0f32; total * ROPE_ANGLES];
    let mut s = vec![0f32; total * ROPE_ANGLES];
    for p in 0..total {
        for j in 0..ROPE_ANGLES {
            let th = p as f64 * inv[j] as f64;
            c[p * ROPE_ANGLES + j] = th.cos() as f32;
            s[p * ROPE_ANGLES + j] = th.sin() as f32;
        }
    }
    (c, s)
}

fn run(total: usize) -> (f32, f32, f32, f64, f64, f64) {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;

    let inv = inv_freq();
    let (k_host, v_host) = build_kv(total, &inv);
    let mut state = 0x1234_abcdu32;
    let q_host: Vec<f32> = (0..N_Q * HEAD_DIM).map(|_| xorshift(&mut state)).collect();
    let scaling = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let want = reference(&q_host, &k_host, &v_host, total, scaling);

    let blocks = total.div_ceil(BLOCK_SIZE);
    let table: Vec<i32> = (0..blocks as i32).collect();
    let slots = blocks * BLOCK_SIZE;
    let bits = |v: &[f32]| -> Vec<u16> { v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect() };

    #[allow(deprecated)]
    let d_table: CudaSlice<i32> = stream.clone_htod(&table).unwrap();
    #[allow(deprecated)]
    let d_start: CudaSlice<i32> = stream.clone_htod(&vec![0i32]).unwrap();
    #[allow(deprecated)]
    let d_ntot: CudaSlice<i32> = stream.clone_htod(&vec![total as i32]).unwrap();
    #[allow(deprecated)]
    let d_inv: CudaSlice<f32> = stream.clone_htod(&inv).unwrap();
    let (cos_pk, sin_pk) = packed_tables(total, &inv);
    #[allow(deprecated)]
    let d_cpk: CudaSlice<f32> = stream.clone_htod(&cos_pk).unwrap();
    #[allow(deprecated)]
    let d_spk: CudaSlice<f32> = stream.clone_htod(&sin_pk).unwrap();
    #[allow(deprecated)]
    let d_q: CudaSlice<u16> = stream.clone_htod(&bits(&q_host)).unwrap();

    let mut quantize = |src: &[f32]| -> (CudaSlice<u8>, CudaSlice<f32>) {
        #[allow(deprecated)]
        let d_src: CudaSlice<u16> = stream.clone_htod(&bits(src)).unwrap();
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
    let (d_k, d_ks) = quantize(&k_host);
    let (d_v, d_vs) = quantize(&v_host);

    let scratch_elems = cuda::flash_splitk_scratch_elems(N_Q as i32, HEAD_DIM as i32) as usize;
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(N_Q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(N_Q * HEAD_DIM).unwrap();

    let read_v = |d_out: &mut CudaSlice<u16>,
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
                N_Q as i32,
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
    assert_eq!(read_v(&mut d_out, &mut d_scratch, &mut d_fan), 0, "read-V kernel");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let got_read: Vec<f32> = stream
        .clone_dtoh(&d_out)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    let derive_v = |packed: bool,
                    d_out: &mut CudaSlice<u16>,
                    d_scratch: &mut CudaSlice<f32>,
                    d_fan: &mut CudaSlice<u32>|
     -> i32 {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = d_k.device_ptr(&stream);
        let (p_ks, _c) = d_ks.device_ptr(&stream);
        let (p_inv, _d) = d_inv.device_ptr(&stream);
        let (p_cpk, _d2) = d_cpk.device_ptr(&stream);
        let (p_spk, _d3) = d_spk.device_ptr(&stream);
        let (p_nt, _e) = d_ntot.device_ptr(&stream);
        let (p_tb, _f) = d_table.device_ptr(&stream);
        let (p_out, _g) = d_out.device_ptr_mut(&stream);
        let (p_sc, _h) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _i) = d_fan.device_ptr_mut(&stream);
        unsafe {
            cuda::flash_decode_derivev_fp8kv_paged(
                raw,
                p_q as *const u16,
                p_k as *const u8,
                p_ks as *const f32,
                if packed { std::ptr::null() } else { p_inv as *const f32 },
                if packed { p_cpk as *const f32 } else { std::ptr::null() },
                if packed { p_spk as *const f32 } else { std::ptr::null() },
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                N_Q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                ROPE_ANGLES as i32,
                1.0 / W_K,
                scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    assert_eq!(derive_v(false, &mut d_out, &mut d_scratch, &mut d_fan), 0, "derive-V kernel");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let got_derive: Vec<f32> = stream
        .clone_dtoh(&d_out)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    assert_eq!(
        derive_v(true, &mut d_out, &mut d_scratch, &mut d_fan),
        0,
        "derive-V packed kernel"
    );
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let got_packed: Vec<f32> = stream
        .clone_dtoh(&d_out)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    let reps = 50;
    for _ in 0..5 {
        read_v(&mut d_out, &mut d_scratch, &mut d_fan);
        derive_v(false, &mut d_out, &mut d_scratch, &mut d_fan);
        derive_v(true, &mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        read_v(&mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let us_read = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let t1 = Instant::now();
    for _ in 0..reps {
        derive_v(false, &mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let us_derive = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let t2 = Instant::now();
    for _ in 0..reps {
        derive_v(true, &mut d_out, &mut d_scratch, &mut d_fan);
    }
    stream.synchronize().unwrap();
    let us_packed = t2.elapsed().as_secs_f64() * 1e6 / reps as f64;

    (
        rel_rms(&got_read, &want),
        rel_rms(&got_derive, &want),
        rel_rms(&got_packed, &want),
        us_read,
        us_derive,
        us_packed,
    )
}

#[test]
fn reconstructing_v_matches_reading_it_and_reads_half_the_kv() {
    for total in [1024usize, 8192, 32768] {
        let (e_read, e_derive, e_packed, us_read, us_derive, us_packed) = run(total);
        let kv_bytes = (total * N_KV * HEAD_DIM) as f64;
        eprintln!(
            "[derivev-flash] ctx {total:6}: read-V {e_read:e} {us_read:8.2}us | invfreq \
             {e_derive:e} {us_derive:8.2}us {:.3}x | packed {e_packed:e} {us_packed:8.2}us \
             {:.3}x",
            us_derive / us_read,
            us_packed / us_read
        );
        let _ = kv_bytes;
        assert!(
            e_read < 5e-2,
            "the read-V baseline is implausible at ctx {total}: {e_read:e}"
        );
        assert!(
            e_packed < e_read * 1.25,
            "the packed-table angle source is worse than reading V at ctx {total}: \
             {e_packed:e} vs {e_read:e}"
        );
        assert!(
            e_derive < e_read * 1.25,
            "reconstructing V is worse than reading it at ctx {total}: {e_derive:e} vs \
             {e_read:e}"
        );
    }
}
