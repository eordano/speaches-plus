#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::time::Instant;

const Q38_NH: usize = 24;
const Q38_NKV: usize = 4;
const Q38_HD: usize = 256;

const L2_COLD_KV_COPIES_16_TIMES_17MB_EVICTS_THE_128MB_L2_LIKE_THE_60MB_OF_WEIGHTS_BETWEEN_LAYERS:
    usize = 16;

fn xorshift(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
}

struct KvArm {
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    ks: CudaSlice<f32>,
    vs: CudaSlice<f32>,
}

const KVSHARE_MAX_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS: usize = 128;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Fused,
    Kvshare,
    Ilp4,
    Gshare,
}

fn bench_depth(
    ctx: &std::sync::Arc<CudaContext>,
    total: usize,
    copies: usize,
    arm: Arm,
) -> (f64, f64, Vec<f32>) {
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let mut state = 0x2545_f491u32;

    let q_host: Vec<u16> = (0..Q38_NH * Q38_HD)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    #[allow(deprecated)]
    let d_q: CudaSlice<u16> = stream.clone_htod(&q_host).unwrap();
    #[allow(deprecated)]
    let d_ntot: CudaSlice<i32> = stream.clone_htod(&vec![total as i32]).unwrap();

    let kv_bytes = total * Q38_NKV * Q38_HD;
    let mut kv_host: Vec<u8> = vec![0u8; kv_bytes];
    let mut fill_state = 0x9e37_79b9u32;
    for chunk in kv_host.chunks_mut(4) {
        fill_state ^= fill_state << 13;
        fill_state ^= fill_state >> 17;
        fill_state ^= fill_state << 5;
        let b = (fill_state & 0x7f7f_7f7f).to_le_bytes();
        chunk.copy_from_slice(&b[..chunk.len()]);
    }
    let sc_host: Vec<f32> = (0..total * Q38_NKV)
        .map(|_| 0.01 + xorshift(&mut state).abs() * 0.01)
        .collect();
    let arms: Vec<KvArm> = (0..copies)
        .map(|_| {
            #[allow(deprecated)]
            let k = stream.clone_htod(&kv_host).unwrap();
            #[allow(deprecated)]
            let v = stream.clone_htod(&kv_host).unwrap();
            #[allow(deprecated)]
            let ks = stream.clone_htod(&sc_host).unwrap();
            #[allow(deprecated)]
            let vs = stream.clone_htod(&sc_host).unwrap();
            KvArm { k, v, ks, vs }
        })
        .collect();

    let scratch_elems = cuda::flash_splitk_scratch_elems(Q38_NH as i32, Q38_HD as i32)
        .max(Q38_NH * KVSHARE_MAX_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS * (Q38_HD + 2))
        .max(cuda::flash_gshare_scratch_elems(Q38_NH as i32, Q38_HD as i32));
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(Q38_NH).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(Q38_NH * Q38_HD).unwrap();
    let scaling = 1.0f32 / (Q38_HD as f32).sqrt();

    let mut go = |kv: &KvArm| -> i32 {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = kv.k.device_ptr(&stream);
        let (p_v, _c) = kv.v.device_ptr(&stream);
        let (p_ks, _d) = kv.ks.device_ptr(&stream);
        let (p_vs, _e) = kv.vs.device_ptr(&stream);
        let (p_nt, _f) = d_ntot.device_ptr(&stream);
        let (p_out, _h) = d_out.device_ptr_mut(&stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(&stream);
        unsafe {
            if arm == Arm::Gshare {
                cuda::flash_decode_fused_fp8kv_gshare(
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
                    Q38_NH as i32,
                    Q38_NKV as i32,
                    Q38_HD as i32,
                    0,
                    0,
                    scaling,
                )
            } else if arm == Arm::Ilp4 {
                cuda::flash_decode_fused_fp8kv_ilp4(
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
                    Q38_NH as i32,
                    Q38_NKV as i32,
                    Q38_HD as i32,
                    0,
                    0,
                    scaling,
                )
            } else if arm == Arm::Kvshare {
                cuda::flash_decode_kvshare_fp8kv_paged(
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
                    Q38_NH as i32,
                    Q38_NKV as i32,
                    Q38_HD as i32,
                    0,
                    0,
                    0,
                    scaling,
                    std::ptr::null(),
                    0,
                    0,
                    0,
                    0,
                )
            } else {
                cuda::flash_decode_fused_fp8kv(
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
                    Q38_NH as i32,
                    Q38_NKV as i32,
                    Q38_HD as i32,
                    0,
                    0,
                    scaling,
                )
            }
        }
    };

    for arm in &arms {
        let rc = go(arm);
        assert_eq!(rc, 0, "launch rc={rc}");
    }
    stream.synchronize().unwrap();

    let reps = 20usize.max(200 / copies);
    let t0 = Instant::now();
    for r in 0..reps {
        for a in &arms {
            let rc = go(a);
            assert_eq!(rc, 0, "rep {r} rc={rc}");
        }
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / (reps * copies) as f64;
    let unique_bytes = (2 * kv_bytes + 2 * total * Q38_NKV * 4) as f64;
    #[allow(deprecated)]
    let out_host: Vec<f32> = stream
        .clone_dtoh(&d_out)
        .unwrap()
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();
    (us, unique_bytes / us / 1e3, out_host)
}

#[test]
#[ignore = "GPU microbench: set NV_Q38_FLASH_FLOOR_TEST=1 -- isolates the qwen3.8 decode \
            attention kernel (24q/4kv hd256 fp8) vs depth, warm and cold L2, fused vs \
            kvshare arms; NV_E4B_FLASH_SPLITS/NV_Q38_ATTN_GQA_STAGE/NV_KVSHARE_TC apply"]
fn q38_flash_decode_kernel_us_vs_depth_warm_and_cold_l2() {
    if std::env::var("NV_Q38_FLASH_FLOOR_TEST").ok().as_deref() != Some("1") {
        panic!(
            "NV_Q38_FLASH_FLOOR_TEST is not 1, so this depth floor executed nothing. It is \
             already #[ignore]d, which makes --ignored the opt-in; a second env gate behind that \
             can only make an --ignored run print `1 passed` for a floor that measured nothing. A \
             suite whose precondition is absent must fail or be ignored, never return green."
        );
    }
    let cold = L2_COLD_KV_COPIES_16_TIMES_17MB_EVICTS_THE_128MB_L2_LIKE_THE_60MB_OF_WEIGHTS_BETWEEN_LAYERS;
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    for &total in &[256usize, 2048, 8192, 32768, 98304, 196608] {
        for &(copies, tag) in &[(1usize, "warmL2"), (cold, "coldL2")] {
            if total >= 98304 && copies > 1 {
                continue;
            }
            let (us_f, gbps_f, out_f) = bench_depth(&ctx, total, copies, Arm::Fused);
            let (us_s, gbps_s, out_s) = bench_depth(&ctx, total, copies, Arm::Kvshare);
            let (us_i, gbps_i, out_i) = bench_depth(&ctx, total, copies, Arm::Ilp4);
            let (us_g, gbps_g, out_g) = bench_depth(&ctx, total, copies, Arm::Gshare);
            let max_mag_f = out_f.iter().map(|a| a.abs()).fold(0f32, f32::max);
            let kv_diff = out_f
                .iter()
                .zip(&out_s)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                kv_diff <= 1e-3 + 0.02 * max_mag_f,
                "kvshare (under the ambient NV_KVSHARE_TC/NV_E4B_FLASH_SPLITS env) diverged \
                 beyond bf16-rounding from fused at total={total}: max|diff|={kv_diff} vs \
                 max|fused|={max_mag_f}"
            );
            let max_diff = out_f
                .iter()
                .zip(&out_i)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let max_mag = out_f.iter().map(|a| a.abs()).fold(0f32, f32::max);
            assert!(
                max_diff <= 1e-3 + 0.02 * max_mag,
                "ilp4 diverged from fused at total={total}: max|diff|={max_diff} vs max|fused|={max_mag}; \
                 both accumulate f32 online-softmax over the same fp8 rows so beyond bf16-rounding \
                 disagreement means an indexing bug, not reordering"
            );
            let g_diff = out_f
                .iter()
                .zip(&out_g)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                g_diff <= 1e-3 + 0.02 * max_mag_f,
                "gshare diverged beyond bf16-rounding from fused at total={total}: \
                 max|diff|={g_diff} vs max|fused|={max_mag_f}"
            );
            eprintln!(
                "FLASH-FLOOR total={total} {tag} fused_us={us_f:.1} fused_gbps={gbps_f:.0} \
                 kvshare_us={us_s:.1} kvshare_gbps={gbps_s:.0} \
                 ilp4_us={us_i:.1} ilp4_gbps={gbps_i:.0} ilp4_maxdiff={max_diff:.2e} \
                 gshare_us={us_g:.1} gshare_gbps={gbps_g:.0} gshare_maxdiff={g_diff:.2e}"
            );
        }
    }
}
