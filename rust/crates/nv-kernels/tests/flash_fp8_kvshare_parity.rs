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
const N_KV_Q36_GROUP8: usize = 2;
const N_Q_Q36_GROUP8: usize = 16;
const BLOCK_SIZE: usize = 16;

struct Fixture {
    stream: Arc<CudaStream>,
    raw: *mut c_void,
    n_q: usize,
    n_kv: usize,
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
    fixture_shaped(total, N_Q, N_KV)
}

fn fixture_shaped(total: usize, n_q: usize, n_kv: usize) -> Fixture {
    let (src, q) = fixture_src(total, n_q, n_kv);
    fixture_from_src(total, n_q, n_kv, src, q)
}

fn fixture_src(total: usize, n_q: usize, n_kv: usize) -> (Vec<u16>, Vec<u16>) {
    let mut state = 0x2545_f491u32;
    let elems = total * n_kv * HEAD_DIM;
    let src: Vec<u16> = (0..elems)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let q: Vec<u16> = (0..n_q * HEAD_DIM)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    (src, q)
}

fn fixture_from_src(
    total: usize,
    n_q: usize,
    n_kv: usize,
    src: Vec<u16>,
    q: Vec<u16>,
) -> Fixture {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;

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
        let mut d_fp8: CudaSlice<u8> = stream.alloc_zeros::<u8>(slots * n_kv * HEAD_DIM).unwrap();
        let mut d_sc: CudaSlice<f32> = stream.alloc_zeros::<f32>(slots * n_kv).unwrap();
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
                    n_kv as i32,
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
        n_q,
        n_kv,
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
    let scratch_elems = f.n_q * 128 * (HEAD_DIM + 2);
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(f.n_q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(f.n_q * HEAD_DIM).unwrap();
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
                f.n_q as i32,
                f.n_kv as i32,
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

fn run_kvshare_hole(f: &Fixture, splits: usize, hole_lo_delta: i32, hole_hi_delta: i32) -> Vec<u16> {
    let stream = &f.stream;
    let mut d_scratch: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(f.n_q * splits.max(128) * (HEAD_DIM + 2))
        .unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(f.n_q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(f.n_q * HEAD_DIM).unwrap();
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
                f.n_q as i32,
                f.n_kv as i32,
                HEAD_DIM as i32,
                0,
                0,
                splits as i32,
                f.scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
                0,
                hole_lo_delta,
                hole_hi_delta,
            )
        }
    };
    assert_eq!(rc, 0, "kvshare hole launch rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out: Vec<u16> = stream.clone_dtoh(&d_out).unwrap();
    out
}

fn run_kvshare(f: &Fixture, splits: usize, time_reps: usize) -> (Vec<u16>, f64) {
    let stream = &f.stream;
    let mut d_scratch: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(f.n_q * splits.max(128) * (HEAD_DIM + 2))
        .unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(f.n_q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(f.n_q * HEAD_DIM).unwrap();
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
                f.n_q as i32,
                f.n_kv as i32,
                HEAD_DIM as i32,
                0,
                0,
                splits as i32,
                f.scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
                0,
                0,
                0,
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
    let mut d_sink: CudaSlice<f32> = stream.alloc_zeros::<f32>(f.n_kv).unwrap();
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
                f.n_kv as i32,
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
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
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

fn the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it(
) -> std::sync::MutexGuard<'static, ()> {
    static ONE_ARM_STATE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE_ARM_STATE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const DIRECT_ARMS: [usize; 2] = [0, 1];

fn set_arm(direct: usize) {
    std::env::set_var("NV_KVSHARE_DIRECT", direct.to_string());
}

const A_BIT_IDENTITY_SUITE_THAT_READS_NV_KVSHARE_TC_OUT_OF_THE_AMBIENT_ENVIRONMENT_REPORTS_THE_HARNESSES_EXPORTS_INSTEAD_OF_THE_DEFAULT_PATH_BECAUSE_THE_TENSOR_CORE_ARM_SUMS_THE_SCORES_IN_A_DIFFERENT_ORDER_BY_DESIGN:
    &str = "run this suite under NV_KVSHARE_TC=1 and it fails on 1 of 6144 bf16 lanes at \
            max|diff| 1.9e-9, which is the tc arm's documented reassociation and not a defect in \
            the default kvshare kernel; the group-8 twin already clears the gate envs and this \
            one must too, because the arm it is asserting about is the DEFAULT one";

#[test]
fn kvshare_group6_hd256_matches_the_per_q_head_fused_kernel_bit_exactly_at_matched_splits() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    let _ = A_BIT_IDENTITY_SUITE_THAT_READS_NV_KVSHARE_TC_OUT_OF_THE_AMBIENT_ENVIRONMENT_REPORTS_THE_HARNESSES_EXPORTS_INSTEAD_OF_THE_DEFAULT_PATH_BECAUSE_THE_TENSOR_CORE_ARM_SUMS_THE_SCORES_IN_A_DIFFERENT_ORDER_BY_DESIGN;
    clear_gate_envs();
    for total in [1000usize, 8192, 32768] {
        let f = fixture(total);
        let want = run_fused(&f);
        for direct in DIRECT_ARMS {
            set_arm(direct);
            let matched = cuda::flash_decode_fp8_splits_pick(24, 0) as usize;
            let (got, _) = run_kvshare(&f, matched, 0);
            let n_diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            assert_eq!(
                n_diff,
                0,
                "ctx {total} direct {direct}: kvshare splits={matched} \
                 vs fused splits={matched} (the launcher's own pick at NH=24) differ in \
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

fn clear_gate_envs() {
    std::env::remove_var("NV_Q38_ATTN_DEPTHGATE");
    std::env::remove_var("NV_Q36_ATTN_DEPTHGATE");
    std::env::remove_var("NV_KVSHARE_TC");
    std::env::remove_var("NV_Q38_DEPTHGATE_SPLITS");
    std::env::remove_var("NV_Q38_ATTN_GQA_STAGE");
    std::env::remove_var("NV_FLASH_SPLITS_RUNG");
}

#[test]
fn kvshare_group8_hd256_matches_the_per_q_head_fused_kernel_bit_exactly_at_matched_splits() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
    let matched = cuda::flash_decode_fp8_splits_pick(N_Q_Q36_GROUP8 as i32, 0) as usize;
    for total in [1000usize, 8192] {
        let f = fixture_shaped(total, N_Q_Q36_GROUP8, N_KV_Q36_GROUP8);
        let want = run_fused(&f);
        for direct in DIRECT_ARMS {
            set_arm(direct);
            let (got, _) = run_kvshare(&f, matched, 0);
            let n_diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            assert_eq!(
                n_diff,
                0,
                "ctx {total} direct {direct}: group-8 kvshare splits={matched} vs fused \
                 splits={matched} (16q/2kv, the qwen36 share) differ in {n_diff}/{} bf16 \
                 lanes, max|diff|={:.3e}; the GROUP template must preserve the fused \
                 kernel's per-warp position order at group 8 exactly as it does at group 6",
                want.len(),
                max_abs_diff(&want, &got)
            );
        }
    }
    set_arm(1);
}

#[test]
fn kvshare_group8_hd256_tc_arm_stays_within_bf16_rounding_of_the_fused_kernel() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
    clear_gate_envs();
    std::env::set_var("NV_KVSHARE_TC", "1");
    for total in [8192usize, 32768] {
        let f = fixture_shaped(total, N_Q_Q36_GROUP8, N_KV_Q36_GROUP8);
        let want = run_fused(&f);
        for splits in [32usize, 64, 128] {
            let (got, us) = run_kvshare(&f, splits, 30);
            let diff = max_abs_diff(&want, &got);
            let unique = (N_KV_Q36_GROUP8 * total * HEAD_DIM * 2) as f64;
            eprintln!(
                "[kvshare-g8-tc] ctx {total:6} splits {splits:3}: {us:9.2} us | \
                 unique {:7.1} GB/s | max|diff| vs fused {diff:.3e}",
                unique / us / 1e3
            );
            assert!(
                diff < 1.5e-2,
                "ctx {total} splits {splits}: group-8 TC max|diff| {diff:.3e} vs fused \
                 exceeds bf16 reorder tolerance"
            );
        }
    }
    clear_gate_envs();
}

#[test]
fn depthgate_runs_the_default_path_bit_identically_at_and_below_the_knee_and_the_tc_arm_above() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
    let knee = cuda::flash_depthgate_knee_tokens();
    assert!(
        knee >= 1024 && knee <= 8192,
        "knee {knee} outside the plausible mid-depth band; the constant no longer matches \
         its measured basis"
    );
    for total in [256usize, knee] {
        let f = fixture(total);
        clear_gate_envs();
        let want = run_fused(&f);
        std::env::set_var("NV_Q38_ATTN_DEPTHGATE", "1");
        let got = run_fused(&f);
        clear_gate_envs();
        let n_diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        assert_eq!(
            n_diff,
            0,
            "ctx {total} <= knee {knee}: the gated dispatch must run the default fused \
             kernel bit-identically below the knee (the deep TC launch must idle), but \
             {n_diff}/{} bf16 lanes differ, max|diff|={:.3e}",
            want.len(),
            max_abs_diff(&want, &got)
        );
    }
    for total in [knee + BLOCK_SIZE, 8192, 32768] {
        let f = fixture(total);
        clear_gate_envs();
        let fused = run_fused(&f);
        std::env::set_var("NV_Q38_ATTN_DEPTHGATE", "1");
        let gated = run_fused(&f);
        clear_gate_envs();
        std::env::set_var("NV_KVSHARE_TC", "1");
        let (tc32, _) = run_kvshare(&f, 32, 0);
        clear_gate_envs();
        let n_diff_tc = gated.iter().zip(&tc32).filter(|(a, b)| a != b).count();
        assert_eq!(
            n_diff_tc,
            0,
            "ctx {total} > knee {knee}: the gated dispatch must be the TC kvshare arm at \
             splits=32 exactly (the shallow fused launch must idle), but {n_diff_tc}/{} \
             bf16 lanes differ from the explicit TC run, max|diff|={:.3e}",
            gated.len(),
            max_abs_diff(&gated, &tc32)
        );
        let diff = max_abs_diff(&fused, &gated);
        assert!(
            diff < 1.5e-2,
            "ctx {total} > knee {knee}: gated TC arm max|diff| {diff:.3e} vs fused \
             exceeds bf16 reorder tolerance"
        );
    }
    for (total, forced_splits, band) in
        [(8192usize, "32", "mid"), (196608, "0", "auto")]
    {
        let f = fixture(total);
        clear_gate_envs();
        std::env::set_var("NV_Q38_ATTN_DEPTHGATE", "1");
        let banded_default = run_fused(&f);
        std::env::set_var("NV_Q38_DEPTHGATE_SPLITS", forced_splits);
        let forced = run_fused(&f);
        clear_gate_envs();
        let n_diff = banded_default
            .iter()
            .zip(&forced)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            n_diff,
            0,
            "ctx {total}: the banded default deep arm must equal the explicitly forced \
             {band} band (NV_Q38_DEPTHGATE_SPLITS={forced_splits}) bit-for-bit, but \
             {n_diff}/{} lanes differ, max|diff|={:.3e}",
            banded_default.len(),
            max_abs_diff(&banded_default, &forced)
        );
        let fused = run_fused(&f);
        let diff = max_abs_diff(&fused, &banded_default);
        assert!(
            diff < 1.5e-2,
            "ctx {total}: banded {band} arm max|diff| {diff:.3e} vs fused exceeds bf16 \
             reorder tolerance"
        );
    }
}

#[test]
fn depthgate_group8_engages_only_via_the_q36_env_and_gates_at_the_same_knee() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
    let knee = cuda::flash_depthgate_knee_tokens();
    let total = 8192usize;
    let f = fixture_shaped(total, N_Q_Q36_GROUP8, N_KV_Q36_GROUP8);
    let fused = run_fused(&f);
    std::env::set_var("NV_Q38_ATTN_DEPTHGATE", "1");
    let q38_env_only = run_fused(&f);
    clear_gate_envs();
    let n_diff_scoped = fused
        .iter()
        .zip(&q38_env_only)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        n_diff_scoped,
        0,
        "NV_Q38_ATTN_DEPTHGATE must not touch the group-8 (qwen36) shape; {n_diff_scoped} \
         lanes moved"
    );
    std::env::set_var("NV_Q36_ATTN_DEPTHGATE", "1");
    let gated = run_fused(&f);
    clear_gate_envs();
    std::env::set_var("NV_KVSHARE_TC", "1");
    let (tc32, _) = run_kvshare(&f, 32, 0);
    clear_gate_envs();
    let n_diff_tc = gated.iter().zip(&tc32).filter(|(a, b)| a != b).count();
    assert_eq!(
        n_diff_tc,
        0,
        "ctx {total} > knee {knee}: the q36-gated group-8 dispatch must be the TC kvshare \
         arm at splits=32 exactly, but {n_diff_tc}/{} lanes differ, max|diff|={:.3e}",
        gated.len(),
        max_abs_diff(&gated, &tc32)
    );
    let below = fixture_shaped(knee, N_Q_Q36_GROUP8, N_KV_Q36_GROUP8);
    let want = run_fused(&below);
    std::env::set_var("NV_Q36_ATTN_DEPTHGATE", "1");
    let got = run_fused(&below);
    clear_gate_envs();
    let n_diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
    assert_eq!(
        n_diff,
        0,
        "ctx {knee} <= knee: group-8 gated dispatch must stay bit-identical to the \
         default fused kernel below the knee; {n_diff} lanes differ"
    );
}

const TREE_HOLE_LO_DELTA_5_HI_DELTA_2_IS_THE_SERVING_SIBLING_ROW_MASK: (i32, i32) = (5, 2);

fn run_mk_hole(
    f: &Fixture,
    m: usize,
    hole_from_row: i32,
    hole_lo_delta: i32,
    hole_hi_delta: i32,
) -> Vec<u16> {
    let stream = &f.stream;
    let mut d_scratch: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(f.n_q * m * 128 * (HEAD_DIM + 2))
        .unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(f.n_q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * f.n_q * HEAD_DIM).unwrap();
    let mut state = 0x1357_9bdfu32;
    let qm: Vec<u16> = (0..m * f.n_q * HEAD_DIM)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    #[allow(deprecated)]
    let d_qm: CudaSlice<u16> = stream.clone_htod(&qm).unwrap();
    let rc = {
        let (p_q, _a) = d_qm.device_ptr(stream);
        let (p_k, _b) = f.d_k.device_ptr(stream);
        let (p_v, _c) = f.d_v.device_ptr(stream);
        let (p_ks, _d) = f.d_ks.device_ptr(stream);
        let (p_vs, _e) = f.d_vs.device_ptr(stream);
        let (p_nt, _f) = f.d_ntot.device_ptr(stream);
        let (p_out, _h) = d_out.device_ptr_mut(stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(stream);
        unsafe {
            cuda::flash_decode_fused_fp8kv_mk_hole(
                f.raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                0,
                m as i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                f.n_q as i32,
                f.n_kv as i32,
                HEAD_DIM as i32,
                0,
                0,
                f.scaling,
                hole_from_row,
                hole_lo_delta,
                hole_hi_delta,
            )
        }
    };
    assert_eq!(rc, 0, "mk hole launch rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out: Vec<u16> = stream.clone_dtoh(&d_out).unwrap();
    out
}

#[test]
fn mk_hole_arm_is_bitwise_the_mk_kernel_when_disabled_and_excludes_hole_rows_only_from_hole_rows()
{
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
    let m = 6usize;
    let hole_from_row = 4i32;
    let (hole_lo_delta, hole_hi_delta) =
        TREE_HOLE_LO_DELTA_5_HI_DELTA_2_IS_THE_SERVING_SIBLING_ROW_MASK;
    for total in [64usize, 8192] {
        let hole_lo = total - hole_lo_delta as usize;
        let hole_hi = total - hole_hi_delta as usize;
        let (src, q) = fixture_src(total, N_Q, N_KV);
        let row_elems = N_KV * HEAD_DIM;
        let mut src_mutated = src.clone();
        for w in &mut src_mutated[hole_lo * row_elems..hole_hi * row_elems] {
            *w ^= 0x0100;
        }
        let f = fixture_from_src(total, N_Q, N_KV, src.clone(), q.clone());
        let f_mut = fixture_from_src(total, N_Q, N_KV, src_mutated, q.clone());
        let disabled = run_mk_hole(&f, m, m as i32, 0, 0);
        let plain = run_mk_hole(&f, m, m as i32, hole_lo_delta, hole_hi_delta);
        assert_eq!(
            disabled, plain,
            "ctx {total}: hole_from_row=m must make the deltas inert"
        );
        let hole = run_mk_hole(&f, m, hole_from_row, hole_lo_delta, hole_hi_delta);
        let hole_mut = run_mk_hole(&f_mut, m, hole_from_row, hole_lo_delta, hole_hi_delta);
        let row_out = N_Q * HEAD_DIM;
        for qi in 0..m {
            let a = &hole[qi * row_out..(qi + 1) * row_out];
            let b = &hole_mut[qi * row_out..(qi + 1) * row_out];
            let n_diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
            if (qi as i32) < hole_from_row {
                let window_end = total - (m - 1) + qi;
                if window_end > hole_lo {
                    assert!(
                        n_diff > 0,
                        "ctx {total} row {qi}: a non-hole row whose window covers the mutated \
                         rows must move when they move; the mask leaked onto a chain row"
                    );
                }
            } else {
                assert_eq!(
                    n_diff, 0,
                    "ctx {total} row {qi}: rewriting the hole rows moved a sibling row's \
                     output; the hole mask failed to exclude them"
                );
            }
        }
        let no_hole = run_mk_hole(&f, m, m as i32, 0, 0);
        assert_ne!(
            hole, no_hole,
            "ctx {total}: the mk hole mask produced the unmasked output, so it never engaged"
        );
        eprintln!("[mk-hole] ctx {total}: disabled arm inert, sibling rows exclude the hole rows");
    }
}

#[test]
fn kvshare_hole_mask_excludes_the_hole_rows_bitwise_and_matches_a_compacted_reference() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
    let (hole_lo_delta, hole_hi_delta) = TREE_HOLE_LO_DELTA_5_HI_DELTA_2_IS_THE_SERVING_SIBLING_ROW_MASK;
    for total in [64usize, 8192] {
        let hole_lo = total - hole_lo_delta as usize;
        let hole_hi = total - hole_hi_delta as usize;
        let (src, q) = fixture_src(total, N_Q, N_KV);
        let row_elems = N_KV * HEAD_DIM;
        let mut src_mutated = src.clone();
        for w in &mut src_mutated[hole_lo * row_elems..hole_hi * row_elems] {
            *w ^= 0x0100;
        }
        for direct in DIRECT_ARMS {
            set_arm(direct);
            let f = fixture_from_src(total, N_Q, N_KV, src.clone(), q.clone());
            let f_mut = fixture_from_src(total, N_Q, N_KV, src_mutated.clone(), q.clone());
            let hole = run_kvshare_hole(&f, 32, hole_lo_delta, hole_hi_delta);
            let hole_mut = run_kvshare_hole(&f_mut, 32, hole_lo_delta, hole_hi_delta);
            let n_diff = hole.iter().zip(&hole_mut).filter(|(a, b)| a != b).count();
            assert_eq!(
                n_diff,
                0,
                "ctx {total} direct {direct}: rewriting every hole-row K/V moved \
                 {n_diff}/{} output lanes; the hole mask leaked masked rows into the softmax",
                hole.len()
            );
            let (no_hole, _) = run_kvshare(&f, 32, 0);
            assert_ne!(
                hole, no_hole,
                "ctx {total} direct {direct}: the hole mask produced the unmasked output, so \
                 it never engaged"
            );
            let mut src_compact = src[..hole_lo * row_elems].to_vec();
            src_compact.extend_from_slice(&src[hole_hi * row_elems..]);
            let f_compact = fixture_from_src(
                total - (hole_hi - hole_lo),
                N_Q,
                N_KV,
                src_compact,
                q.clone(),
            );
            let want = run_fused(&f_compact);
            let diff = max_abs_diff(&want, &hole);
            assert!(
                diff < 1.5e-2,
                "ctx {total} direct {direct}: hole-masked output max|diff| {diff:.3e} vs the \
                 fused kernel over the physically compacted KV exceeds bf16 reorder tolerance"
            );
            eprintln!(
                "[kvshare-hole] ctx {total} direct {direct}: hole rows bitwise excluded, \
                 compacted-reference max|diff|={diff:.3e}"
            );
        }
    }
    set_arm(1);
}

#[test]
fn kvshare_group6_hd256_split128_stays_within_bf16_rounding_of_the_fused_kernel() {
    let _one_arm_state_at_a_time = the_kvshare_gate_envs_are_process_global_and_reread_on_every_launch_so_each_test_in_this_suite_holds_this_lock_for_its_whole_body_or_a_sibling_setting_nv_kvshare_tc_decides_its_arm_for_it();
    clear_gate_envs();
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
