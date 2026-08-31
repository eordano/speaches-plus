#![cfg(feature = "cuda")]

mod common;
use common::htod_f32;
use common::htod_u16;
use common::lcg_unit_f32 as lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const SERVING_SHAPE_Q38_H48_HK16_D128_CONV10240: (usize, usize, usize, usize, usize) =
    (48, 16, 128, 128, 10240);
const SERVING_CHUNK_T_4096_THE_NV_Q38_PREFILL_CHUNK_DEFAULT: usize = 4096;

fn bench_gated() -> bool {
    std::env::var("NV_GDN_SCANV2_BENCH").as_deref() == Ok("1")
}

fn stream_or_skip(test: &str) -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => panic!("{test}: no CUDA device 0: {e}"),
    }
}

fn rand_bf16(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(lo + lcg(seed) * (hi - lo)).to_bits())
        .collect()
}

fn rand_f32(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| lo + lcg(seed) * (hi - lo)).collect()
}

struct ScanFixture {
    t: usize,
    h: usize,
    hk: usize,
    conv_dim: usize,
    key_dim: usize,
    value_dim: usize,
    d_mixed: CudaSlice<u16>,
    d_z: CudaSlice<u16>,
    d_g: CudaSlice<f32>,
    d_beta: CudaSlice<f32>,
    d_nw: CudaSlice<u16>,
    state0: Vec<f32>,
}

fn scan_fixture(stream: &Arc<CudaStream>, t: usize, seed_base: u64) -> ScanFixture {
    let (h, hk, _, _, _) = SERVING_SHAPE_Q38_H48_HK16_D128_CONV10240;
    scan_fixture_shaped(stream, t, h, hk, seed_base)
}

fn scan_fixture_shaped(
    stream: &Arc<CudaStream>,
    t: usize,
    h: usize,
    hk: usize,
    seed_base: u64,
) -> ScanFixture {
    let (d_k, d_v) = (128usize, 128usize);
    let key_dim = hk * d_k;
    let value_dim = h * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let mut seed = seed_base;
    let mixed = rand_bf16(&mut seed, t * conv_dim, -1.5, 1.5);
    let z = rand_bf16(&mut seed, t * value_dim, -1.0, 1.0);
    let g = rand_f32(&mut seed, t * h, 0.62, 0.999);
    let beta = rand_f32(&mut seed, t * h, 0.05, 0.95);
    let nw = rand_bf16(&mut seed, d_v, 0.5, 1.5);
    let state0 = rand_f32(&mut seed, h * d_k * d_v, -0.05, 0.05);
    ScanFixture {
        t,
        h,
        hk,
        conv_dim,
        key_dim,
        value_dim,
        d_mixed: htod_u16(stream, &mixed),
        d_z: htod_u16(stream, &z),
        d_g: htod_f32(stream, &g),
        d_beta: htod_f32(stream, &beta),
        d_nw: htod_u16(stream, &nw),
        state0,
    }
}

struct ScanBufs {
    d_qn: CudaSlice<f32>,
    d_kn: CudaSlice<f32>,
    d_state: CudaSlice<f32>,
    d_core: CudaSlice<u16>,
}

fn scan_bufs(stream: &Arc<CudaStream>, fx: &ScanFixture) -> ScanBufs {
    ScanBufs {
        d_qn: stream.alloc_zeros::<f32>(fx.t * fx.key_dim).unwrap(),
        d_kn: stream.alloc_zeros::<f32>(fx.t * fx.key_dim).unwrap(),
        d_state: htod_f32(stream, &fx.state0),
        d_core: stream.alloc_zeros::<u16>(fx.t * fx.value_dim).unwrap(),
    }
}

fn run_l2norm(stream: &Arc<CudaStream>, fx: &ScanFixture, bufs: &mut ScanBufs) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pm, _1) = fx.d_mixed.device_ptr(stream);
    let (pq, _2) = bufs.d_qn.device_ptr_mut(stream);
    let (pk, _3) = bufs.d_kn.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::gdn_prefill_qk_l2norm_from_mixed(
            raw,
            pm as *const u16,
            pq as *mut f32,
            pk as *mut f32,
            fx.t as i32,
            fx.hk as i32,
            fx.conv_dim as i32,
            fx.key_dim as i32,
            (1.0 / (128f64).sqrt()) as f32,
            1e-6,
        )
    };
    assert_eq!(rc, 0, "l2norm rc");
}

#[derive(Clone, Copy, PartialEq)]
enum ScanKernel {
    V1Block128,
    V2Warp32Shfl,
}

fn run_scan_kernel(
    stream: &Arc<CudaStream>,
    fx: &ScanFixture,
    bufs: &mut ScanBufs,
    which: ScanKernel,
) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pq, _1) = bufs.d_qn.device_ptr(stream);
    let (pk, _2) = bufs.d_kn.device_ptr(stream);
    let (pm, _3) = fx.d_mixed.device_ptr(stream);
    let (pg, _4) = fx.d_g.device_ptr(stream);
    let (pb, _5) = fx.d_beta.device_ptr(stream);
    let (ps, _6) = bufs.d_state.device_ptr_mut(stream);
    let (pc, _7) = bufs.d_core.device_ptr_mut(stream);
    let f = match which {
        ScanKernel::V1Block128 => cuda::gdn_recurrent_stateful_gqa_bf16out,
        ScanKernel::V2Warp32Shfl => cuda::gdn_recurrent_stateful_gqa_bf16out_shfl,
    };
    let rc = unsafe {
        f(
            raw,
            pq as *const f32,
            pk as *const f32,
            pm as *const u16,
            pg as *const f32,
            pb as *const f32,
            ps as *mut f32,
            pc as *mut u16,
            1,
            fx.t as i32,
            fx.h as i32,
            fx.hk as i32,
            128,
            128,
            fx.conv_dim as i32,
            (2 * fx.key_dim) as i32,
        )
    };
    assert_eq!(rc, 0, "scan kernel rc");
}

fn run_rmsgate(stream: &Arc<CudaStream>, fx: &ScanFixture, bufs: &mut ScanBufs) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pz, _1) = fx.d_z.device_ptr(stream);
    let (pn, _2) = fx.d_nw.device_ptr(stream);
    let (pc, _3) = bufs.d_core.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::gdn_prefill_rmsnorm_gate_bf16(
            raw,
            pc as *const u16,
            pz as *const u16,
            pn as *const u16,
            pc as *mut u16,
            (fx.t * fx.h) as i32,
            128,
            1e-6,
        )
    };
    assert_eq!(rc, 0, "rmsgate rc");
}

fn wall_ms<F: FnMut()>(stream: &Arc<CudaStream>, mut f: F, warmup: usize, reps: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        f();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_GDN_SCANV2_BENCH=1 -- prices the gdn prefill scan stage at the q38 serving shape"]
fn gdn_prefill_scan_stage_microbench_at_q38_serving_shape() {
    if !bench_gated() {
        panic!(
            "NV_GDN_SCANV2_BENCH is not 1, so this microbench executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having measured nothing. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    let stream = match stream_or_skip("gdn_prefill_scan_stage_microbench") {
        Some(s) => s,
        None => return,
    };
    let warmup = 3usize;
    let reps = 10usize;

    for t in [512usize, 1024, 2048, SERVING_CHUNK_T_4096_THE_NV_Q38_PREFILL_CHUNK_DEFAULT] {
        let fx = scan_fixture(&stream, t, 0x5eed_0001);
        let mut bufs = scan_bufs(&stream, &fx);
        let l2_ms = wall_ms(&stream, || run_l2norm(&stream, &fx, &mut bufs), warmup, reps);
        let scan_ms = wall_ms(
            &stream,
            || run_scan_kernel(&stream, &fx, &mut bufs, ScanKernel::V1Block128),
            warmup,
            reps,
        );
        let shfl_ms = wall_ms(
            &stream,
            || run_scan_kernel(&stream, &fx, &mut bufs, ScanKernel::V2Warp32Shfl),
            warmup,
            reps,
        );
        let rms_ms = wall_ms(&stream, || run_rmsgate(&stream, &fx, &mut bufs), warmup, reps);

        let l2_bytes = (t * fx.conv_dim.min(2 * fx.key_dim) * 2 + 2 * t * fx.key_dim * 4) as f64;
        let scan_bytes = (2 * t * fx.key_dim * 4
            + t * fx.value_dim * 2
            + 2 * t * fx.h * 4
            + 2 * fx.h * 128 * 128 * 4
            + t * fx.value_dim * 2) as f64;
        let rms_bytes = (3 * t * fx.value_dim * 2) as f64;
        let scan_flops = (t * fx.h * 128 * 128 * 4) as f64;
        let l2_gbs = l2_bytes / l2_ms / 1e6;
        let scan_gbs = scan_bytes / scan_ms / 1e6;
        let scan_tfs = scan_flops / scan_ms / 1e9;
        let scan_ns_tok = scan_ms * 1e6 / t as f64;
        let rms_gbs = rms_bytes / rms_ms / 1e6;
        let shfl_ns_tok = shfl_ms * 1e6 / t as f64;
        let shfl_x = scan_ms / shfl_ms;
        eprintln!(
            "stage-bench t={t} l2norm={l2_ms:.3}ms ({l2_gbs:.0} GB/s) scan_v1={scan_ms:.3}ms \
             ({scan_gbs:.0} GB/s, {scan_tfs:.2} TFLOP/s, {scan_ns_tok:.0} ns/tok) \
             shfl={shfl_ms:.3}ms ({shfl_ns_tok:.0} ns/tok, {shfl_x:.2}x) \
             rmsgate={rms_ms:.3}ms ({rms_gbs:.0} GB/s)",
        );
    }
}

const PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_PREFETCH_AND_GQA_RATIOS: &[(usize, usize, usize)] = &[
    (1, 48, 16),
    (5, 48, 16),
    (64, 16, 16),
    (33, 32, 16),
    (511, 48, 16),
    (4096, 48, 16),
];

fn dtoh_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    #[allow(deprecated)]
    let v = stream.memcpy_dtov(d).unwrap();
    v
}

#[test]
#[ignore = "cuda bitwise gate; set NV_GDN_SCANV2_TEST=1 -- every scan-kernel variant must reproduce V1Block128's out and state bit for bit, which is the whole claim the shfl scan ships on"]
fn gdn_scan_v2_variants_bitwise_match_v1_out_and_state() {
    if std::env::var("NV_GDN_SCANV2_TEST").as_deref() != Ok("1") {
        panic!(
            "NV_GDN_SCANV2_TEST is not 1, so the bitwise gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having compared no bits, which is the only check behind the \
             kernel's bitwise claim. A suite whose precondition is absent must fail or be \
             ignored, never return green."
        );
    }
    let stream = match stream_or_skip("gdn_scan_v2_variants_bitwise_match_v1") {
        Some(s) => s,
        None => return,
    };
    for &(t, h, hk) in PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_PREFETCH_AND_GQA_RATIOS {
        let fx = scan_fixture_shaped(&stream, t, h, hk, 0xabcd_0000 + t as u64);
        let mut ref_out: Option<(Vec<u16>, Vec<f32>)> = None;
        for which in [
            ScanKernel::V1Block128,
            ScanKernel::V2Warp32Shfl,
        ] {
            let mut bufs = scan_bufs(&stream, &fx);
            run_l2norm(&stream, &fx, &mut bufs);
            run_scan_kernel(&stream, &fx, &mut bufs, which);
            stream.synchronize().unwrap();
            let out = common::dtoh_u16(&stream, &bufs.d_core);
            let state = dtoh_f32(&stream, &bufs.d_state);
            match &ref_out {
                None => ref_out = Some((out, state)),
                Some((ro, rs)) => {
                    let name = match which {
                        ScanKernel::V2Warp32Shfl => "shfl",
                        ScanKernel::V1Block128 => unreachable!(),
                    };
                    common::assert_u16_bits(&format!("{name} out t={t} h={h} hk={hk}"), &out, ro);
                    let state_bits: Vec<u32> = state.iter().map(|v| v.to_bits()).collect();
                    let ref_bits: Vec<u32> = rs.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(
                        state_bits, ref_bits,
                        "{name} state bits t={t} h={h} hk={hk}"
                    );
                }
            }
        }
        eprintln!("scan v2 bitwise parity ok t={t} h={h} hk={hk}");
    }
}
