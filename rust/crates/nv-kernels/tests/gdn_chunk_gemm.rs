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

const SERVING_SHAPE_Q38_H48_HK16_D128: (usize, usize) = (48, 16);
const SERVING_CHUNK_T_4096_THE_NV_Q38_PREFILL_CHUNK_DEFAULT: usize = 4096;

const GEMM_VS_SEQUENTIAL_REL_TOL_1_6E2_FOUR_BF16_EPS_THE_GEMM_INPUTS_ARE_BF16_TENSOR_CORE_ROUNDED_WHERE_THE_SCAN_READS_F32:
    f32 = 1.6e-2;

fn parity_gated() -> bool {
    std::env::var("NV_GDN_CHUNK_GEMM_TEST").as_deref() == Ok("1")
}

fn bench_gated() -> bool {
    std::env::var("NV_GDN_CHUNK_GEMM_BENCH").as_deref() == Ok("1")
}

fn stream_or_panic(test: &str) -> Arc<CudaStream> {
    match CudaContext::new(0) {
        Ok(c) => c.default_stream(),
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

struct Fx {
    t: usize,
    h: usize,
    hk: usize,
    conv_dim: usize,
    key_dim: usize,
    value_dim: usize,
    d_mixed: CudaSlice<u16>,
    d_g: CudaSlice<f32>,
    d_beta: CudaSlice<f32>,
    state0: Vec<f32>,
}

fn fixture(stream: &Arc<CudaStream>, t: usize, h: usize, hk: usize, seed_base: u64) -> Fx {
    let key_dim = hk * 128;
    let value_dim = h * 128;
    let conv_dim = 2 * key_dim + value_dim;
    let mut seed = seed_base;
    let mixed = rand_bf16(&mut seed, t * conv_dim, -1.5, 1.5);
    let g = rand_f32(&mut seed, t * h, 0.62, 0.999);
    let beta = rand_f32(&mut seed, t * h, 0.05, 0.95);
    let state0 = rand_f32(&mut seed, h * 128 * 128, -0.05, 0.05);
    Fx {
        t,
        h,
        hk,
        conv_dim,
        key_dim,
        value_dim,
        d_mixed: htod_u16(stream, &mixed),
        d_g: htod_f32(stream, &g),
        d_beta: htod_f32(stream, &beta),
        state0,
    }
}

struct Bufs {
    d_qn: CudaSlice<f32>,
    d_kn: CudaSlice<f32>,
    d_state: CudaSlice<f32>,
    d_core: CudaSlice<u16>,
}

fn bufs(stream: &Arc<CudaStream>, fx: &Fx) -> Bufs {
    Bufs {
        d_qn: stream.alloc_zeros::<f32>(fx.t * fx.key_dim).unwrap(),
        d_kn: stream.alloc_zeros::<f32>(fx.t * fx.key_dim).unwrap(),
        d_state: htod_f32(stream, &fx.state0),
        d_core: stream.alloc_zeros::<u16>(fx.t * fx.value_dim).unwrap(),
    }
}

fn run_l2norm(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pm, _1) = fx.d_mixed.device_ptr(stream);
    let (pq, _2) = b.d_qn.device_ptr_mut(stream);
    let (pk, _3) = b.d_kn.device_ptr_mut(stream);
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

fn run_scan_v2_shfl(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pq, _1) = b.d_qn.device_ptr(stream);
    let (pk, _2) = b.d_kn.device_ptr(stream);
    let (pm, _3) = fx.d_mixed.device_ptr(stream);
    let (pg, _4) = fx.d_g.device_ptr(stream);
    let (pb, _5) = fx.d_beta.device_ptr(stream);
    let (ps, _6) = b.d_state.device_ptr_mut(stream);
    let (pc, _7) = b.d_core.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::gdn_recurrent_stateful_gqa_bf16out_shfl(
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
    assert_eq!(rc, 0, "scan v2 shfl rc");
}

fn run_scan_v1(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pq, _1) = b.d_qn.device_ptr(stream);
    let (pk, _2) = b.d_kn.device_ptr(stream);
    let (pm, _3) = fx.d_mixed.device_ptr(stream);
    let (pg, _4) = fx.d_g.device_ptr(stream);
    let (pb, _5) = fx.d_beta.device_ptr(stream);
    let (ps, _6) = b.d_state.device_ptr_mut(stream);
    let (pc, _7) = b.d_core.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::gdn_recurrent_stateful_gqa_bf16out(
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
    assert_eq!(rc, 0, "scan v1 rc");
}

fn run_chunk_gemm(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs, ws: &mut CudaSlice<f32>) {
    std::env::set_var(
        V3_ENV_NV_Q38_GDN_CHUNK_GEMM_V3_SELECTS_THE_SPLIT_LADDER_PLUS_PARALLEL_OUTPUT_DECOMPOSITION,
        "0",
    );
    run_chunk_gemm_under_caller_pinned_tail_envs(stream, fx, b, ws);
    std::env::remove_var(
        V3_ENV_NV_Q38_GDN_CHUNK_GEMM_V3_SELECTS_THE_SPLIT_LADDER_PLUS_PARALLEL_OUTPUT_DECOMPOSITION,
    );
}

fn run_chunk_gemm_under_caller_pinned_tail_envs(
    stream: &Arc<CudaStream>,
    fx: &Fx,
    b: &mut Bufs,
    ws: &mut CudaSlice<f32>,
) {
    let raw = stream.cu_stream() as *mut c_void;
    let (pq, _1) = b.d_qn.device_ptr(stream);
    let (pk, _2) = b.d_kn.device_ptr(stream);
    let (pm, _3) = fx.d_mixed.device_ptr(stream);
    let (pg, _4) = fx.d_g.device_ptr(stream);
    let (pb, _5) = fx.d_beta.device_ptr(stream);
    let (ps, _6) = b.d_state.device_ptr_mut(stream);
    let (pc, _7) = b.d_core.device_ptr_mut(stream);
    let ws_len = ws.len();
    let (pw, _8) = ws.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::gdn_chunk_gemm_scan_f32(
            raw,
            pq as *const f32,
            pk as *const f32,
            pm as *const u16,
            pg as *const f32,
            pb as *const f32,
            ps as *mut f32,
            pc as *mut u16,
            pw as *mut f32,
            ws_len as i64,
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
    assert_eq!(rc, 0, "chunk gemm rc");
}

const FWDH_ENV_NV_Q38_GDN_CHUNK_GEMM_FWDH_SELECTS_THE_STATE_RESIDENT_TAIL_OVER_THE_STACK_CARRY:
    &str = "NV_Q38_GDN_CHUNK_GEMM_FWDH";

fn run_chunk_gemm_fwdh(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs, ws: &mut CudaSlice<f32>) {
    std::env::set_var(
        FWDH_ENV_NV_Q38_GDN_CHUNK_GEMM_FWDH_SELECTS_THE_STATE_RESIDENT_TAIL_OVER_THE_STACK_CARRY,
        "1",
    );
    run_chunk_gemm(stream, fx, b, ws);
    std::env::remove_var(
        FWDH_ENV_NV_Q38_GDN_CHUNK_GEMM_FWDH_SELECTS_THE_STATE_RESIDENT_TAIL_OVER_THE_STACK_CARRY,
    );
}

const MMA_ENV_NV_Q38_GDN_CHUNK_GEMM_MMA_SELECTS_THE_FRAGMENT_TENSOR_CORE_TAIL_OVER_BOTH_SCALAR_TAILS:
    &str = "NV_Q38_GDN_CHUNK_GEMM_MMA";

fn run_chunk_gemm_mma(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs, ws: &mut CudaSlice<f32>) {
    std::env::set_var(
        MMA_ENV_NV_Q38_GDN_CHUNK_GEMM_MMA_SELECTS_THE_FRAGMENT_TENSOR_CORE_TAIL_OVER_BOTH_SCALAR_TAILS,
        "1",
    );
    run_chunk_gemm(stream, fx, b, ws);
    std::env::remove_var(
        MMA_ENV_NV_Q38_GDN_CHUNK_GEMM_MMA_SELECTS_THE_FRAGMENT_TENSOR_CORE_TAIL_OVER_BOTH_SCALAR_TAILS,
    );
}

const V3_ENV_NV_Q38_GDN_CHUNK_GEMM_V3_SELECTS_THE_SPLIT_LADDER_PLUS_PARALLEL_OUTPUT_DECOMPOSITION:
    &str = "NV_Q38_GDN_CHUNK_GEMM_V3";

fn run_chunk_gemm_v3(stream: &Arc<CudaStream>, fx: &Fx, b: &mut Bufs, ws: &mut CudaSlice<f32>) {
    std::env::set_var(
        V3_ENV_NV_Q38_GDN_CHUNK_GEMM_V3_SELECTS_THE_SPLIT_LADDER_PLUS_PARALLEL_OUTPUT_DECOMPOSITION,
        "1",
    );
    run_chunk_gemm_under_caller_pinned_tail_envs(stream, fx, b, ws);
    std::env::remove_var(
        V3_ENV_NV_Q38_GDN_CHUNK_GEMM_V3_SELECTS_THE_SPLIT_LADDER_PLUS_PARALLEL_OUTPUT_DECOMPOSITION,
    );
}

fn workspace(stream: &Arc<CudaStream>, fx: &Fx) -> CudaSlice<f32> {
    let elems = cuda::gdn_chunk_gemm_scan_workspace_elems(fx.t as i32, fx.h as i32, fx.hk as i32);
    assert!(elems > 0, "workspace_elems must be positive for t={}", fx.t);
    stream.alloc_zeros::<f32>(elems).unwrap()
}

fn dtoh_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    #[allow(deprecated)]
    let v = stream.memcpy_dtov(d).unwrap();
    v
}

fn max_rel(name: &str, a_bits: &[u16], b_bits: &[u16]) -> f32 {
    assert_eq!(a_bits.len(), b_bits.len(), "{name} length");
    let mut d_max = 0f32;
    let mut m_max = 1e-6f32;
    for (x, y) in a_bits.iter().zip(b_bits.iter()) {
        let xf = bf16::from_bits(*x).to_f32();
        let yf = bf16::from_bits(*y).to_f32();
        d_max = d_max.max((xf - yf).abs());
        m_max = m_max.max(xf.abs());
    }
    d_max / m_max
}

fn max_rel_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut d_max = 0f32;
    let mut m_max = 1e-6f32;
    for (x, y) in a.iter().zip(b.iter()) {
        d_max = d_max.max((x - y).abs());
        m_max = m_max.max(x.abs());
    }
    d_max / m_max
}

const PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_SUBCHUNK_SEAMS_TAILS_AND_GQA_RATIOS: &[(
    usize,
    usize,
    usize,
)] = &[
    (1, 48, 16),
    (5, 48, 16),
    (63, 48, 16),
    (64, 16, 16),
    (65, 48, 16),
    (511, 48, 16),
    (513, 32, 16),
    (4096, 48, 16),
];

#[test]
#[ignore = "cuda parity gate; set NV_GDN_CHUNK_GEMM_TEST=1 -- the chunked WY/UT scan against the sequential scan within reassociation drift"]
fn gdn_chunk_gemm_scan_tracks_the_sequential_scan_within_reassociation_drift() {
    if !parity_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_TEST is not 1, so this parity gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having compared nothing. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    let stream = stream_or_panic("gdn_chunk_gemm_scan_tracks_the_sequential_scan");
    for &(t, h, hk) in PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_SUBCHUNK_SEAMS_TAILS_AND_GQA_RATIOS {
        let fx = fixture(&stream, t, h, hk, 0xd6e0_0000 + t as u64);
        let mut b_ref = bufs(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_ref);
        run_scan_v1(&stream, &fx, &mut b_ref);
        stream.synchronize().unwrap();
        let out_ref = common::dtoh_u16(&stream, &b_ref.d_core);
        let st_ref = dtoh_f32(&stream, &b_ref.d_state);

        let mut b_gemm = bufs(&stream, &fx);
        let mut ws = workspace(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_gemm);
        run_chunk_gemm(&stream, &fx, &mut b_gemm, &mut ws);
        stream.synchronize().unwrap();
        let out_gemm = common::dtoh_u16(&stream, &b_gemm.d_core);
        let st_gemm = dtoh_f32(&stream, &b_gemm.d_state);

        let r_out = max_rel(&format!("out t={t}"), &out_ref, &out_gemm);
        let r_st = max_rel_f32(&st_ref, &st_gemm);
        eprintln!(
            "chunk-gemm parity t={t} h={h} hk={hk}: out rel={r_out:.3e} state rel={r_st:.3e}"
        );
        let tol =
            GEMM_VS_SEQUENTIAL_REL_TOL_1_6E2_FOUR_BF16_EPS_THE_GEMM_INPUTS_ARE_BF16_TENSOR_CORE_ROUNDED_WHERE_THE_SCAN_READS_F32;
        assert!(
            r_out <= tol && r_st <= tol,
            "chunk-gemm scan diverged from the sequential scan beyond the bf16-input rounding \
             class (t={t} h={h} hk={hk}): out rel {r_out:.3e} state rel {r_st:.3e} tol {tol:.1e}; \
             a seam/mask/state-carry defect reads orders of magnitude above this"
        );
        let nonzero = out_gemm.iter().filter(|w| **w != 0).count();
        assert!(
            nonzero * 4 > out_gemm.len(),
            "chunk-gemm output mostly zero ({nonzero}/{}), a zeroed buffer would pass a \
             tolerance gate",
            out_gemm.len()
        );
    }
}

#[test]
#[ignore = "cuda parity gate; set NV_GDN_CHUNK_GEMM_TEST=1 -- the state-resident fwdh tail against the sequential scan within reassociation drift, T-flat across sub-chunk seams and tails"]
fn gdn_chunk_gemm_fwdh_tail_tracks_the_sequential_scan_within_reassociation_drift() {
    if !parity_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_TEST is not 1, so this parity gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having compared nothing. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    let stream = stream_or_panic("gdn_chunk_gemm_fwdh_tail_tracks_the_sequential_scan");
    let mut worst: (f32, usize) = (0.0, 0);
    for &(t, h, hk) in PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_SUBCHUNK_SEAMS_TAILS_AND_GQA_RATIOS {
        let fx = fixture(&stream, t, h, hk, 0xfd40_0000 + t as u64);
        let mut b_ref = bufs(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_ref);
        run_scan_v1(&stream, &fx, &mut b_ref);
        stream.synchronize().unwrap();
        let out_ref = common::dtoh_u16(&stream, &b_ref.d_core);
        let st_ref = dtoh_f32(&stream, &b_ref.d_state);

        let mut b_fwdh = bufs(&stream, &fx);
        let mut ws = workspace(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_fwdh);
        run_chunk_gemm_fwdh(&stream, &fx, &mut b_fwdh, &mut ws);
        stream.synchronize().unwrap();
        let out_fwdh = common::dtoh_u16(&stream, &b_fwdh.d_core);
        let st_fwdh = dtoh_f32(&stream, &b_fwdh.d_state);

        let r_out = max_rel(&format!("fwdh out t={t}"), &out_ref, &out_fwdh);
        let r_st = max_rel_f32(&st_ref, &st_fwdh);
        eprintln!("fwdh parity t={t} h={h} hk={hk}: out rel={r_out:.3e} state rel={r_st:.3e}");
        let tol =
            GEMM_VS_SEQUENTIAL_REL_TOL_1_6E2_FOUR_BF16_EPS_THE_GEMM_INPUTS_ARE_BF16_TENSOR_CORE_ROUNDED_WHERE_THE_SCAN_READS_F32;
        assert!(
            r_out <= tol && r_st <= tol,
            "fwdh state-resident tail diverged from the sequential scan beyond the bf16-input \
             rounding class (t={t} h={h} hk={hk}): out rel {r_out:.3e} state rel {r_st:.3e} tol \
             {tol:.1e}; a seam/decay/state-carry defect reads orders of magnitude above this, and \
             error GROWING with t means per-sub-chunk drift compounding through the resident state"
        );
        if r_out > worst.0 {
            worst = (r_out, t);
        }
        let nonzero = out_fwdh.iter().filter(|w| **w != 0).count();
        assert!(
            nonzero * 4 > out_fwdh.len(),
            "fwdh output mostly zero ({nonzero}/{}), a zeroed buffer would pass a tolerance gate",
            out_fwdh.len()
        );
    }
    eprintln!("fwdh parity worst out rel={:.3e} at t={}", worst.0, worst.1);
}

#[test]
#[ignore = "cuda parity gate; set NV_GDN_CHUNK_GEMM_TEST=1 -- the wmma-fragment state-resident tail against the sequential scan within reassociation drift, T-flat across sub-chunk seams and tails"]
fn gdn_chunk_gemm_mma_tail_tracks_the_sequential_scan_within_reassociation_drift() {
    if !parity_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_TEST is not 1, so this parity gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having compared nothing. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    let stream = stream_or_panic("gdn_chunk_gemm_mma_tail_tracks_the_sequential_scan");
    let mut worst: (f32, usize) = (0.0, 0);
    for &(t, h, hk) in PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_SUBCHUNK_SEAMS_TAILS_AND_GQA_RATIOS {
        let fx = fixture(&stream, t, h, hk, 0x33a0_0000 + t as u64);
        let mut b_ref = bufs(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_ref);
        run_scan_v1(&stream, &fx, &mut b_ref);
        stream.synchronize().unwrap();
        let out_ref = common::dtoh_u16(&stream, &b_ref.d_core);
        let st_ref = dtoh_f32(&stream, &b_ref.d_state);

        let mut b_mma = bufs(&stream, &fx);
        let mut ws = workspace(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_mma);
        run_chunk_gemm_mma(&stream, &fx, &mut b_mma, &mut ws);
        stream.synchronize().unwrap();
        let out_mma = common::dtoh_u16(&stream, &b_mma.d_core);
        let st_mma = dtoh_f32(&stream, &b_mma.d_state);

        let r_out = max_rel(&format!("mma out t={t}"), &out_ref, &out_mma);
        let r_st = max_rel_f32(&st_ref, &st_mma);
        eprintln!("mma parity t={t} h={h} hk={hk}: out rel={r_out:.3e} state rel={r_st:.3e}");
        let tol =
            GEMM_VS_SEQUENTIAL_REL_TOL_1_6E2_FOUR_BF16_EPS_THE_GEMM_INPUTS_ARE_BF16_TENSOR_CORE_ROUNDED_WHERE_THE_SCAN_READS_F32;
        assert!(
            r_out <= tol && r_st <= tol,
            "mma fragment tail diverged from the sequential scan beyond the bf16-input rounding \
             class (t={t} h={h} hk={hk}): out rel {r_out:.3e} state rel {r_st:.3e} tol {tol:.1e}; \
             a seam/decay/state-carry defect reads orders of magnitude above this, and error \
             GROWING with t means per-sub-chunk drift compounding through the resident state \
             whose multiplicative uses are bf16-rounded here where fwdh kept them f32"
        );
        if r_out > worst.0 {
            worst = (r_out, t);
        }
        let nonzero = out_mma.iter().filter(|w| **w != 0).count();
        assert!(
            nonzero * 4 > out_mma.len(),
            "mma output mostly zero ({nonzero}/{}), a zeroed buffer would pass a tolerance gate",
            out_mma.len()
        );
    }
    eprintln!("mma parity worst out rel={:.3e} at t={}", worst.0, worst.1);
}

#[test]
#[ignore = "cuda parity gate; set NV_GDN_CHUNK_GEMM_TEST=1 -- the split ladder plus parallel checkpoint-output tail against the sequential scan within reassociation drift, T-flat across sub-chunk seams and tails"]
fn gdn_chunk_gemm_v3_tail_tracks_the_sequential_scan_within_reassociation_drift() {
    if !parity_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_TEST is not 1, so this parity gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having compared nothing. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    let stream = stream_or_panic("gdn_chunk_gemm_v3_tail_tracks_the_sequential_scan");
    let mut worst: (f32, usize) = (0.0, 0);
    for &(t, h, hk) in PARITY_SHAPES_T_H_HK_EDGE_TS_CROSS_SUBCHUNK_SEAMS_TAILS_AND_GQA_RATIOS {
        let fx = fixture(&stream, t, h, hk, 0x77c0_0000 + t as u64);
        let mut b_ref = bufs(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_ref);
        run_scan_v1(&stream, &fx, &mut b_ref);
        stream.synchronize().unwrap();
        let out_ref = common::dtoh_u16(&stream, &b_ref.d_core);
        let st_ref = dtoh_f32(&stream, &b_ref.d_state);

        let mut b_v3 = bufs(&stream, &fx);
        let mut ws = workspace(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_v3);
        run_chunk_gemm_v3(&stream, &fx, &mut b_v3, &mut ws);
        stream.synchronize().unwrap();
        let out_v3 = common::dtoh_u16(&stream, &b_v3.d_core);
        let st_v3 = dtoh_f32(&stream, &b_v3.d_state);

        let r_out = max_rel(&format!("v3 out t={t}"), &out_ref, &out_v3);
        let r_st = max_rel_f32(&st_ref, &st_v3);
        eprintln!("v3 parity t={t} h={h} hk={hk}: out rel={r_out:.3e} state rel={r_st:.3e}");
        let tol =
            GEMM_VS_SEQUENTIAL_REL_TOL_1_6E2_FOUR_BF16_EPS_THE_GEMM_INPUTS_ARE_BF16_TENSOR_CORE_ROUNDED_WHERE_THE_SCAN_READS_F32;
        assert!(
            r_out <= tol && r_st <= tol,
            "v3 split-ladder tail diverged from the sequential scan beyond the bf16-input \
             rounding class (t={t} h={h} hk={hk}): out rel {r_out:.3e} state rel {r_st:.3e} tol \
             {tol:.1e}; a checkpoint-index or staging defect reads orders of magnitude above \
             this, and error GROWING with t means per-sub-chunk drift compounding through the \
             resident state whose bf16 checkpoint image is the same rounding class the mma tail \
             already feeds its fragments"
        );
        if r_out > worst.0 {
            worst = (r_out, t);
        }
        let nonzero = out_v3.iter().filter(|w| **w != 0).count();
        assert!(
            nonzero * 4 > out_v3.len(),
            "v3 output mostly zero ({nonzero}/{}), a zeroed buffer would pass a tolerance gate",
            out_v3.len()
        );
    }
    eprintln!("v3 parity worst out rel={:.3e} at t={}", worst.0, worst.1);
}

const GRAPH_ENV_NV_Q38_GDN_CHUNK_GEMM_GRAPH_REPLAYS_THE_CAPTURED_TAIL_IN_ONE_LAUNCH: &str =
    "NV_Q38_GDN_CHUNK_GEMM_GRAPH";

#[test]
#[ignore = "cuda bitwise gate; set NV_GDN_CHUNK_GEMM_TEST=1 -- the captured graph replay must reproduce the uncaptured chain bit for bit"]
fn gdn_chunk_gemm_graph_replay_is_bitwise_vs_the_uncaptured_chain() {
    if !parity_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_TEST is not 1, so this bitwise gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having compared no bits. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    std::env::remove_var(GRAPH_ENV_NV_Q38_GDN_CHUNK_GEMM_GRAPH_REPLAYS_THE_CAPTURED_TAIL_IN_ONE_LAUNCH);
    let stream = stream_or_panic("gdn_chunk_gemm_graph_replay_is_bitwise");
    for &(t, h, hk) in &[
        (257usize, 48usize, 16usize),
        (SERVING_CHUNK_T_4096_THE_NV_Q38_PREFILL_CHUNK_DEFAULT, 48, 16),
    ] {
        let fx = fixture(&stream, t, h, hk, 0x6a4f_0000 + t as u64);
        let mut ws = workspace(&stream, &fx);
        let mut b_ref = bufs(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b_ref);
        run_chunk_gemm(&stream, &fx, &mut b_ref, &mut ws);
        stream.synchronize().unwrap();
        let out_ref = common::dtoh_u16(&stream, &b_ref.d_core);
        let st_ref: Vec<u32> = dtoh_f32(&stream, &b_ref.d_state)
            .iter()
            .map(|v| v.to_bits())
            .collect();

        std::env::set_var(
            GRAPH_ENV_NV_Q38_GDN_CHUNK_GEMM_GRAPH_REPLAYS_THE_CAPTURED_TAIL_IN_ONE_LAUNCH,
            "1",
        );
        let replays_before = cuda::gdn_chunk_gemm_graph_replays_total();
        for pass in 0..3 {
            let mut b = bufs(&stream, &fx);
            run_l2norm(&stream, &fx, &mut b);
            run_chunk_gemm(&stream, &fx, &mut b, &mut ws);
            stream.synchronize().unwrap();
            let out = common::dtoh_u16(&stream, &b.d_core);
            let st: Vec<u32> = dtoh_f32(&stream, &b.d_state)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let out_diffs = out.iter().zip(&out_ref).filter(|(a, b)| a != b).count();
            let st_diffs = st.iter().zip(&st_ref).filter(|(a, b)| a != b).count();
            assert!(
                out_diffs == 0 && st_diffs == 0,
                "graph pass {pass} at t={t} moved bits vs the uncaptured chain (out {out_diffs}/{}, \
                 state {st_diffs}/{}): replay runs the exact captured kernels on the exact cached \
                 workspace, so any bit delta is a stale-pointer or staging defect, not rounding",
                out.len(),
                st.len()
            );
        }
        let replays = cuda::gdn_chunk_gemm_graph_replays_total() - replays_before;
        std::env::remove_var(
            GRAPH_ENV_NV_Q38_GDN_CHUNK_GEMM_GRAPH_REPLAYS_THE_CAPTURED_TAIL_IN_ONE_LAUNCH,
        );
        assert_eq!(
            replays, 2,
            "t={t}: pass 0 must warm uncaptured and passes 1-2 must replay the graph; 0 replays \
             means capture silently fell back and the bitwise gate above compared the uncaptured \
             chain against itself"
        );
        eprintln!("chunk-gemm graph bitwise t={t} h={h} hk={hk}: 3 armed passes, {replays} replays, all bits equal");
    }
}

#[test]
#[ignore = "cuda refusal gate; set NV_GDN_CHUNK_GEMM_TEST=1 -- the chunked WY/UT wrappers must decline bad shapes and a short workspace instead of launching"]
fn gdn_chunk_gemm_rejects_bad_shapes_and_short_workspace() {
    if !parity_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_TEST is not 1, so this refusal gate executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having checked no refusal. A suite whose precondition is absent \
             must fail or be ignored, never return green."
        );
    }
    let stream = stream_or_panic("gdn_chunk_gemm_rejects_bad_shapes");
    let fx = fixture(&stream, 8, 48, 16, 0xbad0_0001);
    let mut b = bufs(&stream, &fx);
    let mut ws = workspace(&stream, &fx);
    run_l2norm(&stream, &fx, &mut b);
    let raw = stream.cu_stream() as *mut c_void;
    let call = |b: &mut Bufs, ws: &mut CudaSlice<f32>, batch: i32, k: i32, hk: i32, ws_len: i64| {
        let (pq, _1) = b.d_qn.device_ptr(&stream);
        let (pk, _2) = b.d_kn.device_ptr(&stream);
        let (pm, _3) = fx.d_mixed.device_ptr(&stream);
        let (pg, _4) = fx.d_g.device_ptr(&stream);
        let (pb, _5) = fx.d_beta.device_ptr(&stream);
        let (ps, _6) = b.d_state.device_ptr_mut(&stream);
        let (pc, _7) = b.d_core.device_ptr_mut(&stream);
        let (pw, _8) = ws.device_ptr_mut(&stream);
        unsafe {
            cuda::gdn_chunk_gemm_scan_f32(
                raw,
                pq as *const f32,
                pk as *const f32,
                pm as *const u16,
                pg as *const f32,
                pb as *const f32,
                ps as *mut f32,
                pc as *mut u16,
                pw as *mut f32,
                ws_len,
                batch,
                fx.t as i32,
                fx.h as i32,
                hk,
                k,
                128,
                fx.conv_dim as i32,
                (2 * fx.key_dim) as i32,
            )
        }
    };
    let full = ws.len() as i64;
    assert_eq!(call(&mut b, &mut ws, 1, 64, 16, full), -2, "k!=128");
    assert_eq!(call(&mut b, &mut ws, 1, 128, 5, full), -3, "h%hk");
    assert_eq!(call(&mut b, &mut ws, 2, 128, 16, full), -4, "b!=1");
    assert_eq!(call(&mut b, &mut ws, 1, 128, 16, full - 1), -5, "short ws");
    assert_eq!(call(&mut b, &mut ws, 1, 128, 16, full), 0, "valid");
    stream.synchronize().unwrap();
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
#[ignore = "measurement ladder, GPU-time only; set NV_GDN_CHUNK_GEMM_BENCH=1 -- prices the chunked WY/UT scan stage at the q38 serving shape"]
fn gdn_chunk_gemm_stage_microbench_at_q38_serving_shape() {
    if !bench_gated() {
        panic!(
            "NV_GDN_CHUNK_GEMM_BENCH is not 1, so this microbench executed nothing. It is now \
             #[ignore]d, which makes --ignored the opt-in; before that it printed `1 passed` on \
             every default run having measured nothing. A suite whose precondition is absent must \
             fail or be ignored, never return green."
        );
    }
    let stream = stream_or_panic("gdn_chunk_gemm_stage_microbench");
    let (h, hk) = SERVING_SHAPE_Q38_H48_HK16_D128;
    let warmup = 3usize;
    let reps = 10usize;
    for t in [512usize, 1024, 2048, SERVING_CHUNK_T_4096_THE_NV_Q38_PREFILL_CHUNK_DEFAULT] {
        let fx = fixture(&stream, t, h, hk, 0xbe4c_0001);
        let mut b = bufs(&stream, &fx);
        let mut ws = workspace(&stream, &fx);
        run_l2norm(&stream, &fx, &mut b);
        let v1_ms = wall_ms(&stream, || run_scan_v1(&stream, &fx, &mut b), warmup, reps);
        let v2_ms = wall_ms(&stream, || run_scan_v2_shfl(&stream, &fx, &mut b), warmup, reps);
        let gemm_ms = wall_ms(
            &stream,
            || run_chunk_gemm(&stream, &fx, &mut b, &mut ws),
            warmup,
            reps,
        );
        std::env::set_var(
            GRAPH_ENV_NV_Q38_GDN_CHUNK_GEMM_GRAPH_REPLAYS_THE_CAPTURED_TAIL_IN_ONE_LAUNCH,
            "1",
        );
        let graph_ms = wall_ms(
            &stream,
            || run_chunk_gemm(&stream, &fx, &mut b, &mut ws),
            warmup,
            reps,
        );
        std::env::remove_var(
            GRAPH_ENV_NV_Q38_GDN_CHUNK_GEMM_GRAPH_REPLAYS_THE_CAPTURED_TAIL_IN_ONE_LAUNCH,
        );
        let fwdh_ms = wall_ms(
            &stream,
            || run_chunk_gemm_fwdh(&stream, &fx, &mut b, &mut ws),
            warmup,
            reps,
        );
        let mma_ms = wall_ms(
            &stream,
            || run_chunk_gemm_mma(&stream, &fx, &mut b, &mut ws),
            warmup,
            reps,
        );
        let v3_ms = wall_ms(
            &stream,
            || run_chunk_gemm_v3(&stream, &fx, &mut b, &mut ws),
            warmup,
            reps,
        );
        eprintln!(
            "chunk-gemm stage-bench t={t} scan_v1={v1_ms:.3}ms scan_v2_shfl={v2_ms:.3}ms \
             gemm={gemm_ms:.3}ms gemm_graph={graph_ms:.3}ms fwdh={fwdh_ms:.3}ms mma={mma_ms:.3}ms \
             v3={v3_ms:.3}ms speedup_vs_v1={:.2}x speedup_vs_v2={:.2}x graph_vs_v2={:.2}x \
             fwdh_vs_v2={:.2}x mma_vs_v2={:.2}x v3_vs_v2={:.2}x v3_vs_mma={:.2}x ws_mb={:.0}",
            v1_ms / gemm_ms,
            v2_ms / gemm_ms,
            v2_ms / graph_ms,
            v2_ms / fwdh_ms,
            v2_ms / mma_ms,
            v2_ms / v3_ms,
            mma_ms / v3_ms,
            ws.len() as f64 * 4.0 / 1e6
        );
    }
}
