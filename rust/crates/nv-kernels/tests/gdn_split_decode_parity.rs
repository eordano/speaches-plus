#![cfg(feature = "cuda")]

mod common;
use common::dtoh_u16;
use common::htod_f32;
use common::htod_u16;
use common::lcg_unit_f32 as lcg;
use common::rand_bf16;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const STEPS_4_ACCUMULATES_STATE_DRIFT_ACROSS_SEQUENTIAL_TOKENS: usize = 4;
const OUT_REL_L2_TOL_REDUCTION_ORDER_CHANGES_UNDER_THE_8_WAY_K_SPLIT: f32 = 2e-3;
const STATE_REL_L2_TOL_FP32_STATE_DIFFERS_ONLY_BY_DELTA_ULPS: f32 = 1e-3;

fn rand_f32(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| lo + lcg(seed) * (hi - lo)).collect()
}

fn dtoh_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    #[allow(deprecated)]
    let v = stream.memcpy_dtov(d).unwrap();
    v
}

fn bf16_f32(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|b| half::bf16::from_bits(*b).to_f32())
        .collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt() as f32
}

#[allow(clippy::too_many_arguments)]
fn step_ref(
    raw: *mut c_void,
    stream: &Arc<CudaStream>,
    mixed: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    norm_w: &CudaSlice<u16>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n_k: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
) {
    let rc = unsafe {
        let (mp, _g1) = mixed.device_ptr(stream);
        let (zp, _g2) = z.device_ptr(stream);
        let (ap, _g3) = a.device_ptr(stream);
        let (bp, _g4) = b.device_ptr(stream);
        let (alp, _g5) = a_log.device_ptr(stream);
        let (dtp, _g6) = dt_bias.device_ptr(stream);
        let (nwp, _g7) = norm_w.device_ptr(stream);
        let (sp, _g8) = state.device_ptr_mut(stream);
        let (op, _g9) = out.device_ptr_mut(stream);
        cuda::gdn_decode_step_bf16(
            raw,
            mp as *const u16,
            zp as *const u16,
            ap as *const u16,
            bp as *const u16,
            alp as *const u16,
            dtp as *const u16,
            nwp as *const u16,
            sp as *mut f32,
            op as *mut u16,
            n_k as i32,
            n_v as i32,
            d_k as i32,
            d_v as i32,
            1e-6,
        )
    };
    assert_eq!(rc, 0, "gdn_decode_step_bf16 rc={rc}");
}

#[allow(clippy::too_many_arguments)]
fn step_split(
    raw: *mut c_void,
    stream: &Arc<CudaStream>,
    mixed: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    norm_w: &CudaSlice<u16>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n_k: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
) -> i32 {
    let mut qn: CudaSlice<f32> = stream.alloc_zeros(n_k * d_k).unwrap();
    let mut kn: CudaSlice<f32> = stream.alloc_zeros(n_k * d_k).unwrap();
    let mut ge: CudaSlice<f32> = stream.alloc_zeros(n_v).unwrap();
    let mut be: CudaSlice<f32> = stream.alloc_zeros(n_v).unwrap();
    let mut core: CudaSlice<u16> = stream.alloc_zeros(n_v * d_v).unwrap();
    unsafe {
        let (mp, _g1) = mixed.device_ptr(stream);
        let (zp, _g2) = z.device_ptr(stream);
        let (ap, _g3) = a.device_ptr(stream);
        let (bp, _g4) = b.device_ptr(stream);
        let (alp, _g5) = a_log.device_ptr(stream);
        let (dtp, _g6) = dt_bias.device_ptr(stream);
        let (nwp, _g7) = norm_w.device_ptr(stream);
        let (sp, _g8) = state.device_ptr_mut(stream);
        let (op, _g9) = out.device_ptr_mut(stream);
        let (qnp, _g10) = qn.device_ptr_mut(stream);
        let (knp, _g11) = kn.device_ptr_mut(stream);
        let (gep, _g12) = ge.device_ptr_mut(stream);
        let (bep, _g13) = be.device_ptr_mut(stream);
        let (cop, _g14) = core.device_ptr_mut(stream);
        cuda::gdn_decode_step_split_bf16(
            raw,
            mp as *const u16,
            zp as *const u16,
            ap as *const u16,
            bp as *const u16,
            alp as *const u16,
            dtp as *const u16,
            nwp as *const u16,
            sp as *mut f32,
            op as *mut u16,
            qnp as *mut f32,
            knp as *mut f32,
            gep as *mut f32,
            bep as *mut f32,
            cop as *mut u16,
            n_k as i32,
            n_v as i32,
            d_k as i32,
            d_v as i32,
            1e-6,
        )
    }
}

fn run_case(name: &str, n_k: usize, n_v: usize, d_k: usize, d_v: usize) {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("[{name}] skip: no CUDA device");
        return;
    };
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let mixed_len = 2 * key_dim + value_dim;
    let mut seed = 0x243f6a8885a308d3u64 ^ ((n_v * d_k) as u64);

    let a_log_host = rand_bf16(&mut seed, n_v, -1.0, 1.0);
    let dt_bias_host = rand_bf16(&mut seed, n_v, -0.5, 0.5);
    let norm_w_host = rand_bf16(&mut seed, d_v, 0.5, 1.5);
    let state_host = rand_f32(&mut seed, n_v * d_k * d_v, -0.5, 0.5);

    let a_log = htod_u16(&stream, &a_log_host);
    let dt_bias = htod_u16(&stream, &dt_bias_host);
    let norm_w = htod_u16(&stream, &norm_w_host);
    let mut state_ref = htod_f32(&stream, &state_host);
    let mut state_split = htod_f32(&stream, &state_host);
    let mut out_ref: CudaSlice<u16> = stream.alloc_zeros(value_dim).unwrap();
    let mut out_split: CudaSlice<u16> = stream.alloc_zeros(value_dim).unwrap();

    for step in 0..STEPS_4_ACCUMULATES_STATE_DRIFT_ACROSS_SEQUENTIAL_TOKENS {
        let mixed_host = rand_bf16(&mut seed, mixed_len, -1.0, 1.0);
        let z_host = rand_bf16(&mut seed, value_dim, -2.0, 2.0);
        let a_host = rand_bf16(&mut seed, n_v, -1.0, 1.0);
        let b_host = rand_bf16(&mut seed, n_v, -1.0, 1.0);
        let mixed = htod_u16(&stream, &mixed_host);
        let z = htod_u16(&stream, &z_host);
        let a = htod_u16(&stream, &a_host);
        let b = htod_u16(&stream, &b_host);

        step_ref(
            raw, &stream, &mixed, &z, &a, &b, &a_log, &dt_bias, &norm_w, &mut state_ref,
            &mut out_ref, n_k, n_v, d_k, d_v,
        );
        let rc = step_split(
            raw, &stream, &mixed, &z, &a, &b, &a_log, &dt_bias, &norm_w, &mut state_split,
            &mut out_split, n_k, n_v, d_k, d_v,
        );
        assert_eq!(rc, 0, "[{name}] gdn_decode_step_split_bf16 rc={rc}");
        stream.synchronize().unwrap();

        let o_ref = bf16_f32(&dtoh_u16(&stream, &out_ref));
        let o_split = bf16_f32(&dtoh_u16(&stream, &out_split));
        let ref_max = o_ref.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            ref_max > 1e-4,
            "[{name}] step {step}: reference output is numerically dead (max {ref_max:.3e})"
        );
        let err = rel_l2(&o_split, &o_ref);
        eprintln!("[{name}] step {step} out rel_l2(split vs step)={err:.3e}");
        assert!(
            err < OUT_REL_L2_TOL_REDUCTION_ORDER_CHANGES_UNDER_THE_8_WAY_K_SPLIT,
            "[{name}] step {step}: out diverged rel_l2={err:.3e}"
        );
        let idx_ref = o_ref
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
            .unwrap()
            .0;
        let split_max = o_split.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            o_split[idx_ref].abs() >= 0.999 * split_max,
            "[{name}] step {step}: argmax moved (ref idx {idx_ref})"
        );
    }

    let s_ref = dtoh_f32(&stream, &state_ref);
    let s_split = dtoh_f32(&stream, &state_split);
    let s_max = s_ref.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        s_max > 1e-4,
        "[{name}] final state is numerically dead (max {s_max:.3e})"
    );
    let s_err = rel_l2(&s_split, &s_ref);
    eprintln!("[{name}] final state rel_l2(split vs step)={s_err:.3e}");
    assert!(
        s_err < STATE_REL_L2_TOL_FP32_STATE_DIFFERS_ONLY_BY_DELTA_ULPS,
        "[{name}] state diverged rel_l2={s_err:.3e}"
    );
}

#[test]
fn gdn_split_decode_tiny_shape() {
    run_case("tiny", 2, 4, 32, 32);
}

#[test]
fn gdn_split_decode_tiny_v_per_k_3() {
    run_case("tiny_vpk3", 2, 6, 32, 32);
}

#[test]
fn gdn_split_decode_qwen38_shape_48v_16k_128() {
    run_case("qwen38", 16, 48, 128, 128);
}

#[test]
fn gdn_split_decode_unsupported_geometry_returns_fallback_sentinel() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("[fallback] skip: no CUDA device");
        return;
    };
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let (n_k, n_v, d_k, d_v) = (2usize, 4usize, 64usize, 64usize);
    let mut seed = 7u64;
    let mixed = htod_u16(
        &stream,
        &rand_bf16(&mut seed, 2 * n_k * d_k + n_v * d_v, -1.0, 1.0),
    );
    let z = htod_u16(&stream, &rand_bf16(&mut seed, n_v * d_v, -1.0, 1.0));
    let a = htod_u16(&stream, &rand_bf16(&mut seed, n_v, -1.0, 1.0));
    let b = htod_u16(&stream, &rand_bf16(&mut seed, n_v, -1.0, 1.0));
    let a_log = htod_u16(&stream, &rand_bf16(&mut seed, n_v, -1.0, 1.0));
    let dt_bias = htod_u16(&stream, &rand_bf16(&mut seed, n_v, -1.0, 1.0));
    let norm_w = htod_u16(&stream, &rand_bf16(&mut seed, d_v, 0.5, 1.5));
    let mut state = htod_f32(&stream, &rand_f32(&mut seed, n_v * d_k * d_v, -0.5, 0.5));
    let mut out: CudaSlice<u16> = stream.alloc_zeros(n_v * d_v).unwrap();
    let rc = step_split(
        raw, &stream, &mixed, &z, &a, &b, &a_log, &dt_bias, &norm_w, &mut state, &mut out,
        n_k, n_v, d_k, d_v,
    );
    assert_eq!(
        rc, -1,
        "d_k=64 has no split template; the launcher must return the fallback sentinel"
    );
}
