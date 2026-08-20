#![cfg(feature = "cuda")]

mod common;
use common::assert_u16_bits;
use common::dtoh_u16;
use common::htod_f32;
use common::htod_u16;
use common::lcg_unit_f32 as lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const T_CASES: [usize; 8] = [1, 2, 3, 4, 5, 8, 9, 16];
const DECODE_SHAPES: [(usize, usize, usize, usize); 3] =
    [(2, 4, 32, 32), (16, 32, 128, 128), (4, 8, 64, 96)];
const CONV_DIMS: [usize; 2] = [300, 8192];
const CONV_KS: [usize; 2] = [2, 4];

fn gated() -> bool {
    std::env::var("NV_GDN_CHUNK_TEST").as_deref() == Ok("1")
}

fn bench_gated() -> bool {
    std::env::var("NV_GDN_CHUNK_BENCH").as_deref() == Ok("1")
}

fn stream_or_skip(test: &str) -> Option<Arc<CudaStream>> {
    if !gated() {
        eprintln!("{test}: SKIP set NV_GDN_CHUNK_TEST=1 to run");
        return None;
    }
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => panic!("{test}: NV_GDN_CHUNK_TEST=1 set but no CUDA device 0: {e}"),
    }
}

fn rand_bf16(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(lo + lcg(seed) * (hi - lo)).to_bits())
        .collect()
}

fn rand_f32_nonzero(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let mut v = (lcg(seed) * 2.0 - 1.0) * scale;
            if v == 0.0 {
                v = 0.03125 * scale;
            }
            v
        })
        .collect()
}

fn assert_f32_bits(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    let mut diff = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            diff += 1;
            if first.is_none() {
                first = Some((i, *x, x.to_bits(), *y, y.to_bits()));
            }
        }
    }
    assert_eq!(
        diff,
        0,
        "{name}: {diff}/{} f32 words differ, first {first:?}",
        a.len()
    );
}

fn dtoh_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    #[allow(deprecated)]
    let v = stream.memcpy_dtov(d).unwrap();
    v
}

#[test]
#[ignore]
fn gdn_conv_chunk_matches_sequential() {
    let Some(stream) = stream_or_skip("gdn_conv_chunk_matches_sequential") else {
        return;
    };
    let raw = stream.cu_stream() as *mut c_void;
    for conv_dim in CONV_DIMS {
        for k in CONV_KS {
            for t in T_CASES {
                let start = std::time::Instant::now();
                let mut seed = 0x51ed_2701_u64 ^ ((conv_dim * 131 + k * 17 + t) as u64);
                let x_seq = rand_bf16(&mut seed, t * conv_dim, -1.0, 1.0);
                let conv0 = rand_bf16(&mut seed, conv_dim * (k - 1), -1.0, 1.0);
                let w = rand_bf16(&mut seed, conv_dim * k, -0.5, 0.5);
                assert!(
                    conv0.iter().any(|v| bf16::from_bits(*v).to_f32() != 0.0),
                    "conv fixture must seed a non-zero window"
                );

                let d_x = htod_u16(&stream, &x_seq);
                let d_conv = htod_u16(&stream, &conv0);
                let d_w = htod_u16(&stream, &w);
                let mut d_y: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * conv_dim).unwrap();
                let mut d_ckpt: CudaSlice<u16> =
                    stream.alloc_zeros::<u16>(t * conv_dim * (k - 1)).unwrap();
                let rc = {
                    let (px, _1) = d_x.device_ptr(&stream);
                    let (pc, _2) = d_conv.device_ptr(&stream);
                    let (pw, _3) = d_w.device_ptr(&stream);
                    let (py, _4) = d_y.device_ptr_mut(&stream);
                    let (pk, _5) = d_ckpt.device_ptr_mut(&stream);
                    unsafe {
                        cuda::gdn_conv_decode_chunk_silu_bf16(
                            raw,
                            px as *const u16,
                            pc as *const u16,
                            pw as *const u16,
                            py as *mut u16,
                            pk as *mut u16,
                            conv_dim as i32,
                            k as i32,
                            t as i32,
                        )
                    }
                };
                assert_eq!(rc, 0, "chunk conv rc={rc} C={conv_dim} k={k} t={t}");
                stream.synchronize().unwrap();
                let y_a = dtoh_u16(&stream, &d_y);
                let ckpt_a = dtoh_u16(&stream, &d_ckpt);
                let conv_after = dtoh_u16(&stream, &d_conv);
                assert_u16_bits(
                    &format!("conv_state untouched C={conv_dim} k={k} t={t}"),
                    &conv_after,
                    &conv0,
                );

                let mut d_conv_b = htod_u16(&stream, &conv0);
                let mut y_b = vec![0u16; t * conv_dim];
                let mut ckpt_b = vec![0u16; t * conv_dim * (k - 1)];
                for i in 0..t {
                    let d_xi = htod_u16(&stream, &x_seq[i * conv_dim..(i + 1) * conv_dim]);
                    let mut d_yi: CudaSlice<u16> = stream.alloc_zeros::<u16>(conv_dim).unwrap();
                    let rc = {
                        let (px, _1) = d_xi.device_ptr(&stream);
                        let (pc, _2) = d_conv_b.device_ptr_mut(&stream);
                        let (pw, _3) = d_w.device_ptr(&stream);
                        let (py, _4) = d_yi.device_ptr_mut(&stream);
                        unsafe {
                            cuda::gdn_conv_decode_silu_bf16(
                                raw,
                                px as *const u16,
                                pc as *mut u16,
                                pw as *const u16,
                                py as *mut u16,
                                conv_dim as i32,
                                k as i32,
                            )
                        }
                    };
                    assert_eq!(rc, 0, "t==1 conv rc={rc} C={conv_dim} k={k} i={i}");
                    stream.synchronize().unwrap();
                    y_b[i * conv_dim..(i + 1) * conv_dim]
                        .copy_from_slice(&dtoh_u16(&stream, &d_yi));
                    ckpt_b[i * conv_dim * (k - 1)..(i + 1) * conv_dim * (k - 1)]
                        .copy_from_slice(&dtoh_u16(&stream, &d_conv_b));
                }

                assert_u16_bits(&format!("conv y C={conv_dim} k={k} t={t}"), &y_a, &y_b);
                assert_u16_bits(
                    &format!("conv ckpt C={conv_dim} k={k} t={t}"),
                    &ckpt_a,
                    &ckpt_b,
                );
                eprintln!(
                    "conv chunk parity C={conv_dim} k={k} t={t}: bit-exact, {:?}",
                    start.elapsed()
                );
            }
        }
    }
}

struct DecodeFixture {
    mixed: Vec<u16>,
    z: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    norm_w: Vec<u16>,
    state0: Vec<f32>,
}

fn decode_fixture(n_k: usize, n_v: usize, d_k: usize, d_v: usize, t: usize) -> DecodeFixture {
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let mixed_stride = 2 * key_dim + value_dim;
    let mut seed = 0x9e37_79b9_u64 ^ ((n_k * 1009 + n_v * 131 + d_k * 17 + d_v * 7 + t) as u64);
    let mixed = rand_bf16(&mut seed, t * mixed_stride, -1.0, 1.0);
    let z = rand_bf16(&mut seed, t * value_dim, -2.0, 2.0);
    let a = rand_bf16(&mut seed, t * n_v, -2.0, 2.0);
    let b = rand_bf16(&mut seed, t * n_v, -3.0, 3.0);
    let a_log = rand_bf16(&mut seed, n_v, -3.0, 0.5);
    let dt_bias = rand_bf16(&mut seed, n_v, -1.0, 1.0);
    let norm_w = rand_bf16(&mut seed, d_v, 0.5, 1.5);
    let state0 = rand_f32_nonzero(&mut seed, n_v * d_k * d_v, 0.5);
    DecodeFixture {
        mixed,
        z,
        a,
        b,
        a_log,
        dt_bias,
        norm_w,
        state0,
    }
}

#[test]
#[ignore]
fn gdn_decode_chunk_matches_sequential() {
    let Some(stream) = stream_or_skip("gdn_decode_chunk_matches_sequential") else {
        return;
    };
    let raw = stream.cu_stream() as *mut c_void;
    for (n_k, n_v, d_k, d_v) in DECODE_SHAPES {
        for t in T_CASES {
            let start = std::time::Instant::now();
            let key_dim = n_k * d_k;
            let value_dim = n_v * d_v;
            let mixed_stride = 2 * key_dim + value_dim;
            let state_len = n_v * d_k * d_v;
            let fx = decode_fixture(n_k, n_v, d_k, d_v, t);
            assert!(
                fx.state0.iter().all(|v| *v != 0.0),
                "decode fixture must seed a fully non-zero state"
            );

            let d_mixed = htod_u16(&stream, &fx.mixed);
            let d_z = htod_u16(&stream, &fx.z);
            let d_a = htod_u16(&stream, &fx.a);
            let d_b = htod_u16(&stream, &fx.b);
            let d_alog = htod_u16(&stream, &fx.a_log);
            let d_dt = htod_u16(&stream, &fx.dt_bias);
            let d_nw = htod_u16(&stream, &fx.norm_w);
            let d_state_in = htod_f32(&stream, &fx.state0);
            let mut d_ckpt: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * state_len).unwrap();
            let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * value_dim).unwrap();
            let rc = {
                let (pm, _1) = d_mixed.device_ptr(&stream);
                let (pz, _2) = d_z.device_ptr(&stream);
                let (pa, _3) = d_a.device_ptr(&stream);
                let (pb, _4) = d_b.device_ptr(&stream);
                let (pl, _5) = d_alog.device_ptr(&stream);
                let (pd, _6) = d_dt.device_ptr(&stream);
                let (pn, _7) = d_nw.device_ptr(&stream);
                let (ps, _8) = d_state_in.device_ptr(&stream);
                let (pc, _9) = d_ckpt.device_ptr_mut(&stream);
                let (po, _10) = d_out.device_ptr_mut(&stream);
                unsafe {
                    cuda::gdn_decode_chunk_bf16(
                        raw,
                        pm as *const u16,
                        pz as *const u16,
                        pa as *const u16,
                        pb as *const u16,
                        pl as *const u16,
                        pd as *const u16,
                        pn as *const u16,
                        ps as *const f32,
                        pc as *mut f32,
                        po as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        1e-6f32,
                        t as i32,
                    )
                }
            };
            assert_eq!(
                rc, 0,
                "chunk decode rc={rc} shape=({n_k},{n_v},{d_k},{d_v}) t={t}"
            );
            stream.synchronize().unwrap();
            let out_a = dtoh_u16(&stream, &d_out);
            let ckpt_a = dtoh_f32(&stream, &d_ckpt);
            let state_after = dtoh_f32(&stream, &d_state_in);
            assert_f32_bits(
                &format!("state_in untouched ({n_k},{n_v},{d_k},{d_v}) t={t}"),
                &state_after,
                &fx.state0,
            );

            let mut d_state_b = htod_f32(&stream, &fx.state0);
            let mut out_b = vec![0u16; t * value_dim];
            let mut ckpt_b = vec![0f32; t * state_len];
            for i in 0..t {
                let d_mi = htod_u16(&stream, &fx.mixed[i * mixed_stride..(i + 1) * mixed_stride]);
                let d_zi = htod_u16(&stream, &fx.z[i * value_dim..(i + 1) * value_dim]);
                let d_ai = htod_u16(&stream, &fx.a[i * n_v..(i + 1) * n_v]);
                let d_bi = htod_u16(&stream, &fx.b[i * n_v..(i + 1) * n_v]);
                let mut d_oi: CudaSlice<u16> = stream.alloc_zeros::<u16>(value_dim).unwrap();
                let rc = {
                    let (pm, _1) = d_mi.device_ptr(&stream);
                    let (pz, _2) = d_zi.device_ptr(&stream);
                    let (pa, _3) = d_ai.device_ptr(&stream);
                    let (pb, _4) = d_bi.device_ptr(&stream);
                    let (pl, _5) = d_alog.device_ptr(&stream);
                    let (pd, _6) = d_dt.device_ptr(&stream);
                    let (pn, _7) = d_nw.device_ptr(&stream);
                    let (ps, _8) = d_state_b.device_ptr_mut(&stream);
                    let (po, _9) = d_oi.device_ptr_mut(&stream);
                    unsafe {
                        cuda::gdn_decode_step_bf16(
                            raw,
                            pm as *const u16,
                            pz as *const u16,
                            pa as *const u16,
                            pb as *const u16,
                            pl as *const u16,
                            pd as *const u16,
                            pn as *const u16,
                            ps as *mut f32,
                            po as *mut u16,
                            n_k as i32,
                            n_v as i32,
                            d_k as i32,
                            d_v as i32,
                            1e-6f32,
                        )
                    }
                };
                assert_eq!(
                    rc, 0,
                    "t==1 decode rc={rc} shape=({n_k},{n_v},{d_k},{d_v}) i={i}"
                );
                stream.synchronize().unwrap();
                out_b[i * value_dim..(i + 1) * value_dim]
                    .copy_from_slice(&dtoh_u16(&stream, &d_oi));
                ckpt_b[i * state_len..(i + 1) * state_len]
                    .copy_from_slice(&dtoh_f32(&stream, &d_state_b));
            }

            for i in 0..t {
                assert_u16_bits(
                    &format!("decode out ({n_k},{n_v},{d_k},{d_v}) t={t} token={i}"),
                    &out_a[i * value_dim..(i + 1) * value_dim],
                    &out_b[i * value_dim..(i + 1) * value_dim],
                );
                assert_f32_bits(
                    &format!("decode ckpt ({n_k},{n_v},{d_k},{d_v}) t={t} token={i}"),
                    &ckpt_a[i * state_len..(i + 1) * state_len],
                    &ckpt_b[i * state_len..(i + 1) * state_len],
                );
            }
            let nonzero = out_a.iter().filter(|w| **w != 0).count();
            assert!(
                nonzero * 2 > out_a.len(),
                "decode chunk output mostly zero: {nonzero}/{}",
                out_a.len()
            );
            eprintln!(
                "decode chunk parity ({n_k},{n_v},{d_k},{d_v}) t={t}: bit-exact, {:?}",
                start.elapsed()
            );
        }
    }
}

#[test]
#[ignore]
fn gdn_chunk_wrappers_reject_bad_shapes() {
    let Some(stream) = stream_or_skip("gdn_chunk_wrappers_reject_bad_shapes") else {
        return;
    };
    let raw = stream.cu_stream() as *mut c_void;
    let conv_dim = 64usize;
    let k = 4usize;
    let t = 2usize;
    let mut seed = 0xc0ff_ee11_u64;
    let x_seq = rand_bf16(&mut seed, 17 * conv_dim, -1.0, 1.0);
    let conv0 = rand_bf16(&mut seed, conv_dim * 8, -1.0, 1.0);
    let w = rand_bf16(&mut seed, conv_dim * 9, -0.5, 0.5);
    let d_x = htod_u16(&stream, &x_seq);
    let d_conv = htod_u16(&stream, &conv0);
    let d_w = htod_u16(&stream, &w);
    let mut d_y: CudaSlice<u16> = stream.alloc_zeros::<u16>(17 * conv_dim).unwrap();
    let mut d_ckpt: CudaSlice<u16> = stream.alloc_zeros::<u16>(17 * conv_dim * 8).unwrap();
    let conv_rc =
        |conv_dim: i32, k: i32, t: i32, d_y: &mut CudaSlice<u16>, d_ckpt: &mut CudaSlice<u16>| {
            let (px, _1) = d_x.device_ptr(&stream);
            let (pc, _2) = d_conv.device_ptr(&stream);
            let (pw, _3) = d_w.device_ptr(&stream);
            let (py, _4) = d_y.device_ptr_mut(&stream);
            let (pk, _5) = d_ckpt.device_ptr_mut(&stream);
            unsafe {
                cuda::gdn_conv_decode_chunk_silu_bf16(
                    raw,
                    px as *const u16,
                    pc as *const u16,
                    pw as *const u16,
                    py as *mut u16,
                    pk as *mut u16,
                    conv_dim,
                    k,
                    t,
                )
            }
        };
    let c = conv_dim as i32;
    assert_eq!(conv_rc(c, 9, t as i32, &mut d_y, &mut d_ckpt), -2, "k>8");
    assert_eq!(conv_rc(c, 1, t as i32, &mut d_y, &mut d_ckpt), -2, "k<2");
    assert_eq!(conv_rc(c, k as i32, 0, &mut d_y, &mut d_ckpt), -2, "t=0");
    assert_eq!(conv_rc(c, k as i32, 17, &mut d_y, &mut d_ckpt), -2, "t>16");
    assert_eq!(
        conv_rc(0, 99, t as i32, &mut d_y, &mut d_ckpt),
        -2,
        "conv_dim=0 does not mask bad k"
    );
    assert_eq!(
        conv_rc(0, k as i32, t as i32, &mut d_y, &mut d_ckpt),
        0,
        "conv_dim=0 valid dims noop"
    );
    assert_eq!(
        conv_rc(c, k as i32, t as i32, &mut d_y, &mut d_ckpt),
        0,
        "valid"
    );
    stream.synchronize().unwrap();

    let (n_k, n_v, d_k, d_v) = (2usize, 4usize, 32usize, 32usize);
    let fx = decode_fixture(n_k, n_v, d_k, d_v, t);
    let d_mixed = htod_u16(&stream, &fx.mixed);
    let d_z = htod_u16(&stream, &fx.z);
    let d_a = htod_u16(&stream, &fx.a);
    let d_b = htod_u16(&stream, &fx.b);
    let d_alog = htod_u16(&stream, &fx.a_log);
    let d_dt = htod_u16(&stream, &fx.dt_bias);
    let d_nw = htod_u16(&stream, &fx.norm_w);
    let d_state = htod_f32(&stream, &fx.state0);
    let mut d_ck: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * n_v * d_k * d_v).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * n_v * d_v).unwrap();
    let decode_rc = |n_k_arg: i32,
                     n_v_arg: i32,
                     d_k_arg: i32,
                     d_v_arg: i32,
                     t: i32,
                     d_ck: &mut CudaSlice<f32>,
                     d_out: &mut CudaSlice<u16>| {
        let (pm, _1) = d_mixed.device_ptr(&stream);
        let (pz, _2) = d_z.device_ptr(&stream);
        let (pa, _3) = d_a.device_ptr(&stream);
        let (pb, _4) = d_b.device_ptr(&stream);
        let (pl, _5) = d_alog.device_ptr(&stream);
        let (pd, _6) = d_dt.device_ptr(&stream);
        let (pn, _7) = d_nw.device_ptr(&stream);
        let (ps, _8) = d_state.device_ptr(&stream);
        let (pc, _9) = d_ck.device_ptr_mut(&stream);
        let (po, _10) = d_out.device_ptr_mut(&stream);
        unsafe {
            cuda::gdn_decode_chunk_bf16(
                raw,
                pm as *const u16,
                pz as *const u16,
                pa as *const u16,
                pb as *const u16,
                pl as *const u16,
                pd as *const u16,
                pn as *const u16,
                ps as *const f32,
                pc as *mut f32,
                po as *mut u16,
                n_k_arg,
                n_v_arg,
                d_k_arg,
                d_v_arg,
                1e-6f32,
                t,
            )
        }
    };
    let (nk, nv, dk, dv, tt) = (n_k as i32, n_v as i32, d_k as i32, d_v as i32, t as i32);
    assert_eq!(
        decode_rc(nk, nv, dk, 33, tt, &mut d_ck, &mut d_out),
        -2,
        "d_v%32"
    );
    assert_eq!(
        decode_rc(nk, nv, dk, 2048, tt, &mut d_ck, &mut d_out),
        -2,
        "d_v>1024"
    );
    assert_eq!(
        decode_rc(3, nv, dk, dv, tt, &mut d_ck, &mut d_out),
        -2,
        "n_v%n_k!=0"
    );
    assert_eq!(
        decode_rc(1, 1, 1024, 96, tt, &mut d_ck, &mut d_out),
        -2,
        "smem>96KiB"
    );
    assert_eq!(
        decode_rc(nk, nv, dk, dv, 0, &mut d_ck, &mut d_out),
        -2,
        "t=0"
    );
    assert_eq!(
        decode_rc(nk, nv, dk, dv, 17, &mut d_ck, &mut d_out),
        -2,
        "t>16"
    );
    assert_eq!(
        decode_rc(nk, nv, dk, dv, tt, &mut d_ck, &mut d_out),
        0,
        "valid"
    );
    stream.synchronize().unwrap();
    eprintln!("gdn_chunk_wrappers_reject_bad_shapes: all reject paths exercised");
}

const Q38_VERIFY_NK: usize = 16;
const Q38_VERIFY_NV: usize = 48;
const Q38_VERIFY_DK: usize = 128;
const Q38_VERIFY_DV: usize = 128;
const Q38_VERIFY_CONV_DIM: usize = 10240;
const Q38_VERIFY_CONV_K: usize = 4;
const Q38_VERIFY_M_CASES: [usize; 4] = [2, 3, 5, 8];
const Q38_GDN_LAYERS_48_IS_WHAT_A_VERIFY_ROUND_PAYS_THE_PER_LAYER_COST_TIMES: usize = 48;

#[test]
#[ignore]
fn q38_verify_gdn_chunk_ab_at_serving_shapes() {
    if !bench_gated() {
        eprintln!("q38_verify_gdn_chunk_ab_at_serving_shapes: SKIP set NV_GDN_CHUNK_BENCH=1");
        return;
    }
    let stream = match CudaContext::new(0) {
        Ok(c) => c.default_stream(),
        Err(e) => panic!("q38_verify_gdn_chunk_ab: NV_GDN_CHUNK_BENCH=1 but no CUDA device 0: {e}"),
    };
    let raw = stream.cu_stream() as *mut c_void;
    let (n_k, n_v, d_k, d_v) = (Q38_VERIFY_NK, Q38_VERIFY_NV, Q38_VERIFY_DK, Q38_VERIFY_DV);
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let mixed_stride = 2 * key_dim + value_dim;
    let state_len = n_v * d_k * d_v;
    let conv_dim = Q38_VERIFY_CONV_DIM;
    let conv_k = Q38_VERIFY_CONV_K;
    let conv_row = conv_dim * (conv_k - 1);
    let t_max = *Q38_VERIFY_M_CASES.iter().max().unwrap();
    let reps = 200usize;
    let warmup = 20usize;

    let fx = decode_fixture(n_k, n_v, d_k, d_v, t_max);
    let d_mixed = htod_u16(&stream, &fx.mixed);
    let d_z = htod_u16(&stream, &fx.z);
    let d_a = htod_u16(&stream, &fx.a);
    let d_b = htod_u16(&stream, &fx.b);
    let d_alog = htod_u16(&stream, &fx.a_log);
    let d_dt = htod_u16(&stream, &fx.dt_bias);
    let d_nw = htod_u16(&stream, &fx.norm_w);
    let d_state_ro = htod_f32(&stream, &fx.state0);
    let mut d_state_mut = htod_f32(&stream, &fx.state0);
    let mut d_live: CudaSlice<f32> = stream.alloc_zeros::<f32>(state_len).unwrap();
    let mut d_ckpt: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * state_len).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * value_dim).unwrap();
    let mut d_qn: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * key_dim).unwrap();
    let mut d_kn: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * key_dim).unwrap();
    let mut d_ge: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * n_v).unwrap();
    let mut d_be: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * n_v).unwrap();
    let mut d_core: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * value_dim).unwrap();

    let mut seed = 0x1234_5678_u64;
    let x_seq = rand_bf16(&mut seed, t_max * conv_dim, -1.0, 1.0);
    let conv0 = rand_bf16(&mut seed, conv_row, -1.0, 1.0);
    let cw = rand_bf16(&mut seed, conv_dim * conv_k, -0.5, 0.5);
    let d_x = htod_u16(&stream, &x_seq);
    let d_cw = htod_u16(&stream, &cw);
    let d_conv_ro = htod_u16(&stream, &conv0);
    let mut d_conv_mut = htod_u16(&stream, &conv0);
    let mut d_convy: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * conv_dim).unwrap();
    let mut d_convck: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * conv_row).unwrap();
    let mut d_ckpt_dst: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * state_len).unwrap();
    let mut d_convck_dst: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * conv_row).unwrap();

    let timed = |label: &str, m: usize, f: &mut dyn FnMut()| -> f64 {
        for _ in 0..warmup {
            f();
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        let per_round_ms =
            us * Q38_GDN_LAYERS_48_IS_WHAT_A_VERIFY_ROUND_PAYS_THE_PER_LAYER_COST_TIMES as f64
                / 1e3;
        eprintln!("Q38-VERIFY-GDN-AB m={m} {label} us_per_layer={us:.2} ms_per_round_48L={per_round_ms:.2}");
        us
    };

    for m in Q38_VERIFY_M_CASES {
        let mut state_seq = || {
            for i in 0..m {
                let (pm, _1) = d_mixed.device_ptr(&stream);
                let (pz, _2) = d_z.device_ptr(&stream);
                let (pa, _3) = d_a.device_ptr(&stream);
                let (pb, _4) = d_b.device_ptr(&stream);
                let (pl, _5) = d_alog.device_ptr(&stream);
                let (pd, _6) = d_dt.device_ptr(&stream);
                let (pn, _7) = d_nw.device_ptr(&stream);
                let (ps, _8) = d_state_mut.device_ptr_mut(&stream);
                let (po, _9) = d_out.device_ptr_mut(&stream);
                let (pq, _10) = d_qn.device_ptr_mut(&stream);
                let (pk, _11) = d_kn.device_ptr_mut(&stream);
                let (pg, _12) = d_ge.device_ptr_mut(&stream);
                let (pbe, _13) = d_be.device_ptr_mut(&stream);
                let (pc, _14) = d_core.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gdn_decode_step_split_bf16(
                        raw,
                        (pm as usize + i * mixed_stride * 2) as *const u16,
                        (pz as usize + i * value_dim * 2) as *const u16,
                        (pa as usize + i * n_v * 2) as *const u16,
                        (pb as usize + i * n_v * 2) as *const u16,
                        pl as *const u16,
                        pd as *const u16,
                        pn as *const u16,
                        ps as *mut f32,
                        (po as usize + i * value_dim * 2) as *mut u16,
                        pq as *mut f32,
                        pk as *mut f32,
                        pg as *mut f32,
                        pbe as *mut f32,
                        pc as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        1e-6f32,
                    )
                };
                assert_eq!(rc, 0, "gdn_decode_step_split_bf16 rc={rc}");
            }
        };
        timed("state_m_sequential_split_steps", m, &mut state_seq);

        let mut state_chunk = || {
            let (pm, _1) = d_mixed.device_ptr(&stream);
            let (pz, _2) = d_z.device_ptr(&stream);
            let (pa, _3) = d_a.device_ptr(&stream);
            let (pb, _4) = d_b.device_ptr(&stream);
            let (pl, _5) = d_alog.device_ptr(&stream);
            let (pd, _6) = d_dt.device_ptr(&stream);
            let (pn, _7) = d_nw.device_ptr(&stream);
            let (ps, _8) = d_state_ro.device_ptr(&stream);
            let (pck, _9) = d_ckpt.device_ptr_mut(&stream);
            let (plv, _10) = d_live.device_ptr_mut(&stream);
            let (po, _11) = d_out.device_ptr_mut(&stream);
            let (pq, _12) = d_qn.device_ptr_mut(&stream);
            let (pk, _13) = d_kn.device_ptr_mut(&stream);
            let (pg, _14) = d_ge.device_ptr_mut(&stream);
            let (pbe, _15) = d_be.device_ptr_mut(&stream);
            let (pc, _16) = d_core.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_decode_chunk_split_bf16(
                    raw,
                    pm as *const u16,
                    pz as *const u16,
                    pa as *const u16,
                    pb as *const u16,
                    pl as *const u16,
                    pd as *const u16,
                    pn as *const u16,
                    ps as *const f32,
                    pck as *mut f32,
                    plv as *mut f32,
                    po as *mut u16,
                    pq as *mut f32,
                    pk as *mut f32,
                    pg as *mut f32,
                    pbe as *mut f32,
                    pc as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    1e-6f32,
                    m as i32,
                )
            };
            assert_eq!(rc, 0, "gdn_decode_chunk_split_bf16 rc={rc}");
        };
        timed("state_one_chunk_split_scan", m, &mut state_chunk);

        let mut conv_seq = || {
            for i in 0..m {
                let (px, _1) = d_x.device_ptr(&stream);
                let (pc, _2) = d_conv_mut.device_ptr_mut(&stream);
                let (pw, _3) = d_cw.device_ptr(&stream);
                let (py, _4) = d_convy.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gdn_conv_decode_silu_bf16(
                        raw,
                        (px as usize + i * conv_dim * 2) as *const u16,
                        pc as *mut u16,
                        pw as *const u16,
                        (py as usize + i * conv_dim * 2) as *mut u16,
                        conv_dim as i32,
                        conv_k as i32,
                    )
                };
                assert_eq!(rc, 0, "gdn_conv_decode_silu_bf16 rc={rc}");
            }
        };
        timed("conv_m_sequential_steps", m, &mut conv_seq);

        let mut conv_chunk = || {
            let (px, _1) = d_x.device_ptr(&stream);
            let (pc, _2) = d_conv_ro.device_ptr(&stream);
            let (pw, _3) = d_cw.device_ptr(&stream);
            let (py, _4) = d_convy.device_ptr_mut(&stream);
            let (pk, _5) = d_convck.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_conv_decode_chunk_silu_bf16(
                    raw,
                    px as *const u16,
                    pc as *const u16,
                    pw as *const u16,
                    py as *mut u16,
                    pk as *mut u16,
                    conv_dim as i32,
                    conv_k as i32,
                    m as i32,
                )
            };
            assert_eq!(rc, 0, "gdn_conv_decode_chunk_silu_bf16 rc={rc}");
        };
        timed("conv_one_chunk_kernel", m, &mut conv_chunk);

        let rec_row_bytes = state_len * 4;
        let conv_row_bytes = conv_row * 2;
        let mut ckpt_dtod = || {
            let (src_r, _1) = d_ckpt.device_ptr(&stream);
            let (dst_r, _2) = d_ckpt_dst.device_ptr_mut(&stream);
            let (src_c, _3) = d_convck.device_ptr(&stream);
            let (dst_c, _4) = d_convck_dst.device_ptr_mut(&stream);
            for j in 0..m {
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst_c + (j * conv_row_bytes) as u64,
                        src_c + (j * conv_row_bytes) as u64,
                        conv_row_bytes,
                        stream.cu_stream(),
                    )
                    .unwrap();
                    cudarc::driver::result::memcpy_dtod_async(
                        dst_r + (j * rec_row_bytes) as u64,
                        src_r + (j * rec_row_bytes) as u64,
                        rec_row_bytes,
                        stream.cu_stream(),
                    )
                    .unwrap();
                }
            }
        };
        timed("ckpt_row_dtod_fanout", m, &mut ckpt_dtod);
    }
}

#[test]
#[ignore]
fn gdn_chunk_microbench() {
    if !bench_gated() {
        eprintln!("gdn_chunk_microbench: SKIP set NV_GDN_CHUNK_BENCH=1 to run");
        return;
    }
    let stream = match CudaContext::new(0) {
        Ok(c) => c.default_stream(),
        Err(e) => {
            panic!("gdn_chunk_microbench: NV_GDN_CHUNK_BENCH=1 set but no CUDA device 0: {e}")
        }
    };
    let raw = stream.cu_stream() as *mut c_void;
    let reps = 200usize;
    let warmup = 20usize;
    let bench_t: [usize; 5] = [2, 3, 4, 8, 16];

    let (n_k, n_v, d_k, d_v) = (16usize, 32usize, 128usize, 128usize);
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let mixed_stride = 2 * key_dim + value_dim;
    let state_len = n_v * d_k * d_v;
    let t_max = *bench_t.iter().max().unwrap();
    let fx = decode_fixture(n_k, n_v, d_k, d_v, t_max);
    let d_mixed = htod_u16(&stream, &fx.mixed);
    let d_z = htod_u16(&stream, &fx.z);
    let d_a = htod_u16(&stream, &fx.a);
    let d_b = htod_u16(&stream, &fx.b);
    let d_alog = htod_u16(&stream, &fx.a_log);
    let d_dt = htod_u16(&stream, &fx.dt_bias);
    let d_nw = htod_u16(&stream, &fx.norm_w);
    let d_state_ro = htod_f32(&stream, &fx.state0);
    let mut d_state_mut = htod_f32(&stream, &fx.state0);
    let mut d_ckpt: CudaSlice<f32> = stream.alloc_zeros::<f32>(t_max * state_len).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * value_dim).unwrap();

    eprintln!(
        "decode microbench shape=({n_k},{n_v},{d_k},{d_v}) reps={reps} smem_chunk={} KiB grid={n_v} blocks",
        (d_k * d_v + 2 * d_k + 32) * 4 / 1024
    );
    for t in bench_t {
        let chunk_once = |d_ckpt: &mut CudaSlice<f32>, d_out: &mut CudaSlice<u16>| {
            let (pm, _1) = d_mixed.device_ptr(&stream);
            let (pz, _2) = d_z.device_ptr(&stream);
            let (pa, _3) = d_a.device_ptr(&stream);
            let (pb, _4) = d_b.device_ptr(&stream);
            let (pl, _5) = d_alog.device_ptr(&stream);
            let (pd, _6) = d_dt.device_ptr(&stream);
            let (pn, _7) = d_nw.device_ptr(&stream);
            let (ps, _8) = d_state_ro.device_ptr(&stream);
            let (pc, _9) = d_ckpt.device_ptr_mut(&stream);
            let (po, _10) = d_out.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_decode_chunk_bf16(
                    raw,
                    pm as *const u16,
                    pz as *const u16,
                    pa as *const u16,
                    pb as *const u16,
                    pl as *const u16,
                    pd as *const u16,
                    pn as *const u16,
                    ps as *const f32,
                    pc as *mut f32,
                    po as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    1e-6f32,
                    t as i32,
                )
            };
            assert_eq!(rc, 0);
        };
        let seq_once = |d_state_mut: &mut CudaSlice<f32>, d_out: &mut CudaSlice<u16>| {
            for i in 0..t {
                let (pm, _1) = d_mixed.device_ptr(&stream);
                let (pz, _2) = d_z.device_ptr(&stream);
                let (pa, _3) = d_a.device_ptr(&stream);
                let (pb, _4) = d_b.device_ptr(&stream);
                let (pl, _5) = d_alog.device_ptr(&stream);
                let (pd, _6) = d_dt.device_ptr(&stream);
                let (pn, _7) = d_nw.device_ptr(&stream);
                let (ps, _8) = d_state_mut.device_ptr_mut(&stream);
                let (po, _9) = d_out.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gdn_decode_step_bf16(
                        raw,
                        (pm as usize + i * mixed_stride * 2) as *const u16,
                        (pz as usize + i * value_dim * 2) as *const u16,
                        (pa as usize + i * n_v * 2) as *const u16,
                        (pb as usize + i * n_v * 2) as *const u16,
                        pl as *const u16,
                        pd as *const u16,
                        pn as *const u16,
                        ps as *mut f32,
                        (po as usize + i * value_dim * 2) as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        1e-6f32,
                    )
                };
                assert_eq!(rc, 0);
            }
        };

        for _ in 0..warmup {
            chunk_once(&mut d_ckpt, &mut d_out);
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            chunk_once(&mut d_ckpt, &mut d_out);
        }
        stream.synchronize().unwrap();
        let chunk_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        for _ in 0..warmup {
            seq_once(&mut d_state_mut, &mut d_out);
        }
        stream.synchronize().unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            seq_once(&mut d_state_mut, &mut d_out);
        }
        stream.synchronize().unwrap();
        let seq_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;

        eprintln!(
            "decode t={t}: chunk {chunk_us:.1} us vs {t}x t==1 {seq_us:.1} us -> speedup {:.2}x (chunk/step ratio {:.2})",
            seq_us / chunk_us,
            chunk_us / (seq_us / t as f64)
        );
    }

    let conv_dim = 8192usize;
    let k = 4usize;
    let mut seed = 0xbeef_cafe_u64;
    let x_seq = rand_bf16(&mut seed, t_max * conv_dim, -1.0, 1.0);
    let conv0 = rand_bf16(&mut seed, conv_dim * (k - 1), -1.0, 1.0);
    let w = rand_bf16(&mut seed, conv_dim * k, -0.5, 0.5);
    let d_x = htod_u16(&stream, &x_seq);
    let d_w = htod_u16(&stream, &w);
    let d_conv_ro = htod_u16(&stream, &conv0);
    let mut d_conv_mut = htod_u16(&stream, &conv0);
    let mut d_y: CudaSlice<u16> = stream.alloc_zeros::<u16>(t_max * conv_dim).unwrap();
    let mut d_cck: CudaSlice<u16> = stream
        .alloc_zeros::<u16>(t_max * conv_dim * (k - 1))
        .unwrap();

    eprintln!("conv microbench C={conv_dim} K={k} reps={reps}");
    for t in bench_t {
        let chunk_once = |d_y: &mut CudaSlice<u16>, d_cck: &mut CudaSlice<u16>| {
            let (px, _1) = d_x.device_ptr(&stream);
            let (pc, _2) = d_conv_ro.device_ptr(&stream);
            let (pw, _3) = d_w.device_ptr(&stream);
            let (py, _4) = d_y.device_ptr_mut(&stream);
            let (pk, _5) = d_cck.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_conv_decode_chunk_silu_bf16(
                    raw,
                    px as *const u16,
                    pc as *const u16,
                    pw as *const u16,
                    py as *mut u16,
                    pk as *mut u16,
                    conv_dim as i32,
                    k as i32,
                    t as i32,
                )
            };
            assert_eq!(rc, 0);
        };
        let seq_once = |d_conv_mut: &mut CudaSlice<u16>, d_y: &mut CudaSlice<u16>| {
            for i in 0..t {
                let (px, _1) = d_x.device_ptr(&stream);
                let (pc, _2) = d_conv_mut.device_ptr_mut(&stream);
                let (pw, _3) = d_w.device_ptr(&stream);
                let (py, _4) = d_y.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gdn_conv_decode_silu_bf16(
                        raw,
                        (px as usize + i * conv_dim * 2) as *const u16,
                        pc as *mut u16,
                        pw as *const u16,
                        (py as usize + i * conv_dim * 2) as *mut u16,
                        conv_dim as i32,
                        k as i32,
                    )
                };
                assert_eq!(rc, 0);
            }
        };

        for _ in 0..warmup {
            chunk_once(&mut d_y, &mut d_cck);
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            chunk_once(&mut d_y, &mut d_cck);
        }
        stream.synchronize().unwrap();
        let chunk_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        for _ in 0..warmup {
            seq_once(&mut d_conv_mut, &mut d_y);
        }
        stream.synchronize().unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            seq_once(&mut d_conv_mut, &mut d_y);
        }
        stream.synchronize().unwrap();
        let seq_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;

        eprintln!(
            "conv t={t}: chunk {chunk_us:.1} us vs {t}x t==1 {seq_us:.1} us -> speedup {:.2}x (chunk/step ratio {:.2})",
            seq_us / chunk_us,
            chunk_us / (seq_us / t as f64)
        );
    }
}
