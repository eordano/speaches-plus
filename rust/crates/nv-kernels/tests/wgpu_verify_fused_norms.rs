#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::lcg;
use common::rnd_f;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::kernels::verify_fused_norms as vf;
use nv_kernels::wgpu_backend::WgpuError;

const SPLITS: usize = 16;

fn rnd_bf16(state: &mut u64) -> u16 {
    bf16::from_f32(rnd_f(state)).to_bits()
}

fn bf_dec(w: u16) -> f32 {
    f32::from_bits(u32::from(w) << 16)
}

fn e4m3_dec(b: u8) -> f32 {
    let mag = b & 0x7f;
    if mag == 0x7f {
        return f32::NAN;
    }
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f32;
    let v = if e == 0 {
        m * 0.001_953_125
    } else {
        (1.0 + m * 0.125) * (2f32).powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

struct PrepInputs {
    fused: Vec<u16>,
    qw: Vec<u16>,
    kw: Vec<u16>,
    vw: Vec<u16>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    pos: Vec<i32>,
    width: usize,
    q_dim: usize,
    kv_dim: usize,
}

fn prep_inputs(
    k: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    committed: i32,
    seed: u64,
) -> PrepInputs {
    let mut st = seed;
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let width = q_dim + 2 * kv_dim;
    let half = hd / 2;
    let max_pos = 64usize;
    PrepInputs {
        fused: (0..k * width).map(|_| rnd_bf16(&mut st)).collect(),
        qw: (0..hd).map(|_| rnd_bf16(&mut st)).collect(),
        kw: (0..hd).map(|_| rnd_bf16(&mut st)).collect(),
        vw: (0..hd).map(|_| rnd_bf16(&mut st)).collect(),
        cos: (0..max_pos * half).map(|_| rnd_f(&mut st)).collect(),
        sin: (0..max_pos * half).map(|_| rnd_f(&mut st)).collect(),
        pos: (0..k)
            .map(|i| (committed + i as i32) % max_pos as i32)
            .collect(),
        width,
        q_dim,
        kv_dim,
    }
}

struct PrepOut {
    q_out: Vec<u16>,
    kc: Vec<u8>,
    vc: Vec<u8>,
    ks: Vec<f32>,
    vs: Vec<f32>,
}

fn run_prep(
    ctx: &WgpuContext,
    inp: &PrepInputs,
    k: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    ring: usize,
    committed: i32,
    slots: usize,
) -> PrepOut {
    let mut q_out = vec![0u16; k * nq * hd];
    let mut kc = vec![0u8; slots * nkv * hd];
    let mut vc = vec![0u8; slots * nkv * hd];
    let mut ks = vec![0f32; slots * nkv];
    let mut vs = vec![0f32; slots * nkv];
    vf::verify_qkv_prep(
        ctx,
        &inp.fused,
        inp.width,
        0,
        inp.q_dim,
        inp.q_dim + inp.kv_dim,
        &inp.qw,
        &inp.kw,
        &inp.vw,
        1e-6,
        &inp.cos,
        &inp.sin,
        &inp.pos,
        &mut q_out,
        &mut kc,
        &mut vc,
        &mut ks,
        &mut vs,
        &[committed],
        k,
        nq,
        nkv,
        hd,
        ring,
    )
    .expect("wgpu verify_qkv_prep");
    PrepOut {
        q_out,
        kc,
        vc,
        ks,
        vs,
    }
}

fn cpu_prep_reference(
    inp: &PrepInputs,
    k: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    ring: usize,
    committed: i32,
    slots: usize,
) -> PrepOut {
    let half = hd / 2;
    let eps = 1e-6f64;
    let mut q_out = vec![0u16; k * nq * hd];
    let mut kc = vec![0u8; slots * nkv * hd];
    let mut vc = vec![0u8; slots * nkv * hd];
    let mut ks = vec![0f32; slots * nkv];
    let mut vs = vec![0f32; slots * nkv];
    let norm_row = |row: &[u16], w: &[u16]| -> Vec<f32> {
        let sum: f64 = row.iter().map(|x| f64::from(bf_dec(*x)).powi(2)).sum();
        let rms = 1.0 / (sum / hd as f64 + eps).sqrt();
        row.iter()
            .zip(w.iter())
            .map(|(x, wi)| {
                bf16::from_f32((f64::from(bf_dec(*x)) * rms * f64::from(bf_dec(*wi))) as f32)
                    .to_f32()
            })
            .collect()
    };
    let rope_row = |row: &[f32], pos: i32| -> Vec<f32> {
        let crow = pos as usize * half;
        let mut out = vec![0f32; hd];
        for i in 0..half {
            let a = row[i];
            let b = row[i + half];
            let c = inp.cos[crow + i];
            let s = inp.sin[crow + i];
            out[i] = a * c - b * s;
            out[i + half] = a * s + b * c;
        }
        out
    };
    for t in 0..k {
        let base = t * inp.width;
        for h in 0..nq {
            let row = &inp.fused[base + h * hd..base + (h + 1) * hd];
            let normed = norm_row(row, &inp.qw);
            let roped = rope_row(&normed, inp.pos[t]);
            for d in 0..hd {
                q_out[(t * nq + h) * hd + d] = bf16::from_f32(roped[d]).to_bits();
            }
        }
        let mut slot = (committed + t as i32) as usize;
        if ring > 0 {
            slot %= ring;
        }
        for kvh in 0..nkv {
            let krow = &inp.fused[base + inp.q_dim + kvh * hd..base + inp.q_dim + (kvh + 1) * hd];
            let vrow = &inp.fused[base + inp.q_dim + inp.kv_dim + kvh * hd
                ..base + inp.q_dim + inp.kv_dim + (kvh + 1) * hd];
            let kn: Vec<f32> = rope_row(&norm_row(krow, &inp.kw), inp.pos[t])
                .iter()
                .map(|x| bf16::from_f32(*x).to_f32())
                .collect();
            let vn = norm_row(vrow, &inp.vw);
            let amax_k = kn.iter().fold(0f32, |a, b| a.max(b.abs()));
            let amax_v = vn.iter().fold(0f32, |a, b| a.max(b.abs()));
            let (sk, ik) = if amax_k > 0.0 {
                (amax_k / 448.0, 448.0 / amax_k)
            } else {
                (1.0, 1.0)
            };
            let (sv, iv) = if amax_v > 0.0 {
                (amax_v / 448.0, 448.0 / amax_v)
            } else {
                (1.0, 1.0)
            };
            ks[slot * nkv + kvh] = sk;
            vs[slot * nkv + kvh] = sv;
            for d in 0..hd {
                kc[(slot * nkv + kvh) * hd + d] =
                    nv_kernels::wgpu_backend::kernels::kv_fp8::encode_e4m3(kn[d] * ik);
                vc[(slot * nkv + kvh) * hd + d] =
                    nv_kernels::wgpu_backend::kernels::kv_fp8::encode_e4m3(vn[d] * iv);
            }
        }
    }
    PrepOut {
        q_out,
        kc,
        vc,
        ks,
        vs,
    }
}

#[test]
fn qkv_prep_tracks_cpu_reference() {
    let Some(ctx) = ctx_or_skip("qkv_prep_tracks_cpu_reference") else {
        return;
    };
    for (k, nq, nkv, hd, ring, committed) in [
        (4usize, 8usize, 4usize, 256usize, 0usize, 3i32),
        (2, 4, 2, 512, 0, 7),
        (3, 2, 1, 36, 0, 0),
        (2, 3, 3, 68, 0, 5),
        (4, 8, 4, 256, 24, 21),
    ] {
        let inp = prep_inputs(k, nq, nkv, hd, committed, 0x9e37 + hd as u64);
        let slots = if ring > 0 {
            ring
        } else {
            committed as usize + k + 4
        };
        let got = run_prep(ctx, &inp, k, nq, nkv, hd, ring, committed, slots);
        let want = cpu_prep_reference(&inp, k, nq, nkv, hd, ring, committed, slots);

        let mut worst_q = 0f32;
        for (a, b) in got.q_out.iter().zip(want.q_out.iter()) {
            worst_q = worst_q.max((bf_dec(*a) - bf_dec(*b)).abs());
        }
        let mut worst_scale = 0f32;
        for (a, b) in got
            .ks
            .iter()
            .zip(want.ks.iter())
            .chain(got.vs.iter().zip(want.vs.iter()))
        {
            worst_scale = worst_scale.max((a - b).abs());
        }
        let mut worst_kv = 0f32;
        for i in 0..got.kc.len() {
            let slot_head = i / hd;
            let a = e4m3_dec(got.kc[i]) * got.ks[slot_head];
            let b = e4m3_dec(want.kc[i]) * want.ks[slot_head];
            worst_kv = worst_kv.max((a - b).abs());
            let av = e4m3_dec(got.vc[i]) * got.vs[slot_head];
            let bv = e4m3_dec(want.vc[i]) * want.vs[slot_head];
            worst_kv = worst_kv.max((av - bv).abs());
        }
        eprintln!(
            "qkv_prep k={k} nq={nq} nkv={nkv} hd={hd} ring={ring}: worst_q={worst_q:e} \
             worst_scale={worst_scale:e} worst_dequant={worst_kv:e}"
        );
        assert!(
            worst_q < 3e-2,
            "q_out drifts from CPU reference: {worst_q:e}"
        );
        assert!(
            worst_scale < 1e-3,
            "scales drift from CPU reference: {worst_scale:e}"
        );
        assert!(
            worst_kv < 3e-2,
            "dequantized kv drifts from CPU reference: {worst_kv:e}"
        );
        assert!(got.q_out.iter().any(|x| *x != 0), "q_out all zeros");
        assert!(got.kc.iter().any(|x| *x != 0), "kc all zeros");
    }
}

#[test]
fn qkv_prep_is_deterministic_and_respects_ring_slots() {
    let Some(ctx) = ctx_or_skip("qkv_prep_deterministic_ring") else {
        return;
    };
    let (k, nq, nkv, hd, ring, committed) = (4usize, 4usize, 2usize, 128usize, 6usize, 21i32);
    let inp = prep_inputs(k, nq, nkv, hd, committed, 0x1234);
    let a = run_prep(ctx, &inp, k, nq, nkv, hd, ring, committed, ring);
    let b = run_prep(ctx, &inp, k, nq, nkv, hd, ring, committed, ring);
    assert_eq!(a.q_out, b.q_out, "q_out not deterministic");
    assert_eq!(a.kc, b.kc, "kc not deterministic");
    assert_eq!(a.vc, b.vc, "vc not deterministic");
    assert_eq!(
        a.ks.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.ks.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "k_scale not deterministic"
    );

    let expected_slots: Vec<usize> = (0..k).map(|t| (committed as usize + t) % ring).collect();
    for slot in 0..ring {
        let touched = expected_slots.contains(&slot);
        let row = &a.kc[slot * nkv * hd..(slot + 1) * nkv * hd];
        let nonzero = row.iter().any(|x| *x != 0);
        if touched {
            assert!(nonzero, "ring slot {slot} should have been written");
        } else {
            assert!(!nonzero, "ring slot {slot} written outside its turn");
            for h in 0..nkv {
                assert_eq!(
                    a.ks[slot * nkv + h].to_bits(),
                    0,
                    "scale for untouched slot {slot}"
                );
            }
        }
    }
}

#[test]
fn qkv_prep_zero_k_row_yields_unit_scale_and_zero_bytes() {
    let Some(ctx) = ctx_or_skip("qkv_prep_zero_row") else {
        return;
    };
    let (k, nq, nkv, hd) = (1usize, 2usize, 1usize, 64usize);
    let mut inp = prep_inputs(k, nq, nkv, hd, 0, 0x777);
    for d in 0..hd {
        inp.fused[inp.q_dim + d] = 0;
    }
    let got = run_prep(ctx, &inp, k, nq, nkv, hd, 0, 0, 4);
    assert_eq!(
        got.ks[0].to_bits(),
        1.0f32.to_bits(),
        "zero K row must give scale 1.0"
    );
    assert!(
        got.kc[..hd].iter().all(|x| *x & 0x7f == 0),
        "zero K row must quantize to zero-magnitude fp8 bytes"
    );
    assert!(got.vs[0] > 0.0, "V scale should be positive");
}

#[test]
fn qkv_prep_constant_row_ties_quantize_uniformly() {
    let Some(ctx) = ctx_or_skip("qkv_prep_ties") else {
        return;
    };
    let (k, nq, nkv, hd) = (1usize, 1usize, 1usize, 128usize);
    let mut inp = prep_inputs(k, nq, nkv, hd, 0, 0x99);
    let c = bf16::from_f32(0.75).to_bits();
    let one = bf16::from_f32(1.0).to_bits();
    for d in 0..hd {
        inp.fused[inp.q_dim + inp.kv_dim + d] = c;
        inp.vw[d] = one;
    }
    let got = run_prep(ctx, &inp, k, nq, nkv, hd, 0, 0, 4);
    let first = got.vc[0];
    assert!(
        got.vc[..hd].iter().all(|x| *x == first),
        "constant V row must quantize to one repeated fp8 byte, got {:?}",
        &got.vc[..8]
    );
    assert!(got.vs[0] > 0.0);
}

#[test]
fn qkv_prep_rejects_bad_shapes() {
    let Some(ctx) = ctx_or_skip("qkv_prep_rejects") else {
        return;
    };
    let inp = prep_inputs(1, 1, 1, 64, 0, 1);
    let mut q_out = vec![0u16; 64];
    let mut kc = vec![0u8; 4 * 64];
    let mut vc = vec![0u8; 4 * 64];
    let mut ks = vec![0f32; 4];
    let mut vs = vec![0f32; 4];
    let err = vf::verify_qkv_prep(
        ctx,
        &inp.fused,
        inp.width,
        0,
        64,
        128,
        &inp.qw,
        &inp.kw,
        &inp.vw,
        1e-6,
        &inp.cos,
        &inp.sin,
        &inp.pos,
        &mut q_out,
        &mut kc,
        &mut vc,
        &mut ks,
        &mut vs,
        &[0],
        1,
        1,
        1,
        34,
        0,
    )
    .unwrap_err();
    assert!(
        matches!(err, WgpuError::Unsupported(_)),
        "hd=34 must be refused: {err}"
    );
    let err = vf::verify_qkv_prep(
        ctx,
        &inp.fused,
        inp.width,
        0,
        64,
        128,
        &inp.qw,
        &inp.kw,
        &inp.vw,
        1e-6,
        &inp.cos,
        &inp.sin,
        &inp.pos,
        &mut q_out,
        &mut kc,
        &mut vc,
        &mut ks,
        &mut vs,
        &[4],
        1,
        1,
        1,
        64,
        0,
    )
    .unwrap_err();
    assert!(
        matches!(err, WgpuError::Shape(_)),
        "overfull cache must be refused: {err}"
    );
}

fn cpu_rmsnorm2(
    x: &[u16],
    res: &[u16],
    w1: &[u16],
    w2: &[u16],
    batch: usize,
    hidden: usize,
) -> (Vec<u16>, Vec<u16>) {
    let eps = 1e-6f64;
    let mut sum_out = vec![0u16; batch * hidden];
    let mut norm_out = vec![0u16; batch * hidden];
    for row in 0..batch {
        let base = row * hidden;
        let s1: f64 = x[base..base + hidden]
            .iter()
            .map(|v| f64::from(bf_dec(*v)).powi(2))
            .sum();
        let rms1 = 1.0 / (s1 / hidden as f64 + eps).sqrt();
        let mut s: Vec<f32> = Vec::with_capacity(hidden);
        for i in 0..hidden {
            let t = (f64::from(bf_dec(x[base + i])) * rms1 * f64::from(bf_dec(w1[i]))) as f32;
            let tb = bf16::from_f32(t).to_f32();
            s.push(tb + bf_dec(res[base + i]));
        }
        let s2: f64 = s.iter().map(|v| f64::from(*v).powi(2)).sum();
        let rms2 = 1.0 / (s2 / hidden as f64 + eps).sqrt();
        for i in 0..hidden {
            sum_out[base + i] = bf16::from_f32(s[i]).to_bits();
            let sb = f64::from(bf16::from_f32(s[i]).to_f32());
            norm_out[base + i] =
                bf16::from_f32((sb * rms2 * f64::from(bf_dec(w2[i]))) as f32).to_bits();
        }
    }
    (sum_out, norm_out)
}

#[test]
fn rmsnorm2_tracks_cpu_reference_and_is_deterministic() {
    let Some(ctx) = ctx_or_skip("rmsnorm2_cpu_reference") else {
        return;
    };
    for (batch, hidden) in [(4usize, 5376usize), (1, 512), (3, 254), (2, 130)] {
        let mut st = 0x71u64 ^ hidden as u64;
        let x: Vec<u16> = (0..batch * hidden).map(|_| rnd_bf16(&mut st)).collect();
        let res: Vec<u16> = (0..batch * hidden).map(|_| rnd_bf16(&mut st)).collect();
        let w1: Vec<u16> = (0..hidden).map(|_| rnd_bf16(&mut st)).collect();
        let w2: Vec<u16> = (0..hidden).map(|_| rnd_bf16(&mut st)).collect();
        let mut sum_a = vec![0u16; batch * hidden];
        let mut norm_a = vec![0u16; batch * hidden];
        vf::rmsnorm2_residual_bf16(
            ctx,
            &x,
            &res,
            &w1,
            &w2,
            &mut sum_a,
            &mut norm_a,
            batch,
            hidden,
            1e-6,
        )
        .unwrap();
        let mut sum_b = vec![0u16; batch * hidden];
        let mut norm_b = vec![0u16; batch * hidden];
        vf::rmsnorm2_residual_bf16(
            ctx,
            &x,
            &res,
            &w1,
            &w2,
            &mut sum_b,
            &mut norm_b,
            batch,
            hidden,
            1e-6,
        )
        .unwrap();
        assert_eq!(sum_a, sum_b, "sum_out not deterministic");
        assert_eq!(norm_a, norm_b, "normed_out not deterministic");

        let (sum_ref, norm_ref) = cpu_rmsnorm2(&x, &res, &w1, &w2, batch, hidden);
        let mut worst_sum = 0f32;
        let mut worst_norm = 0f32;
        for i in 0..batch * hidden {
            worst_sum = worst_sum.max((bf_dec(sum_a[i]) - bf_dec(sum_ref[i])).abs());
            worst_norm = worst_norm.max((bf_dec(norm_a[i]) - bf_dec(norm_ref[i])).abs());
        }
        eprintln!(
            "rmsnorm2 batch={batch} hidden={hidden}: worst_sum={worst_sum:e} worst_norm={worst_norm:e}"
        );
        assert!(worst_sum < 3e-2, "sum drifts from reference: {worst_sum:e}");
        assert!(
            worst_norm < 3e-2,
            "normed drifts from reference: {worst_norm:e}"
        );
        assert!(norm_a.iter().any(|x| *x != 0), "normed all zeros");
    }
}

#[test]
fn rmsnorm2_zero_w1_passes_residual_through_exactly() {
    let Some(ctx) = ctx_or_skip("rmsnorm2_zero_w1") else {
        return;
    };
    let (batch, hidden) = (2usize, 384usize);
    let mut st = 0x81u64;
    let x: Vec<u16> = (0..batch * hidden).map(|_| rnd_bf16(&mut st)).collect();
    let res: Vec<u16> = (0..batch * hidden)
        .map(|_| bf16::from_f32(rnd_f(&mut st).abs() + 0.125).to_bits())
        .collect();
    let w1 = vec![0u16; hidden];
    let w2: Vec<u16> = (0..hidden).map(|_| rnd_bf16(&mut st)).collect();
    let mut sum_out = vec![0u16; batch * hidden];
    let mut norm_out = vec![0u16; batch * hidden];
    vf::rmsnorm2_residual_bf16(
        ctx,
        &x,
        &res,
        &w1,
        &w2,
        &mut sum_out,
        &mut norm_out,
        batch,
        hidden,
        1e-6,
    )
    .unwrap();
    assert_eq!(
        sum_out, res,
        "with w1=0 the residual must pass through bit-exactly"
    );
}

#[test]
fn rmsnorm_residual_scale_identity_case_is_exact() {
    let Some(ctx) = ctx_or_skip("rmsnorm_residual_scale_identity") else {
        return;
    };
    let (batch, hidden) = (3usize, 256usize);
    let mut st = 0x91u64;
    let x: Vec<u16> = (0..batch * hidden).map(|_| rnd_bf16(&mut st)).collect();
    let res: Vec<u16> = (0..batch * hidden)
        .map(|_| bf16::from_f32(rnd_f(&mut st).abs() + 0.125).to_bits())
        .collect();
    let w = vec![0u16; hidden];
    let mut out = vec![0u16; batch * hidden];
    vf::rmsnorm_residual_scale_bf16(ctx, &x, &res, &w, &mut out, batch, hidden, 1e-6, 1.0).unwrap();
    assert_eq!(
        out, res,
        "with w=0 and scale=1 the residual must pass through bit-exactly"
    );
}

#[test]
fn rmsnorm_residual_scale_tracks_cpu_reference() {
    let Some(ctx) = ctx_or_skip("rmsnorm_residual_scale_cpu") else {
        return;
    };
    let (batch, hidden, scale) = (4usize, 5376usize, std::f32::consts::FRAC_1_SQRT_2);
    let mut st = 0xa1u64;
    let x: Vec<u16> = (0..batch * hidden).map(|_| rnd_bf16(&mut st)).collect();
    let res: Vec<u16> = (0..batch * hidden).map(|_| rnd_bf16(&mut st)).collect();
    let w: Vec<u16> = (0..hidden).map(|_| rnd_bf16(&mut st)).collect();
    let mut out = vec![0u16; batch * hidden];
    vf::rmsnorm_residual_scale_bf16(ctx, &x, &res, &w, &mut out, batch, hidden, 1e-6, scale)
        .unwrap();
    let mut worst = 0f32;
    for row in 0..batch {
        let base = row * hidden;
        let s: f64 = x[base..base + hidden]
            .iter()
            .map(|v| f64::from(bf_dec(*v)).powi(2))
            .sum();
        let rms = 1.0 / (s / hidden as f64 + 1e-6).sqrt();
        for i in 0..hidden {
            let nb = bf16::from_f32(
                (f64::from(bf_dec(x[base + i])) * rms * f64::from(bf_dec(w[i]))) as f32,
            )
            .to_f32();
            let want = (bf_dec(res[base + i]) + nb) * scale;
            worst = worst.max((bf_dec(out[base + i]) - want).abs());
        }
    }
    eprintln!("rmsnorm_residual_scale: worst_abs={worst:e}");
    assert!(worst < 3e-2, "out drifts from reference: {worst:e}");
    assert!(out.iter().any(|x| *x != 0), "out all zeros");
}

#[test]
fn mk_bf16_equals_per_query_fused_calls_when_unwindowed() {
    let Some(ctx) = ctx_or_skip("mk_bf16_vs_per_query") else {
        return;
    };
    for (m, nh, nkv, hd, total) in [
        (2usize, 8usize, 4usize, 256usize, 300usize),
        (4, 8, 4, 128, 129),
        (5, 4, 2, 64, 64),
        (8, 8, 8, 256, 8),
        (3, 4, 2, 68, 200),
    ] {
        let mut st = 0xb1u64 ^ (total as u64) ^ ((m as u64) << 16);
        let q: Vec<f32> = (0..m * nh * hd).map(|_| rnd_f(&mut st)).collect();
        let k: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
        let v: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();

        let elems = fd::flash_splitk_scratch_elems_mk(nh, hd, m, SPLITS).unwrap();
        let mut scratch = vec![0f32; elems];
        let mut mk_out = vec![0u16; m * nh * hd];
        fd::flash_decode_fused_bf16kv_mk(
            ctx,
            &q,
            &k,
            &v,
            &mut mk_out,
            &mut scratch,
            &[total as i32],
            0,
            m,
            nh,
            nkv,
            hd,
            0,
            SPLITS,
        )
        .expect("wgpu mk bf16");

        let single_elems = fd::flash_splitk_scratch_elems(nh, hd, SPLITS).unwrap();
        let mut oracle = vec![0u16; m * nh * hd];
        for i in 0..m {
            let mut s1 = vec![0f32; single_elems];
            let mut o1 = vec![0u16; nh * hd];
            fd::flash_decode_fused_bf16kv(
                ctx,
                &q[i * nh * hd..(i + 1) * nh * hd],
                &k,
                &v,
                &mut o1,
                &mut s1,
                &[total as i32],
                m - 1 - i,
                nh,
                nkv,
                hd,
                0,
                1.0,
                SPLITS,
                0,
            )
            .expect("wgpu per-query fused");
            oracle[i * nh * hd..(i + 1) * nh * hd].copy_from_slice(&o1);
        }
        let bad = mk_out
            .iter()
            .zip(oracle.iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "mk_bf16 vs per-query m={m} nh={nh} hd={hd} total={total}: bitdiff={bad}/{}",
            mk_out.len()
        );
        assert_eq!(
            bad, 0,
            "unwindowed mk must equal per-query fused calls bit-exactly"
        );
    }
}

#[test]
fn mk_fp8_equals_per_query_fused_calls_when_unwindowed() {
    let Some(ctx) = ctx_or_skip("mk_fp8_vs_per_query") else {
        return;
    };
    let (m, nh, nkv, hd, total) = (4usize, 8usize, 4usize, 128usize, 300usize);
    let mut st = 0xc1u64;
    let q: Vec<u16> = (0..m * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
    let k: Vec<u8> = (0..total * nkv * hd)
        .map(|_| {
            let r = lcg(&mut st) >> 32;
            let sign = ((r & 1) as u8) << 7;
            let exp = 5u8 + ((r >> 1) & 3) as u8;
            let mant = ((r >> 4) & 7) as u8;
            sign | (exp << 3) | mant
        })
        .collect();
    let v: Vec<u8> = k.iter().rev().copied().collect();
    let ks: Vec<f32> = (0..total * nkv)
        .map(|_| 1.0 + rnd_f(&mut st) * 0.1)
        .collect();
    let vs: Vec<f32> = (0..total * nkv)
        .map(|_| 1.0 + rnd_f(&mut st) * 0.1)
        .collect();
    let scaling = 0.0625f32;

    let elems = fd::flash_splitk_scratch_elems_mk(nh, hd, m, SPLITS).unwrap();
    let mut scratch = vec![0f32; elems];
    let mut mk_out = vec![0u16; m * nh * hd];
    fd::flash_decode_fused_fp8kv_mk(
        ctx,
        &q,
        &k,
        &v,
        &ks,
        &vs,
        &mut mk_out,
        &mut scratch,
        &[total as i32],
        0,
        m,
        nh,
        nkv,
        hd,
        0,
        scaling,
        SPLITS,
        0,
    )
    .expect("wgpu mk fp8");

    let single_elems = fd::flash_splitk_scratch_elems(nh, hd, SPLITS).unwrap();
    let mut oracle = vec![0u16; m * nh * hd];
    for i in 0..m {
        let mut s1 = vec![0f32; single_elems];
        let mut o1 = vec![0u16; nh * hd];
        fd::flash_decode_fused_fp8kv(
            ctx,
            &q[i * nh * hd..(i + 1) * nh * hd],
            &k,
            &v,
            &ks,
            &vs,
            &mut o1,
            &mut s1,
            &[(total - (m - 1 - i)) as i32],
            nh,
            nkv,
            hd,
            0,
            scaling,
            SPLITS,
            0,
        )
        .expect("wgpu per-query fp8 fused");
        oracle[i * nh * hd..(i + 1) * nh * hd].copy_from_slice(&o1);
    }
    let bad = mk_out
        .iter()
        .zip(oracle.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("mk_fp8 vs per-query m={m}: bitdiff={bad}/{}", mk_out.len());
    assert_eq!(
        bad, 0,
        "unwindowed fp8 mk must equal per-query fused calls bit-exactly"
    );
}

#[test]
fn mk_windowed_tracks_cpu_attention() {
    let Some(ctx) = ctx_or_skip("mk_windowed_cpu") else {
        return;
    };
    let (m, nh, nkv, hd, total, window) = (4usize, 4usize, 2usize, 64usize, 200usize, 32usize);
    let mut st = 0xd1u64;
    let q: Vec<f32> = (0..m * nh * hd).map(|_| rnd_f(&mut st)).collect();
    let k: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
    let v: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();

    let elems = fd::flash_splitk_scratch_elems_mk(nh, hd, m, SPLITS).unwrap();
    let mut scratch = vec![0f32; elems];
    let mut out = vec![0u16; m * nh * hd];
    fd::flash_decode_fused_bf16kv_mk(
        ctx,
        &q,
        &k,
        &v,
        &mut out,
        &mut scratch,
        &[total as i32],
        0,
        m,
        nh,
        nkv,
        hd,
        window,
        SPLITS,
    )
    .expect("wgpu mk bf16 windowed");

    let mut worst = 0f32;
    for qi in 0..m {
        let tq = total - (m - 1) + qi;
        let start = tq.saturating_sub(window);
        for h in 0..nh {
            let kvh = h / (nh / nkv);
            let mut scores: Vec<f64> = Vec::new();
            for p in start..tq {
                let mut dot = 0f64;
                for d in 0..hd {
                    dot += f64::from(q[(qi * nh + h) * hd + d])
                        * f64::from(bf_dec(k[(p * nkv + kvh) * hd + d]));
                }
                scores.push(dot);
            }
            let mmax = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - mmax).exp()).collect();
            let denom: f64 = weights.iter().sum();
            for d in 0..hd {
                let mut acc = 0f64;
                for (idx, p) in (start..tq).enumerate() {
                    acc += weights[idx] * f64::from(bf_dec(v[(p * nkv + kvh) * hd + d]));
                }
                let want = (acc / denom) as f32;
                let got = bf_dec(out[(qi * nh + h) * hd + d]);
                worst = worst.max((got - want).abs());
            }
        }
    }
    eprintln!("mk windowed vs cpu: worst_abs={worst:e}");
    assert!(
        worst < 3e-2,
        "windowed mk drifts from CPU attention: {worst:e}"
    );
    assert!(out.iter().any(|x| *x != 0), "windowed mk output all zeros");
}

#[test]
fn mk_rejects_bad_shapes() {
    let Some(ctx) = ctx_or_skip("mk_rejects") else {
        return;
    };
    let (nh, nkv, hd, total) = (8usize, 4usize, 128usize, 16usize);
    let q = vec![0f32; 8 * nh * hd];
    let k = vec![0u16; total * nkv * hd];
    let v = vec![0u16; total * nkv * hd];
    let elems = fd::flash_splitk_scratch_elems_mk(nh, hd, 8, SPLITS).unwrap();
    let mut scratch = vec![0f32; elems];
    let mut out = vec![0u16; 8 * nh * hd];
    for (m, hd_bad) in [(0usize, hd), (9, hd), (4, 512)] {
        let r = fd::flash_decode_fused_bf16kv_mk(
            ctx,
            &q,
            &k,
            &v,
            &mut out,
            &mut scratch,
            &[total as i32],
            0,
            m,
            nh,
            nkv,
            hd_bad,
            0,
            SPLITS,
        );
        assert!(r.is_err(), "m={m} hd={hd_bad} must be rejected");
    }
    let r = fd::flash_decode_fused_fp8kv_mk(
        ctx,
        &vec![0u16; 4 * nh * hd],
        &vec![0u8; total * nkv * hd],
        &vec![0u8; total * nkv * hd],
        &vec![1f32; total * nkv],
        &vec![1f32; total * nkv],
        &mut out[..4 * nh * hd],
        &mut scratch,
        &[total as i32],
        0,
        4,
        nh,
        nkv,
        hd,
        0,
        1.0,
        SPLITS,
        8,
    );
    assert!(r.is_err(), "fp8 ring without window must be rejected");
}
