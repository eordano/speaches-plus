#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;
use nv_specdecode::wgpu_spec::{sp_wgsl, SpProbe, SpStage, SpecDims, SpecWeights, SP_ENTRIES};

const COVERED: &[&str] = &[
    "sp_embed",
    "sp_rms1",
    "sp_qkv",
    "sp_rope",
    "sp_kvwrite",
    "sp_attn",
    "sp_oproj",
    "sp_rms2",
    "sp_gateup",
    "sp_down",
    "sp_rmsf",
    "sp_logits",
    "sp_argmax",
];

const COMMITTED: usize = 5;
const KB: usize = 3;
const REL: f64 = 1e-5;

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().unwrap_or_else(|e| {
        panic!(
            "wgpu_spec_host_ref needs a wgpu adapter. It is the only host-reference gate on the \
             shipped sp_* kernels, so skipping it would report green over an unexecuted suite: {e}"
        )
    })
}

fn dims() -> SpecDims {
    SpecDims {
        h: 32,
        nh: 4,
        nkv: 2,
        hd: 8,
        inter: 24,
        vocab: 17,
        max_seq: 48,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

fn rnd(i: u64, seed: u64) -> f64 {
    let mut z = (i.wrapping_add(1))
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

fn rnd_vec(n: usize, seed: u64, amp: f64) -> Vec<f32> {
    (0..n).map(|i| (rnd(i as u64, seed) * amp) as f32).collect()
}

fn norm_vec(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| (1.0 + 0.2 * rnd(i as u64, seed)) as f32)
        .collect()
}

fn weights(d: &SpecDims) -> SpecWeights {
    SpecWeights {
        embed: rnd_vec(d.vocab * d.h, 101, 0.25),
        ln1: norm_vec(d.h, 102),
        wq: rnd_vec(d.qdim() * d.h, 103, 0.25),
        wk: rnd_vec(d.kvdim() * d.h, 104, 0.25),
        wv: rnd_vec(d.kvdim() * d.h, 105, 0.25),
        wo: rnd_vec(d.h * d.qdim(), 106, 0.25),
        ln2: norm_vec(d.h, 107),
        wg: rnd_vec(d.inter * d.h, 108, 0.25),
        wu: rnd_vec(d.inter * d.h, 109, 0.25),
        wd: rnd_vec(d.h * d.inter, 110, 0.25),
        lnf: norm_vec(d.h, 111),
        wlm: rnd_vec(d.vocab * d.h, 112, 0.25),
    }
}

fn probe() -> (SpProbe, SpecDims, SpecWeights) {
    let d = dims();
    let w = weights(&d);
    let mut p = SpProbe::new(ctx(), d, &w, KB).unwrap();
    p.set_step(COMMITTED, KB);
    (p, d, w)
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, b| a.max(b.abs()))
}

fn spread(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

fn gate(name: &str, got: &[f32], want: &[f64], rel: f64) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: reference has {} elements, kernel wrote {}",
        want.len(),
        got.len()
    );
    assert!(!want.is_empty(), "{name}: empty comparison");

    let ref_mag = max_abs(want);
    let got_f64: Vec<f64> = got.iter().map(|v| *v as f64).collect();
    let out_mag = max_abs(&got_f64);
    assert!(
        ref_mag > 1e-3,
        "{name}: reference is degenerate (max |ref| = {ref_mag:.3e}); a gate over zeros stays green forever"
    );
    assert!(
        out_mag > 1e-3,
        "{name}: kernel output is degenerate (max |out| = {out_mag:.3e}); nothing was computed"
    );
    let sp = spread(want);
    assert!(
        sp > 1e-3,
        "{name}: reference is constant (spread {sp:.3e}); a constant reference cannot see an indexing bug"
    );

    let tol = rel * (1.0 + ref_mag);
    assert!(
        sp > 1e3 * tol,
        "{name}: tolerance {tol:.3e} is not small against the signal (spread {sp:.3e}); the gate has no resolving power"
    );

    let mut worst = 0.0f64;
    let mut at = 0usize;
    for i in 0..want.len() {
        let e = (got_f64[i] - want[i]).abs();
        if e > worst {
            worst = e;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{name}: max abs error {worst:.3e} at index {at} (kernel {:.9}, host f64 {:.9}) exceeds tol {tol:.3e}",
        got_f64[at],
        want[at]
    );
    worst
}

fn matvec(out_rows: usize, k: usize, xs: &[f32], w: &[f32], rows: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; out_rows * rows];
    for r in 0..out_rows {
        for o in 0..rows {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += xs[r * k + i] as f64 * w[o * k + i] as f64;
            }
            out[r * rows + o] = acc;
        }
    }
    out
}

fn rms_ref(d: &SpecDims, xs: &[f32], gain: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0f64; KB * d.h];
    for r in 0..KB {
        let mut ss = 0.0f64;
        for i in 0..d.h {
            let v = xs[r * d.h + i] as f64;
            ss += v * v;
        }
        let rr = 1.0 / (ss / d.h as f64 + d.eps as f64).sqrt();
        for c in 0..d.h {
            out[r * d.h + c] = xs[r * d.h + c] as f64 * rr * gain[c] as f64;
        }
    }
    out
}

#[test]
fn sp_embed_matches_host_reference() {
    let (p, d, w) = probe();
    let toks: Vec<u32> = vec![4, 0, (d.vocab - 1) as u32];
    p.write_tokens(&toks);
    p.run("sp_embed");
    let got = p.read_stage(SpStage::X);
    let want: Vec<f64> = (0..KB * d.h)
        .map(|i| w.embed[toks[i / d.h] as usize * d.h + i % d.h] as f64)
        .collect();
    let e = gate("sp_embed", &got, &want, REL);
    println!("sp_embed: max abs err {e:.3e}");
}

#[test]
fn sp_rms1_matches_host_reference() {
    let (p, d, w) = probe();
    let xs = rnd_vec(KB * d.h, 201, 1.5);
    p.write_stage(SpStage::X, &xs);
    p.run("sp_rms1");
    let got = p.read_stage(SpStage::Xn);
    let want = rms_ref(&d, &xs, &w.ln1);
    let e = gate("sp_rms1", &got, &want, REL);
    println!("sp_rms1: max abs err {e:.3e}");
}

#[test]
fn sp_qkv_matches_host_reference() {
    let (p, d, w) = probe();
    let xn = rnd_vec(KB * d.h, 202, 1.2);
    p.write_stage(SpStage::Xn, &xn);
    p.run("sp_qkv");
    let eq = gate(
        "sp_qkv/q",
        &p.read_stage(SpStage::Qb),
        &matvec(KB, d.h, &xn, &w.wq, d.qdim()),
        REL,
    );
    let ek = gate(
        "sp_qkv/k",
        &p.read_stage(SpStage::Kbuf),
        &matvec(KB, d.h, &xn, &w.wk, d.kvdim()),
        REL,
    );
    let ev = gate(
        "sp_qkv/v",
        &p.read_stage(SpStage::Vbuf),
        &matvec(KB, d.h, &xn, &w.wv, d.kvdim()),
        REL,
    );
    println!("sp_qkv: max abs err q {eq:.3e} k {ek:.3e} v {ev:.3e}");
}

#[test]
fn sp_rope_matches_host_reference() {
    let (p, d, w) = probe();
    let _ = w;
    let half = d.hd / 2;
    let qs = rnd_vec(KB * d.qdim(), 203, 1.0);
    let ks = rnd_vec(KB * d.kvdim(), 204, 1.0);
    p.write_stage(SpStage::Qb, &qs);
    p.write_stage(SpStage::Kbuf, &ks);
    p.run("sp_rope");

    let rotate = |src: &[f32], heads: usize| -> Vec<f64> {
        let dimf = heads * d.hd;
        let mut out = vec![0.0f64; KB * dimf];
        for row in 0..KB {
            let pos = (COMMITTED + row) as f64;
            for head in 0..heads {
                for j in 0..half {
                    let inv = (d.rope_theta as f64).powf(-((2 * j) as f64) / (d.hd as f64));
                    let ang = pos * inv;
                    let (s, c) = ang.sin_cos();
                    let base = row * dimf + head * d.hd;
                    let a = src[base + j] as f64;
                    let b = src[base + j + half] as f64;
                    out[base + j] = a * c - b * s;
                    out[base + j + half] = a * s + b * c;
                }
            }
        }
        out
    };

    let eq = gate(
        "sp_rope/q",
        &p.read_stage(SpStage::Qb),
        &rotate(&qs, d.nh),
        REL,
    );
    let ek = gate(
        "sp_rope/k",
        &p.read_stage(SpStage::Kbuf),
        &rotate(&ks, d.nkv),
        REL,
    );
    println!("sp_rope: max abs err q {eq:.3e} k {ek:.3e}");
}

#[test]
fn sp_kvwrite_matches_host_reference() {
    let (p, d, _w) = probe();
    let kvdim = d.kvdim();
    let seed_k = rnd_vec(d.max_seq * kvdim, 205, 0.9);
    let seed_v = rnd_vec(d.max_seq * kvdim, 206, 0.9);
    let kb_in = rnd_vec(KB * kvdim, 207, 1.4);
    let vb_in = rnd_vec(KB * kvdim, 208, 1.4);
    p.write_stage(SpStage::Kc, &seed_k);
    p.write_stage(SpStage::Vc, &seed_v);
    p.write_stage(SpStage::Kbuf, &kb_in);
    p.write_stage(SpStage::Vbuf, &vb_in);
    p.run("sp_kvwrite");

    let expect = |seed: &[f32], src: &[f32]| -> Vec<f64> {
        let mut out: Vec<f64> = seed.iter().map(|v| *v as f64).collect();
        for row in 0..KB {
            let slot = COMMITTED + row;
            for r in 0..kvdim {
                out[slot * kvdim + r] = src[row * kvdim + r] as f64;
            }
        }
        out
    };
    let ek = gate(
        "sp_kvwrite/k",
        &p.read_stage(SpStage::Kc),
        &expect(&seed_k, &kb_in),
        REL,
    );
    let ev = gate(
        "sp_kvwrite/v",
        &p.read_stage(SpStage::Vc),
        &expect(&seed_v, &vb_in),
        REL,
    );
    println!("sp_kvwrite: max abs err k {ek:.3e} v {ev:.3e}");
}

struct AttnRef {
    out: Vec<f64>,
    peak_weight: f64,
}

fn attn_ref(d: &SpecDims, q: &[f32], kc: &[f32], vc: &[f32], scale: f64) -> AttnRef {
    let grp = d.nh / d.nkv;
    let qdim = d.qdim();
    let kvdim = d.kvdim();
    let mut out = vec![0.0f64; KB * qdim];
    let mut peak: f64 = 0.0;
    for row in 0..KB {
        let total = COMMITTED + row + 1;
        for head in 0..d.nh {
            let kvh = head / grp;
            let qoff = row * qdim + head * d.hd;
            let mut logits = vec![0.0f64; total];
            for (s, lg) in logits.iter_mut().enumerate() {
                let koff = s * kvdim + kvh * d.hd;
                let mut dt = 0.0f64;
                for i in 0..d.hd {
                    dt += q[qoff + i] as f64 * kc[koff + i] as f64;
                }
                *lg = dt * scale;
            }
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut l = 0.0f64;
            let mut acc = vec![0.0f64; d.hd];
            for (s, lg) in logits.iter().enumerate() {
                let wgt = (lg - m).exp();
                l += wgt;
                let koff = s * kvdim + kvh * d.hd;
                for i in 0..d.hd {
                    acc[i] += wgt * vc[koff + i] as f64;
                }
            }
            peak = peak.max(
                logits
                    .iter()
                    .map(|lg| (lg - m).exp() / l)
                    .fold(0.0, f64::max),
            );
            for i in 0..d.hd {
                out[qoff + i] = acc[i] / l;
            }
        }
    }
    AttnRef {
        out,
        peak_weight: peak,
    }
}

#[test]
fn sp_attn_matches_host_reference_and_carries_the_head_dim_scale() {
    let (p, d, _w) = probe();
    let qs = rnd_vec(KB * d.qdim(), 209, 2.0);
    let kc = rnd_vec(d.max_seq * d.kvdim(), 210, 2.0);
    let vc = rnd_vec(d.max_seq * d.kvdim(), 211, 1.0);
    p.write_stage(SpStage::Qb, &qs);
    p.write_stage(SpStage::Kc, &kc);
    p.write_stage(SpStage::Vc, &vc);
    p.run("sp_attn");
    let got = p.read_stage(SpStage::Ao);

    let scale = 1.0 / (d.hd as f64).sqrt();
    let truth = attn_ref(&d, &qs, &kc, &vc, scale);
    let unscaled = attn_ref(&d, &qs, &kc, &vc, 1.0);

    assert!(
        truth.peak_weight < 0.9,
        "attention is near one-hot (peak softmax weight {:.4}); softmax would be scale-insensitive \
         and this gate could not see a missing 1/sqrt(head_dim)",
        truth.peak_weight
    );

    let tol = REL * (1.0 + max_abs(&truth.out));
    let margin = truth
        .out
        .iter()
        .zip(unscaled.out.iter())
        .fold(0.0f64, |a, (t, u)| a.max((t - u).abs()));
    assert!(
        margin > 1e3 * tol,
        "dropping the 1/sqrt(head_dim) would move the reference by only {margin:.3e}, within \
         {:.0}x of the tolerance {tol:.3e}; this shape cannot certify the scale",
        margin / tol
    );

    let e = gate("sp_attn", &got, &truth.out, REL);
    println!(
        "sp_attn: max abs err {e:.3e}, tol {tol:.3e}, peak softmax weight {:.4}, \
         no-scale margin {margin:.3e} ({:.0}x tol)",
        truth.peak_weight,
        margin / tol
    );
}

#[test]
fn sp_oproj_matches_host_reference() {
    let (p, d, w) = probe();
    let ao = rnd_vec(KB * d.qdim(), 212, 1.0);
    let xs = rnd_vec(KB * d.h, 213, 0.8);
    p.write_stage(SpStage::Ao, &ao);
    p.write_stage(SpStage::X, &xs);
    p.run("sp_oproj");
    let got = p.read_stage(SpStage::H1);
    let mut want = matvec(KB, d.qdim(), &ao, &w.wo, d.h);
    for (i, v) in want.iter_mut().enumerate() {
        *v += xs[i] as f64;
    }
    let e = gate("sp_oproj", &got, &want, REL);
    println!("sp_oproj: max abs err {e:.3e}");
}

#[test]
fn sp_rms2_matches_host_reference() {
    let (p, d, w) = probe();
    let h1 = rnd_vec(KB * d.h, 214, 1.5);
    p.write_stage(SpStage::H1, &h1);
    p.run("sp_rms2");
    let got = p.read_stage(SpStage::Tn);
    let want = rms_ref(&d, &h1, &w.ln2);
    let e = gate("sp_rms2", &got, &want, REL);
    println!("sp_rms2: max abs err {e:.3e}");
}

#[test]
fn sp_gateup_matches_host_reference() {
    let (p, d, w) = probe();
    let tn = rnd_vec(KB * d.h, 215, 1.2);
    p.write_stage(SpStage::Tn, &tn);
    p.run("sp_gateup");
    let got = p.read_stage(SpStage::Act);
    let g = matvec(KB, d.h, &tn, &w.wg, d.inter);
    let u = matvec(KB, d.h, &tn, &w.wu, d.inter);
    let want: Vec<f64> = g
        .iter()
        .zip(u.iter())
        .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
        .collect();
    let e = gate("sp_gateup", &got, &want, REL);
    println!("sp_gateup: max abs err {e:.3e}");
}

#[test]
fn sp_down_matches_host_reference() {
    let (p, d, w) = probe();
    let act = rnd_vec(KB * d.inter, 216, 1.0);
    let h1 = rnd_vec(KB * d.h, 217, 0.8);
    p.write_stage(SpStage::Act, &act);
    p.write_stage(SpStage::H1, &h1);
    p.run("sp_down");
    let got = p.read_stage(SpStage::H2);
    let mut want = matvec(KB, d.inter, &act, &w.wd, d.h);
    for (i, v) in want.iter_mut().enumerate() {
        *v += h1[i] as f64;
    }
    let e = gate("sp_down", &got, &want, REL);
    println!("sp_down: max abs err {e:.3e}");
}

#[test]
fn sp_rmsf_matches_host_reference() {
    let (p, d, w) = probe();
    let h2 = rnd_vec(KB * d.h, 218, 1.5);
    p.write_stage(SpStage::H2, &h2);
    p.run("sp_rmsf");
    let got = p.read_stage(SpStage::Xf);
    let want = rms_ref(&d, &h2, &w.lnf);
    let e = gate("sp_rmsf", &got, &want, REL);
    println!("sp_rmsf: max abs err {e:.3e}");
}

#[test]
fn sp_logits_matches_host_reference() {
    let (p, d, w) = probe();
    let xf = rnd_vec(KB * d.h, 219, 1.1);
    p.write_stage(SpStage::Xf, &xf);
    p.run("sp_logits");
    let got = p.read_stage(SpStage::Logits);
    let want = matvec(KB, d.h, &xf, &w.wlm, d.vocab);
    let e = gate("sp_logits", &got, &want, REL);
    println!("sp_logits: max abs err {e:.3e}");
}

#[test]
fn sp_argmax_matches_host_reference() {
    let (p, d, _w) = probe();
    let logits = rnd_vec(KB * d.vocab, 220, 2.0);
    p.write_stage(SpStage::Logits, &logits);
    p.run("sp_argmax");
    let got = p.read_amax();
    assert_eq!(got.len(), KB);

    let mut want = Vec::with_capacity(KB);
    for row in 0..KB {
        let mut order: Vec<(usize, f64)> = (0..d.vocab)
            .map(|v| (v, logits[row * d.vocab + v] as f64))
            .collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(
            order[0].1 - order[1].1 > 1e-3,
            "row {row}: top-1 and top-2 logits are {:.6} and {:.6}; a near-tie makes argmax \
             ambiguous and the gate meaningless",
            order[0].1,
            order[1].1
        );
        want.push(order[0].0 as u32);
    }
    assert!(
        want.iter().any(|v| *v != want[0]) || KB == 1,
        "every row picked the same index; the gate cannot see a row-stride bug"
    );
    assert_eq!(got, want, "sp_argmax rows differ from the host argmax");
    println!("sp_argmax: rows {got:?}");
}

#[test]
fn every_shipped_sp_entry_has_a_host_reference() {
    let mut found: Vec<String> = sp_wgsl()
        .lines()
        .filter_map(|line| line.trim().strip_prefix("fn "))
        .map(|rest| rest.split('(').next().unwrap().trim().to_string())
        .collect();
    found.sort();
    assert!(
        found.len() >= COVERED.len(),
        "parsed only {} entry points out of the sp_* shader; the parser is broken and this gate \
         would pass vacuously: {found:?}",
        found.len()
    );

    let mut shipped: Vec<String> = SP_ENTRIES.iter().map(|s| s.to_string()).collect();
    shipped.sort();
    let mut covered: Vec<String> = COVERED.iter().map(|s| s.to_string()).collect();
    covered.sort();

    assert_eq!(
        found, shipped,
        "the sp_* shader's entry points and SP_ENTRIES (the recorded pass table) disagree"
    );
    assert_eq!(
        found, covered,
        "a shipped sp_* entry has no host-reference test in this file"
    );
}

#[test]
fn probe_runs_every_entry_the_graph_records() {
    let (p, _d, _w) = probe();
    p.write_tokens(&[1, 2, 3]);
    for entry in SP_ENTRIES {
        p.run(entry);
    }
}

#[test]
#[should_panic(expected = "is not in the shipped pass table")]
fn probe_rejects_an_entry_the_graph_does_not_record() {
    let (p, _d, _w) = probe();
    p.run("sp_not_an_entry");
}
