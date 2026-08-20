#![cfg(feature = "wgpu")]

mod common;
use common::pack;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitG4w as Lcg;

const TAG: &str = "g4w:head_prep";
const ENTRY: &str = "g4w_head_prep";

const WHY: &str = "g4w_head_prep is the fused attention front-end of the Gemma-4 dense graph: one \
     workgroup per head does the q/k norm, the rope, the per-head amax and the E4M3 quantisation \
     of the key and value that the paged cache then stores, replacing four separate dispatches. \
     The entry census records it as named by exactly one test, wgpu_nozi_graph_census, which \
     audits zero-initialisation policy over its shared arrays and asserts nothing numeric -- so \
     no artifact in the tree checks a single value it produces. It is also where the most state \
     lives: the rope position and the cache slot both come in as buffers, and both default to \
     zero, which is precisely the value at which a dropped index is invisible.";

const ORACLE_IS_NOT_THE_KERNEL: &str = "The reference is an f64 host chain written from the \
     definition of each stage: an RMS norm over the head with its own weight, a rotate-half rope \
     against the supplied cosine and sine tables, the head's largest magnitude divided by the \
     E4M3 maximum as the store scale, and a nearest-with-ties-to-even E4M3 code for each element. \
     The E4M3 encoder is a SEARCH over the 256 codes of the format, decoded from the format's own \
     definition -- it is not a transliteration of the shader's bit manipulation, so a defect in \
     that manipulation is what this suite is able to see.";

const NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO: &str = "The rope position and the KV slot are both \
     driven at NON-ZERO values, and a screen asserts it. At position zero every cosine is one and \
     every sine is zero, so the rope is the identity and dropping it entirely passes; at slot \
     zero the destination index slot * n_kv + head collapses to head, so dropping the slot passes \
     too. A fixture that exercised only the default state would gate neither, which is the second \
     of the two ways a test in this tree has been vacuous before. The destination carries more \
     slots than the one written and the rest stay poisoned, so a collapsed slot index is caught \
     by where it writes rather than by what it writes.";

const EPS: f32 = 1e-6;

const E4M3_MAX: f64 = 448.0;

const POISON_U32: u32 = 0xdead_beef;
const POISON_F32: f32 = -1.0e30;

const SLOTS: usize = 4;

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_g4w_head_prep_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn source() -> String {
    let all = nv_models::gemma4_wgpu::nozi_audit_sources();
    let hit = all.into_iter().find(|(t, _)| *t == TAG).unwrap_or_else(|| {
        panic!(
            "gemma4_wgpu::nozi_audit_sources() no longer exposes {TAG}; this gate compiles the \
             SHIPPED text and cannot fall back to a copy"
        )
    });
    let src = hit.1;
    assert!(
        src.contains(&format!("fn {ENTRY}(")),
        "{TAG} no longer declares {ENTRY}; the entry moved and this gate is now testing nothing"
    );
    assert!(
        !src.contains("HP_MAXW"),
        "{TAG} still carries the unsubstituted HP_MAXW placeholder, so the source this gate \
         compiles is not the one build_fuse_pipelines builds"
    );
    src
}

fn e4m3_decode(code: u8) -> Option<f64> {
    if code & 0x7f == 0x7f {
        return None;
    }
    let e = ((code >> 3) & 15) as i32;
    let m = (code & 7) as f64;
    let mag = if e == 0 {
        m * 0.001953125
    } else {
        (2f64).powi(e - 7) * (1.0 + m / 8.0)
    };
    Some(if code & 0x80 != 0 { -mag } else { mag })
}

fn e4m3_encode(x: f32) -> u8 {
    let sign = if x.to_bits() >> 31 == 1 { 0x80u8 } else { 0 };
    if x.is_nan() {
        return sign | 0x7f;
    }
    let a = (x.abs() as f64).min(E4M3_MAX);
    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for code in 0u8..0x7f {
        let v = e4m3_decode(code).expect("code below 0x7f is finite");
        let err = (a - v).abs();
        if err < best_err || (err == best_err && code & 1 == 0) {
            best_err = err;
            best = code;
        }
    }
    sign | best
}

fn dec(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

fn enc(v: f64) -> u16 {
    half::bf16::from_f32(v as f32).to_bits()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HpParams {
    n_q: u32,
    n_kv: u32,
    head_dim: u32,
    half_dim: u32,
    eps: f32,
    words: u32,
    out_words: u32,
    pad0: u32,
}

struct Case {
    label: &'static str,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    pos: usize,
    slot: usize,
    rotary: usize,
    qa: Vec<u16>,
    ka: Vec<u16>,
    va: Vec<u16>,
    qn: Vec<u16>,
    kn: Vec<u16>,
    vn: Vec<u16>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

struct Expect {
    q_out: Vec<f32>,
    kq: Vec<u32>,
    ks: Vec<f32>,
    vq: Vec<u32>,
    vs: Vec<f32>,
    subnormal_codes: usize,
    saturated_codes: usize,
    zero_amax_heads: usize,
}

#[allow(clippy::too_many_arguments)]
impl Case {
    fn build(
        label: &'static str,
        n_q: usize,
        n_kv: usize,
        hd: usize,
        pos: usize,
        slot: usize,
        rotary: usize,
        seed: u64,
    ) -> Self {
        assert!(hd.is_multiple_of(4), "{label}: E4M3 packs four elements per word");
        assert!(hd <= 256, "{label}: the audited head-prep source sizes its shared rows for 256");
        assert!(rotary <= hd / 2, "{label}: more rotary angles than the half dimension");
        let mut rng = Lcg::new(seed);
        let half = hd / 2;
        let mut cos = vec![1.0f32; (pos + 1) * half];
        let mut sin = vec![0.0f32; (pos + 1) * half];
        for p in 0..=pos {
            for i in 0..rotary {
                let inv = 1.0f32 / 10000f32.powf((i as f32 * 2.0) / hd as f32);
                let theta = p as f32 * inv;
                cos[p * half + i] = theta.cos();
                sin[p * half + i] = theta.sin();
            }
        }
        Self {
            label,
            n_q,
            n_kv,
            hd,
            pos,
            slot,
            rotary,
            qa: rng.bf16_vec(n_q * hd, 0.4),
            ka: rng.bf16_vec(n_kv * hd, 0.4),
            va: rng.bf16_vec(n_kv * hd, 0.4),
            qn: rng.norm_vec(hd),
            kn: rng.norm_vec(hd),
            vn: rng.norm_vec(hd),
            cos,
            sin,
        }
    }

    fn zero_head(mut self, which: usize) -> Self {
        for i in 0..self.hd {
            self.va[which * self.hd + i] = 0;
        }
        self
    }

    fn tiny_value_elements(mut self, head: usize, at: &[usize], magnitude: f32) -> Self {
        for i in at {
            self.va[head * self.hd + i] = half::bf16::from_f32(magnitude).to_bits();
        }
        self
    }

    fn normed(&self, src: &[u16], w: &[u16], head: usize) -> Vec<u16> {
        let hd = self.hd;
        let row = &src[head * hd..(head + 1) * hd];
        let sum: f64 = row.iter().map(|b| dec(*b) * dec(*b)).sum();
        let rms = 1.0 / (EPS as f64 + sum / hd as f64).sqrt();
        (0..hd).map(|i| enc(dec(row[i]) * rms * dec(w[i]))).collect()
    }

    fn roped(&self, a: &[u16]) -> Vec<u16> {
        let hd = self.hd;
        let half = hd / 2;
        let base = self.pos * half;
        (0..hd)
            .map(|elem| {
                let (c, s, x, y) = if elem < half {
                    (
                        self.cos[base + elem] as f64,
                        self.sin[base + elem] as f64,
                        dec(a[elem]),
                        dec(a[elem + half]),
                    )
                } else {
                    let pair = elem - half;
                    (
                        self.cos[base + pair] as f64,
                        self.sin[base + pair] as f64,
                        dec(a[pair]),
                        dec(a[elem]),
                    )
                };
                if elem < half {
                    enc(x * c - y * s)
                } else {
                    enc(x * s + y * c)
                }
            })
            .collect()
    }

    fn expect(&self) -> Expect {
        let hd = self.hd;
        let mut e = Expect {
            q_out: vec![POISON_F32; self.n_q * hd],
            kq: vec![POISON_U32; SLOTS * self.n_kv * hd / 4],
            ks: vec![POISON_F32; SLOTS * self.n_kv],
            vq: vec![POISON_U32; SLOTS * self.n_kv * hd / 4],
            vs: vec![POISON_F32; SLOTS * self.n_kv],
            subnormal_codes: 0,
            saturated_codes: 0,
            zero_amax_heads: 0,
        };
        for h in 0..self.n_q {
            let b = self.roped(&self.normed(&self.qa, &self.qn, h));
            for (i, bits) in b.iter().enumerate() {
                e.q_out[h * hd + i] = dec(*bits) as f32;
            }
        }
        for h in 0..self.n_kv {
            for kind in [1usize, 2] {
                let (src, w) = if kind == 1 {
                    (&self.ka, &self.kn)
                } else {
                    (&self.va, &self.vn)
                };
                let a = self.normed(src, w, h);
                let b = if kind == 1 { self.roped(&a) } else { a };
                let amax = b.iter().map(|x| dec(*x).abs()).fold(0f64, f64::max);
                let positive = amax > 0.0;
                if !positive {
                    e.zero_amax_heads += 1;
                }
                let scale = if positive {
                    (amax / E4M3_MAX) as f32
                } else {
                    1.0f32
                };
                let inv = if positive {
                    (E4M3_MAX / amax) as f32
                } else {
                    1.0f32
                };
                let sidx = self.slot * self.n_kv + h;
                let dst = sidx * hd / 4;
                for w4 in 0..hd / 4 {
                    let mut packed = 0u32;
                    for j in 0..4 {
                        let v = (dec(b[w4 * 4 + j]) * inv as f64) as f32;
                        let code = e4m3_encode(v);
                        if code & 0x78 == 0 && code & 7 != 0 {
                            e.subnormal_codes += 1;
                        }
                        if code & 0x7f == 0x7e {
                            e.saturated_codes += 1;
                        }
                        packed |= (code as u32) << (8 * j);
                    }
                    if kind == 1 {
                        e.kq[dst + w4] = packed;
                    } else {
                        e.vq[dst + w4] = packed;
                    }
                }
                if kind == 1 {
                    e.ks[sidx] = scale;
                } else {
                    e.vs[sidx] = scale;
                }
            }
        }
        e
    }

    fn params(&self) -> HpParams {
        HpParams {
            n_q: self.n_q as u32,
            n_kv: self.n_kv as u32,
            head_dim: self.hd as u32,
            half_dim: (self.hd / 2) as u32,
            eps: EPS,
            words: (self.hd / 2) as u32,
            out_words: (self.hd / 4) as u32,
            pad0: 0,
        }
    }
}

struct Got {
    q_out: Vec<f32>,
    kq: Vec<u32>,
    ks: Vec<f32>,
    vq: Vec<u32>,
    vs: Vec<f32>,
}

fn run(ctx: &WgpuContext, pipe: &wgpu::ComputePipeline, c: &Case) -> anyhow::Result<Got> {
    let hd = c.hd;
    let words = SLOTS * c.n_kv * hd / 4;
    let scales = SLOTS * c.n_kv;
    let qa = dispatch::storage_from_slice(ctx, "hp-qa", &pack(&c.qa));
    let ka = dispatch::storage_from_slice(ctx, "hp-ka", &pack(&c.ka));
    let va = dispatch::storage_from_slice(ctx, "hp-va", &pack(&c.va));
    let qn = dispatch::storage_from_slice(ctx, "hp-qn", &pack(&c.qn));
    let kn = dispatch::storage_from_slice(ctx, "hp-kn", &pack(&c.kn));
    let vn = dispatch::storage_from_slice(ctx, "hp-vn", &pack(&c.vn));
    let qout = dispatch::storage_from_slice(ctx, "hp-qout", &vec![POISON_F32; c.n_q * hd]);
    let kq = dispatch::storage_from_slice(ctx, "hp-kq", &vec![POISON_U32; words]);
    let ks = dispatch::storage_from_slice(ctx, "hp-ks", &vec![POISON_F32; scales]);
    let vq = dispatch::storage_from_slice(ctx, "hp-vq", &vec![POISON_U32; words]);
    let vs = dispatch::storage_from_slice(ctx, "hp-vs", &vec![POISON_F32; scales]);
    let cos = dispatch::storage_from_slice(ctx, "hp-cos", &c.cos);
    let sin = dispatch::storage_from_slice(ctx, "hp-sin", &c.sin);
    let pos = dispatch::storage_from_slice(ctx, "hp-pos", &[c.pos as i32]);
    let kvs = dispatch::storage_from_slice(ctx, "hp-kvstart", &[c.slot as i32]);
    let p = dispatch::uniform_from(ctx, "hp-p", &c.params());
    let grid = dispatch::workgroup_count_1d(ctx, (c.n_q + 2 * c.n_kv) as u64, 1);
    dispatch::dispatch(
        ctx,
        pipe,
        &[
            (0, &qa),
            (1, &ka),
            (2, &va),
            (3, &qn),
            (4, &kn),
            (5, &vn),
            (6, &qout),
            (7, &kq),
            (8, &ks),
            (9, &vq),
            (10, &vs),
            (11, &cos),
            (12, &sin),
            (13, &pos),
            (14, &kvs),
            (15, &p),
        ],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let rbu = |b: &wgpu::Buffer, n: usize| -> anyhow::Result<Vec<u32>> {
        dispatch::read_back(ctx, b, n).map_err(|e| anyhow::anyhow!("{e}"))
    };
    let rbf = |b: &wgpu::Buffer, n: usize| -> anyhow::Result<Vec<f32>> {
        dispatch::read_back(ctx, b, n).map_err(|e| anyhow::anyhow!("{e}"))
    };
    Ok(Got {
        q_out: rbf(&qout, c.n_q * hd)?,
        kq: rbu(&kq, words)?,
        ks: rbf(&ks, scales)?,
        vq: rbu(&vq, words)?,
        vs: rbf(&vs, scales)?,
    })
}

fn diff(got: &Got, e: &Expect) -> Option<String> {
    for (i, (g, w)) in got.q_out.iter().zip(e.q_out.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Some(format!("q_out[{i}] is {g:e}, the f64 chain says {w:e}"));
        }
    }
    for (name, gv, wv) in [("k", &got.kq, &e.kq), ("v", &got.vq, &e.vq)] {
        for (i, (g, w)) in gv.iter().zip(wv.iter()).enumerate() {
            if g != w {
                return Some(format!(
                    "{name} E4M3 word {i} is {g:#010x}, the f64 chain says {w:#010x}"
                ));
            }
        }
    }
    for (name, gv, wv) in [("k", &got.ks, &e.ks), ("v", &got.vs, &e.vs)] {
        for (i, (g, w)) in gv.iter().zip(wv.iter()).enumerate() {
            if g.to_bits() != w.to_bits() {
                return Some(format!(
                    "{name} store scale {i} is {g:e}, the f64 chain says {w:e}"
                ));
            }
        }
    }
    None
}

fn corpus() -> Vec<Case> {
    vec![
        Case::build("gemma4 sliding head, partial rope", 4, 2, 128, 37, 2, 16, 0x0000_5111)
            .tiny_value_elements(0, &[5, 6, 70], 1.0e-6),
        Case::build("gemma4 global head at the audited maximum", 2, 1, 256, 11, 1, 32, 0x0000_6222),
        Case::build("full rotary, narrow head", 8, 4, 64, 5, 3, 32, 0x0000_7333).zero_head(1),
    ]
}

fn screen(cases: &[Case]) {
    let mut subnormal = 0usize;
    let mut saturated = 0usize;
    let mut zero_amax = 0usize;
    for c in cases {
        assert!(
            c.pos > 0,
            "{}: the rope position is zero, where every cosine is one and every sine is zero and \
             the rotation is the identity. {NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}",
            c.label
        );
        assert!(
            c.slot > 0 && c.slot < SLOTS,
            "{}: the KV slot must be non-zero and inside the destination, or the slot term of the \
             destination index is untested. {NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}",
            c.label
        );
        assert!(
            c.rotary > 0,
            "{}: no rotary angles, so the rope tables are the identity. \
             {NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}",
            c.label
        );
        let e = c.expect();
        subnormal += e.subnormal_codes;
        saturated += e.saturated_codes;
        zero_amax += e.zero_amax_heads;
    }
    for (what, n) in [
        ("an element that encodes to a subnormal E4M3 code", subnormal),
        ("an element that saturates the E4M3 range", saturated),
        ("a head whose largest magnitude is zero", zero_amax),
    ] {
        assert!(
            n > 0,
            "the corpus never produces {what}, so that branch of the quantiser is untested while \
             every case passes. {NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}"
        );
    }
    eprintln!(
        "[head-prep-oracle] corpus reaches {subnormal} subnormal E4M3 codes, {saturated} \
         saturated ones and {zero_amax} heads with a zero maximum"
    );
}

const MUTANTS: [(&str, &str, &str); 7] = [
    (
        "rope-first-half-adds-the-sine-term-instead-of-subtracting-it",
        "        return fma(a, c, -(b * s));",
        "        return fma(a, c, (b * s));",
    ),
    (
        "value-head-is-rotated-like-a-key",
        "    if (kind == 2u) {",
        "    if (kind == 3u) {",
    ),
    (
        "kv-destination-drops-the-slot",
        "    let sidx = slot * n_kv + head;",
        "    let sidx = head;",
    ),
    (
        "store-scale-and-its-inverse-are-exchanged",
        "    let inv_scale = select(1.0, hp_div_rn(HP_E4M3_MAX, amax), positive);",
        "    let inv_scale = select(1.0, hp_div_rn(amax, HP_E4M3_MAX), positive);",
    ),
    (
        "head-rms-divides-by-the-word-count-instead-of-the-element-count",
        "        let mean = hp_div_rn(sum, f32(hp_params.head_dim));",
        "        let mean = hp_div_rn(sum, f32(hp_params.words));",
    ),
    (
        "head-maximum-takes-the-signed-value",
        "        let a = abs(hp_b_at(d));",
        "        let a = hp_b_at(d);",
    ),
    (
        "query-output-writes-the-low-half-into-both-lanes",
        "                hp_qout[q_base + elem + 1u] = bf16_hi(word);",
        "                hp_qout[q_base + elem + 1u] = bf16_lo(word);",
    ),
];

const SLOT_MUTANT: usize = 2;

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped head-prep source: {from:?}. This gate is \
         worthless if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn pipeline(ctx: &WgpuContext, src: &str) -> anyhow::Result<wgpu::ComputePipeline> {
    dispatch::compute_pipeline(ctx, "g4w-head-prep-probe", src, ENTRY)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_head_prep() {
    let src = source();
    for (name, from, to) in MUTANTS {
        assert!(
            src.contains(from),
            "anchor for mutant {name} is gone from {TAG}: {from:?}. A mutant whose anchor rotted \
             is silently inert, and the GPU tests that would have caught that do not run on a box \
             with no adapter -- which is why this check is CPU-only and unconditional."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
    }
    screen(&corpus());
    eprintln!("{WHY}\n{ORACLE_IS_NOT_THE_KERNEL}\n{NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}");
}

#[test]
fn the_host_e4m3_encoder_round_trips_every_finite_code() {
    for code in 0u8..=0xff {
        let Some(v) = e4m3_decode(code) else {
            continue;
        };
        if v == 0.0 {
            continue;
        }
        assert_eq!(
            e4m3_encode(v as f32),
            code,
            "the host E4M3 encoder does not return code {code:#04x} for its own decoded value \
             {v:e}; a reference that cannot round-trip the format is not a reference. \
             {ORACLE_IS_NOT_THE_KERNEL}"
        );
    }
    assert_eq!(
        e4m3_encode(1.0e30),
        0x7e,
        "the host E4M3 encoder must saturate rather than overflow, as the shipped one does"
    );
    assert_eq!(
        e4m3_encode(-1.0e30),
        0xfe,
        "the host E4M3 encoder must saturate on the negative side too"
    );
}

#[test]
fn head_prep_matches_an_f64_host_chain() {
    let ctx = ctx();
    eprintln!("[head-prep-oracle] adapter: {}\n{WHY}", ctx.info.name);
    let src = source();
    let pipe = pipeline(ctx, &src).expect("clean pipeline");
    for c in corpus() {
        let got = run(ctx, &pipe, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        let e = c.expect();
        if let Some(d) = diff(&got, &e) {
            panic!(
                "[{}] {ENTRY} disagrees with the f64 host chain (n_q={} n_kv={} head_dim={} \
                 pos={} slot={}): {d}. {ORACLE_IS_NOT_THE_KERNEL}",
                c.label, c.n_q, c.n_kv, c.hd, c.pos, c.slot
            );
        }
        eprintln!(
            "[{}] {ENTRY}: {} q heads and {} kv heads match exactly at pos={} slot={} rotary={}",
            c.label, c.n_q, c.n_kv, c.pos, c.slot, c.rotary
        );
    }
}

#[test]
fn every_head_prep_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = source();
    let cases = corpus();
    let expects: Vec<Expect> = cases.iter().map(|c| c.expect()).collect();
    for (name, from, to) in MUTANTS {
        let bad = mutate(&src, from, to);
        let pipe = pipeline(ctx, &bad).expect("mutant pipeline");
        let mut caught_by: Vec<&str> = Vec::new();
        for (c, e) in cases.iter().zip(expects.iter()) {
            let hit = match run(ctx, &pipe, c) {
                Ok(got) => diff(&got, e).is_some(),
                Err(_) => true,
            };
            if hit {
                caught_by.push(c.label);
            }
        }
        assert!(
            !caught_by.is_empty(),
            "mutant {name} was NOT caught by any case in the corpus. A fused attention front-end \
             that a wrong rotation or a wrong destination passes is not a gate. {WHY} \
             {NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}"
        );
        eprintln!("MUTANT {name}: caught by {caught_by:?}");
    }
}

#[test]
fn a_collapsed_slot_index_is_caught_by_where_it_writes() {
    let ctx = ctx();
    let src = source();
    let (name, from, to) = MUTANTS[SLOT_MUTANT];
    assert_eq!(name, "kv-destination-drops-the-slot");
    let bad = mutate(&src, from, to);
    let pipe = pipeline(ctx, &bad).expect("mutant pipeline");
    for c in corpus() {
        let got = run(ctx, &pipe, &c).expect("slot mutant dispatch");
        let sidx0 = c.slot * c.n_kv;
        let touched = got.ks[..c.n_kv].iter().any(|s| s.to_bits() != POISON_F32.to_bits());
        let vacated = got.ks[sidx0..sidx0 + c.n_kv]
            .iter()
            .all(|s| s.to_bits() == POISON_F32.to_bits());
        assert!(
            touched && vacated,
            "[{}] dropping the slot term left the destination unchanged, so slot={} does not \
             separate the two indices and the corpus is back to the default state. \
             {NEITHER_POSITION_NOR_SLOT_MAY_BE_ZERO}",
            c.label,
            c.slot
        );
    }
    eprintln!("MUTANT {name}: every case writes into slot 0 and leaves its own slot poisoned");
}
