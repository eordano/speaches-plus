#![cfg(feature = "wgpu")]

mod common;
use common::CkParams;
use common::pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitVec as Lcg;

const ENTRY: &str = "q3w_attn_decode_m";

fn source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    assert!(
        src.contains(&format!("fn {ENTRY}(")),
        "the shipped prefill source no longer declares {ENTRY}; the entry moved and this gate is \
         now testing nothing"
    );
    src
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdmParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

fn round_bf16(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

struct Case {
    label: &'static str,
    p: AdmParams,
    ck: CkParams,
    q: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

fn reference(c: &Case) -> Vec<f64> {
    let hd = c.p.head_dim as usize;
    let n_kv = c.p.n_kv as usize;
    let n_heads = c.p.n_heads as usize;
    let m = c.ck.m_live as usize;
    let mut out = vec![0f64; m * n_heads * hd];
    for tk in 0..m {
        let total = c.ck.base as usize + tk + 1;
        let qrow = tk * n_heads * hd;
        for h in 0..n_heads {
            let kv = h / c.p.group as usize;
            let mut scores = vec![0f64; total];
            for (t, s) in scores.iter_mut().enumerate() {
                let kbase = (t * n_kv + kv) * hd;
                let mut dot = 0f64;
                for d in (0..hd).rev() {
                    dot += c.kc[kbase + d] as f64 * c.q[qrow + h * hd + d] as f64;
                }
                *s = dot * c.p.scale as f64;
            }
            let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0f64;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                z += *s;
            }
            for d in 0..hd {
                let mut acc = 0f64;
                for t in (0..total).rev() {
                    acc += scores[t] * c.vc[(t * n_kv + kv) * hd + d] as f64;
                }
                out[qrow + h * hd + d] = acc / z;
            }
        }
    }
    out
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (label, n_heads, n_kv, hd, max_seq, base, m) in [
        ("gqa g2 hd32 base5 m4", 4u32, 2u32, 32u32, 24u32, 5u32, 4u32),
        ("mha g1 hd64 base0 m3", 2, 2, 64, 16, 0, 3),
        ("mqa g8 hd128 base9 m2", 8, 1, 128, 16, 9, 2),
        ("gqa g2 hd32 base0 m1", 4, 2, 32, 16, 0, 1),
    ] {
        let mut r = Lcg::new(0xa77e_5000 ^ (base as u64) << 12 ^ (hd as u64) << 4 ^ m as u64);
        let p = AdmParams {
            n_heads,
            n_kv,
            head_dim: hd,
            max_seq,
            group: n_heads / n_kv,
            pad0: 0,
            pad1: 0,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        let ck = CkParams {
            m_live: m,
            base,
            pad0: 0,
            pad1: 0,
        };
        assert!(
            base + m <= max_seq,
            "{label}: the KV cache must hold every position this dispatch reads"
        );
        let q: Vec<f32> = (0..m * n_heads * hd)
            .map(|_| round_bf16(r.next()))
            .collect();
        let kc: Vec<f32> = (0..max_seq * n_kv * hd)
            .map(|_| round_bf16(r.next()))
            .collect();
        let vc: Vec<f32> = (0..max_seq * n_kv * hd)
            .map(|_| round_bf16(r.next()))
            .collect();
        out.push(Case {
            label,
            p,
            ck,
            q,
            kc,
            vc,
        });
    }
    out
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_prefill_attn_oracle needs a real wgpu adapter; a skipped numeric gate reads as \
         a passed one, so this panics rather than returning early",
    )
}

#[test]
fn q3w_attn_decode_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[q3d-pf-attn-oracle] adapter: {}", ctx.info.name);
    let src = source();

    let mut worst_rel = 0f64;
    for c in cases() {
        let hd = c.p.head_dim as usize;
        let n_heads = c.p.n_heads as usize;
        let m = c.ck.m_live as usize;
        let want = reference(&c);

        let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
        assert!(
            ref_max > 1e-3,
            "{}: reference output is degenerate (max |out| {ref_max:e}); a tolerance check \
             against it would pass on any kernel",
            c.label
        );

        let q_b = dispatch::storage_from_slice(ctx, "q", &pack_bf16(&c.q));
        let kc_b = dispatch::storage_from_slice(ctx, "kc", &pack_bf16(&c.kc));
        let vc_b = dispatch::storage_from_slice(ctx, "vc", &pack_bf16(&c.vc));
        let scores_b = dispatch::storage_zeroed(
            ctx,
            "scores",
            (m * n_heads * c.p.max_seq as usize * 4) as u64,
        );
        let out_b = dispatch::storage_zeroed(ctx, "out", (m * n_heads * hd * 4) as u64);
        let par_b = dispatch::uniform_from(ctx, "admp", &c.p);
        let ck_b = dispatch::uniform_from(ctx, "ck", &c.ck);

        dispatch::run(
            ctx,
            "q3d-pf-attn-oracle",
            &src,
            ENTRY,
            &[
                (80, &q_b),
                (81, &kc_b),
                (82, &vc_b),
                (83, &scores_b),
                (84, &out_b),
                (85, &par_b),
                (86, &ck_b),
            ],
            (c.p.n_heads, m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {ENTRY}: {e}", c.label));

        let got: Vec<f32> = dispatch::read_back(ctx, &out_b, m * n_heads * hd)
            .expect("read back prefill attention output");

        let got_max = got.iter().fold(0f32, |a, b| a.max(b.abs()));
        assert!(
            got_max > 1e-3,
            "{}: kernel output is all-but-zero (max |out| {got_max:e}); the comparison would be \
             zeros against zeros",
            c.label
        );

        let mut rel = 0f64;
        let mut worst_at = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (*g as f64 - *w).abs() / ref_max;
            if d > rel {
                rel = d;
                worst_at = i;
            }
        }
        worst_rel = worst_rel.max(rel);
        eprintln!(
            "[q3d-pf-attn-oracle] {}: rel={rel:.3e} at lane {worst_at} (ref_max={ref_max:.4}, \
             rows={m}, base={})",
            c.label, c.ck.base
        );
        assert!(
            rel < 1e-5,
            "{}: {ENTRY} diverged from the f64 host reference (rel {rel:e} at lane {worst_at}). \
             The kernel decodes bf16 into f32 and accumulates in f32; every green run of this \
             suite prints the measured floor, and 1e-5 sits far above it. Contrast the \
             graph-level fixture, whose rel < 0.05 on 4-layer logits stayed green while the \
             softmax denominator was deleted from this kernel's decode twin.",
            c.label
        );
    }
    eprintln!("[q3d-pf-attn-oracle] worst relative error across cases: {worst_rel:.3e}");
}

#[test]
fn the_corpus_contains_a_row_where_the_softmax_denominator_is_not_one() {
    let mut best = 0f64;
    for c in cases() {
        let hd = c.p.head_dim as usize;
        let n_kv = c.p.n_kv as usize;
        let n_heads = c.p.n_heads as usize;
        for tk in 0..c.ck.m_live as usize {
            let total = c.ck.base as usize + tk + 1;
            let qrow = tk * n_heads * hd;
            for h in 0..n_heads {
                let kv = h / c.p.group as usize;
                let mut scores = vec![0f64; total];
                for (t, s) in scores.iter_mut().enumerate() {
                    let kbase = (t * n_kv + kv) * hd;
                    let mut dot = 0f64;
                    for d in 0..hd {
                        dot += c.kc[kbase + d] as f64 * c.q[qrow + h * hd + d] as f64;
                    }
                    *s = dot * c.p.scale as f64;
                }
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = scores.iter().map(|s| (s - mx).exp()).sum();
                best = best.max((z - 1.0).abs());
            }
        }
    }
    eprintln!("[q3d-pf-attn-oracle] worst |z - 1| over the corpus: {best:.4}");
    assert!(
        best > 1.0,
        "every row has a softmax denominator within 1.0 of unity (worst |z-1| {best}); dropping \
         `/ z` would then be a no-op and this suite could not catch it"
    );
}

#[test]
fn the_corpus_exercises_a_nonzero_chunk_base_and_more_than_one_row() {
    let max_base = cases().iter().map(|c| c.ck.base).max().unwrap_or(0);
    let max_m = cases().iter().map(|c| c.ck.m_live).max().unwrap_or(0);
    eprintln!("[q3d-pf-attn-oracle] max base {max_base}, max m_live {max_m}");
    assert!(
        max_base > 0 && max_m > 1,
        "corpus has max base {max_base} and max m_live {max_m}; with base == 0 everywhere the \
         chunk offset is unobservable, and with m_live == 1 everywhere this entry is just its \
         decode twin"
    );
}
