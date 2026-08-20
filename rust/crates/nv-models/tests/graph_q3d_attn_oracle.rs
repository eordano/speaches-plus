#![cfg(feature = "wgpu")]

mod common;
use common::pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const ATTN_TAG: &str = "q3d:attn";
const ENTRY: &str = "q3w_attn_decode";

fn attn_source() -> String {
    let all = nv_models::qwen3_5_dense_wgpu::nozi_audit_sources();
    let hit = all
        .into_iter()
        .find(|(tag, _)| *tag == ATTN_TAG)
        .unwrap_or_else(|| {
            panic!(
                "qwen3_5_dense_wgpu::nozi_audit_sources() no longer exposes {ATTN_TAG}; this gate \
                 compiles the SHIPPED attention text and cannot fall back to a copy"
            )
        });
    let src = hit.1;
    assert!(
        src.contains(&format!("fn {ENTRY}(")),
        "{ATTN_TAG} no longer declares {ENTRY}; the entry moved and this gate is now testing \
         nothing"
    );
    src
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32;
        v - 1.0
    }
}

fn round_bf16(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

fn reference(p: &AdParams, q: &[f32], kc: &[f32], vc: &[f32], pos: u32) -> Vec<f64> {
    let hd = p.head_dim as usize;
    let n_kv = p.n_kv as usize;
    let total = pos as usize + 1;
    let mut out = vec![0f64; p.n_heads as usize * hd];
    for h in 0..p.n_heads as usize {
        let kv = h / p.group as usize;
        let mut scores = vec![0f64; total];
        for (t, s) in scores.iter_mut().enumerate() {
            let kbase = (t * n_kv + kv) * hd;
            let mut dot = 0f64;
            for d in (0..hd).rev() {
                dot += kc[kbase + d] as f64 * q[h * hd + d] as f64;
            }
            *s = dot * p.scale as f64;
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut z = 0f64;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            z += *s;
        }
        for d in 0..hd {
            let mut acc = 0f64;
            for t in (0..total).rev() {
                acc += scores[t] * vc[(t * n_kv + kv) * hd + d] as f64;
            }
            out[h * hd + d] = acc / z;
        }
    }
    out
}

struct Case {
    label: &'static str,
    p: AdParams,
    pos: u32,
    q: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (label, n_heads, n_kv, hd, max_seq, pos) in [
        ("gqa g2 hd32 pos7", 4u32, 2u32, 32u32, 16u32, 7u32),
        ("mha g1 hd64 pos11", 2, 2, 64, 16, 11),
        ("mqa g8 hd128 pos5", 8, 1, 128, 8, 5),
        ("gqa g2 hd32 pos0", 4, 2, 32, 16, 0),
    ] {
        let mut r = Lcg::new(0xa77e_0000 ^ (pos as u64) << 8 ^ hd as u64);
        let p = AdParams {
            n_heads,
            n_kv,
            head_dim: hd,
            max_seq,
            group: n_heads / n_kv,
            pad0: 0,
            pad1: 0,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        let q: Vec<f32> = (0..n_heads * hd).map(|_| round_bf16(r.next())).collect();
        let kc: Vec<f32> = (0..max_seq * n_kv * hd)
            .map(|_| round_bf16(r.next()))
            .collect();
        let vc: Vec<f32> = (0..max_seq * n_kv * hd)
            .map(|_| round_bf16(r.next()))
            .collect();
        out.push(Case {
            label,
            p,
            pos,
            q,
            kc,
            vc,
        });
    }
    out
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_attn_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

#[test]
fn q3w_attn_decode_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[q3d-attn-oracle] adapter: {}", ctx.info.name);
    let src = attn_source();

    let mut worst_rel = 0f64;
    for c in cases() {
        let hd = c.p.head_dim as usize;
        let n_heads = c.p.n_heads as usize;
        let want = reference(&c.p, &c.q, &c.kc, &c.vc, c.pos);

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
        let scores_b =
            dispatch::storage_zeroed(ctx, "scores", (n_heads * c.p.max_seq as usize * 4) as u64);
        let out_b = dispatch::storage_zeroed(ctx, "out", (n_heads * hd * 4) as u64);
        let pos_b = dispatch::storage_from_slice(ctx, "pos", &[c.pos as i32]);
        let par_b = dispatch::uniform_from(ctx, "adp", &c.p);

        dispatch::run(
            ctx,
            "q3d-attn-oracle",
            &src,
            ENTRY,
            &[
                (20, &q_b),
                (21, &kc_b),
                (22, &vc_b),
                (23, &scores_b),
                (24, &out_b),
                (25, &pos_b),
                (26, &par_b),
            ],
            (c.p.n_heads, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {ENTRY}: {e}", c.label));

        let got: Vec<f32> =
            dispatch::read_back(ctx, &out_b, n_heads * hd).expect("read back attention output");

        let got_max = got.iter().fold(0f32, |a, b| a.max(b.abs()));
        assert!(
            got_max > 1e-3,
            "{}: kernel output is all-but-zero (max |out| {got_max:e}); the comparison would be \
             zeros against zeros",
            c.label
        );

        let scale = ref_max.max(1e-6);
        let mut rel = 0f64;
        let mut worst_at = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (*g as f64 - *w).abs() / scale;
            if d > rel {
                rel = d;
                worst_at = i;
            }
        }
        worst_rel = worst_rel.max(rel);
        eprintln!(
            "[q3d-attn-oracle] {}: rel={rel:.3e} at lane {worst_at} (ref_max={ref_max:.4}, \
             total={} positions)",
            c.label,
            c.pos + 1
        );
        assert!(
            rel < 1e-5,
            "{}: {ENTRY} diverged from the f64 host reference (rel {rel:e} at lane {worst_at}). \
             The kernel decodes bf16 into f32 and accumulates in f32; measured worst over this \
             corpus is 1.8e-7, so 1e-5 is ~55x of the rounding floor and anything above it is a \
             defect. Contrast the graph-level fixture, whose rel < 0.05 on 4-layer logits stayed \
             green while the softmax denominator was deleted from this kernel.",
            c.label
        );
    }
    eprintln!("[q3d-attn-oracle] worst relative error across cases: {worst_rel:.3e}");
}

#[test]
fn the_corpus_contains_a_case_where_the_softmax_denominator_is_not_one() {
    let mut best = 0f64;
    for c in cases() {
        let hd = c.p.head_dim as usize;
        let n_kv = c.p.n_kv as usize;
        let total = c.pos as usize + 1;
        for h in 0..c.p.n_heads as usize {
            let kv = h / c.p.group as usize;
            let mut scores = vec![0f64; total];
            for (t, s) in scores.iter_mut().enumerate() {
                let kbase = (t * n_kv + kv) * hd;
                let mut dot = 0f64;
                for d in 0..hd {
                    dot += c.kc[kbase + d] as f64 * c.q[h * hd + d] as f64;
                }
                *s = dot * c.p.scale as f64;
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = scores.iter().map(|s| (s - m).exp()).sum();
            best = best.max((z - 1.0).abs());
        }
    }
    eprintln!("[q3d-attn-oracle] worst |z - 1| over the corpus: {best:.4}");
    assert!(
        best > 1.0,
        "every case has a softmax denominator within 1.0 of unity (worst |z-1| {best}); dropping \
         `/ z` would then be a no-op and this suite could not catch it"
    );
}
