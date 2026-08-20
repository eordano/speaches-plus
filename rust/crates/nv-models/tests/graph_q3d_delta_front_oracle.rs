#![cfg(feature = "wgpu")]

mod common;
use common::bf16;
use common::bf16_bits_from_f64 as bf16_bits;
use common::CkParams;
use common::pack_bf16_from_f64 as pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const QKV_ENTRY: &str = "q3w_delta_qkv";
const QKV_M_ENTRY: &str = "q3w_delta_qkv_m";
const GATING_M_ENTRY: &str = "q3w_delta_gating_m";
const CONV_M_ENTRY: &str = "q3w_delta_conv_m";
const SHIFT_ENTRY: &str = "q3w_delta_conv_shift";
const OUT_M_ENTRY: &str = "q3w_delta_out_m";

const TOL_F32: f64 = 1e-5;

fn decode_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_delta_source();
    assert!(
        src.contains(&format!("fn {QKV_ENTRY}(")),
        "the shipped delta source no longer declares {QKV_ENTRY}; the entry moved and this gate is \
         now testing nothing"
    );
    src
}

fn prefill_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    for e in [
        QKV_M_ENTRY,
        GATING_M_ENTRY,
        CONV_M_ENTRY,
        SHIFT_ENTRY,
        OUT_M_ENTRY,
    ] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "the shipped prefill source no longer declares {e}; the entry moved and this gate is \
             now testing nothing"
        );
    }
    src
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_delta_front_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 32) as u32) as f64 / (1u64 << 31) as f64) - 1.0
    }

    fn f32_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|_| (self.next() * scale) as f32 as f64)
            .collect()
    }
    fn bf16_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| bf16(self.next() * scale)).collect()
    }
}

fn unpack_bf16(words: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for w in words {
        out.push(half::bf16::from_bits((*w & 0xffff) as u16).to_f32() as f64);
        out.push(half::bf16::from_bits((*w >> 16) as u16).to_f32() as f64);
    }
    out
}

fn f32v(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

fn ulp_bf16(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return f32::MIN_POSITIVE as f64;
    }
    (v.abs().log2().floor() - 7.0).exp2()
}

fn assert_nondegenerate(label: &str, what: &str, want: &[f64], got: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {what} readback length {} != reference length {}",
        got.len(),
        want.len()
    );
    let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        ref_max > 1e-3,
        "{label}: the reference {what} is degenerate (max |x| {ref_max:e}); a tolerance check \
         against it would pass on any kernel"
    );
    let got_max = got.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        got_max > 1e-3,
        "{label}: the kernel {what} is all-but-zero (max |x| {got_max:e}); the comparison would be \
         zeros against zeros"
    );
    let distinct: std::collections::BTreeSet<u64> = want.iter().map(|v| v.to_bits()).collect();
    assert!(
        distinct.len() > 2 || want.len() <= 2,
        "{label}: the reference {what} has only {} distinct values over {} lanes; an index bug \
         would permute identical numbers and pass",
        distinct.len(),
        want.len()
    );
}

fn check_rel(label: &str, entry: &str, what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_nondegenerate(label, what, want, got);
    let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
    let mut rel = 0f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs() / ref_max;
        if d > rel {
            rel = d;
            at = i;
        }
    }
    assert!(
        rel < TOL_F32,
        "{label}: {entry} {what} diverged from the f64 host reference (rel {rel:e} at index {at}, \
         ref_max {ref_max:e}). The shader accumulates in f32 and the reference in f64 in the \
         opposite index order; every green run prints the floor this corpus actually reaches, and \
         {TOL_F32:e} sits far above it. Contrast the graph-level fixture, whose rel < 0.05 on \
         4-layer logits is 300x looser than the effect it would have to see."
    );
    rel
}

fn check_bf16_ulp(label: &str, entry: &str, what: &str, got: &[f64], want: &[f64]) -> (f64, usize) {
    assert_nondegenerate(label, what, want, got);
    let mut worst = 0f64;
    let mut at = 0usize;
    let mut exact = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if bf16_bits(*g) == bf16_bits(*w) {
            exact += 1;
        }
        let frac = (g - w).abs() / ulp_bf16(w.abs().max(g.abs()));
        if frac > worst {
            worst = frac;
            at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{label}: {entry} {what} is more than one bf16 ulp from the f64 host reference ({worst:.3} \
         ulp at index {at}: kernel {} vs reference {}). The kernel rounds this value to bf16 \
         before storing it, so one ulp is the output format's own resolution -- the only thing an \
         f32-vs-f64 evaluation can move -- and anything past it is arithmetic, not rounding.",
        got[at],
        want[at]
    );
    (worst, exact)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DqParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DqmParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    mixed_stride: u32,
    pad0: u32,
    scale: f32,
}

#[derive(Clone, Copy)]
struct QkvCase {
    label: &'static str,
    n_v: usize,
    n_k: usize,
    d_k: usize,
    d_v: usize,
    m: usize,
    scale: f64,
}

fn qkv_cases() -> Vec<QkvCase> {
    vec![
        QkvCase {
            label: "nv4 nk2 dk16 dv16 m3",
            n_v: 4,
            n_k: 2,
            d_k: 16,
            d_v: 16,
            m: 3,
            scale: 0.25,
        },
        QkvCase {
            label: "nv2 nk2 dk32 dv64 m1",
            n_v: 2,
            n_k: 2,
            d_k: 32,
            d_v: 64,
            m: 1,
            scale: 1.75,
        },
    ]
}

impl QkvCase {
    fn key_dim(&self) -> usize {
        self.n_k * self.d_k
    }
    fn value_dim(&self) -> usize {
        self.n_v * self.d_v
    }
    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    fn mixed_stride(&self) -> usize {
        self.conv_dim() + 4
    }
    fn mixed(&self) -> Vec<f64> {
        let mut r = Lcg::new(0xdf70_0000 ^ (self.d_k as u64) << 8 ^ self.n_v as u64);
        r.f32_vec(self.m * self.mixed_stride(), 0.9)
    }
}

fn qkv_reference(c: &QkvCase, mixed: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let v_per_k = c.n_v / c.n_k;
    let mut q = vec![0f64; c.m * c.n_v * c.d_k];
    let mut k = vec![0f64; c.m * c.n_v * c.d_k];
    let mut v = vec![0f64; c.m * c.n_v * c.d_v];
    for t in 0..c.m {
        let mb = t * c.mixed_stride();
        for h in 0..c.n_v {
            let kh = h / v_per_k;
            let mut sq = 0f64;
            let mut sk = 0f64;
            for d in (0..c.d_k).rev() {
                let qv = mixed[mb + kh * c.d_k + d];
                let kv = mixed[mb + c.key_dim() + kh * c.d_k + d];
                sq += qv * qv;
                sk += kv * kv;
            }
            let nq = (sq + 1e-6).sqrt();
            let nk = (sk + 1e-6).sqrt();
            for d in 0..c.d_k {
                let qv = mixed[mb + kh * c.d_k + d];
                let kv = mixed[mb + c.key_dim() + kh * c.d_k + d];
                q[(t * c.n_v + h) * c.d_k + d] = (qv / nq) * c.scale;
                k[(t * c.n_v + h) * c.d_k + d] = kv / nk;
            }
            for d in 0..c.d_v {
                v[(t * c.n_v + h) * c.d_v + d] = mixed[mb + 2 * c.key_dim() + h * c.d_v + d];
            }
        }
    }
    (q, k, v)
}

#[test]
fn q3w_delta_qkv_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[q3d-delta-front-oracle] adapter: {}", ctx.info.name);
    let src = decode_source();
    let mut worst = 0f64;
    for c in qkv_cases() {
        let mixed = c.mixed();
        let one = QkvCase { m: 1, ..c };
        let (wq, wk, wv) = qkv_reference(&one, &mixed);

        let mixed_b = dispatch::storage_from_slice(ctx, "dq-mixed", &f32v(&mixed));
        let q_b = dispatch::storage_zeroed(ctx, "dq-q", (c.n_v * c.d_k * 4) as u64);
        let k_b = dispatch::storage_zeroed(ctx, "dq-k", (c.n_v * c.d_k * 4) as u64);
        let v_b = dispatch::storage_zeroed(ctx, "dq-v", (c.n_v * c.d_v * 4) as u64);
        let p_b = dispatch::uniform_from(
            ctx,
            "dq-p",
            &DqParams {
                n_v: c.n_v as u32,
                d_k: c.d_k as u32,
                d_v: c.d_v as u32,
                key_dim: c.key_dim() as u32,
                v_per_k: (c.n_v / c.n_k) as u32,
                pad0: 0,
                pad1: 0,
                scale: c.scale as f32,
            },
        );
        dispatch::run(
            ctx,
            "q3d-delta-front-oracle",
            &src,
            QKV_ENTRY,
            &[
                (10, &mixed_b),
                (11, &q_b),
                (12, &k_b),
                (13, &v_b),
                (14, &p_b),
            ],
            (c.n_v as u32, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {QKV_ENTRY}: {e}", c.label));

        let gq: Vec<f32> = dispatch::read_back(ctx, &q_b, c.n_v * c.d_k).expect("read back q");
        let gk: Vec<f32> = dispatch::read_back(ctx, &k_b, c.n_v * c.d_k).expect("read back k");
        let gv: Vec<f32> = dispatch::read_back(ctx, &v_b, c.n_v * c.d_v).expect("read back v");
        let to64 = |x: &[f32]| x.iter().map(|v| *v as f64).collect::<Vec<f64>>();
        let rq = check_rel(c.label, QKV_ENTRY, "q", &to64(&gq), &wq);
        let rk = check_rel(c.label, QKV_ENTRY, "k", &to64(&gk), &wk);
        let rv = check_rel(c.label, QKV_ENTRY, "v", &to64(&gv), &wv);
        eprintln!(
            "[q3d-delta-front-oracle] {} decode qkv: q rel={rq:.3e} k rel={rk:.3e} v rel={rv:.3e}",
            c.label
        );
        worst = worst.max(rq).max(rk).max(rv);
    }
    eprintln!("[q3d-delta-front-oracle] worst decode qkv relative error: {worst:.3e}");
}

#[test]
fn q3w_delta_qkv_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    let mut worst = 0f64;
    let mut saw_multi = false;
    for c in qkv_cases() {
        saw_multi |= c.m > 1;
        let mixed = c.mixed();
        let (wq, wk, wv) = qkv_reference(&c, &mixed);

        let mixed_b = dispatch::storage_from_slice(ctx, "pdq-mixed", &f32v(&mixed));
        let q_b = dispatch::storage_zeroed(ctx, "pdq-q", (c.m * c.n_v * c.d_k * 4) as u64);
        let k_b = dispatch::storage_zeroed(ctx, "pdq-k", (c.m * c.n_v * c.d_k * 4) as u64);
        let v_b = dispatch::storage_zeroed(ctx, "pdq-v", (c.m * c.n_v * c.d_v * 4) as u64);
        let p_b = dispatch::uniform_from(
            ctx,
            "pdq-p",
            &DqmParams {
                n_v: c.n_v as u32,
                d_k: c.d_k as u32,
                d_v: c.d_v as u32,
                key_dim: c.key_dim() as u32,
                v_per_k: (c.n_v / c.n_k) as u32,
                mixed_stride: c.mixed_stride() as u32,
                pad0: 0,
                scale: c.scale as f32,
            },
        );
        dispatch::run(
            ctx,
            "q3d-delta-front-oracle",
            &src,
            QKV_M_ENTRY,
            &[
                (20, &mixed_b),
                (21, &q_b),
                (22, &k_b),
                (23, &v_b),
                (24, &p_b),
            ],
            (c.n_v as u32, c.m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {QKV_M_ENTRY}: {e}", c.label));

        let n_qk = c.m * c.n_v * c.d_k;
        let n_v = c.m * c.n_v * c.d_v;
        let gq: Vec<f32> = dispatch::read_back(ctx, &q_b, n_qk).expect("read back q");
        let gk: Vec<f32> = dispatch::read_back(ctx, &k_b, n_qk).expect("read back k");
        let gv: Vec<f32> = dispatch::read_back(ctx, &v_b, n_v).expect("read back v");
        let to64 = |x: &[f32]| x.iter().map(|v| *v as f64).collect::<Vec<f64>>();
        let rq = check_rel(c.label, QKV_M_ENTRY, "q", &to64(&gq), &wq);
        let rk = check_rel(c.label, QKV_M_ENTRY, "k", &to64(&gk), &wk);
        let rv = check_rel(c.label, QKV_M_ENTRY, "v", &to64(&gv), &wv);
        eprintln!(
            "[q3d-delta-front-oracle] {} prefill qkv ({} rows): q rel={rq:.3e} k rel={rk:.3e} \
             v rel={rv:.3e}",
            c.label, c.m
        );
        worst = worst.max(rq).max(rk).max(rv);
    }
    assert!(
        saw_multi,
        "every qkv case has m == 1, where `mixed_stride` and the `t * n_v` output stride are both \
         multiplied by zero and deleting them would pass"
    );
    eprintln!("[q3d-delta-front-oracle] worst prefill qkv relative error: {worst:.3e}");
}

#[test]
fn the_qkv_corpus_normalizes_and_scales() {
    let mut worst_norm_gap = 0f64;
    let mut min_scale_gap = f64::INFINITY;
    let mut grouped = false;
    for c in qkv_cases() {
        grouped |= c.n_v / c.n_k > 1;
        min_scale_gap = min_scale_gap.min((c.scale - 1.0).abs());
        let mixed = c.mixed();
        let v_per_k = c.n_v / c.n_k;
        for t in 0..c.m {
            let mb = t * c.mixed_stride();
            for h in 0..c.n_v {
                let kh = h / v_per_k;
                let mut sq = 0f64;
                for d in 0..c.d_k {
                    let qv = mixed[mb + kh * c.d_k + d];
                    sq += qv * qv;
                }
                worst_norm_gap = worst_norm_gap.max(((sq + 1e-6).sqrt() - 1.0).abs());
            }
        }
    }
    eprintln!(
        "[q3d-delta-front-oracle] worst |nq - 1| {worst_norm_gap:.4}, min |scale - 1| \
         {min_scale_gap:.4}"
    );
    assert!(
        worst_norm_gap > 0.25,
        "every q row already has unit norm ({worst_norm_gap}); dividing by nq would be an identity"
    );
    assert!(
        min_scale_gap > 0.1,
        "some case uses scale == 1 (min |scale - 1| {min_scale_gap}); the `* dq_p.scale` multiply \
         is then an identity there"
    );
    assert!(
        grouped,
        "no case has v_per_k > 1, so `h / dq_p.v_per_k` is the identity map and a grouping bug \
         would be invisible"
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GmParams {
    n_v: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct GatingCase {
    label: &'static str,
    n_v: usize,
    m: usize,
}

fn gating_cases() -> Vec<GatingCase> {
    vec![
        GatingCase {
            label: "nv96 m4",
            n_v: 96,
            m: 4,
        },
        GatingCase {
            label: "nv8 m1",
            n_v: 8,
            m: 1,
        },
    ]
}

struct GatingData {
    ab: Vec<f64>,
    alog: Vec<f64>,
    dt: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
}

fn gating_reference(c: &GatingCase, seed: u64) -> GatingData {
    let mut r = Lcg::new(seed);
    let ab = r.bf16_vec(c.m * 2 * c.n_v, 2.5);
    let alog = r.f32_vec(c.n_v, 0.8);
    let dt = r.f32_vec(c.n_v, 1.2);
    let mut g = vec![0f64; c.m * c.n_v];
    let mut beta = vec![0f64; c.m * c.n_v];
    for t in 0..c.m {
        let base = t * 2 * c.n_v;
        for i in 0..c.n_v {
            let a = ab[base + i];
            let b = ab[base + c.n_v + i];
            beta[t * c.n_v + i] = 1.0 / (1.0 + (-b).exp());
            let tt = a + dt[i];
            let sp = tt.max(0.0) + (1.0 + (-tt.abs()).exp()).ln();
            g[t * c.n_v + i] = (sp * -(alog[i].exp())).exp();
        }
    }
    GatingData {
        ab,
        alog,
        dt,
        g,
        beta,
    }
}

#[test]
fn q3w_delta_gating_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    let mut worst = 0f64;
    for c in gating_cases() {
        let d = gating_reference(&c, 0x6a71_0000 ^ c.n_v as u64);
        let ab_b = dispatch::storage_from_slice(ctx, "pdg-ab", &pack_bf16(&d.ab));
        let alog_b = dispatch::storage_from_slice(ctx, "pdg-alog", &f32v(&d.alog));
        let dt_b = dispatch::storage_from_slice(ctx, "pdg-dt", &f32v(&d.dt));
        let g_b = dispatch::storage_zeroed(ctx, "pdg-g", (c.m * c.n_v * 4) as u64);
        let beta_b = dispatch::storage_zeroed(ctx, "pdg-beta", (c.m * c.n_v * 4) as u64);
        let p_b = dispatch::uniform_from(
            ctx,
            "pdg-p",
            &GmParams {
                n_v: c.n_v as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let (gx, _, _) = dispatch::workgroup_count_1d(ctx, c.n_v as u64, 64);
        dispatch::run(
            ctx,
            "q3d-delta-front-oracle",
            &src,
            GATING_M_ENTRY,
            &[
                (30, &ab_b),
                (31, &alog_b),
                (32, &dt_b),
                (33, &g_b),
                (34, &beta_b),
                (35, &p_b),
            ],
            (gx, c.m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {GATING_M_ENTRY}: {e}", c.label));

        let gg: Vec<f32> = dispatch::read_back(ctx, &g_b, c.m * c.n_v).expect("read back g");
        let gb: Vec<f32> = dispatch::read_back(ctx, &beta_b, c.m * c.n_v).expect("read back beta");
        let to64 = |x: &[f32]| x.iter().map(|v| *v as f64).collect::<Vec<f64>>();
        let rg = check_rel(c.label, GATING_M_ENTRY, "g", &to64(&gg), &d.g);
        let rb = check_rel(c.label, GATING_M_ENTRY, "beta", &to64(&gb), &d.beta);
        eprintln!(
            "[q3d-delta-front-oracle] {} gating: g rel={rg:.3e} beta rel={rb:.3e}",
            c.label
        );
        worst = worst.max(rg).max(rb);
    }
    eprintln!("[q3d-delta-front-oracle] worst gating relative error: {worst:.3e}");
}

#[test]
fn the_gating_corpus_separates_the_two_halves_and_decays() {
    for c in gating_cases() {
        let d = gating_reference(&c, 0x6a71_0000 ^ c.n_v as u64);
        let mut swap_moves = 0usize;
        for t in 0..c.m {
            let base = t * 2 * c.n_v;
            for i in 0..c.n_v {
                let a = d.ab[base + i];
                let b = d.ab[base + c.n_v + i];
                let sa = 1.0 / (1.0 + (-a).exp());
                let sb = 1.0 / (1.0 + (-b).exp());
                swap_moves += usize::from((sa - sb).abs() > 0.01);
            }
        }
        let dt_spread = d.dt.iter().fold(0f64, |a, b| a.max((b - d.dt[0]).abs()));
        let alog_spread = d
            .alog
            .iter()
            .fold(0f64, |a, b| a.max((b - d.alog[0]).abs()));
        let g_lo = d.g.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let beta_gap = d.beta.iter().fold(0f64, |a, b| a.max((b - 0.5).abs()));
        let pairs = c.m * c.n_v;
        eprintln!(
            "[q3d-delta-front-oracle] {}: swapping a and b moves beta on {swap_moves}/{pairs} \
             heads, dt spread {dt_spread:.4}, a_log spread {alog_spread:.4}, min g {g_lo:.4}, \
             worst |beta-0.5| {beta_gap:.4}",
            c.label
        );
        assert!(
            swap_moves * 10 >= pairs * 9,
            "{}: reading `a` where the kernel should read `b` would move beta by more than 0.01 on \
             only {swap_moves} of {pairs} heads; the two halves of the packed ab row are too \
             similar for a half-swap to be observable",
            c.label
        );
        assert!(
            dt_spread > 1e-2 && alog_spread > 1e-2,
            "{}: dt or a_log is constant across heads; dropping the per-head index would pass",
            c.label
        );
        assert!(
            g_lo < 0.95,
            "{}: the smallest decay is {g_lo}; with g ~ 1 the recurrence's `state * g` is an \
             identity and this kernel's output would not matter downstream",
            c.label
        );
        assert!(
            beta_gap > 0.2,
            "{}: beta never leaves the neighbourhood of 0.5, so the sigmoid is being evaluated in \
             its linear region only",
            c.label
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CvmParams {
    conv_dim: u32,
    kernel: u32,
    x_words: u32,
    mixed_stride: u32,
}

struct ConvCase {
    label: &'static str,
    conv_dim: usize,
    kernel: usize,
    m: usize,
    live: usize,
}

fn conv_cases() -> Vec<ConvCase> {
    vec![
        ConvCase {
            label: "cd128 ks4 m5 live2(state-carrying)",
            conv_dim: 128,
            kernel: 4,
            m: 5,
            live: 2,
        },
        ConvCase {
            label: "cd48 ks4 m6 live6(all-from-x)",
            conv_dim: 48,
            kernel: 4,
            m: 6,
            live: 6,
        },
        ConvCase {
            label: "cd64 ks2 m3 live1",
            conv_dim: 64,
            kernel: 2,
            m: 3,
            live: 1,
        },
    ]
}

impl ConvCase {
    fn hist(&self) -> usize {
        self.kernel - 1
    }
    fn x_words(&self) -> usize {
        self.conv_dim / 2 + 8
    }
    fn x_elems(&self) -> usize {
        self.x_words() * 2
    }
    fn mixed_stride(&self) -> usize {
        self.conv_dim + 6
    }
}

struct ConvData {
    x: Vec<f64>,
    w: Vec<f64>,
    state: Vec<f64>,
}

fn conv_data(c: &ConvCase) -> ConvData {
    let mut r = Lcg::new(0xc02f_0000 ^ (c.conv_dim as u64) << 8 ^ c.kernel as u64);
    let mut x = vec![0f64; c.m * c.x_elems()];
    for t in 0..c.m {
        for e in 0..c.x_elems() {
            x[t * c.x_elems() + e] = bf16(r.next() * 1.4);
        }
    }
    ConvData {
        x,
        w: r.f32_vec(c.conv_dim * c.kernel, 0.8),
        state: r.f32_vec(c.conv_dim * c.hist(), 0.9),
    }
}

fn conv_reference(c: &ConvCase, d: &ConvData) -> Vec<f64> {
    let hist = c.hist();
    let mut out = vec![0f64; c.m * c.conv_dim];
    for t in 0..c.m {
        for ch in 0..c.conv_dim {
            let mut acc = d.w[ch * c.kernel + hist] * d.x[t * c.x_elems() + ch];
            for j in (0..hist).rev() {
                let idx = t as isize + j as isize - hist as isize;
                let v = if idx < 0 {
                    d.state[ch * hist + t + j]
                } else {
                    d.x[idx as usize * c.x_elems() + ch]
                };
                acc += d.w[ch * c.kernel + j] * v;
            }
            out[t * c.conv_dim + ch] = bf16(acc / (1.0 + (-acc).exp()));
        }
    }
    out
}

fn shift_reference(c: &ConvCase, d: &ConvData) -> Vec<f64> {
    let hist = c.hist();
    let mut state = d.state.clone();
    for ch in 0..c.conv_dim {
        let mut tmp = vec![0f64; hist];
        for (j, slot) in tmp.iter_mut().enumerate() {
            let idx = c.live as isize + j as isize - hist as isize;
            *slot = if idx < 0 {
                d.state[ch * hist + c.live + j]
            } else {
                d.x[idx as usize * c.x_elems() + ch]
            };
        }
        state[ch * hist..ch * hist + hist].copy_from_slice(&tmp);
    }
    state
}

#[test]
fn q3w_delta_conv_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    let mut worst = 0f64;
    for c in conv_cases() {
        let d = conv_data(&c);
        let want = conv_reference(&c, &d);
        let x_b = dispatch::storage_from_slice(ctx, "pcv-x", &pack_bf16(&d.x));
        let w_b = dispatch::storage_from_slice(ctx, "pcv-w", &f32v(&d.w));
        let state_b = dispatch::storage_from_slice(ctx, "pcv-state", &f32v(&d.state));
        let out_b = dispatch::storage_zeroed(ctx, "pcv-out", (c.m * c.mixed_stride() * 4) as u64);
        let p_b = dispatch::uniform_from(
            ctx,
            "pcv-p",
            &CvmParams {
                conv_dim: c.conv_dim as u32,
                kernel: c.kernel as u32,
                x_words: c.x_words() as u32,
                mixed_stride: c.mixed_stride() as u32,
            },
        );
        let (gx, _, _) = dispatch::workgroup_count_1d(ctx, c.conv_dim as u64, 64);
        dispatch::run(
            ctx,
            "q3d-delta-front-oracle",
            &src,
            CONV_M_ENTRY,
            &[
                (10, &x_b),
                (11, &w_b),
                (12, &state_b),
                (13, &out_b),
                (14, &p_b),
            ],
            (gx, c.m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {CONV_M_ENTRY}: {e}", c.label));

        let raw: Vec<f32> = dispatch::read_back(ctx, &out_b, c.m * c.mixed_stride())
            .expect("read back conv output");
        let mut got = Vec::with_capacity(c.m * c.conv_dim);
        for t in 0..c.m {
            for ch in 0..c.conv_dim {
                got.push(raw[t * c.mixed_stride() + ch] as f64);
            }
        }
        let (frac, exact) = check_bf16_ulp(c.label, CONV_M_ENTRY, "conv+silu", &got, &want);
        eprintln!(
            "[q3d-delta-front-oracle] {} conv: {exact}/{} lanes bit-exact, worst {frac:.3} of one \
             bf16 ulp",
            c.label,
            want.len()
        );
        worst = worst.max(frac);
    }
    eprintln!("[q3d-delta-front-oracle] worst conv deviation: {worst:.3} bf16 ulp");
}

#[test]
fn q3w_delta_conv_shift_rolls_the_history_for_the_next_chunk() {
    let ctx = ctx();
    let src = prefill_source();
    let mut saw_carry = false;
    let mut saw_all_from_x = false;
    for c in conv_cases() {
        saw_carry |= c.live < c.hist();
        saw_all_from_x |= c.live >= c.hist();
        let d = conv_data(&c);
        let want = shift_reference(&c, &d);
        assert_ne!(
            want, d.state,
            "{}: the shift reference equals the state it started from, so a kernel that wrote \
             nothing at all would pass",
            c.label
        );
        let x_b = dispatch::storage_from_slice(ctx, "pcv-x", &pack_bf16(&d.x));
        let state_b = dispatch::storage_from_slice(ctx, "pcv-state", &f32v(&d.state));
        let p_b = dispatch::uniform_from(
            ctx,
            "pcv-p",
            &CvmParams {
                conv_dim: c.conv_dim as u32,
                kernel: c.kernel as u32,
                x_words: c.x_words() as u32,
                mixed_stride: c.mixed_stride() as u32,
            },
        );
        let ck_b = dispatch::uniform_from(
            ctx,
            "pcv-ck",
            &CkParams {
                m_live: c.live as u32,
                base: 0,
                pad0: 0,
                pad1: 0,
            },
        );
        let (gx, _, _) = dispatch::workgroup_count_1d(ctx, c.conv_dim as u64, 64);
        dispatch::run(
            ctx,
            "q3d-delta-front-oracle",
            &src,
            SHIFT_ENTRY,
            &[(10, &x_b), (12, &state_b), (14, &p_b), (15, &ck_b)],
            (gx, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {SHIFT_ENTRY}: {e}", c.label));

        let got: Vec<f32> = dispatch::read_back(ctx, &state_b, c.conv_dim * c.hist())
            .expect("read back shifted state");
        let got64: Vec<f64> = got.iter().map(|v| *v as f64).collect();
        assert_nondegenerate(c.label, "shifted state", &want, &got64);
        for (i, (g, w)) in got64.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{}: {SHIFT_ENTRY} left the wrong history at slot {i} ({g} vs {w}). This entry \
                 only moves bf16-decoded words, so the demand is bit-exactness. Nothing in THIS \
                 chunk reads what it writes -- the whole effect lands on the next chunk -- so no \
                 single-chunk fixture can see it being wrong.",
                c.label
            );
        }
        eprintln!(
            "[q3d-delta-front-oracle] {} shift: {} slots bit-exact (live {} vs hist {})",
            c.label,
            want.len(),
            c.live,
            c.hist()
        );
    }
    assert!(
        saw_carry,
        "no case has m_live < kernel-1, so the `pcv_state[c * hist + live + j]` carry branch never \
         runs and a chunk shorter than the conv kernel would be untested"
    );
    assert!(
        saw_all_from_x,
        "no case has m_live >= kernel-1, so the branch that fills the whole history from x -- the \
         one every full chunk takes -- never runs"
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DomParams {
    n_v: u32,
    d_v: u32,
    pad0: u32,
    eps: f32,
}

struct OutCase {
    label: &'static str,
    n_v: usize,
    d_v: usize,
    m: usize,
}

fn out_cases() -> Vec<OutCase> {
    vec![
        OutCase {
            label: "nv4 dv16 m3",
            n_v: 4,
            d_v: 16,
            m: 3,
        },
        OutCase {
            label: "nv2 dv64 m2",
            n_v: 2,
            d_v: 64,
            m: 2,
        },
    ]
}

struct OutData {
    core: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
    want: Vec<f64>,
}

const OUT_EPS: f64 = 1e-6;

fn out_data(c: &OutCase) -> OutData {
    let mut r = Lcg::new(0x0d07_0000 ^ (c.d_v as u64) << 8 ^ c.n_v as u64);
    let vdim = c.n_v * c.d_v;
    let core = r.f32_vec(c.m * vdim, 1.3);
    let w = r.bf16_vec(c.d_v, 1.1);
    let z = r.bf16_vec(c.m * vdim, 2.0);
    let mut want = vec![0f64; c.m * vdim];
    for t in 0..c.m {
        for h in 0..c.n_v {
            let base = t * vdim + h * c.d_v;
            let mut ss = 0f64;
            for d in (0..c.d_v).rev() {
                let v = bf16(core[base + d]);
                ss += v * v;
            }
            let rms = 1.0 / (ss / c.d_v as f64 + OUT_EPS).sqrt();
            for d in 0..c.d_v {
                let v = bf16(core[base + d]);
                let zv = z[base + d];
                let g = bf16(zv / (1.0 + (-zv).exp()));
                let n = bf16(v * rms * w[d]);
                want[base + d] = bf16(n * g);
            }
        }
    }
    OutData { core, w, z, want }
}

#[test]
fn q3w_delta_out_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    let mut worst = 0f64;
    for c in out_cases() {
        let d = out_data(&c);
        let vdim = c.n_v * c.d_v;
        let core_b = dispatch::storage_from_slice(ctx, "pdo-core", &f32v(&d.core));
        let w_b = dispatch::storage_from_slice(ctx, "pdo-w", &pack_bf16(&d.w));
        let z_b = dispatch::storage_from_slice(ctx, "pdo-z", &pack_bf16(&d.z));
        let out_b = dispatch::storage_zeroed(ctx, "pdo-out", (c.m * vdim * 2) as u64);
        let p_b = dispatch::uniform_from(
            ctx,
            "pdo-p",
            &DomParams {
                n_v: c.n_v as u32,
                d_v: c.d_v as u32,
                pad0: 0,
                eps: OUT_EPS as f32,
            },
        );
        dispatch::run(
            ctx,
            "q3d-delta-front-oracle",
            &src,
            OUT_M_ENTRY,
            &[
                (50, &core_b),
                (51, &w_b),
                (52, &z_b),
                (53, &out_b),
                (54, &p_b),
            ],
            (c.n_v as u32, c.m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {OUT_M_ENTRY}: {e}", c.label));

        let words: Vec<u32> =
            dispatch::read_back(ctx, &out_b, c.m * vdim / 2).expect("read back delta out_m output");
        let got = unpack_bf16(&words);
        let (frac, exact) = check_bf16_ulp(c.label, OUT_M_ENTRY, "gated norm", &got, &d.want);
        eprintln!(
            "[q3d-delta-front-oracle] {} out_m: {exact}/{} lanes bit-exact, worst {frac:.3} of one \
             bf16 ulp",
            c.label,
            d.want.len()
        );
        worst = worst.max(frac);
    }
    eprintln!("[q3d-delta-front-oracle] worst out_m deviation: {worst:.3} bf16 ulp");
}

#[test]
fn the_out_m_corpus_normalizes_and_gates() {
    for c in out_cases() {
        let d = out_data(&c);
        let vdim = c.n_v * c.d_v;
        let mut worst_rms_gap = 0f64;
        let mut worst_gate_gap = 0f64;
        for t in 0..c.m {
            for h in 0..c.n_v {
                let base = t * vdim + h * c.d_v;
                let mut ss = 0f64;
                for dd in 0..c.d_v {
                    let v = bf16(d.core[base + dd]);
                    ss += v * v;
                }
                let rms = 1.0 / (ss / c.d_v as f64 + OUT_EPS).sqrt();
                worst_rms_gap = worst_rms_gap.max((rms - 1.0).abs());
                for dd in 0..c.d_v {
                    let zv = d.z[base + dd];
                    let g = zv / (1.0 + (-zv).exp());
                    worst_gate_gap = worst_gate_gap.max((g - 1.0).abs());
                }
            }
        }
        let w_spread = d.w.iter().fold(0f64, |a, b| a.max((b - d.w[0]).abs()));
        eprintln!(
            "[q3d-delta-front-oracle] {}: worst |rms-1| {worst_rms_gap:.4}, worst |swish(z)-1| \
             {worst_gate_gap:.4}, norm-weight spread {w_spread:.4}",
            c.label
        );
        assert!(
            worst_rms_gap > 0.1,
            "{}: rms never leaves 1 ({worst_rms_gap}); the normalization would be an identity",
            c.label
        );
        assert!(
            worst_gate_gap > 0.25,
            "{}: the swish gate never leaves 1 ({worst_gate_gap}); the `* g` multiply would be an \
             identity",
            c.label
        );
        assert!(
            w_spread > 1e-2,
            "{}: the norm weight vector is constant, so a per-lane weight index bug would be \
             invisible",
            c.label
        );
    }
}
