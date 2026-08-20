#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const SEQ_ENTRY: &str = "q3w_delta_scan";
const WY_ENTRY: &str = "q3w_delta_scan_wy";
const WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS: u32 = 4;

const WY_CPU_F64_AGREEMENT_TOL_REL: f64 = 1e-10;

const WY_GPU_TOL_REL_1E4_BOTH_KERNELS_SUM_F32_OVER_DK128_AND_THE_WY_FORM_REGROUPS_A_32_TOKEN_CHUNK:
    f64 = 1e-4;

const G_LOG_CLAMP_1E30_KEEPS_LOG_FINITE_WHEN_GATING_UNDERFLOWS_TO_ZERO: f64 = 1e-30;

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
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

struct Case {
    label: String,
    heads: usize,
    dk: usize,
    dv: usize,
    m: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state0: Vec<f32>,
}

fn l2_normalize_rows_because_unnormalized_k_makes_the_delta_rule_diverge(
    x: &mut [f32],
    row: usize,
    scale: f32,
) {
    for r in x.chunks_mut(row) {
        let n = r.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt() as f32;
        for v in r.iter_mut() {
            *v = *v / n * scale;
        }
    }
}

fn case(heads: usize, dk: usize, dv: usize, m: usize, seed: u64) -> Case {
    let mut r = Lcg::new(seed);
    let g: Vec<f32> = (0..m * heads)
        .map(|_| 0.35 + 0.6 * (r.next() * 0.5 + 0.5))
        .collect();
    let beta: Vec<f32> = (0..m * heads)
        .map(|_| 0.2 + 0.7 * (r.next() * 0.5 + 0.5))
        .collect();
    let mut q = r.vec(m * heads * dk, 0.7);
    let mut k = r.vec(m * heads * dk, 0.7);
    l2_normalize_rows_because_unnormalized_k_makes_the_delta_rule_diverge(
        &mut q,
        dk,
        1.0 / (dk as f32).sqrt(),
    );
    l2_normalize_rows_because_unnormalized_k_makes_the_delta_rule_diverge(&mut k, dk, 1.0);
    Case {
        label: format!("h{heads} dk{dk} dv{dv} m{m}"),
        heads,
        dk,
        dv,
        m,
        q,
        k,
        v: r.vec(m * heads * dv, 0.7),
        g,
        beta,
        state0: r.vec(heads * dk * dv, 0.5),
    }
}

fn seq_reference(c: &Case) -> (Vec<f64>, Vec<f64>) {
    let (heads, dk, dv, m) = (c.heads, c.dk, c.dv, c.m);
    let mut state: Vec<f64> = c.state0.iter().map(|v| *v as f64).collect();
    let mut out = vec![0f64; m * heads * dv];
    for t in 0..m {
        for h in 0..heads {
            let sbase = h * dk * dv;
            let ge = c.g[t * heads + h] as f64;
            let bt = c.beta[t * heads + h] as f64;
            for e in state[sbase..sbase + dk * dv].iter_mut() {
                *e *= ge;
            }
            for j in 0..dv {
                let mut kv = 0f64;
                for i in 0..dk {
                    kv += state[sbase + i * dv + j] * c.k[(t * heads + h) * dk + i] as f64;
                }
                let delta = (c.v[(t * heads + h) * dv + j] as f64 - kv) * bt;
                let mut acc = 0f64;
                for i in 0..dk {
                    let s =
                        state[sbase + i * dv + j] + c.k[(t * heads + h) * dk + i] as f64 * delta;
                    state[sbase + i * dv + j] = s;
                    acc += s * c.q[(t * heads + h) * dk + i] as f64;
                }
                out[(t * heads + h) * dv + j] = acc;
            }
        }
    }
    (out, state)
}

fn wy_reference(c: &Case, chunk: usize) -> (Vec<f64>, Vec<f64>) {
    let (heads, dk, dv, m) = (c.heads, c.dk, c.dv, c.m);
    let mut state: Vec<f64> = c.state0.iter().map(|v| *v as f64).collect();
    let mut out = vec![0f64; m * heads * dv];
    for h in 0..heads {
        let sbase = h * dk * dv;
        let mut c0 = 0usize;
        while c0 < m {
            let n = chunk.min(m - c0);
            let mut cl = vec![0f64; n];
            let mut beta = vec![0f64; n];
            let mut acc = 0f64;
            for t in 0..n {
                let gi = (c0 + t) * heads + h;
                acc += (c.g[gi] as f64).max(G_LOG_CLAMP_1E30_KEEPS_LOG_FINITE_WHEN_GATING_UNDERFLOWS_TO_ZERO).ln();
                cl[t] = acc;
                beta[t] = c.beta[gi] as f64;
            }
            let kat = |t: usize, i: usize| c.k[((c0 + t) * heads + h) * dk + i] as f64;
            let qat = |t: usize, i: usize| c.q[((c0 + t) * heads + h) * dk + i] as f64;
            let mut a = vec![0f64; n * n];
            let mut wqk = vec![0f64; n * n];
            for t in 0..n {
                for s in 0..=t {
                    let mut dkk = 0f64;
                    let mut dqk = 0f64;
                    for i in 0..dk {
                        dkk += kat(s, i) * kat(t, i);
                        dqk += kat(s, i) * qat(t, i);
                    }
                    let decay = (cl[t] - cl[s]).exp();
                    if s < t {
                        a[t * n + s] = beta[t] * decay * dkk;
                    }
                    wqk[t * n + s] = decay * dqk;
                }
            }
            let mut u = vec![0f64; n * dv];
            let mut outp = vec![0f64; n * dv];
            for v in 0..dv {
                let mut acck = vec![0f64; n];
                let mut accq = vec![0f64; n];
                for i in 0..dk {
                    let sv = state[sbase + i * dv + v];
                    for t in 0..n {
                        acck[t] += sv * kat(t, i);
                        accq[t] += sv * qat(t, i);
                    }
                }
                for t in 0..n {
                    let vt = c.v[((c0 + t) * heads + h) * dv + v] as f64;
                    u[t * dv + v] = beta[t] * (vt - cl[t].exp() * acck[t]);
                    outp[t * dv + v] = cl[t].exp() * accq[t];
                }
            }
            for t in 1..n {
                for v in 0..dv {
                    let mut sub = 0f64;
                    for s in 0..t {
                        sub += a[t * n + s] * u[s * dv + v];
                    }
                    u[t * dv + v] -= sub;
                }
            }
            for t in 0..n {
                for v in 0..dv {
                    let mut sum = outp[t * dv + v];
                    for s in 0..=t {
                        sum += wqk[t * n + s] * u[s * dv + v];
                    }
                    out[((c0 + t) * heads + h) * dv + v] = sum;
                }
            }
            let gend = cl[n - 1];
            for i in 0..dk {
                for v in 0..dv {
                    let mut sv = gend.exp() * state[sbase + i * dv + v];
                    for s in 0..n {
                        sv += (gend - cl[s]).exp() * kat(s, i) * u[s * dv + v];
                    }
                    state[sbase + i * dv + v] = sv;
                }
            }
            c0 += n;
        }
    }
    (out, state)
}

fn rel_err(got: &[f64], want: &[f64], what: &str, label: &str) -> f64 {
    assert_eq!(got.len(), want.len());
    let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        ref_max > 1e-3,
        "{label}: reference {what} is degenerate (max |x| {ref_max:e}); the tolerance check \
         would pass on anything"
    );
    let got_max = got.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        got_max > 1e-3,
        "{label}: candidate {what} is all-but-zero (max |x| {got_max:e}); zeros against zeros \
         proves nothing"
    );
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs() / ref_max)
        .fold(0f64, f64::max)
}

#[test]
fn wy_chunked_f64_form_matches_the_sequential_f64_recurrence() {
    let mut worst = 0f64;
    for (heads, dk, dv) in [(2usize, 16usize, 16usize), (2, 128, 128), (3, 32, 64)] {
        for m in [1usize, 5, 32, 67, 256] {
            for chunk in [16usize, 32, 64] {
                let c = case(heads, dk, dv, m, 0x77_5CA0 ^ ((m * 131 + chunk) as u64));
                let (want_out, want_state) = seq_reference(&c);
                let (got_out, got_state) = wy_reference(&c, chunk);
                let ro = rel_err(&got_out, &want_out, "output", &c.label);
                let rs = rel_err(&got_state, &want_state, "state", &c.label);
                assert!(
                    ro < WY_CPU_F64_AGREEMENT_TOL_REL && rs < WY_CPU_F64_AGREEMENT_TOL_REL,
                    "{} chunk={chunk}: the WY/UT-transform algebra diverges from the sequential \
                     delta recurrence in f64 (out rel {ro:e}, state rel {rs:e} vs tol {:e}); \
                     the derivation is wrong, do not trust any kernel built on it",
                    c.label,
                    WY_CPU_F64_AGREEMENT_TOL_REL
                );
                worst = worst.max(ro).max(rs);
            }
        }
    }
    eprintln!("[q3d-scan-wy] worst f64 wy-vs-sequential rel: {worst:.3e}");
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CkParams {
    m_live: u32,
    base: u32,
    pad0: u32,
    pad1: u32,
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "q3d_delta_scan_wy needs a real wgpu adapter; a skipped numeric gate reads as a passed \
         one, so this panics rather than returning early",
    )
}

fn prefill_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    for entry in [SEQ_ENTRY, WY_ENTRY] {
        assert!(
            src.contains(&format!("fn {entry}(")),
            "the shipped prefill source no longer declares {entry}; the entry moved and this \
             gate is now testing nothing"
        );
    }
    src
}

fn run_scan(c: &Case, entry: &str, src: &str) -> (Vec<f32>, Vec<f32>) {
    let ctx = ctx();
    let q = dispatch::storage_from_slice(ctx, "q", &c.q);
    let k = dispatch::storage_from_slice(ctx, "k", &c.k);
    let v = dispatch::storage_from_slice(ctx, "v", &c.v);
    let g = dispatch::storage_from_slice(ctx, "g", &c.g);
    let beta = dispatch::storage_from_slice(ctx, "beta", &c.beta);
    let out = dispatch::storage_zeroed(ctx, "out", (c.m * c.heads * c.dv * 4) as u64);
    let state = dispatch::storage_from_slice(ctx, "state", &c.state0);
    let p = dispatch::uniform_from(
        ctx,
        "rp",
        &RParams {
            heads: c.heads as u32,
            d_k: c.dk as u32,
            d_v: c.dv as u32,
            pad0: 0,
        },
    );
    let ck = dispatch::uniform_from(
        ctx,
        "ck",
        &CkParams {
            m_live: c.m as u32,
            base: 0,
            pad0: 0,
            pad1: 0,
        },
    );
    let grid_y = if entry == WY_ENTRY {
        WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS
    } else {
        1
    };
    dispatch::run(
        ctx,
        "q3d-scan-wy-gate",
        src,
        entry,
        &[
            (40, &q),
            (41, &k),
            (42, &v),
            (43, &g),
            (44, &beta),
            (45, &out),
            (46, &state),
            (47, &p),
            (48, &ck),
        ],
        (c.heads as u32, grid_y, 1),
    )
    .unwrap_or_else(|e| panic!("{}: dispatch {entry}: {e}", c.label));
    let got_out: Vec<f32> =
        dispatch::read_back(ctx, &out, c.m * c.heads * c.dv).expect("read back scan out");
    let got_state: Vec<f32> =
        dispatch::read_back(ctx, &state, c.heads * c.dk * c.dv).expect("read back scan state");
    (got_out, got_state)
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

#[test]
fn q3w_delta_scan_wy_matches_the_f64_reference_and_the_sequential_kernel_at_real_dims() {
    let src = prefill_source();
    let tol = WY_GPU_TOL_REL_1E4_BOTH_KERNELS_SUM_F32_OVER_DK128_AND_THE_WY_FORM_REGROUPS_A_32_TOKEN_CHUNK;
    let mut worst = 0f64;
    let mut saw_partial_tail_chunk = false;
    let mut saw_full_chunks = false;
    for m in [1usize, 6, 32, 67, 256] {
        saw_partial_tail_chunk |= m % 32 != 0;
        saw_full_chunks |= m >= 64;
        let c = case(32, 128, 128, m, 0x3b_77e5 ^ m as u64);
        let (want_out, want_state) = seq_reference(&c);
        let (wy_out, wy_state) = run_scan(&c, WY_ENTRY, &src);
        let (sq_out, sq_state) = run_scan(&c, SEQ_ENTRY, &src);

        let ro = rel_err(&to_f64(&wy_out), &want_out, "output", &c.label);
        let rs = rel_err(&to_f64(&wy_state), &want_state, "state", &c.label);
        assert!(
            ro < tol && rs < tol,
            "{}: {WY_ENTRY} diverged from the f64 sequential reference (out rel {ro:e}, state \
             rel {rs:e} vs tol {tol:e}, error normalized by the reference max). The WY form \
             regroups the recurrence into per-chunk f32 dots over dk=128 and up to 32 tokens, so \
             it cannot be bit-equal to the sequential kernel; 1e-4 is the coop-arm precedent for \
             reassociated f32 prefill numerics and sits ~50x above the floor this suite prints",
            c.label
        );
        let ko = rel_err(&to_f64(&wy_out), &to_f64(&sq_out), "output", &c.label);
        let ks = rel_err(&to_f64(&wy_state), &to_f64(&sq_state), "state", &c.label);
        assert!(
            ko < tol && ks < tol,
            "{}: {WY_ENTRY} and {SEQ_ENTRY} disagree beyond the reassociation budget (out rel \
             {ko:e}, state rel {ks:e} vs tol {tol:e}); both kernels passed the f64 gate so this \
             is a kernel-vs-kernel drift the route A/B would inherit",
            c.label
        );
        eprintln!(
            "[q3d-scan-wy] {}: wy-vs-f64 out {ro:.3e} state {rs:.3e}; wy-vs-seq out {ko:.3e} \
             state {ks:.3e}",
            c.label
        );
        worst = worst.max(ro).max(rs).max(ko).max(ks);
    }
    assert!(
        saw_partial_tail_chunk && saw_full_chunks,
        "the corpus must exercise both full 32-token chunks and a partial tail chunk; without \
         both, the chunk-seam arithmetic is untested"
    );
    eprintln!("[q3d-scan-wy] worst gpu rel: {worst:.3e}");
}
