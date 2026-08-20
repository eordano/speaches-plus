#![cfg(feature = "wgpu")]

mod common;
use common::CkParams;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitVec as Lcg;

const DECODE_ENTRY: &str = "q3w_delta_recurrent";
const PREFILL_ENTRY: &str = "q3w_delta_scan";

fn decode_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_delta_source();
    assert!(
        src.contains(&format!("fn {DECODE_ENTRY}(")),
        "the shipped delta source no longer declares {DECODE_ENTRY}; the entry moved and this \
         gate is now testing nothing"
    );
    src
}

fn prefill_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    assert!(
        src.contains(&format!("fn {PREFILL_ENTRY}(")),
        "the shipped prefill source no longer declares {PREFILL_ENTRY}; the entry moved and this \
         gate is now testing nothing"
    );
    src
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

struct Case {
    label: &'static str,
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

fn reference(c: &Case) -> (Vec<f64>, Vec<f64>) {
    let (heads, dk, dv, m) = (c.heads, c.dk, c.dv, c.m);
    let mut state: Vec<f64> = c.state0.iter().map(|v| *v as f64).collect();
    let mut out = vec![0f64; m * heads * dv];
    for t in 0..m {
        for h in 0..heads {
            let sbase = h * dk * dv;
            let ge = c.g[t * heads + h] as f64;
            let bt = c.beta[t * heads + h] as f64;
            for i in 0..dk {
                for j in 0..dv {
                    state[sbase + i * dv + j] *= ge;
                }
            }
            let mut kv_mem = vec![0f64; dv];
            for j in 0..dv {
                let mut acc = 0f64;
                for i in (0..dk).rev() {
                    acc += state[sbase + i * dv + j] * c.k[(t * heads + h) * dk + i] as f64;
                }
                kv_mem[j] = acc;
            }
            for j in 0..dv {
                let delta = (c.v[(t * heads + h) * dv + j] as f64 - kv_mem[j]) * bt;
                let mut acc = 0f64;
                for i in (0..dk).rev() {
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

fn case(label: &'static str, heads: usize, dk: usize, dv: usize, m: usize, seed: u64) -> Case {
    let mut r = Lcg::new(seed);
    let g: Vec<f32> = (0..m * heads)
        .map(|_| 0.35 + 0.6 * (r.next() * 0.5 + 0.5))
        .collect();
    let beta: Vec<f32> = (0..m * heads)
        .map(|_| 0.2 + 0.7 * (r.next() * 0.5 + 0.5))
        .collect();
    Case {
        label,
        heads,
        dk,
        dv,
        m,
        q: r.vec(m * heads * dk, 0.7),
        k: r.vec(m * heads * dk, 0.7),
        v: r.vec(m * heads * dv, 0.7),
        g,
        beta,
        state0: r.vec(heads * dk * dv, 0.5),
    }
}

fn cases_m1() -> Vec<Case> {
    vec![
        case("m1 h4 dk16 dv16", 4, 16, 16, 1, 0xd317_0001),
        case("m1 h2 dk32 dv64", 2, 32, 64, 1, 0xd317_0002),
    ]
}

fn cases_mrow() -> Vec<Case> {
    vec![
        case("m6 h4 dk16 dv16", 4, 16, 16, 6, 0xd317_1001),
        case("m3 h2 dk32 dv64", 2, 32, 64, 3, 0xd317_1002),
        case("m1 h4 dk16 dv16", 4, 16, 16, 1, 0xd317_1003),
    ]
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_deltanet_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn check(label: &str, entry: &str, what: &str, got: &[f32], want: &[f64], tol: f64) -> f64 {
    let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        ref_max > 1e-3,
        "{label}: reference {what} is degenerate (max |x| {ref_max:e}); a tolerance check \
         against it would pass on any kernel"
    );
    let got_max = got.iter().fold(0f32, |a, b| a.max(b.abs()));
    assert!(
        got_max > 1e-3,
        "{label}: kernel {what} is all-but-zero (max |x| {got_max:e}); the comparison would be \
         zeros against zeros"
    );
    let mut rel = 0f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (*g as f64 - *w).abs() / ref_max;
        if d > rel {
            rel = d;
            at = i;
        }
    }
    assert!(
        rel < tol,
        "{label}: {entry} {what} diverged from the f64 host reference (rel {rel:e} at index {at}, \
         ref_max {ref_max:e}). Both shaders accumulate in f32 over d_k terms; the measured floor \
         on this corpus is printed by every green run of this suite, and {tol:e} sits well above \
         it. Contrast the graph-level fixture, whose rel < 0.05 on 4-layer logits is 300x looser \
         than the effect it would have to see."
    );
    rel
}

const TOL: f64 = 1e-5;

#[test]
fn q3w_delta_recurrent_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[q3d-deltanet-oracle] adapter: {}", ctx.info.name);
    let src = decode_source();
    let mut worst = 0f64;
    for c in cases_m1() {
        assert_eq!(c.m, 1, "the decode entry carries no token loop");
        let (want_out, want_state) = reference(&c);

        let q = dispatch::storage_from_slice(ctx, "q", &c.q);
        let k = dispatch::storage_from_slice(ctx, "k", &c.k);
        let v = dispatch::storage_from_slice(ctx, "v", &c.v);
        let g = dispatch::storage_from_slice(ctx, "g", &c.g);
        let beta = dispatch::storage_from_slice(ctx, "beta", &c.beta);
        let out = dispatch::storage_zeroed(ctx, "out", (c.heads * c.dv * 4) as u64);
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

        dispatch::run(
            ctx,
            "q3d-deltanet-oracle",
            &src,
            DECODE_ENTRY,
            &[
                (30, &q),
                (31, &k),
                (32, &v),
                (33, &g),
                (34, &beta),
                (35, &out),
                (36, &state),
                (37, &p),
            ],
            (c.heads as u32, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {DECODE_ENTRY}: {e}", c.label));

        let got_out: Vec<f32> =
            dispatch::read_back(ctx, &out, c.heads * c.dv).expect("read back delta out");
        let got_state: Vec<f32> =
            dispatch::read_back(ctx, &state, c.heads * c.dk * c.dv).expect("read back delta state");

        let ro = check(c.label, DECODE_ENTRY, "output", &got_out, &want_out, TOL);
        let rs = check(c.label, DECODE_ENTRY, "state", &got_state, &want_state, TOL);
        eprintln!(
            "[q3d-deltanet-oracle] {} decode: out rel={ro:.3e} state rel={rs:.3e}",
            c.label
        );
        worst = worst.max(ro).max(rs);
    }
    eprintln!("[q3d-deltanet-oracle] worst decode relative error: {worst:.3e}");
}

#[test]
fn q3w_delta_scan_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    let mut worst = 0f64;
    let mut saw_multi_token = false;
    for c in cases_mrow() {
        saw_multi_token |= c.m > 1;
        let (want_out, want_state) = reference(&c);

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

        dispatch::run(
            ctx,
            "q3d-deltanet-oracle",
            &src,
            PREFILL_ENTRY,
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
            (c.heads as u32, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {PREFILL_ENTRY}: {e}", c.label));

        let got_out: Vec<f32> =
            dispatch::read_back(ctx, &out, c.m * c.heads * c.dv).expect("read back scan out");
        let got_state: Vec<f32> =
            dispatch::read_back(ctx, &state, c.heads * c.dk * c.dv).expect("read back scan state");

        let ro = check(c.label, PREFILL_ENTRY, "output", &got_out, &want_out, TOL);
        let rs = check(
            c.label,
            PREFILL_ENTRY,
            "state",
            &got_state,
            &want_state,
            TOL,
        );
        eprintln!(
            "[q3d-deltanet-oracle] {} scan: out rel={ro:.3e} state rel={rs:.3e}",
            c.label
        );
        worst = worst.max(ro).max(rs);
    }
    assert!(
        saw_multi_token,
        "every scan case has m_live == 1, where the token loop runs once and the scan is just \
         the decode kernel; the ordering property this entry exists for would be untested"
    );
    eprintln!("[q3d-deltanet-oracle] worst scan relative error: {worst:.3e}");
}

#[test]
fn the_corpus_decays_the_state_and_gates_the_update() {
    let mut worst_g = 1f32;
    let mut worst_beta = 1f32;
    let mut state_energy = 0f32;
    for c in cases_m1().into_iter().chain(cases_mrow()) {
        for v in &c.g {
            worst_g = worst_g.min(*v);
        }
        for v in &c.beta {
            worst_beta = worst_beta.min((*v - 1.0).abs());
        }
        state_energy = state_energy.max(c.state0.iter().fold(0f32, |a, b| a.max(b.abs())));
    }
    eprintln!(
        "[q3d-deltanet-oracle] min g {worst_g:.4}, min |beta-1| {worst_beta:.4}, \
         max |state0| {state_energy:.4}"
    );
    assert!(
        worst_g < 0.95,
        "no case decays the state by more than {worst_g}; with g ~ 1 the `state * g` multiply is \
         an identity and dropping it is invisible"
    );
    assert!(
        worst_beta > 0.05,
        "beta is within {worst_beta} of 1 everywhere; with beta == 1 the `* bt` gate is an \
         identity and dropping it is invisible"
    );
    assert!(
        state_energy > 1e-2,
        "the carried-in state is ~zero ({state_energy}); the decay multiply then has nothing to \
         act on for the first token"
    );
}
