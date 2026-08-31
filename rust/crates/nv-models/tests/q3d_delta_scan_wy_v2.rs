#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const WY_ENTRY: &str = "q3w_delta_scan_wy";
const WY_CK_ENTRY: &str = "q3w_delta_scan_wy_ck";
const WY_O_ENTRY: &str = "q3w_delta_scan_wy_o";
const WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS: u32 = 4;

const WY_GPU_TOL_REL_1E4_THE_WY_ARM_PRECEDENT_FOR_REASSOCIATED_F32_PREFILL_NUMERICS: f64 = 1e-4;

const V2_GATES_BITWISE_BECAUSE_THE_SPLIT_REPLAYS_THE_INLINE_ARMS_OP_ORDER_THROUGH_EXACT_F32_CHECKPOINTS:
    &str = "the ladder keeps the exact u/state arithmetic of the inline WY arm and the o pass \
     recomputes exp(cl), W and q*S_ckpt with the same fma chains over the same f32 bits (state \
     checkpoints are raw f32, cl is checkpointed rather than recomputed), so out and state must \
     match the single-dispatch arm bit for bit; a mismatch is an op-order drift in the split, \
     not tolerable reassociation";

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

fn rel_err(got: &[f64], want: &[f64], what: &str, label: &str) -> f64 {
    assert_eq!(got.len(), want.len());
    let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        ref_max > 1e-3,
        "{label}: reference {what} is degenerate (max |x| {ref_max:e}); the tolerance check \
         would pass on anything"
    );
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs() / ref_max)
        .fold(0f64, f64::max)
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
        "q3d_delta_scan_wy_v2 needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn prefill_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    for entry in [WY_ENTRY, WY_CK_ENTRY, WY_O_ENTRY] {
        assert!(
            src.contains(&format!("fn {entry}(")),
            "the shipped prefill source no longer declares {entry}; the entry moved and this \
             gate is now testing nothing"
        );
    }
    src
}

fn run_scan_v2(c: &Case, src: &str) -> (Vec<f32>, Vec<f32>) {
    let ctx = ctx();
    let chunks = c.m.div_ceil(32);
    let q = dispatch::storage_from_slice(ctx, "q", &c.q);
    let k = dispatch::storage_from_slice(ctx, "k", &c.k);
    let v = dispatch::storage_from_slice(ctx, "v", &c.v);
    let g = dispatch::storage_from_slice(ctx, "g", &c.g);
    let beta = dispatch::storage_from_slice(ctx, "beta", &c.beta);
    let out = dispatch::storage_zeroed(ctx, "out", (c.m * c.heads * c.dv * 4) as u64);
    let state = dispatch::storage_from_slice(ctx, "state", &c.state0);
    let ck_cl = dispatch::storage_zeroed(ctx, "ckcl", (chunks * c.heads * 32 * 4) as u64);
    let ck_s =
        dispatch::storage_zeroed(ctx, "cks", (chunks * c.heads * c.dk * c.dv * 4) as u64);
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
        "q3d-scan-wy-v2-ck",
        src,
        WY_CK_ENTRY,
        &[
            (39, &ck_cl),
            (41, &k),
            (42, &v),
            (43, &g),
            (44, &beta),
            (45, &out),
            (46, &state),
            (47, &p),
            (48, &ck),
            (49, &ck_s),
        ],
        (
            c.heads as u32,
            WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS,
            1,
        ),
    )
    .unwrap_or_else(|e| panic!("{}: dispatch {WY_CK_ENTRY}: {e}", c.label));
    dispatch::run(
        ctx,
        "q3d-scan-wy-v2-o",
        src,
        WY_O_ENTRY,
        &[
            (39, &ck_cl),
            (40, &q),
            (41, &k),
            (45, &out),
            (47, &p),
            (48, &ck),
            (49, &ck_s),
        ],
        (
            c.heads as u32,
            WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS,
            chunks as u32,
        ),
    )
    .unwrap_or_else(|e| panic!("{}: dispatch {WY_O_ENTRY}: {e}", c.label));
    let got_out: Vec<f32> =
        dispatch::read_back(ctx, &out, c.m * c.heads * c.dv).expect("read back v2 out");
    let got_state: Vec<f32> =
        dispatch::read_back(ctx, &state, c.heads * c.dk * c.dv).expect("read back v2 state");
    (got_out, got_state)
}

fn run_scan_wy(c: &Case, src: &str) -> (Vec<f32>, Vec<f32>) {
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
    dispatch::run(
        ctx,
        "q3d-scan-wy-v2-ref",
        src,
        WY_ENTRY,
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
        (
            c.heads as u32,
            WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS,
            1,
        ),
    )
    .unwrap_or_else(|e| panic!("{}: dispatch {WY_ENTRY}: {e}", c.label));
    let got_out: Vec<f32> =
        dispatch::read_back(ctx, &out, c.m * c.heads * c.dv).expect("read back wy out");
    let got_state: Vec<f32> =
        dispatch::read_back(ctx, &state, c.heads * c.dk * c.dv).expect("read back wy state");
    (got_out, got_state)
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn assert_bitwise(got: &[f32], want: &[f32], what: &str, label: &str) {
    assert_eq!(got.len(), want.len());
    let mut mismatches = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            if first.is_none() {
                first = Some((i, *g, *w));
            }
            mismatches += 1;
        }
    }
    assert!(
        mismatches == 0,
        "{label}: {what} of the v2 split differs from the inline WY arm in {mismatches}/{} \
         elements (first at [{}]: v2 {:?} vs wy {:?}). \
         {V2_GATES_BITWISE_BECAUSE_THE_SPLIT_REPLAYS_THE_INLINE_ARMS_OP_ORDER_THROUGH_EXACT_F32_CHECKPOINTS}",
        got.len(),
        first.map(|f| f.0).unwrap_or(0),
        first.map(|f| f.1),
        first.map(|f| f.2),
    );
}

#[test]
fn wy_v2_split_matches_the_inline_wy_arm_bitwise_and_the_f64_reference() {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    if !ctx.caps.workgroup_storage_fits(
        nv_models::qwen3_5_dense_wgpu::WY_SCAN_WORKGROUP_BYTES_TWO_16K_VEC4_TILES_TWO_4K_ROWS_TWO_128B_LANES,
    ) {
        eprintln!(
            "SKIP: adapter workgroup storage {} < the {} B the wy tiles need; the serving route \
             refuses this kernel here (pf_scan_wy_route), so a red on it would test an \
             unreachable configuration",
            ctx.caps.max_compute_workgroup_storage_size,
            nv_models::qwen3_5_dense_wgpu::WY_SCAN_WORKGROUP_BYTES_TWO_16K_VEC4_TILES_TWO_4K_ROWS_TWO_128B_LANES
        );
        return;
    }
    let src = prefill_source();
    let tol = WY_GPU_TOL_REL_1E4_THE_WY_ARM_PRECEDENT_FOR_REASSOCIATED_F32_PREFILL_NUMERICS;
    let mut worst = 0f64;
    let mut saw_partial_tail_chunk = false;
    let mut saw_full_chunks = false;
    for m in [1usize, 6, 32, 67, 256] {
        saw_partial_tail_chunk |= m % 32 != 0;
        saw_full_chunks |= m >= 64;
        let c = case(32, 128, 128, m, 0x3b_77e5 ^ m as u64);
        let (want_out, want_state) = seq_reference(&c);
        let (wy_out, wy_state) = run_scan_wy(&c, &src);
        let (v2_out, v2_state) = run_scan_v2(&c, &src);

        assert_bitwise(&v2_state, &wy_state, "state", &c.label);
        assert_bitwise(&v2_out, &wy_out, "out", &c.label);

        let ro = rel_err(&to_f64(&v2_out), &want_out, "output", &c.label);
        let rs = rel_err(&to_f64(&v2_state), &want_state, "state", &c.label);
        assert!(
            ro < tol && rs < tol,
            "{}: the v2 split diverged from the f64 sequential reference (out rel {ro:e}, \
             state rel {rs:e} vs tol {tol:e}); bitwise identity to a wrong arm would not \
             catch a shared defect, this bound does",
            c.label
        );
        eprintln!(
            "[q3d-scan-wy-v2] {}: bitwise vs wy OK; vs f64 out {ro:.3e} state {rs:.3e}",
            c.label
        );
        worst = worst.max(ro).max(rs);
    }
    assert!(
        saw_partial_tail_chunk && saw_full_chunks,
        "the corpus must exercise both full 32-token chunks and a partial tail chunk; without \
         both, the chunk-seam arithmetic is untested"
    );
    eprintln!("[q3d-scan-wy-v2] worst gpu rel vs f64: {worst:.3e}");
}
