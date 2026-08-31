#![cfg(feature = "wgpu")]

mod common;
use common::bf16;
use common::pack_bf16_from_f64 as pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const SPLIT_ENTRY: &str = "q3w_delta_head_split_dv32";
const QKV_GATED_ENTRY: &str = "q3w_delta_qkv_gated";
const RECURRENT_ENTRY: &str = "q3w_delta_recurrent";
const DK: usize = 128;
const DV: usize = 128;
const SPLIT_GRID_Y: u32 = 4;
const TOL_F32: f64 = 1e-5;

const WHY: &str = "q3w_delta_head_split_dv32 ports the structural idea behind mlx-lm's \
     gated-delta decode kernel (mlx_lm/models/gated_delta.py): instead of one workgroup per \
     value head, each head is split across dv-column groups of workgroups, and each thread \
     keeps its slice of the 128x128 state in registers between the decay/kv_mem pass and the \
     delta-update pass, so the state crosses global memory once each way per token. The \
     shipped fused entries read the state twice per token and dispatch only n_v workgroups; \
     the 512-lane fused_wide arm already measured a wall null, but that arm scaled lanes \
     inside the same n_v workgroups, while this entry scales the workgroup count itself \
     (n_v x 4) the way mlx's grid does. Only a later rate probe can claim a speedup; this \
     suite is the correctness gate.";

const STATE_POLICY: &str = "The oracle runs a MULTI-STEP sequence from a NONZERO uploaded \
     state: a single step from zero state multiplies zeros through the decay and cannot see \
     a defect in how the carried state is read, and a sequence from zero never reads a state \
     slot the kernel did not itself write.";

const BOUNDS_DOC: &str = "TOL_F32=1e-5 relative against an f64 host reference, the same bar \
     graph_q3d_delta_decode_oracle uses for the f32-writing delta entries; the reference \
     accumulates in the opposite index order from the shader, so agreement past reduction \
     reassociation noise (~1e-7 on these corpora) is arithmetic agreement.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "q3d_delta_mlx_port needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn split_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_delta_source();
    for e in [SPLIT_ENTRY, QKV_GATED_ENTRY, RECURRENT_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "the shipped delta source no longer declares {e}; this gate is testing nothing"
        );
    }
    assert!(
        src.contains("@workgroup_size(512)\nfn q3w_delta_head_split_dv32("),
        "{SPLIT_ENTRY} no longer runs 512 lanes; the grid math below (32 cols x 16 chunks, \
         grid.y {SPLIT_GRID_Y}) is stale"
    );
    src
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

fn f32v(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
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
        "{label}: the kernel {what} is all-but-zero (max |x| {got_max:e}); the comparison \
         would be zeros against zeros"
    );
    let distinct: std::collections::BTreeSet<u64> = want.iter().map(|v| v.to_bits()).collect();
    assert!(
        distinct.len() > 2,
        "{label}: the reference {what} has only {} distinct values over {} lanes; an index \
         defect would permute identical numbers and pass",
        distinct.len(),
        want.len()
    );
}

fn rel_error(got: &[f64], want: &[f64]) -> (f64, usize) {
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
    (rel, at)
}

fn check_rel(label: &str, what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_nondegenerate(label, what, want, got);
    let (rel, at) = rel_error(got, want);
    assert!(
        rel < TOL_F32,
        "{label}: {SPLIT_ENTRY} {what} diverged from the f64 host reference (rel {rel:e} at \
         index {at}, kernel {} vs reference {}). {BOUNDS_DOC}",
        got[at],
        want[at]
    );
    rel
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
struct DgParams {
    n_v: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

struct Case {
    label: &'static str,
    n_v: usize,
    v_per_k: usize,
    steps: usize,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            label: "served nv48 vpk3 dk128 dv128 (qwen3.8 geometry, grid (48,4,1))",
            n_v: 48,
            v_per_k: 3,
            steps: 3,
        },
        Case {
            label: "nv4 vpk1 dk128 dv128",
            n_v: 4,
            v_per_k: 1,
            steps: 4,
        },
        Case {
            label: "nv6 vpk3 dk128 dv128 (three heads share one key head)",
            n_v: 6,
            v_per_k: 3,
            steps: 3,
        },
    ]
}

impl Case {
    fn key_dim(&self) -> usize {
        self.n_v / self.v_per_k * DK
    }
    fn mixed_len(&self) -> usize {
        2 * self.key_dim() + self.n_v * DV
    }
}

struct StepData {
    mixed: Vec<f64>,
    ab: Vec<f64>,
}

struct CaseData {
    steps: Vec<StepData>,
    alog: Vec<f64>,
    dt: Vec<f64>,
    state0: Vec<f64>,
}

const SCALE: f64 = 1.7;

fn case_data(c: &Case) -> CaseData {
    let mut r = Lcg::new(0x517d_0000 ^ (c.n_v as u64) << 8 ^ c.v_per_k as u64);
    let steps = (0..c.steps)
        .map(|_| StepData {
            mixed: r.f32_vec(c.mixed_len(), 1.2),
            ab: r.bf16_vec(2 * c.n_v, 2.5),
        })
        .collect();
    CaseData {
        steps,
        alog: r.f32_vec(c.n_v, 0.8),
        dt: r.f32_vec(c.n_v, 1.2),
        state0: r.f32_vec(c.n_v * DK * DV, 0.5),
    }
}

fn gate_reference(c: &Case, d: &CaseData, s: &StepData, h: usize) -> (f64, f64) {
    let a = s.ab[h];
    let b = s.ab[c.n_v + h];
    let beta = 1.0 / (1.0 + (-b).exp());
    let t = a + d.dt[h];
    let sp = t.max(0.0) + (1.0 + (-t.abs()).exp()).ln();
    ((sp * -(d.alog[h].exp())).exp(), beta)
}

fn reference_step(c: &Case, d: &CaseData, s: &StepData, state: &mut [f64]) -> Vec<f64> {
    let key_dim = c.key_dim();
    let n_k = c.n_v / c.v_per_k;
    let mut qn = vec![0f64; n_k * DK];
    let mut kn = vec![0f64; n_k * DK];
    for kh in (0..n_k).rev() {
        let mut sq = 0f64;
        let mut sk = 0f64;
        for e in (0..DK).rev() {
            let q = s.mixed[kh * DK + e];
            let k = s.mixed[key_dim + kh * DK + e];
            sq += q * q;
            sk += k * k;
        }
        let nq = (sq + 1e-6).sqrt();
        let nk_ = (sk + 1e-6).sqrt();
        for e in 0..DK {
            qn[kh * DK + e] = s.mixed[kh * DK + e] / nq * SCALE;
            kn[kh * DK + e] = s.mixed[key_dim + kh * DK + e] / nk_;
        }
    }
    let mut core = vec![0f64; c.n_v * DV];
    for h in (0..c.n_v).rev() {
        let (g, beta) = gate_reference(c, d, s, h);
        let kh = h / c.v_per_k;
        let sb = h * DK * DV;
        let mut kv_mem = vec![0f64; DV];
        for i in (0..DK).rev() {
            for col in (0..DV).rev() {
                state[sb + i * DV + col] *= g;
                kv_mem[col] += state[sb + i * DV + col] * kn[kh * DK + i];
            }
        }
        for col in 0..DV {
            let vv = s.mixed[2 * key_dim + h * DV + col];
            let delta = (vv - kv_mem[col]) * beta;
            for i in 0..DK {
                state[sb + i * DV + col] += kn[kh * DK + i] * delta;
                core[h * DV + col] += state[sb + i * DV + col] * qn[kh * DK + i];
            }
        }
    }
    core
}

fn uniforms(ctx: &WgpuContext, c: &Case) -> (wgpu::Buffer, wgpu::Buffer) {
    let dq_p = dispatch::uniform_from(
        ctx,
        "mlx-dq-p",
        &DqParams {
            n_v: c.n_v as u32,
            d_k: DK as u32,
            d_v: DV as u32,
            key_dim: c.key_dim() as u32,
            v_per_k: c.v_per_k as u32,
            pad0: 0,
            pad1: 0,
            scale: SCALE as f32,
        },
    );
    let dg_p = dispatch::uniform_from(
        ctx,
        "mlx-dg-p",
        &DgParams {
            n_v: c.n_v as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    (dq_p, dg_p)
}

fn run_split_sequence(
    ctx: &WgpuContext,
    src: &str,
    c: &Case,
    d: &CaseData,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let (dq_p, dg_p) = uniforms(ctx, c);
    let alog_b = dispatch::storage_from_slice(ctx, "mlx-alog", &f32v(&d.alog));
    let dt_b = dispatch::storage_from_slice(ctx, "mlx-dt", &f32v(&d.dt));
    let state_b = dispatch::storage_from_slice(ctx, "mlx-state", &f32v(&d.state0));
    let mut cores = Vec::with_capacity(c.steps);
    let mut states = Vec::with_capacity(c.steps);
    for (t, s) in d.steps.iter().enumerate() {
        let mixed_b = dispatch::storage_from_slice(ctx, "mlx-mixed", &f32v(&s.mixed));
        let ab_b = dispatch::storage_from_slice(ctx, "mlx-ab", &pack_bf16(&s.ab));
        let core_b = dispatch::storage_zeroed(ctx, "mlx-core", (c.n_v * DV * 4) as u64);
        dispatch::run(
            ctx,
            "q3d-delta-mlx-port",
            src,
            SPLIT_ENTRY,
            &[
                (10, &mixed_b),
                (14, &dq_p),
                (20, &ab_b),
                (21, &alog_b),
                (22, &dt_b),
                (25, &dg_p),
                (35, &core_b),
                (36, &state_b),
            ],
            (c.n_v as u32, SPLIT_GRID_Y, 1),
        )
        .map_err(|e| format!("step {t}: {e}"))?;
        let core: Vec<f32> =
            dispatch::read_back(ctx, &core_b, c.n_v * DV).map_err(|e| format!("core {t}: {e}"))?;
        let state: Vec<f32> = dispatch::read_back(ctx, &state_b, c.n_v * DK * DV)
            .map_err(|e| format!("state {t}: {e}"))?;
        cores.push(to64(&core));
        states.push(to64(&state));
    }
    Ok((cores, states))
}

fn run_pair_sequence(
    ctx: &WgpuContext,
    src: &str,
    c: &Case,
    d: &CaseData,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let (dq_p, dg_p) = uniforms(ctx, c);
    let dr_p = dispatch::uniform_from(
        ctx,
        "mlx-dr-p",
        &DrParams {
            heads: c.n_v as u32,
            d_k: DK as u32,
            d_v: DV as u32,
            pad0: 0,
        },
    );
    let alog_b = dispatch::storage_from_slice(ctx, "mlx-alog", &f32v(&d.alog));
    let dt_b = dispatch::storage_from_slice(ctx, "mlx-dt", &f32v(&d.dt));
    let state_b = dispatch::storage_from_slice(ctx, "mlx-state", &f32v(&d.state0));
    let q_b = dispatch::storage_zeroed(ctx, "mlx-q", (c.n_v * DK * 4) as u64);
    let k_b = dispatch::storage_zeroed(ctx, "mlx-k", (c.n_v * DK * 4) as u64);
    let v_b = dispatch::storage_zeroed(ctx, "mlx-v", (c.n_v * DV * 4) as u64);
    let g_b = dispatch::storage_zeroed(ctx, "mlx-g", (c.n_v * 4) as u64);
    let beta_b = dispatch::storage_zeroed(ctx, "mlx-beta", (c.n_v * 4) as u64);
    let mut cores = Vec::with_capacity(c.steps);
    let mut states = Vec::with_capacity(c.steps);
    for (t, s) in d.steps.iter().enumerate() {
        let mixed_b = dispatch::storage_from_slice(ctx, "mlx-mixed", &f32v(&s.mixed));
        let ab_b = dispatch::storage_from_slice(ctx, "mlx-ab", &pack_bf16(&s.ab));
        let core_b = dispatch::storage_zeroed(ctx, "mlx-core", (c.n_v * DV * 4) as u64);
        dispatch::run(
            ctx,
            "q3d-delta-mlx-port",
            src,
            QKV_GATED_ENTRY,
            &[
                (10, &mixed_b),
                (11, &q_b),
                (12, &k_b),
                (13, &v_b),
                (14, &dq_p),
                (20, &ab_b),
                (21, &alog_b),
                (22, &dt_b),
                (23, &g_b),
                (24, &beta_b),
                (25, &dg_p),
            ],
            (c.n_v as u32, 1, 1),
        )
        .map_err(|e| format!("{QKV_GATED_ENTRY} step {t}: {e}"))?;
        dispatch::run(
            ctx,
            "q3d-delta-mlx-port",
            src,
            RECURRENT_ENTRY,
            &[
                (30, &q_b),
                (31, &k_b),
                (32, &v_b),
                (33, &g_b),
                (34, &beta_b),
                (35, &core_b),
                (36, &state_b),
                (37, &dr_p),
            ],
            (c.n_v as u32, 1, 1),
        )
        .map_err(|e| format!("{RECURRENT_ENTRY} step {t}: {e}"))?;
        let core: Vec<f32> =
            dispatch::read_back(ctx, &core_b, c.n_v * DV).map_err(|e| format!("core {t}: {e}"))?;
        let state: Vec<f32> = dispatch::read_back(ctx, &state_b, c.n_v * DK * DV)
            .map_err(|e| format!("state {t}: {e}"))?;
        cores.push(to64(&core));
        states.push(to64(&state));
    }
    Ok((cores, states))
}

fn reference_sequence(c: &Case, d: &CaseData) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut state = d.state0.clone();
    let mut cores = Vec::with_capacity(c.steps);
    let mut states = Vec::with_capacity(c.steps);
    for s in &d.steps {
        cores.push(reference_step(c, d, s, &mut state));
        states.push(state.clone());
    }
    (cores, states)
}

#[test]
fn q3w_delta_head_split_dv32_matches_an_f64_host_reference_over_a_carried_state_sequence() {
    let ctx = ctx();
    eprintln!(
        "[q3d-delta-mlx-port] adapter: {}\n{WHY}\n{STATE_POLICY}\n{BOUNDS_DOC}",
        ctx.info.name
    );
    let src = split_source();
    let mut worst = 0f64;
    for c in cases() {
        let d = case_data(&c);
        let (want_core, want_state) = reference_sequence(&c, &d);
        let (got_core, got_state) =
            run_split_sequence(ctx, &src, &c, &d).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        for t in 0..c.steps {
            let rc = check_rel(c.label, &format!("core at step {t}"), &got_core[t], &want_core[t]);
            let rs =
                check_rel(c.label, &format!("state at step {t}"), &got_state[t], &want_state[t]);
            worst = worst.max(rc).max(rs);
            eprintln!(
                "[q3d-delta-mlx-port] {} step {t}: core rel={rc:.3e} state rel={rs:.3e}",
                c.label
            );
        }
    }
    eprintln!("[q3d-delta-mlx-port] worst relative error: {worst:.3e}");
}

#[test]
fn the_split_corpus_decays_and_separates_heads_and_steps() {
    for c in cases() {
        let d = case_data(&c);
        let (_, states) = reference_sequence(&c, &d);
        assert!(
            c.steps >= 3,
            "{}: fewer than three steps never reads a state slot the kernel wrote two tokens \
             ago. {STATE_POLICY}",
            c.label
        );
        for w in states.windows(2) {
            assert_ne!(
                w[0], w[1],
                "{}: the reference state does not change between consecutive steps, so a \
                 kernel that never wrote the state would pass",
                c.label
            );
        }
        let mut g_lo = f64::INFINITY;
        let mut beta_gap = 0f64;
        for s in &d.steps {
            for h in 0..c.n_v {
                let (g, beta) = gate_reference(&c, &d, s, h);
                g_lo = g_lo.min(g);
                beta_gap = beta_gap.max((beta - 0.5).abs());
            }
        }
        assert!(
            g_lo < 0.95,
            "{}: the smallest decay is {g_lo}; with g ~ 1 the `state * g` the split entry \
             keeps in registers is an identity and the decay mutant below would be inert",
            c.label
        );
        assert!(
            beta_gap > 0.2,
            "{}: beta never leaves the neighbourhood of 0.5; the beta-dropped mutant would \
             move almost nothing",
            c.label
        );
        let state_max = d.state0.iter().fold(0f64, |a, b| a.max(b.abs()));
        assert!(
            state_max > 1e-2,
            "{}: the uploaded initial state is all-but-zero ({state_max:e}). {STATE_POLICY}",
            c.label
        );
        eprintln!(
            "[q3d-delta-mlx-port] {}: min g {g_lo:.4}, worst |beta-0.5| {beta_gap:.4}, |state0| \
             max {state_max:.4}",
            c.label
        );
    }
}

#[test]
fn the_split_entry_agrees_with_the_shipped_qkv_gated_plus_recurrent_pair() {
    let ctx = ctx();
    let src = split_source();
    for c in cases() {
        let d = case_data(&c);
        let (pair_core, pair_state) =
            run_pair_sequence(ctx, &src, &c, &d).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        let (split_core, split_state) =
            run_split_sequence(ctx, &src, &c, &d).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        for t in 0..c.steps {
            let (rc, ci) = rel_error(&split_core[t], &pair_core[t]);
            let (rs, si) = rel_error(&split_state[t], &pair_state[t]);
            assert_nondegenerate(c.label, "pair core", &pair_core[t], &split_core[t]);
            assert!(
                rc < TOL_F32 && rs < TOL_F32,
                "{}: step {t}: {SPLIT_ENTRY} disagrees with the shipped \
                 {QKV_GATED_ENTRY}+{RECURRENT_ENTRY} pair beyond reduction-order noise (core \
                 rel {rc:e} at {ci}, state rel {rs:e} at {si}). The two arms compute one \
                 operator from one buffer set, so past this bar one of them is wrong. {WHY}",
                c.label
            );
        }
        eprintln!(
            "[q3d-delta-mlx-port] {}: split entry within {TOL_F32:e} of the shipped pair over \
             {} steps",
            c.label, c.steps
        );
    }
}

const MUTANTS: [(&str, &str, &str); 5] = [
    (
        "split-decay-dropped",
        "let s = dr_state[sbase + (r0 + i) * dv] * ge;",
        "let s = dr_state[sbase + (r0 + i) * dv];",
    ),
    (
        "split-beta-dropped",
        "let delta = bt * (vv - kv_mem);",
        "let delta = vv - kv_mem;",
    ),
    (
        "split-kv-mem-reads-an-unreduced-partial",
        "let kv_mem = fw_part[col_l];",
        "let kv_mem = fw_part[lid.x];",
    ),
    (
        "split-state-written-without-the-delta-update",
        "dr_state[sbase + (r0 + i) * dv] = u;",
        "dr_state[sbase + (r0 + i) * dv] = st[i];",
    ),
    (
        "split-q-scale-dropped",
        "fh_q[lid.x] = (fh_q[lid.x] / nq) * dq_p.scale;",
        "fh_q[lid.x] = fh_q[lid.x] / nq;",
    ),
];

#[test]
fn every_split_mutant_is_caught_and_every_anchor_is_unique_to_the_new_entry() {
    let ctx = ctx();
    let src = split_source();
    let key = format!("fn {SPLIT_ENTRY}(");
    let at = src
        .find(&key)
        .unwrap_or_else(|| panic!("{SPLIT_ENTRY} lost its declaration"));
    let body_end = src[at..]
        .find("\n}\n")
        .map(|e| at + e + 3)
        .unwrap_or_else(|| panic!("no closing brace after `{key}`"));
    let body = &src[at..body_end];
    let c = Case {
        label: "mutant corpus nv6 vpk3 dk128 dv128",
        n_v: 6,
        v_per_k: 3,
        steps: 3,
    };
    let d = case_data(&c);
    let (want_core, want_state) = reference_sequence(&c, &d);
    for (name, from, to) in MUTANTS {
        assert_ne!(from, to, "mutant {name} replaces text with itself");
        assert_eq!(
            src.matches(from).count(),
            1,
            "the anchor of mutant {name} does not occur exactly once in the shipped delta \
             source; a shared anchor would mutate a neighbouring entry and report its redness \
             as this one's"
        );
        assert!(
            body.contains(from),
            "the anchor of mutant {name} sits outside the body of {SPLIT_ENTRY}"
        );
        let bad = src.replace(from, to);
        let caught = match run_split_sequence(ctx, &bad, &c, &d) {
            Err(_) => true,
            Ok((core, state)) => (0..c.steps).any(|t| {
                rel_error(&core[t], &want_core[t]).0 >= TOL_F32
                    || rel_error(&state[t], &want_state[t]).0 >= TOL_F32
            }),
        };
        assert!(
            caught,
            "mutant {name} of {SPLIT_ENTRY} was NOT caught; either the corpus lost the case \
             that saw the defect or the mutation is inert and must be replaced. {WHY}"
        );
        eprintln!("MUTANT {name}: caught");
    }
}
