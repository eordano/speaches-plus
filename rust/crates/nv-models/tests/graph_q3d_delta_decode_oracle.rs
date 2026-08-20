#![cfg(feature = "wgpu")]

mod common;
use common::bf16;
use common::bf16_bits_from_f64 as bf16_bits;
use common::pack_bf16_from_f64 as pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const CONV_ENTRY: &str = "q3w_delta_conv";
const GATING_ENTRY: &str = "q3w_delta_gating";
const OUT_ENTRY: &str = "q3w_delta_out";
const QKV_ENTRY: &str = "q3w_delta_qkv";
const QKV_GATED_ENTRY: &str = "q3w_delta_qkv_gated";
const GATE_LANE_FN: &str = "q3w_gate_lane";
const GATE_VALUES_FN: &str = "q3w_gate_values";
const OUT_TAIL_FN: &str = "q3w_out_tail";

const FUSED_GATE_TRANSITIVITY: &str = "q3w_delta_qkv_gated is one dispatch replacing the \
     q3w_delta_qkv + q3w_delta_gating pair, and its gate is BIT-IDENTITY to that pair run \
     separately on the same buffers. Both entries call the same q3w_split_lane and q3w_gate_lane \
     lane functions, so the identity is trivially true of the shipped text -- which is exactly the \
     point: any numeric defect in the shared lane functions turns the host-reference oracles for \
     q3w_delta_gating (this suite) and q3w_delta_qkv (graph_q3d_delta_front_oracle) red, and this \
     identity extends that coverage to the fused entry; any defect confined to the fused wrapper \
     (a wrong gate-lane index, a dropped lane-0 guard) breaks the identity itself.";

const WHY: &str = "The three DeltaNet DECODE entries of the Qwen3.5-dense graph, dispatched once \
     per token each at q3d-dn-conv, q3d-dn-gating and q3d-dn-out. graph_q3d_delta_front_oracle \
     gates their M-row PREFILL twins -- q3w_delta_conv_m, q3w_delta_gating_m, q3w_delta_out_m -- \
     and that is a gate on different text: the twins live in PREFILL_WGSL and the originals in \
     DELTA_WGSL, so shipped_prefill_source() does not contain one character of these three \
     function bodies and no mutation of them can turn the twin gate red. \
     entries_gated_here_are_absent_from_the_prefill_source proves that mechanically rather than \
     asserting it. DeltaNet is the mixer of 30 of Qwen3.6's 40 layers, so this is the linear- \
     attention path of the model this repo measures most, and until this suite the only thing \
     behind it was tiny_wgpu_decode_matches_cpu_reference at rel < 0.05 on 4-layer logits -- a bar \
     graph_q3d_elementwise_oracle already measured as 36x too loose to see a neighbouring kernel's \
     output quartered.";

const INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM: &str = "The decode entries and their M-row twins share \
     LINES: `let silu = acc / (1.0 + exp(-acc));` is character-identical in q3w_delta_conv and \
     q3w_delta_conv_m, and a mutation anchor is therefore no evidence of anything on its own. What \
     is true, and what the_decode_bodies_gated_here_are_no_part_of_the_prefill_source checks, is \
     that the whole FUNCTION BODY of each entry gated here is absent from shipped_prefill_source() \
     -- different bindings, different indexing, separately maintained. So an edit to one is not an \
     edit to the other, and the twin gate stays green through any defect introduced on this side. \
     The same check also pins every mutant's anchor to the bodies this suite has a corpus for: \
     src.replace rewrites every occurrence, and a mutant that reached an ungated neighbour would \
     be reporting another kernel's redness as this one's.";

const CONV_ORACLE_STATE_POLICY: &str = "q3w_delta_conv CARRIES STATE and this oracle models a \
     SEQUENCE OF DECODE STEPS STARTING FROM A NONZERO CARRIED STATE -- not one step, and not a \
     sequence from zero. Both alternatives are blind in a way this one is not. A single step from \
     zero state cannot see any defect in how the carried history is read, because every tap reads \
     the same zero; a_single_step_from_zero_state_cannot_see_a_history_indexing_defect runs the \
     shipped kernel and the reversed-history mutant in exactly that configuration and asserts they \
     agree BIT FOR BIT, which is what a zero-state test would have concluded. A sequence from zero \
     state recovers most of that, because from step 1 the history holds values the kernel itself \
     wrote -- but it still never reads a history slot the kernel did not produce, so it cannot \
     distinguish a kernel that reads the carried state from one that ignores it on the first step \
     of a continuation, which is every token after a prefill. The reference is therefore a causal \
     depthwise convolution over the CONCATENATED stream [carried history, x_0 .. x_(T-1)], and \
     both the per-step output AND the full state after every step are asserted.";

const BOUNDS_DOC: &str = "TOL_F32=1e-5 for q3w_delta_gating, which writes f32: the same bar the \
     recurrence and the prefill-front oracles use for the same graph, and ~50x above the f32 \
     rounding floor these corpora reach. ONE BF16 ULP for q3w_delta_conv and q3w_delta_out, which \
     round their result to bf16 before storing it: the output format's own resolution is the only \
     thing an f32-versus-f64 evaluation of the same expression can move, so anything past it is \
     arithmetic and not rounding, and a relative bound would be capped at 2^-8 anyway. BIT-EXACT \
     for the conv STATE, which stores a bf16-decoded input verbatim with no arithmetic at all.";

const TOL_F32: f64 = 1e-5;

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_delta_decode_oracle needs a real wgpu adapter; a skipped numeric gate reads as \
         a passed one, so this panics rather than returning early",
    )
}

fn decode_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_delta_source();
    for e in [
        CONV_ENTRY,
        GATING_ENTRY,
        OUT_ENTRY,
        QKV_ENTRY,
        QKV_GATED_ENTRY,
        GATE_LANE_FN,
        GATE_VALUES_FN,
        OUT_TAIL_FN,
    ] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "the shipped delta source no longer declares {e}; the entry moved and this gate is now \
             testing nothing"
        );
    }
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

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn ulp_bf16(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return f32::MIN_POSITIVE as f64;
    }
    (v.abs().log2().floor() - 7.0).exp2()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
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
        "{label}: the reference {what} has only {} distinct values over {} lanes; an index defect \
         would permute identical numbers and pass",
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

fn check_rel(label: &str, entry: &str, what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_nondegenerate(label, what, want, got);
    let (rel, at) = rel_error(got, want);
    assert!(
        rel < TOL_F32,
        "{label}: {entry} {what} diverged from the f64 host reference (rel {rel:e} at index {at}). \
         The reference is written from the operator's definition and evaluated in the OPPOSITE \
         index order from the shader, so it is not the kernel restated. {BOUNDS_DOC}"
    );
    rel
}

fn bf16_ulp_error(got: &[f64], want: &[f64]) -> (f64, usize, usize) {
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
    (worst, at, exact)
}

fn check_bf16_ulp(label: &str, entry: &str, what: &str, got: &[f64], want: &[f64]) -> (f64, usize) {
    assert_nondegenerate(label, what, want, got);
    let (worst, at, exact) = bf16_ulp_error(got, want);
    assert!(
        worst <= 1.0,
        "{label}: {entry} {what} is more than one bf16 ulp from the f64 host reference \
         ({worst:.3} ulp at index {at}: kernel {} vs reference {}). {BOUNDS_DOC}",
        got[at],
        want[at]
    );
    (worst, exact)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CvParams {
    conv_dim: u32,
    kernel: u32,
    pad0: u32,
    pad1: u32,
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
struct DoParams {
    n_v: u32,
    d_v: u32,
    pad0: u32,
    eps: f32,
}

struct ConvCase {
    label: &'static str,
    conv_dim: usize,
    kernel: usize,
    steps: usize,
}

fn conv_cases() -> Vec<ConvCase> {
    vec![
        ConvCase {
            label: "cd128 ks4 6 steps (shipped kernel width)",
            conv_dim: 128,
            kernel: 4,
            steps: 6,
        },
        ConvCase {
            label: "cd64 ks2 5 steps (hist 1, the roll loop body never runs)",
            conv_dim: 64,
            kernel: 2,
            steps: 5,
        },
        ConvCase {
            label: "cd48 ks3 4 steps",
            conv_dim: 48,
            kernel: 3,
            steps: 4,
        },
    ]
}

struct ConvData {
    x: Vec<f64>,
    w: Vec<f64>,
    state0: Vec<f64>,
}

impl ConvCase {
    fn hist(&self) -> usize {
        self.kernel - 1
    }
    fn data(&self, zero_state: bool) -> ConvData {
        let mut r = Lcg::new(0xc0de_0000 ^ (self.conv_dim as u64) << 8 ^ self.kernel as u64);
        let x = r.bf16_vec(self.steps * self.conv_dim, 1.4);
        let w = r.f32_vec(self.conv_dim * self.kernel, 0.8);
        let state0 = if zero_state {
            vec![0f64; self.conv_dim * self.hist()]
        } else {
            r.f32_vec(self.conv_dim * self.hist(), 0.9)
        };
        ConvData { x, w, state0 }
    }
    fn stream(&self, d: &ConvData, ch: usize, i: isize) -> f64 {
        let hist = self.hist() as isize;
        if i < 0 {
            d.state0[ch * self.hist() + (i + hist) as usize]
        } else {
            d.x[i as usize * self.conv_dim + ch]
        }
    }
    fn reference(&self, d: &ConvData) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let hist = self.hist();
        let mut outs = Vec::with_capacity(self.steps);
        let mut states = Vec::with_capacity(self.steps);
        for t in 0..self.steps {
            let mut out = vec![0f64; self.conv_dim];
            let mut state = vec![0f64; self.conv_dim * hist];
            for ch in 0..self.conv_dim {
                let mut acc = 0f64;
                for j in (0..self.kernel).rev() {
                    let i = t as isize - hist as isize + j as isize;
                    acc += d.w[ch * self.kernel + j] * self.stream(d, ch, i);
                }
                out[ch] = bf16(silu(acc));
                for (j, slot) in state[ch * hist..ch * hist + hist].iter_mut().enumerate() {
                    *slot = self.stream(d, ch, t as isize - hist as isize + 1 + j as isize);
                }
            }
            outs.push(out);
            states.push(state);
        }
        (outs, states)
    }
}

fn run_conv_sequence(
    ctx: &WgpuContext,
    src: &str,
    c: &ConvCase,
    d: &ConvData,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let hist = c.hist();
    let state_b = dispatch::storage_from_slice(ctx, "cv-state", &f32v(&d.state0));
    let w_b = dispatch::storage_from_slice(ctx, "cv-w", &f32v(&d.w));
    let p_b = dispatch::uniform_from(
        ctx,
        "cv-p",
        &CvParams {
            conv_dim: c.conv_dim as u32,
            kernel: c.kernel as u32,
            pad0: 0,
            pad1: 0,
        },
    );
    let grid = dispatch::workgroup_count_1d(ctx, c.conv_dim as u64, 64);
    let mut outs = Vec::with_capacity(c.steps);
    let mut states = Vec::with_capacity(c.steps);
    for t in 0..c.steps {
        let row = &d.x[t * c.conv_dim..(t + 1) * c.conv_dim];
        let x_b = dispatch::storage_from_slice(ctx, "cv-x", &pack_bf16(row));
        let out_b = dispatch::storage_zeroed(ctx, "cv-out", (c.conv_dim * 4) as u64);
        dispatch::run(
            ctx,
            "q3d-delta-decode-oracle",
            src,
            CONV_ENTRY,
            &[
                (0, &x_b),
                (1, &w_b),
                (2, &state_b),
                (3, &out_b),
                (4, &p_b),
            ],
            grid,
        )
        .map_err(|e| format!("step {t}: {e}"))?;
        let out: Vec<f32> =
            dispatch::read_back(ctx, &out_b, c.conv_dim).map_err(|e| format!("out {t}: {e}"))?;
        let state: Vec<f32> = dispatch::read_back(ctx, &state_b, c.conv_dim * hist)
            .map_err(|e| format!("state {t}: {e}"))?;
        outs.push(to64(&out));
        states.push(to64(&state));
    }
    Ok((outs, states))
}

#[test]
fn q3w_delta_conv_matches_an_f64_host_reference_over_a_sequence_from_a_carried_state() {
    let ctx = ctx();
    eprintln!(
        "[q3d-delta-decode-oracle] adapter: {}\n{WHY}\n{CONV_ORACLE_STATE_POLICY}\n{BOUNDS_DOC}",
        ctx.info.name
    );
    let src = decode_source();
    let mut worst = 0f64;
    for c in conv_cases() {
        let d = c.data(false);
        let (want_out, want_state) = c.reference(&d);
        let (got_out, got_state) =
            run_conv_sequence(ctx, &src, &c, &d).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        for t in 0..c.steps {
            let (frac, exact) = check_bf16_ulp(
                c.label,
                CONV_ENTRY,
                &format!("conv+silu at step {t}"),
                &got_out[t],
                &want_out[t],
            );
            worst = worst.max(frac);
            for (i, (g, w)) in got_state[t].iter().zip(want_state[t].iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{}: {CONV_ENTRY} left the wrong conv history at slot {i} after step {t} \
                     ({g} vs {w}). The history stores a bf16-decoded input verbatim with no \
                     arithmetic, so the demand is bit-exactness, and what it holds is what the \
                     NEXT token convolves against. {CONV_ORACLE_STATE_POLICY}",
                    c.label
                );
            }
            eprintln!(
                "[q3d-delta-decode-oracle] {} step {t}: {exact}/{} out lanes bit-exact, worst \
                 {frac:.3} of one bf16 ulp, {} state slots bit-exact",
                c.label,
                want_out[t].len(),
                want_state[t].len()
            );
        }
    }
    eprintln!("[q3d-delta-decode-oracle] worst conv deviation: {worst:.3} bf16 ulp");
}

#[test]
fn the_conv_corpus_carries_a_live_state_and_exercises_both_silu_branches() {
    let mut min_state_energy = f64::INFINITY;
    let mut worst_tap_spread = 0f64;
    let mut negative = 0usize;
    let mut total = 0usize;
    let mut saw_empty_roll = false;
    let mut saw_full_roll = false;
    let mut saw_state_fully_rewritten = false;
    for c in conv_cases() {
        saw_empty_roll |= c.hist() == 1;
        saw_full_roll |= c.hist() >= 2;
        saw_state_fully_rewritten |= c.steps > c.hist();
        let d = c.data(false);
        min_state_energy =
            min_state_energy.min(d.state0.iter().fold(0f64, |a, b| a.max(b.abs())));
        for ch in 0..c.conv_dim {
            let taps = &d.w[ch * c.kernel..(ch + 1) * c.kernel];
            let lo = taps.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = taps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            worst_tap_spread = worst_tap_spread.max(hi - lo);
        }
        let (_, states) = c.reference(&d);
        for w in states.windows(2) {
            assert_ne!(
                w[0], w[1],
                "{}: the reference conv history does not change between consecutive steps, so a \
                 kernel that never wrote the state at all would pass",
                c.label
            );
        }
        assert_ne!(
            states[c.steps - 1],
            d.state0,
            "{}: the state after the last step equals the state the sequence started from",
            c.label
        );
        for t in 0..c.steps {
            for ch in 0..c.conv_dim {
                let mut acc = 0f64;
                for j in 0..c.kernel {
                    let i = t as isize - c.hist() as isize + j as isize;
                    acc += d.w[ch * c.kernel + j] * c.stream(&d, ch, i);
                }
                total += 1;
                negative += usize::from(acc < 0.0);
            }
        }
    }
    eprintln!(
        "[q3d-delta-decode-oracle] conv corpus: min |state0| max {min_state_energy:.4}, worst tap \
         spread {worst_tap_spread:.4}, {negative}/{total} accumulators negative"
    );
    assert!(
        min_state_energy > 1e-2,
        "some conv case starts from an all-but-zero carried state ({min_state_energy:e}); the \
         history taps then multiply zeros and this suite would be the zero-state test it exists to \
         avoid. {CONV_ORACLE_STATE_POLICY}"
    );
    assert!(
        worst_tap_spread > 0.1,
        "the conv taps of every channel are nearly equal ({worst_tap_spread}); a kernel that read \
         the wrong tap weight would produce the same number"
    );
    assert!(
        negative * 4 > total,
        "only {negative} of {total} conv accumulators are negative; silu is nearly linear on the \
         positive side and its saturating branch would go untested"
    );
    assert!(
        saw_empty_roll,
        "no case has kernel == 2, so the roll loop's zero-trip boundary is untested"
    );
    assert!(
        saw_full_roll,
        "no case has kernel >= 3, so the roll loop body never executes and a roll that moved the \
         history the wrong way would be invisible"
    );
    assert!(
        saw_state_fully_rewritten,
        "no case runs more steps than the kernel history is deep, so the suite never reaches a \
         step whose whole history was written by the kernel itself"
    );
}

const CONV_HISTORY_REVERSED: (&str, &str, &str) = (
    "conv-history-taps-read-in-reverse",
    "acc = fma(cv_w[c * ks + j], cv_state[c * hist + j], acc);",
    "acc = fma(cv_w[c * ks + j], cv_state[c * hist + hist - 1u - j], acc);",
);

#[test]
fn a_single_step_from_zero_state_cannot_see_a_history_indexing_defect() {
    let ctx = ctx();
    let src = decode_source();
    let (name, from, to) = CONV_HISTORY_REVERSED;
    let bad = mutate(&src, from, to);
    let c = ConvCase {
        label: "cd128 ks4 one step",
        conv_dim: 128,
        kernel: 4,
        steps: 1,
    };
    assert!(
        c.hist() >= 2,
        "reversing a history of {} slot(s) is the identity map, so this proof needs a deeper \
         kernel",
        c.hist()
    );

    let zero = c.data(true);
    let good_zero = run_conv_sequence(ctx, &src, &c, &zero).expect("shipped, zero state");
    let bad_zero = run_conv_sequence(ctx, &bad, &c, &zero).expect("mutant, zero state");
    for (i, (g, b)) in good_zero.0[0].iter().zip(bad_zero.0[0].iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            b.to_bits(),
            "at channel {i} the mutant {name} already disagrees with the shipped kernel from a \
             ZERO state in ONE step ({g} vs {b}). That contradicts the premise of this suite's \
             state policy and means the mutant is reaching something other than the carried \
             history; re-derive it before trusting the sequence gate. {CONV_ORACLE_STATE_POLICY}"
        );
    }

    let live = c.data(false);
    let good_live = run_conv_sequence(ctx, &src, &c, &live).expect("shipped, carried state");
    let bad_live = run_conv_sequence(ctx, &bad, &c, &live).expect("mutant, carried state");
    let moved = good_live.0[0]
        .iter()
        .zip(bad_live.0[0].iter())
        .filter(|(g, b)| g.to_bits() != b.to_bits())
        .count();
    assert!(
        moved * 2 > c.conv_dim,
        "with a NONZERO carried state and one step, {name} moved only {moved} of {} channels. The \
         zero-state arm above proved this defect is invisible without a carried state; if the \
         carried-state arm cannot see it either, this suite has no gate on how the history is \
         read at all. {CONV_ORACLE_STATE_POLICY}",
        c.conv_dim
    );
    eprintln!(
        "[q3d-delta-decode-oracle] {name}: 0/{} channels move from a zero state in one step, \
         {moved}/{} from the carried state",
        c.conv_dim, c.conv_dim
    );
}

struct GatingCase {
    label: &'static str,
    n_v: usize,
}

fn gating_cases() -> Vec<GatingCase> {
    vec![
        GatingCase {
            label: "nv96 (two workgroups, second one half idle)",
            n_v: 96,
        },
        GatingCase {
            label: "nv7 (odd, so a and b sit in opposite word halves)",
            n_v: 7,
        },
        GatingCase {
            label: "nv8",
            n_v: 8,
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

fn gating_data(c: &GatingCase) -> GatingData {
    let mut r = Lcg::new(0x9a71_0000 ^ c.n_v as u64);
    let ab = r.bf16_vec(2 * c.n_v, 2.5);
    let alog = r.f32_vec(c.n_v, 0.8);
    let dt = r.f32_vec(c.n_v, 1.2);
    let mut g = vec![0f64; c.n_v];
    let mut beta = vec![0f64; c.n_v];
    for i in 0..c.n_v {
        let a = ab[i];
        let b = ab[c.n_v + i];
        beta[i] = 1.0 / (1.0 + (-b).exp());
        let t = a + dt[i];
        let sp = t.max(0.0) + (1.0 + (-t.abs()).exp()).ln();
        g[i] = (sp * -(alog[i].exp())).exp();
    }
    GatingData {
        ab,
        alog,
        dt,
        g,
        beta,
    }
}

fn run_gating(
    ctx: &WgpuContext,
    src: &str,
    c: &GatingCase,
    d: &GatingData,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ab_b = dispatch::storage_from_slice(ctx, "dg-ab", &pack_bf16(&d.ab));
    let alog_b = dispatch::storage_from_slice(ctx, "dg-alog", &f32v(&d.alog));
    let dt_b = dispatch::storage_from_slice(ctx, "dg-dt", &f32v(&d.dt));
    let g_b = dispatch::storage_zeroed(ctx, "dg-g", (c.n_v * 4) as u64);
    let beta_b = dispatch::storage_zeroed(ctx, "dg-beta", (c.n_v * 4) as u64);
    let p_b = dispatch::uniform_from(
        ctx,
        "dg-p",
        &DgParams {
            n_v: c.n_v as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    let grid = dispatch::workgroup_count_1d(ctx, c.n_v as u64, 64);
    dispatch::run(
        ctx,
        "q3d-delta-decode-oracle",
        src,
        GATING_ENTRY,
        &[
            (20, &ab_b),
            (21, &alog_b),
            (22, &dt_b),
            (23, &g_b),
            (24, &beta_b),
            (25, &p_b),
        ],
        grid,
    )
    .map_err(|e| format!("{e}"))?;
    let g: Vec<f32> = dispatch::read_back(ctx, &g_b, c.n_v).map_err(|e| format!("g: {e}"))?;
    let beta: Vec<f32> =
        dispatch::read_back(ctx, &beta_b, c.n_v).map_err(|e| format!("beta: {e}"))?;
    Ok((to64(&g), to64(&beta)))
}

#[test]
fn q3w_delta_gating_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = decode_source();
    let mut worst = 0f64;
    for c in gating_cases() {
        let d = gating_data(&c);
        let (g, beta) =
            run_gating(ctx, &src, &c, &d).unwrap_or_else(|e| panic!("{}: {GATING_ENTRY}: {e}", c.label));
        let rg = check_rel(c.label, GATING_ENTRY, "g", &g, &d.g);
        let rb = check_rel(c.label, GATING_ENTRY, "beta", &beta, &d.beta);
        eprintln!(
            "[q3d-delta-decode-oracle] {} gating: g rel={rg:.3e} beta rel={rb:.3e}",
            c.label
        );
        worst = worst.max(rg).max(rb);
    }
    eprintln!("[q3d-delta-decode-oracle] worst gating relative error: {worst:.3e}");
}

#[test]
fn the_gating_corpus_separates_the_two_halves_and_decays() {
    for c in gating_cases() {
        let d = gating_data(&c);
        let mut swap_moves = 0usize;
        for i in 0..c.n_v {
            let sa = 1.0 / (1.0 + (-d.ab[i]).exp());
            let sb = 1.0 / (1.0 + (-d.ab[c.n_v + i]).exp());
            swap_moves += usize::from((sa - sb).abs() > 0.01);
        }
        let dt_spread = d.dt.iter().fold(0f64, |a, b| a.max((b - d.dt[0]).abs()));
        let alog_spread = d
            .alog
            .iter()
            .fold(0f64, |a, b| a.max((b - d.alog[0]).abs()));
        let g_lo = d.g.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let beta_gap = d.beta.iter().fold(0f64, |a, b| a.max((b - 0.5).abs()));
        eprintln!(
            "[q3d-delta-decode-oracle] {}: swapping a and b moves beta on {swap_moves}/{} heads, \
             dt spread {dt_spread:.4}, a_log spread {alog_spread:.4}, min g {g_lo:.4}, worst \
             |beta-0.5| {beta_gap:.4}",
            c.label, c.n_v
        );
        assert!(
            swap_moves * 10 >= c.n_v * 9,
            "{}: reading `a` where the kernel should read `b` would move beta by more than 0.01 on \
             only {swap_moves} of {} heads; the two halves of the packed ab row are too similar \
             for a half-swap to be observable",
            c.label,
            c.n_v
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

struct FusedCase {
    label: &'static str,
    n_v: usize,
    v_per_k: usize,
    d_k: usize,
    d_v: usize,
}

fn fused_cases() -> Vec<FusedCase> {
    vec![
        FusedCase {
            label: "served nv48 vpk3 dk128 dv128 (qwen3.8 geometry)",
            n_v: 48,
            v_per_k: 3,
            d_k: 128,
            d_v: 128,
        },
        FusedCase {
            label: "ragged nv6 vpk3 dk32 dv48",
            n_v: 6,
            v_per_k: 3,
            d_k: 32,
            d_v: 48,
        },
        FusedCase {
            label: "tiny nv4 vpk1 dk16 dv16",
            n_v: 4,
            v_per_k: 1,
            d_k: 16,
            d_v: 16,
        },
    ]
}

struct FusedOut {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn run_split_and_gate(
    ctx: &WgpuContext,
    src: &str,
    c: &FusedCase,
    mixed: &[f64],
    gd: &GatingData,
    fused: bool,
) -> Result<FusedOut, String> {
    let key_dim = c.n_v / c.v_per_k * c.d_k;
    let mixed_b = dispatch::storage_from_slice(ctx, "dq-mixed", &f32v(mixed));
    let q_b = dispatch::storage_zeroed(ctx, "dq-q", (c.n_v * c.d_k * 4) as u64);
    let k_b = dispatch::storage_zeroed(ctx, "dq-k", (c.n_v * c.d_k * 4) as u64);
    let v_b = dispatch::storage_zeroed(ctx, "dq-v", (c.n_v * c.d_v * 4) as u64);
    let dq_p = dispatch::uniform_from(
        ctx,
        "dq-p",
        &DqParams {
            n_v: c.n_v as u32,
            d_k: c.d_k as u32,
            d_v: c.d_v as u32,
            key_dim: key_dim as u32,
            v_per_k: c.v_per_k as u32,
            pad0: 0,
            pad1: 0,
            scale: 1.7,
        },
    );
    let ab_b = dispatch::storage_from_slice(ctx, "dg-ab", &pack_bf16(&gd.ab));
    let alog_b = dispatch::storage_from_slice(ctx, "dg-alog", &f32v(&gd.alog));
    let dt_b = dispatch::storage_from_slice(ctx, "dg-dt", &f32v(&gd.dt));
    let g_b = dispatch::storage_zeroed(ctx, "dg-g", (c.n_v * 4) as u64);
    let beta_b = dispatch::storage_zeroed(ctx, "dg-beta", (c.n_v * 4) as u64);
    let dg_p = dispatch::uniform_from(
        ctx,
        "dg-p",
        &DgParams {
            n_v: c.n_v as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    if fused {
        dispatch::run(
            ctx,
            "q3d-delta-decode-oracle",
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
        .map_err(|e| format!("{QKV_GATED_ENTRY}: {e}"))?;
    } else {
        dispatch::run(
            ctx,
            "q3d-delta-decode-oracle",
            src,
            QKV_ENTRY,
            &[
                (10, &mixed_b),
                (11, &q_b),
                (12, &k_b),
                (13, &v_b),
                (14, &dq_p),
            ],
            (c.n_v as u32, 1, 1),
        )
        .map_err(|e| format!("{QKV_ENTRY}: {e}"))?;
        dispatch::run(
            ctx,
            "q3d-delta-decode-oracle",
            src,
            GATING_ENTRY,
            &[
                (20, &ab_b),
                (21, &alog_b),
                (22, &dt_b),
                (23, &g_b),
                (24, &beta_b),
                (25, &dg_p),
            ],
            dispatch::workgroup_count_1d(ctx, c.n_v as u64, 64),
        )
        .map_err(|e| format!("{GATING_ENTRY}: {e}"))?;
    }
    Ok(FusedOut {
        q: dispatch::read_back(ctx, &q_b, c.n_v * c.d_k).map_err(|e| format!("q: {e}"))?,
        k: dispatch::read_back(ctx, &k_b, c.n_v * c.d_k).map_err(|e| format!("k: {e}"))?,
        v: dispatch::read_back(ctx, &v_b, c.n_v * c.d_v).map_err(|e| format!("v: {e}"))?,
        g: dispatch::read_back(ctx, &g_b, c.n_v).map_err(|e| format!("g: {e}"))?,
        beta: dispatch::read_back(ctx, &beta_b, c.n_v).map_err(|e| format!("beta: {e}"))?,
    })
}

#[test]
fn q3w_delta_qkv_gated_is_bit_identical_to_the_split_plus_gating_pair() {
    let ctx = ctx();
    let src = decode_source();
    for c in fused_cases() {
        assert_eq!(
            c.n_v % c.v_per_k,
            0,
            "{}: n_v must be a whole number of key groups",
            c.label
        );
        let key_dim = c.n_v / c.v_per_k * c.d_k;
        let mut r = Lcg::new(0xf5ed_0000 ^ (c.n_v as u64) << 8 ^ c.d_k as u64);
        let mixed = r.f32_vec(2 * key_dim + c.n_v * c.d_v, 1.2);
        let gd = gating_data(&GatingCase {
            label: c.label,
            n_v: c.n_v,
        });
        let pair = run_split_and_gate(ctx, &src, &c, &mixed, &gd, false)
            .unwrap_or_else(|e| panic!("{}: pair: {e}", c.label));
        let fused = run_split_and_gate(ctx, &src, &c, &mixed, &gd, true)
            .unwrap_or_else(|e| panic!("{}: fused: {e}", c.label));
        for (what, want, got) in [
            ("q", &pair.q, &fused.q),
            ("k", &pair.k, &fused.k),
            ("v", &pair.v, &fused.v),
            ("g", &pair.g, &fused.g),
            ("beta", &pair.beta, &fused.beta),
        ] {
            assert_nondegenerate(c.label, what, &to64(want), &to64(got));
            for (i, (w, gv)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    w.to_bits(),
                    gv.to_bits(),
                    "{}: {what}[{i}] differs between the fused entry and the pair it replaces \
                     ({w} vs {gv}). {FUSED_GATE_TRANSITIVITY}",
                    c.label
                );
            }
        }
        eprintln!(
            "[q3d-delta-decode-oracle] {}: fused {QKV_GATED_ENTRY} bit-identical to \
             {QKV_ENTRY}+{GATING_ENTRY} over {} q/k, {} v, {} g/beta words",
            c.label,
            c.n_v * c.d_k,
            c.n_v * c.d_v,
            c.n_v
        );
    }
}

struct OutCase {
    label: &'static str,
    n_v: usize,
    d_v: usize,
}

fn out_cases() -> Vec<OutCase> {
    vec![
        OutCase {
            label: "nv4 dv16 (shipped value head width)",
            n_v: 4,
            d_v: 16,
        },
        OutCase {
            label: "nv2 dv64",
            n_v: 2,
            d_v: 64,
        },
        OutCase {
            label: "nv3 dv128 (every lane carries a pair)",
            n_v: 3,
            d_v: 128,
        },
    ]
}

const OUT_EPS: f64 = 1e-6;

struct OutData {
    core: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
    want: Vec<f64>,
}

fn out_data(c: &OutCase) -> OutData {
    let mut r = Lcg::new(0x0d17_0000 ^ (c.d_v as u64) << 8 ^ c.n_v as u64);
    let vdim = c.n_v * c.d_v;
    let core = r.f32_vec(vdim, 1.3);
    let w = r.bf16_vec(c.d_v, 1.1);
    let z = r.bf16_vec(vdim, 2.0);
    let mut want = vec![0f64; vdim];
    for h in 0..c.n_v {
        let base = h * c.d_v;
        let mut ss = 0f64;
        for d in (0..c.d_v).rev() {
            let v = bf16(core[base + d]);
            ss += v * v;
        }
        let rms = 1.0 / (ss / c.d_v as f64 + OUT_EPS).sqrt();
        for d in 0..c.d_v {
            let v = bf16(core[base + d]);
            let zv = z[base + d];
            let g = bf16(silu(zv));
            let n = bf16(v * rms * w[d]);
            want[base + d] = bf16(n * g);
        }
    }
    OutData { core, w, z, want }
}

fn run_out(
    ctx: &WgpuContext,
    src: &str,
    c: &OutCase,
    d: &OutData,
) -> Result<Vec<f64>, String> {
    let vdim = c.n_v * c.d_v;
    let core_b = dispatch::storage_from_slice(ctx, "do-core", &f32v(&d.core));
    let w_b = dispatch::storage_from_slice(ctx, "do-w", &pack_bf16(&d.w));
    let z_b = dispatch::storage_from_slice(ctx, "do-z", &pack_bf16(&d.z));
    let out_b = dispatch::storage_zeroed(ctx, "do-out", (vdim * 2) as u64);
    let p_b = dispatch::uniform_from(
        ctx,
        "do-p",
        &DoParams {
            n_v: c.n_v as u32,
            d_v: c.d_v as u32,
            pad0: 0,
            eps: OUT_EPS as f32,
        },
    );
    dispatch::run(
        ctx,
        "q3d-delta-decode-oracle",
        src,
        OUT_ENTRY,
        &[
            (40, &core_b),
            (41, &w_b),
            (42, &z_b),
            (43, &out_b),
            (44, &p_b),
        ],
        (c.n_v as u32, 1, 1),
    )
    .map_err(|e| format!("{e}"))?;
    let words: Vec<u32> =
        dispatch::read_back(ctx, &out_b, vdim / 2).map_err(|e| format!("readback: {e}"))?;
    Ok(unpack_bf16(&words))
}

#[test]
fn q3w_delta_out_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = decode_source();
    let mut worst = 0f64;
    for c in out_cases() {
        let d = out_data(&c);
        let got =
            run_out(ctx, &src, &c, &d).unwrap_or_else(|e| panic!("{}: {OUT_ENTRY}: {e}", c.label));
        let (frac, exact) = check_bf16_ulp(c.label, OUT_ENTRY, "gated norm", &got, &d.want);
        eprintln!(
            "[q3d-delta-decode-oracle] {} out: {exact}/{} lanes bit-exact, worst {frac:.3} of one \
             bf16 ulp",
            c.label,
            d.want.len()
        );
        worst = worst.max(frac);
    }
    eprintln!("[q3d-delta-decode-oracle] worst out deviation: {worst:.3} bf16 ulp");
}

#[test]
fn the_out_corpus_normalizes_gates_and_separates_the_heads() {
    for c in out_cases() {
        assert!(
            c.n_v > 1,
            "{}: with one head the gate vector index `h * dv + e0` collapses to `e0` and reading \
             the wrong head's z would be invisible",
            c.label
        );
        let d = out_data(&c);
        let mut worst_rms_gap = 0f64;
        let mut worst_gate_gap = 0f64;
        for h in 0..c.n_v {
            let base = h * c.d_v;
            let mut ss = 0f64;
            for dd in 0..c.d_v {
                let v = bf16(d.core[base + dd]);
                ss += v * v;
            }
            let rms = 1.0 / (ss / c.d_v as f64 + OUT_EPS).sqrt();
            worst_rms_gap = worst_rms_gap.max((rms - 1.0).abs());
            for dd in 0..c.d_v {
                worst_gate_gap = worst_gate_gap.max((silu(d.z[base + dd]) - 1.0).abs());
            }
        }
        let w_spread = d.w.iter().fold(0f64, |a, b| a.max((b - d.w[0]).abs()));
        eprintln!(
            "[q3d-delta-decode-oracle] {}: worst |rms-1| {worst_rms_gap:.4}, worst |swish(z)-1| \
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
            "{}: the norm weight vector is constant, so a per-lane weight index defect would be \
             invisible",
            c.label
        );
    }
}

const MUTANTS: [(&str, &str, &str, &str); 14] = [
    (
        CONV_ENTRY,
        "conv-history-not-rolled",
        "cv_state[c * hist + j] = cv_state[c * hist + j + 1u];",
        "cv_state[c * hist + j] = cv_state[c * hist + j];",
    ),
    (
        CONV_ENTRY,
        "conv-newest-input-not-appended-to-the-history",
        "cv_state[c * hist + hist - 1u] = xv;",
        "cv_state[c * hist + hist - 1u] = 0.0;",
    ),
    (
        CONV_ENTRY,
        CONV_HISTORY_REVERSED.0,
        CONV_HISTORY_REVERSED.1,
        CONV_HISTORY_REVERSED.2,
    ),
    (
        CONV_ENTRY,
        "conv-current-input-uses-the-oldest-tap-weight",
        "acc = fma(cv_w[c * ks + hist], xv, acc);",
        "acc = fma(cv_w[c * ks], xv, acc);",
    ),
    (
        CONV_ENTRY,
        "conv-silu-dropped",
        "let silu = acc / (1.0 + exp(-acc));",
        "let silu = acc;",
    ),
    (
        GATING_ENTRY,
        "gating-beta-reads-the-decay-half-of-the-row",
        "let b = bf16_decode(u16_at(dg_ab[j >> 1u], j));",
        "let b = bf16_decode(u16_at(dg_ab[i >> 1u], i));",
    ),
    (
        GATING_ENTRY,
        "gating-dt-bias-dropped",
        "let t = a + dg_dt[i];",
        "let t = a;",
    ),
    (
        GATING_ENTRY,
        "gating-decay-sign-flipped",
        "return vec2<f32>(exp(sp * (-exp(dg_alog[i]))), beta);",
        "return vec2<f32>(exp(sp * exp(dg_alog[i])), beta);",
    ),
    (
        GATING_ENTRY,
        "gating-softplus-replaced-by-relu",
        "let sp = max(t, 0.0) + log(1.0 + exp(-abs(t)));",
        "let sp = max(t, 0.0);",
    ),
    (
        GATING_ENTRY,
        "gating-beta-sigmoid-dropped",
        "let beta = 1.0 / (1.0 + exp(-b));",
        "let beta = b;",
    ),
    (
        OUT_ENTRY,
        "out-gate-reads-the-first-head-for-every-head",
        "let z0 = bf16_decode(u16_at(do_z[zi >> 1u], zi));",
        "let z0 = bf16_decode(u16_at(do_z[e0 >> 1u], e0));",
    ),
    (
        OUT_ENTRY,
        "out-rms-divisor-dropped",
        "let rms = inverseSqrt(red0 / f32(dv) + do_p.eps);",
        "let rms = inverseSqrt(red0 + do_p.eps);",
    ),
    (
        OUT_ENTRY,
        "out-swish-gate-dropped",
        "do_out[zi >> 1u] = bf16_pack(n0 * g0, n1 * g1);",
        "do_out[zi >> 1u] = bf16_pack(n0, n1);",
    ),
    (
        OUT_ENTRY,
        "out-reduction-drops-the-odd-half-of-each-pair",
        "do_red[lane] = v0 * v0 + v1 * v1;",
        "do_red[lane] = v0 * v0;",
    ),
];

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped delta source: {from:?}. This gate is worthless \
         if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn conv_disagrees(ctx: &WgpuContext, src: &str) -> Vec<&'static str> {
    let mut caught = Vec::new();
    for c in conv_cases() {
        let d = c.data(false);
        let (want_out, want_state) = c.reference(&d);
        let hit = match run_conv_sequence(ctx, src, &c, &d) {
            Err(_) => true,
            Ok((got_out, got_state)) => (0..c.steps).any(|t| {
                bf16_ulp_error(&got_out[t], &want_out[t]).0 > 1.0
                    || got_state[t]
                        .iter()
                        .zip(want_state[t].iter())
                        .any(|(g, w)| g.to_bits() != w.to_bits())
            }),
        };
        if hit {
            caught.push(c.label);
        }
    }
    caught
}

fn gating_disagrees(ctx: &WgpuContext, src: &str) -> Vec<&'static str> {
    let mut caught = Vec::new();
    for c in gating_cases() {
        let d = gating_data(&c);
        let hit = match run_gating(ctx, src, &c, &d) {
            Err(_) => true,
            Ok((g, beta)) => {
                rel_error(&g, &d.g).0 >= TOL_F32 || rel_error(&beta, &d.beta).0 >= TOL_F32
            }
        };
        if hit {
            caught.push(c.label);
        }
    }
    caught
}

fn out_disagrees(ctx: &WgpuContext, src: &str) -> Vec<&'static str> {
    let mut caught = Vec::new();
    for c in out_cases() {
        let d = out_data(&c);
        let hit = match run_out(ctx, src, &c, &d) {
            Err(_) => true,
            Ok(got) => bf16_ulp_error(&got, &d.want).0 > 1.0,
        };
        if hit {
            caught.push(c.label);
        }
    }
    caught
}

#[test]
fn every_delta_decode_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = decode_source();
    for (entry, name, from, to) in MUTANTS {
        let bad = mutate(&src, from, to);
        let caught = match entry {
            CONV_ENTRY => conv_disagrees(ctx, &bad),
            GATING_ENTRY => gating_disagrees(ctx, &bad),
            _ => out_disagrees(ctx, &bad),
        };
        assert!(
            !caught.is_empty(),
            "mutant {name} of {entry} was NOT caught by any case in this corpus. A kernel that \
             survives this mutation is not gated by this suite; either the corpus lost the case \
             that saw the defect or the mutation is inert and must be replaced by one that is not. \
             {WHY}"
        );
        eprintln!("MUTANT {name} ({entry}): caught by {caught:?}");
    }
}

fn body_of(src: &str, name: &str) -> String {
    let key = format!("fn {name}(");
    let at = src.find(&key).unwrap_or_else(|| {
        panic!("the shipped source no longer declares {name}; this gate cannot locate its body")
    });
    let rest = &src[at..];
    let end = rest.find("\n}\n").unwrap_or_else(|| {
        panic!(
            "no closing brace at column zero after `{key}`; the WGSL layout changed and this \
             extractor is silently returning the rest of the file"
        )
    });
    rest[..end + 3].to_string()
}

#[test]
fn the_decode_bodies_gated_here_are_no_part_of_the_prefill_source() {
    let decode = decode_source();
    let prefill = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    let gated = [CONV_ENTRY, GATING_ENTRY, OUT_ENTRY];
    let bodies: Vec<String> = gated
        .iter()
        .map(|e| body_of(&decode, e))
        .chain([
            body_of(&decode, GATE_LANE_FN),
            body_of(&decode, GATE_VALUES_FN),
            body_of(&decode, OUT_TAIL_FN),
        ])
        .collect();
    assert!(
        bodies[1].contains(&format!("{GATE_LANE_FN}(i)")),
        "{GATING_ENTRY} no longer calls {GATE_LANE_FN}; the gating oracle then no longer \
         exercises the lane function the fused entry shares, and the fused-identity gate loses \
         its numeric anchor. {FUSED_GATE_TRANSITIVITY}"
    );
    assert!(
        bodies[3].contains(&format!("{GATE_VALUES_FN}(i)")),
        "{GATE_LANE_FN} no longer calls {GATE_VALUES_FN}; q3w_delta_head_fused reads its gate \
         through {GATE_VALUES_FN}, so the gating oracle would stop exercising the arithmetic \
         the fused head entry runs. {FUSED_GATE_TRANSITIVITY}"
    );
    assert!(
        bodies[2].contains(&format!("{OUT_TAIL_FN}(h, e0, v0, v1, do_red[0])")),
        "{OUT_ENTRY} no longer calls {OUT_TAIL_FN}; q3w_delta_head_fused writes its gated \
         output through {OUT_TAIL_FN}, so the out oracle would stop exercising the arithmetic \
         the fused head entry runs. {FUSED_GATE_TRANSITIVITY}"
    );
    for (e, body) in gated.iter().zip(bodies.iter()) {
        let gated_text_len = if *e == GATING_ENTRY {
            body.len() + bodies[3].len() + bodies[4].len()
        } else if *e == OUT_ENTRY {
            body.len() + bodies[5].len()
        } else {
            body.len()
        };
        assert!(
            gated_text_len > 200,
            "the extracted body of {e} is only {gated_text_len} bytes; the extractor is not \
             finding the whole function and every containment check below is vacuous"
        );
        assert!(
            !prefill.contains(&format!("fn {e}(")),
            "the prefill source now declares {e} as well; if the decode entry and its M-row twin \
             have become one function body, graph_q3d_delta_front_oracle already gates it and this \
             suite is redundant"
        );
        assert!(
            !prefill.contains(body.as_str()),
            "the whole body of {e} now occurs verbatim in the prefill source, so \
             graph_q3d_delta_front_oracle's corpus does compile this text and the two gates \
             overlap. {INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}"
        );
    }
    for (name, body) in [
        (GATE_LANE_FN, &bodies[3]),
        (GATE_VALUES_FN, &bodies[4]),
        (OUT_TAIL_FN, &bodies[5]),
    ] {
        assert!(
            !prefill.contains(body.as_str()),
            "the whole body of {name} now occurs verbatim in the prefill source. \
             {INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}"
        );
    }
    for (entry, name, from, to) in MUTANTS {
        assert!(
            decode.contains(from),
            "anchor for mutant {name} is gone from the shipped delta source: {from:?}. A mutant \
             whose anchor rotted is silently inert, and the GPU tests that would have caught that \
             do not run on a box with no adapter -- which is why this check is CPU-only."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
        let total = decode.matches(from).count();
        let inside: usize = bodies.iter().map(|b| b.matches(from).count()).sum();
        assert_eq!(
            total, inside,
            "mutant {name} of {entry}: its anchor occurs {total} time(s) in the shipped delta \
             source but only {inside} of those are inside the three bodies this suite gates. \
             {INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}"
        );
    }
    eprintln!("{WHY}\n{INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}\n{CONV_ORACLE_STATE_POLICY}\n{BOUNDS_DOC}");
}
