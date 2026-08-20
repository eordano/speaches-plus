#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const ENTRY: &str = "gow_router_topk";
const MOE_TAG: &str = "gow:moe";

const WHY: &str = "gow_router_topk is one of the three GPT-OSS entries the runtime entry census \
     records as having no gate of their own: the only backstop is \
     tiny_wgpu_decode_matches_cpu_reference, a 2-layer end-to-end fixture asserting rel < 0.05 on \
     the logits, which routes 2 of 4 experts on toy weights whose outputs sit within a few \
     percent of each other. Swapping which two are chosen moves those logits by far less than the \
     bar they are measured against, and the softmax over the chosen values is invisible to that \
     fixture entirely because the weights renormalise. It is also the one entry in the graph whose \
     defect is DISCRETE -- every other shader is a dot product whose error scales with the \
     mutation, while a router that returns the wrong expert id is either right or catastrophically \
     wrong, and no relative tolerance on downstream logits is the instrument for that. This gate \
     needs no checkpoint: the router reads a logits buffer and writes ids and weights, so it is \
     driven directly at every (n_experts, k) the config space allows.";

const GAP: f64 = 0.125;
const WEIGHT_REL: f64 = 1e-5;
const OVERFLOW_BASE: f64 = 96.0;
const MAX_EXPERTS: usize = 256;
const MAX_K: usize = 16;

const BOUNDS_DOC: &str = "GAP=0.125: every logit is a multiple of it, so chosen[j] - m is EXACT \
     in f32 and the only f32-vs-f64 disagreement left in the softmax is exp itself. It is also the \
     minimum separation between any two logits in a case, which makes the top-k a total order no \
     rounding can permute -- only then may the reference scan the experts in the opposite \
     direction and still be required to agree. WEIGHT_REL=1e-5: f32 exp is accurate to a few ulp \
     (~1e-7 relative) and the normalising divide adds one more, and GAP removes every other \
     source; a routed-expert defect moves a weight by O(1), four orders clear of this, so the \
     bound is set by the arithmetic and not by the value it bounds. OVERFLOW_BASE=96.0: f32 exp \
     saturates at ~88.7, which is the only reason the max subtraction exists and therefore the \
     only way a corpus can gate it. MAX_EXPERTS=256 and MAX_K=16 are the shader's own \
     `taken: array<u32, 256>` and `chosen: array<f32, 16>`; a case past either would index out of \
     range rather than test anything.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_gow_router_topk_oracle needs a real wgpu adapter; a skipped numeric gate reads as \
         a passed one, so this panics rather than returning early",
    )
}

fn source() -> String {
    let all = nv_models::gpt_oss_wgpu::nozi_audit_sources();
    let hit = all
        .into_iter()
        .find(|(tag, _)| *tag == MOE_TAG)
        .unwrap_or_else(|| {
            panic!(
                "gpt_oss_wgpu::nozi_audit_sources() no longer exposes {MOE_TAG}; this gate \
                 compiles the SHIPPED text and cannot fall back to a copy"
            )
        });
    let src = hit.1;
    assert!(
        src.contains(&format!("fn {ENTRY}(")),
        "{MOE_TAG} no longer declares {ENTRY}; the entry moved and this gate is now testing \
         nothing"
    );
    src
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtParams {
    n_experts: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
}

struct Case {
    label: &'static str,
    logits: Vec<f64>,
    k: usize,
}

impl Case {
    fn build(label: &'static str, n: usize, k: usize, base: f64, stride: usize) -> Self {
        assert!(
            n <= MAX_EXPERTS,
            "{label}: {n} experts exceeds {MAX_EXPERTS}"
        );
        assert!(
            k <= MAX_K && k <= n,
            "{label}: k={k} out of range for n={n}"
        );
        assert!(
            gcd(stride, n) == 1,
            "{label}: stride {stride} is not coprime with {n}, so the deal repeats values and the \
             top-k stops being a total order"
        );
        let logits: Vec<f64> = (0..n)
            .map(|i| base + ((i * stride) % n) as f64 * GAP)
            .collect();
        let c = Self { label, logits, k };
        c.screen();
        c
    }

    fn screen(&self) {
        for v in &self.logits {
            assert_eq!(
                *v,
                (*v as f32) as f64,
                "{}: logit {v} is not exact in f32, so the f32 subtraction in the softmax would \
                 carry an error this gate would then have to widen WEIGHT_REL for. {BOUNDS_DOC}",
                self.label
            );
        }
        let mut sorted = self.logits.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in sorted.windows(2) {
            assert!(
                w[1] - w[0] >= GAP,
                "{}: logits {} and {} are closer than GAP={GAP}; the top-k is then not a total \
                 order and the reference could disagree with the shader for a legitimate reason. \
                 {BOUNDS_DOC}",
                self.label,
                w[0],
                w[1]
            );
        }
    }

    fn bits(&self) -> Vec<u32> {
        self.logits.iter().map(|v| (*v as f32).to_bits()).collect()
    }

    fn reference(&self) -> (Vec<u32>, Vec<f64>) {
        let n = self.logits.len();
        let mut taken = vec![false; n];
        let mut ids = Vec::with_capacity(self.k);
        let mut chosen = Vec::with_capacity(self.k);
        for _ in 0..self.k {
            let mut best = f64::NEG_INFINITY;
            let mut bi = usize::MAX;
            for i in (0..n).rev() {
                if taken[i] {
                    continue;
                }
                if bi == usize::MAX || self.logits[i] > best {
                    best = self.logits[i];
                    bi = i;
                }
            }
            taken[bi] = true;
            ids.push(bi as u32);
            chosen.push(best);
        }
        let m = chosen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = chosen.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        (ids, e.into_iter().map(|v| v / s).collect())
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn run(ctx: &WgpuContext, src: &str, c: &Case) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
    let logits = dispatch::storage_from_slice(ctx, "rt-logits", &c.bits());
    let ids = dispatch::storage_from_slice(ctx, "rt-ids", &vec![u32::MAX; c.k]);
    let w = dispatch::storage_from_slice(ctx, "rt-w", &vec![f32::NAN; c.k]);
    let p = dispatch::uniform_from(
        ctx,
        "rt-p",
        &RtParams {
            n_experts: c.logits.len() as u32,
            k: c.k as u32,
            pad0: 0,
            pad1: 0,
        },
    );
    let pipe = dispatch::compute_pipeline(ctx, "gow-router-topk-probe", src, ENTRY)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[(0, &logits), (1, &ids), (2, &w), (3, &p)],
        (1, 1, 1),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let got_ids: Vec<u32> =
        dispatch::read_back(ctx, &ids, c.k).map_err(|e| anyhow::anyhow!("{e}"))?;
    let got_w: Vec<f32> = dispatch::read_back(ctx, &w, c.k).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((got_ids, got_w))
}

fn diff(c: &Case, ids: &[u32], w: &[f32]) -> Option<String> {
    let (want_ids, want_w) = c.reference();
    if ids != want_ids.as_slice() {
        return Some(format!("ids {ids:?} != f64 top-k {want_ids:?}"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for id in ids {
        if !seen.insert(*id) {
            return Some(format!("expert {id} routed twice: {ids:?}"));
        }
    }
    for (j, (g, want)) in w.iter().zip(want_w.iter()).enumerate() {
        if !g.is_finite() {
            return Some(format!("weight {j} is non-finite ({g})"));
        }
        let rel = ((*g as f64) - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        if rel > WEIGHT_REL {
            return Some(format!(
                "weight {j} = {g} vs f64 softmax {want} (rel {rel:e} > {WEIGHT_REL:e})"
            ));
        }
    }
    None
}

fn corpus() -> Vec<Case> {
    vec![
        Case::build("shipped-20b 32 experts, top-4", 32, 4, -2.0, 7),
        Case::build("tiny fixture 4 experts, top-2", 4, 2, -0.5, 3),
        Case::build("k == n, every expert routed", 8, 8, -1.0, 3),
        Case::build("k == 1, argmax only", 16, 1, -1.0, 5),
        Case::build("widest the shader can index", MAX_EXPERTS, MAX_K, -8.0, 17),
        Case::build(
            "overflow without the max subtraction",
            12,
            4,
            OVERFLOW_BASE,
            5,
        ),
    ]
}

const MUTANTS: [(&str, &str, &str); 4] = [
    (
        "chosen-expert-not-marked-taken",
        "        taken[bi] = 1u;",
        "        taken[bi] = 0u;",
    ),
    (
        "selection-keeps-the-smallest-instead-of-the-largest",
        "            if (!found || v > best) {",
        "            if (!found || v < best) {",
    ),
    (
        "softmax-max-subtraction-dropped",
        "        let e = exp(chosen[j] - m);",
        "        let e = exp(chosen[j]);",
    ),
    (
        "weights-left-unnormalised",
        "        grt_w[j] = chosen[j] / s;",
        "        grt_w[j] = chosen[j];",
    ),
];

const MAX_SUB_MUTANT: usize = 2;

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped router source: {from:?}. This gate is \
         worthless if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_router() {
    let src = source();
    for (name, from, to) in MUTANTS {
        assert!(
            src.contains(from),
            "anchor for mutant {name} is gone from the shipped router source: {from:?}. A mutant \
             whose anchor rotted is silently inert, and the GPU tests that would have caught that \
             do not run on a box with no adapter -- which is why this check is CPU-only."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
    }
    let c = Case::build("host-side reference sanity", 8, 3, -1.0, 3);
    let (ids, w) = c.reference();
    assert_eq!(
        ids.len(),
        3,
        "the host reference must return exactly k ids, not {ids:?}"
    );
    assert!(
        (w.iter().sum::<f64>() - 1.0).abs() < 1e-12,
        "the host reference softmax does not sum to 1: {w:?}"
    );
    eprintln!("{WHY}\n{BOUNDS_DOC}");
}

#[test]
fn gow_router_topk_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[gow-router-oracle] adapter: {}\n{WHY}", ctx.info.name);
    let src = source();
    for c in corpus() {
        let (ids, w) = run(ctx, &src, &c).unwrap_or_else(|e| panic!("{}: dispatch: {e}", c.label));
        let sum: f32 = w.iter().sum();
        eprintln!(
            "[{}] n={} k={} ids={ids:?} weights sum to {sum}",
            c.label,
            c.logits.len(),
            c.k
        );
        if let Some(d) = diff(&c, &ids, &w) {
            panic!(
                "{}: {ENTRY} disagrees with the f64 host reference: {d}",
                c.label
            );
        }
        assert!(
            ((sum as f64) - 1.0).abs() <= WEIGHT_REL * c.k as f64,
            "{}: routed weights sum to {sum}, not 1; the MoE output is scaled by this sum",
            c.label
        );
    }
}

#[test]
fn every_router_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = source();
    let cases = corpus();
    for (name, from, to) in MUTANTS {
        let bad = mutate(&src, from, to);
        let mut caught_by: Vec<&str> = Vec::new();
        for c in &cases {
            let caught = match run(ctx, &bad, c) {
                Ok((ids, w)) => diff(c, &ids, &w),
                Err(e) => Some(format!("dispatch failed: {e}")),
            };
            if let Some(d) = caught {
                eprintln!("MUTANT {name} caught by [{}]: {d}", c.label);
                caught_by.push(c.label);
            }
        }
        assert!(
            !caught_by.is_empty(),
            "mutant {name} was NOT caught by any case in the corpus. A router that survives this \
             mutation is not gated by this suite; either the corpus lost the case that saw the \
             defect or the mutation is inert and must be replaced by one that is not. {WHY}"
        );
        eprintln!("MUTANT {name}: caught by {caught_by:?}");
    }
}

#[test]
fn only_the_overflow_case_can_see_the_dropped_max_subtraction() {
    let ctx = ctx();
    let src = source();
    let (name, from, to) = MUTANTS[MAX_SUB_MUTANT];
    assert_eq!(name, "softmax-max-subtraction-dropped");
    let bad = mutate(&src, from, to);
    let mut moderate_survivals = 0usize;
    let mut overflow_caught = false;
    for c in corpus() {
        let over = c.logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max) > 88.0;
        let caught = match run(ctx, &bad, &c) {
            Ok((ids, w)) => diff(&c, &ids, &w).is_some(),
            Err(_) => true,
        };
        if over {
            overflow_caught = caught;
        } else if !caught {
            moderate_survivals += 1;
        }
    }
    assert!(
        overflow_caught,
        "the overflow case did not catch the dropped max subtraction; f32 exp no longer saturates \
         at OVERFLOW_BASE={OVERFLOW_BASE}, so this corpus has stopped gating the one line whose \
         only purpose is overflow and OVERFLOW_BASE must be raised"
    );
    assert!(
        moderate_survivals > 0,
        "every moderate case caught the dropped max subtraction too, which contradicts the \
         algebra: exp(x)/sum(exp(x)) is unchanged by subtracting a constant. This suite is then \
         measuring something other than what it names and the overflow case is no longer the \
         reason the mutant is caught"
    );
}
