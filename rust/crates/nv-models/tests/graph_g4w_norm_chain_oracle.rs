#![cfg(feature = "wgpu")]

mod common;
use common::ord;
use common::pack;
use common::unpack;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitG4w as Lcg;

const TAG: &str = "g4w:norm_chain";
const RES_ENTRY: &str = "g4w_norm_res_norm";
const ADD_ENTRY: &str = "g4w_norm_add_norm";

const WHY: &str = "g4w_norm_res_norm and g4w_norm_add_norm are the fused norm chains the Gemma-4 \
     dense graph runs twice per layer under NV_WGPU_FUSE, replacing the rmsnorm / residual-add / \
     rmsnorm triple with one workgroup that keeps the intermediate in shared state. The entry \
     census records both as named by no test in the workspace: the fuse path is selected by \
     default, so the end-to-end fixtures do drive them, but no artifact in the tree checks a \
     number either kernel produces. Fusion is exactly where that matters -- the unfused triple \
     they replace is gated elsewhere, and the whole point of the fused form is that it does the \
     same arithmetic in a different order with a different set of intermediate rounds.";

const ORACLE_IS_NOT_THE_KERNEL: &str = "The reference is an f64 host chain written from the \
     definition: x * inverseSqrt(eps + mean(x^2)) * w for each norm, an addition for the \
     residual, and the layer scalar where the shipped chain applies it. It is not the unfused \
     rmsnorm kernels and not a second copy of the WGSL. It does model the STORES, because they \
     are part of what the fused chain is: every intermediate this kernel parks in a buffer is \
     bf16, so the reference rounds mid, res and out to bf16 exactly where the shader packs them, \
     and it accumulates the residual mean from the UNROUNDED sums because that is what \
     g4w_norm_res_norm does -- a reference that rounded first would be a different kernel.";

const EPSILON_HAS_TO_BE_REACHABLE: &str = "One case drives the chain with hidden states small \
     enough that mean(x^2) sits far below eps. Without it the epsilon term is a rounding-error \
     perturbation of a mean near 1, no bound on a bf16 output can see it, and a chain that \
     dropped it entirely would pass every case -- the same shape of hole as an nvfp4 fixture \
     whose scale bytes never reach a biased exponent of zero. The screen asserts the regime is \
     actually reached rather than assuming the draw got there.";

const NORM_EPS: f32 = 1e-6;

const TINY_MEAN_MAX: f64 = 1e-8;

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_g4w_norm_chain_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
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
    for e in [RES_ENTRY, ADD_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "{TAG} no longer declares {e}; the entry moved and this gate is now testing nothing"
        );
    }
    src
}

fn dec(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

fn enc(v: f64) -> u16 {
    half::bf16::from_f32(v as f32).to_bits()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NcParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
}

struct Case {
    label: &'static str,
    hidden: usize,
    scale: f32,
    x: Vec<u16>,
    w1: Vec<u16>,
    w2: Vec<u16>,
    res: Vec<u16>,
    first_mean: f64,
}

impl Case {
    fn build(label: &'static str, hidden: usize, scale: f32, mag: f32, seed: u64) -> Self {
        assert!(
            hidden.is_multiple_of(2),
            "{label}: the chain packs two elements per word"
        );
        let mut rng = Lcg::new(seed);
        let x = rng.bf16_vec(hidden, mag);
        let w1 = rng.norm_vec(hidden);
        let w2 = rng.norm_vec(hidden);
        let res = rng.bf16_vec(hidden, mag);
        let first_mean =
            x.iter().map(|b| dec(*b) * dec(*b)).sum::<f64>() / hidden as f64;
        Self {
            label,
            hidden,
            scale,
            x,
            w1,
            w2,
            res,
            first_mean,
        }
    }

    fn rms(&self, vals: &[f64]) -> f64 {
        let sum: f64 = vals.iter().map(|v| v * v).sum();
        1.0 / (NORM_EPS as f64 + sum / self.hidden as f64).sqrt()
    }

    fn norm_into_mid(&self) -> Vec<u16> {
        let xf: Vec<f64> = self.x.iter().map(|b| dec(*b)).collect();
        let r = self.rms(&xf);
        (0..self.hidden)
            .map(|i| enc(xf[i] * r * dec(self.w1[i])))
            .collect()
    }

    fn res_norm_reference(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mid = self.norm_into_mid();
        let raw: Vec<f64> = (0..self.hidden)
            .map(|i| dec(mid[i]) + dec(self.res[i]))
            .collect();
        let res_out: Vec<u16> = raw.iter().map(|v| enc(*v)).collect();
        let r2 = self.rms(&raw);
        let x_out = (0..self.hidden)
            .map(|i| enc(dec(res_out[i]) * r2 * dec(self.w2[i])))
            .collect();
        (x_out, mid, res_out)
    }

    fn add_norm_reference(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mid = self.norm_into_mid();
        let out: Vec<u16> = (0..self.hidden)
            .map(|i| enc((dec(self.res[i]) + dec(mid[i])) * self.scale as f64))
            .collect();
        let outf: Vec<f64> = out.iter().map(|b| dec(*b)).collect();
        let r2 = self.rms(&outf);
        let x_out = (0..self.hidden)
            .map(|i| enc(outf[i] * r2 * dec(self.w2[i])))
            .collect();
        (x_out, mid, out)
    }
}

struct Chain {
    x: Vec<u16>,
    mid: Vec<u16>,
    third: Vec<u16>,
}

fn run(ctx: &WgpuContext, src: &str, entry: &str, c: &Case) -> anyhow::Result<Chain> {
    let words = c.hidden / 2;
    let poison = vec![0xdead_beefu32; words];
    let x = dispatch::storage_from_slice(ctx, "nc-x", &pack(&c.x));
    let w1 = dispatch::storage_from_slice(ctx, "nc-w1", &pack(&c.w1));
    let mid = dispatch::storage_from_slice(ctx, "nc-mid", &poison);
    let res = dispatch::storage_from_slice(ctx, "nc-res", &pack(&c.res));
    let w2 = dispatch::storage_from_slice(ctx, "nc-w2", &pack(&c.w2));
    let out = dispatch::storage_from_slice(ctx, "nc-out", &poison);
    let p = dispatch::uniform_from(
        ctx,
        "nc-p",
        &NcParams {
            hidden: c.hidden as u32,
            words: words as u32,
            eps: NORM_EPS,
            scale: c.scale,
        },
    );
    let pipe = dispatch::compute_pipeline(ctx, "g4w-norm-chain-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
        (20, &x),
        (21, &w1),
        (22, &mid),
        (23, &res),
        (24, &w2),
        (26, &p),
    ];
    if entry == ADD_ENTRY {
        binds.push((25, &out));
    }
    dispatch::dispatch(ctx, &pipe, &binds, (1, 1, 1)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let rb = |b: &wgpu::Buffer| -> anyhow::Result<Vec<u16>> {
        let w: Vec<u32> = dispatch::read_back(ctx, b, words).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(unpack(&w, c.hidden))
    };
    let third = if entry == RES_ENTRY { &res } else { &out };
    Ok(Chain {
        x: rb(&x)?,
        mid: rb(&mid)?,
        third: rb(third)?,
    })
}

fn worst_ulp(got: &[u16], want: &[u16]) -> (i64, usize) {
    let mut worst = 0i64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (ord(*g) - ord(*w)).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

const ULP_BOUND: i64 = 1;

const BOUNDS_DOC: &str = "ULP_BOUND=1: every buffer this chain writes is bf16, so the unit of a \
     defect is a bf16 ulp and the tolerance is one of them, per element, against the f64 chain. \
     It is not a norm over the vector -- a norm on a bf16 output saturates at half an ulp and \
     could not distinguish a correct chain from one with a wrong reduction. The one ulp that is \
     allowed is the f32 reduction order inside inverseSqrt's argument, which no host order \
     reproduces exactly; the arithmetic that follows it is elementwise and exact.";

fn check(who: &str, got: &[u16], want: &[u16]) {
    let (d, at) = worst_ulp(got, want);
    assert!(
        d <= ULP_BOUND,
        "{who} differs from the f64 host chain by {d} bf16 ulp at element {at} (got {:e}, want \
         {:e}). {ORACLE_IS_NOT_THE_KERNEL} {BOUNDS_DOC}",
        dec(got[at]),
        dec(want[at])
    );
}

fn chain_ulp(got: &Chain, want: &(Vec<u16>, Vec<u16>, Vec<u16>)) -> i64 {
    worst_ulp(&got.x, &want.0)
        .0
        .max(worst_ulp(&got.mid, &want.1).0)
        .max(worst_ulp(&got.third, &want.2).0)
}

fn caught(got: &Chain, want: &(Vec<u16>, Vec<u16>, Vec<u16>)) -> bool {
    chain_ulp(got, want) > ULP_BOUND
}

const MUTANTS: [(&str, &str, &str, bool, bool); 6] = [
    (
        "rms-mean-divides-by-the-word-count-instead-of-the-element-count",
        "        let mean = nc_div_rn(sum, f32(nc_params.hidden));",
        "        let mean = nc_div_rn(sum, f32(nc_params.words));",
        true,
        true,
    ),
    (
        "second-norm-applies-the-first-norm-weight",
        "        let ww = nc_w2[i];",
        "        let ww = nc_w1[i];",
        true,
        true,
    ),
    (
        "res-norm-residual-add-dropped-on-the-low-half",
        "        let lo = bf16_lo(xw) + bf16_lo(rw);",
        "        let lo = bf16_lo(xw);",
        true,
        false,
    ),
    (
        "res-norm-mean-accumulates-only-the-low-half",
        "        local = local + lo * lo + hi * hi;",
        "        local = local + lo * lo;",
        true,
        false,
    ),
    (
        "add-norm-layer-scalar-dropped",
        "        let lo = (bf16_lo(aw) + bf16_lo(bw)) * scale;",
        "        let lo = (bf16_lo(aw) + bf16_lo(bw));",
        false,
        true,
    ),
    (
        "first-norm-rms-drops-the-epsilon",
        "        nc_v1 = inverseSqrt(nc_params.eps + mean);",
        "        nc_v1 = inverseSqrt(mean);",
        true,
        true,
    ),
];

const EPSILON_MUTANT: usize = 5;

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped norm chain: {from:?}. This gate is worthless \
         if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn corpus() -> Vec<Case> {
    let cases = vec![
        Case::build("gemma4-31b hidden", 5376, 0.9, 0.3, 0x0000_5376),
        Case::build("one workgroup pass per element", 512, 1.0, 0.6, 0x0000_0512),
        Case::build("odd tail past the last full stride", 258, 0.75, 0.4, 0x0000_0258),
        Case::build("vanishing state, epsilon dominates the mean", 512, 0.9, 1e-5, 0x0000_e959),
    ];
    let tiny = cases
        .iter()
        .filter(|c| c.first_mean < TINY_MEAN_MAX)
        .count();
    assert!(
        tiny > 0,
        "no case in the corpus drives mean(x^2) below {TINY_MEAN_MAX:e}, so eps={NORM_EPS:e} is \
         never the term that decides the scale and the epsilon mutant below cannot be caught by \
         anything. {EPSILON_HAS_TO_BE_REACHABLE}"
    );
    cases
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_norm_chain() {
    let src = source();
    for (name, from, to, _, _) in MUTANTS {
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
    let _ = corpus();
    eprintln!("{WHY}\n{ORACLE_IS_NOT_THE_KERNEL}\n{EPSILON_HAS_TO_BE_REACHABLE}");
}

#[test]
fn norm_res_norm_matches_an_f64_host_chain() {
    let ctx = ctx();
    eprintln!("[norm-chain-oracle] adapter: {}\n{WHY}", ctx.info.name);
    let src = source();
    for c in corpus() {
        let got = run(ctx, &src, RES_ENTRY, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        let want = c.res_norm_reference();
        eprintln!(
            "[{}] {RES_ENTRY} hidden={} mean(x^2)={:e}: x {:?} mid {:?} res {:?} ulp",
            c.label,
            c.hidden,
            c.first_mean,
            worst_ulp(&got.x, &want.0).0,
            worst_ulp(&got.mid, &want.1).0,
            worst_ulp(&got.third, &want.2).0
        );
        check(&format!("[{}] {RES_ENTRY} normed output", c.label), &got.x, &want.0);
        check(&format!("[{}] {RES_ENTRY} first-norm intermediate", c.label), &got.mid, &want.1);
        check(&format!("[{}] {RES_ENTRY} updated residual", c.label), &got.third, &want.2);
    }
}

#[test]
fn norm_add_norm_matches_an_f64_host_chain() {
    let ctx = ctx();
    let src = source();
    for c in corpus() {
        let got = run(ctx, &src, ADD_ENTRY, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        let want = c.add_norm_reference();
        eprintln!(
            "[{}] {ADD_ENTRY} hidden={} scale={}: x {:?} mid {:?} out {:?} ulp",
            c.label,
            c.hidden,
            c.scale,
            worst_ulp(&got.x, &want.0).0,
            worst_ulp(&got.mid, &want.1).0,
            worst_ulp(&got.third, &want.2).0
        );
        check(&format!("[{}] {ADD_ENTRY} normed output", c.label), &got.x, &want.0);
        check(&format!("[{}] {ADD_ENTRY} first-norm intermediate", c.label), &got.mid, &want.1);
        check(&format!("[{}] {ADD_ENTRY} scaled sum", c.label), &got.third, &want.2);
    }
}

#[test]
fn every_norm_chain_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = source();
    for (name, from, to, on_res, on_add) in MUTANTS {
        let bad = mutate(&src, from, to);
        let mut caught_by: Vec<String> = Vec::new();
        for c in corpus() {
            if on_res {
                let hit = match run(ctx, &bad, RES_ENTRY, &c) {
                    Ok(g) => caught(&g, &c.res_norm_reference()),
                    Err(_) => true,
                };
                if hit {
                    caught_by.push(format!("{}/{RES_ENTRY}", c.label));
                }
            }
            if on_add {
                let hit = match run(ctx, &bad, ADD_ENTRY, &c) {
                    Ok(g) => caught(&g, &c.add_norm_reference()),
                    Err(_) => true,
                };
                if hit {
                    caught_by.push(format!("{}/{ADD_ENTRY}", c.label));
                }
            }
        }
        assert!(
            !caught_by.is_empty(),
            "mutant {name} was NOT caught by any case in the corpus. A fused norm chain a broken \
             reduction passes is not a gate. {WHY}"
        );
        eprintln!("MUTANT {name}: caught by {caught_by:?}");
    }
}

const EPSILON_DECISIVENESS: i64 = 100;

#[test]
fn the_epsilon_term_is_decisive_only_where_the_mean_falls_below_it() {
    let ctx = ctx();
    let src = source();
    let (name, from, to, _, _) = MUTANTS[EPSILON_MUTANT];
    assert_eq!(name, "first-norm-rms-drops-the-epsilon");
    let bad = mutate(&src, from, to);
    let mut tiny = 0i64;
    let mut ordinary = 0i64;
    for c in corpus() {
        let g = run(ctx, &bad, RES_ENTRY, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        let d = chain_ulp(&g, &c.res_norm_reference());
        eprintln!(
            "[{}] mean(x^2)={:e}, dropping eps moves the chain by {d} bf16 ulp",
            c.label, c.first_mean
        );
        if c.first_mean < TINY_MEAN_MAX {
            tiny = d;
        } else {
            ordinary = ordinary.max(d);
        }
    }
    assert!(
        tiny > EPSILON_DECISIVENESS * ordinary.max(1),
        "dropping eps={NORM_EPS:e} moved the vanishing-state case by {tiny} bf16 ulp and an \
         ordinary case by as much as {ordinary}; the corpus no longer separates the regime where \
         eps decides the scale from the one where it is a rounding perturbation, so the term \
         whose only purpose is the first regime is untested. {EPSILON_HAS_TO_BE_REACHABLE}"
    );
}
