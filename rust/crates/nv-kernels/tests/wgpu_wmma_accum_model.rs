#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::device::WgpuContext;
mod common;
use common::ctx_or_panic as ctx;
use common::{wmma_accum_class, wmma_f16_dot_probe, WmmaAccumClass};

fn require_coop(ctx: &WgpuContext) {
    assert!(
        !ctx.caps.coop_configs.is_empty(),
        "adapter reports no cooperative-matrix configs, so this suite would characterize \
         nothing; the coop gemm route is closed here and its numerics are moot"
    );
}

fn ieee_dot(xs: &[f32], ys: &[f32], c: f32) -> f32 {
    let mut s = c as f64;
    for (x, y) in xs.iter().zip(ys.iter()) {
        s += *x as f64 * *y as f64;
    }
    s as f32
}

#[test]
fn the_adapter_wmma_class_is_one_this_fleet_has_characterized() {
    let ctx = ctx();
    require_coop(ctx);
    let class = wmma_accum_class(ctx);
    eprintln!("  wmma accumulator class: {class:?}");
}

const F32_REPRESENTABLE_EXACT_SUM: bool = true;
const ROUNDING_EDGE: bool = false;

#[test]
fn the_canonical_dot2_corpus_is_bit_pinned_per_accumulator_class() {
    let ctx = ctx();
    require_coop(ctx);
    let class = wmma_accum_class(ctx);
    let p2 = |e: i32| (2f64.powi(e)) as f32;
    let ones = |n: usize| vec![1f32; n];
    let corpus: Vec<(Vec<f32>, f32, u32, bool)> = vec![
        (vec![3.0, -1.0], 0.0, 0x3fffffff, F32_REPRESENTABLE_EXACT_SUM),
        (vec![3.0, -1.0, -1.0], 0.0, 0x3f7ffffd, F32_REPRESENTABLE_EXACT_SUM),
        (vec![3.0, -2.0], 0.0, 0x3f7ffffe, F32_REPRESENTABLE_EXACT_SUM),
        (vec![-3.0, -1.0], 0.0, 0xc0800000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![-3.0, 1.0], 0.0, 0xc0000000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![6.0, -2.0], 0.0, 0x407fffff, F32_REPRESENTABLE_EXACT_SUM),
        (vec![1.5, -0.5], 0.0, 0x3f7fffff, F32_REPRESENTABLE_EXACT_SUM),
        (vec![4096.0, 1.0, -1.0], 0.0, 0x45800000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![4096.0, -1.0], 0.0, 0x457fefff, F32_REPRESENTABLE_EXACT_SUM),
        (vec![4096.0, p2(-10), -p2(-10)], 0.0, 0x45800000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![3.0, -1.0], 5.0, 0x40e00000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![3.0], -1.0, 0x40000001, F32_REPRESENTABLE_EXACT_SUM),
        (vec![3.0, -1.0], -5.0, 0xc0400000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![], 7.0, 0x40e00000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![-4096.0, 1.0], 0.0, 0xc57ff001, F32_REPRESENTABLE_EXACT_SUM),
        (
            vec![3.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            0.0,
            0xbf800001,
            F32_REPRESENTABLE_EXACT_SUM,
        ),
        (vec![4096.0, -p2(-13)], 0.0, 0x457fffff, ROUNDING_EDGE),
        (vec![4096.0, p2(-12), p2(-13)], 0.0, 0x45800000, ROUNDING_EDGE),
        (vec![4096.0, -p2(-12), -p2(-13)], 0.0, 0x457ffffd, ROUNDING_EDGE),
        (vec![p2(-14), -p2(-15)], 0.0, 0x37fffffe, F32_REPRESENTABLE_EXACT_SUM),
        (vec![2.0, 2.0, -1.0, -1.0, -1.0, -1.0], 0.0, 0xb4800000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![-3.0], 0.0, 0xc0400000, F32_REPRESENTABLE_EXACT_SUM),
        (vec![p2(11), p2(11), -1.0], 0.0, 0x457ff000, F32_REPRESENTABLE_EXACT_SUM),
    ];
    let cases: Vec<(Vec<f32>, Vec<f32>, f32)> = corpus
        .iter()
        .map(|(xs, c, _, _)| (xs.clone(), ones(xs.len()), *c))
        .collect();
    let got = wmma_f16_dot_probe(ctx, &cases);
    for (((xs, c, pinned, representable), g), (_, ys, _)) in
        corpus.iter().zip(got.iter()).zip(cases.iter())
    {
        let exact = ieee_dot(xs, ys, *c);
        match class {
            WmmaAccumClass::TruncatingDot2 => assert_eq!(
                g.to_bits(),
                *pinned,
                "terms={xs:?} c={c}: got {:#010x}, the pinned gfx1151-class value is \
                 {pinned:#010x} (ieee would be {:#010x}). The truncating-dot2 accumulator is \
                 deterministic and this corpus is its fingerprint -- a driver or compiler \
                 change moved the datapath, re-characterize before trusting coop numerics",
                g.to_bits(),
                exact.to_bits()
            ),
            WmmaAccumClass::IeeeExact => {
                if *representable {
                    assert_eq!(
                        g.to_bits(),
                        exact.to_bits(),
                        "terms={xs:?} c={c}: got {:#010x}, want the exactly-representable \
                         {:#010x} on an IEEE-exact WMMA accumulator",
                        g.to_bits(),
                        exact.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn the_dot2_lane_pairing_is_where_the_truncation_lives() {
    let ctx = ctx();
    require_coop(ctx);
    let class = wmma_accum_class(ctx);
    let place = |slots: &[(usize, f32)]| -> (Vec<f32>, Vec<f32>) {
        let mut xs = vec![0f32; 16];
        for (k, v) in slots {
            xs[*k] = *v;
        }
        (xs, vec![1f32; 16])
    };
    let mut cases = Vec::new();
    for k in [1usize, 2, 3, 4, 7, 8, 15] {
        cases.push((place(&[(0, 3.0), (k, -1.0)]), k));
    }
    let got = wmma_f16_dot_probe(
        ctx,
        &cases.iter().map(|((xs, ys), _)| (xs.clone(), ys.clone(), 0.0)).collect::<Vec<_>>(),
    );
    for ((_, k), g) in cases.iter().zip(got.iter()) {
        match class {
            WmmaAccumClass::IeeeExact => assert_eq!(
                g.to_bits(),
                2f32.to_bits(),
                "3@0 + -1@{k}: an IEEE-exact accumulator returns 2.0 at every placement"
            ),
            WmmaAccumClass::TruncatingDot2 => {
                let want: u32 = if *k == 1 { 0x3fffffff } else { 0x40000000 };
                assert_eq!(
                    g.to_bits(),
                    want,
                    "3@0 + -1@{k}: the truncation fires exactly when both terms share a dot2 \
                     lane pair (lanes 2j, 2j+1) and nowhere else; got {:#010x}, pinned \
                     {want:#010x}",
                    g.to_bits()
                );
            }
        }
    }
}

#[test]
fn a_random_single_step_never_loses_more_than_one_unit_of_its_operand_scale() {
    let ctx = ctx();
    require_coop(ctx);
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let f16_clean = |r: u64| -> f32 {
        let mant = (r & 0x3ff) as u32;
        let exp = ((r >> 10) % 13) as i32 - 6;
        let sign = if (r >> 20) & 1 == 1 { -1f32 } else { 1f32 };
        sign * (1.0 + mant as f32 / 1024.0) * (2f32).powi(exp)
    };
    let mut cases = Vec::new();
    for _ in 0..512 {
        let p0 = f16_clean(rng());
        let p1 = f16_clean(rng());
        let c = f16_clean(rng());
        cases.push((vec![p0, p1], vec![1.0, 1.0], c));
    }
    let got = wmma_f16_dot_probe(ctx, &cases);
    let unit = 2f64.powi(-23);
    let mut worst = 0f64;
    for ((xs, _, c), g) in cases.iter().zip(got.iter()) {
        let exact = xs[0] as f64 + xs[1] as f64 + *c as f64;
        let scale = xs[0]
            .abs()
            .max(xs[1].abs())
            .max(c.abs())
            .max(exact.abs() as f32) as f64;
        let err = (*g as f64 - exact).abs();
        let ratio = err / (unit * scale);
        worst = worst.max(ratio);
        assert!(
            ratio <= 1.0,
            "step p0={} p1={} c={}: got {:#010x}, exact {exact:e}; error {err:e} is {ratio:.3} \
             of 2^-23 * the largest operand -- every characterized accumulator class stays at \
             or under one unit of the operand scale per step, and the chain ceilings in \
             wgpu_gemm_coop_prefill.rs are triangle inequalities over exactly this bound",
            xs[0],
            xs[1],
            c,
            g.to_bits()
        );
    }
    eprintln!("  worst single-step loss: {worst:.4} of one 2^-23 operand-scale unit");
}
