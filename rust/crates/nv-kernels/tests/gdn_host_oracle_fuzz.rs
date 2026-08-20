mod gdn_host_ref;

use gdn_host_ref::{PlantedBug, K_DIM, V_DIM, ref_rank1_algebra_f64};
mod common;
use common::max_rel_diff;

fn ref_token_major_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    b: usize,
    t_len: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; b * t_len * h * V_DIM];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0f32; K_DIM * V_DIM];
            for t in 0..t_len {
                let row = (bi * t_len + t) * h + hi;
                let qk = row * K_DIM;
                let vo = row * V_DIM;
                let ge = g_exp[row];
                let bt = beta[row];
                for mv in 0..V_DIM {
                    let mut kv_mem = 0f32;
                    for kk in 0..K_DIM {
                        let s = state[kk * V_DIM + mv] * ge;
                        state[kk * V_DIM + mv] = s;
                        kv_mem += s * k[qk + kk];
                    }
                    let delta = (v[vo + mv] - kv_mem) * bt;
                    let mut out_v = 0f32;
                    for kk in 0..K_DIM {
                        let s = state[kk * V_DIM + mv] + k[qk + kk] * delta;
                        state[kk * V_DIM + mv] = s;
                        out_v += s * q[qk + kk];
                    }
                    out[vo + mv] = out_v;
                }
            }
        }
    }
    out
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn inputs(b: usize, t: usize, h: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = b * t * h;
    let vecs = rows * K_DIM;
    let mut rng = Lcg::new(seed);
    let q: Vec<f32> = (0..vecs).map(|_| rng.next_f32()).collect();
    let k: Vec<f32> = (0..vecs).map(|_| rng.next_f32() * 0.125).collect();
    let v: Vec<f32> = (0..rows * V_DIM).map(|_| rng.next_f32()).collect();
    let g: Vec<f32> = (0..rows).map(|_| 0.75 + rng.next_f32().abs() * 0.25).collect();
    let beta: Vec<f32> = (0..rows).map(|_| rng.next_f32().abs()).collect();
    (q, k, v, g, beta)
}

const CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_RECURRENCE_DRIFT: f64 = 5e-4;

#[test]
fn interleaved_f32_and_rank1_f64_recurrences_agree() {
    let mut cases = 0usize;
    for (b, t, h) in [(1usize, 1usize, 1usize), (1, 7, 2), (2, 5, 3), (1, 64, 2)] {
        for seed in [1u64, 0x9e3779b9] {
            let (q, k, v, g, beta) = inputs(b, t, h, seed);
            let a = ref_token_major_f32(&q, &k, &v, &g, &beta, b, t, h);
            let r = ref_rank1_algebra_f64(&q, &k, &v, &g, &beta, b, t, h, PlantedBug::None);
            let d = max_rel_diff(&a, &r);
            assert!(
                d < CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_RECURRENCE_DRIFT,
                "kernel-order f32 reference and rank-1-algebra f64 reference diverged \
                 (b={b} t={t} h={h} seed={seed:#x}: rel {d:.3e}); one of the two no longer \
                 implements the documented gated-delta rule (decay, read, beta-scaled \
                 delta write at k, then out = state^T q)"
            );
            cases += 1;
        }
    }
    assert!(cases >= 8, "shape grid shrank to {cases} cases");
}

#[test]
fn every_planted_recurrence_bug_is_caught() {
    let (b, t, h) = (1usize, 16usize, 2usize);
    for &bug in &[
        PlantedBug::DecayAfterWrite,
        PlantedBug::ReadBeforeDecay,
        PlantedBug::MissingBetaInDelta,
        PlantedBug::WriteUsesQNotK,
    ] {
        let mut caught = false;
        for seed in [3u64, 4, 5] {
            let (q, k, v, g, beta) = inputs(b, t, h, seed);
            let good = ref_rank1_algebra_f64(&q, &k, &v, &g, &beta, b, t, h, PlantedBug::None);
            let bad = ref_rank1_algebra_f64(&q, &k, &v, &g, &beta, b, t, h, bug);
            if max_rel_diff(&good, &bad)
                > 20.0 * CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_RECURRENCE_DRIFT
            {
                caught = true;
                break;
            }
        }
        assert!(
            caught,
            "a planted gated-delta ordering bug survived the fuzz set; a gate that cannot \
             catch its own seeded mutations vouches for nothing (05.2 planted-bug protocol)"
        );
    }
}

#[test]
fn single_token_hides_the_decay_order_bugs_so_multi_token_is_mandatory() {
    let (q, k, v, g, beta) = inputs(1, 1, 1, 9);
    let good = ref_rank1_algebra_f64(&q, &k, &v, &g, &beta, 1, 1, 1, PlantedBug::None);
    let swapped = ref_rank1_algebra_f64(&q, &k, &v, &g, &beta, 1, 1, 1, PlantedBug::ReadBeforeDecay);
    assert!(
        max_rel_diff(&good, &swapped) < 1e-12,
        "with T=1 the state is zero at read time, so decay-vs-read ordering cannot be \
         observed; this pins WHY every GDN suite must include multi-token sequences"
    );
}
