mod gdn_host_ref;

use gdn_host_ref::{ref_rank1_algebra_f64, PlantedBug, K_DIM, V_DIM};

const F64_ALGEBRA_REL_TOL_1E9_REASSOCIATION_ONLY_SO_DOUBLE_PRECISION_MUST_AGREE_TO_PPB: f64 = 1e-9;
const PLANTED_MASK_BUG_MUST_MOVE_OUTPUTS_BY_1E6_OR_THE_GATE_VOUCHES_FOR_NOTHING: f64 = 1e-6;

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

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn inputs(b: usize, t: usize, h: usize, seed: u64, g_lo: f32, g_hi: f32) -> Inputs {
    let rows = b * t * h;
    let mut rng = Lcg::new(seed);
    Inputs {
        q: (0..rows * K_DIM).map(|_| rng.next_f32()).collect(),
        k: (0..rows * K_DIM).map(|_| rng.next_f32() * 0.125).collect(),
        v: (0..rows * V_DIM).map(|_| rng.next_f32()).collect(),
        g: (0..rows)
            .map(|_| g_lo + rng.next_f32().abs() * (g_hi - g_lo))
            .collect(),
        beta: (0..rows).map(|_| rng.next_f32().abs()).collect(),
    }
}

fn rand_state(b: usize, h: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..b * h * K_DIM * V_DIM)
        .map(|_| rng.next_f32() as f64 * 0.25)
        .collect()
}

fn ref_sequential_f64_with_state(
    x: &Inputs,
    b: usize,
    t_len: usize,
    h: usize,
    s0: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; b * t_len * h * V_DIM];
    let mut fin = vec![0f64; b * h * K_DIM * V_DIM];
    for bi in 0..b {
        for hi in 0..h {
            let sbase = (bi * h + hi) * K_DIM * V_DIM;
            let mut state = s0[sbase..sbase + K_DIM * V_DIM].to_vec();
            for t in 0..t_len {
                let row = (bi * t_len + t) * h + hi;
                let qk = row * K_DIM;
                let vo = row * V_DIM;
                let ge = x.g[row] as f64;
                let bt = x.beta[row] as f64;
                for s in state.iter_mut() {
                    *s *= ge;
                }
                let mut kv_mem = vec![0f64; V_DIM];
                for kk in 0..K_DIM {
                    let kv = x.k[qk + kk] as f64;
                    for mv in 0..V_DIM {
                        kv_mem[mv] += state[kk * V_DIM + mv] * kv;
                    }
                }
                let mut delta = vec![0f64; V_DIM];
                for mv in 0..V_DIM {
                    delta[mv] = (x.v[vo + mv] as f64 - kv_mem[mv]) * bt;
                }
                for kk in 0..K_DIM {
                    let w = x.k[qk + kk] as f64;
                    for mv in 0..V_DIM {
                        state[kk * V_DIM + mv] += w * delta[mv];
                    }
                }
                for mv in 0..V_DIM {
                    let mut acc = 0f64;
                    for kk in 0..K_DIM {
                        acc += state[kk * V_DIM + mv] * x.q[qk + kk] as f64;
                    }
                    out[vo + mv] = acc;
                }
            }
            fin[sbase..sbase + K_DIM * V_DIM].copy_from_slice(&state);
        }
    }
    (out, fin)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChunkBug {
    None,
    ADropsTheDecayRatioWhichTheUnitLowerSolveWouldOtherwiseNeverNotice,
    AttnStrictDiagonal,
    NoDecayOnStateCarry,
}

fn chunked_wy_ut_f64(
    x: &Inputs,
    b: usize,
    t_len: usize,
    h: usize,
    c_sub: usize,
    s0: &[f64],
    bug: ChunkBug,
) -> (Vec<f64>, Vec<f64>) {
    assert!(c_sub >= 1);
    let mut out = vec![0f64; b * t_len * h * V_DIM];
    let mut fin = vec![0f64; b * h * K_DIM * V_DIM];
    for bi in 0..b {
        for hi in 0..h {
            let sbase = (bi * h + hi) * K_DIM * V_DIM;
            let mut state = s0[sbase..sbase + K_DIM * V_DIM].to_vec();
            let mut start = 0usize;
            while start < t_len {
                let c = c_sub.min(t_len - start);
                let row_of = |i: usize| (bi * t_len + start + i) * h + hi;
                let cg: Vec<f64> = {
                    let mut acc = 0f64;
                    (0..c)
                        .map(|i| {
                            acc += (x.g[row_of(i)] as f64).ln();
                            acc
                        })
                        .collect()
                };
                let mut a = vec![0f64; c * c];
                for i in 0..c {
                    let bt_i = x.beta[row_of(i)] as f64;
                    for j in 0..i {
                        let mut dot = 0f64;
                        let ki = row_of(i) * K_DIM;
                        let kj = row_of(j) * K_DIM;
                        for d in 0..K_DIM {
                            dot += x.k[ki + d] as f64 * x.k[kj + d] as f64;
                        }
                        let ratio = if bug
                            == ChunkBug::ADropsTheDecayRatioWhichTheUnitLowerSolveWouldOtherwiseNeverNotice
                        {
                            1f64
                        } else {
                            (cg[i] - cg[j]).exp()
                        };
                        a[i * c + j] = bt_i * ratio * dot;
                    }
                }
                let mut tinv = vec![0f64; c * c];
                for i in 0..c {
                    for col in 0..c {
                        let mut v = if i == col { 1f64 } else { 0f64 };
                        for j in 0..i {
                            v -= a[i * c + j] * tinv[j * c + col];
                        }
                        tinv[i * c + col] = v;
                    }
                }
                let mut rhs = vec![0f64; c * (K_DIM + V_DIM)];
                for i in 0..c {
                    let bt_i = x.beta[row_of(i)] as f64;
                    let bexp = cg[i].exp();
                    let ki = row_of(i) * K_DIM;
                    let vi = row_of(i) * V_DIM;
                    for d in 0..K_DIM {
                        rhs[i * (K_DIM + V_DIM) + d] = bt_i * bexp * x.k[ki + d] as f64;
                    }
                    for d in 0..V_DIM {
                        rhs[i * (K_DIM + V_DIM) + K_DIM + d] = bt_i * x.v[vi + d] as f64;
                    }
                }
                let mut wu = vec![0f64; c * (K_DIM + V_DIM)];
                for i in 0..c {
                    for j in 0..c {
                        let tv = tinv[i * c + j];
                        if tv != 0.0 {
                            for d in 0..K_DIM + V_DIM {
                                wu[i * (K_DIM + V_DIM) + d] += tv * rhs[j * (K_DIM + V_DIM) + d];
                            }
                        }
                    }
                }
                let mut attn = vec![0f64; c * c];
                for i in 0..c {
                    let incl = if bug == ChunkBug::AttnStrictDiagonal {
                        i
                    } else {
                        i + 1
                    };
                    for j in 0..incl.min(c) {
                        let mut dot = 0f64;
                        let qi = row_of(i) * K_DIM;
                        let kj = row_of(j) * K_DIM;
                        for d in 0..K_DIM {
                            dot += x.q[qi + d] as f64 * x.k[kj + d] as f64;
                        }
                        attn[i * c + j] = (cg[i] - cg[j]).exp() * dot;
                    }
                }
                let mut o_loc = vec![0f64; c * V_DIM];
                let mut q_eff = vec![0f64; c * K_DIM];
                for i in 0..c {
                    let bexp = cg[i].exp();
                    let qi = row_of(i) * K_DIM;
                    for d in 0..K_DIM {
                        q_eff[i * K_DIM + d] = bexp * x.q[qi + d] as f64;
                    }
                    for j in 0..c {
                        let av = attn[i * c + j];
                        if av != 0.0 {
                            for d in 0..V_DIM {
                                o_loc[i * V_DIM + d] += av * wu[j * (K_DIM + V_DIM) + K_DIM + d];
                            }
                            for d in 0..K_DIM {
                                q_eff[i * K_DIM + d] -= av * wu[j * (K_DIM + V_DIM) + d];
                            }
                        }
                    }
                }
                for i in 0..c {
                    let vo = row_of(i) * V_DIM;
                    for d in 0..V_DIM {
                        let mut acc = o_loc[i * V_DIM + d];
                        for kk in 0..K_DIM {
                            acc += q_eff[i * K_DIM + kk] * state[kk * V_DIM + d];
                        }
                        out[vo + d] = acc;
                    }
                }
                let b_last = if bug == ChunkBug::NoDecayOnStateCarry {
                    1f64
                } else {
                    cg[c - 1].exp()
                };
                let mut kc = vec![0f64; c * K_DIM];
                for i in 0..c {
                    let scale = (cg[c - 1] - cg[i]).exp();
                    let ki = row_of(i) * K_DIM;
                    for d in 0..K_DIM {
                        kc[i * K_DIM + d] = scale * x.k[ki + d] as f64;
                    }
                }
                let mut p = vec![0f64; K_DIM * K_DIM];
                let mut gmat = vec![0f64; K_DIM * V_DIM];
                for i in 0..c {
                    for kk in 0..K_DIM {
                        let kv = kc[i * K_DIM + kk];
                        if kv != 0.0 {
                            for d in 0..K_DIM {
                                p[kk * K_DIM + d] -= kv * wu[i * (K_DIM + V_DIM) + d];
                            }
                            for d in 0..V_DIM {
                                gmat[kk * V_DIM + d] += kv * wu[i * (K_DIM + V_DIM) + K_DIM + d];
                            }
                        }
                    }
                }
                for d in 0..K_DIM {
                    p[d * K_DIM + d] += b_last;
                }
                let mut new_state = gmat;
                for kk in 0..K_DIM {
                    for j in 0..K_DIM {
                        let pv = p[kk * K_DIM + j];
                        if pv != 0.0 {
                            for d in 0..V_DIM {
                                new_state[kk * V_DIM + d] += pv * state[j * V_DIM + d];
                            }
                        }
                    }
                }
                state = new_state;
                start += c;
            }
            fin[sbase..sbase + K_DIM * V_DIM].copy_from_slice(&state);
        }
    }
    (out, fin)
}

fn max_rel_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&p, &r)| (p - r).abs() / p.abs().max(r.abs()).max(1e-3))
        .fold(0.0, f64::max)
}

const SHAPE_GRID_T_CROSSES_SUBCHUNK_SEAMS_AND_TAILS: [(usize, usize, usize); 5] =
    [(1, 1, 1), (1, 3, 2), (1, 64, 2), (2, 130, 2), (1, 257, 3)];
const SUBCHUNK_GRID_1_IS_THE_DEGENERATE_SEQUENTIAL_CASE_64_IS_THE_CUDA_TILE: [usize; 4] =
    [1, 4, 16, 64];

#[test]
fn chunked_wy_ut_matches_the_rank1_reference_from_zero_state() {
    let mut cases = 0usize;
    for (b, t, h) in SHAPE_GRID_T_CROSSES_SUBCHUNK_SEAMS_AND_TAILS {
        for c in SUBCHUNK_GRID_1_IS_THE_DEGENERATE_SEQUENTIAL_CASE_64_IS_THE_CUDA_TILE {
            for (seed, g_lo, g_hi) in [(1u64, 0.75f32, 1.0f32), (0x9e3779b9, 0.1, 0.999)] {
                let x = inputs(b, t, h, seed, g_lo, g_hi);
                let s0 = vec![0f64; b * h * K_DIM * V_DIM];
                let (got, _) = chunked_wy_ut_f64(&x, b, t, h, c, &s0, ChunkBug::None);
                let r = ref_rank1_algebra_f64(&x.q, &x.k, &x.v, &x.g, &x.beta, b, t, h, PlantedBug::None);
                let rf: Vec<f64> = r.iter().map(|&v| v as f64).collect();
                let gf: Vec<f64> = got.clone();
                let d = max_rel_f64(&gf, &rf);
                assert!(
                    d < 1e-6,
                    "chunked WY/UT diverged from the rank-1 f32-output reference beyond its own \
                     f32 cast noise (b={b} t={t} h={h} c={c} seed={seed:#x} rel={d:.3e}); the \
                     chunk algebra no longer implements decay, beta-delta write, then out"
                );
                cases += 1;
            }
        }
    }
    assert!(cases >= 40, "shape grid shrank to {cases} cases");
}

#[test]
fn chunked_wy_ut_matches_the_sequential_recurrence_with_a_nonzero_carried_state() {
    for (b, t, h) in SHAPE_GRID_T_CROSSES_SUBCHUNK_SEAMS_AND_TAILS {
        for c in SUBCHUNK_GRID_1_IS_THE_DEGENERATE_SEQUENTIAL_CASE_64_IS_THE_CUDA_TILE {
            let x = inputs(b, t, h, 0xfeed ^ (t as u64), 0.6, 0.999);
            let s0 = rand_state(b, h, 0xdead ^ (c as u64));
            let (got_o, got_s) = chunked_wy_ut_f64(&x, b, t, h, c, &s0, ChunkBug::None);
            let (ref_o, ref_s) = ref_sequential_f64_with_state(&x, b, t, h, &s0);
            let d_o = max_rel_f64(&got_o, &ref_o);
            let d_s = max_rel_f64(&got_s, &ref_s);
            assert!(
                d_o < F64_ALGEBRA_REL_TOL_1E9_REASSOCIATION_ONLY_SO_DOUBLE_PRECISION_MUST_AGREE_TO_PPB
                    && d_s < F64_ALGEBRA_REL_TOL_1E9_REASSOCIATION_ONLY_SO_DOUBLE_PRECISION_MUST_AGREE_TO_PPB,
                "chunked WY/UT with carried state diverged (b={b} t={t} h={h} c={c}): out rel \
                 {d_o:.3e} state rel {d_s:.3e}; the S0 correction term (Q_eff = b*q - Attn*W, \
                 P = b_C*I - Kc^T*W) is wrong"
            );
        }
    }
}

#[test]
fn every_planted_chunk_mask_bug_is_caught() {
    let (b, t, h, c) = (1usize, 64usize, 2usize, 16usize);
    for bug in [
        ChunkBug::ADropsTheDecayRatioWhichTheUnitLowerSolveWouldOtherwiseNeverNotice,
        ChunkBug::AttnStrictDiagonal,
        ChunkBug::NoDecayOnStateCarry,
    ] {
        let mut caught = false;
        for seed in [3u64, 4, 5] {
            let x = inputs(b, t, h, seed, 0.6, 0.999);
            let s0 = rand_state(b, h, seed);
            let (good_o, good_s) = chunked_wy_ut_f64(&x, b, t, h, c, &s0, ChunkBug::None);
            let (bad_o, bad_s) = chunked_wy_ut_f64(&x, b, t, h, c, &s0, bug);
            if max_rel_f64(&good_o, &bad_o)
                > PLANTED_MASK_BUG_MUST_MOVE_OUTPUTS_BY_1E6_OR_THE_GATE_VOUCHES_FOR_NOTHING
                || max_rel_f64(&good_s, &bad_s)
                    > PLANTED_MASK_BUG_MUST_MOVE_OUTPUTS_BY_1E6_OR_THE_GATE_VOUCHES_FOR_NOTHING
            {
                caught = true;
                break;
            }
        }
        assert!(
            caught,
            "a planted chunk-mask bug survived the grid; a gate that cannot catch its own \
             seeded mutations vouches for nothing (05.2 planted-bug protocol)"
        );
    }
}
