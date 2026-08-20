#![allow(dead_code)]

pub const K_DIM: usize = 128;
pub const V_DIM: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlantedBug {
    None,
    DecayAfterWrite,
    ReadBeforeDecay,
    MissingBetaInDelta,
    WriteUsesQNotK,
}

pub fn ref_rank1_algebra_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    b: usize,
    t_len: usize,
    h: usize,
    bug: PlantedBug,
) -> Vec<f32> {
    let mut out = vec![0f32; b * t_len * h * V_DIM];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0f64; K_DIM * V_DIM];
            for t in 0..t_len {
                let row = (bi * t_len + t) * h + hi;
                let qk = row * K_DIM;
                let vo = row * V_DIM;
                let ge = g_exp[row] as f64;
                let bt = beta[row] as f64;
                let kt: Vec<f64> = (0..K_DIM).map(|i| k[qk + i] as f64).collect();
                let qt: Vec<f64> = (0..K_DIM).map(|i| q[qk + i] as f64).collect();
                let write_vec = if bug == PlantedBug::WriteUsesQNotK { &qt } else { &kt };

                if bug != PlantedBug::DecayAfterWrite && bug != PlantedBug::ReadBeforeDecay {
                    for s in state.iter_mut() {
                        *s *= ge;
                    }
                }

                let mut kv_mem = vec![0f64; V_DIM];
                for kk in 0..K_DIM {
                    let kv = kt[kk];
                    for mv in 0..V_DIM {
                        kv_mem[mv] += state[kk * V_DIM + mv] * kv;
                    }
                }
                if bug == PlantedBug::ReadBeforeDecay {
                    for s in state.iter_mut() {
                        *s *= ge;
                    }
                }

                let mut delta = vec![0f64; V_DIM];
                for mv in 0..V_DIM {
                    let d = v[vo + mv] as f64 - kv_mem[mv];
                    delta[mv] = if bug == PlantedBug::MissingBetaInDelta { d } else { d * bt };
                }

                for kk in 0..K_DIM {
                    let w = write_vec[kk];
                    for mv in 0..V_DIM {
                        state[kk * V_DIM + mv] += w * delta[mv];
                    }
                }
                if bug == PlantedBug::DecayAfterWrite {
                    for s in state.iter_mut() {
                        *s *= ge;
                    }
                }

                for mv in 0..V_DIM {
                    let mut acc = 0f64;
                    for kk in 0..K_DIM {
                        acc += state[kk * V_DIM + mv] * qt[kk];
                    }
                    out[vo + mv] = acc as f32;
                }
            }
        }
    }
    out
}
