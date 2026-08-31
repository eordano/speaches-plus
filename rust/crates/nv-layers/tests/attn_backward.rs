use candle_core::{DType, Device, Tensor, Var};
use nv_layers::attn::{
    sdpa, sdpa_windowed, sdpa_with_sinks, softmax_last_dim_that_keeps_q_and_k_on_the_graph,
    AttnConfig,
};

const F32_MANTISSA_EPS: f64 = 1.19e-7;

const SDPA_CASTS_ITS_INPUTS_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_NO_MATTER_WHAT_DTYPE_THE_VARS_ARE: f64 =
    512.0 * F32_MANTISSA_EPS;

const AN_F64_CENTRAL_DIFFERENCE_AT_THIS_STEP_CARRIES_TRUNCATION_H2_AND_ROUNDOFF_EPS_OVER_H: f64 =
    1e-5;

const THE_CENTRAL_DIFFERENCE_OF_AN_F64_FORWARD_IS_GOOD_TO_ABOUT_TEN_DIGITS: f64 = 1e-7;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn f64_tensor(v: &[f64], shape: &[usize], dev: &Device) -> Tensor {
    Tensor::from_vec(v.to_vec(), shape, dev).unwrap()
}

fn host(t: &Tensor) -> Vec<f64> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn softmax_rows_in_f64(scores: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * cols];
    for r in 0..rows {
        let row = &scores[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        for (c, e) in exps.iter().enumerate() {
            out[r * cols + c] = e / z;
        }
    }
    out
}

const ROWS: usize = 6;
const COLS: usize = 7;

fn masked_score_corpus() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(0x5EED_1234);
    let mut scores = rng.vec(ROWS * COLS, 3.0);
    for r in 0..ROWS {
        for c in 0..COLS {
            if c > r + (COLS - ROWS) {
                scores[r * COLS + c] = f64::NEG_INFINITY;
            }
        }
    }
    let probe = rng.vec(ROWS * COLS, 1.0);
    (scores, probe)
}

fn softmax_probe_loss_in_f64(scores: &[f64], probe: &[f64]) -> f64 {
    softmax_rows_in_f64(scores, ROWS, COLS)
        .iter()
        .zip(probe.iter())
        .map(|(p, w)| p * w)
        .sum()
}

#[test]
fn the_attention_softmax_backward_matches_an_f64_central_difference_and_the_closed_form_jacobian_vector_product(
) {
    let dev = Device::Cpu;
    let (scores, probe) = masked_score_corpus();

    let var = Var::from_tensor(&f64_tensor(&scores, &[ROWS, COLS], &dev)).unwrap();
    let probe_t = f64_tensor(&probe, &[ROWS, COLS], &dev);
    let p = softmax_last_dim_that_keeps_q_and_k_on_the_graph(var.as_tensor()).unwrap();
    assert_eq!(
        p.dtype(),
        DType::F64,
        "the differentiable branch must not silently downcast the probabilities"
    );
    let loss = p.mul(&probe_t).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let analytic = host(grads.get(&var).unwrap_or_else(|| {
        panic!(
            "the score tensor received no gradient entry at all, which is the original defect: \
             candle_nn::ops::softmax_last_dim is installed with apply_op1_no_bwd, so attention \
             probabilities are a graph leaf and q and k train nowhere while v still does."
        )
    }));

    let p_host = softmax_rows_in_f64(&scores, ROWS, COLS);
    let mut closed_form = vec![0f64; ROWS * COLS];
    for r in 0..ROWS {
        let dot: f64 = (0..COLS)
            .map(|c| p_host[r * COLS + c] * probe[r * COLS + c])
            .sum();
        for c in 0..COLS {
            closed_form[r * COLS + c] = p_host[r * COLS + c] * (probe[r * COLS + c] - dot);
        }
    }

    let h = AN_F64_CENTRAL_DIFFERENCE_AT_THIS_STEP_CARRIES_TRUNCATION_H2_AND_ROUNDOFF_EPS_OVER_H;
    let mut worst_numeric = 0f64;
    let mut worst_closed = 0f64;
    for i in 0..ROWS * COLS {
        let mut up = scores.clone();
        let mut down = scores.clone();
        up[i] += h;
        down[i] -= h;
        let numeric = (softmax_probe_loss_in_f64(&up, &probe)
            - softmax_probe_loss_in_f64(&down, &probe))
            / (2.0 * h);
        let scale = numeric.abs().max(1.0);
        worst_numeric = worst_numeric.max((analytic[i] - numeric).abs() / scale);
        worst_closed = worst_closed.max((analytic[i] - closed_form[i]).abs() / scale);
    }

    eprintln!(
        "softmax backward: worst gap vs f64 central difference {worst_numeric:e}, vs closed-form \
         p*(dy-sum(p*dy)) {worst_closed:e}"
    );
    assert!(
        worst_numeric <= THE_CENTRAL_DIFFERENCE_OF_AN_F64_FORWARD_IS_GOOD_TO_ABOUT_TEN_DIGITS,
        "the softmax backward disagrees with an f64 central difference of an independent host \
         softmax by {worst_numeric:e}, over a budget of {:e}",
        THE_CENTRAL_DIFFERENCE_OF_AN_F64_FORWARD_IS_GOOD_TO_ABOUT_TEN_DIGITS
    );
    assert!(
        worst_closed <= THE_CENTRAL_DIFFERENCE_OF_AN_F64_FORWARD_IS_GOOD_TO_ABOUT_TEN_DIGITS,
        "the softmax backward is not dx = p * (dy - sum(p * dy)); worst gap {worst_closed:e}"
    );
}

#[test]
fn the_central_difference_gate_rejects_the_jacobian_that_forgets_to_subtract_the_mean() {
    let (scores, probe) = masked_score_corpus();
    let p_host = softmax_rows_in_f64(&scores, ROWS, COLS);
    let h = AN_F64_CENTRAL_DIFFERENCE_AT_THIS_STEP_CARRIES_TRUNCATION_H2_AND_ROUNDOFF_EPS_OVER_H;

    let mut worst = 0f64;
    for i in 0..ROWS * COLS {
        let mut up = scores.clone();
        let mut down = scores.clone();
        up[i] += h;
        down[i] -= h;
        let numeric = (softmax_probe_loss_in_f64(&up, &probe)
            - softmax_probe_loss_in_f64(&down, &probe))
            / (2.0 * h);
        let elementwise_only = p_host[i] * probe[i];
        worst = worst.max((elementwise_only - numeric).abs() / numeric.abs().max(1.0));
    }
    eprintln!("falsification: p*dy alone misses the central difference by {worst:e}");
    assert!(
        worst > THE_CENTRAL_DIFFERENCE_OF_AN_F64_FORWARD_IS_GOOD_TO_ABOUT_TEN_DIGITS * 1e4,
        "p * dy without the subtracted row mean passed the same central-difference gate the row \
         above uses (worst {worst:e}), so that gate cannot tell a correct softmax Jacobian from a \
         plainly wrong one and proves nothing"
    );
}

const B: usize = 1;
const SQ: usize = 4;
const SK: usize = 6;
const H: usize = 4;
const H_KV: usize = 2;
const D: usize = 5;

fn sdpa_probe_loss_in_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    probe: &[f64],
    scale: f64,
    window: usize,
) -> f64 {
    let group = H / H_KV;
    let off = SK - SQ;
    let mut loss = 0f64;
    for bi in 0..B {
        for hi in 0..H {
            let g = hi / group;
            for i in 0..SQ {
                let last = i + off;
                let first = if window > 0 {
                    (last + 1).saturating_sub(window)
                } else {
                    0
                };
                let mut scores = Vec::with_capacity(last - first + 1);
                for j in first..=last {
                    let dot: f64 = (0..D)
                        .map(|x| {
                            q[((bi * SQ + i) * H + hi) * D + x]
                                * k[((bi * SK + j) * H_KV + g) * D + x]
                        })
                        .sum();
                    scores.push(dot * scale);
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = exps.iter().sum();
                for x in 0..D {
                    let mut acc = 0f64;
                    for (n, j) in (first..=last).enumerate() {
                        acc += (exps[n] / z) * v[((bi * SK + j) * H_KV + g) * D + x];
                    }
                    loss += acc * probe[((bi * SQ + i) * H + hi) * D + x];
                }
            }
        }
    }
    loss
}

fn sdpa_analytic_gradients(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    probe: &[f64],
    scale: f64,
    window: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dev = Device::Cpu;
    let qv = Var::from_tensor(&f64_tensor(q, &[B, SQ, H, D], &dev)).unwrap();
    let kv = Var::from_tensor(&f64_tensor(k, &[B, SK, H_KV, D], &dev)).unwrap();
    let vv = Var::from_tensor(&f64_tensor(v, &[B, SK, H_KV, D], &dev)).unwrap();
    let cfg = AttnConfig {
        num_heads: H,
        num_kv_heads: H_KV,
        head_dim: D,
        softmax_scale: scale as f32,
        causal: true,
    };
    let out = if window > 0 {
        sdpa_windowed(qv.as_tensor(), kv.as_tensor(), vv.as_tensor(), &cfg, window).unwrap()
    } else {
        sdpa(qv.as_tensor(), kv.as_tensor(), vv.as_tensor(), &cfg).unwrap()
    };
    let probe_t = f64_tensor(probe, &[B, SQ, H, D], &dev);
    let grads = out
        .mul(&probe_t)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    let fetch = |label: &str, var: &Var| -> Vec<f64> {
        host(grads.get(var).unwrap_or_else(|| {
            panic!(
                "{label} received no gradient entry at all from nv_layers::attn::sdpa. This is \
                 the original defect: the fused softmax_last_dim is a graph leaf, so q and k are \
                 unreachable from the loss while v still receives one through probs @ v, and a \
                 trainer calling sdpa silently optimises the value projection alone."
            )
        }))
    };
    (fetch("q", &qv), fetch("k", &kv), fetch("v", &vv))
}

fn assert_sdpa_gradient_matches_a_central_difference(window: usize) {
    let mut rng = Lcg(0xC0FFEE_00 + window as u64);
    let q = rng.vec(B * SQ * H * D, 0.7);
    let k = rng.vec(B * SK * H_KV * D, 0.7);
    let v = rng.vec(B * SK * H_KV * D, 0.7);
    let probe = rng.vec(B * SQ * H * D, 1.0);
    let scale = 1.0 / (D as f64).sqrt();

    let (gq, gk, gv) = sdpa_analytic_gradients(&q, &k, &v, &probe, scale, window);
    let h = AN_F64_CENTRAL_DIFFERENCE_AT_THIS_STEP_CARRIES_TRUNCATION_H2_AND_ROUNDOFF_EPS_OVER_H;
    let budget = SDPA_CASTS_ITS_INPUTS_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_NO_MATTER_WHAT_DTYPE_THE_VARS_ARE;

    for (label, base, analytic) in [("q", &q, &gq), ("k", &k, &gk), ("v", &v, &gv)] {
        assert_eq!(
            analytic.len(),
            base.len(),
            "{label}: gradient shape does not match the input shape"
        );
        let mut worst = 0f64;
        let mut worst_at = 0usize;
        let mut any_nonzero = false;
        for i in 0..base.len() {
            let mut up = base.to_vec();
            let mut down = base.to_vec();
            up[i] += h;
            down[i] -= h;
            let numeric = match label {
                "q" => {
                    (sdpa_probe_loss_in_f64(&up, &k, &v, &probe, scale, window)
                        - sdpa_probe_loss_in_f64(&down, &k, &v, &probe, scale, window))
                        / (2.0 * h)
                }
                "k" => {
                    (sdpa_probe_loss_in_f64(&q, &up, &v, &probe, scale, window)
                        - sdpa_probe_loss_in_f64(&q, &down, &v, &probe, scale, window))
                        / (2.0 * h)
                }
                _ => {
                    (sdpa_probe_loss_in_f64(&q, &k, &up, &probe, scale, window)
                        - sdpa_probe_loss_in_f64(&q, &k, &down, &probe, scale, window))
                        / (2.0 * h)
                }
            };
            assert!(
                analytic[i].is_finite(),
                "{label}[{i}]: non-finite gradient {}",
                analytic[i]
            );
            any_nonzero |= analytic[i] != 0.0;
            let gap = (analytic[i] - numeric).abs() / numeric.abs().max(1.0);
            if gap > worst {
                worst = gap;
                worst_at = i;
            }
        }
        eprintln!(
            "sdpa window={window} {label}: {} probed elements, worst gap vs f64 central \
             difference {worst:e}, budget {budget:e}",
            base.len()
        );
        assert!(
            any_nonzero,
            "{label}: every gradient entry is exactly zero, which a graph leaf also produces"
        );
        assert!(
            worst <= budget,
            "window={window} {label}: sdpa's analytic gradient {} disagrees with an f64 central \
             difference of an independent host attention by {worst:e} at element {worst_at}, \
             over an f32 budget of {budget:e}",
            analytic[worst_at]
        );
    }
}

#[test]
fn sdpa_reaches_q_and_k_with_a_gradient_that_matches_an_f64_central_difference() {
    assert_sdpa_gradient_matches_a_central_difference(0);
}

#[test]
fn a_sliding_window_sdpa_reaches_q_and_k_with_a_gradient_that_matches_the_same_oracle() {
    assert_sdpa_gradient_matches_a_central_difference(3);
}

#[test]
fn the_gqa_expansion_and_the_causal_mask_both_change_the_answer_which_is_what_makes_the_rows_above_gates(
) {
    let mut rng = Lcg(0xABCD_EF01);
    let q = rng.vec(B * SQ * H * D, 0.7);
    let k = rng.vec(B * SK * H_KV * D, 0.7);
    let v = rng.vec(B * SK * H_KV * D, 0.7);
    let probe = rng.vec(B * SQ * H * D, 1.0);
    let scale = 1.0 / (D as f64).sqrt();

    let open = sdpa_probe_loss_in_f64(&q, &k, &v, &probe, scale, 0);
    let windowed = sdpa_probe_loss_in_f64(&q, &k, &v, &probe, scale, 3);
    assert!(
        (open - windowed).abs() > 1e-6,
        "the sliding window changes nothing on this fixture, so the windowed row proves nothing"
    );
    assert!(
        H_KV < H && SK > SQ,
        "the fixture must exercise GQA and a key run longer than the query run"
    );

    let (gq, gk, gv) = sdpa_analytic_gradients(&q, &k, &v, &probe, scale, 0);
    for (label, g) in [("q", &gq), ("k", &gk), ("v", &gv)] {
        assert!(
            g.iter().any(|x| x.abs() > 1e-6),
            "{label}: the gradient is numerically indistinguishable from zero, so the row above \
             would pass against a graph leaf"
        );
    }
}

#[test]
fn sdpa_with_sinks_also_reaches_q_and_k_and_the_sink_logit_itself() {
    let dev = Device::Cpu;
    let mut rng = Lcg(0x51_4E_51_4E);
    let q = rng.vec(B * SQ * H * D, 0.7);
    let k = rng.vec(B * SK * H_KV * D, 0.7);
    let v = rng.vec(B * SK * H_KV * D, 0.7);
    let sinks = rng.vec(H, 0.5);
    let qv = Var::from_tensor(&f64_tensor(&q, &[B, SQ, H, D], &dev)).unwrap();
    let kv = Var::from_tensor(&f64_tensor(&k, &[B, SK, H_KV, D], &dev)).unwrap();
    let vv = Var::from_tensor(&f64_tensor(&v, &[B, SK, H_KV, D], &dev)).unwrap();
    let sv = Var::from_tensor(&f64_tensor(&sinks, &[H], &dev)).unwrap();
    let cfg = AttnConfig {
        num_heads: H,
        num_kv_heads: H_KV,
        head_dim: D,
        softmax_scale: 1.0 / (D as f32).sqrt(),
        causal: true,
    };
    let out = sdpa_with_sinks(
        qv.as_tensor(),
        kv.as_tensor(),
        vv.as_tensor(),
        &cfg,
        sv.as_tensor(),
        0,
    )
    .unwrap();
    let grads = out.sum_all().unwrap().backward().unwrap();
    for (label, var) in [("q", &qv), ("k", &kv), ("v", &vv), ("sinks", &sv)] {
        let g = host(grads.get(var).unwrap_or_else(|| {
            panic!("{label} received no gradient entry at all from sdpa_with_sinks")
        }));
        assert!(
            g.iter().all(|x| x.is_finite()),
            "{label}: non-finite gradient through the appended sink column"
        );
    }
}

#[test]
fn serving_still_runs_the_fused_kernel_and_its_probabilities_are_bit_identical() {
    let dev = Device::Cpu;
    let mut rng = Lcg(0x7E57_0007);
    for &(rows, cols) in &[(1usize, 1usize), (3, 8), (6, 7), (17, 33)] {
        let mut scores = rng.vec(rows * cols, 4.0);
        for r in 0..rows {
            for c in 0..cols {
                if c > r + cols.saturating_sub(rows) {
                    scores[r * cols + c] = f64::NEG_INFINITY;
                }
            }
        }
        let f32_scores: Vec<f32> = scores.iter().map(|&x| x as f32).collect();
        let t = Tensor::from_vec(f32_scores, (rows, cols), &dev).unwrap();
        assert!(
            !t.track_op(),
            "a serving score tensor must not be on an autograd graph or this row proves nothing \
             about what the engine runs"
        );
        let fused: Vec<f32> = candle_nn::ops::softmax_last_dim(&t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let served = softmax_last_dim_that_keeps_q_and_k_on_the_graph(&t).unwrap();
        assert!(
            !served.track_op(),
            "the serving branch must return the same graph leaf it always returned"
        );
        let got: Vec<f32> = served.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (a, b)) in fused.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{rows}x{cols} element {i}: the serving softmax moved off the fused kernel \
                 ({a} vs {b}). Nothing served may change bits: this fix is a backward, not a \
                 forward."
            );
        }
    }
}

const THE_QWEN38_FULL_ATTENTION_ROW_AT_SEQ_256: (usize, usize) = (256, 256);

#[test]
fn the_trained_probabilities_are_the_served_probabilities_to_f32_roundoff() {
    let dev = Device::Cpu;
    let mut rng = Lcg(0x1234_5678);
    let budget = 16.0 * F32_MANTISSA_EPS;
    for (rows, cols) in [(ROWS, COLS), THE_QWEN38_FULL_ATTENTION_ROW_AT_SEQ_256] {
        let scores = rng.vec(rows * cols, 4.0);
        let f32_scores: Vec<f32> = scores.iter().map(|&x| x as f32).collect();
        let served_in = Tensor::from_vec(f32_scores.clone(), (rows, cols), &dev).unwrap();
        let tracked_in = Var::from_tensor(&served_in).unwrap();
        let served = host(&softmax_last_dim_that_keeps_q_and_k_on_the_graph(&served_in).unwrap());
        let trained = host(
            &softmax_last_dim_that_keeps_q_and_k_on_the_graph(tracked_in.as_tensor()).unwrap(),
        );
        let mut worst = 0f64;
        for (a, b) in served.iter().zip(trained.iter()) {
            worst = worst.max((a - b).abs() / a.abs().max(1e-3));
        }
        eprintln!("train-vs-serve softmax gap at {rows}x{cols}: {worst:e} (budget {budget:e})");
        assert!(
            worst <= budget,
            "at {rows}x{cols} the composed softmax the trainer runs and the fused softmax serving \
             runs differ by {worst:e} relative, over a budget of {budget:e}. An adapter would \
             then be fitted against a different function than the one the engine serves."
        );
    }
}
