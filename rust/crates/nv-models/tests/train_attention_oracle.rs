use candle_core::{DType, Device, Tensor, Var};
use nv_models::dense_train::naive_sdpa;

const BATCH: usize = 2;
const SEQ: usize = 3;
const STORED: usize = 5;
const N_Q: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 8;

const Q_ROW: usize = SEQ * N_Q * HEAD_DIM;
const KV_ROW: usize = STORED * N_KV * HEAD_DIM;

const F32_MANTISSA_EPS: f64 = 1.0 / (1u64 << 23) as f64;

const GRADIENT_BUDGET_IS_512_F32_MANTISSA_EPS_BECAUSE_NAIVE_SDPA_CASTS_EVERY_INPUT_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_WHATEVER_DTYPE_THE_VARS_CARRY: f64 =
    512.0 * F32_MANTISSA_EPS;

const CENTRAL_DIFFERENCE_STEP_1EM5_LEAVES_F64_TRUNCATION_AND_ROUNDOFF_FIVE_DECADES_UNDER_THAT_BUDGET: f64 =
    1e-5;

const A_WRONG_JACOBIAN_MUST_MISS_THE_CENTRAL_DIFFERENCE_BY_AT_LEAST_THIS_OR_THE_GATE_IS_VACUOUS:
    f64 = 1e-3;

const NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_FUSED_LEAF_SOFTMAX_PRODUCES: &str =
    "received no gradient entry at all from dense_train::naive_sdpa. That is the fused \
     candle_nn::ops::softmax_last_dim leaf: candle installs it with apply_op1_no_bwd, so the \
     probabilities carry BackpropOp::none and q and k are unreachable from the loss while v \
     still trains through the probs @ v matmul that follows. train_runner emits \
     self_attn.q_proj and self_attn.k_proj as DEFAULT LoRA targets, their B factor is \
     zero-initialised, and candle's AdamW::step silently skips a Var with no GradStore entry -- \
     so B stays exactly zero, those modules merge as an exact no-op, and the loss falls the \
     whole time with nothing reporting it. This is the Gemma4 dense LoRA trainer reached by \
     nvk-train, train_runner::run and the /v1/fine_tuning HTTP endpoint. naive_sdpa must score \
     through nv_layers::attn::softmax_last_dim_that_keeps_q_and_k_on_the_graph, which keeps the \
     identical fused call for serving and takes the composed tracked softmax only while an \
     autograd graph is being built.";

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let seeds: [(u64, u64, u64); BATCH] = [(11, 22, 33), (44, 55, 66)];
    let mut qh_all = Vec::new();
    let mut kh_all = Vec::new();
    let mut vh_all = Vec::new();
    for (sq, sk, sv) in seeds {
        qh_all.extend_from_slice(&seeded(Q_ROW, sq));
        kh_all.extend_from_slice(&seeded(KV_ROW, sk));
        vh_all.extend_from_slice(&seeded(KV_ROW, sv));
    }
    (qh_all, kh_all, vh_all)
}

fn cotangents() -> Vec<f32> {
    seeded(BATCH * Q_ROW, 987_654_321)
}

fn visible(i: usize, j: usize, window: Option<usize>) -> bool {
    let qpos = (STORED - SEQ + i) as i64;
    let kpos = j as i64;
    kpos <= qpos
        && match window {
            Some(w) => qpos - kpos < w as i64,
            None => true,
        }
}

fn probabilities_in_f64(
    q: &[f64],
    k: &[f64],
    h: usize,
    i: usize,
    window: Option<usize>,
) -> Vec<f64> {
    let kvh = h / (N_Q / N_KV);
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut logits = vec![f64::NEG_INFINITY; STORED];
    for (j, logit) in logits.iter_mut().enumerate() {
        if !visible(i, j, window) {
            continue;
        }
        let mut dot = 0f64;
        for d in 0..HEAD_DIM {
            dot += q[(i * N_Q + h) * HEAD_DIM + d] * k[(j * N_KV + kvh) * HEAD_DIM + d];
        }
        *logit = dot * scale;
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|l| if l.is_finite() { (l - max).exp() } else { 0.0 })
        .collect();
    let denom: f64 = exps.iter().sum();
    assert!(
        denom > 0.0,
        "row {i} head {h} attends to nothing, so this row is vacuous"
    );
    exps.iter().map(|e| e / denom).collect()
}

fn attention_in_f64(q: &[f64], k: &[f64], v: &[f64], window: Option<usize>) -> Vec<f64> {
    let group = N_Q / N_KV;
    let mut out = vec![0f64; Q_ROW];
    for h in 0..N_Q {
        let kvh = h / group;
        for i in 0..SEQ {
            let p = probabilities_in_f64(q, k, h, i, window);
            for d in 0..HEAD_DIM {
                let mut acc = 0f64;
                for (j, pj) in p.iter().enumerate() {
                    acc += pj * v[(j * N_KV + kvh) * HEAD_DIM + d];
                }
                out[(i * N_Q + h) * HEAD_DIM + d] = acc;
            }
        }
    }
    out
}

fn attention_backward_in_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    dy: &[f64],
    window: Option<usize>,
    softmax_jacobian_subtracts_the_probability_weighted_mean: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let group = N_Q / N_KV;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut dq = vec![0f64; Q_ROW];
    let mut dk = vec![0f64; KV_ROW];
    let mut dv = vec![0f64; KV_ROW];
    for h in 0..N_Q {
        let kvh = h / group;
        for i in 0..SEQ {
            let p = probabilities_in_f64(q, k, h, i, window);
            let mut dp = vec![0f64; STORED];
            for (j, dpj) in dp.iter_mut().enumerate() {
                let mut acc = 0f64;
                for d in 0..HEAD_DIM {
                    acc += dy[(i * N_Q + h) * HEAD_DIM + d] * v[(j * N_KV + kvh) * HEAD_DIM + d];
                }
                *dpj = acc;
            }
            for (j, pj) in p.iter().enumerate() {
                for d in 0..HEAD_DIM {
                    dv[(j * N_KV + kvh) * HEAD_DIM + d] += pj * dy[(i * N_Q + h) * HEAD_DIM + d];
                }
            }
            let mean: f64 = if softmax_jacobian_subtracts_the_probability_weighted_mean {
                p.iter().zip(dp.iter()).map(|(a, b)| a * b).sum()
            } else {
                0.0
            };
            for j in 0..STORED {
                let dlogit = p[j] * (dp[j] - mean) * scale;
                for d in 0..HEAD_DIM {
                    dq[(i * N_Q + h) * HEAD_DIM + d] += dlogit * k[(j * N_KV + kvh) * HEAD_DIM + d];
                    dk[(j * N_KV + kvh) * HEAD_DIM + d] += dlogit * q[(i * N_Q + h) * HEAD_DIM + d];
                }
            }
        }
    }
    (dq, dk, dv)
}

fn weighted(out: &[f64], dy: &[f64]) -> f64 {
    out.iter().zip(dy.iter()).map(|(a, b)| a * b).sum()
}

fn central_difference_over_every_element(x: &[f64], f: &dyn Fn(&[f64]) -> f64) -> Vec<f64> {
    let h = CENTRAL_DIFFERENCE_STEP_1EM5_LEAVES_F64_TRUNCATION_AND_ROUNDOFF_FIVE_DECADES_UNDER_THAT_BUDGET;
    let mut probe = x.to_vec();
    let mut g = vec![0f64; x.len()];
    for i in 0..x.len() {
        let orig = probe[i];
        probe[i] = orig + h;
        let up = f(&probe);
        probe[i] = orig - h;
        let down = f(&probe);
        probe[i] = orig;
        g[i] = (up - down) / (2.0 * h);
    }
    g
}

fn worst_relative(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "gradient and its oracle disagree on element count, so not every element is probed"
    );
    got.iter().zip(want.iter()).fold(0f64, |acc, (g, w)| {
        assert!(g.is_finite(), "non-finite gradient element");
        acc.max((g - w).abs() / w.abs().max(1.0))
    })
}

fn amax(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |a, x| a.max(x.abs()))
}

fn run(window: Option<usize>) -> Vec<(Vec<f32>, Vec<f64>)> {
    let dev = Device::Cpu;
    let (qh_all, kh_all, vh_all) = fixture();
    let q = Tensor::from_vec(qh_all.clone(), (BATCH, SEQ, N_Q * HEAD_DIM), &dev).unwrap();
    let k = Tensor::from_vec(kh_all.clone(), (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    let v = Tensor::from_vec(vh_all.clone(), (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap();

    let got = naive_sdpa(&q, &k, &v, BATCH, N_Q, N_KV, HEAD_DIM, SEQ, window).unwrap();
    assert_eq!(got.dims(), &[BATCH, SEQ, N_Q, HEAD_DIM]);
    let got_flat: Vec<f32> = got
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    (0..BATCH)
        .map(|bi| {
            let got_row = got_flat[bi * Q_ROW..(bi + 1) * Q_ROW].to_vec();
            let want_row = attention_in_f64(
                &f64s(&qh_all[bi * Q_ROW..(bi + 1) * Q_ROW]),
                &f64s(&kh_all[bi * KV_ROW..(bi + 1) * KV_ROW]),
                &f64s(&vh_all[bi * KV_ROW..(bi + 1) * KV_ROW]),
                window,
            );
            (got_row, want_row)
        })
        .collect()
}

fn candle_gradients(window: Option<usize>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dev = Device::Cpu;
    let (qh_all, kh_all, vh_all) = fixture();
    let qv =
        Var::from_tensor(&Tensor::from_vec(qh_all, (BATCH, SEQ, N_Q * HEAD_DIM), &dev).unwrap())
            .unwrap();
    let kv = Var::from_tensor(
        &Tensor::from_vec(kh_all, (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap(),
    )
    .unwrap();
    let vv = Var::from_tensor(
        &Tensor::from_vec(vh_all, (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap(),
    )
    .unwrap();

    let out = naive_sdpa(
        qv.as_tensor(),
        kv.as_tensor(),
        vv.as_tensor(),
        BATCH,
        N_Q,
        N_KV,
        HEAD_DIM,
        SEQ,
        window,
    )
    .unwrap();
    let dy = Tensor::from_vec(cotangents(), (BATCH, SEQ, N_Q, HEAD_DIM), &dev).unwrap();
    let loss = out.mul(&dy).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let take = |var: &Var, name: &str| -> Vec<f32> {
        let g = grads.get(var.as_tensor()).unwrap_or_else(|| {
            panic!("{name} {NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_FUSED_LEAF_SOFTMAX_PRODUCES}")
        });
        g.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    (take(&qv, "q"), take(&kv, "k"), take(&vv, "v"))
}

fn assert_matches(got: &[f32], want: &[f64], label: &str) {
    let mut worst = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(g.is_finite(), "{label}: non-finite output");
        worst = worst.max((*g as f64 - w).abs() / w.abs().max(1e-3));
    }
    assert!(
        worst <= 1e-5,
        "{label}: the attention dense_train.rs trains through differs from an independent \
         f64 attention by {worst:e} relative. Every adapter is fitted through this op, so a \
         defect here is baked into the weights and never reported."
    );
}

fn assert_gradient_matches_a_central_difference(window: Option<usize>, label: &str) {
    let (gq, gk, gv) = candle_gradients(window);
    let (qh_all, kh_all, vh_all) = fixture();
    let dy_all = cotangents();
    let mut worst = [0f64; 3];
    let mut probed = 0usize;
    for bi in 0..BATCH {
        let q64 = f64s(&qh_all[bi * Q_ROW..(bi + 1) * Q_ROW]);
        let k64 = f64s(&kh_all[bi * KV_ROW..(bi + 1) * KV_ROW]);
        let v64 = f64s(&vh_all[bi * KV_ROW..(bi + 1) * KV_ROW]);
        let dy64 = f64s(&dy_all[bi * Q_ROW..(bi + 1) * Q_ROW]);

        let nq = central_difference_over_every_element(&q64, &|p| {
            weighted(&attention_in_f64(p, &k64, &v64, window), &dy64)
        });
        let nk = central_difference_over_every_element(&k64, &|p| {
            weighted(&attention_in_f64(&q64, p, &v64, window), &dy64)
        });
        let nv = central_difference_over_every_element(&v64, &|p| {
            weighted(&attention_in_f64(&q64, &k64, p, window), &dy64)
        });
        probed += nq.len() + nk.len() + nv.len();

        worst[0] = worst[0].max(worst_relative(
            &f64s(&gq[bi * Q_ROW..(bi + 1) * Q_ROW]),
            &nq,
        ));
        worst[1] = worst[1].max(worst_relative(
            &f64s(&gk[bi * KV_ROW..(bi + 1) * KV_ROW]),
            &nk,
        ));
        worst[2] = worst[2].max(worst_relative(
            &f64s(&gv[bi * KV_ROW..(bi + 1) * KV_ROW]),
            &nv,
        ));
    }
    assert_eq!(
        probed,
        BATCH * (Q_ROW + 2 * KV_ROW),
        "{label}: not every element of q, k and v was probed, so the row is a spot check"
    );
    let budget = GRADIENT_BUDGET_IS_512_F32_MANTISSA_EPS_BECAUSE_NAIVE_SDPA_CASTS_EVERY_INPUT_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_WHATEVER_DTYPE_THE_VARS_CARRY;
    eprintln!(
        "{label}: worst relative gap vs f64 central difference over {probed} probed elements -- \
         q {:e}  k {:e}  v {:e}  budget {:e}",
        worst[0], worst[1], worst[2], budget
    );
    for (i, name) in ["q", "k", "v"].iter().enumerate() {
        assert!(
            worst[i] <= budget,
            "{label}: the gradient dense_train::naive_sdpa hands {name} differs from an \
             independent f64 central difference of an independent f64 attention by {:e} \
             relative, budget {:e}. Every LoRA factor on this projection is fitted through \
             exactly this gradient.",
            worst[i],
            budget
        );
    }
}

#[test]
fn the_attention_the_trainer_trains_through_matches_an_independent_f64_attention() {
    for (bi, (got, want)) in run(None).iter().enumerate() {
        assert_matches(got, want, &format!("causal row {bi}"));
    }
}

#[test]
fn the_sliding_window_the_trainer_trains_through_matches_the_same_oracle() {
    for (bi, (got, want)) in run(Some(2)).iter().enumerate() {
        assert_matches(got, want, &format!("window=2 row {bi}"));
    }
}

#[test]
fn the_window_and_the_gqa_mapping_both_change_the_answer_which_is_what_makes_the_rows_above_gates()
{
    let open_rows = run(None);
    let windowed_rows = run(Some(2));
    for bi in 0..BATCH {
        let open = &open_rows[bi].0;
        let windowed = &windowed_rows[bi].0;
        assert!(
            open.iter().zip(windowed.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "row {bi}: the window changes nothing on this fixture, so the windowed row proves nothing"
        );

        let head0 = &open[..HEAD_DIM];
        let head_last = &open[(N_Q - 1) * HEAD_DIM..N_Q * HEAD_DIM];
        assert!(
            head0
                .iter()
                .zip(head_last.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "row {bi}: all query heads produced the same output, so the GQA mapping is untested"
        );
    }
    assert_eq!(
        STORED - SEQ,
        2,
        "key_base must be nonzero or the offset goes untested"
    );

    let (row0, _) = &open_rows[0];
    let (row1, _) = &open_rows[1];
    assert!(
        row0.iter()
            .zip(row1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "both batch rows produced the same output on different inputs, so the batch \
         dimension is copying row 0 instead of attending each row to its own keys"
    );
}

#[test]
fn a_key_run_longer_than_the_query_run_still_lets_every_query_see_its_own_position() {
    assert!(
        STORED > SEQ,
        "this row is about the stored > seq case and needs one"
    );
    for (bi, (got, want)) in run(None).iter().enumerate() {
        assert_matches(got, want, &format!("stored > seq row {bi}"));
        assert!(
            got.iter().all(|g| g.is_finite()),
            "row {bi}: a query row attended to nothing, so its softmax ran over an all -inf row"
        );
    }
}

#[test]
fn serving_still_scores_through_the_fused_kernel_because_no_decoder_ever_hands_naive_sdpa_a_var() {
    let dev = Device::Cpu;
    let (qh_all, kh_all, vh_all) = fixture();
    let q = Tensor::from_vec(qh_all.clone(), (BATCH, SEQ, N_Q * HEAD_DIM), &dev).unwrap();
    let k = Tensor::from_vec(kh_all.clone(), (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    let v = Tensor::from_vec(vh_all.clone(), (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    assert!(
        !q.track_op() && !k.track_op() && !v.track_op(),
        "the serving arm of this row must start from tensors candle would refuse to record an \
         op for, or it is measuring the training branch against itself"
    );
    let served = naive_sdpa(&q, &k, &v, BATCH, N_Q, N_KV, HEAD_DIM, SEQ, None).unwrap();
    assert!(
        !served.track_op(),
        "naive_sdpa left an op on its output for an input carrying no Var, so the composed \
         tracked softmax ran on a serving-shaped call and the fused \
         candle_nn::ops::softmax_last_dim kernel every decoder relies on was bypassed"
    );

    let qv = Var::from_tensor(&q).unwrap();
    let trained = naive_sdpa(
        qv.as_tensor(),
        &k,
        &v,
        BATCH,
        N_Q,
        N_KV,
        HEAD_DIM,
        SEQ,
        None,
    )
    .unwrap();
    assert!(
        trained.track_op(),
        "naive_sdpa produced a graph leaf for an input that IS a Var, which is the whole defect"
    );

    let a: Vec<f32> = served.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = trained.flatten_all().unwrap().to_vec1().unwrap();
    let mut worst = 0f64;
    let mut differing_bits = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            differing_bits += 1;
        }
        worst = worst.max((*x as f64 - *y as f64).abs() / (*x as f64).abs().max(1.0));
    }
    eprintln!(
        "train/serve skew at [{BATCH},{SEQ},{N_Q},{HEAD_DIM}] -- worst relative {worst:e}, \
         {differing_bits} of {} elements differ in any bit",
        a.len()
    );
    assert!(
        worst <= 16.0 * F32_MANTISSA_EPS,
        "the composed softmax the trainer takes and the fused kernel serving takes disagree by \
         {worst:e}, so the trainer is fitting a different function from the one the engine runs"
    );
}

#[test]
fn the_trainer_reaches_q_and_k_with_a_gradient_that_matches_an_f64_central_difference() {
    assert_gradient_matches_a_central_difference(None, "causal");
}

#[test]
fn the_sliding_window_the_trainer_trains_through_reaches_q_and_k_with_the_same_gradient() {
    assert_gradient_matches_a_central_difference(Some(2), "window=2");
}

#[test]
fn the_central_difference_gate_rejects_the_softmax_jacobian_that_forgets_to_subtract_the_probability_weighted_mean(
) {
    let (qh_all, kh_all, vh_all) = fixture();
    let dy_all = cotangents();
    let budget = GRADIENT_BUDGET_IS_512_F32_MANTISSA_EPS_BECAUSE_NAIVE_SDPA_CASTS_EVERY_INPUT_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_WHATEVER_DTYPE_THE_VARS_CARRY;
    let mut right = 0f64;
    let mut wrong = 0f64;
    for bi in 0..BATCH {
        let q64 = f64s(&qh_all[bi * Q_ROW..(bi + 1) * Q_ROW]);
        let k64 = f64s(&kh_all[bi * KV_ROW..(bi + 1) * KV_ROW]);
        let v64 = f64s(&vh_all[bi * KV_ROW..(bi + 1) * KV_ROW]);
        let dy64 = f64s(&dy_all[bi * Q_ROW..(bi + 1) * Q_ROW]);
        let nq = central_difference_over_every_element(&q64, &|p| {
            weighted(&attention_in_f64(p, &k64, &v64, None), &dy64)
        });
        let (dq_ok, _, _) = attention_backward_in_f64(&q64, &k64, &v64, &dy64, None, true);
        let (dq_bad, _, _) = attention_backward_in_f64(&q64, &k64, &v64, &dy64, None, false);
        right = right.max(worst_relative(&dq_ok, &nq));
        wrong = wrong.max(worst_relative(&dq_bad, &nq));
    }
    eprintln!(
        "closed-form softmax Jacobian vs the same central difference -- \
         p*(dy - sum(p*dy)) {right:e}   p*dy {wrong:e}   budget {budget:e}"
    );
    assert!(
        right <= budget,
        "the closed form p * (dy - sum(p * dy)) misses the central difference by {right:e}, so \
         the central difference itself is wrong and no row built on it is a gate"
    );
    assert!(
        wrong > A_WRONG_JACOBIAN_MUST_MISS_THE_CENTRAL_DIFFERENCE_BY_AT_LEAST_THIS_OR_THE_GATE_IS_VACUOUS,
        "the Jacobian that forgets to subtract the probability-weighted mean lands within \
         {wrong:e} of the central difference on this fixture, so the gradient rows above would \
         pass against a wrong backward and are not gates"
    );
}

#[test]
fn no_gradient_the_rows_above_pin_is_zero_uniform_across_heads_or_blind_to_the_window() {
    let (gq_open, gk_open, gv_open) = candle_gradients(None);
    let (gq_win, gk_win, _) = candle_gradients(Some(2));
    for (name, g) in [("q", &gq_open), ("k", &gk_open), ("v", &gv_open)] {
        assert!(
            amax(g) > 1e-3,
            "the {name} gradient is numerically indistinguishable from zero ({:e}), so the \
             central-difference rows above would pass against a graph leaf that returns zeros",
            amax(g)
        );
    }
    assert!(
        gq_open
            .iter()
            .zip(gq_win.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4)
            && gk_open
                .iter()
                .zip(gk_win.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
        "the sliding window changes neither the q nor the k gradient, so the windowed \
         gradient row repeats the causal one"
    );
    let head0 = &gq_open[..HEAD_DIM];
    let head_last = &gq_open[(N_Q - 1) * HEAD_DIM..N_Q * HEAD_DIM];
    assert!(
        head0
            .iter()
            .zip(head_last.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "every query head received the same q gradient, so the GQA expansion is untested on \
         the backward"
    );
    assert!(
        N_KV < N_Q && STORED > SEQ,
        "the gradient rows need a real GQA fan-out and a nonzero key offset to be about the \
         shape dense_train actually trains"
    );
}
