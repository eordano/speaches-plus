use candle_core::{DType, Device, Tensor, Var};
use nv_models::gemma4_moe::{
    kv_that_keeps_this_steps_own_keys_and_values_on_the_graph, naive_sdpa,
};

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

const GEMMA4_MOE_NAIVE_SDPA_MULTIPLIES_ITS_LOGITS_BY_NO_EXPLICIT_ATTENTION_SCALE_UNLIKE_DENSE_TRAIN_SO_AN_ORACLE_THAT_DIVIDED_BY_SQRT_HEAD_DIM_WOULD_PIN_A_DIFFERENT_FUNCTION: f64 =
    1.0;

const NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_FUSED_LEAF_SOFTMAX_PRODUCES: &str =
    "received no gradient entry at all from gemma4_moe::naive_sdpa. That is the fused \
     candle_nn::ops::softmax_last_dim leaf: candle installs it with apply_op1_no_bwd, so the \
     probabilities carry BackpropOp::none and q and k are unreachable from the loss while v \
     still trains through the probs @ v matmul that follows. train_runner emits \
     self_attn.q_proj and self_attn.k_proj as DEFAULT LoRA targets, their B factor is \
     zero-initialised, and candle's AdamW::step silently skips a Var with no GradStore entry -- \
     so B stays exactly zero, those modules merge as an exact no-op, and the loss falls the \
     whole time with nothing reporting it. This is the Gemma4 MoE LoRA trainer reached by \
     Gemma4Moe::forward_with_cache from train_runner::run and from the /v1/fine_tuning HTTP \
     endpoint. naive_sdpa must score through \
     nv_layers::attn::softmax_last_dim_that_keeps_q_and_k_on_the_graph, which keeps the identical \
     fused call for serving and takes the composed tracked softmax only while an autograd graph \
     is being built.";

#[derive(Clone, Copy)]
struct Geometry {
    write_start: usize,
    new_total: usize,
    window: Option<usize>,
    label: &'static str,
}

const PREFILL_CAUSAL_KEY_BASE_ZERO: Geometry = Geometry {
    write_start: 2,
    new_total: STORED,
    window: None,
    label: "causal write_start=2 key_base=0",
};

const ROLLED_SLIDING_CACHE_KEY_BASE_TWO: Geometry = Geometry {
    write_start: 4,
    new_total: 7,
    window: Some(3),
    label: "window=3 write_start=4 key_base=2",
};

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
    (seeded(Q_ROW, 11), seeded(KV_ROW, 22), seeded(KV_ROW, 33))
}

fn cotangents() -> Vec<f32> {
    seeded(Q_ROW, 987_654_321)
}

fn key_base_of(g: Geometry) -> usize {
    g.new_total - STORED
}

fn visible(i: usize, j: usize, g: Geometry) -> bool {
    let qpos = (g.write_start + i) as i64;
    let kpos = (key_base_of(g) + j) as i64;
    kpos <= qpos
        && match g.window {
            Some(w) => qpos - kpos < w as i64,
            None => true,
        }
}

fn probabilities_in_f64(q: &[f64], k: &[f64], h: usize, i: usize, g: Geometry) -> Vec<f64> {
    let kvh = h / (N_Q / N_KV);
    let scale = GEMMA4_MOE_NAIVE_SDPA_MULTIPLIES_ITS_LOGITS_BY_NO_EXPLICIT_ATTENTION_SCALE_UNLIKE_DENSE_TRAIN_SO_AN_ORACLE_THAT_DIVIDED_BY_SQRT_HEAD_DIM_WOULD_PIN_A_DIFFERENT_FUNCTION;
    let mut logits = vec![f64::NEG_INFINITY; STORED];
    for (j, logit) in logits.iter_mut().enumerate() {
        if !visible(i, j, g) {
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
        "{}: row {i} head {h} attends to nothing, so this row is vacuous",
        g.label
    );
    exps.iter().map(|e| e / denom).collect()
}

fn attention_in_f64(q: &[f64], k: &[f64], v: &[f64], g: Geometry) -> Vec<f64> {
    let group = N_Q / N_KV;
    let mut out = vec![0f64; Q_ROW];
    for h in 0..N_Q {
        let kvh = h / group;
        for i in 0..SEQ {
            let p = probabilities_in_f64(q, k, h, i, g);
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
    g: Geometry,
    softmax_jacobian_subtracts_the_probability_weighted_mean: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let group = N_Q / N_KV;
    let scale = GEMMA4_MOE_NAIVE_SDPA_MULTIPLIES_ITS_LOGITS_BY_NO_EXPLICIT_ATTENTION_SCALE_UNLIKE_DENSE_TRAIN_SO_AN_ORACLE_THAT_DIVIDED_BY_SQRT_HEAD_DIM_WOULD_PIN_A_DIFFERENT_FUNCTION;
    let mut dq = vec![0f64; Q_ROW];
    let mut dk = vec![0f64; KV_ROW];
    let mut dv = vec![0f64; KV_ROW];
    for h in 0..N_Q {
        let kvh = h / group;
        for i in 0..SEQ {
            let p = probabilities_in_f64(q, k, h, i, g);
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

fn the_prefill_bias_arm_is_off_so_the_oracle_scores_the_same_function(g: Geometry) {
    assert!(
        nv_models::gemma4::xattn_cfg().is_none(),
        "{}: NV_XATTN_PREFILL is engaged, so naive_sdpa adds a sparsity bias to its full-attention \
         prefill scores that this oracle does not model, and every row here would be comparing \
         two different functions",
        g.label
    );
}

fn served_output(g: Geometry) -> Vec<f32> {
    the_prefill_bias_arm_is_off_so_the_oracle_scores_the_same_function(g);
    let dev = Device::Cpu;
    let (qh, kh, vh) = fixture();
    let q = Tensor::from_vec(qh, (1usize, SEQ, N_Q * HEAD_DIM), &dev).unwrap();
    let k = Tensor::from_vec(kh, (1usize, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    let v = Tensor::from_vec(vh, (1usize, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    let got = naive_sdpa(
        &q,
        &k,
        &v,
        N_Q,
        N_KV,
        HEAD_DIM,
        SEQ,
        g.write_start,
        g.new_total,
        g.window,
    )
    .unwrap();
    assert_eq!(
        got.dims(),
        &[SEQ, N_Q, HEAD_DIM],
        "{}: naive_sdpa changed its output layout, so every index in this oracle is wrong",
        g.label
    );
    got.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn candle_gradients(g: Geometry) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    the_prefill_bias_arm_is_off_so_the_oracle_scores_the_same_function(g);
    let dev = Device::Cpu;
    let (qh, kh, vh) = fixture();
    let qv = Var::from_tensor(&Tensor::from_vec(qh, (1usize, SEQ, N_Q * HEAD_DIM), &dev).unwrap())
        .unwrap();
    let kv =
        Var::from_tensor(&Tensor::from_vec(kh, (1usize, STORED, N_KV * HEAD_DIM), &dev).unwrap())
            .unwrap();
    let vv =
        Var::from_tensor(&Tensor::from_vec(vh, (1usize, STORED, N_KV * HEAD_DIM), &dev).unwrap())
            .unwrap();

    let out = naive_sdpa(
        qv.as_tensor(),
        kv.as_tensor(),
        vv.as_tensor(),
        N_Q,
        N_KV,
        HEAD_DIM,
        SEQ,
        g.write_start,
        g.new_total,
        g.window,
    )
    .unwrap();
    let dy = Tensor::from_vec(cotangents(), (SEQ, N_Q, HEAD_DIM), &dev).unwrap();
    let loss = out.mul(&dy).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let take = |var: &Var, name: &str| -> Vec<f32> {
        let gr = grads.get(var.as_tensor()).unwrap_or_else(|| {
            panic!(
                "{}: {name} {NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_FUSED_LEAF_SOFTMAX_PRODUCES}",
                g.label
            )
        });
        gr.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    (take(&qv, "q"), take(&kv, "k"), take(&vv, "v"))
}

fn assert_forward_matches_an_independent_f64_attention(g: Geometry) {
    let got = served_output(g);
    let (qh, kh, vh) = fixture();
    let want = attention_in_f64(&f64s(&qh), &f64s(&kh), &f64s(&vh), g);
    let mut worst = 0f64;
    for (x, w) in got.iter().zip(want.iter()) {
        assert!(x.is_finite(), "{}: non-finite output", g.label);
        worst = worst.max((*x as f64 - w).abs() / w.abs().max(1e-3));
    }
    eprintln!(
        "{}: forward vs independent f64 attention {worst:e}",
        g.label
    );
    assert!(
        worst <= 1e-5,
        "{}: the attention the MoE trainer trains through differs from an independent f64 \
         attention by {worst:e} relative. Every adapter fitted through Gemma4Moe is fitted \
         through this op, so a defect here is baked into the weights and never reported.",
        g.label
    );
}

fn assert_gradient_matches_a_central_difference(g: Geometry) {
    let (gq, gk, gv) = candle_gradients(g);
    let (qh, kh, vh) = fixture();
    let q64 = f64s(&qh);
    let k64 = f64s(&kh);
    let v64 = f64s(&vh);
    let dy64 = f64s(&cotangents());

    let nq = central_difference_over_every_element(&q64, &|p| {
        weighted(&attention_in_f64(p, &k64, &v64, g), &dy64)
    });
    let nk = central_difference_over_every_element(&k64, &|p| {
        weighted(&attention_in_f64(&q64, p, &v64, g), &dy64)
    });
    let nv = central_difference_over_every_element(&v64, &|p| {
        weighted(&attention_in_f64(&q64, &k64, p, g), &dy64)
    });
    let probed = nq.len() + nk.len() + nv.len();
    assert_eq!(
        probed,
        Q_ROW + 2 * KV_ROW,
        "{}: not every element of q, k and v was probed, so the row is a spot check",
        g.label
    );

    let worst = [
        worst_relative(&f64s(&gq), &nq),
        worst_relative(&f64s(&gk), &nk),
        worst_relative(&f64s(&gv), &nv),
    ];
    let budget = GRADIENT_BUDGET_IS_512_F32_MANTISSA_EPS_BECAUSE_NAIVE_SDPA_CASTS_EVERY_INPUT_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_WHATEVER_DTYPE_THE_VARS_CARRY;
    eprintln!(
        "{}: worst relative gap vs f64 central difference over {probed} probed elements -- \
         q {:e}  k {:e}  v {:e}  budget {:e}",
        g.label, worst[0], worst[1], worst[2], budget
    );
    for (i, name) in ["q", "k", "v"].iter().enumerate() {
        assert!(
            worst[i] <= budget,
            "{}: the gradient gemma4_moe::naive_sdpa hands {name} differs from an independent \
             f64 central difference of an independent f64 attention by {:e} relative, budget \
             {:e}. Every LoRA factor on this projection is fitted through exactly this gradient.",
            g.label,
            worst[i],
            budget
        );
    }
}

fn assert_the_central_difference_rejects_the_jacobian_without_the_mean(g: Geometry) {
    let (qh, kh, vh) = fixture();
    let q64 = f64s(&qh);
    let k64 = f64s(&kh);
    let v64 = f64s(&vh);
    let dy64 = f64s(&cotangents());
    let budget = GRADIENT_BUDGET_IS_512_F32_MANTISSA_EPS_BECAUSE_NAIVE_SDPA_CASTS_EVERY_INPUT_TO_F32_SO_ITS_ANALYTIC_GRADIENT_IS_AN_F32_QUANTITY_WHATEVER_DTYPE_THE_VARS_CARRY;

    let nq = central_difference_over_every_element(&q64, &|p| {
        weighted(&attention_in_f64(p, &k64, &v64, g), &dy64)
    });
    let (dq_ok, _, _) = attention_backward_in_f64(&q64, &k64, &v64, &dy64, g, true);
    let (dq_bad, _, _) = attention_backward_in_f64(&q64, &k64, &v64, &dy64, g, false);
    let right = worst_relative(&dq_ok, &nq);
    let wrong = worst_relative(&dq_bad, &nq);
    eprintln!(
        "{}: closed-form softmax Jacobian vs the same central difference -- \
         p*(dy - sum(p*dy)) {right:e}   p*dy {wrong:e}   budget {budget:e}",
        g.label
    );
    assert!(
        right <= budget,
        "{}: the closed form p * (dy - sum(p * dy)) misses the central difference by {right:e}, \
         so the central difference itself is wrong and no row built on it is a gate",
        g.label
    );
    assert!(
        wrong > A_WRONG_JACOBIAN_MUST_MISS_THE_CENTRAL_DIFFERENCE_BY_AT_LEAST_THIS_OR_THE_GATE_IS_VACUOUS,
        "{}: the Jacobian that forgets to subtract the probability-weighted mean lands within \
         {wrong:e} of the central difference on this fixture, so the gradient rows above would \
         pass against a wrong backward and are not gates",
        g.label
    );
}

#[test]
fn the_attention_the_moe_trainer_trains_through_matches_an_independent_f64_attention() {
    assert_forward_matches_an_independent_f64_attention(PREFILL_CAUSAL_KEY_BASE_ZERO);
}

#[test]
fn the_rolled_sliding_cache_the_moe_trainer_trains_through_matches_the_same_oracle() {
    assert_forward_matches_an_independent_f64_attention(ROLLED_SLIDING_CACHE_KEY_BASE_TWO);
}

#[test]
fn the_moe_trainer_reaches_q_and_k_with_a_gradient_that_matches_an_f64_central_difference() {
    assert_gradient_matches_a_central_difference(PREFILL_CAUSAL_KEY_BASE_ZERO);
}

#[test]
fn the_rolled_sliding_cache_reaches_q_and_k_with_the_same_gradient() {
    assert_gradient_matches_a_central_difference(ROLLED_SLIDING_CACHE_KEY_BASE_TWO);
}

#[test]
fn the_central_difference_gate_rejects_the_softmax_jacobian_that_forgets_to_subtract_the_probability_weighted_mean(
) {
    assert_the_central_difference_rejects_the_jacobian_without_the_mean(
        PREFILL_CAUSAL_KEY_BASE_ZERO,
    );
}

#[test]
fn serving_still_scores_through_the_fused_kernel_because_no_decoder_ever_hands_naive_sdpa_a_var() {
    let g = PREFILL_CAUSAL_KEY_BASE_ZERO;
    the_prefill_bias_arm_is_off_so_the_oracle_scores_the_same_function(g);
    let dev = Device::Cpu;
    let (qh, kh, vh) = fixture();
    let q = Tensor::from_vec(qh, (1usize, SEQ, N_Q * HEAD_DIM), &dev).unwrap();
    let k = Tensor::from_vec(kh, (1usize, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    let v = Tensor::from_vec(vh, (1usize, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    assert!(
        !q.track_op() && !k.track_op() && !v.track_op(),
        "the serving arm of this row must start from tensors candle would refuse to record an \
         op for, or it is measuring the training branch against itself"
    );
    let served = naive_sdpa(
        &q,
        &k,
        &v,
        N_Q,
        N_KV,
        HEAD_DIM,
        SEQ,
        g.write_start,
        g.new_total,
        g.window,
    )
    .unwrap();
    assert!(
        !served.track_op(),
        "gemma4_moe::naive_sdpa left an op on its output for an input carrying no Var, so the \
         composed tracked softmax ran on a serving-shaped call and the fused \
         candle_nn::ops::softmax_last_dim kernel every Gemma4Moe decode relies on was bypassed"
    );

    let qv = Var::from_tensor(&q).unwrap();
    let trained = naive_sdpa(
        qv.as_tensor(),
        &k,
        &v,
        N_Q,
        N_KV,
        HEAD_DIM,
        SEQ,
        g.write_start,
        g.new_total,
        g.window,
    )
    .unwrap();
    assert!(
        trained.track_op(),
        "gemma4_moe::naive_sdpa produced a graph leaf for an input that IS a Var, which is the \
         whole defect"
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
        "train/serve skew at [{SEQ},{N_Q},{HEAD_DIM}] -- worst relative {worst:e}, \
         {differing_bits} of {} elements differ in any bit",
        a.len()
    );
    assert!(
        worst <= 16.0 * F32_MANTISSA_EPS,
        "the composed softmax the MoE trainer takes and the fused kernel Gemma4Moe serving takes \
         disagree by {worst:e}, so the trainer is fitting a different function from the one the \
         engine runs"
    );
}

#[test]
fn no_gradient_the_rows_above_pin_is_zero_uniform_across_heads_or_blind_to_the_window() {
    let (gq_open, gk_open, gv_open) = candle_gradients(PREFILL_CAUSAL_KEY_BASE_ZERO);
    let (gq_win, gk_win, _) = candle_gradients(ROLLED_SLIDING_CACHE_KEY_BASE_TWO);
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
        "the sliding window and the rolled key base change neither the q nor the k gradient, so \
         the second gradient row repeats the first"
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
        N_KV < N_Q
            && STORED > SEQ
            && PREFILL_CAUSAL_KEY_BASE_ZERO.write_start > 0
            && key_base_of(ROLLED_SLIDING_CACHE_KEY_BASE_TWO) > 0,
        "the gradient rows need a real GQA fan-out, a stored run longer than the query run, a \
         nonzero write_start and a nonzero key_base to be about the shape gemma4_moe actually \
         trains and decodes"
    );
}

const KV_N_KV: usize = 2;
const KV_HEAD_DIM: usize = 4;
const KV_PREFIX_OF_AN_INCREMENTAL_DECODE_STEP: usize = 2;

const NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_IN_PLACE_KV_WRITE_PRODUCES: &str =
    "received no gradient entry at all from the KV cache round trip. Gemma4KvCache::write_at \
     stores k and v with Tensor::slice_set, an in-place copy into a buffer new_kv_cache \
     allocated once, so cache.view hands back Narrow(buffer) -- a graph leaf with no link to \
     k_rot. k and v are reachable from the loss only through that write, so self_attn.k_proj and \
     self_attn.v_proj keep their zero-init B forever and merge as an exact no-op while the loss \
     falls. gemma4_moe::attention_forward must splice this step's own live k and v back over the \
     tail of the view through \
     kv_that_keeps_this_steps_own_keys_and_values_on_the_graph.";

fn kv_hosts(prefix: usize) -> (Vec<f32>, Vec<f32>) {
    let per_step = SEQ * KV_N_KV * KV_HEAD_DIM;
    let older = prefix * KV_N_KV * KV_HEAD_DIM;
    let fresh = seeded(per_step, 4242);
    let mut view = seeded(older, 1717);
    view.extend_from_slice(&fresh);
    (view, fresh)
}

fn assert_the_splice_is_bit_identical_and_keeps_the_fresh_kv_on_the_graph(prefix: usize) {
    let dev = Device::Cpu;
    let (view_host, fresh_host) = kv_hosts(prefix);
    let stored = prefix + SEQ;
    let k_view = Tensor::from_vec(
        view_host.clone(),
        (1usize, stored, KV_N_KV, KV_HEAD_DIM),
        &dev,
    )
    .unwrap();
    let fresh = Tensor::from_vec(fresh_host, (1usize, SEQ, KV_N_KV, KV_HEAD_DIM), &dev).unwrap();
    let kvar = Var::from_tensor(&fresh).unwrap();
    let k_new = kvar.as_tensor().clone();

    let (k_full, v_full) = kv_that_keeps_this_steps_own_keys_and_values_on_the_graph(
        &k_view, &k_view, &k_new, &k_new, SEQ,
    )
    .unwrap();
    assert!(
        k_full.track_op() && v_full.track_op(),
        "prefix {prefix}: the spliced cache view is still a graph leaf, so k and v \
         {NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_IN_PLACE_KV_WRITE_PRODUCES}"
    );
    assert_eq!(
        k_full.dims(),
        &[1, stored, KV_N_KV, KV_HEAD_DIM],
        "prefix {prefix}: the splice changed the shape naive_sdpa attends over"
    );

    let got: Vec<f32> = k_full.flatten_all().unwrap().to_vec1().unwrap();
    let differing = got
        .iter()
        .zip(view_host.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "prefix {prefix}: {differing} of {} elements of the spliced key run differ in some bit \
         from the cache view the decoder reads, so training and serving attend over different \
         keys",
        got.len()
    );

    let w = seeded(stored * KV_N_KV * KV_HEAD_DIM, 909);
    let dy = Tensor::from_vec(w.clone(), (1usize, stored, KV_N_KV, KV_HEAD_DIM), &dev).unwrap();
    let loss = k_full.mul(&dy).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let g: Vec<f32> = grads
        .get(kvar.as_tensor())
        .unwrap_or_else(|| {
            panic!("prefix {prefix}: k {NO_GRADIENT_ENTRY_AT_ALL_IS_WHAT_THE_IN_PLACE_KV_WRITE_PRODUCES}")
        })
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let tail = &w[prefix * KV_N_KV * KV_HEAD_DIM..];
    assert_eq!(
        g.len(),
        tail.len(),
        "prefix {prefix}: the gradient does not cover this step's own keys"
    );
    let worst = g
        .iter()
        .zip(tail.iter())
        .fold(0f64, |acc, (a, b)| acc.max((*a as f64 - *b as f64).abs()));
    assert!(
        worst <= 1e-6,
        "prefix {prefix}: d(sum(w * spliced)) / d(this step's k) differs from w over this step's \
         own rows by {worst:e}; the splice is routing the gradient to the wrong positions"
    );
}

#[test]
fn this_steps_own_keys_and_values_come_back_on_the_graph_when_a_gradient_is_being_built() {
    assert_the_splice_is_bit_identical_and_keeps_the_fresh_kv_on_the_graph(0);
}

#[test]
fn an_incremental_decode_step_splices_its_own_kv_over_a_detached_prefix() {
    assert_the_splice_is_bit_identical_and_keeps_the_fresh_kv_on_the_graph(
        KV_PREFIX_OF_AN_INCREMENTAL_DECODE_STEP,
    );
}

#[test]
fn serving_reads_the_kv_cache_view_unchanged_because_no_decoder_ever_hands_the_cache_a_var() {
    let dev = Device::Cpu;
    let prefix = KV_PREFIX_OF_AN_INCREMENTAL_DECODE_STEP;
    let (view_host, fresh_host) = kv_hosts(prefix);
    let stored = prefix + SEQ;
    let k_view = Tensor::from_vec(
        view_host.clone(),
        (1usize, stored, KV_N_KV, KV_HEAD_DIM),
        &dev,
    )
    .unwrap();
    let k_new = Tensor::from_vec(fresh_host, (1usize, SEQ, KV_N_KV, KV_HEAD_DIM), &dev).unwrap();
    assert!(
        !k_view.track_op() && !k_new.track_op(),
        "the serving arm of this row must start from tensors candle would refuse to record an \
         op for, or it is measuring the training branch against itself"
    );

    let (k_full, v_full) = kv_that_keeps_this_steps_own_keys_and_values_on_the_graph(
        &k_view, &k_view, &k_new, &k_new, SEQ,
    )
    .unwrap();
    assert!(
        !k_full.track_op() && !v_full.track_op(),
        "the KV splice left an op on the cache view for an input carrying no Var, so a serving \
         decode is now rebuilding its key run through a cat it never used to take"
    );
    assert_eq!(
        k_full.dims(),
        k_view.dims(),
        "serving got a different key-run shape out of the cache than the view it read"
    );
    let got: Vec<f32> = k_full.flatten_all().unwrap().to_vec1().unwrap();
    let differing = got
        .iter()
        .zip(view_host.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "serving no longer reads the cache view byte for byte: {differing} of {} elements differ",
        got.len()
    );
}
