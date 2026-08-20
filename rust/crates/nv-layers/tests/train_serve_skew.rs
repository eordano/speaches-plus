#![cfg(feature = "cuda")]

mod common;
use common::cuda;
use candle_core::{DType, Device, Tensor};
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};

const HIDDEN: usize = 256;
const ROWS: usize = 8;
const HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const TOKENS: usize = 6;

const F32_MANTISSA_EPS: f32 = 1.19e-7;

fn roundoff_budget_for_a_reduction_of_width(n: usize) -> f32 {
    16.0 * F32_MANTISSA_EPS * (n as f32).sqrt()
}

fn worst_relative_gap(a: &Tensor, b: &Tensor) -> f32 {
    let av: Vec<f32> = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    let bv: Vec<f32> = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    assert_eq!(av.len(), bv.len(), "the two forwards disagree on output SHAPE, not just value");
    let mut worst = 0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        assert!(x.is_finite() && y.is_finite(), "non-finite output: {x} vs {y}");
        worst = worst.max((x - y).abs() / x.abs().max(y.abs()).max(1e-3));
    }
    worst
}

#[test]
fn the_norm_the_trainer_runs_matches_the_norm_serving_runs() {
    let Some(dev) = cuda() else { return };
    let w = Tensor::rand(0.5f32, 1.5f32, HIDDEN, &dev).unwrap();
    let norm = RmsNorm::new(w, 1e-6);
    let budget = roundoff_budget_for_a_reduction_of_width(HIDDEN);

    for dtype in [DType::F32, DType::BF16] {
        let x = Tensor::rand(-2f32, 2f32, (ROWS, HIDDEN), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        assert!(
            !x.track_op(),
            "an input on an autograd graph diverts the CUDA path to the candle one, which \
             would compare forward_candle against itself and pass unconditionally"
        );
        let serving = norm.forward(&x).unwrap();
        let training = norm.forward_candle(&x).unwrap();
        let gap = worst_relative_gap(&serving, &training);
        let allowed = if dtype == DType::BF16 { 8e-3 } else { budget };
        assert!(
            gap <= allowed,
            "{dtype:?}: RmsNorm::forward (what serving runs) and forward_candle (what \
             dense_train.rs runs for every norm in the model) differ by {gap:e} relative, \
             over a budget of {allowed:e}. An adapter is then trained against a different \
             model than the one it is served on."
        );
    }
}

#[test]
fn the_rope_the_trainer_runs_matches_the_rope_serving_runs() {
    let Some(dev) = cuda() else { return };
    let rope = Rope::new(
        RopeConfig { head_dim: HEAD_DIM, max_seq_len: 128, base: 10000.0, kind: RopeKind::Standard },
        &dev,
    )
    .unwrap();
    let pos = Tensor::from_vec((0..TOKENS as u32).collect::<Vec<_>>(), TOKENS, &dev).unwrap();
    let budget = roundoff_budget_for_a_reduction_of_width(HEAD_DIM);

    for dtype in [DType::F32, DType::BF16] {
        let shape = (TOKENS, HEADS, HEAD_DIM);
        let q = Tensor::rand(-2f32, 2f32, shape, &dev).unwrap().to_dtype(dtype).unwrap();
        let k = Tensor::rand(-2f32, 2f32, shape, &dev).unwrap().to_dtype(dtype).unwrap();
        assert!(!q.track_op(), "a tracked input diverts apply() to apply_candle, see above");

        let (q_serve, k_serve) = rope.apply(&q, &k, &pos).unwrap();
        let (q_train, k_train) = rope.apply_candle(&q, &k, &pos).unwrap();
        let allowed = if dtype == DType::BF16 { 8e-3 } else { budget };
        for (label, s, t) in [("q", &q_serve, &q_train), ("k", &k_serve, &k_train)] {
            let gap = worst_relative_gap(s, t);
            assert!(
                gap <= allowed,
                "{dtype:?} {label}: Rope::apply (what serving runs) and apply_candle (what \
                 dense_train.rs calls directly) differ by {gap:e} relative, over a budget of \
                 {allowed:e}. Attention is then trained on different rotations than it is \
                 served with."
            );
        }
    }
}
