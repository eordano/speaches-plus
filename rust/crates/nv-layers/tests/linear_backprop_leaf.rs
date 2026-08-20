#![cfg(feature = "cuda")]

mod common;
use common::cuda;
use candle_core::{DType, Device, Tensor, Var};
use nv_layers::linear::Linear;

const IN: usize = 64;
const OUT: usize = 32;
const ROWS: usize = 8;

fn the_env_var_is_process_global_so_these_rows_must_not_overlap(
) -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn bf16_linear(dev: &Device) -> Linear {
    let w = Tensor::rand(-0.05f32, 0.05f32, (OUT, IN), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    Linear::new(w, None).unwrap()
}

fn tracked_bf16_input(dev: &Device) -> (Var, Tensor) {
    let v = Var::from_tensor(&Tensor::rand(-1f32, 1f32, (ROWS, IN), dev).unwrap()).unwrap();
    let x = v.as_tensor().to_dtype(DType::BF16).unwrap();
    assert!(
        x.track_op(),
        "the input must be on an autograd graph or this suite proves nothing about backprop"
    );
    (v, x)
}

#[test]
fn a_bf16_activation_is_what_arms_the_leaf_and_f32_is_why_training_never_hit_it() {

    let Some(dev) = cuda() else { return };
    let lin = bf16_linear(&dev);

    let v = Var::from_tensor(&Tensor::rand(-1f32, 1f32, (ROWS, IN), &dev).unwrap()).unwrap();
    let out_f32 = lin.forward(v.as_tensor()).unwrap();
    assert!(
        out_f32.track_op(),
        "an F32 activation must take the differentiable fallback"
    );
    assert!(
        out_f32.sum_all().unwrap().backward().unwrap().get(&v).is_some(),
        "the F32 path must deliver a gradient to its input"
    );
}

#[test]
fn the_bf16_fast_path_keeps_the_graph_alive_and_the_hatch_still_returns_a_leaf() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();

    let Some(dev) = cuda() else { return };
    let lin = bf16_linear(&dev);

    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let (v, x) = tracked_bf16_input(&dev);
    let out = lin.forward(&x).unwrap();
    assert_eq!(out.dims(), &[ROWS, OUT]);
    assert!(
        out.track_op(),
        "a BF16 activation through the CUDA fast path came back as a graph leaf, so a \
         training step through this layer contributes no gradient at all"
    );
    let gi = out
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&v)
        .expect("the input received no gradient, so nothing upstream of this layer can train")
        .clone();
    let host: Vec<f32> =
        gi.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    assert!(host.iter().all(|x| x.is_finite()), "non-finite input gradient");
    assert!(
        host.iter().any(|x| *x != 0.0),
        "the input gradient is identically zero, which is what a severed graph looks like \
         when the shapes still line up"
    );

    std::env::set_var("NV_ALLOW_LEAF_GRADIENT_LOSS", "1");
    let (v2, x2) = tracked_bf16_input(&dev);
    let leaked = lin.forward(&x2).unwrap();
    let still_a_leaf = !leaked.track_op();
    let no_grad = leaked.sum_all().unwrap().backward().unwrap().get(&v2).is_none();
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");

    assert!(
        still_a_leaf,
        "NV_ALLOW_LEAF_GRADIENT_LOSS=1 no longer reaches the hand-written kernel, so the \
         positive control above may be passing for an unrelated reason"
    );
    assert!(no_grad, "the superseded path must still be the silent zero it always was");
}

use nv_layers::rope::{Rope, RopeConfig, RopeKind};

const HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const TOKENS: usize = 4;

fn rope_and_inputs(dev: &Device) -> (Rope, Var, Tensor, Tensor, Tensor) {
    let rope = Rope::new(
        RopeConfig { head_dim: HEAD_DIM, max_seq_len: 64, base: 10000.0, kind: RopeKind::Standard },
        dev,
    )
    .unwrap();
    let shape = (TOKENS, HEADS, HEAD_DIM);
    let qv = Var::from_tensor(&Tensor::rand(-1f32, 1f32, shape, dev).unwrap()).unwrap();
    let q = qv.as_tensor().affine(1.0, 0.0).unwrap();
    let k = Tensor::rand(-1f32, 1f32, shape, dev).unwrap();
    let pos = Tensor::from_vec((0..TOKENS as u32).collect::<Vec<_>>(), TOKENS, dev).unwrap();
    assert!(q.track_op(), "q must be on a graph or this proves nothing");
    (rope, qv, q, k, pos)
}

#[test]
fn rope_dispatch_keeps_the_graph_alive_and_the_hatch_still_returns_a_leaf() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };

    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let (rope, qv, q, k, pos) = rope_and_inputs(&dev);
    let (q_rot, _k_rot) = rope.apply(&q, &k, &pos).unwrap();
    assert_eq!(q_rot.dims(), &[TOKENS, HEADS, HEAD_DIM]);
    assert!(
        q_rot.track_op(),
        "Rope::apply on CUDA F32 returned a graph leaf, so training through it delivers no \
         gradient to q_proj or k_proj at all"
    );
    let g = q_rot.sum_all().unwrap().backward().unwrap();
    let host: Vec<f32> = g
        .get(&qv)
        .expect("q received no gradient, so its adapter can never train")
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(host.iter().all(|x| x.is_finite()), "non-finite q gradient");
    assert!(host.iter().any(|x| *x != 0.0), "q gradient is identically zero");

    std::env::set_var("NV_ALLOW_LEAF_GRADIENT_LOSS", "1");
    let (rope2, qv2, q2, k2, pos2) = rope_and_inputs(&dev);
    let (leaked, _) = rope2.apply(&q2, &k2, &pos2).unwrap();
    let still_a_leaf = !leaked.track_op();
    let no_grad = leaked.sum_all().unwrap().backward().unwrap().get(&qv2).is_none();
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");

    assert!(
        still_a_leaf,
        "NV_ALLOW_LEAF_GRADIENT_LOSS=1 no longer reaches apply_cuda_f32, so the row above \
         may be passing for an unrelated reason"
    );
    assert!(no_grad, "the superseded rope path must still be the silent zero it always was");
}
