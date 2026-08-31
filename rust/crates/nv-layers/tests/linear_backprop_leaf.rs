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

fn worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(
    got: &Tensor,
    want: &Tensor,
) -> f32 {
    let g: Vec<f32> = got
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let w: Vec<f32> = want
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(g.len(), w.len(), "gradient SHAPES disagree, not just values");
    let peak = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.0, "the reference gradient is identically zero");
    let floor = 0.05 * peak;
    let mut worst = 0f32;
    for (a, b) in g.iter().zip(w.iter()) {
        assert!(a.is_finite() && b.is_finite(), "non-finite gradient: {a} vs {b}");
        worst = worst.max((a - b).abs() / a.abs().max(b.abs()).max(floor));
    }
    worst
}

const BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_64_REDUCTION: f32 = 5e-2;

#[test]
fn dx_through_the_op_matches_the_f32_dense_backward_to_bf16_tolerance() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let lin = bf16_linear(&dev);
    let w_f32 = lin.weight().unwrap().to_dtype(DType::F32).unwrap();

    let (v, x) = tracked_bf16_input(&dev);
    let out = lin.forward(&x).unwrap();
    let grads = out.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let dx = grads
        .get(&v)
        .expect("the op path delivered no input gradient at all")
        .clone();

    let v_ref = Var::from_tensor(v.as_tensor()).unwrap();
    let out_ref = v_ref
        .as_tensor()
        .matmul(&w_f32.t().unwrap().contiguous().unwrap())
        .unwrap();
    let grads_ref = out_ref.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let dx_ref = grads_ref.get(&v_ref).unwrap().clone();

    let gap = worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(&dx, &dx_ref);
    assert!(
        gap <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_64_REDUCTION,
        "dx from Bf16LinearOp (dy @ W on the bf16 GEMM) is {gap:e} relative away from the \
         f32 dense backward, past bf16 tolerance -- earlier-layer adapters would train on a \
         wrong gradient"
    );
}

#[test]
fn a_var_backed_weight_is_refused_loudly_instead_of_training_as_a_silent_no_op() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let wv = Var::from_tensor(
        &Tensor::rand(-0.05f32, 0.05f32, (OUT, IN), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
    )
    .unwrap();
    assert!(wv.as_tensor().track_op(), "the weight must be trainable or this proves nothing");
    let lin = Linear::new(wv.as_tensor().clone(), None).unwrap();
    let (_v, x) = tracked_bf16_input(&dev);
    let err = lin.forward(&x).expect_err(
        "a trainable bf16 CUDA weight went through the dx-only op without an error: the op \
         detaches it, so it would receive zero gradient while dx and the loss keep moving",
    );
    assert!(
        format!("{err:#}").contains("Bf16LinearOp detaches its weight"),
        "the forward failed, but not on the frozen-weight ensure: {err:#}"
    );
}

#[test]
fn dx_through_forward_rows_matches_the_f32_backward_of_the_narrowed_rows() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let row_off = 8usize;
    let rows = 16usize;
    let lin = bf16_linear(&dev);
    let w_f32 = lin.weight().unwrap().to_dtype(DType::F32).unwrap();

    let (v, x) = tracked_bf16_input(&dev);
    let out = lin.forward_rows(&x, row_off, rows).unwrap();
    assert_eq!(out.dims(), &[ROWS, rows]);
    assert!(
        out.track_op(),
        "forward_rows under autograd returned a graph leaf, so fused-qkv training \
         through row-sliced projections would deliver no gradient at all"
    );
    let grads = out.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let dx = grads
        .get(&v)
        .expect("the rows arm delivered no input gradient at all")
        .clone();

    let v_ref = Var::from_tensor(v.as_tensor()).unwrap();
    let w_rows = w_f32.narrow(0, row_off, rows).unwrap();
    let out_ref = v_ref
        .as_tensor()
        .matmul(&w_rows.t().unwrap().contiguous().unwrap())
        .unwrap();
    let grads_ref = out_ref.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let dx_ref = grads_ref.get(&v_ref).unwrap().clone();

    let gap = worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(&dx, &dx_ref);
    assert!(
        gap <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_64_REDUCTION,
        "dx from forward_rows is {gap:e} relative away from the f32 backward of \
         W[{row_off}..{}]: a stale row_off keeps the (m, in_features) shape and fails \
         only this value check, training upstream adapters on the wrong weight rows",
        row_off + rows
    );
}

#[test]
fn dx_through_forward_dense_det_matches_the_f32_dense_backward() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let lin = bf16_linear(&dev);
    let w_f32 = lin.weight().unwrap().to_dtype(DType::F32).unwrap();

    let (v, x) = tracked_bf16_input(&dev);
    let out = lin.forward_dense_det(&x).unwrap();
    assert_eq!(out.dims(), &[ROWS, OUT]);
    assert!(
        out.track_op(),
        "forward_dense_det under autograd returned a graph leaf, so a deterministic \
         training forward would deliver no gradient at all"
    );
    let grads = out.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let dx = grads
        .get(&v)
        .expect("the det arm delivered no input gradient at all")
        .clone();

    let v_ref = Var::from_tensor(v.as_tensor()).unwrap();
    let out_ref = v_ref
        .as_tensor()
        .matmul(&w_f32.t().unwrap().contiguous().unwrap())
        .unwrap();
    let grads_ref = out_ref.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let dx_ref = grads_ref.get(&v_ref).unwrap().clone();

    let gap = worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(&dx, &dx_ref);
    assert!(
        gap <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_64_REDUCTION,
        "dx from forward_dense_det is {gap:e} relative away from the f32 dense backward"
    );
}

struct CandleOpsLoraHook {
    a: Var,
    b: Var,
    scaling: f64,
}

impl nv_layers::linear::LoraDeltaHook for CandleOpsLoraHook {
    fn in_features(&self) -> usize {
        self.a.as_tensor().dims()[1]
    }
    fn out_features(&self) -> usize {
        self.b.as_tensor().dims()[0]
    }
    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> anyhow::Result<Option<Tensor>> {
        assert!(win.is_none(), "this test hook only serves full-width forwards");
        let x = x2.to_dtype(DType::F32)?;
        let delta = x
            .matmul(&self.a.as_tensor().t()?.contiguous()?)?
            .matmul(&self.b.as_tensor().t()?.contiguous()?)?
            .affine(self.scaling, 0.0)?;
        Ok(Some((y2 + delta.to_dtype(y2.dtype())?)?))
    }
}

#[test]
fn lora_factor_grads_flow_through_the_op_and_match_the_f32_reference() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let rank = 4usize;
    let lin = bf16_linear(&dev);
    let w_f32 = lin.weight().unwrap().to_dtype(DType::F32).unwrap();
    let a0 = Tensor::rand(-0.1f32, 0.1f32, (rank, IN), &dev).unwrap();
    let b0 = Tensor::rand(-0.1f32, 0.1f32, (OUT, rank), &dev).unwrap();
    let hook = std::sync::Arc::new(CandleOpsLoraHook {
        a: Var::from_tensor(&a0).unwrap(),
        b: Var::from_tensor(&b0).unwrap(),
        scaling: 2.0,
    });
    lin.attach_lora(hook.clone()).unwrap();

    let (_v, x) = tracked_bf16_input(&dev);
    let out = lin.forward(&x).unwrap();
    let grads = out.sqr().unwrap().sum_all().unwrap().backward().unwrap();
    let ga = grads
        .get(hook.a.as_tensor())
        .expect("no gradient reached lora_A through the op path")
        .clone();
    let gb = grads
        .get(hook.b.as_tensor())
        .expect("no gradient reached lora_B through the op path")
        .clone();

    let a_ref = Var::from_tensor(&a0).unwrap();
    let b_ref = Var::from_tensor(&b0).unwrap();
    let x_ref = x.detach().to_dtype(DType::F32).unwrap();
    let base_ref = x_ref
        .matmul(&w_f32.t().unwrap().contiguous().unwrap())
        .unwrap();
    let delta_ref = x_ref
        .matmul(&a_ref.as_tensor().t().unwrap().contiguous().unwrap())
        .unwrap()
        .matmul(&b_ref.as_tensor().t().unwrap().contiguous().unwrap())
        .unwrap()
        .affine(2.0, 0.0)
        .unwrap();
    let out_ref = (base_ref + delta_ref).unwrap();
    let grads_ref = out_ref.sqr().unwrap().sum_all().unwrap().backward().unwrap();

    for (name, got, want) in [
        ("lora_A", &ga, grads_ref.get(a_ref.as_tensor()).unwrap()),
        ("lora_B", &gb, grads_ref.get(b_ref.as_tensor()).unwrap()),
    ] {
        let gap = worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(got, want);
        assert!(
            gap <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_64_REDUCTION,
            "{name} gradient through Bf16LinearOp is {gap:e} relative away from the f32 \
             reference -- the adapter would train against a different base than it serves on"
        );
    }
}

fn weight_shaped_grad_entries(grads: &candle_core::backprop::GradStore) -> usize {
    grads
        .get_ids()
        .filter_map(|id| grads.get_id(*id))
        .filter(|g| g.dims() == [OUT, IN] || g.dims() == [IN, OUT])
        .count()
}

#[test]
fn a_backward_leaves_no_weight_shaped_gradient_and_the_superseded_fallback_left_one() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");

    let lin = bf16_linear(&dev);
    let frozen_weight = lin.weight().unwrap().clone();
    let (v, x) = tracked_bf16_input(&dev);
    let grads = lin
        .forward(&x)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert!(grads.get(&v).is_some(), "positive control: dx must still arrive");
    assert!(
        grads.get(&frozen_weight).is_none(),
        "the frozen weight tensor itself acquired a gradient entry"
    );
    assert_eq!(
        weight_shaped_grad_entries(&grads),
        0,
        "a weight-shaped gradient survived the backward: candle built a weight-side graph, \
         which is exactly the retained-transposed-copy + phantom-rhs-grad memory the op \
         exists to kill (on E4B this was ~16.8 GiB per length bucket and an OOM at 84 GiB \
         free)"
    );

    let w_f32_forces_the_superseded_matmul_fallback_under_autograd =
        Tensor::rand(-0.05f32, 0.05f32, (OUT, IN), &dev).unwrap();
    let lin_old = Linear::new(w_f32_forces_the_superseded_matmul_fallback_under_autograd, None)
        .unwrap();
    let (v2, x2) = tracked_bf16_input(&dev);
    let grads_old = lin_old
        .forward(&x2)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert!(grads_old.get(&v2).is_some());
    assert!(
        weight_shaped_grad_entries(&grads_old) > 0,
        "the sensitivity control went green: matmul_fallback under autograd no longer \
         leaves a weight-shaped phantom gradient, so the probe above can no longer fail \
         and proves nothing"
    );
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
