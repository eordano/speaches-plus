#![cfg(feature = "cuda")]

mod common;
use candle_core::{DType, Device, Tensor, Var};
use common::{cuda, detect_major};
use cudarc::driver::CudaContext;
use half::bf16;
use nv_layers::linear::{fp8_rowscale_dequant_bf16_host, fp8_weight_payload, Linear};
use nv_quant::fp8::{supports_fp8, Fp8GemmRunner, Fp8ScaleMode};
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

const IN: usize = 128;
const OUT: usize = 128;
const ROWS: usize = 4;
const BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_128_REDUCTION: f32 = 5e-2;

fn the_env_var_is_process_global_so_these_rows_must_not_overlap(
) -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

struct QuantAndDequantTwin {
    label: &'static str,
    quant: Linear,
    dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses: Linear,
}

fn weight_host() -> Vec<bf16> {
    (0..OUT * IN)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 0.25))
        .collect()
}

fn quant_pairs(dev: &Device) -> Vec<QuantAndDequantTwin> {
    let Ok(ctx) = CudaContext::new(0) else {
        return Vec::new();
    };
    let major = detect_major(&ctx);
    let stream = ctx.default_stream();
    let w_host = weight_host();
    let w = Tensor::from_vec(w_host.clone(), (OUT, IN), dev).unwrap();
    let mut out = Vec::new();

    if supports_fp8(major) {
        let runner = Arc::new(Mutex::new(Fp8GemmRunner::new(stream.clone()).unwrap()));
        let (quant, mode) = match Linear::from_bf16_quantized_fp8_in_mode(
            &w,
            None,
            None,
            dev,
            runner.clone(),
            Fp8ScaleMode::PerOuterRow,
        ) {
            Ok(l) => (l, Fp8ScaleMode::PerOuterRow),
            Err(e) => {
                assert!(
                    format!("{e:#}").contains("OUTER_VEC_32F"),
                    "unexpected fp8 construction failure: {e:#}"
                );
                (
                    Linear::from_bf16_quantized_fp8_in_mode(
                        &w,
                        None,
                        None,
                        dev,
                        runner,
                        Fp8ScaleMode::PerTensor,
                    )
                    .unwrap(),
                    Fp8ScaleMode::PerTensor,
                )
            }
        };
        let (bytes, rows) = fp8_weight_payload(&w_host, OUT, IN, None, mode).unwrap();
        let w_dq = fp8_rowscale_dequant_bf16_host(&bytes, &rows, OUT, IN);
        let twin = Linear::new(Tensor::from_vec(w_dq, (OUT, IN), dev).unwrap(), None).unwrap();
        out.push(QuantAndDequantTwin {
            label: "fp8",
            quant,
            dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses: twin,
        });
    }
    if supports_nvfp4(major) {
        let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream).unwrap()));
        let quant = Linear::from_bf16_quantized_nvfp4(&w, None, dev, runner).unwrap();
        let w_dq = quant
            .dequant_weight()
            .unwrap()
            .expect("nvfp4 storage must expose its host-dequant oracle");
        let twin = Linear::new(w_dq, None).unwrap();
        out.push(QuantAndDequantTwin {
            label: "nvfp4",
            quant,
            dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses: twin,
        });
    }
    if out.is_empty() {
        eprintln!("skip: this SM supports neither fp8 nor nvfp4 residency");
    }
    out
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

fn bits(t: &Tensor) -> Vec<u16> {
    assert_eq!(
        t.dtype(),
        DType::BF16,
        "bit parity is only meaningful on the bf16 output"
    );
    t.to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect()
}

fn fixed_dy(dev: &Device) -> Tensor {
    Tensor::rand(-1f32, 1f32, (ROWS, OUT), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .detach()
}

fn worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(got: &Tensor, want: &Tensor) -> f32 {
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
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite gradient: {a} vs {b}"
        );
        worst = worst.max((a - b).abs() / a.abs().max(b.abs()).max(floor));
    }
    worst
}

#[test]
fn the_training_forward_is_bitwise_the_serving_forward_for_each_quantized_storage() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    for pair in quant_pairs(&dev) {
        let (_v, x) = tracked_bf16_input(&dev);
        let x_serve = x.detach();
        assert!(!x_serve.track_op());
        let y_serve = pair.quant.forward(&x_serve).unwrap();
        assert!(
            !y_serve.track_op(),
            "{}: the serving forward grew a graph, so this test no longer compares \
             serving against training",
            pair.label
        );
        let y_train = pair.quant.forward(&x).unwrap();
        assert!(
            y_train.track_op(),
            "{}: a quantized-storage forward under autograd came back as a graph leaf, \
             so a training step through it contributes no gradient at all",
            pair.label
        );
        assert_eq!(
            bits(&y_serve),
            bits(&y_train.detach()),
            "{}: the QuantLinearOp forward and the no-grad serving forward disagree \
             bitwise on identical input bytes -- train-forward == serve-forward is the \
             entire point of the op",
            pair.label
        );
    }
}

#[test]
fn dx_matches_the_dense_twin_built_from_the_host_dequant_oracle() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    for pair in quant_pairs(&dev) {
        let c = fixed_dy(&dev);
        let (v_q, x_q) = tracked_bf16_input(&dev);
        let grads_q = (pair.quant.forward(&x_q).unwrap() * &c)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let dx_q = grads_q
            .get(&v_q)
            .expect("the quant op path delivered no input gradient at all")
            .clone();

        let v_t = Var::from_tensor(v_q.as_tensor()).unwrap();
        let x_t = v_t.as_tensor().to_dtype(DType::BF16).unwrap();
        let twin = &pair.dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses;
        let grads_t = (twin.forward(&x_t).unwrap() * &c)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let dx_t = grads_t.get(&v_t).unwrap().clone();

        let gap = worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(&dx_q, &dx_t);
        assert!(
            gap <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_128_REDUCTION,
            "{}: dx through QuantLinearOp (dy . W_dequant, straight-through) is {gap:e} \
             relative away from Bf16LinearOp's dx on the host-dequant twin -- same dy, \
             same weight values, same proved GEMM orientation, so any gap here is a \
             wiring error in the backward",
            pair.label
        );
    }
}

#[test]
fn a_finite_difference_of_the_dequant_dense_function_confirms_the_dx_orientation() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    for pair in quant_pairs(&dev) {
        let c = fixed_dy(&dev);
        let (v, x) = tracked_bf16_input(&dev);
        let grads = (pair.quant.forward(&x).unwrap() * &c)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let dx: Vec<f32> = grads
            .get(&v)
            .expect("no input gradient to spot-check")
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        let w_dq: Vec<f32> = pair
            .dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses
            .weight()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let c_h: Vec<f32> = c
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let x_h: Vec<f32> = x
            .detach()
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let loss = |xs: &[f32]| -> f64 {
            let mut l = 0f64;
            for r in 0..ROWS {
                for o in 0..OUT {
                    let mut acc = 0f64;
                    for j in 0..IN {
                        acc += (xs[r * IN + j] as f64) * (w_dq[o * IN + j] as f64);
                    }
                    l += acc * c_h[r * OUT + o] as f64;
                }
            }
            l
        };
        let eps = 1e-2f32;
        for (r, i) in [(0usize, 0usize), (0, IN / 2), (ROWS - 1, IN - 1), (1, 7), (2, 96)] {
            let mut plus = x_h.clone();
            let mut minus = x_h.clone();
            plus[r * IN + i] += eps;
            minus[r * IN + i] -= eps;
            let fd = ((loss(&plus) - loss(&minus)) / (2.0 * eps as f64)) as f32;
            let got = dx[r * IN + i];
            let denom = fd.abs().max(got.abs()).max(0.05);
            assert!(
                (got - fd).abs() / denom <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_128_REDUCTION,
                "{}: dx[{r},{i}] = {got} but the central difference of the dequant-dense \
                 loss (exact for a linear map) says {fd} -- a transposed or swapped \
                 operand in the backward GEMM fails exactly here",
                pair.label
            );
        }
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
fn a_quant_backward_leaves_no_weight_shaped_gradient_and_the_fallback_control_still_leaves_one() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    for pair in quant_pairs(&dev) {
        let (v, x) = tracked_bf16_input(&dev);
        let grads = pair
            .quant
            .forward(&x)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert!(
            grads.get(&v).is_some(),
            "{}: positive control: dx must still arrive",
            pair.label
        );
        assert_eq!(
            weight_shaped_grad_entries(&grads),
            0,
            "{}: a weight-shaped gradient survived the quant backward: candle built a \
             weight-side graph, the exact phantom-dW memory both dx-only ops exist to kill",
            pair.label
        );
    }

    let w_f32_forces_the_superseded_matmul_fallback_under_autograd =
        Tensor::rand(-0.05f32, 0.05f32, (OUT, IN), &dev).unwrap();
    let lin_old =
        Linear::new(w_f32_forces_the_superseded_matmul_fallback_under_autograd, None).unwrap();
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
         leaves a weight-shaped phantom gradient, so the probe above proves nothing"
    );
}

#[test]
fn a_recompute_reruns_the_quantized_serving_gemm_and_the_dequant_dense_forward_is_visibly_not_it() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    for pair in quant_pairs(&dev) {
        let (_v, x) = tracked_bf16_input(&dev);
        let x_serve = x.detach();
        let y_serve = pair.quant.forward(&x_serve).unwrap();
        let first = pair.quant.forward(&x).unwrap();
        let recompute = pair.quant.forward(&x).unwrap();
        assert_eq!(
            bits(&first.detach()),
            bits(&recompute.detach()),
            "{}: two op forwards on identical bytes disagree, so a checkpointing \
             recompute would not reproduce the checkpointed activation",
            pair.label
        );
        assert_eq!(
            bits(&recompute.detach()),
            bits(&y_serve),
            "{}: the recompute did not land on the serving quantized GEMM output",
            pair.label
        );
        let y_dense = pair
            .dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses
            .forward(&x_serve)
            .unwrap();
        assert_ne!(
            bits(&y_serve),
            bits(&y_dense),
            "{}: the dequant-dense forward is bitwise the quantized forward on this \
             input, so the two equalities above can no longer tell whether a recompute \
             routed through fwd or through the dequant helper -- pick an input where \
             activation quantization bites",
            pair.label
        );
    }
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
fn lora_factor_grads_flow_on_the_quant_path_and_match_the_dense_twin_arm() {
    let _serialised = the_env_var_is_process_global_so_these_rows_must_not_overlap();
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let rank = 4usize;
    for pair in quant_pairs(&dev) {
        let a0 = Tensor::rand(-0.1f32, 0.1f32, (rank, IN), &dev).unwrap();
        let b0 = Tensor::rand(-0.1f32, 0.1f32, (OUT, rank), &dev).unwrap();
        let c = fixed_dy(&dev);
        let hook_q = Arc::new(CandleOpsLoraHook {
            a: Var::from_tensor(&a0).unwrap(),
            b: Var::from_tensor(&b0).unwrap(),
            scaling: 2.0,
        });
        pair.quant.attach_lora(hook_q.clone()).unwrap();
        let (_v, x) = tracked_bf16_input(&dev);
        let grads_q = (pair.quant.forward(&x).unwrap() * &c)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        pair.quant.detach_lora();
        let ga = grads_q
            .get(hook_q.a.as_tensor())
            .expect(
                "no gradient reached lora_A through the quant op path: the hook seam did \
                 not fire around the QuantLinearOp base GEMM",
            )
            .clone();
        let gb = grads_q
            .get(hook_q.b.as_tensor())
            .expect("no gradient reached lora_B through the quant op path")
            .clone();

        let hook_t = Arc::new(CandleOpsLoraHook {
            a: Var::from_tensor(&a0).unwrap(),
            b: Var::from_tensor(&b0).unwrap(),
            scaling: 2.0,
        });
        let twin = &pair.dense_twin_built_from_the_same_host_dequant_oracle_the_backward_uses;
        twin.attach_lora(hook_t.clone()).unwrap();
        let grads_t = (twin.forward(&x.detach()).unwrap() * &c)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        twin.detach_lora();

        for (name, got, want_var) in [("lora_A", &ga, &hook_t.a), ("lora_B", &gb, &hook_t.b)] {
            let want = grads_t.get(want_var.as_tensor()).unwrap();
            let gap = worst_rel_gap_floored_at_a_twentieth_of_the_reference_peak(got, want);
            assert!(
                gap <= BF16_GEMM_REL_TOLERANCE_FOR_A_WIDTH_128_REDUCTION,
                "{}: {name} gradient on the quant path is {gap:e} relative away from the \
                 dense-arm delta -- the adapter would train against a different base than \
                 it serves on",
                pair.label
            );
        }
    }
}
