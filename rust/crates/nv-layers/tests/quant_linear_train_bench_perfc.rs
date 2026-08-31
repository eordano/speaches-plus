#![cfg(feature = "cuda")]

mod common;
use candle_core::{DType, Device, Tensor, Var};
use common::{cuda, detect_major};
use cudarc::driver::{CudaContext, CudaStream};
use half::bf16;
use nv_layers::linear::{fp8_rowscale_dequant_bf16_host, fp8_weight_payload, Linear};
use nv_quant::fp8::{supports_fp8, Fp8GemmRunner, Fp8ScaleMode};
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

const IN: usize = 5120;
const OUT: usize = 5120;
const ROWS: usize = 4;
const SERVING_DECODE_ROWS_1_PRICES_THE_M1_GEMV_SHAPE_SERVING_ACTUALLY_RUNS: usize = 1;
const WARMUP: usize = 10;
const ITERS: usize = 100;

struct QuantAndDequantTwin {
    label: &'static str,
    quant: Linear,
    dense_twin_on_the_host_dequant_weights_the_trainers_construct_today: Linear,
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
            dense_twin_on_the_host_dequant_weights_the_trainers_construct_today: twin,
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
            dense_twin_on_the_host_dequant_weights_the_trainers_construct_today: twin,
        });
    }
    assert!(
        !out.is_empty(),
        "this SM supports neither fp8 nor nvfp4 residency, so the bench has nothing to price"
    );
    out
}

fn bench_ms(stream: &Arc<CudaStream>, mut iter: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        iter();
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        iter();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn fwd_serve_ms(stream: &Arc<CudaStream>, lin: &Linear, x_serve: &Tensor) -> f64 {
    bench_ms(stream, || {
        let y = lin.forward(x_serve).unwrap();
        assert!(!y.track_op());
        drop(y);
    })
}

fn fwd_bwd_ms(stream: &Arc<CudaStream>, lin: &Linear, dev: &Device) -> f64 {
    let v = Var::from_tensor(&Tensor::rand(-1f32, 1f32, (ROWS, IN), dev).unwrap()).unwrap();
    bench_ms(stream, || {
        let x = v.as_tensor().to_dtype(DType::BF16).unwrap();
        assert!(x.track_op());
        let grads = lin
            .forward(&x)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert!(grads.get(&v).is_some());
        drop(grads);
    })
}

#[test]
#[ignore = "wall-clock micro-benchmark on a quiet card; set NV_QUANT_LINEAR_TRAIN_BENCH=1"]
fn quantized_resident_forward_backward_is_priced_against_the_host_dequant_dense_twin() {
    if std::env::var("NV_QUANT_LINEAR_TRAIN_BENCH").as_deref() != Ok("1") {
        eprintln!("skip: NV_QUANT_LINEAR_TRAIN_BENCH unset");
        return;
    }
    let Some(dev) = cuda() else { return };
    std::env::remove_var("NV_ALLOW_LEAF_GRADIENT_LOSS");
    let stream = match &dev {
        Device::Cuda(d) => d.cuda_stream(),
        _ => unreachable!(),
    };
    println!(
        "[quant-linear-train-bench] shape {OUT}x{IN} m={ROWS} warmup {WARMUP} iters {ITERS}"
    );
    for pair in quant_pairs(&dev) {
        let x_serve = Tensor::rand(-1f32, 1f32, (ROWS, IN), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .detach();
        let twin = &pair.dense_twin_on_the_host_dequant_weights_the_trainers_construct_today;
        let x_serve_m1 = Tensor::rand(
            -1f32,
            1f32,
            (SERVING_DECODE_ROWS_1_PRICES_THE_M1_GEMV_SHAPE_SERVING_ACTUALLY_RUNS, IN),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .detach();
        let q_m1 = fwd_serve_ms(&stream, &pair.quant, &x_serve_m1);
        let d_m1 = fwd_serve_ms(&stream, twin, &x_serve_m1);
        println!(
            "[quant-linear-train-bench] {label}  fwd_serve_m1 quant {q_m1:.4} ms  dense \
             {d_m1:.4} ms  m1 quant/dense {ratio:.3}",
            label = pair.label,
            ratio = q_m1 / d_m1
        );
        let q_serve = fwd_serve_ms(&stream, &pair.quant, &x_serve);
        let d_serve = fwd_serve_ms(&stream, twin, &x_serve);
        let q_fb = fwd_bwd_ms(&stream, &pair.quant, &dev);
        let d_fb = fwd_bwd_ms(&stream, twin, &dev);
        println!(
            "[quant-linear-train-bench] {label}  fwd_serve quant {q_serve:.4} ms  dense \
             {d_serve:.4} ms  fwd+bwd quant {q_fb:.4} ms  dense {d_fb:.4} ms  fwd+bwd \
             quant/dense {ratio:.3}",
            label = pair.label,
            ratio = q_fb / d_fb
        );
    }
}
