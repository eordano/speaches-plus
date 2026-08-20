#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use half::bf16;
use nv_layers::linear::Linear;
use nv_quant::fp8::{supports_fp8, Fp8GemmRunner};
use std::sync::{Arc, Mutex, OnceLock};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn ctx_and_major() -> Option<(Arc<CudaContext>, i32)> {
    let ctx = CudaContext::new(0).ok()?;
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    Some((ctx, major))
}

fn probe_result(ctx: &Arc<CudaContext>) -> &'static Result<(), String> {
    static R: OnceLock<Result<(), String>> = OnceLock::new();
    R.get_or_init(|| {
        let stream = ctx.default_stream();
        let mut runner = Fp8GemmRunner::new(stream).unwrap();
        runner
            .probe_per_row_scale_support()
            .map_err(|e| format!("{e:#}"))
    })
}

fn test_weight_and_input(m: usize, d: usize, device: &Device) -> (Tensor, Tensor) {
    let w: Vec<bf16> = (0..d * d)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin()))
        .collect();
    let x: Vec<bf16> = (0..m * d)
        .map(|i| bf16::from_f32(((i as f32) * 0.011).cos()))
        .collect();
    (
        Tensor::from_vec(w, (d, d), device).unwrap(),
        Tensor::from_vec(x, (m, d), device).unwrap(),
    )
}

#[test]
fn per_row_probe_predicts_construction_behavior() {
    let Some((ctx, major)) = ctx_and_major() else {
        eprintln!("skip: no CUDA");
        return;
    };
    if !supports_fp8(major) {
        eprintln!("skip: SM {major} lacks FP8");
        return;
    }
    let _env = ENV_LOCK.lock().unwrap();
    assert!(
        std::env::var("NV_FP8_SCALE_MODE").is_err(),
        "test needs the default scale mode; unset NV_FP8_SCALE_MODE"
    );
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let (m, d) = (16usize, 1024usize);
    let (wt, xt) = test_weight_and_input(m, d, &device);
    let runner = Arc::new(Mutex::new(Fp8GemmRunner::new(stream).unwrap()));

    match probe_result(&ctx) {
        Ok(()) => {
            let lin = Linear::from_bf16_quantized_fp8(&wt, None, &device, runner)
                .expect("probe says per-row is served; construction must succeed");
            let y = lin.forward(&xt).expect(
                "probe says per-row is served; the first forward must not be the place it dies",
            );
            assert_eq!(y.dims(), &[m, d]);
            eprintln!("[probe-contract] per-row fp8 served here; construct+forward ok");
        }
        Err(detail) => {
            eprintln!("[probe-contract] probe refused per-row fp8: {detail}");

            {
                use nv_quant::fp8::quantize_e4m3_per_row;
                let stream = ctx.default_stream();
                let wv: Vec<bf16> = (0..d * d)
                    .map(|i| bf16::from_f32(((i as f32) * 0.013).sin()))
                    .collect();
                let xv: Vec<bf16> = (0..m * d)
                    .map(|i| bf16::from_f32(((i as f32) * 0.011).cos()))
                    .collect();
                let (wq, ws) = quantize_e4m3_per_row(&wv, d, d).unwrap();
                let (xq, xs) = quantize_e4m3_per_row(&xv, m, d).unwrap();
                #[allow(deprecated)]
                let a_dev = stream.clone_htod(&xq).unwrap();
                #[allow(deprecated)]
                let b_dev = stream.clone_htod(&wq).unwrap();
                #[allow(deprecated)]
                let as_dev = stream.clone_htod(&xs).unwrap();
                #[allow(deprecated)]
                let bs_dev = stream.clone_htod(&ws).unwrap();
                let mut d_dev = stream.alloc_zeros::<bf16>(m * d).unwrap();
                let r = runner.lock().unwrap().matmul_e4m3_row_scaled(
                    &a_dev, &b_dev, &mut d_dev, m as u64, d as u64, d as u64, &as_dev, &bs_dev,
                );
                assert!(
                    r.is_err(),
                    "the probe refused per-row fp8 but a real per-row GEMM at the probe's \
                     own shape SUCCEEDED -- the probe is wrong, and every per-row test in \
                     the tree is being silently downgraded to tensor mode"
                );
                eprintln!(
                    "[probe-contract] hardware agrees: real per-row GEMM failed: {:#}",
                    r.unwrap_err()
                );
            }
            let err = Linear::from_bf16_quantized_fp8(&wt, None, &device, runner)
                .err()
                .expect(
                    "probe says per-row fp8 is NOT served; construction must refuse at load \
                     instead of deferring the failure to the first request",
                );
            let msg = format!("{err:#}");
            eprintln!("[probe-contract] BOOT-TIME REFUSAL:\n{msg}");
            assert!(
                msg.contains("NV_FP8_SCALE_MODE=tensor"),
                "the refusal must name the escape hatch NV_FP8_SCALE_MODE=tensor, got: {msg}"
            );
            assert!(
                msg.contains("OUTER_VEC_32F"),
                "the refusal must name the unsupported cuBLASLt mode, got: {msg}"
            );
        }
    }
}

#[test]
fn tensor_mode_escape_hatch_constructs_and_forwards() {
    let Some((ctx, major)) = ctx_and_major() else {
        eprintln!("skip: no CUDA");
        return;
    };
    if !supports_fp8(major) {
        eprintln!("skip: SM {major} lacks FP8");
        return;
    }
    let _env = ENV_LOCK.lock().unwrap();
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let (m, d) = (16usize, 256usize);
    let (wt, xt) = test_weight_and_input(m, d, &device);
    let runner = Arc::new(Mutex::new(Fp8GemmRunner::new(stream).unwrap()));

    std::env::set_var("NV_FP8_SCALE_MODE", "tensor");
    let built = Linear::from_bf16_quantized_fp8(&wt, None, &device, runner);
    std::env::remove_var("NV_FP8_SCALE_MODE");
    let lin = built.expect("tensor-mode fp8 must construct everywhere fp8 exists");

    let y = lin.forward(&xt).expect("tensor-mode fp8 forward");
    assert_eq!(y.dims(), &[m, d]);
    let y_ref = Linear::new(wt, None).unwrap().forward(&xt).unwrap();
    let yf: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let yb: Vec<f32> = y_ref
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, e) in yf.iter().zip(yb.iter()) {
        num += ((g - e) as f64).powi(2);
        den += (*e as f64).powi(2);
    }
    let rel = (num / den.max(1e-12)).sqrt();
    eprintln!("[tensor-mode] fp8 vs bf16 rel rms = {rel}");
    assert!(rel < 0.05, "tensor-mode fp8 vs bf16 rel rms {rel} > 0.05");
}
