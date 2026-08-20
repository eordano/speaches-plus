#![cfg(feature = "cuda")]

mod common;
use common::detect_major;
use candle_core::{DType, Device, Tensor};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use half::bf16;
use nv_layers::linear::Linear;
use nv_quant::fp8::{supports_fp8, Fp8GemmRunner};
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

fn bits(t: &Tensor) -> Vec<u16> {
    t.to_dtype(DType::BF16)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn garbage_and_rows(leading: usize, k: usize) -> (Vec<bf16>, Vec<bf16>) {
    let garbage: Vec<bf16> = (0..k).map(|i| bf16::from_f32(500.0 + i as f32)).collect();
    let rows: Vec<bf16> = (0..leading * k)
        .map(|i| bf16::from_f32(((i as f32) * 0.011).cos()))
        .collect();
    (garbage, rows)
}

fn offset_view_and_ref(leading: usize, k: usize, device: &Device) -> (Tensor, Tensor) {
    let (garbage, rows) = garbage_and_rows(leading, k);
    let mut full = garbage;
    full.extend_from_slice(&rows);
    let full_t = Tensor::from_vec(full, (leading + 1, k), device).unwrap();
    let view = full_t.narrow(0, 1, leading).unwrap();
    let ref_t = Tensor::from_vec(rows, (leading, k), device).unwrap();
    (view, ref_t)
}

fn test_weight(n: usize, k: usize, device: &Device) -> Tensor {
    let w: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin()))
        .collect();
    Tensor::from_vec(w, (n, k), device).unwrap()
}

#[test]
fn bf16_forward_honors_input_start_offset_all_leadings() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let _ = detect_major(&ctx);
    let device = Device::new_cuda(0).unwrap();
    let (n, k) = (128usize, 128usize);
    let lin = Linear::new(test_weight(n, k, &device), None).unwrap();

    for leading in [1usize, 4, 32] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        assert_ne!(view.layout().start_offset(), 0);
        let y_view = lin.forward(&view).unwrap();
        let y_ref = lin.forward(&ref_t).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "bf16 forward leading={leading}: offset view diverges from full tensor"
        );
    }
}

#[test]
fn bf16_forward_dense_honors_input_start_offset() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let _ = detect_major(&ctx);
    let device = Device::new_cuda(0).unwrap();
    let (n, k) = (128usize, 128usize);
    let lin = Linear::new(test_weight(n, k, &device), None).unwrap();

    for leading in [1usize, 4, 32] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        let y_view = lin.forward_dense(&view).unwrap();
        let y_ref = lin.forward_dense(&ref_t).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "bf16 forward_dense leading={leading}: offset view diverges"
        );
    }
}

#[test]
fn bf16_pretransposed_forward_honors_input_start_offset() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let _ = detect_major(&ctx);
    let device = Device::new_cuda(0).unwrap();
    let (n, k) = (128usize, 128usize);
    let mut lin = Linear::new(test_weight(n, k, &device), None).unwrap();
    assert!(lin.ensure_pretransposed().unwrap());

    for leading in [4usize, 32] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        let y_view = lin.forward(&view).unwrap();
        let y_ref = lin.forward(&ref_t).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "bf16 pretransposed forward leading={leading}: offset view diverges"
        );
    }
}

#[test]
fn bf16_forward_dense_det_honors_input_start_offset() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let _ = detect_major(&ctx);
    let device = Device::new_cuda(0).unwrap();
    let (n, k) = (128usize, 128usize);
    let lin = Linear::new(test_weight(n, k, &device), None).unwrap();

    for leading in [1usize, 8] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        let y_view = lin.forward_dense_det(&view).unwrap();
        let y_ref = lin.forward_dense_det(&ref_t).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "forward_dense_det leading={leading}: offset view diverges"
        );
    }
}

#[test]
fn bf16_forward_rows_honors_input_start_offset() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let _ = detect_major(&ctx);
    let device = Device::new_cuda(0).unwrap();
    let (n, k) = (128usize, 128usize);
    let lin = Linear::new(test_weight(n, k, &device), None).unwrap();
    let (row_off, rows) = (16usize, 32usize);

    for leading in [1usize, 4] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        let y_view = lin.forward_rows(&view, row_off, rows).unwrap();
        let y_ref = lin.forward_rows(&ref_t, row_off, rows).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "forward_rows leading={leading}: offset view diverges"
        );
    }
}

#[test]
fn nvfp4_forward_honors_input_start_offset() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));
    let (n, k) = (128usize, 128usize);
    let lin =
        Linear::from_bf16_quantized_nvfp4_dev(&test_weight(n, k, &device), None, &device, runner)
            .unwrap();

    for leading in [1usize, 4] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        let y_view = lin.forward(&view).unwrap();
        let y_ref = lin.forward(&ref_t).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "nvfp4 forward leading={leading}: offset view diverges"
        );
    }
}

#[test]
fn fp8_forward_honors_input_start_offset() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_fp8(major) {
        eprintln!("skip: SM {major} lacks FP8");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Fp8GemmRunner::new(stream.clone()).unwrap()));
    let (n, k) = (128usize, 128usize);
    let w = test_weight(n, k, &device);

    let lin = match Linear::from_bf16_quantized_fp8(&w, None, &device, runner.clone()) {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("OUTER_VEC_32F"),
                "unexpected fp8 construction failure: {msg}"
            );
            eprintln!("[fp8-test] per-row refused on this platform; using tensor mode");
            Linear::from_bf16_quantized_fp8_in_mode(
                &w,
                None,
                None,
                &device,
                runner,
                nv_quant::fp8::Fp8ScaleMode::PerTensor,
            )
            .expect("tensor-mode fp8 must construct wherever fp8 exists")
        }
    };

    for leading in [1usize, 4] {
        let (view, ref_t) = offset_view_and_ref(leading, k, &device);
        let y_view = lin.forward(&view).unwrap();
        let y_ref = lin.forward(&ref_t).unwrap();
        assert_eq!(
            bits(&y_view),
            bits(&y_ref),
            "fp8 forward leading={leading}: offset view diverges"
        );
    }
}
