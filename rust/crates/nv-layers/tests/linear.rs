#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;

fn max_rel(got: &[f32], expect: &[f32]) -> f32 {
    let mut m = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        let rel = if e.abs() > 1e-3 { diff / e.abs() } else { diff };
        m = m.max(rel);
    }
    m
}

#[test]
fn linear_bf16_matches_candle_matmul_no_bias() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };

    for &(out_f, in_f, batch) in &[
        (64usize, 128usize, 16usize),
        (128, 64, 32),
        (256, 256, 8),
        (32, 96, 64),
    ] {
        let w = Tensor::randn(0f32, 0.05, (out_f, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let x = Tensor::randn(0f32, 1.0, (batch, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let lin = Linear::new(w.clone(), None).unwrap();
        let got = lin.forward(&x).unwrap();
        assert_eq!(got.dims(), &[batch, out_f]);
        assert_eq!(got.dtype(), DType::BF16);

        let expect = x
            .to_dtype(DType::F32)
            .unwrap()
            .matmul(&w.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap();

        let got_v: Vec<f32> = got
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
        let rel = max_rel(&got_v, &expect_v);
        assert!(
            rel < 0.05,
            "linear bf16 drift {rel} (out={out_f} in={in_f} batch={batch})"
        );
    }
}

#[test]
fn linear_with_bias_adds_bias() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (out_f, in_f, b) = (32usize, 64usize, 8usize);
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let bias = Tensor::randn(0f32, 0.1, out_f, &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::randn(0f32, 1.0, (b, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let lin_nobias = Linear::new(w.clone(), None).unwrap();
    let lin_bias = Linear::new(w.clone(), Some(bias.clone())).unwrap();

    let y0 = lin_nobias.forward(&x).unwrap();
    let y1 = lin_bias.forward(&x).unwrap();

    let y0_v: Vec<f32> = y0
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let y1_v: Vec<f32> = y1
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b_v: Vec<f32> = bias.to_dtype(DType::F32).unwrap().to_vec1().unwrap();

    for batch_i in 0..b {
        for j in 0..out_f {
            let diff = (y1_v[batch_i * out_f + j] - y0_v[batch_i * out_f + j] - b_v[j]).abs();
            assert!(diff < 0.05, "bias add drift {diff} at ({batch_i},{j})");
        }
    }
}

#[test]
fn linear_handles_3d_input() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (out_f, in_f, batch, seq) = (96usize, 64usize, 4usize, 16usize);
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::randn(0f32, 1.0, (batch, seq, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Linear::new(w, None).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[batch, seq, out_f]);
}

#[test]
fn linear_handles_4d_input() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (out_f, in_f) = (64usize, 32usize);
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::randn(0f32, 1.0, (2, 3, 5, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Linear::new(w, None).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 5, out_f]);
}

#[test]
fn linear_rejects_wrong_input_dim() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let w = Tensor::randn(0f32, 0.02, (32usize, 64usize), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::randn(0f32, 1.0, (8usize, 33usize), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Linear::new(w, None).unwrap();
    assert!(lin.forward(&x).is_err());
}

#[test]
fn linear_rejects_non_2d_weight() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let w = Tensor::randn(0f32, 0.02, (2usize, 32usize, 64usize), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    assert!(Linear::new(w, None).is_err());
}

#[test]
fn linear_zero_input_yields_zero_output() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (out_f, in_f, b) = (16usize, 24usize, 4usize);
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::zeros((b, in_f), DType::BF16, &device).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let y = lin.forward(&x).unwrap();
    let vals: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_abs = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(
        max_abs < 1e-6,
        "zero input produced non-zero output {max_abs}"
    );
}

#[test]
fn linear_deterministic_across_runs() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (out_f, in_f, b) = (48usize, 64usize, 8usize);
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::randn(0f32, 1.0, (b, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Linear::new(w, None).unwrap();
    let y1 = lin.forward(&x).unwrap();
    let y2 = lin.forward(&x).unwrap();
    let v1: Vec<f32> = y1
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let v2: Vec<f32> = y2
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "linear non-deterministic");
    }
}

#[test]
fn linear_cpu_fallback_matches_candle() {
    let device = Device::Cpu;
    let (out_f, in_f, b) = (32usize, 48usize, 6usize);
    let w = Tensor::randn(0f32, 0.05, (out_f, in_f), &device).unwrap();
    let x = Tensor::randn(0f32, 1.0, (b, in_f), &device).unwrap();
    let lin = Linear::new(w.clone(), None).unwrap();
    let got = lin.forward(&x).unwrap();
    let expect = x.matmul(&w.t().unwrap()).unwrap();
    let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
    let rel = max_rel(&got_v, &expect_v);
    assert!(rel < 1e-4, "cpu fallback drift {rel}");
}

#[test]
fn linear_large_dimensions_correctness() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (out_f, in_f, b) = (1024usize, 768usize, 64usize);
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x = Tensor::randn(0f32, 1.0, (b, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Linear::new(w.clone(), None).unwrap();
    let got = lin.forward(&x).unwrap();
    assert_eq!(got.dims(), &[b, out_f]);

    let expect = x
        .to_dtype(DType::F32)
        .unwrap()
        .matmul(&w.to_dtype(DType::F32).unwrap().t().unwrap())
        .unwrap();
    let got_v: Vec<f32> = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
    let rel = max_rel(&got_v, &expect_v);
    assert!(rel < 0.05, "large linear drift {rel}");
}

#[test]
fn linear_asymmetric_dims() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    for &(out_f, in_f, b) in &[(3usize, 5usize, 7usize), (16, 7, 11), (97, 41, 13)] {
        let w = Tensor::randn(0f32, 0.05, (out_f, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let x = Tensor::randn(0f32, 1.0, (b, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let lin = Linear::new(w.clone(), None).unwrap();
        let got = lin.forward(&x).unwrap();
        assert_eq!(got.dims(), &[b, out_f]);
        let expect = x
            .to_dtype(DType::F32)
            .unwrap()
            .matmul(&w.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap();
        let got_v: Vec<f32> = got
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
        let rel = max_rel(&got_v, &expect_v);
        assert!(
            rel < 0.1,
            "asymmetric linear drift {rel} for ({out_f},{in_f},{b})"
        );
    }
}

#[test]
fn linear_metadata_accessors() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let w = Tensor::randn(0f32, 0.02, (80usize, 40usize), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Linear::new(w, None).unwrap();
    assert_eq!(lin.in_features(), 40);
    assert_eq!(lin.out_features(), 80);
    assert!(lin.bias().is_none());
}
