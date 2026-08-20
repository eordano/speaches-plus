#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::norm::RmsNorm;

#[test]
fn rmsnorm_bf16_matches_cpu_candle_reference() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };

    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f64;

    let x_cpu_f32 = Tensor::randn(0f32, 1.0, (batch, hidden), &Device::Cpu).unwrap();
    let w_cpu_f32 = Tensor::randn(1f32, 0.1, hidden, &Device::Cpu).unwrap();
    let x_cpu = x_cpu_f32
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let w_cpu = w_cpu_f32
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let mean_sq = x_cpu.sqr().unwrap().mean_keepdim(1).unwrap();
    let eps_t = Tensor::new(eps as f32, &Device::Cpu).unwrap();
    let denom = mean_sq.broadcast_add(&eps_t).unwrap().sqrt().unwrap();
    let normed = x_cpu.broadcast_div(&denom).unwrap();
    let expect = normed.broadcast_mul(&w_cpu).unwrap();

    let x_gpu = x_cpu
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w_gpu = w_cpu
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let rms = RmsNorm::new(w_gpu, eps);
    let got = rms.forward(&x_gpu).unwrap();
    assert_eq!(got.dtype(), DType::BF16);

    let got_v: Vec<f32> = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let mut max_rel = 0f32;
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        let diff = (g - e).abs();
        let rel = if e.abs() > 1e-3 { diff / e.abs() } else { diff };
        max_rel = max_rel.max(rel);
    }
    assert!(max_rel < 0.01, "bf16 rmsnorm drift {max_rel}");
}

#[test]
fn rmsnorm_f32_matches_cpu_reference() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let batch = 2usize;
    let hidden = 512usize;
    let eps = 1e-5f64;

    let x_cpu = Tensor::randn(0f32, 1.0, (batch, hidden), &Device::Cpu).unwrap();
    let w_cpu = Tensor::randn(1f32, 0.1, hidden, &Device::Cpu).unwrap();

    let mean_sq = x_cpu.sqr().unwrap().mean_keepdim(1).unwrap();
    let eps_t = Tensor::new(eps as f32, &Device::Cpu).unwrap();
    let denom = mean_sq.broadcast_add(&eps_t).unwrap().sqrt().unwrap();
    let normed = x_cpu.broadcast_div(&denom).unwrap();
    let expect = normed.broadcast_mul(&w_cpu).unwrap();

    let x_gpu = x_cpu.to_device(&device).unwrap();
    let w_gpu = w_cpu.to_device(&device).unwrap();
    let rms = RmsNorm::new(w_gpu, eps);
    let got = rms.forward(&x_gpu).unwrap();
    assert_eq!(got.dtype(), DType::F32);

    let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0f32;
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    assert!(max_abs < 1e-3, "f32 rmsnorm drift {max_abs}");
}

#[test]
fn rmsnorm_f32_handles_3d_input() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let (b, t, h) = (2usize, 8usize, 256usize);
    let eps = 1e-5f64;
    let x = Tensor::randn(0f32, 1.0, (b, t, h), &device).unwrap();
    let w = Tensor::randn(1f32, 0.1, h, &device).unwrap();
    let rms = RmsNorm::new(w.clone(), eps);
    let got = rms.forward(&x).unwrap();
    assert_eq!(got.dims(), &[b, t, h]);

    let x_cpu = x.to_device(&Device::Cpu).unwrap();
    let w_cpu = w.to_device(&Device::Cpu).unwrap();
    let mean_sq = x_cpu.sqr().unwrap().mean_keepdim(2).unwrap();
    let denom = mean_sq
        .broadcast_add(&Tensor::new(eps as f32, &Device::Cpu).unwrap())
        .unwrap()
        .sqrt()
        .unwrap();
    let expect = x_cpu
        .broadcast_div(&denom)
        .unwrap()
        .broadcast_mul(&w_cpu)
        .unwrap();

    let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_abs = 0f32;
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    assert!(max_abs < 1e-3, "3d rmsnorm drift {max_abs}");
}

#[test]
fn rmsnorm_f32_scale_invariant() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let (b, h) = (2usize, 128usize);
    let eps = 1e-9f64;
    let x = Tensor::randn(0f32, 1.0, (b, h), &device).unwrap();
    let w = Tensor::ones(h, DType::F32, &device).unwrap();
    let rms = RmsNorm::new(w, eps);

    let y1 = rms.forward(&x).unwrap();
    let x2 = (&x * 7.0).unwrap();
    let y2 = rms.forward(&x2).unwrap();

    let v1: Vec<f32> = y1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = y2.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_abs = 0f32;
    for (a, b) in v1.iter().zip(v2.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(max_abs < 1e-3, "scale invariance drift {max_abs}");
}

#[test]
fn rmsnorm_f32_unit_norm_output_when_weight_one() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let (b, h) = (4usize, 256usize);
    let eps = 1e-9f64;
    let x = Tensor::randn(0f32, 1.0, (b, h), &device).unwrap();
    let w = Tensor::ones(h, DType::F32, &device).unwrap();
    let rms = RmsNorm::new(w, eps);
    let y = rms.forward(&x).unwrap();

    let y_cpu = y.to_device(&Device::Cpu).unwrap();
    let vals: Vec<f32> = y_cpu.flatten_all().unwrap().to_vec1().unwrap();
    for row in 0..b {
        let mut sumsq = 0f32;
        for i in 0..h {
            sumsq += vals[row * h + i].powi(2);
        }
        let rms_val = (sumsq / h as f32).sqrt();
        assert!(
            (rms_val - 1.0).abs() < 1e-3,
            "row {row} rms = {rms_val}, expected 1"
        );
    }
}

#[test]
fn rmsnorm_deterministic() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let x = Tensor::randn(0f32, 1.0, (4usize, 256usize), &device).unwrap();
    let w = Tensor::randn(1f32, 0.1, 256usize, &device).unwrap();
    let rms = RmsNorm::new(w, 1e-5);
    let y1 = rms.forward(&x).unwrap();
    let y2 = rms.forward(&x).unwrap();
    let v1: Vec<f32> = y1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = y2.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "rmsnorm non-deterministic");
    }
}

#[test]
fn rmsnorm_cpu_path_matches_reference() {
    let device = Device::Cpu;
    let (b, h) = (3usize, 64usize);
    let eps = 1e-5f64;
    let x = Tensor::randn(0f32, 1.0, (b, h), &device).unwrap();
    let w = Tensor::randn(1f32, 0.05, h, &device).unwrap();
    let rms = RmsNorm::new(w.clone(), eps);
    let got = rms.forward(&x).unwrap();

    let mean_sq = x.sqr().unwrap().mean_keepdim(1).unwrap();
    let denom = mean_sq
        .broadcast_add(&Tensor::new(eps as f32, &device).unwrap())
        .unwrap()
        .sqrt()
        .unwrap();
    let expect = x.broadcast_div(&denom).unwrap().broadcast_mul(&w).unwrap();
    let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_abs = 0f32;
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    assert!(max_abs < 1e-5, "cpu rmsnorm drift {max_abs}");
}
