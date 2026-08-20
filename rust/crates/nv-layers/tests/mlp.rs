#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;

#[test]
fn mlp_forward_shape_and_finite_on_gpu() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };

    let hidden = 128usize;
    let intermediate = 256usize;
    let batch = 2usize;
    let seq = 8usize;

    let mk = |out_f: usize, in_f: usize| -> Linear {
        let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        Linear::new(w, None).unwrap()
    };

    let mlp = Mlp::new(
        mk(intermediate, hidden),
        mk(intermediate, hidden),
        mk(hidden, intermediate),
    )
    .unwrap();

    let x = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let out = mlp.forward(&x).unwrap();
    assert_eq!(out.dims(), &[batch, seq, hidden]);
    assert_eq!(out.dtype(), DType::BF16);

    let vals: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "mlp produced non-finite values"
    );
    let any_nonzero = vals.iter().any(|v| v.abs() > 1e-6);
    assert!(any_nonzero, "mlp produced all zeros");
}

#[test]
fn mlp_swiglu_matches_candle_reference() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (h, i, b, s) = (64usize, 128usize, 2usize, 4usize);

    let gate_w = Tensor::randn(0f32, 0.02, (i, h), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let up_w = Tensor::randn(0f32, 0.02, (i, h), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let down_w = Tensor::randn(0f32, 0.02, (h, i), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let mlp = Mlp::new(
        Linear::new(gate_w.clone(), None).unwrap(),
        Linear::new(up_w.clone(), None).unwrap(),
        Linear::new(down_w.clone(), None).unwrap(),
    )
    .unwrap();

    let x = Tensor::randn(0f32, 1.0, (b, s, h), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let got = mlp.forward(&x).unwrap();

    let xf = x.to_dtype(DType::F32).unwrap();
    let gate_f = gate_w.to_dtype(DType::F32).unwrap();
    let up_f = up_w.to_dtype(DType::F32).unwrap();
    let down_f = down_w.to_dtype(DType::F32).unwrap();

    let x2 = xf.reshape((b * s, h)).unwrap();
    let gate = x2.matmul(&gate_f.t().unwrap()).unwrap();
    let up = x2.matmul(&up_f.t().unwrap()).unwrap();
    let act = candle_nn::ops::silu(&gate).unwrap().mul(&up).unwrap();
    let down = act.matmul(&down_f.t().unwrap()).unwrap();
    let expect = down.reshape((b, s, h)).unwrap();

    let got_v: Vec<f32> = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        let abs = (g - e).abs();
        let rel = if e.abs() > 1e-3 { abs / e.abs() } else { abs };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 0.1,
        "swiglu drift max_abs={max_abs} max_rel={max_rel}"
    );
}

#[test]
fn mlp_zero_input_zero_output() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (h, i) = (32usize, 64usize);
    let mk = |out_f: usize, in_f: usize| -> Linear {
        let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        Linear::new(w, None).unwrap()
    };
    let mlp = Mlp::new(mk(i, h), mk(i, h), mk(h, i)).unwrap();
    let x = Tensor::zeros((2usize, 4usize, h), DType::BF16, &device).unwrap();
    let out = mlp.forward(&x).unwrap();
    let vals: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_abs = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(max_abs < 1e-3, "zero in non-zero out {max_abs}");
}

#[test]
fn mlp_rejects_shape_mismatch_on_construction() {
    let device = Device::Cpu;
    let mk = |out_f: usize, in_f: usize| -> Linear {
        let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device).unwrap();
        Linear::new(w, None).unwrap()
    };
    let gate = mk(64, 32);
    let up = mk(80, 32);
    let down = mk(32, 64);
    assert!(Mlp::new(gate, up, down).is_err());

    let gate = mk(64, 32);
    let up = mk(64, 33);
    let down = mk(32, 64);
    assert!(Mlp::new(gate, up, down).is_err());

    let gate = mk(64, 32);
    let up = mk(64, 32);
    let down = mk(32, 65);
    assert!(Mlp::new(gate, up, down).is_err());
}

#[test]
fn mlp_deterministic() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let (h, i) = (32usize, 64usize);
    let mk = |out_f: usize, in_f: usize| -> Linear {
        let w = Tensor::randn(0f32, 0.02, (out_f, in_f), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        Linear::new(w, None).unwrap()
    };
    let mlp = Mlp::new(mk(i, h), mk(i, h), mk(h, i)).unwrap();
    let x = Tensor::randn(0f32, 1.0, (2usize, 4usize, h), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let y1 = mlp.forward(&x).unwrap();
    let y2 = mlp.forward(&x).unwrap();
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
        assert_eq!(a.to_bits(), b.to_bits(), "mlp non-deterministic");
    }
}
