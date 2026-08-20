#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::attn::{flash_attn, AttnConfig};

fn cpu_attn_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle_core::Result<Tensor> {
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let dims = q.dims();
    let (b, sq, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let sk = k.dims()[1];

    let q_t = q.permute((0, 2, 1, 3))?.contiguous()?;
    let k_t = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v_t = v.permute((0, 2, 1, 3))?.contiguous()?;

    let q_flat = q_t.reshape((b * h, sq, d))?;
    let k_flat = k_t.reshape((b * h, sk, d))?;
    let v_flat = v_t.reshape((b * h, sk, d))?;

    let k_perm = k_flat.permute((0, 2, 1))?.contiguous()?;
    let scale = Tensor::new(softmax_scale, q.device())?;
    let mut scores = q_flat.matmul(&k_perm)?.broadcast_mul(&scale)?;

    if causal {
        let mask = build_causal_mask(sq, sk, q.device())?;
        scores = scores.broadcast_add(&mask)?;
    }

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v_flat)?;
    let out = out
        .reshape((b, h, sq, d))?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    Ok(out)
}

fn build_causal_mask(sq: usize, sk: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut mask = vec![0f32; sq * sk];
    for i in 0..sq {
        for j in 0..sk {
            if j > i + sk.saturating_sub(sq) {
                mask[i * sk + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, sq, sk), device)?.reshape((sq, sk))
}

#[test]
fn flash_attn_matches_cpu_softmax_reference() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };

    let (b, sq, h, d) = (2usize, 16usize, 4usize, 64usize);
    let q = Tensor::randn(0f32, 1.0, (b, sq, h, d), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let k = Tensor::randn(0f32, 1.0, (b, sq, h, d), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let v = Tensor::randn(0f32, 1.0, (b, sq, h, d), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

    let cfg = AttnConfig {
        num_heads: h,
        num_kv_heads: h,
        head_dim: d,
        softmax_scale: 1.0 / (d as f32).sqrt(),
        causal: true,
    };

    let got = flash_attn(&q, &k, &v, &cfg).unwrap();
    let expect = cpu_attn_reference(&q, &k, &v, cfg.softmax_scale, cfg.causal).unwrap();

    let n = got.elem_count();
    let got_v: Vec<f32> = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expect_v: Vec<f32> = expect.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got_v.len(), n);
    assert_eq!(expect_v.len(), n);

    let mut max_abs = 0f32;
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    assert!(max_abs < 0.01, "flash_attn drift {max_abs} exceeds 0.01");
}
