#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::causal_attention_chunked;

fn fixed_tensor(dev: &Device, seed: u64, shape: (usize, usize, usize, usize)) -> Tensor {
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    let mut state = seed | 1;
    let vals: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0) as f32
        })
        .collect();
    Tensor::from_vec(vals, shape, dev).unwrap()
}

fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    total: usize,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    offset: usize,
) -> Vec<f64> {
    let group = n_q / n_kv;
    let mut out = vec![0f64; seq * n_q * hd];
    for h in 0..n_q {
        let kvh = h / group;
        for i in 0..seq {
            let visible = offset + i + 1;
            let qrow = |d: usize| q[(i * n_q + h) * hd + d] as f64;
            let mut logits = vec![0f64; visible];
            for j in 0..visible {
                let mut acc = 0f64;
                for d in 0..hd {
                    acc += qrow(d) * k[(j * n_kv + kvh) * hd + d] as f64;
                }
                logits[j] = acc;
            }
            let m = logits.iter().cloned().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = logits.iter().map(|&x| (x - m).exp()).collect();
            let denom: f64 = exps.iter().sum();
            for d in 0..hd {
                let mut acc = 0f64;
                for j in 0..visible {
                    acc += exps[j] / denom * v[(j * n_kv + kvh) * hd + d] as f64;
                }
                out[(i * n_q + h) * hd + d] = acc;
            }
        }
    }
    let _ = total;
    out
}

fn check_case(dev: &Device, seq: usize, offset: usize, n_q: usize, n_kv: usize, hd: usize) {
    let total = offset + seq;
    let q = fixed_tensor(dev, 7, (1, seq, n_q, hd));
    let k = fixed_tensor(dev, 11, (1, total, n_kv, hd));
    let v = fixed_tensor(dev, 13, (1, total, n_kv, hd));
    let out = causal_attention_chunked(&q, &k, &v, n_q, n_kv, hd, seq, offset).unwrap();
    assert_eq!(out.dims(), &[1, seq, n_q, hd]);
    let got = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let qh = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let kh = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let vh = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = reference(&qh, &kh, &vh, seq, total, n_q, n_kv, hd, offset);
    let mut max_err = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        max_err = max_err.max((*g as f64 - w).abs());
    }
    eprintln!(
        "[chunked-attn] seq={seq} offset={offset} h={n_q}/{n_kv} d={hd} max_err={max_err:.3e}"
    );
    assert!(
        max_err < 5e-3,
        "chunked attention deviates from f64 reference: {max_err}"
    );
}

#[test]
fn chunked_matches_f64_reference() {
    let dev = Device::new_cuda(0).expect("cuda");
    check_case(&dev, 7, 0, 4, 2, 8);
    check_case(&dev, 36, 0, 8, 2, 16);
    check_case(&dev, 5, 9, 4, 1, 8);
    check_case(&dev, 130, 0, 4, 2, 8);
}

#[test]
fn chunked_multi_chunk_equals_single_chunk_math() {
    let dev = Device::new_cuda(0).expect("cuda");
    check_case(&dev, 700, 0, 2, 1, 8);
}

#[test]
fn chunked_is_bitwise_deterministic_at_model_shape() {
    let dev = Device::new_cuda(0).expect("cuda");
    let (seq, n_q, n_kv, hd) = (36, 32, 4, 512);
    let q = fixed_tensor(&dev, 1, (1, seq, n_q, hd))
        .to_dtype(DType::BF16)
        .unwrap();
    let k = fixed_tensor(&dev, 2, (1, seq, n_kv, hd))
        .to_dtype(DType::BF16)
        .unwrap();
    let v = fixed_tensor(&dev, 3, (1, seq, n_kv, hd))
        .to_dtype(DType::BF16)
        .unwrap();
    let mut hashes = std::collections::HashSet::new();
    for _ in 0..20 {
        let out = causal_attention_chunked(&q, &k, &v, n_q, n_kv, hd, seq, 0).unwrap();
        let bytes = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in &bytes {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hashes.insert(h);
    }
    eprintln!(
        "[chunked-attn] determinism at (36,32/4,512): distinct={}",
        hashes.len()
    );
    assert_eq!(
        hashes.len(),
        1,
        "chunked causal attention must be run-to-run deterministic"
    );
}
