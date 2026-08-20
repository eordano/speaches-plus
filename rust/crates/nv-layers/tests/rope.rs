#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::rope::{Rope, RopeConfig, RopeKind};

fn build_rope_tables(max_pos: usize, half_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; max_pos * half_dim];
    let mut sin = vec![0f32; max_pos * half_dim];
    for p in 0..max_pos {
        for i in 0..half_dim {
            let theta = (p as f32) / base.powf((i as f32 * 2.0) / (half_dim as f32 * 2.0));
            cos[p * half_dim + i] = theta.cos();
            sin[p * half_dim + i] = theta.sin();
        }
    }
    (cos, sin)
}

fn cpu_rope_apply(
    x: &mut [f32],
    positions: &[i32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    n_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    for (t, pos) in positions.iter().enumerate() {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let c = cos_tbl[(*pos as usize) * half + i];
                let s = sin_tbl[(*pos as usize) * half + i];
                let a = x[base + i];
                let b = x[base + i + half];
                x[base + i] = a * c - b * s;
                x[base + i + half] = a * s + b * c;
            }
        }
    }
}

#[test]
fn rope_f32_matches_cpu_reference() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };

    let tokens = 8usize;
    let n_heads = 16usize;
    let n_kv_heads = 4usize;
    let head_dim = 64usize;
    let half = head_dim / 2;
    let max_pos = 32usize;
    let base = 10_000.0f32;

    let (cos_tbl, sin_tbl) = build_rope_tables(max_pos, half, base);

    let mut q_host = Vec::with_capacity(tokens * n_heads * head_dim);
    let mut k_host = Vec::with_capacity(tokens * n_kv_heads * head_dim);
    for i in 0..(tokens * n_heads * head_dim) {
        q_host.push((i as f32 * 0.0013).sin());
    }
    for i in 0..(tokens * n_kv_heads * head_dim) {
        k_host.push((i as f32 * 0.0017).cos());
    }
    let positions: Vec<i32> = (0..tokens).map(|i| (i as i32) % (max_pos as i32)).collect();

    let mut q_expect = q_host.clone();
    let mut k_expect = k_host.clone();
    cpu_rope_apply(
        &mut q_expect,
        &positions,
        &cos_tbl,
        &sin_tbl,
        n_heads,
        head_dim,
    );
    cpu_rope_apply(
        &mut k_expect,
        &positions,
        &cos_tbl,
        &sin_tbl,
        n_kv_heads,
        head_dim,
    );

    let cfg = RopeConfig {
        head_dim,
        max_seq_len: max_pos,
        base,
        kind: RopeKind::Standard,
    };
    let rope = Rope::new(cfg, &device).unwrap();

    let q = Tensor::from_vec(q_host, (tokens, n_heads, head_dim), &device).unwrap();
    let k = Tensor::from_vec(k_host, (tokens, n_kv_heads, head_dim), &device).unwrap();
    let pos = Tensor::from_vec(positions, tokens, &device).unwrap();

    let (q_out, k_out) = rope.apply(&q, &k, &pos).unwrap();
    assert_eq!(q_out.dtype(), DType::F32);

    let q_got = q_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let k_got = k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let mut max_q = 0f32;
    let mut max_k = 0f32;
    for (g, e) in q_got.iter().zip(q_expect.iter()) {
        max_q = max_q.max((g - e).abs());
    }
    for (g, e) in k_got.iter().zip(k_expect.iter()) {
        max_k = max_k.max((g - e).abs());
    }
    assert!(max_q < 1e-5, "rope q drift {max_q}");
    assert!(max_k < 1e-5, "rope k drift {max_k}");
}

#[test]
fn rope_position_zero_is_identity() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let head_dim = 64usize;
    let cfg = RopeConfig {
        head_dim,
        max_seq_len: 16,
        base: 10_000.0,
        kind: RopeKind::Standard,
    };
    let rope = Rope::new(cfg, &device).unwrap();
    let q = Tensor::randn(0f32, 1.0, (4usize, 8usize, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (4usize, 2usize, head_dim), &device).unwrap();
    let zeros = Tensor::from_vec(vec![0i32; 4], 4usize, &device).unwrap();
    let (q_out, k_out) = rope.apply(&q, &k, &zeros).unwrap();
    let q_v: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
    let qo_v: Vec<f32> = q_out.flatten_all().unwrap().to_vec1().unwrap();
    let k_v: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
    let ko_v: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in q_v.iter().zip(qo_v.iter()) {
        assert!((a - b).abs() < 1e-5, "rope at pos 0 not identity for q");
    }
    for (a, b) in k_v.iter().zip(ko_v.iter()) {
        assert!((a - b).abs() < 1e-5, "rope at pos 0 not identity for k");
    }
}

#[test]
fn rope_preserves_norm() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let head_dim = 64usize;
    let n_heads = 4usize;
    let n_kv = 4usize;
    let tokens = 8usize;
    let cfg = RopeConfig {
        head_dim,
        max_seq_len: 32,
        base: 10_000.0,
        kind: RopeKind::Standard,
    };
    let rope = Rope::new(cfg, &device).unwrap();
    let q = Tensor::randn(0f32, 1.0, (tokens, n_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (tokens, n_kv, head_dim), &device).unwrap();
    let positions: Vec<i32> = (0..tokens as i32).collect();
    let pos = Tensor::from_vec(positions, tokens, &device).unwrap();
    let (q_out, _) = rope.apply(&q, &k, &pos).unwrap();

    let q_in_v: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
    let q_out_v: Vec<f32> = q_out.flatten_all().unwrap().to_vec1().unwrap();
    let head_stride = head_dim;
    for t in 0..tokens {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_stride;
            let n_in: f32 = q_in_v[base..base + head_dim].iter().map(|v| v * v).sum();
            let n_out: f32 = q_out_v[base..base + head_dim].iter().map(|v| v * v).sum();
            let rel = (n_in - n_out).abs() / (n_in.max(1e-6));
            assert!(rel < 1e-4, "norm not preserved t={t} h={h} rel={rel}");
        }
    }
}

#[test]
fn rope_bf16_candle_path_matches_cpu_reference() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let head_dim = 32usize;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let tokens = 4usize;
    let max_pos = 16usize;
    let base = 10_000f32;
    let half = head_dim / 2;
    let (cos_tbl, sin_tbl) = build_rope_tables(max_pos, half, base);

    let mut q_host = Vec::with_capacity(tokens * n_heads * head_dim);
    let mut k_host = Vec::with_capacity(tokens * n_kv * head_dim);
    for i in 0..(tokens * n_heads * head_dim) {
        q_host.push((i as f32 * 0.0011).sin());
    }
    for i in 0..(tokens * n_kv * head_dim) {
        k_host.push((i as f32 * 0.0017).cos());
    }
    let positions: Vec<i32> = (0..tokens as i32).collect();
    let mut q_expect = q_host.clone();
    let mut k_expect = k_host.clone();
    cpu_rope_apply(
        &mut q_expect,
        &positions,
        &cos_tbl,
        &sin_tbl,
        n_heads,
        head_dim,
    );
    cpu_rope_apply(
        &mut k_expect,
        &positions,
        &cos_tbl,
        &sin_tbl,
        n_kv,
        head_dim,
    );

    let cfg = RopeConfig {
        head_dim,
        max_seq_len: max_pos,
        base,
        kind: RopeKind::Standard,
    };
    let rope = Rope::new(cfg, &device).unwrap();

    let q_f32 = Tensor::from_vec(q_host, (tokens, n_heads, head_dim), &device).unwrap();
    let k_f32 = Tensor::from_vec(k_host, (tokens, n_kv, head_dim), &device).unwrap();
    let q_bf = q_f32.to_dtype(DType::BF16).unwrap();
    let k_bf = k_f32.to_dtype(DType::BF16).unwrap();
    let pos = Tensor::from_vec(positions, tokens, &device).unwrap();
    let (q_out, k_out) = rope.apply(&q_bf, &k_bf, &pos).unwrap();
    assert_eq!(q_out.dtype(), DType::BF16);

    let q_got: Vec<f32> = q_out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let k_got: Vec<f32> = k_out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut max_q = 0f32;
    let mut max_k = 0f32;
    for (g, e) in q_got.iter().zip(q_expect.iter()) {
        max_q = max_q.max((g - e).abs());
    }
    for (g, e) in k_got.iter().zip(k_expect.iter()) {
        max_k = max_k.max((g - e).abs());
    }
    assert!(max_q < 0.02, "bf16 candle-rope q drift {max_q}");
    assert!(max_k < 0.02, "bf16 candle-rope k drift {max_k}");
}

#[test]
fn rope_4d_input_shape_preserved() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let cfg = RopeConfig {
        head_dim: 64,
        max_seq_len: 32,
        base: 10_000.0,
        kind: RopeKind::Standard,
    };
    let rope = Rope::new(cfg, &device).unwrap();
    let b = 2usize;
    let t = 8usize;
    let q = Tensor::randn(0f32, 1.0, (b, t, 4usize, 64usize), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (b, t, 2usize, 64usize), &device).unwrap();
    let positions: Vec<i32> = (0..t as i32).collect();
    let mut tiled = Vec::with_capacity(b * t);
    for _ in 0..b {
        tiled.extend_from_slice(&positions);
    }
    let pos = Tensor::from_vec(tiled, (b, t), &device).unwrap();
    let (q_out, k_out) = rope.apply(&q, &k, &pos).unwrap();
    assert_eq!(q_out.dims(), &[b, t, 4, 64]);
    assert_eq!(k_out.dims(), &[b, t, 2, 64]);
}

#[test]
fn rope_deterministic() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let cfg = RopeConfig {
        head_dim: 64,
        max_seq_len: 32,
        base: 10_000.0,
        kind: RopeKind::Standard,
    };
    let rope = Rope::new(cfg, &device).unwrap();
    let q = Tensor::randn(0f32, 1.0, (8usize, 4usize, 64usize), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (8usize, 2usize, 64usize), &device).unwrap();
    let positions: Vec<i32> = (0..8).collect();
    let pos = Tensor::from_vec(positions, 8usize, &device).unwrap();
    let (q1, _) = rope.apply(&q, &k, &pos).unwrap();
    let (q2, _) = rope.apply(&q, &k, &pos).unwrap();
    let v1: Vec<f32> = q1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = q2.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "rope non-deterministic");
    }
}

#[test]
fn rope_rejects_odd_head_dim() {
    let cfg = RopeConfig {
        head_dim: 65,
        max_seq_len: 16,
        base: 10_000.0,
        kind: RopeKind::Standard,
    };
    assert!(Rope::new(cfg, &Device::Cpu).is_err());
}
