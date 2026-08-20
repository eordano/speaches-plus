#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::max_abs_err;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::rope;

pub fn cpu_rope(
    q: &mut [f32],
    k: &mut [f32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let half_dim = head_dim / 2;
    for token_idx in 0..batch {
        let pos = positions[token_idx] as usize;
        for head_idx in 0..(n_heads + n_kv_heads) {
            for pair_idx in 0..half_dim {
                let c = cos_tbl[pos * half_dim + pair_idx];
                let s = sin_tbl[pos * half_dim + pair_idx];
                if head_idx < n_heads {
                    let base = (token_idx * n_heads + head_idx) * head_dim;
                    let a = q[base + pair_idx];
                    let b = q[base + pair_idx + half_dim];
                    q[base + pair_idx] = a * c - b * s;
                    q[base + pair_idx + half_dim] = a * s + b * c;
                } else {
                    let kv_head = head_idx - n_heads;
                    let base = (token_idx * n_kv_heads + kv_head) * head_dim;
                    let a = k[base + pair_idx];
                    let b = k[base + pair_idx + half_dim];
                    k[base + pair_idx] = a * c - b * s;
                    k[base + pair_idx + half_dim] = a * s + b * c;
                }
            }
        }
    }
}

pub fn tables(rows: usize, half_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos_tbl = vec![0f32; rows * half_dim];
    let mut sin_tbl = vec![0f32; rows * half_dim];
    for p in 0..rows {
        for i in 0..half_dim {
            let inv_freq = 1.0f32 / base.powf(2.0 * i as f32 / (2 * half_dim) as f32);
            let angle = p as f32 * inv_freq;
            cos_tbl[p * half_dim + i] = angle.cos();
            sin_tbl[p * half_dim + i] = angle.sin();
        }
    }
    (cos_tbl, sin_tbl)
}

fn sample(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.0013 + phase).sin() * 2.0)
        .collect()
}

fn exact_count(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want.iter())
        .filter(|(g, e)| g.to_bits() == e.to_bits())
        .count()
}

#[test]
fn wgpu_rope_f32_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_matches_cpu_reference") else {
        return;
    };
    for (batch, n_heads, n_kv_heads, head_dim) in [
        (1usize, 1usize, 1usize, 64usize),
        (5, 32, 8, 128),
        (13, 12, 2, 256),
        (3, 4, 4, 2),
    ] {
        let half_dim = head_dim / 2;
        let rows = 4096usize;
        let (cos_tbl, sin_tbl) = tables(rows, half_dim, 10000.0);
        let positions: Vec<i32> = (0..batch).map(|i| ((i * 977) % rows) as i32).collect();

        let q0 = sample(batch * n_heads * head_dim, 0.0);
        let k0 = sample(batch * n_kv_heads * head_dim, 1.7);
        let mut q = q0.clone();
        let mut k = k0.clone();
        rope::rope_f32(
            ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads,
            head_dim,
        )
        .expect("rope f32");

        let mut wq = q0.clone();
        let mut wk = k0.clone();
        cpu_rope(
            &mut wq, &mut wk, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
        );

        let eq = max_abs_err(&q, &wq);
        let ek = max_abs_err(&k, &wk);
        let exq = exact_count(&q, &wq);
        eprintln!(
            "rope b={batch} h={n_heads} kv={n_kv_heads} d={head_dim}: max_abs q={eq:e} k={ek:e} bit_exact_q={exq}/{}",
            q.len()
        );
        assert!(eq < 1e-5, "q max abs error {eq}");
        assert!(ek < 1e-5, "k max abs error {ek}");
    }
}

#[test]
fn wgpu_rope_f32_preserves_pair_norm() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_preserves_pair_norm") else {
        return;
    };
    let (batch, n_heads, n_kv_heads, head_dim) = (4usize, 8usize, 2usize, 128usize);
    let half_dim = head_dim / 2;
    let (cos_tbl, sin_tbl) = tables(512, half_dim, 10000.0);
    let positions: Vec<i32> = (0..batch).map(|i| (i * 37 + 1) as i32).collect();

    let q0 = sample(batch * n_heads * head_dim, 0.3);
    let k0 = sample(batch * n_kv_heads * head_dim, 2.1);
    let mut q = q0.clone();
    let mut k = k0.clone();
    rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
    )
    .expect("rope f32");

    let mut worst = 0f32;
    for i in 0..q0.len() / head_dim {
        for p in 0..half_dim {
            let a0 = q0[i * head_dim + p];
            let b0 = q0[i * head_dim + p + half_dim];
            let a1 = q[i * head_dim + p];
            let b1 = q[i * head_dim + p + half_dim];
            let n0 = a0 * a0 + b0 * b0;
            let n1 = a1 * a1 + b1 * b1;
            worst = worst.max((n0 - n1).abs());
        }
    }
    eprintln!("rotation norm drift={worst:e}");
    assert!(
        worst < 1e-4,
        "rope is not a rotation: pair norm drift {worst}"
    );
}

#[test]
fn wgpu_rope_f32_uses_the_halved_convention_not_interleaved() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_uses_the_halved_convention_not_interleaved") else {
        return;
    };
    let (batch, n_heads, n_kv_heads, head_dim) = (1usize, 1usize, 0usize, 4usize);
    let half_dim = head_dim / 2;
    let cos_tbl = vec![0f32, 0f32];
    let sin_tbl = vec![1f32, 1f32];
    let positions = vec![0i32];
    let mut q = vec![1f32, 2f32, 3f32, 4f32];
    let mut k: Vec<f32> = Vec::new();
    rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
    )
    .expect("rope f32");

    assert_eq!(half_dim, 2);
    assert_eq!(
        q,
        vec![-3.0, -4.0, 1.0, 2.0],
        "quarter turn must rotate element i against element i+head_dim/2, not i+1"
    );
}

#[test]
fn wgpu_rope_f32_position_zero_is_identity() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_position_zero_is_identity") else {
        return;
    };
    let (batch, n_heads, n_kv_heads, head_dim) = (2usize, 3usize, 1usize, 64usize);
    let half_dim = head_dim / 2;
    let (cos_tbl, sin_tbl) = tables(8, half_dim, 10000.0);
    let positions = vec![0i32; batch];
    let q0 = sample(batch * n_heads * head_dim, 0.9);
    let k0 = sample(batch * n_kv_heads * head_dim, 1.3);
    let mut q = q0.clone();
    let mut k = k0.clone();
    rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
    )
    .expect("rope f32");
    assert_eq!(
        q, q0,
        "position 0 has cos=1 sin=0 and must be a bit-exact no-op"
    );
    assert_eq!(
        k, k0,
        "position 0 has cos=1 sin=0 and must be a bit-exact no-op"
    );
}

#[test]
fn wgpu_rope_f32_only_touches_the_requested_heads() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_only_touches_the_requested_heads") else {
        return;
    };
    let (batch, n_heads, head_dim) = (3usize, 5usize, 32usize);
    let half_dim = head_dim / 2;
    let (cos_tbl, sin_tbl) = tables(64, half_dim, 10000.0);
    let positions: Vec<i32> = (0..batch).map(|i| (i + 7) as i32).collect();
    let q0 = sample(batch * n_heads * head_dim, 0.1);
    let mut q = q0.clone();
    let mut k: Vec<f32> = Vec::new();
    rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, 0, head_dim,
    )
    .expect("rope f32 with no kv heads");

    let mut wq = q0.clone();
    let mut wk: Vec<f32> = Vec::new();
    cpu_rope(
        &mut wq, &mut wk, &cos_tbl, &sin_tbl, &positions, batch, n_heads, 0, head_dim,
    );
    let err = max_abs_err(&q, &wq);
    eprintln!("no-kv max_abs={err:e}");
    assert!(err < 1e-5, "max abs error {err}");
    assert!(k.is_empty());
}

#[test]
fn wgpu_rope_f32_large_dispatch_folds_past_the_workgroup_limit() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_large_dispatch_folds_past_the_workgroup_limit")
    else {
        return;
    };
    let (batch, n_heads, n_kv_heads, head_dim) = (20000usize, 16usize, 16usize, 64usize);
    let half_dim = head_dim / 2;
    let total_pairs = batch * (n_heads + n_kv_heads) * half_dim;
    let groups = total_pairs.div_ceil(256);
    eprintln!(
        "total_pairs={total_pairs} groups={groups} limit={}",
        ctx.caps.max_compute_workgroups_per_dimension
    );
    assert!(groups > ctx.caps.max_compute_workgroups_per_dimension as usize);

    let (cos_tbl, sin_tbl) = tables(4096, half_dim, 10000.0);
    let positions: Vec<i32> = (0..batch).map(|i| (i % 4096) as i32).collect();
    let q0 = sample(batch * n_heads * head_dim, 0.5);
    let k0 = sample(batch * n_kv_heads * head_dim, 1.1);
    let mut q = q0.clone();
    let mut k = k0.clone();
    rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
    )
    .expect("rope f32 large");

    let mut wq = q0.clone();
    let mut wk = k0.clone();
    cpu_rope(
        &mut wq, &mut wk, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
    );
    let eq = max_abs_err(&q, &wq);
    let ek = max_abs_err(&k, &wk);
    eprintln!("folded dispatch max_abs q={eq:e} k={ek:e}");
    assert!(eq < 1e-5 && ek < 1e-5, "q={eq} k={ek}");
}

#[test]
fn wgpu_rope_f32_rejects_bad_shapes() {
    let Some(ctx) = ctx_or_skip("wgpu_rope_f32_rejects_bad_shapes") else {
        return;
    };
    let cos_tbl = vec![1f32; 8];
    let sin_tbl = vec![0f32; 8];
    let positions = vec![0i32, 1];

    let mut q = vec![0f32; 8];
    let mut k: Vec<f32> = Vec::new();
    let e = rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, 2, 1, 0, 3,
    )
    .unwrap_err();
    eprintln!("odd head_dim rejection: {e}");

    let mut q = vec![0f32; 7];
    let e = rope::rope_f32(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, 2, 1, 0, 4,
    )
    .unwrap_err();
    eprintln!("q length rejection: {e}");

    let mut q = vec![0f32; 8];
    let bad_positions = vec![0i32, 99];
    let e = rope::rope_f32(
        ctx,
        &mut q,
        &mut k,
        &cos_tbl,
        &sin_tbl,
        &bad_positions,
        2,
        1,
        0,
        4,
    )
    .unwrap_err();
    eprintln!("position range rejection: {e}");
}
