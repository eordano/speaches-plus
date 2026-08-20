#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::moe_unpermute_scatter as mus;
use nv_kernels::wgpu_backend::WgpuError;

fn cpu_oracle(
    y_sorted: &[u16],
    routing_weights: &[f32],
    inv_perm: &[i32],
    n_tokens: usize,
    k: usize,
    hidden: usize,
    stride: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; n_tokens * hidden];
    for n in 0..n_tokens {
        for h in 0..hidden {
            let mut acc = 0f32;
            for s in 0..k {
                let slot = n * k + s;
                let row = inv_perm[slot] as usize;
                let w = routing_weights[slot];
                let v = bf16::from_bits(y_sorted[row * stride + h]).to_f32();
                acc = w.mul_add(v, acc);
            }
            out[n * hidden + h] = acc;
        }
    }
    out
}

fn sample_rows(rows: usize, stride: usize) -> Vec<u16> {
    (0..rows * stride)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 1.5).to_bits())
        .collect()
}

fn sample_weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 + ((i as f32) * 0.071).cos().abs() * 0.4)
        .collect()
}

fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0f32;
    let mut exact = 0usize;
    for (g, w) in got.iter().zip(want.iter()) {
        if g.to_bits() == w.to_bits() {
            exact += 1;
        }
        max_abs = max_abs.max((g - w).abs());
    }
    eprintln!(
        "{name}: max_abs={max_abs:e} bit_exact={exact}/{}",
        want.len()
    );
    (max_abs, exact)
}

#[test]
fn matches_cpu_oracle_dense() {
    let Some(ctx) = ctx_or_skip("mus_dense") else {
        return;
    };
    let (n_tokens, k, hidden) = (8usize, 4usize, 512usize);
    let stride = hidden;
    let rows = n_tokens * k;

    let mut inv_perm: Vec<i32> = (0..rows as i32).collect();
    inv_perm.swap(0, 7);
    inv_perm.swap(3, 12);
    inv_perm.swap(5, 22);

    let y_sorted = sample_rows(rows, stride);
    let weights = sample_weights(rows);
    let want = cpu_oracle(&y_sorted, &weights, &inv_perm, n_tokens, k, hidden, stride);

    let mut got = vec![0f32; n_tokens * hidden];
    mus::moe_unpermute_scatter(
        ctx, &y_sorted, &weights, &inv_perm, &mut got, n_tokens, k, hidden, stride,
    )
    .unwrap();

    let (max_abs, exact) = compare("mus_dense", &got, &want);
    assert_eq!(exact, want.len(), "max_abs={max_abs:e}");
}

#[test]
fn matches_cpu_oracle_odd_stride() {
    let Some(ctx) = ctx_or_skip("mus_odd_stride") else {
        return;
    };
    let (n_tokens, k, hidden) = (5usize, 3usize, 131usize);
    let stride = 133usize;
    let rows = n_tokens * k;

    let inv_perm: Vec<i32> = (0..rows).map(|i| ((i * 7 + 1) % rows) as i32).collect();
    let y_sorted = sample_rows(rows, stride);
    let weights = sample_weights(rows);
    let want = cpu_oracle(&y_sorted, &weights, &inv_perm, n_tokens, k, hidden, stride);

    let mut got = vec![0f32; n_tokens * hidden];
    mus::moe_unpermute_scatter(
        ctx, &y_sorted, &weights, &inv_perm, &mut got, n_tokens, k, hidden, stride,
    )
    .unwrap();

    let (max_abs, exact) = compare("mus_odd_stride", &got, &want);
    assert_eq!(exact, want.len(), "max_abs={max_abs:e}");
}

#[test]
fn matches_cpu_oracle_repeated_rows() {
    let Some(ctx) = ctx_or_skip("mus_repeated") else {
        return;
    };
    let (n_tokens, k, hidden) = (4usize, 8usize, 1000usize);
    let stride = hidden;
    let rows = 6usize;

    let inv_perm: Vec<i32> = (0..n_tokens * k).map(|i| (i % rows) as i32).collect();
    let y_sorted = sample_rows(rows, stride);
    let weights = sample_weights(n_tokens * k);
    let want = cpu_oracle(&y_sorted, &weights, &inv_perm, n_tokens, k, hidden, stride);

    let mut got = vec![0f32; n_tokens * hidden];
    mus::moe_unpermute_scatter(
        ctx, &y_sorted, &weights, &inv_perm, &mut got, n_tokens, k, hidden, stride,
    )
    .unwrap();

    let (max_abs, exact) = compare("mus_repeated", &got, &want);
    assert_eq!(exact, want.len(), "max_abs={max_abs:e}");
}

#[test]
fn folds_past_the_workgroup_dispatch_limit() {
    let Some(ctx) = ctx_or_skip("mus_fold") else {
        return;
    };
    let (n_tokens, k, hidden) = (40_000usize, 1usize, 257usize);
    let stride = hidden;
    let rows = 97usize;

    let inv_perm: Vec<i32> = (0..n_tokens * k).map(|i| (i % rows) as i32).collect();
    let y_sorted = sample_rows(rows, stride);
    let weights = sample_weights(n_tokens * k);
    let want = cpu_oracle(&y_sorted, &weights, &inv_perm, n_tokens, k, hidden, stride);

    let mut got = vec![0f32; n_tokens * hidden];
    mus::moe_unpermute_scatter(
        ctx, &y_sorted, &weights, &inv_perm, &mut got, n_tokens, k, hidden, stride,
    )
    .unwrap();

    let (max_abs, exact) = compare("mus_fold", &got, &want);
    assert_eq!(exact, want.len(), "max_abs={max_abs:e}");
}

#[test]
fn degenerate_shapes_are_noops() {
    let Some(ctx) = ctx_or_skip("mus_degenerate") else {
        return;
    };
    let mut out: Vec<f32> = Vec::new();
    mus::moe_unpermute_scatter(ctx, &[], &[], &[], &mut out, 0, 4, 4, 4).unwrap();
    assert!(out.is_empty());

    let mut out = vec![7f32; 8];
    mus::moe_unpermute_scatter(ctx, &[], &[], &[], &mut out, 2, 0, 4, 4).unwrap();
    assert_eq!(out, vec![7f32; 8]);

    let mut out: Vec<f32> = Vec::new();
    mus::moe_unpermute_scatter(ctx, &[], &[0.0; 4], &[0; 4], &mut out, 2, 2, 0, 0).unwrap();
    assert!(out.is_empty());
}

#[test]
fn bad_shapes_are_rejected() {
    let Some(ctx) = ctx_or_skip("mus_bad_shapes") else {
        return;
    };
    let mut out = vec![0f32; 8];
    let e = mus::moe_unpermute_scatter(ctx, &[0; 16], &[0.0; 3], &[0; 4], &mut out, 2, 2, 4, 4)
        .unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");

    let e =
        mus::moe_unpermute_scatter(ctx, &[0; 8], &[0.0; 4], &[0, 1, 2, 9], &mut out, 2, 2, 4, 4)
            .unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
}
