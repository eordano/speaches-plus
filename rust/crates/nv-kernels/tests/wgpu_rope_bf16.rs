#![cfg(feature = "wgpu")]

mod common;
use common::require;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::rope_bf16;
use nv_kernels::wgpu_backend::WgpuError;
use common::ctx_or_skip;

fn build_tables(max_pos: usize, half: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let theta = (p as f32) / base.powf((i as f32 * 2.0) / (half as f32 * 2.0));
            cos[p * half + i] = theta.cos();
            sin[p * half + i] = theta.sin();
        }
    }
    (cos, sin)
}

fn sample_bf16(n: usize, seed: f32) -> Vec<u16> {
    (0..n)
        .map(|i| bf16::from_f32(((i as f32) * seed).sin() * 1.75).to_bits())
        .collect()
}

fn cpu_rope_bf16(
    x: &[u16],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    positions: &[i32],
    n_heads: usize,
    head_dim: usize,
) -> Vec<u16> {
    let half = head_dim / 2;
    let mut out = x.to_vec();
    for (t, pos) in positions.iter().enumerate() {
        let row = (*pos as usize) * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let c = cos_tbl[row + i];
                let s = sin_tbl[row + i];
                let a = bf16::from_bits(x[base + i]).to_f32();
                let b = bf16::from_bits(x[base + i + half]).to_f32();
                out[base + i] = bf16::from_f32(a * c - b * s).to_bits();
                out[base + i + half] = bf16::from_f32(a * s + b * c).to_bits();
            }
        }
    }
    out
}

fn report(name: &str, got: &[u16], want: &[u16]) -> (usize, i32, f32) {
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    let mut max_abs = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        if g != w {
            mismatch += 1;
            max_ulp = max_ulp.max((*g as i32 - *w as i32).abs());
            let d = (bf16::from_bits(*g).to_f32() - bf16::from_bits(*w).to_f32()).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    eprintln!(
        "{name}: {mismatch}/{} bf16 words differ, max_ulp={max_ulp}, max_abs={max_abs:e}",
        got.len()
    );
    (mismatch, max_ulp, max_abs)
}

#[test]
fn rope_bf16_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("rope_bf16_matches_cpu_oracle") else {
        return;
    };
    let (batch, n_heads, n_kv_heads, head_dim, max_pos) = (8usize, 16usize, 4usize, 64usize, 32);
    let half = head_dim / 2;
    let (cos_tbl, sin_tbl) = build_tables(max_pos, half, 10_000.0);
    let positions: Vec<i32> = (0..batch)
        .map(|i| (i as i32 * 3) % max_pos as i32)
        .collect();

    let q0 = sample_bf16(batch * n_heads * head_dim, 0.0013);
    let k0 = sample_bf16(batch * n_kv_heads * head_dim, 0.0017);
    let q_want = cpu_rope_bf16(&q0, &cos_tbl, &sin_tbl, &positions, n_heads, head_dim);
    let k_want = cpu_rope_bf16(&k0, &cos_tbl, &sin_tbl, &positions, n_kv_heads, head_dim);

    let mut q = q0.clone();
    let mut k = k0.clone();
    rope_bf16::rope_bf16(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads, head_dim,
    )
    .unwrap();

    let (mq, uq, _) = report("q", &q, &q_want);
    let (mk, uk, _) = report("k", &k, &k_want);
    assert_eq!(mq, 0, "q not bit-exact vs cpu oracle (max_ulp={uq})");
    assert_eq!(mk, 0, "k not bit-exact vs cpu oracle (max_ulp={uk})");
}

#[test]
fn rope_bf16_oop_matches_in_place() {
    let Some(ctx) = ctx_or_skip("rope_bf16_oop_matches_in_place") else {
        return;
    };
    let (batch, n_heads, n_kv_heads, head_dim, max_pos) = (5usize, 6usize, 2usize, 128usize, 17);
    let half = head_dim / 2;
    let (cos_tbl, sin_tbl) = build_tables(max_pos, half, 1_000_000.0);
    let positions: Vec<i32> = (0..batch)
        .map(|i| (i as i32 * 5) % max_pos as i32)
        .collect();

    let q0 = sample_bf16(batch * n_heads * head_dim, 0.0031);
    let k0 = sample_bf16(batch * n_kv_heads * head_dim, 0.0041);
    let q_want = cpu_rope_bf16(&q0, &cos_tbl, &sin_tbl, &positions, n_heads, head_dim);
    let k_want = cpu_rope_bf16(&k0, &cos_tbl, &sin_tbl, &positions, n_kv_heads, head_dim);

    let mut q_out = vec![0u16; q0.len()];
    let mut k_out = vec![0u16; k0.len()];
    rope_bf16::rope_bf16_oop(
        ctx, &q0, &k0, &mut q_out, &mut k_out, &cos_tbl, &sin_tbl, &positions, batch, n_heads,
        n_kv_heads, head_dim,
    )
    .unwrap();

    let mut q_ip = q0.clone();
    let mut k_ip = k0.clone();
    rope_bf16::rope_bf16(
        ctx, &mut q_ip, &mut k_ip, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads,
        head_dim,
    )
    .unwrap();

    assert_eq!(q_out, q_ip, "oop q differs from in-place q");
    assert_eq!(k_out, k_ip, "oop k differs from in-place k");
    let (mq, _, _) = report("oop q", &q_out, &q_want);
    let (mk, _, _) = report("oop k", &k_out, &k_want);
    assert_eq!(mq, 0);
    assert_eq!(mk, 0);
}

#[test]
fn rope_bf16_handles_odd_half_dim_and_single_head() {
    let Some(ctx) = ctx_or_skip("rope_bf16_handles_odd_half_dim_and_single_head") else {
        return;
    };
    for (n_heads, n_kv_heads, head_dim) in [(1usize, 1usize, 6usize), (3, 1, 2), (2, 0, 10)] {
        let batch = 4usize;
        let max_pos = 9usize;
        let half = head_dim / 2;
        let (cos_tbl, sin_tbl) = build_tables(max_pos, half, 500.0);
        let positions: Vec<i32> = (0..batch)
            .map(|i| (i as i32 * 2) % max_pos as i32)
            .collect();
        let q0 = sample_bf16(batch * n_heads * head_dim, 0.07);
        let k0 = sample_bf16(batch * n_kv_heads * head_dim, 0.11);
        let q_want = cpu_rope_bf16(&q0, &cos_tbl, &sin_tbl, &positions, n_heads, head_dim);
        let k_want = cpu_rope_bf16(&k0, &cos_tbl, &sin_tbl, &positions, n_kv_heads, head_dim);
        let mut q = q0.clone();
        let mut k = k0.clone();
        rope_bf16::rope_bf16(
            ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, n_kv_heads,
            head_dim,
        )
        .unwrap();
        assert_eq!(q, q_want, "q mismatch for head_dim={head_dim}");
        assert_eq!(k, k_want, "k mismatch for head_dim={head_dim}");
    }
}

#[test]
fn rope_bf16_folds_past_the_workgroup_limit() {
    let Some(ctx) = ctx_or_skip("rope_bf16_folds_past_the_workgroup_limit") else {
        return;
    };
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    let (n_heads, head_dim, max_pos) = (2usize, 8usize, 64usize);
    let half = head_dim / 2;
    let words_per_token = n_heads * half;
    let batch = (limit * 256 / words_per_token) + 37;
    let (cos_tbl, sin_tbl) = build_tables(max_pos, half, 10_000.0);
    let positions: Vec<i32> = (0..batch).map(|i| (i % max_pos) as i32).collect();
    let q0 = sample_bf16(batch * n_heads * head_dim, 0.0009);

    let mut q = q0.clone();
    let mut k: Vec<u16> = Vec::new();
    rope_bf16::rope_bf16(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, batch, n_heads, 0, head_dim,
    )
    .unwrap();

    let mut checked = 0usize;
    for t in (0..batch).step_by(4093).chain([batch - 1, batch - 2]) {
        let pos = positions[t] as usize;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let c = cos_tbl[pos * half + i];
                let s = sin_tbl[pos * half + i];
                let a = bf16::from_bits(q0[base + i]).to_f32();
                let b = bf16::from_bits(q0[base + i + half]).to_f32();
                assert_eq!(
                    q[base + i],
                    bf16::from_f32(a * c - b * s).to_bits(),
                    "lo half at token {t} head {h} pair {i}"
                );
                assert_eq!(
                    q[base + i + half],
                    bf16::from_f32(a * s + b * c).to_bits(),
                    "hi half at token {t} head {h} pair {i}"
                );
                checked += 2;
            }
        }
    }
    eprintln!("folded dispatch: batch={batch}, checked {checked} elements");
    assert!(checked > 0);
}

#[test]
fn rope_bf16_rejects_bad_shapes() {
    let Some(ctx) = ctx_or_skip("rope_bf16_rejects_bad_shapes") else {
        return;
    };
    let (cos_tbl, sin_tbl) = build_tables(4, 2, 100.0);
    let positions = vec![0i32, 1];
    let mut q = vec![0u16; 2 * 4];
    let mut k = vec![0u16; 2 * 4];

    let e = rope_bf16::rope_bf16(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &positions, 2, 1, 1, 5,
    )
    .unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");

    let bad_pos = vec![0i32, 9];
    let e = rope_bf16::rope_bf16(
        ctx, &mut q, &mut k, &cos_tbl, &sin_tbl, &bad_pos, 2, 1, 1, 4,
    )
    .unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");

    let mut short = vec![0u16; 3];
    let e = rope_bf16::rope_bf16(
        ctx, &mut short, &mut k, &cos_tbl, &sin_tbl, &positions, 2, 1, 1, 4,
    )
    .unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
}
