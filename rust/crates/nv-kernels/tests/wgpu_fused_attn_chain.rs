#![cfg(feature = "wgpu")]

mod common;
use common::bits;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::{fused_attn_chain, kv_fp8, rmsnorm, rope_bf16};

fn sample(n: usize, seed: u32) -> Vec<u16> {
    let s = seed as f32;
    bits(
        &(0..n)
            .map(|i| ((i as f32) * (0.0011 + s * 0.0005) + s).sin() * (2.3 + s * 0.7))
            .collect::<Vec<f32>>(),
    )
}

fn sample_w(n: usize, seed: u32) -> Vec<u16> {
    let s = seed as f32;
    bits(
        &(0..n)
            .map(|i| 1.0 + ((i as f32) * 0.0017 + s).cos() * 0.5)
            .collect::<Vec<f32>>(),
    )
}

fn tables(head_dim: usize, rows: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; rows * half];
    let mut sin = vec![0f32; rows * half];
    for p in 0..rows {
        for i in 0..half {
            let f = 1.0 / theta.powf((i as f32 * 2.0) / (head_dim as f32));
            let a = (p as f32) * f;
            cos[p * half + i] = a.cos();
            sin[p * half + i] = a.sin();
        }
    }
    (cos, sin)
}

fn bf16_slice_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect()
}

const SHAPES: &[(usize, usize)] = &[(4usize, 64usize), (2, 128), (8, 256), (1, 512), (3, 128)];
const EPS: f32 = 1e-6;
const SLOTS: usize = 16;

#[test]
fn fused_q_matches_rms_then_rope_bitwise() {
    let Some(ctx) = ctx_or_skip("fused_q_matches_rms_then_rope_bitwise") else {
        return;
    };
    for (seed, &(n_heads, head_dim)) in SHAPES.iter().enumerate() {
        for &pos in &[0i32, 1, 7, 13] {
            let seed = seed as u32;
            let x = sample(n_heads * head_dim, seed);
            let w = sample_w(head_dim, seed + 5);
            let (cos, sin) = tables(head_dim, SLOTS, 10000.0);
            let positions = [pos];

            let mut normed = vec![0u16; x.len()];
            rmsnorm::rmsnorm_bf16(ctx, &x, &w, &mut normed, n_heads, head_dim, EPS)
                .expect("ref rms");
            let mut roped = normed.clone();
            let mut no_k: Vec<u16> = Vec::new();
            rope_bf16::rope_bf16(
                ctx, &mut roped, &mut no_k, &cos, &sin, &positions, 1, n_heads, 0, head_dim,
            )
            .expect("ref rope");
            let want = bf16_slice_to_f32(&roped);

            let mut got = vec![0f32; x.len()];
            fused_attn_chain::q_rms_rope_f32(
                ctx, &x, &w, &cos, &sin, &positions, &mut got, 1, n_heads, head_dim, EPS,
            )
            .expect("fused q");

            let diff = want
                .iter()
                .zip(got.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!("q n_heads={n_heads} head_dim={head_dim} pos={pos} diff={diff}");
            assert_eq!(
                diff, 0,
                "fused q mismatch n_heads={n_heads} hd={head_dim} pos={pos}"
            );
        }
    }
}

#[test]
fn fused_k_matches_rms_rope_quant_bitwise() {
    let Some(ctx) = ctx_or_skip("fused_k_matches_rms_rope_quant_bitwise") else {
        return;
    };
    for (seed, &(n_kv, head_dim)) in SHAPES.iter().enumerate() {
        for &(pos, start) in &[
            (0i32, 0i32),
            (3, 3),
            (9, 9),
            (SLOTS as i32 - 1, SLOTS as i32 - 1),
        ] {
            let seed = seed as u32 + 100;
            let x = sample(n_kv * head_dim, seed);
            let w = sample_w(head_dim, seed + 5);
            let (cos, sin) = tables(head_dim, SLOTS, 1000000.0);
            let positions = [pos];
            let starts = [start];

            let mut normed = vec![0u16; x.len()];
            rmsnorm::rmsnorm_bf16(ctx, &x, &w, &mut normed, n_kv, head_dim, EPS).expect("ref rms");
            let mut roped = normed.clone();
            let mut no_q: Vec<u16> = Vec::new();
            rope_bf16::rope_bf16(
                ctx, &mut no_q, &mut roped, &cos, &sin, &positions, 1, 0, n_kv, head_dim,
            )
            .expect("ref rope");
            let mut want_fp8 = vec![0u8; SLOTS * n_kv * head_dim];
            let mut want_scales = vec![0f32; SLOTS * n_kv];
            kv_fp8::quantize_kv_fp8(
                ctx,
                &roped,
                &mut want_fp8,
                &mut want_scales,
                &starts,
                1,
                n_kv,
                head_dim,
                0,
            )
            .expect("ref quant");

            let mut got_fp8 = vec![0u8; SLOTS * n_kv * head_dim];
            let mut got_scales = vec![0f32; SLOTS * n_kv];
            fused_attn_chain::k_rms_rope_fp8(
                ctx,
                &x,
                &w,
                &cos,
                &sin,
                &positions,
                &starts,
                &mut got_fp8,
                &mut got_scales,
                1,
                n_kv,
                head_dim,
                0,
                EPS,
            )
            .expect("fused k");

            let db = want_fp8
                .iter()
                .zip(got_fp8.iter())
                .filter(|(a, b)| a != b)
                .count();
            let ds = want_scales
                .iter()
                .zip(got_scales.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!("k n_kv={n_kv} head_dim={head_dim} pos={pos} start={start} fp8_diff={db} scale_diff={ds}");
            assert_eq!(
                db, 0,
                "fused k fp8 bytes mismatch n_kv={n_kv} hd={head_dim}"
            );
            assert_eq!(ds, 0, "fused k scales mismatch n_kv={n_kv} hd={head_dim}");
        }
    }
}

#[test]
fn fused_v_matches_rms_quant_bitwise() {
    let Some(ctx) = ctx_or_skip("fused_v_matches_rms_quant_bitwise") else {
        return;
    };
    for (seed, &(n_kv, head_dim)) in SHAPES.iter().enumerate() {
        for &start in &[0i32, 5, SLOTS as i32 - 1] {
            let seed = seed as u32 + 200;
            let x = sample(n_kv * head_dim, seed);
            let ones = bits(&vec![1.0f32; head_dim]);
            let starts = [start];

            let mut normed = vec![0u16; x.len()];
            rmsnorm::rmsnorm_bf16(ctx, &x, &ones, &mut normed, n_kv, head_dim, EPS)
                .expect("ref rms");
            let mut want_fp8 = vec![0u8; SLOTS * n_kv * head_dim];
            let mut want_scales = vec![0f32; SLOTS * n_kv];
            kv_fp8::quantize_kv_fp8(
                ctx,
                &normed,
                &mut want_fp8,
                &mut want_scales,
                &starts,
                1,
                n_kv,
                head_dim,
                0,
            )
            .expect("ref quant");

            let mut got_fp8 = vec![0u8; SLOTS * n_kv * head_dim];
            let mut got_scales = vec![0f32; SLOTS * n_kv];
            fused_attn_chain::v_rms_fp8(
                ctx,
                &x,
                &ones,
                &starts,
                &mut got_fp8,
                &mut got_scales,
                1,
                n_kv,
                head_dim,
                0,
                EPS,
            )
            .expect("fused v");

            let db = want_fp8
                .iter()
                .zip(got_fp8.iter())
                .filter(|(a, b)| a != b)
                .count();
            let ds = want_scales
                .iter()
                .zip(got_scales.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "v n_kv={n_kv} head_dim={head_dim} start={start} fp8_diff={db} scale_diff={ds}"
            );
            assert_eq!(
                db, 0,
                "fused v fp8 bytes mismatch n_kv={n_kv} hd={head_dim}"
            );
            assert_eq!(ds, 0, "fused v scales mismatch n_kv={n_kv} hd={head_dim}");
        }
    }
}

#[test]
fn fused_v_zero_row_uses_unit_scale_like_unfused() {
    let Some(ctx) = ctx_or_skip("fused_v_zero_row_uses_unit_scale_like_unfused") else {
        return;
    };
    let (n_kv, head_dim) = (2usize, 64usize);
    let x = vec![0u16; n_kv * head_dim];
    let ones = bits(&vec![1.0f32; head_dim]);
    let starts = [0i32];

    let mut want_fp8 = vec![0xffu8; SLOTS * n_kv * head_dim];
    let mut want_scales = vec![-1f32; SLOTS * n_kv];
    kv_fp8::quantize_kv_fp8(
        ctx,
        &x,
        &mut want_fp8,
        &mut want_scales,
        &starts,
        1,
        n_kv,
        head_dim,
        0,
    )
    .expect("ref quant");

    let mut got_fp8 = vec![0xffu8; SLOTS * n_kv * head_dim];
    let mut got_scales = vec![-1f32; SLOTS * n_kv];
    fused_attn_chain::v_rms_fp8(
        ctx,
        &x,
        &ones,
        &starts,
        &mut got_fp8,
        &mut got_scales,
        1,
        n_kv,
        head_dim,
        0,
        EPS,
    )
    .expect("fused v");

    assert_eq!(want_fp8, got_fp8, "zero-row fp8 bytes must match");
    let ds = want_scales
        .iter()
        .zip(got_scales.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(ds, 0, "zero-row scales must match");
    assert_eq!(got_scales[0].to_bits(), 1.0f32.to_bits());
}
