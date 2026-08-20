#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::{graph_decode, residual_scale};

fn sample_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) as f32 / (u32::MAX >> 1) as f32;
        let v = (u - 1.0) * 60.0;
        let b = match s % 997 {
            0 => bf16::from_f32(f32::INFINITY),
            1 => bf16::from_f32(f32::NEG_INFINITY),
            2 => bf16::from_f32(f32::NAN),
            _ => bf16::from_f32(v + (i % 7) as f32 * 0.125),
        };
        out.push(b.to_bits());
    }
    if n > 64 {
        let hot = bf16::from_f32(57.5).to_bits();
        out[n / 3] = hot;
        out[2 * n / 3] = hot;
    }
    out
}

fn pack(x: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; x.len().div_ceil(2)];
    for (i, w) in out.iter_mut().enumerate() {
        let lo = x[2 * i] as u32;
        let hi = x.get(2 * i + 1).copied().unwrap_or(0) as u32;
        *w = lo | (hi << 16);
    }
    out
}

const SHAPES: &[usize] = &[262144, 32768, 4095, 2048, 511];
const CAPS: &[f32] = &[30.0, 0.0];

#[test]
fn folded_stage1_matches_composed_cap_plus_argmax_bitwise() {
    let Some(ctx) = ctx_or_skip("folded_stage1_matches_composed_cap_plus_argmax_bitwise") else {
        return;
    };
    let blocks = graph_decode::ARGMAX_BLOCKS;
    for (si, &n) in SHAPES.iter().enumerate() {
        for &cap in CAPS {
            let x = sample_bf16(
                n,
                0x9e3779b97f4a7c15 ^ ((si as u64) << 17) ^ cap.to_bits() as u64,
            );
            let xw = pack(&x);

            let mut want_f32 = vec![0f32; n];
            residual_scale::tanh_softcap_bf16_to_f32(ctx, &x, &mut want_f32, cap, n)
                .expect("composed cap/cast");
            let mut want_tok = vec![0u32; 1];
            let mut want_pv = vec![0f32; blocks];
            let mut want_pi = vec![0i32; blocks];
            graph_decode::argmax_f32_rows_with_parts(
                ctx,
                &want_f32,
                &mut want_tok,
                Some(&mut want_pv),
                Some(&mut want_pi),
                1,
                n,
            )
            .expect("composed argmax");

            let mut got_f32 = vec![0f32; n];
            let mut got_tok = vec![0u32; 1];
            let mut got_pv = vec![0f32; blocks];
            let mut got_pi = vec![0i32; blocks];
            graph_decode::argmax_softcap_bf16_fold(
                ctx,
                &xw,
                &mut got_f32,
                &mut got_tok,
                Some(&mut got_pv),
                Some(&mut got_pi),
                n,
                cap,
            )
            .expect("fused fold");

            let logit_diff = want_f32
                .iter()
                .zip(got_f32.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                logit_diff, 0,
                "n={n} cap={cap}: {logit_diff} of {n} f32 logits differ bitwise"
            );
            let pv_diff = want_pv
                .iter()
                .zip(got_pv.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(pv_diff, 0, "n={n} cap={cap}: stage1 part values differ");
            assert_eq!(
                want_pi, got_pi,
                "n={n} cap={cap}: stage1 part indices differ"
            );
            assert_eq!(want_tok, got_tok, "n={n} cap={cap}: argmax token differs");
            eprintln!(
                "n={n} cap={cap}: token {} and {} logits bit-identical (fused vs composed)",
                got_tok[0], n
            );
        }
    }
}

#[test]
fn folded_stage1_breaks_ties_toward_the_lower_index() {
    let Some(ctx) = ctx_or_skip("folded_stage1_breaks_ties_toward_the_lower_index") else {
        return;
    };
    let n = 70000usize;
    let mut x = vec![bf16::from_f32(-3.0).to_bits(); n];
    let hot = bf16::from_f32(21.25).to_bits();
    for idx in [69001usize, 40000, 33000, 1500] {
        x[idx] = hot;
    }
    for &cap in CAPS {
        let mut got_f32 = vec![0f32; n];
        let mut got_tok = vec![0u32; 1];
        graph_decode::argmax_softcap_bf16_fold(
            ctx,
            &pack(&x),
            &mut got_f32,
            &mut got_tok,
            None,
            None,
            n,
            cap,
        )
        .expect("fused fold");
        let mut want_f32 = vec![0f32; n];
        residual_scale::tanh_softcap_bf16_to_f32(ctx, &x, &mut want_f32, cap, n).expect("composed");
        let mut want_tok = vec![0u32; 1];
        graph_decode::argmax_f32_rows(ctx, &want_f32, &mut want_tok, 1, n)
            .expect("composed argmax");
        assert_eq!(
            got_tok[0], 1500,
            "cap={cap}: tie must resolve to the lowest index"
        );
        assert_eq!(
            got_tok, want_tok,
            "cap={cap}: fused vs composed tie-break differs"
        );
    }
}
