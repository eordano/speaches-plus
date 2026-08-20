#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::depthwise_conv1d_silu_bf16 as dwc;

fn cpu_oracle(x: &[u16], w: &[u16], b: usize, c: usize, t: usize, k: usize) -> Vec<u16> {
    let mut y = vec![0u16; b * c * t];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..t {
                let mut acc = 0f32;
                for kk in 0..k {
                    let src = ti as isize - (k as isize - 1) + kk as isize;
                    if src >= 0 {
                        let xv = bf16::from_bits(x[(bi * c + ci) * t + src as usize]).to_f32();
                        let wv = bf16::from_bits(w[ci * k + kk]).to_f32();
                        acc = xv.mul_add(wv, acc);
                    }
                }
                let sig = 1.0f32 / (1.0f32 + (-acc).exp());
                y[(bi * c + ci) * t + ti] = bf16::from_f32(acc * sig).to_bits();
            }
        }
    }
    y
}

fn gen_x(n: usize, phase: f32) -> Vec<u16> {
    (0..n)
        .map(|i| bf16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.5).to_bits())
        .collect()
}

fn gen_w(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| bf16::from_f32((((i as f32) * 0.07).cos() - 0.3) * 0.4).to_bits())
        .collect()
}

fn compare(label: &str, got: &[u16], want: &[u16]) -> (usize, i32, f32) {
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    let mut max_abs = 0f32;
    for (g, r) in got.iter().zip(want.iter()) {
        if g != r {
            mismatch += 1;
            max_ulp = max_ulp.max((*g as i32 - *r as i32).abs());
        }
        let d = (bf16::from_bits(*g).to_f32() - bf16::from_bits(*r).to_f32()).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    eprintln!(
        "{label}: {mismatch}/{} bf16 words differ, max_ulp={max_ulp}, max_abs={max_abs:.3e}",
        want.len()
    );
    (mismatch, max_ulp, max_abs)
}

fn run_case(ctx: &WgpuContext, label: &str, b: usize, c: usize, t: usize, k: usize, phase: f32) {
    let n = b * c * t;
    let x = gen_x(n, phase);
    let w = gen_w(c * k);
    let want = cpu_oracle(&x, &w, b, c, t, k);
    let mut got = vec![0u16; n];
    dwc::depthwise_conv1d_silu_bf16(ctx, &x, &w, &mut got, b, c, t, k).unwrap();
    let (mismatch, max_ulp, max_abs) = compare(label, &got, &want);
    assert!(max_ulp <= 1, "{label}: bf16 ulp {max_ulp} > 1");
    assert!(
        mismatch * 1000 <= n.max(1000),
        "{label}: {mismatch}/{n} words differ, over 0.1%"
    );
    assert!(max_abs < 1e-2, "{label}: max_abs {max_abs} too high");
}

#[test]
fn decode_step_t1_k4() {
    let Some(ctx) = ctx_or_skip("decode_step_t1_k4") else {
        return;
    };
    run_case(ctx, "decode T=1", 1, 6144, 1, 4, 0.0);
}

#[test]
fn prefill_t8_k4() {
    let Some(ctx) = ctx_or_skip("prefill_t8_k4") else {
        return;
    };
    run_case(ctx, "prefill T=8", 1, 6144, 8, 4, 0.3);
}

#[test]
fn prefill_t128_k4() {
    let Some(ctx) = ctx_or_skip("prefill_t128_k4") else {
        return;
    };
    run_case(ctx, "prefill T=128", 1, 6144, 128, 4, 0.7);
}

#[test]
fn odd_shapes_and_kernel_widths() {
    let Some(ctx) = ctx_or_skip("odd_shapes_and_kernel_widths") else {
        return;
    };
    run_case(ctx, "B3 C5 T7 K3", 3, 5, 7, 3, 1.1);
    run_case(ctx, "B1 C1 T1 K1", 1, 1, 1, 1, 0.5);
    run_case(ctx, "B2 C3 T5 K2", 2, 3, 5, 2, 2.0);
    run_case(ctx, "B1 C7 T9 K5", 1, 7, 9, 5, 0.9);
    run_case(ctx, "B1 C4 T6 K7", 1, 4, 6, 7, 1.7);
}

#[test]
fn left_edge_is_zero_padded_not_clamped() {
    let Some(ctx) = ctx_or_skip("left_edge_is_zero_padded_not_clamped") else {
        return;
    };
    let (b, c, t, k) = (1usize, 1usize, 4usize, 4usize);
    let x: Vec<u16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|v| bf16::from_f32(*v).to_bits())
        .collect();
    let w: Vec<u16> = [1.0f32, 2.0, 4.0, 8.0]
        .iter()
        .map(|v| bf16::from_f32(*v).to_bits())
        .collect();
    let mut got = vec![0u16; b * c * t];
    dwc::depthwise_conv1d_silu_bf16(ctx, &x, &w, &mut got, b, c, t, k).unwrap();
    let want = cpu_oracle(&x, &w, b, c, t, k);
    assert_eq!(got, want, "zero-padded causal edge mismatch");
    let acc: Vec<f32> = got.iter().map(|v| bf16::from_bits(*v).to_f32()).collect();
    let silu = |a: f32| a / (1.0 + (-a).exp());
    for (i, expect) in [8.0f32, 4.0, 2.0, 1.0].iter().enumerate() {
        let want = bf16::from_f32(silu(*expect)).to_f32();
        assert!(
            (acc[i] - want).abs() <= want.abs() * 0.01,
            "t={i}: got {} want {want}",
            acc[i]
        );
    }
}

#[test]
fn zero_sized_dims_are_a_no_op() {
    let Some(ctx) = ctx_or_skip("zero_sized_dims_are_a_no_op") else {
        return;
    };
    let mut y: Vec<u16> = vec![0xdead; 4];
    dwc::depthwise_conv1d_silu_bf16(ctx, &[], &[], &mut y, 0, 4, 4, 4).unwrap();
    assert_eq!(y, vec![0xdead; 4]);
}

#[test]
fn shape_mismatch_is_reported() {
    let Some(ctx) = ctx_or_skip("shape_mismatch_is_reported") else {
        return;
    };
    let mut y = vec![0u16; 8];
    let err = dwc::depthwise_conv1d_silu_bf16(ctx, &[0u16; 7], &[0u16; 8], &mut y, 1, 2, 4, 4)
        .unwrap_err();
    assert!(format!("{err}").contains("shape mismatch"), "{err}");
}

#[test]
fn large_negative_activations_saturate_toward_zero() {
    let Some(ctx) = ctx_or_skip("large_negative_activations_saturate_toward_zero") else {
        return;
    };
    let (b, c, t, k) = (1usize, 2usize, 8usize, 3usize);
    let n = b * c * t;
    let x: Vec<u16> = (0..n)
        .map(|i| bf16::from_f32(((i as f32) * 0.31).sin() * 12.0).to_bits())
        .collect();
    let w: Vec<u16> = (0..c * k)
        .map(|i| bf16::from_f32(((i as f32) * 0.9).cos() * 3.0).to_bits())
        .collect();
    let want = cpu_oracle(&x, &w, b, c, t, k);
    let mut got = vec![0u16; n];
    dwc::depthwise_conv1d_silu_bf16(ctx, &x, &w, &mut got, b, c, t, k).unwrap();
    let (_, max_ulp, max_abs) = compare("wide range", &got, &want);
    assert!(max_ulp <= 1, "wide range: bf16 ulp {max_ulp} > 1");
    assert!(max_abs < 1e-1, "wide range: max_abs {max_abs}");
}
