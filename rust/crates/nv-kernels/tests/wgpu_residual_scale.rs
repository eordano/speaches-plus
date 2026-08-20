#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::residual_scale;
use nv_kernels::wgpu_backend::WgpuError;
use common::bits as to_bits;

fn from_bits(v: &[u16]) -> Vec<f32> {
    v.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

fn cpu_residual_add_scale(a: &[u16], b: &[u16], scale: f32) -> Vec<u16> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let av = bf16::from_bits(*x).to_f32();
            let bv = bf16::from_bits(*y).to_f32();
            bf16::from_f32((av + bv) * scale).to_bits()
        })
        .collect()
}

fn cpu_scale(x: &[u16], scale: f32) -> Vec<u16> {
    x.iter()
        .map(|v| bf16::from_f32(bf16::from_bits(*v).to_f32() * scale).to_bits())
        .collect()
}

fn cpu_softcap(x: &[u16], cap: f32) -> Vec<f32> {
    let softcap = cap > 0.0 && cap.is_finite();
    let inv_cap = if softcap { 1.0f32 / cap } else { 0.0 };
    x.iter()
        .map(|v| {
            let f = bf16::from_bits(*v).to_f32();
            if softcap {
                (f * inv_cap).tanh() * cap
            } else {
                f
            }
        })
        .collect()
}

fn sample_a(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.00037).sin() * 3.5 - 0.25)
        .collect()
}

fn sample_b(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.00091).cos() * 1.75 + 0.125)
        .collect()
}

fn assert_bit_exact(name: &str, got: &[u16], want: &[u16]) {
    let mut mismatch = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            mismatch += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    eprintln!("{name}: {mismatch}/{} bf16 words differ", want.len());
    assert_eq!(mismatch, 0, "{name}: first mismatch {first:?}");
}

#[test]
fn residual_add_scale_matches_cpu_oracle_bit_exactly() {
    let Some(ctx) = ctx_or_skip("residual_add_scale") else {
        return;
    };
    for n in [1usize, 2, 3, 255, 256, 257, 4096, 12289] {
        let a = to_bits(&sample_a(n));
        let b = to_bits(&sample_b(n));
        for scale in [1.0f32, 0.5, std::f32::consts::SQRT_2, -2.25, 0.0] {
            let want = cpu_residual_add_scale(&a, &b, scale);
            let mut got = vec![0u16; n];
            residual_scale::residual_add_scale_bf16(ctx, &a, &b, &mut got, scale, n).unwrap();
            assert_bit_exact(
                &format!("residual_add_scale n={n} scale={scale}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn scale_out_matches_cpu_oracle_bit_exactly() {
    let Some(ctx) = ctx_or_skip("scale_out") else {
        return;
    };
    for n in [1usize, 5, 512, 513, 8191] {
        let x = to_bits(&sample_a(n));
        for scale in [1.0f32, 0.125, -std::f32::consts::FRAC_1_SQRT_2, 3.0] {
            let want = cpu_scale(&x, scale);
            let mut got = vec![0u16; n];
            residual_scale::scale_out_bf16(ctx, &x, &mut got, scale, n).unwrap();
            assert_bit_exact(&format!("scale_out n={n} scale={scale}"), &got, &want);
        }
    }
}

#[test]
fn scale_inplace_matches_cpu_oracle_bit_exactly() {
    let Some(ctx) = ctx_or_skip("scale_inplace") else {
        return;
    };
    for n in [1usize, 7, 1024, 1025] {
        let x = to_bits(&sample_b(n));
        for scale in [1.0f32, 0.25, -1.5] {
            let want = cpu_scale(&x, scale);
            let mut got = x.clone();
            residual_scale::scale_inplace_bf16(ctx, &mut got, scale, n).unwrap();
            assert_bit_exact(&format!("scale_inplace n={n} scale={scale}"), &got, &want);
        }
    }
}

#[test]
fn scale_of_special_values_is_bit_exact() {
    let Some(ctx) = ctx_or_skip("scale_specials") else {
        return;
    };
    let vals: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        3.0,
        65504.0,
        1e-30,
        -1e-30,
        1.5,
        2.5,
        1.000_976_6,
    ];
    let x = to_bits(&vals);
    let n = x.len();
    for scale in [1.0f32, 2.0, 0.5, -1.0, 0.75] {
        let want = cpu_scale(&x, scale);
        let mut got = vec![0u16; n];
        residual_scale::scale_out_bf16(ctx, &x, &mut got, scale, n).unwrap();
        assert_bit_exact(&format!("scale_specials scale={scale}"), &got, &want);
    }
}

#[test]
fn tanh_softcap_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("tanh_softcap") else {
        return;
    };
    let n = 4097usize;
    let xf: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0013).sin() * 90.0).collect();
    let x = to_bits(&xf);
    for cap in [30.0f32, 50.0, 1.0] {
        let want = cpu_softcap(&x, cap);
        let mut got = vec![0f32; n];
        residual_scale::tanh_softcap_bf16_to_f32(ctx, &x, &mut got, cap, n).unwrap();
        let max_abs = got
            .iter()
            .zip(want.iter())
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        eprintln!("tanh_softcap cap={cap}: max_abs={max_abs:e}");
        assert!(
            max_abs < 1e-5 * cap.max(1.0),
            "cap={cap} max_abs={max_abs:e}"
        );
    }
}

#[test]
fn tanh_softcap_disabled_is_a_pure_cast() {
    let Some(ctx) = ctx_or_skip("tanh_softcap_off") else {
        return;
    };
    let n = 1023usize;
    let xf: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.002).cos() * 12.0).collect();
    let x = to_bits(&xf);
    let want = from_bits(&x);
    for cap in [0.0f32, -1.0, f32::INFINITY] {
        let mut got = vec![0f32; n];
        residual_scale::tanh_softcap_bf16_to_f32(ctx, &x, &mut got, cap, n).unwrap();
        assert_eq!(got, want, "cap={cap} must be a bit-exact widening cast");
    }
}

#[test]
fn zero_length_is_a_no_op_and_shape_errors_report() {
    let Some(ctx) = ctx_or_skip("residual_scale_edges") else {
        return;
    };
    let mut empty: Vec<u16> = Vec::new();
    residual_scale::scale_inplace_bf16(ctx, &mut empty, 2.0, 0).unwrap();
    residual_scale::residual_add_scale_bf16(ctx, &[], &[], &mut empty, 2.0, 0).unwrap();

    let mut y = vec![0u16; 3];
    let e = residual_scale::scale_out_bf16(ctx, &[1, 2], &mut y, 1.0, 3).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
}
