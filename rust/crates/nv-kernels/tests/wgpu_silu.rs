#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::require;
use common::to_bf16;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::silu;
use common::cpu_silu_mul;

fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
}

fn wave(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.0017 + phase).sin() * 6.0)
        .collect()
}

fn oracle_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.0017).sin() * 4.0).collect()
}

fn oracle_gate(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.0023).cos() * 3.0).collect()
}

fn from_bf16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

#[test]
fn wgpu_silu_f32_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_f32_matches_cpu_reference") else {
        return;
    };
    let n = 4096usize;
    let x = wave(n, 0.0);
    let want = cpu_silu(&x);
    let mut got = vec![0f32; n];
    silu::silu_f32(ctx, &x, &mut got, n).expect("silu_f32");
    let err = max_abs_err(&got, &want);
    eprintln!("wgpu_silu_f32: max abs err {err:e}");
    assert!(err < 1e-5, "silu f32 drift {err}");
}

#[test]
fn wgpu_silu_mul_f32_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_mul_f32_matches_cpu_reference") else {
        return;
    };
    let n = 4096usize;
    let x = oracle_x(n);
    let gate = oracle_gate(n);
    let want = cpu_silu_mul(&x, &gate);
    let mut got = vec![0f32; n];
    silu::silu_mul_f32(ctx, &x, &gate, &mut got, n).expect("silu_mul_f32");
    let err = max_abs_err(&got, &want);
    eprintln!("wgpu_silu_mul_f32: max abs err {err:e}");
    assert!(err < 1e-5, "silu_mul f32 drift {err}");
}

#[test]
fn wgpu_silu_bf16_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_bf16_matches_cpu_reference") else {
        return;
    };
    let n = 4096usize;
    let x = wave(n, 0.0);
    let xb = to_bf16(&x);
    let want = cpu_silu(&from_bf16(&xb));
    let mut got_bits = vec![0u16; n];
    silu::silu_bf16(ctx, &xb, &mut got_bits, n).expect("silu_bf16");
    let got = from_bf16(&got_bits);
    let err = max_abs_err(&got, &want);
    eprintln!("wgpu_silu_bf16: max abs err {err:e}");
    assert!(err < 0.05, "silu bf16 drift {err}");
}

#[test]
fn wgpu_silu_mul_bf16_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_mul_bf16_matches_cpu_reference") else {
        return;
    };
    let n = 4096usize;
    let x = oracle_x(n);
    let gate = oracle_gate(n);
    let xb = to_bf16(&x);
    let gb = to_bf16(&gate);
    let want = cpu_silu_mul(&from_bf16(&xb), &from_bf16(&gb));
    let mut got_bits = vec![0u16; n];
    silu::silu_mul_bf16(ctx, &xb, &gb, &mut got_bits, n).expect("silu_mul_bf16");
    let got = from_bf16(&got_bits);
    let err = max_abs_err(&got, &want);
    eprintln!("wgpu_silu_mul_bf16: max abs err {err:e}");
    assert!(err < 0.05, "silu_mul bf16 drift {err}");
}

#[test]
fn wgpu_silu_f32_handles_saturating_and_extreme_inputs() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_f32_handles_saturating_and_extreme_inputs") else {
        return;
    };
    let x: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 6.0, -6.0, 87.9, -87.9, 88.0, -88.0, 100.0, -100.0, 1.0e9, -1.0e9,
        1.0e30, -1.0e30, 3.4e38, -3.4e38,
    ];
    let n = x.len();
    let mut got = vec![0f32; n];
    silu::silu_f32(ctx, &x, &mut got, n).expect("silu_f32");
    for (i, v) in x.iter().enumerate() {
        let g = got[i];
        assert!(g.is_finite(), "x[{i}]={v} produced {g}");
        if *v >= 88.0 {
            assert!(
                (g - v).abs() <= v.abs() * 1e-6,
                "large positive x[{i}]={v} gave {g}"
            );
        } else if *v <= -88.0 {
            assert!(g.abs() < 1e-6, "large negative x[{i}]={v} gave {g}");
        } else {
            let want = v / (1.0 + (-v).exp());
            assert!(
                (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                "x[{i}]={v} gave {g} want {want}"
            );
        }
    }
}

#[test]
fn wgpu_silu_bf16_handles_an_odd_length_tail() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_bf16_handles_an_odd_length_tail") else {
        return;
    };
    for n in [1usize, 3, 5, 255, 257, 513] {
        let x = wave(n, 0.4);
        let xb = to_bf16(&x);
        let want = cpu_silu(&from_bf16(&xb));
        let mut got_bits = vec![0xdeadu16; n];
        silu::silu_bf16(ctx, &xb, &mut got_bits, n).expect("silu_bf16");
        let got = from_bf16(&got_bits);
        let err = max_abs_err(&got, &want);
        assert!(err < 0.05, "n={n} odd tail drift {err}");
    }
}

#[test]
fn wgpu_silu_f32_spans_more_workgroups_than_one_dimension_allows() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_f32_spans_more_workgroups_than_one_dimension_allows")
    else {
        return;
    };
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    let n = limit * (silu::WORKGROUP_SIZE as usize) + 1024;
    let bytes = (n * 4) as u64;
    if bytes > ctx.caps.max_storage_buffer_binding_size || bytes > ctx.caps.max_buffer_size {
        if !require() {
            eprintln!(
                "wgpu_silu_f32_spans_more_workgroups_than_one_dimension_allows: SKIP \
                 (NV_KERNELS_WGPU_ALLOW_SKIP=1) n={n} needs {bytes} bytes per buffer, past \
                 the device limit"
            );
            return;
        }
        panic!(
            "wgpu_silu_f32_spans_more_workgroups_than_one_dimension_allows: n={n} needs \
             {bytes} bytes per buffer, past this device's limit, so the multi-dimension \
             span this test exists to check cannot be exercised. Set \
             NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    }
    let x: Vec<f32> = (0..n)
        .map(|i| (((i % 4096) as f32) * 0.0017).sin() * 6.0)
        .collect();
    let mut got = vec![0f32; n];
    silu::silu_f32(ctx, &x, &mut got, n).expect("silu_f32 large");

    let mut err = 0f32;
    let mut i = 0usize;
    while i < n {
        let want = x[i] / (1.0 + (-x[i]).exp());
        err = err.max((got[i] - want).abs());
        i += 4099;
    }
    for i in [
        0usize,
        n - 1,
        limit * (silu::WORKGROUP_SIZE as usize) - 1,
        limit * (silu::WORKGROUP_SIZE as usize),
    ] {
        let want = x[i] / (1.0 + (-x[i]).exp());
        err = err.max((got[i] - want).abs());
    }
    eprintln!("wgpu_silu_f32 n={n}: sampled max abs err {err:e}");
    assert!(err < 1e-5, "large-n silu drift {err}");
}
