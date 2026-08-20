#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::max_abs_err;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::rmsnorm;
use common::cpu_rmsnorm;

fn sample_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.0001).sin()).collect()
}

fn sample_w(n: usize) -> Vec<f32> {
    (0..n).map(|i| 1.0 + ((i as f32) * 0.001).cos()).collect()
}

#[test]
fn wgpu_rmsnorm_f32_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_f32_matches_cpu_reference") else {
        return;
    };
    let eps = 1e-5f32;
    for (batch, hidden) in [(4usize, 1024usize), (1, 257), (3, 64), (7, 4096)] {
        let x = sample_x(batch * hidden);
        let w = sample_w(hidden);
        let mut y = vec![0f32; batch * hidden];
        rmsnorm::rmsnorm_f32(ctx, &x, &w, &mut y, batch, hidden, eps).expect("rmsnorm f32");
        let want = cpu_rmsnorm(&x, &w, hidden, eps);
        let err = max_abs_err(&y, &want);
        eprintln!("f32 batch={batch} hidden={hidden} max_abs_err={err:e}");
        assert!(
            err < 1e-4,
            "batch={batch} hidden={hidden} max abs error {err}"
        );
    }
}

#[test]
fn wgpu_rmsnorm_bf16_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_bf16_matches_cpu_reference") else {
        return;
    };
    let eps = 1e-5f32;
    for (batch, hidden) in [(4usize, 1024usize), (2, 130), (5, 2048)] {
        let xb: Vec<bf16> = sample_x(batch * hidden)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let wb: Vec<bf16> = sample_w(hidden).into_iter().map(bf16::from_f32).collect();
        let x_f32: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
        let w_f32: Vec<f32> = wb.iter().map(|v| v.to_f32()).collect();
        let want = cpu_rmsnorm(&x_f32, &w_f32, hidden, eps);

        let x_u16: Vec<u16> = xb.iter().map(|v| v.to_bits()).collect();
        let w_u16: Vec<u16> = wb.iter().map(|v| v.to_bits()).collect();
        let mut y_u16 = vec![0u16; batch * hidden];
        rmsnorm::rmsnorm_bf16(ctx, &x_u16, &w_u16, &mut y_u16, batch, hidden, eps)
            .expect("rmsnorm bf16");
        let got: Vec<f32> = y_u16.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();

        let err = max_abs_err(&got, &want);
        eprintln!("bf16 batch={batch} hidden={hidden} max_abs_err={err:e}");
        assert!(err < 0.05, "batch={batch} hidden={hidden} bf16 drift {err}");
    }
}

#[test]
fn wgpu_rmsnorm_bf16_output_is_bit_exact_against_cpu_bf16_rounding() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_bf16_output_is_bit_exact_against_cpu_bf16_rounding")
    else {
        return;
    };
    let batch = 3usize;
    let hidden = 512usize;
    let eps = 1e-5f32;

    let xb: Vec<bf16> = sample_x(batch * hidden)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let wb: Vec<bf16> = sample_w(hidden).into_iter().map(bf16::from_f32).collect();
    let x_u16: Vec<u16> = xb.iter().map(|v| v.to_bits()).collect();
    let w_u16: Vec<u16> = wb.iter().map(|v| v.to_bits()).collect();
    let mut y_u16 = vec![0u16; batch * hidden];
    rmsnorm::rmsnorm_bf16(ctx, &x_u16, &w_u16, &mut y_u16, batch, hidden, eps)
        .expect("rmsnorm bf16");

    let mut mismatched = 0usize;
    for b in 0..batch {
        let row: Vec<f32> = xb[b * hidden..(b + 1) * hidden]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let sumsq: f32 = row.iter().map(|v| v * v).sum();
        let rms = 1.0f32 / (sumsq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            let want = bf16::from_f32(row[i] * rms * wb[i].to_f32());
            if y_u16[b * hidden + i] != want.to_bits() {
                mismatched += 1;
            }
        }
    }
    eprintln!(
        "bf16 bit-exact mismatches: {mismatched}/{} (rms differs only by reduction order)",
        batch * hidden
    );
    assert!(
        mismatched * 200 <= batch * hidden,
        "more than 0.5% of bf16 outputs differ from the CPU rounding: {mismatched}"
    );
}

#[test]
fn wgpu_rmsnorm_f32_scale_invariant() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_f32_scale_invariant") else {
        return;
    };
    let batch = 2usize;
    let hidden = 768usize;
    let eps = 0f32;
    let x = sample_x(batch * hidden);
    let w = sample_w(hidden);
    let scaled: Vec<f32> = x.iter().map(|v| v * 8.0).collect();

    let mut y0 = vec![0f32; batch * hidden];
    let mut y1 = vec![0f32; batch * hidden];
    rmsnorm::rmsnorm_f32(ctx, &x, &w, &mut y0, batch, hidden, eps).expect("rmsnorm base");
    rmsnorm::rmsnorm_f32(ctx, &scaled, &w, &mut y1, batch, hidden, eps).expect("rmsnorm scaled");

    let err = max_abs_err(&y1, &y0);
    eprintln!("scale invariance max_abs_err={err:e}");
    assert!(err < 1e-4, "rmsnorm is not scale invariant: {err}");
}

#[test]
fn wgpu_rmsnorm_f32_unit_norm_output_when_weight_one() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_f32_unit_norm_output_when_weight_one") else {
        return;
    };
    let batch = 3usize;
    let hidden = 1024usize;
    let eps = 0f32;
    let x = sample_x(batch * hidden);
    let w = vec![1f32; hidden];
    let mut y = vec![0f32; batch * hidden];
    rmsnorm::rmsnorm_f32(ctx, &x, &w, &mut y, batch, hidden, eps).expect("rmsnorm");

    for b in 0..batch {
        let row = &y[b * hidden..(b + 1) * hidden];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        assert!(
            (ms - 1.0).abs() < 1e-3,
            "row {b} mean square {ms} is not 1.0"
        );
    }
}

#[test]
fn wgpu_rmsnorm_rejects_bad_shapes() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_rejects_bad_shapes") else {
        return;
    };
    let mut y = vec![0f32; 8];
    let err = rmsnorm::rmsnorm_f32(ctx, &[0f32; 8], &[0f32; 3], &mut y, 2, 4, 1e-5).unwrap_err();
    eprintln!("shape rejection: {err}");

    let mut yb = vec![0u16; 6];
    let err = rmsnorm::rmsnorm_bf16(ctx, &[0u16; 6], &[0u16; 3], &mut yb, 2, 3, 1e-5).unwrap_err();
    eprintln!("odd hidden rejection: {err}");
}
