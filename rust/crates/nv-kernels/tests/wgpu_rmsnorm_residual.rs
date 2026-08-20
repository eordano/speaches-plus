#![cfg(feature = "wgpu")]

mod common;
use common::bits;
use common::ctx_or_skip;
use common::max_abs_err;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::rmsnorm_residual;

const BLOCK: usize = 256;

fn tree_reduce(mut scratch: Vec<f32>) -> f32 {
    let mut stride = BLOCK / 2;
    while stride > 0 {
        for t in 0..stride {
            scratch[t] += scratch[t + stride];
        }
        stride >>= 1;
    }
    scratch[0]
}

fn cpu_rmsnorm_residual_f32(
    x: &[f32],
    residual: &mut [f32],
    weight: &[f32],
    out: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    for row in 0..batch {
        let base = row * hidden;
        let mut scratch = vec![0f32; BLOCK];
        for (t, slot) in scratch.iter_mut().enumerate() {
            let mut local = 0f32;
            let mut i = t;
            while i < hidden {
                let s = x[base + i] + residual[base + i];
                residual[base + i] = s;
                local += s * s;
                i += BLOCK;
            }
            *slot = local;
        }
        let total = tree_reduce(scratch);
        let rms = 1.0f32 / (total / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            out[base + i] = residual[base + i] * rms * weight[i];
        }
    }
}

fn cpu_rmsnorm_residual_bf16(
    x: &[u16],
    residual: &mut [u16],
    weight: &[u16],
    out: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
) {
    let f = |b: u16| bf16::from_bits(b).to_f32();
    for row in 0..batch {
        let base = row * hidden;
        let mut scratch = vec![0f32; BLOCK];
        for (t, slot) in scratch.iter_mut().enumerate() {
            let mut local = 0f32;
            let mut i = t;
            while i < hidden {
                let s = f(x[base + i]) + f(residual[base + i]);
                residual[base + i] = bf16::from_f32(s).to_bits();
                local += s * s;
                i += BLOCK;
            }
            *slot = local;
        }
        let total = tree_reduce(scratch);
        let rms = 1.0f32 / (total / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            let v = f(residual[base + i]) * rms * f(weight[i]);
            out[base + i] = bf16::from_f32(v).to_bits();
        }
    }
}

fn sample_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.0007).sin() * 2.0).collect()
}

fn sample_r(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.0013).cos() * 1.5).collect()
}

fn sample_w(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 1.0 + ((i as f32) * 0.001).cos() * 0.5)
        .collect()
}

fn max_ulp(got: &[u16], want: &[u16]) -> (usize, i32) {
    let mut n = 0usize;
    let mut m = 0i32;
    for (a, b) in got.iter().zip(want.iter()) {
        if a != b {
            n += 1;
            m = m.max((*a as i32 - *b as i32).abs());
        }
    }
    (n, m)
}

#[test]
fn wgpu_rmsnorm_residual_f32_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_residual_f32_matches_cpu_reference") else {
        return;
    };
    let eps = 1e-5f32;
    for (batch, hidden) in [(4usize, 1024usize), (1, 257), (3, 64), (7, 4096), (2, 255)] {
        let x = sample_x(batch * hidden);
        let r0 = sample_r(batch * hidden);
        let w = sample_w(hidden);

        let mut want_res = r0.clone();
        let mut want_out = vec![0f32; batch * hidden];
        cpu_rmsnorm_residual_f32(&x, &mut want_res, &w, &mut want_out, batch, hidden, eps);

        let mut got_res = r0.clone();
        let mut got_out = vec![0f32; batch * hidden];
        rmsnorm_residual::rmsnorm_residual_f32(
            ctx,
            &x,
            &mut got_res,
            &w,
            &mut got_out,
            batch,
            hidden,
            eps,
        )
        .expect("rmsnorm_residual f32");

        let res_diff = got_res
            .iter()
            .zip(want_res.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let err = max_abs_err(&got_out, &want_out);
        eprintln!(
            "f32 batch={batch} hidden={hidden} res_bit_diff={res_diff} out_max_abs_err={err:e}"
        );
        assert_eq!(
            res_diff, 0,
            "residual = x + residual is an exact f32 add and must be bit-exact (batch={batch} hidden={hidden})"
        );
        assert!(
            err < 1e-4,
            "batch={batch} hidden={hidden} out max abs error {err}"
        );
    }
}

#[test]
fn wgpu_rmsnorm_residual_bf16_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_residual_bf16_matches_cpu_reference") else {
        return;
    };
    let eps = 1e-5f32;
    for (batch, hidden) in [(5usize, 2048usize), (1, 256), (3, 64), (2, 4096), (4, 130)] {
        let x = bits(&sample_x(batch * hidden));
        let r0 = bits(&sample_r(batch * hidden));
        let w = bits(&sample_w(hidden));

        let mut want_res = r0.clone();
        let mut want_out = vec![0u16; batch * hidden];
        cpu_rmsnorm_residual_bf16(&x, &mut want_res, &w, &mut want_out, batch, hidden, eps);

        let mut got_res = r0.clone();
        let mut got_out = vec![0u16; batch * hidden];
        rmsnorm_residual::rmsnorm_residual_bf16(
            ctx,
            &x,
            &mut got_res,
            &w,
            &mut got_out,
            batch,
            hidden,
            eps,
        )
        .expect("rmsnorm_residual bf16");

        let res_diff = got_res
            .iter()
            .zip(want_res.iter())
            .filter(|(a, b)| a != b)
            .count();
        let (n, ulp) = max_ulp(&got_out, &want_out);
        eprintln!(
            "bf16 batch={batch} hidden={hidden} res_bit_diff={res_diff} out_words_differ={n} out_max_ulp={ulp}"
        );
        assert_eq!(
            res_diff, 0,
            "residual write is an exact bf16 round-to-nearest-even and must be bit-exact (batch={batch} hidden={hidden})"
        );
        assert!(
            ulp <= 1,
            "batch={batch} hidden={hidden} out differs by {ulp} bf16 ulp ({n} words)"
        );
    }
}

#[test]
fn wgpu_rmsnorm_residual_rejects_bad_shapes() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_residual_rejects_bad_shapes") else {
        return;
    };
    let mut res = vec![0f32; 8];
    let mut out = vec![0f32; 8];
    let e = rmsnorm_residual::rmsnorm_residual_f32(
        ctx, &[0f32; 7], &mut res, &[0f32; 4], &mut out, 2, 4, 1e-5,
    )
    .unwrap_err();
    eprintln!("shape error: {e}");

    let mut resb = vec![0u16; 6];
    let mut outb = vec![0u16; 6];
    let e = rmsnorm_residual::rmsnorm_residual_bf16(
        ctx, &[0u16; 6], &mut resb, &[0u16; 3], &mut outb, 2, 3, 1e-5,
    )
    .unwrap_err();
    eprintln!("odd hidden error: {e}");
}

#[test]
fn wgpu_rmsnorm_residual_empty_is_a_noop() {
    let Some(ctx) = ctx_or_skip("wgpu_rmsnorm_residual_empty_is_a_noop") else {
        return;
    };
    let mut res: Vec<f32> = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    rmsnorm_residual::rmsnorm_residual_f32(ctx, &[], &mut res, &[], &mut out, 0, 0, 1e-5).unwrap();
    assert!(out.is_empty());
}
