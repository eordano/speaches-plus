#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::LcgShift32TwoSided as Lcg;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemm_bf16_small_m as smk;
use nv_kernels::wgpu_backend::kernels::gemv_bf16;
use common::idle_pct;
use common::wait_for_idle;

fn reference_rows(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; m * n];
    for mi in 0..m {
        let mut row = vec![0u16; n];
        gemv_bf16::gemv_bf16(ctx, w, &x[mi * k..(mi + 1) * k], &mut row, n, k)
            .unwrap_or_else(|e| panic!("gemv_bf16 reference row {mi} n={n} k={k}: {e}"));
        out[mi * n..(mi + 1) * n].copy_from_slice(&row);
    }
    out
}

fn check_shape(ctx: &WgpuContext, seed: u64, n: usize, k: usize) {
    for m in 1..=smk::MAX_M as usize {
        let mut rng = Lcg(seed ^ ((m as u64) << 48) ^ ((n as u64) << 24) ^ k as u64);
        let w = rng.bf16_vec(n * k, 1.0);
        let x = rng.bf16_vec(m * k, 2.0);

        let expected = reference_rows(ctx, &w, &x, m, n, k);

        let mut got = vec![0u16; m * n];
        smk::gemm_bf16_small_m(ctx, &w, &x, &mut got, m, n, k)
            .unwrap_or_else(|e| panic!("gemm_bf16_small_m m={m} n={n} k={k}: {e}"));

        let diff = expected
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff,
            0,
            "m={m} n={n} k={k}: {diff}/{} words differ from M sequential gemv_bf16 runs",
            m * n
        );
        eprintln!("gemm_bf16_small_m m={m} n={n} k={k}: bit-exact vs {m}x gemv_bf16");
    }
}

#[test]
fn small_m_matches_sequential_gemv_bf16_bitwise_k2048() {
    let Some(ctx) = ctx_or_skip("small_m_k2048") else {
        return;
    };
    check_shape(ctx, 0xA11C_E001, 2048, 2048);
}

#[test]
fn small_m_matches_sequential_gemv_bf16_bitwise_k5376_prefill_tail_scaled() {
    let Some(ctx) = ctx_or_skip("small_m_k5376") else {
        return;
    };
    check_shape(ctx, 0xA11C_E002, 336, 5376);
}

#[test]
fn small_m_matches_sequential_gemv_bf16_bitwise_non_multiple_of_8_k() {
    let Some(ctx) = ctx_or_skip("small_m_nonmul8") else {
        return;
    };
    check_shape(ctx, 0xA11C_E003, 129, 2050);
    check_shape(ctx, 0xA11C_E004, 37, 1026);
}

#[test]
fn small_m_matches_sequential_gemv_bf16_bitwise_small_shape() {
    let Some(ctx) = ctx_or_skip("small_m_tiny") else {
        return;
    };
    check_shape(ctx, 0xA11C_E005, 9, 16);
}

#[test]
#[ignore = "GPU rate measurement; run explicitly with --ignored"]
fn bench_rows_per_us_table() {
    let Some(ctx) = ctx_or_skip("bench_rows_per_us_table") else {
        return;
    };

    let quiet = wait_for_idle(85, std::time::Duration::from_secs(15 * 60));
    if !quiet {
        eprintln!(
            "bench_rows_per_us_table: PROVISIONAL -- could not reach a quiet window in 15 min"
        );
    }

    let n = 5376usize;
    let k = 2048usize;
    let iters = 200usize;
    let warmup = 20usize;

    println!(
        "gemm_bf16_small_m bench n={n} k={k} iters={iters} quiet_window={} (numbers are {} if quiet_window=false)",
        quiet,
        if quiet { "measured" } else { "PROVISIONAL" }
    );
    println!(
        "{:>3} {:>14} {:>14} {:>10}",
        "M", "batched_us/row", "m_x_gemv1_us/row", "speedup"
    );

    for m in 1..=smk::MAX_M as usize {
        let mut rng = Lcg(0xB0BA_1234 ^ ((m as u64) << 32) ^ k as u64);
        let w = rng.bf16_vec(n * k, 1.0);
        let x = rng.bf16_vec(m * k, 2.0);

        let (_, batched_secs) = smk::gemm_bf16_small_m_probe(ctx, &w, &x, m, n, k, warmup, iters)
            .unwrap_or_else(|e| panic!("probe m={m}: {e}"));
        let batched_us_per_row = batched_secs * 1e6 / (iters as f64 * m as f64);

        let gemv_kernel = if k.is_multiple_of(8) {
            gemv_bf16::GemvKernel::TreeVec8
        } else {
            gemv_bf16::GemvKernel::TreeScalar
        };
        let mut m1_total_secs = 0.0;
        for mi in 0..m {
            let (_, secs) = gemv_bf16::gemv_bf16_probe(
                ctx,
                &w,
                &x[mi * k..(mi + 1) * k],
                n,
                k,
                warmup,
                iters,
                gemv_kernel,
            )
            .unwrap_or_else(|e| panic!("gemv1 probe m={m} row={mi}: {e}"));
            m1_total_secs += secs;
        }
        let m1_us_per_row = m1_total_secs * 1e6 / (iters as f64 * m as f64);
        let speedup = m1_us_per_row / batched_us_per_row;

        println!(
            "{:>3} {:>14.4} {:>14.4} {:>9.2}x",
            m, batched_us_per_row, m1_us_per_row, speedup
        );
    }
}
