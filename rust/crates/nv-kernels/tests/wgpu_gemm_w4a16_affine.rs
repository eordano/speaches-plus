#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::gemm_w4a16_affine as affm;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    m_live: u32,
    x_stride_words: u32,
    y_stride_words: u32,
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn bf16_small(&mut self) -> u16 {
        let v = ((self.next() % 2000) as f32 / 1000.0 - 1.0) * 0.05;
        (v.to_bits() >> 16) as u16
    }
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP {test}: no wgpu adapter: {e}");
            None
        }
    }
}

const SENTINEL: u32 = 0x7fc0;

fn params_for(n: usize, k: usize, gs: usize, m: usize, groups_x: u32) -> Params {
    Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x,
        m_live: m as u32,
        x_stride_words: (k / 2) as u32,
        y_stride_words: n as u32,
    }
}

#[test]
fn affine_mrows_v2_pk_packs_bf16_row_pairs_matching_the_word_per_row_entry() {
    let Some(ctx) =
        ctx_or_skip("affine_mrows_v2_pk_packs_bf16_row_pairs_matching_the_word_per_row_entry")
    else {
        return;
    };
    let (m, n, k, gs) = (5usize, 272usize, 512usize, 64usize);
    let mut r = Lcg(0x7a11ce);
    let words: Vec<u32> = (0..n * k / 8).map(|_| r.next()).collect();
    let scales_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let biases_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let x_pack: Vec<u32> = (0..m * k / 2).map(|_| r.next() & 0x3f803f80).collect();
    let pb = dispatch::storage_from_slice(ctx, "pk-packed", &words);
    let sb = dispatch::storage_from_slice(ctx, "pk-scale", &scales_b);
    let bb = dispatch::storage_from_slice(ctx, "pk-bias", &biases_b);
    let xb = dispatch::storage_from_slice(ctx, "pk-x", &x_pack);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, affm::ROWS_PER_GROUP);
    let flat_y = dispatch::storage_from_slice(ctx, "pk-flat", &vec![SENTINEL; m * n]);
    let flat_p = params_for(n, k, gs, m, groups.0);
    let ub = dispatch::uniform_from(ctx, "pk-flat-p", &flat_p);
    dispatch::run(
        ctx,
        "pk-flat",
        &affm::source(),
        affm::MROWS_ENTRY,
        &[(0, &pb), (1, &sb), (2, &xb), (3, &flat_y), (4, &ub), (affm::BIAS_SLOT, &bb)],
        groups,
    )
    .expect("flat dispatch");
    let flat: Vec<u32> = dispatch::read_back(ctx, &flat_y, m * n).expect("rb flat");

    let v2_y = dispatch::storage_from_slice(ctx, "pk-v2", &vec![0u32; m * n / 2]);
    let v2_groups = dispatch::workgroup_count_1d(ctx, n as u64, affm::MROWS_V2_ROWS_PER_GROUP);
    let v2_p = Params {
        y_stride_words: (n / 2) as u32,
        groups_x: v2_groups.0,
        ..flat_p
    };
    let ub3 = dispatch::uniform_from(ctx, "pk-v2-p", &v2_p);
    dispatch::run(
        ctx,
        "pk-v2",
        &affm::source(),
        affm::MROWS_V2_PK_ENTRY_STAGES_X_IN_THREADGROUP_FOR_64_ROWS,
        &[(0, &pb), (1, &sb), (2, &xb), (3, &v2_y), (4, &ub3), (affm::BIAS_SLOT, &bb)],
        v2_groups,
    )
    .expect("v2 dispatch");
    let v2: Vec<u32> = dispatch::read_back(ctx, &v2_y, m * n / 2).expect("rb v2");
    let mut v2_worst_off = 0u32;
    for t in 0..m {
        for p2 in 0..n / 2 {
            let lo = flat[t * n + 2 * p2] & 0xffff;
            let hi = flat[t * n + 2 * p2 + 1] & 0xffff;
            let want = lo | (hi << 16);
            let got = v2[t * (n / 2) + p2];
            for (gw, ww) in [(got & 0xffff, lo), (got >> 16, hi >> 0)] {
                let (gf, wf) = (bf16_to_f32(gw as u16), bf16_to_f32(ww as u16));
                let off = (gf - wf).abs();
                assert!(
                    off <= wf.abs().max(1e-3) * 0.02,
                    "v2 t{t} pair {p2}: {gf} vs {wf}"
                );
                if got != want {
                    v2_worst_off = v2_worst_off.max(1);
                }
            }
        }
    }
    eprintln!("V2-PK vs flat: value-equal within 2% (bitwise diff present: {v2_worst_off})");

}

#[test]
fn affine_mrows_matches_the_f64_host_reference_and_leaves_the_m_tail_untouched() {
    let Some(ctx) =
        ctx_or_skip("affine_mrows_matches_the_f64_host_reference_and_leaves_the_m_tail_untouched")
    else {
        return;
    };
    let max_m = affm::MAX_M as usize;
    for m in [1usize, 10, 16] {
        for (n, k) in [(64usize, 512usize), (256, 5120), (68, 512)] {
            let gs = 64usize;
            affm::grain_rule(gs).expect("grain rule");
            affm::m_rule(m).expect("m rule");
            let mut r = Lcg(0x9e3779b97f4a7c15 ^ ((n * k + m) as u64));
            let words: Vec<u32> = (0..n * k / 8).map(|_| r.next()).collect();
            let scales_b: Vec<u16> = (0..n * (k / gs)).map(|_| r.bf16_small()).collect();
            let biases_b: Vec<u16> = (0..n * (k / gs)).map(|_| r.bf16_small()).collect();
            let x_b: Vec<u16> = (0..m * k).map(|_| r.bf16_small()).collect();

            let scales_f: Vec<f32> = scales_b.iter().map(|b| bf16_to_f32(*b)).collect();
            let biases_f: Vec<f32> = biases_b.iter().map(|b| bf16_to_f32(*b)).collect();
            let x_f: Vec<f32> = x_b.iter().map(|b| bf16_to_f32(*b)).collect();
            let w = nv_quant::mlx_affine::dequantize_mlx_affine_linear(
                &words, &scales_f, &biases_f, n, k, gs,
            );
            let want: Vec<f64> = (0..m * n)
                .map(|tn| {
                    let (t, row) = (tn / n, tn % n);
                    (0..k)
                        .map(|i| w[row * k + i] as f64 * x_f[t * k + i] as f64)
                        .sum::<f64>()
                })
                .collect();

            let widen = |v: &[u16]| -> Vec<u32> { v.iter().map(|b| *b as u32).collect() };
            let pack_x: Vec<u32> = (0..m * k / 2)
                .map(|i| x_b[2 * i] as u32 | ((x_b[2 * i + 1] as u32) << 16))
                .collect();
            let groups = dispatch::workgroup_count_1d(ctx, n as u64, affm::ROWS_PER_GROUP);
            let params = params_for(n, k, gs, m, groups.0);
            let pb = dispatch::storage_from_slice(ctx, "affm-packed", &words);
            let sb = dispatch::storage_from_slice(ctx, "affm-scale", &widen(&scales_b));
            let bb = dispatch::storage_from_slice(ctx, "affm-bias", &widen(&biases_b));
            let xb = dispatch::storage_from_slice(ctx, "affm-x", &pack_x);
            let yb = dispatch::storage_from_slice(ctx, "affm-y", &vec![SENTINEL; max_m * n]);
            let ub = dispatch::uniform_from(ctx, "affm-params", &params);
            dispatch::run(
                ctx,
                "affm-oracle",
                &affm::source(),
                affm::MROWS_ENTRY,
                &[
                    (0, &pb),
                    (1, &sb),
                    (2, &xb),
                    (3, &yb),
                    (4, &ub),
                    (affm::BIAS_SLOT, &bb),
                ],
                groups,
            )
            .expect("dispatch");
            let got_words: Vec<u32> = dispatch::read_back(ctx, &yb, max_m * n).expect("readback");
            let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs())).max(1e-6);
            let mut worst = 0f64;
            for (tn, gw) in got_words.iter().enumerate() {
                let (t, row) = (tn / n, tn % n);
                if t >= m {
                    assert_eq!(
                        *gw, SENTINEL,
                        "m{m} n{n} k{k}: padded tail row t{t} row {row} was written"
                    );
                    continue;
                }
                let got = bf16_to_f32((gw & 0xffff) as u16) as f64;
                let rel = (got - want[tn]).abs() / ref_max;
                worst = worst.max(rel);
                assert!(
                    rel < 3.0 / 256.0,
                    "m{m} n{n} k{k} t{t} row {row}: kernel {got} vs f64 host {}, rel {rel:e} \
                     exceeds the bf16-output-plus-f32-accumulation budget",
                    want[tn]
                );
            }
            eprintln!("AFFINE-GEMM-ORACLE m{m} n{n} k{k} worst rel {worst:e}");
        }
    }
}

#[test]
#[ignore = "GPU rate measurement; run via nvk.sh probe on an idle GPU -- times the affine M-row entry at M=8 at the qwen3.8 gate_proj shape and reports effective GB/s of packed-weight traffic for comparison against gemv_w4a16_affine dispatched 8 times"]
fn affine_mrows_rate_at_qwen38_gate_proj_shape() {
    let Some(ctx) = ctx_or_skip("affine_mrows_rate_at_qwen38_gate_proj_shape") else {
        return;
    };
    for m in [1usize, 8, 16] {
        rate_one_m(ctx, m);
    }
}

fn rate_one_m(ctx: &nv_kernels::wgpu_backend::WgpuContext, m: usize) {
    rate_one_entry(ctx, m, "mrows", affm::MROWS_ENTRY, affm::ROWS_PER_GROUP, 1, false);
    rate_one_entry(
        ctx,
        m,
        "v2pk",
        affm::MROWS_V2_PK_ENTRY_STAGES_X_IN_THREADGROUP_FOR_64_ROWS,
        affm::MROWS_V2_ROWS_PER_GROUP,
        2,
        false,
    );
}

fn rate_one_entry(
    ctx: &nv_kernels::wgpu_backend::WgpuContext,
    m: usize,
    label: &str,
    entry: &str,
    rows_per_group: u32,
    y_div: usize,
    coltiled: bool,
) {
    let (n, k, gs) = (17408usize, 5120usize, 64usize);
    let mut r = Lcg(0xabcdef12345);
    let words: Vec<u32> = (0..n * k / 8).map(|_| r.next()).collect();
    let scales_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let biases_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let x_pack: Vec<u32> = (0..m * k / 2).map(|_| r.next() & 0x3f803f80).collect();
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = Params {
        y_stride_words: (n / y_div) as u32,
        ..params_for(n, k, gs, m, groups.0)
    };
    let _ = coltiled;
    let pb = dispatch::storage_from_slice(ctx, "affm-rate-packed", &words);
    let sb = dispatch::storage_from_slice(ctx, "affm-rate-scale", &scales_b);
    let bb = dispatch::storage_from_slice(ctx, "affm-rate-bias", &biases_b);
    let xb = dispatch::storage_from_slice(ctx, "affm-rate-x", &x_pack);
    let yb = dispatch::storage_from_slice(ctx, "affm-rate-y", &vec![0u32; m * n / y_div]);
    let ub = dispatch::uniform_from(ctx, "affm-rate-params", &params);
    let binds = [
        (0u32, &pb),
        (1, &sb),
        (2, &xb),
        (3, &yb),
        (4, &ub),
        (affm::BIAS_SLOT, &bb),
    ];
    let source = affm::source();
    let weight_bytes = (n * k / 2 + n * (k / gs) * 4 * 2) as f64;
    for _ in 0..5 {
        dispatch::run(ctx, "affm-rate-warm", &source, entry, &binds, groups).expect("warm");
    }
    let _drain_warm: Vec<u32> = dispatch::read_back(ctx, &yb, 1).expect("drain-warm");
    let laps = 50usize;
    let t0 = std::time::Instant::now();
    for _ in 0..laps {
        dispatch::run(ctx, "affm-rate-lap", &source, entry, &binds, groups).expect("lap");
    }
    let _drain: Vec<u32> = dispatch::read_back(ctx, &yb, 1).expect("drain");
    let per = t0.elapsed().as_secs_f64() / laps as f64;
    eprintln!(
        "AFFINE-GEMM-RATE {label} m{m} n{n} k{k} {:.3} ms/dispatch {:.1} GB/s packed-weight {:.1} tok-GFLOP/s",
        per * 1e3,
        weight_bytes / per / 1e9,
        (2.0 * (m * n * k) as f64) / per / 1e9
    );
}
