#![cfg(feature = "wgpu")]

use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::{compose, dispatch};
mod common;
use common::ctx_or_skip_quiet_unqualified as ctx_or_skip;
use common::wgpu_allow_skip;

const MULTICOL_WGSL: &str = include_str!("../wgsl/gemv_multicol.wgsl");
const Q3D_GEMV_WGSL: &str = include_str!("../wgsl/q3d_gemv_bf16.wgsl");
const MC_ENTRY: &str = "gemv_bf16_mc";
const Q3D_ENTRY: &str = "q3w_gemv_bf16";

const NCOLS_CONST_ANCHOR: &str = "const MC_NCOLS: u32 = 2u;";
const X_STRIDE_EXPR_ANCHOR: &str = "c * mc_p.x_col_stride_words + i";

const KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES: f32 = 1e-2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct McParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    x_col_stride_words: u32,
    y_col_stride_words: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Q3bParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_bf16_unit(&mut self) -> u16 {
        let v = (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        bf16::from_f32(v).to_bits()
    }
}

fn variant_src(m: usize) -> String {
    assert!(
        matches!(m, 1 | 2 | 4 | 8),
        "column-count menu is 1/2/4/8, got {m}"
    );
    let line = format!("const MC_NCOLS: u32 = {m}u;");
    let replaced = MULTICOL_WGSL.replacen(NCOLS_CONST_ANCHOR, &line, 1);
    assert!(
        m == 2 || replaced != MULTICOL_WGSL,
        "NCOLS const anchor missing from gemv_multicol.wgsl; variant compose is vacuous"
    );
    compose(&replaced)
}

fn column_stride_swapped_src(m: usize) -> String {
    let good = variant_src(m);
    let bad = good.replacen(
        X_STRIDE_EXPR_ANCHOR,
        "c * mc_p.y_col_stride_words + i",
        1,
    );
    assert_ne!(
        bad, good,
        "x-stride expression anchor missing; the planted-bug gate is vacuous"
    );
    bad
}

fn gen_w(n: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg::new(seed);
    (0..n * k).map(|_| rng.next_bf16_unit()).collect()
}

fn gen_x_cols(m: usize, k: usize, seed: u64) -> Vec<Vec<u16>> {
    (0..m)
        .map(|c| {
            let mut rng = Lcg::new(seed ^ ((c as u64 + 1) << 40));
            (0..k).map(|_| rng.next_bf16_unit()).collect()
        })
        .collect()
}

fn pack_u16(v: &[u16]) -> Vec<u32> {
    v.chunks(2)
        .map(|c| (c[0] as u32) | ((*c.get(1).unwrap_or(&0) as u32) << 16))
        .collect()
}

fn host_cols(w: &[u16], xcols: &[Vec<u16>], n: usize, k: usize, alpha: f32) -> Vec<Vec<f32>> {
    xcols
        .iter()
        .map(|x| {
            (0..n)
                .map(|row| {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += bf16::from_bits(w[row * k + kk]).to_f64()
                            * bf16::from_bits(x[kk]).to_f64();
                    }
                    (acc * alpha as f64) as f32
                })
                .collect()
        })
        .collect()
}

struct McRun {
    n: usize,
    k: usize,
    m: usize,
    out_f32: bool,
    alpha: f32,
}

fn run_multicol(
    ctx: &WgpuContext,
    label: &str,
    src: &str,
    w: &[u16],
    xcols: &[Vec<u16>],
    c: &McRun,
) -> Vec<Vec<f32>> {
    assert!(c.k % 2 == 0, "bf16 packing needs even k");
    let k_words = c.k / 2;
    let pairs = c.n.div_ceil(2);
    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    let y_stride = if c.out_f32 { c.n } else { pairs };
    let params = McParams {
        n_rows: c.n as u32,
        k_words: k_words as u32,
        groups_x: groups.0,
        out_f32: c.out_f32 as u32,
        w_row_words: k_words as u32,
        x_off_words: 0,
        y_off_words: 0,
        x_col_stride_words: k_words as u32,
        y_col_stride_words: y_stride as u32,
        alpha: c.alpha,
        ..Default::default()
    };
    let x_flat: Vec<u16> = xcols.iter().flatten().copied().collect();
    let wb = dispatch::storage_from_slice(ctx, "mc-w", &pack_u16(w));
    let xb = dispatch::storage_from_slice(ctx, "mc-x", &pack_u16(&x_flat));
    let yb = dispatch::storage_from_slice(ctx, "mc-y", &vec![0x7fc00000u32; c.m * y_stride]);
    let ub = dispatch::uniform_from(ctx, "mc-p", &params);
    dispatch::run(
        ctx,
        label,
        src,
        MC_ENTRY,
        &[(0, &wb), (1, &xb), (2, &ub), (3, &yb)],
        groups,
    )
    .expect("multicol dispatch");
    let words: Vec<u32> = dispatch::read_back(ctx, &yb, c.m * y_stride).expect("read back");
    (0..c.m)
        .map(|col| {
            let seg = &words[col * y_stride..(col + 1) * y_stride];
            if c.out_f32 {
                seg.iter().map(|v| f32::from_bits(*v)).collect()
            } else {
                let mut out = Vec::with_capacity(c.n);
                for (pair, wv) in seg.iter().enumerate() {
                    out.push(bf16::from_bits((*wv & 0xffff) as u16).to_f32());
                    if pair * 2 + 1 < c.n {
                        out.push(bf16::from_bits((*wv >> 16) as u16).to_f32());
                    }
                }
                out
            }
        })
        .collect()
}

fn max_rel_mismatch(got: &[Vec<f32>], want: &[Vec<f32>]) -> (f32, usize, usize) {
    let mut worst = (0f32, 0usize, 0usize);
    for (col, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "column {col} length");
        for (row, (a, b)) in g.iter().zip(w).enumerate() {
            let d = (a - b).abs() / b.abs().max(1e-3);
            if d > worst.0 {
                worst = (d, col, row);
            }
        }
    }
    worst
}

fn assert_cols_match(tag: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    let (d, col, row) = max_rel_mismatch(got, want);
    assert!(
        d < KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES,
        "{tag}: col {col} row {row} rel {d:.3e}: the multicol entry no longer matches the \
         host oracle; per-column inputs are distinct by construction so a column-indexing \
         bug cannot hide"
    );
}

#[test]
fn multicol_matches_host_oracle_with_distinct_columns_and_odd_dead_lane_rows() {
    let Some(ctx) = ctx_or_skip("multicol_parity_fuzz") else {
        return;
    };
    for &m in &[2usize, 4, 8] {
        let src = variant_src(m);
        let label = format!("mc{m}-fuzz");
        for &(n, k, seed, alpha) in &[
            (2usize, 64usize, 1u64, 1.0f32),
            (33, 256, 2, 1.0),
            (63, 512, 3, 1.0),
            (129, 250, 4, 0.5),
            (256, 1024, 5, 1.0),
        ] {
            let w = gen_w(n, k, seed);
            let xcols = gen_x_cols(m, k, seed.wrapping_mul(0x9e37));
            let case = McRun {
                n,
                k,
                m,
                out_f32: true,
                alpha,
            };
            let got = run_multicol(ctx, &label, &src, &w, &xcols, &case);
            let want = host_cols(&w, &xcols, n, k, alpha);
            assert_cols_match(&format!("m={m} n={n} k={k} seed={seed}"), &got, &want);
        }
    }
}

#[test]
fn multicol_matches_host_oracle_at_qwen38_serving_shapes() {
    let Some(ctx) = ctx_or_skip("multicol_parity_serving") else {
        return;
    };
    for &m in &[2usize, 4, 8] {
        let src = variant_src(m);
        let label = format!("mc{m}-serving");
        for &(n, k, seed) in &[(5120usize, 5120usize, 11u64), (12288, 5120, 12)] {
            let w = gen_w(n, k, seed);
            let xcols = gen_x_cols(m, k, seed.wrapping_mul(0x51de));
            let case = McRun {
                n,
                k,
                m,
                out_f32: true,
                alpha: 1.0,
            };
            let got = run_multicol(ctx, &label, &src, &w, &xcols, &case);
            let want = host_cols(&w, &xcols, n, k, 1.0);
            assert_cols_match(&format!("m={m} n={n} k={k}"), &got, &want);
        }
    }
}

#[test]
fn bf16_packed_output_path_matches_at_odd_row_counts() {
    let Some(ctx) = ctx_or_skip("multicol_packed_odd_rows") else {
        return;
    };
    for &m in &[2usize, 4] {
        let src = variant_src(m);
        let label = format!("mc{m}-packed");
        for &(n, k, seed) in &[(33usize, 256usize, 21u64), (127, 512, 22)] {
            let w = gen_w(n, k, seed);
            let xcols = gen_x_cols(m, k, seed.wrapping_mul(0x77));
            let case = McRun {
                n,
                k,
                m,
                out_f32: false,
                alpha: 1.0,
            };
            let got = run_multicol(ctx, &label, &src, &w, &xcols, &case);
            let want = host_cols(&w, &xcols, n, k, 1.0);
            assert_cols_match(&format!("packed m={m} n={n} k={k}"), &got, &want);
        }
    }
}

#[test]
fn planted_column_stride_swap_is_caught_by_the_parity_harness() {
    let Some(ctx) = ctx_or_skip("multicol_planted_bug") else {
        return;
    };
    let m = 4usize;
    let (n, k, seed) = (33usize, 256usize, 31u64);
    assert_ne!(
        k / 2,
        n,
        "the planted swap is only observable when x and y column strides differ"
    );
    let src = column_stride_swapped_src(m);
    let w = gen_w(n, k, seed);
    let xcols = gen_x_cols(m, k, seed.wrapping_mul(0x1337));
    let case = McRun {
        n,
        k,
        m,
        out_f32: true,
        alpha: 1.0,
    };
    let got = run_multicol(ctx, "mc4-stride-swap-planted", &src, &w, &xcols, &case);
    let want = host_cols(&w, &xcols, n, k, 1.0);
    let (d, col, row) = max_rel_mismatch(&got, &want);
    assert!(
        d >= KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES,
        "column-stride swap survived parity (worst rel {d:.3e} at col {col} row {row}); \
         the harness would miss a real column-indexing regression"
    );
}

struct BenchRig {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    groups: (u32, u32, u32),
    dispatches_per_step: usize,
}

fn dispatch_steps(ctx: &WgpuContext, rig: &BenchRig, steps: usize) {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&rig.pipeline);
        for _ in 0..steps {
            for bg in &rig.bind_groups {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(rig.groups.0, rig.groups.1, rig.groups.2);
            }
        }
    }
    ctx.queue.submit([enc.finish()]);
}

fn time_rig(ctx: &WgpuContext, rig: &BenchRig, steps: usize) -> f64 {
    dispatch_steps(ctx, rig, 5);
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    dispatch_steps(ctx, rig, steps);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore]
fn m_singlecol_dispatches_vs_one_multicol_dispatch_weight_amortization_bench() {
    let Some(ctx) = ctx_or_skip("multicol_bench") else {
        return;
    };
    let k = 5120usize;
    let k_words = (k / 2) as u32;
    for &(n, steps) in &[(5120usize, 200usize), (12288, 100)] {
        let pairs = n.div_ceil(2);
        let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
        let weight_bytes = (n * k * 2) as f64;
        let w = gen_w(n, k, 0xbe0c ^ n as u64);
        let wb = dispatch::storage_from_slice(ctx, "mcb-w", &pack_u16(&w));
        for &m in &[2usize, 4, 8] {
            let xcols = gen_x_cols(m, k, 0xabc ^ ((n * m) as u64));
            let x_flat: Vec<u16> = xcols.iter().flatten().copied().collect();
            let xb = dispatch::storage_from_slice(ctx, "mcb-x", &pack_u16(&x_flat));
            let yb = dispatch::storage_from_slice(ctx, "mcb-y", &vec![0u32; m * n]);

            let single_pipeline = dispatch::cached_compute_pipeline(
                ctx,
                "mcb-q3d-single",
                &compose(Q3D_GEMV_WGSL),
                Q3D_ENTRY,
            )
            .expect("q3d pipeline");
            let single_bgs: Vec<wgpu::BindGroup> = (0..m)
                .map(|col| {
                    let p = Q3bParams {
                        n_rows: n as u32,
                        k_words,
                        groups_x: groups.0,
                        out_f32: 1,
                        w_row_words: k_words,
                        x_off_words: col as u32 * k_words,
                        y_off_words: (col * n) as u32,
                        alpha: 1.0,
                        ..Default::default()
                    };
                    let ub = dispatch::uniform_from(ctx, "mcb-q3d-p", &p);
                    dispatch::bind_group(
                        ctx,
                        &single_pipeline,
                        &[(0, &wb), (1, &xb), (2, &ub), (3, &yb)],
                    )
                })
                .collect();
            let single = BenchRig {
                pipeline: single_pipeline,
                bind_groups: single_bgs,
                groups,
                dispatches_per_step: m,
            };

            let mc_params = McParams {
                n_rows: n as u32,
                k_words,
                groups_x: groups.0,
                out_f32: 1,
                w_row_words: k_words,
                x_off_words: 0,
                y_off_words: 0,
                x_col_stride_words: k_words,
                y_col_stride_words: n as u32,
                alpha: 1.0,
                ..Default::default()
            };
            let mc_pipeline = dispatch::cached_compute_pipeline(
                ctx,
                &format!("mcb-mc{m}"),
                &variant_src(m),
                MC_ENTRY,
            )
            .expect("multicol pipeline");
            let mc_ub = dispatch::uniform_from(ctx, "mcb-mc-p", &mc_params);
            let mc_bg = dispatch::bind_group(
                ctx,
                &mc_pipeline,
                &[(0, &wb), (1, &xb), (2, &mc_ub), (3, &yb)],
            );
            let multi = BenchRig {
                pipeline: mc_pipeline,
                bind_groups: vec![mc_bg],
                groups,
                dispatches_per_step: 1,
            };

            dispatch_steps(ctx, &multi, 1);
            ctx.poll_blocking().expect("parity poll");
            let words: Vec<u32> = dispatch::read_back(ctx, &yb, m * n).expect("read back");
            let got: Vec<Vec<f32>> = (0..m)
                .map(|c| {
                    words[c * n..(c + 1) * n]
                        .iter()
                        .map(|v| f32::from_bits(*v))
                        .collect()
                })
                .collect();
            let want = host_cols(&w, &xcols, n, k, 1.0);
            assert_cols_match(&format!("bench pre-flight m={m} n={n} k={k}"), &got, &want);

            let mut best_single = f64::MAX;
            let mut best_multi = f64::MAX;
            for _ in 0..5 {
                best_single = best_single.min(time_rig(ctx, &single, steps));
                best_multi = best_multi.min(time_rig(ctx, &multi, steps));
            }
            let ms_single = best_single * 1e3 / steps as f64;
            let ms_multi = best_multi * 1e3 / steps as f64;
            let eff_single = weight_bytes * m as f64 * steps as f64 / best_single / 1e9;
            let eff_multi = weight_bytes * m as f64 * steps as f64 / best_multi / 1e9;
            eprintln!(
                "-- n={n} k={k} m={m} weight={:.1} MB dispatches/step single={} multi={}",
                weight_bytes / 1e6,
                single.dispatches_per_step,
                multi.dispatches_per_step,
            );
            eprintln!(
                "   {m}x single-col {ms_single:>9.4} ms/step  eff {eff_single:>7.1} GB/s per-col-amortized"
            );
            eprintln!(
                "   1x multi-col  {ms_multi:>9.4} ms/step  eff {eff_multi:>7.1} GB/s per-col-amortized  speedup {:.2}x",
                ms_single / ms_multi
            );
        }
    }
}
