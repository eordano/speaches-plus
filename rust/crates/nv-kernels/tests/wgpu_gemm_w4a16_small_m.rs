#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::pack_u16;
use common::widen_u16;
use std::process::Command;
use std::time::{Duration, Instant};

use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemm_w4a16_small_m as sm;
use nv_kernels::wgpu_backend::kernels::gemv_w4a16 as gw;
use common::LcgShift33W4a16Packs as Lcg;
use common::Params as V4Params;

fn run_v4_row(
    ctx: &WgpuContext,
    packed: &[u32],
    scales: &[u16],
    x_row: &[u16],
    n: usize,
    k: usize,
    gs: usize,
) -> Vec<u16> {
    let source = nv_kernels::wgpu_backend::compose(gw::WGSL);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::ROWS_PER_GROUP);
    let params = V4Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
    };
    let packed_buf = dispatch::storage_from_slice(ctx, "sm-parity-v4-packed", packed);
    let scale_buf = dispatch::storage_from_slice(ctx, "sm-parity-v4-scale", &widen_u16(scales));
    let x_buf = dispatch::storage_from_slice(ctx, "sm-parity-v4-x", &pack_u16(x_row));
    let y_buf = dispatch::storage_zeroed(ctx, "sm-parity-v4-y", (n * 4) as u64);
    let params_buf = dispatch::uniform_from(ctx, "sm-parity-v4-params", &params);
    let pipeline = dispatch::cached_compute_pipeline(
        ctx,
        "nv_kernels_gemm_w4a16_small_m_parity_v4",
        &source,
        gw::V4_ENTRY,
    )
    .unwrap_or_else(|e| panic!("v4 pipeline: {e}"));
    let bindings: Vec<(u32, &wgpu::Buffer)> = vec![
        (1, &scale_buf),
        (3, &y_buf),
        (4, &params_buf),
        (gw::V4_PACKED_SLOT, &packed_buf),
        (gw::V4_X_SLOT, &x_buf),
    ];
    let group = dispatch::bind_group(ctx, &pipeline, &bindings);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, n).expect("read_back");
    words.iter().map(|w| (*w & 0xffff) as u16).collect()
}

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
    gs: usize,
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "tiny",
            n: 24,
            k: 128,
            gs: 32,
        },
        Shape {
            name: "qkv-like",
            n: 96,
            k: 256,
            gs: 32,
        },
        Shape {
            name: "gate_up_concat",
            n: 384 + 384,
            k: 256,
            gs: 32,
        },
        Shape {
            name: "non_mult4_n",
            n: 37,
            k: 128,
            gs: 32,
        },
        Shape {
            name: "wide_k",
            n: 40,
            k: 2560,
            gs: 32,
        },
    ]
}

#[test]
fn small_m_matches_m_sequential_v4_dispatches_bitwise() {
    let Some(ctx) = ctx_or_skip("small_m_matches_m_sequential_v4_dispatches_bitwise") else {
        return;
    };
    let mut failures = Vec::new();
    for shape in shapes() {
        for m in sm::MIN_M..=sm::MAX_M {
            let m = m as usize;
            let mut rng = Lcg::new(
                0x534d_314d ^ ((shape.n as u64) << 40) ^ ((shape.k as u64) << 20) ^ (m as u64),
            );
            let packed = rng.packed(shape.n * (shape.k / 8));
            let scales = rng.scales(shape.n * (shape.k / shape.gs));
            let x_rows: Vec<Vec<u16>> = (0..m).map(|_| rng.bf16_words(shape.k, 1.5)).collect();
            let x_flat: Vec<u16> = x_rows.iter().flatten().copied().collect();

            let want: Vec<Vec<u16>> = x_rows
                .iter()
                .map(|row| run_v4_row(ctx, &packed, &scales, row, shape.n, shape.k, shape.gs))
                .collect();

            let mut y = vec![0u16; m * shape.n];
            sm::gemm_w4a16_small_m(
                ctx, &packed, &scales, &x_flat, &mut y, shape.n, shape.k, shape.gs, m,
            )
            .unwrap_or_else(|e| panic!("{}: M={m}: dispatch failed: {e}", shape.name));

            let mut mismatch = 0usize;
            for t in 0..m {
                for row in 0..shape.n {
                    if y[t * shape.n + row] != want[t][row] {
                        mismatch += 1;
                    }
                }
            }
            eprintln!(
                "gemm_w4a16_small_m {:<16} n={:<5} k={:<5} gs={} M={m} | {mismatch} words off",
                shape.name, shape.n, shape.k, shape.gs
            );
            if mismatch != 0 {
                failures.push(format!("{} M={m}: {mismatch} words mismatched", shape.name));
            }
        }
    }
    assert!(failures.is_empty(), "parity failures: {failures:#?}");
}

fn idle_pct() -> Option<f64> {
    let out = Command::new("top").arg("-l").arg("1").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("CPU usage: ") {
            if let Some(idx) = rest.find("% idle") {
                let start = rest[..idx].rfind(' ').map(|p| p + 1).unwrap_or(0);
                if let Ok(v) = rest[start..idx].parse::<f64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn wait_for_idle(threshold: f64, max_wait: Duration) -> bool {
    let start = Instant::now();
    loop {
        match idle_pct() {
            Some(p) if p >= threshold => return true,
            Some(p) => eprintln!("small_m_bench: idle {p:.1}% < {threshold}%, waiting..."),
            None => eprintln!("small_m_bench: could not parse `top -l 1` idle reading"),
        }
        if start.elapsed() >= max_wait {
            return false;
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn bench_v4_m_times(
    ctx: &WgpuContext,
    packed: &wgpu::Buffer,
    scale: &wgpu::Buffer,
    x_bufs: &[wgpu::Buffer],
    params_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    groups: (u32, u32, u32),
    iters: usize,
) -> f64 {
    let source = nv_kernels::wgpu_backend::compose(gw::WGSL);
    let pipeline = dispatch::cached_compute_pipeline(
        ctx,
        "nv_kernels_gemm_w4a16_small_m_bench_v4",
        &source,
        gw::V4_ENTRY,
    )
    .expect("v4 pipeline");
    let groups_per_m: Vec<wgpu::BindGroup> = x_bufs
        .iter()
        .map(|xb| {
            let bindings: Vec<(u32, &wgpu::Buffer)> = vec![
                (1, scale),
                (3, y_buf),
                (4, params_buf),
                (gw::V4_PACKED_SLOT, packed),
                (gw::V4_X_SLOT, xb),
            ];
            dispatch::bind_group(ctx, &pipeline, &bindings)
        })
        .collect();
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for _ in 0..count {
                for group in &groups_per_m {
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
                }
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(5);
    ctx.poll_blocking().expect("warmup poll");
    let start = Instant::now();
    submit(iters);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore = "GPU rate measurement that waits for an idle card; run explicitly with --ignored"]
fn small_m_bench_rows_per_us_vs_m_gemv_baseline() {
    let Some(ctx) = ctx_or_skip("small_m_bench_rows_per_us_vs_m_gemv_baseline") else {
        return;
    };

    let idle = wait_for_idle(85.0, Duration::from_secs(15 * 60));
    if !idle {
        eprintln!("small_m_bench: PROVISIONAL -- machine did not reach 85% idle within 15 min");
    }

    let bench_shapes = [
        ("qkv", 3072usize, 2560usize),
        ("o", 2560, 2048),
        ("gate_up", 20480, 2560),
        ("down", 2560, 10240),
    ];
    const GS: usize = 32;
    let iters = 100usize;

    println!(
        "\n{:<10} {:>6} {:>4} | {:>10} {:>14} {:>14} {:>10}",
        "shape", "n", "M", "small_m ms", "small_m rows/us", "m*v4 rows/us", "speedup"
    );
    for (name, n, k) in bench_shapes {
        let mut rng = Lcg::new(0x5342_454e ^ ((n as u64) << 32) ^ k as u64);
        let packed = rng.packed(n * (k / 8));
        let scales = rng.scales(n * (k / GS));

        for m in sm::MIN_M..=sm::MAX_M {
            let m = m as usize;
            let x_rows: Vec<Vec<u16>> = (0..m).map(|_| rng.bf16_words(k, 1.5)).collect();
            let x_flat: Vec<u16> = x_rows.iter().flatten().copied().collect();

            let packed_buf = dispatch::storage_from_slice(ctx, "sm-bench-packed", &packed);
            let scale_buf =
                dispatch::storage_from_slice(ctx, "sm-bench-scale", &widen_u16(&scales));
            let x_buf = dispatch::storage_from_slice(ctx, "sm-bench-x", &pack_u16(&x_flat));
            let y_buf = dispatch::storage_zeroed(ctx, "sm-bench-y", ((m * n) as u64) * 4);
            let groups = dispatch::workgroup_count_1d(ctx, n as u64, sm::ROWS_PER_GROUP);
            let params = sm_bench_params(n, k, GS, groups.0);
            let params_buf = dispatch::uniform_from(ctx, "sm-bench-params", &params);
            let entry = sm::entry_for(m as u32).unwrap();
            let source = sm::small_m_source();
            let pipeline = dispatch::cached_compute_pipeline(
                ctx,
                "nv_kernels_gemm_w4a16_small_m_bench",
                &source,
                &entry,
            )
            .unwrap_or_else(|e| panic!("small_m pipeline M={m}: {e}"));
            let bindings: Vec<(u32, &wgpu::Buffer)> = vec![
                (0, &packed_buf),
                (1, &scale_buf),
                (2, &x_buf),
                (3, &y_buf),
                (4, &params_buf),
            ];
            let group = dispatch::bind_group(ctx, &pipeline, &bindings);
            let submit = |count: usize| {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    for _ in 0..count {
                        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
                    }
                }
                ctx.queue.submit([enc.finish()]);
            };
            submit(5);
            ctx.poll_blocking().expect("warmup poll");
            let start = Instant::now();
            submit(iters);
            ctx.poll_blocking().expect("timed poll");
            let small_m_secs = start.elapsed().as_secs_f64();

            let v4_packed = dispatch::storage_from_slice(ctx, "sm-bench-v4-packed", &packed);
            let v4_scale =
                dispatch::storage_from_slice(ctx, "sm-bench-v4-scale", &widen_u16(&scales));
            let v4_x_bufs: Vec<wgpu::Buffer> = x_rows
                .iter()
                .map(|row| dispatch::storage_from_slice(ctx, "sm-bench-v4-x", &pack_u16(row)))
                .collect();
            let v4_y = dispatch::storage_zeroed(ctx, "sm-bench-v4-y", (n * 4) as u64);
            let v4_params_buf = dispatch::uniform_from(
                ctx,
                "sm-bench-v4-params",
                &V4Params {
                    n_rows: n as u32,
                    k_elems: k as u32,
                    gs: GS as u32,
                    w_row_words: (k / 8) as u32,
                    scale_row_stride: (k / GS) as u32,
                    groups_x: groups.0,
                },
            );
            let v4_secs = bench_v4_m_times(
                ctx,
                &v4_packed,
                &v4_scale,
                &v4_x_bufs,
                &v4_params_buf,
                &v4_y,
                groups,
                iters,
            );

            let small_m_ms = small_m_secs * 1e3 / iters as f64;
            let v4_ms = v4_secs * 1e3 / iters as f64;
            let small_m_rows_per_us = (n * m) as f64 / (small_m_ms * 1e3);
            let v4_rows_per_us = (n * m) as f64 / (v4_ms * 1e3);
            let speedup = v4_ms / small_m_ms;
            println!(
                "{name:<10} {n:>6} {m:>4} | {small_m_ms:>10.4} {small_m_rows_per_us:>14.2} {v4_rows_per_us:>14.2} {speedup:>9.2}x{}",
                if idle { "" } else { " (PROVISIONAL)" }
            );
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SmParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    x_stride_words: u32,
    y_stride_words: u32,
}

fn sm_bench_params(n: usize, k: usize, gs: usize, groups_x: u32) -> SmParams {
    SmParams {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x,
        x_stride_words: (k / 2) as u32,
        y_stride_words: n as u32,
    }
}
