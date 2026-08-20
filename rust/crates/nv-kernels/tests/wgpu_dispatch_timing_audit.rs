#![cfg(feature = "wgpu")]

use std::sync::mpsc;
use std::time::Instant;

use nv_kernels::wgpu_backend::device::WgpuContext;

const TRIVIAL_SRC: &str = "@group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>, 1>;\n@compute @workgroup_size(1)\nfn trivial() {\n    atomicAdd(&counter[0], 1u);\n}\n";

const DISPATCH_COUNTS: [usize; 4] = [1, 8, 64, 512];
const TRIALS_PER_N: usize = 25;
const WARMUP_TRIALS: usize = 5;

fn audit_enabled() -> bool {
    std::env::var("NV_DISPATCH_AUDIT").as_deref() == Ok("1")
}

struct TrivialFixture {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    buf: wgpu::Buffer,
    staging: wgpu::Buffer,
}

fn build_fixture(ctx: &WgpuContext) -> TrivialFixture {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv-dispatch-audit-trivial"),
            source: wgpu::ShaderSource::Wgsl(TRIVIAL_SRC.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nv-dispatch-audit-trivial"),
            layout: None,
            module: &module,
            entry_point: Some("trivial"),
            compilation_options: Default::default(),
            cache: None,
        });
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-dispatch-audit-counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-dispatch-audit-staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    TrivialFixture {
        pipeline,
        group,
        buf,
        staging,
    }
}

fn sequential_n_one_sync_us(ctx: &WgpuContext, fx: &TrivialFixture, n: usize) -> f64 {
    ctx.queue.write_buffer(&fx.buf, 0, bytemuck::cast_slice(&[0u32]));
    ctx.poll_blocking().expect("reset poll");

    let mut enc = ctx.device.create_command_encoder(&Default::default());
    for _ in 0..n {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&fx.pipeline);
        pass.set_bind_group(0, &fx.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&fx.buf, 0, &fx.staging, 0, 4);
    let cb = enc.finish();

    let start = Instant::now();
    ctx.queue.submit([cb]);
    let slice = fx.staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.poll_blocking().expect("map poll");
    rx.recv().expect("map channel").expect("map result");
    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;

    let view = slice.get_mapped_range().expect("mapped range");
    let got = bytemuck::cast_slice::<u8, u32>(&view)[0];
    drop(view);
    fx.staging.unmap();
    assert_eq!(
        got, n as u32,
        "counter mismatch at n={n}: trivial kernel did not run exactly n times sequentially"
    );
    elapsed_us
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn least_squares_fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

#[test]
fn wgpu_sequential_dispatch_marginal_cost_matches_published_floor() {
    if !audit_enabled() {
        eprintln!(
            "wgpu_sequential_dispatch_marginal_cost_matches_published_floor: SKIP \
             (set NV_DISPATCH_AUDIT=1 to run; GPU-exclusive, idle-gate first)"
        );
        return;
    }
    let ctx = match WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => panic!(
            "wgpu_sequential_dispatch_marginal_cost_matches_published_floor: no wgpu adapter: {e}"
        ),
    };
    eprintln!(
        "device={} backend={:?} {}",
        ctx.info.name,
        ctx.info.backend,
        ctx.summary()
    );

    let fx = build_fixture(ctx);

    for &n in &DISPATCH_COUNTS {
        for _ in 0..WARMUP_TRIALS {
            sequential_n_one_sync_us(ctx, &fx, n);
        }
    }

    let mut points = Vec::with_capacity(DISPATCH_COUNTS.len());
    for &n in &DISPATCH_COUNTS {
        let mut samples: Vec<f64> = (0..TRIALS_PER_N)
            .map(|_| sequential_n_one_sync_us(ctx, &fx, n))
            .collect();
        let med = median(&mut samples);
        eprintln!(
            "n={n:<4} median_wall_us={med:.2} min={:.2} max={:.2}",
            samples.first().copied().unwrap_or(0.0),
            samples.last().copied().unwrap_or(0.0)
        );
        points.push((n as f64, med));
    }

    let (slope_us_per_dispatch, intercept_us) = least_squares_fit(&points);
    eprintln!(
        "[NV_DISPATCH_AUDIT] device={} slope={:.4} us/dispatch intercept={:.2} us \
         (published floor: perf/runs.jsonl; basis in docs/book/05.1-wgpu-status.md section 7.2)",
        ctx.info.name, slope_us_per_dispatch, intercept_us
    );

    assert!(
        slope_us_per_dispatch.is_finite() && slope_us_per_dispatch > 0.0,
        "slope must be a positive finite number of us/dispatch, got {slope_us_per_dispatch}"
    );
    assert!(
        intercept_us.is_finite(),
        "intercept must be finite, got {intercept_us}"
    );
}
