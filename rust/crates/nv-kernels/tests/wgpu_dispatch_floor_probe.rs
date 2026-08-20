#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const PROBE_WGSL: &str = "
@group(0) @binding(0) var<storage, read_write> buf_a: array<u32>;
@compute @workgroup_size(32)
fn probe_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) { buf_a[0] = buf_a[0] + 1u; }
}
@compute @workgroup_size(32)
fn probe_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) { buf_a[1] = buf_a[1] + 1u; }
}
";

#[test]
#[ignore = "release-only wall-clock probe of the per-dispatch floor inside one compute pass; \
            set NV_DISPATCH_FLOOR_PROBE=1 -- sizes what dispatch-count fusion can buy before \
            anyone builds it"]
fn what_one_dispatch_costs_inside_a_single_pass() {
    if std::env::var("NV_DISPATCH_FLOOR_PROBE").is_err() {
        eprintln!("skip: NV_DISPATCH_FLOOR_PROBE not set");
        return;
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dispatch-floor-probe"),
            source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
        });
    let mk = |entry: &str| {
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    let pa = mk("probe_a");
    let pb = mk("probe_b");
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-buf"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mk_bg = |pl: &wgpu::ComputePipeline| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe-bg"),
            layout: &pl.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    };
    let bga = mk_bg(&pa);
    let bgb = mk_bg(&pb);

    let time_pass = |n: usize, alternate: bool, reps: usize| -> f64 {
        let run = || {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for i in 0..n {
                    let (p, g) = if alternate && i % 2 == 1 {
                        (&pb, &bgb)
                    } else {
                        (&pa, &bga)
                    };
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, g, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            ctx.queue.submit([enc.finish()]);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        };
        run();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            run();
        }
        t0.elapsed().as_secs_f64() / reps as f64
    };

    for &(n, alt, tag) in &[
        (64usize, false, "same-pipeline"),
        (1146, false, "same-pipeline"),
        (64, true, "alternating"),
        (1146, true, "alternating"),
    ] {
        let s = time_pass(n, alt, 20);
        eprintln!(
            "[dispatch-floor] n={n} {tag}: {:.3} ms/pass, {:.2} us/dispatch",
            s * 1e3,
            s * 1e6 / n as f64
        );
    }
    let base = time_pass(64, true, 20);
    let full = time_pass(1146, true, 20);
    eprintln!(
        "[dispatch-floor] marginal alternating dispatch: {:.2} us -- x1082 extra dispatches = {:.3} ms of a decode step",
        (full - base) * 1e6 / (1146.0 - 64.0),
        (full - base) * 1e3
    );
}
