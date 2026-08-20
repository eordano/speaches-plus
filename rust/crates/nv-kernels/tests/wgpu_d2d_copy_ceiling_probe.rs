#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;

const STREAM_WGSL: &str = "
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(256)
fn stream_read(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&src);
    var acc = vec4<u32>(0u);
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        acc = acc ^ src[i];
        i = i + 262144u;
    }
    if (gid.x == 0u) { dst[0] = acc; }
    if (acc.x == 0xdeadbeefu && acc.y == 0x1234u) { dst[1] = acc; }
}
";

#[test]
#[ignore = "release-only wall-clock D2D ceiling probe for the floor ledger; set \
            NV_D2D_CEILING_PROBE=1; prints copy-engine GB/s (2N bytes moved per copy) \
            and shader stream-read GB/s on the active adapter"]
fn d2d_copy_and_stream_read_ceiling() {
    if std::env::var("NV_D2D_CEILING_PROBE").is_err() {
        eprintln!("skip: NV_D2D_CEILING_PROBE not set");
        return;
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    let n_bytes: u64 = 1024 * 1024 * 1024;
    let mk = |label: &str, usage: wgpu::BufferUsages| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: n_bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let src = mk(
        "d2d-src",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let dst = mk(
        "d2d-dst",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let seed: Vec<u8> = (0..4096u32).flat_map(|i| i.to_le_bytes()).collect();
    ctx.queue.write_buffer(&src, 0, &seed);

    let time_reps = |f: &dyn Fn(&mut wgpu::CommandEncoder), reps: usize| -> f64 {
        for warm in 0..2 {
            let _ = warm;
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            f(&mut enc);
            ctx.queue.submit([enc.finish()]);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            f(&mut enc);
            ctx.queue.submit([enc.finish()]);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        t0.elapsed().as_secs_f64() / reps as f64
    };

    let copy_s = time_reps(
        &|enc| enc.copy_buffer_to_buffer(&src, 0, &dst, 0, n_bytes),
        10,
    );
    eprintln!(
        "[d2d-ceiling] copy_buffer_to_buffer {} MiB: {:.3} ms, moved(2N) {:.1} GB/s, one-way {:.1} GB/s",
        n_bytes >> 20,
        copy_s * 1e3,
        (2 * n_bytes) as f64 / copy_s / 1e9,
        n_bytes as f64 / copy_s / 1e9
    );

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("d2d-stream"),
            source: wgpu::ShaderSource::Wgsl(STREAM_WGSL.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_read"),
            layout: None,
            module: &module,
            entry_point: Some("stream_read"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("d2d-stream-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst.as_entire_binding(),
            },
        ],
    });
    let stream_s = time_reps(
        &|enc| {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1024, 1, 1);
        },
        10,
    );
    eprintln!(
        "[d2d-ceiling] shader stream-read {} MiB: {:.3} ms, {:.1} GB/s",
        n_bytes >> 20,
        stream_s * 1e3,
        n_bytes as f64 / stream_s / 1e9
    );
}
