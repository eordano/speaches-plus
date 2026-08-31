#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;

const STREAM_THREADS: u32 = 262144;
const STREAM_WGS: u32 = 1024;
const STREAM_WG_SIZE: u32 = 256;
const SEED_BYTES: usize = 4096;
const DEFAULT_SIZES_MIB: &[u64] = &[256, 512, 1024, 2048, 4096];
const DEFAULT_REPS: usize = 20;

const STREAM_WGSL: &str = "
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;
var<workgroup> part: array<vec4<u32>, 256>;
@compute @workgroup_size(256)
fn stream_read(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = arrayLength(&src);
    var acc = vec4<u32>(0u);
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        acc = acc ^ src[i];
        i = i + 262144u;
    }
    part[lid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { part[lid] = part[lid] ^ part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) { dst[wid.x] = part[0]; }
}
";

fn env_sizes_mib() -> Vec<u64> {
    match std::env::var("NV_D2D_CEILING_SIZES_MIB") {
        Ok(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<u64>().ok())
            .filter(|m| *m > 0)
            .collect(),
        Err(_) => DEFAULT_SIZES_MIB.to_vec(),
    }
}

#[test]
#[ignore = "release-only wall-clock D2D read-ceiling probe: this is the DENOMINATOR the \
            speed-of-light floor ledger divides by, so it sweeps buffer sizes past L2 and \
            xor-reduces every lane into a host-checked sink -- a probe whose loads can be \
            elided reports the launch overhead, not the bandwidth. Set NV_D2D_CEILING_PROBE=1; \
            NV_D2D_CEILING_SIZES_MIB / NV_D2D_CEILING_REPS override the sweep"]
fn d2d_copy_and_stream_read_ceiling() {
    if std::env::var("NV_D2D_CEILING_PROBE").is_err() {
        panic!(
            "NV_D2D_CEILING_PROBE is unset, so this ceiling probe executed nothing -- and it is \
             the DENOMINATOR the speed-of-light floor ledger divides by. It is already \
             #[ignore]d, which makes --ignored the opt-in; a second env gate behind that can only \
             make an --ignored run print `1 passed` for a denominator nobody measured. A suite \
             whose precondition is absent must fail or be ignored, never return green."
        );
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    let limits = ctx.device.limits();
    let reps: usize = std::env::var("NV_D2D_CEILING_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    eprintln!(
        "[d2d-ceiling] limits max_buffer_size {} MiB, max_storage_binding {} MiB, reps {}",
        limits.max_buffer_size >> 20,
        limits.max_storage_buffer_binding_size >> 20,
        reps
    );

    let seed: Vec<u8> = (0..(SEED_BYTES / 4) as u32)
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let mut expect = [0u32; 4];
    for chunk in seed.chunks_exact(16) {
        for (l, w) in chunk.chunks_exact(4).enumerate() {
            expect[l] ^= u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        }
    }

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

    for mib in env_sizes_mib() {
        let n_bytes: u64 = mib << 20;
        if n_bytes > limits.max_buffer_size || n_bytes > limits.max_storage_buffer_binding_size {
            eprintln!("[d2d-ceiling] skip {mib} MiB: over device limits");
            continue;
        }
        assert_eq!(
            n_bytes % (STREAM_THREADS as u64 * 16),
            0,
            "GRID_STRIDE_LOOP_NEEDS_A_WHOLE_NUMBER_OF_PASSES_OR_THE_TAIL_LANES_IDLE: {mib} MiB is \
             not a multiple of {STREAM_THREADS} vec4 lanes"
        );
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let src = mk(
            "d2d-src",
            n_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let dst_bytes = (STREAM_WGS as u64) * 16;
        let dst = mk(
            "d2d-dst",
            n_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = mk(
            "d2d-readback",
            dst_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        ctx.queue.write_buffer(&src, 0, &seed);

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

        let time_reps = |f: &dyn Fn(&mut wgpu::CommandEncoder), reps: usize| -> f64 {
            for _ in 0..2 {
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
            reps.min(10),
        );
        eprintln!(
            "[d2d-ceiling] copy_buffer_to_buffer {mib} MiB: {:.3} ms, moved(2N) {:.1} GB/s, one-way {:.1} GB/s",
            copy_s * 1e3,
            (2 * n_bytes) as f64 / copy_s / 1e9,
            n_bytes as f64 / copy_s / 1e9
        );

        let stream_s = time_reps(
            &|enc| {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(STREAM_WGS, 1, 1);
            },
            reps,
        );
        eprintln!(
            "[d2d-ceiling] shader stream-read {mib} MiB: {:.3} ms, {:.1} GB/s, wgs {} x {} threads",
            stream_s * 1e3,
            n_bytes as f64 / stream_s / 1e9,
            STREAM_WGS,
            STREAM_WG_SIZE
        );

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&dst, 0, &readback, 0, dst_bytes);
        ctx.queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let mut got = [0u32; 4];
        {
            let view = slice
                .get_mapped_range()
                .expect("readback map for the read-sink assertion");
            for chunk in view.chunks_exact(16) {
                for (l, w) in chunk.chunks_exact(4).enumerate() {
                    got[l] ^= u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                }
            }
        }
        readback.unmap();
        assert_eq!(
            got, expect,
            "READ_SINK_XOR_OVER_ALL_LANES_PROVES_THE_LOADS_HAPPENED: a stream-read ceiling whose \
             loads were elided reports dispatch overhead as bandwidth; got {got:?} want {expect:?} \
             at {mib} MiB"
        );
    }
}
