#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;

const ROWS_100544_IS_HALF_THE_GPTOSS_VOCAB_ONE_BALANCED_LMHEAD_CHUNK: u32 = 100544;
const HIDDEN: u32 = 2880;
const K_V4: u32 = HIDDEN / 8;
const K_V4_I8: u32 = HIDDEN / 16;
const ROWS_PER_WG_8_SUBGROUPS_OF_32_MATCHES_THE_SHIPPED_GEMV: u32 = 8;
const REPS: usize = 20;

const PROBE_WGSL: &str = "
@group(0) @binding(0) var<storage, read> wbuf: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> xbuf: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> ybuf: array<u32>;
@group(0) @binding(3) var<storage, read> sbuf: array<f32>;

const ROWS = 100544u;
const KV4 = 360u;
const KV16 = 180u;

fn bf16_lo(w: u32) -> f32 { return bitcast<f32>(w << 16u); }
fn bf16_hi(w: u32) -> f32 { return bitcast<f32>(w & 0xffff0000u); }

fn int8_decode(w: u32, i: u32) -> f32 {
    let b = (w >> (8u * i)) & 0xffu;
    return f32(i32(b << 24u) >> 24u);
}

fn butterfly32(v: f32) -> f32 {
    var a = v;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn dot8(wv: vec4<u32>, xv: vec4<u32>) -> f32 {
    var a = fma(bf16_lo(wv.x), bf16_lo(xv.x), 0.0);
    a = fma(bf16_hi(wv.x), bf16_hi(xv.x), a);
    a = fma(bf16_lo(wv.y), bf16_lo(xv.y), a);
    a = fma(bf16_hi(wv.y), bf16_hi(xv.y), a);
    a = fma(bf16_lo(wv.z), bf16_lo(xv.z), a);
    a = fma(bf16_hi(wv.z), bf16_hi(xv.z), a);
    a = fma(bf16_lo(wv.w), bf16_lo(xv.w), a);
    a = fma(bf16_hi(wv.w), bf16_hi(xv.w), a);
    return a;
}

fn i8_dot16(wv: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    var a = fma(int8_decode(wv.x, 0u), bf16_lo(xa.x), 0.0);
    a = fma(int8_decode(wv.x, 1u), bf16_hi(xa.x), a);
    a = fma(int8_decode(wv.x, 2u), bf16_lo(xa.y), a);
    a = fma(int8_decode(wv.x, 3u), bf16_hi(xa.y), a);
    a = fma(int8_decode(wv.y, 0u), bf16_lo(xa.z), a);
    a = fma(int8_decode(wv.y, 1u), bf16_hi(xa.z), a);
    a = fma(int8_decode(wv.y, 2u), bf16_lo(xa.w), a);
    a = fma(int8_decode(wv.y, 3u), bf16_hi(xa.w), a);
    a = fma(int8_decode(wv.z, 0u), bf16_lo(xb.x), a);
    a = fma(int8_decode(wv.z, 1u), bf16_hi(xb.x), a);
    a = fma(int8_decode(wv.z, 2u), bf16_lo(xb.y), a);
    a = fma(int8_decode(wv.z, 3u), bf16_hi(xb.y), a);
    a = fma(int8_decode(wv.w, 0u), bf16_lo(xb.z), a);
    a = fma(int8_decode(wv.w, 1u), bf16_hi(xb.z), a);
    a = fma(int8_decode(wv.w, 2u), bf16_lo(xb.w), a);
    a = fma(int8_decode(wv.w, 3u), bf16_hi(xb.w), a);
    return a;
}

@compute @workgroup_size(256)
fn lmh_bf16_pure_read(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = wid.x * 8u + sgid;
    let live = row < ROWS;
    let kv4 = select(0u, KV4, live);
    let wbase = select(0u, row * KV4, live);
    var acc = 0u;
    for (var i = lane; i < kv4; i = i + 32u) {
        let wv = wbuf[wbase + i];
        acc = acc + wv.x + wv.y + wv.z + wv.w;
    }
    if (lane == 0u && live) {
        ybuf[row] = acc;
    }
}

@compute @workgroup_size(256)
fn lmh_bf16_dot_butterfly_shipped_shape(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = wid.x * 8u + sgid;
    let live = row < ROWS;
    let kv4 = select(0u, KV4, live);
    let wbase = select(0u, row * KV4, live);
    var acc = 0.0;
    for (var i = lane; i < kv4; i = i + 32u) {
        acc = acc + dot8(wbuf[wbase + i], xbuf[i]);
    }
    let total = butterfly32(acc);
    if (lane == 0u && live) {
        ybuf[row] = bitcast<u32>(total);
    }
}

@compute @workgroup_size(256)
fn lmh_i8_dot_scaled_shipped_shape(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = wid.x * 8u + sgid;
    let live = row < ROWS;
    let kv16 = select(0u, KV16, live);
    let wbase = select(0u, row * KV16, live);
    var acc = 0.0;
    for (var i = lane; i < kv16; i = i + 32u) {
        acc = acc + i8_dot16(wbuf[wbase + i], xbuf[2u * i], xbuf[2u * i + 1u]);
    }
    let total = butterfly32(acc);
    if (lane == 0u && live) {
        ybuf[row] = bitcast<u32>(total * sbuf[row]);
    }
}
";

#[test]
#[ignore = "release-only wall-clock probe: prices the shallow-decode TOP profiled item (the \
            lm_head bf16 GEMV, 16.0% of the depth-256 step) at its REAL work-shape in \
            isolation -- 8 rows/wg, lane-strided vec4 reads, dot8 + 32-lane butterfly -- \
            against a pure-read ceiling and against the existing NV_GOW_LMHEAD_I8 kernel's \
            half-byte dot16 shape. Set NV_GOW_LMHEAD_COST_PROBE=1. The decision it feeds: \
            if the shipped dot lands at the pure-read ceiling, the kernel is exonerated and \
            the only shallow lm_head lever is BYTES, which the dormant i8 arm already halves"]
fn lmhead_gemv_shipped_shape_vs_pure_read_ceiling_vs_i8_half_bytes() {
    if std::env::var("NV_GOW_LMHEAD_COST_PROBE").is_err() {
        panic!(
            "NV_GOW_LMHEAD_COST_PROBE is unset, so the lm_head cost probe executed nothing. \
             It is already #[ignore]d, which makes --ignored the opt-in; a second env gate \
             behind that can only make an --ignored run print `1 passed` for a decision \
             nobody measured. A suite whose precondition is absent must fail or be ignored, \
             never return green."
        );
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    eprintln!("adapter: {}", ctx.summary());
    let rows = ROWS_100544_IS_HALF_THE_GPTOSS_VOCAB_ONE_BALANCED_LMHEAD_CHUNK as u64;
    let bf16_bytes = rows * (HIDDEN as u64) * 2;
    let i8_bytes = rows * (HIDDEN as u64);
    let groups = (rows as u32).div_ceil(ROWS_PER_WG_8_SUBGROUPS_OF_32_MATCHES_THE_SHIPPED_GEMV);

    let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let wbuf = mk("lmh-w", bf16_bytes, st);
    let w8buf = mk("lmh-w8", i8_bytes, st);
    let xbuf = mk("lmh-x", (K_V4 as u64) * 16, st);
    let sbuf = mk("lmh-s", rows * 4, st);
    let ybuf = mk(
        "lmh-y",
        rows * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = mk(
        "lmh-read",
        rows * 4,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );

    let bf16_one = 0x3f80u32;
    let seed: Vec<u8> = (0..(1 << 20) as u32)
        .flat_map(|_| ((bf16_one << 16) | bf16_one).to_le_bytes())
        .collect();
    for (buf, total) in [(&wbuf, bf16_bytes), (&xbuf, (K_V4 as u64) * 16)] {
        let mut off = 0u64;
        while off < total {
            let n = seed.len().min((total - off) as usize);
            ctx.queue.write_buffer(buf, off, &seed[..n]);
            off += n as u64;
        }
    }
    let i8_seed: Vec<u8> = (0..(1 << 20) as u32)
        .flat_map(|_| 0x01010101u32.to_le_bytes())
        .collect();
    let mut off = 0u64;
    while off < i8_bytes {
        let n = i8_seed.len().min((i8_bytes - off) as usize);
        ctx.queue.write_buffer(&w8buf, off, &i8_seed[..n]);
        off += n as u64;
    }
    let ones: Vec<u8> = (0..rows as u32).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    ctx.queue.write_buffer(&sbuf, 0, &ones);

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lmh-cost"),
            source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
        });
    let mk_pipe = |entry: &str| {
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
    let mk_bg = |pipe: &wgpu::ComputePipeline,
                 w: &wgpu::Buffer,
                 bindings_the_auto_layout_kept: &[u32],
                 label: &str| {
        let all: [(u32, &wgpu::Buffer); 4] = [(0, w), (1, &xbuf), (2, &ybuf), (3, &sbuf)];
        let entries: Vec<wgpu::BindGroupEntry> = all
            .iter()
            .filter(|(i, _)| bindings_the_auto_layout_kept.contains(i))
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: *i,
                resource: b.as_entire_binding(),
            })
            .collect();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    };
    let p_pure = mk_pipe("lmh_bf16_pure_read");
    let p_dot = mk_pipe("lmh_bf16_dot_butterfly_shipped_shape");
    let p_i8 = mk_pipe("lmh_i8_dot_scaled_shipped_shape");
    let bg_pure = mk_bg(&p_pure, &wbuf, &[0, 2], "lmh-bg-pure");
    let bg_dot = mk_bg(&p_dot, &wbuf, &[0, 1, 2], "lmh-bg-dot");
    let bg_i8 = mk_bg(&p_i8, &w8buf, &[0, 1, 2, 3], "lmh-bg-i8");

    let time = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| -> f64 {
        for _ in 0..3 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        t0.elapsed().as_secs_f64() / REPS as f64
    };

    let s_pure = time(&p_pure, &bg_pure);
    let s_dot = time(&p_dot, &bg_dot);
    let s_i8 = time(&p_i8, &bg_i8);

    {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&ybuf, 0, &readback, 0, rows * 4);
        ctx.queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range().expect("map lm_head readback");
        let any_nonzero = data.chunks_exact(4).any(|c| c != [0u8; 4]);
        assert!(
            any_nonzero,
            "every row sank zero: the dot chains were elided and the timings measured \
             launch overhead, not the GEMV"
        );
    }

    eprintln!(
        "[lmh-cost] bf16 pure read:      {:.3} ms  {:.1} GB/s (the pattern ceiling for the \
         shipped 8-rows/wg lane-strided shape)",
        s_pure * 1e3,
        bf16_bytes as f64 / s_pure / 1e9
    );
    eprintln!(
        "[lmh-cost] bf16 dot+butterfly:  {:.3} ms  {:.1} GB/s (the shipped \
         gow_fu_gemv_bf16_sg work-shape)",
        s_dot * 1e3,
        bf16_bytes as f64 / s_dot / 1e9
    );
    eprintln!(
        "[lmh-cost] i8 dot16+scale:      {:.3} ms  {:.1} GB/s of i8 bytes -- {:.2}x over the \
         bf16 dot (the NV_GOW_LMHEAD_I8 kernel's work-shape, half the bytes)",
        s_i8 * 1e3,
        i8_bytes as f64 / s_i8 / 1e9,
        s_dot / s_i8
    );
    eprintln!(
        "[lmh-cost] dot-vs-pure gap {:.2}x at rows={rows} k={HIDDEN}: above ~1.15x the kernel \
         owns a mechanism worth fixing; at ~1.0x the kernel is exonerated and bytes are the \
         only lever",
        s_dot / s_pure
    );
}
