#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;

const NPOS: u32 = 122880;
const NKV: u32 = 8;
const HD_WORDS_16_IS_THE_GPTOSS_64B_FP8_ROW: u32 = 16;
const SPLITS: u32 = 64;
const WARPS: u32 = 8;
const REPS: usize = 20;

const PROBE_WGSL: &str = "
@group(0) @binding(0) var<storage, read> kbuf: array<u32>;
@group(0) @binding(1) var<storage, read> vbuf: array<u32>;
@group(0) @binding(2) var<storage, read> ivbuf: array<u32>;
@group(0) @binding(3) var<storage, read_write> sink: array<u32>;
var<workgroup> part: array<u32, 256>;

const NPOS = 122880u;
const NKV = 8u;
const HDW = 16u;
const SPLITS = 64u;
const WARPS = 8u;

fn sgred(x: f32) -> f32 {
    var a = x;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn sgred_pk(x: u32) -> u32 {
    var a = x;
    a = pack2x16float(unpack2x16float(a) + unpack2x16float(subgroupShuffleXor(a, 16u)));
    a = pack2x16float(unpack2x16float(a) + unpack2x16float(subgroupShuffleXor(a, 8u)));
    a = pack2x16float(unpack2x16float(a) + unpack2x16float(subgroupShuffleXor(a, 4u)));
    a = pack2x16float(unpack2x16float(a) + unpack2x16float(subgroupShuffleXor(a, 2u)));
    a = pack2x16float(unpack2x16float(a) + unpack2x16float(subgroupShuffleXor(a, 1u)));
    return a;
}

fn reduce_and_sink(lid: u32, wgflat: u32, acc_in: u32) {
    part[lid] = acc_in;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { part[lid] = part[lid] + part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) { sink[wgflat] = part[0]; }
}

@compute @workgroup_size(256)
fn granule64_split_kv(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * HDW;
        if (lane < HDW) {
            acc = acc + kbuf[rowb + lane];
            acc = acc + vbuf[rowb + lane];
        }
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_sgadd8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * HDW;
        var pt = 0.0;
        if (lane < HDW) {
            acc = acc + kbuf[rowb + lane];
            acc = acc + vbuf[rowb + lane];
            pt = f32(acc & 15u);
        }
        var sg = 0.0;
        sg = sg + sgred(pt);
        sg = sg + sgred(pt + 1.0);
        sg = sg + sgred(pt + 2.0);
        sg = sg + sgred(pt + 3.0);
        sg = sg + sgred(pt + 4.0);
        sg = sg + sgred(pt + 5.0);
        sg = sg + sgred(pt + 6.0);
        sg = sg + sgred(pt + 7.0);
        acc = acc + (u32(sg) & 1u);
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_sgadd1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * HDW;
        var pt = 0.0;
        if (lane < HDW) {
            acc = acc + kbuf[rowb + lane];
            acc = acc + vbuf[rowb + lane];
            pt = f32(acc & 15u);
        }
        var sg = 0.0;
        sg = sg + sgred(pt + 0.0);
        acc = acc + (u32(sg) & 1u);
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_sgadd2(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * HDW;
        var pt = 0.0;
        if (lane < HDW) {
            acc = acc + kbuf[rowb + lane];
            acc = acc + vbuf[rowb + lane];
            pt = f32(acc & 15u);
        }
        var sg = 0.0;
        sg = sg + sgred(pt + 0.0);
        sg = sg + sgred(pt + 1.0);
        acc = acc + (u32(sg) & 1u);
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_sgadd4(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * HDW;
        var pt = 0.0;
        if (lane < HDW) {
            acc = acc + kbuf[rowb + lane];
            acc = acc + vbuf[rowb + lane];
            pt = f32(acc & 15u);
        }
        var sg = 0.0;
        sg = sg + sgred(pt + 0.0);
        sg = sg + sgred(pt + 1.0);
        sg = sg + sgred(pt + 2.0);
        sg = sg + sgred(pt + 3.0);
        acc = acc + (u32(sg) & 1u);
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_sgpk4(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * HDW;
        var pt = 0.0;
        if (lane < HDW) {
            acc = acc + kbuf[rowb + lane];
            acc = acc + vbuf[rowb + lane];
            pt = f32(acc & 15u);
        }
        var sg = 0u;
        sg = sg + sgred_pk(pack2x16float(vec2<f32>(pt, pt + 1.0)));
        sg = sg + sgred_pk(pack2x16float(vec2<f32>(pt + 2.0, pt + 3.0)));
        sg = sg + sgred_pk(pack2x16float(vec2<f32>(pt + 4.0, pt + 5.0)));
        sg = sg + sgred_pk(pack2x16float(vec2<f32>(pt + 6.0, pt + 7.0)));
        acc = acc + (sg & 1u);
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_lanepos(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let blk = SPLITS * WARPS * 32u;
    var acc = 0u;
    var base = (split * WARPS + warp) * 32u;
    loop {
        if (base >= NPOS) { break; }
        let p = base + lane;
        var pt = 0.0;
        if (p < NPOS) {
            let rowb = (p * NKV + kvh) * HDW;
            var kacc = 0u;
            for (var w = 0u; w < HDW; w = w + 1u) {
                kacc = kacc + kbuf[rowb + w];
            }
            acc = acc + kacc;
            pt = f32(kacc & 15u);
        }
        var vs = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let pv = base + i;
            if (pv >= NPOS) { continue; }
            var w8 = 0.0;
            w8 = w8 + subgroupShuffle(pt, 0u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, 8u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, 16u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, 24u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, (i + 1u) & 31u);
            w8 = w8 + subgroupShuffle(pt, (i + 2u) & 31u);
            w8 = w8 + subgroupShuffle(pt, (i + 3u) & 31u);
            w8 = w8 + subgroupShuffle(pt, (i + 4u) & 31u);
            let vrow = (pv * NKV + kvh) * HDW;
            let d = lane & 15u;
            vs = vs + f32(vbuf[vrow + d]) * w8;
        }
        acc = acc + (u32(vs) & 1u);
        base = base + blk;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule64_lanepos_scales(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let blk = SPLITS * WARPS * 32u;
    var acc = 0u;
    var base = (split * WARPS + warp) * 32u;
    loop {
        if (base >= NPOS) { break; }
        let p = base + lane;
        var pt = 0.0;
        if (p < NPOS) {
            let rowb = (p * NKV + kvh) * HDW;
            let ks = bitcast<f32>(kbuf[(p * NKV + kvh) & (arrayLength(&kbuf) - 1u)] | 1u);
            let vs = bitcast<f32>(vbuf[(p * NKV + kvh) & (arrayLength(&vbuf) - 1u)] | 1u);
            var kacc = 0u;
            for (var w = 0u; w < HDW; w = w + 1u) {
                kacc = kacc + kbuf[rowb + w];
            }
            acc = acc + kacc;
            pt = f32(kacc & 15u) * ks * vs;
        }
        var vs2 = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let pv = base + i;
            if (pv >= NPOS) { continue; }
            var w8 = 0.0;
            w8 = w8 + subgroupShuffle(pt, 0u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, 8u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, 16u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, 24u + (i & 7u));
            w8 = w8 + subgroupShuffle(pt, (i + 1u) & 31u);
            w8 = w8 + subgroupShuffle(pt, (i + 2u) & 31u);
            w8 = w8 + subgroupShuffle(pt, (i + 3u) & 31u);
            w8 = w8 + subgroupShuffle(pt, (i + 4u) & 31u);
            let vrow = (pv * NKV + kvh) * HDW;
            let d = lane & 15u;
            vs2 = vs2 + f32(vbuf[vrow + d]) * w8;
        }
        acc = acc + (u32(vs2) & 1u);
        base = base + blk;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}

@compute @workgroup_size(256)
fn granule128_interleaved_kv(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let split = wg.x;
    let kvh = wg.y;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let stride = SPLITS * WARPS;
    var acc = 0u;
    var p = split * WARPS + warp;
    loop {
        if (p >= NPOS) { break; }
        let rowb = (p * NKV + kvh) * (2u * HDW);
        acc = acc + ivbuf[rowb + lane];
        p = p + stride;
    }
    reduce_and_sink(lid, wg.y * SPLITS + wg.x, acc);
}
";

#[test]
#[ignore = "release-only wall-clock probe: prices the gptoss stage1 KV read at its REAL access \
            pattern in isolation -- 64B split-K/V rows (the shipped layout) vs one 128B \
            interleaved row per (pos,kvh), identical total bytes. Set NV_KV_GRANULE_PROBE=1. \
            The decision it feeds: if 128B granules do not approach 2x, the KV-interleave \
            surgery in STATUS is pointless and its theory joins the falsified six"]
fn kv_row_granule_64b_split_vs_128b_interleaved_at_the_stage1_stride_pattern() {
    if std::env::var("NV_KV_GRANULE_PROBE").is_err() {
        panic!(
            "NV_KV_GRANULE_PROBE is unset, so the granule probe executed nothing. It is already \
             #[ignore]d, which makes --ignored the opt-in; a second env gate behind that can only \
             make an --ignored run print `1 passed` for a decision nobody measured. A suite whose \
             precondition is absent must fail or be ignored, never return green."
        );
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    let hdw = HD_WORDS_16_IS_THE_GPTOSS_64B_FP8_ROW;
    let row_words = (NPOS * NKV * hdw) as u64;
    let kv_bytes = row_words * 4;
    let iv_bytes = kv_bytes * 2;
    let wgs = (SPLITS * NKV) as u64;

    let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let kbuf = mk("granule-k", kv_bytes, st);
    let vbuf = mk("granule-v", kv_bytes, st);
    let ivbuf = mk("granule-iv", iv_bytes, st);
    let sink = mk(
        "granule-sink",
        wgs * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = mk(
        "granule-read",
        wgs * 4,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );

    let seed: Vec<u8> = (0..(1 << 20) as u32).flat_map(|i| (i | 1).to_le_bytes()).collect();
    let mut off = 0u64;
    while off < kv_bytes {
        let n = seed.len().min((kv_bytes - off) as usize);
        ctx.queue.write_buffer(&kbuf, off, &seed[..n]);
        ctx.queue.write_buffer(&vbuf, off, &seed[..n]);
        off += n as u64;
    }
    let mut off = 0u64;
    while off < iv_bytes {
        let n = seed.len().min((iv_bytes - off) as usize);
        ctx.queue.write_buffer(&ivbuf, off, &seed[..n]);
        off += n as u64;
    }

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kv-granule"),
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
    let p64 = mk_pipe("granule64_split_kv");
    let p128 = mk_pipe("granule128_interleaved_kv");
    let bg64 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("granule-bg64"),
        layout: &p64.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: kbuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vbuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sink.as_entire_binding(),
            },
        ],
    });
    let bg128 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("granule-bg128"),
        layout: &p128.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ivbuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sink.as_entire_binding(),
            },
        ],
    });

    let time = |pipe: &wgpu::ComputePipeline, bgx: &wgpu::BindGroup| -> f64 {
        for _ in 0..3 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bgx, &[]);
                pass.dispatch_workgroups(SPLITS, NKV, 1);
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
                pass.set_bind_group(0, bgx, &[]);
                pass.dispatch_workgroups(SPLITS, NKV, 1);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        t0.elapsed().as_secs_f64() / REPS as f64
    };

    let psg = mk_pipe("granule64_sgadd8");
    let bgsg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("granule-bgsg"),
        layout: &psg.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: kbuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vbuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sink.as_entire_binding(),
            },
        ],
    });
    let mk_bg_ksv = |pipe: &wgpu::ComputePipeline, label: &str| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: kbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sink.as_entire_binding(),
                },
            ],
        })
    };
    let p1 = mk_pipe("granule64_sgadd1");
    let p2 = mk_pipe("granule64_sgadd2");
    let p4 = mk_pipe("granule64_sgadd4");
    let ppk = mk_pipe("granule64_sgpk4");
    let bg1 = mk_bg_ksv(&p1, "bg1");
    let bg2 = mk_bg_ksv(&p2, "bg2");
    let bg4 = mk_bg_ksv(&p4, "bg4");
    let bgpk = mk_bg_ksv(&ppk, "bgpk");
    let plp = mk_pipe("granule64_lanepos");
    let bglp = mk_bg_ksv(&plp, "bglp");
    let s64 = time(&p64, &bg64);
    let s128 = time(&p128, &bg128);
    let ssg = time(&psg, &bgsg);
    let s1 = time(&p1, &bg1);
    let s2 = time(&p2, &bg2);
    let s4 = time(&p4, &bg4);
    let spk = time(&ppk, &bgpk);
    let slp = time(&plp, &bglp);
    let plps = mk_pipe("granule64_lanepos_scales");
    let bglps = mk_bg_ksv(&plps, "bglps");
    let slps = time(&plps, &bglps);

    {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&sink, 0, &readback, 0, wgs * 4);
        ctx.queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range().expect("map granule readback");
        let any_nonzero = data.chunks_exact(4).any(|c| c != [0u8; 4]);
        assert!(
            any_nonzero,
            "every workgroup sank zero: the xor chains were elided and both timings measured \
             launch overhead, not bandwidth"
        );
    }

    let total = (NPOS as u64 * NKV as u64 * 128) as f64;
    eprintln!(
        "[kv-granule] 64B split K/V:   {:.3} ms  {:.1} GB/s (the shipped stage1 layout)",
        s64 * 1e3,
        total / s64 / 1e9
    );
    eprintln!(
        "[kv-granule] 128B interleave: {:.3} ms  {:.1} GB/s (the proposed layout)",
        s128 * 1e3,
        total / s128 / 1e9
    );
    eprintln!(
        "[kv-granule] 64B + 8 subgroupAdd/round: {:.3} ms  {:.1} GB/s (the real stage1 adds \
         8 folds x subgroupAdd per position)",
        ssg * 1e3,
        total / ssg / 1e9
    );
    eprintln!(
        "[kv-granule] butterfly cost curve: x1 {:.3} ms, x2 {:.3} ms, x4 {:.3} ms, x8 {:.3} ms \
         (pure {:.3} ms) -- per-butterfly {:.1} us",
        s1 * 1e3,
        s2 * 1e3,
        s4 * 1e3,
        ssg * 1e3,
        s64 * 1e3,
        (ssg - s1) * 1e6 / 7.0
    );
    eprintln!(
        "[kv-granule] packed 2xf16 x4 butterflies (the fold-pair fix candidate): {:.3} ms \
         {:.1} GB/s vs x8-f32 {:.3} ms",
        spk * 1e3,
        total / spk / 1e9,
        ssg * 1e3
    );
    eprintln!(
        "[kv-granule] lane-per-position (no reduce, 8 broadcasts/pos in the V pass): {:.3} ms \
         {:.1} GB/s -- the llama.cpp vec-FA shape candidate",
        slp * 1e3,
        total / slp / 1e9
    );
    eprintln!(
        "[kv-granule] lane-per-position + per-position scalar scale loads: {:.3} ms {:.1} GB/s \
         -- the unmodeled cost candidate for the realized-62%-of-prediction residual",
        slps * 1e3,
        total / slps / 1e9
    );
    eprintln!(
        "[kv-granule] interleave speedup {:.2}x, sgadd8 slowdown {:.2}x, at NPOS={NPOS} \
         NKV={NKV} splits={SPLITS} warps={WARPS}",
        s64 / s128,
        ssg / s64
    );
}
