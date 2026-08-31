#![cfg(feature = "wgpu")]

use std::time::Instant;

use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;

const LEN_WORDS: usize = 64;
const LEN_BYTES: u64 = (LEN_WORDS * 4) as u64;
const DISPATCH_COUNTS: [usize; 3] = [64, 256, 1024];
const WALL_REPS: usize = 15;
const GPU_TS_REPS: usize = 5;
const WARMUP: usize = 4;
const FAN_SIBLINGS: usize = 3;
const CHUNKED_SUBMIT_COUNT: usize = 8;
const INDEP_MAX: usize = 1024;

const CHAIN_READ_SRC: &str = "\
@group(0) @binding(0) var<storage, read> src: array<u32, 64>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32, 64>;
@compute @workgroup_size(64)
fn chain_step(@builtin(local_invocation_index) i: u32) { dst[i] = src[i] + 1u; }
";

const CHAIN_RW_SRC: &str = "\
@group(0) @binding(0) var<storage, read_write> src: array<u32, 64>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32, 64>;
@compute @workgroup_size(64)
fn chain_step_rw(@builtin(local_invocation_index) i: u32) { dst[i] = src[i] + 1u; }
";

const BUMP_SRC: &str = "\
@group(0) @binding(0) var<storage, read_write> acc: array<u32, 64>;
@compute @workgroup_size(64)
fn bump(@builtin(local_invocation_index) i: u32) { acc[i] = acc[i] + 1u; }
";

const WIDE_SRC: &str = "\
@group(0) @binding(0) var<storage, read> wsrc: array<u32, 64>;
@group(0) @binding(1) var<storage, read_write> wdst: array<u32, 64>;
@group(0) @binding(2) var<storage, read> w2: array<u32, 64>;
@group(0) @binding(3) var<storage, read> w3: array<u32, 64>;
@group(0) @binding(4) var<storage, read> w4: array<u32, 64>;
@group(0) @binding(5) var<storage, read> w5: array<u32, 64>;
@group(0) @binding(6) var<storage, read> w6: array<u32, 64>;
@group(0) @binding(7) var<storage, read> w7: array<u32, 64>;
@compute @workgroup_size(64)
fn chain_step_wide(@builtin(local_invocation_index) i: u32) {
    wdst[i] = wsrc[i] + 1u + (w2[i] & 0u) + (w3[i] & 0u) + (w4[i] & 0u)
        + (w5[i] & 0u) + (w6[i] & 0u) + (w7[i] & 0u);
}
";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    RawChainOnePass,
    RawChainRwBoundOnePass,
    RawChainWideBindOnePass,
    RawChainIndirectOnePass,
    SameBufferBumpSetOnce,
    SameBufferBumpResetEach,
    IndependentFanoutOnePass,
    Fan4NarrowedReadSiblings,
    Fan4RwBoundSiblings,
    RawChainPassPerDispatch,
    RawChainPassPerDispatchTsEach,
    RawChainOnePassTsInsideEach,
    RawChainEncoderPerDispatch,
    RawChainSubmitPerDispatch,
    RawChainChunkedSubmits,
}

const ARMS: [Arm; 15] = [
    Arm::RawChainOnePass,
    Arm::RawChainRwBoundOnePass,
    Arm::RawChainWideBindOnePass,
    Arm::RawChainIndirectOnePass,
    Arm::SameBufferBumpSetOnce,
    Arm::SameBufferBumpResetEach,
    Arm::IndependentFanoutOnePass,
    Arm::Fan4NarrowedReadSiblings,
    Arm::Fan4RwBoundSiblings,
    Arm::RawChainPassPerDispatch,
    Arm::RawChainPassPerDispatchTsEach,
    Arm::RawChainOnePassTsInsideEach,
    Arm::RawChainEncoderPerDispatch,
    Arm::RawChainSubmitPerDispatch,
    Arm::RawChainChunkedSubmits,
];

fn arm_name(a: Arm) -> &'static str {
    match a {
        Arm::RawChainOnePass => "raw_chain_one_pass",
        Arm::RawChainRwBoundOnePass => "raw_chain_rw_bound_one_pass",
        Arm::RawChainWideBindOnePass => "raw_chain_wide_bind_one_pass",
        Arm::RawChainIndirectOnePass => "raw_chain_indirect_one_pass",
        Arm::SameBufferBumpSetOnce => "same_buffer_bump_set_once",
        Arm::SameBufferBumpResetEach => "same_buffer_bump_reset_each",
        Arm::IndependentFanoutOnePass => "independent_fanout_one_pass",
        Arm::Fan4NarrowedReadSiblings => "fan4_narrowed_read_siblings",
        Arm::Fan4RwBoundSiblings => "fan4_rw_bound_siblings",
        Arm::RawChainPassPerDispatch => "raw_chain_pass_per_dispatch",
        Arm::RawChainPassPerDispatchTsEach => "raw_chain_pass_per_disp_ts_each",
        Arm::RawChainOnePassTsInsideEach => "raw_chain_one_pass_ts_inside",
        Arm::RawChainEncoderPerDispatch => "raw_chain_encoder_per_dispatch",
        Arm::RawChainSubmitPerDispatch => "raw_chain_submit_per_dispatch",
        Arm::RawChainChunkedSubmits => "raw_chain_chunked_submits_8",
    }
}

struct Fixture {
    p_chain: wgpu::ComputePipeline,
    p_chain_rw: wgpu::ComputePipeline,
    p_bump: wgpu::ComputePipeline,
    p_wide: wgpu::ComputePipeline,
    a: wgpu::Buffer,
    c: wgpu::Buffer,
    x: wgpu::Buffer,
    fan_dsts: Vec<wgpu::Buffer>,
    indep_dsts: Vec<wgpu::Buffer>,
    indirect: wgpu::Buffer,
    bg_ab: wgpu::BindGroup,
    bg_ba: wgpu::BindGroup,
    bgrw_ab: wgpu::BindGroup,
    bgrw_ba: wgpu::BindGroup,
    bgw_ab: wgpu::BindGroup,
    bgw_ba: wgpu::BindGroup,
    bg_bump_c: wgpu::BindGroup,
    bg_bump_x: wgpu::BindGroup,
    bg_fan_read: Vec<wgpu::BindGroup>,
    bg_fan_rw: Vec<wgpu::BindGroup>,
    bg_indep: Vec<wgpu::BindGroup>,
    ts: Option<TsRig>,
}

struct TsRig {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    period: f64,
}

fn storage(ctx: &WgpuContext, label: &str) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: LEN_BYTES,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn pipeline(ctx: &WgpuContext, label: &str, src: &str, entry: &str) -> wgpu::ComputePipeline {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
}

fn bg2(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    b0: &wgpu::Buffer,
    b1: &wgpu::Buffer,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pl.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b1.as_entire_binding(),
            },
        ],
    })
}

fn build_fixture(ctx: &WgpuContext) -> Fixture {
    let p_chain = pipeline(ctx, "serbench-chain", CHAIN_READ_SRC, "chain_step");
    let p_chain_rw = pipeline(ctx, "serbench-chain-rw", CHAIN_RW_SRC, "chain_step_rw");
    let p_bump = pipeline(ctx, "serbench-bump", BUMP_SRC, "bump");
    let p_wide = pipeline(ctx, "serbench-wide", WIDE_SRC, "chain_step_wide");

    let a = storage(ctx, "serbench-a");
    let b = storage(ctx, "serbench-b");
    let c = storage(ctx, "serbench-c");
    let x = storage(ctx, "serbench-x");
    let fan_dsts: Vec<wgpu::Buffer> = (0..FAN_SIBLINGS)
        .map(|i| storage(ctx, &format!("serbench-fan-{i}")))
        .collect();
    let indep_dsts: Vec<wgpu::Buffer> = (0..INDEP_MAX)
        .map(|i| storage(ctx, &format!("serbench-ind-{i}")))
        .collect();
    let indirect = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("serbench-indirect"),
        size: 12,
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&indirect, 0, bytemuck::cast_slice(&[1u32, 1, 1]));

    let wides: Vec<&wgpu::Buffer> = (0..6).map(|i| &indep_dsts[INDEP_MAX - 1 - i]).collect();
    let bgw = |b0: &wgpu::Buffer, b1: &wgpu::Buffer| {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b1.as_entire_binding(),
            },
        ];
        for (i, w) in wides.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 2) as u32,
                resource: w.as_entire_binding(),
            });
        }
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p_wide.get_bind_group_layout(0),
            entries: &entries,
        })
    };

    let bg1 = |pl: &wgpu::ComputePipeline, b0: &wgpu::Buffer| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pl.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b0.as_entire_binding(),
            }],
        })
    };

    let ts = if ctx.caps.timestamp_query {
        Some(TsRig {
            qs: ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("serbench-ts"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("serbench-ts-resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("serbench-ts-staging"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period: ctx.queue.get_timestamp_period() as f64,
        })
    } else {
        None
    };

    Fixture {
        bg_ab: bg2(ctx, &p_chain, &a, &b),
        bg_ba: bg2(ctx, &p_chain, &b, &a),
        bgrw_ab: bg2(ctx, &p_chain_rw, &a, &b),
        bgrw_ba: bg2(ctx, &p_chain_rw, &b, &a),
        bgw_ab: bgw(&a, &b),
        bgw_ba: bgw(&b, &a),
        bg_bump_c: bg1(&p_bump, &c),
        bg_bump_x: bg1(&p_bump, &x),
        bg_fan_read: fan_dsts.iter().map(|d| bg2(ctx, &p_chain, &x, d)).collect(),
        bg_fan_rw: fan_dsts
            .iter()
            .map(|d| bg2(ctx, &p_chain_rw, &x, d))
            .collect(),
        bg_indep: indep_dsts
            .iter()
            .map(|d| bg2(ctx, &p_chain, &a, d))
            .collect(),
        p_chain,
        p_chain_rw,
        p_bump,
        p_wide,
        a,
        c,
        x,
        fan_dsts,
        indep_dsts,
        indirect,
        ts,
    }
}

struct Phases {
    encode_us: f64,
    submit_us: f64,
    wall_us: f64,
    gpu_us: f64,
}

fn tsw<'a>(
    rig: Option<&'a TsRig>,
    i: usize,
    n: usize,
    want: bool,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    let rig = rig.filter(|_| want && (i == 0 || i + 1 == n))?;
    Some(wgpu::ComputePassTimestampWrites {
        query_set: &rig.qs,
        beginning_of_pass_write_index: (i == 0).then_some(0),
        end_of_pass_write_index: (i + 1 == n).then_some(1),
    })
}

fn chain_pair<'a>(fx: &'a Fixture, arm: Arm, i: usize) -> (&'a wgpu::ComputePipeline, &'a wgpu::BindGroup) {
    match arm {
        Arm::RawChainRwBoundOnePass => (
            &fx.p_chain_rw,
            if i % 2 == 0 { &fx.bgrw_ab } else { &fx.bgrw_ba },
        ),
        Arm::RawChainWideBindOnePass => (
            &fx.p_wide,
            if i % 2 == 0 { &fx.bgw_ab } else { &fx.bgw_ba },
        ),
        _ => (
            &fx.p_chain,
            if i % 2 == 0 { &fx.bg_ab } else { &fx.bg_ba },
        ),
    }
}

fn read_gpu_span(ctx: &WgpuContext, rig: &TsRig) -> f64 {
    let slice = rig.staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.poll_blocking().expect("ts poll");
    rx.recv().expect("ts map channel").expect("ts map");
    let view = slice.get_mapped_range().expect("ts mapped range");
    let words = bytemuck::cast_slice::<u8, u64>(&view).to_vec();
    drop(view);
    rig.staging.unmap();
    words[1].saturating_sub(words[0]) as f64 * rig.period / 1e3
}

fn run_arm(ctx: &WgpuContext, fx: &Fixture, arm: Arm, n: usize, want_gpu: bool) -> Phases {
    let rig = fx.ts.as_ref().filter(|_| want_gpu);
    let t0 = Instant::now();
    let mut encode_done = t0;
    let mut submit_done = t0;
    let mut cbs: Vec<wgpu::CommandBuffer> = Vec::new();
    let mut submits_inline = false;

    match arm {
        Arm::RawChainOnePass | Arm::RawChainRwBoundOnePass | Arm::RawChainWideBindOnePass => {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: tsw(rig, 0, 1, true),
                });
                for i in 0..n {
                    let (pl, bg) = chain_pair(fx, arm, i);
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            if let Some(r) = rig {
                enc.resolve_query_set(&r.qs, 0..2, &r.resolve, 0);
                enc.copy_buffer_to_buffer(&r.resolve, 0, &r.staging, 0, 16);
            }
            cbs.push(enc.finish());
        }
        Arm::RawChainIndirectOnePass => {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: tsw(rig, 0, 1, true),
                });
                for i in 0..n {
                    let (pl, bg) = chain_pair(fx, arm, i);
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups_indirect(&fx.indirect, 0);
                }
            }
            if let Some(r) = rig {
                enc.resolve_query_set(&r.qs, 0..2, &r.resolve, 0);
                enc.copy_buffer_to_buffer(&r.resolve, 0, &r.staging, 0, 16);
            }
            cbs.push(enc.finish());
        }
        Arm::SameBufferBumpSetOnce | Arm::SameBufferBumpResetEach => {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: tsw(rig, 0, 1, true),
                });
                if arm == Arm::SameBufferBumpSetOnce {
                    pass.set_pipeline(&fx.p_bump);
                    pass.set_bind_group(0, &fx.bg_bump_c, &[]);
                    for _ in 0..n {
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                } else {
                    for _ in 0..n {
                        pass.set_pipeline(&fx.p_bump);
                        pass.set_bind_group(0, &fx.bg_bump_c, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
            }
            if let Some(r) = rig {
                enc.resolve_query_set(&r.qs, 0..2, &r.resolve, 0);
                enc.copy_buffer_to_buffer(&r.resolve, 0, &r.staging, 0, 16);
            }
            cbs.push(enc.finish());
        }
        Arm::IndependentFanoutOnePass => {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: tsw(rig, 0, 1, true),
                });
                pass.set_pipeline(&fx.p_chain);
                for i in 0..n {
                    pass.set_bind_group(0, &fx.bg_indep[i], &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            if let Some(r) = rig {
                enc.resolve_query_set(&r.qs, 0..2, &r.resolve, 0);
                enc.copy_buffer_to_buffer(&r.resolve, 0, &r.staging, 0, 16);
            }
            cbs.push(enc.finish());
        }
        Arm::Fan4NarrowedReadSiblings | Arm::Fan4RwBoundSiblings => {
            let groups = n / (FAN_SIBLINGS + 1);
            let (sib_pl, sib_bgs) = if arm == Arm::Fan4NarrowedReadSiblings {
                (&fx.p_chain, &fx.bg_fan_read)
            } else {
                (&fx.p_chain_rw, &fx.bg_fan_rw)
            };
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: tsw(rig, 0, 1, true),
                });
                for _ in 0..groups {
                    pass.set_pipeline(&fx.p_bump);
                    pass.set_bind_group(0, &fx.bg_bump_x, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(sib_pl);
                    for bg in sib_bgs {
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
            }
            if let Some(r) = rig {
                enc.resolve_query_set(&r.qs, 0..2, &r.resolve, 0);
                enc.copy_buffer_to_buffer(&r.resolve, 0, &r.staging, 0, 16);
            }
            cbs.push(enc.finish());
        }
        Arm::RawChainPassPerDispatch => {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            for i in 0..n {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: tsw(rig, i, n, true),
                });
                let (pl, bg) = chain_pair(fx, arm, i);
                pass.set_pipeline(pl);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            if let Some(r) = rig {
                enc.resolve_query_set(&r.qs, 0..2, &r.resolve, 0);
                enc.copy_buffer_to_buffer(&r.resolve, 0, &r.staging, 0, 16);
            }
            cbs.push(enc.finish());
        }
        Arm::RawChainPassPerDispatchTsEach => {
            let queries = (n * 2) as u32;
            let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("serbench-ts-each"),
                ty: wgpu::QueryType::Timestamp,
                count: queries,
            });
            let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("serbench-ts-each-resolve"),
                size: queries as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            for i in 0..n {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some((i * 2) as u32),
                        end_of_pass_write_index: Some((i * 2 + 1) as u32),
                    }),
                });
                let (pl, bg) = chain_pair(fx, arm, i);
                pass.set_pipeline(pl);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            enc.resolve_query_set(&qs, 0..queries, &resolve, 0);
            cbs.push(enc.finish());
        }
        Arm::RawChainOnePassTsInsideEach => {
            let queries = (n * 2) as u32;
            let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("serbench-ts-inside"),
                ty: wgpu::QueryType::Timestamp,
                count: queries,
            });
            let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("serbench-ts-inside-resolve"),
                size: queries as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for i in 0..n {
                    pass.write_timestamp(&qs, (i * 2) as u32);
                    let (pl, bg) = chain_pair(fx, arm, i);
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.write_timestamp(&qs, (i * 2 + 1) as u32);
                }
            }
            enc.resolve_query_set(&qs, 0..queries, &resolve, 0);
            cbs.push(enc.finish());
        }
        Arm::RawChainEncoderPerDispatch => {
            for i in 0..n {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    let (pl, bg) = chain_pair(fx, arm, i);
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                cbs.push(enc.finish());
            }
        }
        Arm::RawChainSubmitPerDispatch => {
            submits_inline = true;
            for i in 0..n {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    let (pl, bg) = chain_pair(fx, arm, i);
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                ctx.queue.submit([enc.finish()]);
            }
            encode_done = Instant::now();
            submit_done = encode_done;
        }
        Arm::RawChainChunkedSubmits => {
            submits_inline = true;
            let chunk = n.div_ceil(CHUNKED_SUBMIT_COUNT);
            let mut i = 0;
            while i < n {
                let hi = (i + chunk).min(n);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    for j in i..hi {
                        let (pl, bg) = chain_pair(fx, arm, j);
                        pass.set_pipeline(pl);
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
                ctx.queue.submit([enc.finish()]);
                i = hi;
            }
            encode_done = Instant::now();
            submit_done = encode_done;
        }
    }

    if !submits_inline {
        encode_done = Instant::now();
        ctx.queue.submit(cbs.drain(..));
        submit_done = Instant::now();
    }

    let gpu_ts_encoded = !matches!(
        arm,
        Arm::RawChainPassPerDispatchTsEach
            | Arm::RawChainOnePassTsInsideEach
            | Arm::RawChainEncoderPerDispatch
            | Arm::RawChainSubmitPerDispatch
            | Arm::RawChainChunkedSubmits
    );
    let gpu_us = match rig {
        Some(r) if gpu_ts_encoded => read_gpu_span(ctx, r),
        _ => {
            ctx.poll_blocking().expect("wall poll");
            0.0
        }
    };
    let wall = t0.elapsed().as_secs_f64() * 1e6;
    Phases {
        encode_us: (encode_done - t0).as_secs_f64() * 1e6,
        submit_us: (submit_done - encode_done).as_secs_f64() * 1e6,
        wall_us: wall,
        gpu_us,
    }
}

fn zero(ctx: &WgpuContext, buf: &wgpu::Buffer) {
    ctx.queue.write_buffer(buf, 0, &[0u8; LEN_BYTES as usize]);
}

fn assert_uniform(ctx: &WgpuContext, what: &str, buf: &wgpu::Buffer, want: u32) {
    let got = dispatch::read_back::<u32>(ctx, buf, LEN_WORDS).expect("readback");
    assert!(
        got.iter().all(|&v| v == want),
        "{what}: expected all {want}, got first={} min={} max={}",
        got[0],
        got.iter().min().unwrap(),
        got.iter().max().unwrap()
    );
}

fn verify_arm(ctx: &WgpuContext, fx: &Fixture, arm: Arm, n: usize) {
    match arm {
        Arm::SameBufferBumpSetOnce | Arm::SameBufferBumpResetEach => {
            zero(ctx, &fx.c);
            run_arm(ctx, fx, arm, n, false);
            assert_uniform(ctx, arm_name(arm), &fx.c, n as u32);
        }
        Arm::IndependentFanoutOnePass => {
            zero(ctx, &fx.a);
            for d in fx.indep_dsts.iter().take(n) {
                zero(ctx, d);
            }
            run_arm(ctx, fx, arm, n, false);
            assert_uniform(ctx, arm_name(arm), &fx.indep_dsts[0], 1);
            assert_uniform(ctx, arm_name(arm), &fx.indep_dsts[n - 1], 1);
        }
        Arm::Fan4NarrowedReadSiblings | Arm::Fan4RwBoundSiblings => {
            let groups = (n / (FAN_SIBLINGS + 1)) as u32;
            zero(ctx, &fx.x);
            for d in &fx.fan_dsts {
                zero(ctx, d);
            }
            run_arm(ctx, fx, arm, n, false);
            assert_uniform(ctx, arm_name(arm), &fx.x, groups);
            for d in &fx.fan_dsts {
                assert_uniform(ctx, arm_name(arm), d, groups + 1);
            }
        }
        _ => {
            zero(ctx, &fx.a);
            run_arm(ctx, fx, arm, n, false);
            assert_uniform(ctx, arm_name(arm), &fx.a, n as u32);
        }
    }
}

fn median_of(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "GPU wall-clock microbench of the dependent-dispatch serialization floor; \
            set NV_WGPU_SERBENCH=1 and hold the gpu queue lock"]
fn dependent_dispatch_serialization_cost_by_encoding_shape() {
    if std::env::var("NV_WGPU_SERBENCH").as_deref() != Ok("1") {
        panic!(
            "NV_WGPU_SERBENCH is not 1, so this serialization bench executed nothing. It is \
             already #[ignore]d, which makes --ignored the opt-in; a second env gate behind that \
             can only make an --ignored run print `1 passed` for a bench that measured nothing. A \
             suite whose precondition is absent must fail or be ignored, never return green."
        );
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    eprintln!(
        "[serbench] device={} backend={:?} timestamp_query={}",
        ctx.info.name, ctx.info.backend, ctx.caps.timestamp_query
    );
    let fx = build_fixture(ctx);

    let mut slope_rows: Vec<(Arm, Vec<(usize, f64)>)> = Vec::new();
    for &arm in &ARMS {
        if arm == Arm::RawChainOnePassTsInsideEach && !ctx.ts_inside_passes {
            eprintln!("[serbench] {} SKIP: TIMESTAMP_QUERY_INSIDE_PASSES unavailable", arm_name(arm));
            continue;
        }
        let mut walls = Vec::new();
        for &n in &DISPATCH_COUNTS {
            verify_arm(ctx, &fx, arm, n);
            for _ in 0..WARMUP {
                run_arm(ctx, &fx, arm, n, false);
            }
            let mut wall = Vec::new();
            let mut enc = Vec::new();
            let mut sub = Vec::new();
            for _ in 0..WALL_REPS {
                let p = run_arm(ctx, &fx, arm, n, false);
                wall.push(p.wall_us);
                enc.push(p.encode_us);
                sub.push(p.submit_us);
            }
            let mut gpu = Vec::new();
            for _ in 0..GPU_TS_REPS {
                let p = run_arm(ctx, &fx, arm, n, true);
                if p.gpu_us > 0.0 {
                    gpu.push(p.gpu_us);
                }
            }
            let (w, e, s) = (median_of(wall), median_of(enc), median_of(sub));
            let g = if gpu.is_empty() { 0.0 } else { median_of(gpu) };
            eprintln!(
                "[serbench] {:<32} n={:<5} wall={:>9.1}us enc={:>8.1} sub={:>7.1} gpu={:>9.1} per_disp={:>6.2}us gpu_per_disp={:>6.2}us",
                arm_name(arm),
                n,
                w,
                e,
                s,
                g,
                w / n as f64,
                g / n as f64,
            );
            walls.push((n, w));
        }
        slope_rows.push((arm, walls));
    }

    eprintln!("[serbench] marginal us/dispatch (wall slope n=256 -> n=1024):");
    for (arm, walls) in &slope_rows {
        let w256 = walls.iter().find(|(n, _)| *n == 256).unwrap().1;
        let w1024 = walls.iter().find(|(n, _)| *n == 1024).unwrap().1;
        eprintln!(
            "[serbench-slope] {:<32} {:>7.3} us/dispatch",
            arm_name(*arm),
            (w1024 - w256) / 768.0
        );
    }
}

const E4B_LIKE_PASS_COUNT: usize = 474;
const PROF_REPS: usize = 9;

#[test]
#[ignore = "GPU wall-clock check that the in-pass profiler shape removes the pass-split \
            distortion on the shared submit_profiled path; set NV_WGPU_SERBENCH_PROF=1 \
            and hold the gpu queue lock"]
fn profiled_submit_distortion_pass_split_vs_inpass() {
    if std::env::var("NV_WGPU_SERBENCH_PROF").as_deref() != Ok("1") {
        panic!(
            "NV_WGPU_SERBENCH_PROF is not 1, so this profiled bench executed nothing. It is \
             already #[ignore]d, which makes --ignored the opt-in; a second env gate behind that \
             can only make an --ignored run print `1 passed` for a bench that measured nothing. A \
             suite whose precondition is absent must fail or be ignored, never return green. The \
             two timestamp-capability skips below are a different case: those are hardware this \
             box may genuinely not have."
        );
    }
    let ctx = WgpuContext::shared().expect("wgpu adapter required");
    if !ctx.caps.timestamp_query {
        eprintln!("skip: no timestamp_query");
        return;
    }
    if !ctx.ts_inside_passes {
        eprintln!("skip: TIMESTAMP_QUERY_INSIDE_PASSES unavailable on this device");
        return;
    }
    let fx = build_fixture(ctx);
    let n = E4B_LIKE_PASS_COUNT;
    let passes: Vec<dispatch::PassRef<'_>> = (0..n)
        .map(|i| {
            let (pl, bg) = chain_pair(&fx, Arm::RawChainOnePass, i);
            (pl, bg, (1u32, 1u32, 1u32))
        })
        .collect();
    let labels: Vec<String> = (0..n)
        .map(|i| format!("chain_step_{}", if i % 2 == 0 { "ab" } else { "ba" }))
        .collect();

    for round in 0..2 {
        for (mode, inpass) in [("pass_split", false), ("inpass", true)] {
            if inpass {
                std::env::set_var(dispatch::PROFILE_INPASS_ENV, "1");
            } else {
                std::env::remove_var(dispatch::PROFILE_INPASS_ENV);
            }
            dispatch::profile::reset();
            for _ in 0..2 {
                dispatch::submit_profiled_slices(ctx, &passes, &labels).expect("warmup");
            }
            dispatch::profile::reset();
            let mut walls = Vec::new();
            for _ in 0..PROF_REPS {
                let t0 = Instant::now();
                dispatch::submit_profiled_slices(ctx, &passes, &labels).expect("profiled submit");
                walls.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            let seen = dispatch::profile::dispatches();
            let span_sum_ms = dispatch::profile::total_ns() / 1e6;
            assert_eq!(
                seen,
                (n * PROF_REPS) as u64,
                "profiler must record every dispatch in {mode}"
            );
            let wall = median_of(walls);
            eprintln!(
                "[serbench-prof] round={round} mode={mode:<10} n={n} wall={:>9.1}us per_disp={:>6.2}us span_mean={:>6.2}us span_sum/rep={:>7.3}ms",
                wall,
                wall / n as f64,
                span_sum_ms * 1e3 / seen as f64,
                span_sum_ms / PROF_REPS as f64,
            );
        }
    }
    std::env::remove_var(dispatch::PROFILE_INPASS_ENV);
}
