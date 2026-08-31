#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, na, WgpuContext};

const THE_SPIKE_GATES_THE_MSL_ROUND_ON_INTEROP_NOT_MATH: &str =
    "Three questions decide whether the fused-attention escape hatch is buildable: does this \
     adapter grant PASSTHROUGH_SHADERS, does a runtime-compiled MSL kernel bind the same wgpu \
     storage buffers the tower already owns and round a value through them bit-exactly, and can \
     an MSL dispatch share one submitted pass list with a WGSL dispatch -- because if it can, \
     the native-queue design and its per-layer CPU syncs are unnecessary and the sync price is \
     zero by construction. The fragment kernel additionally proves the one operation naga \
     cannot express: per-element access to a cooperative matmul accumulator (exp2 applied \
     between the QK product and the store), the op that killed three flash forms and the \
     vision fusion.";

const SPIKE_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

struct SpikeParams {
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void dv_msl_spike_axpb(
    device float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant SpikeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        y[gid] = fma(x[gid], 2.0f, 1.0f);
    }
}

constexpr constant int SM = 16;
constexpr constant int SN = 32;
constexpr constant int SK = 32;

kernel void dv_msl_spike_frag_exp2(
    device half* a [[buffer(0)]],
    device half* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant SpikeParams& p [[buffer(3)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup half at[SM * SK];
    threadgroup half bt[SN * SK];
    for (uint i = lid; i < uint(SM * SK); i += 128u) {
        at[i] = a[i];
    }
    for (uint i = lid; i < uint(SN * SK); i += 128u) {
        bt[i] = b[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid != 0u) {
        return;
    }
    auto tA = tensor(at, dextents<int32_t, 2>(SK, SM), array<int, 2>{1, SK});
    auto tB = tensor(bt, dextents<int32_t, 2>(SK, SN), array<int, 2>{1, SK});
    constexpr auto d = matmul2d_descriptor(SM, SN, SK, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroup> op;
    auto sA = tA.slice(0, 0);
    auto sB = tB.slice(0, 0);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();
    op.run(sA, sB, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto idx = acc.get_multidimensional_index(i);
            y[uint(idx[1]) * uint(SN) + uint(idx[0])] = exp2(acc[i]);
        }
    }
}
"#;

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpikeParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const WGSL_NEG: &str = "
@group(0) @binding(0) var<storage, read> wx: array<f32>;
@group(0) @binding(1) var<storage, read_write> wy: array<f32>;

@compute @workgroup_size(64)
fn dv_wgsl_neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < arrayLength(&wx)) {
        wy[gid.x] = -wx[gid.x];
    }
}
";

#[test]
#[ignore]
fn the_msl_interop_spike_round_trips_bit_exactly_and_shares_a_pass_list_with_wgsl() {
    let _ = THE_SPIKE_GATES_THE_MSL_ROUND_ON_INTEROP_NOT_MATH;
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dv-msl-spike] adapter: {}", ctx.summary());
    assert!(
        na::supported(ctx),
        "PASSTHROUGH_SHADERS is not granted on this adapter, so the MSL escape hatch is closed \
         at the door -- that is the round's finding"
    );
    if let Some((before, after)) = na::msl4_probe() {
        eprintln!("[dv-msl-spike] msl language version hook: {before:#x} -> {after:#x}");
    }

    let module = na::msl_module(
        ctx,
        "dv-msl-spike",
        &[
            ("dv_msl_spike_axpb", (64, 1, 1)),
            ("dv_msl_spike_frag_exp2", (128, 1, 1)),
        ],
        SPIKE_MSL,
    )
    .expect("passthrough MSL compile");

    let mk_layout = |label: &str, storages: &[(u32, bool)], uniforms: &[u32]| {
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = storages
            .iter()
            .map(|&(b, ro)| storage_entry(b, ro))
            .collect();
        entries.extend(uniforms.iter().map(|&b| uniform_entry(b)));
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        (
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                }),
            bgl,
        )
    };

    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 500.0).collect();
    let xb = dispatch::storage_from_slice(ctx, "dv-spike-x", &x);
    let yb = dispatch::storage_zeroed(ctx, "dv-spike-y", (n * 4) as u64);
    let wyb = dispatch::storage_zeroed(ctx, "dv-spike-wy", (n * 4) as u64);
    let p = SpikeParams { n: n as u32, pad0: 0, pad1: 0, pad2: 0 };
    let pb = dispatch::uniform_from(ctx, "dv-spike-p", &p);

    let (axpb_layout, axpb_bgl) = mk_layout("dv-spike-axpb", &[(0, true), (1, false)], &[2]);
    let axpb = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dv-spike-axpb"),
            layout: Some(&axpb_layout),
            module: &module,
            entry_point: Some("dv_msl_spike_axpb"),
            compilation_options: Default::default(),
            cache: None,
        });
    let axpb_bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dv-spike-axpb"),
        layout: &axpb_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: yb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
        ],
    });

    let wgsl_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dv-spike-wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL_NEG.into()),
        });
    let (neg_layout, neg_bgl) = mk_layout("dv-spike-neg", &[(0, true), (1, false)], &[]);
    let neg = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dv-spike-neg"),
            layout: Some(&neg_layout),
            module: &wgsl_module,
            entry_point: Some("dv_wgsl_neg"),
            compilation_options: Default::default(),
            cache: None,
        });
    let neg_bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dv-spike-neg"),
        layout: &neg_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: yb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wyb.as_entire_binding() },
        ],
    });

    let groups = (n as u32).div_ceil(64);
    dispatch::submit_pass_list(
        ctx,
        vec![
            (&axpb, &axpb_bind, (groups, 1, 1)),
            (&neg, &neg_bind, (groups, 1, 1)),
        ],
    );
    let y: Vec<f32> = dispatch::read_back(ctx, &yb, n).expect("msl readback");
    let wy: Vec<f32> = dispatch::read_back(ctx, &wyb, n).expect("wgsl readback");
    let mut bad = 0usize;
    for i in 0..n {
        let want = x[i].mul_add(2.0, 1.0);
        bad += usize::from(y[i].to_bits() != want.to_bits());
        bad += usize::from(wy[i].to_bits() != (-want).to_bits());
    }
    assert_eq!(
        bad, 0,
        "the MSL dispatch and the WGSL dispatch that consumed its output in the same submitted \
         pass list must both be bit-exact"
    );
    eprintln!(
        "[dv-msl-spike] axpb bit-exact over {n} f32 through a shared wgpu buffer, and the WGSL \
         consumer in the same pass list saw the MSL writes with no extra sync"
    );

    let (sm, sn, sk) = (16usize, 32usize, 32usize);
    let a: Vec<half::f16> = (0..sm * sk)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.125))
        .collect();
    let b: Vec<half::f16> = (0..sn * sk)
        .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.0625))
        .collect();
    let ab = dispatch::storage_from_slice(ctx, "dv-spike-a", &a);
    let bb = dispatch::storage_from_slice(ctx, "dv-spike-b", &b);
    let fy = dispatch::storage_zeroed(ctx, "dv-spike-fy", (sm * sn * 4) as u64);
    let (frag_layout, frag_bgl) =
        mk_layout("dv-spike-frag", &[(0, true), (1, true), (2, false)], &[3]);
    let frag = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dv-spike-frag"),
            layout: Some(&frag_layout),
            module: &module,
            entry_point: Some("dv_msl_spike_frag_exp2"),
            compilation_options: Default::default(),
            cache: None,
        });
    let frag_bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dv-spike-frag"),
        layout: &frag_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ab.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: fy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
        ],
    });
    dispatch::submit_pass_list(ctx, vec![(&frag, &frag_bind, (1, 1, 1))]);
    let got: Vec<f32> = dispatch::read_back(ctx, &fy, sm * sn).expect("frag readback");
    let mut worst = 0f64;
    for m in 0..sm {
        for nn in 0..sn {
            let mut accv = 0f32;
            for k in 0..sk {
                accv += a[m * sk + k].to_f32() * b[nn * sk + k].to_f32();
            }
            let want = accv.exp2();
            let g = got[m * sn + nn];
            let rel = ((g - want).abs() as f64) / (want.abs().max(1e-6) as f64);
            worst = worst.max(rel);
        }
    }
    eprintln!(
        "[dv-msl-spike] per-element exp2 on a cooperative matmul accumulator: worst rel \
         {worst:.3e} over {} cells -- the operation naga cannot express, compiled and correct",
        sm * sn
    );
    assert!(
        worst < 1e-3,
        "the fragment kernel's exp2(QK) disagrees with the host oracle by {worst:.3e}"
    );
}
