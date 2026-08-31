use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use nv_kernels::wgpu_backend::{na, WgpuContext};

pub const MSL_GEMV_ENTRY: &str = "dg_msl_w4_gemv";
pub const MSL_GEMV_ROWS: u32 = 8;

pub const MSL_GEMV_EXISTS_BECAUSE_THE_FIVE_ISSUE_SLOT_BOUND_WAS_NAGAS_NOT_THE_MACHINES: &str =
    "W4_GEMV_BOUND_IS_FIVE_ISSUE_SLOTS_PER_THIRTYTWO_WEIGHTS priced the WGSL kernel, where \
     every activation pair costs an unpack instruction and f16 loads pay convert chains. Raw \
     MSL through the same-queue passthrough reads the activations as device half -- the \
     convert is the load -- and keeps float accumulators exactly as mlx's qmv does, so the \
     inner loop spends its issue slots on the fused multiplies. The geometry is the tuned \
     WGSL kernel's own knee: a weight vec4 is 32 codes, warp per row, eight rows per \
     threadgroup, simd_sum for the row. Scales and biases stay in their widened one-bf16-per\
     -u32 upload so the WGSL arm shares them; the bf16 output rounding mirrors dequant.wgsl's \
     round-to-nearest-even bit for bit.";

const MSL_SRC: &str = "
#include <metal_stdlib>
using namespace metal;

struct DgMslW4P {
    uint n_rows;
    uint w_vec4;
    uint scale_stride;
    uint groups_x;
    uint out_f32;
    uint y_off;
    uint pad0;
    uint pad1;
};

inline float bf16f(uint v) { return as_type<float>(v << 16); }
inline uint bf16e(float x) {
    uint b = as_type<uint>(x);
    uint r = 0x7fffu + ((b >> 16) & 1u);
    return (x != x) ? 0x7fc0u : ((b + r) >> 16);
}

kernel void dg_msl_w4_gemv(
    device const uint4* wq [[buffer(0)]],
    device const uint* sc [[buffer(1)]],
    device const uint* bs [[buffer(2)]],
    device const half* x [[buffer(3)]],
    constant DgMslW4P& p [[buffer(4)]],
    device uint* y [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint rh = tid >> 5;
    const uint lane = tid & 31;
    const uint wg8 = tg.x + tg.y * p.groups_x;
    const uint row0 = wg8 * 8u;
    const uint row = min(row0 + rh, p.n_rows - 1u);
    const bool live = row0 + rh < p.n_rows;
    const uint kw4 = live ? p.w_vec4 : 0u;
    device const uint4* wrow = wq + row * p.w_vec4;
    const uint sbase = row * p.scale_stride;

    float acc = 0.0f;
    for (uint i = lane; i < kw4; i += 32u) {
        uint4 wv = wrow[i];
        device const half* xp = x + i * 32u;
        float qdot = 0.0f;
        float xsum = 0.0f;
        for (uint w = 0; w < 4; ++w) {
            uint ww = wv[w];
            device const half* xw = xp + w * 8u;
            for (uint j = 0; j < 4; ++j) {
                uint bb = (ww >> (8u * j)) & 0xFFu;
                float lo = float(xw[2u * j]);
                float hi = float(xw[2u * j + 1u]);
                xsum += lo + hi;
                qdot = fma(float(bb & 0xFu), lo, qdot);
                qdot = fma(float(bb >> 4u), hi, qdot);
            }
        }
        uint g = i >> 1;
        acc = fma(bf16f(sc[sbase + g]), qdot, fma(bf16f(bs[sbase + g]), xsum, acc));
    }
    acc = simd_sum(acc);
    threadgroup float rows8[8];
    if (lane == 0) rows8[rh] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.out_f32 == 1u) {
        if (tid < 8u && row0 + tid < p.n_rows) {
            y[p.y_off + row0 + tid] = as_type<uint>(rows8[tid]);
        }
    } else if (tid < 4u && row0 + tid * 2u < p.n_rows) {
        float lo = rows8[tid * 2u];
        float hi = (row0 + tid * 2u + 1u < p.n_rows) ? rows8[tid * 2u + 1u] : 0.0f;
        y[(row0 >> 1) + tid] = (bf16e(lo) & 0xFFFFu) | (bf16e(hi) << 16);
    }
}
";

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DgMslW4Params {
    pub n_rows: u32,
    pub w_vec4: u32,
    pub scale_stride: u32,
    pub groups_x: u32,
    pub out_f32: u32,
    pub y_off: u32,
    pub pad0: u32,
    pub pad1: u32,
}

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

pub struct MslGemvPipelines {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub layout: wgpu::BindGroupLayout,
}

type CacheRow = (
    wgpu::Device,
    std::result::Result<Arc<MslGemvPipelines>, String>,
);

fn cache() -> &'static Mutex<Vec<CacheRow>> {
    static CACHE: OnceLock<Mutex<Vec<CacheRow>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn build(ctx: &WgpuContext) -> Result<Arc<MslGemvPipelines>> {
    let _ = MSL_GEMV_EXISTS_BECAUSE_THE_FIVE_ISSUE_SLOT_BOUND_WAS_NAGAS_NOT_THE_MACHINES;
    let module = na::msl_module(ctx, "dg-msl-w4", &[(MSL_GEMV_ENTRY, (256, 1, 1))], MSL_SRC)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, true),
        storage_entry(3, true),
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage_entry(5, false),
    ];
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dg-msl-w4"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dg-msl-w4"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = Arc::new(ctx.device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some(MSL_GEMV_ENTRY),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(MSL_GEMV_ENTRY),
            compilation_options: Default::default(),
            cache: None,
        },
    ));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("dg-msl-w4 pipeline: {err}");
    }
    Ok(Arc::new(MslGemvPipelines {
        pipeline,
        layout: bgl,
    }))
}

pub fn msl_gemv_pipelines(ctx: &'static WgpuContext) -> Result<Arc<MslGemvPipelines>> {
    let mut guard = cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = build(ctx);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push((ctx.device.clone(), stored));
    built
}
