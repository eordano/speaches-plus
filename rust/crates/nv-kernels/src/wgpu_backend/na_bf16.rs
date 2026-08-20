use std::sync::{Arc, Mutex, OnceLock};

use super::device::WgpuContext;
use super::dispatch::profile;
use super::na;
use super::na::{storage_entry, uniform_entry};
use super::{Result, WgpuError};

pub const TILE_N: u32 = 64;
pub const TILE_M: u32 = 16;
pub const K_CHUNK: u32 = 32;
pub const WG_THREADS: u32 = 128;

pub const ENTRY: &str = "na_gemm_bf16";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NaBf16Params {
    pub n_rows: u32,
    pub k_elems: u32,
    pub x_stride_words: u32,
    pub y_stride_words: u32,
    pub dst_word_off: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NaLiveParams {
    pub m_live: u32,
    pub base: u32,
    pub pad0: u32,
    pub pad1: u32,
}

pub const MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

struct NaBf16Params {
    uint n_rows;
    uint k_elems;
    uint x_stride_words;
    uint y_stride_words;
    uint dst_word_off;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct NaLiveParams {
    uint m_live;
    uint base;
    uint pad0;
    uint pad1;
};

constexpr constant int TN = 64;
constexpr constant int TM = 16;
constexpr constant int KC = 32;
constexpr constant int NSG = 4;

static inline uint nb_bf16_encode(float x) {
    uint b = as_type<uint>(x);
    uint r = 0x7fffu + ((b >> 16u) & 1u);
    return (x != x) ? 0x7fc0u : ((b + r) >> 16u);
}

kernel void na_gemm_bf16(
    device bfloat* wb [[buffer(0)]],
    device bfloat* xb [[buffer(1)]],
    device uint* y [[buffer(2)]],
    constant NaBf16Params& np [[buffer(3)]],
    constant NaLiveParams& lp [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    uint mu = min(lp.m_live, uint(TM));
    if (mu == 0u) {
        return;
    }
    threadgroup float ftile[TM * TN];
    int K = int(np.k_elems);
    int NR = int(np.n_rows);
    int m = int(mu);
    int n0 = int(tgid.x) * TN;

    auto tA = tensor(xb,
                     dextents<int32_t, 2>(K, m),
                     array<int, 2>{1, int(np.x_stride_words) * 2});
    auto tW = tensor(wb,
                     dextents<int32_t, 2>(K, NR),
                     array<int, 2>{1, K});

    constexpr auto desc = matmul2d_descriptor(TM, TN, KC, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    auto sA0 = tA.slice(0, 0);
    auto sW0 = tW.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    for (int k0 = int(sgid) * KC; k0 < K; k0 += NSG * KC) {
        auto sA = tA.slice(k0, 0);
        auto sW = tW.slice(k0, n0);
        op.run(sA, sW, acc);
    }

    for (uint i = lid; i < uint(TM * TN); i += 128u) {
        ftile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < uint(NSG); s++) {
        if (sgid == s) {
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    auto idx = acc.get_multidimensional_index(i);
                    ftile[uint(idx[1]) * TN + uint(idx[0])] += acc[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint words = uint(TN / 2) * mu;
    for (uint w = lid; w < words; w += 128u) {
        uint t = w / uint(TN / 2);
        uint rp = w % uint(TN / 2);
        int row = n0 + int(2 * rp);
        if (row >= NR) {
            continue;
        }
        uint lo = nb_bf16_encode(ftile[t * TN + 2 * rp]) & 0xffffu;
        uint hi = (row + 1 < NR) ? nb_bf16_encode(ftile[t * TN + 2 * rp + 1]) : 0u;
        y[np.dst_word_off + t * np.y_stride_words + uint(row >> 1)] = lo | (hi << 16u);
    }
}
"#;

type CacheEntry = (
    wgpu::Device,
    std::result::Result<Arc<wgpu::ComputePipeline>, WgpuError>,
);

fn cache() -> &'static Mutex<Vec<CacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn build_pipeline(ctx: &WgpuContext) -> Result<Arc<wgpu::ComputePipeline>> {
    let module = na::msl_module(ctx, "nv-na-gemm-bf16", &[(ENTRY, (WG_THREADS, 1, 1))], MSL)?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, false),
        uniform_entry(3),
        uniform_entry(4),
    ];
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nv-na-bf16"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv-na-bf16"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = Arc::new(
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nv-na-gemm-bf16"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(ENTRY),
                compilation_options: Default::default(),
                cache: None,
            }),
    );
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!(
            "na bf16 passthrough: {err}"
        )));
    }
    profile::name_pipeline(&pipeline, "na-gemm-bf16:na_gemm_bf16");
    Ok(pipeline)
}

pub fn pipeline(ctx: &WgpuContext) -> Result<Arc<wgpu::ComputePipeline>> {
    let mut guard = cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone();
        }
    }
    let built = if na::supported(ctx) {
        build_pipeline(ctx)
    } else {
        Err(WgpuError::Unsupported(
            "na tensor-ops need the Metal backend with PASSTHROUGH_SHADERS".into(),
        ))
    };
    guard.push((ctx.device.clone(), built.clone()));
    built
}

pub fn available(ctx: &WgpuContext) -> bool {
    pipeline(ctx).is_ok()
}

pub fn shape_ok(n: usize, k: usize, m: usize) -> bool {
    n >= 2 && k >= 32 && k.is_multiple_of(32) && m >= 1 && m <= TILE_M as usize
}

pub fn grid_x(n_rows: u32) -> u32 {
    n_rows.div_ceil(TILE_N)
}

pub fn pipeline_label(p: &wgpu::ComputePipeline) -> Option<&'static str> {
    let guard = cache().lock().ok()?;
    for (_, entry) in guard.iter() {
        if let Ok(pl) = entry {
            if pl.as_ref() == p {
                return Some("na_gemm_bf16");
            }
        }
    }
    None
}
