use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use nv_kernels::wgpu_backend::{na, WgpuContext};

pub const MSL_GEMV_ENTRY: &str = "dg_msl_w4_gemv";
pub const MSL_GEMV_RMS_ENTRY: &str = "dg_msl_w4_gemv_rms";
pub const MSL_GEMV_GSILU_ENTRY: &str = "dg_msl_w4_gemv_gsilu";
pub const MSL_GEMV_BATCH_ENTRY: &str = "dg_msl_w4_gemv_b";
pub const MSL_GEMV_ROWS: u32 = 8;

pub const BHEAD_WINS_ONLY_THE_DISPATCH_CRUMBS_BECAUSE_THE_W4_HEAD_IS_ISSUE_BOUND_NOT_READ_BOUND:
    &str = "dg_msl_w4_gemv_b serves the batched-region lm head as one dispatch looping the batch \
     rows around the shipped dg_msl_w4_gemv dot text -- the i-loop, its simd_sum and the \
     out_f32 identity store unchanged, only the per-row x base, the y row offset and a \
     second rows8 barrier outside the dot (the gsilu class of recompile, not the rms round's \
     branched-dot class; gated to 0/151936 per row against the shipped replay on real \
     weights, tolerance returns for arbitration). What batching CANNOT buy here: the head at \
     m=32 is issue-bound on the mandated scalar chain, not weight-read-bound -- the ~513 \
     GB/s apparent of one replay is the byte rate that falls out of ~2.6 T ops/s of \
     load+convert+extract+fma issue, not a DRAM wall -- so deleting the 31 redundant 116 MB \
     head reads moves nothing. Four reuse geometries measured on real weights, isolated \
     head at 32 rows vs the replay's 8.2 ms: 8-row register blocks around each weight vec4 \
     8.2-8.4 (ALU-invariant), a 24 KB threadgroup x slab 10.8 (occupancy collapse, the \
     W4_GEMV_BOUND slab verdict again), hoisted nibble extracts with u32 pair loads and 32 \
     live accumulators 19.4 (register spill); every geometry held the per-row bits. The \
     shipped row loop wins what is winnable -- 160 dispatches (32 x gather/gemv/argmax \
     pair/stash) become 4, deleting ~0.9 ms of pass-boundary crumbs per bstep, bitwise. A \
     cheaper head needs an arithmetic-class change (a w4-read steel GEMM), which is \
     tolerance-class and returns for arbitration if ever sized.";

pub const GSILU_FUSE_KEEPS_EVERY_REDUCTION_IN_ITS_KERNEL_AND_REPLAYS_THE_SILU_PASS_VERBATIM: &str =
    "dg_msl_w4_gemv_gsilu deletes the standalone dg_silu_mul pass -- one serial-drain pass \
     boundary per layer at m=1 -- by pairing each gate row with its up row in one workgroup \
     (simdgroups 0-3 take gate rows base..base+3, simdgroups 4-7 the up rows inter+base..) and \
     computing silu(g)*u in the epilogue. The per-row dot is the shipped kernel's text \
     unchanged and its simd_sum never crosses a kernel boundary; the epilogue first rounds \
     each acc through the exact bf16e pair-store the shipped kernel writes, so silu's inputs \
     are the same bf16 words the deleted pass read, then mirrors dg_silu_mul token for token \
     (lo / (1.0 + exp(-lo)) * ulo, contraction-free) and rounds once to f16 where the WGSL \
     pack2x16float sat -- the vision gsilu epilogue and the rms round's pack-equivalence \
     census are the precedents. Row-to-simdgroup remapping changes no row's arithmetic, only \
     which workgroup owns it, so the fusion is bitwise-eligible; the gates hold it to \
     0/151936.";

pub const RMS_FUSE_IS_TOLERANCE_CLASS_BECAUSE_SIMD_SUM_COMPILES_TO_A_PER_KERNEL_ORDER: &str =
    "dg_msl_w4_gemv_rms deletes the standalone dg_rms/dg_rmsres passes by recomputing the row \
     norm in every workgroup and folding the residual add into the producer gemv's epilogue. \
     The NORM side is measured bit-exact against the naga-compiled WGSL: the strided \
     sum-of-squares, eight-slot pairing tree, rsqrt(tot/hidden + eps) scalar (0/4M random \
     inputs), the pack2x16float-equivalent f16 numerator (0/1.33M), the recomputed f16 \
     activations in-context (0/768 words), and the fold's res words and unrounded res_f32 \
     sums (0/768, host-exact). What is NOT reproducible is the w4 dot itself: metal's \
     simd_sum lowers to a reduction order chosen per compiled kernel under fast math, so this \
     kernel's f32 accs sit one ulp off the shipped dg_msl_w4_gemv's on about two thirds of \
     rows (12225/17920 measured at the layer-18 gate-up shape; host fma replay matches this \
     kernel's lanes summed sequentially, while no canonical order reproduces the shipped \
     binary's bits). bf16 output rounding hides all but a few boundary words per layer \
     (4/8960 measured), which cascade to full logit divergence at data-dependent context \
     cells -- so the fusion is TOLERANCE-class, never bitwise, whatever the norm math does. \
     It also LOSES on the wall: the per-workgroup norm recompute prices at +0.7 ms/step on \
     the gate-up class against ~0.4 ms of deleted pass boundaries, and the qkv side washes. \
     rms_mode 1 reads bf16 res directly (dg_rms over the embed row), rms_mode 2 reads \
     res_f32 for the squares; fold_res 1 adds this gemv's rounded output into res in place, \
     safe because each word belongs to exactly one epilogue thread and no fold kernel also \
     norms.";

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

struct DgMslW4RmsP {
    uint n_rows;
    uint w_vec4;
    uint scale_stride;
    uint groups_x;
    uint out_f32;
    uint y_off;
    uint rms_mode;
    uint fold_res;
    float eps;
    uint pad0;
    uint pad1;
    uint pad2;
};

inline float bfl(uint w) { return as_type<float>(w << 16); }
inline float bfh(uint w) { return as_type<float>(w & 0xffff0000u); }

kernel void dg_msl_w4_gemv_rms(
    device const uint4* wq [[buffer(0)]],
    device const uint* sc [[buffer(1)]],
    device const uint* bs [[buffer(2)]],
    device const half* x [[buffer(3)]],
    constant DgMslW4RmsP& p [[buffer(4)]],
    device uint* y [[buffer(5)]],
    device const uint* nw [[buffer(6)]],
    device uint* resw [[buffer(7)]],
    device float* resf [[buffer(8)]],
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

    float rms = 1.0f;
    if (p.rms_mode != 0u) {
        const uint words = p.w_vec4 * 16u;
        float local = 0.0f;
        if (p.rms_mode == 1u) {
            for (uint t = tid; t < words; t += 256u) {
                uint xw = resw[t];
                float lo = bfl(xw);
                float hi = bfh(xw);
                local = (local + (lo * lo)) + (hi * hi);
            }
        } else {
            for (uint t = tid; t < words; t += 256u) {
                float lo = resf[2u * t];
                float hi = resf[2u * t + 1u];
                local = (local + (lo * lo)) + (hi * hi);
            }
        }
        float wsum = simd_sum(local);
        threadgroup float warp8[8];
        if ((tid & 31u) == 0u) { warp8[tid >> 5u] = wsum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = ((warp8[0] + warp8[1]) + (warp8[2] + warp8[3]))
            + ((warp8[4] + warp8[5]) + (warp8[6] + warp8[7]));
        rms = rsqrt((tot / float(p.w_vec4 * 32u)) + p.eps);
    }

    float acc = 0.0f;
    if (p.rms_mode == 0u) {
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
    } else {
        for (uint i = lane; i < kw4; i += 32u) {
            uint4 wv = wrow[i];
            float qdot = 0.0f;
            float xsum = 0.0f;
            for (uint w = 0; w < 4; ++w) {
                uint ww = wv[w];
                uint wb = i * 16u + w * 4u;
                for (uint j = 0; j < 4; ++j) {
                    uint bb = (ww >> (8u * j)) & 0xFFu;
                    uint sw = resw[wb + j];
                    uint gw = nw[wb + j];
                    uint hw = as_type<uint>(half2(float2(
                        (bfl(sw) * rms) * bfl(gw),
                        (bfh(sw) * rms) * bfh(gw))));
                    float lo = float(as_type<half2>(hw).x);
                    float hi = float(as_type<half2>(hw).y);
                    xsum += lo + hi;
                    qdot = fma(float(bb & 0xFu), lo, qdot);
                    qdot = fma(float(bb >> 4u), hi, qdot);
                }
            }
            uint g = i >> 1;
            acc = fma(bf16f(sc[sbase + g]), qdot, fma(bf16f(bs[sbase + g]), xsum, acc));
        }
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
        uint yw = (bf16e(lo) & 0xFFFFu) | (bf16e(hi) << 16);
        y[(row0 >> 1) + tid] = yw;
        if (p.fold_res == 1u) {
            uint widx = (row0 >> 1) + tid;
            uint rw = resw[widx];
            float slo = bfl(yw) + bfl(rw);
            float shi = bfh(yw) + bfh(rw);
            resw[widx] = (bf16e(slo) & 0xFFFFu) | (bf16e(shi) << 16);
            resf[2u * widx] = slo;
            resf[2u * widx + 1u] = shi;
        }
    }
}

kernel void dg_msl_w4_gemv_gsilu(
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
    const uint nh = p.n_rows >> 1;
    const uint base = wg8 * 4u;
    const uint pr = base + (rh & 3u);
    const uint lrow = (rh < 4u) ? pr : (nh + pr);
    const uint row = min(lrow, p.n_rows - 1u);
    const bool live = pr < nh;
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
    if (tid < 2u && base + tid * 2u < nh) {
        float g0 = rows8[tid * 2u];
        float g1 = (base + tid * 2u + 1u < nh) ? rows8[tid * 2u + 1u] : 0.0f;
        float u0 = rows8[4u + tid * 2u];
        float u1 = (base + tid * 2u + 1u < nh) ? rows8[4u + tid * 2u + 1u] : 0.0f;
        uint gw = (bf16e(g0) & 0xFFFFu) | (bf16e(g1) << 16);
        uint uw = (bf16e(u0) & 0xFFFFu) | (bf16e(u1) << 16);
        float glo = bfl(gw);
        float ghi = bfh(gw);
        float ulo = bfl(uw);
        float uhi = bfh(uw);
        y[(base >> 1) + tid] = as_type<uint>(half2(float2(
        (glo / (1.0f + exp(-glo))) * ulo,
        (ghi / (1.0f + exp(-ghi))) * uhi)));
    }
}

struct DgMslW4BP {
    uint n_rows;
    uint w_vec4;
    uint scale_stride;
    uint groups_x;
    uint b_rows;
    uint x_stride;
    uint y_stride;
    uint pad0;
};

kernel void dg_msl_w4_gemv_b(
    device const uint4* wq [[buffer(0)]],
    device const uint* sc [[buffer(1)]],
    device const uint* bs [[buffer(2)]],
    device const half* xb [[buffer(3)]],
    constant DgMslW4BP& p [[buffer(4)]],
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
    threadgroup float rows8[8];

    for (uint r = 0; r < p.b_rows; ++r) {
        device const half* x = xb + r * p.x_stride;
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
        if (lane == 0) rows8[rh] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8u && row0 + tid < p.n_rows) {
            y[r * p.y_stride + row0 + tid] = as_type<uint>(rows8[tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DgMslW4BParams {
    pub n_rows: u32,
    pub w_vec4: u32,
    pub scale_stride: u32,
    pub groups_x: u32,
    pub b_rows: u32,
    pub x_stride: u32,
    pub y_stride: u32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DgMslW4RmsParams {
    pub n_rows: u32,
    pub w_vec4: u32,
    pub scale_stride: u32,
    pub groups_x: u32,
    pub out_f32: u32,
    pub y_off: u32,
    pub rms_mode: u32,
    pub fold_res: u32,
    pub eps: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
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
    pub rms_pipeline: Arc<wgpu::ComputePipeline>,
    pub rms_layout: wgpu::BindGroupLayout,
    pub gsilu_pipeline: Arc<wgpu::ComputePipeline>,
    pub batch_pipeline: Arc<wgpu::ComputePipeline>,
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
    let _ = RMS_FUSE_IS_TOLERANCE_CLASS_BECAUSE_SIMD_SUM_COMPILES_TO_A_PER_KERNEL_ORDER;
    let _ = GSILU_FUSE_KEEPS_EVERY_REDUCTION_IN_ITS_KERNEL_AND_REPLAYS_THE_SILU_PASS_VERBATIM;
    let _ = BHEAD_WINS_ONLY_THE_DISPATCH_CRUMBS_BECAUSE_THE_W4_HEAD_IS_ISSUE_BOUND_NOT_READ_BOUND;
    let module = na::msl_module(
        ctx,
        "dg-msl-w4",
        &[
            (MSL_GEMV_ENTRY, (256, 1, 1)),
            (MSL_GEMV_RMS_ENTRY, (256, 1, 1)),
            (MSL_GEMV_GSILU_ENTRY, (256, 1, 1)),
            (MSL_GEMV_BATCH_ENTRY, (256, 1, 1)),
        ],
        MSL_SRC,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, true),
        storage_entry(3, true),
        uniform_entry(4),
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
    let pipeline = Arc::new(
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(MSL_GEMV_ENTRY),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(MSL_GEMV_ENTRY),
                compilation_options: Default::default(),
                cache: None,
            }),
    );
    let gsilu_pipeline = Arc::new(ctx.device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some(MSL_GEMV_GSILU_ENTRY),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(MSL_GEMV_GSILU_ENTRY),
            compilation_options: Default::default(),
            cache: None,
        },
    ));
    let batch_pipeline = Arc::new(ctx.device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some(MSL_GEMV_BATCH_ENTRY),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(MSL_GEMV_BATCH_ENTRY),
            compilation_options: Default::default(),
            cache: None,
        },
    ));
    let rms_entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, true),
        storage_entry(3, true),
        uniform_entry(4),
        storage_entry(5, false),
        storage_entry(6, true),
        storage_entry(7, false),
        storage_entry(8, false),
    ];
    let rms_bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dg-msl-w4-rms"),
            entries: &rms_entries,
        });
    let rms_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dg-msl-w4-rms"),
            bind_group_layouts: &[Some(&rms_bgl)],
            immediate_size: 0,
        });
    let rms_pipeline = Arc::new(ctx.device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some(MSL_GEMV_RMS_ENTRY),
            layout: Some(&rms_layout),
            module: &module,
            entry_point: Some(MSL_GEMV_RMS_ENTRY),
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
        rms_pipeline,
        rms_layout: rms_bgl,
        gsilu_pipeline,
        batch_pipeline,
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
