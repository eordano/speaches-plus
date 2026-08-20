
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nvk_grid.cuh"
#include "nvk_smem_optin.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerBlock;
constexpr int kMaxSharedK = 4096;

__device__ __forceinline__ void moe_gemv_swiglu_body(
    float* xs,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ h,
    int num_experts,
    int N,
    int K
) {
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[k] = __bfloat162float(x[k]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    int slot = blockIdx.y;
    int e = ids[slot];
    if (e < 0 || e >= num_experts) {
        if (lane == 0) h[(size_t)slot * N + n] = __float2bfloat16(0.0f);
        return;
    }

    const uint4* g4 = reinterpret_cast<const uint4*>(gate + ((size_t)e * N + n) * K);
    const uint4* u4 = reinterpret_cast<const uint4*>(up + ((size_t)e * N + n) * K);
    int Kv = K / 8;

    float accg = 0.0f;
    float accu = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pg = __ldg(&g4[v]);
        uint4 pu = __ldg(&u4[v]);
        const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&pg);
        const __nv_bfloat162* upp = reinterpret_cast<const __nv_bfloat162*>(&pu);
        int kb = v * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 gf = __bfloat1622float2(gp[j]);
            float2 uf = __bfloat1622float2(upp[j]);
            float xa = xs[kb + 2 * j];
            float xb = xs[kb + 2 * j + 1];
            accg += gf.x * xa + gf.y * xb;
            accu += uf.x * xa + uf.y * xb;
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        accg += __shfl_xor_sync(0xffffffff, accg, offset);
        accu += __shfl_xor_sync(0xffffffff, accu, offset);
    }
    if (lane == 0) {
        __nv_bfloat16 g = __float2bfloat16(accg);
        __nv_bfloat16 u = __float2bfloat16(accu);
        __nv_bfloat16 s = g / (static_cast<__nv_bfloat16>(1) + hexp(-g));
        h[(size_t)slot * N + n] = s * u;
    }
}

__global__ void moe_gemv_swiglu_m1_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ h,
    int num_experts,
    int N,
    int K
) {
    __shared__ float xs[kMaxSharedK];
    moe_gemv_swiglu_body(xs, gate, up, ids, x, h, num_experts, N, K);
}

__global__ void moe_gemv_swiglu_mb_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ h,
    int num_experts,
    int N,
    int K,
    int k
) {
    __shared__ float xs[kMaxSharedK];
    int b = blockIdx.z;
    moe_gemv_swiglu_body(
        xs,
        gate,
        up,
        ids + (size_t)b * k,
        x + (size_t)b * K,
        h + (size_t)b * k * N,
        num_experts, N, K);
}

__device__ __forceinline__ void moe_gemv_down_tail_body(
    float* hs,
    const __nv_bfloat16* __restrict__ down,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ shared_f32,
    const __nv_bfloat16* __restrict__ resid,
    __nv_bfloat16* __restrict__ out,
    int k,
    int num_experts,
    int N,
    int K
) {
    for (int i = threadIdx.x; i < k * K; i += kBlockDim) {
        hs[i] = __bfloat162float(h[i]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    int Kv = K / 8;
    float acc = 0.0f;
    for (int slot = 0; slot < k; ++slot) {
        int e = ids[slot];
        if (e < 0 || e >= num_experts) continue;
        const uint4* w4 = reinterpret_cast<const uint4*>(down + ((size_t)e * N + n) * K);
        const float* hp = hs + slot * K;
        float a = 0.0f;
        for (int v = lane; v < Kv; v += kWarpSize) {
            uint4 pw = __ldg(&w4[v]);
            const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
            int kb = v * 8;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __bfloat1622float2(wp[j]);
                a += wf.x * hp[kb + 2 * j] + wf.y * hp[kb + 2 * j + 1];
            }
        }
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        acc += weights[slot] * __bfloat162float(__float2bfloat16(a));
    }

    if (lane == 0) {
        float t = acc + shared_f32[n];
        __nv_bfloat16 fb = __float2bfloat16(t);
        float rv = __bfloat162float(resid[n]);
        out[n] = __float2bfloat16(rv + __bfloat162float(fb));
    }
}

__global__ void moe_gemv_down_tail_m1_kernel(
    const __nv_bfloat16* __restrict__ down,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ shared_f32,
    const __nv_bfloat16* __restrict__ resid,
    __nv_bfloat16* __restrict__ out,
    int k,
    int num_experts,
    int N,
    int K
) {
    extern __shared__ float hs[];
    moe_gemv_down_tail_body(
        hs, down, ids, weights, h, shared_f32, resid, out, k, num_experts, N, K);
}

__global__ void moe_gemv_down_tail_mb_kernel(
    const __nv_bfloat16* __restrict__ down,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ shared_f32,
    const __nv_bfloat16* __restrict__ resid,
    __nv_bfloat16* __restrict__ out,
    int k,
    int num_experts,
    int N,
    int K
) {
    extern __shared__ float hs[];
    int b = blockIdx.z;
    moe_gemv_down_tail_body(
        hs,
        down,
        ids + (size_t)b * k,
        weights + (size_t)b * k,
        h + (size_t)b * k * K,
        shared_f32 + (size_t)b * N,
        resid + (size_t)b * N,
        out + (size_t)b * N,
        k, num_experts, N, K);
}

static int moe_smem_limit() { return nvk_max_dynamic_smem_optin(); }

}

extern "C" int nv_kernels_moe_gemv_swiglu_bf16_m1(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    const int32_t* ids,
    const uint16_t* x,
    uint16_t* h,
    int k,
    int num_experts,
    int inter,
    int hidden
) {
    if (k <= 0 || num_experts <= 0 || inter <= 0 || hidden <= 0) return -1;
    if ((hidden & 7) != 0 || hidden > kMaxSharedK) return -1;
    if (k > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((inter + kRowsPerBlock - 1) / kRowsPerBlock), (unsigned)k);
    moe_gemv_swiglu_m1_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        ids,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(h),
        num_experts, inter, hidden);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_gemv_swiglu_bf16_mb(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    const int32_t* ids,
    const uint16_t* x,
    uint16_t* h,
    int b,
    int k,
    int num_experts,
    int inter,
    int hidden
) {
    if (b <= 0 || k <= 0 || num_experts <= 0 || inter <= 0 || hidden <= 0) return -1;
    if ((hidden & 7) != 0 || hidden > kMaxSharedK) return -1;
    if (k > 65535 || b > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((inter + kRowsPerBlock - 1) / kRowsPerBlock),
              (unsigned)k,
              (unsigned)b);
    moe_gemv_swiglu_mb_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        ids,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(h),
        num_experts, inter, hidden, k);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_gemv_down_tail_bf16_m1(
    void* stream,
    const uint16_t* down,
    const int32_t* ids,
    const float* weights,
    const uint16_t* h,
    const float* shared_f32,
    const uint16_t* resid,
    uint16_t* out,
    int k,
    int num_experts,
    int hidden,
    int inter
) {
    if (k <= 0 || num_experts <= 0 || inter <= 0 || hidden <= 0) return -1;
    if ((inter & 7) != 0) return -1;
    size_t smem = (size_t)k * inter * sizeof(float);
    if (smem > 48 * 1024) {
        size_t limit = (size_t)moe_smem_limit();
        if (limit == 0 || smem > limit) return -1;
        static DynamicSmemOptin optin;
        int orc = raise_dynamic_smem_optin_never_lowering_it(
            optin, (const void*)moe_gemv_down_tail_m1_kernel, smem);
        if (orc != 0) return orc;
    }
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((hidden + kRowsPerBlock - 1) / kRowsPerBlock));
    moe_gemv_down_tail_m1_kernel<<<grid, dim3(kBlockDim), smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(down),
        ids,
        weights,
        reinterpret_cast<const __nv_bfloat16*>(h),
        shared_f32,
        reinterpret_cast<const __nv_bfloat16*>(resid),
        reinterpret_cast<__nv_bfloat16*>(out),
        k, num_experts, hidden, inter);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_gemv_down_tail_bf16_mb(
    void* stream,
    const uint16_t* down,
    const int32_t* ids,
    const float* weights,
    const uint16_t* h,
    const float* shared_f32,
    const uint16_t* resid,
    uint16_t* out,
    int b,
    int k,
    int num_experts,
    int hidden,
    int inter
) {
    if (b <= 0 || k <= 0 || num_experts <= 0 || inter <= 0 || hidden <= 0) return -1;
    if ((inter & 7) != 0) return -1;
    if (b > 65535) return NVK_ERR_GRID_AXIS;
    size_t smem = (size_t)k * inter * sizeof(float);
    if (smem > 48 * 1024) {
        size_t limit = (size_t)moe_smem_limit();
        if (limit == 0 || smem > limit) return -1;
        static DynamicSmemOptin optin;
        int orc = raise_dynamic_smem_optin_never_lowering_it(
            optin, (const void*)moe_gemv_down_tail_mb_kernel, smem);
        if (orc != 0) return orc;
    }
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((hidden + kRowsPerBlock - 1) / kRowsPerBlock), 1u, (unsigned)b);
    moe_gemv_down_tail_mb_kernel<<<grid, dim3(kBlockDim), smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(down),
        ids,
        weights,
        reinterpret_cast<const __nv_bfloat16*>(h),
        shared_f32,
        reinterpret_cast<const __nv_bfloat16*>(resid),
        reinterpret_cast<__nv_bfloat16*>(out),
        k, num_experts, hidden, inter);
    return (int)cudaGetLastError();
}
