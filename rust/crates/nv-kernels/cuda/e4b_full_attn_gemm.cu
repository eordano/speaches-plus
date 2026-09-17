#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kSoftmaxThreads256 = 256;
constexpr int kKeyTileMax4096OneScoreTileStaysInRegistersAt16PerThread = 4096;
constexpr int kKeyTileMax8192TheDoubledTileHoldsThirtyTwoScoresPerThread = 8192;

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    v = (threadIdx.x < (kSoftmaxThreads256 / 32)) ? red[threadIdx.x] : -INFINITY;
    if (w == 0) v = warp_max(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    v = (threadIdx.x < (kSoftmaxThreads256 / 32)) ? red[threadIdx.x] : 0.f;
    if (w == 0) v = warp_sum(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

template <int kValuesPerThread>
__global__ void __launch_bounds__(kSoftmaxThreads256) e4b_online_softmax_tile_kernel(
    const float* __restrict__ s,
    __nv_bfloat16* __restrict__ p,
    float* __restrict__ m_run,
    float* __restrict__ l_run,
    float* __restrict__ o_acc,
    int rows_per_group,
    int n,
    int tile_len,
    int key_off,
    int q_pos0,
    int hd,
    float scale,
    int first_tile) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int row = g * rows_per_group + r;
    const int t = r % n;
    const int allowed = q_pos0 + t - key_off + 1;
    const float* srow = s + (size_t)row * tile_len;
    __nv_bfloat16* prow = p + (size_t)row * tile_len;
    float vals[kValuesPerThread];
    float local_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const int j = threadIdx.x + i * kSoftmaxThreads256;
        float x = -INFINITY;
        if (j < tile_len && j < allowed) x = srow[j] * scale;
        vals[i] = x;
        local_max = fmaxf(local_max, x);
    }
    const float tile_max = block_max(local_max, red);
    const float m_old = first_tile ? -INFINITY : m_run[row];
    const float m_new = fmaxf(m_old, tile_max);
    const bool row_live = m_new > -INFINITY;
    float local_sum = 0.f;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const int j = threadIdx.x + i * kSoftmaxThreads256;
        if (j < tile_len) {
            const float e = (row_live && vals[i] > -INFINITY) ? __expf(vals[i] - m_new) : 0.f;
            prow[j] = __float2bfloat16(e);
            local_sum += e;
        }
    }
    const float tile_sum = block_sum(local_sum, red);
    const float alpha = (row_live && m_old > -INFINITY) ? __expf(m_old - m_new) : 0.f;
    const float l_old = first_tile ? 0.f : l_run[row];
    if (threadIdx.x == 0) {
        m_run[row] = m_new;
        l_run[row] = l_old * alpha + tile_sum;
    }
    if (!first_tile) {
        float* orow = o_acc + (size_t)row * hd;
        for (int d = threadIdx.x; d < hd; d += kSoftmaxThreads256) orow[d] *= alpha;
    }
}

__global__ void e4b_attn_finalize_kernel(
    const float* __restrict__ o_acc,
    const float* __restrict__ l_run,
    __nv_bfloat16* __restrict__ out,
    int n,
    int heads,
    int hd) {
    const int row = blockIdx.x;
    const int h = row / n;
    const int t = row % n;
    const float inv = 1.f / l_run[row];
    const float* orow = o_acc + (size_t)row * hd;
    __nv_bfloat16* dst = out + ((size_t)t * heads + h) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = __float2bfloat16(orow[d] * inv);
}

}

extern "C" int nv_kernels_e4b_online_softmax_tile(
    void* stream,
    const float* s,
    uint16_t* p,
    float* m_run,
    float* l_run,
    float* o_acc,
    int rows_per_group,
    int groups,
    int n,
    int tile_len,
    int key_off,
    int q_pos0,
    int hd,
    float scale,
    int first_tile) {
    if (tile_len <= 0 || tile_len > kKeyTileMax8192TheDoubledTileHoldsThirtyTwoScoresPerThread) return -1;
    if (rows_per_group <= 0 || groups <= 0 || n <= 0 || hd <= 0) return -2;
    dim3 grid(rows_per_group, groups);
    if (tile_len <= kKeyTileMax4096OneScoreTileStaysInRegistersAt16PerThread) {
        e4b_online_softmax_tile_kernel<kKeyTileMax4096OneScoreTileStaysInRegistersAt16PerThread / kSoftmaxThreads256>
            <<<grid, kSoftmaxThreads256, 0, (cudaStream_t)stream>>>(
                s, reinterpret_cast<__nv_bfloat16*>(p), m_run, l_run, o_acc, rows_per_group, n, tile_len,
                key_off, q_pos0, hd, scale, first_tile);
    } else {
        e4b_online_softmax_tile_kernel<kKeyTileMax8192TheDoubledTileHoldsThirtyTwoScoresPerThread / kSoftmaxThreads256>
            <<<grid, kSoftmaxThreads256, 0, (cudaStream_t)stream>>>(
                s, reinterpret_cast<__nv_bfloat16*>(p), m_run, l_run, o_acc, rows_per_group, n, tile_len,
                key_off, q_pos0, hd, scale, first_tile);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_e4b_attn_finalize(
    void* stream,
    const float* o_acc,
    const float* l_run,
    uint16_t* out,
    int n,
    int heads,
    int hd) {
    if (n <= 0 || heads <= 0 || hd <= 0) return -1;
    e4b_attn_finalize_kernel<<<n * heads, 256, 0, (cudaStream_t)stream>>>(
        o_acc, l_run, reinterpret_cast<__nv_bfloat16*>(out), n, heads, hd);
    return (int)cudaGetLastError();
}
