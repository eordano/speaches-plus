
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kAppendThreads = 128;

__device__ __forceinline__ void copy_row_bf16(
    const __nv_bfloat16* __restrict__ s,
    __nv_bfloat16* __restrict__ d,
    int row_elems
) {
    if ((row_elems & 7) == 0) {
        const uint4* s4 = reinterpret_cast<const uint4*>(s);
        uint4* d4 = reinterpret_cast<uint4*>(d);
        const int n4 = row_elems >> 3;
        for (int j = threadIdx.x; j < n4; j += blockDim.x) d4[j] = s4[j];
    } else {
        for (int j = threadIdx.x; j < row_elems; j += blockDim.x) d[j] = s[j];
    }
}

__global__ void kv_ring_append_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    const int* __restrict__ pos_dev,
    int cap,
    int row_elems
) {
    const int i = blockIdx.x;
    int slot = (*pos_dev + i) % cap;
    copy_row_bf16(src + (size_t)i * row_elems, dst + (size_t)slot * row_elems, row_elems);
}

__global__ void kv_shift_bf16_kernel(
    __nv_bfloat16* __restrict__ buf,
    int src_row,
    int dst_row,
    int row_elems
) {
    const int i = blockIdx.x;
    copy_row_bf16(
        buf + (size_t)(src_row + i) * row_elems,
        buf + (size_t)(dst_row + i) * row_elems,
        row_elems
    );
}

constexpr int kWarp = 32;
constexpr int kRingWarps = 8;
constexpr int kRingThreads = kWarp * kRingWarps;
constexpr int kMaxHD = 512;
constexpr int kMaxAccPerLane = kMaxHD / kWarp;

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__global__ void attention_bf16_decode_ring_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ ring_meta,
    int cap,
    int window,
    int NH,
    int NKV,
    int HD,
    float scaling
) {
    const int h = blockIdx.x;
    if (h >= NH) return;

    const int ring_start = ring_meta[0];
    const int stored = ring_meta[1];
    const int L = (window > 0 && stored > window) ? window : stored;
    const int i0 = stored - L;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kRingThreads)
        qsh[d] = __bfloat162float(q[(size_t)h * HD + d]) * scaling;
    __syncthreads();

    float acc[kMaxAccPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int i = i0 + warp; i < stored; i += kRingWarps) {
        int slot = ring_start + i;
        if (slot >= cap) slot -= cap;
        const __nv_bfloat16* kp = k + ((size_t)slot * NKV + kvh) * HD;

        float partial = 0.0f;
        for (int d = lane; d < HD; d += kWarp)
            partial += qsh[d] * __bfloat162float(kp[d]);
        float score = warp_sum(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const __nv_bfloat16* vp = v + ((size_t)slot * NKV + kvh) * HD;
        #pragma unroll
        for (int j = 0; j < kMaxAccPerLane; ++j) {
            int d = lane + j * kWarp;
            if (d < HD) acc[j] = acc[j] * corr + w * __bfloat162float(vp[d]);
        }
        m = m_new;
    }

    __shared__ float sm[kRingWarps];
    __shared__ float sl[kRingWarps];
    __shared__ float sacc[kRingWarps][kMaxHD];

    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    #pragma unroll
    for (int j = 0; j < kMaxAccPerLane; ++j) {
        int d = lane + j * kWarp;
        if (d < HD) sacc[warp][d] = acc[j];
    }
    __syncthreads();

    if (warp == 0) {
        float m_glob = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kRingWarps; ++w) m_glob = fmaxf(m_glob, sm[w]);

        float l_glob = 0.0f;
        #pragma unroll
        for (int w = 0; w < kRingWarps; ++w)
            l_glob += (sm[w] == -INFINITY) ? 0.0f : sl[w] * __expf(sm[w] - m_glob);
        float inv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;

        float scale_w[kRingWarps];
        #pragma unroll
        for (int w = 0; w < kRingWarps; ++w)
            scale_w[w] = (sm[w] == -INFINITY) ? 0.0f : __expf(sm[w] - m_glob);

        for (int d = lane; d < HD; d += kWarp) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kRingWarps; ++w) a += sacc[w][d] * scale_w[w];
            out[(size_t)h * HD + d] = __float2bfloat16(a * inv_l);
        }
    }
}

}

extern "C" int nv_kernels_kv_ring_append_bf16(
    void* stream,
    const uint16_t* src,
    uint16_t* dst,
    const int* pos_dev,
    int t,
    int cap,
    int n_kv,
    int head_dim
) {
    if (t <= 0) return 0;
    if (cap <= 0 || n_kv <= 0 || head_dim <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    kv_ring_append_bf16_kernel<<<(unsigned)t, kAppendThreads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(src),
        reinterpret_cast<__nv_bfloat16*>(dst),
        pos_dev, cap, n_kv * head_dim
    );
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_kv_shift_bf16(
    void* stream,
    uint16_t* buf,
    int src_row,
    int dst_row,
    int rows,
    int n_kv,
    int head_dim
) {
    if (rows <= 0) return 0;
    if (n_kv <= 0 || head_dim <= 0 || src_row < 0 || dst_row < 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    kv_shift_bf16_kernel<<<(unsigned)rows, kAppendThreads, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(buf), src_row, dst_row, n_kv * head_dim
    );
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_attention_bf16_decode_ring(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* ring_meta,
    int cap,
    int window,
    int n_q,
    int n_kv,
    int head_dim,
    float scaling
) {
    if (n_q <= 0 || n_kv <= 0) return 0;
    if (head_dim > kMaxHD || (n_q % n_kv) != 0 || cap <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    attention_bf16_decode_ring_kernel<<<(unsigned)n_q, kRingThreads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(out),
        ring_meta, cap, window, n_q, n_kv, head_dim, scaling
    );
    return (int)cudaGetLastError();
}
