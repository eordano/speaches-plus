
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kBlock = 128;
constexpr int kMaxPerThread = 4;

__global__ void attn_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int NH,
    int NKV,
    int HD,
    int TOTAL,
    int START
) {
    int h = blockIdx.x;
    if (h >= NH) return;
    int group = NH / NKV;
    int kvh = h / group;
    int tid = threadIdx.x;

    extern __shared__ float qsh[];
    for (int d = tid; d < HD; d += kBlock) qsh[d] = q[(size_t)h * HD + d];
    __syncthreads();

    float acc[kMaxPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxPerThread; ++i) acc[i] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;
    __shared__ float red[kBlock];

    for (int p = START; p < TOTAL; ++p) {
        const float* kp = k + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        for (int d = tid; d < HD; d += kBlock) partial += qsh[d] * kp[d];
        red[tid] = partial;
        __syncthreads();
        for (int s = kBlock / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float score = red[0];
        __syncthreads();

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const float* vp = v + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxPerThread; ++i) {
            int d = tid + i * kBlock;
            if (d < HD) acc[i] = acc[i] * corr + w * vp[d];
        }
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int i = 0; i < kMaxPerThread; ++i) {
        int d = tid + i * kBlock;
        if (d < HD) out[(size_t)h * HD + d] = acc[i] * inv_l;
    }
}

}

extern "C" int nv_kernels_attn_decode_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int NH,
    int NKV,
    int HD,
    int TOTAL,
    int START
) {
    if (NH <= 0 || TOTAL <= 0 || NKV <= 0) return 0;
    if (HD > kBlock * kMaxPerThread || (NH % NKV) != 0) return -1;
    if (START < 0) START = 0;
    cudaStream_t s = (cudaStream_t)stream;
    size_t shmem = (size_t)HD * sizeof(float);
    attn_decode_kernel<<<(unsigned)NH, kBlock, shmem, s>>>(
        q, k, v, out, NH, NKV, HD, TOTAL, START
    );
    return (int)cudaGetLastError();
}
