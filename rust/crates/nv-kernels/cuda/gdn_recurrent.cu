#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nv_kernels.h"
#include "nvk_smem_optin.cuh"

template <int K_DIM, int V_DIM>
__global__ void gdn_recurrent_kernel_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int B, int T, int H
) {
    int bh = blockIdx.x;
    int b  = bh / H;
    int h  = bh % H;
    if (b >= B) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* state  = smem;
    float* k_buf  = smem + K_DIM * V_DIM;
    float* q_buf  = k_buf + K_DIM;

    for (int idx = tid; idx < K_DIM * V_DIM; idx += blockDim.x) {
        state[idx] = 0.f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        size_t kv_base = ((size_t)b * T + t) * H + h;
        size_t qk_base = (((size_t)b * T + t) * H + h) * K_DIM;
        size_t vo_base = (((size_t)b * T + t) * H + h) * V_DIM;

        float ge = g_exp[kv_base];
        float bt = beta[kv_base];

        for (int i = tid; i < K_DIM; i += blockDim.x) {
            k_buf[i] = k[qk_base + i];
            q_buf[i] = q[qk_base + i];
        }
        __syncthreads();

        if (tid < V_DIM) {
            int my_v = tid;
            float v_t = v[vo_base + my_v];

            float kv_mem = 0.f;
            for (int kk = 0; kk < K_DIM; ++kk) {
                float s = state[kk * V_DIM + my_v] * ge;
                state[kk * V_DIM + my_v] = s;
                kv_mem += s * k_buf[kk];
            }

            float delta = (v_t - kv_mem) * bt;
            float out_v = 0.f;
            for (int kk = 0; kk < K_DIM; ++kk) {
                float s = state[kk * V_DIM + my_v] + k_buf[kk] * delta;
                state[kk * V_DIM + my_v] = s;
                out_v += s * q_buf[kk];
            }
            out[vo_base + my_v] = out_v;
        }
        __syncthreads();
    }
}

extern "C" int nv_kernels_gdn_recurrent_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    const float* g_exp,
    const float* beta,
    float* out,
    int B,
    int T,
    int H,
    int K,
    int V
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (B * T * H == 0) return 0;
    if (K != 128 || V != 128) {
        return -2;
    }
    int blocks = B * H;
    int threads = V;
    size_t smem = (K * V + 2 * K) * sizeof(float);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gdn_recurrent_kernel_f32<128, 128>, smem);
    if (orc != 0) return orc;
    gdn_recurrent_kernel_f32<128, 128><<<blocks, threads, smem, s>>>(
        q, k, v, g_exp, beta, out, B, T, H);
    return (int)cudaGetLastError();
}
