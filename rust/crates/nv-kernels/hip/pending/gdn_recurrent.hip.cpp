#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "nv_kernels.h"

template <int K_DIM, int V_TILE>
__global__ __launch_bounds__(V_TILE) void gdn_recurrent_kernel_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int B, int T, int H, int V_DIM
) {
    int tiles  = V_DIM / V_TILE;
    int bh     = blockIdx.x / tiles;
    int v_tile = blockIdx.x - bh * tiles;
    int b  = bh / H;
    int h  = bh % H;
    if (b >= B) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* state  = smem;
    float* k_buf  = smem + K_DIM * V_TILE;
    float* q_buf  = k_buf + K_DIM;

    for (int idx = tid; idx < K_DIM * V_TILE; idx += blockDim.x) {
        state[idx] = 0.f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        size_t kv_base = ((size_t)b * T + t) * H + h;
        size_t qk_base = (((size_t)b * T + t) * H + h) * K_DIM;
        size_t vo_base = (((size_t)b * T + t) * H + h) * (size_t)V_DIM
                       + (size_t)v_tile * V_TILE;

        float ge = g_exp[kv_base];
        float bt = beta[kv_base];

        for (int i = tid; i < K_DIM; i += blockDim.x) {
            k_buf[i] = k[qk_base + i];
            q_buf[i] = q[qk_base + i];
        }
        __syncthreads();

        if (tid < V_TILE) {
            int my_v = tid;

            float v_t = v[vo_base + my_v];

            float kv_mem = 0.f;
            for (int kk = 0; kk < K_DIM; ++kk) {
                float s = state[kk * V_TILE + my_v] * ge;
                state[kk * V_TILE + my_v] = s;
                kv_mem += s * k_buf[kk];
            }

            float delta = (v_t - kv_mem) * bt;
            float out_v = 0.f;
            for (int kk = 0; kk < K_DIM; ++kk) {
                float s = state[kk * V_TILE + my_v] + k_buf[kk] * delta;
                state[kk * V_TILE + my_v] = s;
                out_v += s * q_buf[kk];
            }
            out[vo_base + my_v] = out_v;
        }
        __syncthreads();
    }
}

namespace {

constexpr int kGdnK = 128;

template <int V_TILE>
constexpr size_t gdn_smem_bytes() {
    return (size_t)(kGdnK * V_TILE + 2 * kGdnK) * sizeof(float);
}

template <int V_TILE>
int gdn_launch(
    hipStream_t s,
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta, float* out,
    int B, int T, int H, int V
) {
    size_t smem = gdn_smem_bytes<V_TILE>();
    int blocks  = B * H * (V / V_TILE);
    (void)hipFuncSetAttribute(
        reinterpret_cast<const void*>(gdn_recurrent_kernel_f32<kGdnK, V_TILE>),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    gdn_recurrent_kernel_f32<kGdnK, V_TILE><<<blocks, V_TILE, smem, s>>>(
        q, k, v, g_exp, beta, out, B, T, H, V);
    return (int)hipGetLastError();
}

int gdn_max_lds_bytes() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return 0;
    int lds = 0;
    if (hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock,
                              dev) != hipSuccess) {
        return 0;
    }
    return lds;
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
    hipStream_t s = (hipStream_t)stream;
    if (B * T * H == 0) return 0;
    if (K != 128 || V != 128) {

        return -2;
    }

    int lds = gdn_max_lds_bytes();
    if (lds <= 0) return -2;

    if ((size_t)lds >= gdn_smem_bytes<128>()) {
        return gdn_launch<128>(s, q, k, v, g_exp, beta, out, B, T, H, V);
    }
    if ((size_t)lds >= gdn_smem_bytes<64>()) {
        return gdn_launch<64>(s, q, k, v, g_exp, beta, out, B, T, H, V);
    }
    if ((size_t)lds >= gdn_smem_bytes<32>()) {
        return gdn_launch<32>(s, q, k, v, g_exp, beta, out, B, T, H, V);
    }
    return -2;
}
