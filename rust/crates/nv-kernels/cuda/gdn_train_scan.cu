#include <cuda_runtime.h>
#include "nv_kernels.h"
#include "nvk_smem_optin.cuh"

namespace {

constexpr int kD = 128;
constexpr int kPad = kD + 1;
constexpr int kThreads = 128;

__device__ __forceinline__ float block_sum_128(float v, float* red4) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) red4[warp] = v;
    __syncthreads();
    float s = red4[0] + red4[1] + red4[2] + red4[3];
    __syncthreads();
    return s;
}

__global__ void __launch_bounds__(kThreads) gdn_train_scan_fwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ ckpt,
    int BN,
    int T,
    int chunk
) {
    const int bn = blockIdx.x;
    if (bn >= BN) return;
    const int j = threadIdx.x;
    extern __shared__ float smem[];
    float* state = smem;
    float* k_buf = state + kD * kD;
    float* q_buf = k_buf + kD;
    const int n_chunks = (T + chunk - 1) / chunk;
    for (int idx = j; idx < kD * kD; idx += kThreads) state[idx] = 0.f;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const size_t row = (size_t)bn * T + t;
        if (t % chunk == 0) {
            float* c = ckpt + ((size_t)(t / chunk) * BN + bn) * kD * kD;
            for (int kk = 0; kk < kD; ++kk) c[kk * kD + j] = state[kk * kD + j];
        }
        k_buf[j] = k[row * kD + j];
        q_buf[j] = q[row * kD + j];
        __syncthreads();
        const float ge = g_exp[row];
        const float bt = beta[row];
        const float v_t = v[row * kD + j];
        float kv_mem = 0.f;
        #pragma unroll 8
        for (int kk = 0; kk < kD; ++kk) {
            float s = state[kk * kD + j] * ge;
            state[kk * kD + j] = s;
            kv_mem += s * k_buf[kk];
        }
        const float delta = (v_t - kv_mem) * bt;
        float o = 0.f;
        #pragma unroll 8
        for (int kk = 0; kk < kD; ++kk) {
            float s = state[kk * kD + j] + k_buf[kk] * delta;
            state[kk * kD + j] = s;
            o += s * q_buf[kk];
        }
        out[row * kD + j] = o;
        __syncthreads();
    }
    (void)n_chunks;
}

__global__ void __launch_bounds__(kThreads) gdn_train_scan_bwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    const float* __restrict__ d_out,
    const float* __restrict__ ckpt,
    float* __restrict__ scratch,
    float* __restrict__ dq,
    float* __restrict__ dk,
    float* __restrict__ dv,
    float* __restrict__ dg,
    float* __restrict__ dbeta,
    int BN,
    int T,
    int chunk
) {
    const int bn = blockIdx.x;
    if (bn >= BN) return;
    const int j = threadIdx.x;
    extern __shared__ float smem[];
    float* G = smem;
    float* k_buf = G + kD * kPad;
    float* q_buf = k_buf + kD;
    float* do_buf = q_buf + kD;
    float* delta_buf = do_buf + kD;
    float* dm_buf = delta_buf + kD;
    float* red4 = dm_buf + kD;
    const int n_chunks = (T + chunk - 1) / chunk;
    float* my_scratch = scratch + (size_t)bn * chunk * kD * kD;

    for (int i = 0; i < kD; ++i) G[i * kPad + j] = 0.f;
    __syncthreads();

    for (int cj = n_chunks - 1; cj >= 0; --cj) {
        const int t0 = cj * chunk;
        const int clen = min(chunk, T - t0);
        const float* c = ckpt + ((size_t)cj * BN + bn) * kD * kD;
        for (int ti = 0; ti < clen; ++ti) {
            const size_t row = (size_t)bn * T + t0 + ti;
            const float* sp = (ti == 0) ? c : my_scratch + (size_t)(ti - 1) * kD * kD;
            k_buf[j] = k[row * kD + j];
            __syncthreads();
            const float ge = g_exp[row];
            const float bt = beta[row];
            const float v_t = v[row * kD + j];
            float kv_mem = 0.f;
            #pragma unroll 8
            for (int kk = 0; kk < kD; ++kk) kv_mem += __ldg(sp + kk * kD + j) * ge * k_buf[kk];
            const float delta = (v_t - kv_mem) * bt;
            if (ti + 1 < clen) {
                float* nx = my_scratch + (size_t)ti * kD * kD;
                #pragma unroll 8
                for (int kk = 0; kk < kD; ++kk) {
                    nx[kk * kD + j] = __ldg(sp + kk * kD + j) * ge + k_buf[kk] * delta;
                }
            }
            __syncthreads();
        }
        for (int ti = clen - 1; ti >= 0; --ti) {
            const size_t row = (size_t)bn * T + t0 + ti;
            const float* sp = (ti == 0) ? c : my_scratch + (size_t)(ti - 1) * kD * kD;
            k_buf[j] = k[row * kD + j];
            q_buf[j] = q[row * kD + j];
            do_buf[j] = d_out[row * kD + j];
            __syncthreads();
            const float ge = g_exp[row];
            const float bt = beta[row];
            const float v_t = v[row * kD + j];
            const float do_j = do_buf[j];
            float m = 0.f;
            float ddelta = 0.f;
            #pragma unroll 8
            for (int i = 0; i < kD; ++i) {
                m += __ldcg(sp + i * kD + j) * ge * k_buf[i];
                float gij = G[i * kPad + j] + q_buf[i] * do_j;
                G[i * kPad + j] = gij;
                ddelta += k_buf[i] * gij;
            }
            const float delta = (v_t - m) * bt;
            const float dm = -bt * ddelta;
            delta_buf[j] = delta;
            dm_buf[j] = dm;
            dv[row * kD + j] = bt * ddelta;
            const float dbeta_sum = block_sum_128(ddelta * (v_t - m), red4);
            const float do_dot_delta = block_sum_128(do_j * delta, red4);
            if (j == 0) dbeta[row] = dbeta_sum;
            const int i = j;
            const float k_i = k_buf[i];
            const float* sprow = sp + (size_t)i * kD;
            float dq_i = 0.f;
            float dk_i = 0.f;
            float dalpha = 0.f;
            #pragma unroll 8
            for (int jj = 0; jj < kD; ++jj) {
                const float spv = __ldcg(sprow + jj);
                dq_i += do_buf[jj] * spv;
                const float gij = G[i * kPad + jj];
                dk_i += delta_buf[jj] * gij + dm_buf[jj] * ge * spv;
                const float dd = gij + k_i * dm_buf[jj];
                dalpha += dd * spv;
                G[i * kPad + jj] = ge * dd;
            }
            dq[row * kD + i] = ge * dq_i + k_i * do_dot_delta;
            dk[row * kD + i] = dk_i;
            const float dalpha_sum = block_sum_128(dalpha, red4);
            if (j == 0) dg[row] = dalpha_sum;
            __syncthreads();
        }
    }
}

}

extern "C" int nv_kernels_gdn_train_scan_fwd_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    const float* g_exp,
    const float* beta,
    float* out,
    float* ckpt,
    int bn,
    int t,
    int dk,
    int dv,
    int chunk
) {
    if (bn <= 0 || t <= 0) return 0;
    if (dk != kD || dv != kD || chunk <= 0) return -2;
    size_t smem = (size_t)(kD * kD + 2 * kD) * sizeof(float);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)gdn_train_scan_fwd_kernel, smem);
    if (orc != 0) return orc;
    gdn_train_scan_fwd_kernel<<<bn, kThreads, smem, (cudaStream_t)stream>>>(
        q, k, v, g_exp, beta, out, ckpt, bn, t, chunk);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_train_scan_bwd_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    const float* g_exp,
    const float* beta,
    const float* d_out,
    const float* ckpt,
    float* scratch,
    float* dq,
    float* dk,
    float* dv,
    float* dg,
    float* dbeta,
    int bn,
    int t,
    int dkd,
    int dvd,
    int chunk
) {
    if (bn <= 0 || t <= 0) return 0;
    if (dkd != kD || dvd != kD || chunk <= 0) return -2;
    size_t smem = (size_t)(kD * kPad + 5 * kD + 4) * sizeof(float);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)gdn_train_scan_bwd_kernel, smem);
    if (orc != 0) return orc;
    gdn_train_scan_bwd_kernel<<<bn, kThreads, smem, (cudaStream_t)stream>>>(
        q, k, v, g_exp, beta, d_out, ckpt, scratch, dq, dk, dv, dg, dbeta, bn, t, chunk);
    return (int)cudaGetLastError();
}
