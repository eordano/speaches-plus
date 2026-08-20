
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "nv_kernels.h"
#include "nvk_smem_optin.cuh"
#include "nvk_gdn_conv.cuh"

__device__ inline float gdn_softplus_safe(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

#define GDN_CONV_MAX_K NVK_GDN_CONV_MAX_K
#define GDN_CHUNK_T_MAX 16

#define gdn_conv_step_silu nvk_gdn_conv_step_silu

__global__ void gdn_conv_decode_silu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x_new,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    int C, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const __nv_bfloat16* w_row = w + (size_t)c * K;
    __nv_bfloat16* s_row = conv_state + (size_t)c * (K - 1);
    y[c] = gdn_conv_step_silu(s_row, x_new[c], w_row, K);
    for (int i = 0; i < K - 2; ++i) {
        s_row[i] = s_row[i + 1];
    }
    s_row[K - 2] = x_new[c];
}

__global__ void gdn_conv_decode_chunk_silu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x_seq,
    const __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ ckpt_conv,
    int C, int K, int t
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const __nv_bfloat16* w_row = w + (size_t)c * K;
    const __nv_bfloat16* s_row = conv_state + (size_t)c * (K - 1);
    __nv_bfloat16 win[GDN_CONV_MAX_K - 1];
    for (int i = 0; i < K - 1; ++i) {
        win[i] = s_row[i];
    }
    for (int i = 0; i < t; ++i) {
        __nv_bfloat16 xb = x_seq[(size_t)i * C + c];
        y[(size_t)i * C + c] = gdn_conv_step_silu(win, xb, w_row, K);
        for (int j = 0; j < K - 2; ++j) {
            win[j] = win[j + 1];
        }
        win[K - 2] = xb;
        __nv_bfloat16* ck = ckpt_conv + ((size_t)i * C + c) * (K - 1);
        for (int j = 0; j < K - 1; ++j) {
            ck[j] = win[j];
        }
    }
}

__device__ inline float gdn_block_sum(float v, float* red32) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) red32[warp] = v;
    __syncthreads();
    int n_warps = blockDim.x >> 5;
    float total = 0.f;
    if (threadIdx.x < 32) {
        float x = (threadIdx.x < n_warps) ? red32[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            x += __shfl_down_sync(0xffffffffu, x, off);
        }
        if (threadIdx.x == 0) red32[0] = x;
    }
    __syncthreads();
    total = red32[0];
    __syncthreads();
    return total;
}

__device__ __forceinline__ void gdn_decode_token_ab_scalars(
    const __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ z,
    float a_h,
    float b_h,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    float* s_head,
    __nv_bfloat16* __restrict__ out,
    int n_k, int n_v, int d_k, int d_v, float rms_eps,
    float* qn, float* kn, float* red32,
    int h, int tid
) {
    int v_per_k = n_v / n_k;
    int kh = h / v_per_k;
    int key_dim = n_k * d_k;

    float sq = 0.f;
    float sk = 0.f;
    for (int i = tid; i < d_k; i += blockDim.x) {
        float qv = __bfloat162float(mixed[(size_t)kh * d_k + i]);
        float kv = __bfloat162float(mixed[(size_t)key_dim + kh * d_k + i]);
        qn[i] = qv;
        kn[i] = kv;
        sq += qv * qv;
        sk += kv * kv;
    }
    __syncthreads();
    float sq_tot = gdn_block_sum(sq, red32);
    float sk_tot = gdn_block_sum(sk, red32);
    float q_mul = (1.f / sqrtf(sq_tot + 1e-6f)) * (1.f / sqrtf((float)d_k));
    float k_mul = 1.f / sqrtf(sk_tot + 1e-6f);
    for (int i = tid; i < d_k; i += blockDim.x) {
        qn[i] *= q_mul;
        kn[i] *= k_mul;
    }
    __syncthreads();

    float a_v = a_h + __bfloat162float(dt_bias[h]);
    float sp = gdn_softplus_safe(a_v);
    float g = -__expf(__bfloat162float(A_log[h])) * sp;
    float g_exp = expf(g);
    float beta_b = __bfloat162float(
        __float2bfloat16(1.f / (1.f + expf(-b_h))));

    float v_t = __bfloat162float(mixed[(size_t)2 * key_dim + (size_t)h * d_v + tid]);

    float kv_mem = 0.f;
    for (int kk = 0; kk < d_k; ++kk) {
        float s = s_head[(size_t)kk * d_v + tid] * g_exp;
        s_head[(size_t)kk * d_v + tid] = s;
        kv_mem += s * kn[kk];
    }
    float delta = (v_t - kv_mem) * beta_b;
    float out_v = 0.f;
    for (int kk = 0; kk < d_k; ++kk) {
        float s = s_head[(size_t)kk * d_v + tid] + kn[kk] * delta;
        s_head[(size_t)kk * d_v + tid] = s;
        out_v += s * qn[kk];
    }

    float core = __bfloat162float(__float2bfloat16(out_v));
    float var = gdn_block_sum(core * core, red32) / (float)d_v;
    float denom = sqrtf(var + rms_eps);
    float normed = __bfloat162float(__float2bfloat16(
        core / denom * __bfloat162float(norm_w[tid])));
    float zv = __bfloat162float(z[(size_t)h * d_v + tid]);
    float gate = __bfloat162float(__float2bfloat16(zv / (1.f + expf(-zv))));
    out[(size_t)h * d_v + tid] = __float2bfloat16(normed * gate);
}

__device__ __forceinline__ void gdn_decode_token(
    const __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    float* s_head,
    __nv_bfloat16* __restrict__ out,
    int n_k, int n_v, int d_k, int d_v, float rms_eps,
    float* qn, float* kn, float* red32,
    int h, int tid
) {
    gdn_decode_token_ab_scalars(
        mixed, z,
        __bfloat162float(a[h]), __bfloat162float(b[h]),
        A_log, dt_bias, norm_w,
        s_head, out,
        n_k, n_v, d_k, d_v, rms_eps,
        qn, kn, red32, h, tid);
}

__global__ void gdn_decode_step_ab_fused_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ a_w,
    const __nv_bfloat16* __restrict__ b_w,
    const __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int hidden, int n_k, int n_v, int d_k, int d_v, float rms_eps
) {
    extern __shared__ float smem[];
    float* qn = smem;
    float* kn = smem + d_k;
    float* red32 = smem + 2 * d_k;

    int h = blockIdx.x;
    int tid = threadIdx.x;

    float sa = 0.f;
    float sb = 0.f;
    const __nv_bfloat16* a_row = a_w + (size_t)h * hidden;
    const __nv_bfloat16* b_row = b_w + (size_t)h * hidden;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float xv = __bfloat162float(x[i]);
        sa += xv * __bfloat162float(a_row[i]);
        sb += xv * __bfloat162float(b_row[i]);
    }
    float a_tot = gdn_block_sum(sa, red32);
    float b_tot = gdn_block_sum(sb, red32);
    float a_h = __bfloat162float(__float2bfloat16(a_tot));
    float b_h = __bfloat162float(__float2bfloat16(b_tot));

    float* s_head = state + (size_t)h * d_k * d_v;
    gdn_decode_token_ab_scalars(
        mixed, z, a_h, b_h,
        A_log, dt_bias, norm_w,
        s_head, out,
        n_k, n_v, d_k, d_v, rms_eps,
        qn, kn, red32, h, tid);
}

__global__ void gdn_decode_step_bf16_kernel(
    const __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int n_k, int n_v, int d_k, int d_v
    , float rms_eps
) {
    extern __shared__ float smem[];
    float* qn = smem;
    float* kn = smem + d_k;
    float* red32 = smem + 2 * d_k;

    int h = blockIdx.x;
    int tid = threadIdx.x;
    float* s_head = state + (size_t)h * d_k * d_v;
    gdn_decode_token(
        mixed, z, a, b, A_log, dt_bias, norm_w,
        s_head, out,
        n_k, n_v, d_k, d_v, rms_eps,
        qn, kn, red32, h, tid);
}

__global__ void gdn_decode_chunk_bf16_kernel(
    const __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    const float* __restrict__ state_in,
    float* __restrict__ ckpt_state,
    __nv_bfloat16* __restrict__ out,
    int n_k, int n_v, int d_k, int d_v, float rms_eps, int t
) {
    extern __shared__ float smem[];
    float* s_tile = smem;
    float* qn = smem + (size_t)d_k * d_v;
    float* kn = qn + d_k;
    float* red32 = kn + d_k;

    int h = blockIdx.x;
    int tid = threadIdx.x;
    size_t mixed_stride = (size_t)2 * n_k * d_k + (size_t)n_v * d_v;

    const float* g_head = state_in + (size_t)h * d_k * d_v;
    for (int kk = 0; kk < d_k; ++kk) {
        s_tile[(size_t)kk * d_v + tid] = g_head[(size_t)kk * d_v + tid];
    }
    for (int i = 0; i < t; ++i) {
        gdn_decode_token(
            mixed + (size_t)i * mixed_stride,
            z + (size_t)i * n_v * d_v,
            a + (size_t)i * n_v,
            b + (size_t)i * n_v,
            A_log, dt_bias, norm_w,
            s_tile,
            out + (size_t)i * n_v * d_v,
            n_k, n_v, d_k, d_v, rms_eps,
            qn, kn, red32, h, tid);
        float* ck = ckpt_state + ((size_t)i * n_v + h) * d_k * d_v;
        for (int kk = 0; kk < d_k; ++kk) {
            ck[(size_t)kk * d_v + tid] = s_tile[(size_t)kk * d_v + tid];
        }
    }
}

template <int K_DIM, int V_DIM, int COLS, int SUBS>
__global__ void gdn_decode_split_step_kernel(
    const float* __restrict__ qn,
    const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ mixed,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ core,
    int n_k, int n_v
) {
    const int VSPLIT = V_DIM / COLS;
    const int KPS = K_DIM / SUBS;
    int vblock = blockIdx.x % VSPLIT;
    int h = blockIdx.x / VSPLIT;
    int v_per_k = n_v / n_k;
    int kh = h / v_per_k;
    int key_dim = n_k * K_DIM;
    int tid = threadIdx.x;
    int col = tid / SUBS;
    int sub = tid % SUBS;
    int v_glob = vblock * COLS + col;

    float* g_state = state + (size_t)h * K_DIM * V_DIM;
    float st[KPS];
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        st[j] = g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob];
    }

    const float* q_head = qn + (size_t)kh * K_DIM;
    const float* k_head = kn + (size_t)kh * K_DIM;
    float ge = g_exp[h];
    float bt = beta[h];
    float v_t = __bfloat162float(
        mixed[(size_t)2 * key_dim + (size_t)h * V_DIM + v_glob]);

    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        float s = st[j] * ge;
        st[j] = s;
        kv += s * k_head[sub * KPS + j];
    }
#pragma unroll
    for (int off = SUBS / 2; off > 0; off >>= 1) {
        kv += __shfl_xor_sync(0xffffffffu, kv, off);
    }
    float delta = (v_t - kv) * bt;
    float ov = 0.f;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        float s = st[j] + k_head[sub * KPS + j] * delta;
        st[j] = s;
        ov += s * q_head[sub * KPS + j];
    }
#pragma unroll
    for (int off = SUBS / 2; off > 0; off >>= 1) {
        ov += __shfl_xor_sync(0xffffffffu, ov, off);
    }
    if (sub == 0) {
        core[(size_t)h * V_DIM + v_glob] = __float2bfloat16(ov);
    }
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob] = st[j];
    }
}

template <int K_DIM, int V_DIM, int COLS, int SUBS>
__global__ void gdn_decode_split_step_lanecol_kernel(
    const float* __restrict__ qn,
    const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ mixed,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ core,
    int n_k, int n_v
) {
    const int VSPLIT = V_DIM / COLS;
    const int KPS = K_DIM / SUBS;
    __shared__ float red[COLS * SUBS];
    int vblock = blockIdx.x % VSPLIT;
    int h = blockIdx.x / VSPLIT;
    int v_per_k = n_v / n_k;
    int kh = h / v_per_k;
    int key_dim = n_k * K_DIM;
    int tid = threadIdx.x;
    int col = tid % COLS;
    int sub = tid / COLS;
    int v_glob = vblock * COLS + col;

    float* g_state = state + (size_t)h * K_DIM * V_DIM;
    float st[KPS];
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        st[j] = g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob];
    }

    const float* q_head = qn + (size_t)kh * K_DIM;
    const float* k_head = kn + (size_t)kh * K_DIM;
    float ge = g_exp[h];
    float bt = beta[h];
    float v_t = __bfloat162float(
        mixed[(size_t)2 * key_dim + (size_t)h * V_DIM + v_glob]);

    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        float s = st[j] * ge;
        st[j] = s;
        kv += s * k_head[sub * KPS + j];
    }
    red[sub * COLS + col] = kv;
    __syncthreads();
#pragma unroll
    for (int off = SUBS / 2; off > 0; off >>= 1) {
        if (sub < off) red[sub * COLS + col] += red[(sub + off) * COLS + col];
        __syncthreads();
    }
    kv = red[col];
    __syncthreads();
    float delta = (v_t - kv) * bt;
    float ov = 0.f;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        float s = st[j] + k_head[sub * KPS + j] * delta;
        st[j] = s;
        ov += s * q_head[sub * KPS + j];
    }
    red[sub * COLS + col] = ov;
    __syncthreads();
#pragma unroll
    for (int off = SUBS / 2; off > 0; off >>= 1) {
        if (sub < off) red[sub * COLS + col] += red[(sub + off) * COLS + col];
        __syncthreads();
    }
    if (sub == 0) {
        core[(size_t)h * V_DIM + v_glob] = __float2bfloat16(red[col]);
    }
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob] = st[j];
    }
}

__global__ void gdn_decode_chunk_split_prep_kernel(
    const __nv_bfloat16* __restrict__ mixed,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ qn,
    float* __restrict__ kn,
    float* __restrict__ g_exp,
    float* __restrict__ beta,
    int n_k, int n_v, int d_k, int d_v
) {
    extern __shared__ float smem[];
    float* red32 = smem;
    int kh = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int key_dim = n_k * d_k;
    size_t mixed_stride = (size_t)2 * key_dim + (size_t)n_v * d_v;
    const __nv_bfloat16* mixed_row = mixed + (size_t)row * mixed_stride;
    float* qn_row = qn + (size_t)row * key_dim;
    float* kn_row = kn + (size_t)row * key_dim;

    float sq = 0.f;
    float sk = 0.f;
    float qv = 0.f;
    float kv = 0.f;
    for (int i = tid; i < d_k; i += blockDim.x) {
        qv = __bfloat162float(mixed_row[(size_t)kh * d_k + i]);
        kv = __bfloat162float(mixed_row[(size_t)key_dim + kh * d_k + i]);
        sq += qv * qv;
        sk += kv * kv;
    }
    __syncthreads();
    float sq_tot = gdn_block_sum(sq, red32);
    float sk_tot = gdn_block_sum(sk, red32);
    float q_mul = (1.f / sqrtf(sq_tot + 1e-6f)) * (1.f / sqrtf((float)d_k));
    float k_mul = 1.f / sqrtf(sk_tot + 1e-6f);
    if (tid < d_k) {
        qn_row[(size_t)kh * d_k + tid] = qv * q_mul;
        kn_row[(size_t)kh * d_k + tid] = kv * k_mul;
    }

    int v_per_k = n_v / n_k;
    if (tid < v_per_k) {
        int h = kh * v_per_k + tid;
        float a_v = __bfloat162float(a[(size_t)row * n_v + h])
            + __bfloat162float(dt_bias[h]);
        float sp = gdn_softplus_safe(a_v);
        float g = -__expf(__bfloat162float(A_log[h])) * sp;
        g_exp[(size_t)row * n_v + h] = expf(g);
        beta[(size_t)row * n_v + h] = __bfloat162float(
            __float2bfloat16(1.f / (1.f + expf(-__bfloat162float(b[(size_t)row * n_v + h])))));
    }
}

template <int K_DIM, int V_DIM, int COLS, int SUBS>
__global__ void gdn_decode_chunk_split_lanecol_kernel(
    const float* __restrict__ qn,
    const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ mixed,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    const float* __restrict__ state_in,
    float* __restrict__ ckpt_state,
    float* __restrict__ live_state_out,
    __nv_bfloat16* __restrict__ core,
    int n_k, int n_v, int t
) {
    const int VSPLIT = V_DIM / COLS;
    const int KPS = K_DIM / SUBS;
    __shared__ float red[COLS * SUBS];
    int vblock = blockIdx.x % VSPLIT;
    int h = blockIdx.x / VSPLIT;
    int v_per_k = n_v / n_k;
    int kh = h / v_per_k;
    int key_dim = n_k * K_DIM;
    size_t mixed_stride = (size_t)2 * key_dim + (size_t)n_v * V_DIM;
    int tid = threadIdx.x;
    int col = tid % COLS;
    int sub = tid / COLS;
    int v_glob = vblock * COLS + col;

    const float* g_state = state_in + (size_t)h * K_DIM * V_DIM;
    float st[KPS];
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        st[j] = g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob];
    }

    for (int i = 0; i < t; ++i) {
        const float* q_head = qn + (size_t)i * key_dim + (size_t)kh * K_DIM;
        const float* k_head = kn + (size_t)i * key_dim + (size_t)kh * K_DIM;
        float ge = g_exp[(size_t)i * n_v + h];
        float bt = beta[(size_t)i * n_v + h];
        float v_t = __bfloat162float(
            mixed[(size_t)i * mixed_stride + (size_t)2 * key_dim + (size_t)h * V_DIM + v_glob]);

        float kv = 0.f;
#pragma unroll
        for (int j = 0; j < KPS; ++j) {
            float s = st[j] * ge;
            st[j] = s;
            kv += s * k_head[sub * KPS + j];
        }
        red[sub * COLS + col] = kv;
        __syncthreads();
#pragma unroll
        for (int off = SUBS / 2; off > 0; off >>= 1) {
            if (sub < off) red[sub * COLS + col] += red[(sub + off) * COLS + col];
            __syncthreads();
        }
        kv = red[col];
        __syncthreads();
        float delta = (v_t - kv) * bt;
        float ov = 0.f;
#pragma unroll
        for (int j = 0; j < KPS; ++j) {
            float s = st[j] + k_head[sub * KPS + j] * delta;
            st[j] = s;
            ov += s * q_head[sub * KPS + j];
        }
        red[sub * COLS + col] = ov;
        __syncthreads();
#pragma unroll
        for (int off = SUBS / 2; off > 0; off >>= 1) {
            if (sub < off) red[sub * COLS + col] += red[(sub + off) * COLS + col];
            __syncthreads();
        }
        if (sub == 0) {
            core[(size_t)i * n_v * V_DIM + (size_t)h * V_DIM + v_glob] =
                __float2bfloat16(red[col]);
        }
        __syncthreads();
        float* ck = ckpt_state + ((size_t)i * n_v + h) * K_DIM * V_DIM;
#pragma unroll
        for (int j = 0; j < KPS; ++j) {
            ck[(size_t)(sub * KPS + j) * V_DIM + v_glob] = st[j];
        }
    }
    float* live = live_state_out + (size_t)h * K_DIM * V_DIM;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        live[(size_t)(sub * KPS + j) * V_DIM + v_glob] = st[j];
    }
}

__global__ void gdn_decode_chunk_split_gate_kernel(
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out,
    int n_v, int d_v, float rms_eps
) {
    extern __shared__ float smem[];
    float* red32 = smem;
    int h = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    size_t base = (size_t)row * n_v * d_v + (size_t)h * d_v + tid;
    float c = __bfloat162float(core[base]);
    float var = gdn_block_sum(c * c, red32) / (float)d_v;
    float denom = sqrtf(var + rms_eps);
    float normed = __bfloat162float(__float2bfloat16(
        c / denom * __bfloat162float(norm_w[tid])));
    float zv = __bfloat162float(z[base]);
    float gate = __bfloat162float(__float2bfloat16(zv / (1.f + expf(-zv))));
    out[base] = __float2bfloat16(normed * gate);
}

extern "C" int nv_kernels_gdn_decode_chunk_split_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    const float* state_in,
    float* ckpt_state,
    float* live_state_out,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int t
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (n_v <= 0 || t <= 0) return 0;
    if (n_k <= 0 || n_v % n_k != 0) return -1;
    if (d_k != 128 || d_v != 128) return -1;
    if (n_v / n_k > d_k) return -1;
    if (t > 16) return -1;
    size_t red_smem = 32 * sizeof(float);
    gdn_decode_chunk_split_prep_kernel<<<dim3((unsigned)n_k, (unsigned)t), d_k, red_smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(mixed),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        qn_scratch, kn_scratch, g_exp_scratch, beta_scratch,
        n_k, n_v, d_k, d_v);
    gdn_decode_chunk_split_lanecol_kernel<128, 128, 32, 8>
        <<<n_v * (128 / 32), 256, 0, s>>>(
            qn_scratch, kn_scratch,
            reinterpret_cast<const __nv_bfloat16*>(mixed),
            g_exp_scratch, beta_scratch,
            state_in, ckpt_state, live_state_out,
            reinterpret_cast<__nv_bfloat16*>(core_scratch),
            n_k, n_v, t);
    gdn_decode_chunk_split_gate_kernel<<<dim3((unsigned)n_v, (unsigned)t), d_v, red_smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(core_scratch),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(norm_w),
        reinterpret_cast<__nv_bfloat16*>(out),
        n_v, d_v, rms_eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_decode_step_split_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (n_v <= 0) return 0;
    if (n_k <= 0 || n_v % n_k != 0) return -2;
    if (d_k != d_v) return -1;
    if (d_k != 128 && d_k != 32) return -1;
    if (n_v / n_k > d_k) return -1;
    size_t red_smem = 32 * sizeof(float);
    gdn_decode_chunk_split_prep_kernel<<<dim3((unsigned)n_k, 1u), d_k, red_smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(mixed),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        qn_scratch, kn_scratch, g_exp_scratch, beta_scratch,
        n_k, n_v, d_k, d_v);
    static const bool lanecol_off_env_nv_q38_gdn_step_v1 =
        getenv("NV_Q38_GDN_STEP_V1") != nullptr;
    if (d_k == 128 && !lanecol_off_env_nv_q38_gdn_step_v1) {
        gdn_decode_split_step_lanecol_kernel<128, 128, 32, 8>
            <<<n_v * (128 / 32), 256, 0, s>>>(
                qn_scratch, kn_scratch,
                reinterpret_cast<const __nv_bfloat16*>(mixed),
                g_exp_scratch, beta_scratch, state,
                reinterpret_cast<__nv_bfloat16*>(core_scratch),
                n_k, n_v);
    } else if (d_k == 128) {
        gdn_decode_split_step_kernel<128, 128, 16, 8>
            <<<n_v * (128 / 16), 128, 0, s>>>(
                qn_scratch, kn_scratch,
                reinterpret_cast<const __nv_bfloat16*>(mixed),
                g_exp_scratch, beta_scratch, state,
                reinterpret_cast<__nv_bfloat16*>(core_scratch),
                n_k, n_v);
    } else {
        gdn_decode_split_step_kernel<32, 32, 8, 4>
            <<<n_v * (32 / 8), 32, 0, s>>>(
                qn_scratch, kn_scratch,
                reinterpret_cast<const __nv_bfloat16*>(mixed),
                g_exp_scratch, beta_scratch, state,
                reinterpret_cast<__nv_bfloat16*>(core_scratch),
                n_k, n_v);
    }
    gdn_decode_chunk_split_gate_kernel<<<dim3((unsigned)n_v, 1u), d_v, red_smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(core_scratch),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(norm_w),
        reinterpret_cast<__nv_bfloat16*>(out),
        n_v, d_v, rms_eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_conv_decode_silu_bf16(
    void* stream,
    const uint16_t* x_new,
    uint16_t* conv_state,
    const uint16_t* w,
    uint16_t* y,
    int conv_dim,
    int k
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (conv_dim <= 0) return 0;
    if (k < 2) return -2;
    constexpr int BLOCK = 256;
    int grid = (conv_dim + BLOCK - 1) / BLOCK;
    gdn_conv_decode_silu_bf16_kernel<<<grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_new),
        reinterpret_cast<__nv_bfloat16*>(conv_state),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(y),
        conv_dim, k);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_decode_step_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (n_v <= 0) return 0;
    if (n_k <= 0 || d_k <= 0 || d_v <= 0) return -2;
    if (n_v % n_k != 0) return -2;
    if (d_v % 32 != 0 || d_v > 1024) return -2;
    size_t smem = (size_t)(2 * d_k + 32) * sizeof(float);
    if (smem > 96 * 1024) return -2;
    static DynamicSmemOptin optin_gdn_decode_step_bf16_kernel;
    int orc_gdn_decode_step_bf16_kernel = raise_dynamic_smem_optin_never_lowering_it(
        optin_gdn_decode_step_bf16_kernel, (const void*)gdn_decode_step_bf16_kernel, smem);
    if (orc_gdn_decode_step_bf16_kernel != 0) return orc_gdn_decode_step_bf16_kernel;
    gdn_decode_step_bf16_kernel<<<n_v, d_v, smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(mixed),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        reinterpret_cast<const __nv_bfloat16*>(norm_w),
        state,
        reinterpret_cast<__nv_bfloat16*>(out),
        n_k, n_v, d_k, d_v, rms_eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_decode_step_ab_fused_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* a_w,
    const uint16_t* b_w,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    int hidden,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (n_v <= 0) return 0;
    if (hidden <= 0 || n_k <= 0 || d_k <= 0 || d_v <= 0) return -2;
    if (n_v % n_k != 0) return -2;
    if (d_v % 32 != 0 || d_v > 1024) return -2;
    size_t smem = (size_t)(2 * d_k + 32) * sizeof(float);
    if (smem > 96 * 1024) return -2;
    static DynamicSmemOptin optin_gdn_decode_step_ab_fused_bf16_kernel;
    int orc_gdn_decode_step_ab_fused_bf16_kernel = raise_dynamic_smem_optin_never_lowering_it(
        optin_gdn_decode_step_ab_fused_bf16_kernel,
        (const void*)gdn_decode_step_ab_fused_bf16_kernel, smem);
    if (orc_gdn_decode_step_ab_fused_bf16_kernel != 0)
        return orc_gdn_decode_step_ab_fused_bf16_kernel;
    gdn_decode_step_ab_fused_bf16_kernel<<<n_v, d_v, smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(a_w),
        reinterpret_cast<const __nv_bfloat16*>(b_w),
        reinterpret_cast<const __nv_bfloat16*>(mixed),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        reinterpret_cast<const __nv_bfloat16*>(norm_w),
        state,
        reinterpret_cast<__nv_bfloat16*>(out),
        hidden, n_k, n_v, d_k, d_v, rms_eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_conv_decode_chunk_silu_bf16(
    void* stream,
    const uint16_t* x_seq,
    const uint16_t* conv_state,
    const uint16_t* w,
    uint16_t* y,
    uint16_t* ckpt_conv,
    int conv_dim,
    int k,
    int t
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (k < 2 || k > GDN_CONV_MAX_K) return -2;
    if (t < 1 || t > GDN_CHUNK_T_MAX) return -2;
    if (conv_dim <= 0) return 0;
    constexpr int BLOCK = 256;
    int grid = (conv_dim + BLOCK - 1) / BLOCK;
    gdn_conv_decode_chunk_silu_bf16_kernel<<<grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_seq),
        reinterpret_cast<const __nv_bfloat16*>(conv_state),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(y),
        reinterpret_cast<__nv_bfloat16*>(ckpt_conv),
        conv_dim, k, t);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_decode_chunk_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    const float* state_in,
    float* ckpt_state,
    uint16_t* out,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int t
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (n_v <= 0) return 0;
    if (n_k <= 0 || d_k <= 0 || d_v <= 0) return -2;
    if (n_v % n_k != 0) return -2;
    if (d_v % 32 != 0 || d_v > 1024) return -2;
    if (t < 1 || t > GDN_CHUNK_T_MAX) return -2;
    size_t smem = ((size_t)d_k * d_v + 2 * (size_t)d_k + 32) * sizeof(float);
    if (smem > 96 * 1024) return -2;
    static DynamicSmemOptin optin_gdn_decode_chunk_bf16_kernel;
    int orc_gdn_decode_chunk_bf16_kernel = raise_dynamic_smem_optin_never_lowering_it(
        optin_gdn_decode_chunk_bf16_kernel, (const void*)gdn_decode_chunk_bf16_kernel, smem);
    if (orc_gdn_decode_chunk_bf16_kernel != 0) return orc_gdn_decode_chunk_bf16_kernel;
    gdn_decode_chunk_bf16_kernel<<<n_v, d_v, smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(mixed),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        reinterpret_cast<const __nv_bfloat16*>(norm_w),
        state_in,
        ckpt_state,
        reinterpret_cast<__nv_bfloat16*>(out),
        n_k, n_v, d_k, d_v, rms_eps, t);
    return (int)cudaGetLastError();
}
