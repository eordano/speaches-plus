#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nv_kernels.h"

__global__ void gdn_prefill_qk_l2norm_from_mixed_kernel(
    const __nv_bfloat16* __restrict__ mixed,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int HK,
    int conv_dim,
    int key_dim,
    float q_scale,
    float l2_eps
) {
    const int K_DIM = 128;
    int kh = blockIdx.x % HK;
    size_t bt = blockIdx.x / HK;
    int i = threadIdx.x;
    size_t base = bt * (size_t)conv_dim + (size_t)kh * K_DIM;
    float qv = __bfloat162float(mixed[base + i]);
    float kv = __bfloat162float(mixed[base + key_dim + i]);
    float qs = qv * qv;
    float ks = kv * kv;
    for (int off = 16; off > 0; off >>= 1) {
        qs += __shfl_xor_sync(0xffffffffu, qs, off);
        ks += __shfl_xor_sync(0xffffffffu, ks, off);
    }
    __shared__ float warp_q[4];
    __shared__ float warp_k[4];
    int warp = i >> 5;
    if ((i & 31) == 0) {
        warp_q[warp] = qs;
        warp_k[warp] = ks;
    }
    __syncthreads();
    float q_sum = warp_q[0] + warp_q[1] + warp_q[2] + warp_q[3];
    float k_sum = warp_k[0] + warp_k[1] + warp_k[2] + warp_k[3];
    float q_inv = rsqrtf(q_sum + l2_eps) * q_scale;
    float k_inv = rsqrtf(k_sum + l2_eps);
    size_t o = (bt * (size_t)HK + kh) * K_DIM + i;
    q_out[o] = qv * q_inv;
    k_out[o] = kv * k_inv;
}

extern "C" int nv_kernels_gdn_prefill_qk_l2norm_from_mixed(
    void* stream,
    const uint16_t* mixed,
    float* q_out,
    float* k_out,
    int BT,
    int HK,
    int conv_dim,
    int key_dim,
    float q_scale,
    float l2_eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (BT * HK == 0) return 0;
    if (key_dim != HK * 128) return -2;
    dim3 grid((unsigned)(BT * HK));
    gdn_prefill_qk_l2norm_from_mixed_kernel<<<grid, 128, 0, s>>>(
        (const __nv_bfloat16*)mixed, q_out, k_out, HK, conv_dim, key_dim, q_scale, l2_eps);
    return (int)cudaGetLastError();
}

template <int K_DIM, int V_DIM, int COLS, int SUBS>
__global__ void gdn_recurrent_stateful_gqa_regstate_kernel(
    const float* __restrict__ qn,
    const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ mixed,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    float* __restrict__ state_inout,
    __nv_bfloat16* __restrict__ out,
    int B, int T, int H, int HK,
    int conv_dim, int v_channel_offset
) {
    const int VSPLIT = V_DIM / COLS;
    const int KPS = K_DIM / SUBS;
    int vblock = blockIdx.x % VSPLIT;
    int bh = blockIdx.x / VSPLIT;
    int h = bh % H;
    int b = bh / H;
    if (b >= B) return;
    int kh = h / (H / HK);

    int tid = threadIdx.x;
    int col = tid / SUBS;
    int sub = tid % SUBS;
    int v_glob = vblock * COLS + col;

    float st[KPS];
    float* g_state = state_inout + (size_t)bh * K_DIM * V_DIM;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        st[j] = g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob];
    }

    __shared__ float k_buf[K_DIM];
    __shared__ float q_buf[K_DIM];

    size_t qk_base0 = (((size_t)b * T) * HK + kh) * K_DIM;
    float k_next = kn[qk_base0 + tid];
    float q_next = qn[qk_base0 + tid];
    float ge_next = g_exp[(size_t)b * T * H + h];
    float bt_next = beta[(size_t)b * T * H + h];
    float v_next = __bfloat162float(
        mixed[(size_t)b * T * conv_dim + v_channel_offset + (size_t)h * V_DIM + v_glob]);

    for (int t = 0; t < T; ++t) {
        size_t bt_idx = (size_t)b * T + t;
        float ge = ge_next;
        float bt_c = bt_next;
        float v_t = v_next;
        __syncthreads();
        k_buf[tid] = k_next;
        q_buf[tid] = q_next;
        __syncthreads();
        if (t + 1 < T) {
            size_t nb = bt_idx + 1;
            size_t qk_base = (nb * HK + kh) * K_DIM;
            k_next = kn[qk_base + tid];
            q_next = qn[qk_base + tid];
            ge_next = g_exp[nb * H + h];
            bt_next = beta[nb * H + h];
            v_next = __bfloat162float(
                mixed[nb * conv_dim + v_channel_offset + (size_t)h * V_DIM + v_glob]);
        }

        float kv = 0.f;
#pragma unroll
        for (int j = 0; j < KPS; ++j) {
            float s = st[j] * ge;
            st[j] = s;
            kv += s * k_buf[sub * KPS + j];
        }
#pragma unroll
        for (int off = SUBS / 2; off > 0; off >>= 1) {
            kv += __shfl_xor_sync(0xffffffffu, kv, off);
        }
        float delta = (v_t - kv) * bt_c;
        float ov = 0.f;
#pragma unroll
        for (int j = 0; j < KPS; ++j) {
            float s = st[j] + k_buf[sub * KPS + j] * delta;
            st[j] = s;
            ov += s * q_buf[sub * KPS + j];
        }
#pragma unroll
        for (int off = SUBS / 2; off > 0; off >>= 1) {
            ov += __shfl_xor_sync(0xffffffffu, ov, off);
        }
        if (sub == 0) {
            out[(bt_idx * H + h) * V_DIM + v_glob] = __float2bfloat16(ov);
        }
    }

#pragma unroll
    for (int j = 0; j < KPS; ++j) {
        g_state[(size_t)(sub * KPS + j) * V_DIM + v_glob] = st[j];
    }
}

extern "C" int nv_kernels_gdn_recurrent_stateful_gqa_bf16out(
    void* stream,
    const float* qn,
    const float* kn,
    const uint16_t* mixed,
    const float* g_exp,
    const float* beta,
    float* state_inout,
    uint16_t* out,
    int B,
    int T,
    int H,
    int HK,
    int K,
    int V,
    int conv_dim,
    int v_channel_offset
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (B * T * H == 0) return 0;
    if (K != 128 || V != 128) return -2;
    if (HK <= 0 || H % HK != 0) return -3;
    const int COLS = 16;
    const int SUBS = 8;
    const int VSPLIT = 128 / COLS;
    int blocks = B * H * VSPLIT;
    gdn_recurrent_stateful_gqa_regstate_kernel<128, 128, COLS, SUBS>
        <<<blocks, COLS * SUBS, 0, s>>>(
            qn, kn, (const __nv_bfloat16*)mixed, g_exp, beta, state_inout,
            (__nv_bfloat16*)out, B, T, H, HK, conv_dim, v_channel_offset);
    return (int)cudaGetLastError();
}

__global__ void gdn_prefill_rmsnorm_gate_kernel(
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ gated,
    float rms_eps
) {
    const int V_DIM = 128;
    size_t row = blockIdx.x;
    int i = threadIdx.x;
    size_t idx = row * V_DIM + i;
    float x = __bfloat162float(core[idx]);
    float ss = x * x;
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, off);
    }
    __shared__ float warp_ss[4];
    int warp = i >> 5;
    if ((i & 31) == 0) {
        warp_ss[warp] = ss;
    }
    __syncthreads();
    float var = (warp_ss[0] + warp_ss[1] + warp_ss[2] + warp_ss[3]) / (float)V_DIM;
    float inv = rsqrtf(var + rms_eps);
    float w = __bfloat162float(norm_weight[i]);
    __nv_bfloat16 normed = __float2bfloat16(x * inv * w);
    float zf = __bfloat162float(z[idx]);
    float gate_f = zf / (1.0f + expf(-zf));
    __nv_bfloat16 gate = __float2bfloat16(gate_f);
    gated[idx] = __hmul(normed, gate);
}

extern "C" int nv_kernels_gdn_prefill_rmsnorm_gate_bf16(
    void* stream,
    const uint16_t* core,
    const uint16_t* z,
    const uint16_t* norm_weight,
    uint16_t* gated,
    int rows,
    int v_dim,
    float rms_eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (rows == 0) return 0;
    if (v_dim != 128) return -2;
    gdn_prefill_rmsnorm_gate_kernel<<<rows, 128, 0, s>>>(
        (const __nv_bfloat16*)core, (const __nv_bfloat16*)z,
        (const __nv_bfloat16*)norm_weight, (__nv_bfloat16*)gated, rms_eps);
    return (int)cudaGetLastError();
}
__global__ void gdn_conv1d_silu_bt_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ state_in,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ state_out,
    int B, int T, int C, int K
) {
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int t = blockIdx.x % T;
    int b = blockIdx.x / T;
    int P = K - 1;
    float acc = 0.f;
    for (int j = 0; j < K; ++j) {
        int ti = t + j - P;
        float v;
        if (ti >= 0) v = __bfloat162float(x[((size_t)b * T + ti) * C + c]);
        else if (state_in) v = __bfloat162float(state_in[((size_t)b * C + c) * P + (P + ti)]);
        else v = 0.f;
        acc += v * __bfloat162float(w[c * K + j]);
    }
    float sig = 1.f / (1.f + expf(-acc));
    y[((size_t)b * T + t) * C + c] = __float2bfloat16(acc * sig);
    if (t == 0 && state_out) {
        for (int p = 0; p < P; ++p) {
            int ti = T - P + p;
            float v;
            if (ti >= 0) v = __bfloat162float(x[((size_t)b * T + ti) * C + c]);
            else if (state_in) v = __bfloat162float(state_in[((size_t)b * C + c) * P + (P + ti)]);
            else v = 0.f;
            state_out[((size_t)b * C + c) * P + p] = __float2bfloat16(v);
        }
    }
}

extern "C" int nv_kernels_gdn_conv1d_silu_bt_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* state_in,
    const uint16_t* w,
    uint16_t* y,
    uint16_t* state_out,
    int B,
    int T,
    int C,
    int K
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (B * T * C == 0) return 0;
    if (K < 2 || K > 8) return -2;
    dim3 grid((unsigned)(B * T), (unsigned)((C + 255) / 256));
    gdn_conv1d_silu_bt_kernel<<<grid, 256, 0, s>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)state_in, (const __nv_bfloat16*)w,
        (__nv_bfloat16*)y, (__nv_bfloat16*)state_out, B, T, C, K);
    return (int)cudaGetLastError();
}
