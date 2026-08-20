#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_hip_wave.h"

namespace {

constexpr int kBlock = 128;
constexpr int kMaxPerThread = 4;

template <int BLOCK>
__device__ inline float gd_block_sum(float v) {
    constexpr int kW = nv_hip::kWave;
    constexpr int kWaves = BLOCK / kW;
    static_assert(BLOCK >= kW && BLOCK % kW == 0,
                  "BLOCK must be a whole number of wavefronts");
    __shared__ float wave_sums[kWaves];
    __shared__ float total;
    int lane = threadIdx.x & (kW - 1);
    int wave = threadIdx.x / kW;
    v = nv_hip::wave_sum<kW>(v);
    if (lane == 0) wave_sums[wave] = v;
    __syncthreads();
    if (wave == 0) {
        float s = (lane < kWaves) ? wave_sums[lane] : 0.0f;
        s = nv_hip::lane_group_sum<kWaves>(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

constexpr int kLdsFallback = 64 * 1024;

int gd_max_smem_bytes() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return kLdsFallback;
    int v = nv_hip::host_max_lds_bytes(dev);
    return (v > 0) ? v : kLdsFallback;
}

__global__ void incr_pos_kernel(int* pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) pos[0] += 1;
}

__global__ void write_kv_f32_kernel(
    const float* __restrict__ src_k,
    const float* __restrict__ src_v,
    float* __restrict__ cache_k,
    float* __restrict__ cache_v,
    const int* __restrict__ pos,
    int NKV,
    int HD
) {
    int kvh = blockIdx.x;
    if (kvh >= NKV) return;
    int slot = pos[0] - 1;
    if (slot < 0) return;
    size_t dst = ((size_t)slot * NKV + kvh) * HD;
    size_t src = (size_t)kvh * HD;
    for (int d = threadIdx.x; d < HD; d += kBlock) {
        cache_k[dst + d] = src_k[src + d];
        cache_v[dst + d] = src_v[src + d];
    }
}

__global__ void attn_decode_dev_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int* __restrict__ pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    int h = blockIdx.x;
    if (h >= NH) return;
    int total = pos[0];
    int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    int group = NH / NKV;
    int kvh = h / group;
    int tid = threadIdx.x;

    extern __shared__ float qsh[];
    for (int d = tid; d < HD; d += kBlock) qsh[d] = q[(size_t)h * HD + d];
    __syncthreads();

    float acc[kMaxPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxPerThread; ++i) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    __shared__ float red[kBlock];

    for (int p = start; p < total; ++p) {
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

__global__ void cast_bf16_f32_kernel(const __hip_bfloat16* __restrict__ x,
                                     float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

__global__ void cast_f32_bf16_kernel(const float* __restrict__ x,
                                     __hip_bfloat16* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

__global__ void rms_no_weight_kernel(const __hip_bfloat16* __restrict__ x,
                                     float* __restrict__ y, int rows, int dim,
                                     float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    const __hip_bfloat16* xr = x + (size_t)r * dim;
    float* yr = y + (size_t)r * dim;
    float partial = 0.0f;
    for (int d = tid; d < dim; d += kBlock) {
        float v = __bfloat162float(xr[d]);
        partial += v * v;
    }
    float inv = rsqrtf(gd_block_sum<kBlock>(partial) / dim + eps);
    for (int d = tid; d < dim; d += kBlock) {
        yr[d] = __bfloat162float(xr[d]) * inv;
    }
}

__global__ void gelu_mul_bf16f32_kernel(const __hip_bfloat16* __restrict__ gate,
                                        const float* __restrict__ pli,
                                        __hip_bfloat16* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    const float k = 0.7978845608028654f;
    float t = tanhf(k * (g + 0.044715f * g * g * g));
    float gelu = 0.5f * g * (1.0f + t);
    y[i] = __float2bfloat16(gelu * pli[i]);
}

__global__ void cast_scale_bf16_f32_kernel(const __hip_bfloat16* __restrict__ x,
                                           float* __restrict__ y, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]) * scale;
}

__global__ void add_scale_f32_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ y, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (a[i] + b[i]) * scale;
}

extern "C" int nv_kernels_cast_scale_bf16_f32(void* stream, const uint16_t* x, float* y,
                                              float scale, int n) {
    if (n <= 0) return 0;
    int blocks = (n + kBlock - 1) / kBlock;
    cast_scale_bf16_f32_kernel<<<blocks, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x), y, scale, n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_add_scale_f32(void* stream, const float* a, const float* b,
                                        float* y, float scale, int n) {
    if (n <= 0) return 0;
    int blocks = (n + kBlock - 1) / kBlock;
    add_scale_f32_kernel<<<blocks, kBlock, 0, (hipStream_t)stream>>>(a, b, y, scale, n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_cast_bf16_f32(void* stream, const uint16_t* x, float* y, int n) {
    if (n <= 0) return 0;
    int blocks = (n + kBlock - 1) / kBlock;
    cast_bf16_f32_kernel<<<blocks, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x), y, n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_cast_f32_bf16(void* stream, const float* x, uint16_t* y, int n) {
    if (n <= 0) return 0;
    int blocks = (n + kBlock - 1) / kBlock;
    cast_f32_bf16_kernel<<<blocks, kBlock, 0, (hipStream_t)stream>>>(
        x, reinterpret_cast<__hip_bfloat16*>(y), n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rms_no_weight_bf16_f32(void* stream, const uint16_t* x,
                                                 float* y, int rows, int dim, float eps) {
    if (rows <= 0 || dim <= 0) return 0;
    rms_no_weight_kernel<<<(unsigned)rows, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x), y, rows, dim, eps);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_gelu_mul_bf16f32(void* stream, const uint16_t* gate,
                                           const float* pli, uint16_t* y, int n) {
    if (n <= 0) return 0;
    int blocks = (n + kBlock - 1) / kBlock;
    gelu_mul_bf16f32_kernel<<<blocks, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(gate), pli,
        reinterpret_cast<__hip_bfloat16*>(y), n);
    return (int)hipGetLastError();
}

__global__ void incr_pos_rope_kernel(int* pos, int* rope_pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        rope_pos[0] = pos[0];
        pos[0] += 1;
    }
}

constexpr int kArgmaxBlocks = 256;

__global__ void argmax_bf16_stage1_kernel(
    const __hip_bfloat16* __restrict__ logits,
    int n,
    float* __restrict__ part_val,
    int* __restrict__ part_idx
) {
    __shared__ float sval[kBlock];
    __shared__ int sidx[kBlock];
    int tid = threadIdx.x;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int i = blockIdx.x * kBlock + tid; i < n; i += gridDim.x * kBlock) {
        float v = __bfloat162float(logits[i]);
        if (v > best || (v == best && i < bidx)) {
            best = v;
            bidx = i;
        }
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_val[blockIdx.x] = sval[0];
        part_idx[blockIdx.x] = sidx[0];
    }
}

__global__ void argmax_bf16_stage2_kernel(
    const float* __restrict__ part_val,
    const int* __restrict__ part_idx,
    int nparts,
    const int* __restrict__ pos,
    uint32_t* __restrict__ token_out,
    uint32_t* __restrict__ ring,
    int ring_mask
) {
    __shared__ float sval[kArgmaxBlocks];
    __shared__ int sidx[kArgmaxBlocks];
    int tid = threadIdx.x;
    if (tid < nparts) {
        sval[tid] = part_val[tid];
        sidx[tid] = part_idx[tid];
    } else {
        sval[tid] = -INFINITY;
        sidx[tid] = 0x7fffffff;
    }
    __syncthreads();
    for (int s = kArgmaxBlocks / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        uint32_t t = (uint32_t)sidx[0];
        token_out[0] = t;
        if (ring != nullptr) ring[(pos[0] - 1) & ring_mask] = t;
    }
}

__global__ void argmax_f32_rows_stage1_kernel(
    const float* __restrict__ logits,
    int n,
    float* __restrict__ part_val,
    int* __restrict__ part_idx
) {
    __shared__ float sval[kBlock];
    __shared__ int sidx[kBlock];
    const float* row = logits + (size_t)blockIdx.y * n;
    int tid = threadIdx.x;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int i = blockIdx.x * kBlock + tid; i < n; i += gridDim.x * kBlock) {
        float v = row[i];
        if (isfinite(v) && (v > best || (v == best && i < bidx))) {
            best = v;
            bidx = i;
        }
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_val[(size_t)blockIdx.y * kArgmaxBlocks + blockIdx.x] = sval[0];
        part_idx[(size_t)blockIdx.y * kArgmaxBlocks + blockIdx.x] = sidx[0];
    }
}

__global__ void argmax_f32_rows_stage2_kernel(
    const float* __restrict__ part_val,
    const int* __restrict__ part_idx,
    uint32_t* __restrict__ out
) {
    __shared__ float sval[kArgmaxBlocks];
    __shared__ int sidx[kArgmaxBlocks];
    int tid = threadIdx.x;
    sval[tid] = part_val[(size_t)blockIdx.x * kArgmaxBlocks + tid];
    sidx[tid] = part_idx[(size_t)blockIdx.x * kArgmaxBlocks + tid];
    __syncthreads();
    for (int s = kArgmaxBlocks / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = (sidx[0] == 0x7fffffff) ? 0u : (uint32_t)sidx[0];
    }
}

extern "C" int nv_kernels_argmax_f32_rows(
    void* stream,
    const float* logits,
    int rows,
    int n,
    float* part_val,
    int* part_idx,
    uint32_t* out
) {
    if (rows <= 0 || n <= 0) return -1;
    dim3 g1(kArgmaxBlocks, rows);
    argmax_f32_rows_stage1_kernel<<<g1, kBlock, 0, (hipStream_t)stream>>>(
        logits, n, part_val, part_idx);
    argmax_f32_rows_stage2_kernel<<<rows, kArgmaxBlocks, 0, (hipStream_t)stream>>>(
        part_val, part_idx, out);
    return (int)hipGetLastError();
}

__global__ void rmsnorm_bf16w_f32out_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
    float* __restrict__ y,
    int rows,
    int dim,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    const __hip_bfloat16* xr = x + (size_t)r * dim;
    float* yr = y + (size_t)r * dim;
    float partial = 0.0f;
    for (int d = tid; d < dim; d += kBlock) {
        float v = __bfloat162float(xr[d]);
        partial += v * v;
    }
    float inv = rsqrtf(gd_block_sum<kBlock>(partial) / dim + eps);
    for (int d = tid; d < dim; d += kBlock) {
        yr[d] = __bfloat162float(xr[d]) * inv * __bfloat162float(w[d]);
    }
}

__global__ void rmsnorm_add_scale_bf16_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
    const __hip_bfloat16* __restrict__ res,
    __hip_bfloat16* __restrict__ y,
    float* __restrict__ rstd_out,
    const __hip_bfloat16* __restrict__ next_w,
    __hip_bfloat16* __restrict__ normed_out,
    int rows,
    int dim,
    float eps,
    float scale,
    float eps_next
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    const __hip_bfloat16* xr = x + (size_t)r * dim;
    const __hip_bfloat16* rr = res + (size_t)r * dim;
    __hip_bfloat16* yr = y + (size_t)r * dim;
    float partial = 0.0f;
    for (int d = tid; d < dim; d += kBlock) {
        float v = __bfloat162float(xr[d]);
        partial += v * v;
    }

    extern __shared__ float s_of[];
    const bool stage = (normed_out != nullptr);
    float inv = rsqrtf(gd_block_sum<kBlock>(partial) / dim + eps);
    float out_sq = 0.0f;
    for (int d = tid; d < dim; d += kBlock) {
        float normed = __bfloat162float(xr[d]) * inv * __bfloat162float(w[d]);
        float out = (__bfloat162float(rr[d]) + normed) * scale;
        __hip_bfloat16 ob = __float2bfloat16(out);
        yr[d] = ob;

        float of = __bfloat162float(ob);
        if (stage) s_of[d] = of;
        out_sq += of * of;
    }
    if (rstd_out != nullptr || normed_out != nullptr) {
        float total = gd_block_sum<kBlock>(out_sq);
        float inv2 = rsqrtf(total / dim + eps_next);
        if (rstd_out != nullptr && tid == 0) rstd_out[r] = inv2;
        if (normed_out != nullptr) {
            __hip_bfloat16* nr = normed_out + (size_t)r * dim;
            for (int d = tid; d < dim; d += kBlock) {
                nr[d] = __float2bfloat16(s_of[d] * inv2 * __bfloat162float(next_w[d]));
            }
        }
    }
}

constexpr int kPrepMaxHD = 512;

__global__ void qkv_prep_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ qw,
    const __hip_bfloat16* __restrict__ kw,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ rope_pos,
    const int* __restrict__ cache_pos,
    int delta,
    float* __restrict__ q_out,
    __hip_bfloat16* __restrict__ kcache,
    __hip_bfloat16* __restrict__ vcache,
    int NH,
    int NKV,
    int HD,
    float eps
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const bool has_kv = (kw != nullptr);
    const int total_heads = has_kv ? NH + 2 * NKV : NH;
    if (head >= total_heads) return;
    const int kind = head < NH ? 0 : (head < NH + NKV ? 1 : 2);
    const __hip_bfloat16* xr = qkv + (size_t)head * HD;

    __shared__ float ns[kPrepMaxHD];
    float partial = 0.0f;
    for (int d = tid; d < HD; d += kBlock) {
        float v = __bfloat162float(xr[d]);
        partial += v * v;
    }
    float inv = rsqrtf(gd_block_sum<kBlock>(partial) / HD + eps);
    const __hip_bfloat16* w = (kind == 0) ? qw : (kind == 1 ? kw : nullptr);
    for (int d = tid; d < HD; d += kBlock) {
        float n = __bfloat162float(xr[d]) * inv;
        if (w != nullptr) n *= __bfloat162float(w[d]);
        ns[d] = n;
    }
    __syncthreads();

    const int half = HD >> 1;
    if (kind == 0) {
        const int p = rope_pos[0] - delta;
        const float* cr = cos_tbl + (size_t)p * half;
        const float* sr = sin_tbl + (size_t)p * half;
        float* out = q_out + (size_t)head * HD;
        for (int d = tid; d < HD; d += kBlock) {
            int i = (d < half) ? d : d - half;
            float a = ns[i];
            float b = ns[i + half];
            out[d] = (d < half) ? (a * cr[i] - b * sr[i]) : (a * sr[i] + b * cr[i]);
        }
        return;
    }
    const int slot = cache_pos[0] - 1 - delta;
    if (slot < 0) return;
    const int kvh = head - NH - (kind == 2 ? NKV : 0);
    const size_t dst = ((size_t)slot * NKV + kvh) * HD;
    if (kind == 1) {
        const int p = rope_pos[0] - delta;
        const float* cr = cos_tbl + (size_t)p * half;
        const float* sr = sin_tbl + (size_t)p * half;
        for (int d = tid; d < HD; d += kBlock) {
            int i = (d < half) ? d : d - half;
            float a = ns[i];
            float b = ns[i + half];
            float r = (d < half) ? (a * cr[i] - b * sr[i]) : (a * sr[i] + b * cr[i]);
            kcache[dst + d] = __float2bfloat16(r);
        }
    } else {
        for (int d = tid; d < HD; d += kBlock) {
            vcache[dst + d] = __float2bfloat16(ns[d]);
        }
    }
}

__global__ void rstd_bf16_kernel(const __hip_bfloat16* __restrict__ x,
                                 float* __restrict__ rstd_out, int rows,
                                 int dim, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const __hip_bfloat16* xr = x + (size_t)r * dim;
    float partial = 0.0f;
    for (int d = threadIdx.x; d < dim; d += kBlock) {
        float v = __bfloat162float(xr[d]);
        partial += v * v;
    }
    float total = gd_block_sum<kBlock>(partial);
    if (threadIdx.x == 0) rstd_out[r] = rsqrtf(total / dim + eps);
}

__global__ void rms_apply_bf16_kernel(const __hip_bfloat16* __restrict__ x,
                                      const __hip_bfloat16* __restrict__ w,
                                      const float* __restrict__ rstd,
                                      __hip_bfloat16* __restrict__ y, int n,
                                      int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]) * rstd[i / dim] * __bfloat162float(w[i % dim]);
    y[i] = __float2bfloat16(v);
}

extern "C" int nv_kernels_incr_pos_rope(void* stream, int* pos, int* rope_pos) {
    incr_pos_rope_kernel<<<1, 1, 0, (hipStream_t)stream>>>(pos, rope_pos);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_argmax_bf16(
    void* stream,
    const uint16_t* logits,
    int n,
    float* part_val,
    int* part_idx,
    const int* pos,
    uint32_t* token_out,
    uint32_t* ring,
    int ring_mask
) {
    if (n <= 0) return -1;
    argmax_bf16_stage1_kernel<<<kArgmaxBlocks, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(logits), n, part_val, part_idx);
    argmax_bf16_stage2_kernel<<<1, kArgmaxBlocks, 0, (hipStream_t)stream>>>(
        part_val, part_idx, kArgmaxBlocks, pos, token_out, ring, ring_mask);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_argmax_bf16_parts() { return kArgmaxBlocks; }

__global__ void token_map_u32_kernel(
    const uint32_t* __restrict__ map,
    const uint32_t* __restrict__ idx,
    uint32_t* __restrict__ out
) {
    out[0] = map[idx[0]];
}

extern "C" int nv_kernels_token_map_u32(
    void* stream,
    const uint32_t* map,
    const uint32_t* idx,
    uint32_t* out
) {
    token_map_u32_kernel<<<1, 1, 0, (hipStream_t)stream>>>(map, idx, out);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rmsnorm_bf16w_f32out(void* stream, const uint16_t* x,
                                               const uint16_t* w, float* y,
                                               int rows, int dim, float eps) {
    if (rows <= 0 || dim <= 0) return 0;
    rmsnorm_bf16w_f32out_kernel<<<(unsigned)rows, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(w), y, rows, dim, eps);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rmsnorm_add_scale_bf16(void* stream, const uint16_t* x,
                                                 const uint16_t* w, const uint16_t* res,
                                                 uint16_t* y, float* rstd_out,
                                                 const uint16_t* next_w,
                                                 uint16_t* normed_out, int rows,
                                                 int dim, float eps, float scale,
                                                 float eps_next) {
    if (rows <= 0 || dim <= 0) return 0;
    if (normed_out != nullptr && next_w == nullptr) return -1;
    size_t smem = (normed_out != nullptr) ? (size_t)dim * sizeof(float) : 0;
    size_t lds_limit = (size_t)gd_max_smem_bytes();
    size_t lds_reserve = 1024;
    if (lds_limit <= lds_reserve || smem > lds_limit - lds_reserve) return -1;
    rmsnorm_add_scale_bf16_kernel<<<(unsigned)rows, kBlock, smem, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(w),
        reinterpret_cast<const __hip_bfloat16*>(res),
        reinterpret_cast<__hip_bfloat16*>(y), rstd_out,
        reinterpret_cast<const __hip_bfloat16*>(next_w),
        reinterpret_cast<__hip_bfloat16*>(normed_out), rows, dim, eps, scale, eps_next);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_qkv_prep(void* stream, const uint16_t* qkv,
                                   const uint16_t* qw, const uint16_t* kw,
                                   const float* cos_tbl, const float* sin_tbl,
                                   const int* rope_pos, const int* cache_pos,
                                   int delta,
                                   float* q_out, uint16_t* kcache, uint16_t* vcache,
                                   int NH, int NKV, int HD, float eps) {
    if (NH <= 0 || HD <= 0 || HD > kPrepMaxHD || (HD & 1) != 0) return -1;
    if (kw != nullptr && (NKV <= 0 || cache_pos == nullptr || kcache == nullptr ||
                          vcache == nullptr)) return -1;
    int blocks = (kw != nullptr) ? NH + 2 * NKV : NH;
    qkv_prep_kernel<<<(unsigned)blocks, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(qkv),
        reinterpret_cast<const __hip_bfloat16*>(qw),
        reinterpret_cast<const __hip_bfloat16*>(kw),
        cos_tbl, sin_tbl, rope_pos, cache_pos, delta, q_out,
        reinterpret_cast<__hip_bfloat16*>(kcache),
        reinterpret_cast<__hip_bfloat16*>(vcache), NH, NKV, HD, eps);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rstd_bf16(void* stream, const uint16_t* x, float* rstd_out,
                                    int rows, int dim, float eps) {
    if (rows <= 0 || dim <= 0) return 0;
    rstd_bf16_kernel<<<(unsigned)rows, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x), rstd_out, rows, dim, eps);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rms_apply_bf16(void* stream, const uint16_t* x,
                                         const uint16_t* w, const float* rstd,
                                         uint16_t* y, int n, int dim) {
    if (n <= 0) return 0;
    int blocks = (n + kBlock - 1) / kBlock;
    rms_apply_bf16_kernel<<<blocks, kBlock, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(w), rstd,
        reinterpret_cast<__hip_bfloat16*>(y), n, dim);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_incr_pos(void* stream, int* pos) {
    incr_pos_kernel<<<1, 1, 0, (hipStream_t)stream>>>(pos);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_write_kv_f32(
    void* stream,
    const float* src_k,
    const float* src_v,
    float* cache_k,
    float* cache_v,
    const int* pos,
    int NKV,
    int HD
) {
    if (NKV <= 0 || HD <= 0) return 0;
    write_kv_f32_kernel<<<(unsigned)NKV, kBlock, 0, (hipStream_t)stream>>>(
        src_k, src_v, cache_k, cache_v, pos, NKV, HD
    );
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_attn_decode_dev_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const int* pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > kBlock * kMaxPerThread || (NH % NKV) != 0) return -1;
    size_t shmem = (size_t)HD * sizeof(float);
    attn_decode_dev_kernel<<<(unsigned)NH, kBlock, shmem, (hipStream_t)stream>>>(
        q, k, v, out, pos, NH, NKV, HD, WINDOW
    );
    return (int)hipGetLastError();
}

__global__ void multi_zero_bf16_kernel(const ulonglong2* __restrict__ list, int n) {
    const int b = blockIdx.x;
    if (b >= n) return;
    const ulonglong2 e = list[b];
    uint4* p4 = reinterpret_cast<uint4*>((void*)e.x);
    const size_t n4 = (size_t)e.y / 8;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x) p4[i] = z;
}

extern "C" int nv_kernels_multi_zero_bf16(void* stream, const void* list, int n) {
    if (n <= 0) return 0;
    if (list == nullptr) return -1;
    multi_zero_bf16_kernel<<<(unsigned)n, 256, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const ulonglong2*>(list), n);
    return (int)hipGetLastError();
}
