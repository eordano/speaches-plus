#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kBlock = 128;
constexpr int kParts = 256;
constexpr int kMaxRows = 8;

__global__ void spec_begin_kernel(const int* pos, int* rows_pos, int* pos_end, int m) {
    int i = threadIdx.x;
    if (i < m) rows_pos[i] = pos[0] + i;
    if (i == 0) pos_end[0] = pos[0] + m;
}

__global__ void spec_qkv_prep_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    float* __restrict__ q_out,
    __nv_bfloat16* __restrict__ cache_k,
    __nv_bfloat16* __restrict__ cache_v,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ rows_pos,
    int nh,
    int nkv,
    int hd
) {
    int row = blockIdx.x;
    int head = blockIdx.y;
    int pair = threadIdx.x;
    int half = hd / 2;
    if (pair >= half) return;
    int p = rows_pos[row];
    const __nv_bfloat16* src = qkv + (size_t)row * (nh + 2 * nkv) * hd + (size_t)head * hd;
    if (head >= nh + nkv) {
        size_t dst = ((size_t)p * nkv + (head - nh - nkv)) * hd;
        cache_v[dst + pair] = src[pair];
        cache_v[dst + pair + half] = src[pair + half];
        return;
    }
    float c = cos_tbl[(size_t)p * half + pair];
    float s = sin_tbl[(size_t)p * half + pair];
    float a = __bfloat162float(src[pair]);
    float b = __bfloat162float(src[pair + half]);
    if (head < nh) {
        size_t base = ((size_t)row * nh + head) * hd;
        q_out[base + pair] = a * c - b * s;
        q_out[base + pair + half] = a * s + b * c;
    } else {
        size_t dst = ((size_t)p * nkv + (head - nh)) * hd;
        cache_k[dst + pair] = __float2bfloat16(a * c - b * s);
        cache_k[dst + pair + half] = __float2bfloat16(a * s + b * c);
    }
}

__global__ void spec_silu_mul_halves_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ act,
    int rows,
    int inter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * inter) return;
    int r = idx / inter;
    int j = idx - r * inter;
    __nv_bfloat16 v = gate_up[(size_t)r * 2 * inter + j];
    __nv_bfloat16 s = v / (static_cast<__nv_bfloat16>(1) + hexp(-v));
    act[idx] = s * gate_up[(size_t)r * 2 * inter + inter + j];
}

__device__ inline void argmax_merge(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

__global__ void spec_argmax_rows_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int n,
    float* __restrict__ part_val,
    int* __restrict__ part_idx
) {
    __shared__ float sval[kBlock];
    __shared__ int sidx[kBlock];
    const __nv_bfloat16* row = logits + (size_t)blockIdx.y * n;
    int tid = threadIdx.x;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int i = blockIdx.x * kBlock + tid; i < n; i += gridDim.x * kBlock) {
        float v = __bfloat162float(row[i]);
        if (isfinite(v)) argmax_merge(best, bidx, v, i);
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (tid < s) argmax_merge(sval[tid], sidx[tid], sval[tid + s], sidx[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        part_val[(size_t)blockIdx.y * kParts + blockIdx.x] = sval[0];
        part_idx[(size_t)blockIdx.y * kParts + blockIdx.x] = sidx[0];
    }
}

__global__ void spec_accept_kernel(
    const float* __restrict__ part_val,
    const int* __restrict__ part_idx,
    int m,
    int* __restrict__ pos,
    int* __restrict__ toks,
    uint32_t* __restrict__ ring,
    int ring_mask,
    int pad
) {
    __shared__ float sval[kParts];
    __shared__ int sidx[kParts];
    __shared__ int out[kMaxRows];
    int tid = threadIdx.x;
    for (int r = 0; r < m; ++r) {
        sval[tid] = part_val[(size_t)r * kParts + tid];
        sidx[tid] = part_idx[(size_t)r * kParts + tid];
        __syncthreads();
        for (int s = kParts / 2; s > 0; s >>= 1) {
            if (tid < s) argmax_merge(sval[tid], sidx[tid], sval[tid + s], sidx[tid + s]);
            __syncthreads();
        }
        if (tid == 0) out[r] = (sidx[0] == 0x7fffffff) ? 0 : sidx[0];
        __syncthreads();
    }
    if (tid == 0) {
        int p = pos[0];
        int n = 1;
        while (n < m && out[n - 1] == pad) ++n;
        for (int i = 0; i < n; ++i) ring[(p + i) & ring_mask] = (uint32_t)out[i];
        toks[0] = out[n - 1];
        pos[0] = p + n;
    }
}

}

extern "C" int nv_kernels_voxtral_spec_begin(
    void* stream, const int* pos, int* rows_pos, int* pos_end, int m
) {
    if (m < 1 || m > kMaxRows) return -1;
    spec_begin_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(pos, rows_pos, pos_end, m);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_spec_qkv_prep(
    void* stream,
    const uint16_t* qkv,
    float* q_out,
    uint16_t* cache_k,
    uint16_t* cache_v,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* rows_pos,
    int m,
    int nh,
    int nkv,
    int hd
) {
    if (m < 1 || m > kMaxRows || (hd & 1) || hd / 2 > 1024) return -1;
    dim3 grid((unsigned)m, (unsigned)(nh + 2 * nkv));
    spec_qkv_prep_kernel<<<grid, hd / 2, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), q_out, reinterpret_cast<__nv_bfloat16*>(cache_k),
        reinterpret_cast<__nv_bfloat16*>(cache_v), cos_tbl, sin_tbl, rows_pos, nh, nkv, hd);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_spec_silu_mul_halves(
    void* stream, const uint16_t* gate_up, uint16_t* act, int rows, int inter
) {
    if (rows < 1 || inter < 1) return -1;
    int n = rows * inter;
    spec_silu_mul_halves_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_up),
        reinterpret_cast<__nv_bfloat16*>(act), rows, inter);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_spec_argmax_parts() { return kParts; }

extern "C" int nv_kernels_voxtral_spec_accept(
    void* stream,
    const uint16_t* logits,
    int m,
    int n,
    float* part_val,
    int* part_idx,
    int* pos,
    int* toks,
    uint32_t* ring,
    int ring_mask,
    int pad
) {
    if (m < 1 || m > kMaxRows || n <= 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 g1(kParts, (unsigned)m);
    spec_argmax_rows_kernel<<<g1, kBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits), n, part_val, part_idx);
    spec_accept_kernel<<<1, kParts, 0, s>>>(
        part_val, part_idx, m, pos, toks, ring, ring_mask, pad);
    return (int)cudaGetLastError();
}
