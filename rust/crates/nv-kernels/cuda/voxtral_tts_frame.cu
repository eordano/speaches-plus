#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kArgmaxBlock = 1024;
constexpr int kMaxFlowTokens = 4;

__device__ inline float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline __nv_bfloat16 rn(float v) { return __float2bfloat16_rn(v); }
__device__ inline float round_bf(float v) { return bf(rn(v)); }

__global__ void tts_begin_kernel(
    const int* pos, int* rows_pos, int* pos_end, const int* fi,
    const __nv_bfloat16* noise, __nv_bfloat16* x, int noise_rows, int n
) {
    int i = threadIdx.x;
    if (i == 0) {
        rows_pos[0] = pos[0];
        pos_end[0] = pos[0] + 1;
    }
    int slot = fi[0] % noise_rows;
    for (int j = i; j < n; j += blockDim.x) x[j] = noise[(size_t)slot * n + j];
}

__device__ inline void argmax_merge(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

__global__ void tts_semantic_argmax_kernel(
    const __nv_bfloat16* __restrict__ logits, int rows, const int* fi, int ring_slots, int nc,
    uint32_t* ring, int end_token
) {
    __shared__ float sval[kArgmaxBlock];
    __shared__ int sidx[kArgmaxBlock];
    int tid = threadIdx.x;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int i = 1 + tid; i < rows; i += blockDim.x) argmax_merge(best, bidx, bf(logits[i]), i);
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int s = kArgmaxBlock / 2; s > 0; s >>= 1) {
        if (tid < s) argmax_merge(sval[tid], sidx[tid], sval[tid + s], sidx[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        ring[(size_t)(fi[0] % ring_slots) * nc] = (sidx[0] == 0x7fffffff) ? (uint32_t)end_token : (uint32_t)sidx[0];
    }
}

__global__ void tts_flow_attn_kernel(
    const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
    int S, int nh, int nkv, int hd, float scale
) {
    extern __shared__ float sm[];
    int b = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    int kvh = h / (nh / nkv);
    int row_stride = (nh + 2 * nkv) * hd;
    float* q = sm;
    float* k = q + S * hd;
    float* v = k + S * hd;
    float* sc = v + S * hd;
    for (int s = 0; s < S; ++s) {
        const __nv_bfloat16* row = qkv + (size_t)(b * S + s) * row_stride;
        q[s * hd + lane] = bf(row[h * hd + lane]);
        k[s * hd + lane] = bf(row[(nh + kvh) * hd + lane]);
        v[s * hd + lane] = bf(row[(nh + nkv + kvh) * hd + lane]);
    }
    __syncthreads();
    int warp = lane / 32, wl = lane % 32, nw = blockDim.x / 32;
    for (int p = warp; p < S * S; p += nw) {
        int i = p / S, j = p % S;
        float acc = 0.0f;
        for (int e = wl; e < hd; e += 32) acc += q[i * hd + e] * k[j * hd + e];
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (wl == 0) sc[p] = acc * scale;
    }
    __syncthreads();
    for (int i = 0; i < S; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < S; ++j) m = fmaxf(m, sc[i * S + j]);
        float den = 0.0f, o = 0.0f;
        for (int j = 0; j < S; ++j) {
            float p = expf(sc[i * S + j] - m);
            den += p;
            o += p * v[j * hd + lane];
        }
        out[(size_t)(b * S + i) * nh * hd + h * hd + lane] = rn(o / den);
    }
}

__global__ void tts_fsq_kernel(
    const __nv_bfloat16* x, int n, float levels, const int* fi, int ring_slots, int nc,
    uint32_t* ring, int end_token, int n_special
) {
    int j = threadIdx.x;
    if (j >= n) return;
    uint32_t* frame = ring + (size_t)(fi[0] % ring_slots) * nc;
    float c = fminf(fmaxf(bf(x[j]), -1.0f), 1.0f);
    float y = round_bf(__fadd_rn(c, 1.0f));
    y = round_bf(__fmul_rn(y, 0.5f));
    y = round_bf(__fmul_rn(y, levels));
    float r = fminf(fmaxf(rintf(y), 0.0f), levels);
    uint32_t code = (frame[0] == (uint32_t)end_token) ? 0u : (uint32_t)r;
    frame[1 + j] = code + (uint32_t)n_special;
}

__global__ void tts_embed_sum_kernel(
    const __nv_bfloat16* __restrict__ table, const int* __restrict__ offsets,
    const uint32_t* __restrict__ ring, const int* __restrict__ teacher, const int* teacher_len,
    const int* fi, int ring_slots, int nc, __nv_bfloat16* out, int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) return;
    int f = fi[0];
    const uint32_t* gen = ring + (size_t)(f % ring_slots) * nc;
    const int* tch = teacher + (size_t)f * nc;
    bool forced = f < teacher_len[0];
    float acc = 0.0f;
    for (int c = 0; c < nc; ++c) {
        int code = forced ? tch[c] : (int)gen[c];
        acc += bf(table[((size_t)offsets[c] + code) * d + j]);
    }
    out[j] = rn(acc);
}

__global__ void tts_advance_kernel(int* pos, int* fi) {
    pos[0] += 1;
    fi[0] += 1;
}

}

extern "C" int nv_kernels_voxtral_tts_begin(
    void* stream, const int* pos, int* rows_pos, int* pos_end, const int* fi,
    const uint16_t* noise, uint16_t* x, int noise_rows, int n
) {
    if (noise_rows < 1 || n < 1) return -1;
    tts_begin_kernel<<<1, 64, 0, (cudaStream_t)stream>>>(
        pos, rows_pos, pos_end, fi, reinterpret_cast<const __nv_bfloat16*>(noise),
        reinterpret_cast<__nv_bfloat16*>(x), noise_rows, n);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_semantic_argmax(
    void* stream, const uint16_t* logits, int rows, const int* fi, int ring_slots, int nc,
    uint32_t* ring, int end_token
) {
    if (rows < 2 || ring_slots < 1 || nc < 1) return -1;
    tts_semantic_argmax_kernel<<<1, kArgmaxBlock, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits), rows, fi, ring_slots, nc, ring, end_token);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_flow_attn(
    void* stream, const uint16_t* qkv, uint16_t* out, int b, int s, int nh, int nkv, int hd, float scale
) {
    if (b < 1 || s < 1 || s > kMaxFlowTokens || nh < 1 || nkv < 1 || nh % nkv || hd % 32 || hd > 1024) return -1;
    size_t smem = (size_t)(3 * s * hd + s * s) * sizeof(float);
    dim3 grid((unsigned)b, (unsigned)nh);
    tts_flow_attn_kernel<<<grid, hd, smem, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<__nv_bfloat16*>(out),
        s, nh, nkv, hd, scale);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_fsq(
    void* stream, const uint16_t* x, int n, float levels, const int* fi, int ring_slots, int nc,
    uint32_t* ring, int end_token, int n_special
) {
    if (n < 1 || n > 1024 || nc != n + 1) return -1;
    tts_fsq_kernel<<<1, n, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), n, levels, fi, ring_slots, nc, ring, end_token,
        n_special);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_embed_sum(
    void* stream, const uint16_t* table, const int* offsets, const uint32_t* ring, const int* teacher,
    const int* teacher_len, const int* fi, int ring_slots, int nc, uint16_t* out, int d
) {
    if (d < 1 || nc < 1) return -1;
    tts_embed_sum_kernel<<<(d + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(table), offsets, ring, teacher, teacher_len, fi,
        ring_slots, nc, reinterpret_cast<__nv_bfloat16*>(out), d);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_advance(void* stream, int* pos, int* fi) {
    tts_advance_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(pos, fi);
    return (int)cudaGetLastError();
}
