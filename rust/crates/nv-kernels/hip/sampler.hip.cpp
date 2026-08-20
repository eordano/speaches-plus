#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>
#include <stdint.h>
#include <math.h>
#include "nv_kernels.h"

__device__ inline uint64_t splitmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ inline float u64_to_unit_float(uint64_t r) {
    uint32_t mant = (uint32_t)(r >> 40);
    return (float)mant * (1.0f / (float)(1u << 24));
}

template <int BLOCK>
__device__ inline float block_reduce_max(float v, float* scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = scratch[threadIdx.x + stride];
            if (other > scratch[threadIdx.x]) scratch[threadIdx.x] = other;
        }
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

template <int BLOCK>
__device__ inline float block_reduce_sum(float v, float* scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

template <int BLOCK>
__device__ inline uint32_t block_reduce_sum_u32(uint32_t v, uint32_t* scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    uint32_t result = scratch[0];
    __syncthreads();
    return result;
}

template <int BLOCK>
__global__ void sampler_kernel(
    const float* __restrict__ logits,
    const uint64_t* __restrict__ seeds,
    float* __restrict__ probs_out,
    uint32_t* __restrict__ token_out,
    size_t vocab,
    float temperature,
    uint32_t top_k,
    float top_p
) {
    __shared__ float fscratch[BLOCK];
    __shared__ uint32_t uscratch[BLOCK];
    __shared__ float threshold_shared;
    __shared__ float total_shared;
    __shared__ float target_shared;
    __shared__ uint32_t winner_shared;

    size_t row = blockIdx.x;
    const float* row_logits = logits + row * vocab;
    float* row_probs = probs_out + row * vocab;

    float inv_t;
    if (temperature <= 0.0f) {
        inv_t = 1.0e6f;
    } else {
        inv_t = 1.0f / temperature;
    }

    float local_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = row_logits[i] * inv_t;
        if (v > local_max) local_max = v;
    }
    float row_max = block_reduce_max<BLOCK>(local_max, fscratch);

    float local_sum = 0.f;
    for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = row_logits[i] * inv_t;
        float e = expf(v - row_max);
        row_probs[i] = e;
        local_sum += e;
    }
    float row_sum = block_reduce_sum<BLOCK>(local_sum, fscratch);
    float inv_sum = (row_sum > 0.f) ? (1.0f / row_sum) : 0.f;

    for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
        row_probs[i] = row_probs[i] * inv_sum;
    }
    __syncthreads();

    if (top_k > 0 && (size_t)top_k < vocab) {
        float lo = 0.f;
        float hi = 1.0f + 1e-6f;
        for (int iter = 0; iter < 40; ++iter) {
            float mid = 0.5f * (lo + hi);
            uint32_t local_count = 0;
            for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
                if (row_probs[i] >= mid) local_count += 1u;
            }
            uint32_t total_count = block_reduce_sum_u32<BLOCK>(local_count, uscratch);
            if (total_count > top_k) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (threadIdx.x == 0) threshold_shared = hi;
        __syncthreads();
        float thr = threshold_shared;
        for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
            if (row_probs[i] < thr) row_probs[i] = 0.f;
        }
        __syncthreads();

        float local_sum2 = 0.f;
        for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
            local_sum2 += row_probs[i];
        }
        float sum2 = block_reduce_sum<BLOCK>(local_sum2, fscratch);
        float inv2 = (sum2 > 0.f) ? (1.0f / sum2) : 0.f;
        for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
            row_probs[i] = row_probs[i] * inv2;
        }
        __syncthreads();
    }

    if (top_p < 1.0f && top_p > 0.0f) {
        float lo = 0.f;
        float hi = 1.f;
        for (int iter = 0; iter < 40; ++iter) {
            float mid = 0.5f * (lo + hi);
            float local_mass = 0.f;
            for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
                float p = row_probs[i];
                if (p >= mid) local_mass += p;
            }
            float mass = block_reduce_sum<BLOCK>(local_mass, fscratch);
            if (mass > top_p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (threadIdx.x == 0) threshold_shared = lo;
        __syncthreads();
        float thr = threshold_shared;
        for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
            if (row_probs[i] < thr) row_probs[i] = 0.f;
        }
        __syncthreads();

        float local_sum3 = 0.f;
        for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
            local_sum3 += row_probs[i];
        }
        float sum3 = block_reduce_sum<BLOCK>(local_sum3, fscratch);
        float inv3 = (sum3 > 0.f) ? (1.0f / sum3) : 0.f;
        for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
            row_probs[i] = row_probs[i] * inv3;
        }
        __syncthreads();
    }

    float local_total = 0.f;
    for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
        local_total += row_probs[i];
    }
    float total = block_reduce_sum<BLOCK>(local_total, fscratch);

    if (threadIdx.x == 0) {
        uint64_t seed = seeds[row];
        uint64_t mixed = splitmix64(seed ^ (0x9E3779B97F4A7C15ULL + row));
        float u = u64_to_unit_float(mixed);
        if (u >= 1.0f) u = 0.99999994f;
        total_shared = total;
        target_shared = u * total;
        winner_shared = 0xFFFFFFFFu;
    }
    __syncthreads();

    float target = target_shared;

    float local_partial = 0.f;
    for (size_t i = threadIdx.x; i < vocab; i += BLOCK) {
        local_partial += row_probs[i];
    }
    fscratch[threadIdx.x] = local_partial;
    __syncthreads();

    if (threadIdx.x == 0) {
        float cum = 0.f;
        int found_tid = -1;
        float prefix_before = 0.f;
        for (int t = 0; t < BLOCK; ++t) {
            float seg = fscratch[t];
            if (cum + seg >= target) {
                found_tid = t;
                prefix_before = cum;
                break;
            }
            cum += seg;
        }
        if (found_tid < 0) {
            found_tid = BLOCK - 1;
            prefix_before = total_shared;
        }
        uscratch[0] = (uint32_t)found_tid;
        fscratch[0] = prefix_before;
    }
    __syncthreads();

    uint32_t found_tid = uscratch[0];
    float prefix_before = fscratch[0];
    __syncthreads();

    if (threadIdx.x == found_tid) {
        float cum = prefix_before;
        uint32_t pick = 0xFFFFFFFFu;
        for (size_t i = (size_t)threadIdx.x; i < vocab; i += BLOCK) {
            float p = row_probs[i];
            cum += p;
            if (cum >= target && p > 0.f) {
                pick = (uint32_t)i;
                break;
            }
        }
        if (pick == 0xFFFFFFFFu) {
            for (size_t i = vocab; i > 0; --i) {
                if (row_probs[i - 1] > 0.f) {
                    pick = (uint32_t)(i - 1);
                    break;
                }
            }
        }
        winner_shared = pick;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        token_out[row] = winner_shared;
    }
}

extern "C" int nv_kernels_sampler_topk_topp(
    void* stream,
    const float* logits,
    const uint64_t* seeds,
    float* probs_out,
    uint32_t* token_out,
    size_t batch,
    size_t vocab,
    float temperature,
    uint32_t top_k,
    float top_p
) {
    hipStream_t s = (hipStream_t)stream;
    if (batch == 0 || vocab == 0) return 0;
    constexpr int BLOCK = 256;
    sampler_kernel<BLOCK><<<(int)batch, BLOCK, 0, s>>>(
        logits, seeds, probs_out, token_out,
        vocab, temperature, top_k, top_p);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_softmax_topk_topp(
    void* stream,
    const float* logits,
    float* probs_out,
    uint32_t* indices_out,
    size_t batch,
    size_t vocab,
    size_t k,
    float p
) {
    (void)stream;
    (void)logits;
    (void)probs_out;
    (void)indices_out;
    (void)batch;
    (void)vocab;
    (void)k;
    (void)p;
    return -1;
}
