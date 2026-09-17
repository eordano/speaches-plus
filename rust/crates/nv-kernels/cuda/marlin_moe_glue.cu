#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kMoeBlock8 = 8;
constexpr int kAlignThreads = 256;
constexpr int kAlignMaxExperts = 1024;
constexpr int kAlignMaxAssignmentsIs1MBecauseTheOneBlockKernelLoopsOverAssignmentsAndOnlyItsPerExpertSmemIsBounded = 1 << 20;

__global__ void marlin_moe_align_block8_small_kernel(
    const int32_t* __restrict__ topk_ids,
    int assignments,
    int num_experts,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_past_padded
) {
    __shared__ int count[kAlignMaxExperts];
    __shared__ int block_start[kAlignMaxExperts];
    __shared__ int cursor[kAlignMaxExperts];
    __shared__ int warp_sums[kAlignThreads / 32];
    __shared__ int total_blocks;
    const int tid = threadIdx.x;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        count[e] = 0;
        cursor[e] = 0;
    }
    __syncthreads();
    for (int i = tid; i < assignments; i += blockDim.x) atomicAdd(&count[topk_ids[i]], 1);
    __syncthreads();
    int running = 0;
    for (int base = 0; base < num_experts; base += kAlignThreads) {
        const int e = base + tid;
        const int nb = (e < num_experts) ? (count[e] + kMoeBlock8 - 1) / kMoeBlock8 : 0;
        int v = nb;
        const int lane = tid & 31;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, v, off);
            if (lane >= off) v += n;
        }
        if (lane == 31) warp_sums[tid >> 5] = v;
        __syncthreads();
        int warp_prefix = 0;
        for (int w = 0; w < (tid >> 5); ++w) warp_prefix += warp_sums[w];
        int chunk_total = 0;
        for (int w = 0; w < kAlignThreads / 32; ++w) chunk_total += warp_sums[w];
        if (e < num_experts) block_start[e] = running + warp_prefix + v - nb;
        running += chunk_total;
        __syncthreads();
    }
    if (tid == 0) {
        total_blocks = running;
        num_tokens_past_padded[0] = running * kMoeBlock8;
    }
    __syncthreads();
    const int blocks = total_blocks;
    for (int i = tid; i < blocks * kMoeBlock8; i += blockDim.x) sorted_ids[i] = assignments;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        const int nb = (count[e] + kMoeBlock8 - 1) / kMoeBlock8;
        for (int b = 0; b < nb; ++b) expert_ids[block_start[e] + b] = e;
    }
    __syncthreads();
    for (int i = tid; i < assignments; i += blockDim.x) {
        const int e = topk_ids[i];
        const int slot = atomicAdd(&cursor[e], 1);
        sorted_ids[block_start[e] * kMoeBlock8 + slot] = i;
    }
    __syncthreads();
    for (int e = tid; e < num_experts; e += blockDim.x) {
        int32_t* seg = sorted_ids + block_start[e] * kMoeBlock8;
        const int n = count[e];
        for (int a = 1; a < n; ++a) {
            const int32_t v = seg[a];
            int b = a - 1;
            while (b >= 0 && seg[b] > v) {
                seg[b + 1] = seg[b];
                --b;
            }
            seg[b + 1] = v;
        }
    }
}

__global__ void marlin_moe_silu_mul_halves_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ act,
    int rows,
    int inter
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * inter) return;
    const int r = idx / inter;
    const int j = idx - r * inter;
    const float g = __bfloat162float(gate_up[(size_t)r * 2 * inter + j]);
    const float u = __bfloat162float(gate_up[(size_t)r * 2 * inter + inter + j]);
    const float sg = g / (1.0f + __expf(-g));
    act[idx] = __float2bfloat16(sg * u);
}

__global__ void marlin_moe_sum_slots_into_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    float* __restrict__ out,
    int n_tokens,
    int slots,
    int hidden
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * hidden) return;
    const int t = idx / hidden;
    const int h = idx - t * hidden;
    float acc = 0.0f;
    for (int s = 0; s < slots; ++s) acc += __bfloat162float(rows[(size_t)(t * slots + s) * hidden + h]);
    out[idx] = acc;
}

__global__ void marlin_moe_place_rows_by_inv_perm_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int32_t* __restrict__ inv_perm,
    __nv_bfloat16* __restrict__ y_down,
    int n_rows,
    int hidden
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * hidden) return;
    const int r = idx / hidden;
    const int h = idx - r * hidden;
    y_down[(size_t)inv_perm[r] * hidden + h] = rows[idx];
}

__global__ void split_rows_bf16_four_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ a,
    __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ c,
    __nv_bfloat16* __restrict__ d,
    int rows,
    int na,
    int nb,
    int nc,
    int nd,
    int n_src
) {
    const int n_out = na + nb + nc + nd;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n_out) return;
    const int r = idx / n_out;
    const int col = idx - r * n_out;
    const __nv_bfloat16 v = src[r * n_src + col];
    if (col < na) a[r * na + col] = v;
    else if (col < na + nb) b[r * nb + (col - na)] = v;
    else if (col < na + nb + nc) c[r * nc + (col - na - nb)] = v;
    else d[r * nd + (col - na - nb - nc)] = v;
}

}

extern "C" int nv_kernels_split_rows_bf16_four(
    void* stream,
    const uint16_t* src,
    uint16_t* a,
    uint16_t* b,
    uint16_t* c,
    uint16_t* d,
    int rows,
    int na,
    int nb,
    int nc,
    int nd,
    int n_src
) {
    if (src == nullptr || a == nullptr || b == nullptr || c == nullptr || rows < 1 || na < 1 || nb < 1 || nc < 1 || nd < 0) return -1;
    if ((nd > 0) != (d != nullptr)) return -2;
    if (na + nb + nc + nd > n_src) return -3;
    const int total = rows * (na + nb + nc + nd);
    const int threads = 256;
    split_rows_bf16_four_kernel<<<(total + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(src), reinterpret_cast<__nv_bfloat16*>(a), reinterpret_cast<__nv_bfloat16*>(b), reinterpret_cast<__nv_bfloat16*>(c), reinterpret_cast<__nv_bfloat16*>(d), rows, na, nb, nc, nd, n_src);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_marlin_moe_place_rows_by_inv_perm(
    void* stream,
    const uint16_t* rows_bf16,
    const int32_t* inv_perm,
    uint16_t* y_down,
    int n_rows,
    int hidden
) {
    if (rows_bf16 == nullptr || inv_perm == nullptr || y_down == nullptr || n_rows < 1 || hidden < 1) return -1;
    const int total = n_rows * hidden;
    const int threads = 256;
    marlin_moe_place_rows_by_inv_perm_kernel<<<(total + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(rows_bf16), inv_perm, reinterpret_cast<__nv_bfloat16*>(y_down), n_rows, hidden);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_marlin_moe_silu_mul_halves(
    void* stream,
    const uint16_t* gate_up_bf16,
    uint16_t* act_bf16,
    int rows,
    int inter
) {
    if (gate_up_bf16 == nullptr || act_bf16 == nullptr || rows < 1 || inter < 1) return -1;
    const int total = rows * inter;
    const int threads = 256;
    marlin_moe_silu_mul_halves_kernel<<<(total + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_up_bf16), reinterpret_cast<__nv_bfloat16*>(act_bf16), rows, inter);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_marlin_moe_align_block8_small(
    void* stream,
    const int32_t* topk_ids,
    int assignments,
    int num_experts,
    int32_t* sorted_ids,
    int32_t* expert_ids,
    int32_t* num_tokens_past_padded
) {
    if (topk_ids == nullptr || sorted_ids == nullptr || expert_ids == nullptr || num_tokens_past_padded == nullptr) return -1;
    if (assignments < 1 || assignments > kAlignMaxAssignmentsIs1MBecauseTheOneBlockKernelLoopsOverAssignmentsAndOnlyItsPerExpertSmemIsBounded || num_experts < 1 || num_experts > kAlignMaxExperts) return -1;
    marlin_moe_align_block8_small_kernel<<<1, kAlignThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        topk_ids, assignments, num_experts, sorted_ids, expert_ids, num_tokens_past_padded);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_marlin_moe_sum_slots_into_f32(
    void* stream,
    const uint16_t* rows_bf16,
    float* out,
    int n_tokens,
    int slots,
    int hidden
) {
    if (rows_bf16 == nullptr || out == nullptr || n_tokens < 1 || slots < 1 || hidden < 1) return -1;
    const int total = n_tokens * hidden;
    const int threads = 256;
    marlin_moe_sum_slots_into_f32_kernel<<<(total + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(rows_bf16), out, n_tokens, slots, hidden);
    return (int)cudaGetLastError();
}
