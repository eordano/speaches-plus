#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nv_kernels.h"
#include "nvk_norm_m1.cuh"

template <int BLOCK>
__device__ inline float
writeout_smem_tree_sum_is_the_stock_reduction_so_the_wide_256_rung_stays_bitwise(float v) {
    __shared__ float scratch[BLOCK];
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

template <int BLOCK, int VPT>
__global__ void rmsnorm_residual_writeout_m1_wide_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ res_out,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    const __nv_bfloat16* row_res_in = res_in + row * hidden;
    __nv_bfloat16* row_res_out = res_out + row * hidden;
    __nv_bfloat16* row_out = out + row * hidden;

    float sv[VPT];
    float wv[VPT];
    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        size_t i = (size_t)threadIdx.x + (size_t)u * (size_t)BLOCK;
        sv[u] = 0.f;
        wv[u] = 0.f;
        if (i < hidden) {
            sv[u] = __bfloat162float(row_x[i]) + __bfloat162float(row_res_in[i]);
            wv[u] = __bfloat162float(weight[i]);
        }
    }

    float local = 0.f;
    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        size_t i = (size_t)threadIdx.x + (size_t)u * (size_t)BLOCK;
        if (i < hidden) {
            float s = sv[u];
            __nv_bfloat16 sb = __float2bfloat16(s);
            row_res_out[i] = sb;
            sv[u] = __bfloat162float(sb);
            local += s * s;
        }
    }
    float sum =
        writeout_smem_tree_sum_is_the_stock_reduction_so_the_wide_256_rung_stays_bitwise<BLOCK>(
            local);
    float rms = rsqrtf(sum / (float)hidden + eps);

    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        size_t i = (size_t)threadIdx.x + (size_t)u * (size_t)BLOCK;
        if (i < hidden) row_out[i] = __float2bfloat16(sv[u] * rms * wv[u]);
    }
}

template <int BLOCK, int VPT>
static void rmsnorm_residual_writeout_m1_wide_launch_one(
    cudaStream_t s,
    dim3 grid,
    const __nv_bfloat16* x,
    const __nv_bfloat16* res_in,
    const __nv_bfloat16* weight,
    __nv_bfloat16* res_out,
    __nv_bfloat16* out,
    size_t hidden,
    float eps
) {
    rmsnorm_residual_writeout_m1_wide_kernel_bf16<BLOCK, VPT><<<grid, BLOCK, 0, s>>>(
        x, res_in, weight, res_out, out, hidden, eps);
}

static bool rmsnorm_residual_writeout_m1_wide_launch(
    cudaStream_t s,
    size_t batch,
    const __nv_bfloat16* x,
    const __nv_bfloat16* res_in,
    const __nv_bfloat16* weight,
    __nv_bfloat16* res_out,
    __nv_bfloat16* out,
    size_t hidden,
    float eps
) {
    int block = nvk_norm_m1_wide_block_for_row(nv_kernels_norm_m1_wide_block(), batch, hidden);
    if (block == kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape) return false;
    dim3 grid((unsigned)batch);
    if (block == 256) {
        rmsnorm_residual_writeout_m1_wide_launch_one<
            256, kNormM1WideVptForBlock256IsTwentyFourSoHiddenUpTo6144Stages>(
            s, grid, x, res_in, weight, res_out, out, hidden, eps);
    } else if (block == 512) {
        rmsnorm_residual_writeout_m1_wide_launch_one<
            512, kNormM1WideVptForBlock512IsTwelveSoHiddenUpTo6144Stages>(
            s, grid, x, res_in, weight, res_out, out, hidden, eps);
    } else {
        rmsnorm_residual_writeout_m1_wide_launch_one<
            1024, kNormM1WideVptForBlock1024IsEightSoHiddenUpTo8192Stages>(
            s, grid, x, res_in, weight, res_out, out, hidden, eps);
    }
    return true;
}

__device__ float
    g_norm_m1_split_writeout_partials_one_row_per_rung_because_the_arm_serves_batch_one_only
        [kNormM1SplitLadderRungs1To32IsSixPowersOfTwo]
        [kNormM1SplitCtasMustBePowersOfTwoUpTo32SoTheLadderStaysAWholeNumberOfWavesOnA188SmCard];
__device__ unsigned
    g_norm_m1_split_writeout_arrivals_per_rung_monotonic_so_no_launch_has_to_reset_it_and_one_rung_never_sees_two_grid_sizes
        [kNormM1SplitLadderRungs1To32IsSixPowersOfTwo];

template <int BLOCK, int SPLITS>
__global__ void rmsnorm_residual_writeout_m1_split_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ res_out,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    __shared__ float scratch[BLOCK];
    float local = 0.f;
    for (size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x; i < hidden;
         i += (size_t)BLOCK * SPLITS) {
        float xv = __bfloat162float(x[i]);
        float rv = __bfloat162float(res_in[i]);
        float s = xv + rv;
        res_out[i] = __float2bfloat16(s);
        local += s * s;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum =
        nvk_norm_m1_split_grid_fan_in_one_launch_because_a_second_dependent_launch_costs_more_than_the_norm_kernel_it_splits<
            SPLITS>(
            scratch[0],
            g_norm_m1_split_writeout_partials_one_row_per_rung_because_the_arm_serves_batch_one_only
                [nvk_norm_m1_split_rung(SPLITS)],
            &g_norm_m1_split_writeout_arrivals_per_rung_monotonic_so_no_launch_has_to_reset_it_and_one_rung_never_sees_two_grid_sizes
                [nvk_norm_m1_split_rung(SPLITS)]);
    float rms = rsqrtf(sum / (float)hidden + eps);

    for (size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x; i < hidden;
         i += (size_t)BLOCK * SPLITS) {
        float s = __bfloat162float(res_out[i]);
        float w = __bfloat162float(weight[i]);
        out[i] = __float2bfloat16(s * rms * w);
    }
}

static bool rmsnorm_residual_writeout_m1_split_launch(
    cudaStream_t s,
    size_t batch,
    const __nv_bfloat16* x,
    const __nv_bfloat16* res_in,
    const __nv_bfloat16* weight,
    __nv_bfloat16* res_out,
    __nv_bfloat16* out,
    size_t hidden,
    float eps
) {
    int ctas = nvk_norm_m1_split_ctas_for_row(nv_kernels_norm_m1_split_ctas(), batch, hidden);
    if (ctas == kNormM1SplitOffIsTheStockSingleCtaPerRowGrid) return false;
    constexpr int BLOCK = kNormM1SplitBlockIs256SoTheSplitOneRungIsBitwiseAgainstTheStockKernel;
#define NVK_RMSNORM_RESIDUAL_WRITEOUT_SPLIT_LAUNCH(N)                              \
    rmsnorm_residual_writeout_m1_split_kernel_bf16<BLOCK, N><<<N, BLOCK, 0, s>>>(  \
        x, res_in, weight, res_out, out, hidden, eps)
    NVK_NORM_M1_SPLIT_DISPATCH_THE_LADDER_IS_TEMPLATED_SO_THE_GRID_STRIDE_STAYS_COMPILE_TIME_AND_THE_ROW_LOOP_KEEPS_THE_STOCK_KERNELS_UNROLL(
        ctas, NVK_RMSNORM_RESIDUAL_WRITEOUT_SPLIT_LAUNCH)
#undef NVK_RMSNORM_RESIDUAL_WRITEOUT_SPLIT_LAUNCH
    return true;
}

template <int BLOCK>
__global__ void rmsnorm_residual_writeout_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ res_out,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    __shared__ float scratch[BLOCK];
    __shared__ float row_rms;

    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    const __nv_bfloat16* row_res_in = res_in + row * hidden;
    __nv_bfloat16* row_res_out = res_out + row * hidden;
    __nv_bfloat16* row_out = out + row * hidden;

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float xv = __bfloat162float(row_x[i]);
        float rv = __bfloat162float(row_res_in[i]);
        float s = xv + rv;
        row_res_out[i] = __float2bfloat16(s);
        local += s * s;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_rms = rsqrtf(scratch[0] / (float)hidden + eps);
    }
    __syncthreads();

    float rms = row_rms;

    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float s = __bfloat162float(row_res_out[i]);
        float w = __bfloat162float(weight[i]);
        float v = s * rms * w;
        row_out[i] = __float2bfloat16(v);
    }
}

extern "C" int nv_kernels_rmsnorm_residual_writeout_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* res_in,
    const uint16_t* weight,
    uint16_t* res_out,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    if (rmsnorm_residual_writeout_m1_split_launch(
            s, batch,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(res_in),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(res_out),
            reinterpret_cast<__nv_bfloat16*>(out), hidden, eps)) {
        return (int)cudaGetLastError();
    }
    if (rmsnorm_residual_writeout_m1_wide_launch(
            s, batch,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(res_in),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(res_out),
            reinterpret_cast<__nv_bfloat16*>(out), hidden, eps)) {
        return (int)cudaGetLastError();
    }
    rmsnorm_residual_writeout_kernel_bf16<BLOCK><<<(int)batch, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(res_in),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(res_out),
        reinterpret_cast<__nv_bfloat16*>(out),
        hidden, eps);
    return (int)cudaGetLastError();
}
