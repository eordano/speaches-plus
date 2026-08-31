#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include "nv_kernels.h"
#include "nvk_norm_m1.cuh"
#include "nvk_pdl.cuh"

template <typename T>
__device__ inline float to_f32(T x);

template <>
__device__ inline float to_f32<float>(float x) { return x; }

template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_f32(float x);

template <>
__device__ inline float from_f32<float>(float x) { return x; }

template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <int BLOCK>
__device__ inline float block_sum(float v) {
    constexpr int kWarp = 32;
    constexpr int kWarps = BLOCK / kWarp;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

template <typename T, int BLOCK>
__global__ void rmsnorm_kernel(const T* __restrict__ x,
                               const T* __restrict__ weight,
                               T* __restrict__ y,
                               size_t hidden,
                               float eps) {
    size_t row = blockIdx.x;
    const T* row_x = x + row * hidden;
    T* row_y = y + row * hidden;

    NVK_PDL_PROLOG();

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float v = to_f32<T>(row_x[i]);
        local += v * v;
    }
    float sum = block_sum<BLOCK>(local);
    float rms = rsqrtf(sum / (float)hidden + eps);

    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float v = to_f32<T>(row_x[i]) * rms * to_f32<T>(weight[i]);
        row_y[i] = from_f32<T>(v);
    }

    NVK_PDL_EPILOG();
}

static int g_norm_m1_wide_block_selected_once_at_first_launch_so_the_default_path_pays_no_getenv = -2;

extern "C" int nv_kernels_norm_m1_wide_block() {
    if (g_norm_m1_wide_block_selected_once_at_first_launch_so_the_default_path_pays_no_getenv == -2) {
        const char* e = getenv("NV_NORM_M1_WIDE");
        int p = e ? atoi(e) : 0;
        g_norm_m1_wide_block_selected_once_at_first_launch_so_the_default_path_pays_no_getenv =
            (nvk_norm_m1_wide_vpt(p) > 0)
                ? p
                : kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    }
    return g_norm_m1_wide_block_selected_once_at_first_launch_so_the_default_path_pays_no_getenv;
}

extern "C" int nv_kernels_norm_m1_wide_force(int block) {
    g_norm_m1_wide_block_selected_once_at_first_launch_so_the_default_path_pays_no_getenv =
        (nvk_norm_m1_wide_vpt(block) > 0)
            ? block
            : kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    return g_norm_m1_wide_block_selected_once_at_first_launch_so_the_default_path_pays_no_getenv;
}

static int g_norm_m1_split_ctas_selected_once_at_first_launch_so_the_default_path_pays_no_getenv = -2;

static inline int nvk_norm_m1_split_cap_clamp(int p) {
    if (p < 1 ||
        p > kNormM1SplitCtasMustBePowersOfTwoUpTo32SoTheLadderStaysAWholeNumberOfWavesOnA188SmCard ||
        (p & (p - 1)) != 0) {
        return kNormM1SplitOffIsTheStockSingleCtaPerRowGrid;
    }
    return p;
}

extern "C" int nv_kernels_norm_m1_split_ctas() {
    if (g_norm_m1_split_ctas_selected_once_at_first_launch_so_the_default_path_pays_no_getenv == -2) {
        const char* e = getenv("NV_NORM_M1_SPLIT");
        g_norm_m1_split_ctas_selected_once_at_first_launch_so_the_default_path_pays_no_getenv =
            nvk_norm_m1_split_cap_clamp(e ? atoi(e) : 0);
    }
    return g_norm_m1_split_ctas_selected_once_at_first_launch_so_the_default_path_pays_no_getenv;
}

extern "C" int nv_kernels_norm_m1_split_force(int ctas) {
    g_norm_m1_split_ctas_selected_once_at_first_launch_so_the_default_path_pays_no_getenv =
        nvk_norm_m1_split_cap_clamp(ctas);
    return g_norm_m1_split_ctas_selected_once_at_first_launch_so_the_default_path_pays_no_getenv;
}

extern "C" int nv_kernels_norm_m1_wide_block_for_launch(size_t batch, size_t hidden) {
    return nvk_norm_m1_wide_block_for_row(nv_kernels_norm_m1_wide_block(), batch, hidden);
}

extern "C" int nv_kernels_norm_m1_split_ctas_for_launch(size_t batch, size_t hidden) {
    return nvk_norm_m1_split_ctas_for_row(nv_kernels_norm_m1_split_ctas(), batch, hidden);
}

template <typename T, int BLOCK, int VPT>
__global__ void rmsnorm_m1_wide_kernel(const T* __restrict__ x,
                                       const T* __restrict__ weight,
                                       T* __restrict__ y,
                                       size_t hidden,
                                       float eps) {
    size_t row = blockIdx.x;
    const T* row_x = x + row * hidden;
    T* row_y = y + row * hidden;

    NVK_PDL_PROLOG();

    float xv[VPT];
    float wv[VPT];
    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        size_t i = (size_t)threadIdx.x + (size_t)u * (size_t)BLOCK;
        xv[u] = 0.f;
        wv[u] = 0.f;
        if (i < hidden) {
            xv[u] = to_f32<T>(row_x[i]);
            wv[u] = to_f32<T>(weight[i]);
        }
    }

    float local = 0.f;
    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        size_t i = (size_t)threadIdx.x + (size_t)u * (size_t)BLOCK;
        if (i < hidden) local += xv[u] * xv[u];
    }
    float sum = block_sum<BLOCK>(local);
    float rms = rsqrtf(sum / (float)hidden + eps);

    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        size_t i = (size_t)threadIdx.x + (size_t)u * (size_t)BLOCK;
        if (i < hidden) row_y[i] = from_f32<T>(xv[u] * rms * wv[u]);
    }

    NVK_PDL_EPILOG();
}

template <typename T, int BLOCK, int VPT>
static void rmsnorm_m1_wide_launch_one(cudaStream_t s,
                                       dim3 grid,
                                       const T* x,
                                       const T* weight,
                                       T* y,
                                       size_t hidden,
                                       float eps) {
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(&cfg, rmsnorm_m1_wide_kernel<T, BLOCK, VPT>, x, weight, y, hidden, eps);
    } else {
        rmsnorm_m1_wide_kernel<T, BLOCK, VPT><<<grid, BLOCK, 0, s>>>(x, weight, y, hidden, eps);
    }
}

template <typename T>
static bool rmsnorm_m1_wide_launch(cudaStream_t s,
                                   size_t batch,
                                   const T* x,
                                   const T* weight,
                                   T* y,
                                   size_t hidden,
                                   float eps) {
    int block = nvk_norm_m1_wide_block_for_row(nv_kernels_norm_m1_wide_block(), batch, hidden);
    if (block == kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape) return false;
    dim3 grid((unsigned)batch);
    if (block == 256) {
        rmsnorm_m1_wide_launch_one<T, 256,
                                   kNormM1WideVptForBlock256IsTwentyFourSoHiddenUpTo6144Stages>(
            s, grid, x, weight, y, hidden, eps);
    } else if (block == 512) {
        rmsnorm_m1_wide_launch_one<T, 512,
                                   kNormM1WideVptForBlock512IsTwelveSoHiddenUpTo6144Stages>(
            s, grid, x, weight, y, hidden, eps);
    } else {
        rmsnorm_m1_wide_launch_one<T, 1024,
                                   kNormM1WideVptForBlock1024IsEightSoHiddenUpTo8192Stages>(
            s, grid, x, weight, y, hidden, eps);
    }
    return true;
}

__device__ float g_norm_m1_split_partials_one_row_per_rung_because_the_arm_serves_batch_one_only
    [kNormM1SplitLadderRungs1To32IsSixPowersOfTwo]
    [kNormM1SplitCtasMustBePowersOfTwoUpTo32SoTheLadderStaysAWholeNumberOfWavesOnA188SmCard];
__device__ unsigned
    g_norm_m1_split_arrivals_per_rung_monotonic_so_no_launch_has_to_reset_it_and_one_rung_never_sees_two_grid_sizes
        [kNormM1SplitLadderRungs1To32IsSixPowersOfTwo];

template <typename T, int BLOCK, int SPLITS>
__global__ void rmsnorm_m1_split_kernel(const T* __restrict__ x,
                                        const T* __restrict__ weight,
                                        T* __restrict__ y,
                                        size_t hidden,
                                        float eps) {
    float local = 0.f;
    for (size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x; i < hidden;
         i += (size_t)BLOCK * SPLITS) {
        float v = to_f32<T>(x[i]);
        local += v * v;
    }
    float sum =
        nvk_norm_m1_split_grid_fan_in_one_launch_because_a_second_dependent_launch_costs_more_than_the_norm_kernel_it_splits<
            SPLITS>(
            block_sum<BLOCK>(local),
            g_norm_m1_split_partials_one_row_per_rung_because_the_arm_serves_batch_one_only
                [nvk_norm_m1_split_rung(SPLITS)],
            &g_norm_m1_split_arrivals_per_rung_monotonic_so_no_launch_has_to_reset_it_and_one_rung_never_sees_two_grid_sizes
                [nvk_norm_m1_split_rung(SPLITS)]);
    float rms = rsqrtf(sum / (float)hidden + eps);

    for (size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x; i < hidden;
         i += (size_t)BLOCK * SPLITS) {
        float v = to_f32<T>(x[i]) * rms * to_f32<T>(weight[i]);
        y[i] = from_f32<T>(v);
    }
}

template <typename T>
static bool rmsnorm_m1_split_launch(cudaStream_t s,
                                    size_t batch,
                                    const T* x,
                                    const T* weight,
                                    T* y,
                                    size_t hidden,
                                    float eps) {
    int ctas = nvk_norm_m1_split_ctas_for_row(nv_kernels_norm_m1_split_ctas(), batch, hidden);
    if (ctas == kNormM1SplitOffIsTheStockSingleCtaPerRowGrid) return false;
    constexpr int BLOCK = kNormM1SplitBlockIs256SoTheSplitOneRungIsBitwiseAgainstTheStockKernel;
#define NVK_RMSNORM_SPLIT_LAUNCH(N) \
    rmsnorm_m1_split_kernel<T, BLOCK, N><<<N, BLOCK, 0, s>>>(x, weight, y, hidden, eps)
    NVK_NORM_M1_SPLIT_DISPATCH_THE_LADDER_IS_TEMPLATED_SO_THE_GRID_STRIDE_STAYS_COMPILE_TIME_AND_THE_ROW_LOOP_KEEPS_THE_STOCK_KERNELS_UNROLL(
        ctas, NVK_RMSNORM_SPLIT_LAUNCH)
#undef NVK_RMSNORM_SPLIT_LAUNCH
    return true;
}

extern "C" int nv_kernels_rmsnorm_f32(
    void* stream,
    const float* x,
    const float* weight,
    float* y,
    size_t batch,
    size_t hidden,
    float eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    if (rmsnorm_m1_split_launch<float>(s, batch, x, weight, y, hidden, eps)) {
        return (int)cudaGetLastError();
    }
    if (rmsnorm_m1_wide_launch<float>(s, batch, x, weight, y, hidden, eps)) {
        return (int)cudaGetLastError();
    }
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)batch), dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(&cfg, rmsnorm_kernel<float, BLOCK>, x, weight, y, hidden, eps);
    } else {
        rmsnorm_kernel<float, BLOCK><<<(int)batch, BLOCK, 0, s>>>(x, weight, y, hidden, eps);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rmsnorm_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* y,
    size_t batch,
    size_t hidden,
    float eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    if (rmsnorm_m1_split_launch<__nv_bfloat16>(
            s, batch,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(y), hidden, eps)) {
        return (int)cudaGetLastError();
    }
    if (rmsnorm_m1_wide_launch<__nv_bfloat16>(
            s, batch,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(y), hidden, eps)) {
        return (int)cudaGetLastError();
    }
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)batch), dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(
            &cfg, rmsnorm_kernel<__nv_bfloat16, BLOCK>,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(y),
            hidden,
            eps);
    } else {
        rmsnorm_kernel<__nv_bfloat16, BLOCK><<<(int)batch, BLOCK, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(y),
            hidden,
            eps);
    }
    return (int)cudaGetLastError();
}
