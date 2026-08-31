#pragma once

#include <cstddef>

static constexpr int kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape = 0;
static constexpr int kNormM1WideVptForBlock256IsTwentyFourSoHiddenUpTo6144Stages = 24;
static constexpr int kNormM1WideVptForBlock512IsTwelveSoHiddenUpTo6144Stages = 12;
static constexpr int kNormM1WideVptForBlock1024IsEightSoHiddenUpTo8192Stages = 8;

static constexpr size_t
    kNormM1ArmsServeBatchOneOnlyBecauseAMultiRowNormAlreadyHasOneCtaPerRowAndWideningItMeasuredPlus12PercentOnGemma4HeadDimNormsAnd2Point6xOnQwen35PrefillNorm =
        1;

static constexpr int kNormM1SplitOffIsTheStockSingleCtaPerRowGrid = 0;
static constexpr int kNormM1SplitBlockIs256SoTheSplitOneRungIsBitwiseAgainstTheStockKernel = 256;
static constexpr int
    kNormM1SplitCtasMustBePowersOfTwoUpTo32SoTheLadderStaysAWholeNumberOfWavesOnA188SmCard = 32;
static constexpr int kNormM1SplitLadderRungs1To32IsSixPowersOfTwo = 6;

static constexpr int nvk_norm_m1_split_rung(int ctas) {
    return ctas <= 1 ? 0 : 1 + nvk_norm_m1_split_rung(ctas >> 1);
}

#ifdef __CUDACC__
template <int SPLITS>
__device__ inline float
nvk_norm_m1_split_grid_fan_in_one_launch_because_a_second_dependent_launch_costs_more_than_the_norm_kernel_it_splits(
    float block_partial,
    float* partials_for_this_rung,
    unsigned* arrivals_for_this_rung
) {
    __shared__ float row_sum;
    if (SPLITS == 1) {
        if (threadIdx.x == 0) row_sum = block_partial;
    } else if (threadIdx.x == 0) {
        partials_for_this_rung[blockIdx.x] = block_partial;
        __threadfence();
        unsigned old = atomicAdd(arrivals_for_this_rung, 1u);
        unsigned target = old - (old & (unsigned)(SPLITS - 1)) + (unsigned)SPLITS;
        volatile unsigned* arrivals = arrivals_for_this_rung;
        while ((int)(*arrivals - target) < 0) __nanosleep(64);
        __threadfence();
        volatile float* partials = partials_for_this_rung;
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < SPLITS; ++j) s += partials[j];
        row_sum = s;
    }
    __syncthreads();
    return row_sum;
}

#define NVK_NORM_M1_SPLIT_DISPATCH_THE_LADDER_IS_TEMPLATED_SO_THE_GRID_STRIDE_STAYS_COMPILE_TIME_AND_THE_ROW_LOOP_KEEPS_THE_STOCK_KERNELS_UNROLL( \
    ctas, LAUNCH)                                                                              \
    switch (ctas) {                                                                            \
        case 1: LAUNCH(1); break;                                                              \
        case 2: LAUNCH(2); break;                                                               \
        case 4: LAUNCH(4); break;                                                               \
        case 8: LAUNCH(8); break;                                                               \
        case 16: LAUNCH(16); break;                                                             \
        default: LAUNCH(32); break;                                                             \
    }
#endif

static inline int nvk_norm_m1_wide_vpt(int block) {
    switch (block) {
        case 256: return kNormM1WideVptForBlock256IsTwentyFourSoHiddenUpTo6144Stages;
        case 512: return kNormM1WideVptForBlock512IsTwelveSoHiddenUpTo6144Stages;
        case 1024: return kNormM1WideVptForBlock1024IsEightSoHiddenUpTo8192Stages;
        default: return 0;
    }
}

extern "C" int nv_kernels_norm_m1_wide_block();
extern "C" int nv_kernels_norm_m1_wide_force(int block);
extern "C" int nv_kernels_norm_m1_split_ctas();
extern "C" int nv_kernels_norm_m1_split_force(int ctas);

static inline int nvk_norm_m1_wide_block_for_row(int cap, size_t batch, size_t hidden) {
    if (batch >
        kNormM1ArmsServeBatchOneOnlyBecauseAMultiRowNormAlreadyHasOneCtaPerRowAndWideningItMeasuredPlus12PercentOnGemma4HeadDimNormsAnd2Point6xOnQwen35PrefillNorm) {
        return kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    }
    if (nvk_norm_m1_wide_vpt(cap) <= 0) return kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    int chosen = kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    const int ladder[3] = {256, 512, 1024};
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        int c = ladder[r];
        if (c <= cap && (size_t)c <= hidden) chosen = c;
    }
    if (chosen == kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape) {
        return kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    }
    if (hidden > (size_t)chosen * (size_t)nvk_norm_m1_wide_vpt(chosen)) {
        return kNormM1WideOffIsTheStock256ThreadSingleCtaPerRowShape;
    }
    return chosen;
}

static inline int nvk_norm_m1_split_ctas_for_row(int cap, size_t batch, size_t hidden) {
    if (batch >
        kNormM1ArmsServeBatchOneOnlyBecauseAMultiRowNormAlreadyHasOneCtaPerRowAndWideningItMeasuredPlus12PercentOnGemma4HeadDimNormsAnd2Point6xOnQwen35PrefillNorm) {
        return kNormM1SplitOffIsTheStockSingleCtaPerRowGrid;
    }
    if (cap < 1 ||
        cap > kNormM1SplitCtasMustBePowersOfTwoUpTo32SoTheLadderStaysAWholeNumberOfWavesOnA188SmCard) {
        return kNormM1SplitOffIsTheStockSingleCtaPerRowGrid;
    }
    if ((cap & (cap - 1)) != 0) return kNormM1SplitOffIsTheStockSingleCtaPerRowGrid;
    size_t chunks =
        (hidden + (size_t)kNormM1SplitBlockIs256SoTheSplitOneRungIsBitwiseAgainstTheStockKernel - 1) /
        (size_t)kNormM1SplitBlockIs256SoTheSplitOneRungIsBitwiseAgainstTheStockKernel;
    int n = cap;
    while (n > 1 && (size_t)n > chunks) n >>= 1;
    return n;
}
