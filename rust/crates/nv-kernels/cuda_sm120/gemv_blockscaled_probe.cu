
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"

namespace {

using ElementA = cutlass::float_e2m1_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::float_e2m1_t;
using ElementD = cutlass::float_e2m1_t;
using ElementSFA = cutlass::float_e4m3_t;
using ElementSFB = cutlass::float_e4m3_t;
using ElementSFD = cutlass::float_e4m3_t;
using LayoutOutput = cutlass::layout::ColumnMajor;
using LayoutSFD = cutlass::layout::ColumnMajor;

using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int kVectorSize = 16;
static constexpr int kElementsPerAccess =
    128 / cutlass::sizeof_bits<ElementA>::value;

using ThreadShape = cutlass::gemm::GemmShape<16, 8>;

using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,
    ElementSFD,
    LayoutOutput,
    LayoutSFD
>;

using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA, LayoutA,
    ElementB,
    ElementD,
    ElementAccumulatorMainloop,
    EpilogueOp,
    kElementsPerAccess,
    0,
    0,
    ElementSFA,
    ElementSFB,
    kVectorSize
>;

using GemvOp = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

__device__ int probe_size() {
    return static_cast<int>(sizeof(GemvOp));
}

}

extern "C" int nv_kernels_gemv_blockscaled_probe(int* out_size) {
    if (out_size) *out_size = static_cast<int>(reinterpret_cast<uintptr_t>(&probe_size) & 0x7fffffff);
    return 0;
}
