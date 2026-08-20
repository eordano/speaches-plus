
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nv_kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "flashinfer/math.cuh"

extern "C" int nv_kernels_cutlass_flashinfer_probe(int* out_cutlass_status,
                                                   int* out_flashinfer_max_e2m1_x32) {
    *out_cutlass_status = static_cast<int>(cutlass::Status::kSuccess);

    float v = 6.0f * 32.0f;
    *out_flashinfer_max_e2m1_x32 = static_cast<int>(v);
    return 0;
}
