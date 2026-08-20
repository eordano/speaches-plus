#include <hip/hip_runtime.h>
#include "nv_kernels.h"

namespace {
constexpr int kCutlassStatusSuccessEquivalent = 0;
constexpr int kFlashinferMaxE2m1TimesBlock32  = 192;
}

extern "C" int nv_kernels_cutlass_flashinfer_probe(int* out_cutlass_status,
                                                   int* out_flashinfer_max_e2m1_x32) {
    if (out_cutlass_status) {
        *out_cutlass_status = kCutlassStatusSuccessEquivalent;
    }
    if (out_flashinfer_max_e2m1_x32) {
        *out_flashinfer_max_e2m1_x32 = kFlashinferMaxE2m1TimesBlock32;
    }
    return 0;
}
