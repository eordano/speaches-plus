#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stddef.h>

#include "nv_kernels.h"

namespace nvk_moe_grouped_fp4_rocm {

using ElementPacked = uint8_t;
using ElementSFType = uint8_t;
using ElementD      = uint16_t;

constexpr int kNvfp4BlockSize = 16;

constexpr int kRocmNotImplemented = -1001;

struct GroupStride {
    int64_t major;
    int64_t minor;
    int64_t batch;
};

__global__ void get_group_gemm_starts(
    ElementPacked const** a_ptrs,
    ElementPacked const** b_ptrs,
    ElementD** out_ptrs,
    ElementSFType const** a_scales_ptrs,
    ElementSFType const** b_scales_ptrs,
    float const** alpha_ptrs,
    GroupStride* a_strides,
    GroupStride* b_strides,
    GroupStride* c_strides,
    int64_t a_stride_val,
    int64_t b_stride_val,
    int64_t c_stride_val,
    ElementPacked* a_base,
    ElementPacked* b_base,
    ElementD* out_base,
    ElementSFType* a_scales_base,
    ElementSFType* b_scales_base,
    float* alphas_base,
    int32_t const* expert_offsets,
    int32_t const* sf_offsets,
    int32_t const* problem_sizes,
    int32_t const* active_expert_indices,
    int num_experts,
    int N, int K
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int64_t expert_offset = static_cast<int64_t>(expert_offsets[e]);
    int64_t sf_offset     = static_cast<int64_t>(sf_offsets[e]);
    constexpr int64_t group_size = kNvfp4BlockSize;
    int64_t n = static_cast<int64_t>(problem_sizes[e * 3 + 1]);
    int64_t k = static_cast<int64_t>(problem_sizes[e * 3 + 2]);

    int64_t half_k  = k / 2;
    int64_t group_k = k / group_size;

    int64_t e_global = active_expert_indices ? (int64_t)active_expert_indices[e]
                                             : (int64_t)e;

    a_ptrs[e]        = a_base + expert_offset * half_k;

    b_ptrs[e]        = b_base + e_global * n * half_k;

    out_ptrs[e]      = out_base + expert_offset * n;

    a_scales_ptrs[e] = a_scales_base + sf_offset * group_k;

    b_scales_ptrs[e] = b_scales_base + e_global * n * group_k;

    alpha_ptrs[e]    = alphas_base + e_global;

    a_strides[e] = GroupStride{a_stride_val, 1, 0};
    b_strides[e] = GroupStride{b_stride_val, 1, 0};
    c_strides[e] = GroupStride{c_stride_val, 1, 0};
}

}

using namespace nvk_moe_grouped_fp4_rocm;

extern "C" int nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16(
    void* stream,
    const void* a_packed,
    const void* a_scales,
    const void* b_packed,
    const void* b_scales,
    const float* alphas,
    void* d_bf16,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    const int32_t* active_expert_indices,
    int N,
    int K,
    int num_experts,
    int64_t a_row_stride_elems,
    int64_t b_row_stride_elems,
    int64_t c_row_stride_elems,
    void* meta_scratch,
    size_t meta_scratch_bytes,
    void* gemm_workspace,
    size_t gemm_workspace_bytes,
    size_t* required_workspace
) {
    (void)stream;
    (void)a_packed;
    (void)a_scales;
    (void)b_packed;
    (void)b_scales;
    (void)alphas;
    (void)d_bf16;
    (void)expert_offsets;
    (void)sf_offsets;
    (void)problem_sizes;
    (void)active_expert_indices;
    (void)N;
    (void)K;
    (void)num_experts;
    (void)a_row_stride_elems;
    (void)b_row_stride_elems;
    (void)c_row_stride_elems;
    (void)meta_scratch;
    (void)meta_scratch_bytes;
    (void)gemm_workspace;
    (void)gemm_workspace_bytes;

    if (required_workspace) *required_workspace = 0;
    return kRocmNotImplemented;
}
