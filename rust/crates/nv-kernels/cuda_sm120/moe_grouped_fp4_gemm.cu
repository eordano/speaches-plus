
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <climits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "nv_kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

namespace nvk_moe_grouped_fp4_sm120 {

using namespace cute;

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
using ElementType     = cutlass::float_e2m1_t;
using ElementSFType   = cutlass::float_ue4m3_t;
using ElementA        = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB        = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC        = cutlass::bfloat16_t;
using ElementD        = ElementC;
using ElementAcc      = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

static constexpr int AlignmentA = 32;
static constexpr int AlignmentB = 32;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape  = Shape<_1, _1, _1>;

using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAcc, ElementC, ElementAcc>;

template <class MmaTileShapeT>
struct GemmCfg {
  using MmaTileShape = MmaTileShapeT;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc,
          ElementAcc, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutD*, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB,
          LayoutB*, AlignmentB, ElementAcc, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                            CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

using MmaTileShape       = Shape<_128, _128, _128>;
using MmaTileShapeDecode = Shape<_128, _128, _128>;

using PrefillCfg = GemmCfg<MmaTileShape>;
using DecodeCfg  = GemmCfg<MmaTileShapeDecode>;

using StrideA     = PrefillCfg::StrideA;
using StrideB     = PrefillCfg::StrideB;
using StrideC     = PrefillCfg::StrideC;
using StrideD     = PrefillCfg::StrideD;
using LayoutSFA   = PrefillCfg::LayoutSFA;
using LayoutSFB   = PrefillCfg::LayoutSFB;
using ScaleConfig = PrefillCfg::ScaleConfig;

using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;

__global__ void __get_group_gemm_starts(
    ElementType const** a_ptrs,
    ElementType const** b_ptrs,
    ElementD** out_ptrs,
    ElementSFType const** a_scales_ptrs,
    ElementSFType const** b_scales_ptrs,
    float const** alpha_ptrs,
    LayoutSFA* layout_sfa,
    LayoutSFB* layout_sfb,
    StrideA* a_strides,
    StrideB* b_strides,
    StrideC* c_strides,
    int64_t a_stride_val,
    int64_t b_stride_val,
    int64_t c_stride_val,
    ElementType* a_base,
    ElementType* b_base,
    ElementD* out_base,
    ElementSFType* a_scales_base,
    ElementSFType* b_scales_base,
    float* alphas_base,
    int32_t const* expert_offsets,
    int32_t const* sf_offsets,
    int32_t const* problem_sizes,
    int32_t const* active_expert_indices,
    int N, int K
) {
    int e = threadIdx.x;
    if (e >= gridDim.x * blockDim.x) return;

    int64_t expert_offset = static_cast<int64_t>(expert_offsets[e]);
    int64_t sf_offset     = static_cast<int64_t>(sf_offsets[e]);
    constexpr int64_t group_size = 16;
    int64_t m = static_cast<int64_t>(problem_sizes[e * 3]);
    int64_t n = static_cast<int64_t>(problem_sizes[e * 3 + 1]);
    int64_t k = static_cast<int64_t>(problem_sizes[e * 3 + 2]);

    int64_t half_k  = k / 2;
    int64_t group_k = k / group_size;

    int64_t group_k_pad = ((group_k + 3) / 4) * 4;
    int64_t n_pad       = ((n + 127) / 128) * 128;

    int64_t e_global = active_expert_indices ? (int64_t)active_expert_indices[e]
                                              : (int64_t)e;

    a_ptrs[e]        = a_base + expert_offset * half_k;

    b_ptrs[e]        = b_base + e_global * n * half_k;

    out_ptrs[e]      = out_base + expert_offset * n;

    a_scales_ptrs[e] = a_scales_base + sf_offset * group_k_pad;

    b_scales_ptrs[e] = b_scales_base + e_global * n_pad * group_k_pad;

    alpha_ptrs[e]    = alphas_base + e_global;

    a_strides[e] = cute::make_stride(a_stride_val, cute::_1{}, cute::_0{});
    b_strides[e] = cute::make_stride(b_stride_val, cute::_1{}, cute::_0{});
    c_strides[e] = cute::make_stride(c_stride_val, cute::_1{}, cute::_0{});

    layout_sfa[e] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                         static_cast<int>(k), 1));
    layout_sfb[e] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                         static_cast<int>(k), 1));
}

template <typename T>
T* take(uint8_t*& scratch, size_t& offset, int count) {

    offset = (offset + 127) & ~size_t{127};
    T* p = reinterpret_cast<T*>(scratch + offset);
    offset += sizeof(T) * static_cast<size_t>(count);
    return p;
}

template <class Cfg>
static int launch_grouped_fp4(
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
    static_assert(std::is_same<typename Cfg::LayoutSFA, LayoutSFA>::value,
                  "decode tile LayoutSFA must match reference");
    static_assert(std::is_same<typename Cfg::LayoutSFB, LayoutSFB>::value,
                  "decode tile LayoutSFB must match reference");
    static_assert(std::is_same<typename Cfg::StrideA, StrideA>::value,
                  "decode tile StrideA must match reference");
    static_assert(std::is_same<typename Cfg::StrideB, StrideB>::value,
                  "decode tile StrideB must match reference");
    static_assert(std::is_same<typename Cfg::StrideC, StrideC>::value,
                  "decode tile StrideC must match reference");

    using Gemm       = typename Cfg::Gemm;
    using GemmKernel = typename Cfg::GemmKernel;

    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (meta_scratch_bytes < 64 * 1024) {
        return -10;
    }
    uint8_t* scratch = static_cast<uint8_t*>(meta_scratch);
    size_t off = 0;
    auto a_ptrs        = take<ElementType const*>(scratch, off, num_experts);
    auto b_ptrs        = take<ElementType const*>(scratch, off, num_experts);
    auto out_ptrs      = take<ElementD*>(scratch, off, num_experts);
    auto a_scales_ptrs = take<ElementSFType const*>(scratch, off, num_experts);
    auto b_scales_ptrs = take<ElementSFType const*>(scratch, off, num_experts);
    auto alpha_ptrs    = take<float const*>(scratch, off, num_experts);
    auto layout_sfa    = take<LayoutSFA>(scratch, off, num_experts);
    auto layout_sfb    = take<LayoutSFB>(scratch, off, num_experts);
    auto a_strides     = take<StrideA>(scratch, off, num_experts);
    auto b_strides     = take<StrideB>(scratch, off, num_experts);
    auto c_strides     = take<StrideC>(scratch, off, num_experts);

    if (off > meta_scratch_bytes) return -11;

    __get_group_gemm_starts<<<1, num_experts, 0, s>>>(
        a_ptrs, b_ptrs, out_ptrs,
        a_scales_ptrs, b_scales_ptrs, alpha_ptrs,
        layout_sfa, layout_sfb,
        a_strides, b_strides, c_strides,
        a_row_stride_elems, b_row_stride_elems, c_row_stride_elems,
        const_cast<ElementType*>(reinterpret_cast<ElementType const*>(a_packed)),
        const_cast<ElementType*>(reinterpret_cast<ElementType const*>(b_packed)),
        reinterpret_cast<ElementD*>(d_bf16),
        const_cast<ElementSFType*>(reinterpret_cast<ElementSFType const*>(a_scales)),
        const_cast<ElementSFType*>(reinterpret_cast<ElementSFType const*>(b_scales)),
        const_cast<float*>(alphas),
        expert_offsets, sf_offsets, problem_sizes,
        active_expert_indices,
        N, K
    );
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return (int)err;
    }

    static std::unordered_map<int, int> sm_count_cache;
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = dev_id;
    if (sm_count_cache.find(dev_id) == sm_count_cache.end()) {
        sm_count_cache[dev_id] =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev_id);
    }
    hw_info.sm_count = std::min(sm_count_cache[dev_id], INT_MAX);

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    using RasterOrderField = std::remove_reference_t<decltype(scheduler.raster_order)>;
    scheduler.raster_order = RasterOrderField::AlongM;

    auto* problem_sizes_as_shapes =
        const_cast<UnderlyingProblemShape*>(
            reinterpret_cast<UnderlyingProblemShape const*>(problem_sizes));

    typename GemmKernel::MainloopArguments mainloop_args{
        a_ptrs,
        a_strides,
        b_ptrs,
        b_strides,
        a_scales_ptrs,
        layout_sfa,
        b_scales_ptrs,
        layout_sfb};

    typename GemmKernel::EpilogueArguments epi_args{
        {},
        nullptr,
        c_strides,
        out_ptrs,
        c_strides};
    auto& fusion_args = epi_args.thread;
    fusion_args.alpha_ptr_array = const_cast<float**>(alpha_ptrs);
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.beta = 0.0f;

    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, problem_sizes_as_shapes, nullptr},
        mainloop_args,
        epi_args,
        hw_info,
        scheduler};

    struct GemmCacheEntry {
        Gemm op;
        size_t need_ws;
    };
    static std::mutex g_gemm_cache_mu;
    static std::unordered_map<uint64_t, std::unique_ptr<GemmCacheEntry>> g_gemm_cache;

    uint64_t key = 1469598103934665603ull;
    auto mix = [&key](uint64_t v) {
        key ^= v;
        key *= 1099511628211ull;
    };
    mix(reinterpret_cast<uint64_t>(meta_scratch));
    mix(reinterpret_cast<uint64_t>(gemm_workspace));
    mix(reinterpret_cast<uint64_t>(problem_sizes));
    mix(reinterpret_cast<uint64_t>(a_packed));
    mix(reinterpret_cast<uint64_t>(b_packed));
    mix(reinterpret_cast<uint64_t>(d_bf16));
    mix(reinterpret_cast<uint64_t>(active_expert_indices));
    mix((uint64_t)(uint32_t)N | ((uint64_t)(uint32_t)K << 32));
    mix((uint64_t)(uint32_t)num_experts);
    mix((uint64_t)a_row_stride_elems);
    mix((uint64_t)b_row_stride_elems);
    mix((uint64_t)c_row_stride_elems);

    {
        std::lock_guard<std::mutex> lock(g_gemm_cache_mu);
        auto it = g_gemm_cache.find(key);
        if (it != g_gemm_cache.end()) {
            if (a_packed == nullptr && b_packed == nullptr && d_bf16 == nullptr) {
                if (required_workspace) *required_workspace = it->second->need_ws;
                return 0;
            }
            auto status = it->second->op.run(s);
            if (status != cutlass::Status::kSuccess) {
                std::fprintf(stderr,
                             "moe_grouped_fp4_gemm_sm120: cached run failed: %d\n",
                             (int)status);
                g_gemm_cache.erase(it);
                return -16;
            }
            return 0;
        }
    }

    Gemm gemm_op;
    size_t need_ws = Gemm::get_workspace_size(args);
    if (required_workspace) *required_workspace = need_ws;
    if (a_packed == nullptr && b_packed == nullptr && d_bf16 == nullptr) {

        return 0;
    }
    if (need_ws > gemm_workspace_bytes) {
        return -12;
    }

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "moe_grouped_fp4_gemm_sm120: can_implement failed: %d\n",
                     (int)status);
        return -13;
    }
    status = gemm_op.initialize(args, gemm_workspace, s);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "moe_grouped_fp4_gemm_sm120: initialize failed: %d\n",
                     (int)status);
        return -14;
    }
    status = gemm_op.run(s);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "moe_grouped_fp4_gemm_sm120: run failed: %d\n",
                     (int)status);
        return -15;
    }
    {
        std::lock_guard<std::mutex> lock(g_gemm_cache_mu);
        auto entry = std::make_unique<GemmCacheEntry>();
        entry->op = gemm_op;
        entry->need_ws = need_ws;
        g_gemm_cache.emplace(key, std::move(entry));
    }
    return 0;
}

}

using namespace nvk_moe_grouped_fp4_sm120;

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
    return launch_grouped_fp4<PrefillCfg>(
        stream, a_packed, a_scales, b_packed, b_scales, alphas, d_bf16,
        expert_offsets, sf_offsets, problem_sizes, active_expert_indices,
        N, K, num_experts, a_row_stride_elems, b_row_stride_elems,
        c_row_stride_elems, meta_scratch, meta_scratch_bytes, gemm_workspace,
        gemm_workspace_bytes, required_workspace);
}

extern "C" int nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode(
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
    return launch_grouped_fp4<DecodeCfg>(
        stream, a_packed, a_scales, b_packed, b_scales, alphas, d_bf16,
        expert_offsets, sf_offsets, problem_sizes, active_expert_indices,
        N, K, num_experts, a_row_stride_elems, b_row_stride_elems,
        c_row_stride_elems, meta_scratch, meta_scratch_bytes, gemm_workspace,
        gemm_workspace_bytes, required_workspace);
}
