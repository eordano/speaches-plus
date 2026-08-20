
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "nv_kernels.h"

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"

namespace flashinfer {
namespace gemm {

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM)

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 1, 1, 1, _1SM)

}
}

extern "C" int nv_kernels_cutlass_fp4_gemm_sm120_bf16(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
) {
    using namespace flashinfer::gemm;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    try {
        CutlassGemmConfig cfg{};
        size_t needed = genericFp4GemmKernelLauncher<
            __nv_bfloat16,
            cute::Int<128>, cute::Int<128>, cute::Int<128>,
            cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM
        >(
            d_bf16,
            a_fp4, b_fp4,
            a_sf, b_sf,
            global_sf,
            m, n, k,1,
            cfg,
            static_cast<char*>(workspace), workspace_bytes,
            s,
nullptr
        );
        if (required_workspace) *required_workspace = needed;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nv_kernels_cutlass_fp4_gemm_sm120_bf16: %s\n", e.what());
        return -1;
    } catch (...) {
        return -2;
    }
}

extern "C" int nv_kernels_cutlass_fp4_gemm_sm120_bf16_streamk(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
) {
    using namespace flashinfer::gemm;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    try {
        CutlassGemmConfig cfg{};
        size_t needed = genericFp4GemmKernelLauncherStreamK<
            __nv_bfloat16,
            cute::Int<128>, cute::Int<128>, cute::Int<128>,
            cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM
        >(
            d_bf16,
            a_fp4, b_fp4,
            a_sf, b_sf,
            global_sf,
            m, n, k,1,
            cfg,
            static_cast<char*>(workspace), workspace_bytes,
            s,
nullptr
        );
        if (required_workspace) *required_workspace = needed;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nv_kernels_cutlass_fp4_gemm_sm120_bf16_streamk: %s\n", e.what());
        return -1;
    } catch (...) {
        return -2;
    }
}

namespace {
template <typename Gemm>
size_t run_fp4_streamk_splits(void* D, const void* A, const void* B, const void* input_sf,
                              const void* weight_sf, const float* global_sf, int m, int n, int k,
                              int splits, char* workspace, size_t workspace_bytes,
                              cudaStream_t stream) {
    Gemm gemm;
    auto args = flashinfer::gemm::prepareGemmArgsImpl<Gemm>(D, A, B, input_sf, weight_sf, global_sf,
                                                            m, n, k, 1);
    args.scheduler.splits = splits;
    if (gemm.get_workspace_size(args) > workspace_bytes) {
        throw std::runtime_error("[FP4 gemm splits] workspace insufficient");
    }
    auto ci = gemm.can_implement(args);
    if (ci != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("[FP4 gemm splits] can_implement: ") +
                                 cutlass::cutlassGetStatusString(ci));
    }
    auto is = gemm.initialize(args, workspace, stream);
    if (is != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("[FP4 gemm splits] initialize: ") +
                                 cutlass::cutlassGetStatusString(is));
    }
    auto rs = gemm.run(args, workspace, stream, nullptr, true);
    if (rs != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("[FP4 gemm splits] run: ") +
                                 cutlass::cutlassGetStatusString(rs));
    }
    return gemm.get_workspace_size(args);
}
}

extern "C" int nv_kernels_cutlass_fp4_gemm_sm120_bf16_tiled(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    int tile,
    int stream_k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
) {
    using namespace flashinfer::gemm;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    try {
        CutlassGemmConfig cfg{};
        size_t needed = 0;
        char* ws = static_cast<char*>(workspace);
        if (tile == 0 && stream_k >= 2) {
            needed = run_fp4_streamk_splits<Fp4Gemm___nv_bfloat16_128_128_128_StreamK>(
                d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, stream_k, ws,
                workspace_bytes, s);
            if (required_workspace) *required_workspace = needed;
            return 0;
        }
        switch (tile * 2 + (stream_k ? 1 : 0)) {
            case 0:
                needed = genericFp4GemmKernelLauncher<
                    __nv_bfloat16, cute::Int<128>, cute::Int<128>, cute::Int<128>,
                    cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
                    d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, 1, cfg,
                    ws, workspace_bytes, s, nullptr);
                break;
            case 1:
                needed = genericFp4GemmKernelLauncherStreamK<
                    __nv_bfloat16, cute::Int<128>, cute::Int<128>, cute::Int<128>,
                    cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
                    d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, 1, cfg,
                    ws, workspace_bytes, s, nullptr);
                break;
            case 2:
                needed = genericFp4GemmKernelLauncher<
                    __nv_bfloat16, cute::Int<128>, cute::Int<128>, cute::Int<256>,
                    cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
                    d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, 1, cfg,
                    ws, workspace_bytes, s, nullptr);
                break;
            case 3:
                needed = genericFp4GemmKernelLauncherStreamK<
                    __nv_bfloat16, cute::Int<128>, cute::Int<128>, cute::Int<256>,
                    cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
                    d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, 1, cfg,
                    ws, workspace_bytes, s, nullptr);
                break;
            case 4:
                needed = genericFp4GemmKernelLauncher<
                    __nv_bfloat16, cute::Int<128>, cute::Int<256>, cute::Int<128>,
                    cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
                    d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, 1, cfg,
                    ws, workspace_bytes, s, nullptr);
                break;
            case 5:
                needed = genericFp4GemmKernelLauncherStreamK<
                    __nv_bfloat16, cute::Int<128>, cute::Int<256>, cute::Int<128>,
                    cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
                    d_bf16, a_fp4, b_fp4, a_sf, b_sf, global_sf, m, n, k, 1, cfg,
                    ws, workspace_bytes, s, nullptr);
                break;
            default:
                return -3;
        }
        if (required_workspace) *required_workspace = needed;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nv_kernels_cutlass_fp4_gemm_sm120_bf16_tiled(tile=%d sk=%d): %s\n",
                     tile, stream_k, e.what());
        return -1;
    } catch (...) {
        return -2;
    }
}
