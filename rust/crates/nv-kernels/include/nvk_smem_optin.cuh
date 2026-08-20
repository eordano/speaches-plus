#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <mutex>

struct DynamicSmemOptin {
    std::mutex serialize_raises;
    std::atomic<size_t> granted{0};
};

static int raise_dynamic_smem_optin_never_lowering_it(
    DynamicSmemOptin& state,
    const void* kernel,
    size_t smem
) {
    if (smem <= state.granted.load(std::memory_order_acquire)) return 0;
    std::lock_guard<std::mutex> held(state.serialize_raises);
    if (smem <= state.granted.load(std::memory_order_relaxed)) return 0;
    cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return -3;
    state.granted.store(smem, std::memory_order_release);
    return 0;
}

static int nvk_max_dynamic_smem_optin() {
    static const int limit = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
            cudaSuccess) {
            return 0;
        }
        return v;
    }();
    return limit;
}
