#include <cuda_runtime.h>
#include <stdint.h>
#include "nv_kernels.h"

extern "C" int nv_kernels_l2_info(int* l2_bytes, int* max_persist_bytes, int* max_window_bytes) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return (int)e;
    e = cudaDeviceGetAttribute(l2_bytes, cudaDevAttrL2CacheSize, dev);
    if (e != cudaSuccess) return (int)e;
    e = cudaDeviceGetAttribute(max_persist_bytes, cudaDevAttrMaxPersistingL2CacheSize, dev);
    if (e != cudaSuccess) return (int)e;
    return (int)cudaDeviceGetAttribute(max_window_bytes, cudaDevAttrMaxAccessPolicyWindowSize, dev);
}

extern "C" int nv_kernels_l2_persist_limit(size_t bytes) {
    return (int)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);
}

extern "C" int nv_kernels_stream_access_policy_window(
    void* stream,
    const void* base,
    size_t bytes,
    float hit_ratio
) {
    cudaStreamAttrValue v;
    v.accessPolicyWindow.base_ptr = const_cast<void*>(base);
    v.accessPolicyWindow.num_bytes = bytes;
    v.accessPolicyWindow.hitRatio = hit_ratio;
    v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    return (int)cudaStreamSetAttribute(
        (cudaStream_t)stream, cudaStreamAttributeAccessPolicyWindow, &v);
}
