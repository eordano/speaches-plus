#include <cuda_runtime.h>
#include <stdint.h>

namespace batched_state_copy {

constexpr int kThreads256 = 256;
constexpr int kBlocksPerDesc32SplitsALargeStateCopyAcrossEnoughSmsToUseMemoryBandwidthNotTheCopyEngine = 32;

__global__ void __launch_bounds__(kThreads256)
batched_copy_u32_kernel(
    const uint64_t* __restrict__ dsts,
    const uint64_t* __restrict__ srcs,
    const uint32_t* __restrict__ nwords,
    int blocks_per_desc
) {
    const int d = blockIdx.x / blocks_per_desc;
    const int sub = blockIdx.x - d * blocks_per_desc;
    const uint32_t n = nwords[d];
    const uint32_t n4 = n >> 2;
    const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(srcs[d]);
    uint4* __restrict__ t4 = reinterpret_cast<uint4*>(dsts[d]);
    for (uint32_t i = (uint32_t)sub * kThreads256 + threadIdx.x; i < n4; i += (uint32_t)blocks_per_desc * kThreads256) {
        t4[i] = s4[i];
    }
    if (sub == 0 && threadIdx.x == 0) {
        for (uint32_t i = n4 << 2; i < n; ++i)
            reinterpret_cast<uint32_t*>(dsts[d])[i] = reinterpret_cast<const uint32_t*>(srcs[d])[i];
    }
}

}

extern "C" int nv_kernels_batched_copy_u32(
    void* stream,
    const uint64_t* dsts,
    const uint64_t* srcs,
    const uint32_t* nwords,
    int count
) {
    using namespace batched_state_copy;
    if (count <= 0) return 0;
    const int bpd = kBlocksPerDesc32SplitsALargeStateCopyAcrossEnoughSmsToUseMemoryBandwidthNotTheCopyEngine;
    batched_copy_u32_kernel<<<(unsigned)(count * bpd), kThreads256, 0, (cudaStream_t)stream>>>(
        dsts, srcs, nwords, bpd);
    return (int)cudaGetLastError();
}
