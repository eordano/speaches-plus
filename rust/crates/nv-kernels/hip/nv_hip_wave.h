#pragma once

#include <hip/hip_runtime.h>

#if defined(__GFX9__)
#define NV_HIP_WAVE 64
#elif defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
#define NV_HIP_WAVE 32
#elif defined(__AMDGCN__) && defined(__HIP_DEVICE_COMPILE__)
#error "nv_hip_wave.h: unrecognised AMDGCN target family; define NV_HIP_WAVE explicitly"
#else
#define NV_HIP_WAVE 32
#endif

#define NV_HIP_WAVE_FALLBACK 64

namespace nv_hip {

constexpr int kWave = NV_HIP_WAVE;

using lane_mask_t = unsigned long long;

constexpr lane_mask_t kFullMask =
    (kWave == 64) ? ~(lane_mask_t)0 : (lane_mask_t)0xffffffffull;

template <int WAVE>
__device__ inline float wave_sum(float v) {
#pragma unroll
    for (int o = WAVE / 2; o > 0; o >>= 1) {
        v += __shfl_xor_sync(kFullMask, v, o, WAVE);
    }
    return v;
}

template <int WAVE>
__device__ inline float wave_max(float v) {
#pragma unroll
    for (int o = WAVE / 2; o > 0; o >>= 1) {
        float o_v = __shfl_xor_sync(kFullMask, v, o, WAVE);
        v = o_v > v ? o_v : v;
    }
    return v;
}

template <int N>
__device__ inline float lane_group_sum(float v) {
#pragma unroll
    for (int o = N / 2; o > 0; o >>= 1) {
        v += __shfl_xor_sync(kFullMask, v, o, kWave);
    }
    return v;
}

__device__ inline lane_mask_t wave_ballot(int pred) {
    return (lane_mask_t)__ballot(pred);
}

__device__ inline int lane_mask_popc(lane_mask_t m) {
    return __popcll((unsigned long long)m);
}

__device__ inline int lane_mask_ffs(lane_mask_t m) {
    return __ffsll((unsigned long long)m);
}

__device__ inline lane_mask_t lane_bit(int lane) {
    return (lane_mask_t)1 << lane;
}

inline int host_wave_size(int device) {
    int v = 0;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, device) !=
        hipSuccess) {
        return 0;
    }
    return v;
}

inline int host_wave_size() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return NV_HIP_WAVE_FALLBACK;
    int w = host_wave_size(dev);
    return w > 0 ? w : NV_HIP_WAVE_FALLBACK;
}

inline int wave_aligned_block(int desired, int max_block) {
    int wave = host_wave_size();
    if (wave <= 0) wave = NV_HIP_WAVE_FALLBACK;
    if (max_block < wave) max_block = wave;
    if (desired < 1) desired = 1;
    if (desired > max_block) desired = max_block;
    int b = ((desired + wave - 1) / wave) * wave;
    if (b > max_block) b = (max_block / wave) * wave;
    if (b < wave) b = wave;
    return b;
}

inline int host_max_lds_bytes(int device) {
    int v = 0;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerBlock,
                              device) != hipSuccess) {
        return 0;
    }
    return v;
}

}
