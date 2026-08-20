#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "nv_kernels.h"
#include "nv_hip_wave.h"

template <typename T>
__device__ inline float to_f32(T x);

template <>
__device__ inline float to_f32<float>(float x) { return x; }

template <>
__device__ inline float to_f32<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_f32(float x);

template <>
__device__ inline float from_f32<float>(float x) { return x; }

template <>
__device__ inline __hip_bfloat16 from_f32<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

template <int BLOCK>
__device__ inline float block_sum(float v) {
    constexpr int kWarp = nv_hip::kWave;
    constexpr int kWarps = BLOCK / kWarp;
    static_assert(BLOCK >= kWarp && BLOCK % kWarp == 0, "BLOCK must be a whole number of wavefronts");
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    v = nv_hip::wave_sum<kWarp>(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        s = nv_hip::lane_group_sum<kWarps>(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

template <typename T, int BLOCK>
__global__ void rmsnorm_kernel(const T* __restrict__ x,
                               const T* __restrict__ weight,
                               T* __restrict__ y,
                               size_t hidden,
                               float eps) {
    size_t row = blockIdx.x;
    const T* row_x = x + row * hidden;
    T* row_y = y + row * hidden;

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float v = to_f32<T>(row_x[i]);
        local += v * v;
    }
    float sum = block_sum<BLOCK>(local);
    float rms = rsqrtf(sum / (float)hidden + eps);

    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float v = to_f32<T>(row_x[i]) * rms * to_f32<T>(weight[i]);
        row_y[i] = from_f32<T>(v);
    }
}

extern "C" int nv_kernels_rmsnorm_f32(
    void* stream,
    const float* x,
    const float* weight,
    float* y,
    size_t batch,
    size_t hidden,
    float eps
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    rmsnorm_kernel<float, BLOCK><<<(int)batch, BLOCK, 0, s>>>(x, weight, y, hidden, eps);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rmsnorm_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* y,
    size_t batch,
    size_t hidden,
    float eps
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    rmsnorm_kernel<__hip_bfloat16, BLOCK><<<(int)batch, BLOCK, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(weight),
        reinterpret_cast<__hip_bfloat16*>(y),
        hidden,
        eps);
    return (int)hipGetLastError();
}
