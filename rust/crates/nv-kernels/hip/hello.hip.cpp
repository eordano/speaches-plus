#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include "nv_kernels.h"

__global__ void nv_kernels_hello_kernel(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}

extern "C" int nv_kernels_hello_launch(void* stream, float* out, size_t n) {
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    hipStream_t s = (hipStream_t)stream;
    nv_kernels_hello_kernel<<<grid, block, 0, s>>>(out, n);
    return (int)hipGetLastError();
}

#define NV_KERNELS_CAPABILITY_NOT_APPLICABLE (-1000)

extern "C" int nv_kernels_capability(int* sm_major, int* sm_minor) {
    if (sm_major) *sm_major = 0;
    if (sm_minor) *sm_minor = 0;
    return NV_KERNELS_CAPABILITY_NOT_APPLICABLE;
}

extern "C" int nv_kernels_device_info(int* wave_size, char* arch_name,
                                      int arch_name_len) {
    int dev = 0;
    hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) return (int)err;
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, dev);
    if (err != hipSuccess) return (int)err;
    if (wave_size) *wave_size = prop.warpSize;
    if (arch_name && arch_name_len > 0) {
        int i = 0;
        while (i < arch_name_len - 1 && prop.gcnArchName[i] != '\0') {
            arch_name[i] = prop.gcnArchName[i];
            ++i;
        }
        arch_name[i] = '\0';
    }
    return 0;
}
