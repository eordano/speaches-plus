#include <cuda_runtime.h>
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
    cudaStream_t s = (cudaStream_t)stream;
    nv_kernels_hello_kernel<<<grid, block, 0, s>>>(out, n);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_capability(int* sm_major, int* sm_minor) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return (int)err;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return (int)err;
    *sm_major = prop.major;
    *sm_minor = prop.minor;
    return 0;
}
