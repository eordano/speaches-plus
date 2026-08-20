#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

static inline bool nvk_pdl_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("NV_PDL");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v == 1;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define NVK_PDL_PROLOG() cudaGridDependencySynchronize()
#define NVK_PDL_EPILOG() cudaTriggerProgrammaticLaunchCompletion()
#else
#define NVK_PDL_PROLOG() ((void)0)
#define NVK_PDL_EPILOG() ((void)0)
#endif

#define NVK_PDL_ATTR(cfg_var, grid_v, block_v, smem_v, stream_v)          \
    cudaLaunchAttribute cfg_var##_attr;                                   \
    cfg_var##_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization; \
    cfg_var##_attr.val.programmaticStreamSerializationAllowed = 1;        \
    cudaLaunchConfig_t cfg_var = {};                                      \
    cfg_var.gridDim = (grid_v);                                           \
    cfg_var.blockDim = (block_v);                                         \
    cfg_var.dynamicSmemBytes = (smem_v);                                  \
    cfg_var.stream = (stream_v);                                          \
    cfg_var.attrs = &cfg_var##_attr;                                      \
    cfg_var.numAttrs = 1;
