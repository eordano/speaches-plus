#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "nvk_smem_optin.cuh"

namespace {

constexpr int kGroupWarpSize = 32;
constexpr int kGroupWarpsPerBlock = 8;
constexpr int kGroupBlockDim = kGroupWarpSize * kGroupWarpsPerBlock;

constexpr int GROUP_K_TILE_IS_A_WHOLE_NUMBER_OF_WARP_STRIDES_SO_A_LANES_CHUNK_ORDER_IS_THE_SAME_SEQUENCE_THE_UNTILED_KERNEL_WALKS =
    kGroupWarpSize;

constexpr size_t GROUP_STAGED_X_TILE_BYTES_STAY_UNDER_THIS_SO_MORE_THAN_ONE_BLOCK_PER_SM_KEEPS_THE_WEIGHT_LOADS_IN_FLIGHT =
    49152;

__device__ __forceinline__ void group_deq_e4m3_bytes16(const int4& raw, float* wf) {
    const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        wf[2 * t] = f.x;
        wf[2 * t + 1] = f.y;
    }
}

template <int TM, int R>
__global__ void gemv_e4m3_group_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int tile_v
) {
    extern __shared__ float2 xg[];
    const int lane = threadIdx.x & (kGroupWarpSize - 1);
    const int warp = threadIdx.x / kGroupWarpSize;
    const int nbase = blockIdx.x * (kGroupWarpsPerBlock * R) + warp;
    const int Kv = K >> 4;
    const int rowpad = tile_v * 9;

    float acc[R][TM];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < TM; ++j) acc[r][j] = 0.0f;
    }

    const int4* wrow[R];
    bool live[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        int n = nbase + r * kGroupWarpsPerBlock;
        live[r] = n < N;
        wrow[r] = reinterpret_cast<const int4*>(wq + (size_t)(live[r] ? n : 0) * K);
    }

    for (int t0 = 0; t0 < Kv; t0 += tile_v) {
        const int tv = (Kv - t0) < tile_v ? (Kv - t0) : tile_v;
        __syncthreads();
        int4 head[R];
        const bool head_live = lane < tv;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            head[r] = (live[r] && head_live) ? __ldcs(&wrow[r][t0 + lane])
                                             : make_int4(0, 0, 0, 0);
        }
        const int pairs = tv * 8;
        for (int j = 0; j < TM; ++j) {
            const __nv_bfloat162* xr =
                reinterpret_cast<const __nv_bfloat162*>(x + (size_t)j * K) + (size_t)t0 * 8;
            float2* dst = xg + j * rowpad;
            for (int p = threadIdx.x; p < pairs; p += kGroupBlockDim) {
                dst[(p >> 3) * 9 + (p & 7)] = __bfloat1622float2(xr[p]);
            }
        }
        __syncthreads();

        for (int v = lane; v < tv; v += kGroupWarpSize) {
            float wf[R][16];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int4 raw = (v == lane) ? head[r]
                                       : (live[r] ? __ldcs(&wrow[r][t0 + v])
                                                  : make_int4(0, 0, 0, 0));
                group_deq_e4m3_bytes16(raw, wf[r]);
            }
            const float2* base = xg + v * 9;
#pragma unroll
            for (int j = 0; j < TM; ++j) {
                const float2* xp = base + j * rowpad;
                float xv[16];
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    float2 f = xp[p];
                    xv[2 * p] = f.x;
                    xv[2 * p + 1] = f.y;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        acc[r][j] += wf[r][2 * p] * xv[2 * p] + wf[r][2 * p + 1] * xv[2 * p + 1];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!live[r]) continue;
        int n = nbase + r * kGroupWarpsPerBlock;
        float rs = __ldg(&row_scale[n]);
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            float a = acc[r][j];
#pragma unroll
            for (int offset = kGroupWarpSize / 2; offset > 0; offset >>= 1) {
                a += __shfl_xor_sync(0xffffffff, a, offset);
            }
            if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
        }
    }
}

int group_tile_v(int M, int Kv) {
    const int stride =
        GROUP_K_TILE_IS_A_WHOLE_NUMBER_OF_WARP_STRIDES_SO_A_LANES_CHUNK_ORDER_IS_THE_SAME_SEQUENCE_THE_UNTILED_KERNEL_WALKS;
    size_t per_stride = (size_t)M * stride * 9 * sizeof(float2);
    if (per_stride == 0) return 0;
    size_t cap =
        GROUP_STAGED_X_TILE_BYTES_STAY_UNDER_THIS_SO_MORE_THAN_ONE_BLOCK_PER_SM_KEEPS_THE_WEIGHT_LOADS_IN_FLIGHT;
    size_t limit = (size_t)nvk_max_dynamic_smem_optin();
    if (limit < cap) cap = limit;
    int strides = (int)(cap / per_stride);
    if (strides < 1) return 0;
    int tile = strides * stride;
    if (tile > Kv) {
        tile = ((Kv + stride - 1) / stride) * stride;
    }
    return tile;
}

template <int TM, int R>
int launch_group(
    cudaStream_t s,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int tile_v
) {
    size_t smem = (size_t)TM * tile_v * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_e4m3_group_kernel<TM, R>, smem);
    if (orc != 0) return orc;
    const int rows_per_block = kGroupWarpsPerBlock * R;
    dim3 grid((unsigned)((N + rows_per_block - 1) / rows_per_block));
    gemv_e4m3_group_kernel<TM, R><<<grid, dim3(kGroupBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), N, K, tile_v);
    return (int)cudaGetLastError();
}

template <int R>
int dispatch_group_m(
    cudaStream_t s,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M,
    int tile_v
) {
    switch (M) {
        case 1: return launch_group<1, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 2: return launch_group<2, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 3: return launch_group<3, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 4: return launch_group<4, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 5: return launch_group<5, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 6: return launch_group<6, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 7: return launch_group<7, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 8: return launch_group<8, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 9: return launch_group<9, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 10: return launch_group<10, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 11: return launch_group<11, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 12: return launch_group<12, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 13: return launch_group<13, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 14: return launch_group<14, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 15: return launch_group<15, R>(s, wq, row_scale, x, y, N, K, tile_v);
        case 16: return launch_group<16, R>(s, wq, row_scale, x, y, N, K, tile_v);
        default: return -1;
    }
}

}  // namespace

extern "C" int nv_kernels_gemv_e4m3_group(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M,
    int rows_per_warp
) {
    if (N <= 0 || K <= 0 || M <= 0) return -1;
    if ((K & 15) != 0 || M > 16) return -1;
    const int Kv = K >> 4;
    const int tile_v = group_tile_v(M, Kv);
    if (tile_v <= 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    switch (rows_per_warp) {
        case 1: return dispatch_group_m<1>(s, wq, row_scale, x, y, N, K, M, tile_v);
        case 2: return dispatch_group_m<2>(s, wq, row_scale, x, y, N, K, M, tile_v);
        case 4: return dispatch_group_m<4>(s, wq, row_scale, x, y, N, K, M, tile_v);
        default: return -1;
    }
}

extern "C" int nv_kernels_gemv_e4m3_group_tile_v(int M, int K) {
    if (M <= 0 || K <= 0 || (K & 15) != 0) return 0;
    return group_tile_v(M, K >> 4);
}

extern "C" int nv_kernels_gemv_e4m3_untiled_staging_reaches_this_m(int K) {
    if (K <= 0 || (K & 15) != 0) return 0;
    size_t per_row = (size_t)(K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (per_row == 0) return 0;
    size_t limit = (size_t)nvk_max_dynamic_smem_optin();
    int m = (int)(limit / per_row);
    return m > 16 ? 16 : m;
}
