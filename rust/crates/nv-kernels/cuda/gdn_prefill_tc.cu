#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include "nv_kernels.h"

namespace gdn_prefill_tc {

static std::atomic<long long>
    g_launches_counted_so_an_engagement_gate_never_infers_arm_selection_from_bit_deltas{0};

constexpr int TC_CHUNK_64_MATCHES_THE_CHUNK_GEMM_SUBCHUNK_SO_WORKSPACE_LAYOUT_IS_SHARED = 64;
constexpr int TC_HEAD_DIM_128_IS_THE_Q38_GDN_KEY_AND_VALUE_WIDTH = 128;
constexpr int TC_THREADS_256_EIGHT_WARPS_TWO_16X16_TILES_EACH_COVER_THE_64X64_PRODUCTS = 256;
constexpr int TC_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS = 136;
constexpr int TC_SCRATCH_LD_68_PADS_THE_64_COL_F32_STAGE_SO_STORES_SPREAD_BANKS = 68;
constexpr int TC_TINV_LD_72_PADS_THE_64_COL_BF16_TINV_SO_FRAGMENT_LOADS_SPREAD_BANKS = 72;
constexpr int TC_RHS_LD_264_PADS_THE_256_COL_RHS_AND_FITS_THE_DEAD_KR_QR_STAGE = 264;
constexpr int TC_WARP_STORE_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS = 20;
constexpr int TC_AB_LD_64_UNPADDED_SO_A_BF16_PLUS_ALL_NEUMANN_BLOCKS_FIT_THE_DEAD_SCR_BYTES = 64;
constexpr int TC_BLK_LD_16_ONE_16X16_NEUMANN_BLOCK_PER_512_BYTES = 16;
constexpr int TC_LOWER_TILES_10_OF_16_THE_MASKS_ZERO_EVERY_UPPER_TILE_SO_MMA_SKIPS_THEM = 10;
__device__ constexpr int TC_LOW_MI[TC_LOWER_TILES_10_OF_16_THE_MASKS_ZERO_EVERY_UPPER_TILE_SO_MMA_SKIPS_THEM] =
    {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
__device__ constexpr int TC_LOW_NJ[TC_LOWER_TILES_10_OF_16_THE_MASKS_ZERO_EVERY_UPPER_TILE_SO_MMA_SKIPS_THEM] =
    {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
constexpr int TC_NEUMANN_BLOCKS_6_STRICTLY_LOWER_16X16_BLOCKS_OF_THE_64X64_TRIANGLE = 6;
__device__ constexpr int TC_NEU_MI[TC_NEUMANN_BLOCKS_6_STRICTLY_LOWER_16X16_BLOCKS_OF_THE_64X64_TRIANGLE] =
    {1, 2, 3, 2, 3, 3};
__device__ constexpr int TC_NEU_MJ[TC_NEUMANN_BLOCKS_6_STRICTLY_LOWER_16X16_BLOCKS_OF_THE_64X64_TRIANGLE] =
    {0, 0, 0, 1, 1, 2};

__device__ __forceinline__ void tc_cp_async16(void* dst_smem, const void* src_gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}

__device__ __forceinline__ void tc_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void tc_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ void tc_fused_head_kernel(
    const __nv_bfloat16* __restrict__ kr, const __nv_bfloat16* __restrict__ qr,
    const __nv_bfloat16* __restrict__ rhs, const float* __restrict__ beta,
    const float* __restrict__ cg,
    __nv_bfloat16* __restrict__ attn, __nv_bfloat16* __restrict__ wu, int T, int H, int HK
) {
    using namespace nvcuda;
    const int C = TC_CHUNK_64_MATCHES_THE_CHUNK_GEMM_SUBCHUNK_SO_WORKSPACE_LAYOUT_IS_SHARED;
    const int D = TC_HEAD_DIM_128_IS_THE_Q38_GDN_KEY_AND_VALUE_WIDTH;
    const int ALD = TC_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int SLD = TC_SCRATCH_LD_68_PADS_THE_64_COL_F32_STAGE_SO_STORES_SPREAD_BANKS;
    const int TLD = TC_TINV_LD_72_PADS_THE_64_COL_BF16_TINV_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int RLD = TC_RHS_LD_264_PADS_THE_256_COL_RHS_AND_FITS_THE_DEAD_KR_QR_STAGE;
    const int WLD = TC_WARP_STORE_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS;
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int kh = h / (H / HK);
    const size_t sh = (size_t)s * H + h;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ unsigned char smem_raw[];
    float* scr = (float*)smem_raw;
    float* af = scr + C * SLD;
    float* cgs = af + C * SLD;
    float* betas = cgs + C;
    __nv_bfloat16* kq = (__nv_bfloat16*)(betas + C);
    __nv_bfloat16* kst = kq;
    __nv_bfloat16* qst = kq + C * ALD;
    __nv_bfloat16* rst = kq;
    __nv_bfloat16* tb = kq + 2 * C * ALD;

    {
        const __nv_bfloat16* krow = kr + (((size_t)s * HK + kh) * C) * D;
        const __nv_bfloat16* qrow = qr + (((size_t)s * HK + kh) * C) * D;
        for (int i = tid; i < C * 16; i += (int)blockDim.x) {
            int c = i >> 4;
            int seg = i & 15;
            tc_cp_async16(kst + c * ALD + seg * 8, krow + (size_t)c * D + seg * 8);
            tc_cp_async16(qst + c * ALD + seg * 8, qrow + (size_t)c * D + seg * 8);
        }
        tc_cp_async_commit();
        if (tid < C) {
            cgs[tid] = cg[sh * C + tid];
            int t = s * C + tid;
            betas[tid] = t < T ? beta[(size_t)t * H + h] : 0.f;
        }
        tc_cp_async_wait<0>();
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fkt;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> facc;

    for (int tile = wid; tile < TC_LOWER_TILES_10_OF_16_THE_MASKS_ZERO_EVERY_UPPER_TILE_SO_MMA_SKIPS_THEM;
         tile += 8) {
        int tm = TC_LOW_MI[tile] * 16;
        int tn = TC_LOW_NJ[tile] * 16;
        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt < D / 16; ++kt) {
            wmma::load_matrix_sync(fa, kst + tm * ALD + kt * 16, ALD);
            wmma::load_matrix_sync(fkt, kst + tn * ALD + kt * 16, ALD);
            wmma::mma_sync(facc, fa, fkt, facc);
        }
        wmma::store_matrix_sync(scr + tm * SLD + tn, facc, SLD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < C * C; idx += (int)blockDim.x) {
        int i = idx >> 6;
        int j = idx & 63;
        int t_i = s * C + i;
        float a_v = 0.f;
        if (t_i < T && j < i) {
            a_v = betas[i] * expf(cgs[i] - cgs[j]) * scr[i * SLD + j];
        }
        af[i * SLD + j] = a_v;
    }
    __syncthreads();

    for (int tile = wid; tile < TC_LOWER_TILES_10_OF_16_THE_MASKS_ZERO_EVERY_UPPER_TILE_SO_MMA_SKIPS_THEM;
         tile += 8) {
        int tm = TC_LOW_MI[tile] * 16;
        int tn = TC_LOW_NJ[tile] * 16;
        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt < D / 16; ++kt) {
            wmma::load_matrix_sync(fa, qst + tm * ALD + kt * 16, ALD);
            wmma::load_matrix_sync(fkt, kst + tn * ALD + kt * 16, ALD);
            wmma::mma_sync(facc, fa, fkt, facc);
        }
        wmma::store_matrix_sync(scr + tm * SLD + tn, facc, SLD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < C * C; idx += (int)blockDim.x) {
        int i = idx >> 6;
        int j = idx & 63;
        int t_i = s * C + i;
        float o_v = 0.f;
        if (t_i < T && j <= i) {
            o_v = expf(cgs[i] - cgs[j]) * scr[i * SLD + j];
        }
        attn[(sh * C + i) * C + j] = __float2bfloat16(o_v);
    }
    __syncthreads();

    {
        const __nv_bfloat16* rrow = rhs + sh * C * 256;
        for (int i = tid; i < C * 32; i += (int)blockDim.x) {
            int c = i >> 5;
            int seg = i & 31;
            tc_cp_async16(rst + c * RLD + seg * 8, rrow + (size_t)c * 256 + seg * 8);
        }
        tc_cp_async_commit();
    }

    const int ABLD = TC_AB_LD_64_UNPADDED_SO_A_BF16_PLUS_ALL_NEUMANN_BLOCKS_FIT_THE_DEAD_SCR_BYTES;
    const int BLD = TC_BLK_LD_16_ONE_16X16_NEUMANN_BLOCK_PER_512_BYTES;
    __nv_bfloat16* ab = (__nv_bfloat16*)scr;
    __nv_bfloat16* blk_ti = ab + C * ABLD;
    __nv_bfloat16* blk_m = blk_ti + 4 * 256;
    __nv_bfloat16* blk_m2 = blk_m + 6 * 256;
    __nv_bfloat16* blk_m3 = blk_m2 + 3 * 256;

    for (int idx = tid; idx < C * C; idx += (int)blockDim.x) {
        int i = idx >> 6;
        int j = idx & 63;
        ab[i * ABLD + j] = __float2bfloat16(af[i * SLD + j]);
    }
    for (int idx = tid; idx < C * TLD; idx += (int)blockDim.x) {
        tb[idx] = __float2bfloat16(0.f);
    }
    if (tid < C) {
        int g = tid >> 4;
        int c = tid & 15;
        const float* dbase = af + (g * 16) * SLD + g * 16;
        float tcol[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float v = (r == c) ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (j < r) v -= dbase[r * SLD + j] * tcol[j];
            }
            tcol[r] = v;
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            blk_ti[g * 256 + r * BLD + c] = __float2bfloat16(tcol[r]);
        }
    }
    __syncthreads();

    auto mma_16 = [&](const __nv_bfloat16* pa, const __nv_bfloat16* pb, int ldb, bool accum) {
        if (!accum) wmma::fill_fragment(facc, 0.f);
        wmma::load_matrix_sync(fa, pa, BLD);
        wmma::load_matrix_sync(fb, pb, ldb);
        wmma::mma_sync(facc, fa, fb, facc);
    };
    auto store_blk_bf16 = [&](__nv_bfloat16* dst, int ldd) {
        float* stg = af + wid * 16 * TC_WARP_STORE_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS;
        wmma::store_matrix_sync(
            stg, facc, TC_WARP_STORE_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS,
            wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            dst[(e >> 4) * ldd + (e & 15)] = __float2bfloat16(
                stg[(e >> 4) * TC_WARP_STORE_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS +
                    (e & 15)]);
        }
        __syncwarp();
    };

    if (wid < TC_NEUMANN_BLOCKS_6_STRICTLY_LOWER_16X16_BLOCKS_OF_THE_64X64_TRIANGLE) {
        int mi = TC_NEU_MI[wid];
        int mj = TC_NEU_MJ[wid];
        mma_16(blk_ti + mi * 256, ab + (mi * 16) * ABLD + mj * 16, ABLD, false);
        store_blk_bf16(blk_m + wid * 256, BLD);
    }
    __syncthreads();

    if (wid == 0) {
        mma_16(blk_m + 3 * 256, blk_m + 0 * 256, BLD, false);
        store_blk_bf16(blk_m2 + 0 * 256, BLD);
    } else if (wid == 1) {
        mma_16(blk_m + 5 * 256, blk_m + 3 * 256, BLD, false);
        store_blk_bf16(blk_m2 + 2 * 256, BLD);
    } else if (wid == 2) {
        mma_16(blk_m + 4 * 256, blk_m + 0 * 256, BLD, false);
        mma_16(blk_m + 5 * 256, blk_m + 1 * 256, BLD, true);
        store_blk_bf16(blk_m2 + 1 * 256, BLD);
    }
    __syncthreads();

    if (wid == 0) {
        mma_16(blk_m + 5 * 256, blk_m2 + 0 * 256, BLD, false);
        store_blk_bf16(blk_m3, BLD);
    }
    __syncthreads();

    for (int e = tid; e < 6 * 256; e += (int)blockDim.x) {
        int b = e >> 8;
        int r = (e >> 4) & 15;
        int c2 = e & 15;
        float v = -__bfloat162float(blk_m[b * 256 + r * BLD + c2]);
        int bi = TC_NEU_MI[b];
        int bj = TC_NEU_MJ[b];
        int d = bi - bj;
        if (d >= 2) {
            int m2i = bj == 0 ? bi - 2 : 2;
            v += __bfloat162float(blk_m2[m2i * 256 + r * BLD + c2]);
        }
        if (d == 3) {
            v -= __bfloat162float(blk_m3[r * BLD + c2]);
        }
        blk_m[b * 256 + r * BLD + c2] = __float2bfloat16(v);
    }
    __syncthreads();

    if (wid < TC_NEUMANN_BLOCKS_6_STRICTLY_LOWER_16X16_BLOCKS_OF_THE_64X64_TRIANGLE) {
        int mi = TC_NEU_MI[wid];
        int mj = TC_NEU_MJ[wid];
        mma_16(blk_m + wid * 256, blk_ti + mj * 256, BLD, false);
        store_blk_bf16(tb + (mi * 16) * TLD + mj * 16, TLD);
    } else {
        for (int e = (wid - 6) * 32 + lane; e < 4 * 256; e += 64) {
            int g = e >> 8;
            int r = (e >> 4) & 15;
            int c2 = e & 15;
            tb[(g * 16 + r) * TLD + g * 16 + c2] = blk_ti[g * 256 + r * BLD + c2];
        }
    }
    tc_cp_async_wait<0>();
    __syncthreads();

    const int m0 = (wid >> 1) * 16;
    const int half = (wid & 1) * 128;
    float* wscr = af + wid * 16 * WLD;
    for (int nt = 0; nt < 8; ++nt) {
        int n0 = half + nt * 16;
        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt <= m0 / 16; ++kt) {
            wmma::load_matrix_sync(fa, tb + m0 * TLD + kt * 16, TLD);
            wmma::load_matrix_sync(fb, rst + kt * 16 * RLD + half + nt * 16, RLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        wmma::store_matrix_sync(wscr, facc, WLD, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            int r = idx >> 4;
            int cc = idx & 15;
            wu[(sh * C + m0 + r) * 256 + n0 + cc] = __float2bfloat16(wscr[r * WLD + cc]);
        }
        __syncwarp();
    }
}

constexpr size_t
    TC_SMEM_BYTES_87K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP_AND_FITS_ONE_BLOCK_PER_SM =
        (size_t)(2 * 64 * 68 + 2 * 64) * sizeof(float) +
        (size_t)(2 * 64 * 136 + 2 * 64 * 72) * sizeof(__nv_bfloat16);

static_assert(64 * TC_RHS_LD_264_PADS_THE_256_COL_RHS_AND_FITS_THE_DEAD_KR_QR_STAGE <=
                  2 * 64 * TC_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS,
              "the rhs stage must alias the dead kr+qr reservation, never grow past it");
static_assert(8 * 16 * TC_WARP_STORE_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS <=
                  64 * TC_SCRATCH_LD_68_PADS_THE_64_COL_F32_STAGE_SO_STORES_SPREAD_BANKS,
              "the per-warp wu store bounce must alias the dead A reservation, never grow past it");
static_assert((64 * TC_AB_LD_64_UNPADDED_SO_A_BF16_PLUS_ALL_NEUMANN_BLOCKS_FIT_THE_DEAD_SCR_BYTES +
               14 * 256) * sizeof(__nv_bfloat16) <=
                  64 * TC_SCRATCH_LD_68_PADS_THE_64_COL_F32_STAGE_SO_STORES_SPREAD_BANKS *
                      sizeof(float),
              "the bf16 A image plus the 14 neumann blocks must alias the dead qk scratch, never "
              "grow past it");

}

extern "C" long long nv_kernels_gdn_prefill_tc_head_launches_total() {
    return gdn_prefill_tc::g_launches_counted_so_an_engagement_gate_never_infers_arm_selection_from_bit_deltas
        .load(std::memory_order_relaxed);
}

extern "C" int nv_kernels_gdn_prefill_tc_fused_head(
    void* stream,
    const uint16_t* kr,
    const uint16_t* qr,
    const uint16_t* rhs,
    const float* beta,
    const float* cg,
    uint16_t* attn,
    uint16_t* wu,
    int T,
    int H,
    int HK,
    int n
) {
    using namespace gdn_prefill_tc;
    if (T <= 0 || H <= 0 || HK <= 0 || n <= 0) return -1;
    if (H % HK != 0) return -3;
    if (rhs == nullptr) return -2;
    static std::once_flag smem_once;
    static cudaError_t smem_rc = cudaSuccess;
    std::call_once(smem_once, []() {
        smem_rc = cudaFuncSetAttribute(
            tc_fused_head_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)TC_SMEM_BYTES_87K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP_AND_FITS_ONE_BLOCK_PER_SM);
    });
    if (smem_rc != cudaSuccess) return (int)smem_rc;
    dim3 grid(n, H);
    tc_fused_head_kernel<<<
        grid, TC_THREADS_256_EIGHT_WARPS_TWO_16X16_TILES_EACH_COVER_THE_64X64_PRODUCTS,
        TC_SMEM_BYTES_87K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP_AND_FITS_ONE_BLOCK_PER_SM,
        (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)kr, (const __nv_bfloat16*)qr, (const __nv_bfloat16*)rhs, beta, cg,
        (__nv_bfloat16*)attn, (__nv_bfloat16*)wu, T, H, HK);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    gdn_prefill_tc::g_launches_counted_so_an_engagement_gate_never_infers_arm_selection_from_bit_deltas
        .fetch_add(1, std::memory_order_relaxed);
    return 0;
}
