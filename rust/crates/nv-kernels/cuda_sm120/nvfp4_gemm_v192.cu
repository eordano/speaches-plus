#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nv_kernels.h"

namespace {

constexpr int TILE_M_192_RETIRES_THE_PARTIAL_WAVE_THE_128_ROW_CTA_LEAVES_ON_188_SMS = 192;
constexpr int TILE_N = 128;
constexpr int TILE_K_NIBBLES = 128;
constexpr int STAGES = 3;
constexpr int ROW_STRIDE_80B_SPREADS_8_CONSECUTIVE_ROWS_OVER_ALL_32_BANKS = 80;
constexpr int WARPS_M = 6;
constexpr int WARPS_N = 2;
constexpr int M_ITERS = 2;
constexpr int N_ITERS = 8;
constexpr int THREADS = WARPS_M * WARPS_N * 32;

constexpr int TILE_M = TILE_M_192_RETIRES_THE_PARTIAL_WAVE_THE_128_ROW_CTA_LEAVES_ON_188_SMS;
constexpr int ROW_STRIDE = ROW_STRIDE_80B_SPREADS_8_CONSECUTIVE_ROWS_OVER_ALL_32_BANKS;
constexpr int SMEM_A_STAGE = TILE_M * ROW_STRIDE;
constexpr int SMEM_B_STAGE = TILE_N * ROW_STRIDE;
constexpr int SFA_BLOCKS_2_MTILES_X_2_KTILES_BECAUSE_192_SPANS_A_128_BLOCK_BOUNDARY = 4;
constexpr int SFB_BLOCKS = 2;
constexpr int SMEM_SFA_STAGE = SFA_BLOCKS_2_MTILES_X_2_KTILES_BECAUSE_192_SPANS_A_128_BLOCK_BOUNDARY * 512;
constexpr int SMEM_SFB_STAGE = SFB_BLOCKS * 512;
constexpr int SMEM_STAGE =
    SMEM_A_STAGE + SMEM_B_STAGE + SMEM_SFA_STAGE + SMEM_SFB_STAGE;
constexpr int SMEM_TOTAL = SMEM_STAGE * STAGES;

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(src));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_wait_all_but_one() {
    asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* addr) {
    unsigned s = (unsigned)__cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ int sf_offset_within_512b_block(int row) {
    return ((row & 31) << 4) + (((row >> 5) & 3) << 2);
}

__device__ __forceinline__ void mma_nvfp4_16x8x64_sf4x(float* acc, uint32_t a0,
                                                       uint32_t a1, uint32_t a2,
                                                       uint32_t a3, uint32_t b0,
                                                       uint32_t b1, uint32_t sfa,
                                                       uint32_t sfb) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%0,%1,%2,%3},"
        "{%10},"
        "{%11,%12},"
        "{%13},"
        "{%14,%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z),
          "h"(z), "r"(sfb), "h"(z), "h"(z));
#endif
}

struct StagePtrs {
    char* a;
    char* b;
    char* sfa;
    char* sfb;
};

__device__ __forceinline__ StagePtrs stage_ptrs(char* smem, int stage) {
    char* base = smem + (size_t)stage * SMEM_STAGE;
    StagePtrs p;
    p.a = base;
    p.b = base + SMEM_A_STAGE;
    p.sfa = base + SMEM_A_STAGE + SMEM_B_STAGE;
    p.sfb = base + SMEM_A_STAGE + SMEM_B_STAGE + SMEM_SFA_STAGE;
    return p;
}

struct CopyDesc {
    const uint8_t* src;
    int dst_off;
    int kt_shift;
};

__device__ __forceinline__ void issue_from_descs(char* smem, int slot,
                                                 const CopyDesc* d, int nd,
                                                 int kt_shift_a, int kt) {
    char* base = smem + (size_t)slot * SMEM_STAGE;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j < nd && d[j].src) {
            cp_async16(base + d[j].dst_off, d[j].src + ((size_t)kt << kt_shift_a));
        }
    }
}

__global__ void __launch_bounds__(THREADS, 1) nvfp4_gemm_v192_kernel(
    const uint8_t* __restrict__ a_fp4, const uint8_t* __restrict__ a_sf,
    const uint8_t* __restrict__ b_fp4, const uint8_t* __restrict__ b_sf,
    const float* __restrict__ global_sf, __nv_bfloat16* __restrict__ d, int m,
    int n, int k, int tiles_m) {
    extern __shared__ char smem[];
    int tm = blockIdx.x % tiles_m;
    int tn = blockIdx.x / tiles_m;
    int tm_base = tm * TILE_M;
    int tn_base = tn * TILE_N;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int wm = warp % WARPS_M;
    int wn = warp / WARPS_M;
    int kt_count = k / TILE_K_NIBBLES;
    int k_tiles = k >> 6;
    size_t row_bytes = (size_t)(k >> 1);

    CopyDesc dab[4];
    {
        int row = tid >> 2, c = tid & 3, gr = tm_base + row;
        dab[0].src = gr < m ? a_fp4 + (size_t)gr * row_bytes + c * 16 : nullptr;
        dab[0].dst_off = row * ROW_STRIDE + c * 16;
        dab[0].kt_shift = 6;
    }
    {
        int i1 = tid + THREADS;
        int row = i1 >> 2, c = i1 & 3, gr = tm_base + row;
        dab[1].src = gr < m ? a_fp4 + (size_t)gr * row_bytes + c * 16 : nullptr;
        dab[1].dst_off = row * ROW_STRIDE + c * 16;
        dab[1].kt_shift = 6;
    }
    {
        int row = tid >> 2, c = tid & 3, gr = tn_base + row;
        dab[2].src = gr < n ? b_fp4 + (size_t)gr * row_bytes + c * 16 : nullptr;
        dab[2].dst_off = SMEM_A_STAGE + row * ROW_STRIDE + c * 16;
        dab[2].kt_shift = 6;
    }
    {
        int i = tid + THREADS;
        if (i < TILE_N * 4) {
            int row = i >> 2, c = i & 3, gr = tn_base + row;
            dab[3].src = gr < n ? b_fp4 + (size_t)gr * row_bytes + c * 16 : nullptr;
            dab[3].dst_off = SMEM_A_STAGE + row * ROW_STRIDE + c * 16;
            dab[3].kt_shift = 6;
        } else {
            int sfi = i - TILE_N * 4;
            int blk = sfi >> 5, chunk = sfi & 31;
            if (blk < SFA_BLOCKS_2_MTILES_X_2_KTILES_BECAUSE_192_SPANS_A_128_BLOCK_BOUNDARY) {
                int part = blk >> 1, g = blk & 1;
                int mtile = (tm_base >> 7) + part;
                dab[3].src = mtile * 128 < m
                    ? a_sf + ((size_t)(mtile * k_tiles + g) << 9) + chunk * 16
                    : nullptr;
                dab[3].dst_off =
                    SMEM_A_STAGE + SMEM_B_STAGE + blk * 512 + chunk * 16;
            } else if (blk < 6) {
                int g = blk & 1;
                int ntile = tn_base >> 7;
                dab[3].src = ntile * 128 < n
                    ? b_sf + ((size_t)(ntile * k_tiles + g) << 9) + chunk * 16
                    : nullptr;
                dab[3].dst_off = SMEM_A_STAGE + SMEM_B_STAGE + SMEM_SFA_STAGE +
                                 (blk - 4) * 512 + chunk * 16;
            } else {
                dab[3].src = nullptr;
                dab[3].dst_off = 0;
            }
            dab[3].kt_shift = 10;
        }
    }
    bool d3_none = dab[3].src == nullptr && dab[3].dst_off == 0;

#pragma unroll
    for (int slot = 0; slot < STAGES; ++slot) {
        char* base = smem + (size_t)slot * SMEM_STAGE;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (!dab[j].src && !(j == 3 && d3_none)) {
                *(uint4*)(base + dab[j].dst_off) = uint4{0, 0, 0, 0};
            }
        }
    }
    __syncthreads();

    for (int ps = 0; ps < STAGES - 1; ++ps) {
        if (ps < kt_count) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (dab[j].src) {
                    cp_async16(smem + (size_t)ps * SMEM_STAGE + dab[j].dst_off,
                               dab[j].src + ((size_t)ps << dab[j].kt_shift));
                }
            }
        }
        cp_commit();
    }

    float acc[M_ITERS][N_ITERS][4];
#pragma unroll
    for (int mi = 0; mi < M_ITERS; ++mi) {
#pragma unroll
        for (int ni = 0; ni < N_ITERS; ++ni) {
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[mi][ni][v] = 0.f;
        }
    }

    uint32_t afrag[2][M_ITERS][4];
    uint32_t sfa[2][M_ITERS];
    uint32_t bfrag[2][N_ITERS][2];
    uint32_t sfb[2][N_ITERS];
    int lr = lane & 7;
    int sel = lane >> 3;

    cp_wait_all_but_one();
    __syncthreads();

    {
        StagePtrs p = stage_ptrs(smem, 0);
#pragma unroll
        for (int mi = 0; mi < M_ITERS; ++mi) {
            int ra = wm * 16 + mi * 96 + (sel & 1) * 8 + lr;
            ldmatrix_x4(afrag[0][mi],
                        p.a + ra * ROW_STRIDE + (sel >> 1) * 16);
            int gsf = tm_base + wm * 16 + mi * 96 + 8 * (lane & 1) + (lane >> 2);
            int part = (gsf >> 7) - (tm_base >> 7);
            sfa[0][mi] = *(const uint32_t*)(p.sfa + part * 2 * 512 +
                                            sf_offset_within_512b_block(gsf));
        }
#pragma unroll
        for (int nb = 0; nb < N_ITERS / 2; ++nb) {
            int ni = 2 * nb;
            int rb = wn * 64 + (ni + (sel >> 1)) * 8 + lr;
            uint32_t q[4];
            ldmatrix_x4(q, p.b + rb * ROW_STRIDE + (sel & 1) * 16);
            bfrag[0][ni][0] = q[0];
            bfrag[0][ni][1] = q[1];
            bfrag[0][ni + 1][0] = q[2];
            bfrag[0][ni + 1][1] = q[3];
        }
#pragma unroll
        for (int ni = 0; ni < N_ITERS; ++ni) {
            int rb = wn * 64 + ni * 8 + (lane >> 2);
            sfb[0][ni] = *(const uint32_t*)(p.sfb +
                                            sf_offset_within_512b_block(rb));
        }
    }

    for (int kt = 0; kt < kt_count; ++kt) {
        StagePtrs pc = stage_ptrs(smem, kt % STAGES);
#pragma unroll
        for (int mi = 0; mi < M_ITERS; ++mi) {
            int ra = wm * 16 + mi * 96 + (sel & 1) * 8 + lr;
            ldmatrix_x4(afrag[1][mi],
                        pc.a + ra * ROW_STRIDE + 32 + (sel >> 1) * 16);
            int gsf = tm_base + wm * 16 + mi * 96 + 8 * (lane & 1) + (lane >> 2);
            int part = (gsf >> 7) - (tm_base >> 7);
            sfa[1][mi] = *(const uint32_t*)(pc.sfa + (part * 2 + 1) * 512 +
                                            sf_offset_within_512b_block(gsf));
        }
#pragma unroll
        for (int nb = 0; nb < N_ITERS / 2; ++nb) {
            int ni = 2 * nb;
            int rb = wn * 64 + (ni + (sel >> 1)) * 8 + lr;
            uint32_t q[4];
            ldmatrix_x4(q, pc.b + rb * ROW_STRIDE + 32 + (sel & 1) * 16);
            bfrag[1][ni][0] = q[0];
            bfrag[1][ni][1] = q[1];
            bfrag[1][ni + 1][0] = q[2];
            bfrag[1][ni + 1][1] = q[3];
        }
#pragma unroll
        for (int ni = 0; ni < N_ITERS; ++ni) {
            int rb = wn * 64 + ni * 8 + (lane >> 2);
            sfb[1][ni] = *(const uint32_t*)(pc.sfb + 512 +
                                            sf_offset_within_512b_block(rb));
        }
        int nxt = kt + STAGES - 1;
        bool issue = nxt < kt_count;
        char* ibase = smem + (size_t)(nxt % STAGES) * SMEM_STAGE;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (issue && dab[j].src) {
                cp_async16(ibase + dab[j].dst_off,
                           dab[j].src + ((size_t)nxt << dab[j].kt_shift));
            }
        }
#pragma unroll
        for (int ni = 0; ni < N_ITERS; ++ni) {
#pragma unroll
            for (int mi = 0; mi < M_ITERS; ++mi) {
                mma_nvfp4_16x8x64_sf4x(acc[mi][ni], afrag[0][mi][0],
                                       afrag[0][mi][1], afrag[0][mi][2],
                                       afrag[0][mi][3], bfrag[0][ni][0],
                                       bfrag[0][ni][1], sfa[0][mi], sfb[0][ni]);
            }
        }

#pragma unroll
        for (int j = 2; j < 4; ++j) {
            if (issue && dab[j].src) {
                cp_async16(ibase + dab[j].dst_off,
                           dab[j].src + ((size_t)nxt << dab[j].kt_shift));
            }
        }
        cp_commit();
        cp_wait_all_but_one();
        __syncthreads();

        if (kt + 1 < kt_count) {
            StagePtrs pn = stage_ptrs(smem, (kt + 1) % STAGES);
#pragma unroll
            for (int mi = 0; mi < M_ITERS; ++mi) {
                int ra = wm * 16 + mi * 96 + (sel & 1) * 8 + lr;
                ldmatrix_x4(afrag[0][mi],
                            pn.a + ra * ROW_STRIDE + (sel >> 1) * 16);
                int gsf =
                    tm_base + wm * 16 + mi * 96 + 8 * (lane & 1) + (lane >> 2);
                int part = (gsf >> 7) - (tm_base >> 7);
                sfa[0][mi] = *(const uint32_t*)(pn.sfa + part * 2 * 512 +
                                                sf_offset_within_512b_block(gsf));
            }
#pragma unroll
            for (int nb = 0; nb < N_ITERS / 2; ++nb) {
                int ni = 2 * nb;
                int rb = wn * 64 + (ni + (sel >> 1)) * 8 + lr;
                uint32_t q[4];
                ldmatrix_x4(q, pn.b + rb * ROW_STRIDE + (sel & 1) * 16);
                bfrag[0][ni][0] = q[0];
                bfrag[0][ni][1] = q[1];
                bfrag[0][ni + 1][0] = q[2];
                bfrag[0][ni + 1][1] = q[3];
            }
#pragma unroll
            for (int ni = 0; ni < N_ITERS; ++ni) {
                int rb = wn * 64 + ni * 8 + (lane >> 2);
                sfb[0][ni] = *(const uint32_t*)(pn.sfb +
                                                sf_offset_within_512b_block(rb));
            }
        }
#pragma unroll
        for (int ni = 0; ni < N_ITERS; ++ni) {
#pragma unroll
            for (int mi = 0; mi < M_ITERS; ++mi) {
                mma_nvfp4_16x8x64_sf4x(acc[mi][ni], afrag[1][mi][0],
                                       afrag[1][mi][1], afrag[1][mi][2],
                                       afrag[1][mi][3], bfrag[1][ni][0],
                                       bfrag[1][ni][1], sfa[1][mi], sfb[1][ni]);
            }
        }
    }

    float alpha = __ldg(global_sf);
#pragma unroll
    for (int mi = 0; mi < M_ITERS; ++mi) {
#pragma unroll
        for (int ni = 0; ni < N_ITERS; ++ni) {
            int r0 = tm_base + wm * 16 + mi * 96 + (lane >> 2);
            int c0 = tn_base + wn * 64 + ni * 8 + ((lane & 3) << 1);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int r = r0 + h * 8;
                if (r >= m || c0 >= n) continue;
                float v0 = alpha * acc[mi][ni][2 * h];
                float v1 = alpha * acc[mi][ni][2 * h + 1];
                if (c0 + 1 < n) {
                    __nv_bfloat162 pair =
                        __floats2bfloat162_rn(v0, v1);
                    *(__nv_bfloat162*)(d + (size_t)r * n + c0) = pair;
                } else {
                    d[(size_t)r * n + c0] = __float2bfloat16_rn(v0);
                }
            }
        }
    }
}

}

extern "C" int nv_kernels_nvfp4_gemm_sm120_v192_bf16(
    void* stream, const void* a_fp4, const void* a_sf, const void* b_fp4,
    const void* b_sf, const float* global_sf, void* d_bf16, int m, int n,
    int k) {
    if (m <= 0 || n <= 0 || k < TILE_K_NIBBLES || (k % TILE_K_NIBBLES) != 0) {
        return -3;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    static int attr_rc = cudaFuncSetAttribute(
        nvfp4_gemm_v192_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_TOTAL);
    if (attr_rc != cudaSuccess) return -4;
    int tiles_m = (m + TILE_M - 1) / TILE_M;
    int tiles_n = (n + TILE_N - 1) / TILE_N;
    nvfp4_gemm_v192_kernel<<<tiles_m * tiles_n, THREADS, SMEM_TOTAL, s>>>(
        static_cast<const uint8_t*>(a_fp4), static_cast<const uint8_t*>(a_sf),
        static_cast<const uint8_t*>(b_fp4), static_cast<const uint8_t*>(b_sf),
        global_sf, static_cast<__nv_bfloat16*>(d_bf16), m, n, k, tiles_m);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
