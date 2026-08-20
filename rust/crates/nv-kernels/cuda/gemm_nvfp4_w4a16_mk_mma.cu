#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr int kNvfp4Block = 16;
constexpr int kKTile = 128;
constexpr int kMMax = 16;
constexpr int kXStrideHalves = kKTile + 8;
constexpr int kWStrideBytes = kKTile / 2 + 16;
constexpr int kScTilesPerStage = kKTile / 64;
constexpr int kWarps8NeedsEnoughCtasToCoverAllSms = 132;
#define NV_UE4M3_SUBNORMAL_STEP 0.001953125f

__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
                 "l"(src_gmem),
                 "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float decode_ue4m3_scale(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

__device__ __forceinline__ unsigned decode_fp4_pair_scaled_bf16x2(uint8_t b, float sf) {
    constexpr unsigned kE2m1MagAsE4m3Bytes0to3 = 0x3C383000u;
    constexpr unsigned kE2m1MagAsE4m3Bytes4to7 = 0x4C484440u;
    constexpr unsigned kSignByteTable = 0x00008000u;
    unsigned mag_sel = ((unsigned)b & 0x07u) | ((((unsigned)b >> 4) & 0x07u) << 4);
    unsigned sgn_sel = (((unsigned)b >> 3) & 0x01u) | ((((unsigned)b >> 7) & 0x01u) << 4);
    unsigned e4m3x2 =
        (__byte_perm(kE2m1MagAsE4m3Bytes0to3, kE2m1MagAsE4m3Bytes4to7, mag_sel) |
         __byte_perm(kSignByteTable, 0u, sgn_sel)) &
        0xFFFFu;
    __half2_raw hr =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)e4m3x2, __NV_E4M3);
    float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&hr));
    __nv_bfloat162 packed = __floats2bfloat162_rn(wf.x * sf, wf.y * sf);
    return *reinterpret_cast<unsigned*>(&packed);
}

template <int WARPS, bool DUAL, bool MLE8>
__global__ void __launch_bounds__(WARPS * 32) gemm_nvfp4_w4a16_mk_mma_kernel(
    const uint8_t* __restrict__ wq_a,
    const uint8_t* __restrict__ sc_a,
    const uint8_t* __restrict__ wq_b,
    const uint8_t* __restrict__ sc_b,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y_a,
    __nv_bfloat16* __restrict__ y_b,
    float alpha_a,
    float alpha_b,
    int M,
    int N,
    int K
) {
#if __CUDA_ARCH__ >= 800
    constexpr int kRows = WARPS * 8;
    constexpr int kArms = DUAL ? 2 : 1;
    constexpr int kXRows = MLE8 ? 8 : kMMax;
    constexpr int kNBuf = (DUAL && !MLE8) ? 2 : 3;
    constexpr int kPrefetch = kNBuf - 1;
    __shared__ __align__(16) __nv_bfloat16 xs[kNBuf][kXRows * kXStrideHalves];
    __shared__ __align__(16) uint8_t ws_tile[kNBuf][kArms * kRows * kWStrideBytes];
    __shared__ __align__(16) uint8_t scs[kNBuf][kArms * kScTilesPerStage * 512];

    const uint8_t* wq[2] = {wq_a, wq_b};
    const uint8_t* sc[2] = {sc_a, sc_b};

    int n_base = (int)blockIdx.x * kRows;
    int lane = threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int tg = lane & 3;
    int gid = lane >> 2;
    int nthreads = WARPS * 32;

    int kb_total = K / kNvfp4Block;
    int k_tiles_sc = (kb_total + 3) >> 2;
    int m_tile = n_base >> 7;

    float acc[kArms][4];
    #pragma unroll
    for (int a = 0; a < kArms; ++a)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[a][i] = 0.0f;

    int r_mine = n_base + warp * 8 + gid;
    int sc_local_row = (r_mine & 31) * 16 + ((r_mine >> 5) & 3) * 4;

    auto stage_load = [&](int buf, int k0) {
        int kt = min(kKTile, K - k0);
        constexpr int kXSegsPerRow = kKTile / 8;
        for (int i = threadIdx.x; i < kXRows * kXSegsPerRow; i += nthreads) {
            int row = i / kXSegsPerRow;
            int seg = i % kXSegsPerRow;
            int valid = (row < M && seg * 8 < kt) ? 16 : 0;
            const __nv_bfloat16* src =
                valid ? (x + (size_t)row * K + k0 + seg * 8) : x;
            cp_async16(&xs[buf][row * kXStrideHalves + seg * 8], src, valid);
        }
        constexpr int kWSegsPerRow = kKTile / 32;
        for (int i = threadIdx.x; i < kArms * kRows * kWSegsPerRow; i += nthreads) {
            int arm = i / (kRows * kWSegsPerRow);
            int rem = i % (kRows * kWSegsPerRow);
            int row = rem / kWSegsPerRow;
            int seg = rem % kWSegsPerRow;
            int valid = (seg * 16 < kt / 2) ? 16 : 0;
            const uint8_t* src =
                valid ? (wq[arm] + (size_t)(n_base + row) * (K / 2) + k0 / 2 + seg * 16)
                      : wq[arm];
            cp_async16(
                &ws_tile[buf][(arm * kRows + row) * kWStrideBytes + seg * 16],
                src,
                valid);
        }
        int sc_tile0 = (k0 / kNvfp4Block) >> 2;
        for (int i = threadIdx.x; i < kArms * kScTilesPerStage * 32; i += nthreads) {
            int arm = i / (kScTilesPerStage * 32);
            int rem = i % (kScTilesPerStage * 32);
            int tile = rem / 32;
            int seg = rem % 32;
            int valid = (sc_tile0 + tile < k_tiles_sc) ? 16 : 0;
            const uint8_t* src =
                valid
                    ? (sc[arm] +
                       ((size_t)m_tile * k_tiles_sc + sc_tile0 + tile) * 512 +
                       seg * 16)
                    : sc[arm];
            cp_async16(
                &scs[buf][(arm * kScTilesPerStage + tile) * 512 + seg * 16],
                src,
                valid);
        }
    };

    int nstages = (K + kKTile - 1) / kKTile;
    for (int p = 0; p < kPrefetch && p < nstages; ++p) {
        stage_load(p % kNBuf, p * kKTile);
        cp_async_commit();
    }

    for (int s = 0; s < nstages; ++s) {
        int k0 = s * kKTile;
        int buf = s % kNBuf;
        int ahead = s + kPrefetch;
        if (ahead < nstages) {
            stage_load(ahead % kNBuf, ahead * kKTile);
            cp_async_commit();
        }
        int outstanding_beyond_s = min(ahead, nstages - 1) - s;
        if (outstanding_beyond_s >= 2) {
            cp_async_wait<2>();
        } else if (outstanding_beyond_s == 1) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        int kt = min(kKTile, K - k0);
        int chunks = kt / kNvfp4Block;
        const __nv_bfloat16* xrow0 = &xs[buf][gid * kXStrideHalves + tg * 2];
        const __nv_bfloat16* xrow1 =
            MLE8 ? xrow0 : &xs[buf][(gid + 8) * kXStrideHalves + tg * 2];
        const uint8_t* wrow0 = &ws_tile[buf][(warp * 8 + gid) * kWStrideBytes];
        const uint8_t* scbuf0 = &scs[buf][sc_local_row];
        for (int c = 0; c < chunks; ++c) {
            int kc = c * kNvfp4Block;
            unsigned a0 = *reinterpret_cast<const unsigned*>(xrow0 + kc);
            unsigned a2 = *reinterpret_cast<const unsigned*>(xrow0 + kc + 8);
            unsigned a1 = MLE8 ? 0u : *reinterpret_cast<const unsigned*>(xrow1 + kc);
            unsigned a3 =
                MLE8 ? 0u : *reinterpret_cast<const unsigned*>(xrow1 + kc + 8);
            int sc_off = (c >> 2) * 512 + (c & 3);
            #pragma unroll
            for (int a = 0; a < kArms; ++a) {
                const uint8_t* wrow = wrow0 + a * kRows * kWStrideBytes;
                float sf = decode_ue4m3_scale(
                    scbuf0[a * kScTilesPerStage * 512 + sc_off]);
                unsigned b0 = decode_fp4_pair_scaled_bf16x2(wrow[kc / 2 + tg], sf);
                unsigned b1 =
                    decode_fp4_pair_scaled_bf16x2(wrow[kc / 2 + tg + 4], sf);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[a][0]), "+f"(acc[a][1]), "+f"(acc[a][2]),
                      "+f"(acc[a][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    int ncol = n_base + warp * 8 + tg * 2;
    if (ncol + 1 < N) {
        __nv_bfloat16* ys[2] = {y_a, y_b};
        float alphas[2] = {alpha_a, alpha_b};
        #pragma unroll
        for (int a = 0; a < kArms; ++a) {
            __nv_bfloat16* y = ys[a];
            float alpha = alphas[a];
            if (gid < M) {
                y[(size_t)gid * N + ncol] = __float2bfloat16(acc[a][0] * alpha);
                y[(size_t)gid * N + ncol + 1] =
                    __float2bfloat16(acc[a][1] * alpha);
            }
            if (!MLE8 && gid + 8 < M) {
                y[(size_t)(gid + 8) * N + ncol] =
                    __float2bfloat16(acc[a][2] * alpha);
                y[(size_t)(gid + 8) * N + ncol + 1] =
                    __float2bfloat16(acc[a][3] * alpha);
            }
        }
    }
#endif
}

template <int WARPS, bool DUAL, bool MLE8>
int launch_mk_mma(
    cudaStream_t stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const uint16_t* x,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int M,
    int N,
    int K
) {
    int rows = WARPS * 8;
    dim3 grid((unsigned)(N / rows), 1u);
    gemm_nvfp4_w4a16_mk_mma_kernel<WARPS, DUAL, MLE8>
        <<<grid, dim3(WARPS * 32), 0, stream>>>(
            wq_a, sc_a, wq_b, sc_b,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y_a),
            reinterpret_cast<__nv_bfloat16*>(y_b),
            alpha_a, alpha_b, M, N, K);
    return (int)cudaGetLastError();
}

template <bool DUAL, bool MLE8>
int launch_mk_mma_pick_warps(
    cudaStream_t stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const uint16_t* x,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int M,
    int N,
    int K
) {
    if (N % 64 == 0 && N / 64 >= kWarps8NeedsEnoughCtasToCoverAllSms) {
        return launch_mk_mma<8, DUAL, MLE8>(
            stream, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, K);
    }
    if (N % 32 == 0) {
        return launch_mk_mma<4, DUAL, MLE8>(
            stream, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, K);
    }
    return -1;
}

}

extern "C" int nv_kernels_gemm_nvfp4_w4a16_mk_dual(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const uint16_t* x,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int M,
    int N,
    int K
) {
    if (M <= 0 || M > kMMax) return -1;
    if (N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    if ((wq_b == nullptr) != (y_b == nullptr)) return -1;
    if ((wq_b == nullptr) != (sc_b == nullptr)) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    bool dual = wq_b != nullptr;
    bool mle8 = M <= 8;
    if (dual && mle8) {
        return launch_mk_mma_pick_warps<true, true>(
            s, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, K);
    }
    if (dual) {
        return launch_mk_mma_pick_warps<true, false>(
            s, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, K);
    }
    if (mle8) {
        return launch_mk_mma_pick_warps<false, true>(
            s, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, K);
    }
    return launch_mk_mma_pick_warps<false, false>(
        s, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, K);
}
