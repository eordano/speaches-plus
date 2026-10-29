#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <mma.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <array>
#include "nv_kernels.h"

namespace gdn_chunk_gemm {

constexpr int SUBCHUNK_64_MATCHES_FLA_CHUNK_SIZE_AND_KEEPS_DECAY_RATIOS_ABOVE_F32_UNDERFLOW = 64;
constexpr int STACK_ROWS_192_IS_STATE_128_PLUS_OUT_64_SO_THE_CARRY_IS_ONE_GEMM_PER_SUBCHUNK =
    192;
constexpr size_t LT_WORKSPACE_BYTES_32M_MATCHES_THE_NV_QUANT_DET_HANDLE = 32u * 1024 * 1024;

struct WsLayout {
    int n;
    size_t cg, kr, qr, attn, rhs, wu, kc, stack_a, aw_fp8, ak_fp8, aw_scale, ak_scale;
    size_t total;
};

WsLayout layout(int T, int H, int HK) {
    const int C = SUBCHUNK_64_MATCHES_FLA_CHUNK_SIZE_AND_KEEPS_DECAY_RATIOS_ABOVE_F32_UNDERFLOW;
    const int R = STACK_ROWS_192_IS_STATE_128_PLUS_OUT_64_SO_THE_CARRY_IS_ONE_GEMM_PER_SUBCHUNK;
    WsLayout l{};
    l.n = (T + C - 1) / C;
    size_t P = (size_t)l.n * C;
    size_t off = 0;
    auto take_f32 = [&off](size_t elems) {
        size_t at = off;
        off += elems;
        return at;
    };
    auto take_bf16 = [&off](size_t elems) {
        size_t at = off;
        off += (elems + 1) / 2;
        return at;
    };
    l.cg = take_f32(P * H);
    l.kr = take_bf16(P * HK * 128);
    l.qr = take_bf16(P * HK * 128);
    l.attn = take_bf16((size_t)l.n * H * C * C);
    l.rhs = take_bf16((size_t)l.n * H * C * 256);
    l.wu = take_bf16((size_t)l.n * H * C * 256);
    l.kc = take_bf16(P * H * 128);
    l.stack_a = take_f32((size_t)l.n * H * R * 128);
    auto take_fp8 = [&off](size_t elems) {
        off = (off + 3) & ~(size_t)3;
        size_t at = off;
        off += (elems + 3) / 4;
        return at;
    };
    l.aw_fp8 = take_fp8((size_t)l.n * H * C * 128);
    l.ak_fp8 = take_fp8((size_t)l.n * H * C * 128);
    l.aw_scale = take_f32((size_t)l.n * H * C);
    l.ak_scale = take_f32((size_t)l.n * H * C);
    l.total = off;
    return l;
}

enum GemmMode {
    GEMM_F32_ALL = 0,
    GEMM_BF16_IN_F32_OUT_TENSOR_CORES_WITH_F32_ACCUM = 1,
    GEMM_BF16_ALL_TENSOR_CORES_WITH_F32_ACCUM = 2,
};

struct PlanKey {
    std::array<int64_t, 13> v;
    bool operator<(const PlanKey& o) const { return v < o.v; }
};

struct Plan {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t algo;
};

static bool prof_armed_nv_gdn_chunk_gemm_prof_1_syncs_every_stage_so_never_in_timed_runs() {
    static int armed = -1;
    if (armed < 0) {
        const char* v = getenv("NV_GDN_CHUNK_GEMM_PROF");
        armed = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return armed == 1;
}

struct StageProf {
    bool on;
    cudaStream_t s;
    std::chrono::steady_clock::time_point t0;
    void lap(const char* name) {
        if (!on) return;
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[gdn-chunk-gemm-prof] %s %.1f us\n", name,
                std::chrono::duration<double, std::micro>(t1 - t0).count());
        t0 = t1;
    }
};

static cublasLtHandle_t g_handle = nullptr;
static std::mutex g_mu;
static std::map<PlanKey, Plan> g_plans;

static bool ckpt_lowp_mma_env_read_per_call_opt_in_the_fp8_mma_two_blocks_per_sm_perf_kernel() {
    const char* v = getenv("NV_Q38_GDN_V3_CKPT_LOWP_MMA");
    return v && strcmp(v, "1") == 0;
}

static bool ckpt_fp8_env_read_per_call_opt_in_native_e4m3_mma_fp8_state_and_operands_no_convert() {
    const char* v = getenv("NV_Q38_GDN_V3_CKPT_FP8");
    return v && strcmp(v, "1") == 0;
}

static bool ckpt_kstream_env_read_per_call_opt_in_bf16_two_blocks_per_sm_one_reused_operand_buffer() {
    const char* v = getenv("NV_Q38_GDN_V3_CKPT_KSTREAM");
    return v && strcmp(v, "1") == 0;
}

static int run_tc_head(
    cudaStream_t s, const WsLayout& l, float* workspace, const float* beta,
    int T, int H, int HK, StageProf* prof
) {
    __nv_bfloat16* kr = (__nv_bfloat16*)(workspace + l.kr);
    __nv_bfloat16* qr = (__nv_bfloat16*)(workspace + l.qr);
    __nv_bfloat16* rhs = (__nv_bfloat16*)(workspace + l.rhs);
    __nv_bfloat16* attn = (__nv_bfloat16*)(workspace + l.attn);
    __nv_bfloat16* wu = (__nv_bfloat16*)(workspace + l.wu);
    float* cg = workspace + l.cg;
    int rc = nv_kernels_gdn_prefill_tc_fused_head(
        (void*)s, (const uint16_t*)kr, (const uint16_t*)qr, (const uint16_t*)rhs, beta, cg,
        (uint16_t*)attn, (uint16_t*)wu, T, H, HK, l.n);
    if (rc != 0) return rc;
    if (prof) prof->lap("tc_fused_head");
    return 0;
}

static cublasStatus_t plan_for(
    int m, int n, int k, bool trans_a_rm, bool trans_b_rm, GemmMode mode,
    int lda, int ldb, int ldd,
    long long sa, long long sb, long long sd, int batch, Plan** out
) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_handle) {
        cublasStatus_t st = cublasLtCreate(&g_handle);
        if (st != CUBLAS_STATUS_SUCCESS) return st;
    }
    PlanKey key{{m, n, k, trans_a_rm ? 1 : 0, trans_b_rm ? 1 : 0, (int64_t)mode, lda, ldb, ldd,
                 sa, sb, sd, batch}};
    auto it = g_plans.find(key);
    if (it != g_plans.end()) {
        *out = &it->second;
        return CUBLAS_STATUS_SUCCESS;
    }
    Plan p{};
    cublasStatus_t st = cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    cublasOperation_t op_a = trans_b_rm ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_a_rm ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    cudaDataType_t ab_ty = mode == GEMM_F32_ALL ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t c_ty =
        mode == GEMM_BF16_ALL_TENSOR_CORES_WITH_F32_ACCUM ? CUDA_R_16BF : CUDA_R_32F;
    int a_rows = trans_b_rm ? k : n;
    int a_cols = trans_b_rm ? n : k;
    int b_rows = trans_a_rm ? m : k;
    int b_cols = trans_a_rm ? k : m;
    st = cublasLtMatrixLayoutCreate(&p.la, ab_ty, a_rows, a_cols, ldb);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    st = cublasLtMatrixLayoutCreate(&p.lb, ab_ty, b_rows, b_cols, lda);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    st = cublasLtMatrixLayoutCreate(&p.lc, c_ty, n, m, ldd);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    if (batch > 1) {
        auto set_batch = [&](cublasLtMatrixLayout_t l, long long stride) {
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                             sizeof(batch));
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &stride, sizeof(stride));
        };
        set_batch(p.la, sb);
        set_batch(p.lb, sa);
        set_batch(p.lc, sd);
    }
    cublasLtMatmulPreference_t pref;
    st = cublasLtMatmulPreferenceCreate(&pref);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    size_t ws = LT_WORKSPACE_BYTES_32M_MATCHES_THE_NV_QUANT_DET_HANDLE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                         sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    st = cublasLtMatmulAlgoGetHeuristic(g_handle, p.desc, p.la, p.lb, p.lc, p.lc, pref, 1,
                                        &heur, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    if (returned < 1) return CUBLAS_STATUS_NOT_SUPPORTED;
    p.algo = heur.algo;
    auto ins = g_plans.emplace(key, p);
    *out = &ins.first->second;
    return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t gemm_rm(
    cudaStream_t stream, float* ltws,
    int m, int n, int k, bool trans_a_rm, bool trans_b_rm, GemmMode mode,
    float alpha, const void* a, int lda, long long sa,
    const void* b, int ldb, long long sb,
    float beta, const void* c, void* d, int ldd, long long sd, int batch
) {
    Plan* p = nullptr;
    cublasStatus_t st =
        plan_for(m, n, k, trans_a_rm, trans_b_rm, mode, lda, ldb, ldd, sa, sb, sd, batch, &p);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    return cublasLtMatmul(g_handle, p->desc, &alpha, b, p->la, a, p->lb, &beta, c, p->lc, d,
                          p->lc, &p->algo, ltws,
                          LT_WORKSPACE_BYTES_32M_MATCHES_THE_NV_QUANT_DET_HANDLE, stream);
}

__global__ void cumsum_log_decay_kernel(
    const float* __restrict__ g_exp, float* __restrict__ cg, int T, int H
) {
    const int C = 64;
    int h = blockIdx.x % H;
    int s = blockIdx.x / H;
    int c = threadIdx.x;
    int t = s * C + c;
    const float LOG_DECAY_FLOOR_NEG80_KEEPS_CG_FINITE_WHEN_G_EXP_UNDERFLOWS_TO_ZERO_BECAUSE_LOGF_0_IS_NEG_INF_AND_INF_MINUS_INF_POISONS_THE_RATIO =
        -80.0f;
    __shared__ float buf[C];
    buf[c] = t < T
        ? fmaxf(
              logf(g_exp[(size_t)t * H + h]),
              LOG_DECAY_FLOOR_NEG80_KEEPS_CG_FINITE_WHEN_G_EXP_UNDERFLOWS_TO_ZERO_BECAUSE_LOGF_0_IS_NEG_INF_AND_INF_MINUS_INF_POISONS_THE_RATIO)
        : 0.f;
    __syncthreads();
    for (int off = 1; off < C; off <<= 1) {
        float add = c >= off ? buf[c - off] : 0.f;
        __syncthreads();
        buf[c] += add;
        __syncthreads();
    }
    cg[((size_t)s * H + h) * C + c] = buf[c];
}

__global__ void materialize_rhs_qb_kc_kernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ mixed, const float* __restrict__ beta,
    const float* __restrict__ cg, __nv_bfloat16* __restrict__ rhs,
    __nv_bfloat16* __restrict__ kc, int T, int H, int HK, int conv_dim, int v_channel_offset,
    __nv_bfloat16* __restrict__ qr, __nv_bfloat16* __restrict__ kr
) {
    const int C = 64;
    int h = blockIdx.x % H;
    int row = blockIdx.x / H;
    int s = row / C;
    int c = row % C;
    int i = threadIdx.x;
    int t = s * C + c;
    size_t sh = (size_t)s * H + h;
    size_t rhs_at = (sh * C + c) * 256;
    size_t kc_at = (sh * C + c) * 128;
    int kh = h / (H / HK);
    const bool kv_group_leader_writes_the_rearranged_qk_once_per_kh = (qr != nullptr) && (h % (H / HK) == 0);
    size_t qr_dst = (((size_t)s * HK + kh) * C + c) * 128 + i;
    if (t >= T) {
        rhs[rhs_at + i] = __float2bfloat16(0.f);
        rhs[rhs_at + 128 + i] = __float2bfloat16(0.f);
        kc[kc_at + i] = __float2bfloat16(0.f);
        if (kv_group_leader_writes_the_rearranged_qk_once_per_kh) {
            qr[qr_dst] = __float2bfloat16(0.f);
            kr[qr_dst] = __float2bfloat16(0.f);
        }
        return;
    }
    float cg_c = cg[sh * C + c];
    float cg_last = cg[sh * C + (C - 1)];
    float bt = beta[(size_t)t * H + h];
    float kv = kn[((size_t)t * HK + kh) * 128 + i];
    float qv = qn[((size_t)t * HK + kh) * 128 + i];
    float vv = __bfloat162float(mixed[(size_t)t * conv_dim + v_channel_offset + (size_t)h * 128 + i]);
    rhs[rhs_at + i] = __float2bfloat16(bt * expf(cg_c) * kv);
    rhs[rhs_at + 128 + i] = __float2bfloat16(bt * vv);
    kc[kc_at + i] = __float2bfloat16(expf(cg_last - cg_c) * kv);
    if (kv_group_leader_writes_the_rearranged_qk_once_per_kh) {
        qr[qr_dst] = __float2bfloat16(qv);
        kr[qr_dst] = __float2bfloat16(kv);
    }
}

constexpr int MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID = 32;
constexpr int MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP = 256;
constexpr int MMA_STATE_LD_40_PADS_THE_32_COL_TILE_SO_FRAGMENT_LOADS_SPREAD_BANKS = 40;
constexpr int MMA_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS = 136;
constexpr int MMA_ATTN_LD_72_PADS_THE_64_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS = 72;

__device__ __forceinline__ void mma_cp_async16(void* dst_smem, const void* src_gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}

__device__ __forceinline__ void mma_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void mma_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

constexpr size_t
    V3_CHECKPOINTS_ALIAS_THE_STACK_A_RESERVATION_STATE_16384_PLUS_VNEW_8192_BF16_IS_HALF_ITS_F32_ROWS =
        (size_t)128 * 128;

__global__ void gdn_lowp_quantize_operands_kernel(
    const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ kc,
    __nv_fp8_e4m3* __restrict__ aw_fp8, __nv_fp8_e4m3* __restrict__ ak_fp8,
    float* __restrict__ aw_scale, float* __restrict__ ak_scale, int H
) {
    const int C = 64;
    const int D = 128;
    const int sc = blockIdx.x;
    const int h = blockIdx.y;
    const size_t sh = (size_t)sc * H + h;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    __shared__ float aws[64];
    __shared__ float aks[64];
    for (int c = wid; c < C; c += 8) {
        float ma = 0.f, mk = 0.f;
        for (int col = lane; col < D; col += 32) {
            ma = fmaxf(ma, fabsf(__bfloat162float(wu[sh * C * 256 + (size_t)c * 256 + col])));
            mk = fmaxf(mk, fabsf(__bfloat162float(kc[(sh * C + c) * D + col])));
        }
        for (int o = 16; o > 0; o >>= 1) {
            ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
            mk = fmaxf(mk, __shfl_xor_sync(0xffffffffu, mk, o));
        }
        if (lane == 0) {
            aws[c] = (ma > 1e-12f) ? 448.f / ma : 1.f;
            aks[c] = (mk > 1e-12f) ? 448.f / mk : 1.f;
        }
    }
    __syncthreads();
    if (tid < C) {
        aw_scale[sh * C + tid] = aws[tid];
        ak_scale[sh * C + tid] = aks[tid];
    }
    for (int i = tid; i < C * D; i += (int)blockDim.x) {
        int c = i / D, col = i % D;
        float wa = __bfloat162float(wu[sh * C * 256 + (size_t)c * 256 + col]) * aws[c];
        float ka = __bfloat162float(kc[(sh * C + c) * D + col]) * aks[c];
        aw_fp8[(sh * C + c) * D + col] = (__nv_fp8_e4m3)fmaxf(-448.f, fminf(448.f, wa));
        ak_fp8[(sh * C + c) * D + col] = (__nv_fp8_e4m3)fmaxf(-448.f, fminf(448.f, ka));
    }
}

constexpr int V3_LOWP_OPERAND_LD_128_FP8_UNPADDED_KEEPS_16B_CP_ASYNC_ALIGNMENT = 128;

__device__ __forceinline__ unsigned lowp_fp8x2_to_half2_bits(unsigned packed16) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed16, __NV_E4M3);
    return *reinterpret_cast<unsigned*>(&hr);
}
__device__ __forceinline__ unsigned lowp_bf16x2_to_half2_bits(unsigned lo_bits, unsigned hi_bits) {
    __nv_bfloat16 lo = *reinterpret_cast<__nv_bfloat16*>(&lo_bits);
    __nv_bfloat16 hi = *reinterpret_cast<__nv_bfloat16*>(&hi_bits);
    __half2 h = __floats2half2_rn(__bfloat162float(lo), __bfloat162float(hi));
    return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ unsigned lowp_scale_half2(unsigned h2bits, float inv0, float inv1) {
    __half2 h = *reinterpret_cast<__half2*>(&h2bits);
    float2 f = __half22float2(h);
    __half2 r = __floats2half2_rn(f.x * inv0, f.y * inv1);
    return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ void lowp_mma_m16n8k16_f16(
    float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void lowp_mma_m16n8k32_e4m3(
    float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned lowp_quant_e4m3x4(float f0, float f1, float f2, float f3) {
    __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f0, f1), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f2, f3), __NV_SATFINITE, __NV_E4M3);
    return (unsigned)lo | ((unsigned)hi << 16);
}

constexpr size_t V3_LOWP_SMEM_BYTES_45K_FP8_OPERANDS_PLUS_BF16_STATE_LET_TWO_BLOCKS_CORESIDE_WITH_THE_CP_ASYNC_DOUBLE_BUFFER_AND_SLD40_KEPT =
    (size_t)(128 * 40 + 64 * 40 + 64 * 40) * sizeof(__nv_bfloat16) +
    (size_t)(64 * 128 + 64 * 128) * sizeof(__nv_fp8_e4m3) +
    (size_t)(2 * 64) * sizeof(float);

__global__ void __launch_bounds__(256, 2) v3_ladder_state_checkpoint_lowp_kernel(
    const __nv_fp8_e4m3* __restrict__ aw_fp8, const __nv_fp8_e4m3* __restrict__ ak_fp8,
    const float* __restrict__ aw_scale, const float* __restrict__ ak_scale,
    const __nv_bfloat16* __restrict__ wu, const float* __restrict__ cg,
    float* __restrict__ state_inout, __nv_bfloat16* __restrict__ schk,
    __nv_bfloat16* __restrict__ vchk, int H, int n
) {
    using namespace nvcuda;
    const int C = 64;
    const int D = 128;
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    const int SLD = MMA_STATE_LD_40_PADS_THE_32_COL_TILE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int AL = V3_LOWP_OPERAND_LD_128_FP8_UNPADDED_KEEPS_16B_CP_ASYNC_ALIGNMENT;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int j0 = tile * TC;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* S = (__nv_bfloat16*)smem_raw;
    __nv_fp8_e4m3* Aw = (__nv_fp8_e4m3*)(S + D * SLD);
    __nv_fp8_e4m3* Ak = Aw + C * AL;
    __nv_bfloat16* Us = (__nv_bfloat16*)(Ak + C * AL);
    __nv_bfloat16* Vn = Us + C * SLD;
    float* awsc = (float*)(Vn + C * SLD);
    float* aksc = awsc + C;

    auto stage_fp8 = [&](__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src) {
        for (int i = tid; i < C * 8; i += (int)blockDim.x) {
            int c = i >> 3;
            int seg = i & 7;
            mma_cp_async16(dst + c * AL + seg * 16, src + (size_t)c * 128 + seg * 16);
        }
    };
    auto stage_us = [&](const __nv_bfloat16* wrow) {
        for (int i = tid; i < C * 4; i += (int)blockDim.x) {
            int c = i >> 2;
            int seg = i & 3;
            mma_cp_async16(Us + c * SLD + seg * 8, wrow + (size_t)c * 256 + D + j0 + seg * 8);
        }
    };
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int j = idx % TC;
        __nv_bfloat16 vb = __float2bfloat16(state_inout[((size_t)h * D + k) * D + j0 + j]);
        S[k * SLD + j] = vb;
        schk[((size_t)h * D + k) * D + j0 + j] = vb;
    }
    {
        stage_fp8(Aw, aw_fp8 + (size_t)h * C * 128);
        stage_us(wu + (size_t)h * C * 256);
        mma_cp_async_commit();
        stage_fp8(Ak, ak_fp8 + (size_t)h * C * 128);
        mma_cp_async_commit();
    }

    const int sm0 = wid * 16;

    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        float b_last = expf(cg[sh * C + (C - 1)]);
        if (tid < C) {
            awsc[tid] = aw_scale[sh * C + tid];
            aksc[tid] = ak_scale[sh * C + tid];
        }
        mma_cp_async_wait<1>();
        __syncthreads();

        {
            const int gid = lane >> 2;
            const int tg = lane & 3;
            const int mt = wid >> 1;
            const int ntb = (wid & 1) * 2;
            const int r0 = mt * 16 + gid;
            const int r8 = r0 + 8;
            float c0[4] = {0.f, 0.f, 0.f, 0.f};
            float c1[4] = {0.f, 0.f, 0.f, 0.f};
            for (int kt = 0; kt < D / 16; ++kt) {
                int kc = kt * 16 + tg * 2;
                unsigned a0 = lowp_fp8x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&Aw[(size_t)r0 * AL + kc]));
                unsigned a2 = lowp_fp8x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&Aw[(size_t)r0 * AL + kc + 8]));
                unsigned a1 = lowp_fp8x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&Aw[(size_t)r8 * AL + kc]));
                unsigned a3 = lowp_fp8x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&Aw[(size_t)r8 * AL + kc + 8]));
                int nn0 = ntb * 8 + gid;
                int nn1 = (ntb + 1) * 8 + gid;
                unsigned b00 = lowp_bf16x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&S[kc * SLD + nn0]),
                    *reinterpret_cast<const unsigned short*>(&S[(kc + 1) * SLD + nn0]));
                unsigned b01 = lowp_bf16x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&S[(kc + 8) * SLD + nn0]),
                    *reinterpret_cast<const unsigned short*>(&S[(kc + 9) * SLD + nn0]));
                unsigned b10 = lowp_bf16x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&S[kc * SLD + nn1]),
                    *reinterpret_cast<const unsigned short*>(&S[(kc + 1) * SLD + nn1]));
                unsigned b11 = lowp_bf16x2_to_half2_bits(
                    *reinterpret_cast<const unsigned short*>(&S[(kc + 8) * SLD + nn1]),
                    *reinterpret_cast<const unsigned short*>(&S[(kc + 9) * SLD + nn1]));
                lowp_mma_m16n8k16_f16(c0, a0, a1, a2, a3, b00, b01);
                lowp_mma_m16n8k16_f16(c1, a0, a1, a2, a3, b10, b11);
            }
            float inv0 = 1.0f / awsc[r0];
            float inv8 = 1.0f / awsc[r8];
            int co = tg * 2;
            const float* cc[2] = {c0, c1};
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                int nbase = (ntb + nt2) * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = (e < 2) ? r0 : r8;
                    float invv = (e < 2) ? inv0 : inv8;
                    int col = nbase + co + (e & 1);
                    float scrv = cc[nt2][e] * invv;
                    __nv_bfloat16 vb =
                        __float2bfloat16(__bfloat162float(Us[rr * SLD + col]) - scrv);
                    Vn[rr * SLD + col] = vb;
                    vchk[(sh * C + rr) * D + j0 + col] = vb;
                }
            }
        }
        __syncthreads();

        if (sc + 1 < n) {
            size_t sh1 = (size_t)(sc + 1) * H + h;
            stage_fp8(Aw, aw_fp8 + sh1 * C * 128);
            stage_us(wu + sh1 * C * 256);
            mma_cp_async_commit();
            mma_cp_async_wait<1>();
        } else {
            mma_cp_async_wait<0>();
        }
        __syncthreads();

        {
            const int gid = lane >> 2;
            const int tg = lane & 3;
            float cc0[4] = {0.f, 0.f, 0.f, 0.f};
            float cc1[4] = {0.f, 0.f, 0.f, 0.f};
            float cc2[4] = {0.f, 0.f, 0.f, 0.f};
            float cc3[4] = {0.f, 0.f, 0.f, 0.f};
            float* ccs[4] = {cc0, cc1, cc2, cc3};
            const unsigned char* Akb = reinterpret_cast<const unsigned char*>(Ak);
            for (int kt = 0; kt < C / 16; ++kt) {
                int cc = kt * 16 + tg * 2;
                int d0 = sm0 + gid;
                int d8 = d0 + 8;
                auto la = [&](int cidx, int d) -> unsigned {
                    unsigned raw = (unsigned)Akb[(size_t)(cidx + 1) * AL + d] << 8 |
                                   (unsigned)Akb[(size_t)cidx * AL + d];
                    return lowp_scale_half2(lowp_fp8x2_to_half2_bits(raw), 1.f / aksc[cidx],
                                            1.f / aksc[cidx + 1]);
                };
                unsigned a0 = la(cc, d0);
                unsigned a1 = la(cc, d8);
                unsigned a2 = la(cc + 8, d0);
                unsigned a3 = la(cc + 8, d8);
                for (int nt2 = 0; nt2 < 4; ++nt2) {
                    int n = nt2 * 8 + gid;
                    unsigned b0 = lowp_bf16x2_to_half2_bits(
                        *reinterpret_cast<const unsigned short*>(&Vn[cc * SLD + n]),
                        *reinterpret_cast<const unsigned short*>(&Vn[(cc + 1) * SLD + n]));
                    unsigned b1 = lowp_bf16x2_to_half2_bits(
                        *reinterpret_cast<const unsigned short*>(&Vn[(cc + 8) * SLD + n]),
                        *reinterpret_cast<const unsigned short*>(&Vn[(cc + 9) * SLD + n]));
                    lowp_mma_m16n8k16_f16(ccs[nt2], a0, a1, a2, a3, b0, b1);
                }
            }
            int co = tg * 2;
            for (int nt2 = 0; nt2 < 4; ++nt2) {
                int nbase = nt2 * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = sm0 + gid + ((e < 2) ? 0 : 8);
                    int jj = nbase + co + (e & 1);
                    float snew = __bfloat162float(S[rr * SLD + jj]) * b_last + ccs[nt2][e];
                    S[rr * SLD + jj] = __float2bfloat16(snew);
                }
            }
            __syncwarp();
        }
        for (int idx = lane; idx < 16 * TC; idx += 32) {
            int r = idx / TC;
            int j = idx % TC;
            if (sc + 1 < n) {
                schk[(((size_t)(sc + 1) * H + h) * D + sm0 + r) * D + j0 + j] =
                    S[(sm0 + r) * SLD + j];
            }
        }
        __syncthreads();
        if (sc + 1 < n) {
            stage_fp8(Ak, ak_fp8 + ((size_t)(sc + 1) * H + h) * C * 128);
            mma_cp_async_commit();
        }
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int j = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + j] = __bfloat162float(S[k * SLD + j]);
    }
}

constexpr size_t V3_FP8_SMEM_BYTES_44K_FP8_STATE_AND_OPERANDS_THROUGH_THE_NATIVE_E4M3_MMA_NO_CONVERT =
    (size_t)(128 * 40 + 64 * 40 + 64 * 40) * sizeof(__nv_bfloat16) +
    (size_t)(128 * 40 + 64 * 128 + 64 * 128 + 64 * 40) * sizeof(__nv_fp8_e4m3) +
    (size_t)(2 * 64 + 8) * sizeof(float);

__device__ __forceinline__ float lowp_block_absmax(float v, float* red, int tid) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    float m = 0.f;
    for (int w = 0; w < 8; ++w) m = fmaxf(m, red[w]);
    return m;
}

__global__ void __launch_bounds__(256, 2) v3_ladder_state_checkpoint_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ aw_fp8, const __nv_fp8_e4m3* __restrict__ ak_fp8,
    const float* __restrict__ aw_scale, const float* __restrict__ ak_scale,
    const __nv_bfloat16* __restrict__ wu, const float* __restrict__ cg,
    float* __restrict__ state_inout, __nv_bfloat16* __restrict__ schk,
    __nv_bfloat16* __restrict__ vchk, int H, int n
) {
    const int C = 64;
    const int D = 128;
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    const int SLD = MMA_STATE_LD_40_PADS_THE_32_COL_TILE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int AL = V3_LOWP_OPERAND_LD_128_FP8_UNPADDED_KEEPS_16B_CP_ASYNC_ALIGNMENT;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int j0 = tile * TC;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* S = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Us = S + D * SLD;
    __nv_bfloat16* Vn = Us + C * SLD;
    __nv_fp8_e4m3* Sf8 = (__nv_fp8_e4m3*)(Vn + C * SLD);
    __nv_fp8_e4m3* Aw = Sf8 + D * SLD;
    __nv_fp8_e4m3* Ak = Aw + C * AL;
    __nv_fp8_e4m3* Vf8 = Ak + C * AL;
    float* awsc = (float*)(Vf8 + C * SLD);
    float* aksc = awsc + C;
    float* red = aksc + C;

    auto stage_fp8 = [&](__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src) {
        for (int i = tid; i < C * 8; i += (int)blockDim.x) {
            int c = i >> 3, seg = i & 7;
            mma_cp_async16(dst + c * AL + seg * 16, src + (size_t)c * 128 + seg * 16);
        }
    };
    auto stage_us = [&](const __nv_bfloat16* wrow) {
        for (int i = tid; i < C * 4; i += (int)blockDim.x) {
            int c = i >> 2, seg = i & 3;
            mma_cp_async16(Us + c * SLD + seg * 8, wrow + (size_t)c * 256 + D + j0 + seg * 8);
        }
    };

    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC, j = idx % TC;
        __nv_bfloat16 vb = __float2bfloat16(state_inout[((size_t)h * D + k) * D + j0 + j]);
        S[k * SLD + j] = vb;
        schk[((size_t)h * D + k) * D + j0 + j] = vb;
    }
    {
        stage_fp8(Aw, aw_fp8 + (size_t)h * C * 128);
        stage_us(wu + (size_t)h * C * 256);
        mma_cp_async_commit();
        stage_fp8(Ak, ak_fp8 + (size_t)h * C * 128);
        mma_cp_async_commit();
    }

    const int sm0 = wid * 16;
    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        float b_last = expf(cg[sh * C + (C - 1)]);
        if (tid < C) {
            awsc[tid] = aw_scale[sh * C + tid];
            aksc[tid] = ak_scale[sh * C + tid];
        }
        mma_cp_async_wait<1>();
        __syncthreads();

        float smax = 0.f;
        for (int idx = tid; idx < D * TC; idx += blockDim.x)
            smax = fmaxf(smax, fabsf(__bfloat162float(S[(idx / TC) * SLD + idx % TC])));
        smax = lowp_block_absmax(smax, red, tid);
        float sscale = (smax > 1e-12f) ? 448.f / smax : 1.f;
        float sinv = 1.f / sscale;
        __syncthreads();
        for (int idx = tid; idx < D * TC; idx += blockDim.x) {
            int k = idx / TC, j = idx % TC;
            Sf8[k * SLD + j] = (__nv_fp8_e4m3)(__bfloat162float(S[k * SLD + j]) * sscale);
        }
        __syncthreads();

        {
            const int mt = wid >> 1;
            const int ntb = (wid & 1) * 2;
            const int r0 = mt * 16 + gid, r8 = r0 + 8;
            float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
            for (int kt = 0; kt < D / 32; ++kt) {
                int kk = kt * 32 + tg * 4;
                unsigned a0 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r0 * AL + kk]);
                unsigned a1 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r8 * AL + kk]);
                unsigned a2 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r0 * AL + kk + 16]);
                unsigned a3 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r8 * AL + kk + 16]);
                auto lb = [&](int nn) -> uint2 {
                    const unsigned char* p = reinterpret_cast<const unsigned char*>(Sf8);
                    unsigned lo = 0, hi = 0;
                    for (int e = 0; e < 4; ++e) {
                        lo |= (unsigned)p[(size_t)(kk + e) * SLD + nn] << (8 * e);
                        hi |= (unsigned)p[(size_t)(kk + 16 + e) * SLD + nn] << (8 * e);
                    }
                    return make_uint2(lo, hi);
                };
                uint2 b0v = lb(ntb * 8 + gid);
                uint2 b1v = lb((ntb + 1) * 8 + gid);
                lowp_mma_m16n8k32_e4m3(c0, a0, a1, a2, a3, b0v.x, b0v.y);
                lowp_mma_m16n8k32_e4m3(c1, a0, a1, a2, a3, b1v.x, b1v.y);
            }
            float inv0 = sinv / awsc[r0], inv8 = sinv / awsc[r8];
            const float* cc[2] = {c0, c1};
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                int nbase = (ntb + nt2) * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = (e < 2) ? r0 : r8;
                    float invv = (e < 2) ? inv0 : inv8;
                    int col = nbase + tg * 2 + (e & 1);
                    __nv_bfloat16 vb = __float2bfloat16(__bfloat162float(Us[rr * SLD + col]) -
                                                        cc[nt2][e] * invv);
                    Vn[rr * SLD + col] = vb;
                    vchk[(sh * C + rr) * D + j0 + col] = vb;
                }
            }
        }
        __syncthreads();

        if (sc + 1 < n) {
            size_t sh1 = (size_t)(sc + 1) * H + h;
            stage_fp8(Aw, aw_fp8 + sh1 * C * 128);
            stage_us(wu + sh1 * C * 256);
            mma_cp_async_commit();
            mma_cp_async_wait<1>();
        } else {
            mma_cp_async_wait<0>();
        }

        float vmax = 0.f;
        for (int idx = tid; idx < C * TC; idx += blockDim.x) {
            int c = idx / TC, j = idx % TC;
            vmax = fmaxf(vmax, fabsf(__bfloat162float(Vn[c * SLD + j]) / aksc[c]));
        }
        vmax = lowp_block_absmax(vmax, red, tid);
        float vscale = (vmax > 1e-12f) ? 448.f / vmax : 1.f;
        float vinv = 1.f / vscale;
        __syncthreads();
        for (int idx = tid; idx < C * TC; idx += blockDim.x) {
            int c = idx / TC, j = idx % TC;
            Vf8[c * SLD + j] = (__nv_fp8_e4m3)(__bfloat162float(Vn[c * SLD + j]) / aksc[c] * vscale);
        }
        __syncthreads();

        {
            float cc0[4] = {0, 0, 0, 0}, cc1[4] = {0, 0, 0, 0};
            float cc2[4] = {0, 0, 0, 0}, cc3[4] = {0, 0, 0, 0};
            float* ccs[4] = {cc0, cc1, cc2, cc3};
            const unsigned char* Akb = reinterpret_cast<const unsigned char*>(Ak);
            const unsigned char* Vb = reinterpret_cast<const unsigned char*>(Vf8);
            for (int kt = 0; kt < C / 32; ++kt) {
                int cbase = kt * 32 + tg * 4;
                int d0 = sm0 + gid, d8 = d0 + 8;
                auto la = [&](int d) -> uint2 {
                    unsigned lo = 0, hi = 0;
                    for (int e = 0; e < 4; ++e) {
                        lo |= (unsigned)Akb[(size_t)(cbase + e) * AL + d] << (8 * e);
                        hi |= (unsigned)Akb[(size_t)(cbase + 16 + e) * AL + d] << (8 * e);
                    }
                    return make_uint2(lo, hi);
                };
                uint2 a0v = la(d0);
                uint2 a1v = la(d8);
                for (int nt2 = 0; nt2 < 4; ++nt2) {
                    int nn = nt2 * 8 + gid;
                    unsigned lo = 0, hi = 0;
                    for (int e = 0; e < 4; ++e) {
                        lo |= (unsigned)Vb[(size_t)(cbase + e) * SLD + nn] << (8 * e);
                        hi |= (unsigned)Vb[(size_t)(cbase + 16 + e) * SLD + nn] << (8 * e);
                    }
                    lowp_mma_m16n8k32_e4m3(ccs[nt2], a0v.x, a1v.x, a0v.y, a1v.y, lo, hi);
                }
            }
            for (int nt2 = 0; nt2 < 4; ++nt2) {
                int nbase = nt2 * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = sm0 + gid + ((e < 2) ? 0 : 8);
                    int jj = nbase + tg * 2 + (e & 1);
                    float snew = __bfloat162float(S[rr * SLD + jj]) * b_last + ccs[nt2][e] * vinv;
                    S[rr * SLD + jj] = __float2bfloat16(snew);
                }
            }
            __syncwarp();
        }
        for (int idx = lane; idx < 16 * TC; idx += 32) {
            int r = idx / TC, j = idx % TC;
            if (sc + 1 < n)
                schk[(((size_t)(sc + 1) * H + h) * D + sm0 + r) * D + j0 + j] = S[(sm0 + r) * SLD + j];
        }
        __syncthreads();
        if (sc + 1 < n) {
            stage_fp8(Ak, ak_fp8 + ((size_t)(sc + 1) * H + h) * C * 128);
            mma_cp_async_commit();
        }
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC, j = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + j] = __bfloat162float(S[k * SLD + j]);
    }
}

__device__ __forceinline__ unsigned pack_bf16x2(unsigned short lo, unsigned short hi) {
    return (unsigned)lo | ((unsigned)hi << 16);
}
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr size_t V3_BF16_SMEM_BYTES_54K_FULL_BF16_OPERANDS_NATIVE_MMA_NO_CONVERT_NO_PREPASS_ONE_BLOCK_PER_SM_ISOLATES_THE_M16N8K16_HALF_WIDTH_PER_STEP_COST =
    (size_t)(128 * 40 + 2 * 64 * 136 + 64 * 40 + 64 * 40) * sizeof(__nv_bfloat16);

__global__ void v3_ladder_state_checkpoint_bf16_kernel(
    const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ kc,
    const float* __restrict__ cg, float* __restrict__ state_inout,
    __nv_bfloat16* __restrict__ schk, __nv_bfloat16* __restrict__ vchk, int H, int n
) {
    const int C = 64;
    const int D = 128;
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    const int SLD = MMA_STATE_LD_40_PADS_THE_32_COL_TILE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int ALD = MMA_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int j0 = tile * TC;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* S = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Aw = S + D * SLD;
    __nv_bfloat16* Ak = Aw + C * ALD;
    __nv_bfloat16* Us = Ak + C * ALD;
    __nv_bfloat16* Vn = Us + C * SLD;

    auto stage_wide = [&](__nv_bfloat16* dst, const __nv_bfloat16* src_row0, int src_ld) {
        for (int i = tid; i < C * 16; i += (int)blockDim.x) {
            int c = i >> 4, seg = i & 15;
            mma_cp_async16(dst + c * ALD + seg * 8, src_row0 + (size_t)c * src_ld + seg * 8);
        }
    };
    auto stage_us = [&](const __nv_bfloat16* wrow) {
        for (int i = tid; i < C * 4; i += (int)blockDim.x) {
            int c = i >> 2, seg = i & 3;
            mma_cp_async16(Us + c * SLD + seg * 8, wrow + (size_t)c * 256 + D + j0 + seg * 8);
        }
    };
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC, j = idx % TC;
        __nv_bfloat16 vb = __float2bfloat16(state_inout[((size_t)h * D + k) * D + j0 + j]);
        S[k * SLD + j] = vb;
        schk[((size_t)h * D + k) * D + j0 + j] = vb;
    }
    {
        stage_wide(Aw, wu + (size_t)h * C * 256, 256);
        stage_us(wu + (size_t)h * C * 256);
        mma_cp_async_commit();
        stage_wide(Ak, kc + (size_t)h * C * D, D);
        mma_cp_async_commit();
    }

    const int sm0 = wid * 16;

    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        float b_last = expf(cg[sh * C + (C - 1)]);
        mma_cp_async_wait<1>();
        __syncthreads();

        {
            const int gid = lane >> 2;
            const int tg = lane & 3;
            const int mt = wid >> 1;
            const int ntb = (wid & 1) * 2;
            const int r0 = mt * 16 + gid;
            const int r8 = r0 + 8;
            float c0[4] = {0.f, 0.f, 0.f, 0.f};
            float c1[4] = {0.f, 0.f, 0.f, 0.f};
            for (int kt = 0; kt < D / 16; ++kt) {
                int kk = kt * 16 + tg * 2;
                unsigned a0 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r0 * ALD + kk]);
                unsigned a2 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r0 * ALD + kk + 8]);
                unsigned a1 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r8 * ALD + kk]);
                unsigned a3 = *reinterpret_cast<const unsigned*>(&Aw[(size_t)r8 * ALD + kk + 8]);
                int nn0 = ntb * 8 + gid;
                int nn1 = (ntb + 1) * 8 + gid;
                unsigned b00 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[kk * SLD + nn0]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 1) * SLD + nn0]));
                unsigned b01 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 8) * SLD + nn0]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 9) * SLD + nn0]));
                unsigned b10 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[kk * SLD + nn1]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 1) * SLD + nn1]));
                unsigned b11 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 8) * SLD + nn1]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 9) * SLD + nn1]));
                mma_m16n8k16_bf16(c0, a0, a1, a2, a3, b00, b01);
                mma_m16n8k16_bf16(c1, a0, a1, a2, a3, b10, b11);
            }
            int co = tg * 2;
            const float* cc[2] = {c0, c1};
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                int nbase = (ntb + nt2) * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = (e < 2) ? r0 : r8;
                    int col = nbase + co + (e & 1);
                    __nv_bfloat16 vb =
                        __float2bfloat16(__bfloat162float(Us[rr * SLD + col]) - cc[nt2][e]);
                    Vn[rr * SLD + col] = vb;
                    vchk[(sh * C + rr) * D + j0 + col] = vb;
                }
            }
        }
        __syncthreads();

        if (sc + 1 < n) {
            size_t sh1 = (size_t)(sc + 1) * H + h;
            stage_wide(Aw, wu + sh1 * C * 256, 256);
            stage_us(wu + sh1 * C * 256);
            mma_cp_async_commit();
            mma_cp_async_wait<1>();
        } else {
            mma_cp_async_wait<0>();
        }
        __syncthreads();

        {
            const int gid = lane >> 2;
            const int tg = lane & 3;
            float cc0[4] = {0.f, 0.f, 0.f, 0.f};
            float cc1[4] = {0.f, 0.f, 0.f, 0.f};
            float cc2[4] = {0.f, 0.f, 0.f, 0.f};
            float cc3[4] = {0.f, 0.f, 0.f, 0.f};
            float* ccs[4] = {cc0, cc1, cc2, cc3};
            for (int kt = 0; kt < C / 16; ++kt) {
                int ck = kt * 16 + tg * 2;
                int d0 = sm0 + gid;
                int d8 = d0 + 8;
                auto la = [&](int cidx, int d) -> unsigned {
                    return pack_bf16x2(
                        *reinterpret_cast<const unsigned short*>(&Ak[(size_t)cidx * ALD + d]),
                        *reinterpret_cast<const unsigned short*>(&Ak[(size_t)(cidx + 1) * ALD + d]));
                };
                unsigned a0 = la(ck, d0);
                unsigned a1 = la(ck, d8);
                unsigned a2 = la(ck + 8, d0);
                unsigned a3 = la(ck + 8, d8);
                for (int nt2 = 0; nt2 < 4; ++nt2) {
                    int nn = nt2 * 8 + gid;
                    unsigned b0 = pack_bf16x2(
                        *reinterpret_cast<const unsigned short*>(&Vn[ck * SLD + nn]),
                        *reinterpret_cast<const unsigned short*>(&Vn[(ck + 1) * SLD + nn]));
                    unsigned b1 = pack_bf16x2(
                        *reinterpret_cast<const unsigned short*>(&Vn[(ck + 8) * SLD + nn]),
                        *reinterpret_cast<const unsigned short*>(&Vn[(ck + 9) * SLD + nn]));
                    mma_m16n8k16_bf16(ccs[nt2], a0, a1, a2, a3, b0, b1);
                }
            }
            int co = tg * 2;
            for (int nt2 = 0; nt2 < 4; ++nt2) {
                int nbase = nt2 * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = sm0 + gid + ((e < 2) ? 0 : 8);
                    int jj = nbase + co + (e & 1);
                    float snew = __bfloat162float(S[rr * SLD + jj]) * b_last + ccs[nt2][e];
                    S[rr * SLD + jj] = __float2bfloat16(snew);
                }
            }
            __syncwarp();
        }
        for (int idx = lane; idx < 16 * TC; idx += 32) {
            int r = idx / TC, j = idx % TC;
            if (sc + 1 < n)
                schk[(((size_t)(sc + 1) * H + h) * D + sm0 + r) * D + j0 + j] = S[(sm0 + r) * SLD + j];
        }
        __syncthreads();
        if (sc + 1 < n) {
            stage_wide(Ak, kc + ((size_t)(sc + 1) * H + h) * C * D, D);
            mma_cp_async_commit();
        }
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC, j = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + j] = __bfloat162float(S[k * SLD + j]);
    }
}

constexpr size_t V3_KSTREAM_SMEM_BYTES_37K_BF16_STATE_ONE_REUSED_OPERAND_BUFFER_TIME_MULTIPLEXED_AW_THEN_AK_LETS_TWO_BLOCKS_CORESIDE_AND_BANK_THE_STRAGGLER_WAVE =
    (size_t)(128 * 40 + 64 * 136 + 64 * 40 + 64 * 40) * sizeof(__nv_bfloat16);

__global__ void __launch_bounds__(256, 2) v3_ladder_state_checkpoint_kstream_kernel(
    const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ kc,
    const float* __restrict__ cg, float* __restrict__ state_inout,
    __nv_bfloat16* __restrict__ schk, __nv_bfloat16* __restrict__ vchk, int H, int n
) {
    const int C = 64;
    const int D = 128;
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    const int SLD = MMA_STATE_LD_40_PADS_THE_32_COL_TILE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int ALD = MMA_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int j0 = tile * TC;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* S = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Op = S + D * SLD;
    __nv_bfloat16* Us = Op + C * ALD;
    __nv_bfloat16* Vn = Us + C * SLD;

    auto stage_wide = [&](const __nv_bfloat16* src_row0, int src_ld) {
        for (int i = tid; i < C * 16; i += (int)blockDim.x) {
            int c = i >> 4, seg = i & 15;
            mma_cp_async16(Op + c * ALD + seg * 8, src_row0 + (size_t)c * src_ld + seg * 8);
        }
    };
    auto stage_us = [&](const __nv_bfloat16* wrow) {
        for (int i = tid; i < C * 4; i += (int)blockDim.x) {
            int c = i >> 2, seg = i & 3;
            mma_cp_async16(Us + c * SLD + seg * 8, wrow + (size_t)c * 256 + D + j0 + seg * 8);
        }
    };
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC, j = idx % TC;
        __nv_bfloat16 vb = __float2bfloat16(state_inout[((size_t)h * D + k) * D + j0 + j]);
        S[k * SLD + j] = vb;
        schk[((size_t)h * D + k) * D + j0 + j] = vb;
    }

    const int sm0 = wid * 16;
    const int gid = lane >> 2;
    const int tg = lane & 3;

    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        float b_last = expf(cg[sh * C + (C - 1)]);
        const __nv_bfloat16* wrow = wu + sh * C * 256;
        stage_wide(wrow, 256);
        stage_us(wrow);
        mma_cp_async_commit();
        mma_cp_async_wait<0>();
        __syncthreads();

        {
            const int mt = wid >> 1;
            const int ntb = (wid & 1) * 2;
            const int r0 = mt * 16 + gid;
            const int r8 = r0 + 8;
            float c0[4] = {0.f, 0.f, 0.f, 0.f};
            float c1[4] = {0.f, 0.f, 0.f, 0.f};
            for (int kt = 0; kt < D / 16; ++kt) {
                int kk = kt * 16 + tg * 2;
                unsigned a0 = *reinterpret_cast<const unsigned*>(&Op[(size_t)r0 * ALD + kk]);
                unsigned a2 = *reinterpret_cast<const unsigned*>(&Op[(size_t)r0 * ALD + kk + 8]);
                unsigned a1 = *reinterpret_cast<const unsigned*>(&Op[(size_t)r8 * ALD + kk]);
                unsigned a3 = *reinterpret_cast<const unsigned*>(&Op[(size_t)r8 * ALD + kk + 8]);
                int nn0 = ntb * 8 + gid;
                int nn1 = (ntb + 1) * 8 + gid;
                unsigned b00 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[kk * SLD + nn0]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 1) * SLD + nn0]));
                unsigned b01 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 8) * SLD + nn0]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 9) * SLD + nn0]));
                unsigned b10 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[kk * SLD + nn1]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 1) * SLD + nn1]));
                unsigned b11 = pack_bf16x2(
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 8) * SLD + nn1]),
                    *reinterpret_cast<const unsigned short*>(&S[(kk + 9) * SLD + nn1]));
                mma_m16n8k16_bf16(c0, a0, a1, a2, a3, b00, b01);
                mma_m16n8k16_bf16(c1, a0, a1, a2, a3, b10, b11);
            }
            int co = tg * 2;
            const float* cc[2] = {c0, c1};
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                int nbase = (ntb + nt2) * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = (e < 2) ? r0 : r8;
                    int col = nbase + co + (e & 1);
                    __nv_bfloat16 vb =
                        __float2bfloat16(__bfloat162float(Us[rr * SLD + col]) - cc[nt2][e]);
                    Vn[rr * SLD + col] = vb;
                    vchk[(sh * C + rr) * D + j0 + col] = vb;
                }
            }
        }
        __syncthreads();

        stage_wide(kc + sh * C * D, D);
        mma_cp_async_commit();
        mma_cp_async_wait<0>();
        __syncthreads();

        {
            float cc0[4] = {0.f, 0.f, 0.f, 0.f};
            float cc1[4] = {0.f, 0.f, 0.f, 0.f};
            float cc2[4] = {0.f, 0.f, 0.f, 0.f};
            float cc3[4] = {0.f, 0.f, 0.f, 0.f};
            float* ccs[4] = {cc0, cc1, cc2, cc3};
            for (int kt = 0; kt < C / 16; ++kt) {
                int ck = kt * 16 + tg * 2;
                int d0 = sm0 + gid;
                int d8 = d0 + 8;
                auto la = [&](int cidx, int d) -> unsigned {
                    return pack_bf16x2(
                        *reinterpret_cast<const unsigned short*>(&Op[(size_t)cidx * ALD + d]),
                        *reinterpret_cast<const unsigned short*>(&Op[(size_t)(cidx + 1) * ALD + d]));
                };
                unsigned a0 = la(ck, d0);
                unsigned a1 = la(ck, d8);
                unsigned a2 = la(ck + 8, d0);
                unsigned a3 = la(ck + 8, d8);
                for (int nt2 = 0; nt2 < 4; ++nt2) {
                    int nn = nt2 * 8 + gid;
                    unsigned b0 = pack_bf16x2(
                        *reinterpret_cast<const unsigned short*>(&Vn[ck * SLD + nn]),
                        *reinterpret_cast<const unsigned short*>(&Vn[(ck + 1) * SLD + nn]));
                    unsigned b1 = pack_bf16x2(
                        *reinterpret_cast<const unsigned short*>(&Vn[(ck + 8) * SLD + nn]),
                        *reinterpret_cast<const unsigned short*>(&Vn[(ck + 9) * SLD + nn]));
                    mma_m16n8k16_bf16(ccs[nt2], a0, a1, a2, a3, b0, b1);
                }
            }
            int co = tg * 2;
            for (int nt2 = 0; nt2 < 4; ++nt2) {
                int nbase = nt2 * 8;
                for (int e = 0; e < 4; ++e) {
                    int rr = sm0 + gid + ((e < 2) ? 0 : 8);
                    int jj = nbase + co + (e & 1);
                    float snew = __bfloat162float(S[rr * SLD + jj]) * b_last + ccs[nt2][e];
                    S[rr * SLD + jj] = __float2bfloat16(snew);
                }
            }
            __syncwarp();
        }
        for (int idx = lane; idx < 16 * TC; idx += 32) {
            int r = idx / TC, j = idx % TC;
            if (sc + 1 < n)
                schk[(((size_t)(sc + 1) * H + h) * D + sm0 + r) * D + j0 + j] = S[(sm0 + r) * SLD + j];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC, j = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + j] = __bfloat162float(S[k * SLD + j]);
    }
}

constexpr int V3_OUT_SCR_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS = 20;

__global__ void v3_output_from_checkpoints_kernel(
    const __nv_bfloat16* __restrict__ schk, const __nv_bfloat16* __restrict__ vchk,
    const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ qr,
    const float* __restrict__ cg, __nv_bfloat16* __restrict__ out, int T, int H, int HK
) {
    using namespace nvcuda;
    const int C = 64;
    const int D = 128;
    const int BLD = MMA_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int ATLD = MMA_ATTN_LD_72_PADS_THE_64_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int SCRLD = V3_OUT_SCR_LD_20_PADS_THE_16_COL_WARP_TILE_SO_F32_STORES_SPREAD_BANKS;
    const int sc = blockIdx.x;
    const int h = blockIdx.y;
    const int kh = h / (H / HK);
    const size_t sh = (size_t)sc * H + h;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ unsigned char smem_raw[];
    float* ec = (float*)smem_raw;
    float* scrw = ec + C;
    __nv_bfloat16* Sc = (__nv_bfloat16*)(scrw + 8 * 16 * SCRLD);
    __nv_bfloat16* Qe = Sc + D * BLD;
    __nv_bfloat16* Vs = Qe + C * BLD;
    __nv_bfloat16* At = Vs + C * BLD;

    for (int i = tid; i < D * 16; i += (int)blockDim.x) {
        int k = i >> 4;
        int seg = i & 15;
        mma_cp_async16(Sc + k * BLD + seg * 8, schk + (sh * D + k) * D + seg * 8);
    }
    for (int i = tid; i < C * 16; i += (int)blockDim.x) {
        int c = i >> 4;
        int seg = i & 15;
        mma_cp_async16(Qe + c * BLD + seg * 8,
                       qr + (((size_t)sc * HK + kh) * C + c) * D + seg * 8);
        mma_cp_async16(Vs + c * BLD + seg * 8, vchk + (sh * C + c) * D + seg * 8);
    }
    for (int i = tid; i < C * 8; i += (int)blockDim.x) {
        int c = i >> 3;
        int seg = i & 7;
        mma_cp_async16(At + c * ATLD + seg * 8, attn + (sh * C + c) * C + seg * 8);
    }
    if (tid < C) ec[tid] = expf(cg[sh * C + tid]);
    mma_cp_async_commit();
    mma_cp_async_wait<0>();
    __syncthreads();

    for (int idx = tid; idx < C * D; idx += blockDim.x) {
        int c = idx >> 7;
        int d = idx & 127;
        Qe[c * BLD + d] = __float2bfloat16(ec[c] * __bfloat162float(Qe[c * BLD + d]));
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> facc;

    const int m0 = (wid & 3) * 16;
    const int nbase = (wid >> 2) * 16;
    float* myscr = scrw + wid * 16 * SCRLD;
    for (int nt = 0; nt < 4; ++nt) {
        int nc = nbase + nt * 32;
        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt < D / 16; ++kt) {
            wmma::load_matrix_sync(fa, Qe + m0 * BLD + kt * 16, BLD);
            wmma::load_matrix_sync(fb, Sc + kt * 16 * BLD + nc, BLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        for (int kt = 0; kt < C / 16; ++kt) {
            wmma::load_matrix_sync(fa, At + m0 * ATLD + kt * 16, ATLD);
            wmma::load_matrix_sync(fb, Vs + kt * 16 * BLD + nc, BLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        wmma::store_matrix_sync(myscr, facc, SCRLD, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            int r = idx >> 4;
            int cc = idx & 15;
            int t = sc * C + m0 + r;
            if (t < T) {
                out[((size_t)t * H + h) * D + nc + cc] = __float2bfloat16(myscr[r * SCRLD + cc]);
            }
        }
        __syncwarp();
    }
}

constexpr size_t V3_OUT_SMEM_BYTES_87K_ONE_SHOT_STAGE_NO_PIPELINE_BLOCKS_OVERLAP_EACH_OTHER =
    (size_t)(64 + 8 * 16 * 20) * sizeof(float) +
    (size_t)(128 * 136 + 2 * 64 * 136 + 64 * 72) * sizeof(__nv_bfloat16);

static int run_v3_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, float* state_inout, __nv_bfloat16* out,
    int T, int H, int HK, StageProf* prof
) {
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    static std::once_flag smem_once;
    static cudaError_t smem_rc = cudaSuccess;
    std::call_once(smem_once, []() {
        smem_rc = cudaFuncSetAttribute(
            v3_output_from_checkpoints_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_OUT_SMEM_BYTES_87K_ONE_SHOT_STAGE_NO_PIPELINE_BLOCKS_OVERLAP_EACH_OTHER);
        if (smem_rc != cudaSuccess) return;
        smem_rc = cudaFuncSetAttribute(
            v3_ladder_state_checkpoint_lowp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_LOWP_SMEM_BYTES_45K_FP8_OPERANDS_PLUS_BF16_STATE_LET_TWO_BLOCKS_CORESIDE_WITH_THE_CP_ASYNC_DOUBLE_BUFFER_AND_SLD40_KEPT);
        if (smem_rc != cudaSuccess) return;
        smem_rc = cudaFuncSetAttribute(
            v3_ladder_state_checkpoint_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_FP8_SMEM_BYTES_44K_FP8_STATE_AND_OPERANDS_THROUGH_THE_NATIVE_E4M3_MMA_NO_CONVERT);
        if (smem_rc != cudaSuccess) return;
        smem_rc = cudaFuncSetAttribute(
            v3_ladder_state_checkpoint_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_BF16_SMEM_BYTES_54K_FULL_BF16_OPERANDS_NATIVE_MMA_NO_CONVERT_NO_PREPASS_ONE_BLOCK_PER_SM_ISOLATES_THE_M16N8K16_HALF_WIDTH_PER_STEP_COST);
        if (smem_rc != cudaSuccess) return;
        smem_rc = cudaFuncSetAttribute(
            v3_ladder_state_checkpoint_kstream_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_KSTREAM_SMEM_BYTES_37K_BF16_STATE_ONE_REUSED_OPERAND_BUFFER_TIME_MULTIPLEXED_AW_THEN_AK_LETS_TWO_BLOCKS_CORESIDE_AND_BANK_THE_STRAGGLER_WAVE);
    });
    if (smem_rc != cudaSuccess) return (int)smem_rc;
    __nv_bfloat16* attn = (__nv_bfloat16*)(workspace + l.attn);
    __nv_bfloat16* wu = (__nv_bfloat16*)(workspace + l.wu);
    __nv_bfloat16* kc = (__nv_bfloat16*)(workspace + l.kc);
    __nv_bfloat16* qr = (__nv_bfloat16*)(workspace + l.qr);
    float* cg = workspace + l.cg;
    __nv_bfloat16* schk = (__nv_bfloat16*)(workspace + l.stack_a);
    __nv_bfloat16* vchk =
        schk + (size_t)l.n * H * V3_CHECKPOINTS_ALIAS_THE_STACK_A_RESERVATION_STATE_16384_PLUS_VNEW_8192_BF16_IS_HALF_ITS_F32_ROWS;
    dim3 lgrid(128 / TC, H);
    cudaError_t ce;
    if (getenv("NV_Q38_GDN_V3_CKPT_LOWP_MMA_OCC")) {
        int occ_lowp = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ_lowp, v3_ladder_state_checkpoint_lowp_kernel, 256,
            (int)V3_LOWP_SMEM_BYTES_45K_FP8_OPERANDS_PLUS_BF16_STATE_LET_TWO_BLOCKS_CORESIDE_WITH_THE_CP_ASYNC_DOUBLE_BUFFER_AND_SLD40_KEPT);
        int occ_bf16 = 0, occ_ks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ_bf16, v3_ladder_state_checkpoint_bf16_kernel, 256,
            (int)V3_BF16_SMEM_BYTES_54K_FULL_BF16_OPERANDS_NATIVE_MMA_NO_CONVERT_NO_PREPASS_ONE_BLOCK_PER_SM_ISOLATES_THE_M16N8K16_HALF_WIDTH_PER_STEP_COST);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ_ks, v3_ladder_state_checkpoint_kstream_kernel, 256,
            (int)V3_KSTREAM_SMEM_BYTES_37K_BF16_STATE_ONE_REUSED_OPERAND_BUFFER_TIME_MULTIPLEXED_AW_THEN_AK_LETS_TWO_BLOCKS_CORESIDE_AND_BANK_THE_STRAGGLER_WAVE);
        fprintf(stderr, "[v3-lowp-occ] lowp_mma=%d/SM bf16=%d/SM kstream=%d/SM smem_bf16=%zu smem_ks=%zu\n",
                occ_lowp, occ_bf16, occ_ks,
                V3_BF16_SMEM_BYTES_54K_FULL_BF16_OPERANDS_NATIVE_MMA_NO_CONVERT_NO_PREPASS_ONE_BLOCK_PER_SM_ISOLATES_THE_M16N8K16_HALF_WIDTH_PER_STEP_COST,
                V3_KSTREAM_SMEM_BYTES_37K_BF16_STATE_ONE_REUSED_OPERAND_BUFFER_TIME_MULTIPLEXED_AW_THEN_AK_LETS_TWO_BLOCKS_CORESIDE_AND_BANK_THE_STRAGGLER_WAVE);
    }
    const bool ckpt_fp8 =
        ckpt_fp8_env_read_per_call_opt_in_native_e4m3_mma_fp8_state_and_operands_no_convert();
    if (ckpt_kstream_env_read_per_call_opt_in_bf16_two_blocks_per_sm_one_reused_operand_buffer()) {
        v3_ladder_state_checkpoint_kstream_kernel<<<
            lgrid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
            V3_KSTREAM_SMEM_BYTES_37K_BF16_STATE_ONE_REUSED_OPERAND_BUFFER_TIME_MULTIPLEXED_AW_THEN_AK_LETS_TWO_BLOCKS_CORESIDE_AND_BANK_THE_STRAGGLER_WAVE,
            s>>>(wu, kc, cg, state_inout, schk, vchk, H, l.n);
    } else if (ckpt_lowp_mma_env_read_per_call_opt_in_the_fp8_mma_two_blocks_per_sm_perf_kernel() ||
        ckpt_fp8) {
        __nv_fp8_e4m3* aw8 = (__nv_fp8_e4m3*)(workspace + l.aw_fp8);
        __nv_fp8_e4m3* ak8 = (__nv_fp8_e4m3*)(workspace + l.ak_fp8);
        float* awsc = workspace + l.aw_scale;
        float* aksc = workspace + l.ak_scale;
        gdn_lowp_quantize_operands_kernel<<<dim3(l.n, H), 256, 0, s>>>(
            wu, kc, aw8, ak8, awsc, aksc, H);
        ce = cudaGetLastError();
        if (ce != cudaSuccess) return (int)ce;
        if (prof) prof->lap("v3_lowp_quantize");
        if (ckpt_fp8) {
            v3_ladder_state_checkpoint_fp8_kernel<<<
                lgrid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
                V3_FP8_SMEM_BYTES_44K_FP8_STATE_AND_OPERANDS_THROUGH_THE_NATIVE_E4M3_MMA_NO_CONVERT,
                s>>>(aw8, ak8, awsc, aksc, wu, cg, state_inout, schk, vchk, H, l.n);
        } else {
            v3_ladder_state_checkpoint_lowp_kernel<<<
                lgrid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
                V3_LOWP_SMEM_BYTES_45K_FP8_OPERANDS_PLUS_BF16_STATE_LET_TWO_BLOCKS_CORESIDE_WITH_THE_CP_ASYNC_DOUBLE_BUFFER_AND_SLD40_KEPT,
                s>>>(aw8, ak8, awsc, aksc, wu, cg, state_inout, schk, vchk, H, l.n);
        }
    } else {
        v3_ladder_state_checkpoint_bf16_kernel<<<
            lgrid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
            V3_BF16_SMEM_BYTES_54K_FULL_BF16_OPERANDS_NATIVE_MMA_NO_CONVERT_NO_PREPASS_ONE_BLOCK_PER_SM_ISOLATES_THE_M16N8K16_HALF_WIDTH_PER_STEP_COST,
            s>>>(wu, kc, cg, state_inout, schk, vchk, H, l.n);
    }
    ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    if (prof) prof->lap("v3_ladder_state_checkpoint");
    dim3 ogrid(l.n, H);
    v3_output_from_checkpoints_kernel<<<
        ogrid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
        V3_OUT_SMEM_BYTES_87K_ONE_SHOT_STAGE_NO_PIPELINE_BLOCKS_OVERLAP_EACH_OTHER, s>>>(
        schk, vchk, attn, qr, cg, out, T, H, HK);
    ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    if (prof) prof->lap("v3_output_from_checkpoints");
    return 0;
}

}

extern "C" long long nv_kernels_gdn_chunk_gemm_scan_workspace_elems(int T, int H, int HK) {
    if (T <= 0 || H <= 0 || HK <= 0) return 0;
    return (long long)gdn_chunk_gemm::layout(T, H, HK).total;
}

constexpr long long TREE_MAT_ELEMS_128X128_ONE_CHUNK_CARRY_TRANSITION_IS_A_FULL_STATE_SQUARE_MATRIX =
    128 * 128;

extern "C" long long nv_kernels_gdn_scan_tree_compose_floor_workspace_elems(int n, int H) {
    if (n <= 0 || H <= 0) return 0;
    long long mats = (long long)n * H * 2 + (long long)(n / 2 + 1) * H * 2;
    return mats * TREE_MAT_ELEMS_128X128_ONE_CHUNK_CARRY_TRANSITION_IS_A_FULL_STATE_SQUARE_MATRIX / 2;
}

extern "C" int nv_kernels_gdn_scan_tree_compose_floor(
    void* stream, float* workspace, long long workspace_elems, int n, int H
) {
    using namespace gdn_chunk_gemm;
    if (n <= 0 || H <= 0) return -2;
    if (workspace == nullptr ||
        workspace_elems < nv_kernels_gdn_scan_tree_compose_floor_workspace_elems(n, H))
        return -5;
    cudaStream_t s = (cudaStream_t)stream;
    const long long ME =
        TREE_MAT_ELEMS_128X128_ONE_CHUNK_CARRY_TRANSITION_IS_A_FULL_STATE_SQUARE_MATRIX;
    __nv_bfloat16* cur_m = (__nv_bfloat16*)workspace;
    __nv_bfloat16* cur_n = cur_m + (long long)n * H * ME;
    __nv_bfloat16* nxt_m = cur_n + (long long)n * H * ME;
    __nv_bfloat16* nxt_n = nxt_m + (long long)(n / 2 + 1) * H * ME;
    float* ltws = workspace;
    const GemmMode bf_all = GEMM_BF16_ALL_TENSOR_CORES_WITH_F32_ACCUM;
    cublasStatus_t st;
    st = gemm_rm(s, ltws, 128, 128, 64, false, false, bf_all, 1.f, nxt_m, 64, (long long)0,
                 nxt_n, 128, (long long)0, 0.f, cur_m, cur_m, 128, ME, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    st = gemm_rm(s, ltws, 128, 128, 64, false, false, bf_all, 1.f, nxt_m, 64, (long long)0,
                 nxt_n, 128, (long long)0, 0.f, cur_n, cur_n, 128, ME, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    for (int sweep = 0; sweep < 2; ++sweep) {
        __nv_bfloat16* am = cur_m;
        __nv_bfloat16* an = cur_n;
        __nv_bfloat16* bm = nxt_m;
        __nv_bfloat16* bn = nxt_n;
        for (int m = n / 2; m >= 1; m /= 2) {
            st = gemm_rm(s, ltws, 128, 128, 128, false, false, bf_all, 1.f, am + ME, 128,
                         2 * ME, am, 128, 2 * ME, 0.f, bm, bm, 128, ME, m * H);
            if (st != CUBLAS_STATUS_SUCCESS) return -6;
            st = gemm_rm(s, ltws, 128, 128, 128, false, false, bf_all, 1.f, am + ME, 128,
                         2 * ME, an, 128, 2 * ME, 1.f, an + ME, bn, 128, ME, m * H);
            if (st != CUBLAS_STATUS_SUCCESS) return -6;
            __nv_bfloat16* tm = am;
            __nv_bfloat16* tn = an;
            am = bm;
            an = bn;
            bm = tm;
            bn = tn;
        }
    }
    st = gemm_rm(s, ltws, 64, 128, 128, false, false, bf_all, 1.f, nxt_m, 128, ME, cur_m, 128,
                 ME, 0.f, cur_n, cur_n, 128, (long long)(ME / 2), n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    return 0;
}

extern "C" int nv_kernels_gdn_chunk_gemm_scan_f32(
    void* stream,
    const float* qn,
    const float* kn,
    const uint16_t* mixed,
    const float* g_exp,
    const float* beta,
    float* state_inout,
    uint16_t* out,
    float* workspace,
    long long workspace_elems,
    int B,
    int T,
    int H,
    int HK,
    int K,
    int V,
    int conv_dim,
    int v_channel_offset
) {
    using namespace gdn_chunk_gemm;
    cudaStream_t s = (cudaStream_t)stream;
    if (B * T * H == 0) return 0;
    if (K != 128 || V != 128) return -2;
    if (HK <= 0 || H % HK != 0) return -3;
    if (B != 1) return -4;
    WsLayout l = layout(T, H, HK);
    if (workspace == nullptr || (long long)l.total > workspace_elems) return -5;
    const int C = SUBCHUNK_64_MATCHES_FLA_CHUNK_SIZE_AND_KEEPS_DECAY_RATIOS_ABOVE_F32_UNDERFLOW;
    const int n = l.n;
    float* cg = workspace + l.cg;
    __nv_bfloat16* kr = (__nv_bfloat16*)(workspace + l.kr);
    __nv_bfloat16* qr = (__nv_bfloat16*)(workspace + l.qr);
    __nv_bfloat16* rhs = (__nv_bfloat16*)(workspace + l.rhs);
    __nv_bfloat16* kc = (__nv_bfloat16*)(workspace + l.kc);

    StageProf prof{prof_armed_nv_gdn_chunk_gemm_prof_1_syncs_every_stage_so_never_in_timed_runs(),
                   s, std::chrono::steady_clock::now()};
    cumsum_log_decay_kernel<<<n * H, C, 0, s>>>(g_exp, cg, T, H);
    materialize_rhs_qb_kc_kernel<<<n * C * H, 128, 0, s>>>(
        qn, kn, (const __nv_bfloat16*)mixed, beta, cg, rhs, kc, T, H, HK,
        conv_dim, v_channel_offset, qr, kr);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    prof.lap("materialize");

    int rc = run_tc_head(s, l, workspace, beta, T, H, HK, prof.on ? &prof : nullptr);
    if (rc != 0) return rc;
    return run_v3_tail(s, l, workspace, state_inout, (__nv_bfloat16*)out, T, H, HK,
                       prof.on ? &prof : nullptr);
}
