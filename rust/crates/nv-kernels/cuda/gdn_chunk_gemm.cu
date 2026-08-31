#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
    size_t cg, kr, qr, kk, qk, amat, tinv, attn, rhs, wu, kc, stack_a, stack_c, stack_d, sbeta,
        sstate, ltws;
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
    l.kk = take_f32((size_t)l.n * HK * C * C);
    l.qk = take_f32((size_t)l.n * HK * C * C);
    l.amat = take_f32((size_t)l.n * H * C * C);
    l.tinv = take_bf16((size_t)l.n * H * C * C);
    l.attn = take_bf16((size_t)l.n * H * C * C);
    l.rhs = take_bf16((size_t)l.n * H * C * 256);
    l.wu = take_bf16((size_t)l.n * H * C * 256);
    l.kc = take_bf16(P * H * 128);
    l.stack_a = take_f32((size_t)l.n * H * R * 128);
    l.stack_c = take_f32((size_t)l.n * H * R * 128);
    l.stack_d = take_f32((size_t)l.n * H * R * 128);
    l.sbeta = take_f32(P * H);
    l.sstate = take_f32((size_t)H * 128 * 128);
    l.ltws = take_f32(LT_WORKSPACE_BYTES_32M_MATCHES_THE_NV_QUANT_DET_HANDLE / sizeof(float));
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

static bool fwdh_env_read_per_call_so_one_process_can_ab_stack_carry_vs_state_resident() {
    const char*
        FWDH_REPLACES_THE_STACK_CARRY_WITH_ONE_STATE_RESIDENT_KERNEL_BECAUSE_THE_STACKS_COST_10X_SCAN_TRAFFIC =
            "NV_Q38_GDN_CHUNK_GEMM_FWDH";
    const char* v = getenv(
        FWDH_REPLACES_THE_STACK_CARRY_WITH_ONE_STATE_RESIDENT_KERNEL_BECAUSE_THE_STACKS_COST_10X_SCAN_TRAFFIC);
    return v && strcmp(v, "1") == 0;
}

static bool mma_env_read_per_call_so_one_process_can_ab_fragment_tail_vs_the_scalar_tails() {
    const char*
        MMA_REPLACES_THE_SCALAR_FMA_SUBCHUNK_GEMMS_WITH_BF16_FRAGMENT_TENSOR_CORE_TILES_THE_ONE_CAUSE_LEFT_BY_THE_THREE_BANKED_NEGATIVES =
            "NV_Q38_GDN_CHUNK_GEMM_MMA";
    const char* v = getenv(
        MMA_REPLACES_THE_SCALAR_FMA_SUBCHUNK_GEMMS_WITH_BF16_FRAGMENT_TENSOR_CORE_TILES_THE_ONE_CAUSE_LEFT_BY_THE_THREE_BANKED_NEGATIVES);
    return v && strcmp(v, "1") == 0;
}

static bool v3_env_read_per_call_so_one_process_can_ab_split_ladder_vs_inline_out_tails() {
    const char*
        V3_DEFAULT_ON_THE_CHAMPION_TAIL_ROWS_Q38_TAILCHAMP_10_OF_10_P_0_002_OVER_MMA_AND_ZERO_RESTORES_THE_OPT_IN_MMA_FWDH_GRAPH_SCALAR_SELECTION =
            "NV_Q38_GDN_CHUNK_GEMM_V3";
    const char* v = getenv(
        V3_DEFAULT_ON_THE_CHAMPION_TAIL_ROWS_Q38_TAILCHAMP_10_OF_10_P_0_002_OVER_MMA_AND_ZERO_RESTORES_THE_OPT_IN_MMA_FWDH_GRAPH_SCALAR_SELECTION);
    return !(v && strcmp(v, "0") == 0);
}

static bool graph_env_read_per_call_so_one_process_can_ab_replay_vs_uncaptured() {
    const char*
        GRAPH_REPLAYS_THE_TAIL_IN_ONE_LAUNCH_BECAUSE_THE_78_LAUNCH_CHAIN_IS_SUBMISSION_BOUND =
            "NV_Q38_GDN_CHUNK_GEMM_GRAPH";
    const char* v = getenv(
        GRAPH_REPLAYS_THE_TAIL_IN_ONE_LAUNCH_BECAUSE_THE_78_LAUNCH_CHAIN_IS_SUBMISSION_BOUND);
    return v && strcmp(v, "1") == 0;
}

enum GraphState {
    WARMUP_RAN_UNCAPTURED_SO_LT_PLANS_AND_MODULES_EXIST_BEFORE_CAPTURE = 0,
    GRAPH_READY = 1,
    CAPTURE_FAILED_STAY_UNCAPTURED = 2,
};

struct GraphKey {
    std::array<int64_t, 4> v;
    bool operator<(const GraphKey& o) const { return v < o.v; }
};

struct GraphEntry {
    GraphState state;
    cudaGraphExec_t exec;
};

static std::mutex g_graph_mu;
static std::map<GraphKey, GraphEntry> g_graphs;
static cudaStream_t
    g_capture_stream_side_allocated_because_the_callers_legacy_stream_cannot_begin_capture =
        nullptr;
static std::atomic<long long> g_graph_replays{0};

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

__global__ void rearrange_qk_padded_kernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    __nv_bfloat16* __restrict__ qr, __nv_bfloat16* __restrict__ kr, int T, int HK
) {
    const int C = 64;
    int kh = blockIdx.x % HK;
    int row = blockIdx.x / HK;
    int s = row / C;
    int c = row % C;
    int i = threadIdx.x;
    int t = s * C + c;
    size_t dst = (((size_t)s * HK + kh) * C + c) * 128 + i;
    if (t < T) {
        size_t src = ((size_t)t * HK + kh) * 128 + i;
        qr[dst] = __float2bfloat16(qn[src]);
        kr[dst] = __float2bfloat16(kn[src]);
    } else {
        qr[dst] = __float2bfloat16(0.f);
        kr[dst] = __float2bfloat16(0.f);
    }
}

__global__ void materialize_rhs_qb_kc_kernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ mixed, const float* __restrict__ beta,
    const float* __restrict__ cg, __nv_bfloat16* __restrict__ rhs, float* __restrict__ stack_a,
    __nv_bfloat16* __restrict__ kc, int T, int H, int HK, int conv_dim, int v_channel_offset
) {
    const int C = 64;
    const int R = 192;
    int h = blockIdx.x % H;
    int row = blockIdx.x / H;
    int s = row / C;
    int c = row % C;
    int i = threadIdx.x;
    int t = s * C + c;
    size_t sh = (size_t)s * H + h;
    size_t rhs_at = (sh * C + c) * 256;
    size_t qb_at = sh * R * 128 + (size_t)(128 + c) * 128;
    size_t kc_at = (sh * C + c) * 128;
    if (t >= T) {
        rhs[rhs_at + i] = __float2bfloat16(0.f);
        rhs[rhs_at + 128 + i] = __float2bfloat16(0.f);
        if (stack_a != nullptr) stack_a[qb_at + i] = 0.f;
        kc[kc_at + i] = __float2bfloat16(0.f);
        return;
    }
    int kh = h / (H / HK);
    float cg_c = cg[sh * C + c];
    float cg_last = cg[sh * C + (C - 1)];
    float bt = beta[(size_t)t * H + h];
    float kv = kn[((size_t)t * HK + kh) * 128 + i];
    float qv = qn[((size_t)t * HK + kh) * 128 + i];
    float vv = __bfloat162float(mixed[(size_t)t * conv_dim + v_channel_offset + (size_t)h * 128 + i]);
    rhs[rhs_at + i] = __float2bfloat16(bt * expf(cg_c) * kv);
    rhs[rhs_at + 128 + i] = __float2bfloat16(bt * vv);
    if (stack_a != nullptr) stack_a[qb_at + i] = expf(cg_c) * qv;
    kc[kc_at + i] = __float2bfloat16(expf(cg_last - cg_c) * kv);
}

__global__ void mask_scale_a_attn_kernel(
    const float* __restrict__ kk, const float* __restrict__ qk,
    const float* __restrict__ beta, const float* __restrict__ cg,
    float* __restrict__ amat, __nv_bfloat16* __restrict__ attn, int T, int H, int HK
) {
    const int C = 64;
    int i = blockIdx.x % C;
    int h = (blockIdx.x / C) % H;
    int s = blockIdx.x / (C * H);
    int j = threadIdx.x;
    int kh = h / (H / HK);
    size_t sh = (size_t)s * H + h;
    int t_i = s * C + i;
    int t_j = s * C + j;
    size_t src = (((size_t)s * HK + kh) * C + i) * C + j;
    size_t dst = (sh * C + i) * C + j;
    float a_v = 0.f;
    float o_v = 0.f;
    if (t_i < T && t_j < T) {
        float ratio = expf(cg[sh * C + i] - cg[sh * C + j]);
        if (j < i) {
            a_v = beta[(size_t)t_i * H + h] * ratio * kk[src];
        }
        if (j <= i) {
            o_v = ratio * qk[src];
        }
    }
    amat[dst] = a_v;
    attn[dst] = __float2bfloat16(o_v);
}

__global__ void unit_lower_triangular_inverse_bf16out_kernel(
    const float* __restrict__ amat, __nv_bfloat16* __restrict__ tinv
) {
    const int C = 64;
    int c = threadIdx.x;
    const float* a = amat + (size_t)blockIdx.x * C * C;
    __nv_bfloat16* t = tinv + (size_t)blockIdx.x * C * C;
    __shared__ float a_sh[C][C];
    __shared__ float t_sh[C][C];
    for (int i = 0; i < C; ++i) {
        a_sh[i][c] = a[i * C + c];
    }
    __syncthreads();
    for (int i = 0; i < C; ++i) {
        float v = (i == c) ? 1.f : 0.f;
        for (int j = 0; j < i; ++j) {
            v -= a_sh[i][j] * t_sh[j][c];
        }
        t_sh[i][c] = v;
        __syncthreads();
    }
    for (int i = 0; i < C; ++i) {
        t[i * C + c] = __float2bfloat16(t_sh[i][c]);
    }
}

__global__ void add_carry_decay_to_diagonal_kernel(
    const float* __restrict__ cg, float* __restrict__ stack_a, int H
) {
    const int C = 64;
    const int R = 192;
    int d = threadIdx.x;
    size_t sh = (size_t)blockIdx.x;
    float b_last = expf(cg[sh * C + (C - 1)]);
    stack_a[sh * R * 128 + (size_t)d * 128 + d] += b_last;
}

__global__ void writeback_out_bf16_kernel(
    const float* __restrict__ stack_d, __nv_bfloat16* __restrict__ out, int T, int H
) {
    const int C = 64;
    const int R = 192;
    int h = blockIdx.x % H;
    int t = blockIdx.x / H;
    int i = threadIdx.x;
    int s = t / C;
    int c = t % C;
    size_t src = ((size_t)s * H + h) * R * 128 + (size_t)(128 + c) * 128 + i;
    out[((size_t)t * H + h) * 128 + i] = __float2bfloat16(stack_d[src]);
}

constexpr int FWDH_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID = 32;
constexpr int FWDH_THREADS_512_SIXTEEN_WARPS_HIDE_SMEM_LATENCY_AT_ONE_BLOCK_PER_SM =
    512;

__global__ void fwdh_state_resident_carry_kernel(
    const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ qr,
    const float* __restrict__ cg, float* __restrict__ state_inout,
    __nv_bfloat16* __restrict__ out, int T, int H, int HK, int n
) {
    const int C = 64;
    const int D = 128;
    const int TC = FWDH_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    const int KB = 16;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int j0 = tile * TC;
    const int kh = h / (H / HK);
    const int tid = threadIdx.x;
    const int j = tid & 31;
    const int wid = tid >> 5;
    extern __shared__ unsigned char smem_raw[];
    float* S = (float*)smem_raw;
    float* vnew = S + D * TC;
    float* ec = vnew + C * TC;
    __nv_bfloat16* wsm = (__nv_bfloat16*)(ec + C);
    __nv_bfloat16* qsm = wsm + C * D;
    __nv_bfloat16* ksm = qsm + C * D;
    __nv_bfloat16* atsm = ksm + C * D;
    __nv_bfloat16* usm = atsm + C * C;
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int jj = idx % TC;
        S[idx] = state_inout[((size_t)h * D + k) * D + j0 + jj];
    }
    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        if (tid < C) ec[tid] = expf(cg[sh * C + tid]);
        const __nv_bfloat162* wrow = (const __nv_bfloat162*)(wu + sh * C * 256);
        for (int idx = tid; idx < C * 64; idx += blockDim.x) {
            int c = idx >> 6;
            int p = idx & 63;
            ((__nv_bfloat162*)wsm)[idx] = wrow[c * 128 + p];
        }
        for (int idx = tid; idx < C * (TC / 2); idx += blockDim.x) {
            int c = idx / (TC / 2);
            int p = idx % (TC / 2);
            ((__nv_bfloat162*)usm)[idx] = wrow[c * 128 + (128 + j0) / 2 + p];
        }
        const __nv_bfloat162* qrow =
            (const __nv_bfloat162*)(qr + (((size_t)sc * HK + kh) * C) * 128);
        const __nv_bfloat162* krow = (const __nv_bfloat162*)(kc + sh * C * 128);
        for (int idx = tid; idx < C * 64; idx += blockDim.x) {
            ((__nv_bfloat162*)qsm)[idx] = qrow[idx];
            ((__nv_bfloat162*)ksm)[idx] = krow[idx];
        }
        const __nv_bfloat162* arow = (const __nv_bfloat162*)(attn + sh * C * C);
        for (int idx = tid; idx < C * (C / 2); idx += blockDim.x) {
            ((__nv_bfloat162*)atsm)[idx] = arow[idx];
        }
        __syncthreads();
        const int c0 = wid * 4;
        float acc[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[r] = 0.f;
        for (int kb = 0; kb < D; kb += KB) {
            float sreg[KB];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) sreg[kk] = S[(kb + kk) * TC + j];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __nv_bfloat162* w2 = (const __nv_bfloat162*)(wsm + (c0 + r) * D + kb);
#pragma unroll
                for (int kk = 0; kk < KB / 2; ++kk) {
                    float2 wf = __bfloat1622float2(w2[kk]);
                    acc[r] += wf.x * sreg[2 * kk];
                    acc[r] += wf.y * sreg[2 * kk + 1];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int c = c0 + r;
            vnew[c * TC + j] = __bfloat162float(usm[c * TC + j]) - acc[r];
        }
        __syncthreads();
        float accq[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) accq[r] = 0.f;
        for (int kb = 0; kb < D; kb += KB) {
            float sreg[KB];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) sreg[kk] = S[(kb + kk) * TC + j];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __nv_bfloat162* q2 = (const __nv_bfloat162*)(qsm + (c0 + r) * D + kb);
#pragma unroll
                for (int kk = 0; kk < KB / 2; ++kk) {
                    float2 qf = __bfloat1622float2(q2[kk]);
                    accq[r] += qf.x * sreg[2 * kk];
                    accq[r] += qf.y * sreg[2 * kk + 1];
                }
            }
        }
        float acco[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) acco[r] = 0.f;
        for (int mb = 0; mb < C; mb += KB) {
            float vreg[KB];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) vreg[kk] = vnew[(mb + kk) * TC + j];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __nv_bfloat162* a2 = (const __nv_bfloat162*)(atsm + (c0 + r) * C + mb);
#pragma unroll
                for (int kk = 0; kk < KB / 2; ++kk) {
                    float2 af = __bfloat1622float2(a2[kk]);
                    acco[r] += af.x * vreg[2 * kk];
                    acco[r] += af.y * vreg[2 * kk + 1];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int c = c0 + r;
            int t = sc * C + c;
            if (t < T) {
                out[((size_t)t * H + h) * D + j0 + j] =
                    __float2bfloat16(ec[c] * accq[r] + acco[r]);
            }
        }
        __syncthreads();
        const int k0 = wid * 8;
        float acck[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) acck[r] = 0.f;
        for (int cb = 0; cb < C; cb += KB) {
            float vreg[KB];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) vreg[kk] = vnew[(cb + kk) * TC + j];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                const __nv_bfloat162* k2 = (const __nv_bfloat162*)(ksm + (cb + kk) * D + k0);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float2 kf = __bfloat1622float2(k2[r]);
                    acck[2 * r] += kf.x * vreg[kk];
                    acck[2 * r + 1] += kf.y * vreg[kk];
                }
            }
        }
        float b_last = ec[C - 1];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int k = k0 + r;
            S[k * TC + j] = b_last * S[k * TC + j] + acck[r];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int jj = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + jj] = S[idx];
    }
}

constexpr size_t FWDH_SMEM_BYTES_84K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP =
    (size_t)(128 * 32 + 64 * 32 + 64) * sizeof(float) +
    (size_t)(3 * 64 * 128 + 64 * 64 + 64 * 32) * sizeof(__nv_bfloat16);

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

__global__ void mma_fragment_state_resident_carry_kernel(
    const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ qr,
    const float* __restrict__ cg, float* __restrict__ state_inout,
    __nv_bfloat16* __restrict__ out, int T, int H, int HK, int n
) {
    using namespace nvcuda;
    const int C = 64;
    const int D = 128;
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    const int SLD = MMA_STATE_LD_40_PADS_THE_32_COL_TILE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int ALD = MMA_OPERAND_LD_136_PADS_THE_128_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int ATLD = MMA_ATTN_LD_72_PADS_THE_64_COL_STAGE_SO_FRAGMENT_LOADS_SPREAD_BANKS;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int j0 = tile * TC;
    const int kh = h / (H / HK);
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    extern __shared__ unsigned char smem_raw[];
    float* S = (float*)smem_raw;
    float* scr = S + D * SLD;
    float* ec = scr + C * SLD;
    __nv_bfloat16* Sb = (__nv_bfloat16*)(ec + C);
    __nv_bfloat16* Ab0 = Sb + D * SLD;
    __nv_bfloat16* Ab1 = Ab0 + C * ALD;
    __nv_bfloat16* At = Ab1 + C * ALD;
    __nv_bfloat16* Us = At + C * ATLD;
    __nv_bfloat16* Vn = Us + C * SLD;

    auto stage_wide = [&](__nv_bfloat16* dst, const __nv_bfloat16* src_row0, int src_ld) {
        for (int i = tid; i < C * 16; i += (int)blockDim.x) {
            int c = i >> 4;
            int seg = i & 15;
            mma_cp_async16(dst + c * ALD + seg * 8, src_row0 + (size_t)c * src_ld + seg * 8);
        }
    };

    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int j = idx % TC;
        float v = state_inout[((size_t)h * D + k) * D + j0 + j];
        S[k * SLD + j] = v;
        Sb[k * SLD + j] = __float2bfloat16(v);
    }
    {
        const __nv_bfloat16* arow = attn + (size_t)h * C * C;
        for (int i = tid; i < C * 8; i += (int)blockDim.x) {
            int c = i >> 3;
            int seg = i & 7;
            mma_cp_async16(At + c * ATLD + seg * 8, arow + (size_t)c * C + seg * 8);
        }
        mma_cp_async_commit();
        const __nv_bfloat16* wrow = wu + (size_t)h * C * 256;
        stage_wide(Ab0, wrow, 256);
        for (int i = tid; i < C * 4; i += (int)blockDim.x) {
            int c = i >> 2;
            int seg = i & 3;
            mma_cp_async16(Us + c * SLD + seg * 8, wrow + (size_t)c * 256 + D + j0 + seg * 8);
        }
        mma_cp_async_commit();
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> fat;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> facc;

    const int m0 = (wid >> 1) * 16;
    const int n0 = (wid & 1) * 16;
    const int sm0 = wid * 16;

    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        __nv_bfloat16* AbW = (sc & 1) ? Ab1 : Ab0;
        __nv_bfloat16* AbQ = (sc & 1) ? Ab0 : Ab1;
        const __nv_bfloat16* qrow = qr + (((size_t)sc * HK + kh) * C) * D;
        stage_wide(AbQ, qrow, D);
        mma_cp_async_commit();
        if (tid < C) ec[tid] = expf(cg[sh * C + tid]);
        mma_cp_async_wait<1>();
        __syncthreads();

        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt < D / 16; ++kt) {
            wmma::load_matrix_sync(fa, AbW + m0 * ALD + kt * 16, ALD);
            wmma::load_matrix_sync(fb, Sb + kt * 16 * SLD + n0, SLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        wmma::store_matrix_sync(scr + m0 * SLD + n0, facc, SLD, wmma::mem_row_major);
        __syncthreads();

        const __nv_bfloat16* krow = kc + sh * C * D;
        stage_wide(AbW, krow, D);
        mma_cp_async_commit();
        for (int idx = tid; idx < C * TC; idx += blockDim.x) {
            int c = idx / TC;
            int j = idx % TC;
            Vn[c * SLD + j] =
                __float2bfloat16(__bfloat162float(Us[c * SLD + j]) - scr[c * SLD + j]);
        }
        mma_cp_async_wait<1>();
        __syncthreads();

        for (int idx = tid; idx < C * D; idx += blockDim.x) {
            int c = idx >> 7;
            int d = idx & 127;
            AbQ[c * ALD + d] =
                __float2bfloat16(ec[c] * __bfloat162float(AbQ[c * ALD + d]));
        }
        __syncthreads();

        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt < D / 16; ++kt) {
            wmma::load_matrix_sync(fa, AbQ + m0 * ALD + kt * 16, ALD);
            wmma::load_matrix_sync(fb, Sb + kt * 16 * SLD + n0, SLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        for (int kt = 0; kt < C / 16; ++kt) {
            wmma::load_matrix_sync(fa, At + m0 * ATLD + kt * 16, ATLD);
            wmma::load_matrix_sync(fb, Vn + kt * 16 * SLD + n0, SLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        wmma::store_matrix_sync(scr + m0 * SLD + n0, facc, SLD, wmma::mem_row_major);
        __syncthreads();

        for (int idx = tid; idx < C * TC; idx += blockDim.x) {
            int c = idx / TC;
            int j = idx % TC;
            int t = sc * C + c;
            if (t < T) {
                out[((size_t)t * H + h) * D + j0 + j] = __float2bfloat16(scr[c * SLD + j]);
            }
        }
        if (sc + 1 < n) {
            size_t sh1 = (size_t)(sc + 1) * H + h;
            const __nv_bfloat16* arow = attn + sh1 * C * C;
            for (int i = tid; i < C * 8; i += (int)blockDim.x) {
                int c = i >> 3;
                int seg = i & 7;
                mma_cp_async16(At + c * ATLD + seg * 8, arow + (size_t)c * C + seg * 8);
            }
            mma_cp_async_commit();
            const __nv_bfloat16* wrow = wu + sh1 * C * 256;
            stage_wide(AbQ, wrow, 256);
            for (int i = tid; i < C * 4; i += (int)blockDim.x) {
                int c = i >> 2;
                int seg = i & 3;
                mma_cp_async16(Us + c * SLD + seg * 8,
                               wrow + (size_t)c * 256 + D + j0 + seg * 8);
            }
            mma_cp_async_commit();
            mma_cp_async_wait<2>();
        } else {
            mma_cp_async_wait<0>();
        }
        __syncthreads();

        float b_last = ec[C - 1];
        for (int nt = 0; nt < TC / 16; ++nt) {
            wmma::load_matrix_sync(facc, S + sm0 * SLD + nt * 16, SLD, wmma::mem_row_major);
            for (int i = 0; i < facc.num_elements; ++i) facc.x[i] *= b_last;
            for (int kt = 0; kt < C / 16; ++kt) {
                wmma::load_matrix_sync(fat, AbW + kt * 16 * ALD + sm0, ALD);
                wmma::load_matrix_sync(fb, Vn + kt * 16 * SLD + nt * 16, SLD);
                wmma::mma_sync(facc, fat, fb, facc);
            }
            wmma::store_matrix_sync(S + sm0 * SLD + nt * 16, facc, SLD, wmma::mem_row_major);
        }
        __syncwarp();
        for (int idx = tid & 31; idx < 16 * TC; idx += 32) {
            int r = idx / TC;
            int j = idx % TC;
            Sb[(sm0 + r) * SLD + j] = __float2bfloat16(S[(sm0 + r) * SLD + j]);
        }
        __syncthreads();
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int j = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + j] = S[k * SLD + j];
    }
}

constexpr size_t MMA_SMEM_BYTES_93K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP =
    (size_t)(128 * 40 + 64 * 40 + 64) * sizeof(float) +
    (size_t)(128 * 40 + 2 * 64 * 136 + 64 * 72 + 64 * 40 + 64 * 40) * sizeof(__nv_bfloat16);

constexpr size_t
    V3_CHECKPOINTS_ALIAS_THE_STACK_A_RESERVATION_STATE_16384_PLUS_VNEW_8192_BF16_IS_HALF_ITS_F32_ROWS =
        (size_t)128 * 128;

__global__ void v3_ladder_state_checkpoint_kernel(
    const __nv_bfloat16* __restrict__ wu, const __nv_bfloat16* __restrict__ kc,
    const float* __restrict__ cg, float* __restrict__ state_inout,
    __nv_bfloat16* __restrict__ schk, __nv_bfloat16* __restrict__ vchk, int H, int n
) {
    using namespace nvcuda;
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
    extern __shared__ unsigned char smem_raw[];
    float* S = (float*)smem_raw;
    float* scr = S + D * SLD;
    float* ecl = scr + C * SLD;
    __nv_bfloat16* Sb = (__nv_bfloat16*)(ecl + 4);
    __nv_bfloat16* Aw = Sb + D * SLD;
    __nv_bfloat16* Ak = Aw + C * ALD;
    __nv_bfloat16* Us0 = Ak + C * ALD;
    __nv_bfloat16* Us1 = Us0 + C * SLD;
    __nv_bfloat16* Vn = Us1 + C * SLD;

    auto stage_wide = [&](__nv_bfloat16* dst, const __nv_bfloat16* src_row0, int src_ld) {
        for (int i = tid; i < C * 16; i += (int)blockDim.x) {
            int c = i >> 4;
            int seg = i & 15;
            mma_cp_async16(dst + c * ALD + seg * 8, src_row0 + (size_t)c * src_ld + seg * 8);
        }
    };
    auto stage_u = [&](__nv_bfloat16* dst, const __nv_bfloat16* wrow) {
        for (int i = tid; i < C * 4; i += (int)blockDim.x) {
            int c = i >> 2;
            int seg = i & 3;
            mma_cp_async16(dst + c * SLD + seg * 8, wrow + (size_t)c * 256 + D + j0 + seg * 8);
        }
    };

    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int j = idx % TC;
        float v = state_inout[((size_t)h * D + k) * D + j0 + j];
        S[k * SLD + j] = v;
        __nv_bfloat16 vb = __float2bfloat16(v);
        Sb[k * SLD + j] = vb;
        schk[((size_t)h * D + k) * D + j0 + j] = vb;
    }
    {
        const __nv_bfloat16* wrow = wu + (size_t)h * C * 256;
        stage_wide(Aw, wrow, 256);
        stage_u(Us0, wrow);
        mma_cp_async_commit();
        stage_wide(Ak, kc + (size_t)h * C * D, D);
        mma_cp_async_commit();
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> fat;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> facc;

    const int m0 = (wid >> 1) * 16;
    const int n0 = (wid & 1) * 16;
    const int sm0 = wid * 16;

    for (int sc = 0; sc < n; ++sc) {
        size_t sh = (size_t)sc * H + h;
        __nv_bfloat16* Us = (sc & 1) ? Us1 : Us0;
        if (tid == 0) ecl[0] = expf(cg[sh * C + (C - 1)]);
        mma_cp_async_wait<1>();
        __syncthreads();

        wmma::fill_fragment(facc, 0.f);
        for (int kt = 0; kt < D / 16; ++kt) {
            wmma::load_matrix_sync(fa, Aw + m0 * ALD + kt * 16, ALD);
            wmma::load_matrix_sync(fb, Sb + kt * 16 * SLD + n0, SLD);
            wmma::mma_sync(facc, fa, fb, facc);
        }
        wmma::store_matrix_sync(scr + m0 * SLD + n0, facc, SLD, wmma::mem_row_major);
        __syncthreads();

        for (int idx = tid; idx < C * TC; idx += blockDim.x) {
            int c = idx / TC;
            int j = idx % TC;
            __nv_bfloat16 v =
                __float2bfloat16(__bfloat162float(Us[c * SLD + j]) - scr[c * SLD + j]);
            Vn[c * SLD + j] = v;
            vchk[(sh * C + c) * D + j0 + j] = v;
        }
        if (sc + 1 < n) {
            size_t sh1 = (size_t)(sc + 1) * H + h;
            const __nv_bfloat16* wrow = wu + sh1 * C * 256;
            stage_wide(Aw, wrow, 256);
            stage_u((sc & 1) ? Us0 : Us1, wrow);
            mma_cp_async_commit();
            mma_cp_async_wait<1>();
        } else {
            mma_cp_async_wait<0>();
        }
        __syncthreads();

        float b_last = ecl[0];
        for (int nt = 0; nt < TC / 16; ++nt) {
            wmma::load_matrix_sync(facc, S + sm0 * SLD + nt * 16, SLD, wmma::mem_row_major);
            for (int i = 0; i < facc.num_elements; ++i) facc.x[i] *= b_last;
            for (int kt = 0; kt < C / 16; ++kt) {
                wmma::load_matrix_sync(fat, Ak + kt * 16 * ALD + sm0, ALD);
                wmma::load_matrix_sync(fb, Vn + kt * 16 * SLD + nt * 16, SLD);
                wmma::mma_sync(facc, fat, fb, facc);
            }
            wmma::store_matrix_sync(S + sm0 * SLD + nt * 16, facc, SLD, wmma::mem_row_major);
        }
        __syncwarp();
        for (int idx = tid & 31; idx < 16 * TC; idx += 32) {
            int r = idx / TC;
            int j = idx % TC;
            __nv_bfloat16 vb = __float2bfloat16(S[(sm0 + r) * SLD + j]);
            Sb[(sm0 + r) * SLD + j] = vb;
            if (sc + 1 < n) {
                schk[(((size_t)(sc + 1) * H + h) * D + sm0 + r) * D + j0 + j] = vb;
            }
        }
        __syncthreads();
        if (sc + 1 < n) {
            stage_wide(Ak, kc + ((size_t)(sc + 1) * H + h) * C * D, D);
            mma_cp_async_commit();
        }
    }
    for (int idx = tid; idx < D * TC; idx += blockDim.x) {
        int k = idx / TC;
        int j = idx % TC;
        state_inout[((size_t)h * D + k) * D + j0 + j] = S[k * SLD + j];
    }
}

constexpr size_t V3_LADDER_SMEM_BYTES_91K_TWO_PRODUCT_CHAIN_FITS_UNDER_THE_MMA_TAILS_93K =
    (size_t)(128 * 40 + 64 * 40 + 4) * sizeof(float) +
    (size_t)(128 * 40 + 2 * 64 * 136 + 2 * 64 * 40 + 64 * 40) * sizeof(__nv_bfloat16);

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

static int run_head(
    cudaStream_t s, const WsLayout& l, float* workspace, const float* beta,
    int T, int H, int HK, StageProf* prof
) {
    const int C = SUBCHUNK_64_MATCHES_FLA_CHUNK_SIZE_AND_KEEPS_DECAY_RATIOS_ABOVE_F32_UNDERFLOW;
    const int n = l.n;
    float* cg = workspace + l.cg;
    __nv_bfloat16* kr = (__nv_bfloat16*)(workspace + l.kr);
    __nv_bfloat16* qr = (__nv_bfloat16*)(workspace + l.qr);
    float* kk = workspace + l.kk;
    float* qk = workspace + l.qk;
    float* amat = workspace + l.amat;
    __nv_bfloat16* tinv = (__nv_bfloat16*)(workspace + l.tinv);
    __nv_bfloat16* attn = (__nv_bfloat16*)(workspace + l.attn);
    __nv_bfloat16* rhs = (__nv_bfloat16*)(workspace + l.rhs);
    __nv_bfloat16* wu = (__nv_bfloat16*)(workspace + l.wu);
    float* ltws = workspace + l.ltws;
    const GemmMode bf_in = GEMM_BF16_IN_F32_OUT_TENSOR_CORES_WITH_F32_ACCUM;
    const GemmMode bf_all = GEMM_BF16_ALL_TENSOR_CORES_WITH_F32_ACCUM;

    cublasStatus_t st;
    st = gemm_rm(s, ltws, C, C, 128, false, true, bf_in, 1.f, kr, 128, (long long)C * 128, kr,
                 128, (long long)C * 128, 0.f, kk, kk, C, (long long)C * C, n * HK);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    st = gemm_rm(s, ltws, C, C, 128, false, true, bf_in, 1.f, qr, 128, (long long)C * 128, kr,
                 128, (long long)C * 128, 0.f, qk, qk, C, (long long)C * C, n * HK);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    if (prof) prof->lap("kk_qk_gemms");

    mask_scale_a_attn_kernel<<<n * H * C, C, 0, s>>>(kk, qk, beta, cg, amat, attn, T, H, HK);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    if (prof) prof->lap("mask_scale");
    unit_lower_triangular_inverse_bf16out_kernel<<<n * H, C, 0, s>>>(amat, tinv);
    ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    if (prof) prof->lap("tri_inv");

    st = gemm_rm(s, ltws, C, 256, C, false, false, bf_all, 1.f, tinv, C, (long long)C * C, rhs,
                 256, (long long)C * 256, 0.f, wu, wu, 256, (long long)C * 256, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    if (prof) prof->lap("wu_gemm");
    return 0;
}

static int run_fwdh_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, float* state_inout, __nv_bfloat16* out,
    int T, int H, int HK
) {
    const int TC = FWDH_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    static std::once_flag smem_once;
    static cudaError_t smem_rc = cudaSuccess;
    std::call_once(smem_once, []() {
        smem_rc = cudaFuncSetAttribute(
            fwdh_state_resident_carry_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)FWDH_SMEM_BYTES_84K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP);
    });
    if (smem_rc != cudaSuccess) return (int)smem_rc;
    __nv_bfloat16* attn = (__nv_bfloat16*)(workspace + l.attn);
    __nv_bfloat16* wu = (__nv_bfloat16*)(workspace + l.wu);
    __nv_bfloat16* kc = (__nv_bfloat16*)(workspace + l.kc);
    __nv_bfloat16* qr = (__nv_bfloat16*)(workspace + l.qr);
    float* cg = workspace + l.cg;
    dim3 grid(128 / TC, H);
    fwdh_state_resident_carry_kernel<<<
        grid, FWDH_THREADS_512_SIXTEEN_WARPS_HIDE_SMEM_LATENCY_AT_ONE_BLOCK_PER_SM,
        FWDH_SMEM_BYTES_84K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP, s>>>(
        wu, kc, attn, qr, cg, state_inout, out, T, H, HK, l.n);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    return 0;
}

static int run_mma_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, float* state_inout, __nv_bfloat16* out,
    int T, int H, int HK
) {
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    static std::once_flag smem_once;
    static cudaError_t smem_rc = cudaSuccess;
    std::call_once(smem_once, []() {
        smem_rc = cudaFuncSetAttribute(
            mma_fragment_state_resident_carry_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)MMA_SMEM_BYTES_93K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP);
    });
    if (smem_rc != cudaSuccess) return (int)smem_rc;
    __nv_bfloat16* attn = (__nv_bfloat16*)(workspace + l.attn);
    __nv_bfloat16* wu = (__nv_bfloat16*)(workspace + l.wu);
    __nv_bfloat16* kc = (__nv_bfloat16*)(workspace + l.kc);
    __nv_bfloat16* qr = (__nv_bfloat16*)(workspace + l.qr);
    float* cg = workspace + l.cg;
    dim3 grid(128 / TC, H);
    mma_fragment_state_resident_carry_kernel<<<
        grid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
        MMA_SMEM_BYTES_93K_NEEDS_THE_OPT_IN_ATTRIBUTE_ABOVE_THE_48K_STATIC_CAP, s>>>(
        wu, kc, attn, qr, cg, state_inout, out, T, H, HK, l.n);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    return 0;
}

static int run_v3_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, float* state_inout, __nv_bfloat16* out,
    int T, int H, int HK, StageProf* prof
) {
    const int TC = MMA_TILE_COLS_32_BY_H48_HEADS_MAKES_192_BLOCKS_ONE_WAVE_ON_THE_188_SM_GRID;
    static std::once_flag smem_once;
    static cudaError_t smem_rc = cudaSuccess;
    std::call_once(smem_once, []() {
        smem_rc = cudaFuncSetAttribute(
            v3_ladder_state_checkpoint_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_LADDER_SMEM_BYTES_91K_TWO_PRODUCT_CHAIN_FITS_UNDER_THE_MMA_TAILS_93K);
        if (smem_rc != cudaSuccess) return;
        smem_rc = cudaFuncSetAttribute(
            v3_output_from_checkpoints_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)V3_OUT_SMEM_BYTES_87K_ONE_SHOT_STAGE_NO_PIPELINE_BLOCKS_OVERLAP_EACH_OTHER);
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
    v3_ladder_state_checkpoint_kernel<<<
        lgrid, MMA_THREADS_256_EIGHT_WARPS_ONE_16X16_WMMA_OUTPUT_TILE_PER_WARP,
        V3_LADDER_SMEM_BYTES_91K_TWO_PRODUCT_CHAIN_FITS_UNDER_THE_MMA_TAILS_93K, s>>>(
        wu, kc, cg, state_inout, schk, vchk, H, l.n);
    cudaError_t ce = cudaGetLastError();
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

static int run_stack_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, const float* state0,
    int T, int H, int HK, StageProf* prof
) {
    const int C = SUBCHUNK_64_MATCHES_FLA_CHUNK_SIZE_AND_KEEPS_DECAY_RATIOS_ABOVE_F32_UNDERFLOW;
    const int R = STACK_ROWS_192_IS_STATE_128_PLUS_OUT_64_SO_THE_CARRY_IS_ONE_GEMM_PER_SUBCHUNK;
    const int n = l.n;
    float* cg = workspace + l.cg;
    __nv_bfloat16* attn = (__nv_bfloat16*)(workspace + l.attn);
    __nv_bfloat16* wu = (__nv_bfloat16*)(workspace + l.wu);
    __nv_bfloat16* kc = (__nv_bfloat16*)(workspace + l.kc);
    float* stack_a = workspace + l.stack_a;
    float* stack_c = workspace + l.stack_c;
    float* stack_d = workspace + l.stack_d;
    float* ltws = workspace + l.ltws;
    const GemmMode bf_in = GEMM_BF16_IN_F32_OUT_TENSOR_CORES_WITH_F32_ACCUM;

    cublasStatus_t st;
    st = gemm_rm(s, ltws, C, 128, C, false, false, bf_in, 1.f, attn, C, (long long)C * C,
                 wu + 128, 256, (long long)C * 256, 0.f, stack_c + (size_t)128 * 128,
                 stack_c + (size_t)128 * 128, 128, (long long)R * 128, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    st = gemm_rm(s, ltws, C, 128, C, false, false, bf_in, -1.f, attn, C, (long long)C * C, wu,
                 256, (long long)C * 256, 1.f, stack_a + (size_t)128 * 128,
                 stack_a + (size_t)128 * 128, 128, (long long)R * 128, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    if (prof) prof->lap("oloc_qeff_gemms");
    st = gemm_rm(s, ltws, 128, 128, C, true, false, bf_in, -1.f, kc, 128, (long long)C * 128, wu,
                 256, (long long)C * 256, 0.f, stack_a, stack_a, 128, (long long)R * 128, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;
    st = gemm_rm(s, ltws, 128, 128, C, true, false, bf_in, 1.f, kc, 128, (long long)C * 128,
                 wu + 128, 256, (long long)C * 256, 0.f, stack_c, stack_c, 128,
                 (long long)R * 128, n * H);
    if (st != CUBLAS_STATUS_SUCCESS) return -6;

    add_carry_decay_to_diagonal_kernel<<<n * H, 128, 0, s>>>(cg, stack_a, H);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    if (prof) prof->lap("p_g_gemms_diag");

    for (int sc = 0; sc < n; ++sc) {
        size_t at = (size_t)sc * H * R * 128;
        const float* s_prev = sc == 0 ? state0 : stack_d + (size_t)(sc - 1) * H * R * 128;
        long long s_stride = sc == 0 ? (long long)128 * 128 : (long long)R * 128;
        st = gemm_rm(s, ltws, R, 128, 128, false, false, GEMM_F32_ALL, 1.f, stack_a + at, 128,
                     (long long)R * 128, s_prev, 128, s_stride, 1.f, stack_c + at, stack_d + at,
                     128, (long long)R * 128, H);
        if (st != CUBLAS_STATUS_SUCCESS) return -6;
    }
    if (prof) prof->lap("sequential_carry_gemms");
    return 0;
}

static int run_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, const float* beta, const float* state0,
    int T, int H, int HK, StageProf* prof
) {
    int rc = run_head(s, l, workspace, beta, T, H, HK, prof);
    if (rc != 0) return rc;
    return run_stack_tail(s, l, workspace, state0, T, H, HK, prof);
}

static int run_graph_tail(
    cudaStream_t s, const WsLayout& l, float* workspace, const float* sbeta, const float* sstate,
    int T, int H, int HK
) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphKey key{{T, H, HK, (int64_t)(uintptr_t)workspace}};
    auto it = g_graphs.find(key);
    if (it == g_graphs.end()) {
        int rc = run_tail(s, l, workspace, sbeta, sstate, T, H, HK, nullptr);
        if (rc == 0) {
            g_graphs.emplace(
                key,
                GraphEntry{WARMUP_RAN_UNCAPTURED_SO_LT_PLANS_AND_MODULES_EXIST_BEFORE_CAPTURE,
                           nullptr});
        }
        return rc;
    }
    GraphEntry& e = it->second;
    if (e.state == WARMUP_RAN_UNCAPTURED_SO_LT_PLANS_AND_MODULES_EXIST_BEFORE_CAPTURE) {
        cudaStream_t& cs =
            g_capture_stream_side_allocated_because_the_callers_legacy_stream_cannot_begin_capture;
        if (cs == nullptr && cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) {
            cs = nullptr;
            e.state = CAPTURE_FAILED_STAY_UNCAPTURED;
            fprintf(stderr, "[gdn-chunk-gemm-graph] side stream create failed; staying uncaptured\n");
        } else if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            e.state = CAPTURE_FAILED_STAY_UNCAPTURED;
            fprintf(stderr, "[gdn-chunk-gemm-graph] begin capture failed; staying uncaptured\n");
        } else {
            int rc = run_tail(cs, l, workspace, sbeta, sstate, T, H, HK, nullptr);
            cudaGraph_t graph = nullptr;
            cudaError_t ce = cudaStreamEndCapture(cs, &graph);
            if (rc != 0 || ce != cudaSuccess || graph == nullptr) {
                if (graph != nullptr) cudaGraphDestroy(graph);
                e.state = CAPTURE_FAILED_STAY_UNCAPTURED;
                fprintf(stderr, "[gdn-chunk-gemm-graph] capture failed rc=%d ce=%d; staying uncaptured\n",
                        rc, (int)ce);
            } else {
                size_t n_nodes = 0;
                cudaGraphGetNodes(graph, nullptr, &n_nodes);
                cudaGraphExec_t exec = nullptr;
                ce = cudaGraphInstantiate(&exec, graph, 0);
                cudaGraphDestroy(graph);
                if (ce != cudaSuccess || exec == nullptr) {
                    e.state = CAPTURE_FAILED_STAY_UNCAPTURED;
                    fprintf(stderr, "[gdn-chunk-gemm-graph] instantiate failed ce=%d; staying uncaptured\n",
                            (int)ce);
                } else {
                    fprintf(stderr, "[gdn-chunk-gemm-graph] captured T=%d H=%d HK=%d nodes=%zu\n",
                            T, H, HK, n_nodes);
                    e.exec = exec;
                    e.state = GRAPH_READY;
                }
            }
        }
    }
    if (e.state != GRAPH_READY) {
        return run_tail(s, l, workspace, sbeta, sstate, T, H, HK, nullptr);
    }
    cudaError_t ce = cudaGraphLaunch(e.exec, s);
    if (ce != cudaSuccess) return (int)ce;
    g_graph_replays.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

}

extern "C" long long nv_kernels_gdn_chunk_gemm_scan_workspace_elems(int T, int H, int HK) {
    if (T <= 0 || H <= 0 || HK <= 0) return 0;
    return (long long)gdn_chunk_gemm::layout(T, H, HK).total;
}

extern "C" long long nv_kernels_gdn_chunk_gemm_graph_replays_total() {
    return gdn_chunk_gemm::g_graph_replays.load(std::memory_order_relaxed);
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
    float* stack_a = workspace + l.stack_a;
    float* stack_d = workspace + l.stack_d;

    StageProf prof{prof_armed_nv_gdn_chunk_gemm_prof_1_syncs_every_stage_so_never_in_timed_runs(),
                   s, std::chrono::steady_clock::now()};
    const bool v3_armed =
        v3_env_read_per_call_so_one_process_can_ab_split_ladder_vs_inline_out_tails();
    const bool mma_armed =
        !v3_armed && mma_env_read_per_call_so_one_process_can_ab_fragment_tail_vs_the_scalar_tails();
    const bool fwdh_armed =
        !v3_armed && !mma_armed &&
        fwdh_env_read_per_call_so_one_process_can_ab_stack_carry_vs_state_resident();
    cumsum_log_decay_kernel<<<n * H, C, 0, s>>>(g_exp, cg, T, H);
    rearrange_qk_padded_kernel<<<n * C * HK, 128, 0, s>>>(qn, kn, qr, kr, T, HK);
    materialize_rhs_qb_kc_kernel<<<n * C * H, 128, 0, s>>>(
        qn, kn, (const __nv_bfloat16*)mixed, beta, cg, rhs,
        (v3_armed || mma_armed || fwdh_armed) ? nullptr : stack_a, kc, T, H, HK, conv_dim,
        v_channel_offset);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    prof.lap("materialize");

    if (v3_armed) {
        int rc = run_head(s, l, workspace, beta, T, H, HK, prof.on ? &prof : nullptr);
        if (rc != 0) return rc;
        rc = run_v3_tail(s, l, workspace, state_inout, (__nv_bfloat16*)out, T, H, HK,
                         prof.on ? &prof : nullptr);
        if (rc != 0) return rc;
        return 0;
    }

    if (mma_armed) {
        int rc = run_head(s, l, workspace, beta, T, H, HK, prof.on ? &prof : nullptr);
        if (rc != 0) return rc;
        rc = run_mma_tail(s, l, workspace, state_inout, (__nv_bfloat16*)out, T, H, HK);
        if (rc != 0) return rc;
        prof.lap("mma_fragment_state_resident_carry");
        return 0;
    }

    if (fwdh_armed) {
        int rc = run_head(s, l, workspace, beta, T, H, HK, prof.on ? &prof : nullptr);
        if (rc != 0) return rc;
        rc = run_fwdh_tail(s, l, workspace, state_inout, (__nv_bfloat16*)out, T, H, HK);
        if (rc != 0) return rc;
        prof.lap("fwdh_state_resident_carry");
        return 0;
    }

    const bool graph_armed =
        !prof.on && graph_env_read_per_call_so_one_process_can_ab_replay_vs_uncaptured();
    int rc;
    if (graph_armed) {
        float* sbeta = workspace + l.sbeta;
        float* sstate = workspace + l.sstate;
        ce = cudaMemcpyAsync(sbeta, beta, (size_t)T * H * sizeof(float),
                             cudaMemcpyDeviceToDevice, s);
        if (ce != cudaSuccess) return (int)ce;
        ce = cudaMemcpyAsync(sstate, state_inout, (size_t)H * 128 * 128 * sizeof(float),
                             cudaMemcpyDeviceToDevice, s);
        if (ce != cudaSuccess) return (int)ce;
        rc = run_graph_tail(s, l, workspace, sbeta, sstate, T, H, HK);
    } else {
        rc = run_tail(s, l, workspace, beta, state_inout, T, H, HK, &prof);
    }
    if (rc != 0) return rc;

    writeback_out_bf16_kernel<<<T * H, 128, 0, s>>>(stack_d, (__nv_bfloat16*)out, T, H);
    ce = cudaGetLastError();
    if (ce != cudaSuccess) return (int)ce;
    const int R = STACK_ROWS_192_IS_STATE_128_PLUS_OUT_64_SO_THE_CARRY_IS_ONE_GEMM_PER_SUBCHUNK;
    ce = cudaMemcpy2DAsync(state_inout, (size_t)128 * 128 * sizeof(float),
                           stack_d + (size_t)(n - 1) * H * R * 128,
                           (size_t)R * 128 * sizeof(float), (size_t)128 * 128 * sizeof(float), H,
                           cudaMemcpyDeviceToDevice, s);
    if (ce != cudaSuccess) return (int)ce;
    prof.lap("writeback_and_state_copy");
    return 0;
}
