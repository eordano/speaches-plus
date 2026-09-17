#include "laguna_prefill_fp8_attn.h"
#include <cuda_runtime.h>
#include "laguna_prefill_fp8_nan_upcast.cuh"
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace flashinfer;
using Query = __nv_bfloat16;
using KV = __nv_fp8_e4m3;
using Output = __nv_bfloat16;
using Params = SinglePrefillParams<Query, KV, Output>;
using Variant = DefaultAttention<false, false, false, false>;

static_assert(sizeof(Query) == 2 && sizeof(KV) == 1 && sizeof(Output) == 2);
static_assert(sizeof(LagunaPrefillFp8CacheView) == 48 && offsetof(LagunaPrefillFp8CacheView, k) == 16);

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}

static void eager_stream(cudaStream_t stream) {
    cudaStreamCaptureStatus state;
    check(cudaStreamIsCapturing(stream, &state));
    require(state == cudaStreamCaptureStatusNone, "prefill adapter capture is unsupported");
}

static int failed(char* error, size_t size, const char* message) {
    if (error && size) std::snprintf(error, size, "%s", message);
    return -1;
}

extern "C" int laguna_prefill_fp8_prepare(LagunaPrefillFp8Plan* out, uint32_t m, uint32_t s,
                                     uint32_t capacity, void* raw_stream, char* error, size_t size) {
    if (out) *out = {};
    try {
        require(out, "plan storage is required");
        require(m > 0 && m <= 8192 && s >= m && s <= capacity && capacity <= 262144,
                "requires 1<=M<=8192 and M<=S<=capacity<=262144");
        auto stream = static_cast<cudaStream_t>(raw_stream);
        eager_stream(stream);
        LagunaPrefillFp8Plan p{};
        p.q_len = m; p.kv_len = s; p.capacity = capacity; p.stream = raw_stream;
        check(cudaGetDevice(&p.device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, p.device));
        require(prop.major == 12 && prop.minor == 0, "bounded probe requires SM120");
        p.sm_count = prop.multiProcessorCount;
        int max_smem = 0;
        check(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, p.device));
        p.tile_q = FA2DetermineCtaTileQ(static_cast<int64_t>(m) * 6, 128);
        DISPATCH_CTA_TILE_Q(p.tile_q, CTA_TILE_Q, {
            constexpr uint32_t WQ = get_num_warps_q(CTA_TILE_Q);
            constexpr uint32_t WK = get_num_warps_kv(CTA_TILE_Q);
            constexpr uint32_t MQ = get_num_mma_q(CTA_TILE_Q);
            constexpr uint32_t q_bytes = CTA_TILE_Q * 128 * sizeof(Query);
            constexpr uint32_t kv_mma_bytes = 256 * 16 * WK * sizeof(KV);
            const int resident = max_smem >= 2 * (q_bytes + kv_mma_bytes) ? 2 : 1;
            require(max_smem / resident > q_bytes, "mixed prefill shared memory cannot hold query tile");
            const uint32_t max_mk = std::min<uint32_t>(8 / MQ,
                (max_smem / resident - q_bytes) / kv_mma_bytes);
            DISPATCH_NUM_MMA_KV(max_mk, MK, {
                using Traits = KernelTraits<MaskMode::kCausal, CTA_TILE_Q, MQ, MK, 8, 8,
                    WQ, WK, PosEncodingMode::kNone, Query, KV, Output, float, int32_t, Variant>;
                if constexpr (Traits::IsInvalid()) {
                    throw std::runtime_error("pinned prefill selected invalid traits");
                } else {
                    auto kernel = SinglePrefillWithKVCacheKernel<Traits, Params>;
                    p.shared_bytes = sizeof(typename Traits::SharedStorage);
                    p.tile_kv = Traits::CTA_TILE_KV;
                    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, p.shared_bytes));
                    int active = 0;
                    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, WQ * WK * 32, p.shared_bytes));
                    require(active > 0, "prefill kernel has zero occupancy");
                    p.active_blocks_per_sm = active;
                    cudaFuncAttributes attributes{};
                    check(cudaFuncGetAttributes(&attributes, kernel));
                    p.registers_per_thread = attributes.numRegs;
                    const uint32_t base = 8 * ceil_div(m * 6, CTA_TILE_Q);
                    const uint32_t max_chunks = active * p.sm_count / base;
                    p.splits = max_chunks ? ceil_div(s, std::max<uint32_t>(ceil_div(s, max_chunks), 256)) : 1;
                    p.splits = std::max<uint32_t>(1, p.splits);
                    p.ctas = base * p.splits;
                    p.scratch_bytes = p.splits > 1 ? uint64_t(p.splits) * m * 48 * (128 * sizeof(Output) + sizeof(float)) : 0;
                }
            });
        });
        p.abi = 2;
        p.cache_elem_enum = 2; p.scale_mode = LAGUNA_PREFILL_FP8_SCALE_FIXED_ONE;
        p.k_scale = p.v_scale = 1.0f;
        *out = p;
        if (error && size) error[0] = 0;
        return 0;
    } catch (const std::exception& e) { return failed(error, size, e.what()); }
      catch (...) { return failed(error, size, "unknown prefill prepare failure"); }
}

extern "C" int laguna_prefill_fp8_run(const LagunaPrefillFp8Plan* p, const uint16_t* q,
                                 const LagunaPrefillFp8CacheView* cache, uint16_t* o,
                                 void* scratch, size_t scratch_bytes, void* raw_stream,
                                 char* error, size_t size) {
    try {
        require(p && p->abi == 2 && raw_stream == p->stream, "valid immutable plan and its stream required");
        require(cache && cache->cache_elem_enum == 2 && cache->scale_mode == LAGUNA_PREFILL_FP8_SCALE_FIXED_ONE &&
                cache->k_scale == 1.0f && cache->v_scale == 1.0f && !cache->k_row_scales && !cache->v_row_scales,
                "E4M3 FixedOne cache required; per-token/head scales are incompatible");
        const auto* k = reinterpret_cast<const uint8_t*>(cache->k);
        const auto* v = reinterpret_cast<const uint8_t*>(cache->v);
        require(q && k && v && o && scratch, "Q/K/V/O and non-null split scratch required");
        require(scratch_bytes >= p->scratch_bytes, "insufficient split scratch");
        require((reinterpret_cast<uintptr_t>(scratch) & 15) == 0, "split scratch must be 16-byte aligned");
        require(((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                  reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(o)) & 15) == 0,
                "Q/K/V/O must be 16-byte aligned");
        int device = -1;
        check(cudaGetDevice(&device));
        require(device == p->device, "prefill plan belongs to another CUDA device");
        auto stream = static_cast<cudaStream_t>(raw_stream);
        eager_stream(stream);
        Params args{};
        args.lse = nullptr;
        args.q = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(q));
        args.k = reinterpret_cast<KV*>(const_cast<uint8_t*>(k));
        args.v = reinterpret_cast<KV*>(const_cast<uint8_t*>(v));
        args.o = reinterpret_cast<__nv_bfloat16*>(o);
        args.group_size = uint_fastdiv(6);
        args.qo_len = p->q_len; args.kv_len = p->kv_len;
        args.num_qo_heads = 48; args.num_kv_heads = 8;
        args.q_stride_n = 6144; args.q_stride_h = 128;
        args.k_stride_n = args.v_stride_n = 1024;
        args.k_stride_h = args.v_stride_h = 128;
        args.head_dim = 128; args.window_left = -1;
        args.sm_scale = 1.0f / std::sqrt(128.0f);
        args.rope_rcp_scale = args.rope_rcp_theta = 1.0f;
        check(SinglePrefillWithKVCacheDispatched<128, 128, PosEncodingMode::kNone, false,
            MaskMode::kCausal, Variant, Params>(args, static_cast<__nv_bfloat16*>(scratch), stream));
        if (error && size) error[0] = 0;
        return 0;
    } catch (const std::exception& e) { return failed(error, size, e.what()); }
      catch (...) { return failed(error, size, "unknown prefill launch failure"); }
}
