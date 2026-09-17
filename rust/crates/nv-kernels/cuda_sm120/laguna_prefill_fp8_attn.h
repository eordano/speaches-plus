#pragma once
#include <stddef.h>
#include <stdint.h>

enum {
    LAGUNA_PREFILL_FP8_SCALE_FIXED_ONE = 1,
    LAGUNA_PREFILL_FP8_SCALE_PER_TOKEN_HEAD = 2
};

typedef struct {
    uint32_t cache_elem_enum, scale_mode;
    float k_scale, v_scale;
    uint64_t k, v, k_row_scales, v_row_scales;
} LagunaPrefillFp8CacheView;

typedef struct {
    uint32_t abi, q_len, kv_len, capacity, tile_q, tile_kv, splits, ctas;
    uint32_t sm_count, active_blocks_per_sm, shared_bytes, registers_per_thread;
    uint64_t scratch_bytes;
    int device;
    void* stream;
    uint32_t cache_elem_enum, scale_mode;
    float k_scale, v_scale;
} LagunaPrefillFp8Plan;

#ifdef __cplusplus
extern "C" {
#endif
int laguna_prefill_fp8_prepare(LagunaPrefillFp8Plan*, uint32_t, uint32_t, uint32_t, void*, char*, size_t);
int laguna_prefill_fp8_run(const LagunaPrefillFp8Plan*, const uint16_t*, const LagunaPrefillFp8CacheView*, uint16_t*, void*, size_t, void*, char*, size_t);
#ifdef __cplusplus
}
#endif
