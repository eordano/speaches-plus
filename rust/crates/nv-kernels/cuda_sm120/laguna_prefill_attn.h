#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t abi, q_len, kv_len, capacity, tile_q, tile_kv, splits, ctas;
    uint32_t sm_count, active_blocks_per_sm, shared_bytes, registers_per_thread;
    uint64_t scratch_bytes;
    int device;
    void* stream;
} LagunaPrefillPlan;

#ifdef __cplusplus
extern "C" {
#endif
int laguna_prefill_prepare(LagunaPrefillPlan*, uint32_t, uint32_t, uint32_t, void*, char*, size_t);
int laguna_prefill_run(const LagunaPrefillPlan*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, void*, size_t, void*, char*, size_t);
#ifdef __cplusplus
}
#endif
