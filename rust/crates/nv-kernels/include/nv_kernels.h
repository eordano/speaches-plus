#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int nv_kernels_hello_launch(void* stream, float* out, size_t n);

int nv_kernels_capability(int* sm_major, int* sm_minor);

int nv_kernels_rmsnorm_f32(
    void* stream,
    const float* x,
    const float* weight,
    float* y,
    size_t batch,
    size_t hidden,
    float eps
);

int nv_kernels_rmsnorm_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* y,
    size_t batch,
    size_t hidden,
    float eps
);

int nv_kernels_gather_rows_e4m3_bf16(void* stream, const uint8_t* weight, const float* row_scale, const int32_t* token_ids, uint16_t* out, int tokens, int hidden, int vocab);
int nv_kernels_rmsnorm_e4m3_weight_bf16(void* stream, const uint16_t* x, const uint8_t* weight, const float* weight_scale, uint16_t* out, size_t batch, size_t hidden, float eps);
int nv_kernels_rmsnorm_residual_e4m3_weight_bf16(void* stream, const uint16_t* x, uint16_t* residual, const uint8_t* weight, const float* weight_scale, uint16_t* out, size_t batch, size_t hidden, float eps);

int nv_kernels_rope_f32(
    void* stream,
    float* q,
    float* k,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    size_t batch,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
);

int nv_kernels_rope_bf16(
    void* stream,
    uint16_t* q,
    uint16_t* k,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    size_t batch,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
);

int nv_kernels_softmax_topk_topp(
    void* stream,
    const float* logits,
    float* probs_out,
    uint32_t* indices_out,
    size_t batch,
    size_t vocab,
    size_t k,
    float p
);

int nv_kernels_silu_f32(
    void* stream,
    const float* x,
    float* y,
    size_t n
);

int nv_kernels_silu_bf16(
    void* stream,
    const uint16_t* x,
    uint16_t* y,
    size_t n
);

int nv_kernels_silu_mul_f32(
    void* stream,
    const float* x,
    const float* gate,
    float* y,
    size_t n
);

int nv_kernels_silu_mul_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* gate,
    uint16_t* y,
    size_t n
);

int nv_kernels_silu_mul_bf16_candle(
    void* stream,
    const uint16_t* x,
    const uint16_t* gate,
    uint16_t* y,
    size_t n
);

int nv_kernels_gelu_tanh_mul_bf16(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* y,
    size_t n
);

int nv_kernels_gemv_bf16(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K
);

int nv_kernels_gemm_bf16_mk(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_gemv_w4a16(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int GS
);

int nv_kernels_marlin_workspace_elems(int* out_elems);

int nv_kernels_marlin_repack_w4a16(
    void* stream,
    const void* b_q_packed,
    void* b_q_marlin_out,
    int k,
    int n,
    int num_bits
);

int nv_kernels_marlin_gemm_w4a16(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales,
    void* c_out,
    void* workspace,
    int m,
    int n,
    int k,
    int group_size,
    int a_is_bf16
);

int nv_kernels_marlin_gemm_w4a16_prezeroed(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales,
    void* c_out,
    void* workspace,
    int m,
    int n,
    int k,
    int group_size,
    int a_is_bf16
);

int nv_kernels_marlin_gemm_w4a16_m1_tail(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales,
    void* c_out,
    void* workspace,
    int n,
    int k,
    int group_size,
    int kind,
    const uint16_t* res,
    const uint16_t* w,
    uint16_t* h_out,
    float* rstd_out,
    const uint16_t* next_w,
    uint16_t* normed_out,
    const float* pli,
    uint16_t* gelu_out,
    float eps,
    float scale,
    float eps_next,
    unsigned int* counter);

int nv_kernels_marlin_gemm_fp8_w8a16(void* stream, const void* a_bf16, const void* b_q_marlin, const void* b_scales_bf16_per_channel, void* c_out, void* c_tmp_f32, long long c_tmp_floats, void* workspace, int m, int n, int k);
int nv_kernels_marlin_nvfp4_c_tmp_floats(long long* out_floats);
int nv_kernels_marlin_gemm_nvfp4_w4a16(void* stream, const void* a_bf16, const void* b_q_marlin, const void* b_scales_e5m3_bytes, const float* global_scale_dev, void* c_out, void* c_tmp_f32, long long c_tmp_floats, void* workspace, int m, int n, int k);

int nv_kernels_marlin_moe_nvfp4_c_tmp_floats(long long* out_floats);
int nv_kernels_marlin_moe_workspace_ints(int* out_ints);
int nv_kernels_marlin_moe_nvfp4_w4a16(void* stream, const void* a_bf16, const void* b_q_stacked, const void* b_scales_stacked, const float* global_scales_per_expert, void* c_out, void* c_tmp_f32, long long c_tmp_floats, void* workspace, const int* sorted_token_ids, const int* expert_ids, const int* num_tokens_past_padded, const float* topk_weights, int num_experts, int top_k, int mul_topk_weights, int size_m, int size_n, int size_k);

int nv_kernels_marlin_moe_align_block8_small(void* stream, const int32_t* topk_ids, int assignments, int num_experts, int32_t* sorted_ids, int32_t* expert_ids, int32_t* num_tokens_past_padded);
int nv_kernels_marlin_moe_place_rows_by_inv_perm(void* stream, const uint16_t* rows_bf16, const int32_t* inv_perm, uint16_t* y_down, int n_rows, int hidden);
int nv_kernels_marlin_moe_silu_mul_halves(void* stream, const uint16_t* gate_up_bf16, uint16_t* act_bf16, int rows, int inter);
int nv_kernels_split_rows_bf16_four(void* stream, const uint16_t* src, uint16_t* a, uint16_t* b, uint16_t* c, uint16_t* d, int rows, int na, int nb, int nc, int nd, int n_src);
int nv_kernels_marlin_moe_sum_slots_into_f32(void* stream, const uint16_t* rows_bf16, float* out, int n_tokens, int slots, int hidden);

int nv_kernels_marlin_int4_fp8_preprocess(
    void* stream,
    const void* qweight,
    void* out,
    long long numel_i32
);

int nv_kernels_marlin_repack_w4a8(
    void* stream,
    const void* b_q_packed,
    void* b_q_marlin_out,
    int k,
    int n,
    int num_bits
);

int nv_kernels_marlin_gemm_w4a8_fp8(
    void* stream,
    const void* a_fp8,
    const void* a_scales_f32,
    const void* b_q_marlin,
    const void* b_scales_bf16,
    void* c_out_bf16,
    void* workspace,
    int m,
    int n,
    int k,
    int group_size,
    int c_prezeroed
);

int nv_kernels_multi_zero_bf16(void* stream, const void* list, int n);

int nv_kernels_gemv_i8_normed_mk_max_m(int K);

int nv_kernels_gemv_i8_normed_mk(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_gemv_i8_mk_h(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_normx_mk(
    void* stream,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* xn,
    int K,
    int M
);

int nv_kernels_gemv_i8_prenormed_mk(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* xn,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_rowquant_e4m3(
    void* stream,
    const uint16_t* w,
    uint8_t* wq,
    float* row_scale,
    int N,
    int K
);

int nv_kernels_gemv_e4m3_mk_h(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_gemv_e4m3_mk(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_gemv_e4m3_group(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M,
    int rows_per_warp
);

int nv_kernels_gemm_e4m3_mrows_f16mma(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M
);

int nv_kernels_gemm_nvfp4_mrows_f16mma(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* x,
    uint16_t* y,
    float alpha,
    int M,
    int N,
    int K
);

int nv_kernels_gemm_nvfp4_mrows_f16mma_dual(
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
    int N_b,
    int K
);

int nv_kernels_batched_copy_u32(
    void* stream,
    const uint64_t* dsts,
    const uint64_t* srcs,
    const uint32_t* nwords,
    int count
);

int nv_kernels_gemv_bf16_rows(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int rows
);

int nv_kernels_gemv_e4m3_group_tile_v(int M, int K);

int nv_kernels_gemv_e4m3_untiled_staging_reaches_this_m(int K);

int nv_kernels_gemv_e4m3_mk_dual(
    void* stream,
    const uint8_t* wq_a,
    const float* rs_a,
    const uint8_t* wq_b,
    const float* rs_b,
    const uint16_t* x,
    uint16_t* y_a,
    uint16_t* y_b,
    int n_a,
    int n_b,
    int K,
    int M
);

int nv_kernels_scale_rowcol_bf16(
    void* stream,
    uint16_t* d,
    const float* row_scale_m,
    const float* col_scale_n,
    int M,
    int N
);

int nv_kernels_marlin_gemm_w4a16_ex(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales,
    void* c_out,
    void* c_tmp,
    void* workspace,
    int m,
    int n,
    int k,
    int group_size,
    int a_is_bf16,
    int use_atomic_add,
    int use_fp32_reduce
);

int nv_kernels_attn_decode_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int NH,
    int NKV,
    int HD,
    int TOTAL,
    int START
);

int nv_kernels_incr_pos(void* stream, int* pos);
int nv_kernels_write_kv_f32(
    void* stream,
    const float* src_k,
    const float* src_v,
    float* cache_k,
    float* cache_v,
    const int* pos,
    int NKV,
    int HD
);
int nv_kernels_attn_decode_dev_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const int* pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
);
int nv_kernels_flash_decode_dev_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const int* pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
);

int nv_kernels_cast_bf16_f32(void* stream, const uint16_t* x, float* y, int n);
int nv_kernels_cast_f32_bf16(void* stream, const float* x, uint16_t* y, int n);
int nv_kernels_rms_no_weight_bf16_f32(
    void* stream, const uint16_t* x, float* y, int rows, int dim, float eps);
int nv_kernels_gelu_mul_bf16f32(
    void* stream, const uint16_t* gate, const float* pli, uint16_t* y, int n);
int nv_kernels_cast_scale_bf16_f32(
    void* stream, const uint16_t* x, float* y, float scale, int n);
int nv_kernels_add_scale_f32(
    void* stream, const float* a, const float* b, float* y, float scale, int n);
int nv_kernels_incr_pos_rope(void* stream, int* pos, int* rope_pos);
int nv_kernels_argmax_bf16(
    void* stream, const uint16_t* logits, int n, float* part_val, int* part_idx,
    const int* pos, uint32_t* token_out, uint32_t* ring, int ring_mask);
int nv_kernels_argmax_bf16_parts(void);
int nv_kernels_rmsnorm_bf16w_f32out(
    void* stream, const uint16_t* x, const uint16_t* w, float* y,
    int rows, int dim, float eps);
int nv_kernels_rmsnorm_add_scale_bf16(
    void* stream, const uint16_t* x, const uint16_t* w, const uint16_t* res,
    uint16_t* y, float* rstd_out, const uint16_t* next_w, uint16_t* normed_out,
    int rows, int dim, float eps, float scale, float eps_next);
int nv_kernels_qkv_prep(
    void* stream, const uint16_t* qkv, const uint16_t* qw, const uint16_t* kw,
    const float* cos_tbl, const float* sin_tbl, const int* rope_pos,
    const int* cache_pos, int delta, float* q_out, uint16_t* kcache,
    uint16_t* vcache, int NH, int NKV, int HD, float eps);
int nv_kernels_rstd_bf16(
    void* stream, const uint16_t* x, float* rstd_out, int rows, int dim,
    float eps);
int nv_kernels_rms_apply_bf16(
    void* stream, const uint16_t* x, const uint16_t* w, const float* rstd,
    uint16_t* y, int n, int dim);
int nv_kernels_gemv_bf16_normed(
    void* stream, const uint16_t* W, const uint16_t* x, const uint16_t* wn,
    const float* rstd, uint16_t* y, int N, int K);
int nv_kernels_gemv_w4a16_gelu_pli(
    void* stream, const uint32_t* packed, const uint16_t* scale,
    const uint16_t* x, const float* pli, uint16_t* y, int N, int K, int GS);
int nv_kernels_rowquant_i8(
    void* stream, const uint16_t* w, int8_t* wq, float* row_scale, int N, int K);
int nv_kernels_gemv_i8_normed(
    void* stream, const int8_t* wq, const float* row_scale, const uint16_t* x,
    const uint16_t* wn, const float* rstd, uint16_t* y, int N, int K);
int nv_kernels_flash_decode_dev_f32_bf16out(
    void* stream, const float* q, const float* k, const float* v, uint16_t* out,
    const int* pos, int NH, int NKV, int HD, int WINDOW);
int nv_kernels_flash_splitk_scratch_elems(int NH, int HD);
int nv_kernels_flash_depthgate_knee_tokens(void);
int nv_kernels_flash_decode_fp8_splits_pick(int NH, int WINDOW);
int nv_kernels_flash_decode_fp8_splits_pick_for_a_lane_group(
    int NH, int WINDOW, int lanes_in_flight);
int nv_kernels_flash_sm_count(void);
int nv_kernels_flash_decode_splitk_bf16kv(
    void* stream, const float* q, const uint16_t* k, const uint16_t* v,
    uint16_t* out, const int* pos, float* scratch, int NH, int NKV, int HD,
    int WINDOW);
int nv_kernels_flash_decode_fused_bf16kv(
    void* stream, const float* q, const uint16_t* k, const uint16_t* v,
    uint16_t* out, const int* pos, int delta, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW);

int nv_kernels_flash_decode_gqa_fp8kv_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int splits,
    float scaling,
    const int* block_table,
    int block_size
);

int nv_kernels_kvshare_bw_probe(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    const int* block_table,
    int block_size,
    float* sink,
    int total,
    int NKV,
    int SPLITS,
    int mode
);

int nv_kernels_flash_decode_kvshare_fp8kv_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int splits,
    float scaling,
    const int* block_table,
    int block_size,
    int delta,
    int hole_lo_delta,
    int hole_hi_delta
);
int nv_kernels_flash_decode_kvshare_fp8kv_paged_qk_tc(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int splits,
    float scaling,
    const int* block_table,
    int block_size,
    int delta,
    int hole_lo_delta,
    int hole_hi_delta
);

int nv_kernels_flash_decode_rowshared_tctile_fp8kv_mk_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int splits,
    int splits_cap_scratch_rows_over_m,
    float scaling,
    const int* block_table,
    int block_size,
    int delta,
    int M,
    int hole_from_row,
    int hole_lo_delta,
    int hole_hi_delta
);

int nv_kernels_flash_decode_fused_tc_fp8kv_mk_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int splits,
    int splits_cap_scratch_rows_over_m,
    float scaling,
    const int* block_table,
    int block_size,
    int delta,
    int M,
    int hole_from_row,
    int hole_lo_delta,
    int hole_hi_delta
);


int nv_kernels_flash_decode_derivev_fp8kv_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const float* k_scales,
    const float* inv_freq,
    const float* cos_pk,
    const float* sin_pk,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int rope_angles,
    float w_inv,
    float scaling,
    const int* block_table,
    int block_size
);

int nv_kernels_flash_decode_fused_fp8kv_paged(
    void* stream, const uint16_t* q, const uint8_t* k_fp8, const uint8_t* v_fp8,
    const float* k_scales, const float* v_scales, uint16_t* out,
    const int* n_total_dev, float* scratch, unsigned int* fan_in,
    int NH, int NKV, int HD, int WINDOW, int RING, float scaling,
    const int* block_table, int block_size
);

int nv_kernels_flash_decode_fused_fp8kv_mk_paged(
    void* stream, const uint16_t* q, const uint8_t* k_fp8, const uint8_t* v_fp8,
    const float* k_scales, const float* v_scales, uint16_t* out,
    const int* n_total_dev, int delta, int M, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    float scaling, const int* block_table, int block_size
);

int nv_kernels_flash_decode_fused_fp8kv(
    void* stream, const uint16_t* q, const uint8_t* k_fp8,
    const uint8_t* v_fp8, const float* k_scales, const float* v_scales,
    uint16_t* out, const int* n_total_dev, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    float scaling);

int nv_kernels_flash_decode_fused_fp8kv_lane_group(
    void* stream, const uint16_t* q, const uint8_t* k_fp8,
    const uint8_t* v_fp8, const float* k_scales, const float* v_scales,
    uint16_t* out, const int* n_total_dev, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    float scaling, int lanes_in_flight);

int nv_kernels_flash_splitk_scratch_elems_mk(int NH, int HD, int M);

int nv_kernels_flash_decode_fused_bf16kv_mk(
    void* stream, const float* q, const uint16_t* k, const uint16_t* v,
    uint16_t* out, const int* pos, int delta, int M, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW);

int nv_kernels_flash_decode_fused_fp8kv_mk(
    void* stream, const uint16_t* q, const uint8_t* k_fp8,
    const uint8_t* v_fp8, const float* k_scales, const float* v_scales,
    uint16_t* out, const int* n_total_dev, int delta, int M, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    float scaling);
int nv_kernels_flash_decode_fused_fp8kv_mk_hole(
    void* stream, const uint16_t* q, const uint8_t* k_fp8,
    const uint8_t* v_fp8, const float* k_scales, const float* v_scales,
    uint16_t* out, const int* n_total_dev, int delta, int M, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    float scaling, int hole_from_row, int hole_lo_delta, int hole_hi_delta);
int nv_kernels_write_kv_bf16(
    void* stream, const float* src_k, const float* src_v, uint16_t* cache_k,
    uint16_t* cache_v, const int* pos, int NKV, int HD);

int nv_kernels_nvfp4_quantize_row_bf16(
    void* stream,
    const uint16_t* x,
    uint8_t* packed_out,
    uint8_t* scales_out,
    float stored_global,
    int K
);

int nv_kernels_nvfp4_gemv_bf16act(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint16_t* x_bf16,
    uint16_t* y,
    float alpha,
    int N,
    int K
);

int nv_kernels_nvfp4_gemv_bf16act_mrows(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint16_t* x_bf16,
    uint16_t* y,
    float alpha,
    int N,
    int K,
    int M
);

int nv_kernels_nvfp4_gemv_prenorm_fused(void);

int nv_kernels_nvfp4_gemv_prenorm_fused_force(int on);

int nv_kernels_nvfp4_gemv_prenorm_fuses_this_row(int k);

int nv_kernels_nvfp4_gemv_bf16act_prenorm(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint16_t* x_raw,
    const uint16_t* norm_w,
    uint16_t* xn_scratch,
    uint16_t* y,
    float alpha,
    int N,
    int K,
    float eps
);

int nv_kernels_nvfp4_gemv_bf16(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint8_t* x_packed,
    const uint8_t* x_scales,
    uint16_t* y,
    float alpha,
    int N,
    int K
);

int nv_kernels_cutlass_flashinfer_probe(int* out_cutlass_status,
                                        int* out_flashinfer_max_e2m1_x32);

int nv_kernels_quantize_kv_fp8(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* x_fp8_base,
    float* scales_base,
    const int* start_dev,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
);

int nv_kernels_quantize_kv_fp8_unit_scale(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* x_fp8_base,
    float* scales_base,
    const int* start_dev,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
);

int nv_kernels_dequantize_kv_fp8(
    void* stream,
    const uint8_t* x_fp8,
    const float* scales,
    uint16_t* x_bf16,
    int start,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
);

int nv_kernels_gemv_nvfp4_w4a16_dual_m1(
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
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a16_dual_m1_v2(
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
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a16_silu_gate_up_in_m1_v2(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* y,
    float alpha,
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a16_gateup_pair_silu_out_m1_v3(
    void* stream,
    const uint8_t* wq_g,
    const uint8_t* sc_g,
    const uint8_t* wq_u,
    const uint8_t* sc_u,
    const uint16_t* x,
    uint16_t* act,
    float alpha_g,
    float alpha_u,
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a16_down_act_in_m1_v3(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* act,
    uint16_t* y,
    float alpha,
    int N,
    int K
);


int nv_kernels_gemm_nvfp4_w4a16_mk_dual(
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
    int y_stride,
    int K
);

int nv_kernels_gemv_nvfp4_w4a16_silu_gate_up_in_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* y,
    float alpha,
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a8_dual_m1(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a8_dual_argmax_m1(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    uint64_t* packed_best_scratch,
    uint32_t* out_id,
    float* out_logit,
    float alpha_a,
    float alpha_b,
    int N,
    int N_bot,
    int bot_id_base,
    int K
);

int nv_kernels_silu_mul_rowquant_i8_m1(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    int8_t* act_q8,
    float* act_dequant_scale,
    int K
);

int nv_kernels_gemv_nvfp4_w4a8_down_residual_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    int N,
    int K
);

int nv_kernels_gemv_nvfp4_w4a8_down_residual_m1_rstd_emit(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    float* rstd_ssq_count_pack,
    float rstd_eps,
    int N,
    int K
);

int nv_kernels_silu_mul_rowquant_i8_mk_partials_len(int M, int K);

int nv_kernels_silu_mul_rowquant_i8_mk(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* act_staged_bf16,
    float* partial_absmax,
    int8_t* act_q8,
    float* act_dequant_scales,
    int M,
    int K
);

int nv_kernels_silu_mul_stage_partial_absmax_m1(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* act_staged_bf16,
    float* partial_absmax,
    int K
);

int nv_kernels_gemv_nvfp4_w4a8_down_residual_quant_prologue_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* act_staged_bf16,
    const float* partial_absmax,
    int num_partials,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    int N,
    int K
);

int nv_kernels_rmsnorm_residual_writeout_rowquant_i8_m1(
    void* stream,
    const uint16_t* x,
    const uint16_t* res_in,
    const uint16_t* weight,
    uint16_t* res_out,
    int8_t* out_q8,
    float* out_dequant_scale,
    int hidden,
    float eps
);

int nv_kernels_gemv_nvfp4_w4a8_dual_mk(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_dequant_scales,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int M,
    int N,
    int N_b,
    int K
);

int nv_kernels_gemv_nvfp4_w4a8_down_residual_mk(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_dequant_scales,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    int M,
    int N,
    int K
);

int nv_kernels_qkv_norm_rope_kvstore_fp8_decode(
    void* stream,
    const uint16_t* q_raw,
    const uint16_t* k_raw,
    const uint16_t* v_raw,
    const uint16_t* q_norm_w,
    const uint16_t* k_norm_w,
    const float* cos_tab,
    const float* sin_tab,
    const int* pos_dev,
    uint8_t* k_fp8_base,
    uint8_t* v_fp8_base,
    float* k_scales_base,
    float* v_scales_base,
    uint16_t* q_out,
    uint16_t* q_sig_out,
    int n_q,
    int n_kv,
    int hd,
    int q_row_stride,
    int rotary_dim,
    float eps
);

int nv_kernels_qkv_norm_rope_kvstore_fp8_rows(
    void* stream,
    const uint16_t* q_raw,
    const uint16_t* k_raw,
    const uint16_t* v_raw,
    const uint16_t* q_norm_w,
    const uint16_t* k_norm_w,
    const float* cos_tab,
    const float* sin_tab,
    const int* positions,
    uint8_t* k_fp8_base,
    uint8_t* v_fp8_base,
    float* k_scales_base,
    float* v_scales_base,
    uint16_t* q_out,
    uint16_t* q_sig_out,
    int rows,
    int n_q,
    int n_kv,
    int hd,
    int q_row_stride,
    int rotary_dim,
    float eps
);

int nv_kernels_quantize_kv_fp8_paged(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* x_fp8_base,
    float* scales_base,
    const int* start_dev,
    const int* block_table,
    int block_size,
    int n_tokens,
    int n_kv,
    int head_dim
);

int nv_kernels_dequantize_kv_fp8_paged(
    void* stream,
    const uint8_t* x_fp8_base,
    const float* scales_base,
    uint16_t* x_bf16_out,
    const int* block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim
);

int nv_kernels_derive_v_from_k_fp8_paged(
    void* stream,
    const uint8_t* k_fp8_base,
    const float* k_scales_base,
    const float* cos_tab,
    const float* sin_tab,
    const float* inv_freq,
    uint16_t* v_bf16_out,
    const int* block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim,
    int rope_angles,
    int angle_mode,
    int pos_base,
    float w_inv
);

int nv_kernels_copy_kv_block_fp8(
    void* stream,
    const uint8_t* fp8_base,
    const float* scales_base,
    uint8_t* fp8_dst_base,
    float* scales_dst_base,
    int src_block,
    int dst_block,
    int block_size,
    int n_kv,
    int head_dim
);

int nv_kernels_attention_fp8_decode(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    int n_q,
    int n_kv,
    int head_dim,
    const int* n_total_dev,
    int max_total,
    int sliding_window,
    float scaling
);

int nv_kernels_attention_fp8_decode_gscores(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    int n_q,
    int n_kv,
    int head_dim,
    const int* n_total_dev,
    int max_total,
    int sliding_window,
    float scaling,
    float* scores_gmem
);

int nv_kernels_kv_ring_append_bf16(
    void* stream,
    const uint16_t* src,
    uint16_t* dst,
    const int* pos_dev,
    int t,
    int cap,
    int n_kv,
    int head_dim
);

int nv_kernels_kv_shift_bf16(
    void* stream,
    uint16_t* buf,
    int src_row,
    int dst_row,
    int rows,
    int n_kv,
    int head_dim
);

int nv_kernels_attention_bf16_decode_ring(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* ring_meta,
    int cap,
    int window,
    int n_q,
    int n_kv,
    int head_dim,
    float scaling
);

int nv_kernels_residual_add_scale_bf16(
    void* stream,
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* y,
    float scale,
    size_t n
);

int nv_kernels_scale_inplace_bf16(
    void* stream,
    uint16_t* y,
    float scale,
    size_t n
);

int nv_kernels_scale_out_bf16(
    void* stream,
    const uint16_t* x,
    uint16_t* y,
    float scale,
    size_t n
);

int nv_kernels_gelu_tanh_mul_fused_bf16(
    void* stream,
    const uint16_t* fused,
    uint16_t* y,
    int inter,
    size_t tot_pairs
);

int nv_kernels_tanh_softcap_bf16_to_f32(
    void* stream,
    const uint16_t* x,
    float* y,
    float cap,
    size_t n
);

int nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16(
    void* stream,
    const void* a_packed,
    const void* a_scales,
    const void* b_packed,
    const void* b_scales,
    const float* alphas,
    void* d_bf16,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    const int32_t* active_expert_indices,
    int N,
    int K,
    int num_experts,
    int64_t a_row_stride_elems,
    int64_t b_row_stride_elems,
    int64_t c_row_stride_elems,
    void* meta_scratch,
    size_t meta_scratch_bytes,
    void* gemm_workspace,
    size_t gemm_workspace_bytes,
    size_t* required_workspace
);

int nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode(
    void* stream,
    const void* a_packed,
    const void* a_scales,
    const void* b_packed,
    const void* b_scales,
    const float* alphas,
    void* d_bf16,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    const int32_t* active_expert_indices,
    int N,
    int K,
    int num_experts,
    int64_t a_row_stride_elems,
    int64_t b_row_stride_elems,
    int64_t c_row_stride_elems,
    void* meta_scratch,
    size_t meta_scratch_bytes,
    void* gemm_workspace,
    size_t gemm_workspace_bytes,
    size_t* required_workspace
);

int nv_kernels_moe_grouped_fp4_gemv_m1_bf16(
    void* stream,
    const uint8_t* a_packed,
    const uint8_t* a_scales,
    const uint8_t* b_packed,
    const uint8_t* b_scales,
    const float* alphas,
    uint16_t* d_bf16,
    const int32_t* group_expert_ids,
    int num_groups,
    int num_experts_total,
    int n,
    int k,
    int a_tile_stride_rows,
    long long d_group_stride_elems
);

int nv_kernels_moe_grouped_fp4_gemv_gate_up_silu_quant_small_m(
    void* stream,
    const uint8_t* a_packed,
    const uint8_t* a_scales,
    const uint8_t* gate_w,
    const uint8_t* gate_sf,
    const float* gate_alphas,
    const uint8_t* up_w,
    const uint8_t* up_sf,
    const float* up_alphas,
    const float* dn_mini,
    const int32_t* group_expert_ids,
    uint8_t* y_act_fp4,
    uint8_t* y_act_sf,
    int num_groups,
    int num_experts_total,
    int n,
    int k,
    int a_tile_stride_rows
);

int nv_kernels_moe_router_gemv_sg_xstage_small_m(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* wg_bf16,
    const uint16_t* wsg_bf16,
    float* logits_f32,
    uint16_t* sg_out_bf16,
    uint16_t* x_stage_bf16,
    int m,
    int E,
    int K
);

int nv_kernels_cutlass_fp4_gemm_sm120_bf16(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
);

int nv_kernels_gdn_recurrent_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    const float* g_exp,
    const float* beta,
    float* out,
    int B,
    int T,
    int H,
    int K,
    int V
);

int nv_kernels_gdn_train_scan_fwd_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    const float* g_exp,
    const float* beta,
    float* out,
    float* ckpt,
    int bn,
    int t,
    int dk,
    int dv,
    int chunk);

int nv_kernels_gdn_train_scan_bwd_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    const float* g_exp,
    const float* beta,
    const float* d_out,
    const float* ckpt,
    float* scratch,
    float* dq,
    float* dk,
    float* dv,
    float* dg,
    float* dbeta,
    int bn,
    int t,
    int dkd,
    int dvd,
    int chunk);

int nv_kernels_moe_slot_weighted_sum_bf16_f32(void* stream, const uint16_t* rows, const float* w, float* out, int tokens, int top_k, int hidden);
int nv_kernels_moe_slot_weight_grad_bf16_f32(void* stream, const uint16_t* rows, const float* d_out, float* dw, int tokens, int top_k, int hidden);
int nv_kernels_moe_slot_scatter_grad_f32_bf16(void* stream, const float* d_out, const float* w, uint16_t* d_rows, int tokens, int top_k, int hidden);
int nv_kernels_moe_slot_sum_pairs_bf16_f32(void* stream, const uint16_t* a, const uint16_t* b, float* out, int tokens, int top_k, int hidden);

int nv_kernels_attn_masked_softmax_rows_bf16(void* stream, const float* s, uint16_t* p, long rows, int t, int window, float scale);
int nv_kernels_attn_masked_softmax_bwd_rows_bf16(void* stream, const float* s, const float* dp, const float* delta, uint16_t* p, uint16_t* ds, long rows, int t, int window, float scale);

int nv_kernels_gdn_train_head_fwd(void* stream, const uint16_t* x, const float* w, float* packed, int b, int t, int conv_dim, int key_dim, int n_v, int v_per_k, int conv_k, float q_scale);
int nv_kernels_gdn_train_head_bwd(void* stream, const uint16_t* x, const float* w, const float* g_packed, float* dy_scratch, uint16_t* dx, int b, int t, int conv_dim, int key_dim, int n_v, int v_per_k, int conv_k, float q_scale);

int nv_kernels_ce_rows_bf16_fwd_dlogits(void* stream, const uint16_t* logits, const uint32_t* targets, const float* weights, float* loss_rows, uint16_t* dlogits, long rows, int vocab);
int nv_kernels_ce_rows_scale_dlogits(void* stream, uint16_t* dlogits, const float* row_scale, long rows, int vocab);

int nv_kernels_rmsnorm_train_fwd(void* stream, const void* x, const float* w, void* y, float* rstd_out, long rows, int dim, float eps, int x_is_bf16);
int nv_kernels_rmsnorm_train_bwd(void* stream, const void* x, const float* w, const float* rstd, const void* g, void* dx, long rows, int dim, int x_is_bf16);

int nv_kernels_rope_train_fwd(void* stream, const void* x, void* y, const float* cos_tbl, const float* sin_tbl, const uint32_t* pos, long tokens, int heads, int head_dim, int rotary_dim, float scale, int x_is_bf16);
int nv_kernels_rope_train_bwd(void* stream, const void* g, void* dx, const float* cos_tbl, const float* sin_tbl, const uint32_t* pos, long tokens, int heads, int head_dim, int rotary_dim, float scale, int x_is_bf16);

int nv_kernels_gdn_prefill_qk_l2norm_from_mixed(
    void* stream,
    const uint16_t* mixed,
    float* q_out,
    float* k_out,
    int BT,
    int HK,
    int conv_dim,
    int key_dim,
    float q_scale,
    float l2_eps
);

int nv_kernels_gdn_recurrent_stateful_gqa_bf16out(
    void* stream,
    const float* qn,
    const float* kn,
    const uint16_t* mixed,
    const float* g_exp,
    const float* beta,
    float* state_inout,
    uint16_t* out,
    int B,
    int T,
    int H,
    int HK,
    int K,
    int V,
    int conv_dim,
    int v_channel_offset
);


int nv_kernels_gdn_recurrent_stateful_gqa_bf16out_shfl(
    void* stream,
    const float* qn,
    const float* kn,
    const uint16_t* mixed,
    const float* g_exp,
    const float* beta,
    float* state_inout,
    uint16_t* out,
    int B,
    int T,
    int H,
    int HK,
    int K,
    int V,
    int conv_dim,
    int v_channel_offset
);



long long nv_kernels_gdn_chunk_gemm_scan_workspace_elems(int T, int H, int HK);

long long nv_kernels_gdn_chunk_gemm_graph_replays_total(void);

int nv_kernels_gdn_chunk_gemm_scan_f32(
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
);

long long nv_kernels_gdn_scan_tree_compose_floor_workspace_elems(int n, int H);

int nv_kernels_gdn_scan_tree_compose_floor(
    void* stream,
    float* workspace,
    long long workspace_elems,
    int n,
    int H
);

long long nv_kernels_gdn_prefill_tc_head_launches_total(void);

int nv_kernels_gdn_prefill_tc_fused_head(
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
);

int nv_kernels_gdn_conv1d_silu_bt_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* state_in,
    const uint16_t* w,
    uint16_t* y,
    uint16_t* state_out,
    int B,
    int T,
    int C,
    int K
);

int nv_kernels_gdn_conv1d_silu_bt_fp8_weights(
    void* stream,
    const uint16_t* x,
    const uint16_t* state_in,
    const uint8_t* w,
    const float* weight_scales,
    uint16_t* y,
    uint16_t* state_out,
    int B,
    int T,
    int C,
    int K
);

int nv_kernels_gdn_prefill_rmsnorm_gate_bf16(
    void* stream,
    const uint16_t* core,
    const uint16_t* z,
    const uint16_t* norm_weight,
    uint16_t* gated,
    int rows,
    int v_dim,
    float rms_eps
);

int nv_kernels_norm_m1_wide_block(void);

int nv_kernels_norm_m1_wide_batch_cap(void);

int nv_kernels_norm_m1_wide_batch_cap_force(int rows);

int nv_kernels_norm_m1_wide_force(int block);

int nv_kernels_norm_m1_split_ctas(void);

int nv_kernels_norm_m1_split_force(int ctas);

int nv_kernels_norm_m1_wide_block_for_launch(size_t batch, size_t hidden);

int nv_kernels_norm_m1_split_ctas_for_launch(size_t batch, size_t hidden);

int nv_kernels_rmsnorm_residual_bf16(
    void* stream,
    const uint16_t* x,
    uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps
);

int nv_kernels_rmsnorm_residual_f32(
    void* stream,
    const float* x,
    float* residual,
    const float* weight,
    float* out,
    size_t batch,
    size_t hidden,
    float eps
);

int nv_kernels_gdn_conv_decode_silu_bf16(
    void* stream,
    const uint16_t* x_new,
    uint16_t* conv_state,
    const uint16_t* w,
    uint16_t* y,
    int conv_dim,
    int k
);

int nv_kernels_gdn_conv_decode_silu_fp8_weights(
    void* stream,
    const uint16_t* x_new,
    uint16_t* conv_state,
    const uint8_t* w,
    const float* weight_scales,
    uint16_t* y,
    int conv_dim,
    int k
);

int nv_kernels_gdn_conv_decode_silu_group_bf16(
    void* stream,
    const uint16_t* x_new,
    uint16_t* const* conv_states,
    const uint16_t* w,
    uint16_t* y,
    int conv_dim,
    int k,
    int lanes,
    int x_lane_stride,
    int y_lane_stride
);

int nv_kernels_gdn_conv_decode_silu_group_fp8_weights(
    void* stream,
    const uint16_t* x_new,
    uint16_t* const* conv_states,
    const uint8_t* w,
    const float* weight_scales,
    uint16_t* y,
    int conv_dim,
    int k,
    int lanes,
    int x_lane_stride,
    int y_lane_stride
);

int nv_kernels_gdn_decode_step_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
);

int nv_kernels_gdn_decode_step_group_vec4_arm_set(int arm);

int nv_kernels_gdn_decode_step_single_lane_arms_set(int vec4_arm, int one_launch);

int nv_kernels_gdn_decode_step_split_group_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* const* states,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int lanes,
    int ab_lane_stride
);

int nv_kernels_gdn_decode_step_group_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* const* states,
    uint16_t* out,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int lanes,
    int mixed_lane_stride,
    int z_lane_stride,
    int ab_lane_stride,
    int out_lane_stride
);

int nv_kernels_gdn_decode_step_split_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
);

int nv_kernels_gdn_decode_step_split_overlap_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
);

int nv_kernels_gdn_state_overlap_join_pending(void* stream);

int nv_kernels_gdn_state_overlap_pending_clear(void);

int nv_kernels_rmsnorm_residual_writeout_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* res_in,
    const uint16_t* weight,
    uint16_t* res_out,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps
);

int nv_kernels_gdn_decode_step_ab_fused_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* a_w,
    const uint16_t* b_w,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    float* state,
    uint16_t* out,
    int hidden,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps
);

int nv_kernels_gemv_e4m3_qkvz_conv_m1(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* conv_w,
    uint16_t* conv_state,
    uint16_t* mixed_out,
    uint16_t* z_out,
    int N,
    int K,
    int conv_dim,
    int K_c
);

int nv_kernels_gdn_conv_decode_chunk_silu_bf16(
    void* stream,
    const uint16_t* x_seq,
    const uint16_t* conv_state,
    const uint16_t* w,
    uint16_t* y,
    uint16_t* ckpt_conv,
    int conv_dim,
    int k,
    int t
);

int nv_kernels_gdn_conv_decode_chunk_silu_fp8_weights(
    void* stream,
    const uint16_t* x_seq,
    const uint16_t* conv_state,
    const uint8_t* w,
    const float* weight_scales,
    uint16_t* y,
    uint16_t* ckpt_conv,
    int conv_dim,
    int k,
    int t
);

int nv_kernels_gdn_decode_chunk_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    const float* state_in,
    float* ckpt_state,
    uint16_t* out,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int t,
    int ab_row_stride
);

int nv_kernels_gdn_decode_chunk_split_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    const float* state_in,
    float* ckpt_state,
    float* live_state_out,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int t,
    int ab_row_stride
);

int nv_kernels_gdn_conv_decode_chunk_silu_tree_bf16(
    void* stream,
    const uint16_t* x_seq,
    const uint16_t* conv_state,
    const uint16_t* w,
    uint16_t* y,
    uint16_t* ckpt_conv,
    const int* parents,
    int conv_dim,
    int k,
    int t
);

int nv_kernels_gdn_conv_decode_chunk_silu_tree_fp8_weights(
    void* stream,
    const uint16_t* x_seq,
    const uint16_t* conv_state,
    const uint8_t* w,
    const float* weight_scales,
    uint16_t* y,
    uint16_t* ckpt_conv,
    const int* parents,
    int conv_dim,
    int k,
    int t
);

int nv_kernels_gdn_decode_chunk_split_tree_bf16(
    void* stream,
    const uint16_t* mixed,
    const uint16_t* z,
    const uint16_t* a_in,
    const uint16_t* b_in,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    const uint16_t* norm_w,
    const float* state_in,
    float* ckpt_state,
    float* live_state_out,
    uint16_t* out,
    float* qn_scratch,
    float* kn_scratch,
    float* g_exp_scratch,
    float* beta_scratch,
    uint16_t* core_scratch,
    const int* parents,
    int n_k,
    int n_v,
    int d_k,
    int d_v,
    float rms_eps,
    int t,
    int ab_row_stride
);

int nv_kernels_gdn_gating_bf16(
    void* stream,
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    float* g_out,
    uint16_t* beta_out,
    size_t tokens,
    size_t num_heads
);

int nv_kernels_gdn_gating_f32(
    void* stream,
    const float* a,
    const float* b,
    const float* A_log,
    const float* dt_bias,
    float* g_out,
    float* beta_out,
    size_t tokens,
    size_t num_heads
);

int nv_kernels_silu_mul_quantize_nvfp4_bf16_per_expert(
    void* stream,
    const uint16_t* y_gate_bf16,
    const uint16_t* y_up_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
);

int nv_kernels_silu_mul_quantize_nvfp4_bf16_per_expert_strided(
    void* stream,
    const uint16_t* y_gate_bf16,
    const uint16_t* y_up_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
);

int nv_kernels_quantize_nvfp4_bf16_per_expert(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
);

int nv_kernels_quantize_nvfp4_bf16_per_expert_strided(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
);

int nv_kernels_quantize_nvfp4_bf16(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    int m_padded,
    int m_logical,
    int K
);

int nv_kernels_quantize_nvfp4_bf16_rows(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    int m_rows,
    int K
);

int nv_kernels_rmsnorm_quantize_nvfp4_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* weight_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    float eps,
    int m_padded,
    int m_logical,
    int K
);

int nv_kernels_depthwise_conv1d_silu_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* w_bf16,
    uint16_t* y_bf16,
    int B, int C, int T, int K
);

int nv_kernels_depthwise_conv1d_silu_fp8_weights(
    void* stream,
    const uint16_t* x_bf16,
    const uint8_t* w_fp8,
    const float* weight_scales,
    uint16_t* y_bf16,
    int B, int C, int T, int K
);

int nv_kernels_copy_cols_bf16(
    void* stream,
    const uint16_t* src,
    uint16_t* dst,
    int rows,
    int width,
    long long src_stride,
    long long dst_stride,
    long long src_off,
    long long dst_off
);

int nv_kernels_mul_sigmoid_rowgate_f32(
    void* stream,
    const float* x,
    const float* gate_logits,
    float* y,
    int rows,
    int hidden
);

int nv_kernels_gather_rows_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const int32_t* src_idx,
    uint16_t* out_bf16,
    int m_total_padded,
    int hidden,
    int n_tokens
);

int nv_kernels_gather_rows_bf16_strided(
    void* stream,
    const uint16_t* x_bf16,
    const int32_t* src_idx,
    uint16_t* out_bf16,
    int m_total_padded,
    int hidden,
    int n_tokens,
    int row_stride
);

int nv_kernels_moe_unpermute_scatter_tail(
    void* stream,
    const uint16_t* y_sorted_bf16,
    const float* topk_weights,
    const int32_t* inv_perm,
    const float* shared_f32,
    const uint16_t* resid_bf16,
    uint16_t* out_bf16,
    int n_tokens,
    int k,
    int hidden,
    int y_sorted_row_stride
);

int nv_kernels_moe_unpermute_scatter_shared_sig_tail(
    void* stream,
    const uint16_t* y_sorted_bf16,
    const float* topk_weights,
    const int32_t* inv_perm,
    const uint16_t* shared_gate_logit_bf16,
    uint16_t* out_bf16,
    int n_tokens,
    int k,
    int hidden,
    int y_sorted_row_stride
);

int nv_kernels_moe_gemv_swiglu_bf16_m1(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    const int32_t* ids,
    const uint16_t* x,
    uint16_t* h,
    int k,
    int num_experts,
    int inter,
    int hidden
);

int nv_kernels_moe_gemv_swiglu_bf16_mb(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    const int32_t* ids,
    const uint16_t* x,
    uint16_t* h,
    int b,
    int k,
    int num_experts,
    int inter,
    int hidden
);

int nv_kernels_moe_gemv_down_tail_bf16_m1(
    void* stream,
    const uint16_t* down,
    const int32_t* ids,
    const float* weights,
    const uint16_t* h,
    const float* shared_f32,
    const uint16_t* resid,
    uint16_t* out,
    int k,
    int num_experts,
    int hidden,
    int inter
);

int nv_kernels_moe_gemv_down_tail_bf16_mb(
    void* stream,
    const uint16_t* down,
    const int32_t* ids,
    const float* weights,
    const uint16_t* h,
    const float* shared_f32,
    const uint16_t* resid,
    uint16_t* out,
    int b,
    int k,
    int num_experts,
    int hidden,
    int inter
);

int nv_kernels_moe_unpermute_scatter(
    void* stream,
    const uint16_t* y_sorted_bf16,
    const float* topk_weights,
    const int32_t* inv_perm,
    float* y_acc_f32,
    int n_tokens,
    int k,
    int hidden,
    int y_sorted_row_stride
);

int nv_kernels_moe_route_topk(
    void* stream,
    const float* logits,
    const float* bias,
    int32_t* topk_ids,
    float* topk_weights,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling
);

int nv_kernels_moe_route_topk_shared_tail(
    void* stream,
    const float* logits,
    const float* bias,
    int32_t* topk_ids,
    float* topk_weights,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling,
    int shared_tail_id
);

int nv_kernels_moe_route_gather_quant_small_m(
    void* stream,
    const float* logits,
    const float* bias,
    const uint16_t* x_bf16,
    const float* globals_gu,
    const float* globals_dn,
    int32_t* topk_ids,
    float* topk_weights,
    float* gu_mini,
    float* dn_mini,
    uint8_t* x_fp4,
    uint8_t* x_sf,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling,
    int shared_tail_id,
    int hidden,
    int min_tile,
    int quant_ctas
);

int nv_kernels_gather_f32_by_ids(
    void* stream,
    const float* src,
    const int32_t* ids,
    float* dst,
    int n
);

int nv_kernels_moe_permute(
    void* stream,
    const int32_t* topk_ids,
    int32_t* permuted_token_idx,
    int32_t* expert_offsets,
    int32_t* inv_perm,
    int32_t* scratch_counts,
    int n_tokens,
    int k,
    int num_experts
);

int nv_kernels_moe_permute_padded(
    void* stream,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* permuted_token_idx,
    int32_t* expert_offsets,
    int32_t* inv_perm,
    int32_t* scratch_counts,
    int n_tokens,
    int k,
    int num_experts,
    int tile_rows,
    int padded_capacity
);

int nv_kernels_dflash_accept_f32(
    void* stream,
    const float* logits,
    const uint32_t* drafts,
    uint32_t* row_argmax,
    uint32_t* out,
    float* part_val,
    int* part_idx,
    int m,
    int vocab
);
int nv_kernels_dflash_accept_parts(void);

int nv_kernels_softplus_gate_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int groups,
    int hd
);

int nv_kernels_laguna_rope_scale_bf16(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* pos_base,
    int t,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale
);

int nv_kernels_gemv_bf16_qkvg_normed(
    void* stream, const uint16_t* Wq, const uint16_t* Wk, const uint16_t* Wv,
    const uint16_t* Wg, const uint16_t* x, const uint16_t* wn,
    const float* rstd, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K);

int nv_kernels_gemv_q8_qkvg_normed(
    void* stream, int fp8, const void* Wq, const float* Sq, const void* Wk,
    const float* Sk, const void* Wv, const float* Sv, const uint16_t* Wg,
    const uint16_t* x, const uint16_t* wn, const float* rstd, uint16_t* yq,
    uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng,
    int K);

int nv_kernels_laguna_rstd256_bf16(
    void* stream, const uint16_t* x, float* rstd_out, int dim, float eps);

int nv_kernels_laguna_qk_normrope_bf16(
    void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out,
    uint16_t* k_out, const uint16_t* qw, const uint16_t* kw,
    const float* cos_tbl, const float* sin_tbl, const int* pos_base,
    int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale,
    float eps_q, float eps_k);
int nv_kernels_laguna_qk_normrope_bf16_rows(void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out, const uint16_t* qw, const uint16_t* kw, const float* cos_tbl, const float* sin_tbl, const int* positions, int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale, float eps_q, float eps_k, int rows);
int nv_kernels_laguna_qk_normrope_fp8_weight_rows(void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out, const uint8_t* qw, const float* qs, const uint8_t* kw, const float* ks, const float* cos_tbl, const float* sin_tbl, const int* positions, int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale, float eps_q, float eps_k, int rows);
int nv_kernels_laguna_qk_normrope_bf16_base_rows(void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out, const uint16_t* qw, const uint16_t* kw, const float* cos_tbl, const float* sin_tbl, const int* pos_base, int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale, float eps_q, float eps_k, int rows);
int nv_kernels_laguna_qk_normrope_fp8_weight_base_rows(void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out, const uint8_t* qw, const float* qs, const uint8_t* kw, const float* ks, const float* cos_tbl, const float* sin_tbl, const int* pos_base, int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale, float eps_q, float eps_k, int rows);
int nv_kernels_laguna_gemv_fp8_qkvg_fp8_norm_rows(void* stream, const uint8_t* wq, const float* sq, const uint8_t* wk, const float* sk, const uint8_t* wv, const float* sv, const uint8_t* wg, const float* sg, const uint16_t* x, const uint8_t* wn, const float* sn, float eps, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int nq, int nk, int nv, int ng, int k, int rows);
int nv_kernels_laguna_gemv_fp8_qkvg_rows_rstd(void* stream, const uint8_t* wq, const float* sq, const uint8_t* wk, const float* sk, const uint8_t* wv, const float* sv, const uint8_t* wg, const float* sg, const uint16_t* x, const void* wn, const float* sn, const float* rstd, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int nq, int nk, int nv, int ng, int k, int rows);
int nv_kernels_laguna_prefill_qk_normrope_bf16(void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out, const uint16_t* qw, const uint16_t* kw, const float* cos_tbl, const float* sin_tbl, const int* positions, int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale, float eps_q, float eps_k, int rows);
int nv_kernels_laguna_prefill_qk_normrope_fp8_weight(void* stream, const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out, const uint8_t* qw, const uint8_t* kw, const float* qs, const float* ks, const float* cos_tbl, const float* sin_tbl, const int* positions, int n_q, int n_kv, int head_dim, int rotary_dim, float rot_scale, float eps_q, float eps_k, int rows);
int nv_kernels_laguna_prefill_gate_bf16(void* stream, const uint16_t* attn, const uint16_t* gate, uint16_t* out, int rows, int n_q, int head_dim);

int nv_kernels_laguna_flash_decode_gqa_scratch_elems(int n_kv);
int nv_kernels_laguna_flash_decode_gqa(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* total_ptr,
    int delta,
    float* scratch,
    unsigned int* fan_in,
    int n_q,
    int n_kv,
    int head_dim,
    int window,
    float scale
);

int nv_kernels_laguna_flash_decode_gqa_rows(void* stream, const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, const int* total_ptr, int delta, float* scratch, unsigned int* fan_in, int n_q, int n_kv, int head_dim, int rows, int window, float scale);
int nv_kernels_laguna_flash_decode_verify(
    void* stream,
    const uint16_t* q,
    const uint8_t* k,
    const uint8_t* v,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* committed,
    float* scratch,
    unsigned int* fan_in,
    int n_q,
    int n_kv,
    int head_dim,
    int rows,
    int capacity,
    int splits,
    float scale,
    int kv_is_fp8
);
int nv_kernels_laguna_gemv_bf16_qkvg_normed_rows(void* stream, const uint16_t* Wq, const uint16_t* Wk, const uint16_t* Wv, const uint16_t* Wg, const uint16_t* x, const uint16_t* wn, const float* rstd, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng, int K, int rows);
int nv_kernels_laguna_gemv_bf16_rows(void* stream, const uint16_t* W, const uint16_t* x, uint16_t* y, int N, int K, int rows);
int nv_kernels_laguna_gemv_q8_qkvg_normed_rows(void* stream, int fp8, const void* Wq, const float* Sq, const void* Wk, const float* Sk, const void* Wv, const float* Sv, const uint16_t* Wg, const uint16_t* x, const uint16_t* wn, const float* rstd, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng, int K, int rows);
int nv_kernels_laguna_gemv_q8_qkvg_normed_rows_rstd_fused(void* stream, int fp8, const void* Wq, const float* Sq, const void* Wk, const float* Sk, const void* Wv, const float* Sv, const uint16_t* Wg, const uint16_t* x, const uint16_t* wn, float eps, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng, int K, int rows);
int nv_kernels_laguna_gemv_fp8_qkvg_normed_rows_rstd_fused(void* stream, const uint8_t* Wq, const float* Sq, const uint8_t* Wk, const float* Sk, const uint8_t* Wv, const float* Sv, const uint8_t* Wg, const float* Sg, const uint16_t* x, const uint16_t* wn, float eps, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng, int K, int rows);
int nv_kernels_laguna_gemv_q8_rows(void* stream, int fp8, const void* W, const float* S, const uint16_t* x, uint16_t* y, int N, int K, int rows);
int nv_kernels_laguna_gemv_q8_rows_gated(void* stream, int fp8, const void* W, const float* S, const uint16_t* attn, const uint16_t* gate, int gate_heads, int gate_hd, uint16_t* y, int N, int K, int rows);
int nv_kernels_laguna_flash_decode_verify_window(void* stream, const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, const int* committed, float* scratch, unsigned int* fan_in, int n_q, int n_kv, int head_dim, int rows, int capacity, int window, int rows_per_block, float scale);
int nv_kernels_laguna_flash_decode_verify_fused_tc(void* stream, const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, const int* committed, float* scratch, unsigned int* fan_in, int n_q, int n_kv, int head_dim, int rows, int capacity, int splits, float scale);
int nv_kernels_laguna_flash_decode_verify_fused_tc_fp8(void* stream, const uint16_t* q, const uint8_t* k, const uint8_t* v, const float* k_scales, const float* v_scales, uint16_t* out, const int* committed, float* scratch, unsigned int* fan_in, int n_q, int n_kv, int head_dim, int rows, int capacity, int splits, float scale, int kvq);
int nv_kernels_quantize_kv_int8(void* stream, const uint16_t* x_bf16, uint8_t* x_int8_base, float* scales_base, const int* start_dev, int n_tokens, int n_kv, int head_dim, int ring);
int nv_kernels_dequantize_kv_int8(void* stream, const uint8_t* x_int8, const float* scales, uint16_t* x_bf16, int start, int n_tokens, int n_kv, int head_dim, int ring);
int nv_kernels_kv_append_int8(void* stream, const uint16_t* k_new, const uint16_t* v_new, uint8_t* kc, uint8_t* vc, float* k_scale, float* v_scale, const int* n_committed, int K, int NKV, int HD, int ring);
int nv_kernels_laguna_rstd256_bf16_rows(void* stream, const uint16_t* x, float* rstd_out, int dim, float eps, int rows);

int nv_kernels_laguna_seqlens_prep(
    void* stream,
    const int* meta,
    int* cu_full,
    int* cu_slide,
    int t
);

int nv_kernels_prof_timestamp(
    void* stream,
    unsigned long long* out
);

int nv_kernels_softplus_gate_exact_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int groups,
    int hd
);

int nv_kernels_gemv_w4a16_m1_proto(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int GS,
    int variant
);

int nv_kernels_gemv_w4a16_m1_proto_geglu(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int n_gate_rows_then_up_rows,
    int k,
    int gs,
    int variant);

int nv_kernels_gemv_w4a16_m1_proto_post_norm(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int n,
    int k,
    int gs,
    int variant,
    const uint16_t* res,
    const uint16_t* w,
    uint16_t* h_out,
    float* rstd_out,
    const uint16_t* next_w,
    uint16_t* normed_out,
    float eps,
    float res_scale,
    float eps_next,
    unsigned int* counter);

int nv_kernels_sampler_topk_topp(
    void* stream,
    const float* logits,
    const uint64_t* seeds,
    float* probs_out,
    uint32_t* token_out,
    size_t batch,
    size_t vocab,
    float temperature,
    uint32_t top_k,
    float top_p
);

int nv_kernels_tree_verify_attn_bf16(
    void* stream,
    const uint16_t* q,
    const uint16_t* kc,
    const uint16_t* vc,
    const int* n_committed,
    const unsigned char* mask,
    const int* positions,
    uint16_t* out,
    int NH,
    int NKV,
    int HD,
    int K,
    int window
);

int nv_kernels_gqa512_scratch_elems(int NH, int M, int splits);

int nv_kernels_gqa512_verify_bf16(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* pos,
    int delta,
    int M,
    float* scratch,
    int NH,
    int NKV,
    int HD,
    int splits
);

int nv_kernels_gqa512_verify_fp8(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scale,
    const float* v_scale,
    uint16_t* out,
    const int* pos,
    int delta,
    int M,
    float* scratch,
    int NH,
    int NKV,
    int HD,
    int splits,
    float scaling
);

int nv_kernels_kv_append_bf16(
    void* stream,
    const uint16_t* k_new,
    const uint16_t* v_new,
    uint16_t* kc,
    uint16_t* vc,
    const int* n_committed,
    int K,
    int NKV,
    int HD
);

int nv_kernels_kv_compact_bf16(
    void* stream,
    uint16_t* kc,
    uint16_t* vc,
    uint16_t* sk,
    uint16_t* sv,
    const int* path,
    int base,
    int A,
    int stride
);

int nv_kernels_tree_verify_attn_fp8(
    void* stream,
    const uint16_t* q,
    const uint8_t* kc,
    const uint8_t* vc,
    const float* k_scale,
    const float* v_scale,
    const int* n_committed,
    const unsigned char* mask,
    const int* positions,
    uint16_t* out,
    int NH, int NKV, int HD, int K, int window, int ring
);

int nv_kernels_kv_append_fp8(
    void* stream,
    const uint16_t* k_new,
    const uint16_t* v_new,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    const int* n_committed,
    int K, int NKV, int HD, int ring
);

int nv_kernels_kv_append_fp8_unit_scale(
    void* stream,
    const uint16_t* k_new,
    const uint16_t* v_new,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    const int* n_committed,
    int K, int NKV, int HD, int ring
);

int nv_kernels_verify_qkv_prep(
    void* stream,
    const uint16_t* qkv,
    long long qkv_stride,
    long long q_off,
    long long k_off,
    long long v_off,
    const uint16_t* qw,
    const uint16_t* kw,
    const uint16_t* vw,
    float eps,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    uint16_t* q_out,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    const int32_t* n_committed,
    int K, int NQ, int NKV, int HD, int ring
);

int nv_kernels_rmsnorm2_residual_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* residual,
    const uint16_t* w1,
    const uint16_t* w2,
    uint16_t* sum_out,
    uint16_t* normed_out,
    size_t batch,
    size_t hidden,
    float eps
);

int nv_kernels_rmsnorm_residual_scale_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* residual,
    const uint16_t* w,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps,
    float scale
);

int nv_kernels_kv_compact_fp8(
    void* stream,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    uint8_t* sk,
    uint8_t* sv,
    float* ssk,
    float* ssv,
    const int* path,
    int base, int A, int NKV, int HD, int ring
);

int nv_kernels_rope_bf16_oop(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    size_t batch,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
);

int nv_kernels_attn_gate_sigmoid_mul_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* q_raw,
    uint16_t* out,
    size_t rows,
    size_t n_heads,
    size_t head_dim
);

int nv_kernels_rope_bf16_partial_oop(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    size_t batch,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t rotary_dim
);
int nv_kernels_token_map_u32(
    void* stream, const uint32_t* map, const uint32_t* idx, uint32_t* out);
int nv_kernels_argmax_f32_rows(
    void* stream, const float* logits, int rows, int n, float* part_val,
    int* part_idx, uint32_t* out);
int nv_kernels_cutlass_fp4_gemm_sm120_bf16_streamk(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
);
int nv_kernels_e4m3_pertensor_quant_two_segments(
    void* stream,
    const uint16_t* x0_bf16,
    const uint16_t* x1_bf16,
    long long n_each,
    uint8_t* q0_out,
    uint8_t* q1_out,
    float* scale_out,
    unsigned int* amax_scratch
);
int nv_kernels_dx_twin_nvfp4_requant_transposed_kmajor(
    void* stream,
    const uint8_t* w_packed,
    const uint8_t* w_scales_swizzled,
    float alpha,
    int n_out,
    int k_in,
    uint32_t* kmajor_out,
    uint8_t* scales_linear_out
);
int nv_kernels_dx_twin_nvfp4_scales_to_marlin(
    void* stream,
    const uint8_t* scales_linear,
    int n,
    int kb,
    float global_scale,
    uint8_t* bytes_e5m3_out,
    float* global_out,
    unsigned int* vmin_scratch,
    int* flags_out
);
int nv_kernels_dequant_weight_nvfp4_swizzled_to_bf16_rowmajor(
    void* stream,
    const uint8_t* w_packed,
    const uint8_t* w_scales_swizzled,
    uint16_t* out_bf16,
    float alpha,
    int n,
    int k
);
int nv_kernels_dequant_weight_e4m3_rowscale_to_bf16_rowmajor(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    uint16_t* out_bf16,
    int n,
    int k
);
int nv_kernels_cutlass_fp4_gemm_sm120_bf16_tiled(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    int tile,
    int stream_k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
);
int nv_kernels_nvfp4_gemm_sm120_v192_bf16(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k
);
int nv_kernels_quantize_nvfp4_silu_candle_bf16(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint8_t* packed_out,
    uint8_t* scales_out,
    float stored_global,
    int m_padded,
    int m_logical,
    int K,
    int sf_blk_rows
);
int nv_kernels_quantize_nvfp4_bf16_sf96(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    int m_logical,
    int K
);
int nv_kernels_nvfp4_cfx_shape_index(int m, int n, int k);
int nv_kernels_nvfp4_cfx_load(
    const void* cubin,
    size_t cubin_bytes,
    int m, int n, int k
);
int nv_kernels_nvfp4_cfx_ready(int m, int n, int k);
int nv_kernels_nvfp4_cfx_gemm_bf16(
    void* stream,
    const void* a_fp4,
    const void* a_sf96,
    const void* b_fp4,
    const void* b_sf128,
    const float* alpha_dev,
    void* d_bf16,
    int m, int n, int k
);

int nv_kernels_e4b_online_softmax_tile(
    void* stream, const float* s, uint16_t* p, float* m_run, float* l_run, float* o_acc,
    int rows_per_group, int groups, int n, int tile_len, int key_off, int q_pos0, int hd,
    float scale, int first_tile);
int nv_kernels_e4b_attn_finalize(
    void* stream, const float* o_acc, const float* l_run, uint16_t* out, int n, int heads, int hd);

int nv_kernels_flash_decode_gqa4_hd512_scratch_elems(int NH, int capacity);
int nv_kernels_flash_decode_fused_gqa4_hd512(
    void* stream, const float* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
    const int* pos, int delta, float* scratch, unsigned int* fan_in, int NH, int NKV, int HD,
    int WINDOW, int capacity);

int nv_kernels_e4b_prefill_qkv_prep(
    void* stream, const uint16_t* qkv, int qkv_row_stride, const uint16_t* qw, const uint16_t* kw,
    const float* cos_tbl, const float* sin_tbl, const int* pos, float* q_out, float* kcache,
    float* vcache, uint16_t* ktwin, uint16_t* vtwin, int n, int write_off, int cap, int NH, int NKV,
    int HD, float eps);

#ifdef __cplusplus
}
#endif
